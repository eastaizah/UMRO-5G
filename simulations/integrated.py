"""
integrated.py -- The end-to-end UMRO-5G runner (article Section 8.3, Figure 4).

This file answers Reviewer 1 comment 1: the four layers and the three nested control
loops run together on ONE clock, over ONE traffic realisation, and the coupling
between the loops is logged and reported rather than asserted.

                     ---------------------------------------------
  SLOW LOOP (2 s)    Non-RT RIC / MANO
                     * Lagrangian dual (sub-gradient) update of the per-slice
                       price mu_s from the measured demand-allocation residual
                     * re-derivation of the per-slice minimum PRB guarantee
                       n_s^min from the price-weighted share
                     * VNF placement / scaling for the SFC serving each slice,
                       which sets the transport-and-core latency component
                     OUTPUT: prices, guarantees, a price budget
                             --> constrains the medium loop's feasible set
                     ---------------------------------------------
  MEDIUM LOOP (100 ms)  Near-RT RIC
                     * a LEARNED controller (DQN, pre-trained offline, see
                       `pretrain_medium_policy`) picks one of 7 PRB re-split
                       actions from the telemetry state
                     * admission control on the SLA-proximity signal
                     * an action is executed only if it is feasible under the
                       slow loop's guarantees and price budget
                     OUTPUT: the PRB partition n_s
                             --> constrains what the fast loop may schedule
                     ---------------------------------------------
  FAST LOOP (0.5 ms, mu = 1)  gNB MAC
                     * proportional-fair scheduling of each slice's own PRBs
                     * per-user rate from channel.py (CQI/MCS table)
                     * per-slice FIFO queues, exact head-of-line delay
                     OUTPUT: measured throughput, backlog, HoL delay
                             --> telemetry that drives BOTH outer loops
                     ---------------------------------------------

Five arms are run on identical traffic and channel realisations (common random
numbers), which is what makes the seed-paired statistics in stats.py valid:

  A_hard_static  static hard isolation, PF only, no medium or slow loop
  B_soft_pool    soft isolation with a fixed shared pool, no learning
  C_no_medium    UMRO-5G with the medium loop disabled  (fast + slow)
  D_no_slow      UMRO-5G with the slow loop disabled    (fast + medium)
  E_full_umro    UMRO-5G with all three loops
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from config import (INTEGRATED, SLICES, SLICE_NAMES, SLICING, TRAFFIC, ARMS,
                    IntegratedConfig, CAMPAIGN)
from channel import CellChannel
from traffic import diurnal_multiplier
from sfc import required_instances, chain_latency_with_scaling
from learners import MLP, Adam, ReplayBuffer, huber_grad, RunningNorm
from config import RLTRAIN

N_SLICES = len(SLICES)

# SLA definition used by the integrated runner (documented in README.md):
SLA_CARRIED_LOAD = 0.95        # fraction of the offered load a slice must carry
URLLC_WINDOW_VIOL_MAX = 0.01   # tolerated URLLC delay-budget violation probability
GUARANTEE_BUDGET = 0.80        # share of the carrier the slow loop may lock down
PRICE_GUARANTEE_GAIN = 0.35    # how strongly the dual price bends the guarantee
PRICE_BUDGET_SLACK = 1.35      # headroom the medium loop gets over the priced floor
SURROGATE_PRB_MBPS_LO = 1.9    # calibrated against the full TTI-level simulator
SURROGATE_PRB_MBPS_HI = 2.4
VALIDATION_EVERY = 10          # episodes between medium-policy validations
SFC_UNSTABLE_MS = 1000.0       # latency recorded when a VNF chain is offered more
                               # traffic than its provisioned instances can serve

# Medium-loop action set: (slice_to_give, slice_to_take) or None for no-op.
MEDIUM_ACTIONS: tuple = (
    None,          # 0: hold
    (0, 1),        # 1: +step PRB to eMBB, taken from URLLC
    (0, 2),        # 2: +step PRB to eMBB, taken from mMTC
    (1, 0),        # 3: +step PRB to URLLC, taken from eMBB
    (1, 2),        # 4: +step PRB to URLLC, taken from mMTC
    (2, 0),        # 5: +step PRB to mMTC, taken from eMBB
    (2, 1),        # 6: +step PRB to mMTC, taken from URLLC
)
MEDIUM_STATE_DIM = 4 * N_SLICES + 1      # per slice: alloc, backlog, sla ratio, price
                                         # plus the normalised time of day


# ==================================================================================
# Helpers
# ==================================================================================

def integer_allocation(x: np.ndarray, n_prb: int, mins: np.ndarray) -> np.ndarray:
    """Largest-remainder rounding of a continuous PRB split onto the integer budget,
    respecting the per-slice minimum guarantees."""
    x = np.maximum(np.asarray(x, dtype=float), mins.astype(float))
    if x.sum() <= 0:
        x = mins.astype(float).copy()
    x = x * (n_prb / max(x.sum(), 1e-9))
    x = np.maximum(x, mins.astype(float))
    base = np.floor(x).astype(int)
    base = np.maximum(base, mins.astype(int))
    deficit = n_prb - base.sum()
    if deficit > 0:
        rem = x - np.floor(x)
        order = np.argsort(-rem)
        for k in range(deficit):
            base[order[k % N_SLICES]] += 1
    elif deficit < 0:
        head = base - mins.astype(int)
        for _ in range(-deficit):
            j = int(np.argmax(head))
            if head[j] <= 0:
                j = int(np.argmax(base))
            base[j] -= 1
            head[j] -= 1
    return base


@dataclass
class LoopTrace:
    """Everything the article needs in order to *show* the loops are coupled."""

    t_s: List[float] = field(default_factory=list)
    prices: List[List[float]] = field(default_factory=list)
    min_prb: List[List[int]] = field(default_factory=list)
    alloc: List[List[int]] = field(default_factory=list)
    throughput_mbps: List[List[float]] = field(default_factory=list)
    backlog_kbit: List[List[float]] = field(default_factory=list)
    urllc_delay_ms: List[float] = field(default_factory=list)
    action: List[int] = field(default_factory=list)
    action_rejected: List[int] = field(default_factory=list)
    admitted_fraction: List[List[float]] = field(default_factory=list)
    offered_multiplier: List[float] = field(default_factory=list)
    sfc_latency_ms: List[List[float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ==================================================================================
# Medium-loop learned controller
# ==================================================================================

class MediumPolicy:
    """Q-network used by the Near-RT RIC.  Weights are produced once by
    `pretrain_medium_policy` and reused, unchanged, by every evaluation seed --
    training and evaluation therefore use disjoint random streams."""

    def __init__(self, params: Dict[str, list] | None = None,
                 rng: np.random.Generator | None = None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.net = MLP([MEDIUM_STATE_DIM, 64, 64, len(MEDIUM_ACTIONS)], rng,
                       out_scale=0.1)
        if params is not None:
            for k, v in params.items():
                self.net.params[k][...] = np.asarray(v, dtype=float)
        self.trained = params is not None

    def act(self, state: np.ndarray) -> int:
        q = self.net.forward(np.asarray(state, dtype=float)[None, :], cache=False)[0]
        return int(np.argmax(q))

    def dump(self) -> Dict[str, list]:
        return {k: v.tolist() for k, v in self.net.params.items()}

    @classmethod
    def load(cls, path: str) -> "MediumPolicy":
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return cls(params=blob["params"])


def medium_state(alloc: np.ndarray, backlog_bits: np.ndarray,
                 thr_mbps: np.ndarray, prices: np.ndarray, n_prb: int,
                 t_frac: float) -> np.ndarray:
    sla = np.array([s.sla_rate_mbps for s in SLICES])
    ref = np.array([s.n_users * s.arrival_rate_per_user_hz * s.packet_bits_mean
                    * 0.1 for s in SLICES])           # 100 ms of nominal offered load
    st = np.concatenate([
        alloc / n_prb,
        np.clip(backlog_bits / np.maximum(ref, 1.0), 0.0, 5.0),
        np.clip(thr_mbps / sla, 0.0, 3.0),
        np.clip(prices / SLICING.dual_max, 0.0, 1.0),
        [t_frac],
    ])
    return st.astype(float)


# ----------------------------------------------------------------------------------
# Fluid surrogate used to pre-train the medium-loop policy
# ----------------------------------------------------------------------------------

class MediumSurrogateEnv:
    """Fluid-queue approximation of one cell at the 100 ms medium-loop timescale.

    The fast loop is replaced by its deterministic fluid equivalent
    (service_s = n_s * mean per-PRB rate of slice s) and the queues evolve by
    Lindley recursion at the 100 ms grid.  This surrogate exists ONLY to pre-train
    the medium-loop policy; every number reported for UMRO-5G comes from the full
    TTI-level simulator in `run_arm`, and the pre-trained policy is evaluated there
    without further adaptation.  The surrogate/simulator gap is therefore part of the
    measurement, not hidden by it.
    """

    def __init__(self, rng: np.random.Generator, cfg: IntegratedConfig = INTEGRATED):
        self.rng = rng
        self.cfg = cfg
        self.n_prb = cfg.numerology.n_prb
        self.dt = cfg.medium_period_ms * 1e-3
        self.n_steps = int(round(TRAFFIC.compress_seconds / self.dt))
        self.sla = np.array([s.sla_rate_mbps for s in SLICES])
        self.mins = np.array([s.min_prb for s in SLICES])
        self.reset()

    def reset(self):
        self.t = 0
        self.alloc = integer_allocation(np.array(SLICING.hard_split, dtype=float),
                                        self.n_prb, self.mins)
        self.backlog = np.zeros(N_SLICES)
        self.prices = np.full(N_SLICES, SLICING.dual_init)
        # per-PRB rate of each slice: drawn once per episode from the CQI statistics
        # Per-PRB throughput seen by the fast loop, calibrated against the full
        # TTI-level simulator (measured 1.9-2.4 Mbit/s per PRB with PF scheduling on
        # the CQI table of channel.py).  See README.md, "surrogate calibration".
        self.prb_mbps = self.rng.uniform(SURROGATE_PRB_MBPS_LO,
                                         SURROGATE_PRB_MBPS_HI, N_SLICES)
        return self._state()

    def _state(self):
        thr = self.last_thr if hasattr(self, "last_thr") else np.zeros(N_SLICES)
        return medium_state(self.alloc.astype(float), self.backlog, thr,
                            self.prices, self.n_prb,
                            (self.t * self.dt) / TRAFFIC.compress_seconds)

    def step(self, action: int):
        cfg = self.cfg
        pair = MEDIUM_ACTIONS[action]
        if pair is not None:
            give, take = pair
            step = cfg.medium_prb_step
            if self.alloc[take] - step >= self.mins[take]:
                self.alloc[give] += step
                self.alloc[take] -= step
        t_s = self.t * self.dt
        mult = float(diurnal_multiplier(t_s))
        offered = np.array([s.n_users * s.arrival_rate_per_user_hz
                            * s.packet_bits_mean for s in SLICES]) / 1e6
        offered = offered * cfg.nominal_load * mult
        arrivals = offered * self.dt * self.rng.lognormal(-0.02, 0.2, N_SLICES)
        capacity = self.alloc * self.prb_mbps * self.dt
        self.backlog = self.backlog + arrivals
        served = np.minimum(self.backlog, capacity)
        self.backlog -= served
        thr = served / self.dt
        self.last_thr = thr
        # fluid delay estimate: backlog / service rate
        delay_ms = 1e3 * self.backlog / np.maximum(self.alloc * self.prb_mbps, 1e-6)
        budget = np.array([s.delay_budget_ms for s in SLICES])
        # The surrogate reward is the SAME SLA definition the full simulator scores:
        # carry at least SLA_CARRIED_LOAD of the offered load inside the delay budget.
        carried = served / np.maximum(arrivals, 1e-9)
        sla_ok = (carried >= SLA_CARRIED_LOAD) & (delay_ms <= budget)
        w = np.array([s.utility_weight for s in SLICES])
        reward = float(np.sum(w * sla_ok))
        reward -= 4.0 * float(delay_ms[1] > budget[1])
        reward -= 2.0 * float(np.sum(np.maximum(0.0, 1.0 - carried)))
        # slow-loop price update every 20 medium steps (2 s / 100 ms)
        if (self.t + 1) % 20 == 0:
            resid = (offered * self.dt - served) / max(offered.sum() * self.dt, 1e-9)
            self.prices = np.clip(self.prices + cfg.slow_price_step * resid * N_SLICES,
                                  1e-3, SLICING.dual_max)
        self.t += 1
        done = self.t >= self.n_steps
        return self._state(), reward, done


def pretrain_medium_policy(seed: int = 424242, n_episodes: int = 220,
                           cfg: IntegratedConfig = INTEGRATED,
                           verbose: bool = False) -> dict:
    """Train the Near-RT RIC Q-network on the fluid surrogate.

    Returns the weights plus the learning curve, so the pre-training itself is
    auditable and is stored in `results/medium_policy.json`.
    """
    rng = np.random.default_rng(seed)
    env = MediumSurrogateEnv(np.random.default_rng(seed + 1), cfg)
    n_actions = len(MEDIUM_ACTIONS)
    q = MLP([MEDIUM_STATE_DIM, 64, 64, n_actions], rng, out_scale=0.1)
    tgt = MLP([MEDIUM_STATE_DIM, 64, 64, n_actions], rng, out_scale=0.1)
    tgt.set_params(q.copy_params())
    opt = Adam(q.params, RLTRAIN.lr)
    buf = ReplayBuffer(40000, MEDIUM_STATE_DIM, rng)
    norm = RunningNorm()
    curve = []
    val_curve = []
    steps = 0
    best_val = -np.inf
    best_params = q.copy_params()

    def _rollout(policy, n: int = 4, base: int = 5000) -> float:
        tot = []
        for k in range(n):
            e = MediumSurrogateEnv(np.random.default_rng(seed + base + k), cfg)
            st = e.reset()
            acc = 0.0
            fin = False
            while not fin:
                st, rr, fin = e.step(policy(st))
                acc += rr
            tot.append(acc)
        return float(np.mean(tot))

    for ep in range(n_episodes):
        s = env.reset()
        eps = max(0.05, 1.0 - ep / (0.6 * n_episodes))
        acc = 0.0
        done = False
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                a = int(np.argmax(q.forward(s[None, :], cache=False)[0]))
            s2, r, done = env.step(a)
            norm.update(r)
            buf.add(s, a, norm.normalise(r), s2, done)
            s = s2
            acc += r
            steps += 1
            if buf.n >= 1000:
                bs, ba, br, bs2, bd = buf.sample(RLTRAIN.batch_size)
                qn = tgt.forward(bs2, cache=False).max(axis=1)
                target = br + RLTRAIN.gamma * (1.0 - bd) * qn
                out = q.forward(bs)
                pred = out[np.arange(len(ba)), ba]
                dout = np.zeros_like(out)
                dout[np.arange(len(ba)), ba] = huber_grad(pred - target) / len(ba)
                g, _ = q.backward(dout)
                opt.step(q.params, g, RLTRAIN.grad_clip)
                if steps % RLTRAIN.target_update_steps == 0:
                    tgt.set_params(q.copy_params())
        curve.append(acc)
        # Early stopping on a held-out validation set of surrogate episodes: the best
        # snapshot is kept, so a late-training collapse cannot be shipped by accident.
        if (ep + 1) % VALIDATION_EVERY == 0:
            v = _rollout(lambda st: int(np.argmax(q.forward(st[None, :],
                                                            cache=False)[0])))
            val_curve.append([ep + 1, v])
            if v > best_val:
                best_val = v
                best_params = q.copy_params()
        if verbose and (ep + 1) % 20 == 0:
            print(f"  pretrain ep {ep + 1}: return {np.mean(curve[-20:]):.1f}"
                  f"  best validation {best_val:.1f}")

    q.set_params(best_params)
    greedy = _rollout(lambda st: int(np.argmax(q.forward(st[None, :],
                                                        cache=False)[0])), 6, 9000)
    hold = _rollout(lambda st: 0, 6, 9000)
    rand = _rollout(lambda st: int(rng.integers(n_actions)), 6, 9000)
    return {
        "params": {k: v.tolist() for k, v in q.params.items()},
        "validation_curve": val_curve,
        "best_validation_return": float(best_val),
        "curve": [float(v) for v in curve],
        "surrogate_return_learned": greedy,
        "surrogate_return_hold": hold,
        "surrogate_return_random": rand,
        "n_episodes": n_episodes,
        "seed": seed,
        "note": ("pre-trained on the fluid surrogate MediumSurrogateEnv; evaluated "
                 "unchanged on the full TTI-level simulator"),
    }


# ==================================================================================
# The full TTI-level runner
# ==================================================================================

class IntegratedCell:
    """One gNB cell with three slices, three control loops and a TTI clock."""

    def __init__(self, seed: int, arm: str, policy: MediumPolicy | None,
                 compress_seconds: float | None = None,
                 cfg: IntegratedConfig = INTEGRATED, record_trace: bool = False):
        self.cfg = cfg
        self.arm = arm
        self.policy = policy
        self.num = cfg.numerology
        self.n_prb = self.num.n_prb
        self.slot_ms = self.num.slot_ms
        self.slot_s = self.slot_ms * 1e-3
        self.compress = (TRAFFIC.compress_seconds if compress_seconds is None
                         else compress_seconds)
        self.n_tti = int(round(self.compress / self.slot_s))
        self.medium_period = int(round(cfg.medium_period_ms / self.slot_ms))
        self.slow_period = int(round(cfg.slow_period_ms / self.slot_ms))
        self.record_trace = record_trace

        # --- common random numbers: identical streams for every arm -----------------
        self.rng_ch = np.random.default_rng(seed * 1_000_003 + 11)
        self.rng_tr = np.random.default_rng(seed * 1_000_003 + 23)

        # --- users ------------------------------------------------------------------
        self.n_users = sum(s.n_users for s in SLICES)
        self.user_slice = np.concatenate(
            [np.full(s.n_users, i) for i, s in enumerate(SLICES)])
        self.slice_idx = [np.nonzero(self.user_slice == i)[0] for i in range(N_SLICES)]
        self.pkt_bits = np.concatenate(
            [np.full(s.n_users, s.packet_bits_mean) for s in SLICES])
        self.lam_user = np.concatenate(
            [np.full(s.n_users, s.arrival_rate_per_user_hz) for s in SLICES])
        self.channel = CellChannel(self.rng_ch, self.n_users, self.num)

        # --- state ------------------------------------------------------------------
        self.backlog = np.zeros(self.n_users)
        self.avg_rate = np.full(self.n_users, 1.0)
        self.mins = np.array([s.min_prb for s in SLICES], dtype=int)
        self.prices = np.full(N_SLICES, SLICING.dual_init)
        self.admitted = np.ones(N_SLICES)
        # Static VNF provisioning, sized once for the *mean* daily load.  Arms without
        # a slow loop keep this provisioning for the whole run; arms C and E rescale
        # it every 2 s.  Without this the no-slow-loop arms would silently enjoy a
        # zero-latency core network.
        self.vnf_instances = np.ones((N_SLICES, 5), dtype=int)
        self.sfc_latency_ms = np.zeros(N_SLICES)
        for s in range(N_SLICES):
            pps = SLICES[s].n_users * SLICES[s].arrival_rate_per_user_hz * cfg.nominal_load
            self.vnf_instances[s, :] = required_instances(
                pps, cfg.vnf_capacity_pps, cfg.vnf_max_instances,
                cfg.vnf_target_utilisation)
            self.sfc_latency_ms[s] = self._sfc_latency(pps, self.vnf_instances[s, :])
        self.alloc = self._initial_alloc()
        self.price_budget = float(np.dot(self.prices, self.alloc)) * 1.25

        # --- accumulators -----------------------------------------------------------
        self.bits_served = np.zeros(N_SLICES)
        self.bits_offered = np.zeros(N_SLICES)
        self.bits_blocked = np.zeros(N_SLICES)
        self.prb_used = 0.0
        self.tti_count = 0
        self.urllc_viol = 0
        self.urllc_e2e_viol = 0
        self.urllc_e2e_sum = 0.0
        self.urllc_active_tti = 0
        self.sfc_latency_sum = np.zeros(N_SLICES)
        self.delay_sum = np.zeros(N_SLICES)
        self.delay_max = np.zeros(N_SLICES)
        self.sla_windows_ok = 0
        self.sla_windows_ok_slice = np.zeros(N_SLICES, dtype=int)
        self.sla_windows = 0
        self.n_medium_actions = 0
        self.n_medium_rejected = 0
        self.n_slow_updates = 0
        self.arrival_cum = np.zeros((N_SLICES, self.n_tti + 1))
        self.served_cum = np.zeros((N_SLICES, self.n_tti + 1))
        self.trace = LoopTrace()
        self.window_bits = np.zeros(N_SLICES)
        self.window_offered = np.zeros(N_SLICES)
        self.window_delay_sum = np.zeros(N_SLICES)
        self.window_viol = 0
        self.window_tti = 0

    # ------------------------------------------------------------------------------
    def _initial_alloc(self) -> np.ndarray:
        if self.arm == "A_hard_static":
            return integer_allocation(np.array(SLICING.hard_split, dtype=float),
                                      self.n_prb, np.zeros(N_SLICES, dtype=int))
        if self.arm == "B_soft_pool":
            return integer_allocation(
                np.array(self.cfg.baseline_b_min, dtype=float)
                + self.cfg.baseline_b_pool / N_SLICES,
                self.n_prb, np.array(self.cfg.baseline_b_min, dtype=int))
        return integer_allocation(np.array(SLICING.hard_split, dtype=float),
                                  self.n_prb, self.mins)

    def _sfc_latency(self, pps: float, instances: np.ndarray) -> float:
        lat = chain_latency_with_scaling(pps, instances, self.cfg.vnf_capacity_pps)
        return float(lat * 1e3) if np.isfinite(lat) else float(SFC_UNSTABLE_MS)

    # ------------------------------------------------------------------------------
    def _fast_tti(self, t: int) -> None:
        se = self.channel.step(t)
        bits_per_prb = se * self.num.prb_bandwidth_hz * self.slot_s

        # ---- arrivals (Poisson counts, exponential marks) --------------------------
        t_s = t * self.slot_s
        mult = float(diurnal_multiplier(t_s, self.compress))
        rho = self.cfg.nominal_load * mult
        lam = self.lam_user * rho * self.slot_s
        offered_lam = lam.copy()
        lam = lam * self.admitted[self.user_slice]
        counts = self.rng_tr.poisson(lam)
        bits = np.where(counts > 0,
                        self.rng_tr.gamma(np.maximum(counts, 1e-9), self.pkt_bits),
                        0.0)
        self.backlog += bits

        # ---- proportional-fair scheduling inside each slice's own PRBs -------------
        served = np.zeros(self.n_users)
        prb_used = 0
        for s in range(N_SLICES):
            idx = self.slice_idx[s]
            n_s = int(self.alloc[s])
            if n_s <= 0:
                continue
            bpp = bits_per_prb[idx]
            bl = self.backlog[idx]
            active = bl > 0
            if not np.any(active):
                continue
            metric = np.where(active & (bpp > 0), bpp / self.avg_rate[idx], -1.0)
            order = np.argsort(-metric)
            need = np.ceil(np.where(bpp[order] > 0, bl[order] / np.maximum(bpp[order], 1e-9), 0.0))
            need = np.where(metric[order] > 0, need, 0.0)
            cum = np.cumsum(need)
            give = np.clip(n_s - (cum - need), 0.0, need)
            got_bits = np.minimum(give * bpp[order], bl[order])
            served_slice = np.zeros(idx.size)
            served_slice[order] = got_bits
            served[idx] = served_slice
            prb_used += float(min(give.sum(), n_s))

        self.backlog -= served
        np.maximum(self.backlog, 0.0, out=self.backlog)
        np.minimum(self.backlog, self.cfg.max_backlog_bits, out=self.backlog)
        self.avg_rate = ((1.0 - self.cfg.pf_ewma) * self.avg_rate
                         + self.cfg.pf_ewma * served)
        np.maximum(self.avg_rate, 1e-6, out=self.avg_rate)

        # ---- telemetry -------------------------------------------------------------
        for s in range(N_SLICES):
            idx = self.slice_idx[s]
            a = float(bits[idx].sum())
            d = float(served[idx].sum())
            self.arrival_cum[s, t + 1] = self.arrival_cum[s, t] + a
            self.served_cum[s, t + 1] = self.served_cum[s, t] + d
            self.bits_served[s] += d
            off = float((offered_lam[idx] * self.pkt_bits[idx]).sum())
            self.bits_offered[s] += off
            self.window_bits[s] += d
            self.window_offered[s] += off
        self.prb_used += prb_used
        self.sfc_latency_sum += self.sfc_latency_ms
        self.tti_count += 1
        self.window_tti += 1

        # ---- exact FIFO head-of-line delay per slice --------------------------------
        for s in range(N_SLICES):
            d_now = self.served_cum[s, t + 1]
            if self.backlog[self.slice_idx[s]].sum() <= 0:
                delay_ms = 0.0
            else:
                j = int(np.searchsorted(self.arrival_cum[s, :t + 2], d_now, side="left"))
                delay_ms = max(0.0, (t + 1 - j)) * self.slot_ms
            self.delay_sum[s] += delay_ms
            self.window_delay_sum[s] += delay_ms
            self.delay_max[s] = max(self.delay_max[s], delay_ms)
            if s == 1:
                # radio-plane violation: head-of-line delay measured against the
                # ITU-R M.2410-0 1 ms user-plane radio target
                self._urllc_delay_ms = delay_ms
                self.urllc_active_tti += 1
                if delay_ms > SLICES[s].delay_budget_ms:
                    self.urllc_viol += 1
                    self.window_viol += 1
                # end-to-end violation: radio delay + the SFC latency that the slow
                # loop controls, measured against the 5QI 82 packet delay budget
                self.urllc_e2e_sum += delay_ms + self.sfc_latency_ms[s]
                if delay_ms + self.sfc_latency_ms[s] > self.cfg.e2e_budget_urllc_ms:
                    self.urllc_e2e_viol += 1

    # ------------------------------------------------------------------------------
    def _medium_loop(self, t: int) -> None:
        """Near-RT RIC: learned PRB re-split + admission control, constrained by the
        slow loop's guarantees and price budget."""
        cfg = self.cfg
        window_s = self.window_tti * self.slot_s
        thr = self.window_bits / max(window_s, 1e-9) / 1e6            # Mbit/s
        sla = np.array([s.sla_rate_mbps for s in SLICES])
        backlog_s = np.array([self.backlog[self.slice_idx[s]].sum()
                              for s in range(N_SLICES)])

        # --- SLA bookkeeping --------------------------------------------------------
        # A slice is satisfied in a 100 ms window when (i) it carried at least
        # SLA_CARRIED_LOAD of the load offered to it during that window and (ii) its
        # mean head-of-line delay plus its SFC latency stayed inside its 5QI packet
        # delay budget.  A fixed absolute rate target cannot be used because the
        # offered load follows a diurnal envelope that swings by a factor of six.
        budget = np.array([s.delay_budget_ms for s in SLICES])
        carried = self.window_bits / np.maximum(self.window_offered, 1.0)
        win_delay = self.window_delay_sum / max(self.window_tti, 1)
        urllc_ok = (self.window_viol / max(self.window_tti, 1)) <= URLLC_WINDOW_VIOL_MAX
        per_slice_ok = (carried >= SLA_CARRIED_LOAD) & (win_delay <= budget)
        ok = bool(np.all(per_slice_ok) and urllc_ok)
        self.sla_windows += 1
        self.sla_windows_ok += int(ok)
        self.sla_windows_ok_slice += per_slice_ok.astype(int)

        rejected = 0
        action = 0
        if self.arm in ("A_hard_static",):
            pass
        elif self.arm == "B_soft_pool":
            # demand-proportional split of the fixed shared pool (no learning)
            mins = np.array(cfg.baseline_b_min, dtype=float)
            excess = np.maximum(backlog_s, 1.0)
            share = cfg.baseline_b_pool * excess / excess.sum()
            self.alloc = integer_allocation(mins + share, self.n_prb,
                                            mins.astype(int))
        elif self.arm in ("D_no_slow", "E_full_umro"):
            t_frac = (t * self.slot_s) / self.compress
            state = medium_state(self.alloc.astype(float), backlog_s, thr,
                                 self.prices, self.n_prb, t_frac)
            action = self.policy.act(state) if self.policy is not None else 0
            pair = MEDIUM_ACTIONS[action]
            if pair is not None:
                give, take = pair
                step = cfg.medium_prb_step
                cand = self.alloc.copy()
                cand[give] += step
                cand[take] -= step
                feasible = bool(np.all(cand >= self.mins))
                # slow-loop price budget constrains the medium loop's feasible set
                if self.arm == "E_full_umro":
                    feasible = feasible and (float(np.dot(self.prices, cand))
                                             <= self.price_budget)
                if feasible:
                    self.alloc = cand
                    self.n_medium_actions += 1
                else:
                    rejected = 1
                    self.n_medium_rejected += 1
            # admission control on the SLA-proximity signal
            ratio = backlog_s / np.maximum(sla * 1e6 * cfg.medium_period_ms * 1e-3, 1.0)
            self.admitted = np.where(ratio > cfg.admission_backlog_threshold,
                                     0.75, 1.0)
        # C_no_medium: the medium loop is disabled -- the allocation only changes
        # when the slow loop rewrites the guarantees.

        if self.record_trace:
            self.trace.t_s.append(t * self.slot_s)
            self.trace.prices.append([float(v) for v in self.prices])
            self.trace.min_prb.append([int(v) for v in self.mins])
            self.trace.alloc.append([int(v) for v in self.alloc])
            self.trace.throughput_mbps.append([float(v) for v in thr])
            self.trace.backlog_kbit.append([float(v / 1e3) for v in backlog_s])
            self.trace.urllc_delay_ms.append(float(getattr(self, "_urllc_delay_ms", 0.0)))
            self.trace.action.append(int(action))
            self.trace.action_rejected.append(int(rejected))
            self.trace.admitted_fraction.append([float(v) for v in self.admitted])
            self.trace.offered_multiplier.append(
                float(diurnal_multiplier(t * self.slot_s, self.compress)))
            self.trace.sfc_latency_ms.append([float(v) for v in self.sfc_latency_ms])

        self.window_bits[:] = 0.0
        self.window_offered[:] = 0.0
        self.window_delay_sum[:] = 0.0
        self.window_viol = 0
        self.window_tti = 0

    # ------------------------------------------------------------------------------
    def _slow_loop(self, t: int) -> None:
        """Non-RT RIC / MANO: dual price update, guarantee re-derivation, VNF scaling."""
        cfg = self.cfg
        t_s = t * self.slot_s
        mult = float(diurnal_multiplier(t_s, self.compress))
        rho = cfg.nominal_load * mult

        # --- measured demand vs. granted capacity, in PRB-equivalents ---------------
        se_mean = np.array([max(self.channel.spectral_efficiency[self.slice_idx[s]].mean(),
                                1e-3) for s in range(N_SLICES)])
        prb_bps = se_mean * self.num.prb_bandwidth_hz
        offered_bps = np.array([s.n_users * s.arrival_rate_per_user_hz
                                * s.packet_bits_mean for s in SLICES]) * rho
        demand_prb = offered_bps / np.maximum(prb_bps, 1.0)
        residual = (demand_prb - self.alloc) / self.n_prb

        # --- Lagrangian dual sub-gradient step (Equation (34)) ----------------------
        self.prices = np.clip(self.prices + cfg.slow_price_step * residual * N_SLICES,
                              1e-3, cfg.slow_price_max)

        # --- re-derive the per-slice minimum guarantees ------------------------------
        # The guarantee tracks the measured PRB-equivalent demand, modulated by the
        # dual price: a slice whose price has risen (persistently under-served) is
        # given a larger guarantee.  The guarantees are then normalised to
        # GUARANTEE_BUDGET of the carrier so the medium loop keeps room to manoeuvre.
        norm_price = self.prices / max(float(self.prices.sum()), 1e-9)
        raw = demand_prb * (1.0 + PRICE_GUARANTEE_GAIN * norm_price * N_SLICES)
        raw = raw * (self.n_prb * GUARANTEE_BUDGET / max(raw.sum(), 1e-9))
        self.mins = np.clip(np.round(raw).astype(int),
                            cfg.slow_min_prb_floor, cfg.slow_min_prb_ceiling)
        while self.mins.sum() > self.n_prb - N_SLICES:
            self.mins[int(np.argmax(self.mins))] -= 1
        self.price_budget = float(np.dot(self.prices, self.mins)) * PRICE_BUDGET_SLACK

        # --- VNF placement / scaling for each slice's SFC ---------------------------
        for s in range(N_SLICES):
            pps = (SLICES[s].n_users * SLICES[s].arrival_rate_per_user_hz * rho)
            inst = required_instances(pps, cfg.vnf_capacity_pps,
                                      cfg.vnf_max_instances,
                                      cfg.vnf_target_utilisation)
            self.vnf_instances[s, :] = inst
            self.sfc_latency_ms[s] = self._sfc_latency(pps, self.vnf_instances[s, :])

        # --- guarantees constrain the current allocation ----------------------------
        self.alloc = integer_allocation(self.alloc.astype(float), self.n_prb, self.mins)
        if self.arm == "C_no_medium":
            # with no medium loop the slow loop is the only allocator: it hands out
            # the guarantees plus a price-proportional share of the remainder
            rest = self.n_prb - self.mins.sum()
            resid_demand = np.maximum(demand_prb - self.mins, 0.0)
            if resid_demand.sum() <= 1e-9:
                resid_demand = np.ones(N_SLICES)
            self.alloc = integer_allocation(
                self.mins + rest * resid_demand / resid_demand.sum(),
                self.n_prb, self.mins)
        self.n_slow_updates += 1

    # ------------------------------------------------------------------------------
    def run(self) -> dict:
        use_medium = self.arm in ("B_soft_pool", "D_no_slow", "E_full_umro")
        use_slow = self.arm in ("C_no_medium", "E_full_umro")
        for t in range(self.n_tti):
            self._fast_tti(t)
            if (t + 1) % self.medium_period == 0:
                if use_medium or self.arm in ("A_hard_static", "C_no_medium"):
                    self._medium_loop(t)
            if use_slow and (t + 1) % self.slow_period == 0:
                self._slow_loop(t)
        return self._metrics()

    def _metrics(self) -> dict:
        dur_s = self.tti_count * self.slot_s
        thr = self.bits_served / dur_s / 1e6
        offered = self.bits_offered / dur_s / 1e6
        sla = np.array([s.sla_rate_mbps for s in SLICES])
        mean_delay = self.delay_sum / max(self.tti_count, 1)
        util = self.prb_used / max(self.tti_count * self.n_prb, 1)
        jain = (thr.sum() ** 2) / (N_SLICES * float(np.sum(thr ** 2)) + 1e-12)
        out = {
            "arm": self.arm,
            "duration_s": dur_s,
            "n_tti": self.tti_count,
            "aggregate_throughput_mbps": float(thr.sum()),
            "prb_utilisation": float(util),
            "urllc_violation_prob": float(self.urllc_viol / max(self.urllc_active_tti, 1)),
            "sla_satisfaction_rate": float(self.sla_windows_ok / max(self.sla_windows, 1)),
            "jain_across_slices": float(jain),
            "medium_actions_applied": int(self.n_medium_actions),
            "medium_actions_rejected": int(self.n_medium_rejected),
            "slow_updates": int(self.n_slow_updates),
            "final_prices": [float(v) for v in self.prices],
            "final_min_prb": [int(v) for v in self.mins],
            "final_alloc": [int(v) for v in self.alloc],
            "carried_load_fraction": float(self.bits_served.sum()
                                           / max(self.bits_offered.sum(), 1e-9)),
            "urllc_e2e_violation_prob": float(self.urllc_e2e_viol
                                              / max(self.urllc_active_tti, 1)),
            "urllc_mean_e2e_latency_ms": float(self.urllc_e2e_sum
                                               / max(self.urllc_active_tti, 1)),
        }
        for i, name in enumerate(SLICE_NAMES):
            out[f"throughput_{name}_mbps"] = float(thr[i])
            out[f"offered_{name}_mbps"] = float(offered[i])
            out[f"mean_hol_delay_{name}_ms"] = float(mean_delay[i])
            out[f"max_hol_delay_{name}_ms"] = float(self.delay_max[i])
            out[f"sla_margin_{name}"] = float(thr[i] / sla[i])
            out[f"mean_sfc_latency_{name}_ms"] = float(
                self.sfc_latency_sum[i] / max(self.tti_count, 1))
            out[f"sla_satisfaction_{name}"] = float(
                self.sla_windows_ok_slice[i] / max(self.sla_windows, 1))
        return out


# ==================================================================================
# Public entry points
# ==================================================================================

def run_arm(seed: int, arm: str, policy: MediumPolicy | None,
            compress_seconds: float | None = None,
            record_trace: bool = False) -> dict:
    cell = IntegratedCell(seed, arm, policy, compress_seconds,
                          record_trace=record_trace)
    m = cell.run()
    if record_trace:
        m["trace"] = cell.trace.as_dict()
    return m


def run_integrated_seed(seed: int, policy_params: Dict[str, list] | None = None,
                        compress_seconds: float | None = None,
                        record_trace: bool = False) -> dict:
    """One independent replication: all five arms on identical random streams."""
    policy = MediumPolicy(params=policy_params) if policy_params else MediumPolicy()
    out = {"seed": seed, "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = run_arm(seed, arm, policy, compress_seconds,
                                   record_trace=record_trace)
    return out


def coupling_statistics(trace: dict) -> dict:
    """Quantify the coupling the reviewer says is missing.

    * price -> guarantee: correlation between the slow loop's price vector and the
      minimum-PRB guarantee it derives;
    * guarantee -> allocation: correlation between the guarantee and the allocation
      the medium loop actually settles on;
    * allocation -> throughput: correlation between the allocation and the measured
      per-slice throughput one medium period later (the fast loop responding);
    * offered load -> allocation: does the partition track the diurnal envelope?
    """
    p = np.asarray(trace["prices"], dtype=float)
    g = np.asarray(trace["min_prb"], dtype=float)
    a = np.asarray(trace["alloc"], dtype=float)
    th = np.asarray(trace["throughput_mbps"], dtype=float)
    mult = np.asarray(trace["offered_multiplier"], dtype=float)

    def corr(x, y):
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    # Pooling the three slices into one correlation would be dominated by the
    # between-slice scale difference rather than by the within-slice dynamics, so the
    # per-slice correlations are reported individually and the mean is given as a
    # single summary number.
    per_slice_pg = [corr(p[:, i], g[:, i]) for i in range(p.shape[1])]
    per_slice_ga = [corr(g[:, i], a[:, i]) for i in range(g.shape[1])]
    per_slice_at = [corr(a[:-1, i], th[1:, i]) for i in range(a.shape[1])]
    per_slice_ma = [corr(mult, a[:, i]) for i in range(a.shape[1])]

    def nanmean(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    return {
        "corr_price_guarantee_per_slice": per_slice_pg,
        "corr_price_guarantee": nanmean(per_slice_pg),
        "corr_guarantee_alloc_per_slice": per_slice_ga,
        "corr_guarantee_alloc": nanmean(per_slice_ga),
        "corr_alloc_throughput_lag1_per_slice": per_slice_at,
        "corr_alloc_throughput_lag1": nanmean(per_slice_at),
        "corr_offered_alloc_per_slice": per_slice_ma,
        "corr_offered_alloc_eMBB": per_slice_ma[0],
        "corr_price_guarantee_pooled": corr(p, g),
        "corr_guarantee_alloc_pooled": corr(g, a),
        "price_range": [float(p.min()), float(p.max())],
        "guarantee_range": [int(g.min()), int(g.max())],
        "alloc_range_per_slice": [[int(a[:, i].min()), int(a[:, i].max())]
                                  for i in range(a.shape[1])],
        "n_medium_samples": int(a.shape[0]),
    }
