"""
scheduler.py -- MAC schedulers evaluated in the article's Section 8.6.

All four schedulers consume the *same* pre-generated spectral-efficiency trace for a
given seed (common random numbers), so the comparison is paired at the TTI level as
well as at the seed level.

  round_robin      cyclic PRB assignment; perfect long-run fairness.
  max_rate         greedy argmax of instantaneous rate.
  proportional_fair  argmax of r_k(t) / R_k(t) with an EWMA of the delivered rate.
  learned          a genuinely trained policy: a 2-layer MLP scores each user from
                   (normalised instantaneous SE, normalised average rate, backlog) and
                   PRBs go to the top-scoring users.  Trained online by REINFORCE with
                   a moving-average baseline and an entropy bonus.  It is a real
                   policy-gradient learner, not a hand-tuned heuristic -- but it is a
                   *policy-gradient* scheduler, not a DQN, and the paper must say so.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from config import SCHED, SchedulerConfig, NumerologyConfig, NUM_MU1_SCHED
from channel import rayleigh_se_matrix
from config import CHANNEL


# ----------------------------------------------------------------------------------
# Learned scheduler
# ----------------------------------------------------------------------------------

class PolicyGradientScheduler:
    """Softmax policy over per-user scores produced by a small MLP.

    Features per user: [se_norm, avgrate_norm, log1p(avgrate), se/avgrate ratio].
    The action is a sample of `n_prb` users without replacement (a Plackett-Luce
    top-k draw).  The reward is the increment in the sum-log-rate network utility,
    which is the objective proportional fairness maximises -- so the learner is given
    the *objective*, not the *rule*, and must discover an allocation policy itself.
    """

    N_FEATURES = 4

    def __init__(self, rng: np.random.Generator, cfg: SchedulerConfig = SCHED):
        self.rng = rng
        self.cfg = cfg
        h = cfg.hidden_units
        s = 0.6
        self.W1 = rng.normal(0.0, s / np.sqrt(self.N_FEATURES), (self.N_FEATURES, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0.0, s / np.sqrt(h), (h, 1))
        self.b2 = np.zeros(1)
        self.baseline = 0.0
        self._cache = None

    def _features(self, se: np.ndarray, avg: np.ndarray) -> np.ndarray:
        se_n = se / (se.mean() + 1e-9)
        av_n = avg / (avg.mean() + 1e-9)
        return np.stack([se_n, av_n, np.log1p(avg), se_n / (av_n + 1e-6)], axis=1)

    def scores(self, se: np.ndarray, avg: np.ndarray) -> np.ndarray:
        x = self._features(se, avg)
        z1 = x @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = (a1 @ self.W2 + self.b2).ravel()
        self._cache = (x, z1, a1, z2)
        return z2

    def select(self, se: np.ndarray, avg: np.ndarray, n_prb: int,
               explore: bool) -> np.ndarray:
        z = self.scores(se, avg)
        z = z - z.max()
        if not explore:
            return np.argsort(-z)[:n_prb]
        # Gumbel top-k == sampling without replacement from the softmax
        g = self.rng.gumbel(size=z.shape)
        return np.argsort(-(z + g))[:n_prb]

    def update(self, chosen: np.ndarray, reward: float) -> None:
        x, z1, a1, z2 = self._cache
        z = z2 - z2.max()
        p = np.exp(z)
        p /= p.sum()
        onehot = np.zeros_like(p)
        onehot[chosen] = 1.0 / max(len(chosen), 1)
        self.baseline = 0.99 * self.baseline + 0.01 * reward
        adv = reward - self.baseline
        # d/dz of log pi  (softmax over users, averaged over the chosen set)
        dz = adv * (onehot - p)
        # entropy bonus keeps the policy from collapsing onto one user
        dz += self.cfg.entropy_coeff * (-p * (np.log(p + 1e-12) + 1.0))
        dz = dz.reshape(-1, 1)
        gW2 = a1.T @ dz
        gb2 = dz.sum(axis=0)
        da1 = dz @ self.W2.T
        dz1 = da1 * (1.0 - a1 ** 2)
        gW1 = x.T @ dz1
        gb1 = dz1.sum(axis=0)
        lr = self.cfg.learn_rate
        for p_, g_ in ((self.W1, gW1), (self.b1, gb1), (self.W2, gW2), (self.b2, gb2)):
            np.clip(g_, -5.0, 5.0, out=g_)
            p_ += lr * g_


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------

def jains_fairness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    denom = n * np.sum(x ** 2)
    if denom <= 0:
        return 1.0
    return float((x.sum() ** 2) / denom)


# ----------------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------------

ALGORITHMS = ("round_robin", "max_rate", "proportional_fair", "learned")


def run_scheduling_seed(seed: int, n_tti: int | None = None,
                        numerology: NumerologyConfig = NUM_MU1_SCHED,
                        cfg: SchedulerConfig = SCHED) -> dict:
    """One independent replication.  Returns per-algorithm aggregate metrics."""
    n_tti = cfg.n_tti if n_tti is None else n_tti
    rng = np.random.default_rng(seed)
    n_u = cfg.n_users
    n_prb = numerology.n_prb
    prb_bw = numerology.prb_bandwidth_hz
    slot_s = numerology.slot_ms * 1e-3

    mean_snr_db = rng.uniform(cfg.snr_db_low, cfg.snr_db_high, n_u)
    coherence_tti = max(1, int(round(CHANNEL.coherence_ms / numerology.slot_ms)))
    se_trace = rayleigh_se_matrix(rng, mean_snr_db, n_tti, coherence_tti)

    results: Dict[str, dict] = {}
    for algo in ALGORITHMS:
        delivered = np.zeros(n_u)              # cumulative bits
        avg = np.full(n_u, 1e-3)               # EWMA of delivered rate (bit/TTI)
        policy = (PolicyGradientScheduler(np.random.default_rng(seed + 777), cfg)
                  if algo == "learned" else None)
        prev_util = float(np.sum(np.log(avg)))
        rr_ptr = 0
        measured_from = cfg.warmup_tti if algo == "learned" else 0
        measured_bits = np.zeros(n_u)

        for t in range(n_tti):
            se = se_trace[t]
            bits_per_prb = se * prb_bw * slot_s
            if algo == "round_robin":
                chosen = (rr_ptr + np.arange(n_prb)) % n_u
                rr_ptr = (rr_ptr + n_prb) % n_u
            elif algo == "max_rate":
                chosen = np.argsort(-bits_per_prb)[:n_prb]
            elif algo == "proportional_fair":
                metric = bits_per_prb / avg
                chosen = np.argsort(-metric)[:n_prb]
            else:
                chosen = policy.select(bits_per_prb, avg, n_prb,
                                       explore=t < n_tti - cfg.warmup_tti)

            got = np.zeros(n_u)
            np.add.at(got, chosen, bits_per_prb[chosen])
            delivered += got
            if t >= measured_from:
                measured_bits += got
            avg = (1.0 - cfg.pf_ewma) * avg + cfg.pf_ewma * got
            if policy is not None:
                util = float(np.sum(np.log(np.maximum(avg, 1e-9))))
                policy.update(np.unique(chosen), util - prev_util)
                prev_util = util

        n_meas = max(n_tti - measured_from, 1)
        rate_mbps = measured_bits / (n_meas * slot_s) / 1e6
        order = np.argsort(rate_mbps)
        n_edge = max(1, int(round(cfg.cell_edge_fraction * n_u)))
        results[algo] = {
            "throughput_mbps": float(rate_mbps.sum()),
            "jain": jains_fairness(rate_mbps),
            "p5_mbps": float(np.percentile(rate_mbps, cfg.cell_edge_percentile)),
            "cell_edge_mbps": float(rate_mbps[order[:n_edge]].mean()),
            "mean_user_mbps": float(rate_mbps.mean()),
            "starved_users": int(np.sum(rate_mbps < 0.01)),
        }
    return {"seed": seed, "algorithms": results}
