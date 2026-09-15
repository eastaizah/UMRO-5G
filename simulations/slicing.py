"""
slicing.py -- PRB allocation across slices.

Three allocators, matching the three arms of the article's Section 8.4:

  hard_isolation   fixed partition, no sharing.
  soft_isolation   per-slice minimum guarantee plus a shared pool split in proportion
                   to instantaneous demand (Equation (10) of the article).
  lagrangian_dual  UMRO-5G: dual decomposition of
                       max  sum_s w_s * U_s(n_s)   s.t.  sum_s n_s <= N,  n_s >= n_s^min
                   with a projected sub-gradient update of the single price
                   multiplier (Equation (34) of the article), and utility weights
                   w_s re-scaled every timeslot by the slice's SLA proximity.

`U_s(n) = log(1 + a_s n)` with `a_s` the per-PRB rate the slice currently sees, so the
stationary point of the Lagrangian is available in closed form:
    dU/dn = w_s a_s / (1 + a_s n) = lambda   ->   n_s(lambda) = w_s/lambda - 1/a_s
which is exactly per-slice water-filling against the price `lambda`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import SLICES, SLICING, SliceSpec


@dataclass
class DualState:
    """Persistent state of the Lagrangian allocator (survives across timeslots)."""

    price: float = SLICING.dual_init
    step: float = SLICING.dual_step
    n_updates: int = 0
    residual: float = 0.0
    converged: bool = False


def hard_isolation(n_prb: int, split: Sequence[int] | None = None) -> np.ndarray:
    """Static partition.  Ignores demand entirely."""
    split = SLICING.hard_split if split is None else split
    alloc = np.asarray(split, dtype=np.float64)
    total = alloc.sum()
    if total > n_prb:
        alloc = alloc * (n_prb / total)
    return alloc


def soft_isolation(demand_prb: np.ndarray, n_prb: int,
                   minimums: Sequence[int] | None = None,
                   pool: int | None = None) -> np.ndarray:
    """Minimum guarantees + proportional share of a fixed pool (Equation (10))."""
    minimums = SLICING.soft_min if minimums is None else minimums
    mins = np.asarray(minimums, dtype=np.float64)
    pool_size = float(n_prb - mins.sum()) if pool is None else float(pool)
    pool_size = max(0.0, min(pool_size, n_prb - mins.sum()))

    excess = np.maximum(demand_prb - mins, 0.0)
    tot = excess.sum()
    if tot <= 1e-12:
        share = np.zeros_like(mins)
    else:
        share = pool_size * excess / tot
    return mins + share


def lagrangian_dual(demand_prb: np.ndarray, per_prb_rate: np.ndarray,
                    weights: np.ndarray, n_prb: int,
                    minimums: Sequence[int] | None = None,
                    state: DualState | None = None,
                    n_iterations: int | None = None) -> tuple[np.ndarray, DualState]:
    """Projected sub-gradient ascent on the dual of the PRB-allocation problem.

    Returns (allocation, dual state).  The dual state carries the price across
    timeslots, which is what makes this a *warm-started* dual method rather than an
    independent solve per slot.
    """
    minimums = SLICING.soft_min if minimums is None else minimums
    mins = np.asarray(minimums, dtype=np.float64)
    a = np.maximum(np.asarray(per_prb_rate, dtype=np.float64), 1e-9)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    dem = np.maximum(np.asarray(demand_prb, dtype=np.float64), 0.0)
    upper = np.maximum(dem, mins)

    st = DualState() if state is None else state
    n_it = SLICING.n_dual_iterations if n_iterations is None else n_iterations

    alloc = np.clip(np.zeros_like(mins) + mins, mins, upper)
    for _ in range(n_it):
        lam = max(st.price, 1e-6)
        # per-slice best response  n_s(lambda) = w_s/lambda - 1/a_s
        alloc = w / lam - 1.0 / a
        alloc = np.clip(alloc, mins, upper)
        residual = alloc.sum() - n_prb
        # exponentiated sub-gradient on the budget-normalised residual
        st.price = float(np.clip(st.price * np.exp(st.step * residual / n_prb),
                                 1e-6, SLICING.dual_max))
        st.step *= SLICING.dual_step_decay
        st.n_updates += 1
        st.residual = float(residual)
        if abs(residual) <= SLICING.dual_tolerance * n_prb:
            st.converged = True
            break
    else:
        st.converged = abs(st.residual) <= SLICING.dual_tolerance * n_prb

    # The dual solution can be infeasible by a small residual (the primal is an
    # integer programme relaxed to the box); project onto the simplex sum = n_prb.
    alloc = _project_to_budget(alloc, mins, upper, n_prb)
    return alloc, st


def _project_to_budget(alloc: np.ndarray, mins: np.ndarray, upper: np.ndarray,
                       n_prb: int) -> np.ndarray:
    """Scale the discretionary part of the allocation so the budget holds exactly."""
    alloc = np.clip(alloc, mins, upper)
    total = alloc.sum()
    if abs(total - n_prb) < 1e-9:
        return alloc
    if total > n_prb:
        free = alloc - mins
        surplus = total - n_prb
        f = free.sum()
        if f > 1e-12:
            alloc = alloc - free * min(1.0, surplus / f)
        # if still over budget the minimums alone exceed N: scale them down
        if alloc.sum() > n_prb + 1e-9:
            alloc = alloc * (n_prb / alloc.sum())
    else:
        head = upper - alloc
        deficit = n_prb - total
        h = head.sum()
        if h > 1e-12:
            alloc = alloc + head * min(1.0, deficit / h)
    return alloc


def sla_proximity_weights(base_weights: np.ndarray, achieved_mbps: np.ndarray,
                          sla_mbps: np.ndarray) -> np.ndarray:
    """w_s <- w_s * (1 + gain * relative SLA shortfall).  A slice comfortably above
    its SLA keeps its nominal weight; a slice below it bids harder for PRBs."""
    sla = np.maximum(np.asarray(sla_mbps, dtype=np.float64), 1e-9)
    gap = np.clip((sla - np.asarray(achieved_mbps, dtype=np.float64)) / sla, 0.0, 1.0)
    return np.asarray(base_weights, dtype=np.float64) * (
        1.0 + SLICING.sla_proximity_gain * gap)


# ----------------------------------------------------------------------------------
# Multi-slice Monte Carlo experiment (article Section 8.4)
# ----------------------------------------------------------------------------------

STRATEGIES = ("hard", "soft", "umro")


def _demand_and_rate(rng: np.random.Generator, rho: float, n_prb: int,
                     slot_ms: float, prb_bw_hz: float,
                     slices: Sequence[SliceSpec]) -> tuple[np.ndarray, np.ndarray]:
    """One Monte Carlo draw: instantaneous PRB demand and per-PRB rate per slice."""
    from channel import se_from_snr_db, path_loss_db, noise_power_dbm
    from config import CHANNEL

    per_prb_rate = np.zeros(len(slices))
    demand = np.zeros(len(slices))
    eirp = CHANNEL.tx_power_dbm + CHANNEL.bs_antenna_gain_db - 10 * np.log10(n_prb)
    n_dbm = noise_power_dbm(prb_bw_hz)
    for i, s in enumerate(slices):
        d = rng.uniform(CHANNEL.min_distance_m, CHANNEL.cell_radius_m, s.n_users)
        sh = rng.normal(0.0, CHANNEL.shadowing_sigma_db, s.n_users)
        fade_db = 10.0 * np.log10(np.maximum(rng.exponential(1.0, s.n_users), 1e-12))
        snr_db = eirp - path_loss_db(d) - sh + fade_db - n_dbm
        se = se_from_snr_db(snr_db)
        mean_se = float(se.mean())
        rate_bps = max(mean_se * prb_bw_hz, 1e3)
        per_prb_rate[i] = rate_bps / 1e6                     # Mbit/s per PRB
        offered_mbps = (s.n_users * s.arrival_rate_per_user_hz * rho
                        * s.packet_bits_mean) / 1e6
        # Poisson burstiness on the number of active users in this slot
        active = rng.poisson(max(s.n_users * rho, 1e-9))
        burst = active / max(s.n_users * rho, 1e-9) if s.n_users * rho > 0 else 1.0
        demand[i] = offered_mbps * burst / per_prb_rate[i]
    return demand, per_prb_rate


def run_multislice_seed(seed: int, n_prb: int, slot_ms: float, prb_bw_hz: float,
                        loads: Sequence[float], n_mc: int) -> dict:
    """One independent replication of the multi-slice Monte Carlo experiment.

    All `n_mc` iterations are averaged into a single per-load point estimate: this is
    the whole reason the independent sample size equals the number of seeds and NOT
    seeds x iterations.
    """
    slices = SLICES
    n_s = len(slices)
    sla = np.array([s.sla_rate_mbps for s in slices])
    base_w = np.array([s.utility_weight for s in slices])

    out: dict[str, dict[str, list]] = {
        st: {"throughput_mbps": [], "utilisation": [], "urllc_violation": [],
             "per_slice_mbps": [], "jain": [], "dual_price": [],
             "dual_converged": []}
        for st in STRATEGIES
    }

    for rho in loads:
        # Common random numbers *within* a load point: the same channel and demand
        # draws are replayed for all three strategies.
        draw_rng = np.random.default_rng(seed * 100003 + int(round(rho * 10)))
        demands = np.zeros((n_mc, n_s))
        rates = np.zeros((n_mc, n_s))
        for m in range(n_mc):
            demands[m], rates[m] = _demand_and_rate(draw_rng, rho, n_prb, slot_ms,
                                                    prb_bw_hz, slices)

        for strat in STRATEGIES:
            thr = np.zeros(n_mc)
            util = np.zeros(n_mc)
            viol = np.zeros(n_mc)
            per_slice = np.zeros((n_mc, n_s))
            jain = np.zeros(n_mc)
            dual = DualState()
            achieved = sla.copy()
            for m in range(n_mc):
                dem, rate = demands[m], rates[m]
                if strat == "hard":
                    alloc = hard_isolation(n_prb)
                elif strat == "soft":
                    alloc = soft_isolation(dem, n_prb)
                else:
                    w = sla_proximity_weights(base_w, achieved, sla)
                    alloc, dual = lagrangian_dual(dem, rate, w, n_prb, state=dual)
                served = np.minimum(alloc, dem) * rate            # Mbit/s
                per_slice[m] = served
                thr[m] = served.sum()
                util[m] = float(np.minimum(alloc, dem).sum() / n_prb)
                # URLLC violation: the URLLC slice does not get the PRBs its
                # instantaneous demand needs within this slot.
                deficit = max(0.0, dem[1] - alloc[1])
                viol[m] = float(deficit / max(dem[1], 1e-9) > 0.05)
                s_sum = served.sum()
                jain[m] = (s_sum ** 2) / (n_s * np.sum(served ** 2) + 1e-12)
                achieved = 0.9 * achieved + 0.1 * served

            out[strat]["throughput_mbps"].append(float(thr.mean()))
            out[strat]["utilisation"].append(float(util.mean()))
            out[strat]["urllc_violation"].append(float(viol.mean()))
            out[strat]["per_slice_mbps"].append([float(v) for v in per_slice.mean(axis=0)])
            out[strat]["jain"].append(float(jain.mean()))
            out[strat]["dual_price"].append(float(dual.price))
            out[strat]["dual_converged"].append(bool(dual.converged))

    return {"seed": seed, "loads": list(loads), "strategies": out}
