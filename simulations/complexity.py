"""
complexity.py -- Wall-clock decision-time benchmark (article Section 8.8, Table 12).

Reviewer 1 comment 9 challenges the article's asserted "100-1000x speedup".  Nothing
here is asserted: three decision procedures are executed on the same instances and
`time.perf_counter` is read around each call.  The speedup is whatever the ratio of
the measured medians turns out to be, and the branch-and-bound timeouts are reported
as timeouts rather than hidden behind an em dash.

The three procedures
--------------------
learned_inference : one forward pass of the trained policy network.  The state vector
    is the N-user channel/queue state and the output is one score per user, so the
    network is [N -> 64 -> 64 -> N] and its cost grows linearly in N.  This is exactly
    the operation a Near-RT RIC xApp performs per decision.

water_filling : classical convex power allocation over N parallel channels.  Solved by
    bisection on the water level to `wf_tolerance`; each bisection step is O(N), so
    the procedure is O(N log(1/eps)).

branch_and_bound : exact solution of a 0-1 MINLP -- a strongly correlated binary
    knapsack (c_i = w_i + 100), which is the classical worst case for branch and
    bound.  Depth-first search with the LP (fractional-knapsack) upper bound and
    incumbent pruning.  A wall-clock timeout of `bnb_timeout_s` is enforced; when it
    fires the run is recorded as `timed_out = True` and the elapsed time is the
    timeout, i.e. a LOWER BOUND on the true solution time.

Timing hygiene: BLAS is pinned to one thread by run_all.py, every method is warmed up
before measurement, and the reported statistic is the MEDIAN over `n_repeats`
repetitions, which is robust to operating-system scheduling noise.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from config import COMPLEXITY, ComplexityConfig

METHODS = ("learned_inference", "water_filling", "branch_and_bound")


# ----------------------------------------------------------------------------------
# 1. Learned inference
# ----------------------------------------------------------------------------------

def make_policy_net(n: int, rng: np.random.Generator,
                    cfg: ComplexityConfig = COMPLEXITY) -> List[np.ndarray]:
    h1, h2 = cfg.dqn_hidden
    return [
        rng.normal(0, 1 / np.sqrt(n), (n, h1)), np.zeros(h1),
        rng.normal(0, 1 / np.sqrt(h1), (h1, h2)), np.zeros(h2),
        rng.normal(0, 1 / np.sqrt(h2), (h2, n)), np.zeros(n),
    ]


def policy_forward(net: List[np.ndarray], x: np.ndarray) -> np.ndarray:
    a = np.tanh(x @ net[0] + net[1])
    a = np.tanh(a @ net[2] + net[3])
    return a @ net[4] + net[5]


# ----------------------------------------------------------------------------------
# 2. Water-filling
# ----------------------------------------------------------------------------------

def water_filling(gains: np.ndarray, total_power: float,
                  cfg: ComplexityConfig = COMPLEXITY) -> np.ndarray:
    """p_i = max(0, mu - 1/g_i) with mu chosen so that sum_i p_i = P.

    Bisection on the water level mu; each evaluation is O(N)."""
    inv = 1.0 / np.maximum(gains, 1e-12)
    lo = 0.0
    hi = total_power + float(inv.max())
    mid = 0.5 * (lo + hi)
    for _ in range(cfg.wf_max_iter):
        mid = 0.5 * (lo + hi)
        p = np.maximum(mid - inv, 0.0)
        s = float(p.sum())
        if abs(s - total_power) <= cfg.wf_tolerance * total_power:
            break
        if s > total_power:
            hi = mid
        else:
            lo = mid
    return np.maximum(mid - inv, 0.0)


# ----------------------------------------------------------------------------------
# 3. Branch and bound
# ----------------------------------------------------------------------------------

def _lp_bound(values: np.ndarray, weights: np.ndarray, idx: int,
              cap: float, acc: float) -> float:
    """Fractional-knapsack upper bound over items idx.. (already ratio-sorted)."""
    n = values.size
    b = acc
    c = cap
    i = idx
    while i < n and weights[i] <= c:
        c -= weights[i]
        b += values[i]
        i += 1
    if i < n and c > 0:
        b += values[i] * c / weights[i]
    return b


def branch_and_bound_knapsack(values: np.ndarray, weights: np.ndarray,
                              capacity: float, timeout_s: float
                              ) -> tuple[float, bool, int]:
    """Depth-first branch and bound.  Returns (best value, timed_out, nodes)."""
    order = np.argsort(-(values / weights))
    v = np.ascontiguousarray(values[order], dtype=np.float64)
    w = np.ascontiguousarray(weights[order], dtype=np.float64)
    n = int(v.size)
    best = 0.0
    nodes = 0
    t0 = time.perf_counter()
    stack = [(0, capacity, 0.0)]
    timed_out = False
    while stack:
        nodes += 1
        if (nodes & 0x3FF) == 0 and time.perf_counter() - t0 > timeout_s:
            timed_out = True
            break
        i, cap, acc = stack.pop()
        if acc > best:
            best = acc
        if i >= n:
            continue
        if _lp_bound(v, w, i, cap, acc) <= best:
            continue
        stack.append((i + 1, cap, acc))
        if w[i] <= cap:
            stack.append((i + 1, cap - w[i], acc + v[i]))
    return best, timed_out, nodes


# ----------------------------------------------------------------------------------
# Benchmark driver
# ----------------------------------------------------------------------------------

def run_complexity_seed(seed: int, n_repeats: int | None = None,
                        cfg: ComplexityConfig = COMPLEXITY) -> dict:
    """One independent replication of the whole N sweep for all three methods."""
    n_repeats = cfg.n_repeats if n_repeats is None else n_repeats
    rng = np.random.default_rng(seed)
    rows = []
    for n in cfg.n_sweep:
        net = make_policy_net(n, rng, cfg)
        x = rng.normal(0.0, 1.0, (1, n))
        gains = rng.exponential(1.0, n)
        w_int = rng.integers(1, 1001, n).astype(np.float64)
        v_int = w_int + 100.0
        capacity = 0.5 * float(w_int.sum())

        policy_forward(net, x)
        t = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            policy_forward(net, x)
            t.append(time.perf_counter() - t0)
        dqn_ms = np.array(t) * 1e3

        water_filling(gains, 1.0, cfg)
        t = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            water_filling(gains, 1.0, cfg)
            t.append(time.perf_counter() - t0)
        wf_ms = np.array(t) * 1e3

        # Branch and bound is repeated only while it is cheap; once an instance
        # times out, repeating it just burns `bnb_timeout_s` again for no extra
        # information, so a single measurement is kept and flagged as a lower bound.
        t = []
        timeouts = 0
        nodes = []
        max_reps = max(1, min(3, n_repeats))
        for _ in range(max_reps):
            t0 = time.perf_counter()
            _, to, nd = branch_and_bound_knapsack(v_int, w_int, capacity,
                                                  cfg.bnb_timeout_s)
            t.append(time.perf_counter() - t0)
            timeouts += int(to)
            nodes.append(nd)
            if to:
                break
        bnb_reps = len(t)
        bnb_ms = np.array(t) * 1e3

        rows.append({
            "n": int(n),
            "learned_inference_ms_median": float(np.median(dqn_ms)),
            "learned_inference_ms_mean": float(dqn_ms.mean()),
            "water_filling_ms_median": float(np.median(wf_ms)),
            "water_filling_ms_mean": float(wf_ms.mean()),
            "branch_and_bound_ms_median": float(np.median(bnb_ms)),
            "branch_and_bound_ms_mean": float(bnb_ms.mean()),
            "bnb_timed_out": bool(timeouts > 0),
            "bnb_timeout_fraction": float(timeouts / bnb_reps),
            "bnb_nodes_median": float(np.median(nodes)),
            "speedup_vs_water_filling": float(np.median(wf_ms) / max(np.median(dqn_ms), 1e-12)),
            "speedup_vs_branch_and_bound": float(np.median(bnb_ms) / max(np.median(dqn_ms), 1e-12)),
            "n_repeats": int(n_repeats),
            "bnb_repeats": int(bnb_reps),
        })
    return {"seed": seed, "rows": rows,
            "timeout_s": cfg.bnb_timeout_s,
            "note": ("branch-and-bound entries flagged bnb_timed_out = True are LOWER "
                     "BOUNDS: the search was aborted at the timeout without proving "
                     "optimality, so the true decision time is larger and the "
                     "corresponding speedup figure is a lower bound as well")}
