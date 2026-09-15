"""
sfc.py -- Service Function Chain latency: Jackson analytical model vs. a
discrete-event simulation.

Analytical model
----------------
An open tandem of M/M/1 queues.  Burke's theorem makes the departure process of each
stationary M/M/1 queue Poisson with the same rate, so the chain is a Jackson network
and the per-VNF sojourn times are independent with mean 1/(mu_i - lambda).  The
end-to-end mean latency is therefore
        E[D] = sum_i 1 / (mu_i - lambda),        lambda < min_i mu_i
and the end-to-end delay is a hypoexponential random variable, whose tail is used for
the violation probability.

Discrete-event simulation
-------------------------
A FIFO single-server queue satisfies the Lindley recursion
        d_i = max(a_i, d_{i-1}) + s_i
which, substituting c_i = cumsum(s)_i and e_i = d_i - c_i, becomes
        e_i = max(a_i - c_{i-1}, e_{i-1})
i.e. a running maximum.  The recursion is therefore *exact* -- not an approximation --
and can be evaluated with `np.maximum.accumulate`, which is what makes a 200 000-packet
five-stage chain affordable at 30 seeds.  The simulator is event-driven in the sense
that every packet's arrival, service-start and departure epoch is materialised; the
vectorisation only removes the Python-level event loop.
"""

from __future__ import annotations

import numpy as np

from config import SFC, SFCConfig


# ----------------------------------------------------------------------------------
# Analytical
# ----------------------------------------------------------------------------------

def jackson_mean_latency_s(lam_pps: float, mu_pps: np.ndarray) -> float:
    mu = np.asarray(mu_pps, dtype=np.float64)
    if lam_pps >= mu.min():
        return float("inf")
    return float(np.sum(1.0 / (mu - lam_pps)))


def jackson_latency_variance_s2(lam_pps: float, mu_pps: np.ndarray) -> float:
    mu = np.asarray(mu_pps, dtype=np.float64)
    if lam_pps >= mu.min():
        return float("inf")
    return float(np.sum(1.0 / (mu - lam_pps) ** 2))


def hypoexponential_ccdf(t_s: float, lam_pps: float, mu_pps: np.ndarray) -> float:
    """P(D > t) for a sum of independent Exp(mu_i - lambda) sojourn times.

    Uses the standard partial-fraction expansion, valid when the rates are distinct
    (they are, by construction of SFC.service_rates_pps).
    """
    mu = np.asarray(mu_pps, dtype=np.float64)
    if lam_pps >= mu.min():
        return 1.0
    r = mu - lam_pps
    n = r.size
    ccdf = 0.0
    for i in range(n):
        coeff = 1.0
        for j in range(n):
            if j != i:
                coeff *= r[j] / (r[j] - r[i])
        ccdf += coeff * np.exp(-r[i] * t_s)
    return float(np.clip(ccdf, 0.0, 1.0))


def max_arrival_rate_for_budget(budget_s: float, mu_pps: np.ndarray,
                                tol: float = 1e-4) -> float:
    """Largest lambda whose analytical mean end-to-end latency stays inside the
    budget (bisection)."""
    mu = np.asarray(mu_pps, dtype=np.float64)
    lo, hi = 0.0, mu.min() - 1e-6
    if jackson_mean_latency_s(lo, mu) > budget_s:
        return 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if jackson_mean_latency_s(mid, mu) <= budget_s:
            lo = mid
        else:
            hi = mid
    return lo


# ----------------------------------------------------------------------------------
# Discrete-event simulation
# ----------------------------------------------------------------------------------

def simulate_chain(rng: np.random.Generator, lam_pps: float, mu_pps: np.ndarray,
                   n_packets: int, warmup_fraction: float = SFC.warmup_fraction
                   ) -> dict:
    """Exact FIFO tandem-queue simulation.  Returns end-to-end delay statistics."""
    mu = np.asarray(mu_pps, dtype=np.float64)
    inter = rng.exponential(1.0 / lam_pps, n_packets)
    arrivals = np.cumsum(inter)
    entry = arrivals.copy()

    t = arrivals
    for m in mu:
        s = rng.exponential(1.0 / m, n_packets)
        c = np.cumsum(s)
        c_shift = np.concatenate(([0.0], c[:-1]))
        e = np.maximum.accumulate(t - c_shift)
        t = e + c                                 # departure epochs from this stage

    delay = t - entry
    w = int(n_packets * warmup_fraction)
    d = delay[w:]
    return {
        "mean_s": float(d.mean()),
        "std_s": float(d.std(ddof=1)),
        "p50_s": float(np.percentile(d, 50)),
        "p95_s": float(np.percentile(d, 95)),
        "p99_s": float(np.percentile(d, 99)),
        "n_measured": int(d.size),
        "delays_s": d,
    }


def run_sfc_seed(seed: int, n_packets: int | None = None,
                 cfg: SFCConfig = SFC) -> dict:
    """One independent replication over the whole arrival-rate sweep."""
    n_packets = cfg.n_packets_des if n_packets is None else n_packets
    mu = np.asarray(cfg.service_rates_pps, dtype=np.float64)
    budget_s = cfg.urllc_budget_ms * 1e-3
    rows = []
    for k, lam in enumerate(cfg.arrival_sweep_pps):
        rng = np.random.default_rng(seed * 7919 + k)
        ana_mean = jackson_mean_latency_s(lam, mu)
        ana_viol = hypoexponential_ccdf(budget_s, lam, mu)
        if not np.isfinite(ana_mean):
            rows.append({"lambda_pps": float(lam), "stable": False,
                         "analytical_ms": None, "des_ms": None,
                         "rel_error_pct": None, "des_p99_ms": None,
                         "analytical_violation": 1.0, "des_violation": 1.0,
                         "rho_bottleneck": float(lam / mu.min())})
            continue
        des = simulate_chain(rng, lam, mu, n_packets, cfg.warmup_fraction)
        des_viol = float(np.mean(des["delays_s"] > budget_s))
        rows.append({
            "lambda_pps": float(lam),
            "stable": True,
            "rho_bottleneck": float(lam / mu.min()),
            "analytical_ms": ana_mean * 1e3,
            "des_ms": des["mean_s"] * 1e3,
            "rel_error_pct": 100.0 * (des["mean_s"] - ana_mean) / ana_mean,
            "des_p95_ms": des["p95_s"] * 1e3,
            "des_p99_ms": des["p99_s"] * 1e3,
            "analytical_violation": ana_viol,
            "des_violation": des_viol,
        })
    return {
        "seed": seed,
        "rows": rows,
        "max_lambda_for_1ms_pps": max_arrival_rate_for_budget(budget_s, mu),
        "bottleneck_mu_pps": float(mu.min()),
    }


# ----------------------------------------------------------------------------------
# VNF placement / scaling used by the integrated runner's slow loop
# ----------------------------------------------------------------------------------

def required_instances(offered_pps: float, capacity_pps: float | None = None,
                       max_instances: int | None = None,
                       target_utilisation: float | None = None,
                       cfg: SFCConfig = SFC) -> int:
    """Smallest instance count keeping the VNF at or below its target utilisation."""
    capacity_pps = cfg.vnf_cpu_capacity_pps if capacity_pps is None else capacity_pps
    max_instances = cfg.vnf_max_instances if max_instances is None else max_instances
    target = (cfg.vnf_target_utilisation if target_utilisation is None
              else target_utilisation)
    need = offered_pps / (capacity_pps * target)
    return int(min(max_instances, max(1, np.ceil(need))))


def chain_latency_with_scaling(offered_pps: float, instances: np.ndarray,
                               capacity_pps: float | None = None,
                               cfg: SFCConfig = SFC) -> float:
    """Mean chain latency in seconds when VNF i runs `instances[i]` parallel servers,
    modelled as a single M/M/1 queue of aggregate rate instances[i]*capacity."""
    capacity_pps = cfg.vnf_cpu_capacity_pps if capacity_pps is None else capacity_pps
    mu = np.asarray(instances, dtype=np.float64) * capacity_pps
    return jackson_mean_latency_s(offered_pps, mu)
