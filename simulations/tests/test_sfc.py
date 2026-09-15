import numpy as np
import pytest

from sfc import (jackson_mean_latency_s, jackson_latency_variance_s2,
                 hypoexponential_ccdf, max_arrival_rate_for_budget,
                 simulate_chain, required_instances, chain_latency_with_scaling,
                 run_sfc_seed)
from config import SFC


MU = np.array(SFC.service_rates_pps, dtype=float)


def test_single_queue_reduces_to_mm1():
    mu = np.array([5000.0])
    lam = 3000.0
    assert jackson_mean_latency_s(lam, mu) == pytest.approx(1.0 / (5000.0 - 3000.0))


def test_latency_diverges_at_the_bottleneck():
    assert not np.isfinite(jackson_mean_latency_s(MU.min(), MU))
    assert not np.isfinite(jackson_mean_latency_s(MU.min() + 10, MU))
    assert np.isfinite(jackson_mean_latency_s(MU.min() - 1, MU))


def test_latency_is_increasing_in_the_arrival_rate():
    lams = [500, 1000, 2000, 3000, 3800]
    v = [jackson_mean_latency_s(l, MU) for l in lams]
    assert all(v[i] < v[i + 1] for i in range(len(v) - 1))


def test_hypoexponential_ccdf_is_a_survival_function():
    lam = 2000.0
    ts = np.linspace(0, 0.05, 40)
    vals = [hypoexponential_ccdf(t, lam, MU) for t in ts]
    assert vals[0] == pytest.approx(1.0, abs=1e-9)
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))
    assert vals[-1] < 1e-3


def test_ccdf_mean_matches_the_analytical_mean():
    """E[D] = integral_0^inf P(D > t) dt -- checks the two formulas agree."""
    lam = 2500.0
    t = np.linspace(0, 0.2, 40001)
    ccdf = np.array([hypoexponential_ccdf(v, lam, MU) for v in t])
    mean_from_tail = np.trapezoid(ccdf, t)
    assert mean_from_tail == pytest.approx(jackson_mean_latency_s(lam, MU), rel=1e-3)


def test_des_matches_the_analytical_model_within_monte_carlo_error():
    rng = np.random.default_rng(5)
    lam = 2000.0
    des = simulate_chain(rng, lam, MU, 200_000)
    ana = jackson_mean_latency_s(lam, MU)
    assert des["mean_s"] == pytest.approx(ana, rel=0.05)


def test_des_variance_matches_the_analytical_variance():
    rng = np.random.default_rng(6)
    lam = 1500.0
    des = simulate_chain(rng, lam, MU, 200_000)
    var = jackson_latency_variance_s2(lam, MU)
    assert des["std_s"] ** 2 == pytest.approx(var, rel=0.15)


def test_des_percentiles_are_ordered():
    rng = np.random.default_rng(7)
    d = simulate_chain(rng, 1800.0, MU, 60_000)
    assert d["p50_s"] < d["p95_s"] < d["p99_s"]


@pytest.mark.parametrize("budget", [2e-3, 5e-3, SFC.e2e_budget_ms * 1e-3])
def test_max_arrival_rate_respects_the_budget(budget):
    lam = max_arrival_rate_for_budget(budget, MU)
    assert jackson_mean_latency_s(lam, MU) <= budget + 1e-9
    assert jackson_mean_latency_s(lam + 1.0, MU) > budget


def test_zero_load_latency_is_the_floor_of_the_chain():
    floor = jackson_mean_latency_s(0.0, MU)
    assert floor == pytest.approx(float(np.sum(1.0 / MU)))
    assert max_arrival_rate_for_budget(0.5 * floor, MU) == 0.0


def test_scaling_reduces_latency():
    inst1 = np.ones(5)
    inst3 = np.full(5, 3)
    pps = 200.0
    assert (chain_latency_with_scaling(pps, inst3, 250.0)
            < chain_latency_with_scaling(pps, inst1, 250.0))


def test_required_instances_respects_target_utilisation_and_cap():
    assert required_instances(100.0, 250.0, 8, 0.7) == 1
    assert required_instances(400.0, 250.0, 8, 0.7) == 3
    assert required_instances(1e9, 250.0, 8, 0.7) == 8


def test_seed_run_is_reproducible():
    a = run_sfc_seed(3, 20_000)
    b = run_sfc_seed(3, 20_000)
    assert a["rows"][2]["des_ms"] == b["rows"][2]["des_ms"]
