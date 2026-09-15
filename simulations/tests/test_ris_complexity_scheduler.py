import numpy as np
import pytest

from ris import (alternating_optimization, run_ris_seed, _crandn, _steering,
                 _quantise, _pathloss_lin, ARMS as RIS_ARMS)
from config import RIS
from complexity import (water_filling, branch_and_bound_knapsack, make_policy_net,
                        policy_forward, run_complexity_seed)
from config import COMPLEXITY
from scheduler import (jains_fairness, PolicyGradientScheduler, run_scheduling_seed,
                       ALGORITHMS)
from config import SCHED, NUM_MU1_SCHED


# ------------------------------------------------------------------ RIS

def test_steering_vector_has_unit_modulus():
    v = _steering(32, 0.4)
    assert np.allclose(np.abs(v), 1.0)


def test_quantisation_is_a_no_op_for_continuous_phases():
    th = np.linspace(0, 2 * np.pi, 17)
    assert np.allclose(_quantise(th, 0), th)


def test_quantisation_produces_the_right_number_of_levels():
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 4000)
    q = _quantise(th, 2)
    assert len(np.unique(np.round(q, 9))) <= 2 ** 2 + 1


def test_pathloss_decreases_with_distance_and_exponent():
    assert _pathloss_lin(200.0, 3.5, RIS) < _pathloss_lin(50.0, 3.5, RIS)
    assert _pathloss_lin(200.0, 3.5, RIS) < _pathloss_lin(200.0, 2.2, RIS)


def test_alternating_optimization_beats_random_phases_and_no_ris():
    rng = np.random.default_rng(1)
    M, N = 16, 64
    h_d = _crandn(rng, M) * 0.05
    G = _crandn(rng, N, M)
    h_r = _crandn(rng, N)
    g_no = float(np.vdot(h_d, h_d).real)
    th = rng.uniform(0, 2 * np.pi, N)
    h_rand = h_d + G.conj().T @ (np.exp(-1j * th) * h_r)
    g_rand = float(np.vdot(h_rand, h_rand).real)
    g_ao, iters, conv = alternating_optimization(h_d, G, h_r, RIS)
    assert g_ao > g_rand > g_no
    assert 1 <= iters <= RIS.ao_max_iter


def test_alternating_optimization_converges_and_reports_it():
    rng = np.random.default_rng(2)
    M, N = 8, 32
    h_d = _crandn(rng, M) * 0.1
    G = _crandn(rng, N, M)
    h_r = _crandn(rng, N)
    g, iters, conv = alternating_optimization(h_d, G, h_r, RIS)
    assert conv is True
    assert iters < RIS.ao_max_iter


def test_ao_gain_grows_with_the_number_of_elements():
    rng = np.random.default_rng(3)
    M = 8
    h_d = _crandn(rng, M) * 0.05
    gains = []
    for N in (16, 64, 256):
        r = np.random.default_rng(4)
        G = _crandn(r, N, M) / np.sqrt(N)
        h_r = _crandn(r, N)
        g, _, _ = alternating_optimization(h_d, G, h_r, RIS)
        gains.append(g)
    assert gains[0] < gains[1] < gains[2]


def test_ris_seed_run_structure():
    r = run_ris_seed(1, n_realizations=6)
    assert len(r["rows"]) == len(RIS.n_ris_sweep)
    for row in r["rows"]:
        for arm in RIS_ARMS:
            assert row[f"{arm}_se_bps_hz"] >= 0.0
            assert 0.0 <= row[f"{arm}_urllc_violation"] <= 1.0
        assert row["alternating_opt_se_bps_hz"] >= row["no_ris_se_bps_hz"]
    assert "csi" in r["assumptions"]


# ------------------------------------------------------------ complexity

def test_water_filling_satisfies_the_kkt_conditions():
    rng = np.random.default_rng(5)
    g = rng.exponential(1.0, 50)
    p = water_filling(g, 10.0, COMPLEXITY)
    assert p.sum() == pytest.approx(10.0, rel=1e-6)
    assert np.all(p >= -1e-12)
    active = p > 1e-9
    level = p[active] + 1.0 / g[active]
    assert np.allclose(level, level[0], rtol=1e-5)


def test_water_filling_gives_more_power_to_better_channels():
    g = np.array([0.1, 1.0, 10.0])
    p = water_filling(g, 5.0, COMPLEXITY)
    assert p[2] >= p[1] >= p[0]


def test_branch_and_bound_matches_brute_force_on_small_instances():
    rng = np.random.default_rng(6)
    for _ in range(6):
        n = 12
        w = rng.integers(1, 40, n).astype(float)
        v = w + 5.0
        cap = 0.5 * w.sum()
        best, to, nodes = branch_and_bound_knapsack(v, w, cap, 10.0)
        brute = 0.0
        for mask in range(1 << n):
            sel = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
            if sel @ w <= cap:
                brute = max(brute, float(sel @ v))
        assert not to
        assert best == pytest.approx(brute)


def test_branch_and_bound_honours_the_timeout():
    rng = np.random.default_rng(7)
    n = 400
    w = rng.integers(1, 1001, n).astype(float)
    v = w + 100.0
    import time
    t0 = time.perf_counter()
    _, to, _ = branch_and_bound_knapsack(v, w, 0.5 * w.sum(), 0.5)
    el = time.perf_counter() - t0
    assert to is True
    assert el < 3.0


def test_policy_forward_shapes():
    rng = np.random.default_rng(8)
    net = make_policy_net(40, rng, COMPLEXITY)
    out = policy_forward(net, rng.normal(0, 1, (1, 40)))
    assert out.shape == (1, 40)


def test_complexity_rows_are_reported_for_every_size():
    r = run_complexity_seed(1, n_repeats=2)
    assert [row["n"] for row in r["rows"]] == list(COMPLEXITY.n_sweep)
    for row in r["rows"]:
        assert row["learned_inference_ms_median"] > 0
        assert row["water_filling_ms_median"] > 0
        assert row["branch_and_bound_ms_median"] > 0
        assert isinstance(row["bnb_timed_out"], bool)


# -------------------------------------------------------------- scheduler

def test_jain_index_bounds():
    assert jains_fairness(np.ones(10)) == pytest.approx(1.0)
    one_hot = np.zeros(10)
    one_hot[0] = 1.0
    assert jains_fairness(one_hot) == pytest.approx(0.1)
    assert 0 < jains_fairness(np.arange(1, 11)) < 1


def test_policy_gradient_scheduler_selects_the_right_number_of_prbs():
    rng = np.random.default_rng(9)
    pol = PolicyGradientScheduler(rng, SCHED)
    se = rng.exponential(1.0, 20)
    avg = np.full(20, 1.0)
    sel = pol.select(se, avg, 7, explore=True)
    assert len(sel) == 7
    assert len(np.unique(sel)) == 7
    assert np.all((sel >= 0) & (sel < 20))


def test_policy_gradient_update_changes_the_parameters():
    rng = np.random.default_rng(10)
    pol = PolicyGradientScheduler(rng, SCHED)
    se = rng.exponential(1.0, 12)
    avg = np.full(12, 1.0)
    before = pol.W1.copy()
    sel = pol.select(se, avg, 4, explore=True)
    pol.update(np.unique(sel), 3.0)
    assert not np.allclose(before, pol.W1)


def test_scheduling_ordering_is_the_expected_one():
    r = run_scheduling_seed(11, n_tti=600)["algorithms"]
    assert r["max_rate"]["throughput_mbps"] > r["proportional_fair"]["throughput_mbps"]
    assert r["proportional_fair"]["throughput_mbps"] > r["round_robin"]["throughput_mbps"]
    assert r["round_robin"]["jain"] > r["max_rate"]["jain"]
    assert r["proportional_fair"]["jain"] > r["max_rate"]["jain"]
    assert r["round_robin"]["p5_mbps"] > r["max_rate"]["p5_mbps"]


def test_all_scheduling_algorithms_are_present_and_finite():
    r = run_scheduling_seed(12, n_tti=200)["algorithms"]
    assert set(r) == set(ALGORITHMS)
    for algo, m in r.items():
        for k, v in m.items():
            assert np.isfinite(v), (algo, k)
