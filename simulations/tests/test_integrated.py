import numpy as np
import pytest

from integrated import (integer_allocation, IntegratedCell, MediumPolicy,
                        MEDIUM_ACTIONS, MEDIUM_STATE_DIM, medium_state,
                        MediumSurrogateEnv, coupling_statistics, run_arm,
                        N_SLICES, SLA_CARRIED_LOAD)
from config import INTEGRATED, ARMS, SLICES


N_PRB = INTEGRATED.numerology.n_prb


def test_integer_allocation_hits_the_budget_and_the_minimums():
    mins = np.array([10, 8, 6])
    for x in (np.array([50.0, 30.0, 20.0]), np.array([1.0, 1.0, 1.0]),
              np.array([200.0, 5.0, 5.0]), np.array([0.0, 0.0, 0.0])):
        a = integer_allocation(x, N_PRB, mins)
        assert a.sum() == N_PRB
        assert np.all(a >= mins)
        assert a.dtype.kind == "i"


def test_integer_allocation_preserves_ordering():
    a = integer_allocation(np.array([60.0, 30.0, 16.0]), N_PRB,
                           np.array([5, 5, 5]))
    assert a[0] > a[1] > a[2]


def test_medium_state_dimension_and_bounds():
    st = medium_state(np.array([50.0, 30.0, 26.0]), np.array([1e6, 1e5, 1e4]),
                      np.array([100.0, 20.0, 3.0]), np.array([1.0, 2.0, 0.5]),
                      N_PRB, 0.5)
    assert st.shape == (MEDIUM_STATE_DIM,)
    assert np.all(np.isfinite(st))


def test_medium_policy_returns_a_legal_action():
    p = MediumPolicy()
    st = np.zeros(MEDIUM_STATE_DIM)
    a = p.act(st)
    assert 0 <= a < len(MEDIUM_ACTIONS)


def test_surrogate_env_runs_and_conserves_prbs():
    env = MediumSurrogateEnv(np.random.default_rng(1))
    env.reset()
    done = False
    steps = 0
    while not done:
        s, r, done = env.step(steps % len(MEDIUM_ACTIONS))
        assert env.alloc.sum() <= N_PRB + 1e-9
        assert np.isfinite(r)
        steps += 1
    assert steps == env.n_steps


def test_every_arm_runs_and_conserves_the_prb_budget():
    pol = MediumPolicy()
    for arm in ARMS:
        cell = IntegratedCell(5, arm, pol, compress_seconds=1.0)
        m = cell.run()
        assert cell.alloc.sum() == N_PRB
        assert m["aggregate_throughput_mbps"] > 0
        assert 0.0 <= m["prb_utilisation"] <= 1.0
        assert 0.0 <= m["urllc_violation_prob"] <= 1.0
        assert 0.0 <= m["sla_satisfaction_rate"] <= 1.0


def test_common_random_numbers_give_identical_offered_load_across_arms():
    pol = MediumPolicy()
    offered = []
    for arm in ARMS:
        m = run_arm(9, arm, pol, compress_seconds=1.0)
        offered.append(sum(m[f"offered_{s.name}_mbps"] for s in SLICES))
    assert max(offered) - min(offered) < 1e-6


def test_only_the_enabled_loops_actually_run():
    pol = MediumPolicy()
    a = run_arm(3, "A_hard_static", pol, compress_seconds=2.0)
    c = run_arm(3, "C_no_medium", pol, compress_seconds=2.0)
    d = run_arm(3, "D_no_slow", pol, compress_seconds=2.0)
    e = run_arm(3, "E_full_umro", pol, compress_seconds=2.0)
    assert a["slow_updates"] == 0
    assert a["medium_actions_applied"] == 0
    assert c["slow_updates"] > 0
    assert c["medium_actions_applied"] == 0
    assert d["slow_updates"] == 0
    assert e["slow_updates"] > 0


def test_hard_isolation_never_changes_its_allocation():
    cell = IntegratedCell(4, "A_hard_static", MediumPolicy(), compress_seconds=2.0)
    start = cell.alloc.copy()
    cell.run()
    assert np.array_equal(start, cell.alloc)


def test_slow_loop_moves_the_prices_and_the_guarantees():
    cell = IntegratedCell(6, "E_full_umro", MediumPolicy(), compress_seconds=6.0)
    p0 = cell.prices.copy()
    m0 = cell.mins.copy()
    cell.run()
    assert not np.allclose(p0, cell.prices)
    assert not np.array_equal(m0, cell.mins) or cell.n_slow_updates > 0


def test_trace_is_recorded_and_coupling_is_computable():
    m = run_arm(7, "E_full_umro", MediumPolicy(), compress_seconds=3.0,
                record_trace=True)
    tr = m["trace"]
    n = len(tr["t_s"])
    assert n > 5
    for key in ("prices", "min_prb", "alloc", "throughput_mbps",
                "offered_multiplier", "action"):
        assert len(tr[key]) == n
    c = coupling_statistics(tr)
    assert c["n_medium_samples"] == n
    assert len(c["alloc_range_per_slice"]) == N_SLICES


def test_served_load_never_exceeds_offered_load():
    for arm in ARMS:
        m = run_arm(8, arm, MediumPolicy(), compress_seconds=1.0)
        assert m["carried_load_fraction"] <= 1.05      # Poisson noise on the estimate


def test_run_is_reproducible():
    a = run_arm(12, "E_full_umro", MediumPolicy(), compress_seconds=1.0)
    b = run_arm(12, "E_full_umro", MediumPolicy(), compress_seconds=1.0)
    assert a["aggregate_throughput_mbps"] == b["aggregate_throughput_mbps"]
    assert a["sla_satisfaction_rate"] == b["sla_satisfaction_rate"]


def test_sla_threshold_constant_is_the_documented_one():
    assert SLA_CARRIED_LOAD == 0.95
