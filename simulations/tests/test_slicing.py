import numpy as np
import pytest

from slicing import (hard_isolation, soft_isolation, lagrangian_dual, DualState,
                     sla_proximity_weights, _project_to_budget, run_multislice_seed,
                     STRATEGIES)
from config import SLICING, NUM_MU0, TRAFFIC


N_PRB = 100


def test_hard_isolation_is_static_and_within_budget():
    a = hard_isolation(N_PRB)
    b = hard_isolation(N_PRB)
    assert np.array_equal(a, b)
    assert a.sum() <= N_PRB + 1e-9
    assert np.array_equal(a, np.array(SLICING.hard_split, dtype=float))


def test_soft_isolation_respects_minimums_and_the_pool():
    dem = np.array([70.0, 10.0, 5.0])
    a = soft_isolation(dem, N_PRB)
    mins = np.array(SLICING.soft_min, dtype=float)
    assert np.all(a >= mins - 1e-9)
    assert a.sum() == pytest.approx(mins.sum() + SLICING.soft_pool)


def test_soft_isolation_favours_the_hungrier_slice():
    a = soft_isolation(np.array([80.0, 12.0, 5.0]), N_PRB)
    b = soft_isolation(np.array([12.0, 80.0, 5.0]), N_PRB)
    assert a[0] > b[0]
    assert b[1] > a[1]


def test_soft_isolation_with_no_excess_demand_gives_only_the_minimums():
    mins = np.array(SLICING.soft_min, dtype=float)
    a = soft_isolation(mins.copy(), N_PRB)
    assert np.allclose(a, mins)


def test_lagrangian_respects_the_budget_and_the_minimums():
    dem = np.array([80.0, 25.0, 15.0])
    rate = np.array([1.2, 0.9, 0.4])
    w = np.array([1.0, 6.0, 0.6])
    alloc, st = lagrangian_dual(dem, rate, w, N_PRB)
    assert alloc.sum() == pytest.approx(N_PRB, abs=1e-6)
    assert np.all(alloc >= np.array(SLICING.soft_min) - 1e-9)
    assert np.all(alloc <= np.maximum(dem, SLICING.soft_min) + 1e-9)


def test_lagrangian_gives_more_to_the_higher_weighted_slice():
    dem = np.array([90.0, 90.0, 90.0])
    rate = np.array([1.0, 1.0, 1.0])
    lo, _ = lagrangian_dual(dem, rate, np.array([1.0, 1.0, 1.0]), N_PRB,
                            minimums=(0, 0, 0), n_iterations=400)
    hi, _ = lagrangian_dual(dem, rate, np.array([1.0, 8.0, 1.0]), N_PRB,
                            minimums=(0, 0, 0), n_iterations=400)
    assert hi[1] > lo[1]
    assert hi[0] < lo[0]


def test_lagrangian_dual_price_converges_on_a_feasible_instance():
    dem = np.array([70.0, 40.0, 30.0])
    rate = np.array([1.1, 0.9, 0.5])
    w = np.array([1.0, 3.0, 0.6])
    st = DualState()
    for _ in range(40):
        alloc, st = lagrangian_dual(dem, rate, w, N_PRB, state=st,
                                    minimums=(0, 0, 0), n_iterations=200)
    assert abs(alloc.sum() - N_PRB) < 1e-6
    assert st.converged


def test_dual_price_is_higher_when_demand_is_higher():
    """The dual price is the shadow cost of a PRB, so it must increase with the
    demand pressure.  (Its absolute level depends on the utility scale, which is why
    the test compares two demand levels rather than an arbitrary starting value.)"""
    rate = np.array([1.0, 1.0, 1.0])
    w = np.array([1.0, 1.0, 1.0])

    def settle(dem):
        st = DualState()
        for _ in range(12):
            _, st = lagrangian_dual(np.array(dem), rate, w, N_PRB, state=st,
                                    minimums=(0, 0, 0), n_iterations=200)
        return st.price

    assert settle([200.0, 200.0, 200.0]) > settle([40.0, 25.0, 15.0])


def test_dual_state_warm_start_is_carried_across_calls():
    dem = np.array([50.0, 30.0, 20.0])
    rate = np.array([1.0, 1.0, 1.0])
    w = np.array([1.0, 1.0, 1.0])
    st = DualState()
    _, st = lagrangian_dual(dem, rate, w, N_PRB, state=st)
    n1 = st.n_updates
    _, st = lagrangian_dual(dem, rate, w, N_PRB, state=st)
    assert st.n_updates > n1


def test_projection_hits_the_budget_exactly():
    mins = np.array([10.0, 5.0, 5.0])
    upper = np.array([80.0, 60.0, 60.0])
    for start in ([70.0, 50.0, 40.0], [10.0, 5.0, 5.0], [12.0, 6.0, 6.0]):
        a = _project_to_budget(np.array(start), mins, upper, N_PRB)
        assert a.sum() == pytest.approx(N_PRB, abs=1e-6)
        assert np.all(a >= mins - 1e-9)


def test_sla_proximity_boosts_only_under_served_slices():
    base = np.array([1.0, 1.0, 1.0])
    sla = np.array([100.0, 10.0, 2.0])
    ach = np.array([120.0, 5.0, 2.0])
    w = sla_proximity_weights(base, ach, sla)
    assert w[0] == pytest.approx(1.0)         # above SLA -> nominal weight
    assert w[1] > 1.0                          # below SLA -> bids harder
    assert w[2] == pytest.approx(1.0)


def test_multislice_seed_structure_and_reproducibility():
    a = run_multislice_seed(3, N_PRB, NUM_MU0.slot_ms, NUM_MU0.prb_bandwidth_hz,
                            TRAFFIC.load_sweep[:3], 20)
    b = run_multislice_seed(3, N_PRB, NUM_MU0.slot_ms, NUM_MU0.prb_bandwidth_hz,
                            TRAFFIC.load_sweep[:3], 20)
    assert set(a["strategies"]) == set(STRATEGIES)
    assert a["strategies"]["umro"]["throughput_mbps"] == b["strategies"]["umro"]["throughput_mbps"]
    for st in STRATEGIES:
        assert len(a["strategies"][st]["throughput_mbps"]) == 3
        assert all(0.0 <= u <= 1.0001 for u in a["strategies"][st]["utilisation"])
        assert all(0.0 <= v <= 1.0 for v in a["strategies"][st]["urllc_violation"])


def test_utilisation_is_non_decreasing_in_load_for_hard_isolation():
    r = run_multislice_seed(5, N_PRB, NUM_MU0.slot_ms, NUM_MU0.prb_bandwidth_hz,
                            (0.2, 0.5, 0.9), 40)
    u = r["strategies"]["hard"]["utilisation"]
    assert u[0] < u[1] < u[2]
