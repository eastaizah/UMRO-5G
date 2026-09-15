import numpy as np
import pytest

from energy import (HoltForecaster, simulate_day, run_energy_seed, STRATEGIES,
                    DEMAND_SCALES, DEFAULT_MARGIN, _slot_power_w, _n_slots)
from config import ENERGY
from traffic import daily_load_profile


def test_holt_tracks_a_linear_trend_exactly_in_the_limit():
    f = HoltForecaster(alpha=0.9, beta=0.9)
    for t in range(60):
        f.update(10.0 + 2.0 * t)
    assert f.predict() == pytest.approx(10.0 + 2.0 * 60, rel=0.02)


def test_holt_converges_on_a_constant_series():
    f = HoltForecaster()
    for _ in range(80):
        f.update(7.0)
    assert f.predict() == pytest.approx(7.0, abs=1e-2)


def test_holt_first_update_initialises_the_level():
    f = HoltForecaster()
    assert f.predict() == 0.0
    f.update(4.0)
    assert f.predict() == pytest.approx(4.0)


def test_power_model_bounds():
    cfg = ENERGY
    full = _slot_power_w(cfg.n_oru, cfg.n_oru * cfg.per_oru_capacity_mbps, cfg)
    idle = _slot_power_w(cfg.n_oru, 0.0, cfg)
    assert full == pytest.approx(cfg.n_oru * cfg.p_active_w + cfg.p_overhead_w)
    assert idle == pytest.approx(cfg.n_oru * cfg.p_active_static_w + cfg.p_overhead_w)
    assert idle < full


def test_sleeping_an_oru_saves_power():
    cfg = ENERGY
    a = _slot_power_w(cfg.n_oru, 100.0, cfg)
    b = _slot_power_w(cfg.n_oru - 5, 100.0, cfg)
    assert b < a


def test_always_on_keeps_every_oru_active():
    r = simulate_day(np.random.default_rng(1), "always_on", 1.0, 0.0)
    assert r["mean_active_oru"] == pytest.approx(ENERGY.n_oru)
    assert r["n_wakeups"] == 0


def test_sleeping_strategies_use_less_energy_than_always_on():
    rng_seed = 4
    base = simulate_day(np.random.default_rng(rng_seed), "always_on", 0.5, 0.15)
    stat = simulate_day(np.random.default_rng(rng_seed), "static_sleep", 0.5, 0.15)
    pred = simulate_day(np.random.default_rng(rng_seed), "predictive_sleep", 0.5, 0.15)
    assert stat["energy_j"] < base["energy_j"]
    assert pred["energy_j"] < base["energy_j"]


def test_energy_efficiency_definition():
    r = simulate_day(np.random.default_rng(2), "predictive_sleep", 0.8, 0.15)
    assert r["energy_efficiency_mbit_per_j"] == pytest.approx(
        r["served_mbit"] / r["energy_j"])


def test_served_never_exceeds_offered_and_blocking_is_a_fraction():
    r = simulate_day(np.random.default_rng(3), "predictive_sleep", 1.3, 0.0)
    assert r["served_mbit"] <= r["offered_mbit"] + 1e-6
    assert 0.0 <= r["mean_blocking"] <= 1.0
    assert 0.0 <= r["max_blocking"] <= 1.0


def test_larger_safety_margin_reduces_blocking():
    seed = 9
    lo = simulate_day(np.random.default_rng(seed), "predictive_sleep", 1.0, 0.0)
    hi = simulate_day(np.random.default_rng(seed), "predictive_sleep", 1.0, 0.65)
    assert hi["max_blocking"] <= lo["max_blocking"] + 1e-9
    assert hi["mean_active_oru"] >= lo["mean_active_oru"]


def test_common_random_numbers_give_identical_demand_across_strategies():
    seed = 11
    a = simulate_day(np.random.default_rng(seed), "always_on", 1.0, 0.15)
    b = simulate_day(np.random.default_rng(seed), "static_sleep", 1.0, 0.15)
    assert a["offered_mbit"] == pytest.approx(b["offered_mbit"])


def test_diurnal_profile_is_normalised_and_periodic():
    n = _n_slots(ENERGY)
    p = daily_load_profile(n, ENERGY.slot_minutes)
    assert p.mean() == pytest.approx(1.0, rel=0.02)
    assert p.min() < 0.5 < p.max()


def test_forecast_error_is_reported_and_finite():
    r = simulate_day(np.random.default_rng(5), "predictive_sleep", 1.0, 0.15)
    assert np.isfinite(r["forecast_mape_pct"])
    assert r["forecast_mape_pct"] >= 0.0


def test_seed_run_shape_and_reproducibility():
    a = run_energy_seed(21)
    b = run_energy_seed(21)
    assert set(a["curves"]) == set(STRATEGIES)
    assert len(a["curves"]["always_on"]) == len(DEMAND_SCALES)
    assert (a["curves"]["predictive_sleep"][0]["energy_j"]
            == b["curves"]["predictive_sleep"][0]["energy_j"])
    assert a["default_margin"] == DEFAULT_MARGIN
