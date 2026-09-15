"""
energy.py -- O-RU energy model and adaptive cell sleeping (article Section 8.9,
Figure 6).

Power model (EARTH / Auer et al., "How much energy is needed to run a wireless
network?", IEEE Wireless Commun. 2011):

    P_site(t) = n_active(t) * (P_static + P_slope * u(t))
              + (N - n_active(t)) * P_sleep
              + P_overhead
    E         = sum_t P_site(t) * T_slot  +  n_wakeups * E_wake

where u(t) is the fractional utilisation of the active O-RUs.  A sleeping O-RU still
draws P_sleep and costs E_wake to bring back up, so aggressive sleeping is not free.

Three strategies:
  always_on   all O-RUs powered, no sleeping.
  static      a fixed fraction sleeps inside a fixed night window (open loop).
  predictive  one-slot-ahead demand forecast drives the activation decision, with a
              configurable safety margin.

FORECASTER HONESTY NOTE
-----------------------
The forecaster is **Holt's linear-trend exponential smoothing**, a two-parameter
recursive predictor.  It is NOT an LSTM.  The article's current claim of "LSTM-based
traffic prediction" is not reproduced by this code and must be reworded; see
work/sim_results.md.
"""

from __future__ import annotations

import numpy as np

from config import ENERGY, EnergyConfig
from traffic import daily_load_profile

STRATEGIES = ("always_on", "static_sleep", "predictive_sleep")


# ----------------------------------------------------------------------------------
# Forecaster
# ----------------------------------------------------------------------------------

class HoltForecaster:
    """Holt's linear trend method (double exponential smoothing).

        level_t = a * y_t + (1 - a) * (level_{t-1} + trend_{t-1})
        trend_t = b * (level_t - level_{t-1}) + (1 - b) * trend_{t-1}
        yhat_{t+1} = level_t + trend_t
    """

    def __init__(self, alpha: float = ENERGY.holt_alpha,
                 beta: float = ENERGY.holt_beta):
        self.alpha = alpha
        self.beta = beta
        self.level: float | None = None
        self.trend: float = 0.0

    def predict(self) -> float:
        if self.level is None:
            return 0.0
        return max(0.0, self.level + self.trend)

    def update(self, y: float) -> None:
        if self.level is None:
            self.level = float(y)
            self.trend = 0.0
            return
        prev = self.level
        self.level = self.alpha * y + (1.0 - self.alpha) * (prev + self.trend)
        self.trend = self.beta * (self.level - prev) + (1.0 - self.beta) * self.trend


# ----------------------------------------------------------------------------------
# Day simulation
# ----------------------------------------------------------------------------------

def _n_slots(cfg: EnergyConfig) -> int:
    return int(round(24 * 60 / cfg.slot_minutes))


def _slot_power_w(n_active: int, served_mbps: float, cfg: EnergyConfig) -> float:
    cap = max(n_active, 1) * cfg.per_oru_capacity_mbps
    u = float(np.clip(served_mbps / cap, 0.0, 1.0))
    return (n_active * (cfg.p_active_static_w + cfg.p_load_slope_w * u)
            + (cfg.n_oru - n_active) * cfg.p_sleep_w
            + cfg.p_overhead_w)


def simulate_day(rng: np.random.Generator, strategy: str, demand_scale: float,
                 margin: float, cfg: EnergyConfig = ENERGY) -> dict:
    """One 24-hour day at 15-minute resolution.

    `demand_scale` scales the whole diurnal envelope; sweeping it is what traces the
    energy-efficiency / throughput curve of Figure 6.
    """
    n_slots = _n_slots(cfg)
    slot_s = cfg.slot_minutes * 60.0
    profile = daily_load_profile(n_slots, cfg.slot_minutes, rng, cfg.demand_noise_cv)

    full_capacity = cfg.n_oru * cfg.per_oru_capacity_mbps
    # Nominal mean demand: at demand_scale = 1.0 the daily *peak* fills 85 % of the
    # deployed capacity.  [CHOSEN] documented in README.
    peak_target = 0.85 * full_capacity
    mean_demand = demand_scale * peak_target / max(profile.max(), 1e-9)
    demand = profile * mean_demand

    hours = (np.arange(n_slots) * cfg.slot_minutes) / 60.0
    fc = HoltForecaster()
    prev_active = cfg.n_oru
    energy_j = 0.0
    served_total_mbit = 0.0
    offered_total_mbit = 0.0
    n_wakeups = 0
    active_trace = np.zeros(n_slots, dtype=int)
    forecast_trace = np.zeros(n_slots)
    blocking = np.zeros(n_slots)

    for t in range(n_slots):
        if strategy == "always_on":
            n_active = cfg.n_oru
        elif strategy == "static_sleep":
            in_window = (cfg.static_sleep_start_hour <= hours[t] < cfg.static_sleep_end_hour)
            if in_window:
                n_active = max(1, int(round(cfg.n_oru * (1.0 - cfg.static_sleep_fraction))))
            else:
                n_active = cfg.n_oru
        elif strategy == "predictive_sleep":
            pred = fc.predict()
            if t == 0:
                n_active = cfg.n_oru
            else:
                need = pred * (1.0 + margin) / cfg.per_oru_capacity_mbps
                n_active = int(np.clip(np.ceil(need), 1, cfg.n_oru))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        forecast_trace[t] = fc.predict()
        if n_active > prev_active:
            n_wakeups += (n_active - prev_active)
            energy_j += (n_active - prev_active) * cfg.wake_energy_j

        capacity = n_active * cfg.per_oru_capacity_mbps
        served = min(demand[t], capacity)
        blocking[t] = (demand[t] - served) / max(demand[t], 1e-9)

        energy_j += _slot_power_w(n_active, served, cfg) * slot_s
        served_total_mbit += served * slot_s
        offered_total_mbit += demand[t] * slot_s
        active_trace[t] = n_active
        prev_active = n_active
        fc.update(demand[t])

    mean_served_mbps = served_total_mbit / (n_slots * slot_s)
    ee = served_total_mbit / energy_j                     # Mbit / J
    mape = float(np.mean(np.abs(forecast_trace[1:] - demand[1:])
                         / np.maximum(demand[1:], 1e-9)) * 100.0)
    max_block = float(blocking.max())
    sla_ok = bool(mean_served_mbps >= cfg.sla_rate_mbps and max_block <= cfg.sla_blocking_max)
    return {
        "strategy": strategy,
        "demand_scale": float(demand_scale),
        "margin": float(margin),
        "energy_kwh": energy_j / 3.6e6,
        "energy_j": energy_j,
        "served_mbit": served_total_mbit,
        "offered_mbit": offered_total_mbit,
        "mean_throughput_mbps": mean_served_mbps,
        "peak_throughput_mbps": float(np.minimum(demand, active_trace * cfg.per_oru_capacity_mbps).max()),
        "energy_efficiency_mbit_per_j": ee,
        "mean_active_oru": float(active_trace.mean()),
        "n_wakeups": int(n_wakeups),
        "max_blocking": max_block,
        "mean_blocking": float(blocking.mean()),
        "forecast_mape_pct": mape,
        "sla_feasible": sla_ok,
        "spectral_efficiency_bps_hz": mean_served_mbps * 1e6
        / (cfg.n_oru * 40e6),           # 40 MHz per O-RU, see config.NUM_MU1
    }


DEMAND_SCALES = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30)

# Operating point of the predictive strategy for the demand sweep.
#
# SELECTION RULE, applied once and then frozen: the SMALLEST margin in
# ENERGY.margin_sweep for which every seed satisfies the SLA at the nominal demand
# point.  A pilot over 30 seeds gave peak blocking of 15.7 % at a 15 % margin, 7.4 %
# at 30 %, 1.1 % at 45 % and 0.03 % at 65 %, against a 1 % blocking limit; 0.65 is
# therefore the smallest SLA-feasible margin and is the value used.  Smaller margins
# buy more energy efficiency but violate the SLA, and the margin sweep in the results
# shows the whole trade-off explicitly rather than hiding it behind one point.
DEFAULT_MARGIN = 0.65


def run_energy_seed(seed: int, cfg: EnergyConfig = ENERGY) -> dict:
    """One independent replication: every strategy on the *same* demand realisation
    (common random numbers) for every point of the demand sweep."""
    curves = {s: [] for s in STRATEGIES}
    for i, scale in enumerate(DEMAND_SCALES):
        for strategy in STRATEGIES:
            # identical rng stream => identical demand trace across strategies
            rng = np.random.default_rng(seed * 104729 + i)
            curves[strategy].append(simulate_day(rng, strategy, scale,
                                                 DEFAULT_MARGIN, cfg))
    # Safety-margin sensitivity for the predictive strategy at nominal demand
    margin_rows = []
    for j, m in enumerate(cfg.margin_sweep):
        rng = np.random.default_rng(seed * 104729 + 5)     # nominal scale index 5
        margin_rows.append(simulate_day(rng, "predictive_sleep", 1.00, m, cfg))
    return {"seed": seed, "demand_scales": list(DEMAND_SCALES),
            "curves": curves, "margin_sweep": margin_rows,
            "default_margin": DEFAULT_MARGIN}
