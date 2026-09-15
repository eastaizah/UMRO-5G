"""
traffic.py -- Offered-load generation.

Two generators:
  * `PoissonSliceTraffic`  -- per-slice Poisson packet arrivals on a TTI clock, used by
    the integrated runner and the multi-slice Monte Carlo.
  * `diurnal_multiplier`   -- the compressed 24-hour envelope that makes the scenario
    non-stationary so that the medium and slow control loops have something to track.
"""

from __future__ import annotations

import numpy as np

from config import SLICES, TRAFFIC, SliceSpec


def diurnal_multiplier(t_seconds: np.ndarray | float,
                       compress_seconds: float | None = None) -> np.ndarray:
    """Load multiplier at simulated time `t_seconds`.

    The 24 hourly anchors of `TRAFFIC.diurnal_hourly` are linearly interpolated on a
    circular clock and the whole day is compressed into `compress_seconds`.
    """
    compress = TRAFFIC.compress_seconds if compress_seconds is None else compress_seconds
    t = np.asarray(t_seconds, dtype=np.float64)
    hours = (t / compress) * 24.0
    anchors = np.asarray(TRAFFIC.diurnal_hourly, dtype=np.float64)
    x = np.mod(hours, 24.0)
    i0 = np.floor(x).astype(int) % 24
    i1 = (i0 + 1) % 24
    frac = x - np.floor(x)
    return anchors[i0] * (1.0 - frac) + anchors[i1] * frac


def diurnal_hour_of(t_seconds: float, compress_seconds: float) -> float:
    return float(np.mod((t_seconds / compress_seconds) * 24.0, 24.0))


class PoissonSliceTraffic:
    """Poisson packet arrivals for one slice on a TTI clock.

    At load factor `rho` and diurnal multiplier `m`, the per-TTI arrival count is
    Poisson with mean  n_users * lambda_user * rho * m * T_tti.  Packet sizes are
    exponentially distributed with the slice's mean, so the per-TTI arrived *bits*
    are a compound Poisson variable; we sample the count and then the aggregate size
    as a Gamma(count, mean) draw, which is exact for exponential marks.
    """

    def __init__(self, rng: np.random.Generator, spec: SliceSpec, slot_ms: float):
        self.rng = rng
        self.spec = spec
        self.slot_s = slot_ms * 1e-3

    def arrivals(self, rho: float, multiplier: float = 1.0) -> tuple[int, float]:
        """Return (n_packets, total_bits) arriving in one TTI."""
        lam = (self.spec.n_users * self.spec.arrival_rate_per_user_hz
               * rho * multiplier * self.slot_s)
        n = int(self.rng.poisson(lam))
        if n == 0:
            return 0, 0.0
        bits = float(self.rng.gamma(shape=n, scale=self.spec.packet_bits_mean))
        return n, bits

    def arrivals_block(self, rho: np.ndarray, multiplier: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised version over a block of TTIs."""
        lam = (self.spec.n_users * self.spec.arrival_rate_per_user_hz
               * np.asarray(rho) * np.asarray(multiplier) * self.slot_s)
        n = self.rng.poisson(lam)
        bits = np.where(n > 0,
                        self.rng.gamma(shape=np.maximum(n, 1e-9),
                                       scale=self.spec.packet_bits_mean),
                        0.0)
        return n, bits


def offered_bits_per_tti(spec: SliceSpec, rho: float, multiplier: float,
                         slot_ms: float) -> float:
    """Deterministic mean offered load, in bits per TTI (used for capacity planning
    inside the slow loop, never for the measured results)."""
    return (spec.n_users * spec.arrival_rate_per_user_hz * rho * multiplier
            * spec.packet_bits_mean * slot_ms * 1e-3)


def offered_mbps(spec: SliceSpec, rho: float, multiplier: float = 1.0) -> float:
    return (spec.n_users * spec.arrival_rate_per_user_hz * rho * multiplier
            * spec.packet_bits_mean) / 1e6


def total_offered_mbps(rho: float, multiplier: float = 1.0) -> float:
    return sum(offered_mbps(s, rho, multiplier) for s in SLICES)


def daily_load_profile(n_slots: int, slot_minutes: float,
                       rng: np.random.Generator | None = None,
                       noise_cv: float = 0.0) -> np.ndarray:
    """Normalised offered-load trace over one day at `slot_minutes` resolution.

    Returned values are multiples of the daily mean (mean of the noiseless trace over
    a full day is 1.0 by construction of TRAFFIC.diurnal_hourly).
    """
    hours = (np.arange(n_slots) * slot_minutes) / 60.0
    anchors = np.asarray(TRAFFIC.diurnal_hourly, dtype=np.float64)
    x = np.mod(hours, 24.0)
    i0 = np.floor(x).astype(int) % 24
    i1 = (i0 + 1) % 24
    frac = x - np.floor(x)
    base = anchors[i0] * (1.0 - frac) + anchors[i1] * frac
    base = base / anchors.mean()
    if rng is not None and noise_cv > 0.0:
        # multiplicative log-normal jitter with the requested coefficient of variation
        sigma = np.sqrt(np.log(1.0 + noise_cv ** 2))
        base = base * rng.lognormal(-0.5 * sigma ** 2, sigma, size=n_slots)
    return base
