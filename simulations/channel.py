"""
channel.py -- Link-level model.

Chain:  distance -> 3GPP macro path loss -> log-normal shadowing -> Rayleigh block
fading -> SNR -> CQI selection against a 3GPP NR CQI/MCS table -> discrete spectral
efficiency -> per-PRB rate.

Everything here is deterministic given a numpy Generator, so a seed fully determines
a realisation.  No global RNG is ever used.
"""

from __future__ import annotations

import numpy as np

from config import CHANNEL, CQI_TABLE_256QAM, NumerologyConfig

# --- CQI table, pre-extracted -------------------------------------------------------
CQI_INDEX = np.array([r[0] for r in CQI_TABLE_256QAM], dtype=np.int32)
CQI_SE = np.array([r[3] for r in CQI_TABLE_256QAM], dtype=np.float64)
CQI_CODE_RATE = np.array([r[2] / 1024.0 for r in CQI_TABLE_256QAM], dtype=np.float64)
CQI_MOD_ORDER = np.array([r[1] for r in CQI_TABLE_256QAM], dtype=np.int32)


def cqi_snr_thresholds_db() -> np.ndarray:
    """SNR required to support each CQI, derived from the attenuated / truncated
    Shannon bound  SE = bw_eff * log2(1 + SNR / sinr_eff)  (Mogensen et al., 2007).

    Inverting for the SNR that exactly delivers CQI k's spectral efficiency:
        SNR_req(k) = sinr_eff * (2^(SE_k / bw_eff) - 1)

    The resulting thresholds are monotone in k by construction, which is the property
    the scheduler relies on.
    """
    lin = CHANNEL.sinr_efficiency * (np.power(2.0, CQI_SE / CHANNEL.bw_efficiency) - 1.0)
    return 10.0 * np.log10(lin)


SNR_THRESHOLDS_DB = cqi_snr_thresholds_db()


def se_from_snr_db(snr_db: np.ndarray | float) -> np.ndarray:
    """Discrete spectral efficiency (bit/s/Hz) via CQI selection.

    Returns 0.0 when the SNR cannot support CQI 1 (link outage), which is the
    behaviour of a real MCS selector at target BLER 0.1.
    """
    snr_db = np.asarray(snr_db, dtype=np.float64)
    # index of the highest threshold that is <= snr_db
    idx = np.searchsorted(SNR_THRESHOLDS_DB, snr_db, side="right") - 1
    se = np.where(idx >= 0, CQI_SE[np.clip(idx, 0, len(CQI_SE) - 1)], 0.0)
    return se


def cqi_from_snr_db(snr_db: np.ndarray | float) -> np.ndarray:
    """CQI index in 0..15 (0 = out of range)."""
    snr_db = np.asarray(snr_db, dtype=np.float64)
    idx = np.searchsorted(SNR_THRESHOLDS_DB, snr_db, side="right") - 1
    return np.where(idx >= 0, CQI_INDEX[np.clip(idx, 0, len(CQI_INDEX) - 1)], 0)


def shannon_se(snr_db: np.ndarray | float) -> np.ndarray:
    """Continuous attenuated-Shannon reference, capped at the top CQI.  Used only as a
    sanity bound in the unit tests -- never for reported results."""
    snr = np.power(10.0, np.asarray(snr_db, dtype=np.float64) / 10.0)
    se = CHANNEL.bw_efficiency * np.log2(1.0 + snr / CHANNEL.sinr_efficiency)
    return np.minimum(se, CHANNEL.max_spectral_efficiency)


def path_loss_db(distance_m: np.ndarray | float) -> np.ndarray:
    """3GPP TR 36.814 macro-cell path loss, 2 GHz."""
    d_km = np.maximum(np.asarray(distance_m, dtype=np.float64),
                      CHANNEL.min_distance_m) / 1000.0
    return CHANNEL.pl_intercept_db + CHANNEL.pl_slope_db * np.log10(d_km)


def noise_power_dbm(bandwidth_hz: float) -> float:
    return (CHANNEL.thermal_noise_dbm_per_hz + 10.0 * np.log10(bandwidth_hz)
            + CHANNEL.noise_figure_db)


def sample_positions(rng: np.random.Generator, n_users: int) -> np.ndarray:
    """Uniform in the annulus [min_distance, cell_radius] (uniform *area* density)."""
    r_min, r_max = CHANNEL.min_distance_m, CHANNEL.cell_radius_m
    u = rng.random(n_users)
    return np.sqrt(u * (r_max ** 2 - r_min ** 2) + r_min ** 2)


class CellChannel:
    """Per-user wideband SNR with Rayleigh block fading.

    Large-scale state (distance, shadowing) is drawn once per user and held for the
    whole run.  Small-scale fading is redrawn every `coherence_tti` TTIs, giving the
    block-fading model the article's Section 8.5 describes ("coherence time 10 ms").
    """

    def __init__(self, rng: np.random.Generator, n_users: int,
                 numerology: NumerologyConfig):
        self.rng = rng
        self.n_users = n_users
        self.num = numerology
        self.prb_bw_hz = numerology.prb_bandwidth_hz

        self.distance_m = sample_positions(rng, n_users)
        self.shadow_db = rng.normal(0.0, CHANNEL.shadowing_sigma_db, n_users)

        eirp_dbm = CHANNEL.tx_power_dbm + CHANNEL.bs_antenna_gain_db
        # Power is spread evenly over the carrier; per-PRB EIRP:
        per_prb_dbm = eirp_dbm - 10.0 * np.log10(numerology.n_prb)
        n_dbm = noise_power_dbm(self.prb_bw_hz)
        self.mean_snr_db = (per_prb_dbm + CHANNEL.ue_antenna_gain_db
                            - path_loss_db(self.distance_m) - self.shadow_db - n_dbm)

        self.coherence_tti = max(1, int(round(CHANNEL.coherence_ms / numerology.slot_ms)))
        self._tti = -1
        self._fading_db = np.zeros(n_users)
        self._snr_db = self.mean_snr_db.copy()
        self._se = se_from_snr_db(self._snr_db)

    def step(self, tti: int) -> np.ndarray:
        """Advance to absolute TTI `tti`; return the per-user spectral efficiency."""
        if self._tti < 0 or (tti // self.coherence_tti) != (self._tti // self.coherence_tti):
            # |h|^2 ~ Exp(1)  ->  Rayleigh envelope
            g = self.rng.exponential(1.0, self.n_users)
            self._fading_db = 10.0 * np.log10(np.maximum(g, 1e-12))
            self._snr_db = self.mean_snr_db + self._fading_db
            self._se = se_from_snr_db(self._snr_db)
        self._tti = tti
        return self._se

    @property
    def spectral_efficiency(self) -> np.ndarray:
        return self._se

    @property
    def snr_db(self) -> np.ndarray:
        return self._snr_db

    def prb_rate_bps(self) -> np.ndarray:
        """Bits per second a user would get from one PRB at the current CQI."""
        return self._se * self.prb_bw_hz

    def prb_bits_per_tti(self) -> np.ndarray:
        """Bits carried by one PRB in one TTI."""
        return self._se * self.prb_bw_hz * (self.num.slot_ms * 1e-3)


def rayleigh_se_matrix(rng: np.random.Generator, mean_snr_db: np.ndarray,
                       n_tti: int, coherence_tti: int) -> np.ndarray:
    """Vectorised (n_tti, n_users) spectral-efficiency trace for the scheduling
    experiment, where every TTI must be replayed identically across schedulers."""
    n_users = mean_snr_db.shape[0]
    n_blocks = int(np.ceil(n_tti / coherence_tti))
    g = rng.exponential(1.0, size=(n_blocks, n_users))
    fading_db = 10.0 * np.log10(np.maximum(g, 1e-12))
    snr_db = mean_snr_db[None, :] + fading_db
    se_blocks = se_from_snr_db(snr_db)
    return np.repeat(se_blocks, coherence_tti, axis=0)[:n_tti]
