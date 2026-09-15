"""
ris.py -- RIS-assisted downlink (article Section 8.10, Figure 7).

Signal model
------------
    y_k = (h_{d,k}^H + h_{r,k}^H  Theta  G) w  s + n

    h_{d,k} in C^M   direct gNB -> UE k  (blocked macro link, Rayleigh)
    G       in C^NxM gNB -> RIS          (near-LoS, Rician)
    h_{r,k} in C^N   RIS -> UE k         (Rician)
    Theta   = diag(e^{j theta_1}, ..., e^{j theta_N}),  |theta_n| unit modulus

STATED ASSUMPTIONS (these must appear verbatim in the article):
  * Perfect instantaneous CSI of h_d, G and h_r at the gNB.  No channel-estimation
    overhead or error is modelled.  This makes every RIS number reported here an
    UPPER BOUND on what an implementation can achieve.
  * Continuous phase shifts by default (`RIS.phase_bits = 0`); the code also supports
    b-bit uniform quantisation and the quantisation loss is reported separately.
  * Narrowband flat fading; one scheduled UE per resource, users served in TDMA, so
    the system spectral efficiency is the mean of the per-user spectral efficiencies.
  * Path-loss exponents: 3.5 (direct, blocked), 2.2 (gNB-RIS, near-LoS), 2.8 (RIS-UE).
  * Transmit power is calibrated so the *no-RIS* arm sees the article's -10 dB
    average receive SNR; all three arms then use the same power.

Alternating optimization
------------------------
  1. active beamformer: w = h(Theta)/||h(Theta)||          (MRT, optimal for SU-MISO)
  2. passive phases:    theta_n = angle(h_d^H w) - angle(c_n),
                        c_n = conj(h_{r,n}) * (G w)_n      (closed-form co-phasing)
  Iterated until the relative change of ||h(Theta)||^2 falls below RIS.ao_tolerance
  (1e-6) or RIS.ao_max_iter (30) iterations are reached.  Both the achieved iteration
  count and the convergence flag are reported.
"""

from __future__ import annotations

import numpy as np

from config import RIS, RISConfig

ARMS = ("no_ris", "random_phase", "alternating_opt")


def _crandn(rng: np.random.Generator, *shape: int) -> np.ndarray:
    """Circularly symmetric complex Gaussian, unit variance."""
    return (rng.normal(0.0, np.sqrt(0.5), shape)
            + 1j * rng.normal(0.0, np.sqrt(0.5), shape))


def _steering(n: int, angle_rad: float) -> np.ndarray:
    """Half-wavelength uniform linear array response."""
    return np.exp(1j * np.pi * np.arange(n) * np.sin(angle_rad))


def _pathloss_lin(d_m: float, exponent: float, cfg: RISConfig) -> float:
    pl_db = cfg.ref_pathloss_db + 10.0 * exponent * np.log10(max(d_m, 1.0))
    return float(10.0 ** (-pl_db / 10.0))


def _rician(rng: np.random.Generator, los: np.ndarray, k_lin: float) -> np.ndarray:
    nlos = _crandn(rng, *los.shape)
    return (np.sqrt(k_lin / (1.0 + k_lin)) * los
            + np.sqrt(1.0 / (1.0 + k_lin)) * nlos)


def _quantise(theta: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return theta
    levels = 2 ** bits
    step = 2.0 * np.pi / levels
    return np.round(theta / step) * step


def alternating_optimization(h_d: np.ndarray, G: np.ndarray, h_r: np.ndarray,
                             cfg: RISConfig = RIS) -> tuple[float, int, bool]:
    """Return (||h_eff||^2, iterations, converged)."""
    n = h_r.size
    theta = np.zeros(n)
    prev = -np.inf
    converged = False
    it = 0
    gain = 0.0
    for it in range(1, cfg.ao_max_iter + 1):
        # --- step 1: MRT for the current phases
        h_eff = h_d + G.conj().T @ (np.exp(-1j * theta) * h_r)
        # (h_r^H Theta G)^H = G^H Theta^H h_r
        norm = np.linalg.norm(h_eff)
        if norm < 1e-12:
            break
        w = h_eff / norm
        # --- step 2: closed-form co-phasing
        a = np.vdot(h_d, w)                       # h_d^H w
        gw = G @ w                                # (N,)
        c = np.conj(h_r) * gw                     # per-element contribution
        theta = np.angle(a) - np.angle(c)
        theta = _quantise(theta, cfg.phase_bits)
        h_eff = h_d + G.conj().T @ (np.exp(-1j * theta) * h_r)
        gain = float(np.vdot(h_eff, h_eff).real)
        if prev > 0 and abs(gain - prev) <= cfg.ao_tolerance * prev:
            converged = True
            break
        prev = gain
    return gain, it, converged


def run_ris_seed(seed: int, n_realizations: int | None = None,
                 cfg: RISConfig = RIS) -> dict:
    """One independent replication over the whole N_RIS sweep."""
    n_real = cfg.n_realizations if n_realizations is None else n_realizations
    M, K = cfg.n_bs_antennas, cfg.n_users
    k_lin = 10.0 ** (cfg.rician_k_db / 10.0)
    elem_gain = 10.0 ** (cfg.element_gain_db / 10.0)

    beta_d = _pathloss_lin(cfg.d_bs_ue_m, cfg.pl_exponent_direct, cfg)
    beta_g = _pathloss_lin(cfg.d_bs_ris_m, cfg.pl_exponent_bs_ris, cfg)
    beta_r = _pathloss_lin(cfg.d_ris_ue_m, cfg.pl_exponent_ris_ue, cfg)

    # Power calibration: E[||h_d||^2] = M * beta_d  ->  set P/sigma^2 so the mean
    # no-RIS receive SNR equals cfg.direct_snr_db.
    snr_target = 10.0 ** (cfg.direct_snr_db / 10.0)
    p_over_sigma2 = snr_target / (M * beta_d)

    rows = []
    for n_ris in cfg.n_ris_sweep:
        rng = np.random.default_rng(seed * 15485863 + n_ris)
        se = {arm: np.zeros(n_real * K) for arm in ARMS}
        ao_iters = []
        ao_conv = []
        idx = 0
        for r in range(n_real):
            # gNB-RIS channel: shared by all UEs in this realisation
            psi = rng.uniform(-np.pi / 2, np.pi / 2)
            phi = rng.uniform(-np.pi / 2, np.pi / 2)
            los_G = np.outer(_steering(n_ris, phi), _steering(M, psi).conj())
            G = np.sqrt(beta_g * elem_gain) * _rician(rng, los_G, k_lin)
            for k in range(K):
                h_d = np.sqrt(beta_d) * _crandn(rng, M)
                omega = rng.uniform(-np.pi / 2, np.pi / 2)
                h_r = np.sqrt(beta_r) * _rician(rng, _steering(n_ris, omega), k_lin)

                # --- arm 1: no RIS
                g0 = float(np.vdot(h_d, h_d).real)
                # --- arm 2: random phases
                th = rng.uniform(0.0, 2.0 * np.pi, n_ris)
                th = _quantise(th, cfg.phase_bits)
                h_rand = h_d + G.conj().T @ (np.exp(-1j * th) * h_r)
                g1 = float(np.vdot(h_rand, h_rand).real)
                # --- arm 3: alternating optimization
                g2, it, conv = alternating_optimization(h_d, G, h_r, cfg)
                ao_iters.append(it)
                ao_conv.append(conv)

                for arm, g in (("no_ris", g0), ("random_phase", g1),
                               ("alternating_opt", g2)):
                    se[arm][idx] = np.log2(1.0 + p_over_sigma2 * g)
                idx += 1

        row = {"n_ris": int(n_ris),
               "ao_mean_iterations": float(np.mean(ao_iters)),
               "ao_converged_fraction": float(np.mean(ao_conv))}
        for arm in ARMS:
            v = se[arm]
            row[f"{arm}_se_bps_hz"] = float(v.mean())
            row[f"{arm}_se_std"] = float(v.std(ddof=1))
            row[f"{arm}_throughput_mbps"] = float(v.mean() * cfg.bandwidth_mhz)
            row[f"{arm}_urllc_violation"] = float(
                np.mean(v < cfg.urllc_rate_req_bps_per_hz))
        row["ao_gain_pct_vs_no_ris"] = 100.0 * (
            row["alternating_opt_se_bps_hz"] / max(row["no_ris_se_bps_hz"], 1e-12) - 1.0)
        row["random_gain_pct_vs_no_ris"] = 100.0 * (
            row["random_phase_se_bps_hz"] / max(row["no_ris_se_bps_hz"], 1e-12) - 1.0)
        rows.append(row)

    return {"seed": seed, "rows": rows,
            "assumptions": {
                "csi": "perfect instantaneous CSI (upper bound)",
                "phase_resolution": ("continuous" if cfg.phase_bits <= 0
                                     else f"{cfg.phase_bits}-bit uniform"),
                "pathloss_exponents": {"direct": cfg.pl_exponent_direct,
                                       "bs_ris": cfg.pl_exponent_bs_ris,
                                       "ris_ue": cfg.pl_exponent_ris_ue},
                "rician_k_db": cfg.rician_k_db,
                "scheduling": "single UE per resource, TDMA across K users",
            }}
