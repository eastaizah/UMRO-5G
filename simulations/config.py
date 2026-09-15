"""
config.py -- Every scenario parameter used anywhere in the UMRO-5G simulation package.

Design rule (enforced by review, not by the interpreter): no numerical constant that
describes the *scenario* may appear outside this file.  Modules import the dataclass
instances defined at the bottom.  Loop counters, array indices, and unit conversions
are the only literals allowed elsewhere.

Provenance of each value is recorded in the `SOURCE` comment next to it:
  [3GPP]  taken from a 3GPP specification or technical report
  [PAPER] taken from the current draft of the UMRO-5G article (kept for continuity)
  [CHOSEN] the article was ambiguous or silent; a defensible value was chosen here
           and is documented in simulations/README.md
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Dict, List

# ----------------------------------------------------------------------------------
# 0. Global campaign settings
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class CampaignConfig:
    """Replication design.  The *unit of independent replication is the seed*."""

    n_seeds: int = 30               # [CHOSEN] >= 30 so Student-t inference is sound
    n_seeds_quick: int = 4          # smoke-test replication count
    seed_base: int = 20250001       # [CHOSEN] master seed; seed_i = seed_base + i
    # Monte Carlo iterations *inside* one seed.  These reduce the variance of that
    # seed's point estimate.  They do NOT increase the independent sample size.
    n_monte_carlo: int = 1000       # [PAPER] kept identical to the original draft
    n_monte_carlo_quick: int = 60
    alpha: float = 0.05             # family-wise significance level before correction
    n_workers: int = 24             # [CHOSEN] matches the 24-core reference machine


# ----------------------------------------------------------------------------------
# 1. Radio / channel
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelConfig:
    """Link-level model: 3GPP macro path loss + log-normal shadowing + Rayleigh fading,
    mapped to spectral efficiency through a 3GPP NR CQI/MCS table."""

    carrier_ghz: float = 2.0                 # [3GPP] TR 36.814 Case 1 macro carrier
    # 3GPP TR 36.814 Table A.2.1.1-3 macro-cell path loss:  PL = a + b*log10(d_km)
    pl_intercept_db: float = 128.1           # [3GPP]
    pl_slope_db: float = 37.6                # [3GPP]
    shadowing_sigma_db: float = 8.0          # [3GPP] TR 36.814 macro NLOS
    min_distance_m: float = 35.0             # [3GPP] minimum gNB-UE 2D distance
    cell_radius_m: float = 500.0             # [CHOSEN] dense-urban macro ISD/2

    tx_power_dbm: float = 46.0               # [3GPP] 40 W macro gNB total conducted power
    bs_antenna_gain_db: float = 15.0         # [CHOSEN] typical 3-sector macro antenna
    ue_antenna_gain_db: float = 0.0          # [CHOSEN]
    noise_figure_db: float = 7.0             # [3GPP] TR 38.901 UE noise figure
    thermal_noise_dbm_per_hz: float = -174.0 # physical constant at 290 K

    # Rayleigh block fading: the channel is redrawn every `coherence_tti` TTIs.
    coherence_ms: float = 10.0               # [PAPER] Section 8.5 "coherence time 10 ms"

    # Attenuated / truncated Shannon bound used to derive the CQI SNR thresholds
    # (Mogensen et al., "LTE capacity compared to the Shannon bound", VTC-Spring 2007):
    #     SE = bw_eff * log2(1 + SNR / sinr_eff)
    bw_efficiency: float = 0.88              # [3GPP-derived]
    sinr_efficiency: float = 0.72            # [3GPP-derived]
    max_spectral_efficiency: float = 7.4063  # CQI 15 of the 256QAM table


# 3GPP TS 38.214 Table 5.2.2.1-3 (CQI table 2, up to 256QAM), target BLER 0.1.
# (cqi_index, modulation_order, code_rate_x1024, spectral_efficiency_bps_per_hz)
CQI_TABLE_256QAM: Tuple[Tuple[int, int, int, float], ...] = (
    (1, 2, 78, 0.1523),
    (2, 2, 193, 0.3770),
    (3, 2, 449, 0.8770),
    (4, 4, 378, 1.4766),
    (5, 4, 490, 1.9141),
    (6, 4, 616, 2.4063),
    (7, 6, 466, 2.7305),
    (8, 6, 567, 3.3223),
    (9, 6, 666, 3.9023),
    (10, 6, 772, 4.5234),
    (11, 6, 873, 5.1152),
    (12, 8, 711, 5.5547),
    (13, 8, 797, 6.2266),
    (14, 8, 885, 6.9141),
    (15, 8, 948, 7.4063),
)


@dataclass(frozen=True)
class NumerologyConfig:
    """One 5G NR numerology.  `n_prb` is *derived* from bandwidth and SCS but is stored
    explicitly so that the value used in the paper's tables is auditable."""

    mu: int
    scs_khz: float
    bandwidth_mhz: float
    n_prb: int
    slot_ms: float                # = TTI in this package (slot-level scheduling)
    label: str

    @property
    def prb_bandwidth_hz(self) -> float:
        return 12.0 * self.scs_khz * 1e3


# [PAPER] Section 8.4 of the draft: B = 20 MHz, mu = 0, 100 PRBs.
NUM_MU0 = NumerologyConfig(mu=0, scs_khz=15.0, bandwidth_mhz=20.0, n_prb=100,
                           slot_ms=1.0, label="mu0-20MHz")
# [CHOSEN] The integrated runner is required to use a 0.5 ms TTI (mu = 1).  At 30 kHz
# SCS a 20 MHz carrier only carries 51 PRBs, which would silently halve the cell
# capacity relative to the rest of the article.  A 40 MHz carrier at mu = 1 carries
# 106 PRBs, preserving the ~100-PRB cell the article describes.  Documented in README.
NUM_MU1 = NumerologyConfig(mu=1, scs_khz=30.0, bandwidth_mhz=40.0, n_prb=106,
                           slot_ms=0.5, label="mu1-40MHz")
# [PAPER] Section 8.6 scheduling experiment: 20 PRBs, 30 kHz SCS, mu = 1.
NUM_MU1_SCHED = NumerologyConfig(mu=1, scs_khz=30.0, bandwidth_mhz=7.2, n_prb=20,
                                 slot_ms=0.5, label="mu1-sched-20prb")


# ----------------------------------------------------------------------------------
# 2. Slices and traffic
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceSpec:
    name: str
    n_users: int
    packet_bits_mean: float          # mean packet size in bits
    arrival_rate_per_user_hz: float  # Poisson rate per user at load factor rho = 1
    delay_budget_ms: float           # 3GPP 5QI packet delay budget
    min_prb: int                     # slow-loop minimum guarantee (initial value)
    utility_weight: float            # w_s in the cross-layer objective
    sla_rate_mbps: float             # per-slice aggregate SLA throughput floor
    priority: int                    # 0 = highest


# 3GPP TS 23.501 Table 5.7.4-1 standardised 5QI packet delay budgets:
#   5QI 1  (conversational voice)      100 ms
#   5QI 9  (default eMBB bearer)       300 ms  -> 50 ms used here for the buffer model
#   5QI 82 (discrete automation URLLC)  10 ms
#   5QI 85 (electricity distribution)    5 ms
# The URLLC "1 ms" figure quoted in the article is the *radio* one-way user-plane
# latency target of ITU-R M.2410-0, not an end-to-end PDB; both are modelled.
SLICES: Tuple[SliceSpec, ...] = (
    SliceSpec(name="eMBB", n_users=30, packet_bits_mean=12000.0,
              arrival_rate_per_user_hz=520.0, delay_budget_ms=50.0,
              min_prb=30, utility_weight=1.0, sla_rate_mbps=110.0, priority=2),
    SliceSpec(name="URLLC", n_users=12, packet_bits_mean=1600.0,
              arrival_rate_per_user_hz=2200.0, delay_budget_ms=1.0,
              min_prb=20, utility_weight=6.0, sla_rate_mbps=26.0, priority=0),
    SliceSpec(name="mMTC", n_users=60, packet_bits_mean=400.0,
              arrival_rate_per_user_hz=250.0, delay_budget_ms=100.0,
              min_prb=10, utility_weight=0.6, sla_rate_mbps=3.5, priority=1),
)

SLICE_NAMES: Tuple[str, ...] = tuple(s.name for s in SLICES)


@dataclass(frozen=True)
class TrafficConfig:
    """Poisson arrivals per slice plus the compressed-diurnal envelope."""

    load_sweep: Tuple[float, ...] = tuple(round(0.1 * k, 1) for k in range(1, 11))
    # Compressed 24-hour profile: 24 hourly multipliers of the nominal offered load.
    # Shape follows the canonical mobile-network diurnal curve (night trough ~03:00,
    # daytime plateau, evening peak ~21:00).  [CHOSEN] normalised so mean == 1.0.
    diurnal_hourly: Tuple[float, ...] = (
        0.42, 0.31, 0.25, 0.24, 0.28, 0.38, 0.55, 0.78,
        1.02, 1.18, 1.24, 1.26, 1.30, 1.28, 1.22, 1.20,
        1.26, 1.38, 1.52, 1.60, 1.55, 1.32, 0.95, 0.62,
    )
    # A 24-hour day is compressed into `compress_seconds` of simulated time so that
    # the 2 s slow loop sees a non-stationary target it must track.
    compress_seconds: float = 60.0        # [CHOSEN] 24 h -> 60 s  (1440x compression)
    compress_seconds_quick: float = 6.0


# ----------------------------------------------------------------------------------
# 3. Slicing / PRB allocation
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SlicingConfig:
    hard_split: Tuple[int, ...] = (50, 30, 20)      # [PAPER] eMBB / URLLC / mMTC
    soft_min: Tuple[int, ...] = (30, 20, 10)        # [PAPER] minimum guarantees
    soft_pool: int = 40                             # [PAPER] shared pool
    # Lagrangian dual (Equation 34 of the article)
    # Exponentiated (multiplicative) sub-gradient:
    #     mu <- clip( mu * exp(step * (sum_s n_s - N) / N), mu_min, mu_max )
    # Normalising the residual by the PRB budget makes the step size scale-free;
    # an additive update on an unnormalised residual oscillates.
    dual_step: float = 0.55                         # [CHOSEN] sub-gradient step size
    dual_step_decay: float = 0.995                  # [CHOSEN]
    dual_init: float = 1.0                          # [CHOSEN]
    dual_max: float = 25.0                          # [CHOSEN] multiplier clip
    n_dual_iterations: int = 60                     # [CHOSEN] per Monte Carlo draw
    dual_tolerance: float = 1e-4                    # relative primal-residual stop
    sla_proximity_gain: float = 2.5                 # [CHOSEN] weight boost per unit SLA gap


# ----------------------------------------------------------------------------------
# 4. Schedulers
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerConfig:
    n_users: int = 50                    # [PAPER] Section 8.6
    n_tti: int = 1000                    # [PAPER] 1000 TTIs
    n_tti_quick: int = 120
    snr_db_low: float = 0.0              # [PAPER] average SNR U(0, 25) dB per user
    snr_db_high: float = 25.0
    pf_ewma: float = 0.01                # [CHOSEN] PF averaging constant (1/100 TTIs)
    cell_edge_percentile: float = 5.0    # 5th percentile == cell-edge definition used
    cell_edge_fraction: float = 0.10     # worst 10 % of users = "cell edge" set
    # Learned (policy-gradient) scheduler
    learn_rate: float = 5e-3             # [CHOSEN]
    entropy_coeff: float = 1e-3          # [CHOSEN]
    hidden_units: int = 16               # [CHOSEN]
    warmup_tti: int = 200                # [CHOSEN] steps before the policy is evaluated


# ----------------------------------------------------------------------------------
# 5. Service function chain
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class SFCConfig:
    # [PAPER] Section 8.7: Firewall -> NAT -> Load Balancer -> DPI -> Proxy
    vnf_names: Tuple[str, ...] = ("Firewall", "NAT", "LoadBalancer", "DPI", "Proxy")
    # [CHOSEN] The article states the service rates inside an OMML equation object
    # that the text extraction cannot recover, and the values implied by its own
    # numbers are mutually inconsistent: a five-stage M/M/1 chain whose bottleneck
    # saturates inside a 50-350 packet/s sweep cannot deliver 0.65 ms at 200
    # packet/s, because its zero-load latency alone would be 8.8 ms.  A
    # self-consistent parameterisation is used instead, preserving the ratios
    # between the five VNFs.  The deviation is recorded in README.md and in
    # work/sim_results.md.
    service_rates_pps: Tuple[float, ...] = (5000.0, 8000.0, 6000.0, 4000.0, 7000.0)
    arrival_sweep_pps: Tuple[float, ...] = (500.0, 1000.0, 1500.0, 2000.0, 2500.0,
                                            3000.0, 3500.0, 3800.0)
    n_packets_des: int = 200_000          # [CHOSEN] packets per DES replication
    n_packets_des_quick: int = 20_000
    warmup_fraction: float = 0.10         # transient discarded before measurement
    urllc_budget_ms: float = 1.0          # [PAPER] ITU-R M.2410-0 radio budget
    e2e_budget_ms: float = 10.0           # [3GPP] TS 23.501 5QI 82 packet delay budget
    # VNF placement / scaling used by the slow loop of the integrated runner
    vnf_cpu_capacity_pps: float = 250.0   # [CHOSEN] one VNF instance = 250 pps
    vnf_max_instances: int = 8            # [CHOSEN]
    vnf_target_utilisation: float = 0.70  # [PAPER] "target 70-80 % of max load"


# ----------------------------------------------------------------------------------
# 6. Energy / cell sleeping  (Figure 6, article Section 8.9)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyConfig:
    n_oru: int = 19                       # [CHOSEN] 19-cell hexagonal cluster
    p_active_w: float = 200.0             # [PAPER] Figure 3 caption
    p_sleep_w: float = 20.0               # [PAPER] Figure 3 caption
    p_overhead_w: float = 500.0           # [PAPER] Figure 3 caption (BBU pool + cooling)
    # Load-dependent slope of the active O-RU (EARTH model, Auer et al. 2011):
    #   P = p_active_static + slope * offered_load_fraction
    p_active_static_w: float = 130.0      # [CHOSEN] consistent with p_active at full load
    p_load_slope_w: float = 70.0          # [CHOSEN] 130 + 70 = 200 W at full load
    wake_energy_j: float = 45.0           # [CHOSEN] transition (sleep -> active) cost
    wake_latency_slots: int = 1           # [CHOSEN] one 15-min slot to wake
    slot_minutes: float = 15.0            # 96 slots per day
    per_oru_capacity_mbps: float = 260.0  # [CHOSEN] from ChannelConfig at mean SE
    # Static scheduled sleep: fixed night window
    static_sleep_start_hour: float = 0.0  # [CHOSEN]
    static_sleep_end_hour: float = 6.0    # [CHOSEN]
    static_sleep_fraction: float = 0.50   # [CHOSEN] half the O-RUs sleep in the window
    # Predictive sleep: Holt double exponential smoothing (level + trend).
    # NOTE: this is *not* an LSTM.  The article's claim of "LSTM-based prediction"
    # is not reproduced here; see work/sim_results.md.
    holt_alpha: float = 0.45              # [CHOSEN] level smoothing
    holt_beta: float = 0.15               # [CHOSEN] trend smoothing
    # Safety-margin sweep: traces the energy-efficiency / throughput trade-off curve.
    margin_sweep: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30,
                                       0.45, 0.65, 0.85)
    demand_noise_cv: float = 0.12         # [CHOSEN] coefficient of variation of demand
    sla_rate_mbps: float = 100.0          # [PAPER] R_eMBB >= 100 Mbps
    sla_blocking_max: float = 0.01        # [CHOSEN] <=1 % offered load unserved


# ----------------------------------------------------------------------------------
# 7. RIS  (Figure 7, article Section 8.10)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RISConfig:
    n_bs_antennas: int = 64               # [PAPER] M = 64
    n_users: int = 10                     # [PAPER] K = 10
    n_ris_sweep: Tuple[int, ...] = (16, 32, 64, 128, 256, 512)   # [PAPER]
    bandwidth_mhz: float = 20.0           # [PAPER]
    direct_snr_db: float = -10.0          # [PAPER] SNR = -10 dB (direct link, cell edge)
    # Geometry / large-scale fading
    pl_exponent_direct: float = 3.5       # [CHOSEN] blocked BS-UE macro link
    pl_exponent_bs_ris: float = 2.2       # [CHOSEN] near-LoS BS-RIS link
    pl_exponent_ris_ue: float = 2.8       # [CHOSEN] RIS-UE link
    d_bs_ue_m: float = 200.0              # [CHOSEN]
    d_bs_ris_m: float = 180.0             # [CHOSEN]
    d_ris_ue_m: float = 40.0              # [CHOSEN] RIS deployed close to the UE cluster
    ref_pathloss_db: float = 30.0         # [CHOSEN] free-space loss at d0 = 1 m, 2 GHz
    rician_k_db: float = 10.0             # [CHOSEN] cascade Rician factor
    element_gain_db: float = 5.0          # [CHOSEN] per-element reflection gain
    # Alternating optimization
    ao_max_iter: int = 100
    ao_tolerance: float = 1e-5            # relative change in the objective
    phase_bits: int = 0                   # 0 = continuous phases; >0 = 2^bits levels
    n_realizations: int = 300             # Monte Carlo channel draws per seed
    n_realizations_quick: int = 40
    # [CHOSEN] URLLC rate requirement used for the outage definition.  A 1600-bit
    # URLLC packet delivered inside the ITU-R M.2410-0 1 ms radio budget over an
    # 8 MHz mini-slot allocation needs 1600 bit / 1 ms / 8 MHz = 0.2 bit/s/Hz.
    # Wide-bandwidth, short-duration allocation is exactly how NR mini-slot URLLC
    # transmission works, so this is the operationally meaningful threshold at the
    # cell-edge SNR of this experiment.
    urllc_rate_req_bps_per_hz: float = 0.2


# ----------------------------------------------------------------------------------
# 8. Complexity benchmark  (article Section 8.8)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComplexityConfig:
    n_sweep: Tuple[int, ...] = (10, 50, 100, 200, 500, 1000)   # [PAPER]
    n_repeats: int = 40                   # timing repetitions per (N, method, seed)
    n_repeats_quick: int = 5
    bnb_timeout_s: float = 10.0           # [CHOSEN] the article wrote ">10 000 ms"
    wf_tolerance: float = 1e-9            # water-filling bisection tolerance
    wf_max_iter: int = 200
    dqn_hidden: Tuple[int, int] = (64, 64)   # inference network used for the timing


# ----------------------------------------------------------------------------------
# 9. Learning-based controllers  (article Section 8.5)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RLEnvConfig:
    """Multi-cell power-control environment used by all four learners."""

    n_cells: int = 4                      # [PAPER] 4 cells
    n_users: int = 20                     # [PAPER] 20 users (5 per cell)
    n_power_levels: int = 3               # [CHOSEN] 3 levels -> 81 joint actions
    p_max_w: float = 20.0                 # [CHOSEN] per-cell transmit power
    bandwidth_hz: float = 20e6
    noise_dbm: float = -95.0              # [CHOSEN] in-band noise floor
    episode_steps: int = 40               # [CHOSEN]
    urllc_users_per_cell: int = 1         # [CHOSEN]
    urllc_queue_budget_bits: float = 3000.0
    urllc_penalty: float = 8.0            # lambda_URLLC of Equation (35)
    arrival_bits_mean: float = 11000.0    # [CHOSEN] per user per step while the cell is busy
    coherence_steps: int = 4              # [CHOSEN]
    # Energy term of the reward: transmitting at full power costs radio energy and
    # raises inter-cell interference, so the optimal power depends on the *state*
    # (backlog and traffic activity) rather than being uniformly maximal.
    energy_cost_weight: float = 3.0       # [CHOSEN] reward units per normalised watt
    # Per-cell on/off traffic burstiness (2-state Markov chain).  A cell that has just
    # gone idle should drop its power; that is the behaviour an agent must learn.
    burst_on_to_off: float = 0.10         # [CHOSEN]
    burst_off_to_on: float = 0.14         # [CHOSEN]  -> ~58 % duty cycle
    # Cell adjacency graph used by the GNN (ring topology over 4 cells)
    adjacency: Tuple[Tuple[int, ...], ...] = ((1, 3), (0, 2), (1, 3), (2, 0))


@dataclass(frozen=True)
class RLTrainConfig:
    n_episodes: int = 700                 # [CHOSEN] reduced from the draft's 2000 so a
    n_episodes_quick: int = 40            #          30-seed x 4-algorithm campaign fits
    gamma: float = 0.95
    lr: float = 1.5e-3
    batch_size: int = 64
    replay_capacity: int = 20_000
    warmup_transitions: int = 800
    target_update_steps: int = 200
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 300
    hidden: Tuple[int, int] = (64, 64)
    grad_clip: float = 5.0
    train_every: int = 1
    # Federated learning
    fedavg_period_episodes: int = 10      # [PAPER] aggregation every 10 episodes
    # QMIX monotonic mixer
    mixer_hidden: int = 32
    # GNN
    gnn_message_rounds: int = 2           # [PAPER] "2-layer message-passing"
    gnn_hidden: int = 48
    # Convergence detection
    convergence_window: int = 50          # episodes in the moving average
    # A run counts as having reached a plateau when the total drift of its
    # evaluation return over the final fifth of training is below this fraction of
    # the plateau level.  5 % is a conventional plateau definition and is loose
    # enough not to be dominated by evaluation noise at 8 evaluation scenarios; the
    # three sub-criteria are reported separately so a reader can apply a different
    # threshold to the same data.
    convergence_tolerance: float = 0.05


# ----------------------------------------------------------------------------------
# 10. Integrated end-to-end runner  (article Section 8.3)
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegratedConfig:
    numerology: NumerologyConfig = NUM_MU1
    fast_period_tti: int = 1              # fast loop: every TTI (0.5 ms)
    medium_period_ms: float = 100.0       # medium loop: Near-RT RIC
    slow_period_ms: float = 2000.0        # slow loop: Non-RT RIC / MANO
    nominal_load: float = 0.62            # [CHOSEN] mean offered load at diurnal mult. 1
    pf_ewma: float = 0.005                # [CHOSEN] 1/200 TTIs at 0.5 ms = 100 ms window
    # Medium loop
    medium_n_actions: int = 7             # discrete re-split actions of the RIC policy
    medium_prb_step: int = 6              # PRBs moved per re-split action
    admission_backlog_threshold: float = 3.0   # x SLA backlog -> block new sessions
    # Slow loop
    slow_price_step: float = 0.08         # [CHOSEN] dual sub-gradient step
    slow_price_max: float = 20.0
    slow_min_prb_floor: int = 8           # never guarantee fewer PRBs than this
    slow_min_prb_ceiling: int = 60
    # VNF placement / scaling handled by the slow loop.  The Section 8.7 SFC study
    # uses a small chain (hundreds of packets per second); the integrated cell carries
    # thousands of packets per second, so the per-instance capacity is scaled
    # accordingly and stated separately.
    vnf_capacity_pps: float = 1800.0      # [CHOSEN] one VNF instance
    vnf_max_instances: int = 20           # [CHOSEN]
    vnf_target_utilisation: float = 0.70  # [PAPER] "target 70-80 % of max load"
    # 3GPP TS 23.501 Table 5.7.4-1, 5QI 82 (discrete automation): PDB = 10 ms.
    # The 1 ms figure quoted in the article is the ITU-R M.2410-0 *radio* user-plane
    # target and is applied here to the radio head-of-line delay only.
    e2e_budget_urllc_ms: float = 10.0
    # Telemetry
    telemetry_ewma: float = 0.05
    # Baseline B fixed shared pool
    baseline_b_min: Tuple[int, ...] = (30, 20, 10)
    baseline_b_pool: int = 46             # 106 - 60
    # Queue / delay model
    max_backlog_bits: float = 4e7         # buffer size before drops


ARMS: Tuple[str, ...] = (
    "A_hard_static",       # static hard isolation, PF only, no outer loops
    "B_soft_pool",         # soft isolation, fixed shared pool, no learning
    "C_no_medium",         # UMRO-5G, medium loop disabled (fast + slow)
    "D_no_slow",           # UMRO-5G, slow loop disabled (fast + medium)
    "E_full_umro",         # UMRO-5G, all three loops
)

ARM_LABELS: Dict[str, str] = {
    "A_hard_static": "Baseline A: hard isolation (PF only)",
    "B_soft_pool": "Baseline B: soft isolation, fixed pool",
    "C_no_medium": "Baseline C: UMRO-5G without the medium loop",
    "D_no_slow": "Baseline D: UMRO-5G without the slow loop",
    "E_full_umro": "UMRO-5G (all three loops)",
}


# ----------------------------------------------------------------------------------
# 11. Plotting
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlotConfig:
    dpi: int = 300
    # Okabe-Ito colour-blind-safe qualitative palette (Okabe & Ito, 2008).
    palette: Tuple[str, ...] = (
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#D55E00",  # vermillion
        "#56B4E9",  # sky blue
        "#000000",  # black
    )
    markers: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
    linestyles: Tuple[str, ...] = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)),
                                   (0, (5, 1)), (0, (1, 1)))
    column_width_in: float = 3.5      # MDPI single-column width
    double_width_in: float = 7.2      # MDPI full text width
    base_font_pt: float = 8.0


# ----------------------------------------------------------------------------------
# Instances (import these)
# ----------------------------------------------------------------------------------

CAMPAIGN = CampaignConfig()
CHANNEL = ChannelConfig()
TRAFFIC = TrafficConfig()
SLICING = SlicingConfig()
SCHED = SchedulerConfig()
SFC = SFCConfig()
ENERGY = EnergyConfig()
RIS = RISConfig()
COMPLEXITY = ComplexityConfig()
RLENV = RLEnvConfig()
RLTRAIN = RLTrainConfig()
INTEGRATED = IntegratedConfig()
PLOT = PlotConfig()


def seeds(n: int | None = None) -> List[int]:
    """The seed list.  Common random numbers: every arm of every experiment is driven
    by the same seed list, which is what makes the paired statistical tests valid."""
    n = CAMPAIGN.n_seeds if n is None else n
    return [CAMPAIGN.seed_base + i for i in range(n)]


def snapshot() -> Dict[str, dict]:
    """Serialisable record of every parameter, written next to every result file."""
    return {
        "campaign": asdict(CAMPAIGN),
        "channel": asdict(CHANNEL),
        "traffic": asdict(TRAFFIC),
        "slicing": asdict(SLICING),
        "scheduler": asdict(SCHED),
        "sfc": asdict(SFC),
        "energy": asdict(ENERGY),
        "ris": asdict(RIS),
        "complexity": asdict(COMPLEXITY),
        "rl_env": asdict(RLENV),
        "rl_train": asdict(RLTRAIN),
        "integrated": asdict(INTEGRATED),
        "slices": [asdict(s) for s in SLICES],
        "numerologies": {n.label: asdict(n)
                         for n in (NUM_MU0, NUM_MU1, NUM_MU1_SCHED)},
        "cqi_table": [list(r) for r in CQI_TABLE_256QAM],
    }
