# UMRO-5G simulation package

Executable code, configuration, dependencies, raw results and figure-generation
scripts for the simulation section of *UMRO-5G: A Unified Framework for Management
and Resource Orchestration*.

Everything reported in the article's Section 8 is produced by this package. Nothing
in `results/` is hand-edited, and nothing in the article's Section 8 is a number that
does not appear in `results/` or in `../work/sim_results.md`.

---

## 1. Quick start

```bash
python -m pip install -r requirements.txt

# smoke run, about 1.5 minutes: 4 seeds, short horizons
python run_all.py --quick --out results_quick

# full campaign, about 16 minutes on 24 cores: 30 seeds
python run_all.py --seeds 30 --out results --workers 24

# regenerate every figure from the committed results
python plot_figures.py --results results --figures ../figures

# regenerate the human-readable report the article is written from
python make_report.py --results results --out ../work/sim_results.md

# verify the models
python -m pytest tests -q
```

Run the commands from inside `simulations/`. `run_all.py` and `plot_figures.py` both
resolve paths relative to the working directory.

Useful flags:

| Flag | Meaning |
|---|---|
| `--seeds N` | number of independent replications (default 30) |
| `--out DIR` | output directory for JSON/CSV (default `results`) |
| `--quick` | smoke run: 4 seeds, 60 Monte Carlo iterations, 6 s compressed day, 40 RL episodes |
| `--only a,b,c` | run a subset of the stages (`pretrain,integrated,slicing,learners,scheduling,sfc,energy,ris,complexity`) |
| `--workers K` | worker processes (default 24) |

---

## 2. Directory map

```
simulations/
  README.md            this file
  requirements.txt     pinned dependencies
  config.py            EVERY scenario parameter, as frozen dataclasses
  channel.py           path loss + shadowing + Rayleigh block fading -> CQI/MCS -> spectral efficiency
  traffic.py           Poisson per-slice arrivals; the compressed 24 h diurnal envelope
  slicing.py           hard / soft / Lagrangian-dual PRB allocators + the multi-slice Monte Carlo
  scheduler.py         Round Robin, Max-Rate, Proportional Fair, and a policy-gradient scheduler
  sfc.py               Jackson M/M/1 tandem analytics + an exact FIFO discrete-event simulator
  energy.py            O-RU power model; always-on / static / predictive sleeping
  ris.py               RIS channel; no-RIS / random-phase / alternating optimization
  complexity.py        wall-clock decision-time benchmark (learned inference, water-filling, branch and bound)
  learners.py          real DQN, QMIX, FedAvg and GNN-DRL agents in NumPy (hand-written backprop)
  integrated.py        the end-to-end three-loop UMRO-5G runner + its five ablation arms
  stats.py             per-seed aggregation, t confidence intervals, paired tests, corrections, power
  run_all.py           campaign driver (multiprocessing over seeds)
  plot_figures.py      regenerates every figure from results/
  make_report.py       regenerates ../work/sim_results.md from results/
  tests/               pytest unit tests (124 tests)
  results/             committed raw output of the full 30-seed run
../figures/            300 dpi PNGs consumed by the article
../work/sim_results.md the human-readable report the writing agents consume
```

### What lands in `results/`

| File | Content |
|---|---|
| `run_manifest.json` | timestamp, wall-clock per stage, seed list, platform, library versions |
| `config_snapshot.json` | every parameter used, serialised from `config.py` |
| `medium_policy.json` | pre-trained Near-RT RIC Q-network weights + its training and validation curves |
| `integrated.json` | per-seed values, summaries, paired-test families, coupling statistics |
| `integrated_trace_seed0.json` | full medium-loop trace of the first seed (prices, guarantees, allocations) |
| `slicing.json`, `learners.json`, `scheduling.json`, `sfc.json`, `energy.json`, `ris.json`, `complexity.json` | one file per experiment |
| `*.csv` | the same content flattened for spreadsheet inspection |

---

## 3. Seed handling and the sample-size rule

**The unit of independent replication is the seed.**

`config.seeds()` returns `seed_base + i` for `i = 0 .. n-1` with `seed_base = 20250001`.
The *same* seed list drives *every arm of every experiment* (common random numbers), so:

* arm-to-arm comparisons are **paired** at the seed level;
* the paired t-test, the Wilcoxon signed-rank test and Cohen's `d_z` in `stats.py` are
  applied to the seed-wise difference vector;
* the degrees of freedom are `n_seeds - 1 = 29`.

Each seed internally runs `n_monte_carlo` iterations (1000 by default) and **averages
them into a single number**. That averaging reduces the variance of that seed's point
estimate; it does **not** create independent samples. The effective sample size is
therefore 30, not 30 000. `stats.describe()` records `n_monte_carlo_per_seed` as a
separate field precisely so this distinction stays visible in the raw output, and
`tests/test_stats.py::test_degrees_of_freedom_are_seeds_minus_one_not_iterations`
asserts that changing it cannot change the inference.

Within a seed, every random stream is derived deterministically from that seed, so a
run is bit-for-bit reproducible on the same NumPy version. Determinism is asserted in
`tests/test_channel.py`, `tests/test_integrated.py`, `tests/test_sfc.py` and
`tests/test_energy.py`.

Two stages deviate and say so:

* **`complexity`** uses 10 seeds instead of 30. It measures wall-clock decision time,
  so it is run **sequentially** on an otherwise idle machine; running 24 replications
  concurrently would measure CPU contention rather than algorithmic cost. All
  confidence intervals for that stage carry `n = 10`, `df = 9`.
* **`pretrain`** is a single run, not a replicated one. It produces the Near-RT RIC
  policy that all 30 evaluation seeds then share unchanged (see Section 5).

---

## 4. Expected runtime

Reference machine: Windows 11, 24 logical cores, Python 3.13.0, NumPy 2.4.4.
BLAS is pinned to one thread per process by `run_all.py`.

Measured on the committed run; `results/run_manifest.json` carries the authoritative
per-stage timings of whichever run produced the committed results.

| Stage | Tasks | Wall clock (full run) |
|---|---|---|
| `pretrain` | 1 (sequential) | 56 s |
| `integrated` | 30 seeds x 5 arms | 151 s |
| `slicing` | 30 seeds x 10 loads x 1000 iterations x 3 strategies | 12 s |
| `learners` | 4 algorithms x 30 seeds x 700 episodes | 379 s |
| `scheduling` | 30 seeds x 4 algorithms x 1000 TTIs | 1 s |
| `sfc` | 30 seeds x 8 arrival rates x 200 000 packets | 1 s |
| `energy` | 30 seeds x 8 demand points x 3 strategies x 96 slots | < 1 s |
| `ris` | 30 seeds x 6 element counts x 300 realisations x 10 UEs | 24 s |
| `complexity` | 10 seeds x 6 sizes (**sequential**) | 325 s |
| **total** | | **about 16 minutes** |

`--quick` completes in about 90 s, of which about 70 s is the sequential complexity
benchmark (its branch-and-bound timeouts are wall-clock and cannot be shortened
without changing the measurement). Use
`--only pretrain,integrated,slicing,learners,scheduling,sfc,energy,ris` to skip the
timing benchmark; the rest of the campaign then takes about 10 minutes.

---

## 5. What each experiment does

### 5.1 Integrated end-to-end evaluation (`integrated.py`, Figure 4)

A single slotted simulation in which four layers and three nested loops share one
clock and one traffic realisation.

* **Fast loop, every TTI (0.5 ms, numerology mu = 1)** — proportional-fair scheduling
  inside each slice's own PRBs, per-user rate from the CQI/MCS table, per-slice FIFO
  queues, and an *exact* head-of-line delay obtained by inverting the cumulative
  arrival curve against the cumulative service curve.
* **Medium loop, every 100 ms (Near-RT RIC)** — a pre-trained Q-network reads the
  telemetry state (per-slice allocation, backlog, SLA ratio, dual price, time of day)
  and picks one of seven PRB re-split actions; admission control throttles a slice
  whose backlog exceeds its SLA-proximity threshold.
* **Slow loop, every 2 s (Non-RT RIC / MANO)** — an exponentiated sub-gradient step on
  the per-slice dual price from the measured demand-allocation residual, re-derivation
  of the minimum PRB guarantees from the price-modulated demand, and VNF placement and
  scaling for each slice's service function chain.

**The loops are coupled, and the coupling is measured, not asserted.** The slow loop
emits prices and guarantees; the medium loop may only apply an action that is feasible
under those guarantees *and* under the slow loop's price budget (rejected actions are
counted); the medium loop's partition is the hard limit on what the fast loop can
schedule; and the fast loop's measured throughput, backlog and delay are the telemetry
both outer loops consume. `coupling_statistics()` reports the correlations along that
chain from the recorded trace, which is written to
`results/integrated_trace_seed0.json`.

Five arms run on identical channel and traffic streams:

| Arm | Fast | Medium | Slow |
|---|---|---|---|
| `A_hard_static` | yes | no | no |
| `B_soft_pool` | yes | fixed-pool rule, no learning | no |
| `C_no_medium` | yes | **no** | yes |
| `D_no_slow` | yes | yes | **no** |
| `E_full_umro` | yes | yes | yes |

C and D are the ablations that turn "the framework was evaluated as a whole" into a
measurement of what each loop contributes.

### 5.2 Learning-based controllers (`learners.py`, Figure 5)

Four **real** agents, implemented in NumPy with hand-derived gradients that are
checked against central finite differences in `tests/test_learners.py`:

* `DQN` — centralised Deep Q-Network over the joint action space (3^4 = 81 actions),
  uniform experience replay, separate target network, epsilon-greedy with linear
  decay, Huber loss, gradient clipping;
* `MADRL-QMIX` — one Q-network per cell plus a QMIX monotonic mixing network whose
  weights come from hypernetworks conditioned on the global state and are forced
  non-negative, guaranteeing `dQ_tot / dQ_i >= 0` (asserted in the tests);
* `FedAvg-DQN` — one local DQN per cell with its own replay buffer, uniform parameter
  averaging every 10 episodes;
* `GNN-DRL` — two rounds of normalised-adjacency message passing over the cell graph
  with shared self/neighbour weights, then a shared per-node Q head.

There are **no tabular proxy agents anywhere in this package.** The environment is a
four-cell interference-limited downlink with per-user queues, bursty on/off traffic and
an explicit energy cost, so the optimal transmit power genuinely depends on the state.

Convergence is *measured*, not assumed. `convergence_diagnostics()` requires three
conditions simultaneously (improvement over the random-policy floor by more than one
plateau standard deviation, a relative slope below 2 % over the last fifth of
training, and progress relative to the first fifth). An agent that fails any of them
is reported as **not converged**.

The headline metric is a **normalised score**:

```
score = (return - random_policy_return) / (myopic_oracle_return - random_policy_return)
```

`0` is uniformly random power control and `1` is a per-slot greedy oracle that has
full instantaneous per-user CSI, which the agents do not. The oracle is therefore an
upper reference, not the optimum of the Markov decision process, and the article must
describe it that way.

Training and evaluation use **disjoint seed streams**: training scenarios come from
`seed`, evaluation from `900000 + seed`, and the eight evaluation scenarios are held
fixed for the whole run so the learning curve measures policy improvement rather than
scenario noise.

### 5.3 Multi-slice Monte Carlo (`slicing.py`)

Reproduces the article's Section 8.4: a 100-PRB, 20 MHz, mu = 0 cell, offered load
swept from 0.1 to 1.0, three allocators (hard isolation, soft isolation with a fixed
pool, Lagrangian dual). Within a load point all three allocators replay the *same*
demand and channel draws.

### 5.4 Scheduling comparison (`scheduler.py`)

50 users, 20 PRBs at 30 kHz, mean SNR uniform on 0-25 dB, 1000 TTIs. All four
schedulers consume the *same* pre-generated spectral-efficiency trace, so the
comparison is paired at the TTI level as well as at the seed level. Reported metrics:
aggregate throughput, Jain fairness, 5th-percentile user rate, and the mean rate of the
worst 10 % of users ("cell edge").

### 5.5 SFC latency (`sfc.py`)

A five-VNF open tandem of M/M/1 queues. The analytical arm is the Jackson-network mean
plus the exact hypoexponential tail; the simulation arm is a FIFO discrete-event
simulator built on the Lindley recursion, which is **exact** rather than approximate
(the vectorisation removes the Python event loop, not the events).

### 5.6 Energy and adaptive cell sleeping (`energy.py`, Figure 6)

Nineteen O-RUs, 96 fifteen-minute slots, the EARTH load-dependent power model, an
explicit wake-up energy cost, and three strategies. The demand envelope is swept to
trace the energy-efficiency/throughput curve, and the predictive strategy's safety
margin is swept separately.

**The forecaster is Holt's linear-trend exponential smoothing. It is NOT an LSTM.**
The article's current claim of "LSTM-based traffic prediction" is not reproduced by
this code and must be reworded.

### 5.7 RIS (`ris.py`, Figure 7)

M = 64 gNB antennas, K = 10 UEs, N_RIS in {16, 32, 64, 128, 256, 512}. Direct
gNB-UE path (Rayleigh, exponent 3.5) plus the gNB-RIS-UE cascade (Rician with a 10 dB
factor, exponents 2.2 and 2.8). Three arms: no RIS, random phases, and alternating
optimization (MRT active beamformer / closed-form per-element co-phasing, iterated
until the relative change of the effective channel gain falls below `1e-5` or 100
iterations, both reported).

Stated assumptions, reproduced in the article: **perfect instantaneous CSI**
(so every RIS number is an upper bound), continuous phase shifts by default, narrowband
flat fading, and one scheduled UE per resource.

### 5.8 Complexity benchmark (`complexity.py`)

`time.perf_counter` around three real decision procedures at N = 10 .. 1000:
a forward pass of the policy network, bisection water-filling, and exact branch and
bound on a strongly correlated binary knapsack (the classical hard family). A 10 s
wall-clock timeout is enforced; timed-out entries are flagged and their times are
reported as **lower bounds**. The speedup factors are ratios of measured medians, not
assertions.

---

## 6. Choices made where the article was ambiguous

Every parameter in `config.py` carries a `[3GPP]`, `[PAPER]` or `[CHOSEN]` marker.
The `[CHOSEN]` ones that materially affect a reported number are:

| Choice | Value | Why |
|---|---|---|
| Integrated-runner numerology | 40 MHz, mu = 1, 106 PRBs | The 0.5 ms TTI requires mu = 1. At 30 kHz a 20 MHz carrier holds only 51 PRBs, which would silently halve the cell relative to the rest of the article; 40 MHz restores the ~100-PRB cell the article describes. Section 8.4 keeps the article's 20 MHz / mu = 0 / 100 PRB configuration. |
| CQI SNR thresholds | derived, not tabulated | Obtained by inverting the attenuated-and-truncated Shannon bound `SE = 0.88 log2(1 + SNR/0.72)` (Mogensen et al., VTC-Spring 2007) at each CQI's spectral efficiency, giving a monotone, reproducible mapping rather than an unsourced threshold table. |
| Day compression | 24 h into 60 s | Gives the 2 s slow loop 30 updates and the 100 ms medium loop 600 updates over a full diurnal cycle at an affordable 120 000 TTIs per arm per seed. |
| SLA definition (integrated) | carry >= 95 % of the offered load within the 5QI delay budget, per 100 ms window | A fixed absolute rate target is meaningless under a diurnal envelope that swings by a factor of six. |
| URLLC budgets | 1 ms radio, 10 ms end-to-end | 1 ms is the ITU-R M.2410-0 *radio* user-plane target, not an end-to-end packet delay budget; the end-to-end figure is the 3GPP TS 23.501 Table 5.7.4-1 5QI 82 PDB. Reviewer 2 raised exactly this point. |
| SFC service rates | (5000, 8000, 6000, 4000, 7000) packets/s, lambda swept 500-3800 | The article's rates live inside an OMML equation object that text extraction cannot recover, and its own reported numbers are mutually inconsistent: a five-stage M/M/1 chain that saturates inside a 50-350 packet/s sweep has a zero-load latency of 8.8 ms and cannot deliver the article's stated 0.65 ms at 200 packet/s. The ratios between the five VNFs are preserved. |
| Dual update | exponentiated sub-gradient on the budget-normalised residual | An additive update on an unnormalised residual oscillates instead of converging. |
| Medium-loop pre-training | fluid surrogate, then frozen | See Section 7. |
| VNF instance capacity (integrated) | 1800 packets/s, up to 20 instances | The Section 8.7 chain carries hundreds of packets per second; the integrated cell carries tens of thousands. Reported separately from the Section 8.7 parameters. |
| Predictive-sleep safety margin | 65 % | Chosen by a stated rule: the smallest margin in the swept set for which every seed satisfies the SLA at the nominal demand point. Smaller margins give higher energy efficiency but violate the blocking limit; the full margin sweep is reported so the trade-off is visible rather than hidden behind one point. |
| URLLC rate requirement (RIS study) | 0.2 bit/s/Hz | A 1600-bit URLLC packet delivered inside the 1 ms radio budget over an 8 MHz mini-slot allocation. |
| RL convergence criterion | improvement + <= 5 % plateau drift + progress | All three must hold; the three sub-criteria are also reported separately so a different stability threshold can be applied to the same data. |
| Complexity replications | 10 seeds | Sequential wall-clock measurement, see Section 3. |
| RL episodes | 700 | Reduced from the draft's 2000 so that a 4-algorithm x 30-seed campaign fits the runtime budget. Reported, and the convergence diagnostics state explicitly whether 700 was enough for each algorithm. |

---

## 7. Honesty notes (read before writing about these results)

1. **The medium-loop controller is pre-trained on a fluid surrogate.**
   `MediumSurrogateEnv` replaces the fast loop by its deterministic fluid equivalent at
   the 100 ms grid, with the per-PRB rate calibrated (1.9-2.4 Mbit/s per PRB) against
   the full TTI-level simulator. The resulting Q-network is then evaluated **unchanged**
   on the full simulator by all 30 seeds. This gives clean train/test separation, but it
   also means the surrogate-to-simulator gap is part of the measured result. The
   surrogate returns of the learned, hold-everything and random policies are all stored
   in `results/medium_policy.json` so the reader can see how much the pre-training
   actually achieved.
2. **The learned scheduler is a policy-gradient (REINFORCE) scheduler, not a DQN.**
   It is genuinely trained online and is given the objective rather than the rule, but
   the article must not call it "DRL scheduler" without qualification.
3. **The forecaster in `energy.py` is Holt exponential smoothing, not an LSTM.**
4. **The RIS results assume perfect CSI** and are therefore upper bounds.
5. **The branch-and-bound decision times at large N are timeouts**, so both those times
   and the speedups computed from them are lower bounds.
6. **The "% of optimum" normaliser for the RL agents is a myopic full-CSI oracle**, not
   the optimum of the Markov decision process.
7. Any comparison that does not survive the Bonferroni correction is reported in
   `../work/sim_results.md` as **not significant**, together with the wording the
   article should use instead.

---

## 8. Reproducing the exact committed results

```bash
python -m pip install -r requirements.txt      # exact pins
python run_all.py --seeds 30 --out results --workers 24
python plot_figures.py --results results --figures ../figures
python -m pytest tests -q
```

`results/run_manifest.json` records the platform, the Python and library versions, the
seed list and the per-stage wall clock of the committed run. Numbers are bit-for-bit
reproducible on the same NumPy version; a different NumPy minor version can change the
last digits of the Monte Carlo estimates (the RNG algorithms are stable, but reduction
orders in vectorised operations are not guaranteed across versions). The conclusions,
confidence intervals and significance verdicts are not sensitive to that.
