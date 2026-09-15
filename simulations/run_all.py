"""
run_all.py -- Campaign driver.

    python run_all.py --seeds 30 --out results
    python run_all.py --quick                      (smoke run, ~1 minute)
    python run_all.py --only integrated,learners   (subset of the stages)

Design notes
------------
* The unit of independent replication is the SEED.  Every stage maps the same seed
  list onto its workers, so all arms of all experiments share common random numbers
  and the paired tests in stats.py are valid.
* Work is distributed over seeds with `multiprocessing`, using the "spawn" start
  method so the driver behaves identically on Windows and POSIX.
* BLAS is pinned to a single thread in the parent and in every worker.  Without this,
  24 workers x N BLAS threads oversubscribes the machine and the complexity
  benchmark's timings become meaningless.
* The COMPLEXITY stage is deliberately NOT parallelised: it measures wall-clock
  decision time, and running 24 of them at once would measure contention instead.
  It therefore uses fewer replications; that reduction is reported.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import platform
import sys
import time
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

import config as C
from config import CAMPAIGN, TRAFFIC, ARMS, SLICE_NAMES, seeds as seed_list

# Complexity replication count: the timing stage runs sequentially, so it uses fewer
# replications than the rest of the campaign.  Documented in README.md and reported
# in work/sim_results.md.
N_SEEDS_COMPLEXITY = 10
N_SEEDS_COMPLEXITY_QUICK = 2

STAGES = ("pretrain", "integrated", "slicing", "learners", "scheduling",
          "sfc", "energy", "ris", "complexity")


# ==================================================================================
# Worker entry points (module level so they pickle under the spawn start method)
# ==================================================================================

def _init_worker() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def work_integrated(args: tuple) -> dict:
    from integrated import run_integrated_seed
    seed, params, compress, trace = args
    return run_integrated_seed(seed, params, compress, record_trace=trace)


def work_slicing(args: tuple) -> dict:
    from slicing import run_multislice_seed
    from config import NUM_MU0, TRAFFIC as T
    seed, n_mc = args
    return run_multislice_seed(seed, NUM_MU0.n_prb, NUM_MU0.slot_ms,
                               NUM_MU0.prb_bandwidth_hz, T.load_sweep, n_mc)


def work_learners(args: tuple) -> dict:
    from learners import train_agent
    algo, seed, n_episodes = args
    return train_agent(algo, seed, n_episodes=n_episodes)


def work_scheduling(args: tuple) -> dict:
    from scheduler import run_scheduling_seed
    seed, n_tti = args
    return run_scheduling_seed(seed, n_tti)


def work_sfc(args: tuple) -> dict:
    from sfc import run_sfc_seed
    seed, n_packets = args
    return run_sfc_seed(seed, n_packets)


def work_energy(args: tuple) -> dict:
    from energy import run_energy_seed
    (seed,) = args
    return run_energy_seed(seed)


def work_ris(args: tuple) -> dict:
    from ris import run_ris_seed
    seed, n_real = args
    return run_ris_seed(seed, n_real)


# ==================================================================================
# Aggregation
# ==================================================================================

def _per_seed(records: List[dict], getter: Callable[[dict], float]) -> List[float]:
    return [float(getter(r)) for r in records]


def aggregate_integrated(records: List[dict], n_mc: int,
                         trace: dict | None = None) -> dict:
    import stats as S
    from integrated import coupling_statistics

    metrics = ["aggregate_throughput_mbps", "prb_utilisation",
               "urllc_violation_prob", "urllc_e2e_violation_prob",
               "sla_satisfaction_rate", "carried_load_fraction",
               "urllc_mean_e2e_latency_ms", "jain_across_slices",
               "medium_actions_applied", "medium_actions_rejected",
               "slow_updates"]
    for name in SLICE_NAMES:
        metrics += [f"throughput_{name}_mbps", f"mean_hol_delay_{name}_ms",
                    f"sla_satisfaction_{name}", f"mean_sfc_latency_{name}_ms"]

    per_seed: Dict[str, Dict[str, List[float]]] = {}
    for m in metrics:
        per_seed[m] = {arm: _per_seed(records, lambda r, a=arm, k=m: r["arms"][a][k])
                       for arm in ARMS}

    summaries = {m: S.summarise_arms(per_seed[m], m, n_mc) for m in metrics}

    # Family of comparisons: every arm against the full framework, for the three
    # headline metrics.  One family -> one Bonferroni correction.
    family_metrics = ["sla_satisfaction_rate", "aggregate_throughput_mbps",
                      "prb_utilisation"]
    comps = []
    for m in family_metrics:
        comps += S.all_pairwise(per_seed[m], m, reference="E_full_umro",
                                higher_is_better=True)
    family = S.build_family(comps, family_name="integrated_end_to_end")

    # URLLC violation probability: lower is better, own family
    viol = S.build_family(
        S.all_pairwise(per_seed["urllc_violation_prob"], "urllc_violation_prob",
                       reference="E_full_umro", higher_is_better=False),
        family_name="integrated_urllc_violation")

    normality = {m: S.normality_check(
        np.array(per_seed[m]["E_full_umro"]) - np.array(per_seed[m]["A_hard_static"]))
        for m in family_metrics}

    coupling = coupling_statistics(trace) if trace else None

    return {"per_seed": per_seed, "summaries": summaries, "family": family,
            "family_urllc": viol, "normality": normality, "coupling": coupling,
            "n_seeds": len(records)}


def aggregate_slicing(records: List[dict], n_mc: int) -> dict:
    import stats as S
    from slicing import STRATEGIES
    loads = records[0]["loads"]
    out: dict = {"loads": loads, "per_load": {}, "n_seeds": len(records)}
    families = []
    for li, load in enumerate(loads):
        block: dict = {}
        for metric, better in (("throughput_mbps", True), ("utilisation", True),
                               ("urllc_violation", False)):
            per = {st: [r["strategies"][st][metric][li] for r in records]
                   for st in STRATEGIES}
            block[metric] = {
                "summary": S.summarise_arms(per, metric, n_mc),
                "per_seed": per,
            }
            if abs(load - 1.0) < 1e-9 or abs(load - 0.8) < 1e-9:
                families += S.all_pairwise(per, f"{metric}@rho={load}",
                                           reference="umro", higher_is_better=better)
        out["per_load"][f"{load}"] = block
    out["family"] = S.build_family(families, family_name="multislice_high_load")
    out["dual_converged_fraction"] = float(np.mean(
        [np.mean(r["strategies"]["umro"]["dual_converged"]) for r in records]))
    return out


def aggregate_learners(records: List[dict], n_mc: int) -> dict:
    import stats as S
    from learners import ALGORITHMS
    by_algo: Dict[str, List[dict]] = {a: [] for a in ALGORITHMS}
    for r in records:
        by_algo[r["algorithm"]].append(r)

    per_seed_score = {a: [r["normalised_score"] for r in by_algo[a]]
                      for a in ALGORITHMS}
    per_seed_ep95 = {a: [float(r["diagnostics"]["episodes_to_95pct"])
                         for r in by_algo[a]] for a in ALGORITHMS}
    converged = {a: float(np.mean([r["diagnostics"]["converged"] for r in by_algo[a]]))
                 for a in ALGORITHMS}
    drift = {a: [abs(r["diagnostics"]["relative_slope"]) for r in by_algo[a]]
             for a in ALGORITHMS}
    improved = {a: float(np.mean([r["diagnostics"]["improved_over_random"]
                                  for r in by_algo[a]])) for a in ALGORITHMS}
    stable = {a: float(np.mean([r["diagnostics"]["stable"] for r in by_algo[a]]))
              for a in ALGORITHMS}

    curves = {}
    for a in ALGORITHMS:
        arr = np.array([r["normalised_score_curve"] for r in by_algo[a]])
        eps = by_algo[a][0]["eval_episodes"]
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0, ddof=1)
        n = arr.shape[0]
        from scipy import stats as sps
        hw = sps.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        curves[a] = {"episodes": eps, "mean": mean.tolist(), "sd": sd.tolist(),
                     "ci_halfwidth": hw.tolist(), "n_seeds": int(n)}

    comps = S.all_pairwise(per_seed_score, "normalised_score",
                           reference="GNN-DRL", higher_is_better=True)
    family = S.build_family(comps, family_name="learner_final_score")
    return {"summaries": S.summarise_arms(per_seed_score, "normalised_score", n_mc),
            "plateau_drift": S.summarise_arms(drift, "abs_relative_slope", n_mc),
            "episodes_to_95pct": S.summarise_arms(per_seed_ep95,
                                                  "episodes_to_95pct", n_mc),
            "converged_fraction": converged,
            "improved_over_random_fraction": improved,
            "stable_fraction": stable,
            "curves": curves, "family": family,
            "per_seed": per_seed_score,
            "random_floor": float(np.mean([r["random_floor"] for r in records])),
            "max_power_reference": float(np.mean([r["max_power_reference"]
                                                  for r in records])),
            "n_seeds": len(by_algo[ALGORITHMS[0]])}


def aggregate_scheduling(records: List[dict], n_mc: int) -> dict:
    import stats as S
    from scheduler import ALGORITHMS
    metrics = ("throughput_mbps", "jain", "p5_mbps", "cell_edge_mbps",
               "mean_user_mbps", "starved_users")
    per_seed = {m: {a: [r["algorithms"][a][m] for r in records] for a in ALGORITHMS}
                for m in metrics}
    comps = []
    for m, better in (("throughput_mbps", True), ("jain", True),
                      ("p5_mbps", True), ("cell_edge_mbps", True)):
        comps += S.all_pairwise(per_seed[m], m, reference="proportional_fair",
                                higher_is_better=better)
    return {"summaries": {m: S.summarise_arms(per_seed[m], m, n_mc) for m in metrics},
            "per_seed": per_seed,
            "family": S.build_family(comps, family_name="scheduling"),
            "n_seeds": len(records)}


def aggregate_sfc(records: List[dict], n_mc: int) -> dict:
    import stats as S
    lambdas = [row["lambda_pps"] for row in records[0]["rows"]]
    out = {"lambdas": lambdas, "rows": [], "n_seeds": len(records)}
    for i, lam in enumerate(lambdas):
        ana = [r["rows"][i]["analytical_ms"] for r in records]
        des = [r["rows"][i]["des_ms"] for r in records]
        err = [r["rows"][i]["rel_error_pct"] for r in records]
        if any(v is None for v in ana):
            out["rows"].append({"lambda_pps": lam, "stable": False})
            continue
        cmp_ = S.paired_compare(des, ana, "mean_latency_ms", "DES", "analytical")
        out["rows"].append({
            "lambda_pps": lam,
            "stable": True,
            "rho_bottleneck": records[0]["rows"][i]["rho_bottleneck"],
            "analytical": S.describe(ana, "analytical_ms", n_monte_carlo_per_seed=n_mc).as_dict(),
            "des": S.describe(des, "des_ms", n_monte_carlo_per_seed=n_mc).as_dict(),
            "rel_error_pct": S.describe(err, "rel_error_pct").as_dict(),
            "paired_test": cmp_.as_dict(),
            "analytical_violation": S.describe(
                [r["rows"][i]["analytical_violation"] for r in records],
                "analytical_violation").as_dict(),
            "des_violation": S.describe(
                [r["rows"][i]["des_violation"] for r in records],
                "des_violation").as_dict(),
        })
    out["max_lambda_for_1ms_pps"] = float(np.mean(
        [r["max_lambda_for_1ms_pps"] for r in records]))
    out["bottleneck_mu_pps"] = records[0]["bottleneck_mu_pps"]
    return out


def aggregate_energy(records: List[dict], n_mc: int) -> dict:
    import stats as S
    from energy import STRATEGIES, DEMAND_SCALES
    from config import ENERGY
    out: dict = {"demand_scales": list(DEMAND_SCALES), "points": {}, "n_seeds": len(records)}
    for si, scale in enumerate(DEMAND_SCALES):
        block = {}
        for st in STRATEGIES:
            block[st] = {
                k: S.describe([r["curves"][st][si][k] for r in records], k).as_dict()
                for k in ("energy_efficiency_mbit_per_j", "mean_throughput_mbps",
                          "energy_kwh", "mean_active_oru", "max_blocking",
                          "forecast_mape_pct", "spectral_efficiency_bps_hz")
            }
            block[st]["sla_feasible_fraction"] = float(np.mean(
                [r["curves"][st][si]["sla_feasible"] for r in records]))
        out["points"][f"{scale}"] = block

    # Headline family: predictive vs the two baselines at the nominal demand point
    nominal = list(DEMAND_SCALES).index(1.00)
    per_seed = {st: [r["curves"][st][nominal]["energy_efficiency_mbit_per_j"]
                     for r in records] for st in STRATEGIES}
    thr_seed = {st: [r["curves"][st][nominal]["mean_throughput_mbps"]
                     for r in records] for st in STRATEGIES}
    comps = S.all_pairwise(per_seed, "energy_efficiency_mbit_per_j",
                           reference="predictive_sleep", higher_is_better=True)
    comps += S.all_pairwise(thr_seed, "mean_throughput_mbps",
                            reference="predictive_sleep", higher_is_better=True)
    out["family"] = S.build_family(comps, family_name="energy_nominal_point")
    out["per_seed_ee_nominal"] = per_seed
    out["per_seed_throughput_nominal"] = thr_seed
    out["margin_sweep"] = [
        {"margin": ENERGY.margin_sweep[j],
         "ee": S.describe([r["margin_sweep"][j]["energy_efficiency_mbit_per_j"]
                           for r in records], "ee").as_dict(),
         "throughput": S.describe([r["margin_sweep"][j]["mean_throughput_mbps"]
                                   for r in records], "thr").as_dict(),
         "max_blocking": S.describe([r["margin_sweep"][j]["max_blocking"]
                                     for r in records], "block").as_dict(),
         "sla_feasible_fraction": float(np.mean(
             [r["margin_sweep"][j]["sla_feasible"] for r in records]))}
        for j in range(len(ENERGY.margin_sweep))]
    return out


def aggregate_ris(records: List[dict], n_mc: int) -> dict:
    import stats as S
    from ris import ARMS as RIS_ARMS
    from config import RIS as RISCFG
    out: dict = {"n_ris_sweep": list(RISCFG.n_ris_sweep), "points": {},
                 "n_seeds": len(records),
                 "assumptions": records[0]["assumptions"]}
    families = []
    for i, n_ris in enumerate(RISCFG.n_ris_sweep):
        block = {}
        for arm in RIS_ARMS:
            block[arm] = {
                "se_bps_hz": S.describe([r["rows"][i][f"{arm}_se_bps_hz"]
                                         for r in records], "se", n_monte_carlo_per_seed=n_mc).as_dict(),
                "throughput_mbps": S.describe([r["rows"][i][f"{arm}_throughput_mbps"]
                                               for r in records], "thr").as_dict(),
                "urllc_violation": S.describe([r["rows"][i][f"{arm}_urllc_violation"]
                                               for r in records], "viol").as_dict(),
            }
        block["ao_mean_iterations"] = S.describe(
            [r["rows"][i]["ao_mean_iterations"] for r in records], "iters").as_dict()
        block["ao_converged_fraction"] = float(np.mean(
            [r["rows"][i]["ao_converged_fraction"] for r in records]))
        block["ao_gain_pct"] = S.describe(
            [r["rows"][i]["ao_gain_pct_vs_no_ris"] for r in records], "gain").as_dict()
        block["random_gain_pct"] = S.describe(
            [r["rows"][i]["random_gain_pct_vs_no_ris"] for r in records], "gain").as_dict()
        out["points"][str(n_ris)] = block
        if n_ris in (64, 512):
            per = {arm: [r["rows"][i][f"{arm}_se_bps_hz"] for r in records]
                   for arm in RIS_ARMS}
            families += S.all_pairwise(per, f"se_bps_hz@N={n_ris}",
                                       reference="alternating_opt",
                                       higher_is_better=True)
            perv = {arm: [r["rows"][i][f"{arm}_urllc_violation"] for r in records]
                    for arm in RIS_ARMS}
            families += S.all_pairwise(perv, f"urllc_violation@N={n_ris}",
                                       reference="alternating_opt",
                                       higher_is_better=False)
    out["family"] = S.build_family(families, family_name="ris")
    return out


def aggregate_complexity(records: List[dict]) -> dict:
    import stats as S
    from config import COMPLEXITY as CX
    out: dict = {"n_sweep": list(CX.n_sweep), "rows": [],
                 "timeout_s": CX.bnb_timeout_s, "n_seeds": len(records),
                 "note": records[0]["note"]}
    for i, n in enumerate(CX.n_sweep):
        row = {"n": n}
        for m in ("learned_inference", "water_filling", "branch_and_bound"):
            row[m] = S.describe([r["rows"][i][f"{m}_ms_median"] for r in records],
                                m).as_dict()
        row["bnb_timeout_fraction"] = float(np.mean(
            [r["rows"][i]["bnb_timed_out"] for r in records]))
        row["bnb_nodes_median"] = float(np.median(
            [r["rows"][i]["bnb_nodes_median"] for r in records]))
        row["speedup_vs_water_filling"] = S.describe(
            [r["rows"][i]["speedup_vs_water_filling"] for r in records], "sp").as_dict()
        row["speedup_vs_branch_and_bound"] = S.describe(
            [r["rows"][i]["speedup_vs_branch_and_bound"] for r in records], "sp").as_dict()
        out["rows"].append(row)
    return out


# ==================================================================================
# CSV export (csv module -- pandas is deliberately not a dependency)
# ==================================================================================

def write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def export_csvs(out: Path, blobs: Dict[str, dict]) -> None:
    if "integrated" in blobs:
        b = blobs["integrated"]
        rows = []
        for metric, arms in b["per_seed"].items():
            for arm, vals in arms.items():
                s = b["summaries"][metric][arm]
                rows.append([metric, arm, s["n_seeds"], s["mean"], s["sd"],
                             s["ci_low"], s["ci_high"]])
        write_csv(out / "integrated_summary.csv",
                  ["metric", "arm", "n_seeds", "mean", "sd", "ci_low", "ci_high"], rows)
        write_csv(out / "integrated_comparisons.csv",
                  list(b["family"]["rows"][0].keys()),
                  [list(r.values()) for r in b["family"]["rows"]])
        per_seed_rows = []
        for metric, arms in b["per_seed"].items():
            for arm, vals in arms.items():
                for i, v in enumerate(vals):
                    per_seed_rows.append([metric, arm, i, v])
        write_csv(out / "integrated_per_seed.csv",
                  ["metric", "arm", "seed_index", "value"], per_seed_rows)

    if "learners" in blobs:
        b = blobs["learners"]
        rows = []
        for algo, c in b["curves"].items():
            for i, ep in enumerate(c["episodes"]):
                rows.append([algo, ep, c["mean"][i], c["sd"][i], c["ci_halfwidth"][i],
                             c["n_seeds"]])
        write_csv(out / "learner_curves.csv",
                  ["algorithm", "episode", "mean_score", "sd", "ci_halfwidth",
                   "n_seeds"], rows)

    if "scheduling" in blobs:
        b = blobs["scheduling"]
        rows = []
        for metric, arms in b["summaries"].items():
            for arm, s in arms.items():
                rows.append([metric, arm, s["mean"], s["sd"], s["ci_low"], s["ci_high"]])
        write_csv(out / "scheduling_summary.csv",
                  ["metric", "algorithm", "mean", "sd", "ci_low", "ci_high"], rows)

    if "complexity" in blobs:
        b = blobs["complexity"]
        rows = []
        for r in b["rows"]:
            rows.append([r["n"],
                         r["learned_inference"]["mean"], r["learned_inference"]["ci_halfwidth"],
                         r["water_filling"]["mean"], r["water_filling"]["ci_halfwidth"],
                         r["branch_and_bound"]["mean"], r["branch_and_bound"]["ci_halfwidth"],
                         r["bnb_timeout_fraction"],
                         r["speedup_vs_water_filling"]["mean"],
                         r["speedup_vs_branch_and_bound"]["mean"]])
        write_csv(out / "complexity.csv",
                  ["n", "learned_ms", "learned_ci", "waterfilling_ms", "waterfilling_ci",
                   "bnb_ms", "bnb_ci", "bnb_timeout_fraction",
                   "speedup_vs_wf", "speedup_vs_bnb"], rows)

    if "energy" in blobs:
        b = blobs["energy"]
        rows = []
        for scale, block in b["points"].items():
            for st, d in block.items():
                rows.append([scale, st,
                             d["mean_throughput_mbps"]["mean"],
                             d["energy_efficiency_mbit_per_j"]["mean"],
                             d["energy_efficiency_mbit_per_j"]["ci_halfwidth"],
                             d["sla_feasible_fraction"], d["max_blocking"]["mean"]])
        write_csv(out / "energy.csv",
                  ["demand_scale", "strategy", "throughput_mbps", "ee_mbit_per_j",
                   "ee_ci", "sla_feasible_fraction", "max_blocking"], rows)

    if "ris" in blobs:
        b = blobs["ris"]
        rows = []
        for n_ris, block in b["points"].items():
            for arm in ("no_ris", "random_phase", "alternating_opt"):
                d = block[arm]
                rows.append([n_ris, arm, d["se_bps_hz"]["mean"],
                             d["se_bps_hz"]["ci_halfwidth"],
                             d["urllc_violation"]["mean"],
                             d["urllc_violation"]["ci_halfwidth"]])
        write_csv(out / "ris.csv",
                  ["n_ris", "arm", "se_bps_hz", "se_ci", "urllc_violation",
                   "viol_ci"], rows)

    if "sfc" in blobs:
        b = blobs["sfc"]
        rows = []
        for r in b["rows"]:
            if not r.get("stable"):
                rows.append([r["lambda_pps"], "", "", "", "", ""])
                continue
            rows.append([r["lambda_pps"], r["rho_bottleneck"],
                         r["analytical"]["mean"], r["des"]["mean"],
                         r["rel_error_pct"]["mean"], r["paired_test"]["p_raw"]])
        write_csv(out / "sfc.csv",
                  ["lambda_pps", "rho_bottleneck", "analytical_ms", "des_ms",
                   "rel_error_pct", "paired_p"], rows)

    if "slicing" in blobs:
        b = blobs["slicing"]
        rows = []
        for load, block in b["per_load"].items():
            for metric, d in block.items():
                for strat, s in d["summary"].items():
                    rows.append([load, metric, strat, s["mean"], s["sd"],
                                 s["ci_low"], s["ci_high"]])
        write_csv(out / "slicing.csv",
                  ["load", "metric", "strategy", "mean", "sd", "ci_low", "ci_high"],
                  rows)


# ==================================================================================
# Driver
# ==================================================================================

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="UMRO-5G simulation campaign")
    ap.add_argument("--seeds", type=int, default=CAMPAIGN.n_seeds)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--quick", action="store_true",
                    help="smoke run: few seeds, short horizons")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated subset of " + ",".join(STAGES))
    ap.add_argument("--workers", type=int, default=CAMPAIGN.n_workers)
    args = ap.parse_args(argv)

    quick = args.quick
    n_seeds = CAMPAIGN.n_seeds_quick if quick else args.seeds
    n_mc = CAMPAIGN.n_monte_carlo_quick if quick else CAMPAIGN.n_monte_carlo
    compress = (TRAFFIC.compress_seconds_quick if quick else TRAFFIC.compress_seconds)
    n_ep = None
    if quick:
        from config import RLTRAIN
        n_ep = RLTRAIN.n_episodes_quick
    stages = ([s.strip() for s in args.only.split(",") if s.strip()]
              if args.only else list(STAGES))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = seed_list(n_seeds)
    ctx = get_context("spawn")
    t_start = time.time()
    timings: Dict[str, float] = {}
    blobs: Dict[str, dict] = {}

    print(f"UMRO-5G campaign: {n_seeds} seeds, {n_mc} Monte Carlo iterations/seed, "
          f"{args.workers} workers, quick={quick}")
    print(f"stages: {stages}")

    # ---- stage 0: pre-train the Near-RT RIC policy --------------------------------
    policy_path = out / "medium_policy.json"
    if "pretrain" in stages or not policy_path.exists():
        from integrated import pretrain_medium_policy
        t0 = time.time()
        n_pre = 40 if quick else 300
        blob = pretrain_medium_policy(n_episodes=n_pre)
        with open(policy_path, "w", encoding="utf-8") as fh:
            json.dump(blob, fh)
        timings["pretrain"] = time.time() - t0
        print(f"  pretrain: {timings['pretrain']:.1f}s  "
              f"learned={blob['surrogate_return_learned']:.1f} "
              f"hold={blob['surrogate_return_hold']:.1f} "
              f"random={blob['surrogate_return_random']:.1f}")
    with open(policy_path, "r", encoding="utf-8") as fh:
        policy_params = json.load(fh)["params"]

    def run_stage(name: str, fn, payload) -> List[dict]:
        t0 = time.time()
        if len(payload) == 1 or args.workers <= 1:
            recs = [fn(p) for p in payload]
        else:
            with ctx.Pool(min(args.workers, len(payload)),
                          initializer=_init_worker) as pool:
                recs = pool.map(fn, payload)
        timings[name] = time.time() - t0
        print(f"  {name}: {timings[name]:.1f}s  ({len(payload)} tasks)")
        return recs

    if "integrated" in stages:
        payload = [(s, policy_params, compress, i == 0) for i, s in enumerate(sd)]
        recs = run_stage("integrated", work_integrated, payload)
        trace = recs[0]["arms"]["E_full_umro"].pop("trace", None)
        for r in recs[1:]:
            for a in ARMS:
                r["arms"][a].pop("trace", None)
        blobs["integrated"] = aggregate_integrated(recs, n_mc, trace)
        if trace is not None:
            with open(out / "integrated_trace_seed0.json", "w", encoding="utf-8") as fh:
                json.dump({"seed": recs[0]["seed"], "arm": "E_full_umro",
                           "trace": trace}, fh)

    if "slicing" in stages:
        recs = run_stage("slicing", work_slicing, [(s, n_mc) for s in sd])
        blobs["slicing"] = aggregate_slicing(recs, n_mc)

    if "learners" in stages:
        from learners import ALGORITHMS
        payload = [(a, s, n_ep) for a in ALGORITHMS for s in sd]
        recs = run_stage("learners", work_learners, payload)
        blobs["learners"] = aggregate_learners(recs, 1)

    if "scheduling" in stages:
        from config import SCHED
        n_tti = SCHED.n_tti_quick if quick else SCHED.n_tti
        recs = run_stage("scheduling", work_scheduling, [(s, n_tti) for s in sd])
        blobs["scheduling"] = aggregate_scheduling(recs, n_tti)

    if "sfc" in stages:
        from config import SFC
        n_pk = SFC.n_packets_des_quick if quick else SFC.n_packets_des
        recs = run_stage("sfc", work_sfc, [(s, n_pk) for s in sd])
        blobs["sfc"] = aggregate_sfc(recs, n_pk)

    if "energy" in stages:
        recs = run_stage("energy", work_energy, [(s,) for s in sd])
        blobs["energy"] = aggregate_energy(recs, 1)

    if "ris" in stages:
        from config import RIS
        n_real = RIS.n_realizations_quick if quick else RIS.n_realizations
        recs = run_stage("ris", work_ris, [(s, n_real) for s in sd])
        blobs["ris"] = aggregate_ris(recs, n_real)

    if "complexity" in stages:
        from config import COMPLEXITY
        from complexity import run_complexity_seed
        n_cx = N_SEEDS_COMPLEXITY_QUICK if quick else N_SEEDS_COMPLEXITY
        reps = COMPLEXITY.n_repeats_quick if quick else COMPLEXITY.n_repeats
        t0 = time.time()
        print("  complexity: running SEQUENTIALLY (wall-clock measurement)")
        recs = [run_complexity_seed(s, reps) for s in seed_list(n_cx)]
        timings["complexity"] = time.time() - t0
        print(f"  complexity: {timings['complexity']:.1f}s ({n_cx} seeds)")
        blobs["complexity"] = aggregate_complexity(recs)

    # ---- persist -------------------------------------------------------------------
    for name, blob in blobs.items():
        with open(out / f"{name}.json", "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=1, default=float)
    export_csvs(out, blobs)

    # A partial run (--only ...) must not erase the record of the stages it did not
    # re-execute, otherwise the manifest stops describing the results on disk.
    manifest_path = out / "run_manifest.json"
    prior: dict = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                prior = json.load(fh)
        except (json.JSONDecodeError, OSError):
            prior = {}
    merged_timings = dict(prior.get("stage_seconds", {}))
    merged_timings.update(timings)
    merged_stages = sorted(set(prior.get("stages", [])) | set(stages),
                           key=lambda s: STAGES.index(s) if s in STAGES else 99)

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": (time.time() - t_start
                         + sum(v for k, v in merged_timings.items()
                               if k not in timings)),
        "wall_clock_this_invocation_s": time.time() - t_start,
        "stages_this_invocation": stages,
        "stage_seconds": merged_timings,
        "n_seeds": n_seeds,
        "n_seeds_complexity": (N_SEEDS_COMPLEXITY_QUICK if quick
                               else N_SEEDS_COMPLEXITY),
        "n_monte_carlo_per_seed": n_mc,
        "compress_seconds": compress,
        "quick": quick,
        "seeds": sd,
        "stages": merged_stages,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "sample_size_rule": ("the unit of independent replication is the seed; "
                             "Monte Carlo iterations are averaged within a seed and "
                             "do NOT increase the independent sample size"),
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(out / "config_snapshot.json", "w", encoding="utf-8") as fh:
        json.dump(C.snapshot(), fh, indent=1, default=str)

    print(f"total wall clock: {manifest['wall_clock_s']:.1f}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
