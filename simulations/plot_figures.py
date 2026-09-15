"""
plot_figures.py -- Regenerates every figure from the committed result files.

    python plot_figures.py --results results --figures ../figures

Rules enforced here (they are review requirements, not preferences):
  * 300 dpi PNG.
  * NO title inside the image: the figure title lives in the article's caption.
    In particular, none of the stale "Fig. 9 / Fig. 10 / Fig. 11" strings that the
    previous figures carried are reproduced.
  * Okabe-Ito colour-blind-safe palette, and every series is additionally
    distinguished by marker and line style so the figures survive greyscale printing.
  * Sized for an MDPI text column and readable at that size.
  * Error bands / bars are 95 % Student-t confidence intervals over the SEEDS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from config import PLOT, ARM_LABELS, SLICE_NAMES

C = PLOT.palette
M = PLOT.markers
L = PLOT.linestyles


def _style() -> None:
    plt.rcParams.update({
        "font.size": PLOT.base_font_pt,
        "axes.labelsize": PLOT.base_font_pt,
        "axes.titlesize": PLOT.base_font_pt,
        "xtick.labelsize": PLOT.base_font_pt - 1,
        "ytick.labelsize": PLOT.base_font_pt - 1,
        "legend.fontsize": PLOT.base_font_pt - 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "lines.markersize": 3.6,
        "figure.constrained_layout.use": True,
        "savefig.dpi": PLOT.dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _load(results: Path, name: str):
    p = results / f"{name}.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ==================================================================================
# Figure 4 -- integrated end-to-end evaluation
# ==================================================================================

SHORT = {
    "A_hard_static": "A\nhard iso.",
    "B_soft_pool": "B\nsoft pool",
    "C_no_medium": "C\nno medium",
    "D_no_slow": "D\nno slow",
    "E_full_umro": "E\nfull UMRO",
}


def fig4_integrated(results: Path, figures: Path) -> None:
    b = _load(results, "integrated")
    if b is None:
        return
    trace_blob = _load(results, "integrated_trace_seed0")
    arms = list(SHORT.keys())
    fig, ax = plt.subplots(2, 2, figsize=(PLOT.double_width_in, 4.6))

    # (a) SLA satisfaction rate with 95 % CI
    s = b["summaries"]["sla_satisfaction_rate"]
    x = np.arange(len(arms))
    mean = [s[a]["mean"] for a in arms]
    err = [s[a]["ci_halfwidth"] for a in arms]
    bars = ax[0, 0].bar(x, mean, yerr=err, capsize=2.5,
                        color=[C[i % len(C)] for i in range(len(arms))],
                        edgecolor="black", linewidth=0.4)
    bars[-1].set_hatch("//")
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels([SHORT[a] for a in arms])
    ax[0, 0].set_ylabel("SLA satisfaction rate")
    ax[0, 0].set_ylim(0, 1.05)
    ax[0, 0].text(0.02, 0.94, "(a)", transform=ax[0, 0].transAxes, fontweight="bold", va="top")

    # (b) per-slice throughput, stacked
    bottom = np.zeros(len(arms))
    for i, name in enumerate(SLICE_NAMES):
        sm = b["summaries"][f"throughput_{name}_mbps"]
        vals = np.array([sm[a]["mean"] for a in arms])
        ax[0, 1].bar(x, vals, bottom=bottom, label=name, color=C[i],
                     edgecolor="black", linewidth=0.4)
        bottom += vals
    tot = b["summaries"]["aggregate_throughput_mbps"]
    ax[0, 1].errorbar(x, [tot[a]["mean"] for a in arms],
                      yerr=[tot[a]["ci_halfwidth"] for a in arms],
                      fmt="none", ecolor="black", capsize=2.5, elinewidth=0.8)
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels([SHORT[a] for a in arms])
    ax[0, 1].set_ylabel("Throughput (Mbit/s)")
    ax[0, 1].set_ylim(0, 1.32 * float(np.max(bottom)))
    ax[0, 1].legend(loc="upper center", frameon=False, ncol=3, fontsize=6,
                    handlelength=1.2, columnspacing=1.0)
    ax[0, 1].text(0.02, 0.94, "(b)", transform=ax[0, 1].transAxes, fontweight="bold", va="top")

    # (c) PRB utilisation vs URLLC violation probability
    u = b["summaries"]["prb_utilisation"]
    v = b["summaries"]["urllc_violation_prob"]
    for i, a in enumerate(arms):
        ax[1, 0].errorbar(u[a]["mean"], 100 * v[a]["mean"],
                          xerr=u[a]["ci_halfwidth"],
                          yerr=100 * v[a]["ci_halfwidth"],
                          marker=M[i % len(M)], color=C[i % len(C)],
                          capsize=2, markersize=5,
                          label=SHORT[a].replace("\n", " "))
    ax[1, 0].set_xlabel("PRB utilisation")
    ax[1, 0].set_ylabel("URLLC radio violation (%)")
    ax[1, 0].legend(loc="best", frameon=False, fontsize=5.5)
    ax[1, 0].text(0.02, 0.94, "(c)", transform=ax[1, 0].transAxes, fontweight="bold", va="top")

    # (d) loop coupling over the compressed day
    if trace_blob is not None:
        tr = trace_blob["trace"]
        t = np.asarray(tr["t_s"])
        alloc = np.asarray(tr["alloc"], dtype=float)
        price = np.asarray(tr["prices"], dtype=float)
        mult = np.asarray(tr["offered_multiplier"], dtype=float)
        hours = 24.0 * t / max(t.max(), 1e-9)
        for i, name in enumerate(SLICE_NAMES):
            ax[1, 1].plot(hours, alloc[:, i], color=C[i], linestyle=L[i],
                          label=f"{name} PRBs")
        ax[1, 1].set_xlabel("Time of the compressed day (h)")
        ax[1, 1].set_ylabel("PRBs held by the slice")
        ax2 = ax[1, 1].twinx()
        ax2.plot(hours, mult, color=C[6], linestyle=":", linewidth=1.0,
                 label="offered load")
        ax2.plot(hours, price[:, 0], color=C[4], linestyle="-.", linewidth=1.0,
                 label="eMBB price")
        ax2.set_ylabel("Offered-load multiplier / dual price")
        ax2.grid(False)
        h1, l1 = ax[1, 1].get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax[1, 1].legend(h1 + h2, l1 + l2, loc="upper left", frameon=False,
                        fontsize=5.2, ncol=2)
        ax[1, 1].text(0.02, 0.94, "(d)", transform=ax[1, 1].transAxes,
                      fontweight="bold", va="top")
    fig.savefig(figures / "fig4_integrated.png", dpi=PLOT.dpi)
    plt.close(fig)


# ==================================================================================
# Figure 5 -- learning-based controller convergence
# ==================================================================================

def fig5_convergence(results: Path, figures: Path) -> None:
    b = _load(results, "learners")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(PLOT.double_width_in, 2.5),
                           gridspec_kw={"width_ratios": [1.6, 1.0]})
    algos = list(b["curves"].keys())
    for i, a in enumerate(algos):
        c = b["curves"][a]
        ep = np.asarray(c["episodes"], dtype=float)
        m = np.asarray(c["mean"])
        hw = np.asarray(c["ci_halfwidth"])
        ax[0].plot(ep, m, color=C[i], linestyle=L[i], marker=M[i],
                   markevery=max(1, len(ep) // 10), label=a)
        ax[0].fill_between(ep, m - hw, m + hw, color=C[i], alpha=0.15, linewidth=0)
    ax[0].axhline(0.0, color="black", linewidth=0.7, linestyle=(0, (1, 2)))
    ax[0].axhline(1.0, color=C[6], linewidth=0.7, linestyle=(0, (4, 2)))
    ax[0].annotate("uniformly random power control", xy=(ep[0], 0.0),
                   xytext=(4, 4), textcoords="offset points", ha="left",
                   fontsize=5.5, color="black")
    ax[0].annotate("myopic full-CSI oracle", xy=(ep[-1], 1.0),
                   xytext=(-2, -9), textcoords="offset points", ha="right",
                   fontsize=5.5, color=C[6])
    ax[0].set_ylim(top=1.14)
    ax[0].set_xlabel("Training episode")
    ax[0].set_ylabel("Normalised evaluation score")
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.13, 0.97), frameon=False,
                 ncol=2, handlelength=1.8, columnspacing=1.2)
    ax[0].text(0.015, 0.97, "(a)", transform=ax[0].transAxes,
           fontweight="bold", va="top")

    s = b["summaries"]
    x = np.arange(len(algos))
    mean = [s[a]["mean"] for a in algos]
    err = [s[a]["ci_halfwidth"] for a in algos]
    ax[1].bar(x, mean, yerr=err, capsize=2.5, color=[C[i] for i in range(len(algos))],
              edgecolor="black", linewidth=0.4)
    conv = b["converged_fraction"]
    for i, a in enumerate(algos):
        ax[1].text(i, max(mean[i] + err[i], 0) + 0.015,
                   f"{100 * conv[a]:.0f}%", ha="center", fontsize=5.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([a.replace("-", "-\n") for a in algos], fontsize=5.5)
    ax[1].set_ylabel("Final normalised score")
    ax[1].text(0.015, 0.97, "(b)", transform=ax[1].transAxes,
           fontweight="bold", va="top")
    fig.savefig(figures / "fig5_convergence.png", dpi=PLOT.dpi)
    plt.close(fig)


# ==================================================================================
# Figure 6 -- energy efficiency vs throughput
# ==================================================================================

ENERGY_LABEL = {"always_on": "Always-on", "static_sleep": "Static scheduled sleep",
                "predictive_sleep": "Predictive sleep (UMRO-5G)"}


def fig6_energy(results: Path, figures: Path) -> None:
    b = _load(results, "energy")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(PLOT.double_width_in, 2.5))
    scales = [str(s) for s in b["demand_scales"]]
    for i, st in enumerate(ENERGY_LABEL):
        thr, ee, hw, feas = [], [], [], []
        for s in scales:
            d = b["points"][s][st]
            thr.append(d["mean_throughput_mbps"]["mean"])
            ee.append(d["energy_efficiency_mbit_per_j"]["mean"])
            hw.append(d["energy_efficiency_mbit_per_j"]["ci_halfwidth"])
            feas.append(d["sla_feasible_fraction"])
        thr, ee, hw, feas = map(np.asarray, (thr, ee, hw, feas))
        order = np.argsort(thr)
        ax[0].plot(thr[order], ee[order], color=C[i], linestyle=L[i], marker=M[i],
                   label=ENERGY_LABEL[st])
        ax[0].fill_between(thr[order], (ee - hw)[order], (ee + hw)[order],
                           color=C[i], alpha=0.15, linewidth=0)
        ok = feas >= 0.5
        if ok.any():
            ax[0].scatter(thr[ok], ee[ok], s=42, facecolors="none",
                          edgecolors=C[i], linewidths=1.0, zorder=5)
    ax[0].set_xlabel("System throughput (Mbit/s)")
    ax[0].set_ylabel("Energy efficiency (Mbit/J)")
    ax[0].legend(loc="best", frameon=False)
    ax[0].text(0.015, 0.97, "(a)", transform=ax[0].transAxes,
           fontweight="bold", va="top")
    ax[0].set_title("open rings mark SLA-feasible operating points",
                    fontsize=5.4, loc="right", pad=2.0)

    m = b["margin_sweep"]
    marg = [100 * r["margin"] for r in m]
    ee = np.array([r["ee"]["mean"] for r in m])
    hw = np.array([r["ee"]["ci_halfwidth"] for r in m])
    blk = np.array([100 * r["max_blocking"]["mean"] for r in m])
    ax[1].errorbar(marg, ee, yerr=hw, color=C[2], marker=M[2], capsize=2,
                   label="Energy efficiency")
    ax[1].set_xlabel("Predictive-sleep safety margin (%)")
    ax[1].set_ylabel("Energy efficiency (Mbit/J)")
    ax3 = ax[1].twinx()
    ax3.plot(marg, blk, color=C[4], linestyle="--", marker=M[4],
             label="Peak blocking")
    ax3.set_ylabel("Peak blocking (%)")
    ax3.grid(False)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, loc="best", frameon=False)
    ax[1].text(0.048, 0.965, "(b)", transform=ax[1].transAxes,
           fontweight="bold", va="top")
    fig.savefig(figures / "fig6_energy.png", dpi=PLOT.dpi)
    plt.close(fig)


# ==================================================================================
# Figure 7 -- RIS
# ==================================================================================

RIS_LABEL = {"no_ris": "No RIS", "random_phase": "Random phases",
             "alternating_opt": "Alternating optimization"}


def fig7_ris(results: Path, figures: Path) -> None:
    b = _load(results, "ris")
    if b is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(PLOT.double_width_in, 2.5))
    n_sweep = b["n_ris_sweep"]
    for i, arm in enumerate(RIS_LABEL):
        se = np.array([b["points"][str(n)][arm]["se_bps_hz"]["mean"] for n in n_sweep])
        hw = np.array([b["points"][str(n)][arm]["se_bps_hz"]["ci_halfwidth"]
                       for n in n_sweep])
        vi = np.array([100 * b["points"][str(n)][arm]["urllc_violation"]["mean"]
                       for n in n_sweep])
        vh = np.array([100 * b["points"][str(n)][arm]["urllc_violation"]["ci_halfwidth"]
                       for n in n_sweep])
        ax[0].errorbar(n_sweep, se, yerr=hw, color=C[i], linestyle=L[i], marker=M[i],
                       capsize=2, label=RIS_LABEL[arm])
        ax[1].errorbar(n_sweep, vi, yerr=vh, color=C[i], linestyle=L[i], marker=M[i],
                       capsize=2, label=RIS_LABEL[arm])
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xticks(n_sweep)
        a.set_xticklabels([str(n) for n in n_sweep])
        a.set_xlabel("Number of RIS elements $N_{\\mathrm{RIS}}$")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0),
                                                    numticks=12))
    ax[0].yaxis.set_minor_locator(ticker.NullLocator())
    ax[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax[0].set_ylabel("Spectral efficiency (bit/s/Hz)")
    ax[1].set_ylabel("URLLC rate-outage probability (%)")
    ax[1].set_ylim(-3, 103)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0.11, 0.99), frameon=False)
    ax[0].text(0.015, 0.97, "(a)", transform=ax[0].transAxes,
           fontweight="bold", va="top")
    ax[1].text(0.015, 0.97, "(b)", transform=ax[1].transAxes,
           fontweight="bold", va="top")
    fig.savefig(figures / "fig7_ris.png", dpi=PLOT.dpi)
    plt.close(fig)


# ==================================================================================
# Supporting figures (not numbered in the article, kept for the repository)
# ==================================================================================

def fig_supporting(results: Path, figures: Path) -> None:
    sl = _load(results, "slicing")
    if sl is not None:
        fig, ax = plt.subplots(1, 3, figsize=(PLOT.double_width_in, 2.2))
        loads = sl["loads"]
        labels = {"hard": "Hard isolation", "soft": "Soft isolation",
                  "umro": "UMRO-5G (Lagrangian)"}
        for k, (metric, ylab) in enumerate((("throughput_mbps", "Aggregate throughput (Mbit/s)"),
                                            ("utilisation", "PRB utilisation"),
                                            ("urllc_violation", "URLLC violation prob."))):
            for i, st in enumerate(labels):
                m = [sl["per_load"][str(l)][metric]["summary"][st]["mean"] for l in loads]
                h = [sl["per_load"][str(l)][metric]["summary"][st]["ci_halfwidth"]
                     for l in loads]
                ax[k].errorbar(loads, m, yerr=h, color=C[i], linestyle=L[i],
                               marker=M[i], capsize=2, label=labels[st])
            ax[k].set_xlabel(r"Offered load $\rho$")
            ax[k].set_ylabel(ylab)
        ax[0].legend(loc="best", frameon=False)
        fig.savefig(figures / "figS1_multislice.png", dpi=PLOT.dpi)
        plt.close(fig)

    cx = _load(results, "complexity")
    if cx is not None:
        fig, a = plt.subplots(figsize=(PLOT.column_width_in, 2.4))
        n = [r["n"] for r in cx["rows"]]
        names = {"learned_inference": "Learned inference",
                 "water_filling": "Water-filling",
                 "branch_and_bound": "Branch and bound"}
        for i, key in enumerate(names):
            m = [r[key]["mean"] for r in cx["rows"]]
            h = [r[key]["ci_halfwidth"] for r in cx["rows"]]
            a.errorbar(n, m, yerr=h, color=C[i], linestyle=L[i], marker=M[i],
                       capsize=2, label=names[key])
        to = [r["bnb_timeout_fraction"] for r in cx["rows"]]
        for xi, t in zip(n, to):
            if t > 0:
                a.axvline(xi, color=C[4], alpha=0.12, linewidth=6)
        a.axhline(cx["timeout_s"] * 1e3, color="black", linewidth=0.7,
                  linestyle=(0, (1, 2)))
        a.text(n[0], cx["timeout_s"] * 1e3 * 1.15,
               f"branch-and-bound timeout ({cx['timeout_s']:.0f} s)", fontsize=5.2)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Problem size $N$ (users)")
        a.set_ylabel("Decision time (ms)")
        a.legend(loc="best", frameon=False)
        fig.savefig(figures / "figS2_complexity.png", dpi=PLOT.dpi)
        plt.close(fig)

    sf = _load(results, "sfc")
    if sf is not None:
        fig, a = plt.subplots(figsize=(PLOT.column_width_in, 2.4))
        rows = [r for r in sf["rows"] if r.get("stable")]
        lam = [r["lambda_pps"] for r in rows]
        an = [r["analytical"]["mean"] for r in rows]
        de = [r["des"]["mean"] for r in rows]
        dh = [r["des"]["ci_halfwidth"] for r in rows]
        a.plot(lam, an, color=C[0], linestyle=L[0], marker=M[0],
               label="Jackson analytical")
        a.errorbar(lam, de, yerr=dh, color=C[1], linestyle=L[1], marker=M[1],
                   capsize=2, label="Discrete-event simulation")
        a.axhline(1.0, color="black", linewidth=0.7, linestyle=(0, (1, 2)))
        a.text(lam[0], 1.08, "1 ms URLLC radio budget", fontsize=5.2)
        a.set_xlabel(r"Arrival rate $\lambda$ (packets/s)")
        a.set_ylabel("End-to-end chain latency (ms)")
        a.set_yscale("log")
        a.legend(loc="best", frameon=False)
        fig.savefig(figures / "figS3_sfc.png", dpi=PLOT.dpi)
        plt.close(fig)

    sc = _load(results, "scheduling")
    if sc is not None:
        fig, a = plt.subplots(figsize=(PLOT.column_width_in, 2.4))
        algos = list(sc["summaries"]["throughput_mbps"].keys())
        thr = [sc["summaries"]["throughput_mbps"][k]["mean"] for k in algos]
        the = [sc["summaries"]["throughput_mbps"][k]["ci_halfwidth"] for k in algos]
        jai = [sc["summaries"]["jain"][k]["mean"] for k in algos]
        for i, k in enumerate(algos):
            a.errorbar(jai[i], thr[i], yerr=the[i], color=C[i], marker=M[i],
                       markersize=6, capsize=2, label=k.replace("_", " "))
        a.set_xlabel("Jain fairness index")
        a.set_ylabel("Aggregate throughput (Mbit/s)")
        a.legend(loc="best", frameon=False)
        fig.savefig(figures / "figS4_scheduling.png", dpi=PLOT.dpi)
        plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="../figures")
    args = ap.parse_args(argv)
    results = Path(args.results)
    figures = Path(args.figures)
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    fig4_integrated(results, figures)
    fig5_convergence(results, figures)
    fig6_energy(results, figures)
    fig7_ris(results, figures)
    fig_supporting(results, figures)
    print(f"figures written to {figures.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
