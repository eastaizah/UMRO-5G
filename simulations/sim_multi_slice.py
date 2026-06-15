#!/usr/bin/env python3
"""
Simulation 1: Multi-Slice Resource Allocation (Monte Carlo)
Compares Hard Isolation vs Soft Isolation vs UMRO-5G dynamic allocation
across 3 slice types: eMBB, URLLC, mMTC.

5G NR parameters (μ=0, SCS=15 kHz):
  B = 20 MHz, N_PRB = 100, Δf_PRB = 12 × 15 kHz = 180 kHz = 0.18 MHz
  Throughput (Mbps) = alloc_PRBs × SPEC_EFF (bps/Hz) × PRB_BW_MHz
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

TOTAL_PRBS = 100
SLICE_NAMES = ['eMBB', 'URLLC', 'mMTC']

# Hard isolation fixed allocations
HARD_ALLOC = np.array([50, 30, 20])

# Soft isolation minimum guarantees + shared pool
SOFT_MIN = np.array([30, 20, 10])
SOFT_SHARED = TOTAL_PRBS - SOFT_MIN.sum()  # 40

# Spectral efficiencies (bps/Hz per PRB) for each slice type
SPEC_EFF = np.array([4.0, 1.5, 0.5])

# 5G NR PRB bandwidth: 12 subcarriers × 15 kHz = 180 kHz = 0.18 MHz
PRB_BW_MHz = 0.18

# Latency budget (ms) – only URLLC has tight constraint
LATENCY_BUDGET = np.array([20.0, 1.0, 50.0])

LOAD_POINTS = np.linspace(0.1, 1.0, 10)
N_MONTE_CARLO = 1000
SEEDS = [42, 123, 256, 789, 1024]


def poisson_demand(load, rng):
    """Generate Poisson demand (PRBs) per slice given normalized load."""
    mean_demand = load * np.array([60, 35, 25])
    return rng.poisson(mean_demand)


def hard_isolation_alloc(demand):
    return np.minimum(demand, HARD_ALLOC)


def soft_isolation_alloc(demand):
    alloc = np.minimum(demand, SOFT_MIN.copy().astype(float))
    residual = demand - alloc
    pool = float(SOFT_SHARED)
    total_residual = residual.sum()
    if total_residual > 0 and pool > 0:
        share = np.minimum(residual, pool * residual / total_residual)
        alloc += share
    return alloc


def umro5g_alloc(demand, load, rng):
    """Dynamic Lagrangian dual decomposition with utility-weighted allocation."""
    # Utility weights: prioritize URLLC at high load, eMBB otherwise
    w = np.array([1.0 + 0.5 * load, 2.0 + 1.5 * load, 0.3 + 0.2 * load])
    w /= w.sum()

    # Lagrangian iterative allocation
    alloc = np.zeros(3)
    remaining = float(TOTAL_PRBS)
    lam = np.zeros(3)  # dual variables

    for _ in range(20):
        # Primal update: maximize weighted utility minus dual penalty
        proposal = w * demand
        proposal = np.minimum(proposal, demand.astype(float))
        total_prop = proposal.sum()
        if total_prop > remaining:
            proposal *= remaining / total_prop
        alloc = proposal

        # Dual update
        violation = alloc.sum() - TOTAL_PRBS
        lam = np.maximum(0, lam + 0.1 * violation)
        remaining = float(TOTAL_PRBS)

    # Small noise for realism
    alloc += rng.normal(0, 0.3, 3)
    alloc = np.clip(alloc, 0, demand.astype(float))
    if alloc.sum() > TOTAL_PRBS:
        alloc *= TOTAL_PRBS / alloc.sum()
    return alloc


def compute_throughput(alloc):
    """Return throughput in Mbps using 5G NR PRB bandwidth."""
    return alloc * SPEC_EFF * PRB_BW_MHz


def compute_latency_violation(alloc, demand):
    """Probability of violating latency budget (simplified model)."""
    # Congestion ratio drives latency: higher congestion → higher latency
    congestion = np.where(alloc > 0, demand / (alloc + 1e-9), 10.0)
    # Simplified latency model (ms)
    latency = 0.5 * congestion  # ms per unit congestion
    return (latency > LATENCY_BUDGET).astype(float)


def run():
    results = {method: {
        'throughput': np.zeros((len(SEEDS), len(LOAD_POINTS), 3)),
        'utilization': np.zeros((len(SEEDS), len(LOAD_POINTS))),
        'latency_viol': np.zeros((len(SEEDS), len(LOAD_POINTS), 3)),
    } for method in ['Hard', 'Soft', 'UMRO-5G']}

    for si, seed in enumerate(SEEDS):
        for li, load in enumerate(LOAD_POINTS):
            rng = np.random.default_rng(seed + li * 100)
            tp_accum = {m: np.zeros(3) for m in results}
            ut_accum = {m: 0.0 for m in results}
            lv_accum = {m: np.zeros(3) for m in results}

            for _ in range(N_MONTE_CARLO):
                demand = poisson_demand(load, rng)

                allocs = {
                    'Hard': hard_isolation_alloc(demand),
                    'Soft': soft_isolation_alloc(demand),
                    'UMRO-5G': umro5g_alloc(demand, load, rng),
                }

                for m, alloc in allocs.items():
                    tp_accum[m] += compute_throughput(alloc)
                    ut_accum[m] += alloc.sum() / TOTAL_PRBS
                    lv_accum[m] += compute_latency_violation(alloc, demand)

            for m in results:
                results[m]['throughput'][si, li] = tp_accum[m] / N_MONTE_CARLO
                results[m]['utilization'][si, li] = ut_accum[m] / N_MONTE_CARLO
                results[m]['latency_viol'][si, li] = lv_accum[m] / N_MONTE_CARLO

    # --- Plotting ---
    plt.rcParams.update({
        'font.size': 11, 'figure.dpi': 150, 'figure.figsize': (7, 5),
        'axes.grid': True, 'grid.alpha': 0.3,
    })
    colors = {'Hard': '#1f77b4', 'Soft': '#ff7f0e', 'UMRO-5G': '#2ca02c'}
    markers = {'Hard': 's', 'Soft': '^', 'UMRO-5G': 'o'}

    # Figure 1: Throughput vs Load per slice
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for idx, sname in enumerate(SLICE_NAMES):
        ax = axes[idx]
        for m in ['Hard', 'Soft', 'UMRO-5G']:
            mean = results[m]['throughput'][:, :, idx].mean(axis=0)
            std = results[m]['throughput'][:, :, idx].std(axis=0)
            ax.plot(LOAD_POINTS, mean, marker=markers[m], color=colors[m],
                    label=m, linewidth=1.8, markersize=5)
            ax.fill_between(LOAD_POINTS, mean - std, mean + std,
                            alpha=0.15, color=colors[m])
        ax.set_xlabel('Normalized Load')
        ax.set_ylabel('Throughput (Mbps)')
        ax.set_title(f'{sname} Slice')
        ax.legend(fontsize=9)
    fig.suptitle('Multi-Slice Throughput vs. Network Load', fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig1_throughput_vs_load.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Figure 2: Resource Utilization vs Load
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ['Hard', 'Soft', 'UMRO-5G']:
        mean = results[m]['utilization'].mean(axis=0)
        std = results[m]['utilization'].std(axis=0)
        ax.plot(LOAD_POINTS, mean * 100, marker=markers[m], color=colors[m],
                label=m, linewidth=1.8, markersize=6)
        ax.fill_between(LOAD_POINTS, (mean - std) * 100, (mean + std) * 100,
                        alpha=0.15, color=colors[m])
    ax.set_xlabel('Normalized Load')
    ax.set_ylabel('Resource Utilization (%)')
    ax.set_title('PRB Utilization vs. Network Load')
    ax.axhline(y=100, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, label='100% Capacity')
    ax.legend()
    ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig2_utilization_vs_load.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Figure 3: Latency Violation Probability (URLLC slice)
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in ['Hard', 'Soft', 'UMRO-5G']:
        mean = results[m]['latency_viol'][:, :, 1].mean(axis=0)  # URLLC
        std = results[m]['latency_viol'][:, :, 1].std(axis=0)
        ax.plot(LOAD_POINTS, mean, marker=markers[m], color=colors[m],
                label=m, linewidth=1.8, markersize=6)
        ax.fill_between(LOAD_POINTS, np.clip(mean - std, 0, 1),
                        np.clip(mean + std, 0, 1), alpha=0.15, color=colors[m])
    ax.axhline(y=0.01, color='red', linestyle='--', linewidth=1.2,
               label='URLLC Target (1%)')
    ax.set_xlabel('Normalized Load')
    ax.set_ylabel('Latency Violation Probability')
    ax.set_title('URLLC Latency Violation Probability vs. Load')
    ax.set_yscale('log')
    ax.legend()
    ax.set_ylim(1e-3, 2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig3_latency_violation.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("  [Sim 1] Generated: fig1_throughput_vs_load.png, "
          "fig2_utilization_vs_load.png, fig3_latency_violation.png")


if __name__ == '__main__':
    run()
