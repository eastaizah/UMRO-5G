#!/usr/bin/env python3
"""
Simulation 5: Computational Complexity Comparison
Compares DRL inference time vs convex optimization vs MILP solver time
across different problem sizes.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

PROBLEM_SIZES = [10, 20, 50, 100, 200, 500, 1000]
N_REPS = 100
HIDDEN_UNITS = 128


def drl_inference(n_users, rng):
    """Neural network forward pass (2 hidden layers, 128 units)."""
    state = rng.normal(0, 1, n_users * 2)  # channel + allocation state
    w1 = rng.normal(0, np.sqrt(2.0 / (n_users * 2)), (n_users * 2, HIDDEN_UNITS))
    b1 = np.zeros(HIDDEN_UNITS)
    w2 = rng.normal(0, np.sqrt(2.0 / HIDDEN_UNITS), (HIDDEN_UNITS, HIDDEN_UNITS))
    b2 = np.zeros(HIDDEN_UNITS)
    w3 = rng.normal(0, np.sqrt(2.0 / HIDDEN_UNITS), (HIDDEN_UNITS, n_users))
    b3 = np.zeros(n_users)

    start = time.perf_counter()
    for _ in range(N_REPS):
        h1 = np.maximum(0, state @ w1 + b1)
        h2 = np.maximum(0, h1 @ w2 + b2)
        _ = h2 @ w3 + b3  # output: allocation per user
    elapsed = time.perf_counter() - start
    return elapsed / N_REPS


def water_filling(n_users, rng, epsilon=1e-6, max_iter=500):
    """Iterative water-filling for power allocation (convex optimization)."""
    channel_gains = rng.rayleigh(1.0, n_users)**2
    total_power = float(n_users)  # total power budget
    noise = 0.1 * np.ones(n_users)

    start = time.perf_counter()
    for _ in range(N_REPS):
        # Binary search on water level
        lo, hi = 0.0, total_power * 10.0
        for _ in range(max_iter):
            mu = (lo + hi) / 2.0
            power = np.maximum(0, mu - noise / channel_gains)
            if power.sum() > total_power:
                hi = mu
            else:
                lo = mu
            if hi - lo < epsilon:
                break
    elapsed = time.perf_counter() - start
    return elapsed / N_REPS


def milp_branch_bound(n_users, rng):
    """Simplified branch-and-bound for discrete allocation (MILP approx)."""
    channel_gains = rng.rayleigh(1.0, n_users)**2
    n_levels = 5
    total_budget = n_users * 2

    start = time.perf_counter()
    for _ in range(N_REPS):
        # Greedy relaxation + rounding (simplified B&B)
        rates = np.zeros((n_users, n_levels))
        for u in range(n_users):
            for l in range(n_levels):
                power = (l + 1) / n_levels * 2.0
                rates[u, l] = np.log2(1 + power * channel_gains[u] / 0.1)

        # LP relaxation: fractional assignment
        best_alloc = np.zeros(n_users, dtype=int)
        remaining = total_budget

        # Sort by marginal gain
        marginal = rates[:, -1] - rates[:, 0]
        order = np.argsort(-marginal)

        for u in order:
            if remaining <= 0:
                break
            best_level = min(n_levels - 1, remaining)
            best_alloc[u] = best_level
            remaining -= best_level + 1

        # Branch iterations (simplified)
        n_branches = min(int(np.log2(n_users + 1)) * n_users, 5000)
        for _ in range(n_branches):
            # Random perturbation and check
            idx = rng.integers(0, n_users)
            old_level = best_alloc[idx]
            new_level = rng.integers(0, n_levels)
            delta_budget = new_level - old_level
            if delta_budget <= remaining:
                if rates[idx, new_level] > rates[idx, old_level]:
                    remaining -= delta_budget
                    best_alloc[idx] = new_level

    elapsed = time.perf_counter() - start
    return elapsed / N_REPS


def run():
    rng = np.random.default_rng(42)

    times_drl = []
    times_convex = []
    times_milp = []

    for n in PROBLEM_SIZES:
        print(f"    Problem size N={n}...")
        t_drl = drl_inference(n, rng)
        t_cvx = water_filling(n, rng)
        t_milp = milp_branch_bound(n, rng)

        times_drl.append(t_drl)
        times_convex.append(t_cvx)
        times_milp.append(t_milp)

    # --- Plotting ---
    plt.rcParams.update({
        'font.size': 11, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3,
    })

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.loglog(PROBLEM_SIZES, np.array(times_drl) * 1000, 'o-',
              color='#2ca02c', linewidth=2, markersize=7,
              label='DRL (NN Forward Pass)', markeredgecolor='black',
              markeredgewidth=0.5)
    ax.loglog(PROBLEM_SIZES, np.array(times_convex) * 1000, 's--',
              color='#1f77b4', linewidth=2, markersize=7,
              label='Convex (Water-Filling)', markeredgecolor='black',
              markeredgewidth=0.5)
    ax.loglog(PROBLEM_SIZES, np.array(times_milp) * 1000, '^-.',
              color='#d62728', linewidth=2, markersize=7,
              label='MILP (Branch & Bound)', markeredgecolor='black',
              markeredgewidth=0.5)

    # Reference lines for complexity classes
    n_ref = np.array(PROBLEM_SIZES, dtype=float)
    ax.loglog(n_ref, 0.003 * n_ref, ':', color='gray', alpha=0.4, linewidth=1)
    ax.text(800, 0.003 * 800 * 1.3, 'O(n)', color='gray', fontsize=9, alpha=0.6)
    ax.loglog(n_ref, 0.00001 * n_ref**2, ':', color='gray', alpha=0.4, linewidth=1)
    ax.text(500, 0.00001 * 500**2 * 1.3, 'O(n²)', color='gray', fontsize=9, alpha=0.6)

    ax.set_xlabel('Problem Size (Number of Users)')
    ax.set_ylabel('Decision Time (ms)')
    ax.set_title('Computational Complexity: DRL vs. Convex vs. MILP')
    ax.legend(fontsize=10)

    # Annotate real-time constraint
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.text(12, 1.15, 'Real-time constraint (1 ms)', color='red',
            fontsize=9, alpha=0.7)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig7_complexity.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    print("  [Sim 5] Generated: fig7_complexity.png")


if __name__ == '__main__':
    run()
