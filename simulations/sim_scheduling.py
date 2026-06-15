#!/usr/bin/env python3
"""
Simulation 3: Scheduling Algorithm Comparison
Compares Round Robin, Max Rate, Proportional Fair, and DRL-based scheduling
for 50 users across 20 PRBs over 1000 TTIs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

N_USERS = 50
N_PRBS = 20
N_TTIS = 1000
SCS = 30e3          # Subcarrier spacing (Hz)
BW_PER_PRB = 12 * SCS  # 12 subcarriers per PRB
SNR_RANGE = (0, 25)  # dB
SEEDS = list(range(10))

# DRL scheduling Q-table (pre-trained approximation)
N_POWER_LEVELS = 5


def generate_channel(n_users, n_prbs, rng):
    """Rayleigh fading channel gains."""
    h = rng.rayleigh(1.0, (n_users, n_prbs))
    # Path loss varies per user (distance-based, simplified)
    distance = rng.uniform(50, 500, n_users)  # meters
    path_loss_db = 128.1 + 37.6 * np.log10(distance / 1000)  # COST-231
    path_loss_linear = 10 ** (-path_loss_db / 10)
    return h * np.sqrt(path_loss_linear[:, None])


def get_snr(channel, rng):
    """Compute SNR per user per PRB."""
    noise_power = 1e-13  # -100 dBm
    tx_power = 0.2  # 200 mW per PRB
    return tx_power * channel**2 / noise_power


def shannon_rate(snr):
    """Shannon capacity in bps."""
    return BW_PER_PRB * np.log2(1 + snr)


def round_robin(snr, n_ttis, rng):
    """Round Robin scheduling."""
    throughput = np.zeros(N_USERS)
    for tti in range(n_ttis):
        for prb in range(N_PRBS):
            user = (tti * N_PRBS + prb) % N_USERS
            throughput[user] += shannon_rate(snr[user, prb])
    return throughput / n_ttis


def max_rate(snr, n_ttis, rng):
    """Max Rate (greedy) scheduling."""
    rates = shannon_rate(snr)
    throughput = np.zeros(N_USERS)
    for tti in range(n_ttis):
        # New channel each TTI
        h = rng.rayleigh(1.0, (N_USERS, N_PRBS))
        dist = rng.uniform(50, 500, N_USERS)
        pl_db = 128.1 + 37.6 * np.log10(dist / 1000)
        pl = 10 ** (-pl_db / 10)
        tti_snr = 0.2 * (h * np.sqrt(pl[:, None]))**2 / 1e-13
        tti_rates = shannon_rate(tti_snr)

        for prb in range(N_PRBS):
            best_user = np.argmax(tti_rates[:, prb])
            throughput[best_user] += tti_rates[best_user, prb]
    return throughput / n_ttis


def proportional_fair(snr, n_ttis, rng):
    """Proportional Fair scheduling."""
    throughput = np.zeros(N_USERS)
    avg_throughput = np.ones(N_USERS) * 1e-6  # avoid div by 0
    alpha_ewma = 0.01

    for tti in range(n_ttis):
        h = rng.rayleigh(1.0, (N_USERS, N_PRBS))
        dist = rng.uniform(50, 500, N_USERS)
        pl_db = 128.1 + 37.6 * np.log10(dist / 1000)
        pl = 10 ** (-pl_db / 10)
        tti_snr = 0.2 * (h * np.sqrt(pl[:, None]))**2 / 1e-13
        tti_rates = shannon_rate(tti_snr)

        tti_tp = np.zeros(N_USERS)
        for prb in range(N_PRBS):
            pf_metric = tti_rates[:, prb] / avg_throughput
            best_user = np.argmax(pf_metric)
            tti_tp[best_user] += tti_rates[best_user, prb]

        throughput += tti_tp
        avg_throughput = (1 - alpha_ewma) * avg_throughput + alpha_ewma * (tti_tp + 1e-9)

    return throughput / n_ttis


def drl_scheduler(snr, n_ttis, rng):
    """DRL-based scheduler (simplified Q-learning policy)."""
    throughput = np.zeros(N_USERS)
    avg_throughput = np.ones(N_USERS) * 1e-6
    alpha_ewma = 0.01

    # Pre-trained Q-table approximation: PF-like but with learned bias
    # toward fairness + total throughput balance
    fairness_weight = 0.6
    rate_weight = 0.4

    for tti in range(n_ttis):
        h = rng.rayleigh(1.0, (N_USERS, N_PRBS))
        dist = rng.uniform(50, 500, N_USERS)
        pl_db = 128.1 + 37.6 * np.log10(dist / 1000)
        pl = 10 ** (-pl_db / 10)
        tti_snr = 0.2 * (h * np.sqrt(pl[:, None]))**2 / 1e-13
        tti_rates = shannon_rate(tti_snr)

        tti_tp = np.zeros(N_USERS)
        for prb in range(N_PRBS):
            # DRL-learned metric: adaptive combination
            norm_rate = tti_rates[:, prb] / (tti_rates[:, prb].max() + 1e-9)
            norm_fair = 1.0 / (avg_throughput / (avg_throughput.min() + 1e-9) + 1e-3)

            # Adaptive weights (simulating learned policy)
            load = np.std(avg_throughput) / (np.mean(avg_throughput) + 1e-9)
            w_f = fairness_weight + 0.2 * load
            w_r = rate_weight - 0.1 * load
            drl_metric = w_r * norm_rate + w_f * norm_fair

            best_user = np.argmax(drl_metric)
            tti_tp[best_user] += tti_rates[best_user, prb]

        throughput += tti_tp
        avg_throughput = (1 - alpha_ewma) * avg_throughput + alpha_ewma * (tti_tp + 1e-9)

    return throughput / n_ttis


def jains_fairness(throughput):
    n = len(throughput)
    return (throughput.sum())**2 / (n * (throughput**2).sum() + 1e-12)


def run():
    schedulers = {
        'Round Robin': round_robin,
        'Max Rate': max_rate,
        'Proportional Fair': proportional_fair,
        'DRL Scheduler': drl_scheduler,
    }

    results = {name: {
        'avg_tp': [], 'fairness': [], 'p5_tp': [], 'cell_edge_tp': [],
        'throughputs': [],
    } for name in schedulers}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        channel = generate_channel(N_USERS, N_PRBS, rng)
        snr = get_snr(channel, rng)

        for name, sched_fn in schedulers.items():
            sched_rng = np.random.default_rng(seed)
            tp = sched_fn(snr, N_TTIS, sched_rng)

            results[name]['avg_tp'].append(tp.mean())
            results[name]['fairness'].append(jains_fairness(tp))
            results[name]['p5_tp'].append(np.percentile(tp, 5))
            # Cell-edge: bottom 10% users
            sorted_tp = np.sort(tp)
            results[name]['cell_edge_tp'].append(sorted_tp[:5].mean())
            results[name]['throughputs'].append(tp)

    # --- Plotting ---
    plt.rcParams.update({
        'font.size': 11, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3,
    })
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    sched_names = list(schedulers.keys())
    x = np.arange(len(sched_names))
    width = 0.6

    # Bar chart helper
    def bar_chart(metric_key, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(7, 5))
        means = [np.mean(results[n][metric_key]) for n in sched_names]
        stds = [np.std(results[n][metric_key]) for n in sched_names]
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sched_names, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{mean:.2e}' if mean > 1e3 else f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, filename),
                    bbox_inches='tight', dpi=200)
        plt.close(fig)

    bar_chart('avg_tp', 'Average Throughput (bps)', 
              'Average User Throughput', 'fig5a_avg_throughput.png')
    bar_chart('fairness', "Jain's Fairness Index",
              "Jain's Fairness Index", 'fig5b_fairness.png')
    bar_chart('p5_tp', '5th Percentile Throughput (bps)',
              '5th Percentile Throughput', 'fig5c_p5_throughput.png')
    bar_chart('cell_edge_tp', 'Cell-Edge Throughput (bps)',
              'Cell-Edge User Throughput (Bottom 10%)', 'fig5d_cell_edge.png')

    # CDF plot
    fig, ax = plt.subplots(figsize=(8, 5.5))
    linestyles = ['-', '--', '-.', ':']
    for idx, name in enumerate(sched_names):
        all_tp = np.concatenate(results[name]['throughputs'])
        sorted_tp = np.sort(all_tp)
        cdf = np.arange(1, len(sorted_tp) + 1) / len(sorted_tp)
        ax.plot(sorted_tp, cdf, color=colors[idx], linestyle=linestyles[idx],
                label=name, linewidth=2)
    ax.set_xlabel('User Throughput (bps)')
    ax.set_ylabel('CDF')
    ax.set_title('CDF of User Throughput')
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig5e_cdf_throughput.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    print("  [Sim 3] Generated: fig5a-e scheduling comparison figures")


if __name__ == '__main__':
    run()
