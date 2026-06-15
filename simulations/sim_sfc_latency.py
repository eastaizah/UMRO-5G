#!/usr/bin/env python3
"""
Simulation 4: SFC End-to-End Latency Sensitivity Analysis
Analyzes SFC latency under Jackson queuing network model.
Compares analytical M/M/1 vs discrete-event simulation for chains of 3, 5, and 7 VNFs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# VNF configurations
VNF_NAMES_5 = ['Firewall', 'NAT', 'Load Bal.', 'DPI', 'Proxy']
SERVICE_RATES_5 = np.array([500, 800, 1000, 600, 400])  # packets/s

# 3-VNF chain (subset)
SERVICE_RATES_3 = np.array([500, 1000, 400])

# 7-VNF chain (extended)
SERVICE_RATES_7 = np.array([500, 800, 1000, 600, 400, 700, 350])

ARRIVAL_RATES = np.linspace(50, 350, 15)
N_PACKETS_SIM = 5000  # per arrival rate point
URLLC_THRESHOLD_MS = 1.0


def analytical_mm1_latency(arrival_rate, service_rates):
    """E2E latency under M/M/1 Jackson network (sum of sojourn times)."""
    latencies = []
    for mu in service_rates:
        if arrival_rate >= mu:
            latencies.append(np.inf)
        else:
            latencies.append(1.0 / (mu - arrival_rate))
    return sum(latencies) * 1000  # convert to ms


def des_sfc_latency(arrival_rate, service_rates, rng, n_packets=N_PACKETS_SIM):
    """Discrete-event simulation of SFC with M/M/1 queues at each VNF."""
    n_vnfs = len(service_rates)

    # Generate inter-arrival times
    inter_arrivals = rng.exponential(1.0 / arrival_rate, n_packets)
    arrival_times = np.cumsum(inter_arrivals)

    # Process through each VNF
    departure_times = arrival_times.copy()

    for vnf_idx in range(n_vnfs):
        mu = service_rates[vnf_idx]
        service_times = rng.exponential(1.0 / mu, n_packets)

        vnf_arrival = departure_times.copy()
        vnf_departure = np.zeros(n_packets)

        for i in range(n_packets):
            if i == 0:
                start = vnf_arrival[i]
            else:
                start = max(vnf_arrival[i], vnf_departure[i - 1])
            vnf_departure[i] = start + service_times[i]

        departure_times = vnf_departure

    e2e_latency = departure_times - arrival_times
    # Discard warmup (first 10%)
    warmup = int(0.1 * n_packets)
    return np.mean(e2e_latency[warmup:]) * 1000  # ms


def run():
    chain_configs = {
        '3-VNF': SERVICE_RATES_3,
        '5-VNF': SERVICE_RATES_5,
        '7-VNF': SERVICE_RATES_7,
    }

    analytical_results = {name: [] for name in chain_configs}
    sim_results = {name: [] for name in chain_configs}
    sim_std = {name: [] for name in chain_configs}

    for name, rates in chain_configs.items():
        for lam in ARRIVAL_RATES:
            # Analytical
            lat_a = analytical_mm1_latency(lam, rates)
            analytical_results[name].append(lat_a)

            # Simulation (multiple runs for CI)
            lat_sims = []
            for seed in range(5):
                rng = np.random.default_rng(42 + seed + int(lam * 10))
                lat_s = des_sfc_latency(lam, rates, rng)
                lat_sims.append(lat_s)
            sim_results[name].append(np.mean(lat_sims))
            sim_std[name].append(np.std(lat_sims))

    # --- Plotting ---
    plt.rcParams.update({
        'font.size': 11, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3,
    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    chain_colors = {'3-VNF': '#1f77b4', '5-VNF': '#ff7f0e', '7-VNF': '#2ca02c'}
    chain_order = ['3-VNF', '5-VNF', '7-VNF']

    for idx, name in enumerate(chain_order):
        ax = axes[idx]
        rates = chain_configs[name]
        stability_limit = min(rates)

        valid = ARRIVAL_RATES < stability_limit * 0.98
        ana = np.array(analytical_results[name])
        sim = np.array(sim_results[name])
        std = np.array(sim_std[name])

        # Cap infinite values for plotting
        ana_plot = np.where(np.isfinite(ana), ana, np.nan)
        sim_plot = np.where(valid, sim, np.nan)
        std_plot = np.where(valid, std, np.nan)

        ax.plot(ARRIVAL_RATES, ana_plot, '-', color=chain_colors[name],
                linewidth=2, label='Analytical (M/M/1)')
        ax.plot(ARRIVAL_RATES[valid], sim_plot[valid], 'o',
                color=chain_colors[name], markersize=5,
                markeredgecolor='black', markeredgewidth=0.5,
                label='DES')
        ax.fill_between(ARRIVAL_RATES[valid],
                        (sim - std)[valid], (sim + std)[valid],
                        alpha=0.15, color=chain_colors[name])

        # URLLC threshold
        ax.axhline(y=URLLC_THRESHOLD_MS, color='red', linestyle='--',
                    linewidth=1.5, label='URLLC (1 ms)')

        # Stability limit
        ax.axvline(x=stability_limit, color='gray', linestyle=':',
                    linewidth=1.2, alpha=0.7,
                    label=f'Stability limit ({stability_limit} pkt/s)')

        ax.set_xlabel('Arrival Rate (packets/s)')
        if idx == 0:
            ax.set_ylabel('E2E Latency (ms)')
        ax.set_title(f'{name} Chain')
        ax.legend(fontsize=8, loc='upper left')
        ax.set_ylim(0, min(50, max(ana_plot[np.isfinite(ana_plot)]) * 1.3)
                     if np.any(np.isfinite(ana_plot)) else 50)

    fig.suptitle('SFC End-to-End Latency: Analytical vs. Simulation',
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig6_sfc_latency.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    # Combined plot: all chains together
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in chain_order:
        rates = chain_configs[name]
        stability_limit = min(rates)
        valid = ARRIVAL_RATES < stability_limit * 0.98
        ana = np.array(analytical_results[name])
        ana_plot = np.where(np.isfinite(ana), ana, np.nan)

        ax.plot(ARRIVAL_RATES, ana_plot, linewidth=2, color=chain_colors[name],
                label=f'{name} (Analytical)')
        sim = np.array(sim_results[name])
        ax.plot(ARRIVAL_RATES[valid], sim[valid], 'o', color=chain_colors[name],
                markersize=5, markeredgecolor='black', markeredgewidth=0.5,
                label=f'{name} (Simulation)')

    ax.axhline(y=URLLC_THRESHOLD_MS, color='red', linestyle='--',
               linewidth=1.5, label='URLLC Threshold (1 ms)')
    ax.set_xlabel('Arrival Rate (packets/s)')
    ax.set_ylabel('E2E Latency (ms)')
    ax.set_title('SFC Latency Comparison Across Chain Lengths')
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 30)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'fig6b_sfc_latency_combined.png'),
                bbox_inches='tight', dpi=200)
    plt.close(fig)

    print("  [Sim 4] Generated: fig6_sfc_latency.png, fig6b_sfc_latency_combined.png")


if __name__ == '__main__':
    run()
