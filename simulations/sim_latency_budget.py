#!/usr/bin/env python3
"""
Fig. 11: End-to-End Latency Budget Decomposition Across UMRO-5G Layers
for URLLC Service (1 ms Budget).

Horizontal stacked bar chart – each bar is one scenario, segments are
latency components. All values in microseconds (μs).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

# Component definitions: (label, color)
COMPONENTS = [
    ('L1 Processing (PHY/MAC @ DU)',       '#1f4e79'),
    ('L1 Transmission (Fronthaul)',         '#2980b9'),
    ('L2 VNF Processing (UPF/AMF)',         '#1e8449'),
    ('L2 Transport (Fronthaul+Midhaul)',    '#58d68d'),
    ('L3 DRL Inference (Near-RT RIC)',      '#e67e22'),
    ('L4 Orchestration Overhead',           '#c0392b'),
    ('Propagation + UE Processing',         '#7f8c8d'),
    ('Safety Margin',                       '#f4d03f'),
]

# Scenarios: rows = scenarios, cols = component values (μs)
# Order matches COMPONENTS above
SCENARIOS = [
    {
        'name': 'Baseline 5G\n(no UMRO-5G)',
        'values': [150, 80, 200, 150, 0, 50, 220, 150],
    },
    {
        'name': 'UMRO-5G\n(L1+L2 only)',
        'values': [100, 50, 150, 100, 0, 0, 200, 400],
    },
    {
        'name': 'Full UMRO-5G\n(all layers)',
        'values': [100, 50, 150, 100, 100, 0, 200, 300],
    },
]

DEADLINE_US = 1000   # URLLC deadline in μs


def run():
    plt.rcParams.update({
        'font.size': 10.5, 'axes.grid': False,
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    y_positions = np.arange(len(SCENARIOS))
    bar_height = 0.52

    for si, scenario in enumerate(SCENARIOS):
        left = 0.0
        for ci, (comp_label, color) in enumerate(COMPONENTS):
            val = scenario['values'][ci]
            if val > 0:
                ax.barh(y_positions[si], val, left=left, height=bar_height,
                        color=color, edgecolor='white', linewidth=0.6)
                # Label inside segment if wide enough
                if val >= 60:
                    ax.text(left + val / 2, y_positions[si],
                            f'{val}', ha='center', va='center',
                            fontsize=7.5, color='white', fontweight='bold')
            left += val

        # Total label at end of bar
        total = sum(scenario['values'])
        ax.text(total + 15, y_positions[si],
                f'{total} μs', ha='left', va='center', fontsize=9)

    # Deadline reference line
    ax.axvline(x=DEADLINE_US, color='red', linestyle='--', linewidth=1.8,
               label=f'URLLC Deadline ({DEADLINE_US} μs = 1 ms)')
    ax.text(DEADLINE_US + 20, len(SCENARIOS) - 0.05,
            '1 ms Deadline', color='red', fontsize=9, va='top')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s['name'] for s in SCENARIOS], fontsize=10)
    ax.set_xlabel('Latency Component (μs)')
    ax.set_xlim(0, 1200)
    ax.set_title(
        'Fig. 11: End-to-End Latency Budget Decomposition Across UMRO-5G Layers\n'
        'for URLLC Service (1 ms Budget)',
        fontsize=11, fontweight='bold',
    )

    # Legend
    patches = [mpatches.Patch(color=c, label=lbl) for lbl, c in COMPONENTS]
    ax.legend(handles=patches, loc='lower center',
              bbox_to_anchor=(0.5, -0.38), ncol=4,
              fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'fig11_latency_budget.png')
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  [Sim 11] Generated: fig11_latency_budget.png')


if __name__ == '__main__':
    run()
