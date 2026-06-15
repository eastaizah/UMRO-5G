#!/usr/bin/env python3
"""
Fig. 8: UMRO-5G Four-Layer Hierarchical Architecture diagram.
Generates a conceptual architecture figure showing four horizontal layers,
inter-layer interfaces, and three nested control loops.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURE_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)


LAYERS = [
    {
        'label': 'Layer 1\nInfrastructure',
        'color': '#1f77b4',
        'text': ('O-RU Arrays / PRBs  |  COTS Servers (CPU/GPU)\n'
                 'Fronthaul / Midhaul / Backhaul'),
    },
    {
        'label': 'Layer 2\nVirtualization\n& Slicing',
        'color': '#2ca02c',
        'text': ('NFVI (KVM / Kubernetes)  |  SDN Controller\n'
                 'NSI_eMBB  \u2295  NSI_URLLC  \u2295  NSI_mMTC'),
    },
    {
        'label': 'Layer 3\nIntelligence',
        'color': '#ff7f0e',
        'text': ('Near-RT RIC (xApps: Traffic Steering, QoS Opt.)\n'
                 'Non-RT RIC (rApps: Policy, ML Training)\n'
                 'DQN / QMIX / FedDQN / GNN-DRL Engines'),
    },
    {
        'label': 'Layer 4\nOrchestration',
        'color': '#d62728',
        'text': ('ETSI MANO (NFVO, VNFM, VIM)  |  3GPP SA5 (CSMF, NSMF, NSSMF)\n'
                 'SLA Engine  |  ZSM Intent Interface'),
    },
]

INTERFACES = [
    'I\u2081\u2082: PRB telemetry / Resource reservation',
    'I\u2082\u2083: E2/O1 KPIs / RRM actions',
    'I\u2083\u2084: A1/O1 Policy / Lifecycle events',
]

CONTROL_LOOPS = [
    {'label': 'Fast Loop\n(<10 ms)',   'span': (0, 0),   'color': '#1f77b4'},
    {'label': 'Medium Loop\n(10 ms\u20131 s)', 'span': (0, 2), 'color': '#2ca02c'},
    {'label': 'Slow Loop\n(>1 s)',     'span': (0, 3),   'color': '#d62728'},
]


def run():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.3, 10.3)
    ax.axis('off')

    layer_height = 1.8
    layer_gap = 0.55
    left_label_w = 1.65
    box_x = left_label_w + 0.05
    box_w = 8.2
    n = len(LAYERS)

    layer_bottoms = []
    for i in range(n):
        y_bottom = i * (layer_height + layer_gap)
        layer_bottoms.append(y_bottom)

    # Draw layer boxes
    for i, layer in enumerate(LAYERS):
        y = layer_bottoms[i]

        # Main content box
        rect = FancyBboxPatch(
            (box_x, y), box_w, layer_height,
            boxstyle='round,pad=0.05',
            facecolor=layer['color'], edgecolor='white',
            linewidth=1.5, alpha=0.82, zorder=2,
        )
        ax.add_patch(rect)

        # Bold layer label on left
        ax.text(
            box_x - 0.12, y + layer_height / 2,
            layer['label'],
            ha='right', va='center',
            fontsize=10, fontweight='bold', color=layer['color'],
        )

        # Component text inside box
        ax.text(
            box_x + box_w / 2, y + layer_height / 2,
            layer['text'],
            ha='center', va='center',
            fontsize=8.5, color='white', linespacing=1.6,
        )

    # Inter-layer interface arrows and labels
    for i, iface_label in enumerate(INTERFACES):
        y_arrow = layer_bottoms[i] + layer_height
        mid_y = y_arrow + layer_gap / 2
        ax.annotate(
            '', xy=(box_x + box_w * 0.5, y_arrow + layer_gap),
            xytext=(box_x + box_w * 0.5, y_arrow),
            arrowprops=dict(arrowstyle='<->', color='#444444',
                            lw=1.4, mutation_scale=14),
            zorder=3,
        )
        ax.text(
            box_x + box_w * 0.5 + 0.18, mid_y,
            iface_label,
            ha='left', va='center',
            fontsize=8, color='#333333', style='italic',
        )

    # Control loops on the right side
    right_x = box_x + box_w + 0.18
    loop_label_x = right_x + 0.55
    for li, loop in enumerate(CONTROL_LOOPS):
        i_lo, i_hi = loop['span']
        y_lo = layer_bottoms[i_lo]
        y_hi = layer_bottoms[i_hi] + layer_height
        mid_y = (y_lo + y_hi) / 2
        height = y_hi - y_lo
        c = loop['color']

        # Bracket: vertical line + horizontal ticks
        ax.plot([right_x, right_x], [y_lo + 0.05, y_hi - 0.05],
                color=c, lw=2.2, zorder=4)
        ax.plot([right_x, right_x + 0.18], [y_lo + 0.05, y_lo + 0.05],
                color=c, lw=2.2, zorder=4)
        ax.plot([right_x, right_x + 0.18], [y_hi - 0.05, y_hi - 0.05],
                color=c, lw=2.2, zorder=4)
        ax.text(
            loop_label_x, mid_y,
            loop['label'],
            ha='left', va='center',
            fontsize=8.5, color=c, fontweight='bold',
        )

    # Title
    ax.set_title(
        'Fig. 8: UMRO-5G Four-Layer Hierarchical Architecture\n'
        'with Three Nested Control Loops',
        fontsize=12, fontweight='bold', pad=14,
    )

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, 'fig8_umro5g_architecture.png')
    fig.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print('  [Sim 8] Generated: fig8_umro5g_architecture.png')


if __name__ == '__main__':
    run()
