"""Regenerate Figure 3, the URLLC end-to-end latency budget decomposition.

The submitted image carries a baked-in title, "Fig. 11: ...", inherited from a
different manuscript whose figure numbering does not match this one. The caption
in the document carries the title, so nothing is drawn inside the axes here.

The component values are the illustrative deployment budget of the submitted
article, unchanged. They are a design allocation, not a measurement, and the
caption says so. The one figure that is measured -- the Layer 3 inference time --
is reported in Section 8.8 as 5-15 us, comfortably inside the 100 us the budget
allocates to it; the allocation is left at the conservative 100 us.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# colour-blind-safe (Okabe-Ito), ordered as the components are stacked
COMPONENTS = [
    ('L1 processing (PHY/MAC at O-DU)',      '#0072B2'),
    ('L1 transmission (fronthaul)',          '#56B4E9'),
    ('L2 VNF processing (UPF/AMF)',          '#009E73'),
    ('L2 transport (fronthaul + midhaul)',   '#66DDAA'),
    ('L3 inference (Near-RT RIC)',           '#E69F00'),
    ('L4 orchestration overhead',            '#D55E00'),
    ('Propagation and UE processing',        '#999999'),
    ('Remaining margin',                     '#F0E442'),
]

SCENARIOS = [
    ('Baseline 5G\n(no UMRO-5G)',        [150,  80, 200, 150,   0,  50, 220, 150]),
    ('UMRO-5G\n(L1 and L2 only)',        [100,  50, 150, 100,   0,   0, 200, 400]),
    ('Full UMRO-5G\n(all layers)',       [100,  50, 150, 100, 100,   0, 200, 300]),
]

BUDGET = 1000.0

fig, ax = plt.subplots(figsize=(7.2, 2.9))
ypos = np.arange(len(SCENARIOS))

for i, (label, colour) in enumerate(COMPONENTS):
    widths = np.array([vals[i] for _, vals in SCENARIOS], dtype=float)
    lefts = np.array([sum(vals[:i]) for _, vals in SCENARIOS], dtype=float)
    ax.barh(ypos, widths, left=lefts, height=0.58, color=colour,
            edgecolor='white', linewidth=0.6, label=label, zorder=2)
    for y, (w, l) in enumerate(zip(widths, lefts)):
        if w >= 90:                      # only label segments wide enough to hold it
            ax.text(l + w / 2, y, f'{int(w)}', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold', zorder=3)

ax.axvline(BUDGET, color='#CC0000', linestyle='--', linewidth=1.4, zorder=4)
ax.text(BUDGET + 12, len(SCENARIOS) - 0.35, '1 ms deployment target',
        color='#CC0000', fontsize=7.5, va='center')

ax.set_yticks(ypos)
ax.set_yticklabels([s for s, _ in SCENARIOS], fontsize=8)
ax.set_xlabel('Latency contribution (µs)', fontsize=8.5)
ax.set_xlim(0, 1180)
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ('top', 'right', 'left'):
    ax.spines[side].set_visible(False)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=3,
          fontsize=6.8, frameon=False, handlelength=1.4, columnspacing=1.2)

fig.tight_layout()
fig.savefig('figures/fig3_latency_budget.png', dpi=300, bbox_inches='tight')
print('wrote figures/fig3_latency_budget.png')
for name, vals in SCENARIOS:
    assert sum(vals) == BUDGET, (name, sum(vals))
print('all scenarios sum to the 1000 us budget')
