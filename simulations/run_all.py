#!/usr/bin/env python3
"""
Unified Runner: Executes all 5G orchestration simulations sequentially
and prints a summary of generated figures.
"""

import os
import sys
import time
import importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURE_DIR, exist_ok=True)

sys.path.insert(0, SCRIPT_DIR)

SIMULATIONS = [
    ('sim_multi_slice', 'Simulation 1: Multi-Slice Resource Allocation (Monte Carlo)'),
    ('sim_drl_convergence', 'Simulation 2: DRL Convergence Comparison'),
    ('sim_scheduling', 'Simulation 3: Scheduling Algorithm Comparison'),
    ('sim_sfc_latency', 'Simulation 4: SFC Latency Sensitivity Analysis'),
    ('sim_complexity', 'Simulation 5: Computational Complexity Comparison'),
    ('sim_architecture_fig', 'Simulation 6: UMRO-5G Architecture Diagram (Fig. 8)'),
    ('sim_ris_optimization', 'Simulation 7: RIS Phase Shift Optimization (Fig. 9)'),
    ('sim_ee_tradeoff', 'Simulation 8: Energy Efficiency–Throughput Trade-off (Fig. 10)'),
    ('sim_latency_budget', 'Simulation 9: Latency Budget Decomposition (Fig. 11)'),
]


def main():
    print("=" * 70)
    print("  5G Resource Management & Orchestration — Simulation Suite")
    print("  Target: MDPI Computation (Q2) Survey Article")
    print("=" * 70)

    total_start = time.time()
    results = []

    for module_name, description in SIMULATIONS:
        print(f"\n{'─' * 60}")
        print(f"  {description}")
        print(f"{'─' * 60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.run()
            elapsed = time.time() - t0
            results.append((description, elapsed, 'OK'))
            print(f"  ✓ Completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results.append((description, elapsed, f'FAILED: {e}'))
            print(f"  ✗ Failed after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    for desc, elapsed, status in results:
        marker = '✓' if status == 'OK' else '✗'
        print(f"  {marker} {desc}")
        print(f"    Time: {elapsed:.1f}s | Status: {status}")

    print(f"\n  Total time: {total_elapsed:.1f}s")

    # List generated figures
    print(f"\n  Generated figures in {FIGURE_DIR}/:")
    if os.path.isdir(FIGURE_DIR):
        figures = sorted(f for f in os.listdir(FIGURE_DIR) if f.endswith('.png'))
        for fig in figures:
            size_kb = os.path.getsize(os.path.join(FIGURE_DIR, fig)) / 1024
            print(f"    • {fig} ({size_kb:.0f} KB)")
        print(f"\n  Total: {len(figures)} figure(s)")
    else:
        print("    (no figures directory found)")

    print(f"{'=' * 70}")

    # Exit with error if any simulation failed
    if any(s != 'OK' for _, _, s in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
