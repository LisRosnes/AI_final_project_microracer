#!/usr/bin/env python3
"""
Visualize Optuna tuning results for ReflexAgent.

Creates plots showing:
1. Survival time distribution (histogram)
2. Objective value progression (best vs trial)
3. Top 10 best configurations
4. Parameter importance for best configs
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def parse_tune_optuna_log(log_file):
    """Parse tune_optuna.log and extract trial results."""
    results = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse lines like: "Trial 10 finished with value: 58.0"
            if "Trial" in line and "finished with value:" in line:
                try:
                    parts = line.split("Trial ")[1].split(" finished with value: ")
                    trial_id = int(parts[0])
                    value_str = parts[1].split(" and parameters:")[0]
                    value = float(value_str)
                    results.append({'trial': trial_id, 'value': value})
                except (IndexError, ValueError):
                    continue
    
    return results


def plot_survival_distribution(results, output_file="survival_distribution.png"):
    """Plot histogram of survival times."""
    if not results:
        print("❌ No results to plot")
        return
    
    values = [r['value'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.1f}')
    ax1.axvline(np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.1f}')
    ax1.set_xlabel('Steps Before Crash', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Survival Time Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Convergence plot
    best_values = []
    current_best = 0
    for r in results:
        current_best = max(current_best, r['value'])
        best_values.append(current_best)
    
    ax2.plot(best_values, linewidth=2, color='darkblue')
    ax2.fill_between(range(len(best_values)), best_values, alpha=0.3)
    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Best Objective Value (Steps)', fontsize=12)
    ax2.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved to {output_file}")
    plt.close()


def print_statistics(results):
    """Print summary statistics."""
    if not results:
        print("❌ No results")
        return
    
    values = [r['value'] for r in results]
    
    print("\n" + "="*60)
    print("OPTIMIZATION STATISTICS")
    print("="*60)
    print(f"Total trials: {len(results)}")
    print(f"Best value: {max(values):.0f} steps")
    print(f"Mean value: {np.mean(values):.1f} ± {np.std(values):.1f} steps")
    print(f"Median value: {np.median(values):.0f} steps")
    print(f"Min value: {min(values):.0f} steps")
    print(f"Max value: {max(values):.0f} steps")
    
    # Distribution
    print(f"\nSurvival distribution:")
    thresholds = [50, 100, 150, 200]
    for threshold in thresholds:
        count = sum(1 for v in values if v >= threshold)
        pct = 100 * count / len(values)
        print(f"  ≥{threshold} steps: {count} configs ({pct:.1f}%)")
    
    print("="*60 + "\n")


def main():
    log_file = Path("/Users/calebfikes/Documents/Projects/AI_final_project_microracer/tune_optuna_500.log")
    output_dir = Path("/Users/calebfikes/Documents/Projects/AI_final_project_microracer")
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        print("   Run: python tune_optuna.py --search 500")
        return
    
    print("📊 Parsing tune_optuna log...")
    results = parse_tune_optuna_log(log_file)
    
    if not results:
        print("❌ Could not parse results from log")
        return
    
    print(f"✅ Found {len(results)} trial results")
    
    print_statistics(results)
    plot_survival_distribution(results, output_dir / "survival_distribution.png")


if __name__ == "__main__":
    main()
