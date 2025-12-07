#!/usr/bin/env python3
"""
Monitor the Optuna tuning run.
Shows progress and best stable config found so far.
"""

import os
import re
import sys

def monitor_tuning(log_file='tune_optuna.log'):
    """Monitor tuning progress from log file."""
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract trial progress
    trials_match = re.search(r'(\d+)/500', content)
    if trials_match:
        current_trial = int(trials_match.group(1))
        progress = (current_trial / 500) * 100
        print(f"\n{'='*70}")
        print(f"TUNING PROGRESS: {current_trial}/500 trials ({progress:.1f}%)")
        print(f"{'='*70}")
    else:
        print("Tuning not yet started or in progress...")
        return
    
    # Count stable configs found
    stable_count = content.count("Stable configs (0% crashes):")
    
    # Try to extract best stable speed from most recent results section
    results_section = content.split("OPTIMIZATION RESULTS")[-1] if "OPTIMIZATION RESULTS" in content else ""
    
    if "Stable configs (0% crashes):" in results_section:
        lines = results_section.split('\n')
        for line in lines:
            if "Stable configs (0% crashes):" in line:
                print(f"\nFinal result: {line.strip()}")
            elif "Mean stable speed:" in line:
                print(f"{line.strip()}")
    else:
        # Count crashes vs non-crashes in trials
        trial_lines = [l for l in content.split('\n') if 'Trial' in l and 'finished with value:' in l]
        if trial_lines:
            crashed = sum(1 for l in trial_lines if 'value: 0.0' in l)
            non_crashed = len(trial_lines) - crashed
            print(f"\nTrials crashed so far: {crashed}")
            print(f"Trials with any success: {non_crashed}")
            if non_crashed > 0:
                print(f"✅ Found stable configurations! Optimization in progress...")
            else:
                print(f"⏳ Still searching... (Optuna will adapt search space)")
    
    # Show ETA
    if current_trial > 0:
        time_per_trial = 0.5  # rough estimate in seconds
        remaining_trials = 500 - current_trial
        eta_minutes = (remaining_trials * time_per_trial * 5) / 60
        print(f"\nEstimated time remaining: ~{eta_minutes:.0f} minutes")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    monitor_tuning()
