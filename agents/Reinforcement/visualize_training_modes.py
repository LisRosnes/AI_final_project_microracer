

#!/usr/bin/env python3
"""
Visualize and compare different DDPG training modes.
Focus on learning effectiveness and final performance.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('default')
colors = {
    'easy': '#2ecc71',
    'hard': '#e74c3c', 
    'curriculum': '#3498db',
    'imitation': '#f39c12'
}

def load_metadata(mode):
    """Load training metadata for a given mode."""
    log_path = Path('../../logs') / f'ddpg_{mode}_improved_metadata.json'
    if not log_path.exists():
        return None
    with open(log_path) as f:
        return json.load(f)

def extract_metrics(metadata, mode):
    """Extract key metrics from metadata."""
    if metadata is None:
        return None
    
    # Learning curve
    steps = metadata['learning_curve']['steps']
    rewards = metadata['learning_curve']['rewards']
    
    # Evaluation history
    eval_history = metadata['early_stopping'].get('eval_history', [])
    eval_steps = [e['step'] for e in eval_history]
    eval_rewards = [e['reward'] for e in eval_history]
    eval_success_rates = [e['success_rate'] for e in eval_history]
    
    # Best performance
    best_eval_reward = metadata['early_stopping'].get('best_eval_reward', None)
    
    # Steps to first completion
    first_completion_step = None
    for eval_point in eval_history:
        if eval_point['success_rate'] > 0:
            first_completion_step = eval_point['step']
            break
    
    # Final metrics
    final_reward = metadata.get('final_avg_reward', None)
    total_iterations = metadata.get('total_iterations', None)
    
    # Imitation specific
    imitation_data = None
    if mode == 'imitation' and 'imitation' in metadata:
        imitation_data = {
            'steps': metadata['imitation']['imitation_steps'],
            'final_loss': metadata['imitation']['final_imitation_loss'],
            'losses': metadata['imitation'].get('imitation_losses', [])
        }
    
    return {
        'mode': mode,
        'steps': steps,
        'rewards': rewards,
        'eval_steps': eval_steps,
        'eval_rewards': eval_rewards,
        'eval_success_rates': eval_success_rates,
        'best_eval_reward': best_eval_reward,
        'first_completion_step': first_completion_step,
        'final_reward': final_reward,
        'total_iterations': total_iterations,
        'imitation': imitation_data
    }

def plot_learning_curves(metrics_dict, save_path=None):
    """Plot learning curves for all modes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for mode, metrics in metrics_dict.items():
        if metrics is None:
            continue
        ax.plot(metrics['steps'], metrics['rewards'], 
               label=mode.capitalize(), color=colors[mode], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Reward (40 episodes)', fontsize=12, fontweight='bold')
    ax.set_title('Learning Curves: Training Mode Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig

def plot_success_rate_progression(metrics_dict, save_path=None):
    """Plot success rate on hard track over training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for mode, metrics in metrics_dict.items():
        if metrics is None or len(metrics['eval_steps']) == 0:
            continue
        
        # Convert to percentage
        success_pct = [sr * 100 for sr in metrics['eval_success_rates']]
        
        ax.plot(metrics['eval_steps'], success_pct,
               label=mode.capitalize(), color=colors[mode], 
               linewidth=2.5, marker='o', markersize=5, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate on Hard Track (%)', fontsize=12, fontweight='bold')
    ax.set_title('Track Completion Success Rate Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig

def plot_best_eval_reward(metrics_dict, save_path=None):
    """Plot best evaluation reward trajectory."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for mode, metrics in metrics_dict.items():
        if metrics is None or len(metrics['eval_steps']) == 0:
            continue
        
        # Compute cumulative best
        best_so_far = []
        current_best = -np.inf
        for reward in metrics['eval_rewards']:
            current_best = max(current_best, reward)
            best_so_far.append(current_best)
        
        ax.plot(metrics['eval_steps'], best_so_far,
               label=mode.capitalize(), color=colors[mode], 
               linewidth=2.5, marker='s', markersize=4, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Evaluation Reward Achieved', fontsize=12, fontweight='bold')
    ax.set_title('Peak Performance Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.tight_layout()
    return fig

def plot_final_performance_comparison(metrics_dict, save_path=None):
    """Bar charts comparing final performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    modes = []
    best_rewards = []
    best_success_rates = []
    steps_to_first = []
    
    for mode, metrics in metrics_dict.items():
        if metrics is None:
            continue
        
        modes.append(mode.capitalize())
        best_rewards.append(metrics['best_eval_reward'] if metrics['best_eval_reward'] else 0)
        
        # Success rate at the point where best reward was achieved
        best_eval_reward = metrics['best_eval_reward']
        success_at_best = 0
        if best_eval_reward and len(metrics['eval_rewards']) > 0:
            # Find the evaluation point where best reward occurred
            for i, reward in enumerate(metrics['eval_rewards']):
                if reward == best_eval_reward:
                    success_at_best = metrics['eval_success_rates'][i] * 100
                    break
        best_success_rates.append(success_at_best)
        
        # Steps to first completion
        if metrics['first_completion_step']:
            steps_to_first.append(metrics['first_completion_step'])
        else:
            steps_to_first.append(None)  # Never completed
    
    # Plot 1: Best eval reward
    axes[0].bar(modes, best_rewards, color=[colors[m.lower()] for m in modes], alpha=0.8)
    axes[0].set_ylabel('Best Eval Reward', fontsize=11, fontweight='bold')
    axes[0].set_title('Best Evaluation Performance', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(best_rewards):
        axes[0].text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Plot 2: Success rate at best reward point
    axes[1].bar(modes, best_success_rates, color=[colors[m.lower()] for m in modes], alpha=0.8)
    axes[1].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes[1].set_title('Success Rate at Best Performance', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(best_success_rates):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Plot 3: Steps to first completion
    valid_modes = [m for m, s in zip(modes, steps_to_first) if s is not None]
    valid_steps = [s for s in steps_to_first if s is not None]
    valid_colors = [colors[m.lower()] for m, s in zip(modes, steps_to_first) if s is not None]
    
    if valid_modes:
        axes[2].bar(valid_modes, valid_steps, color=valid_colors, alpha=0.8)
        axes[2].set_ylabel('Training Steps', fontsize=11, fontweight='bold')
        axes[2].set_title('Steps to First Completion', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(valid_steps):
            axes[2].text(i, v + max(valid_steps)*0.02, f'{v:,}', ha='center', fontweight='bold', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'No completions', ha='center', va='center', 
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Steps to First Completion', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_steps_to_best_performance(metrics_dict, save_path=None):
    """Plot steps required to reach best performance (100% success rate)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    modes = []
    steps_to_100 = []
    steps_to_best = []
    best_achieved = []
    
    for mode, metrics in metrics_dict.items():
        if metrics is None or len(metrics['eval_success_rates']) == 0:
            continue
        
        modes.append(mode.capitalize())
        
        # Find when model first reaches 100% success rate
        step_100 = None
        for i, sr in enumerate(metrics['eval_success_rates']):
            if sr >= 1.0:  # 100% success
                step_100 = metrics['eval_steps'][i]
                break
        
        # Find the step where best performance was achieved
        if metrics['best_eval_reward'] and len(metrics['eval_rewards']) > 0:
            best_idx = metrics['eval_rewards'].index(metrics['best_eval_reward'])
            step_best = metrics['eval_steps'][best_idx]
            best_sr = metrics['eval_success_rates'][best_idx] * 100
        else:
            step_best = None
            best_sr = 0
        
        steps_to_100.append(step_100)
        steps_to_best.append(step_best)
        best_achieved.append(best_sr)
    
    # Create bar chart
    x = np.arange(len(modes))
    width = 0.35
    
    # Bars for steps to 100%
    bars_100 = []
    bars_best = []
    
    for i, (s100, sbest) in enumerate(zip(steps_to_100, steps_to_best)):
        color = colors[modes[i].lower()]
        
        # Bar for 100% success (if achieved)
        if s100 is not None:
            bars_100.append(ax.bar(x[i] - width/2, s100, width, 
                                  label='100% Success' if i == 0 else '', 
                                  color=color, alpha=0.9, edgecolor='black', linewidth=1.5))
            ax.text(x[i] - width/2, s100 + max([s for s in steps_to_100 if s is not None])*0.02, 
                   f'{s100:,}', ha='center', fontweight='bold', fontsize=9)
        
        # Bar for best performance (if different from 100%)
        if sbest is not None and (s100 is None or sbest != s100):
            bars_best.append(ax.bar(x[i] + width/2, sbest, width,
                                   label='Best Reward' if i == 0 and s100 is not None else '', 
                                   color=color, alpha=0.6, edgecolor='black', linewidth=1.5))
            ax.text(x[i] + width/2, sbest + max([s for s in steps_to_best if s is not None])*0.02,
                   f'{sbest:,}\n({best_achieved[i]:.0f}%)', ha='center', 
                   fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Training Mode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Steps', fontsize=12, fontweight='bold')
    ax.set_title('Steps to Reach Best Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    handles = []
    labels = []
    if any(s100 is not None for s100 in steps_to_100):
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='gray', alpha=0.9, edgecolor='black', linewidth=1.5))
        labels.append('Steps to 100% Success')
    if any(s100 is None or sbest != s100 for s100, sbest in zip(steps_to_100, steps_to_best) if sbest is not None):
        from matplotlib.patches import Patch
        handles.append(Patch(facecolor='gray', alpha=0.6, edgecolor='black', linewidth=1.5))
        labels.append('Steps to Best Reward (if <100%)')
    
    if handles:
        ax.legend(handles, labels, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_summary_table(metrics_dict, save_path=None):
    """Create a summary comparison table."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Mode', 'Total Steps', 'Best Eval\nReward', 'Final Success\nRate (%)', 
               'Steps to\nFirst Success', 'Final Avg\nReward']
    rows = []
    
    for mode, metrics in metrics_dict.items():
        if metrics is None:
            continue
        
        final_success = metrics['eval_success_rates'][-1] * 100 if metrics['eval_success_rates'] else 0
        steps_first = f"{metrics['first_completion_step']:,}" if metrics['first_completion_step'] else "Never"
        
        row = [
            mode.capitalize(),
            f"{metrics['total_iterations']:,}",
            f"{metrics['best_eval_reward']:.2f}" if metrics['best_eval_reward'] else "N/A",
            f"{final_success:.1f}%",
            steps_first,
            f"{metrics['final_reward']:.2f}" if metrics['final_reward'] else "N/A"
        ]
        rows.append(row)
    
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', 
                    loc='center', colWidths=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by mode
    for i, (mode, _) in enumerate(metrics_dict.items(), start=1):
        if metrics_dict[mode] is not None:
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(colors[mode])
                table[(i, j)].set_alpha(0.3)
    
    plt.title('Training Mode Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def main():
    print("="*70)
    print("DDPG Training Modes Visualization")
    print("="*70)
    
    # Load all modes
    modes = ['easy', 'hard', 'curriculum', 'imitation']
    metrics_dict = {}
    
    for mode in modes:
        print(f"Loading {mode} mode data...", end=' ')
        metadata = load_metadata(mode)
        if metadata:
            metrics_dict[mode] = extract_metrics(metadata, mode)
            print("✓")
        else:
            print("✗ (not found)")
            metrics_dict[mode] = None
    
    print()
    
    # Create output directory
    output_dir = Path('../../logs/training_mode_comparisons')
    output_dir.mkdir(exist_ok=True)
    print(f"Saving plots to: {output_dir}\n")
    
    # Generate plots
    print("Generating visualizations...")
    
    print("  1. Learning curves...", end=' ')
    plot_learning_curves(metrics_dict, output_dir / 'learning_curves.png')
    print("✓")
    
    print("  2. Success rate progression...", end=' ')
    plot_success_rate_progression(metrics_dict, output_dir / 'success_rate_progression.png')
    print("✓")
    
    print("  3. Best eval reward trajectory...", end=' ')
    plot_best_eval_reward(metrics_dict, output_dir / 'best_eval_reward.png')
    print("✓")
    
    print("  4. Final performance comparison...", end=' ')
    plot_final_performance_comparison(metrics_dict, output_dir / 'final_performance.png')
    print("✓")
    
    print("  5. Steps to best performance...", end=' ')
    plot_steps_to_best_performance(metrics_dict, output_dir / 'steps_to_best_performance.png')
    print("✓")
    
    print("  6. Summary table...", end=' ')
    plot_summary_table(metrics_dict, output_dir / 'summary_table.png')
    print("✓")
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*70}\n")
    
    # Print summary to console
    print("QUICK SUMMARY:")
    print("-" * 70)
    for mode, metrics in metrics_dict.items():
        if metrics is None:
            continue
        print(f"\n{mode.upper()}:")
        print(f"  Best eval reward: {metrics['best_eval_reward']:.2f}")
        if metrics['eval_success_rates']:
            print(f"  Final success rate: {metrics['eval_success_rates'][-1]*100:.1f}%")
        if metrics['first_completion_step']:
            print(f"  First completion at: {metrics['first_completion_step']:,} steps")
        else:
            print(f"  First completion: Never")
    
    plt.show()

if __name__ == '__main__':
    main()
