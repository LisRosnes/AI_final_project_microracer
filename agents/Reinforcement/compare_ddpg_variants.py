"""Compare DDPG Training Modes: Easy vs Hard vs Curriculum.

This script evaluates DDPG models trained with different difficulty
progressions on the same hard track configuration.

Usage:
    python compare_ddpg_variants.py
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import sys
from pathlib import Path

# Add MicroRacer root to path to find tracks.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tracks


def evaluate_model(model, num_episodes=50, verbose=False):
    """
    Evaluate a model over multiple episodes.
    
    Note: racer.completation is a status code:
        0 = in progress, 1 = completed, 2 = border crash, 3 = wrong direction, 4 = too slow
    
    Returns:
        dict: Contains success_rate, avg_completion, avg_reward, avg_steps, lap_times
    """
    results = {
        'completions': [],  # Status codes (0-4)
        'rewards': [],
        'steps': [],
        'lap_times': [],
        'successes': 0,
        'border_crashes': 0,
        'wrong_direction': 0,
        'too_slow': 0
    }
    
    for ep in range(num_episodes):
        racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
        state = racer.reset()
        
        episode_reward = 0
        episode_steps = 0
        start_time = time.time()
        
        while not racer.done and episode_steps < 2000:
            # Get action from model (no noise during evaluation)
            state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = tf.squeeze(model(state_tensor)).numpy()
            
            # Step environment
            state, reward, done = racer.step(action)
            episode_reward += reward
            episode_steps += 1
        
        lap_time = time.time() - start_time
        
        # Track completion status (status code, not percentage!)
        completion_status = racer.completation
        results['completions'].append(completion_status)
        results['rewards'].append(episode_reward)
        results['steps'].append(episode_steps)
        results['lap_times'].append(lap_time)
        
        # Categorize outcome
        if completion_status == 1:
            results['successes'] += 1
        elif completion_status == 2:
            results['border_crashes'] += 1
        elif completion_status == 3:
            results['wrong_direction'] += 1
        elif completion_status == 4:
            results['too_slow'] += 1
        
        if verbose and (ep + 1) % 10 == 0:
            status_str = {0: 'In Progress', 1: 'Completed', 2: 'Border Crash', 
                         3: 'Wrong Dir', 4: 'Too Slow'}.get(completion_status, 'Unknown')
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Status: {status_str}, "
                  f"Reward: {episode_reward:.2f}")
    
    # Calculate summary statistics
    summary = {
        'success_rate': results['successes'] / num_episodes,
        'avg_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'avg_steps': np.mean(results['steps']),
        'avg_lap_time': np.mean(results['lap_times']),
        'num_episodes': num_episodes,
        'successes': results['successes'],
        'border_crashes': results['border_crashes'],
        'wrong_direction': results['wrong_direction'],
        'too_slow': results['too_slow'],
        'raw_results': results
    }
    
    return summary


def load_training_metadata():
    """Load training time and learning curve data if available."""
    metadata = {}
    
    # Try to load metadata for each training mode
    # First try the improved metadata, then fall back to old training metadata
    for mode in ['easy', 'hard', 'curriculum', 'imitation']:
        improved_meta_file = f"../../logs/ddpg_{mode}_improved_metadata.json"
        old_meta_file = f"../../logs/ddpg_{mode}_training_metadata.json"
        
        # Try improved file first
        if os.path.exists(improved_meta_file):
            try:
                with open(improved_meta_file, 'r') as f:
                    metadata[mode] = json.load(f)
                continue
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to load {improved_meta_file}: {e}")
                print(f"   Trying fallback to old metadata file...")
        
        # Fall back to old file
        if os.path.exists(old_meta_file):
            try:
                with open(old_meta_file, 'r') as f:
                    metadata[mode] = json.load(f)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Failed to load {old_meta_file}: {e}")
                print(f"   Skipping metadata for {mode} mode")
    
    return metadata


def compare_models(num_episodes=50):
    """
    Load and compare DDPG models trained with Easy, Hard, Curriculum, and Imitation modes.
    """
    print("="*80)
    print("COMPARING DDPG TRAINING MODES: Easy vs Hard vs Curriculum vs Imitation")
    print("="*80)
    print(f"Evaluation episodes: {num_episodes}")
    print(f"Evaluation track: HARD (obstacles + chicanes)\n")
    
    # Load training metadata (wall time, learning curves)
    training_metadata = load_training_metadata()
    
    # Load models for each training mode
    print("Loading models...")
    models = {}
    model_configs = {
        'easy': 'weights/ddpg_actor_easy_best',
        'hard': 'weights/ddpg_actor_hard_best',
        'curriculum': 'weights/ddpg_actor_curriculum_best',
        'imitation': 'weights/ddpg_actor_imitation_best'
    }
    
    for mode, path in model_configs.items():
        try:
            models[mode] = keras.models.load_model(path)
            print(f"✓ Loaded {mode.upper()} mode model")
        except Exception as e:
            print(f"✗ Could not load {mode.upper()} mode: {e}")
            models[mode] = None
    
    # Check if at least one model loaded
    loaded_models = {k: v for k, v in models.items() if v is not None}
    if not loaded_models:
        print("\nError: No models found. Please train the models first.")
        print("\nExpected paths:")
        for mode, path in model_configs.items():
            print(f"  {mode}: {path}")
        return
    
    results = {}
    
    # Evaluate each loaded model
    for mode, model in loaded_models.items():
        print("\n" + "-"*80)
        print(f"Evaluating {mode.upper()} mode (trained on {mode} track)...")
        print("-"*80)
        results[mode] = evaluate_model(model, num_episodes, verbose=True)
    
    # Print comparison
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Display results based on how many models loaded
    if len(results) >= 2:
        # Define metrics to compare
        metrics = [
            ('Success Rate', 'success_rate', lambda x: f"{x:.2%}", True),
            ('Avg Reward', 'avg_reward', lambda x: f"{x:.2f}", True),
            ('Std Reward', 'std_reward', lambda x: f"{x:.2f}", False),
            ('Avg Steps', 'avg_steps', lambda x: f"{x:.1f}", False),
            ('Avg Lap Time (s)', 'avg_lap_time', lambda x: f"{x:.2f}", False),
            ('Successes', 'successes', lambda x: f"{x}", True),
            ('Border Crashes', 'border_crashes', lambda x: f"{x}", False),
            ('Wrong Direction', 'wrong_direction', lambda x: f"{x}", False),
            ('Too Slow', 'too_slow', lambda x: f"{x}", False),
        ]
        
        # Print header
        modes_list = list(results.keys())
        col_width = 20
        header = f"{'Metric':<25}"
        for mode in modes_list:
            header += f"{mode.capitalize():<{col_width}}"
        header += "Winner"
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for name, key, fmt, higher_better in metrics:
            row = f"{name:<25}"
            
            # Get values for all models
            values = {mode: results[mode][key] for mode in modes_list}
            
            # Find winner
            if higher_better:
                best_mode = max(values, key=values.get)
            else:
                best_mode = min(values, key=values.get)
            
            # Print values
            for mode in modes_list:
                row += f"{fmt(values[mode]):<{col_width}}"
            row += f"{best_mode.capitalize()} ⭐"
            print(row)
        
        # Statistical significance
        from scipy import stats
        print("\n" + "-"*70)
        print(f"Statistical Test (Reward comparison):")
        
        if len(results) == 2:
            # Two-sample t-test
            modes_list = list(results.keys())
            rewards_1 = results[modes_list[0]]['raw_results']['rewards']
            rewards_2 = results[modes_list[1]]['raw_results']['rewards']
            t_stat, p_value = stats.ttest_ind(rewards_1, rewards_2)
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
        else:
            # ANOVA for multiple groups
            all_rewards = [results[mode]['raw_results']['rewards'] for mode in modes_list]
            f_stat, p_value = stats.f_oneway(*all_rewards)
            print(f"  F-statistic: {f_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print(f"  Result: Significant difference (p < 0.05) ✓")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")
    else:
        # Only one model loaded
        print("\nSingle Model Results:")
        mode = list(results.keys())[0]
        for key, value in results[mode].items():
            if key != 'raw_results':
                print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"../../logs/ddpg_comparison_{timestamp}.json"
    
    # Remove raw_results for cleaner JSON
    for model_name in results:
        if 'raw_results' in results[model_name]:
            results[model_name]['raw_results'] = {
                k: v if isinstance(v, int) else [float(x) for x in v] 
                for k, v in results[model_name]['raw_results'].items()
            }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    
    # Generate visualization
    visualize_comparison(results, training_metadata)
    
    print("\n" + "="*80)


def visualize_comparison(results, training_metadata=None):
    """Create comparison visualizations for multiple training modes."""
    if len(results) < 2:
        print("Need at least 2 models for comparison visualization")
        return
    
    modes = list(results.keys())
    num_modes = len(modes)
    
    # Color scheme for up to 4 modes
    colors = {'easy': '#2ecc71', 'hard': '#e74c3c', 'curriculum': '#3498db', 'imitation': '#9b59b6'}
    mode_colors = [colors.get(mode, '#95a5a6') for mode in modes]
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'DDPG Training Mode Comparison: {" vs ".join([m.upper() for m in modes])}', 
                 fontsize=18, fontweight='bold')
    
    # 1. Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    success_rates = [results[mode]['success_rate'] for mode in modes]
    bars = ax1.bar(modes, success_rates, color=mode_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Success Rate', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, max(success_rates) * 1.2 if max(success_rates) > 0 else 1)
    ax1.set_title('Success Rate on Hard Track', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Average Reward Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    avg_rewards = [results[mode]['avg_reward'] for mode in modes]
    bars = ax2.bar(modes, avg_rewards, color=mode_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Average Reward', fontweight='bold', fontsize=12)
    ax2.set_title('Average Reward Performance', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Reward Distribution (Violin Plot)
    ax3 = fig.add_subplot(gs[0, 2])
    reward_data = [results[mode]['raw_results']['rewards'] for mode in modes]
    parts = ax3.violinplot(reward_data, positions=range(num_modes), showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], mode_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    ax3.set_xticks(range(num_modes))
    ax3.set_xticklabels(modes)
    ax3.set_ylabel('Episode Reward', fontweight='bold', fontsize=12)
    ax3.set_title('Reward Distribution', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Completion Status Stacked Bar
    ax4 = fig.add_subplot(gs[1, 0])
    status_labels = ['Completed', 'Border Crash', 'Wrong Dir', 'Too Slow']
    status_colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    bottoms = [0] * num_modes
    for i, label in enumerate(status_labels):
        status_key = ['successes', 'border_crashes', 'wrong_direction', 'too_slow'][i]
        values = [results[mode][status_key] for mode in modes]
        ax4.bar(modes, values, bottom=bottoms, label=label, color=status_colors[i], 
               alpha=0.7, edgecolor='black', linewidth=1)
        bottoms = [b + v for b, v in zip(bottoms, values)]
    
    ax4.set_ylabel('Episode Count', fontweight='bold', fontsize=12)
    ax4.set_title('Completion Status Distribution', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Learning Curves (if metadata available)
    ax5 = fig.add_subplot(gs[1, 1:])
    has_curves = all(mode in training_metadata and 'learning_curve' in training_metadata[mode] 
                     for mode in modes)
    
    if has_curves:
        for mode, color in zip(modes, mode_colors):
            curve = training_metadata[mode]['learning_curve']
            ax5.plot(curve['steps'], curve['rewards'], 
                    color=color, alpha=0.8, linewidth=2.5, label=mode.upper())
            
            # Add curriculum transitions if available
            if mode == 'curriculum' and 'curriculum' in training_metadata[mode]:
                transitions = training_metadata[mode]['curriculum'].get('transitions', [])
                for trans in transitions:
                    step = trans['step']
                    ax5.axvline(x=step, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax5.text(step, ax5.get_ylim()[1] * 0.95, f" {trans['name']}", 
                            rotation=0, fontsize=8, va='top')
        
        ax5.set_xlabel('Training Steps', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Avg Reward', fontweight='bold', fontsize=12)
        ax5.set_title('Learning Curves During Training', fontweight='bold')
        ax5.legend(fontsize=11, loc='lower right')
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Learning curve data unavailable', 
                ha='center', va='center', transform=ax5.transAxes,
                fontsize=14, style='italic')
        ax5.set_title('Learning Curves During Training', fontweight='bold')
        ax5.axis('off')
    
    # 6. Training Time Comparison
    ax6 = fig.add_subplot(gs[2, 0])
    has_time = all(mode in training_metadata and 'training_time_seconds' in training_metadata[mode] 
                   for mode in modes)
    
    if has_time:
        times = [training_metadata[mode]['training_time_seconds'] / 3600 for mode in modes]
        bars = ax6.bar(modes, times, color=mode_colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Training Time (hours)', fontweight='bold', fontsize=12)
        ax6.set_title('Training Wall Time', fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}h', ha='center', va='bottom', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Training time\ndata unavailable', 
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=12, style='italic')
        ax6.set_title('Training Wall Time', fontweight='bold')
        ax6.axis('off')
    
    # 7. Average Steps Comparison
    ax7 = fig.add_subplot(gs[2, 1])
    avg_steps = [results[mode]['avg_steps'] for mode in modes]
    bars = ax7.bar(modes, avg_steps, color=mode_colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax7.set_ylabel('Average Steps per Episode', fontweight='bold', fontsize=12)
    ax7.set_title('Episode Length', fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 8. Performance Summary Radar/Spider Chart
    ax8 = fig.add_subplot(gs[2, 2], projection='polar')
    
    # Normalize metrics for radar chart
    categories = ['Success\nRate', 'Avg\nReward', 'Avg\nSteps']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    for mode, color in zip(modes, mode_colors):
        # Normalize values to 0-1 scale
        success_norm = results[mode]['success_rate']
        reward_norm = (results[mode]['avg_reward'] + 3) / 10  # Normalize reward range
        steps_norm = min(results[mode]['avg_steps'] / 1000, 1)  # Normalize steps
        
        values = [success_norm, reward_norm, steps_norm]
        values += values[:1]
        
        ax8.plot(angles, values, 'o-', linewidth=2, label=mode.upper(), color=color, alpha=0.7)
        ax8.fill(angles, values, alpha=0.2, color=color)
    
    ax8.set_xticks(angles[:-1])
    ax8.set_xticklabels(categories, fontsize=10)
    ax8.set_ylim(0, 1)
    ax8.set_title('Overall Performance', fontweight='bold', pad=20)
    ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax8.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"../../logs/ddpg_modes_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {plot_file}")
    plt.show(block=False)
    plt.pause(0.001)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare DDPG Training Modes')
    parser.add_argument('--episodes', '-n', type=int, default=50,
                        help='Number of evaluation episodes (default: 50)')
    args = parser.parse_args()
    
    compare_models(num_episodes=args.episodes)

