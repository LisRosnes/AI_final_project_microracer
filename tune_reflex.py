#!/usr/bin/env python3
"""
Hyperparameter tuning for ReflexAgent using parallel grid search.

Strategy: Grid search with multiprocessing for parallel evaluation.
- Generates all combinations of hyperparameters from param_grid
- Evaluates each config over multiple episodes in parallel
- Ranks results by average speed (primary metric)
- Saves best config to best_config.py for later use

Usage:
    python tune_reflex.py              # Full search (64 configs × 3 eps)
    python tune_reflex.py --quick      # Quick test (2 configs × 2 eps)
    python tune_reflex.py --test       # Same as --quick

Note: For this scale (64 configs), grid search + multiprocessing is optimal.
Optuna was evaluated but offers minimal benefit for small search spaces.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import itertools
from reflex_agent import ReflexAgent
import tracks
import warnings
warnings.filterwarnings('ignore')


def run_episode(agent, max_steps=500):
    """Run single episode, return metrics."""
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
    state = racer.reset()
    
    episode_reward = 0
    total_speed = 0
    step_count = 0
    crashed = False
    
    done = False
    while not done and step_count < max_steps:
        action = agent.act(state)
        state, reward, done = racer.step(action)
        
        if state is not None and len(state) > 4:
            total_speed += state[4]
        
        episode_reward += reward
        step_count += 1
    
    if racer.completation != 0:  # Any failure
        crashed = True
    
    avg_speed = total_speed / max(step_count, 1)
    
    return {
        'reward': episode_reward,
        'avg_speed': avg_speed,
        'steps': step_count,
        'crashed': crashed,
    }


def evaluate_config(config, num_episodes=5):
    """Evaluate a single hyperparameter configuration."""
    agent = ReflexAgent()
    
    # Apply config
    for param_name, value in config.items():
        setattr(agent, param_name, value)
    
    # Run multiple episodes
    results = []
    for _ in range(num_episodes):
        result = run_episode(agent)
        results.append(result)
    
    # Aggregate results
    avg_speed = np.mean([r['avg_speed'] for r in results])
    crash_rate = np.sum([r['crashed'] for r in results]) / num_episodes
    avg_reward = np.mean([r['reward'] for r in results])
    
    return {
        'config': config,
        'avg_speed': avg_speed,
        'crash_rate': crash_rate,
        'avg_reward': avg_reward,
        'num_episodes': num_episodes,
    }


def tune_hyperparameters(param_grid, num_episodes=5, num_workers=None):
    """
    Search hyperparameter space in parallel.
    
    Args:
        param_grid: Dict of parameter_name -> list of values
        num_episodes: Episodes per configuration
        num_workers: Number of parallel workers (default: CPU count)
    
    Returns:
        List of results sorted by avg_speed (descending)
    """
    
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Parallel workers: {num_workers}")
    print(f"Episodes per config: {num_episodes}")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    configs = [
        dict(zip(param_names, values))
        for values in itertools.product(*param_values)
    ]
    
    total_configs = len(configs)
    total_episodes = total_configs * num_episodes
    print(f"\nTotal configurations to test: {total_configs}")
    print(f"Total episodes: {total_episodes}")
    print(f"Estimated time: ~{total_episodes * 0.5 / 60:.1f} minutes (0.5s per episode)")
    print(f"\nStarting parallel evaluation...")
    print(f"{'='*80}\n")
    
    # Run in parallel with progress tracking
    from functools import partial
    eval_func = partial(evaluate_config, num_episodes=num_episodes)
    
    with Pool(num_workers) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(eval_func, configs, chunksize=2)):
            results.append(result)
            # Print progress every 5 configs
            if (i + 1) % 5 == 0:
                best_so_far = max(results, key=lambda x: x['avg_speed'])
                print(
                    f"[{i+1:3d}/{total_configs}] "
                    f"Best so far: speed={best_so_far['avg_speed']:.4f}, "
                    f"crashes={best_so_far['crash_rate']:.0%}, "
                    f"reward={best_so_far['avg_reward']:.2f}"
                )
    
    # Sort by avg_speed (descending)
    results.sort(key=lambda x: x['avg_speed'], reverse=True)
    
    return results


def print_results(results, top_n=10):
    """Print results in a table."""
    print("\n" + "="*100)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*100)
    print(f"\nTop {min(top_n, len(results))} configurations by average speed:\n")
    
    print(f"{'Rank':<5} {'Avg Speed':<12} {'Crash Rate':<12} {'Avg Reward':<12} {'Config':<50}")
    print("-" * 100)
    
    for i, result in enumerate(results[:top_n]):
        config_str = ', '.join([
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result['config'].items()
        ])[:47] + "..."
        
        print(
            f"{i+1:<5} "
            f"{result['avg_speed']:<12.4f} "
            f"{result['crash_rate']:<12.1%} "
            f"{result['avg_reward']:<12.2f} "
            f"{config_str:<50}"
        )
    
    # Print best config details
    best = results[0]
    print("\n" + "="*100)
    print("BEST CONFIGURATION")
    print("="*100)
    print(f"Average Speed: {best['avg_speed']:.4f}")
    print(f"Crash Rate: {best['crash_rate']:.1%}")
    print(f"Average Reward: {best['avg_reward']:.2f}")
    print(f"\nHyperparameters:")
    for param, value in best['config'].items():
        print(f"  {param} = {value}")
    print("="*100)
    
    return best


def save_best_config(best_result, filename='best_config.py'):
    """Save best config as Python code."""
    with open(filename, 'w') as f:
        f.write("# Best hyperparameters found by tuning\n")
        f.write("# Usage: from best_config import BEST_CONFIG\n")
        f.write("#        agent = ReflexAgent()\n")
        f.write("#        for param, value in BEST_CONFIG.items():\n")
        f.write("#            setattr(agent, param, value)\n\n")
        f.write("BEST_CONFIG = {\n")
        for param, value in best_result['config'].items():
            f.write(f"    '{param}': {value},\n")
        f.write("}\n")
        f.write(f"\n# Performance:\n")
        f.write(f"# Average Speed: {best_result['avg_speed']:.4f}\n")
        f.write(f"# Crash Rate: {best_result['crash_rate']:.1%}\n")
        f.write(f"# Average Reward: {best_result['avg_reward']:.2f}\n")
    
    print(f"\nBest config saved to {filename}")


if __name__ == '__main__':
    import sys
    
    # Quick test mode with minimal configs
    if '--quick' in sys.argv or '--test' in sys.argv:
        print("\n🚀 QUICK TEST MODE (2 configurations)\n")
        param_grid = {
            'K_heading': [0.5, 0.7],
            'K_center': [0.3],
            'beta_s': [0.6],
            'v_max': [1.2],
            'K_speed': [0.9],
            'd_caution': [5.0],
        }
        num_episodes = 2
    else:
        # Full search with reduced space
        param_grid = {
            # Steering parameters
            'K_heading': [0.5, 0.7],
            'K_center': [0.3, 0.5],
            'beta_s': [0.6, 0.8],
            
            # Speed parameters
            'v_max': [1.2, 1.5],
            'K_speed': [0.9, 1.1],
            'd_caution': [5.0, 6.0],
        }
        num_episodes = 3
    
    # Run tuning
    results = tune_hyperparameters(
        param_grid,
        num_episodes=num_episodes,
        num_workers=None,  # Use all CPU cores
    )
    
    # Print and save results
    best = print_results(results, top_n=15)
    save_best_config(best)
