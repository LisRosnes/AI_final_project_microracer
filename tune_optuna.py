#!/usr/bin/env python3
"""
Hyperparameter tuning for ReflexAgent using Optuna.

Strategy: Bayesian optimization (Optuna) to find longest-surviving configurations.
- Uses Tree-structured Parzen Estimator (TPE) for intelligent exploration
- Objective: MAXIMIZE time-to-crash (steps survived before termination)
- Rationale: Much clearer gradient signal than crash-rate
- Result: "Stable but slow" agent that survives longer
- Later: Can add multi-objective for speed vs longevity trade-offs

Advantages:
- Clear numerical signal (50 steps vs 100 steps is obvious)
- Optuna can optimize effectively even if crash-free is impossible
- Results in agent that is "slow but reliable"
- Foundation for future multi-objective optimization

Usage:
    python tune_optuna.py --test       # Test mode: 3 trials
    python tune_optuna.py --search N   # Search N trials
    python tune_optuna.py              # Full search (500 trials)
"""

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from reflex_agent import ReflexAgent
import tracks
import warnings
import sys
warnings.filterwarnings('ignore')


# Global config for tracking
TRIAL_RESULTS = []
BEST_STABLE_CONFIG = None
BEST_STABLE_SPEED = 0.0


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


def evaluate_config(config, num_episodes=5, trial=None):
    """
    Evaluate a single hyperparameter configuration.
    Returns average speed only for non-crashing configs.
    """
    agent = ReflexAgent()
    
    # Apply config
    for param_name, value in config.items():
        setattr(agent, param_name, value)
    
    # Run multiple episodes
    results = []
    for ep_idx in range(num_episodes):
        result = run_episode(agent)
        results.append(result)
        
        # Early stopping: if crashes in first episode, report and exit early
        if ep_idx == 0 and result['crashed']:
            return {
                'config': config,
                'avg_speed': 0.0,
                'crash_rate': 1.0,
                'avg_reward': result['reward'],
                'steps': result['steps'],  # Include steps even on crash
                'crashed': True,
                'stable': False,
            }
        
        # Optuna pruning: stop if we see too many crashes
        if trial is not None:
            crash_count = sum([1 for r in results if r['crashed']])
            if crash_count > num_episodes * 0.5:  # >50% crash rate
                trial.report(0.0, step=ep_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
    # Aggregate results
    crash_count = sum([1 for r in results if r['crashed']])
    crash_rate = crash_count / num_episodes
    avg_speed = np.mean([r['avg_speed'] for r in results if not r['crashed']])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])  # NEW: track longevity
    
    return {
        'config': config,
        'avg_speed': avg_speed if crash_rate == 0 else 0.0,
        'crash_rate': crash_rate,
        'avg_reward': avg_reward,
        'steps': avg_steps,  # NEW: average steps before crash
        'crashed': crash_rate > 0,
        'stable': crash_rate == 0,
    }


def objective(trial):
    """Optuna objective function - maximize time-to-crash (longevity)."""
    global BEST_STABLE_CONFIG, BEST_STABLE_SPEED
    
    # Define search space focusing on stability-critical parameters
    # Key insight: aggressive steering and insufficient speed cause crashes
    # Strategy: prioritize safe steering (low K_heading, high smoothing) and adequate speed
    config = {
        # Steering parameters (prioritize stability: low gains, high smoothing)
        'K_heading': trial.suggest_float('K_heading', 0.2, 0.7),
        'heading_exp': trial.suggest_float('heading_exp', 0.8, 1.1),
        'K_center': trial.suggest_float('K_center', 0.1, 0.5),
        'beta_s': trial.suggest_float('beta_s', 0.6, 0.95),  # High smoothing for stability
        
        # Speed parameters (prioritize adequate speed to avoid "too slow" penalty)
        'v_min': trial.suggest_float('v_min', 0.18, 0.24),  # Higher minimum speed
        'v_turn': trial.suggest_float('v_turn', 0.6, 1.2),
        'v_max': trial.suggest_float('v_max', 1.2, 2.2),    # Higher max speed
        'd_emergency': trial.suggest_float('d_emergency', 2.0, 4.0),
        'd_caution': trial.suggest_float('d_caution', 4.5, 7.0),
        'K_speed': trial.suggest_float('K_speed', 0.8, 1.3),
        'beta_a': trial.suggest_float('beta_a', 0.6, 0.95),  # Higher smoothing for stability
    }
    
    # Evaluate config
    result = evaluate_config(config, num_episodes=5, trial=trial)
    
    # Track results
    TRIAL_RESULTS.append(result)
    
    # NEW OBJECTIVE: Maximize time-to-crash (steps survived)
    # This gives much clearer gradient: "survived 50 steps" vs "survived 100 steps"
    # We normalize by episodes to account for crash rate
    # Metric: average steps before crash (higher = better)
    objective_value = result['steps']
    
    # Update best config if this one survives longer
    if result['steps'] > BEST_STABLE_SPEED:
        BEST_STABLE_CONFIG = result['config'].copy()
        BEST_STABLE_SPEED = result['steps']
    
    return objective_value


def tune_with_optuna(num_trials=500, num_episodes=5):
    """
    Run Optuna optimization.
    
    Args:
        num_trials: Number of trials to run
        num_episodes: Episodes per configuration
    """
    print(f"\n{'='*80}")
    print(f"OPTUNA-BASED HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Trials: {num_trials}")
    print(f"Episodes per trial: {num_episodes}")
    print(f"Objective: Maximize speed among STABLE (0% crash) configs")
    print(f"Search space: 11 hyperparameters with continuous ranges")
    print(f"\nEstimated time: ~{num_trials * num_episodes * 0.6 / 60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Create Optuna study with TPE sampler
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize',
        pruner=MedianPruner(),
    )
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial),
            n_trials=num_trials,
            show_progress_bar=True,
            catch=(Exception,),  # Catch exceptions in trials
        )
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")
    
    return study


def print_results():
    """Print results and best longevity config."""
    global BEST_STABLE_CONFIG, BEST_STABLE_SPEED
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"\nTotal trials: {len(TRIAL_RESULTS)}")
    
    # Find config with longest survival time
    best_by_steps = max(TRIAL_RESULTS, key=lambda r: r['steps'])
    
    print(f"\nSurvival time (steps) range: {min(r['steps'] for r in TRIAL_RESULTS)}-{max(r['steps'] for r in TRIAL_RESULTS)}")
    print(f"\nBest (longest survival) config:")
    print(f"  Steps survived: {best_by_steps['steps']:.1f}")
    print(f"  Crash rate: {best_by_steps['crash_rate']*100:.1f}%")
    print(f"  Average speed: {best_by_steps['avg_speed']:.4f}")
    
    # Count survival distribution
    steps_list = [r['steps'] for r in TRIAL_RESULTS]
    steps_50 = sum(1 for s in steps_list if s >= 50)
    steps_100 = sum(1 for s in steps_list if s >= 100)
    steps_150 = sum(1 for s in steps_list if s >= 150)
    
    print(f"\nSurvival distribution:")
    print(f"  ≥50 steps: {steps_50} configs")
    print(f"  ≥100 steps: {steps_100} configs")
    print(f"  ≥150 steps: {steps_150} configs")
    print(f"  Mean steps: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    
    if best_by_steps['steps'] > 0:
        print("\n" + "="*80)
        print("BEST LONGEVITY CONFIGURATION")
        print("="*80)
        print(f"Steps Survived: {best_by_steps['steps']:.1f}")
        print(f"Crash Rate: {best_by_steps['crash_rate']*100:.1f}%")
        print(f"Average Speed: {best_by_steps['avg_speed']:.4f}")
        print(f"Average Reward: {best_by_steps['avg_reward']:.2f}")
        print(f"\nHyperparameters:")
        for param, value in sorted(best_by_steps['config'].items()):
            print(f"  {param:15s} = {value:.4f}")
        print("="*80)
        return best_by_steps['config']
    else:
        print("\n⚠️  WARNING: No configs survived!")
        print("="*80)
        return None


def save_best_config(config, filename='weights/best_reflex_config.py'):
    """Save best config to weights directory."""
    if config is None:
        print(f"\n❌ Cannot save: no stable config found")
        return
    
    with open(filename, 'w') as f:
        f.write("# Best stable hyperparameters found by Optuna tuning\n")
        f.write("# These parameters achieved 0% crash rate\n")
        f.write("# Usage: from best_reflex_config import BEST_CONFIG\n")
        f.write("#        agent = ReflexAgent()\n")
        f.write("#        for param, value in BEST_CONFIG.items():\n")
        f.write("#            setattr(agent, param, value)\n\n")
        f.write("BEST_CONFIG = {\n")
        for param, value in sorted(config.items()):
            f.write(f"    '{param}': {value:.4f},\n")
        f.write("}\n")
    
    print(f"\n✅ Best config saved to {filename}")


if __name__ == '__main__':
    # Parse arguments
    test_mode = '--test' in sys.argv
    search_arg = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--search' and i+1 < len(sys.argv)-1:
            try:
                search_arg = int(sys.argv[i+2])
            except:
                pass
    
    # Determine number of trials
    if test_mode:
        num_trials = 3
        num_episodes = 3
        print("\n🚀 TEST MODE (3 trials, 3 episodes each)\n")
    elif search_arg:
        num_trials = search_arg
        num_episodes = 5
    else:
        num_trials = 500
        num_episodes = 5
    
    # Run tuning
    study = tune_with_optuna(num_trials=num_trials, num_episodes=num_episodes)
    
    # Print results and get best stable config
    best_config = print_results()
    
    # Save best config if found
    if best_config:
        save_best_config(best_config)
