#!/usr/bin/env python3
"""Optuna tuning runner for the FGMReflexAgent (simplified).

This file provides a compact entrypoint `tune_with_optuna()` and a CLI so
you can run tuning from the command line or import the function from Python.
It uses the generic `utils.eval_viz` helpers for evaluation and visualization.
"""
import argparse
import json
import os
import subprocess
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from .fgm_agent import FGMReflexAgent
from utils.eval_viz import evaluate_agent, visualize_agent


TRIAL_RESULTS = []
RACER_KWARGS = {}  # Global racer config, set from env-config


def run_episode(agent, max_steps=500):
    # inline runner used by evaluate_config; prefer raw lidar in evaluate
    import tracks
    # Use global RACER_KWARGS if set, otherwise default
    racer_kwargs = dict(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    track_width = None
    if RACER_KWARGS:
        for k, v in RACER_KWARGS.items():
            if k == 'track_width':
                track_width = v
            elif k != 'seed':
                racer_kwargs[k] = v
    racer = tracks.Racer(**racer_kwargs)
    if track_width is not None:
        racer.track_width = track_width
    state = racer.reset()
    steps = 0
    distance = 0.0
    while not racer.done and steps < max_steps:
        try:
            raw_lidar = tracks.lidar_grid(racer.carx, racer.cary, racer.carvx, racer.carvy, racer.map)
            # Compute current speed from velocity components
            current_speed = np.sqrt(racer.carvx**2 + racer.carvy**2)
            action = agent.act(raw_lidar, current_speed=current_speed)
        except Exception:
            action = agent.act(state)
        state, reward, done = racer.step(action)
        steps += 1
        try:
            distance += float(reward)
        except Exception:
            pass
    crashed = racer.completation != 1
    return {'steps': steps, 'distance': distance, 'crashed': crashed}


def evaluate_config(config, num_episodes=20, trial=None):
    agent = FGMReflexAgent(
        bubble_radius_factor=config.get('bubble_radius_factor', 0.4),
        gap_min_width=int(config.get('gap_min_width', 3)),
        steering_gain=config.get('steering_gain', 1.4),
        max_speed_straight=config.get('max_speed_straight', 0.95),
        max_speed_turn=config.get('max_speed_turn', 0.40),
        curvature_threshold_factor=config.get('curvature_threshold_factor', 0.6),
        accel_scale=config.get('accel_scale', 1.0),
    )

    results = []
    for i in range(num_episodes):
        r = run_episode(agent)
        results.append(r)
        if trial is not None:
            steps_so_far = int(np.median([x['steps'] for x in results]))
            trial.report(steps_so_far, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()

    avg_distance = float(np.mean([r['distance'] for r in results]))
    avg_steps = float(np.mean([r['steps'] for r in results]))
    crash_rate = sum(1 for r in results if r['crashed']) / float(num_episodes)
    return {'config': config, 'avg_distance': avg_distance, 'avg_steps': avg_steps, 'crash_rate': crash_rate}


def save_best_config(config, filename='weights/best_fgm_config_optuna.py'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('# Best FGM hyperparameters from Optuna\n')
        f.write('BEST_CONFIG = {\n')
        for k, v in sorted(config.items()):
            if isinstance(v, str):
                f.write(f"    '{k}': '{v}',\n")
            else:
                f.write(f"    '{k}': {float(v):.6f},\n")
        f.write('}\n')
    print('Saved best FGM config to', filename)


def tune_with_optuna(trials=300, episodes=20, eval_episodes=100, env_config=None, 
                      output_config='weights/best_fgm_config_optuna.py', optimize_crash_rate=False):
    global EPISODES_PER_TRIAL, RACER_KWARGS
    EPISODES_PER_TRIAL = episodes
    
    # Load env config if provided
    if env_config and os.path.exists(env_config):
        with open(env_config, 'r') as f:
            RACER_KWARGS = json.load(f)
        print(f'Loaded env config from {env_config}: {RACER_KWARGS}')
    
    sampler = TPESampler(seed=42, multivariate=True)
    # Always maximize: score = total_steps - 1000 * crashes
    study = optuna.create_study(sampler=sampler, direction='maximize', pruner=MedianPruner())
    
    best_crash_rate = 1.0  # Track best crash rate for early stopping

    def objective(trial):
        nonlocal best_crash_rate
        config = {
            # Refined ranges based on previous tuning results
            'bubble_radius_factor': trial.suggest_float('bubble_radius_factor', 0.45, 0.5),
            'gap_min_width': trial.suggest_int('gap_min_width', 1, 3),
            'steering_gain': trial.suggest_float('steering_gain', 1.5, 4.0),  # Allow sharper steering
            'max_speed_straight': trial.suggest_float('max_speed_straight', 0.3, 0.5),
            'max_speed_turn': trial.suggest_float('max_speed_turn', 0.05, 0.15),
            'curvature_threshold_factor': trial.suggest_float('curvature_threshold_factor', 0.5, 0.8),
            'accel_scale': trial.suggest_float('accel_scale', 0.1, 0.4),
            'max_speed': trial.suggest_float('max_speed', 0.2, 1.0),  # Speed limiter
        }
        result = evaluate_config(config, num_episodes=EPISODES_PER_TRIAL, trial=trial)
        TRIAL_RESULTS.append(result)
        
        crash_rate = result['crash_rate']
        if crash_rate < best_crash_rate:
            best_crash_rate = crash_rate
            print(f'  New best crash_rate: {crash_rate:.2%}')
        
        # Score = total_steps - 1000 * num_crashes
        # This heavily penalizes crashes while still rewarding longer runs
        num_crashes = int(crash_rate * EPISODES_PER_TRIAL)
        total_steps = result['avg_steps'] * EPISODES_PER_TRIAL
        score = total_steps - 1000 * num_crashes
        return score

    # Custom callback for early stopping at 0% crash rate
    class ZeroCrashCallback:
        def __call__(self, study, trial):
            if best_crash_rate == 0.0:
                print('Reached 0% crash rate! Stopping early.')
                study.stop()

    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=True, callbacks=[ZeroCrashCallback()])
    except KeyboardInterrupt:
        print('Interrupted')

    if TRIAL_RESULTS:
        # Sort by crash_rate (ascending), then by avg_steps (descending)
        best = min(TRIAL_RESULTS, key=lambda r: (r.get('crash_rate', 1.0), -r.get('avg_steps', 0.0)))
        print('Best avg_distance:', best['avg_distance'], 'avg_steps:', best['avg_steps'], 'crash_rate:', best['crash_rate'])
        save_best_config(best['config'], filename=output_config)
        # Evaluate and visualize using utils
        agent = FGMReflexAgent()
        for k, v in best['config'].items():
            try:
                setattr(agent, k, v)
            except Exception:
                pass
        evaluate_agent(agent, num_episodes=eval_episodes, out_path='logs/eval_fgm100.json')
        visualize_agent(agent, out='best_fgm_run.mp4', max_steps=1200)
    else:
        print('No trial results to report')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=300)
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--env-config', type=str, default=None, help='Path to env config JSON (e.g. fgm_env.json)')
    p.add_argument('--output-config', type=str, default='weights/best_fgm_config_optuna.py', help='Output config file')
    p.add_argument('--optimize-crash-rate', action='store_true', help='Optimize for lowest crash rate instead of distance')
    args = p.parse_args()
    tune_with_optuna(trials=args.trials, episodes=args.episodes, eval_episodes=args.eval_episodes,
                     env_config=args.env_config, output_config=args.output_config, 
                     optimize_crash_rate=args.optimize_crash_rate)
