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


def run_episode(agent, max_steps=500):
    # inline runner used by evaluate_config; prefer raw lidar in evaluate
    import tracks
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    state = racer.reset()
    steps = 0
    distance = 0.0
    while not racer.done and steps < max_steps:
        try:
            raw_lidar = tracks.lidar_grid(racer.carx, racer.cary, racer.carvx, racer.carvy, racer.map)
            action = agent.act(raw_lidar)
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


def tune_with_optuna(trials=300, episodes=10, eval_episodes=100):
    global EPISODES_PER_TRIAL
    EPISODES_PER_TRIAL = episodes
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction='maximize', pruner=MedianPruner())

    def objective(trial):
        config = {
            'bubble_radius_factor': trial.suggest_float('bubble_radius_factor', 0.2, 0.6),
            'gap_min_width': trial.suggest_int('gap_min_width', 2, 6),
            'steering_gain': trial.suggest_float('steering_gain', 0.8, 2.0),
            'max_speed_straight': trial.suggest_float('max_speed_straight', 0.6, 1.0),
            'max_speed_turn': trial.suggest_float('max_speed_turn', 0.2, 0.6),
            'curvature_threshold_factor': trial.suggest_float('curvature_threshold_factor', 0.3, 0.8),
            'accel_scale': trial.suggest_float('accel_scale', 0.05, 1.0),
        }
        result = evaluate_config(config, num_episodes=EPISODES_PER_TRIAL, trial=trial)
        TRIAL_RESULTS.append(result)
        score = result['avg_distance'] + (result['avg_steps'] * 1e-4)
        return score

    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print('Interrupted')

    if TRIAL_RESULTS:
        best = max(TRIAL_RESULTS, key=lambda r: (r.get('avg_distance', 0.0), r.get('avg_steps', 0.0)))
        print('Best avg_distance:', best['avg_distance'], 'avg_steps:', best['avg_steps'], 'crash_rate:', best['crash_rate'])
        save_best_config(best['config'], filename='weights/best_fgm_config_optuna.py')
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
    args = p.parse_args()
    tune_with_optuna(trials=args.trials, episodes=args.episodes, eval_episodes=args.eval_episodes)
