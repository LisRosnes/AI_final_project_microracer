#!/usr/bin/env python3
"""Evaluate a saved ReflexAgent config over N episodes and print summary.

Usage:
  python3 eval_model.py --config weights/best_reflex_config_optuna_focused.py --episodes 100
"""
import argparse
import importlib.util
import json
import numpy as np
from agents.reflex.reflex_agent import ReflexAgent
import tracks


def load_best_config(path):
    spec = importlib.util.spec_from_file_location('best_config_tmp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'BEST_CONFIG'):
        return mod.BEST_CONFIG
    # fallback: try to find any dict named BEST_CONFIG or CONFIG
    for name in ('BEST_CONFIG', 'CONFIG'):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f'No BEST_CONFIG found in {path}')


def run_episode(agent, racer, max_steps=2000):
    state = racer.reset()
    agent.reset()
    steps = 0
    speeds = []
    while not racer.done and steps < max_steps:
        # ensure state shape
        if state is None or (hasattr(state, '__len__') and len(state) < 5):
            safe_state = np.array([0., 0., 0., 0., 0.])
        else:
            safe_state = state
        action = agent.act(safe_state)
        state, reward, done = racer.step(action)
        steps += 1
        if state is not None and hasattr(state, '__len__') and len(state) > 4:
            try:
                speeds.append(float(state[4]))
            except Exception:
                pass
    # consider crash if completation != 1
    crashed = racer.completation != 1
    mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    return steps, crashed, mean_speed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='weights/best_reflex_config.py', help='Path to config file (defines BEST_CONFIG)')
    p.add_argument('--episodes', '-n', type=int, default=100, help='Number of episodes to run')
    p.add_argument('--max-steps', type=int, default=2000, help='Max steps per episode')
    p.add_argument('--out', '-o', help='Optional JSON output file to save stats')
    args = p.parse_args()

    best = load_best_config(args.config)
    agent = ReflexAgent()
    for k, v in best.items():
        try:
            setattr(agent, k, v)
        except Exception:
            setattr(agent, k, v)

    # create racer
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)

    steps_list = []
    crashes = 0
    speeds_all = []

    for i in range(args.episodes):
        steps, crashed, mean_speed = run_episode(agent, racer, max_steps=args.max_steps)
        steps_list.append(steps)
        speeds_all.append(mean_speed)
        crashes += 1 if crashed else 0
        print(f'Episode {i+1:3d}: steps={steps:4d}  crashed={crashed}  mean_speed={mean_speed:.3f}')

    mean_steps = float(np.mean(steps_list))
    std_steps = float(np.std(steps_list))
    mean_speed_all = float(np.mean(speeds_all)) if len(speeds_all) > 0 else 0.0

    summary = {
        'episodes': args.episodes,
        'mean_steps': mean_steps,
        'std_steps': std_steps,
        'crashes': crashes,
        'mean_speed': mean_speed_all,
    }

    print('\nSummary:')
    print(json.dumps(summary, indent=2))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'summary': summary, 'steps': steps_list}, f, indent=2)
        print('Saved results to', args.out)


if __name__ == '__main__':
    main()
