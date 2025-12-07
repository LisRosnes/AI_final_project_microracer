#!/usr/bin/env python3
"""Visualize a single run using a saved ReflexAgent config.

Usage:
  python3 viz_model.py --config weights/best_reflex_config_optuna_focused.py --out best_reflex_run.mp4

The script will attempt to save an MP4 using ffmpeg; if that's not
available it will fall back to GIF via Pillow.
"""
import argparse
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from agents.reflex.reflex_agent import ReflexAgent
import tracks


def load_best_config(path):
    spec = importlib.util.spec_from_file_location('best_config_tmp', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'BEST_CONFIG'):
        return mod.BEST_CONFIG
    for name in ('BEST_CONFIG', 'CONFIG'):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f'No BEST_CONFIG found in {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', '-c', default='weights/best_reflex_config.py', help='Path to config file (defines BEST_CONFIG)')
    p.add_argument('--out', '-o', default='best_reflex_run.mp4', help='Output filename (mp4 or gif)')
    p.add_argument('--max-steps', type=int, default=800, help='Max frames/steps to render')
    args = p.parse_args()

    best = load_best_config(args.config)
    agent = ReflexAgent()
    for k, v in best.items():
        try:
            setattr(agent, k, v)
        except Exception:
            setattr(agent, k, v)

    # Create racer and initial state
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    state = racer.reset()
    cs, csin, csout = racer.cs, racer.csin, racer.csout
    obs_pos = getattr(racer, 'obs_pos', [])

    # Prepare plotting
    xd = [racer.carx]
    yd = [racer.cary]

    fig = plt.figure(dpi=180)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
    xs = 2 * np.pi * np.linspace(0,1,200)
    ax.plot(csin(xs)[:,0], csin(xs)[:,1], color='black')
    ax.plot(csout(xs)[:,0], csout(xs)[:,1], color='black')
    for i in range(len(obs_pos)):
        ax.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.6, color='crimson')
    ax.set_aspect('equal')
    line, = ax.plot([], [], lw=1.5, marker='s', markersize=3, color='red')

    axl = fig.add_subplot(gs[0, 1])
    axl.set_xlim([-1.3,1.3])
    axl.set_ylim([-1.3,1.3])
    axl.set_aspect('equal')
    axl.axis(False)
    label = axl.text(0.01, 0.99, '', transform=axl.transAxes, va='top')

    # action / control trace subplot (time vs accel / steer)
    ax_actions = fig.add_subplot(gs[1, :])
    ax_actions.set_xlabel('time (steps)')
    ax_actions.set_ylabel('action')
    ax_actions.set_xlim(0, args.max_steps)
    ax_actions.set_ylim(-1.1, 1.1)
    ax_actions.grid(True, alpha=0.25)
    accel_line, = ax_actions.plot([], [], label='accel', color='tab:blue')
    steer_line, = ax_actions.plot([], [], label='steer', color='tab:orange')
    ax_actions.legend(loc='upper right')

    times = []
    accel_vals = []
    steer_vals = []

    def init():
        line.set_data([], [])
        label.set_text('')
        return line, label

    def animate_frame(frame_idx):
        nonlocal state
        if racer.done or frame_idx >= args.max_steps:
            return line, label
        if state is None or (hasattr(state, '__len__') and len(state) < 5):
            safe_state = np.array([0., 0., 0., 0., 0.])
        else:
            safe_state = state
        action = agent.act(safe_state)
        action = np.array(action, dtype=float)
        a_acc = float(action[0]) if action.shape[0] > 0 else 0.0
        a_steer = float(action[1]) if action.shape[0] > 1 else 0.0
        times.append(frame_idx)
        accel_vals.append(a_acc)
        steer_vals.append(a_steer)
        state, reward, done = racer.step(action)
        xd.append(racer.carx)
        yd.append(racer.cary)
        line.set_data(xd, yd)
        stext = ''
        if state is not None and hasattr(state, '__len__') and len(state) > 4:
            try:
                stext = f"speed: {state[4]:.2f}"
            except Exception:
                stext = ''
        if getattr(racer, 'completation', None) == 1:
            stext += '  - Completed'
        elif getattr(racer, 'completation', None) == 2:
            stext += '  - Off road'
        elif getattr(racer, 'completation', None) == 3:
            stext += '  - Wrong direction'
        elif getattr(racer, 'completation', None) == 4:
            stext += '  - Under speed limit'
        label.set_text(stext)
        return line, label

    # Choose writer
    out_lower = args.out.lower()
    use_mp4 = out_lower.endswith('.mp4')
    try:
        writer = animation.FFMpegWriter(fps=20)
    except Exception:
        writer = PillowWriter(fps=20)
        use_mp4 = False

    steps = 0
    try:
        if use_mp4:
            with writer.saving(fig, args.out, dpi=120):
                fig.canvas.draw()
                while not racer.done and steps < args.max_steps:
                    animate_frame(steps)
                    # update action traces
                    if len(times) > 0:
                        accel_line.set_data(times, accel_vals)
                        steer_line.set_data(times, steer_vals)
                        ax_actions.set_xlim(0, max(args.max_steps, times[-1]+5))
                    fig.canvas.draw()
                    writer.grab_frame()
                    steps += 1
            print('Saved animation to', args.out)
        else:
            out_gif = args.out if out_lower.endswith('.gif') else args.out.rsplit('.', 1)[0] + '.gif'
            with writer.saving(fig, out_gif, dpi=120):
                fig.canvas.draw()
                while not racer.done and steps < args.max_steps:
                    animate_frame(steps)
                    if len(times) > 0:
                        accel_line.set_data(times, accel_vals)
                        steer_line.set_data(times, steer_vals)
                        ax_actions.set_xlim(0, max(args.max_steps, times[-1]+5))
                    fig.canvas.draw()
                    writer.grab_frame()
                    steps += 1
            print('Saved animation to', out_gif)
    except KeyboardInterrupt:
        print('Interrupted during save; partial file may exist.')
    except Exception as e:
        print('Failed to save animation:', e)


if __name__ == '__main__':
    main()
