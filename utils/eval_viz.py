#!/usr/bin/env python3
"""Generic evaluation and visualization helpers for agents.

Provides evaluate_agent() and visualize_agent() helpers used by tuning
scripts. These are deliberately lightweight and depend only on `tracks` for
the Racer simulator.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter


def evaluate_agent(agent, num_episodes=100, racer_kwargs=None, out_path=None, max_steps=2000, seed=None):
    import tracks
    if racer_kwargs is None:
        racer_kwargs = dict(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)

    steps_list = []
    crashes = 0
    speeds_all = []

    for i in range(num_episodes):
        # for reproducibility allow per-episode seeding: seed + i
        if seed is not None:
            try:
                import numpy as _np
                _np.random.seed(int(seed) + i)
            except Exception:
                pass
        racer = tracks.Racer(**racer_kwargs)
        state = racer.reset()
        if hasattr(agent, 'reset'):
            try:
                agent.reset()
            except Exception:
                pass
        steps = 0
        speeds = []
        while not racer.done and steps < max_steps:
            # prefer raw lidar for agents that are explicitly FGM-like; otherwise
            # construct a compact state [direction, distl, dist, distr, speed]
            # using the current simulator state so pretrained models receive the
            # same speed feature they were trained with.
            raw_lidar = None
            try:
                raw_lidar = tracks.lidar_grid(racer.carx, racer.cary, racer.carvx, racer.carvy, racer.map)
            except Exception:
                raw_lidar = None

            # build compact state if possible using current 'state' for speed
            safe_state = state if (state is not None and hasattr(state, '__len__') and len(state) >= 5) else np.array([0., 0., 0., 0., 0.])
            compact_state = safe_state

            prefer_raw = False
            try:
                clsname = agent.__class__.__name__.lower()
                modname = getattr(agent.__class__, '__module__', '')
                if 'fgm' in clsname or 'fgm' in modname or 'follow' in clsname:
                    prefer_raw = True
            except Exception:
                prefer_raw = False

            action = None
            # Try preferred input first, but fall back on the other if it fails
            if prefer_raw and raw_lidar is not None:
                try:
                    action = agent.act(raw_lidar)
                except Exception:
                    try:
                        action = agent.act(compact_state)
                    except Exception:
                        action = np.array([0., 0.])
            else:
                # prefer compact state (good for pretrained models expecting 5-dim inputs)
                try:
                    action = agent.act(compact_state)
                except Exception:
                    if raw_lidar is not None:
                        try:
                            action = agent.act(raw_lidar)
                        except Exception:
                            action = np.array([0., 0.])
                    else:
                        action = np.array([0., 0.])
            state, reward, done = racer.step(action)
            steps += 1
            if state is not None and hasattr(state, '__len__') and len(state) > 4:
                try:
                    speeds.append(float(state[4]))
                except Exception:
                    pass
        crashed = racer.completation != 1
        steps_list.append(steps)
        crashes += 1 if crashed else 0
        speeds_all.append(float(np.mean(speeds)) if len(speeds) > 0 else 0.0)
        print(f'Episode {i+1:3d}: steps={steps:4d}  crashed={crashed}  mean_speed={speeds_all[-1]:.3f}')

    summary = {
        'episodes': num_episodes,
        'mean_steps': float(np.mean(steps_list)),
        'std_steps': float(np.std(steps_list)),
        'crashes': int(crashes),
        'mean_speed': float(np.mean(speeds_all)) if len(speeds_all) > 0 else 0.0,
    }

    print('\nSummary:')
    print(json.dumps(summary, indent=2))

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'summary': summary, 'steps': steps_list}, f, indent=2)
        print('Saved results to', out_path)

    return summary


def visualize_agent(agent, out='agent_run.mp4', racer_kwargs=None, max_steps=800, save_frames=8, seed=None):
    import tracks
    if racer_kwargs is None:
        racer_kwargs = dict(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)

    # allow deterministic visualization by seeding numpy
    if seed is not None:
        try:
            import numpy as _np
            _np.random.seed(int(seed))
        except Exception:
            pass
    racer = tracks.Racer(**racer_kwargs)
    state = racer.reset()
    cs, csin, csout = racer.cs, racer.csin, racer.csout
    obs_pos = getattr(racer, 'obs_pos', [])

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

    ax_actions = fig.add_subplot(gs[1, :])
    ax_actions.set_xlabel('time (steps)')
    ax_actions.set_ylabel('action')
    ax_actions.set_xlim(0, max_steps)
    ax_actions.set_ylim(-1.1, 1.1)
    ax_actions.grid(True, alpha=0.25)
    accel_line, = ax_actions.plot([], [], label='accel', color='tab:blue')
    steer_line, = ax_actions.plot([], [], label='steer', color='tab:orange')
    ax_actions.legend(loc='upper right')

    times = []
    accel_vals = []
    steer_vals = []

    def animate_frame():
        nonlocal state
        if racer.done:
            return False
        if state is None or (hasattr(state, '__len__') and len(state) < 5):
            safe_state = np.array([0., 0., 0., 0., 0.])
        else:
            safe_state = state
        # Prefer raw lidar for FGM-like agents, otherwise pass compact state
        raw_lidar = None
        try:
            raw_lidar = tracks.lidar_grid(racer.carx, racer.cary, racer.carvx, racer.carvy, racer.map)
        except Exception:
            raw_lidar = None

        prefer_raw = False
        try:
            clsname = agent.__class__.__name__.lower()
            modname = getattr(agent.__class__, '__module__', '')
            if 'fgm' in clsname or 'fgm' in modname or 'follow' in clsname:
                prefer_raw = True
        except Exception:
            prefer_raw = False

        if prefer_raw and raw_lidar is not None:
            try:
                action = agent.act(raw_lidar)
            except Exception:
                action = agent.act(safe_state)
        else:
            try:
                action = agent.act(safe_state)
            except Exception:
                if raw_lidar is not None:
                    action = agent.act(raw_lidar)
                else:
                    action = np.array([0., 0.])
        action = np.array(action, dtype=float)
        a_acc = float(action[0]) if action.shape[0] > 0 else 0.0
        a_steer = float(action[1]) if action.shape[0] > 1 else 0.0
        times.append(steps)
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
        return True

    out_lower = out.lower()
    use_mp4 = out_lower.endswith('.mp4')
    try:
        writer = animation.FFMpegWriter(fps=20)
    except Exception:
        writer = PillowWriter(fps=20)
        use_mp4 = False

    os.makedirs('logs/fgm_frames', exist_ok=True)

    steps = 0
    try:
        if use_mp4:
            with writer.saving(fig, out, dpi=120):
                while not racer.done and steps < max_steps:
                    alive = animate_frame()
                    if len(times) > 0:
                        accel_line.set_data(list(range(len(accel_vals))), accel_vals)
                        steer_line.set_data(list(range(len(steer_vals))), steer_vals)
                        ax_actions.set_xlim(0, max(max_steps, len(times)+5))
                    fig.canvas.draw()
                    writer.grab_frame()
                    if steps < save_frames:
                        frame_path = f'logs/fgm_frames/frame_{steps+1:03d}.png'
                        fig.savefig(frame_path)
                    steps += 1
            print('Saved animation to', out)
        else:
            out_gif = out if out_lower.endswith('.gif') else out.rsplit('.', 1)[0] + '.gif'
            with writer.saving(fig, out_gif, dpi=120):
                while not racer.done and steps < max_steps:
                    alive = animate_frame()
                    if len(times) > 0:
                        accel_line.set_data(list(range(len(accel_vals))), accel_vals)
                        steer_line.set_data(list(range(len(steer_vals))), steer_vals)
                        ax_actions.set_xlim(0, max(max_steps, len(times)+5))
                    fig.canvas.draw()
                    writer.grab_frame()
                    if steps < save_frames:
                        frame_path = f'logs/fgm_frames/frame_{steps+1:03d}.png'
                        fig.savefig(frame_path)
                    steps += 1
            print('Saved animation to', out_gif)
    except KeyboardInterrupt:
        print('Interrupted during save; partial file may exist.')
    except Exception as e:
        print('Failed to save animation:', e)
