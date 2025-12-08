#!/usr/bin/env python3
"""Visualize a single run using a saved ReflexAgent config.

Usage:
  python3 viz_model.py --config weights/best_reflex_config_optuna_focused.py --out best_reflex_run.mp4

The script will attempt to save an MP4 using ffmpeg; if that's not
available it will fall back to GIF via Pillow.
"""
import argparse
import importlib.util
import importlib
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
# helper: try to import the backward-compat shim or common implementations
try:
    from reflex_agent import ReflexAgent
except Exception:
    try:
        from agents.fgm import FGMReflexAgent as ReflexAgent
    except Exception:
        # leave ReflexAgent undefined if not available; we'll support
        # loading agents via --agent-path instead.
        ReflexAgent = None
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
    p.add_argument('--agent-path', '-a', default=None,
                   help="Optional agent spec: 'module:Class' or path to a saved actor (keras) model (e.g. weights/ppo_actor_model_car). If provided, this agent will be used instead of the ReflexAgent + config.")
    p.add_argument('--zoom-radius', type=float, default=0.25, help='Half-width of zoomed view (in world units)')
    p.add_argument('--env-config', '-e', default=None, help='Optional path to JSON file containing racer kwargs (e.g. {"obstacles": true, "turn_limit": true}). If omitted the default in this script is used.')
    p.add_argument('--out', '-o', default='best_reflex_run.mp4', help='Output filename (mp4 or gif)')
    p.add_argument('--max-steps', type=int, default=800, help='Max frames/steps to render')
    p.add_argument('--verbose', action='store_true', help='Print progress and writer diagnostics')
    args = p.parse_args()

    agent = None
    # If an agent path is supplied, prefer that (module:Class or saved keras model)
    if args.agent_path:
        spec = args.agent_path
        if ':' in spec:
            modname, clsname = spec.split(':', 1)
            mod = importlib.import_module(modname)
            AgentClass = getattr(mod, clsname)
            agent = AgentClass()
        else:
            # treat as a weights path -> try to load a keras model and wrap it
            if os.path.exists(spec) or spec.startswith('weights'):
                try:
                    # import keras lazily
                    from tensorflow import keras

                    model = keras.models.load_model(spec)

                    class KerasActorWrapper:
                        def __init__(self, model):
                            self.model = model

                        def act(self, state):
                            import numpy as _np

                            def _as_array(x):
                                # handle dicts, lists/tuples, tensors
                                try:
                                    # dict -> take first value
                                    if isinstance(x, dict):
                                        x = next(iter(x.values()))
                                    # try to convert to numpy array
                                    a = _np.array(x, dtype=float)
                                    return a
                                except Exception:
                                    # fallback: if it's a list/tuple of arrays, try to concat
                                    if isinstance(x, (list, tuple)):
                                        for elem in x:
                                            try:
                                                a = _np.array(elem, dtype=float)
                                                return a
                                            except Exception:
                                                continue
                                    raise

                            s = _np.array(state, dtype=float)
                            if s.ndim == 1:
                                s = s.reshape(1, -1)
                            try:
                                out = self.model.predict(s, verbose=0)
                            except Exception:
                                out = self.model(s)

                            # Normalize output into a flat numeric 1D array
                            try:
                                arr = _as_array(out)
                            except Exception:
                                # Last-ditch: try to coerce via list
                                try:
                                    arr = _np.array(list(out), dtype=float)
                                except Exception:
                                    raise RuntimeError('Unable to interpret model output as numeric array')

                            arr = arr.reshape(-1)
                            if arr.size < 2:
                                # pad with zeros if model returned a single value
                                arr = _np.pad(arr, (0, 2 - arr.size), 'constant')
                            return arr[:2]

                    agent = KerasActorWrapper(model)
                except Exception as e:
                    raise RuntimeError(f'Failed to load keras model from {spec}: {e}')
            else:
                raise RuntimeError(f'Unknown agent-path spec or file not found: {spec}')
    else:
        # fallback to config + ReflexAgent (existing behavior)
        best = load_best_config(args.config)
        if ReflexAgent is None:
            raise RuntimeError('No ReflexAgent implementation available; supply --agent-path instead')
        agent = ReflexAgent()
        for k, v in best.items():
            try:
                setattr(agent, k, v)
            except Exception:
                setattr(agent, k, v)

    # Determine whether we should pass the full 19-beam LiDAR to the agent.
    # FGMReflexAgent prefers the raw lidar (length 19). Detect common signals:
    # - agent.N == 19
    # - class name contains 'fgm'
    use_full_lidar = False
    try:
        if hasattr(agent, 'N') and int(getattr(agent, 'N')) == 19:
            use_full_lidar = True
        elif 'fgm' in agent.__class__.__name__.lower():
            use_full_lidar = True
    except Exception:
        use_full_lidar = False

    # Default racer/environment kwargs used when --env-config is not supplied.
    # You can change the defaults here if you want different visualization behavior.
    # Default location: this variable is defined in this file as DEFAULT_RACER_KWARGS.
    DEFAULT_RACER_KWARGS = dict(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)

    # Load env config from JSON file if provided, else use defaults
    if args.env_config:
        if os.path.exists(args.env_config):
            try:
                with open(args.env_config, 'r') as f:
                    racer_kwargs = json.load(f)
                print('Loaded racer kwargs from', args.env_config)
            except Exception as e:
                raise RuntimeError(f'Failed to load env-config JSON from {args.env_config}: {e}')
        else:
            raise RuntimeError(f'env-config file not found: {args.env_config}')
    else:
        racer_kwargs = DEFAULT_RACER_KWARGS
        print('Using default racer kwargs defined in viz_model.py: DEFAULT_RACER_KWARGS')
    # If a seed is provided in the env JSON it will be used to seed numpy and
    # python's random before creating the racer so track generation is
    # deterministic. The JSON key name is `seed` (integer).
    seed = None
    try:
        seed = racer_kwargs.pop('seed')
    except Exception:
        seed = None
    if seed is not None:
        try:
            import random as _rnd
            _rnd.seed(int(seed))
        except Exception:
            pass
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
        print(f'Using seed={seed} for deterministic track generation')

    # Create racer and initial state
    racer = tracks.Racer(**racer_kwargs)
    state = racer.reset()
    cs, csin, csout = racer.cs, racer.csin, racer.csout
    obs_pos = getattr(racer, 'obs_pos', [])

    # Prepare plotting
    xd = [racer.carx]
    yd = [racer.cary]

    fig = plt.figure(dpi=180)
    # Make a 3-row layout: large top row for maps, then two short stacked rows
    # for actions (above) and speed (below), both spanning the full width.
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.8, 0.8])
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
    # draw full map on the zoom axis but we'll change the view limits each frame
    axl.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
    axl.plot(csin(xs)[:,0], csin(xs)[:,1], color='black', linewidth=0.6)
    axl.plot(csout(xs)[:,0], csout(xs)[:,1], color='black', linewidth=0.6)
    for i in range(len(obs_pos)):
        axl.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.0, color='crimson')
    axl.set_aspect('equal')
    axl.axis(False)
    # label text (kept in the zoom axis)
    label = axl.text(0.01, 0.99, '', transform=axl.transAxes, va='top')
    # zoom handles: a path line and a car marker that we'll update every frame
    zoom_path_line, = axl.plot([], [], lw=1.2, color='red')
    zoom_car_marker, = axl.plot([], [], marker='o', markersize=5, color='yellow', markeredgecolor='black')
    # zoom radius
    ZOOM_RADIUS = args.zoom_radius

    # action / control trace subplot (time vs accel / steer)
    # stacked above the speed subplot, both spanning the full width.
    ax_actions = fig.add_subplot(gs[1, :])
    ax_speed = fig.add_subplot(gs[2, :])

    # actions axis
    ax_actions.set_xlabel('time (steps)')
    ax_actions.set_ylabel('action')
    ax_actions.set_xlim(0, args.max_steps)
    ax_actions.set_ylim(-1.1, 1.1)
    ax_actions.grid(True, alpha=0.25)
    accel_line, = ax_actions.plot([], [], label='accel', color='tab:blue')
    steer_line, = ax_actions.plot([], [], label='steer', color='tab:orange')
    ax_actions.legend(loc='upper right')

    # speed axis (separate subplot)
    ax_speed.set_xlabel('time (steps)')
    ax_speed.set_ylabel('speed')
    ax_speed.set_xlim(0, args.max_steps)
    ax_speed.grid(True, alpha=0.25)
    speed_line, = ax_speed.plot([], [], label='speed', color='tab:green')
    try:
        ax_speed.legend(loc='upper right')
    except Exception:
        pass

    times = []
    accel_vals = []
    steer_vals = []
    speed_vals = []

    def init():
        line.set_data([], [])
        label.set_text('')
        try:
            zoom_path_line.set_data([], [])
            zoom_car_marker.set_data([], [])
        except Exception:
            pass
        try:
            accel_line.set_data([], [])
            steer_line.set_data([], [])
            speed_line.set_data([], [])
        except Exception:
            pass
        return line, label

    def animate_frame(frame_idx):
        nonlocal state
        if racer.done or frame_idx >= args.max_steps:
            return line, label
        if state is None or (hasattr(state, '__len__') and len(state) < 5):
            safe_state = np.array([0., 0., 0., 0., 0.])
        else:
            safe_state = state
        # If this agent is the FGM reflex type, prefer passing the full 19-beam
        # lidar array directly (it yields better behavior). Otherwise pass the
        # compact state returned by Racer.observe().
        if use_full_lidar:
            try:
                lidar_signal = tracks.lidar_grid(racer.carx, racer.cary, racer.carvx, racer.carvy, racer.map)
                state_for_agent = lidar_signal
            except Exception:
                state_for_agent = safe_state
        else:
            state_for_agent = safe_state

        action = agent.act(state_for_agent)
        action = np.array(action, dtype=float)
        a_acc = float(action[0]) if action.shape[0] > 0 else 0.0
        a_steer = float(action[1]) if action.shape[0] > 1 else 0.0
        times.append(frame_idx)
        accel_vals.append(a_acc)
        steer_vals.append(a_steer)
        # record speed for plotting (state[4] when available)
        spd = 0.0
        if state is not None and hasattr(state, '__len__') and len(state) > 4:
            try:
                spd = float(state[4])
            except Exception:
                spd = 0.0
        speed_vals.append(spd)
        state, reward, done = racer.step(action)
        xd.append(racer.carx)
        yd.append(racer.cary)
        line.set_data(xd, yd)
        # update zoomed-in view: path and car marker
        try:
            zoom_path_line.set_data(xd, yd)
            zoom_car_marker.set_data([racer.carx], [racer.cary])
            cx = float(racer.carx)
            cy = float(racer.cary)
            axl.set_xlim(cx - ZOOM_RADIUS, cx + ZOOM_RADIUS)
            axl.set_ylim(cy - ZOOM_RADIUS, cy + ZOOM_RADIUS)
        except Exception:
            pass
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
        if args.verbose:
            print('Using FFMpegWriter')
    except Exception:
        writer = PillowWriter(fps=20)
        use_mp4 = False
        if args.verbose:
            print('FFMpegWriter unavailable, using PillowWriter (GIF)')

    steps = 0
    try:
        if use_mp4:
            if args.verbose:
                print('Beginning save to', args.out)
            with writer.saving(fig, args.out, dpi=120):
                fig.canvas.draw()
                while not racer.done and steps < args.max_steps:
                    animate_frame(steps)
                    # update action traces
                    if len(times) > 0:
                        accel_line.set_data(times, accel_vals)
                        steer_line.set_data(times, steer_vals)
                        speed_line.set_data(times, speed_vals)
                        xlim_max = max(args.max_steps, times[-1] + 5)
                        ax_actions.set_xlim(0, xlim_max)
                        ax_speed.set_xlim(0, xlim_max)
                        # adjust speed y-limits to fit data (with small padding)
                        try:
                            smin = min(speed_vals)
                            smax = max(speed_vals)
                            if smin == smax:
                                ax_speed.set_ylim(smin - 0.5, smax + 0.5)
                            else:
                                pad = max(0.1, 0.05 * (smax - smin))
                                ax_speed.set_ylim(max(0.0, smin - pad), smax + pad)
                        except Exception:
                            pass
                    fig.canvas.draw()
                    writer.grab_frame()
                    steps += 1
                    if args.verbose and steps % 10 == 0:
                        print('Wrote frame', steps, flush=True)
            print('Saved animation to', args.out)
        else:
            out_gif = args.out if out_lower.endswith('.gif') else args.out.rsplit('.', 1)[0] + '.gif'
            if args.verbose:
                print('Beginning save to', out_gif)
            with writer.saving(fig, out_gif, dpi=120):
                fig.canvas.draw()
                while not racer.done and steps < args.max_steps:
                    animate_frame(steps)
                    if len(times) > 0:
                        accel_line.set_data(times, accel_vals)
                        steer_line.set_data(times, steer_vals)
                        speed_line.set_data(times, speed_vals)
                        xlim_max = max(args.max_steps, times[-1] + 5)
                        ax_actions.set_xlim(0, xlim_max)
                        ax_speed.set_xlim(0, xlim_max)
                        try:
                            smin = min(speed_vals)
                            smax = max(speed_vals)
                            if smin == smax:
                                ax_speed.set_ylim(smin - 0.5, smax + 0.5)
                            else:
                                pad = max(0.1, 0.05 * (smax - smin))
                                ax_speed.set_ylim(max(0.0, smin - pad), smax + pad)
                        except Exception:
                            pass
                    fig.canvas.draw()
                    writer.grab_frame()
                    steps += 1
                    if args.verbose and steps % 10 == 0:
                        print('Wrote frame', steps, flush=True)
            print('Saved animation to', out_gif)
    except KeyboardInterrupt:
        print('Interrupted during save; partial file may exist.')
    except Exception as e:
        print('Failed to save animation:', e)


if __name__ == '__main__':
    main()
