"""Visualize the Mimic Agent using the project's native Matplotlib plotter."""
import argparse
import ast
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for saving videos
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import os
import sys

# --- IMPORTS ---
# 1. Ensure we can find your custom modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 2. Import the environment and your agent
try:
    import tracks 
    from mimic_agent import MimicAgent
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure 'tracks.py' and 'mimic_agent.py' are in this folder.")
    sys.exit(1)

def main():
    p = argparse.ArgumentParser(description="Visualize Mimic Agent (Matplotlib)")
    p.add_argument('--params', '-p', default='best_mimic_params.txt', help='Path to best params file')
    p.add_argument('--out', '-o', default='mimic_run.mp4', help='Output filename (mp4 or gif)')
    p.add_argument('--zoom-radius', type=float, default=0.25, help='Half-width of zoomed view')
    p.add_argument('--max-steps', type=int, default=1000, help='Max frames to render')
    p.add_argument('--verbose', action='store_true', help='Print progress')
    args = p.parse_args()

    # ---------------------------------------------------------
    # 1. SETUP MIMIC AGENT
    # ---------------------------------------------------------
    # Load parameters
    try:
        with open(args.params, "r") as f:
            params = ast.literal_eval(f.read())
            if args.verbose: print(f"Loaded params: {params}")
    except Exception:
        print("Warning: Params file not found. Using defaults.")
        params = {'speed_threshold': 10.0, 'accel_threshold': 1.0}

    # Initialize Agent
    agent = MimicAgent(
        speed_threshold=params.get('speed_threshold', 10.0),
        accel_threshold=params.get('accel_threshold', 1.0)
    )
    # ---------------------------------------------------------

    # 2. SETUP ENVIRONMENT (Microracer "tracks")
    # We use default racer kwargs + ensure deterministic seed if needed
    racer_kwargs = dict(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    racer = tracks.Racer(**racer_kwargs)
    state = racer.reset()
    
    # 3. PREPARE PLOTTING (Copied from viz_model.py)
    csin, csout = racer.csin, racer.csout
    obs_pos = getattr(racer, 'obs_pos', [])
    xd = [racer.carx]
    yd = [racer.cary]

    fig = plt.figure(dpi=180)
    # Layout: Map (Top Left), Zoomed (Top Right), Actions (Mid), Speed (Bot)
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 0.8, 0.8])
    
    # -- Full Map --
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
    xs = 2 * np.pi * np.linspace(0,1,200)
    ax.plot(csin(xs)[:,0], csin(xs)[:,1], color='black')
    ax.plot(csout(xs)[:,0], csout(xs)[:,1], color='black')
    for i in range(len(obs_pos)):
        ax.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.6, color='crimson')
    ax.set_aspect('equal')
    line, = ax.plot([], [], lw=1.5, marker='s', markersize=3, color='red')

    # -- Zoomed View --
    axl = fig.add_subplot(gs[0, 1])
    axl.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
    axl.plot(csin(xs)[:,0], csin(xs)[:,1], color='black', linewidth=0.6)
    axl.plot(csout(xs)[:,0], csout(xs)[:,1], color='black', linewidth=0.6)
    for i in range(len(obs_pos)):
        axl.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.0, color='crimson')
    axl.set_aspect('equal')
    axl.axis(False)
    zoom_path_line, = axl.plot([], [], lw=1.2, color='red')
    zoom_car_marker, = axl.plot([], [], marker='o', markersize=5, color='yellow', markeredgecolor='black')
    ZOOM_RADIUS = args.zoom_radius

    # -- Graphs --
    ax_actions = fig.add_subplot(gs[1, :])
    ax_speed = fig.add_subplot(gs[2, :])
    
    # Setup axes
    ax_actions.set_xlim(0, args.max_steps)
    ax_actions.set_ylim(-1.1, 1.1)
    accel_line, = ax_actions.plot([], [], label='accel', color='tab:blue')
    steer_line, = ax_actions.plot([], [], label='steer', color='tab:orange')
    ax_actions.legend(loc='upper right')
    
    ax_speed.set_xlim(0, args.max_steps)
    speed_line, = ax_speed.plot([], [], label='speed', color='tab:green')

    times, accel_vals, steer_vals, speed_vals = [], [], [], []

    def animate_frame(frame_idx):
        nonlocal state
        if racer.done or frame_idx >= args.max_steps:
            return
        
        # Prepare state for agent
        if state is None or (hasattr(state, '__len__') and len(state) < 5):
            safe_state = np.array([0., 0., 0., 0., 0.])
        else:
            safe_state = state

        # --- GET ACTION FROM MIMIC AGENT ---
        # Note: MimicAgent expects an observation. 
        # Tracks environment returns a vector (not pixels).
        # Assuming your MimicAgent can handle this vector input.
        action = agent.act(safe_state)
        # -----------------------------------

        # Ensure action is float format [accel, steer]
        if isinstance(action, (int, np.integer)):
             # Handle case where agent returns discrete int
             # (You might need to map this if your agent is discrete)
             a_acc, a_steer = 0.0, 0.0 
        else:
             a_acc = float(action[0])
             a_steer = float(action[1])

        times.append(frame_idx)
        accel_vals.append(a_acc)
        steer_vals.append(a_steer)
        
        # Get speed
        spd = 0.0
        if state is not None and len(state) > 4:
            spd = float(state[4])
        speed_vals.append(spd)

        # Step Environment
        state, reward, done = racer.step(action)
        
        # Update Plots
        xd.append(racer.carx)
        yd.append(racer.cary)
        
        # Update Main Map
        line.set_data(xd, yd)
        
        # Update Zoomed View (The Magic Part)
        zoom_path_line.set_data(xd, yd)
        zoom_car_marker.set_data([racer.carx], [racer.cary])
        cx, cy = float(racer.carx), float(racer.cary)
        axl.set_xlim(cx - ZOOM_RADIUS, cx + ZOOM_RADIUS)
        axl.set_ylim(cy - ZOOM_RADIUS, cy + ZOOM_RADIUS)

    # 4. SAVE VIDEO
    # Determine Writer
    out_lower = args.out.lower()
    if out_lower.endswith('.mp4'):
        try:
            writer = animation.FFMpegWriter(fps=20)
        except:
            writer = PillowWriter(fps=20)
            args.out = args.out.replace('.mp4', '.gif')
    else:
        writer = PillowWriter(fps=20)

    print(f"Generating animation to {args.out}...")
    
    with writer.saving(fig, args.out, dpi=120):
        steps = 0
        while not racer.done and steps < args.max_steps:
            animate_frame(steps)
            
            # Update Graph Data
            if len(times) > 0:
                accel_line.set_data(times, accel_vals)
                steer_line.set_data(times, steer_vals)
                speed_line.set_data(times, speed_vals)
                
                # Auto-scale Speed Graph
                smax = max(speed_vals) if speed_vals else 1.0
                ax_speed.set_ylim(0, smax + 1.0)

            writer.grab_frame()
            steps += 1
            if args.verbose and steps % 20 == 0:
                print(f"Frame {steps}/{args.max_steps}")

    print("Done!")

if __name__ == "__main__":
    main()