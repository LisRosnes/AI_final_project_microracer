#!/usr/bin/env python3
# Small visualization runner for the best ReflexAgent config
# Produces best_reflex_run.gif in the project root

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import tracks
from reflex_agent import ReflexAgent
from weights.best_reflex_config import BEST_CONFIG

# Apply config to agent
agent = ReflexAgent()
for k, v in BEST_CONFIG.items():
    if hasattr(agent, k):
        setattr(agent, k, v)
    else:
        # If param not present as attribute, still set it (some params exist in agent)
        setattr(agent, k, v)

class CallableAgent:
    def __init__(self, agent):
        self.agent = agent
    def __call__(self, state_batch):
        # state_batch expected shape (1, ...)
        state = state_batch[0]
        action = self.agent.act(state)
        # Return in same shape as Keras models: (1, 2)
        return np.array([action])

callable_agent = CallableAgent(agent)

# Create racer and shared map
racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
state = racer.reset()
cs, csin, csout = racer.cs, racer.csin, racer.csout
obs_pos = racer.obs_pos

# Prepare plotting
xd = [racer.carx]
yd = [racer.cary]

fig, axs = plt.subplots(ncols=2, dpi=180)
ax = axs[0]
ax.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
xs = 2 * np.pi * np.linspace(0,1,200)
ax.plot(csin(xs)[:,0], csin(xs)[:,1], color='black')
ax.plot(csout(xs)[:,0], csout(xs)[:,1], color='black')
for i in range(len(obs_pos)):
    ax.plot(obs_pos[i,:2], obs_pos[i,2:], lw=1.6, color='crimson')
ax.set_aspect('equal')
line, = ax.plot([], [], lw=1.5, marker='s', markersize=3, color='red')

axl = axs[1]
axl.set_xlim([-1.3,1.3])
axl.set_ylim([-1.3,1.3])
axl.set_aspect('equal')
axl.axis(False)
label = axl.text(0.01, 0.99, '', transform=axl.transAxes, va='top')


def init():
    line.set_data([], [])
    label.set_text('')
    return line, label

max_steps = 2000

# Animation generator: step until done or max_steps

def animate_frame(frame_idx):
    global state
    if racer.done or frame_idx >= max_steps:
        return line, label
    # Ensure state is a valid observation for the agent
    if state is None or (hasattr(state, '__len__') and len(state) < 5):
        safe_state = np.array([0., 0., 0., 0., 0.])
    else:
        safe_state = state

    # actor expects batch; normalize action shape
    action_raw = callable_agent(np.expand_dims(safe_state, 0))
    action_arr = np.asarray(action_raw)
    if action_arr.ndim == 2:
        a = action_arr[0]
    elif action_arr.ndim == 1:
        a = action_arr
    else:
        a = action_arr.flatten()[:2]

    # a is array [accel, steer]
    state, reward, done = racer.step(a)
    xd.append(racer.carx)
    yd.append(racer.cary)
    line.set_data(xd, yd)
    # status text
    stext = ''
    if state is not None and hasattr(state, '__len__') and len(state) > 4:
        try:
            stext = f"speed: {state[4]:.2f}"
        except Exception:
            stext = ''
    if racer.completation == 1:
        stext += '  - Completed'
    elif racer.completation == 2:
        stext += '  - Off road'
    elif racer.completation == 3:
        stext += '  - Wrong direction'
    elif racer.completation == 4:
        stext += '  - Under speed limit'
    label.set_text(stext)
    return line, label

max_frames = 800
out_mp4 = 'best_reflex_run.mp4'
out_gif = 'best_reflex_run.gif'

# Try to use ffmpeg/FFMpegWriter (faster, produces MP4). Fall back to Pillow GIF.
try:
    writer = animation.FFMpegWriter(fps=20)
    use_mp4 = True
except Exception:
    writer = PillowWriter(fps=20)
    use_mp4 = False

steps = 0
try:
    if use_mp4:
        with writer.saving(fig, out_mp4, dpi=120):
            # draw initial canvas
            fig.canvas.draw()
            while not racer.done and steps < max_frames:
                animate_frame(steps)
                fig.canvas.draw()
                writer.grab_frame()
                steps += 1
        print('Saved animation to', out_mp4)
    else:
        # Pillow GIF fallback
        with writer.saving(fig, out_gif, dpi=120):
            fig.canvas.draw()
            while not racer.done and steps < max_frames:
                animate_frame(steps)
                fig.canvas.draw()
                writer.grab_frame()
                steps += 1
        print('Saved animation to', out_gif)
except KeyboardInterrupt:
    print('Interrupted during save; partial file may exist.')
except Exception as e:
    print('Failed to save animation:', e)

print('Done')
