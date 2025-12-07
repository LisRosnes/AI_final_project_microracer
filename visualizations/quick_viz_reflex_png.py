#!/usr/bin/env python3
# Quick static visualization for best ReflexAgent config
# Saves a single PNG (best_reflex_run.png) — much faster than an animated GIF

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tracks
from reflex_agent import ReflexAgent
from weights.best_reflex_config import BEST_CONFIG

# apply best config
agent = ReflexAgent()
for k, v in BEST_CONFIG.items():
    setattr(agent, k, v)

# run a single episode and record trajectory
racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
state = racer.reset()
xs = [racer.carx]
ys = [racer.cary]
max_steps = 2000
steps = 0
while not racer.done and steps < max_steps:
    # ensure safe state
    if state is None or (hasattr(state, '__len__') and len(state) < 5):
        safe_state = np.array([0., 0., 0., 0., 0.])
    else:
        safe_state = state
    action = agent.act(safe_state)
    state, reward, done = racer.step(action)
    xs.append(racer.carx)
    ys.append(racer.cary)
    steps += 1

# static plot (small figure for speed)
fig, ax = plt.subplots(figsize=(6,6), dpi=100)
ax.imshow(np.rot90(racer.map), extent=[-1.3,1.3,-1.3,1.3], cmap='gray')
# plot borders
xs_border = 2 * np.pi * np.linspace(0,1,200)
ax.plot(racer.csin(xs_border)[:,0], racer.csin(xs_border)[:,1], color='black', lw=1)
ax.plot(racer.csout(xs_border)[:,0], racer.csout(xs_border)[:,1], color='black', lw=1)
# obstacles
for i in range(len(racer.obs_pos)):
    ax.plot(racer.obs_pos[i,:2], racer.obs_pos[i,2:], lw=1.2, color='crimson')
# path
ax.plot(xs, ys, color='red', lw=1.5, marker='.', markersize=2)
ax.set_title(f"Reflex run — steps={steps}  result={racer.completation}")
ax.set_aspect('equal')
ax.set_xlim([-1.3,1.3])
ax.set_ylim([-1.3,1.3])
out_path = 'best_reflex_run.png'
fig.tight_layout()
fig.savefig(out_path)
print('Saved static visualization to', out_path)
