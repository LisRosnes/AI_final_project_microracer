---
paper: https://arxiv.org/abs/2203.10494v1
repository: https://github.com/asperti/MicroRacer/
---

# MicroRacer
A didactic car-racer micro environment for Deep Reinforcement Learning


## Aim and motivation
MicroRacer is a simple environment inspired by car racing and especially meant for the didactics of Deep Reiforcement Learning.
The complexity of the environment has been explicitly calibrated to allow to experiment with many different methods, networks and hyperparameters settings
without the need of sophisticated software and no fear of getting bored by too long training times.

## The MicroRacer environment
MicroRacer generates new random circular tracks at each episode. The Random track is defined by CubicSplines delimiting the inner and outer border; the number of turns and the width of the track are configurable. From this description, we derive a dense matrix of points of dimension 1300x1300 providing information about positions inside the track. This is the actual definition of the track used by the environment.
![micro_racer](https://user-images.githubusercontent.com/15980090/135791705-cd678320-c189-43b5-84fe-1ceb0dd01f0d.png)

## State and actions
MicroRacer **does not** intend to model realistic car dynamics. The model is explicitly meant to be as simple as possible, with the minimal amount of complexity that still makes learning interesting and challenging.

The **state** information available to actors is composed by:
  1. a lidar-like vision of the track from the car's frontal perspective. This is an array of 19 values, expressing the distance of the car from the track's borders along uniformly spaced angles in the range -30°,+30°. 
  2. the car scalar velocity.
  
The actor (the car) has no global knowledge of the track, and no information about its absolute or relative position/direction w.r.t. the track.

The actor is supposed to answer with two actions, both in the range [-1,1]: 
  1. acceleration/deceleration
  2. turning angle
Maximum values for acceleration and turning angles can be configured. 

## Available learning models
We currently equip the code with basic actors trained with DDPG, PPO, SAC and DSAC (weights included). Students are supposed to develop their own models. 

## Requirements
The project just requires basic libraries: tensorflow, numpy, matplotlib, scipy.interpolate (for Cubic Splines) and cython.

There are two requirements manifests in the repository to help with different environments:

- `requirements.txt` — a conservative, pinned set of packages intended for maximum compatibility and reproducibility. Use this when you need a stable, older or CPU-only environment, or when reproducibility is important.
- `requirements-modern.txt` — a more up-to-date requirements file that pins fewer older versions and targets modern Python/TensorFlow stacks (for example Python 3.10+ and recent TensorFlow releases). Use this when you want newer packages, GPU-enabled TensorFlow builds, or you don't need strict backward compatibility.

Notes and recommendations:
- If you use conda, it's often easiest to create an environment and install TensorFlow and ffmpeg from `conda-forge` (these packages can be sensitive to platform and binary compatibility). Example:

```bash
conda create -n microracer python=3.10
conda activate microracer
conda install -c conda-forge ffmpeg tensorflow
pip install -r requirements.txt   # or requirements-modern.txt as appropriate
```

- If you prefer a pip/venv workflow, create a virtualenv and install one of the requirement files:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # or requirements-modern.txt
```

Use `requirements-modern.txt` when you're running a recent Python interpreter and want the latest compatible packages (or GPU support). If you need maximum portability or reproducibility (CI, older OS, or colleagues with older Python), use `requirements.txt` instead.

## Plans for future work
We are extremely interested in collaborations, especially with colleagues teaching DRL at other Universities.
We plan to organize soon a Championship.

## Race examples

Racers in the legend in order from top to bottom: DDPG, TD3, SAC, PPO.

https://user-images.githubusercontent.com/93991100/157229341-4240de02-38d6-4aca-b50d-a3b9b998d171.mp4

&nbsp;


Racers in the legend in order from top to bottom: SAC, DSAC.


https://user-images.githubusercontent.com/93991100/157229440-ceb2be76-593c-4c10-9a5e-c4e771f9fdbc.mp4




## FGM (Follow-the-Gap) reflex: tuning, evaluation and visualization

The repository contains a fast reflex agent based on the Follow-the-Gap (FGM) method. The canonical implementation and Optuna tuning entrypoint live under the `agents.fgm` package.

Quick commands (using a conda environment that has the dependencies installed, e.g. `env`):

Run Optuna tuning (example):

```bash
conda run -n old-ml-env --no-capture-output python3 -m agents.fgm.tune --trials 300 --episodes 10 --eval-episodes 100
```

Evaluate a saved config (example):

```bash
conda run -n env --no-capture-output python3 -m agents.fgm.tune --eval --config weights/best_fgm_config_optuna.py --episodes 100 --out logs/eval_fgm100.json
```

Create a visualization (MP4):

```bash
conda run -n env --no-capture-output python3 -m agents.fgm.tune --viz --config weights/best_fgm_config_optuna.py --out best_fgm_run.mp4
```

Programmatic usage (small example):

```python
from agents.fgm.fgm_agent import FGMReflexAgent
from utils.eval_viz import evaluate_agent, visualize_agent

agent = FGMReflexAgent()
# Optionally load or set parameters from a saved config in `weights/`
evaluate_agent(agent, num_episodes=10, out_path='logs/eval_sample.json')
visualize_agent(agent, out='sample_run.mp4')
```






