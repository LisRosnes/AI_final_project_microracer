# DDPG Agent Documentation

## Overview
Deep Deterministic Policy Gradient (DDPG) reinforcement learning agent for MicroRacer. Trains a neural network to control acceleration and steering using continuous actions.

## Training Modes

The DDPG implementation supports three training strategies:

- **Easy**: Simple track (no obstacles, no chicanes)
- **Hard**: Full track (obstacles + chicanes)
- **Curriculum**: Progressive difficulty (Easy → Medium → Hard)

## Quick Start

### 1. Train a Model

Edit `DDPG.py` to set training mode:
```python
TRAINING_MODE = 'curriculum'  # Options: 'easy', 'hard', 'curriculum'
```

Run training:
```bash
cd agents/Reinforcement
python3 DDPG.py
```

**Output**: Trained models saved to `weights/ddpg_actor_{mode}_best`

### 2. Compare Models

Evaluate all trained variants on the hard track:
```bash
python3 compare_ddpg_variants.py
```

**Output**: Performance comparison and statistical analysis

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gamma` | 0.99 | Discount factor for future rewards |
| `tau` | 0.005 | Soft update rate for target networks |
| `actor_lr` | 0.0003 | Actor network learning rate |
| `critic_lr` | 0.0003 | Critic network learning rate |
| `batch_size` | 64 | Training batch size |
| `buffer_dim` | 100000 | Replay buffer capacity |
| `total_iterations` | 50000 | Training iterations (5000 for quick test) |

## Architecture

**Actor Network** (State → Action):
- Input: 5-state vector (LIDAR + position)
- Shared layers: 128 → 128
- Acceleration tower: 64 → 32 → 1 (tanh)
- Steering tower: 64 → 32 → 1 (tanh)
- Output: [acceleration, steering] ∈ [-1, 1]²

**Critic Network** (State, Action → Q-value):
- State path: 128 → 128
- Action path: 128
- Combined: 256 → 128 → 1
- Output: Estimated Q-value

## Curriculum Training

Automatically progresses through difficulty levels:

1. **Level 1 (Easy)**: No obstacles, no chicanes
   - Advance at 80% success rate, min 5k steps
2. **Level 2 (Medium)**: Obstacles only
   - Advance at 70% success rate, min 8k steps
3. **Level 3 (Hard)**: Full difficulty
   - Trains to convergence (60% target)

Configure in `CURRICULUM_CONFIG` dictionary.

## File Structure

```
agents/Reinforcement/
├── DDPG.py                      # Main training script
├── DDPG2.py                     # Variant implementation
├── DDPG_bootstrapped.py         # Bootstrapped training
├── compare_ddpg_variants.py     # Model comparison tool
└── weights/
    ├── ddpg_actor_easy_best     # Easy mode weights
    ├── ddpg_actor_hard_best     # Hard mode weights
    └── ddpg_actor_curriculum_best  # Curriculum weights
```

## Training Tips

1. **Quick Testing**: Set `QUICK_TEST = True` for 5k iterations
2. **Early Stopping**: Enable with `use_early_stopping = True`
3. **Weight Loading**: Set `load_weights = True` to continue training
4. **Monitoring**: Check `logs/ddpg_{mode}_improved_metadata.json` for metrics

## Evaluation Metrics

The comparison script reports:
- **Success Rate**: % of completed laps
- **Average Reward**: Mean episode reward
- **Border Crashes**: Collisions with track edges
- **Wrong Direction**: Driving backwards
- **Too Slow**: Insufficient speed violations

Statistical significance tested using t-tests (2 models) or ANOVA (3+ models).
