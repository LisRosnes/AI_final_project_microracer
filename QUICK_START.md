# Quick Start Guide

## Project Structure

This project compares rule-based (ReflexAgent) vs learned (PPO) policies for autonomous racing in MicroRacer.

### Core Files

- **`reflex_agent.py`** - ReflexAgent implementation (steering + speed control from lidar)
- **`tune_reflex.py`** - Hyperparameter tuning using parallel grid search
- **`test_agents.py`** - Comprehensive testing and comparison script
- **`best_config.py`** - Best hyperparameters found by tuning (auto-generated)

### Quick Commands

```bash
# Smoke test (verify everything works)
python test_agents.py --smoke

# Compare Reflex vs PPO (10 episodes each)
python test_agents.py

# Quick comparison (3 episodes each)
python test_agents.py --quick

# Test just Reflex agent (5 episodes)
python test_agents.py --reflex 5

# Test just PPO agent (5 episodes)
python test_agents.py --ppo 5

# Tune ReflexAgent hyperparameters (full search: 64 configs × 3 eps)
python tune_reflex.py

# Quick tuning test (2 configs × 2 eps)
python tune_reflex.py --quick
```

## What's Included

### 1. ReflexAgent (`reflex_agent.py`)
Rule-based steering and speed control using lidar input. 
- **Steering**: Find open direction, apply nonlinear heading correction, add centering bias
- **Speed**: Distance-based thresholds (emergency/caution/straight) with PID smoothing
- **Hyperparameters**: 8 tunable parameters for precise control

### 2. Tuning Script (`tune_reflex.py`)
Grid search across hyperparameter space in parallel.
- Tests 64 configurations × 3 episodes = 192 total runs
- Uses multiprocessing.Pool for parallelization
- Ranks by average speed (primary metric)
- Saves best config to `best_config.py`

### 3. Test Suite (`test_agents.py`)
Evaluates agents and compares performance.
- Smoke test: Verify imports, models, environment
- Baseline test: 10 episodes each, detailed metrics
- Comparison: Speed, longevity, stability, reward

## Key Results

From tuning (192 episodes):
- **Best Reflex Speed**: 0.3915 avg (tuned K_heading=0.7, K_center=0.3, beta_s=0.8, v_max=1.5, K_speed=0.9, d_caution=6.0)
- **PPO Speed**: 0.2252 avg
- **PPO Longevity**: 408 steps avg (vs 9.6 for Reflex)
- **PPO Stability**: 70% crash-free (vs 0% for Reflex)

**Conclusion**: Rule-based agents hit fundamental limits. PPO's learned policy far outperforms, especially in stability and longevity.

## Dependencies

- Python 3.13+
- numpy, tensorflow 2.8+, (MicroRacer environment comes with `tracks.py`)

## Further Work

If extending:
1. ReflexAgent parameters can be tuned via `tune_reflex.py`
2. MicroRacer environment variations in `test_agents.py` (obstacles, chicanes, turn_limit)
3. Custom evaluation metrics in `evaluate_agent()` function
