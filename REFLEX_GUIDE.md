# ReflexAgent Implementation Guide

## Overview

ReflexAgent is a rule-based autonomous racing agent that uses lidar input to make steering and speed decisions. It serves as a baseline for comparing against learned policies (like PPO).

## Architecture

### Input Processing
- **Observation**: [direction, distl, dist, distr, speed] (compressed 5-element vector)
  - `direction`: Heading angle to goal
  - `distl`, `dist`, `distr`: Left, center, right lidar distances
  - `speed`: Current velocity
- **Lidar Estimation**: Reconstructs 19-element lidar from 3-distance input by interpolation

### Steering Control
```
1. Find open direction: max distance in lidar field
2. Compute heading: K_heading × direction^heading_exp
3. Add centering: K_center × (center - left) correction
4. Apply smoothing: exponential moving average (beta_s)
5. Output: steer ∈ [-1, 1]
```

**Key Parameters**:
- `K_heading`: Proportional gain for heading correction (0.4-0.8)
- `heading_exp`: Nonlinearity exponent (0.8-1.2)
- `K_center`: Centering correction gain (0.2-0.6)
- `beta_s`: Steering smoothing factor (0.5-0.9)

### Speed Control
```
1. Measure forward distance from lidar
2. Set target speed based on distance:
   - Emergency (d < d_emergency): v_min
   - Caution (d_emergency < d < d_caution): v_turn
   - Clear (d > d_caution): v_max
3. Accelerate aggressively if v < v_min (MicroRacer minimum)
4. Use PID to target speed: accel = K_speed × (v_target - v)
5. Apply smoothing: exponential moving average (beta_a)
6. Output: accel ∈ [-1, 1]
```

**Key Parameters**:
- `v_min`: Minimum required speed (0.15-0.25)
- `v_turn`: Speed in tight turns (0.5-1.0)
- `v_max`: Speed on straights (1.0-2.0)
- `d_emergency`: Distance threshold for emergency braking (2.0-4.0)
- `d_caution`: Distance threshold for caution (4.0-6.0)
- `K_speed`: Speed control gain (0.8-1.2)
- `beta_a`: Acceleration smoothing (0.5-0.9)

## Lidar Reconstruction

Since MicroRacer provides only 3-point lidar (left/center/right), ReflexAgent reconstructs the 19-beam lidar:
- Angle range: -30° to +30° (60° total)
- Beam count: 19 (3.33° spacing)
- Method: Linear interpolation between known distances
- Purpose: Estimate full lidar for detailed steering decisions

## Tuning Strategy

### Parameter Grid
Current grid search covers:
- K_heading: [0.5, 0.7]
- K_center: [0.3, 0.5]
- beta_s: [0.6, 0.8]
- v_max: [1.2, 1.5]
- K_speed: [0.9, 1.1]
- d_caution: [5.0, 6.0]

Total: 2^6 = 64 configurations

### Why Grid Search?
- Parallelizes well with multiprocessing.Pool
- 192 total episodes runs in ~10 minutes
- Simple, interpretable, no external dependencies
- Optuna evaluated but offers minimal benefit at this scale

### Running Tuning
```bash
python tune_reflex.py                  # Full search (64 × 3 = 192 eps)
python tune_reflex.py --quick          # Quick test (2 × 2 = 4 eps)
```

Results saved to `best_config.py` automatically.

## Known Limitations

1. **Rule-based constraints**: Hand-crafted rules can't adapt to novel situations like learned policies
2. **Minimum speed requirement**: MicroRacer terminates if speed < 0.15 for too long
3. **Lidar limitations**: Only 3 distance measurements; reconstruction is approximate
4. **No lookahead**: Decisions based only on current state, no planning

## Comparison with PPO

| Metric | Reflex (Tuned) | PPO (Trained) |
|--------|---|---|
| Avg Speed | 0.155 | 0.168 |
| Longevity | 9.6 steps | 361 steps |
| Stability | 0% crash-free | 70% crash-free |
| Reward | -2.93 | +0.69 |

**Key Finding**: PPO dominates on all metrics. The reflex agent's rule-based design is fundamentally limited.

## Code Structure

### Main Class: ReflexAgent
```python
class ReflexAgent:
    def __init__(self):
        # Initialize hyperparameters
        # Precompute lidar beam angles
    
    def act(state) -> [accel, steer]:
        # Main decision function
    
    def _estimate_lidar(state) -> [19 distances]:
        # Reconstruct full lidar
    
    def _compute_steering(lidar, direction) -> steer:
        # Steering logic
    
    def _compute_acceleration(lidar, speed) -> accel:
        # Speed control logic
```

### Helper Functions (test_agents.py)
- `smoke_test()`: Verify imports and models
- `evaluate_agent()`: Run agent for N episodes, collect metrics
- `compare_agents()`: Side-by-side comparison

### Tuning Functions (tune_reflex.py)
- `run_episode()`: Single episode evaluation
- `evaluate_config()`: Test one hyperparameter set
- `tune_hyperparameters()`: Parallel grid search
- `print_results()`: Format and display results
- `save_best_config()`: Write best config to file

## Extending the Code

### Add New Hyperparameter
1. Add to `ReflexAgent.__init__()` with default value
2. Add to `param_grid` in `tune_reflex.py`
3. Use in steering or speed computation

### Change Tuning Grid
Edit `param_grid` in `tune_reflex.py` before running.
Note: Each parameter value doubles the search space.

### Custom Evaluation Metric
Modify `evaluate_config()` in `tune_reflex.py` to track new metrics beyond speed/reward.

## Performance Notes

- **Tuning time**: ~5-10 minutes for full 64 × 3 grid on 16-core machine
- **Episode time**: ~0.3-1.0s per episode depending on track complexity
- **Memory**: Negligible; tuning uses <100MB total
- **Bottleneck**: MicroRacer simulation (not tuning logic)

## References

- MicroRacer environment: `tracks.py`
- Hyperparameter tuning: `tune_reflex.py`
- Testing harness: `test_agents.py`
- Best config: `best_config.py` (auto-generated)
