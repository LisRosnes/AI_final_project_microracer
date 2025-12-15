# DDPG Model Comparison Tool

## Purpose
Evaluates and compares DDPG models trained with different difficulty progressions (Easy, Hard, Curriculum) on a standardized hard track.

## Usage

### Basic Comparison
```bash
cd agents/Reinforcement
python3 compare_ddpg_variants.py
```

### What It Does
1. Loads all available trained models from `weights/`
2. Evaluates each on **hard track** (obstacles + chicanes) for 50 episodes
3. Computes performance metrics and statistical significance
4. Generates comparison plots and saves results to JSON

## Output

### Console Report
```
COMPARING DDPG TRAINING MODES
Evaluation episodes: 50
Evaluation track: HARD (obstacles + chicanes)

RESULTS SUMMARY
Metric                  Easy                Hard                Curriculum          Winner
Success Rate            45.00%              62.00%              78.00%              Curriculum ⭐
Avg Reward              125.34              198.52              245.67              Curriculum ⭐
Border Crashes          18                  12                  7                   Curriculum ⭐

Statistical Test: ANOVA F=12.45, p=0.0001 ***
```

### Files Generated
- **JSON**: `logs/ddpg_comparison_{timestamp}.json` - Raw results
- **Plots**: Performance comparisons (if matplotlib available)

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Success Rate** | % of episodes that completed the lap |
| **Avg Reward** | Mean cumulative reward per episode |
| **Std Reward** | Reward standard deviation (consistency) |
| **Avg Steps** | Mean steps per episode |
| **Successes** | Count of completed laps (status=1) |
| **Border Crashes** | Track boundary collisions (status=2) |
| **Wrong Direction** | Backward driving violations (status=3) |
| **Too Slow** | Insufficient speed terminations (status=4) |

## Expected Models

The script looks for these weight files:
```
weights/ddpg_actor_easy_best
weights/ddpg_actor_hard_best
weights/ddpg_actor_curriculum_best
```

If a model is missing, it will be skipped with a warning.

## Statistical Tests

- **2 models**: Two-sample t-test on rewards
- **3+ models**: One-way ANOVA on rewards
- Significance levels: * (p<0.05), ** (p<0.01), *** (p<0.001)

## Customization

Edit the script to modify:
```python
def compare_models(num_episodes=50):  # Change episode count
    ...
```

Or evaluate on different track configurations:
```python
racer = tracks.Racer(
    obstacles=True,    # Change difficulty
    chicanes=True,
    turn_limit=True
)
```

## Troubleshooting

**No models found**:
- Train models first using `DDPG.py` with different `TRAINING_MODE` settings
- Check that weights are saved with `save_weights = True`

**scipy not found**:
- Statistical tests will be skipped
- Install: `pip install scipy`

**Metadata errors**:
- Script will run without training metadata (time, learning curves)
- Check `logs/ddpg_{mode}_improved_metadata.json` exists and is valid JSON
