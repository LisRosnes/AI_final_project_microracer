# Best hyperparameters found by Optuna tuning (500 trials, longevity objective)
# Trial 306: Survived 60 steps (longest-living config found)
# 
# Usage: from best_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_center': 0.3053,
    'K_heading': 0.2118,
    'K_speed': 0.9246,
    'beta_a': 0.6150,
    'beta_s': 0.8444,
    'd_caution': 5.9600,
    'd_emergency': 2.0899,
    'heading_exp': 0.9371,
    'v_max': 1.7734,
    'v_min': 0.1955,
    'v_turn': 1.0962,
}

# Performance (10-episode test):
# Average Speed: 0.2702 ± 0.0811
# Average Steps: 10.9 ± 7.7
# Crash Rate: 100.0%
# Average Reward: -2.89 ± 0.09
#
# Note: 100% crash rate is fundamental limitation of reactive control
# on dense MicroRacer tracks. This config maximizes time-to-crash.
