# Best stable hyperparameters found by Optuna tuning
# These parameters achieved 0% crash rate
# Usage: from best_reflex_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_imbalance': 0.5618,
    'K_shrink': 0.8427,
    'K_speed': 1.4657,
    'alpha_accel': 0.7904,
    'alpha_steer': 0.9495,
    'brake_center_threshold': 4.1122,
    'max_speed_cap': 2.2284,
    'max_steer_delta': 0.0325,
    'shrink_threshold': 1.0131,
    'side_min_threshold': 2.5139,
    'steer_deadband': 0.0414,
    'v_target': 0.4575,
}
