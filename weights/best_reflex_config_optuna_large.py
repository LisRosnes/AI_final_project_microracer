# Best stable hyperparameters found by Optuna tuning
# These parameters achieved 0% crash rate
# Usage: from best_reflex_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_imbalance': 0.3625,
    'K_shrink': 0.7492,
    'K_speed': 1.2718,
    'alpha_accel': 0.8799,
    'alpha_steer': 0.9415,
    'brake_center_threshold': 3.5657,
    'max_speed_cap': 1.8927,
    'max_steer_delta': 0.0451,
    'shrink_threshold': 0.5116,
    'side_min_threshold': 1.8152,
    'steer_deadband': 0.0559,
    'v_target': 0.7150,
}
