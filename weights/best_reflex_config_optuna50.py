# Best stable hyperparameters found by Optuna tuning
# These parameters achieved 0% crash rate
# Usage: from best_reflex_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_imbalance': 0.4142,
    'K_shrink': 0.3080,
    'K_speed': 1.5791,
    'alpha_accel': 0.7861,
    'alpha_steer': 0.8945,
    'brake_center_threshold': 3.8181,
    'max_speed_cap': 1.8443,
    'max_steer_delta': 0.1212,
    'shrink_threshold': 0.8393,
    'side_min_threshold': 2.3337,
    'steer_deadband': 0.0702,
    'v_target': 0.6075,
}
