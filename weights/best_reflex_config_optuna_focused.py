# Best stable hyperparameters found by Optuna tuning
# These parameters achieved 0% crash rate
# Usage: from best_reflex_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_imbalance': 0.5058,
    'K_shrink': 0.9279,
    'K_speed': 1.2558,
    'alpha_accel': 0.7878,
    'alpha_steer': 0.9685,
    'brake_center_threshold': 3.3302,
    'max_speed_cap': 1.6074,
    'max_steer_delta': 0.0566,
    'shrink_threshold': 1.0186,
    'side_min_threshold': 1.9191,
    'steer_deadband': 0.0491,
    'v_target': 0.7364,
}
