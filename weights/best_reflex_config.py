# Best stable hyperparameters found by Optuna tuning
# These parameters achieved 0% crash rate
# Usage: from best_reflex_config import BEST_CONFIG
#        agent = ReflexAgent()
#        for param, value in BEST_CONFIG.items():
#            setattr(agent, param, value)

BEST_CONFIG = {
    'K_center': 0.1907,
    'K_heading': 0.4299,
    'K_speed': 1.2569,
    'beta_a': 0.7499,
    'beta_s': 0.6708,
    'd_caution': 6.8353,
    'd_emergency': 3.3104,
    'heading_exp': 1.0268,
    'v_max': 2.1159,
    'v_min': 0.2069,
    'v_turn': 0.7990,
}
