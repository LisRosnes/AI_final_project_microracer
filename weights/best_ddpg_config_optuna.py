# Best DDPG hyperparameters from Optuna
# Metrics:
#   avg_reward: 6.8147
#   avg_steps: 856.20
#   completion_rate: 1.0000
#   crash_rate: 0.0000

BEST_CONFIG = {
    'actor_lr': 0.000138,
    'batch_size': 32.000000,
    'buffer_size': 100000.000000,
    'critic_lr': 0.000249,
    'gamma': 0.951773,
    'noise_accel_mult': 1.837177,
    'noise_scale': 0.139193,
    'noise_steer_mult': 0.694643,
    'tau': 0.001023,
}
