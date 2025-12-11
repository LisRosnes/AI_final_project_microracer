# Best DDPG2 hyperparameters from Optuna
# Metrics:
#   config: {'gamma': 0.9988112985455313, 'tau': 0.00505356469827428, 'critic_lr': 0.0007226393614486144, 'actor_lr': 0.002937380642021175, 'buffer_size': 100000, 'batch_size': 128, 'param_noise_stddev': 0.10833499495437282}
#   avg_reward: 2.5200363199670064
#   avg_steps: 361.3
#   crash_rate: 0.6
#   completion_rate: 0.4

BEST_CONFIG = {
    'actor_lr': 0.002937380642021175,
    'batch_size': 128,
    'buffer_size': 100000,
    'critic_lr': 0.0007226393614486144,
    'gamma': 0.9988112985455313,
    'param_noise_stddev': 0.10833499495437282,
    'tau': 0.00505356469827428,
}
