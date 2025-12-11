# Best PPO hyperparameters from Optuna
# Metrics:
#   config: {'gamma': 0.9609223284133016, 'gae_lambda': 0.9762711648817055, 'policy_clip': 0.2719159674653176, 'target_entropy': 0.0402014676442065, 'target_kl': 0.032757102597342096, 'actor_lr': 0.00010776685849271864, 'critic_lr': 0.00038624034184440395, 'batch_size': 128, 'epochs': 6, 'sigma': 0.13160206417334422}
#   avg_reward: 5.291433036561852
#   avg_steps: 898.4333333333333
#   crash_rate: 0.16666666666666666
#   completion_rate: 0.8333333333333334

BEST_CONFIG = {
    'actor_lr': 0.00010776685849271864,
    'batch_size': 128,
    'critic_lr': 0.00038624034184440395,
    'epochs': 6,
    'gae_lambda': 0.9762711648817055,
    'gamma': 0.9609223284133016,
    'policy_clip': 0.2719159674653176,
    'sigma': 0.13160206417334422,
    'target_entropy': 0.0402014676442065,
    'target_kl': 0.032757102597342096,
}
