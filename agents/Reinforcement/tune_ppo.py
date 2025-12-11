#!/usr/bin/env python3
"""Optuna hyperparameter tuning for PPO agent."""
import argparse
import json
import os
import sys
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow import keras
from keras import backend as K

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tracks

TRIAL_RESULTS = []

BASE_ACTOR_WEIGHTS = "weights/ppo_actor_model_car"
BASE_CRITIC_WEIGHTS = "weights/ppo_critic_model_car"


class PPOActor(tf.keras.Model):
    """PPO actor network."""
    def __init__(self, num_states=5, num_actions=2, sigma=0.2):
        super().__init__()
        self.sigma = sigma
        self.num_actions = num_actions
        self.d1 = layers.Dense(64, activation="tanh")
        self.d2 = layers.Dense(64, activation="tanh")
        self.m = layers.Dense(num_actions, activation="tanh")
        
    def call(self, s):
        out = self.d1(s)
        out = self.d2(out)
        mu = self.m(out)
        return mu, self.sigma
    
    @property  
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.m.trainable_variables


class PPOCritic(tf.keras.Model):
    """PPO critic network."""
    def __init__(self):
        super().__init__()
        self.d1 = layers.Dense(64, activation="tanh")
        self.d2 = layers.Dense(64, activation="tanh")
        self.o = layers.Dense(1)
        
    def call(self, inputs):
        out = self.d1(inputs)
        out = self.d2(out)
        q = self.o(out)
        return q
    
    @property
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.o.trainable_variables


class PPOBuffer:
    """Trajectory buffer for PPO."""
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.val = []
        self.logp = []
        self.batch_size = batch_size

    def sample_batch(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return (np.array(self.states), np.array(self.actions), np.array(self.rewards),
                np.array(self.dones), np.array(self.val), np.array(self.logp), batches)
            
    def record(self, state, action, reward, done, val, logp):
        self.states.append(tf.squeeze(state))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.val.append(tf.squeeze(val))
        self.logp.append(tf.squeeze(logp))
       
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear() 
        self.val.clear()
        self.logp.clear()


class PPOAgent:
    """PPO Agent with tunable hyperparameters."""
    def __init__(self, config, load_base_weights=True):
        self.num_states = 5
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1
        
        # Hyperparameters from config
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.policy_clip = tf.constant(config.get('policy_clip', 0.25), dtype=tf.float32)
        self.target_entropy = tf.constant(config.get('target_entropy', 0.01), dtype=tf.float32)
        self.target_kl = config.get('target_kl', 0.01)
        self.actor_lr = config.get('actor_lr', 3e-4)
        self.critic_lr = config.get('critic_lr', 3e-4)
        self.batch_size = int(config.get('batch_size', 64))
        self.epochs = int(config.get('epochs', 10))
        self.sigma = config.get('sigma', 0.2)
        
        # Create models
        self.actor_model = PPOActor(self.num_states, self.num_actions, self.sigma)
        self.critic_model = PPOCritic()
        self.buffer = PPOBuffer(self.batch_size)
        
        # Build models
        self.critic_model(layers.Input(shape=(self.num_states,)))
        self.actor_model(layers.Input(shape=(self.num_states,)))
        
        # Load base weights if requested
        if load_base_weights:
            try:
                if os.path.exists(BASE_ACTOR_WEIGHTS):
                    loaded_actor = keras.models.load_model(BASE_ACTOR_WEIGHTS, compile=False)
                    self.actor_model.set_weights(loaded_actor.get_weights())
                    print(f"Loaded actor weights from {BASE_ACTOR_WEIGHTS}")
                if os.path.exists(BASE_CRITIC_WEIGHTS):
                    loaded_critic = keras.models.load_model(BASE_CRITIC_WEIGHTS, compile=False)
                    self.critic_model.set_weights(loaded_critic.get_weights())
                    print(f"Loaded critic weights from {BASE_CRITIC_WEIGHTS}")
            except Exception as e:
                print(f"Warning: Could not load base weights: {e}")
        
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_model.compile(optimizer=self.actor_optimizer)
        self.critic_model.compile(loss="mse", optimizer=self.critic_optimizer)
    
    def get_action_and_logp(self, states, actions=None):
        """Get action and log probability."""
        mu, sigma = self.actor_model(states)
        dist = tfp.distributions.Normal(mu, sigma)
        if actions is None:
            # Use reparameterization trick 
            actions = mu + sigma * tfp.distributions.Normal(0, 1).sample(self.num_actions)   
        log_p = dist.log_prob(actions)
        
        if len(log_p.shape) > 1:
            log_p = tf.reduce_sum(log_p, 1)
        else:
            log_p = tf.reduce_sum(log_p)
        log_p = tf.expand_dims(log_p, 1)
        
        valid_action = K.clip(actions, self.lower_bound, self.upper_bound)
        return valid_action, log_p
    
    def gae(self, values, rewards, masks, lastvalue):
        """Compute Generalized Advantage Estimation."""
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                nextvalue = lastvalue
            else:
                nextvalue = values[i + 1]
            delta = rewards[i] + self.gamma * nextvalue * masks[i] - values[i]  
            gae = delta + self.gamma * self.gae_lambda * masks[i] * gae
            returns.insert(0, gae + values[i])
        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        return np.array(returns), advantages
    
    def update_networks(self, last_value=0):
        """Update actor and critic networks."""
        states, actions, rewards, dones, values, old_logp, batches = self.buffer.sample_batch()
        returns, advantages = self.gae(values, rewards, dones, last_value)
        
        # Train using mini-batches
        for batch in batches:
            s_batch = tf.convert_to_tensor(states[batch], dtype=tf.float32)
            a_batch = tf.convert_to_tensor(actions[batch], dtype=tf.float32)
            adv_batch = tf.expand_dims(tf.convert_to_tensor(advantages.numpy()[batch], dtype=tf.float32), 1)
            ret_batch = tf.expand_dims(tf.convert_to_tensor(returns[batch], dtype=tf.float32), 1)
            ologp_batch = tf.expand_dims(tf.convert_to_tensor(old_logp[batch], dtype=tf.float32), 1)
            
            for e in range(self.epochs):
                with tf.GradientTape() as tape:
                    tape.watch(self.actor_model.trainable_variables)
                    _, logp_batch = self.get_action_and_logp(tf.stack(s_batch), tf.stack(a_batch))
                    ratio = tf.exp(logp_batch - ologp_batch)
                    weighted_ratio = ratio * adv_batch
                    weighted_clipped_ratio = tf.clip_by_value(
                        ratio, 
                        clip_value_min=1 - self.policy_clip, 
                        clip_value_max=1 + self.policy_clip
                    ) * adv_batch
                    min_wr = tf.minimum(weighted_ratio, weighted_clipped_ratio) - self.target_entropy * logp_batch
                    loss = -tf.reduce_mean(min_wr)
                
                grad = tape.gradient(loss, self.actor_model.trainable_variables)
                self.actor_model.optimizer.apply_gradients(zip(grad, self.actor_model.trainable_variables))
                
                self.critic_model.train_on_batch(s_batch, ret_batch)
                
                # Early stopping with KL divergence
                _, logp = self.get_action_and_logp(s_batch, a_batch)
                kl = tf.reduce_mean(ologp_batch - logp)
                if kl > 1.5 * self.target_kl:
                    break
        
        self.buffer.clear()
    
    def act(self, state):
        """Get action for evaluation (deterministic)."""
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        mu, _ = self.actor_model(tf_state)
        action = K.clip(mu, self.lower_bound, self.upper_bound)
        return [np.squeeze(action.numpy())]


def run_episode(agent, max_steps=2000, train=True):
    """Run a single episode."""
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    state = racer.reset()
    steps = 0
    total_reward = 0.0
    
    while not racer.done and steps < max_steps:
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        
        if train:
            action, logp = agent.get_action_and_logp(tf_state)
            value = agent.critic_model(tf_state)
            action = tf.squeeze(action)
            next_state, reward, done = racer.step(action.numpy())
            agent.buffer.record(tf_state, action, reward, not done, value, logp)
        else:
            action = agent.act(state)[0]
            next_state, reward, done = racer.step(action)
        
        steps += 1
        total_reward += reward
        state = next_state
    
    if train:
        agent.update_networks(last_value=0)
    
    return {
        'steps': steps,
        'reward': total_reward,
        'crashed': racer.completation != 1,
        'completion': racer.completation
    }


def train_agent(agent, num_episodes=20):
    """Train agent for a specified number of episodes."""
    return [run_episode(agent, train=True) for _ in range(num_episodes)]


def evaluate_agent(agent, num_episodes=10):
    """Evaluate agent without training."""
    results = [run_episode(agent, train=False) for _ in range(num_episodes)]
    return {
        'avg_reward': float(np.mean([r['reward'] for r in results])),
        'avg_steps': float(np.mean([r['steps'] for r in results])),
        'crash_rate': sum(1 for r in results if r['crashed']) / float(num_episodes),
        'completion_rate': sum(1 for r in results if r['completion'] == 1) / float(num_episodes),
        'results': results
    }


def save_best_config(config, metrics, filename='weights/best_ppo_config_optuna.py'):
    """Save best configuration to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('# Best PPO hyperparameters from Optuna\n# Metrics:\n')
        for k, v in metrics.items():
            if k != 'results':
                f.write(f"#   {k}: {v}\n")
        f.write('\nBEST_CONFIG = {\n')
        for k, v in sorted(config.items()):
            f.write(f"    '{k}': {v},\n")
        f.write('}\n')


def tune_with_optuna(trials=50, train_episodes=10, eval_episodes=20, load_base_weights=True):
    """Run Optuna hyperparameter tuning."""
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    def objective(trial):
        config = {
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.9, 0.99),
            'policy_clip': trial.suggest_float('policy_clip', 0.1, 0.3),
            'target_entropy': trial.suggest_float('target_entropy', 0.001, 0.1),
            'target_kl': trial.suggest_float('target_kl', 0.005, 0.05),
            'actor_lr': trial.suggest_float('actor_lr', 1e-4, 1e-2, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'epochs': trial.suggest_int('epochs', 5, 15),
            'sigma': trial.suggest_float('sigma', 0.1, 0.5),
        }
        
        agent = PPOAgent(config, load_base_weights=load_base_weights)
        train_results = train_agent(agent, num_episodes=train_episodes)
        
        for i, result in enumerate(train_results):
            trial.report(result['reward'], step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        eval_metrics = evaluate_agent(agent, num_episodes=eval_episodes)
        result = {'config': config, **eval_metrics}
        TRIAL_RESULTS.append(result)
        
        score = eval_metrics['completion_rate'] * 10.0 + eval_metrics['avg_reward']
        print(f"Trial {trial.number}: score={score:.4f}, completion={eval_metrics['completion_rate']:.2f}")
        return score
    
    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print('Interrupted by user')
    
    if TRIAL_RESULTS:
        best = max(TRIAL_RESULTS, key=lambda r: r.get('completion_rate', 0) * 10.0 + r.get('avg_reward', 0))
        print('\n' + '='*60 + '\nBEST PPO CONFIGURATION:\n' + '='*60)
        print(f"Completion rate: {best['completion_rate']:.4f}")
        print(f"Average reward: {best['avg_reward']:.4f}\n")
        save_best_config(best['config'], best)
        
        with open('logs/tune_ppo_results.json', 'w') as f:
            json.dump(TRIAL_RESULTS, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune PPO hyperparameters with Optuna')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--train-episodes', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--no-base-weights', action='store_true')
    args = parser.parse_args()
    
    print('='*60 + '\nPPO Hyperparameter Tuning\n' + '='*60)
    tune_with_optuna(args.trials, args.train_episodes, args.eval_episodes, not args.no_base_weights)
