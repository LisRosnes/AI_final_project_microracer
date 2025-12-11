#!/usr/bin/env python3
"""Optuna hyperparameter tuning for DDPG agent.

This script tunes DDPG hyperparameters while using pre-trained weights as initialization.
It uses Optuna for efficient hyperparameter search.
"""
import argparse
import json
import os
import sys
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tracks

TRIAL_RESULTS = []

# Base weights to load
BASE_ACTOR_WEIGHTS = "weights/ddpg_actor_model_car"
BASE_CRITIC_WEIGHTS = "weights/ddpg_critic_model_car"

# Alternative h5 weights if SavedModel format doesn't work
BASE_ACTOR_WEIGHTS_H5 = "weights/ddpg_actor_weigths_32_car3_split.h5"
BASE_CRITIC_WEIGHTS_H5 = "weights/ddpg_critic_weigths_32_car3_split.h5"


def get_actor(num_states=5, num_actions=2, train_acceleration=True, train_direction=True):
    """Create actor network with separate towers for acceleration and direction."""
    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(inputs)
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
    out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)

    out2 = layers.Dense(32, activation="relu", trainable=train_direction)(inputs)
    out2 = layers.Dense(32, activation="relu", trainable=train_direction)(out2)
    out2 = layers.Dense(1, activation='tanh', trainable=train_direction)(out2)

    outputs = layers.concatenate([out1, out2])
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model


def get_critic(num_states=5, num_actions=2):
    """Create critic network."""
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs, name="critic")
    return model


class DDPGBuffer:
    """Replay buffer for DDPG."""
    def __init__(self, buffer_capacity=50000, batch_size=64, num_states=5, num_actions=2):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        s, a, r, T, sn = obs_tuple
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = tf.squeeze(s)
        self.action_buffer[index] = a
        self.reward_buffer[index] = r
        self.done_buffer[index] = T
        self.next_state_buffer[index] = tf.squeeze(sn)

        self.buffer_counter += 1

    def sample_batch(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        s = self.state_buffer[batch_indices]
        a = self.action_buffer[batch_indices]
        r = self.reward_buffer[batch_indices]
        T = self.done_buffer[batch_indices]
        sn = self.next_state_buffer[batch_indices]
        return (s, a, r, T, sn)


@tf.function
def update_target(target_weights, weights, tau):
    """Slowly update target network parameters."""
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPGAgent:
    """DDPG Agent with tunable hyperparameters."""
    def __init__(self, config, load_base_weights=True):
        self.num_states = 5
        self.num_actions = 2
        self.upper_bound = 1
        self.lower_bound = -1
        
        # Hyperparameters from config
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.critic_lr = config.get('critic_lr', 0.001)
        self.actor_lr = config.get('actor_lr', 0.001)
        self.buffer_size = int(config.get('buffer_size', 50000))
        self.batch_size = int(config.get('batch_size', 64))
        self.noise_scale = config.get('noise_scale', 0.1)
        self.noise_accel_mult = config.get('noise_accel_mult', 2.0)
        self.noise_steer_mult = config.get('noise_steer_mult', 0.5)
        
        # Create models
        self.actor_model = get_actor(self.num_states, self.num_actions)
        self.critic_model = get_critic(self.num_states, self.num_actions)
        
        # Load base weights if requested
        if load_base_weights:
            try:
                if os.path.exists(BASE_ACTOR_WEIGHTS):
                    self.actor_model = keras.models.load_model(BASE_ACTOR_WEIGHTS)
                    print(f"Loaded actor weights from {BASE_ACTOR_WEIGHTS}")
                elif os.path.exists(BASE_ACTOR_WEIGHTS_H5):
                    self.actor_model.load_weights(BASE_ACTOR_WEIGHTS_H5)
                    print(f"Loaded actor weights from {BASE_ACTOR_WEIGHTS_H5}")
                    
                if os.path.exists(BASE_CRITIC_WEIGHTS):
                    self.critic_model = keras.models.load_model(BASE_CRITIC_WEIGHTS)
                    print(f"Loaded critic weights from {BASE_CRITIC_WEIGHTS}")
                elif os.path.exists(BASE_CRITIC_WEIGHTS_H5):
                    self.critic_model.load_weights(BASE_CRITIC_WEIGHTS_H5)
                    print(f"Loaded critic weights from {BASE_CRITIC_WEIGHTS_H5}")
            except Exception as e:
                print(f"Warning: Could not load base weights: {e}")
                print("Starting from scratch")
        
        # Create target models
        self.target_actor = get_actor(self.num_states, self.num_actions)
        self.target_critic = get_critic(self.num_states, self.num_actions)
        self.target_actor.trainable = False
        self.target_critic.trainable = False
        
        # Initialize target models with same weights
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        
        # Optimizers
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        
        self.critic_model.compile(loss='mse', optimizer=self.critic_optimizer)
        
        # Create compound model for actor training
        state_input = layers.Input(shape=(self.num_states,))
        a = self.actor_model(state_input)
        q = self.target_critic([state_input, a])
        self.aux_model = tf.keras.Model(state_input, q)
        self.aux_model.add_loss(-q)
        self.aux_model.compile(optimizer=self.actor_optimizer)
        
        # Replay buffer
        self.buffer = DDPGBuffer(self.buffer_size, self.batch_size, self.num_states, self.num_actions)
    
    def policy(self, state):
        """Get action with exploration noise."""
        sampled_action = tf.squeeze(self.actor_model(state))
        noise = np.random.normal(scale=self.noise_scale, size=2)
        noise[0] *= self.noise_accel_mult
        noise[1] *= self.noise_steer_mult
        
        sampled_action = sampled_action.numpy()
        sampled_action += noise
        
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
    
    def act(self, state):
        """Get action without noise (for evaluation)."""
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.squeeze(self.actor_model(tf_state))
        return [np.squeeze(action.numpy())]
    
    def train_step(self):
        """Perform one training step."""
        if self.buffer.buffer_counter < self.batch_size:
            return None, None
        
        states, actions, rewards, dones, newstates = self.buffer.sample_batch()
        targetQ = rewards + (1 - dones) * self.gamma * (
            self.target_critic([newstates, self.target_actor(newstates)])
        )
        loss1 = self.critic_model.train_on_batch([states, actions], targetQ)
        loss2 = self.aux_model.train_on_batch(states)
        
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
        
        return loss1, loss2


def run_episode(agent, max_steps=2000, train=True):
    """Run a single episode."""
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    state = racer.reset()
    steps = 0
    total_reward = 0.0
    
    while not racer.done and steps < max_steps:
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        
        if train:
            action = agent.policy(tf_state)[0]
        else:
            action = agent.act(state)[0]
        
        next_state, reward, done = racer.step(action)
        steps += 1
        total_reward += reward
        
        if train:
            fail = done and len(next_state) < agent.num_states
            agent.buffer.record((state, action, reward, fail, next_state))
            agent.train_step()
        
        state = next_state
    
    completion = racer.completation
    crashed = completion != 1
    
    return {
        'steps': steps,
        'reward': total_reward,
        'crashed': crashed,
        'completion': completion
    }


def train_agent(agent, num_episodes=20):
    """Train agent for a specified number of episodes."""
    results = []
    for ep in range(num_episodes):
        result = run_episode(agent, train=True)
        results.append(result)
    return results


def evaluate_agent(agent, num_episodes=10):
    """Evaluate agent without training."""
    results = []
    for ep in range(num_episodes):
        result = run_episode(agent, train=False)
        results.append(result)
    
    avg_reward = float(np.mean([r['reward'] for r in results]))
    avg_steps = float(np.mean([r['steps'] for r in results]))
    crash_rate = sum(1 for r in results if r['crashed']) / float(num_episodes)
    completion_rate = sum(1 for r in results if r['completion'] == 1) / float(num_episodes)
    
    return {
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'crash_rate': crash_rate,
        'completion_rate': completion_rate,
        'results': results
    }


def save_best_config(config, metrics, filename='weights/best_ddpg_config_optuna.py'):
    """Save best configuration to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('# Best DDPG hyperparameters from Optuna\n')
        f.write('# Metrics:\n')
        f.write(f"#   avg_reward: {metrics.get('avg_reward', 0):.4f}\n")
        f.write(f"#   avg_steps: {metrics.get('avg_steps', 0):.2f}\n")
        f.write(f"#   completion_rate: {metrics.get('completion_rate', 0):.4f}\n")
        f.write(f"#   crash_rate: {metrics.get('crash_rate', 0):.4f}\n")
        f.write('\n')
        f.write('BEST_CONFIG = {\n')
        for k, v in sorted(config.items()):
            if isinstance(v, str):
                f.write(f"    '{k}': '{v}',\n")
            else:
                f.write(f"    '{k}': {float(v):.6f},\n")
        f.write('}\n')
    print(f'Saved best DDPG config to {filename}')


def tune_with_optuna(trials=50, train_episodes=10, eval_episodes=20, load_base_weights=True):
    """Run Optuna hyperparameter tuning.
    optimize for completion rate and average reward."""
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(
        sampler=sampler,
        direction='maximize',
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    )
    
    def objective(trial):
        config = {
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'tau': trial.suggest_float('tau', 0.001, 0.01, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True),
            'actor_lr': trial.suggest_float('actor_lr', 1e-4, 1e-2, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 30000, 50000, 100000]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'noise_scale': trial.suggest_float('noise_scale', 0.05, 0.3),
            'noise_accel_mult': trial.suggest_float('noise_accel_mult', 1.0, 3.0),
            'noise_steer_mult': trial.suggest_float('noise_steer_mult', 0.3, 1.0),
        }
        
        # Create agent
        agent = DDPGAgent(config, load_base_weights=load_base_weights)
        
        # Train agent
        print(f"Trial {trial.number}: Training for {train_episodes} episodes...")
        train_results = train_agent(agent, num_episodes=train_episodes)
        
        # Report intermediate values for pruning
        for i, result in enumerate(train_results):
            trial.report(result['reward'], step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Evaluate agent
        print(f"Trial {trial.number}: Evaluating for {eval_episodes} episodes...")
        eval_metrics = evaluate_agent(agent, num_episodes=eval_episodes)
        
        result = {
            'config': config,
            **eval_metrics
        }
        TRIAL_RESULTS.append(result)
        
        # Score: prioritize completion rate and reward
        score = eval_metrics['completion_rate'] * 10.0 + eval_metrics['avg_reward']
        
        print(f"Trial {trial.number}: score={score:.4f}, "
              f"completion={eval_metrics['completion_rate']:.2f}, "
              f"reward={eval_metrics['avg_reward']:.4f}")
        
        return score
    
    try:
        study.optimize(objective, n_trials=trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print('Interrupted by user')
    
    # Save results
    if TRIAL_RESULTS:
        best = max(TRIAL_RESULTS, key=lambda r: (
            r.get('completion_rate', 0) * 10.0 + r.get('avg_reward', 0)
        ))
        print('\n' + '='*60)
        print('BEST CONFIGURATION:')
        print('='*60)
        print(f"Completion rate: {best['completion_rate']:.4f}")
        print(f"Average reward: {best['avg_reward']:.4f}")
        print(f"Average steps: {best['avg_steps']:.2f}")
        print(f"Crash rate: {best['crash_rate']:.4f}")
        print('\nHyperparameters:')
        for k, v in sorted(best['config'].items()):
            print(f"  {k}: {v}")
        print('='*60)
        
        save_best_config(best['config'], best, filename='weights/best_ddpg_config_optuna.py')
        
        # Save full results
        results_file = 'logs/tune_ddpg_results.json'
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(TRIAL_RESULTS, f, indent=2)
        print(f'\nSaved all trial results to {results_file}')
    else:
        print('No trial results to report')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune DDPG hyperparameters with Optuna')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--train-episodes', type=int, default=10,
                        help='Number of training episodes per trial (default: 10)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Number of evaluation episodes per trial (default: 20)')
    parser.add_argument('--no-base-weights', action='store_true',
                        help='Do not load base weights (train from scratch)')
    
    args = parser.parse_args()
    
    print('='*60)
    print('DDPG Hyperparameter Tuning with Optuna')
    print('='*60)
    print(f"Trials: {args.trials}")
    print(f"Training episodes per trial: {args.train_episodes}")
    print(f"Evaluation episodes per trial: {args.eval_episodes}")
    print(f"Load base weights: {not args.no_base_weights}")
    if not args.no_base_weights:
        print(f"Base actor weights: {BASE_ACTOR_WEIGHTS}")
        print(f"Base critic weights: {BASE_CRITIC_WEIGHTS}")
    print('='*60)
    
    tune_with_optuna(
        trials=args.trials,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        load_base_weights=not args.no_base_weights
    )
