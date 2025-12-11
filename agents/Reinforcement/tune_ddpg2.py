#!/usr/bin/env python3
"""Optuna hyperparameter tuning for DDPG2 agent (with parameter noise)."""
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tracks

TRIAL_RESULTS = []
BASE_ACTOR_WEIGHTS = "weights/ddpg2_actor_model_car"
BASE_CRITIC_WEIGHTS = "weights/ddpg2_critic_model_car"


def get_actor(num_states=5, num_actions=2):
    """Create DDPG2 actor with layer normalization."""
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)
    return tf.keras.Model(inputs, outputs, name="ddpg2_actor")


def get_critic(num_states=5, num_actions=2):
    """Create DDPG2 critic."""
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    return tf.keras.Model([state_input, action_input], outputs, name="ddpg2_critic")


class DDPG2Buffer:
    def __init__(self, buffer_capacity=50000, batch_size=64, num_states=5, num_actions=2):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((buffer_capacity, num_states))
        self.action_buffer = np.zeros((buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((buffer_capacity, 1))
        self.done_buffer = np.zeros((buffer_capacity, 1))
        self.next_state_buffer = np.zeros((buffer_capacity, num_states))

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
        return (self.state_buffer[batch_indices], self.action_buffer[batch_indices],
                self.reward_buffer[batch_indices], self.done_buffer[batch_indices],
                self.next_state_buffer[batch_indices])


@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


class DDPG2Agent:
    """DDPG2 Agent with parameter noise."""
    def __init__(self, config, load_base_weights=True):
        self.num_states, self.num_actions = 5, 2
        self.upper_bound, self.lower_bound = 1, -1
        
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.critic_lr = config.get('critic_lr', 0.001)
        self.actor_lr = config.get('actor_lr', 0.001)
        self.buffer_size = int(config.get('buffer_size', 50000))
        self.batch_size = int(config.get('batch_size', 64))
        self.param_noise_stddev = config.get('param_noise_stddev', 0.2)
        
        self.actor_model = get_actor(self.num_states, self.num_actions)
        self.critic_model = get_critic(self.num_states, self.num_actions)
        
        if load_base_weights and os.path.exists(BASE_ACTOR_WEIGHTS):
            try:
                self.actor_model = keras.models.load_model(BASE_ACTOR_WEIGHTS, compile=False)
                self.critic_model = keras.models.load_model(BASE_CRITIC_WEIGHTS, compile=False)
            except Exception as e:
                print(f"Warning: {e}")
        
        self.target_actor = get_actor(self.num_states, self.num_actions)
        self.target_critic = get_critic(self.num_states, self.num_actions)
        self.target_actor.trainable = False
        self.target_critic.trainable = False
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_model.compile(loss='mse', optimizer=self.critic_optimizer)
        
        state_input = layers.Input(shape=(self.num_states,))
        a = self.actor_model(state_input)
        q = self.target_critic([state_input, a])
        self.aux_model = tf.keras.Model(state_input, q)
        self.aux_model.add_loss(-q)
        self.aux_model.compile(optimizer=self.actor_optimizer)
        
        self.buffer = DDPG2Buffer(self.buffer_size, self.batch_size, self.num_states, self.num_actions)
    
    def add_param_noise(self):
        """Add parameter noise to actor."""
        for layer in self.actor_model.layers:
            if hasattr(layer, 'kernel'):
                noise = np.random.normal(0, self.param_noise_stddev, layer.kernel.shape)
                layer.kernel.assign_add(noise)
    
    def policy(self, state):
        """Get action with parameter noise."""
        sampled_action = tf.squeeze(self.actor_model(state)).numpy()
        legal_action = np.clip(sampled_action, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
    
    def act(self, state):
        """Get action without noise."""
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = tf.squeeze(self.actor_model(tf_state))
        return [np.squeeze(action.numpy())]
    
    def train_step(self):
        if self.buffer.buffer_counter < self.batch_size:
            return None, None
        
        states, actions, rewards, dones, newstates = self.buffer.sample_batch()
        targetQ = rewards + (1 - dones) * self.gamma * self.target_critic([newstates, self.target_actor(newstates)])
        loss1 = self.critic_model.train_on_batch([states, actions], targetQ)
        loss2 = self.aux_model.train_on_batch(states)
        
        update_target(self.target_actor.variables, self.actor_model.variables, self.tau)
        update_target(self.target_critic.variables, self.critic_model.variables, self.tau)
        return loss1, loss2


def run_episode(agent, max_steps=2000, train=True):
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True, low_speed_termination=False)
    state = racer.reset()
    steps, total_reward = 0, 0.0
    
    if train:
        agent.add_param_noise()
    
    while not racer.done and steps < max_steps:
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action = agent.policy(tf_state)[0] if train else agent.act(state)[0]
        next_state, reward, done = racer.step(action)
        steps += 1
        total_reward += reward
        
        if train:
            fail = done and len(next_state) < agent.num_states
            agent.buffer.record((state, action, reward, fail, next_state))
            agent.train_step()
        
        state = next_state
    
    return {'steps': steps, 'reward': total_reward, 'crashed': racer.completation != 1, 'completion': racer.completation}


def train_agent(agent, num_episodes=20):
    return [run_episode(agent, train=True) for _ in range(num_episodes)]


def evaluate_agent(agent, num_episodes=10):
    results = [run_episode(agent, train=False) for _ in range(num_episodes)]
    return {
        'avg_reward': float(np.mean([r['reward'] for r in results])),
        'avg_steps': float(np.mean([r['steps'] for r in results])),
        'crash_rate': sum(1 for r in results if r['crashed']) / float(num_episodes),
        'completion_rate': sum(1 for r in results if r['completion'] == 1) / float(num_episodes),
        'results': results
    }


def save_best_config(config, metrics, filename='weights/best_ddpg2_config_optuna.py'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write('# Best DDPG2 hyperparameters from Optuna\n# Metrics:\n')
        for k, v in metrics.items():
            if k != 'results':
                f.write(f"#   {k}: {v}\n")
        f.write('\nBEST_CONFIG = {\n')
        for k, v in sorted(config.items()):
            f.write(f"    '{k}': {v},\n")
        f.write('}\n')


def tune_with_optuna(trials=50, train_episodes=10, eval_episodes=20, load_base_weights=True):
    sampler = TPESampler(seed=42, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction='maximize', pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    
    def objective(trial):
        config = {
            'gamma': trial.suggest_float('gamma', 0.95, 0.999),
            'tau': trial.suggest_float('tau', 0.001, 0.01, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-4, 1e-2, log=True),
            'actor_lr': trial.suggest_float('actor_lr', 1e-4, 1e-2, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 30000, 50000, 100000]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'param_noise_stddev': trial.suggest_float('param_noise_stddev', 0.05, 0.5),
        }
        
        agent = DDPG2Agent(config, load_base_weights=load_base_weights)
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
        print('\n' + '='*60 + '\nBEST DDPG2 CONFIGURATION:\n' + '='*60)
        print(f"Completion rate: {best['completion_rate']:.4f}")
        print(f"Average reward: {best['avg_reward']:.4f}\n")
        save_best_config(best['config'], best)
        
        with open('logs/tune_ddpg2_results.json', 'w') as f:
            json.dump(TRIAL_RESULTS, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune DDPG2 hyperparameters with Optuna')
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--train-episodes', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=20)
    parser.add_argument('--no-base-weights', action='store_true')
    args = parser.parse_args()
    
    print('='*60 + '\nDDPG2 Hyperparameter Tuning\n' + '='*60)
    tune_with_optuna(args.trials, args.train_episodes, args.eval_episodes, not args.no_base_weights)
