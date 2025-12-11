#!/usr/bin/env python3
"""Compare all RL models (DDPG, DDPG2, DSAC, PPO, SAC, TD3) to find the best performer.

Performance is defined as the first model that can complete the lap without crashing.
Uses Optuna to systematically compare and optimize each model.

Usage:
    python agents/Reinforcement/compare_all_models.py --episodes 25 --trials 10
"""
import argparse
import json
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_probability as tfp
from datetime import datetime

# Optuna is optional (only needed for hyperparameter optimization)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Note: Optuna not installed. Hyperparameter optimization features disabled.")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import tracks


# Model weight paths
MODEL_WEIGHTS = {
    'DDPG': {
        'actor': 'weights/ddpg_actor_model_car',
        'critic': 'weights/ddpg_critic_model_car'
    },
    'DDPG2': {
        'actor': 'weights/ddpg2_actor_model_car',
        'critic': 'weights/ddpg2_critic_model_car'
    },
    'DSAC': {
        'actor': 'weights/dsac_actor_model_car',
        'critic': 'weights/dsac_critic_model_car'
    },
    'PPO': {
        'actor': 'weights/ppo_actor_model_car',
        'critic': 'weights/ppo_critic_model_car'
    },
    'SAC': {
        'actor': 'weights/sac_actor_model_car',
        'critic': 'weights/sac_critic_model_car',
        'critic2': 'weights/sac_critic2_model_car'
    },
    'TD3': {
        'actor': 'weights/td3_actor_model_car',
        'critic': 'weights/td3_critic_model_car',
        'critic2': 'weights/td3_critic2_model_car'
    }
}


# ====================
# DDPG Model Functions
# ====================
def get_ddpg_actor(num_states=5, num_actions=2):
    """Create DDPG actor network with split towers."""
    inputs = layers.Input(shape=(num_states,))
    out1 = layers.Dense(32, activation="relu")(inputs)
    out1 = layers.Dense(32, activation="relu")(out1)
    out1 = layers.Dense(1, activation='tanh')(out1)

    out2 = layers.Dense(32, activation="relu")(inputs)
    out2 = layers.Dense(32, activation="relu")(out2)
    out2 = layers.Dense(1, activation='tanh')(out2)

    outputs = layers.concatenate([out1, out2])
    model = tf.keras.Model(inputs, outputs, name="ddpg_actor")
    return model


def get_ddpg_critic(num_states=5, num_actions=2):
    """Create DDPG critic network."""
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs, name="ddpg_critic")
    return model


# ====================
# DDPG2 Model Functions
# ====================
def get_ddpg2_actor(num_states=5, num_actions=2):
    """Create DDPG2 actor network (with layer normalization)."""
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.LayerNormalization()(out)
    out = layers.Dense(64, activation="relu")(out)
    out = layers.LayerNormalization()(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)

    model = tf.keras.Model(inputs, outputs, name="ddpg2_actor")
    return model


# ====================
# TD3 Model Functions
# ====================
def get_td3_actor(num_states=5, num_actions=2):
    """Create TD3 actor network."""
    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(64, activation="relu")(inputs)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(num_actions, activation="tanh")(out)

    model = tf.keras.Model(inputs, outputs, name="td3_actor")
    return model


def get_td3_critic(num_states=5, num_actions=2):
    """Create TD3 critic network."""
    state_input = layers.Input(shape=(num_states,))
    action_input = layers.Input(shape=(num_actions,))

    concat = layers.Concatenate()([state_input, action_input])
    out = layers.Dense(64, activation="relu")(concat)
    out = layers.Dense(64, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs, name="td3_critic")
    return model


# ====================
# PPO Model Classes
# ====================
class PPOActor(tf.keras.Model):
    """PPO actor network - mirrors PPO.py Get_actor class."""
    def __init__(self, num_states=5, num_actions=2):
        super().__init__()
        self.d1 = layers.Dense(64, activation="tanh")
        self.d2 = layers.Dense(64, activation="tanh")
        self.m = layers.Dense(num_actions, activation="tanh")
        
    def call(self, s):
        out = self.d1(s)
        out = self.d2(out)
        mu = self.m(out)
        sigma = 0.13160206417334422  # optimized sigma from PPO.py
        return mu, sigma
    
    @property  
    def trainable_variables(self):
        return self.d1.trainable_variables + \
                self.d2.trainable_variables + \
                self.m.trainable_variables


class PPOCritic(tf.keras.Model):
    """PPO critic network - mirrors PPO.py Get_critic class."""
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


# ====================
# SAC/DSAC Model Classes
# ====================
class SACActor(tf.keras.Model):
    """SAC/DSAC actor network."""
    def __init__(self, num_states=5, num_actions=2):
        super().__init__()
        self.num_actions = num_actions
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.m = layers.Dense(num_actions)
        self.s = layers.Dense(num_actions)
        
    def call(self, inputs, deterministic=False):
        out = self.d1(inputs)
        out = self.d2(out)
        mu = self.m(out)
        log_sigma = self.s(out)
        sigma = tf.exp(log_sigma)
        
        if deterministic:
            # During evaluation, use mean action
            return tf.tanh(mu)
        
        # Sample action during training
        dist = tfp.distributions.Normal(mu, sigma)
        action = mu + sigma * tfp.distributions.Normal(0, 1).sample(self.num_actions)
        valid_action = tf.tanh(action)
        
        return valid_action


class SACCritic(tf.keras.Model):
    """SAC/DSAC critic network."""
    def __init__(self, num_states=5, num_actions=2):
        super().__init__()
        self.d1 = layers.Dense(64, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.o = layers.Dense(1)
        
    def call(self, state, action):
        concat = layers.Concatenate()([state, action])
        out = self.d1(concat)
        out = self.d2(out)
        q = self.o(out)
        return q


# ====================
# Agent Wrappers
# ====================
class ModelAgent:
    """Unified wrapper for all model types."""
    
    def __init__(self, model_type='DDPG'):
        self.model_type = model_type
        self.num_states = 5
        self.num_actions = 2
        self.actor_model = None
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate model based on type."""
        weights = MODEL_WEIGHTS.get(self.model_type, {})
        actor_path = weights.get('actor')
        
        if not actor_path or not os.path.exists(actor_path):
            print(f"Warning: Actor weights not found for {self.model_type} at {actor_path}")
            self._create_new_model()
            return
        
        try:
            # Try to load saved model
            if self.model_type == 'DDPG':
                self.actor_model = keras.models.load_model(actor_path)
            elif self.model_type == 'DDPG2':
                self.actor_model = keras.models.load_model(actor_path)
            elif self.model_type == 'TD3':
                self.actor_model = keras.models.load_model(actor_path)
            elif self.model_type == 'PPO':
                self.actor_model = PPOActor()
                self.actor_model.load_weights(actor_path)
            elif self.model_type in ['SAC', 'DSAC']:
                self.actor_model = SACActor()
                self.actor_model.load_weights(actor_path)
            
            print(f"Loaded {self.model_type} actor from {actor_path}")
        except Exception as e:
            print(f"Error loading {self.model_type} weights: {e}")
            self._create_new_model()
    
    def _create_new_model(self):
        """Create a new untrained model if weights don't exist."""
        print(f"Creating new {self.model_type} model from scratch")
        
        if self.model_type == 'DDPG':
            self.actor_model = get_ddpg_actor()
        elif self.model_type == 'DDPG2':
            self.actor_model = get_ddpg2_actor()
        elif self.model_type == 'TD3':
            self.actor_model = get_td3_actor()
        elif self.model_type == 'PPO':
            self.actor_model = PPOActor()
        elif self.model_type in ['SAC', 'DSAC']:
            self.actor_model = SACActor()
    
    def act(self, state):
        """Get action from the model."""
        if self.actor_model is None:
            return [0.0, 0.0]
        
        # Ensure state is the right shape
        if not isinstance(state, (np.ndarray, list)):
            state = np.array([0., 0., 0., 0., 0.])
        elif len(state) < self.num_states:
            state = np.array([0., 0., 0., 0., 0.])
        else:
            state = np.array(state[:self.num_states])
        
        tf_state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
        
        try:
            if self.model_type == 'PPO':
                mu, sigma = self.actor_model(tf_state)
                action = mu  # Use mean for evaluation
            elif self.model_type in ['SAC', 'DSAC']:
                action = self.actor_model(tf_state, deterministic=True)
            else:
                action = self.actor_model(tf_state)
            
            action = tf.squeeze(action).numpy()
            return [np.squeeze(action)]
        except Exception as e:
            print(f"Error getting action from {self.model_type}: {e}")
            return [0.0, 0.0]


# ====================
# Evaluation Functions
# ====================
def run_episode(agent, max_steps=2000, track_config=None):
    """Run a single episode and return results."""
    if track_config is None:
        track_config = {
            'obstacles': True,
            'turn_limit': True,
            'chicanes': True,
            'low_speed_termination': False
        }
    
    racer = tracks.Racer(**track_config)
    state = racer.reset()
    steps = 0
    total_reward = 0.0
    speeds = []
    
    while not racer.done and steps < max_steps:
        # Get action
        action = agent.act(state)[0]
        
        # Step environment
        next_state, reward, done = racer.step(action)
        steps += 1
        total_reward += reward
        
        # Track speed if available
        if next_state is not None and len(next_state) >= 5:
            speeds.append(float(next_state[4]))
        
        state = next_state
    
    # Check if lap was completed
    completion = racer.completation
    crashed = completion != 1
    
    return {
        'steps': steps,
        'reward': total_reward,
        'crashed': crashed,
        'completion': completion,
        'avg_speed': float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    }


def evaluate_model(model_type, num_episodes=25, track_config=None):
    """Evaluate a specific model type."""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_type}")
    print(f"{'='*60}")
    
    agent = ModelAgent(model_type=model_type)
    results = []
    
    for ep in range(num_episodes):
        result = run_episode(agent, track_config=track_config)
        results.append(result)
        
        if (ep + 1) % 5 == 0:
            completed = sum(1 for r in results if not r['crashed'])
            print(f"  Episode {ep + 1}/{num_episodes}: {completed} completed, "
                  f"{len(results) - completed} crashed")
    
    # Compute metrics
    crashes = sum(1 for r in results if r['crashed'])
    completions = sum(1 for r in results if r['completion'] == 1)
    crash_rate = crashes / float(num_episodes)
    completion_rate = completions / float(num_episodes)
    avg_reward = float(np.mean([r['reward'] for r in results]))
    avg_steps = float(np.mean([r['steps'] for r in results]))
    avg_speed = float(np.mean([r['avg_speed'] for r in results]))
    
    # Find first completion (if any)
    first_completion_episode = None
    for i, r in enumerate(results):
        if not r['crashed']:
            first_completion_episode = i + 1
            break
    
    metrics = {
        'model_type': model_type,
        'num_episodes': num_episodes,
        'crashes': crashes,
        'completions': completions,
        'crash_rate': crash_rate,
        'completion_rate': completion_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_speed': avg_speed,
        'first_completion_episode': first_completion_episode,
        'results': results
    }
    
    print(f"\nResults for {model_type}:")
    print(f"  Completion rate: {completion_rate:.2%}")
    print(f"  Crash rate: {crash_rate:.2%}")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Avg speed: {avg_speed:.3f}")
    if first_completion_episode:
        print(f"  First completion: Episode {first_completion_episode}")
    else:
        print(f"  First completion: Never completed")
    
    return metrics


def compare_all_models(num_episodes=25, track_config=None, output_file=None):
    """Compare all models and find the best performer."""
    model_types = ['DDPG', 'DDPG2', 'DSAC', 'PPO', 'SAC', 'TD3']
    
    print(f"\n{'='*60}")
    print(f"COMPARING ALL RL MODELS")
    print(f"{'='*60}")
    print(f"Episodes per model: {num_episodes}")
    print(f"Track config: {track_config}")
    print()
    
    all_results = {}
    
    for model_type in model_types:
        try:
            metrics = evaluate_model(model_type, num_episodes, track_config)
            all_results[model_type] = metrics
        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            all_results[model_type] = {
                'model_type': model_type,
                'error': str(e),
                'completion_rate': 0.0
            }
    
    # Find the best model
    print(f"\n{'='*60}")
    print(f"FINAL COMPARISON")
    print(f"{'='*60}\n")
    
    # Sort by completion rate (primary) and first completion (secondary)
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    
    if not valid_results:
        print("No models successfully evaluated!")
        return all_results
    
    # Find first model to complete
    first_to_complete = None
    earliest_episode = float('inf')
    
    for model_type, metrics in valid_results.items():
        first_ep = metrics.get('first_completion_episode')
        if first_ep is not None and first_ep < earliest_episode:
            earliest_episode = first_ep
            first_to_complete = model_type
    
    # Sort by completion rate
    sorted_models = sorted(valid_results.items(), 
                          key=lambda x: (x[1]['completion_rate'], -x[1].get('crash_rate', 1.0)),
                          reverse=True)
    
    print("Rankings by completion rate:")
    print(f"{'Rank':<6} {'Model':<8} {'Completion':<12} {'Crash Rate':<12} {'Avg Reward':<12} {'First Complete':<15}")
    print("-" * 75)
    
    for rank, (model_type, metrics) in enumerate(sorted_models, 1):
        first_ep = metrics.get('first_completion_episode', 'Never')
        print(f"{rank:<6} {model_type:<8} {metrics['completion_rate']:>10.1%}  "
              f"{metrics['crash_rate']:>10.1%}  {metrics['avg_reward']:>10.2f}  {str(first_ep):>13}")
    
    print()
    
    if first_to_complete:
        print(f"🏆 WINNER: {first_to_complete}")
        print(f"   First model to complete a lap (episode {earliest_episode})")
        print(f"   Completion rate: {valid_results[first_to_complete]['completion_rate']:.1%}")
    else:
        print("No model successfully completed a lap")
        best_model = sorted_models[0][0]
        print(f"🏆 BEST PERFORMER: {best_model}")
        print(f"   Highest completion rate: {sorted_models[0][1]['completion_rate']:.1%}")
    
    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"logs/model_comparison_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'track_config': track_config,
        'first_to_complete': first_to_complete,
        'earliest_completion_episode': int(earliest_episode) if earliest_episode != float('inf') else None,
        'rankings': [
            {
                'rank': rank,
                'model': model_type,
                'completion_rate': metrics['completion_rate'],
                'crash_rate': metrics['crash_rate'],
                'avg_reward': metrics['avg_reward']
            }
            for rank, (model_type, metrics) in enumerate(sorted_models, 1)
        ],
        'detailed_results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Compare all RL models')
    parser.add_argument('--episodes', type=int, default=25,
                       help='Number of episodes to evaluate each model')
    parser.add_argument('--obstacles', action='store_true', default=True,
                       help='Enable obstacles on track')
    parser.add_argument('--turn-limit', action='store_true', default=True,
                       help='Enable turn limit')
    parser.add_argument('--chicanes', action='store_true', default=True,
                       help='Enable chicanes')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (default: logs/model_comparison_<timestamp>.json)')
    
    args = parser.parse_args()
    
    track_config = {
        'obstacles': args.obstacles,
        'turn_limit': args.turn_limit,
        'chicanes': args.chicanes,
        'low_speed_termination': False
    }
    
    results = compare_all_models(
        num_episodes=args.episodes,
        track_config=track_config,
        output_file=args.output
    )
    
    return results


if __name__ == '__main__':
    main()
