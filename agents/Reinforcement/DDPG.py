from datetime import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Add MicroRacer root to path to find tracks.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tracks 

########################################
###### TRAINING MODE CONFIGURATION #####
TRAINING_MODE = 'easy'  # Options: 'easy', 'hard', 'curriculum'

# Curriculum training configuration
CURRICULUM_CONFIG = {
    'level_1': {
        'name': 'Easy',
        'obstacles': False,
        'chicanes': False,
        'track_width': 0.1,
        'min_success_rate': 0.80,  # 80% completion rate to advance
        'min_steps': 5000,          # Minimum exploration time before evaluation
        'eval_episodes': 10         # Episodes to test mastery
    },
    'level_2': {
        'name': 'Medium', 
        'obstacles': True,
        'chicanes': False,
        'min_success_rate': 0.70,   # 70% completion rate to advance
        'min_steps': 8000,
        'eval_episodes': 10
    },
    'level_3': {
        'name': 'Hard',
        'obstacles': True,
        'chicanes': True,
        'min_success_rate': 0.60,   # Final level - trains to convergence/max iterations
        'min_steps': 10000,
        'eval_episodes': 10
    }
}

# Initialize racer based on training mode
if TRAINING_MODE == 'curriculum':
    initial_config = CURRICULUM_CONFIG['level_1']
    racer = tracks.Racer(
        obstacles=initial_config['obstacles'],
        chicanes=initial_config['chicanes'],
        track_width=initial_config['track_width'],
        turn_limit=True,
        low_speed_termination=True
    )
    print(f"🎓 CURRICULUM MODE: Starting with {initial_config['name']} track")
elif TRAINING_MODE == 'easy':
    racer = tracks.Racer(
        obstacles=False,
        chicanes=False,
        turn_limit=True,
        low_speed_termination=True
    )
    print("🟢 EASY MODE: Training on easy track (no obstacles, no chicanes)")
elif TRAINING_MODE == 'hard':
    racer = tracks.Racer(
        obstacles=True,
        chicanes=True,
        track_width=0.1,
        turn_limit=True,
        low_speed_termination=True
    )
    print("🔴 HARD MODE: Training on hard track (obstacles + chicanes)")
else:
    raise ValueError(f"Invalid TRAINING_MODE: {TRAINING_MODE}. Must be 'easy', 'hard', or 'curriculum'")

########################################
###### IMPROVED HYPERPARAMETERS ########

# Learning rates - FIXED: Made equal for better stability
gamma = 0.99  # Slightly higher discount factor
tau = 0.005  # Faster target network updates
critic_lr = 0.0003  # Equal learning rates
actor_lr = 0.0003

num_states = 5
num_actions = 2
print("State Space dim: {}, Action Space dim: {}".format(num_states, num_actions))

upper_bound = 1
lower_bound = -1
print("Min and Max Value of Action: {}".format(lower_bound, upper_bound))

# Buffer configuration
buffer_dim = 100000
batch_size = 64  # Increased from 32
warmup_steps = 2000  # NEW: Warmup before training starts

# Training configuration
QUICK_TEST = False
total_iterations = 5000 if QUICK_TEST else 50000

# Early stopping configuration
use_early_stopping = False
eval_frequency = 2000
eval_episodes = 10
patience = 10
min_improvement = 0.1

curriculum_patience_multiplier = 1.5
curriculum_grace_steps = 4000
is_training = True

load_weights = False
save_weights = True

# Weight file naming
mode_suffix = TRAINING_MODE

if QUICK_TEST:
    weights_file_actor = f"weights/ddpg_actor_{mode_suffix}_test"
    weights_file_critic = f"weights/ddpg_critic_{mode_suffix}_test"
    best_weights_file_actor = f"weights/ddpg_actor_{mode_suffix}_test_best"
    best_weights_file_critic = f"weights/ddpg_critic_{mode_suffix}_test_best"
    print(f"⚡ QUICK TEST MODE - Saving to *_{mode_suffix}_test weights")
else:
    weights_file_actor = f"weights/ddpg_actor_{mode_suffix}"
    weights_file_critic = f"weights/ddpg_critic_{mode_suffix}"
    best_weights_file_actor = f"weights/ddpg_actor_{mode_suffix}_best"
    best_weights_file_critic = f"weights/ddpg_critic_{mode_suffix}_best"
    print(f"🏋️ FULL TRAINING MODE - Saving to *_{mode_suffix}_best weights")

########################################
###### IMPROVED ACTOR ARCHITECTURE #####
def get_actor(train_acceleration=True, train_direction=True):
    """Improved actor with shared feature extraction."""
    inputs = layers.Input(shape=(num_states,))
    
    # Shared feature extraction - NEW
    shared = layers.Dense(128, activation="relu")(inputs)
    shared = layers.Dense(128, activation="relu")(shared)
    
    # Acceleration tower
    out1 = layers.Dense(64, activation="relu", trainable=train_acceleration)(shared)
    out1 = layers.Dense(32, activation="relu", trainable=train_acceleration)(out1)
    out1 = layers.Dense(1, activation='tanh', trainable=train_acceleration)(out1)
    
    # Steering tower
    out2 = layers.Dense(64, activation="relu", trainable=train_direction)(shared)
    out2 = layers.Dense(32, activation="relu", trainable=train_direction)(out2)
    out2 = layers.Dense(1, activation='tanh', trainable=train_direction)(out2)
    
    outputs = layers.concatenate([out1, out2])
    model = tf.keras.Model(inputs, outputs, name="actor")
    return model

########################################
###### IMPROVED CRITIC ARCHITECTURE ####
def get_critic():
    """Improved critic with more capacity."""
    # State processing
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)
    state_out = layers.Dense(128, activation="relu")(state_out)
    
    # Action processing
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(128, activation="relu")(action_input)
    
    # Combine
    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(128, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    
    model = tf.keras.Model([state_input, action_input], outputs, name="critic")
    return model

########################################
###### REPLAY BUFFER ###################
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
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
        return ((s, a, r, T, sn))

@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))

########################################
###### IMPROVED POLICY WITH BETTER NOISE
def policy(state, current_iteration=0, max_iterations=50000, verbose=False):
    """Policy with improved exploration noise."""
    sampled_action = tf.squeeze(actor_model(state))
    
    # Decay noise from 1.0 to 0.1 over training (more aggressive decay)
    decay_factor = max(0.1, 1.0 - 0.9 * (current_iteration / max_iterations))
    
    # Base noise scale
    noise = np.random.normal(scale=0.15 * decay_factor, size=2)
    
    # FIXED: Better noise balance - steering gets MORE exploration
    noise[0] *= 1.0   # Acceleration noise
    noise[1] *= 1.5   # Steering noise (more important!)
    
    sampled_action = sampled_action.numpy()
    sampled_action += noise
    
    if verbose and sampled_action[0] < 0:
        print("decelerating")
    
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)
    return [np.squeeze(legal_action)]

# Create models
actor_model = get_actor()
critic_model = get_critic()

# Target models
target_actor = get_actor()
target_critic = get_critic()
target_actor.trainable = False
target_critic.trainable = False

def compose(actor, critic):
    state_input = layers.Input(shape=(num_states))
    a = actor(state_input)
    q = critic([state_input, a])
    
    m = tf.keras.Model(state_input, q)
    m.add_loss(-tf.reduce_mean(q))  # Maximize Q-value
    return m

aux_model = compose(actor_model, target_critic)

## TRAINING ##
if __name__ == '__main__':
    if load_weights:
        critic_model = keras.models.load_model(weights_file_critic)
        actor_model = keras.models.load_model(weights_file_actor)
    
    # Initialize target networks
    target_actor_weights = actor_model.get_weights()
    target_critic_weights = critic_model.get_weights()
    target_actor.set_weights(target_actor_weights)
    target_critic.set_weights(target_critic_weights)
    
    # Optimizers
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
    
    critic_model.compile(loss='mse', optimizer=critic_optimizer)
    aux_model.compile(optimizer=actor_optimizer)
    
    buffer = Buffer(buffer_dim, batch_size)
    
    def step(action):
        """Wrapper for environment step."""
        n = 1
        t = np.random.randint(0, n)
        state, reward, done = racer.step(action)
        for i in range(t):
            if not done:
                state, t_r, done = racer.step([0, 0])
                reward += t_r
        return (state, reward, done)
    
    def evaluate_policy(actor, num_episodes=10, verbose=False, eval_difficulty='hard', current_level=1):
        """Evaluate the current policy."""
        total_rewards = []
        total_steps = []
        successes = 0
        
        # Determine difficulty for evaluation
        if eval_difficulty == 'current' and TRAINING_MODE == 'curriculum':
            if current_level == 1:
                config = CURRICULUM_CONFIG['level_1']
            elif current_level == 2:
                config = CURRICULUM_CONFIG['level_2']
            else:
                config = CURRICULUM_CONFIG['level_3']
            eval_racer = tracks.Racer(
                obstacles=config['obstacles'],
                chicanes=config['chicanes'],
                # track_width=config['track_width'],
                turn_limit=True,
                low_speed_termination=True
            )
        else:
            eval_racer = tracks.Racer(obstacles=True, chicanes=True, turn_limit=True, low_speed_termination=True)
        
        for ep in range(num_episodes):
            eval_state = eval_racer.reset()
            episode_reward = 0
            steps = 0
            
            while not eval_racer.done and steps < 1000:
                state_tensor = tf.expand_dims(tf.convert_to_tensor(eval_state), 0)
                action = tf.squeeze(actor(state_tensor, training=False)).numpy()
                action = np.clip(action, lower_bound, upper_bound)
                
                eval_state, reward, done = eval_racer.step(action)
                episode_reward += reward
                steps += 1
            
            total_rewards.append(episode_reward)
            total_steps.append(steps)
            if eval_racer.completation == 1:
                successes += 1
        
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(total_steps)
        success_rate = successes / num_episodes
        
        if verbose:
            print(f"  Eval: Avg Reward={avg_reward:.2f}, Avg Steps={avg_steps:.1f}, Success Rate={success_rate:.1%}")
        
        return {
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'success_rate': success_rate,
            'rewards': total_rewards
        }
    
    def train(total_iterations=total_iterations, training_start_time=None):
        global actor_model, critic_model, target_actor, target_critic, aux_model, buffer, racer
        
        # History tracking
        ep_reward_list = []
        avg_reward_list = []
        
        # NEW: Training metrics
        critic_losses = []
        actor_losses = []
        avg_q_values = []
        
        # Early stopping tracking
        best_eval_reward = -np.inf
        evaluations_without_improvement = 0
        eval_history = []
        stopped_early = False
        best_iteration = 0
        last_curriculum_transition_step = 0
        
        # Curriculum tracking
        current_curriculum_level = 1
        curriculum_transitions = []
        
        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting training with WARMUP of {warmup_steps} steps")
        print(f"{'='*60}\n")
        
        while i < total_iterations:
            
            prev_state = racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[num_states-1]
            done = False
            prev_theta = racer.cartheta  # NEW: Track theta for progress reward
            
            while not(done):
                i = i + 1
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                
                # Use random exploration during warmup for better initial data collection
                if buffer.buffer_counter < warmup_steps:
                    # Random actions: acceleration in [0.5, 1.0], steering constrained
                    # CRITICAL: Limit steering to prevent backward theta movement
                    # At episode start, cartheta ~0, so we need small steering to stay on track
                    action = np.array([
                        np.random.uniform(0.5, 1.0),     # Strong forward acceleration
                        np.random.uniform(-0.3, 0.3)     # Conservative steering (±30%)
                    ])
                else:
                    action = policy(tf_prev_state, current_iteration=i, max_iterations=total_iterations)[0]
                
                state, reward, done = step(action)
                
                # IMPROVED REWARD SHAPING
                # 1. Completion bonus
                if done and racer.completation == 1:
                    reward += 15.0  # Increased completion bonus
                    print("🏁 Episode completed! Bonus reward added.")
                
                # 2. FIXED: Track actual progress along track (theta-based)
                if not done:
                    current_theta = racer.cartheta
                    theta_progress = current_theta - prev_theta
                    
                    # Handle wrap-around at 0/2π
                    if theta_progress < -np.pi:
                        theta_progress += 2 * np.pi
                    elif theta_progress > np.pi:
                        theta_progress -= 2 * np.pi
                    
                    # Reward forward progress
                    if theta_progress > 0:
                        reward += theta_progress * 3.0  # Progress reward
                    
                    prev_theta = current_theta
                
                # 3. NEW: Reward clipping for stability
                reward = np.clip(reward, -5, 5)
                
                fail = done and len(state) < num_states
                buffer.record((prev_state, action, reward, fail, state))
                
                if not(done):
                    mean_speed += state[num_states-1]
                
                episodic_reward += reward
                
                # TRAINING WITH WARMUP
                if buffer.buffer_counter > warmup_steps:
                    states, actions, rewards, dones, newstates = buffer.sample_batch()
                    
                    # Train critic
                    targetQ = rewards + (1-dones) * gamma * (target_critic([newstates, target_actor(newstates)]))
                    critic_loss = critic_model.train_on_batch([states, actions], targetQ)
                    critic_losses.append(critic_loss)
                    
                    # Train actor
                    actor_loss = aux_model.train_on_batch(states)
                    actor_losses.append(actor_loss)
                    
                    # Track Q-values for monitoring
                    current_q = np.mean(critic_model([states, actions]).numpy())
                    avg_q_values.append(current_q)
                    
                    # Update target networks
                    update_target(target_actor.variables, actor_model.variables, tau)
                    update_target(target_critic.variables, critic_model.variables, tau)
                
                # Warmup progress updates
                if buffer.buffer_counter <= warmup_steps and i % 500 == 0:
                    print(f"🔄 Warmup (random exploration): {buffer.buffer_counter}/{warmup_steps} samples collected ({100*buffer.buffer_counter/warmup_steps:.1f}%)")
                
                prev_state = state
                
                if i % 100 == 0:
                    avg_reward_list.append(avg_reward)
                
                # CURRICULUM PROGRESSION - Performance-based transitions
                if (TRAINING_MODE == 'curriculum' and 
                    current_curriculum_level < 3 and 
                    i % eval_frequency == 0 and 
                    i > 0 and 
                    buffer.buffer_counter > warmup_steps):
                    
                    current_level_key = f'level_{current_curriculum_level}'
                    current_config = CURRICULUM_CONFIG[current_level_key]
                    steps_in_level = i - last_curriculum_transition_step
                    
                    # Check for transition: must meet minimum steps
                    if steps_in_level >= current_config['min_steps']:
                        # Evaluate mastery on current difficulty
                        print(f"\n{'='*60}")
                        print(f"🎯 CURRICULUM CHECK at step {i} (Level {current_curriculum_level}: {current_config['name']})")
                        print(f"   Steps in this level: {steps_in_level}")
                        print(f"   Testing mastery with {current_config['eval_episodes']} episodes...")
                        
                        mastery_results = evaluate_policy(
                            actor_model, 
                            num_episodes=current_config['eval_episodes'],
                            verbose=False,
                            eval_difficulty='current',
                            current_level=current_curriculum_level
                        )
                        
                        success_rate = mastery_results['success_rate']
                        avg_mastery_reward = mastery_results['avg_reward']
                        
                        print(f"   Success Rate: {success_rate:.1%} (need {current_config['min_success_rate']:.1%})")
                        print(f"   Avg Reward: {avg_mastery_reward:.2f}")
                        
                        # Check if ready to advance
                        if success_rate >= current_config['min_success_rate']:
                            current_curriculum_level += 1
                            next_level_key = f'level_{current_curriculum_level}'
                            config = CURRICULUM_CONFIG[next_level_key]
                            
                            racer = tracks.Racer(
                                obstacles=config['obstacles'],
                                chicanes=config['chicanes'],
                                track_width=config['track_width'],
                                turn_limit=True,
                                low_speed_termination=True
                            )
                            
                            transition_msg = f"📈 CURRICULUM TRANSITION → Level {current_curriculum_level}: {config['name']} track"
                            print(f"   ✅ MASTERY ACHIEVED! Advancing to next level.")
                            print(transition_msg)
                            print(f"   [PAUSE] Resetting early stopping (grace period: {curriculum_grace_steps} steps)")
                            print(f"{'='*60}\n")
                            
                            curriculum_transitions.append({
                                'step': i,
                                'level': current_curriculum_level,
                                'name': config['name'],
                                'avg_reward': avg_reward,
                                'mastery_success_rate': success_rate,
                                'mastery_reward': avg_mastery_reward
                            })
                            last_curriculum_transition_step = i
                            evaluations_without_improvement = 0
                            best_eval_reward = -np.inf
                            
                            # IMPORTANT: Break to start new episode with new racer
                            break
                        else:
                            print(f"   ⏳ Not ready yet. Continue training on {current_config['name']} level.")
                            print(f"{'='*60}\n")
                
                # Periodic evaluation and best weight tracking
                if i % eval_frequency == 0 and i > 0 and buffer.buffer_counter > warmup_steps:
                    steps_since_transition = i - last_curriculum_transition_step
                    in_grace_period = (TRAINING_MODE == 'curriculum' and 
                                      steps_since_transition < curriculum_grace_steps and 
                                      steps_since_transition > 0)
                    
                    print(f"\n{'='*60}")
                    print(f"Evaluation at step {i}")
                    if in_grace_period:
                        print(f"[PAUSE] Grace period: {curriculum_grace_steps - steps_since_transition} steps remaining")
                    
                    # NEW: Print recent training metrics
                    if len(critic_losses) > 0:
                        recent_critic_loss = np.mean(critic_losses[-100:])
                        recent_actor_loss = np.mean(actor_losses[-100:])
                        recent_q = np.mean(avg_q_values[-100:])
                        print(f"  Recent Critic Loss: {recent_critic_loss:.4f}")
                        print(f"  Recent Actor Loss: {recent_actor_loss:.4f}")
                        print(f"  Recent Avg Q-value: {recent_q:.4f}")
                    
                    eval_difficulty = 'current' if TRAINING_MODE == 'curriculum' else 'hard'
                    eval_results = evaluate_policy(actor_model, num_episodes=eval_episodes, 
                                                  verbose=True, eval_difficulty=eval_difficulty,
                                                  current_level=current_curriculum_level)
                    eval_reward = eval_results['avg_reward']
                    eval_history.append({'step': i, 'reward': eval_reward, 'success_rate': eval_results['success_rate'],
                                        'curriculum_level': current_curriculum_level if TRAINING_MODE == 'curriculum' else None})
                    
                    if in_grace_period and use_early_stopping:
                        print(f"  [PAUSE] Skipping early stopping check (adapting to new difficulty)")
                        print(f"{'='*60}\n")
                        continue
                    
                    # Track and save best weights
                    if eval_reward > best_eval_reward + min_improvement:
                        improvement = eval_reward - best_eval_reward
                        best_eval_reward = eval_reward
                        best_iteration = i
                        evaluations_without_improvement = 0
                        
                        print(f"  🌟 New best model! Reward improved by {improvement:.2f} (was {best_eval_reward-improvement:.2f}, now {best_eval_reward:.2f})")
                        actor_model.save(best_weights_file_actor)
                        critic_model.save(best_weights_file_critic)
                        print(f"  💾 Saved best weights at step {i}")
                    else:
                        evaluations_without_improvement += 1
                        effective_patience = patience * curriculum_patience_multiplier if TRAINING_MODE == 'curriculum' else patience
                        print(f"  No improvement ({evaluations_without_improvement}/{effective_patience})")
                        
                        # Only trigger early stopping if enabled
                        if use_early_stopping and evaluations_without_improvement >= effective_patience:
                            print(f"\n{'='*60}")
                            print(f"⚠️ EARLY STOPPING TRIGGERED")
                            print(f"No improvement for {effective_patience} evaluations ({effective_patience * eval_frequency} steps)")
                            print(f"Best reward: {best_eval_reward:.2f} at step {best_iteration}")
                            print(f"Current reward: {eval_reward:.2f}")
                            if TRAINING_MODE == 'curriculum':
                                print(f"Curriculum level at stopping: {current_curriculum_level}")
                            print(f"{'='*60}\n")
                            stopped_early = True
                            break
                    
                    print(f"{'='*60}\n")
            
            ep_reward_list.append(episodic_reward)
            avg_reward = np.mean(ep_reward_list[-40:])
            print("Episode {}: Iterations {}, Avg. Reward = {:.2f}, Last reward = {:.2f}. Avg. speed = {:.3f}".format(
                ep, i, avg_reward, episodic_reward, mean_speed/i))
            print("\n")
            
            if ep > 0 and ep % 40 == 0:
                print("## Evaluating policy ##")
                tracks.metrics_run(actor_model, 10)
            ep += 1
            
            if stopped_early:
                break
        
        if total_iterations > 0:
            # Restore best model
            if use_early_stopping and os.path.exists(best_weights_file_actor):
                print(f"\n{'='*60}")
                print("Restoring best model weights...")
                actor_model = keras.models.load_model(best_weights_file_actor)
                critic_model = keras.models.load_model(best_weights_file_critic)
                print(f"  Restored from step {best_iteration} (reward: {best_eval_reward:.2f})")
                print(f"{'='*60}\n")
            
            if save_weights:
                print(f"Saving final weights to {weights_file_actor}")
                critic_model.save(weights_file_critic)
                actor_model.save(weights_file_actor)
            
            # Save metadata
            elapsed_seconds = (datetime.now() - training_start_time).total_seconds() if training_start_time else None
            metadata = {
                'training_mode': TRAINING_MODE,
                'training_time_seconds': elapsed_seconds,
                'total_iterations': int(i),
                'final_avg_reward': float(avg_reward),
                'improvements': {
                    'shared_feature_extraction': True,
                    'improved_noise_balance': True,
                    'theta_based_progress_reward': True,
                    'reward_clipping': True,
                    'warmup_steps': warmup_steps,
                    'increased_network_capacity': True,
                    'equal_learning_rates': True
                },
                'early_stopping': {
                    'enabled': use_early_stopping,
                    'stopped_early': stopped_early,
                    'best_iteration': int(best_iteration),
                    'best_eval_reward': float(best_eval_reward),
                    'eval_history': eval_history
                },
                'learning_curve': {
                    'steps': [j*100 for j in range(len(avg_reward_list))],
                    'rewards': [float(r) for r in avg_reward_list]
                },
                'training_metrics': {
                    'critic_losses': [float(x) for x in critic_losses[-1000:]],  # Last 1000
                    'actor_losses': [float(x) for x in actor_losses[-1000:]],
                    'avg_q_values': [float(x) for x in avg_q_values[-1000:]]
                },
                'curriculum': {
                    'enabled': TRAINING_MODE == 'curriculum',
                    'transitions': curriculum_transitions if TRAINING_MODE == 'curriculum' else [],
                    'final_level': current_curriculum_level if TRAINING_MODE == 'curriculum' else None
                }
            }
            
            logs_dir = os.path.join('..', '..', 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            metadata_path = os.path.join(logs_dir, f'ddpg_{TRAINING_MODE}_improved_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Plotting
            plt.figure(figsize=(12, 6))
            plt.plot(avg_reward_list, linewidth=2, color='#3498db')
            
            if TRAINING_MODE == 'curriculum' and curriculum_transitions:
                for transition in curriculum_transitions:
                    step_index = transition['step'] // 100
                    plt.axvline(x=step_index, color='red', linestyle='--', alpha=0.7)
                    plt.text(step_index, plt.ylim()[1] * 0.95, 
                            f" → {transition['name']}", 
                            rotation=0, verticalalignment='top', fontsize=9)
            
            plt.xlabel("Training steps x100", fontweight='bold')
            plt.ylabel("Avg. Episodic Reward", fontweight='bold')
            
            title = f"IMPROVED DDPG Training Progress ({TRAINING_MODE.upper()} mode)"
            if TRAINING_MODE == 'curriculum':
                title += f"\nLevels: Easy→Medium→Hard"
            plt.title(title)
            plt.ylim(-3.5, 7)
            plt.grid(True, alpha=0.3)
            plt.show(block=False)
            plt.pause(0.001)
            
            print("\n" + "="*60)
            print("### IMPROVED DDPG Training Complete ###")
            print(f"Training mode: {TRAINING_MODE.upper()}")
            print(f"Trained over {i} steps")
            if TRAINING_MODE == 'curriculum':
                print(f"Curriculum transitions: {len(curriculum_transitions)}")
                for t in curriculum_transitions:
                    print(f"  Step {t['step']}: → {t['name']} (avg reward: {t['avg_reward']:.2f})")
            if stopped_early:
                print(f"⚠️ Stopped early due to no improvement")
                print(f"🌟 Best model from step {best_iteration} (reward: {best_eval_reward:.2f})")
            print(f"Final avg reward (last 40 eps): {avg_reward:.2f}")
            print("="*60)
    
    if is_training:
        start_t = datetime.now()
        train(training_start_time=start_t)
        end_t = datetime.now()
        print("Time elapsed: {}".format(end_t-start_t))
    
    tracks.newrun([actor_model])