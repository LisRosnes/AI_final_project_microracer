from datetime import datetime
import sys
import os

# Set matplotlib to non-interactive backend to prevent hanging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[DEBUG] Starting DDPG.py...", flush=True)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import regularizers
import numpy as np
import json

print("[DEBUG] Imports successful", flush=True)

# Add MicroRacer root to path to find tracks.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tracks
print("[DEBUG] Tracks module imported", flush=True)

# Import FGM agent for imitation learning
try:
    from agents.fgm.fgm_agent import FGMReflexAgent
    print("[DEBUG] FGM agent imported successfully", flush=True)
except Exception as e:
    print(f"[DEBUG] FGM agent import failed: {e}", flush=True)
    FGMReflexAgent = None 

########################################
###### TRAINING MODE CONFIGURATION #####
TRAINING_MODE = 'easy'  # Options: 'easy', 'hard', 'curriculum', 'imitation'
print(f"[DEBUG] Training mode set to: {TRAINING_MODE}", flush=True)

# Curriculum training configuration
# NEW: Sampling-based curriculum to prevent catastrophic forgetting
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
    },
    # Sampling schedule: probabilities [easy, medium, hard] at different training phases
    'sampling_schedule': {
        0.0: [0.70, 0.25, 0.05],    # Early: 70% easy, 25% medium, 5% hard
        0.2: [0.50, 0.35, 0.15],    # 20% through: shift toward medium
        0.4: [0.30, 0.40, 0.30],    # 40% through: balanced mix
        0.6: [0.15, 0.35, 0.50],    # 60% through: shift toward hard
        0.8: [0.05, 0.25, 0.70],    # 80% through: mostly hard
        1.0: [0.00, 0.15, 0.85],    # Final: 85% hard, maintain some medium
    }
}

def get_curriculum_probabilities(progress):
    """Get sampling probabilities for [easy, medium, hard] based on training progress.
    
    Args:
        progress: Float between 0.0 and 1.0 indicating training progress
    
    Returns:
        List of probabilities [p_easy, p_medium, p_hard] that sum to 1.0
    """
    schedule = CURRICULUM_CONFIG['sampling_schedule']
    schedule_points = sorted(schedule.keys())
    
    # Find the two schedule points to interpolate between
    if progress <= schedule_points[0]:
        return schedule[schedule_points[0]]
    if progress >= schedule_points[-1]:
        return schedule[schedule_points[-1]]
    
    # Linear interpolation between schedule points
    for i in range(len(schedule_points) - 1):
        if schedule_points[i] <= progress < schedule_points[i + 1]:
            t0, t1 = schedule_points[i], schedule_points[i + 1]
            p0, p1 = schedule[t0], schedule[t1]
            
            # Interpolate
            alpha = (progress - t0) / (t1 - t0)
            probs = [p0[j] * (1 - alpha) + p1[j] * alpha for j in range(3)]
            return probs
    
    return schedule[schedule_points[-1]]

def sample_difficulty_level(progress):
    """Sample a difficulty level based on current training progress.
    
    Args:
        progress: Float between 0.0 and 1.0
    
    Returns:
        Integer 1, 2, or 3 representing easy, medium, or hard
    """
    probs = get_curriculum_probabilities(progress)
    return np.random.choice([1, 2, 3], p=probs)

def create_racer_for_level(level):
    """Create a racer instance for the specified difficulty level.
    
    Args:
        level: Integer 1 (easy), 2 (medium), or 3 (hard)
    
    Returns:
        tracks.Racer instance configured for that difficulty
    """
    level_key = f'level_{level}'
    config = CURRICULUM_CONFIG[level_key]
    racer = tracks.Racer(
        obstacles=config['obstacles'],
        chicanes=config['chicanes'],
        turn_limit=True,
        low_speed_termination=True
    )
    # Set track width if specified (for easy tracks)
    if 'track_width' in config:
        racer.track_width = config['track_width']
    return racer

# Imitation learning configuration
IMITATION_CONFIG = {
    'imitation_steps': 3000,        # Number of steps for imitation learning phase
    'imitation_batch_size': 64,     # Batch size for supervised learning
    'imitation_lr': 0.001,          # Learning rate for imitation phase
    'max_imitation_loss': 0.05,     # Max MSE loss to transition to RL
    'demo_buffer_size': 5000,       # Size of demonstration buffer
    'eval_frequency': 500,          # How often to evaluate imitation performance
}

# Better config for easy track imitation learning
BEST_FGM_CONFIG = {
    'accel_scale': 1.0,  # Full acceleration for easy track
    'bubble_radius_factor': 0.5981,
    'curvature_threshold_factor': 0.6725,
    'gap_min_width': 2.0,
    'max_speed_straight': 0.95,  # Higher speeds on easy track
    'max_speed_turn': 0.50,
    'steering_gain': 1.6095,
}


# Initialize racer based on training mode
if TRAINING_MODE == 'curriculum':
    # Start with easy track - will be re-sampled each episode
    racer = create_racer_for_level(1)
    print("🎓 CURRICULUM MODE: Sampling-based difficulty selection")
    print("   This prevents catastrophic forgetting by mixing difficulties throughout training")
    print("   Starting with Easy track (will sample from Easy/Medium/Hard based on progress)")
elif TRAINING_MODE == 'easy':
    racer = tracks.Racer(
        obstacles=False,
        chicanes=False,
        turn_limit=True,
        low_speed_termination=True
    )
    racer.track_width=0.1
    print("🟢 EASY MODE: Training on easy track (no obstacles, no chicanes)")
elif TRAINING_MODE == 'imitation':
    # Start with very easy track for imitation learning
    racer = tracks.Racer(
        obstacles=False,
        chicanes=False,
        turn_limit=False,  # Disable turn limit for imitation phase
        low_speed_termination=False  # Disable speed termination too
    )
    racer.track_width = 0.10  # Moderately wider track
    print("🎯 IMITATION MODE: Starting with imitation learning from reflex agent")
    print("   Phase 1: Supervised learning on easy track (wide, no termination limits)")
    print("   Phase 2: RL fine-tuning on hard track")
elif TRAINING_MODE == 'hard':
    racer = tracks.Racer(
        obstacles=True,
        chicanes=True,
        turn_limit=True,
        low_speed_termination=True
    )
    print("🔴 HARD MODE: Training on hard track (obstacles + chicanes)")
else:
    raise ValueError(f"Invalid TRAINING_MODE: {TRAINING_MODE}. Must be 'easy', 'hard', 'curriculum', or 'imitation'")

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
total_iterations = 5000 if QUICK_TEST else 75000

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
resume_from_checkpoint = True  # NEW: Resume from checkpoint if available

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
    
    # Imitation learning components
    if TRAINING_MODE == 'imitation':
        # Create a simple reflex agent that works with 5D state [direction, distl, dist, distr, speed]
        class SimpleReflexAgent:
            """Simple reflex agent optimized for 5D state space - focuses on centerline following."""
            def __init__(self, steering_gain=1.5, base_speed=0.7):
                self.steering_gain = steering_gain
                self.base_speed = base_speed
            
            def act(self, state):
                """Return [throttle, steering] based on 5D state."""
                if len(state) < 5:
                    return np.array([0.0, 0.0])
                
                direction, distl, dist, distr, speed = state[:5]
                
                # Simple steering: just follow the direction to centerline
                steering = -direction * self.steering_gain
                steering = float(np.clip(steering, -1.0, 1.0))
                
                # Simple throttle: slow down when turning hard
                turn_amount = abs(steering)
                
                if turn_amount < 0.3:
                    # Mostly straight
                    throttle = self.base_speed
                elif turn_amount < 0.6:
                    # Moderate turn
                    throttle = self.base_speed * 0.7
                else:
                    # Sharp turn
                    throttle = self.base_speed * 0.5
                
                # Also slow down if too close to walls
                if dist > 0 and dist < 0.15:
                    throttle *= 0.6
                
                throttle = float(np.clip(throttle, 0.0, 1.0))
                
                return np.array([throttle, steering])
        
        fgm_teacher = SimpleReflexAgent(
            steering_gain=1.5,
            base_speed=0.7
        )
        print("🤖 Simple Reflex teacher agent initialized for 5D state space")
        print(f"   Strategy: Centerline following with adaptive throttle")
        print(f"   Steering gain: 1.5")
        print(f"   Base speed: 0.7")
        print(f"   Note: Teacher doesn't need perfect completion - just demonstrates basic behaviors")
        
        # Demonstration buffer for imitation learning
        class DemonstrationBuffer:
            def __init__(self, capacity):
                self.capacity = capacity
                self.states = []
                self.actions = []
                self.counter = 0
            
            def add(self, state, action):
                if len(self.states) < self.capacity:
                    self.states.append(state)
                    self.actions.append(action)
                else:
                    idx = self.counter % self.capacity
                    self.states[idx] = state
                    self.actions[idx] = action
                self.counter += 1
            
            def sample_batch(self, batch_size):
                indices = np.random.choice(len(self.states), min(batch_size, len(self.states)))
                return np.array([self.states[i] for i in indices]), np.array([self.actions[i] for i in indices])
            
            def size(self):
                return len(self.states)
        
        demo_buffer = DemonstrationBuffer(IMITATION_CONFIG['demo_buffer_size'])
        imitation_losses = []
        print(f"📚 Demonstration buffer created (capacity: {IMITATION_CONFIG['demo_buffer_size']})")
        
        # Test FGM teacher agent before starting imitation learning
        def test_fgm_teacher(teacher_agent, test_racer, num_episodes=5):
            """Test the FGM teacher agent to verify it's working correctly."""
            print(f"\n{'='*60}")
            print(f"🔬 TESTING FGM TEACHER AGENT")
            print(f"{'='*60}")
            print(f"Running {num_episodes} test episodes to verify teacher performance...\n")
            
            test_rewards = []
            test_completions = []
            test_steps = []
            action_stats = {'throttle': [], 'steering': []}
            
            for ep in range(num_episodes):
                state = test_racer.reset()
                done = False
                episode_reward = 0
                steps = 0
                episode_throttles = []
                episode_steerings = []
                
                try:
                    while not done and steps < 1000:
                        action = teacher_agent.act(state)
                        episode_throttles.append(action[0])
                        episode_steerings.append(action[1])
                        
                        state, reward, done = test_racer.step(action)
                        episode_reward += reward
                        steps += 1
                except IndexError:
                    # Car went out of bounds - treat as failed episode
                    print(f"  Episode {ep+1}/{num_episodes}: Out of bounds - ❌ FAILED")
                    test_rewards.append(-3.0)
                    completed = 0
                    test_completions.append(completed)
                    test_steps.append(steps)
                    if len(episode_throttles) > 0:
                        action_stats['throttle'].extend(episode_throttles)
                        action_stats['steering'].extend(episode_steerings)
                    continue
                
                test_rewards.append(episode_reward)
                completed = 1 if test_racer.completation == 1 else 0
                test_completions.append(completed)
                test_steps.append(steps)
                action_stats['throttle'].extend(episode_throttles)
                action_stats['steering'].extend(episode_steerings)
                
                completion_str = "✅ COMPLETED" if test_racer.completation == 1 else "❌ FAILED"
                print(f"  Episode {ep+1}/{num_episodes}: Reward={episode_reward:.2f}, Steps={steps}, {completion_str}")
            
            # Summary statistics
            avg_reward = np.mean(test_rewards)
            avg_steps = np.mean(test_steps)
            completion_rate = sum(test_completions) / num_episodes
            avg_throttle = np.mean(action_stats['throttle'])
            avg_steering_abs = np.mean(np.abs(action_stats['steering']))
            
            print(f"\n📊 FGM Teacher Performance Summary:")
            print(f"   Avg Reward: {avg_reward:.2f}")
            print(f"   Avg Steps: {avg_steps:.1f}")
            print(f"   Completion Rate: {completion_rate:.1%}")
            print(f"   Avg Throttle: {avg_throttle:.3f}")
            print(f"   Avg |Steering|: {avg_steering_abs:.3f}")
            print(f"   Throttle range: [{np.min(action_stats['throttle']):.3f}, {np.max(action_stats['throttle']):.3f}]")
            print(f"   Steering range: [{np.min(action_stats['steering']):.3f}, {np.max(action_stats['steering']):.3f}]")
            
            if completion_rate < 0.1:
                print(f"\n⚠️  WARNING: Teacher has low completion rate ({completion_rate:.1%})")
                print(f"   This is okay - imitation learning can still learn basic behaviors!")
                print(f"   The RL phase will improve on the teacher's performance.")
            elif completion_rate >= 0.5:
                print(f"\n✨ Teacher shows strong performance! Good for imitation learning.")
            else:
                print(f"\n✓ Teacher shows moderate performance. Sufficient for bootstrapping.")
            
            print(f"{'='*60}\n")
            
            return {
                'avg_reward': avg_reward,
                'completion_rate': completion_rate,
                'avg_steps': avg_steps
            }
        
        # Run the test
        teacher_stats = test_fgm_teacher(fgm_teacher, racer, num_episodes=5)
        
    else:
        fgm_teacher = None
        demo_buffer = None
        imitation_losses = []
    
    def step(action):
        """Wrapper for environment step with bounds checking."""
        n = 1
        t = np.random.randint(0, n)
        try:
            state, reward, done = racer.step(action)
            for i in range(t):
                if not done:
                    state, t_r, done = racer.step([0, 0])
                    reward += t_r
        except IndexError:
            # Car went out of bounds (moved too far in one step)
            state = None
            reward = -3
            done = True
            racer.done = True
            racer.completation = 2  # crossing border
        return (state, reward, done)
    
    def evaluate_policy(actor, num_episodes=10, verbose=False, eval_difficulty='hard', current_level=1):
        """Evaluate the current policy."""
        total_rewards = []
        total_steps = []
        successes = 0
        
        # Determine difficulty for evaluation
        for ep in range(num_episodes):
            # CRITICAL FIX: Create NEW racer for EACH episode to test on different tracks
            # Otherwise all 10 episodes test on the SAME randomly generated track!
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
        current_curriculum_level = 1  # Legacy variable (not used in sampling mode)
        curriculum_transitions = []  # Legacy variable (not used in sampling mode)
        curriculum_difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}  # Track sampled difficulties
        
        i = 0
        mean_speed = 0
        ep = 0
        avg_reward = 0
        
        # IMITATION LEARNING PHASE
        if TRAINING_MODE == 'imitation':
            print(f"\n{'='*60}")
            print(f"🎯 PHASE 1: IMITATION LEARNING FROM FGM AGENT")
            print(f"{'='*60}")
            print(f"Collecting demonstrations from FGM teacher...")
            print(f"Target: {IMITATION_CONFIG['imitation_steps']} training steps")
            print(f"Max acceptable loss: {IMITATION_CONFIG['max_imitation_loss']}")
            print(f"{'='*60}\n")
            
            # Create a separate optimizer for imitation learning
            imitation_optimizer = tf.keras.optimizers.Adam(IMITATION_CONFIG['imitation_lr'])
            
            # Collect demonstrations while running FGM agent
            demo_episodes = 0
            while demo_buffer.size() < IMITATION_CONFIG['demo_buffer_size']:
                state = racer.reset()
                done = False
                steps_in_episode = 0
                
                try:
                    while not done and steps_in_episode < 500:
                        # Get FGM teacher's action
                        teacher_action = fgm_teacher.act(state)
                        
                        # Store demonstration
                        demo_buffer.add(state, teacher_action)
                        
                        # Take action in environment
                        state, reward, done = racer.step(teacher_action)
                        steps_in_episode += 1
                except IndexError:
                    # Car went out of bounds - skip this episode
                    print(f"  [Skipped episode {demo_episodes+1} - out of bounds]")
                    demo_episodes += 1
                    continue
                
                demo_episodes += 1
                if demo_episodes % 5 == 0:
                    print(f"  Collected {demo_buffer.size()}/{IMITATION_CONFIG['demo_buffer_size']} demonstrations ({demo_episodes} episodes)")
            
            print(f"\n✅ Demonstration collection complete: {demo_buffer.size()} samples\n")
            print(f"Starting supervised learning phase...\n")
            
            # Train actor to imitate FGM using supervised learning
            imitation_step = 0
            recent_losses = []
            recent_critic_losses = []
            
            # CRITICAL FIX: Also populate replay buffer with demonstrations for critic training
            print(f"Populating replay buffer with demonstrations for critic training...")
            demo_states_all, demo_actions_all = demo_buffer.sample_batch(demo_buffer.size())
            for idx in range(min(len(demo_states_all), buffer.buffer_capacity)):
                # For imitation demos, assume positive reward for good actions
                # This gives the critic something meaningful to learn
                demo_reward = 0.5  # Moderate positive reward for demonstration data
                demo_done = False
                next_idx = min(idx + 1, len(demo_states_all) - 1)
                buffer.record((demo_states_all[idx], demo_actions_all[idx], demo_reward, 
                              demo_done, demo_states_all[next_idx]))
            print(f"  ✅ Replay buffer populated with {min(len(demo_states_all), buffer.buffer_capacity)} demonstration transitions\n")
            
            while imitation_step < IMITATION_CONFIG['imitation_steps']:
                # Sample batch of demonstrations
                demo_states, demo_actions = demo_buffer.sample_batch(IMITATION_CONFIG['imitation_batch_size'])
                
                # Train actor to match teacher's actions (supervised learning)
                with tf.GradientTape() as tape:
                    predicted_actions = actor_model(demo_states, training=True)
                    # MSE loss between predicted and teacher actions
                    imitation_loss = tf.reduce_mean(tf.square(predicted_actions - demo_actions))
                
                # Update actor
                gradients = tape.gradient(imitation_loss, actor_model.trainable_variables)
                imitation_optimizer.apply_gradients(zip(gradients, actor_model.trainable_variables))
                
                imitation_losses.append(float(imitation_loss))
                recent_losses.append(float(imitation_loss))
                
                # CRITICAL FIX: Train critic on demonstration data to avoid actor-critic mismatch
                # Sample from replay buffer which now contains demonstration transitions
                states, actions, rewards, dones, newstates = buffer.sample_batch()
                
                # Train critic with demonstration Q-values
                with tf.GradientTape() as critic_tape:
                    target_actions = target_actor(newstates)
                    target_q = rewards + (1 - dones) * gamma * target_critic([newstates, target_actions])
                    current_q = critic_model([states, actions], training=True)
                    critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
                
                critic_gradients = critic_tape.gradient(critic_loss, critic_model.trainable_variables)
                critic_model.optimizer.apply_gradients(zip(critic_gradients, critic_model.trainable_variables))
                recent_critic_losses.append(float(critic_loss))
                
                # Update target networks gradually during imitation
                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)
                
                imitation_step += 1
                
                # Periodic evaluation of imitation performance
                if imitation_step % IMITATION_CONFIG['eval_frequency'] == 0:
                    avg_loss = np.mean(recent_losses)
                    avg_critic_loss = np.mean(recent_critic_losses) if recent_critic_losses else 0
                    recent_losses = []
                    recent_critic_losses = []
                    print(f"  Step {imitation_step}/{IMITATION_CONFIG['imitation_steps']}: Avg Imitation Loss = {avg_loss:.6f}, Critic Loss = {avg_critic_loss:.6f}")
                    
                    # Check if we've achieved good enough imitation
                    if avg_loss < IMITATION_CONFIG['max_imitation_loss']:
                        print(f"\n🌟 Imitation loss threshold reached! ({avg_loss:.6f} < {IMITATION_CONFIG['max_imitation_loss']})")
                        print(f"   Transitioning to RL phase early at step {imitation_step}\n")
                        break
            
            # Target networks already updated during imitation training loop
            # Final sync to ensure consistency
            target_actor.set_weights(actor_model.get_weights())
            target_critic.set_weights(critic_model.get_weights())
            
            print(f"\n{'='*60}")
            print(f"✅ IMITATION PHASE COMPLETE")
            print(f"   Final actor loss: {np.mean(imitation_losses[-100:]):.6f}")
            print(f"   Final critic loss: {np.mean(recent_critic_losses[-100:]) if recent_critic_losses else 0:.6f}")
            print(f"   Trained for {imitation_step} steps")
            print(f"   Both actor AND critic are now trained on demonstrations")
            print(f"{'='*60}\n")
            
            # Switch to hard track for RL training
            print(f"🔄 Switching to HARD track for RL fine-tuning...\n")
            racer = tracks.Racer(
                obstacles=True,
                chicanes=True,
                turn_limit=True,
                low_speed_termination=True
            )
            
            # CRITICAL FIX: Reset replay buffer to avoid corruption from domain shift
            # The buffer contains easy-track demonstrations that don't apply to hard track
            print(f"🔄 Resetting replay buffer to avoid domain shift corruption...")
            old_buffer_size = buffer.buffer_counter
            buffer = Buffer(buffer_dim, batch_size)
            print(f"   ✅ Buffer reset complete (cleared {old_buffer_size} easy-track samples)")
            print(f"   ℹ️  Will collect new {warmup_steps} warmup samples on HARD track\n")
            
            print(f"{'='*60}")
            print(f"🎯 PHASE 2: REINFORCEMENT LEARNING ON HARD TRACK")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"🚀 Starting training with WARMUP of {warmup_steps} steps")
            print(f"{'='*60}\n")
        
        while i < total_iterations:
            
            # CURRICULUM: Sample difficulty for this episode based on training progress
            sampled_level = None
            if TRAINING_MODE == 'curriculum':
                progress = i / total_iterations
                sampled_level = sample_difficulty_level(progress)
                racer = create_racer_for_level(sampled_level)
                
                # Track difficulty distribution
                level_names = {1: 'easy', 2: 'medium', 3: 'hard'}
                curriculum_difficulty_counts[level_names[sampled_level]] += 1
                
                # Log sampling probabilities periodically
                if ep % 50 == 0:
                    probs = get_curriculum_probabilities(progress)
                    print(f"[Curriculum] Episode {ep}, Step {i}/{total_iterations} ({100*progress:.1f}%) - "
                          f"Sampled: Level {sampled_level}, Probs: Easy={probs[0]:.2f}, Med={probs[1]:.2f}, Hard={probs[2]:.2f}")
            
            prev_state = racer.reset()
            episodic_reward = 0
            mean_speed += prev_state[num_states-1]
            done = False
            prev_theta = racer.cartheta  # NEW: Track theta for progress reward
            current_episode_level = sampled_level if TRAINING_MODE == 'curriculum' else None
            
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
                
                # IMPROVED REWARD SHAPING - FIXED ORDER
                # Track theta for progress (before modifying reward)
                if not done:
                    current_theta = racer.cartheta
                    theta_progress = current_theta - prev_theta
                    
                    # Handle wrap-around at 0/2π
                    if theta_progress < -np.pi:
                        theta_progress += 2 * np.pi
                    elif theta_progress > np.pi:
                        theta_progress -= 2 * np.pi
                    
                    if theta_progress > 0:
                        reward += theta_progress * 1.0  # Moderate progress reward

                    prev_theta = current_theta
                    # Add completion bonus BEFORE clipping
                    if done and racer.completation == 1:
                        reward += 10.0  # Strong but not overwhelming
                        print("🏁 Episode completed! Bonus reward added.")

                    # Clip to focused range (allows completion bonus to matter)
                    reward = np.clip(reward, -5, 5)
                
                fail = done and (state is None or len(state) < num_states)
                
                # Handle terminal states: use prev_state if state is None (out of bounds)
                next_state = state if state is not None else prev_state
                buffer.record((prev_state, action, reward, fail, next_state))
                
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
                
                # CURRICULUM: Track difficulty distribution for monitoring
                # (Old transition logic removed - now using continuous sampling)
                
                # Periodic evaluation and best weight tracking
                if i % eval_frequency == 0 and i > 0 and buffer.buffer_counter > warmup_steps:
                    # Grace period no longer needed with sampling-based curriculum
                    in_grace_period = False
                    
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
                    
                    # Always evaluate on hard track (final target) for curriculum
                    eval_difficulty = 'hard'
                    eval_results = evaluate_policy(actor_model, num_episodes=eval_episodes, 
                                                  verbose=True, eval_difficulty=eval_difficulty,
                                                  current_level=3)  # Always evaluate against hard
                    eval_reward = eval_results['avg_reward']
                    
                    # Track current sampling probabilities for curriculum
                    curriculum_probs = None
                    if TRAINING_MODE == 'curriculum':
                        progress = i / total_iterations
                        curriculum_probs = get_curriculum_probabilities(progress)
                    
                    eval_history.append({
                        'step': i, 
                        'reward': eval_reward, 
                        'success_rate': eval_results['success_rate'],
                        'curriculum_probs': curriculum_probs
                    })
                    
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
                                progress = i / total_iterations
                                probs = get_curriculum_probabilities(progress)
                                print(f"Final curriculum sampling: Easy={probs[0]:.2f}, Med={probs[1]:.2f}, Hard={probs[2]:.2f}")
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
                    'equal_learning_rates': True,
                    'imitation_learning': TRAINING_MODE == 'imitation'
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
                    'type': 'sampling-based' if TRAINING_MODE == 'curriculum' else None,
                    'schedule': CURRICULUM_CONFIG.get('sampling_schedule') if TRAINING_MODE == 'curriculum' else None,
                    'prevented_catastrophic_forgetting': True if TRAINING_MODE == 'curriculum' else False
                },
                'imitation': {
                    'enabled': TRAINING_MODE == 'imitation',
                    'imitation_losses': [float(x) for x in imitation_losses] if TRAINING_MODE == 'imitation' else [],
                    'imitation_steps': len(imitation_losses) if TRAINING_MODE == 'imitation' else 0,
                    'final_imitation_loss': float(np.mean(imitation_losses[-100:])) if TRAINING_MODE == 'imitation' and len(imitation_losses) > 0 else None
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
            
            # No transition markers for sampling-based curriculum
            # (difficulties are mixed throughout training)
            
            plt.xlabel("Training steps x100", fontweight='bold')
            plt.ylabel("Avg. Episodic Reward", fontweight='bold')
            
            title = f"IMPROVED DDPG Training Progress ({TRAINING_MODE.upper()} mode)"
            if TRAINING_MODE == 'curriculum':
                title += f"\nSampling-based curriculum (prevents catastrophic forgetting)"
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
                total_episodes = sum(curriculum_difficulty_counts.values())
                print(f"Sampling-based curriculum - Difficulty distribution:")
                print(f"  Easy:   {curriculum_difficulty_counts['easy']:4d} episodes ({100*curriculum_difficulty_counts['easy']/max(total_episodes, 1):.1f}%)")
                print(f"  Medium: {curriculum_difficulty_counts['medium']:4d} episodes ({100*curriculum_difficulty_counts['medium']/max(total_episodes, 1):.1f}%)")
                print(f"  Hard:   {curriculum_difficulty_counts['hard']:4d} episodes ({100*curriculum_difficulty_counts['hard']/max(total_episodes, 1):.1f}%)")
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