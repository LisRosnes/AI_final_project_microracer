"""Bootstrapped DDPG: Train DDPG initialized with reflex agent demonstrations.

This script imports the tuned DDPG components from DDPG.py and adds bootstrapping.
The goal is to overcome initial exploration barriers in RL by pre-filling the replay
buffer with reflex agent demonstrations.

Usage:
    python DDPG_bootstrapped.py
"""
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

# Add necessary paths for imports
# Need to add MicroRacer root to find tracks.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Add agents directory to find fgm module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tracks
from fgm.fgm_agent import FGMReflexAgent

# Import tuned FGM config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'weights'))
from best_fgm_config_optuna import BEST_CONFIG as FGM_CONFIG

# Import all DDPG components from your tuned implementation
import DDPG

# Use the same hyperparameters and configuration from your tuned DDPG
gamma = DDPG.gamma
tau = DDPG.tau
critic_lr = DDPG.critic_lr
aux_lr = DDPG.aux_lr
num_states = DDPG.num_states
num_actions = DDPG.num_actions
upper_bound = DDPG.upper_bound
lower_bound = DDPG.lower_bound
buffer_dim = DDPG.buffer_dim
batch_size = DDPG.batch_size

# Use functions from DDPG.py
get_actor = DDPG.get_actor
get_critic = DDPG.get_critic
Buffer = DDPG.Buffer
update_target = DDPG.update_target
compose = DDPG.compose

# We need our own policy function that uses our local actor_model
def policy(state, current_iteration=0, max_iterations=50000, verbose=False):
    """Policy with decaying exploration noise (identical to DDPG.py but uses local actor_model)"""
    sampled_action = tf.squeeze(actor_model(state))
    
    # Decay noise from 1.0 to 0.2 over training
    decay_factor = max(0.2, 1.0 - 0.8 * (current_iteration / max_iterations))
    noise = np.random.normal(scale=0.1 * decay_factor, size=2)
    
    # We may change the amount of noise for actions during training
    noise[0] *= 2
    noise[1] *= .5
    # Adding noise to action
    sampled_action = sampled_action.numpy()
    sampled_action += noise
    #in verbose mode, we may print information about selected actions
    if verbose and sampled_action[0] < 0:
        print("decelerating")

    #Finally, we ensure actions are within bounds
    legal_action = np.clip(sampled_action, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]

# Create our own racer instance for training
# Use easier track configuration for better training
racer = tracks.Racer(
    obstacles=False,      # Disable obstacles
    chicanes=False,       # Disable chicanes (tight turns)
    turn_limit=True,      # Keep turn limit check
    low_speed_termination=True  # Keep low speed termination
)

# Training configuration
QUICK_TEST = False  # Set to False for full training run

if QUICK_TEST:
    imitation_episodes = 10          # Quick test: just 10 episodes for imitation
    rl_iterations = 500              # Quick test: 500 RL iterations
    print("⚡ QUICK TEST MODE - Reduced iterations for faster testing")
else:
    imitation_episodes = 1000    # Full training: 1000 episodes for imitation
    rl_iterations = 50000        # Full training: 50k RL iterations

# Imitation Learning parameters
use_imitation = False             # Set to False to skip imitation phase (reflex agent performs poorly)
imitation_batch_size = 64         # Batch size for imitation learning
imitation_learning_rate = 0.001   # Learning rate for imitation

# Switching criteria (when to transition from imitation to RL)
action_similarity_threshold = 0.85  # Switch when 85% action similarity achieved
min_imitation_episodes = 5          # Minimum episodes before considering switch
evaluate_every_n_episodes = 5       # Check similarity every N episodes

is_training = True  # Set to True for training, False for visualization
load_weights = False  # Set to True to continue from checkpoint
save_weights = True  # Save the bootstrapped model

# Separate weights files to compare with standard DDPG
weights_file_actor = "weights/ddpg_bootstrapped_actor_model_car"
weights_file_critic = "weights/ddpg_bootstrapped_critic_model_car"

print("\n" + "="*60)
print("DDPG TRAINING (Imitation Disabled - Reflex Agent Too Weak)")
print("="*60)
print(f"Using tuned hyperparameters from DDPG.py")
print(f"  gamma={gamma}, tau={tau}")
print(f"  critic_lr={critic_lr}, actor_lr={aux_lr}")
print(f"\n🏁 Track Config: Easy mode (no obstacles, no chicanes)")
print(f"\nSkipping imitation phase - reflex agent performance too poor")
print(f"Training with standard DDPG for {rl_iterations} iterations")
print(f"\nWeights will be saved to: {weights_file_actor}")
print("="*60 + "\n")


# Environment step function (from DDPG.py)
def step(action):
    """Environment step with optional empty actions"""
    n = 1
    t = np.random.randint(0, n)
    state, reward, done = racer.step(action)
    for i in range(t):
        if not done:
            state, t_r, done = racer.step([0, 0])
            reward += t_r
    return (state, reward, done)


# Creating models using DDPG.py's architecture
actor_model = get_actor()
critic_model = get_critic()

# We create the target model for double learning
target_actor = get_actor()
target_critic = get_critic()
target_actor.trainable = False
target_critic.trainable = False

# Compose actor and critic using DDPG's method
aux_model = compose(actor_model, target_critic)

## TRAINING ##
if load_weights:
    try:
        critic_model = keras.models.load_model(weights_file_critic)
        actor_model = keras.models.load_model(weights_file_actor)
        print("Loaded existing weights")
    except:
        print("Could not load weights, starting fresh")

# Making the weights equal initially
target_actor_weights = actor_model.get_weights()
target_critic_weights = critic_model.get_weights()
target_actor.set_weights(target_actor_weights)
target_critic.set_weights(target_critic_weights)

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
aux_optimizer = tf.keras.optimizers.Adam(aux_lr)

critic_model.compile(loss='mse', optimizer=critic_optimizer)
aux_model.compile(optimizer=aux_optimizer)

buffer = Buffer(buffer_dim, batch_size)

# Create separate optimizer for imitation learning
imitation_optimizer = tf.keras.optimizers.Adam(imitation_learning_rate)


def compute_action_similarity(actor_actions, reflex_actions):
    """
    Compute similarity between actor and reflex agent actions.
    Returns a value between 0 and 1, where 1 means identical actions.
    """
    # Use mean absolute error normalized by action range
    action_range = upper_bound - lower_bound
    mae = np.mean(np.abs(actor_actions - reflex_actions))
    # Convert to similarity (1 - normalized_error)
    similarity = max(0, 1 - (mae / action_range))
    return similarity


def imitation_learning_phase(max_episodes=1000):
    """
    Phase 1: Train actor to mimic reflex agent behavior via behavioral cloning.
    Stops when action similarity threshold is reached or max episodes hit.
    """
    print("\n" + "="*60)
    print("PHASE 1: IMITATION LEARNING (Behavioral Cloning)")
    print("="*60)
    print(f"Training actor to mimic reflex agent...")
    print(f"Target similarity: {action_similarity_threshold:.1%}")
    print(f"Evaluating every {evaluate_every_n_episodes} episodes")
    print(f"Using tuned FGM parameters: {FGM_CONFIG}\n")
    
    reflex_agent = FGMReflexAgent(**FGM_CONFIG)
    
    # Storage for imitation learning
    imitation_states = []
    imitation_actions = []
    
    similarity_history = []
    loss_history = []
    
    for ep in range(max_episodes):
        prev_state = racer.reset()
        done = False
        episode_states = []
        episode_actions = []
        
        while not done:
            # Get reflex agent's action
            reflex_action = reflex_agent.act(prev_state)
            
            # Store state-action pair
            episode_states.append(prev_state)
            episode_actions.append(reflex_action)
            
            # Step environment (also store in RL buffer for later)
            state, reward, done = step(reflex_action)
            fail = done and len(state) < num_states
            buffer.record((prev_state, reflex_action, reward, fail, state))
            
            prev_state = state
        
        # Add episode data to training set
        imitation_states.extend(episode_states)
        imitation_actions.extend(episode_actions)
        
        # Train on collected data in batches
        if len(imitation_states) >= imitation_batch_size:
            # Sample a batch
            indices = np.random.choice(len(imitation_states), imitation_batch_size, replace=False)
            batch_states = np.array([imitation_states[i] for i in indices])
            batch_actions = np.array([imitation_actions[i] for i in indices])
            
            # Train actor to predict reflex actions (supervised learning)
            with tf.GradientTape() as tape:
                predicted_actions = actor_model(batch_states, training=True)
                # MSE loss between predicted and reflex actions
                loss = tf.reduce_mean(tf.square(predicted_actions - batch_actions))
            
            gradients = tape.gradient(loss, actor_model.trainable_variables)
            imitation_optimizer.apply_gradients(zip(gradients, actor_model.trainable_variables))
            loss_history.append(float(loss))
        
        # Evaluate similarity periodically
        if (ep + 1) % evaluate_every_n_episodes == 0:
            # Test on a held-out evaluation
            eval_reflex_actions = []
            eval_actor_actions = []
            
            for _ in range(10):  # 10 evaluation episodes
                eval_state = racer.reset()
                eval_done = False
                while not eval_done:
                    reflex_action = reflex_agent.act(eval_state)
                    actor_action = actor_model(np.expand_dims(eval_state, 0), training=False).numpy()[0]
                    
                    eval_reflex_actions.append(reflex_action)
                    eval_actor_actions.append(actor_action)
                    
                    eval_state, _, eval_done = step(reflex_action)
            
            # Compute similarity
            similarity = compute_action_similarity(
                np.array(eval_actor_actions),
                np.array(eval_reflex_actions)
            )
            similarity_history.append(similarity)
            
            avg_loss = np.mean(loss_history[-100:]) if loss_history else 0
            print(f"Episode {ep+1}/{max_episodes}: "
                  f"Similarity={similarity:.3f}, "
                  f"Avg Loss={avg_loss:.4f}, "
                  f"Buffer size={buffer.buffer_counter}")
            
            # Check if we should switch to RL
            if ep >= min_imitation_episodes and similarity >= action_similarity_threshold:
                print(f"\n✓ Similarity threshold reached! ({similarity:.3f} >= {action_similarity_threshold})")
                print(f"  Completed {ep+1} imitation episodes")
                print(f"  Switching to RL fine-tuning...")
                break
    
    print(f"\n" + "="*60)
    print("PHASE 1 COMPLETE")
    print("="*60)
    print(f"Episodes completed: {ep+1}")
    print(f"Final similarity: {similarity_history[-1]:.3f}" if similarity_history else "N/A")
    print(f"Total transitions collected: {len(imitation_states)}")
    print(f"Buffer size: {buffer.buffer_counter}")
    print("="*60 + "\n")
    
    # Update target networks with imitation-learned weights
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())
    
    return {
        'episodes': ep + 1,
        'final_similarity': similarity_history[-1] if similarity_history else 0,
        'similarity_history': similarity_history,
        'loss_history': loss_history
    }


def train(training_start_time=None):
    """Two-phase training: Imitation learning followed by RL fine-tuning"""
    
    # Phase 1: Imitation Learning
    imitation_results = None
    if use_imitation:
        imitation_results = imitation_learning_phase(max_episodes=imitation_episodes)
    else:
        print("\nSkipping imitation phase - training from scratch\n")
    
    # History of rewards per episode
    ep_reward_list = []
    # Average reward history of last few episodes
    avg_reward_list = []
    
    i = 0
    mean_speed = 0
    ep = 0
    avg_reward = 0
    
    print("="*60)
    print("PHASE 2: RL FINE-TUNING (DDPG Training)")
    print("="*60)
    print(f"Starting RL training for {rl_iterations} iterations\n")
    
    while i < rl_iterations:
        prev_state = racer.reset()
        episodic_reward = 0
        mean_speed += prev_state[num_states-1]
        done = False
        prev_position = None  # Track position for distance reward
        
        while not(done):
            i = i + 1
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            # Policy with noise decay
            action = policy(tf_prev_state, current_iteration=i, max_iterations=rl_iterations)[0]
            # Get state and reward from the environment
            state, reward, done = step(action)
            
            # REWARD SHAPING: Add completion bonus
            if done and racer.completation == 1:
                reward += 10.0  # Completion bonus
                print("🏁 Episode completed! Bonus reward added.")
            
            # REWARD SHAPING: Add distance progress reward
            if prev_position is not None and not done:
                current_pos = (racer.carx, racer.cary)
                distance_moved = np.sqrt((current_pos[0] - prev_position[0])**2 + 
                                        (current_pos[1] - prev_position[1])**2)
                reward += distance_moved * 0.5
            
            if not done:
                prev_position = (racer.carx, racer.cary)
            
            # We distinguish between termination with failure and successful termination
            fail = done and len(state) < num_states
            buffer.record((prev_state, action, reward, fail, state))
            if not(done):
                mean_speed += state[num_states-1]
        
            episodic_reward += reward

            # Train on batch
            if buffer.buffer_counter > batch_size:
                states, actions, rewards, dones, newstates = buffer.sample_batch()
                targetQ = rewards + (1-dones)*gamma*(target_critic([newstates, target_actor(newstates)]))
                loss1 = critic_model.train_on_batch([states, actions], targetQ)
                loss2 = aux_model.train_on_batch(states)

                update_target(target_actor.variables, actor_model.variables, tau)
                update_target(target_critic.variables, critic_model.variables, tau)
            
            prev_state = state
            
            if i % 100 == 0:
                avg_reward_list.append(avg_reward)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode {}: Iterations {}, Avg. Reward = {:.2f}, Last reward = {:.2f}, Avg. speed = {:.2f}".format(
            ep, i, avg_reward, episodic_reward, mean_speed/i))
        
        if ep > 0 and ep % 40 == 0:
            print("\n## Evaluating policy ##")
            tracks.metrics_run(actor_model, 10)
            print()
        
        ep += 1

    if rl_iterations > 0:
        if save_weights:
            print(f"\nSaving weights to {weights_file_actor}")
            critic_model.save(weights_file_critic)
            actor_model.save(weights_file_actor)
        
        # Save training metadata for comparison
        elapsed_seconds = (datetime.now() - training_start_time).total_seconds() if training_start_time else None
        metadata = {
            'training_time_seconds': elapsed_seconds,
            'total_rl_iterations': i,
            'imitation_episodes': imitation_results['episodes'] if imitation_results else 0,
            'imitation_similarity': imitation_results['final_similarity'] if imitation_results else 0,
            'final_avg_reward': avg_reward,
            'learning_curve': {
                'steps': [j*100 for j in range(len(avg_reward_list))],
                'rewards': avg_reward_list
            },
            'imitation_learning': imitation_results if imitation_results else None
        }
        # Create logs directory if it doesn't exist (in MicroRacer/logs/)
        logs_dir = os.path.join('..', '..', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        metadata_path = os.path.join(logs_dir, 'ddpg_bootstrapped_training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create two-panel plot: imitation + RL
        if imitation_results and imitation_results.get('similarity_history'):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Imitation learning progress
            ax1.plot(imitation_results['similarity_history'], linewidth=2, color='#3498db')
            ax1.axhline(y=action_similarity_threshold, color='r', linestyle='--', 
                       label=f'Threshold ({action_similarity_threshold:.1%})')
            ax1.set_xlabel(f"Evaluation Step (every {evaluate_every_n_episodes} episodes)", fontweight='bold')
            ax1.set_ylabel("Action Similarity", fontweight='bold')
            ax1.set_title("Phase 1: Imitation Learning Progress")
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # RL training progress
            ax2.plot(avg_reward_list, linewidth=2, color='#e74c3c')
            ax2.set_xlabel("Training steps x100", fontweight='bold')
            ax2.set_ylabel("Avg. Episodic Reward", fontweight='bold')
            ax2.set_title("Phase 2: RL Fine-tuning Progress")
            ax2.set_ylim(-3.5, 7)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('Imitation + RL DDPG Training', fontsize=16, fontweight='bold')
        else:
            plt.figure(figsize=(10, 6))
            plt.plot(avg_reward_list)
            plt.xlabel("Training steps x100")
            plt.ylabel("Avg. Episodic Reward")
            plt.title("RL Training Progress (No Imitation)")
            plt.ylim(-3.5, 7)
            plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(logs_dir, 'bootstrapped_ddpg_training.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show(block=False)
        plt.pause(0.001)
        
        print("\n" + "="*60)
        print("### IMITATION + RL DDPG Training ended ###")
        if imitation_results:
            print(f"Imitation episodes: {imitation_results['episodes']}")
            print(f"Final similarity: {imitation_results['final_similarity']:.3f}")
        print(f"RL training steps: {i}")
        print(f"Final avg reward (last 40 eps): {avg_reward:.2f}")
        print("="*60)


if __name__ == "__main__":
    if is_training:
        start_t = datetime.now()
        train(training_start_time=start_t)
        end_t = datetime.now()
        print(f"\nTime elapsed: {end_t-start_t}")
    else:
        # Load and visualize
        if load_weights:
            actor_model = keras.models.load_model(weights_file_actor)
        tracks.newrun([actor_model])
