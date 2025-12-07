#!/usr/bin/env python3
"""
Test script for PPO agent.
Loads pre-trained PPO weights and visualizes it racing on 2 different tracks.
"""

import numpy as np
import tensorflow as tf
import tracks
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_ppo_model(weights_path="weights/ppo_actor_model_car"):
    """Load the pre-trained PPO actor model."""
    print(f"Loading PPO model from: {weights_path}")
    try:
        model = tf.keras.models.load_model(weights_path)
        print("✓ PPO model loaded successfully")
        return model
    except Exception as e:
        print(f"✗ Failed to load PPO model: {e}")
        raise


def run_episode(actor, racer, max_steps=500):
    """Run a single episode with the given actor."""
    state = racer.reset()
    episode_reward = 0
    steps = 0
    done = False
    
    track_data = {
        'x': [racer.carx],
        'y': [racer.cary],
        'rewards': [],
        'speeds': [],
    }
    
    while not done and steps < max_steps:
        # Get action from actor
        state_input = np.expand_dims(state, 0)
        action_output = actor(state_input)
        
        # Handle both single and multiple output formats
        if len(action_output) > 1:
            action = action_output[0]
        else:
            action = action_output
        
        action = action[0].numpy()
        
        # Step environment
        state, reward, done = racer.step(action)
        
        # Track data
        track_data['x'].append(racer.carx)
        track_data['y'].append(racer.cary)
        track_data['rewards'].append(reward)
        
        # Only append speed if state is valid and has enough elements
        if state is not None and len(state) > 4:
            track_data['speeds'].append(state[4])
        else:
            track_data['speeds'].append(0)
        
        episode_reward += reward
        steps += 1
    
    return track_data, episode_reward, steps, racer.completation


def visualize_two_tracks(actor):
    """Run and visualize PPO agent on 2 different tracks side by side."""
    print("\n" + "=" * 70)
    print("Running PPO agent on 2 different tracks...")
    print("=" * 70)
    
    # Create two racers with different random tracks
    racer1 = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
    racer2 = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
    
    print("\nGenerating track 1...")
    track_data_1, reward_1, steps_1, completion_1 = run_episode(actor, racer1)
    
    print("\nGenerating track 2...")
    track_data_2, reward_2, steps_2, completion_2 = run_episode(actor, racer2)
    
    # Print results
    completion_names = {
        0: "Running",
        1: "Completed lap",
        2: "Off road",
        3: "Wrong direction",
        4: "Too slow",
    }
    
    print("\n" + "=" * 70)
    print("TRACK 1 RESULTS")
    print("=" * 70)
    print(f"Steps completed: {steps_1}")
    print(f"Episode reward: {reward_1:.2f}")
    print(f"Status: {completion_names.get(completion_1, 'Unknown')}")
    if len(track_data_1['speeds']) > 0:
        print(f"Average speed: {np.mean(track_data_1['speeds']):.3f}")
        print(f"Max speed: {np.max(track_data_1['speeds']):.3f}")
    
    print("\n" + "=" * 70)
    print("TRACK 2 RESULTS")
    print("=" * 70)
    print(f"Steps completed: {steps_2}")
    print(f"Episode reward: {reward_2:.2f}")
    print(f"Status: {completion_names.get(completion_2, 'Unknown')}")
    if len(track_data_2['speeds']) > 0:
        print(f"Average speed: {np.mean(track_data_2['speeds']):.3f}")
        print(f"Max speed: {np.max(track_data_2['speeds']):.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
    fig.suptitle('PPO Agent Racing on 2 Different Tracks', fontsize=14, fontweight='bold')
    
    # Track 1 visualization
    ax1 = axes[0]
    ax1.set_title(f'Track 1 - {completion_names.get(completion_1, "Unknown")}\nReward: {reward_1:.2f}')
    ax1.imshow(np.rot90(racer1.map), extent=[-1.3, 1.3, -1.3, 1.3], cmap='gray', vmin=-1, vmax=1)
    
    # Plot borders
    xs = 2 * np.pi * np.linspace(0, 1, 200)
    ax1.plot(racer1.csin(xs)[:, 0], racer1.csin(xs)[:, 1], color='black', linewidth=1)
    ax1.plot(racer1.csout(xs)[:, 0], racer1.csout(xs)[:, 1], color='black', linewidth=1)
    
    # Plot obstacles
    for i in range(len(racer1.obs_pos)):
        ax1.plot(racer1.obs_pos[i, :2], racer1.obs_pos[i, 2:], lw=2, color='crimson')
    
    # Plot trajectory
    ax1.plot(track_data_1['x'], track_data_1['y'], lw=2, color='lime', label='Trajectory', marker='o', markersize=2, markevery=10)
    ax1.plot(track_data_1['x'][0], track_data_1['y'][0], 'go', markersize=10, label='Start')
    ax1.plot(track_data_1['x'][-1], track_data_1['y'][-1], 'r*', markersize=15, label='End')
    
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # Track 2 visualization
    ax2 = axes[1]
    ax2.set_title(f'Track 2 - {completion_names.get(completion_2, "Unknown")}\nReward: {reward_2:.2f}')
    ax2.imshow(np.rot90(racer2.map), extent=[-1.3, 1.3, -1.3, 1.3], cmap='gray', vmin=-1, vmax=1)
    
    # Plot borders
    ax2.plot(racer2.csin(xs)[:, 0], racer2.csin(xs)[:, 1], color='black', linewidth=1)
    ax2.plot(racer2.csout(xs)[:, 0], racer2.csout(xs)[:, 1], color='black', linewidth=1)
    
    # Plot obstacles
    for i in range(len(racer2.obs_pos)):
        ax2.plot(racer2.obs_pos[i, :2], racer2.obs_pos[i, 2:], lw=2, color='crimson')
    
    # Plot trajectory
    ax2.plot(track_data_2['x'], track_data_2['y'], lw=2, color='lime', label='Trajectory', marker='o', markersize=2, markevery=10)
    ax2.plot(track_data_2['x'][0], track_data_2['y'][0], 'go', markersize=10, label='Start')
    ax2.plot(track_data_2['x'][-1], track_data_2['y'][-1], 'r*', markersize=15, label='End')
    
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create speed/reward plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    fig.suptitle('PPO Agent Performance Metrics', fontsize=14, fontweight='bold')
    
    # Track 1 speed
    axes[0, 0].plot(track_data_1['speeds'], color='blue', linewidth=1)
    axes[0, 0].set_title('Track 1 - Speed Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Speed')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Track 1 rewards
    axes[0, 1].plot(track_data_1['rewards'], color='green', linewidth=1)
    axes[0, 1].set_title('Track 1 - Reward Per Step')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Track 2 speed
    axes[1, 0].plot(track_data_2['speeds'], color='blue', linewidth=1)
    axes[1, 0].set_title('Track 2 - Speed Over Time')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Speed')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Track 2 rewards
    axes[1, 1].plot(track_data_2['rewards'], color='green', linewidth=1)
    axes[1, 1].set_title('Track 2 - Reward Per Step')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  PPO Agent Test & Visualization".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝\n")
    
    # Load model
    ppo_actor = load_ppo_model()
    
    # Run visualization
    visualize_two_tracks(ppo_actor)
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
