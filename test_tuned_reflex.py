#!/usr/bin/env python3
"""
Test the best stable reflex agent found by Optuna tuning.
Compares against PPO to show improvement/stability.
"""

import numpy as np
from reflex_agent import ReflexAgent
import tracks
import tensorflow as tf
import warnings
import sys
warnings.filterwarnings('ignore')


def evaluate_agent(agent, agent_name, num_episodes=10):
    """Evaluate agent over multiple episodes."""
    print(f"\nEvaluating {agent_name} ({num_episodes} episodes)...\n")
    
    speeds = []
    rewards = []
    steps_list = []
    crash_count = 0
    
    for ep in range(num_episodes):
        racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
        state = racer.reset()
        
        episode_reward = 0
        total_speed = 0
        step_count = 0
        done = False
        
        while not done and step_count < 500:
            action = agent.act(state)
            state, reward, done = racer.step(action)
            
            if state is not None and len(state) > 4:
                total_speed += state[4]
            
            episode_reward += reward
            step_count += 1
        
        if racer.completation != 0:
            crash_count += 1
        
        avg_speed = total_speed / max(step_count, 1)
        speeds.append(avg_speed)
        rewards.append(episode_reward)
        steps_list.append(step_count)
        
        status = "CRASH" if racer.completation != 0 else "OK"
        print(f"  Ep {ep+1:2d}: speed={avg_speed:.4f}, reward={episode_reward:7.2f}, steps={step_count:3d}  [{status}]")
    
    print(f"\n{'─'*70}")
    print(f"{agent_name} Results ({num_episodes} episodes):")
    print(f"  Average Speed:    {np.mean(speeds):.4f} ± {np.std(speeds):.4f}")
    print(f"  Average Reward:   {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
    print(f"  Average Steps:    {np.mean(steps_list):6.1f} ± {np.std(steps_list):5.1f}")
    print(f"  Crash Rate:       {crash_count}/{num_episodes} ({100*crash_count/num_episodes:.0f}%)")
    print(f"{'─'*70}\n")
    
    return {
        'speed': np.mean(speeds),
        'speed_std': np.std(speeds),
        'reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'steps': np.mean(steps_list),
        'steps_std': np.std(steps_list),
        'crashes': crash_count,
        'crash_rate': crash_count / num_episodes,
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("TESTING BEST STABLE REFLEX CONFIG (from Optuna tuning)")
    print("="*70)
    
    # Load best stable config
    print("\n1. OPTUNA-TUNED REFLEX AGENT (Stable)")
    print("─"*70)
    try:
        from weights.best_reflex_config import BEST_CONFIG
        reflex_agent = ReflexAgent()
        for param, value in BEST_CONFIG.items():
            setattr(reflex_agent, param, value)
        print(f"Loaded best stable config from weights/best_reflex_config.py")
        print(f"\nHyperparameters:")
        for param, value in sorted(BEST_CONFIG.items()):
            print(f"  {param:15s} = {value:.4f}")
        reflex_results = evaluate_agent(reflex_agent, "Tuned Reflex (Stable)", num_episodes=10)
    except FileNotFoundError:
        print("⚠️  best_reflex_config.py not found!")
        print("Make sure tune_optuna.py has completed and found a stable config.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Load and test PPO
    print("\n2. PPO AGENT (Pre-trained)")
    print("─"*70)
    try:
        ppo_actor = tf.keras.models.load_model('weights/ppo_actor_model_car')
        
        class PPOAgent:
            def __init__(self, actor_model):
                self.actor = actor_model
            
            def act(self, state):
                state_input = np.expand_dims(state, 0).astype(np.float32)
                action_output = self.actor(state_input)
                if len(action_output) > 1:
                    action = action_output[0]
                else:
                    action = action_output
                return action[0].numpy()
        
        ppo_agent = PPOAgent(ppo_actor)
        ppo_results = evaluate_agent(ppo_agent, "PPO (Pre-trained)", num_episodes=10)
        
        # Comparison
        print("\n" + "="*70)
        print("COMPARISON")
        print("="*70)
        
        print(f"\nSpeed:")
        print(f"  Tuned Reflex: {reflex_results['speed']:.4f}")
        print(f"  PPO:          {ppo_results['speed']:.4f}")
        if reflex_results['speed'] > ppo_results['speed']:
            print(f"  ✅ Reflex is {100*(reflex_results['speed']/ppo_results['speed']-1):.1f}% faster")
        else:
            print(f"  PPO is {100*(ppo_results['speed']/reflex_results['speed']-1):.1f}% faster")
        
        print(f"\nStability:")
        reflex_stable_pct = 100 * (1 - reflex_results['crash_rate'])
        ppo_stable_pct = 100 * (1 - ppo_results['crash_rate'])
        print(f"  Tuned Reflex: {reflex_stable_pct:.0f}% crash-free")
        print(f"  PPO:          {ppo_stable_pct:.0f}% crash-free")
        if reflex_results['crash_rate'] == 0:
            print(f"  ✅ Reflex is perfectly stable (0% crashes)")
        
        print(f"\nLongevity:")
        print(f"  Tuned Reflex: {reflex_results['steps']:.0f} steps")
        print(f"  PPO:          {ppo_results['steps']:.0f} steps")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Could not load PPO model: {e}\n")
