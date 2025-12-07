#!/usr/bin/env python3
"""
Comprehensive agent testing and comparison.
- Smoke tests: Verify environment, imports, models load
- Baseline: Compare Tuned ReflexAgent vs PPO agent
- Evaluation: Detailed metrics, stability analysis, recommendations

Usage:
    python test_agents.py              # Full comparison (10 eps each)
    python test_agents.py --quick      # Quick test (3 eps each)
    python test_agents.py --smoke      # Smoke test only
    python test_agents.py --reflex N   # Test just reflex with N episodes
    python test_agents.py --ppo N      # Test just PPO with N episodes
"""

import numpy as np
from reflex_agent import ReflexAgent
import tracks
import tensorflow as tf
import warnings
import sys
warnings.filterwarnings('ignore')


def smoke_test():
    """Verify all dependencies and models load correctly."""
    print("\n" + "="*70)
    print("SMOKE TEST")
    print("="*70)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Imports
    tests_total += 1
    try:
        print("\n[1/5] Testing imports...", end=" ")
        import numpy
        import tensorflow
        import tracks
        from reflex_agent import ReflexAgent
        print("✓")
        tests_passed += 1
    except Exception as e:
        print(f"✗ {e}")
    
    # Test 2: ReflexAgent instantiation
    tests_total += 1
    try:
        print("[2/5] Testing ReflexAgent creation...", end=" ")
        agent = ReflexAgent()
        print("✓")
        tests_passed += 1
    except Exception as e:
        print(f"✗ {e}")
    
    # Test 3: ReflexAgent.act()
    tests_total += 1
    try:
        print("[3/5] Testing ReflexAgent.act()...", end=" ")
        agent = ReflexAgent()
        state = np.array([0.0, 1.0, 1.0, 1.0, 0.2])  # Mock state
        action = agent.act(state)
        assert len(action) == 2, f"Expected 2 actions, got {len(action)}"
        print("✓")
        tests_passed += 1
    except Exception as e:
        print(f"✗ {e}")
    
    # Test 4: MicroRacer environment
    tests_total += 1
    try:
        print("[4/5] Testing MicroRacer environment...", end=" ")
        racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
        state = racer.reset()
        assert state is not None, "Reset returned None"
        print("✓")
        tests_passed += 1
    except Exception as e:
        print(f"✗ {e}")
    
    # Test 5: PPO model loading
    tests_total += 1
    try:
        print("[5/5] Testing PPO model loading...", end=" ")
        ppo_actor = tf.keras.models.load_model('weights/ppo_actor_model_car')
        assert ppo_actor is not None, "Model is None"
        print("✓")
        tests_passed += 1
    except Exception as e:
        print(f"✗ {e}")
    
    print(f"\n{'─'*70}")
    print(f"Smoke test: {tests_passed}/{tests_total} passed")
    print(f"{'─'*70}\n")
    
    return tests_passed == tests_total


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


def compare_agents(num_episodes=10):
    """Compare tuned ReflexAgent vs PPO."""
    print("\n" + "="*70)
    print("AGENT COMPARISON")
    print("="*70)
    
    # Load tuned reflex agent
    print("\n1. TUNED REFLEX AGENT")
    print("─"*70)
    try:
        from best_config import BEST_CONFIG
        reflex_agent = ReflexAgent()
        for param, value in BEST_CONFIG.items():
            setattr(reflex_agent, param, value)
        print(f"Loaded best config from best_config.py")
        reflex_results = evaluate_agent(reflex_agent, "Tuned Reflex", num_episodes=num_episodes)
    except Exception as e:
        print(f"Warning: Could not load best_config.py ({e})")
        print("Running with default ReflexAgent instead...\n")
        reflex_agent = ReflexAgent()
        reflex_results = evaluate_agent(reflex_agent, "Reflex (Default)", num_episodes=num_episodes)
    
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
        ppo_results = evaluate_agent(ppo_agent, "PPO (Pre-trained)", num_episodes=num_episodes)
        
        # Detailed comparison
        print("\n" + "="*70)
        print("DETAILED COMPARISON")
        print("="*70)
        
        print("\nSpeed Performance:")
        print(f"  Tuned Reflex: {reflex_results['speed']:.4f} ± {reflex_results['speed_std']:.4f}")
        print(f"  PPO:          {ppo_results['speed']:.4f} ± {ppo_results['speed_std']:.4f}")
        speed_diff = reflex_results['speed'] - ppo_results['speed']
        speed_pct = (speed_diff / ppo_results['speed']) * 100 if ppo_results['speed'] > 0 else 0
        print(f"  Delta:        {speed_diff:+.4f} ({speed_pct:+.1f}%)")
        
        print("\nLongevity (steps before crash):")
        print(f"  Tuned Reflex: {reflex_results['steps']:.0f} ± {reflex_results['steps_std']:.0f}")
        print(f"  PPO:          {ppo_results['steps']:.0f} ± {ppo_results['steps_std']:.0f}")
        steps_ratio = reflex_results['steps'] / ppo_results['steps'] if ppo_results['steps'] > 0 else 0
        print(f"  Ratio:        {steps_ratio:.1%} (Reflex vs PPO)")
        
        print("\nStability (crash-free runs):")
        reflex_stable = (1.0 - reflex_results['crash_rate']) * 100
        ppo_stable = (1.0 - ppo_results['crash_rate']) * 100
        print(f"  Tuned Reflex: {100-reflex_results['crash_rate']*100:.0f}% stable ({reflex_results['crashes']}/{num_episodes} crashes)")
        print(f"  PPO:          {100-ppo_results['crash_rate']*100:.0f}% stable ({ppo_results['crashes']}/{num_episodes} crashes)")
        
        print("\nReward (cumulative per episode):")
        print(f"  Tuned Reflex: {reflex_results['reward']:7.2f} ± {reflex_results['reward_std']:6.2f}")
        print(f"  PPO:          {ppo_results['reward']:7.2f} ± {ppo_results['reward_std']:6.2f}")
        
        print("\n" + "="*70)
        print(f"{'─'*70}\n")
        
    except Exception as e:
        print(f"Error loading PPO model: {e}\n")


if __name__ == '__main__':
    # Parse arguments
    smoke_only = '--smoke' in sys.argv
    quick_mode = '--quick' in sys.argv
    reflex_only = '--reflex' in sys.argv
    ppo_only = '--ppo' in sys.argv
    
    # Determine number of episodes
    num_episodes = 3 if quick_mode else 10
    
    # Check for episode override
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ['--reflex', '--ppo'] and i+1 < len(sys.argv)-1:
            try:
                num_episodes = int(sys.argv[i+2])
            except:
                pass
    
    # Run tests
    if smoke_only:
        smoke_test()
    elif reflex_only:
        if not smoke_test():
            print("Smoke test failed! Exiting.")
            sys.exit(1)
        try:
            from best_config import BEST_CONFIG
            agent = ReflexAgent()
            for param, value in BEST_CONFIG.items():
                setattr(agent, param, value)
            evaluate_agent(agent, "Tuned Reflex", num_episodes=num_episodes)
        except Exception as e:
            print(f"Error: {e}")
    elif ppo_only:
        if not smoke_test():
            print("Smoke test failed! Exiting.")
            sys.exit(1)
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
            evaluate_agent(ppo_agent, "PPO", num_episodes=num_episodes)
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Full test
        if not smoke_test():
            print("Smoke test failed! Exiting.")
            sys.exit(1)
        compare_agents(num_episodes=num_episodes)
