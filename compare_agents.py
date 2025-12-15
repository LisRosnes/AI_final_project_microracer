import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tracks
from mimic_agent import MimicAgent

# --- SETUP ---
N_LAPS = 3
TRACK_ARGS = dict(obstacles=True, turn_limit=True, chicanes=True)

def run_lap(agent_name, agent_act_fn):
    env = tracks.Racer(**TRACK_ARGS)
    state = env.reset()
    steps = 0
    start_time = time.time()
    
    while not env.done and steps < 2000:
        # Get action from the agent function
        action = agent_act_fn(state)
        state, reward, done = env.step(action)
        steps += 1
        
    lap_time = time.time() - start_time
    if env.completation < 1.0:
        print(f"   {agent_name}: CRASHED at {env.completation*100:.1f}% track completion.")
        return 999.0 # Penalty for crashing
    else:
        print(f"   {agent_name}: Finished in {lap_time:.4f} seconds.")
        return lap_time

def main():
    print("=== LOADING AGENTS ===")
    
    # 1. Load DDPG (The Teacher)
    print("Loading DDPG (Teacher)...")
    try:
        ddpg_raw = tf.saved_model.load("weights/ddpg_actor_model_car")
        def ddpg_act(state):
            # Helper to format state for DDPG
            if hasattr(state, '__len__') and len(state) >= 5:
                s = np.array([state[:5]], dtype=np.float32)
            else:
                s = np.zeros((1, 5), dtype=np.float32)
            out = ddpg_raw(s)
            if isinstance(out, (list, tuple)): return out[0].numpy()[0]
            elif isinstance(out, dict): return next(iter(out.values())).numpy()[0]
            else: return out.numpy()[0]
    except Exception as e:
        print(f"Error loading DDPG: {e}")
        return

    # 2. Load Mimic (The Student)
    print("Loading Mimic (Student)...")
    student_agent = MimicAgent(model_path="mimic_student_model.h5")
    
    print("\n=== STARTING RACE (Average of 3 Laps) ===")
    
    ddpg_times = []
    mimic_times = []
    
    for i in range(N_LAPS):
        print(f"\n--- Lap {i+1} ---")
        ddpg_times.append(run_lap("DDPG (Teacher)", ddpg_act))
        mimic_times.append(run_lap("Mimic (Student)", student_agent.act))

    avg_ddpg = np.mean(ddpg_times)
    avg_mimic = np.mean(mimic_times)
    
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    print(f"DDPG Avg Time:   {avg_ddpg:.4f} s")
    print(f"Mimic Avg Time:  {avg_mimic:.4f} s")
    print("-" * 30)
    
    if avg_mimic < avg_ddpg:
        print("RESULT: The Student is FASTER! (Amazing!)")
    elif abs(avg_mimic - avg_ddpg) < 1.0:
        print("RESULT: The Student matches the Teacher perfectly.")
        print("(This is the ideal outcome for Imitation Learning).")
    else:
        print("RESULT: The Student is slightly slower (Safety trade-off).")

if __name__ == "__main__":
    main()