import optuna
import tracks
import numpy as np
import time
from mimic_agent import MimicAgent

# --- CONFIGURATION ---
N_TRIALS = 50           # Number of parameter combinations to try
MAX_STEPS = 2000        # Max steps per lap

def objective(trial):
    """
    Optuna will run this function 50 times with different S and A values.
    It returns the LAP TIME (lower is better).
    """
    # 1. Suggest Parameters
    # Search for a Speed threshold between 5 and 25
    s_try = trial.suggest_float("speed_threshold", 5.0, 25.0)
    
    # Search for an Acceleration threshold between 0.1 and 5.0
    a_try = trial.suggest_float("accel_threshold", 0.1, 5.0)
    
    # 2. Initialize Agent with these specific params
    agent = MimicAgent(speed_threshold=s_try, accel_threshold=a_try)
    
    # 3. Initialize Environment
    # Using the same settings as your eval_model.py
    racer = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True) 
    state = racer.reset()
    agent.reset()
    
    steps = 0
    start_time = time.time()
    
    # 4. Run the Lap
    while not racer.done and steps < MAX_STEPS:
        action = agent.act(state)
        state, reward, done = racer.step(action)
        steps += 1
    
    lap_duration = time.time() - start_time
    
    # 5. Penalties (Crucial for Optuna)
    # If the car crashed (completation < 1.0), give a massive penalty (9999 seconds).
    # This forces Optuna to find "Safe" parameters.
    if racer.completation < 1.0:
        return 9999.0 
    
    # If it took too long (stuck), penalize.
    if steps >= MAX_STEPS:
        return 9999.0

    return lap_duration

if __name__ == "__main__":
    print("Starting Optuna Tuning for Mimic (DDPG + Reflex)...")
    
    # We want to MINIMIZE lap time
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n" + "="*40)
    print("       RESULTS       ")
    print("="*40)
    print(f"Best Lap Time: {study.best_value:.4f} seconds")
    print("Best Parameters to use:")
    print(f"  S (Speed Threshold): {study.best_params['speed_threshold']:.4f}")
    print(f"  A (Accel Threshold): {study.best_params['accel_threshold']:.4f}")
    print("="*40)
    
    # Save the best params to a file so you can copy them easily
    with open("best_mimic_params.txt", "w") as f:
        f.write(str(study.best_params))