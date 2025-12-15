import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tracks
import os
from mimic_agent import MimicAgent

# --- CONFIG ---
OUTPUT_MODEL = "mimic_student_model.h5"
N_EPISODES = 60 

def main():
    # --- AUTO-DELETE FIX ---
    # We must delete the old model first, otherwise MimicAgent will 
    # load it and refuse to initialize the Teacher (DDPG/Reflex) components.
    if os.path.exists(OUTPUT_MODEL):
        print(f"Removing old {OUTPUT_MODEL} to force Teacher Mode...")
        os.remove(OUTPUT_MODEL)

    # 1. INITIALIZE "RELAXED" HYBRID TEACHER
    # Speed=25.0, Accel=5.0 to let DDPG drive fast
    print("1. INITIALIZING HYBRID TEACHER (Speed=25.0, Accel=5.0)...")
    teacher = MimicAgent(speed_threshold=25.0, accel_threshold=5.0)
    teacher.mode = "TEACHER" 
    
    # 2. COLLECT DATA
    env = tracks.Racer(obstacles=True, turn_limit=True, chicanes=True)
    all_states, all_actions = [], []
    
    print(f"2. COLLECTING DATA (Running {N_EPISODES} laps)...")
    for ep in range(N_EPISODES):
        state = env.reset()
        teacher.reset()
        done = False
        steps = 0
        while not done and steps < 2000:
            action = teacher.act(state)
            
            if hasattr(state, '__len__') and len(state) >= 5:
                all_states.append(np.array(state[:5], dtype=np.float32))
                all_actions.append(action)
            
            state, reward, done = env.step(action)
            steps += 1
        if (ep+1) % 10 == 0: print(f"   Collected episode {ep+1}...")

    X = np.array(all_states)
    y = np.array(all_actions)
    print(f"   Done! Collected {len(X)} samples.")

    # 3. TRAIN STUDENT
    print("3. TRAINING STUDENT NETWORK...")
    model = models.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='tanh')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train
    model.fit(X, y, epochs=25, batch_size=32, validation_split=0.1, verbose=1)
    
    # 4. SAVE
    model.save(OUTPUT_MODEL)
    print(f"4. SUCCESS! Saved FAST Hybrid student to {OUTPUT_MODEL}")

if __name__ == "__main__":
    main()