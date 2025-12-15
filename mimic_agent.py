import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Try to import ReflexAgent (Fallback logic for Teacher Mode)
try:
    from agents.fgm.fgm_agent import FGMReflexAgent as ReflexAgent
except ImportError:
    try:
        from reflex_agent import ReflexAgent
    except:
        print("Warning: ReflexAgent not found. Teacher mode might fail.")

class MimicAgent:
    def __init__(self, speed_threshold=9.4, accel_threshold=3.45, model_path="mimic_student_model.h5"):
        self.mode = "TEACHER" # Default to Teacher (Switching)
        self.model = None
        
        # 1. Try to Load the Trained Student Model
        cwd = os.getcwd()
        full_path = os.path.join(cwd, model_path)
        
        if os.path.exists(full_path):
            try:
                # --- THE FIX IS HERE: compile=False ---
                self.model = keras.models.load_model(full_path, compile=False)
                self.mode = "STUDENT"
                print(f"[MimicAgent] Found {model_path}! Mode: STUDENT (Neural Network)")
            except Exception as e:
                print(f"[MimicAgent] Error loading student model: {e}")

        # 2. If Student is missing, Setup Teacher (Switching Logic)
        if self.mode == "TEACHER":
            print(f"[MimicAgent] Student model not found. Mode: TEACHER (Switching S={speed_threshold}, A={accel_threshold})")
            self.s_thresh = speed_threshold
            self.a_thresh = accel_threshold
            self.reflex_agent = ReflexAgent()
            
            # Load DDPG (Teacher's Expert Brain)
            try:
                self.ddpg_model = tf.saved_model.load("weights/ddpg_actor_model_car")
            except:
                self.ddpg_model = None
            
            self.last_speed = 0.0

    def reset(self):
        self.last_speed = 0.0
        if hasattr(self, 'reflex_agent'):
            if hasattr(self.reflex_agent, "reset"):
                self.reflex_agent.reset()

    def get_acceleration(self, current_speed):
        accel = current_speed - self.last_speed
        self.last_speed = current_speed
        return accel

    def act(self, state):
        # --- MODE 1: STUDENT (The Trained Brain) ---
        if self.mode == "STUDENT":
            if hasattr(state, '__len__') and len(state) >= 5:
                clean_state = np.array(state[:5], dtype=np.float32)
            else:
                clean_state = np.zeros(5, dtype=np.float32)
            
            state_batch = np.expand_dims(clean_state, axis=0)
            action = self.model.predict(state_batch, verbose=0)[0]
            return action

        # --- MODE 2: TEACHER (The Switching Logic) ---
        if hasattr(state, '__len__') and len(state) > 4:
            current_speed = state[4]
        else:
            current_speed = 0.0
        
        current_accel = self.get_acceleration(current_speed)

        # Switching Logic
        should_switch = (current_speed > self.s_thresh) or (abs(current_accel) > self.a_thresh)

        if should_switch:
            return self.reflex_agent.act(state)
        else:
            if self.ddpg_model:
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                model_output = self.ddpg_model(state_tensor)
                if isinstance(model_output, dict):
                    return next(iter(model_output.values())).numpy()[0]
                elif isinstance(model_output, list):
                    return model_output[0].numpy()[0]
                else:
                    return model_output.numpy()[0]
            else:
                return self.reflex_agent.act(state)