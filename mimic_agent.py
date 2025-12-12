import numpy as np
import tensorflow as tf

# 1. FIX IMPORT: Bypass the broken reflex_agent.py and import directly from the source
try:
    from agents.fgm.fgm_agent import FGMReflexAgent as ReflexAgent
except ImportError:
    # Fallback if python path behaves differently
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'agents', 'fgm'))
    from agents.fgm.fgm_agent import FGMReflexAgent as ReflexAgent

class MimicAgent:
    def __init__(self, speed_threshold=10.0, accel_threshold=1.0):
        # Initialize Reflex Agent (Safety Net)
        self.reflex_agent = ReflexAgent()
        
        # 2. FIX MODEL LOADING: Use tf.saved_model.load for Keras 3 compatibility
        weight_path = "weights/ddpg_actor_model_car"
        try:
            self.ddpg_model = tf.saved_model.load(weight_path)
            # print("Successfully loaded DDPG weights via tf.saved_model.load")
        except Exception as e:
            print(f"Error loading DDPG weights: {e}")
            self.ddpg_model = None

        # Switching Parameters
        self.s_thresh = speed_threshold
        self.a_thresh = accel_threshold
        
        # State tracking
        self.last_speed = 0.0
        self.current_mode = "DDPG" 

    def reset(self):
        """Reset internal state"""
        self.last_speed = 0.0
        if hasattr(self.reflex_agent, "reset"):
            self.reflex_agent.reset()

    def get_acceleration(self, current_speed):
        accel = current_speed - self.last_speed
        self.last_speed = current_speed
        return accel

    def act(self, state):
        # Extract Speed (Index 4 is standard)
        if hasattr(state, '__len__') and len(state) > 4:
            current_speed = state[4]
        else:
            current_speed = 0.0
        
        current_accel = self.get_acceleration(current_speed)

        # --- SWITCHING LOGIC ---
        should_switch = (current_speed > self.s_thresh) or (abs(current_accel) > self.a_thresh)

        if should_switch:
            self.current_mode = "REFLEX"
            action = self.reflex_agent.act(state)
        else:
            self.current_mode = "DDPG"
            if self.ddpg_model:
                # Prepare input tensor: Shape (1, 5) for the model
                # Note: We must cast to float32 for TensorFlow
                state_tensor = tf.convert_to_tensor([state], dtype=tf.float32)
                
                # Call the loaded model directly (it's a callable object now)
                # DDPG typically returns a tensor of actions
                model_output = self.ddpg_model(state_tensor)
                
                # Handle output: it might be a tensor, a list, or a dict depending on saving format
                if isinstance(model_output, dict):
                    # If it returns a dict (common in SavedModel), grab the first output
                    action = next(iter(model_output.values())).numpy()[0]
                elif isinstance(model_output, list):
                    action = model_output[0].numpy()[0]
                else:
                    action = model_output.numpy()[0]
            else:
                # Fallback if model failed to load
                action = self.reflex_agent.act(state)

        return action