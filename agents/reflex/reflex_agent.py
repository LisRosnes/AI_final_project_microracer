#!/usr/bin/env python3
"""
Reflex Agent moved into package `agents.reflex`.
"""

import numpy as np


class ReflexAgent:
    """
    A reflex-based driving agent for MicroRacer.
    Uses 19 lidar distances and speed to output steering and acceleration.
    """
    
    def __init__(self):
        """Initialize the reflex agent with precomputed angles and parameters."""
        
        # ====== Lidar Configuration ======
        self.N = 19  # Number of lidar beams
        self.phi_max_deg = 30  # Maximum angle in degrees
        
        # Precompute beam angles (in degrees)
        self.beam_angles = np.array([
            -self.phi_max_deg + i * (2 * self.phi_max_deg / (self.N - 1))
            for i in range(self.N)
        ])
        
        # Key indices
        self.k_center = self.N // 2  # Index 9 for N=19
        
        # ====== STEERING HYPERPARAMETERS ======
        self.K_heading = 0.6           # Heading gain (reduced for safety)
        self.heading_exp = 1.2         # Nonlinearity on heading
        self.K_center = 0.4            # Center correction gain (reduced)
        self.beta_s = 0.6              # Steering smoothing factor (increased for stability)
        
        # ====== SPEED HYPERPARAMETERS ======
        self.v_min = 0.2               # Absolute minimum speed
        self.v_turn = 0.6              # Safe speed in moderate turns
        self.v_max = 1.5               # Desired speed on straights
        
        self.d_emergency = 2.5          # Immediate braking threshold
        self.d_caution = 5.0            # Slow down if below this
        self.d_straight = 15.0          # Full speed if above this
        self.c_turn_thresh = 0.5        # Curvature threshold for turns
        
        self.K_speed = 1.0              # Speed control gain (increased for faster response)
        self.a_max_brake = 1.0          # Max braking magnitude
        self.a_max_accel = 1.0          # Max acceleration magnitude
        self.beta_a = 0.3               # Acceleration smoothing factor (decreased for quicker response)
        
        # ====== STATE (for smoothing) ======
        self.prev_steer = 0.0
        self.prev_accel = 0.0
    
    def act(self, state):
        """
        Compute action from state.
        
        Args:
            state: numpy array [direction, distl, dist, distr, speed]
                  from the observe() function in tracks.py
                  
        Returns:
            (acceleration, steering) both in [-1, 1]
        """
        # Extract state components from observe() output
        # observe() returns: [dir, distl, dist, distr, v]
        # where:
        #   dir = angle of maximum lidar distance
        #   distl, dist, distr = adjacent distances around max
        #   v = current speed
        
        if len(state) >= 5:
            direction, distl, dist, distr, speed = state[:5]
        else:
            # Fallback for unexpected state format
            return np.array([0.0, 0.0])
        
        # Reconstruct lidar from compressed observation
        lidar = self._estimate_lidar(direction, distl, dist, distr)
        
        # ====== STEERING CONTROL ======
        steering = self._compute_steering(lidar)
        
        # ====== SPEED CONTROL ======
        acceleration = self._compute_acceleration(lidar, speed)
        
        # Clip to [-1, 1]
        steering = np.clip(steering, -1.0, 1.0)
        acceleration = np.clip(acceleration, -1.0, 1.0)
        
        return np.array([acceleration, steering])
    
    def _estimate_lidar(self, direction, distl, dist, distr):
        """
        Estimate full 19-element lidar from the compressed observation.
        
        The observe() function returns the maximum distance direction and 
        the distances on either side. We reconstruct a plausible full lidar array.
        """
        lidar = np.ones(self.N) * dist * 0.7  # Base case - slightly more conservative
        
        # Place the center readings
        lidar[self.k_center] = dist
        
        # The direction tells us the angle of max distance
        # We can use this to understand the track geometry
        
        # Place left and right observations
        # distl and distr are adjacent to the peak
        left_idx = self.k_center - 1
        right_idx = self.k_center + 1
        
        if left_idx >= 0:
            lidar[left_idx] = distl
        if right_idx < self.N:
            lidar[right_idx] = distr
        
        # Smooth the profile
        for i in range(self.N):
            if i == 0:
                lidar[i] = max(lidar[i], distl * 0.9)
            elif i < self.k_center:
                # Interpolate on left side
                alpha = i / self.k_center
                lidar[i] = (1 - alpha) * distl + alpha * dist
            elif i > self.k_center:
                # Interpolate on right side
                alpha = (i - self.k_center) / (self.N - self.k_center - 1)
                lidar[i] = (1 - alpha) * dist + alpha * distr
        
        return lidar
    
    def _compute_steering(self, lidar):
        """
        Compute steering action based on lidar.
        
        Returns:
            steering in approximately [-1, 1] (before clipping)
        """
        # Find most open direction (but with some inertia)
        k_max = np.argmax(lidar)
        theta_open = self.beam_angles[k_max]
        
        # Normalize open direction (in range [-1, 1])
        h = theta_open / self.phi_max_deg
        
        # Heading steering - very conservative
        # Use smaller K_heading and power to avoid aggressive turning
        s_heading = (self.K_heading * 
                    np.sign(h) * (np.abs(h) ** self.heading_exp) * 0.5)
        
        # Left and right averages
        d_left = np.mean(lidar[0:self.k_center])
        d_right = np.mean(lidar[self.k_center:self.N])
        
        # Asymmetry - very weak centering
        d_sum = d_left + d_right + 1e-6
        asym = (d_right - d_left) / d_sum
        
        # Centering steering - very weak
        s_center = self.K_center * asym * 0.3
        
        # Combine with strong smoothing
        s_raw = s_heading + s_center
        
        # Apply heavy smoothing for stability
        steering = (1 - self.beta_s) * s_raw + self.beta_s * self.prev_steer
        self.prev_steer = steering
        
        return steering
    
    def _compute_acceleration(self, lidar, speed):
        """
        Compute acceleration based on lidar and speed.
        
        Simple strategy: always try to maintain a minimum safe speed.
        """
        # Forward distance
        center_indices = np.arange(
            max(0, self.k_center - 1),
            min(self.N, self.k_center + 2)
        )
        d_forward = np.mean(lidar[center_indices])
        
        # If road is blocked, brake hard
        if d_forward < self.d_emergency:
            acceleration = -1.0  # Full brake
        # If approaching obstacle or turn, slow down
        elif d_forward < self.d_caution:
            v_target = self.v_turn
            acceleration = np.clip(self.K_speed * (v_target - speed), -1.0, 1.0)
        # Otherwise, maintain good speed
        else:
            # Always try to accelerate a bit to maintain speed
            # This prevents the "too slow" termination
            if speed < 0.2:
                acceleration = 0.8  # Accelerate quickly
            else:
                v_target = self.v_max
                a_raw = self.K_speed * (v_target - speed)
                acceleration = np.clip(a_raw, -1.0, 1.0)
        
        # Apply minimal smoothing for quick response
        acceleration = (
            (1 - self.beta_a) * acceleration + 
            self.beta_a * self.prev_accel
        )
        self.prev_accel = acceleration
        
        return acceleration
    
    def reset(self):
        """Reset internal state for a new episode."""
        self.prev_steer = 0.0
        self.prev_accel = 0.0
