#!/usr/bin/env python3
"""
Simplified ReflexAgent implementation (prototype).

The previous, more complex implementation is commented out below. This new
agent reduces the 19-beam lidar into 5 sector signals and uses simple
linear rules for steering and acceleration. No arctan or nonlinear maps are
used so the outputs can reach the full [-1, 1] range.
"""

import numpy as np


# ----- Previous implementation (commented out) -----
"""
<OLD IMPLEMENTATION COMMENTED OUT FOR BREVITY>
The full original class body was here; it has been intentionally commented
out to keep a record while we prototype a much simpler controller.
"""


class ReflexAgent:
    """A minimal reflex agent that maps 19-beam lidar into 5 sectors.

    Behavior summary:
    - compress 19 beams into 5 sector averages: left(0), left-mid(1), center(2), right-mid(3), right(4)
    - steering: linear combination of sector imbalance and short-term shrinking of side sectors
    - braking: triggered by low center clearance or by rapid shrinking of side-mid sectors
    - outputs are linear and clipped to [-1,1]; full braking (accel=-1) is possible
    """

    def __init__(self):
        # lidar geometry (kept for compatibility with _estimate_lidar)
        self.N = 19
        self.phi_max_deg = 30
        self.k_center = self.N // 2

        # Simple gains (easy to tune)
        self.K_imbalance = 1.0    # steering gain based on left-right imbalance
        self.K_shrink = 1.5       # steering added when a side-mid sector shortens quickly
        self.K_speed = 1.0        # throttle gain towards v_target

        # thresholds
        self.brake_center_threshold = 2.5   # if center sector < this => full brake
        self.shrink_threshold = 0.5         # if sector decreased by more than this => react
        self.side_min_threshold = 1.0       # keep side beams at least this large

        # target speeds (we don't care about being fast)
        self.v_target = 0.5

        # smoothing / memory
        self.prev_sectors = np.ones(5) * 10.0
        self.prev_steer = 0.0
        self.prev_accel = 0.0
        self.alpha_steer = 0.6   # smoothing for steering (0..1)
        self.alpha_accel = 0.6   # smoothing for accel
        # Maximum speed cap for acceleration: when current speed >= cap,
        # the agent will not issue positive acceleration. Default inf (no cap).
        self.max_speed_cap = float('inf')

    def act(self, state):
        # state: [direction, distl, dist, distr, speed]
        if len(state) >= 5:
            direction, distl, dist, distr, speed = state[:5]
        else:
            return np.array([0.0, 0.0])

        # reconstruct a lightweight lidar (reuse same idea as before)
        lidar = self._estimate_lidar(direction, distl, dist, distr)

        # compress into 5 sector averages
        sectors = self._sectors_from_lidar(lidar)

        # steering: imbalance between right and left (positive => steer right)
        left = sectors[0]
        left_mid = sectors[1]
        center = sectors[2]
        right_mid = sectors[3]
        right = sectors[4]

        denom = (left + right + 1e-6)
        imbalance = (right - left) / denom

        # detect quick shrinking in side-mid sectors (approaching an obstacle)
        shrink1 = max(0.0, self.prev_sectors[1] - left_mid)
        shrink3 = max(0.0, self.prev_sectors[3] - right_mid)

        shrink_effect = 0.0
        if shrink1 > self.shrink_threshold:
            # left-mid is shortening -> steer right
            shrink_effect += self.K_shrink * (shrink1 / (self.prev_sectors[1] + 1e-6))
        if shrink3 > self.shrink_threshold:
            # right-mid shortening -> steer left (negative)
            shrink_effect -= self.K_shrink * (shrink3 / (self.prev_sectors[3] + 1e-6))

        raw_steer = self.K_imbalance * imbalance + shrink_effect

        # small behaviour to keep sides above minimum: gentle turn away from very-close side
        if left < self.side_min_threshold and right > left:
            raw_steer += 0.3  # nudge right
        if right < self.side_min_threshold and left > right:
            raw_steer -= 0.3  # nudge left

        # smoothing
        steering = (1 - self.alpha_steer) * raw_steer + self.alpha_steer * self.prev_steer
        steering = np.clip(steering, -1.0, 1.0)

        # acceleration / braking
        # full brake if center is dangerously small
        if center < self.brake_center_threshold:
            accel = -1.0
        else:
            # also brake if side-mid sectors shrank significantly
            if (shrink1 > self.shrink_threshold and left_mid < center) or (shrink3 > self.shrink_threshold and right_mid < center):
                accel = -0.8
            else:
                # gentle speed control toward v_target
                raw_acc = self.K_speed * (self.v_target - speed)
                accel = np.clip(raw_acc, -1.0, 1.0)

        accel = (1 - self.alpha_accel) * accel + self.alpha_accel * self.prev_accel
        accel = np.clip(accel, -1.0, 1.0)

        # save memory
        self.prev_sectors = sectors.copy()
        self.prev_steer = steering
        self.prev_accel = accel

        return np.array([accel, steering])

    def _estimate_lidar(self, direction, distl, dist, distr):
        # Simple reconstruction similar to original but compact
        N = self.N
        k_center = self.k_center
        lidar = np.ones(N) * dist * 0.7
        lidar[k_center] = dist
        left_idx = k_center - 1
        right_idx = k_center + 1
        if left_idx >= 0:
            lidar[left_idx] = distl
        if right_idx < N:
            lidar[right_idx] = distr
        # linear interpolation to fill
        for i in range(N):
            if i < k_center:
                alpha = i / k_center
                lidar[i] = (1 - alpha) * distl + alpha * dist
            elif i > k_center:
                alpha = (i - k_center) / (N - k_center - 1)
                lidar[i] = (1 - alpha) * dist + alpha * distr
        return lidar

    def _sectors_from_lidar(self, lidar):
        # split into five ranges and average
        N = self.N
        k = self.k_center
        # indices: 0..k-1 left side, k center, k+1..N-1 right
        # define sector boundaries
        s0 = np.mean(lidar[0:4])            # left (0..3)
        s1 = np.mean(lidar[4:9])            # left-mid (4..8)
        s2 = lidar[k]                       # center
        s3 = np.mean(lidar[10:15])         # right-mid (10..14)
        s4 = np.mean(lidar[15:19])         # right (15..18)
        return np.array([s0, s1, s2, s3, s4])

    def reset(self):
        self.prev_sectors = np.ones(5) * 10.0
        self.prev_steer = 0.0
        self.prev_accel = 0.0

