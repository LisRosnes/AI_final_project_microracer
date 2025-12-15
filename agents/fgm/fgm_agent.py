#!/usr/bin/env python3
"""FGM reflex agent (compact, uses full 19-beam LiDAR if available).

This is a lightly refactored copy of the original `fgm_reflex.py` moved into
the `agents.fgm` package. The agent supports being called with either the
full 19-beam lidar array or the compact state [direction, distl, dist, distr, speed].
"""
import numpy as np


class FGMReflexAgent:
    def __init__(self,
                 bubble_radius_factor=0.4,
                 gap_min_width=3,
                 steering_gain=1.4,
                 max_speed_straight=0.95,
                 max_speed_turn=0.40,
                 curvature_threshold_factor=0.6,
                 accel_scale=1.0):
        self.bubble_radius_factor = bubble_radius_factor
        self.gap_min_width = int(gap_min_width)
        self.steering_gain = float(steering_gain)
        self.max_speed_straight = float(max_speed_straight)
        self.max_speed_turn = float(max_speed_turn)
        self.curvature_threshold_factor = float(curvature_threshold_factor)
        self.accel_scale = float(accel_scale)

        # lidar geometry
        self.N = 19
        self.k_center = self.N // 2

    def act(self, state, max_range=None):
        """Return [throttle, steering]. Accepts either raw 19-beam lidar or compact state."""
        lidar = None
        if hasattr(state, '__len__') and len(state) >= self.N:
            lidar = np.asarray(state[:self.N], dtype=float)
        elif hasattr(state, '__len__') and len(state) >= 5:
            direction, distl, dist, distr, speed = state[:5]
            lidar = self._estimate_lidar(direction, distl, dist, distr)
        else:
            return np.array([0.0, 0.0])

        if max_range is None or max_range <= 0:
            max_range = float(np.max(lidar)) if np.any(lidar > 0) else 10.0

        bubble_radius = self.bubble_radius_factor * max_range
        lidar_clean = np.where(lidar < bubble_radius, 0.0, lidar)

        gaps = []
        in_gap = False
        start = 0
        for i in range(self.N):
            if lidar_clean[i] > 0 and not in_gap:
                in_gap = True
                start = i
            elif (lidar_clean[i] <= 0 or i == self.N - 1) and in_gap:
                end = i - 1
                if i == self.N - 1 and lidar_clean[i] > 0:
                    end = i
                gaps.append((start, end))
                in_gap = False

        filtered = []
        for s, e in gaps:
            width = e - s + 1
            if width >= self.gap_min_width:
                filtered.append((s, e))

        if len(filtered) == 0:
            return np.array([0.0, 0.0])

        largest = max(filtered, key=lambda se: (se[1] - se[0] + 1))
        s_idx, e_idx = largest
        center_idx = (s_idx + e_idx) // 2
        mid_idx = self.k_center
        direction_error = (center_idx - mid_idx) / float(mid_idx)

        steering = self.steering_gain * direction_error
        steering = -steering
        steering = float(np.clip(steering, -1.0, 1.0))

        forward_distance = float(lidar[mid_idx])
        curvature_threshold = self.curvature_threshold_factor * max_range
        if forward_distance > curvature_threshold:
            throttle = float(np.clip(self.max_speed_straight, -1.0, 1.0))
        else:
            throttle = float(np.clip(self.max_speed_turn, -1.0, 1.0))

        throttle = float(np.clip(throttle * self.accel_scale, -1.0, 1.0))

        return np.array([throttle, steering])

    def _estimate_lidar(self, direction, distl, dist, distr):
        N = self.N
        k_center = self.k_center
        lidar = np.ones(N) * (dist * 0.7 if dist is not None else 0.0)
        lidar[k_center] = dist
        left_idx = k_center - 1
        right_idx = k_center + 1
        if left_idx >= 0:
            lidar[left_idx] = distl
        if right_idx < N:
            lidar[right_idx] = distr
        for i in range(N):
            if i < k_center:
                alpha = i / float(k_center)
                lidar[i] = (1 - alpha) * distl + alpha * dist
            elif i > k_center:
                alpha = (i - k_center) / float(N - k_center - 1)
                lidar[i] = (1 - alpha) * dist + alpha * distr
        return lidar


if __name__ == '__main__':
    try:
        import tracks
        racer = tracks.Racer(obstacles=False, turn_limit=True, chicanes=True, low_speed_termination=False)
        agent = FGMReflexAgent()
        state = racer.reset()
        steps = 0
        while not racer.done and steps < 500:
            action = agent.act(state)
            state, reward, done = racer.step(action)
            steps += 1
        print('Finished sanity run, steps=', steps, 'completation=', getattr(racer, 'completation', None))
    except Exception as e:
        print('Sanity run failed:', e)
