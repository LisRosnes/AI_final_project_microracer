#!/usr/bin/env python3
"""Train an agent with a curriculum of increasingly difficult tracks.

This script contains a compact, self-contained DDPG-style trainer adapted
from the project's `agents/Reinforcement/DDPG.py` but wrapped so it can be
used programmatically to alter the Racer parameters between epochs.

Usage (example):
  python3 train_curriculum.py --epochs 100 --episodes-per-epoch 20

Notes:
- The trainer accepts either a path to a saved Keras actor model or will
  build a fresh actor+critic pair. The default networks are small and fast.
- Curriculum steps are defined in the `CURRICULUM_STEPS` list. Hyperparams
  exposed by CLI allow tuning promotion/demotion thresholds, number of
  narrowing steps, and logging frequency.
"""
import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

import tracks
import multiprocessing as mp
import queue

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow import keras
except Exception:
    tf = None


# If TensorFlow is available, enable memory growth on GPUs so the process
# doesn't pre-allocate all GPU memory (useful on shared GPU nodes / SLURM).
if tf is not None:
    try:
        gpus = tf.config.list_physical_devices('GPU') if hasattr(tf.config, 'list_physical_devices') else []
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    # fallback for TF versions where experimental API differs
                    try:
                        tf.config.set_logical_device_configuration(gpu, [])
                    except Exception:
                        pass
            # Optional: allow setting an explicit memory cap (in MB) via env var
            # TF_GPU_MEM_MB: integer number of megabytes to reserve for this process
            try:
                mem_mb = os.environ.get('TF_GPU_MEM_MB')
                if mem_mb is not None:
                    try:
                        mem_limit = int(float(mem_mb))
                        # apply to the first visible physical GPU (logical visible devices
                        # may be remapped by CUDA_VISIBLE_DEVICES in the environment)
                        try:
                            tf.config.set_logical_device_configuration(
                                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=mem_limit)])
                        except Exception:
                            # fallback to experimental API if present
                            try:
                                tf.config.experimental.set_virtual_device_configuration(
                                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=mem_limit)])
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        # Best-effort; don't fail import if GPU config cannot be set
        pass


@dataclass
class CurriculumStep:
    label: str
    curves: int
    track_width: float
    chicanes: bool
    obstacles: bool
    turn_limit: bool


DEFAULT_CURRICULUM = [
    # Ultra-wide almost straight loop (easiest starting point)
    CurriculumStep('ultra_wide', curves=0, track_width=0.8, chicanes=False, obstacles=False, turn_limit=True),
    # Very wide loop
    CurriculumStep('very_wide', curves=0, track_width=0.6, chicanes=False, obstacles=False, turn_limit=True),
    # Wide loop
    CurriculumStep('wide_loop', curves=0, track_width=0.5, chicanes=False, obstacles=False, turn_limit=True),
    # Slightly curved, still wide
    CurriculumStep('gentle_curves', curves=4, track_width=0.5, chicanes=False, obstacles=False, turn_limit=True),
    # Add moderate curves
    CurriculumStep('moderate_curves', curves=8, track_width=0.5, chicanes=False, obstacles=False, turn_limit=True),
    # More curves, start narrowing
    CurriculumStep('more_curves_wide', curves=10, track_width=0.4, chicanes=False, obstacles=False, turn_limit=True),
    # More curves, narrower
    CurriculumStep('more_curves_narrow', curves=12, track_width=0.3, chicanes=False, obstacles=False, turn_limit=True),
    # Tight curves, narrow
    CurriculumStep('tight_curves', curves=14, track_width=0.2, chicanes=False, obstacles=False, turn_limit=True),
    # Chicanes intro (wider)
    CurriculumStep('chicanes_easy', curves=12, track_width=0.2, chicanes=True, obstacles=False, turn_limit=True),
    # Chicanes harder
    CurriculumStep('chicanes', curves=14, track_width=0.15, chicanes=True, obstacles=False, turn_limit=True),
    # Obstacles intro
    CurriculumStep('obstacles_easy', curves=12, track_width=0.15, chicanes=True, obstacles=True, turn_limit=True),
    # Obstacles harder
    CurriculumStep('obstacles', curves=14, track_width=0.1, chicanes=True, obstacles=True, turn_limit=True),
    # Final: harder turns + obstacles
    CurriculumStep('final_hard', curves=18, track_width=0.1, chicanes=True, obstacles=True, turn_limit=True),
]


class SimpleBuffer:
    def __init__(self, state_dim, action_dim, capacity=200000, batch_size=64):
        self.capacity = int(capacity)
        self.batch_size = int(batch_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)

    def record(self, s, a, r, d, sn):
        idx = self.ptr % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.dones[idx] = 1.0 if d else 0.0
        self.next_states[idx] = sn if sn is not None else np.zeros(self.state_dim, dtype=np.float32)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        idxs = np.random.randint(0, self.size, size=self.batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.dones[idxs], self.next_states[idxs])


class DDPGCurriculumTrainer:
    def __init__(self, actor_path: Optional[str] = None, lr_actor=1.38e-4, lr_critic=2.49e-4, gamma=0.951773, tau=1.023e-3, batch_size=32):
        if tf is None:
            raise RuntimeError('TensorFlow is required to run the trainer')
        self.state_dim = 5  # tracks.observe() returns 5-element compact state
        self.action_dim = 2
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer = SimpleBuffer(self.state_dim, self.action_dim, capacity=100000, batch_size=batch_size)
        self.episode_rewards = []  # track rewards per episode for debugging

        # actor/critic
        if actor_path and os.path.exists(actor_path):
            self.actor = keras.models.load_model(actor_path)
        else:
            self.actor = self._build_actor()

        self.critic = self._build_critic()
        self.target_actor = keras.models.clone_model(self.actor)
        self.target_critic = keras.models.clone_model(self.critic)
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(lr_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr_critic)
        # compile critic so train_on_batch can be used
        self.critic.compile(optimizer=self.critic_optimizer, loss='mse')

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(self.action_dim, activation='tanh')(x)
        model = keras.Model(inputs, out)
        return model

    def _build_critic(self):
        s_in = layers.Input(shape=(self.state_dim,))
        a_in = layers.Input(shape=(self.action_dim,))
        x = layers.Concatenate()([layers.Dense(32, activation='relu')(s_in), layers.Dense(32, activation='relu')(a_in)])
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(1)(x)
        return keras.Model([s_in, a_in], out)

    def policy(self, state, noise_scale=0.1):
        s = np.asarray(state, dtype=np.float32).reshape(1, -1)
        a = self.actor.predict(s, verbose=0)[0]
        a = a + np.random.normal(scale=noise_scale, size=self.action_dim)
        return np.clip(a, -1.0, 1.0)

    def update_target(self):
        # soft update
        aw = np.array(self.actor.get_weights(), dtype=object)
        taw = self.target_actor.get_weights()
        # use keras set_weights with interpolation
        new_taw = [self.tau * w + (1 - self.tau) * tw for w, tw in zip(self.actor.get_weights(), taw)]
        self.target_actor.set_weights(new_taw)
        new_tcw = [self.tau * w + (1 - self.tau) * tw for w, tw in zip(self.critic.get_weights(), self.target_critic.get_weights())]
        self.target_critic.set_weights(new_tcw)

    def train_step(self):
        if self.buffer.size < self.batch_size:
            return
        s, a, r, d, sn = self.buffer.sample()
        # compute target Q
        target_actions = self.target_actor.predict(sn, verbose=0)
        target_q = self.target_critic.predict([sn, target_actions], verbose=0)
        y = r + (1 - d) * self.gamma * target_q

        # train critic
        self.critic.train_on_batch([s, a], y)

        # train actor via policy gradient (maximize Q)
        with tf.GradientTape() as tape:
            actions_pred = self.actor(s)
            q_vals = self.critic([s, actions_pred])
            actor_loss = -tf.reduce_mean(q_vals)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

        # soft update
        self.update_target()

    def save_actor(self, path):
        try:
            self.actor.save(path)
        except Exception:
            # try saving weights only
            self.actor.save_weights(path + '.h5')

    def save_critic(self, path):
        try:
            self.critic.save(path)
        except Exception:
            self.critic.save_weights(path + '.h5')


def run_episode_and_record(trainer: DDPGCurriculumTrainer, racer: tracks.Racer, max_steps=2000, noise_scale=0.139193, step_idx=0):
    state = racer.reset()
    done = False
    steps = 0
    speeds = []
    episode_reward = 0.0
    while not done and steps < max_steps:
        safe_state = state if (state is not None and len(state) >= trainer.state_dim) else np.zeros(trainer.state_dim, dtype=np.float32)
        action = trainer.policy(safe_state, noise_scale=noise_scale)
        state_next, reward, done = racer.step(action)
        # Add bonus reward for successful steps on step 0 to encourage learning
        # The environment gives -3 for crash, so we need meaningful bonus for progress
        if step_idx == 0 and not done:
            reward += 0.1  # Stronger bonus to encourage safe navigation
        episode_reward += float(reward)
        fail = done and (state_next is None or (hasattr(state_next, '__len__') and len(state_next) < trainer.state_dim))
        trainer.buffer.record(safe_state, action, reward, fail, state_next if state_next is not None else np.zeros(trainer.state_dim))
        trainer.train_step()
        state = state_next
        steps += 1
        if state is not None and hasattr(state, '__len__') and len(state) > 4:
            try:
                speeds.append(float(state[4]))
            except Exception:
                pass
    crashed = racer.completation != 1
    mean_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    return steps, crashed, mean_speed, episode_reward


def worker_loop(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue, seed: Optional[int] = None):
    """Worker process: waits for 'run' tasks, executes episodes, and puts results on result_queue.

    Task dict format: {'cmd': 'run', 'step_cfg': CurriculumStep (as dict), 'episodes': int}
    Result: one dict per episode: {'worker': id, 'episode': idx, 'transitions': [(s,a,r,d,sn), ...], 'crashed': bool, 'mean_speed': float, 'steps': int}
    """
    # Workers should avoid importing tensorflow to not touch GPU
    import tracks as _tracks
    import numpy as _np

    if seed is not None:
        try:
            _np.random.seed(int(seed) + worker_id)
        except Exception:
            pass

    while True:
        try:
            task = task_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if task is None:
            break
        cmd = task.get('cmd')
        if cmd == 'stop':
            break
        if cmd != 'run':
            continue
        step_cfg = task.get('step_cfg')
        episodes = int(task.get('episodes', 1))
        # build racer from step config
        cfg = step_cfg
        racer = _tracks.Racer(obstacles=cfg['obstacles'], turn_limit=cfg['turn_limit'], chicanes=cfg['chicanes'], low_speed_termination=False)
        racer.curves = int(cfg['curves'])
        racer.track_width = float(cfg['track_width'])

        for ep in range(episodes):
            state = racer.reset()
            done = False
            steps = 0
            transitions = []
            speeds = []
            while not done and steps < 2000:
                safe_state = state if (state is not None and hasattr(state, '__len__') and len(state) >= 5) else _np.zeros(5, dtype=_np.float32)
                # simple random policy for data collection in worker; main learner not used here
                action = _np.zeros(2, dtype=_np.float32)
                # random small noise to encourage diversity
                action += _np.random.normal(scale=0.05, size=2)
                action = _np.clip(action, -1.0, 1.0)
                state_next, reward, done = racer.step(action)
                fail = done and (state_next is None or (hasattr(state_next, '__len__') and len(state_next) < 5))
                transitions.append((safe_state.tolist(), action.tolist(), float(reward), bool(fail), (state_next.tolist() if (state_next is not None and hasattr(state_next, '__len__')) else [0.0]*5)))
                state = state_next
                steps += 1
                if state is not None and hasattr(state, '__len__') and len(state) > 4:
                    try:
                        speeds.append(float(state[4]))
                    except Exception:
                        pass
            crashed = racer.completation != 1
            mean_speed = float(_np.mean(speeds)) if len(speeds) > 0 else 0.0
            # send result
            result = dict(worker=worker_id, episode=ep, transitions=transitions, crashed=crashed, mean_speed=mean_speed, steps=steps)
            try:
                result_queue.put(result)
            except Exception:
                pass



def evaluate_agent(trainer: DDPGCurriculumTrainer, racer: tracks.Racer, episodes=20, max_steps=2000):
    successes = 0
    crashes = 0
    mean_steps = []
    for _ in range(episodes):
        state = racer.reset()
        steps = 0
        done = False
        while not done and steps < max_steps:
            safe_state = state if (state is not None and len(state) >= trainer.state_dim) else np.zeros(trainer.state_dim, dtype=np.float32)
            # deterministic evaluation (no noise)
            s = np.asarray(safe_state, dtype=np.float32).reshape(1, -1)
            a = trainer.actor.predict(s, verbose=0)[0]
            state, reward, done = racer.step(a)
            steps += 1
        mean_steps.append(steps)
        if racer.completation == 1:
            successes += 1
        else:
            crashes += 1
    return {'episodes': episodes, 'successes': successes, 'crashes': crashes, 'mean_steps': float(np.mean(mean_steps))}


def build_racer_from_step(step: CurriculumStep, seed: Optional[int] = None):
    if seed is not None:
        np.random.seed(int(seed))
    racer = tracks.Racer(obstacles=step.obstacles, turn_limit=step.turn_limit, chicanes=step.chicanes, low_speed_termination=False)
    racer.curves = step.curves
    racer.track_width = step.track_width
    return racer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor-path', type=str, default=None, help='Optional path to saved Keras actor model to initialize')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--episodes-per-epoch', type=int, default=20)
    p.add_argument('--eval-episodes', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64, help='Training batch size for the replay buffer')
    p.add_argument('--lr-actor', type=float, default=5e-5, help='Actor learning rate')
    p.add_argument('--lr-critic', type=float, default=1e-4, help='Critic learning rate')
    p.add_argument('--narrow-steps', type=int, default=3, help='Number of narrowing steps from start width to min width')
    p.add_argument('--promote-threshold', type=float, default=0.90)
    p.add_argument('--demote-threshold', type=float, default=0.05)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--save-dir', type=str, default='weights/curriculum_runs')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=0, help='Number of parallel worker processes for environment rollouts')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # build curriculum sequence: insert narrowing steps between step 1 and step 2
    curriculum = list(DEFAULT_CURRICULUM)
    # create narrowing sequence from curriculum[0].track_width down to curriculum[1].track_width
    start_w = curriculum[0].track_width
    end_w = curriculum[1].track_width
    narrow_steps = args.narrow_steps
    widths = np.linspace(start_w, end_w, num=narrow_steps + 2)[1:-1] if narrow_steps > 0 else []
    # build intermediate steps keeping other flags from the second step
    intermediates = [CurriculumStep(f'narrow_{i+1}', curves=curriculum[1].curves, track_width=float(w), chicanes=False, obstacles=False, turn_limit=True) for i, w in enumerate(widths)]
    # final curriculum order: wide_loop, moderate_curves, narrow... , chicanes, obstacles, final_hard
    curriculum = [curriculum[0], curriculum[1]] + intermediates + curriculum[2:]

    trainer = DDPGCurriculumTrainer(actor_path=args.actor_path, batch_size=args.batch_size, 
                                     lr_actor=args.lr_actor, lr_critic=args.lr_critic)

    # optionally spawn worker processes for parallel rollouts
    num_workers = int(args.num_workers)
    workers = []
    task_queues = []
    result_queue = None
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)
        result_queue = mp.Queue()
        for wid in range(num_workers):
            tq = mp.Queue()
            p = mp.Process(target=worker_loop, args=(wid, tq, result_queue, args.seed))
            p.daemon = True
            p.start()
            workers.append(p)
            task_queues.append(tq)

    current_idx = 0
    best_success_rate = -1.0
    os.makedirs('weights', exist_ok=True)
    report_every = args.log_every
    warmup_epochs_remaining = 0  # Track warmup period after promotion (2 epochs)
    for epoch in range(1, args.epochs + 1):
        step_cfg = curriculum[current_idx]
        racer = build_racer_from_step(step_cfg, seed=args.seed)
        # run training episodes for this epoch (possibly via workers)
        crash_count = 0
        steps_list = []
        speeds = []
        if num_workers > 0:
            # distribute episodes across workers
            total_eps = int(args.episodes_per_epoch)
            base = total_eps // num_workers
            extras = total_eps % num_workers
            expected_msgs = total_eps
            for wid, tq in enumerate(task_queues):
                eps = base + (1 if wid < extras else 0)
                if eps <= 0:
                    # send a no-op
                    tq.put({'cmd': 'run', 'step_cfg': step_cfg.__dict__, 'episodes': 0})
                else:
                    tq.put({'cmd': 'run', 'step_cfg': step_cfg.__dict__, 'episodes': eps})

            received = 0
            # collect episode results
            while received < expected_msgs:
                try:
                    msg = result_queue.get(timeout=30)
                except queue.Empty:
                    print('Warning: timed out waiting for worker results')
                    break
                received += 1
                # process message
                transitions = msg.get('transitions', [])
                crashed = bool(msg.get('crashed', False))
                mean_speed = float(msg.get('mean_speed', 0.0))
                steps_list.append(int(msg.get('steps', 0)))
                speeds.append(mean_speed)
                crash_count += 1 if crashed else 0
                # feed transitions into buffer and do training steps
                for tr in transitions:
                    s, a, r, d, sn = tr
                    trainer.buffer.record(np.array(s, dtype=np.float32), np.array(a, dtype=np.float32), float(r), bool(d), np.array(sn, dtype=np.float32))
                # run some training iterations after each episode
                for _ in range(max(1, len(transitions) // max(1, trainer.batch_size))):
                    trainer.train_step()
        else:
            for ep in range(args.episodes_per_epoch):
                steps, crashed, mean_speed, episode_reward = run_episode_and_record(trainer, racer, max_steps=2000, noise_scale=0.139193, step_idx=current_idx)
                crash_count += 1 if crashed else 0
                steps_list.append(steps)
                speeds.append(mean_speed)
                trainer.episode_rewards.append(episode_reward)

        crash_rate = crash_count / float(len(steps_list))

        # evaluate performance deterministically
        eval_racer = build_racer_from_step(step_cfg, seed=args.seed)
        eval_stats = evaluate_agent(trainer, eval_racer, episodes=args.eval_episodes)
        success_rate = eval_stats['successes'] / float(eval_stats['episodes'])

        promoted = False
        demoted = False
        # promotion/demotion rules
        # skip demotion during warmup period (2 epochs after promotion)
        if success_rate >= args.promote_threshold and current_idx < len(curriculum) - 1:
            current_idx += 1
            promoted = True
            warmup_epochs_remaining = 2  # Disable demotion for 2 epochs after promotion
        elif success_rate <= args.demote_threshold and current_idx > 0 and warmup_epochs_remaining <= 0:
            # Don't demote during warmup period
            current_idx -= 1
            demoted = True
        
        # Decrement warmup counter each epoch
        if warmup_epochs_remaining > 0:
            warmup_epochs_remaining -= 1

        # periodic reporting or on change
        if epoch % report_every == 0 or promoted or demoted:
            mean_reward = np.mean(trainer.episode_rewards) if trainer.episode_rewards else 0
            print(f"Epoch {epoch:3d} | step={current_idx} ({step_cfg.label}) | crash_rate={crash_rate:.3f} | eval_success={success_rate:.3f} | mean_reward={mean_reward:.3f} | mean_steps={np.mean(steps_list):.1f} | mean_speed={np.mean(speeds):.3f}")
            if promoted:
                print(f"  PROMOTED -> step {current_idx} ({curriculum[current_idx].label})")
                print(f"  [Warmup: demotion disabled for 2 epochs]")
            if demoted:
                print(f"  DEMOTED -> step {current_idx} ({curriculum[current_idx].label})")
            if warmup_epochs_remaining > 0 and not promoted:
                print(f"  [Warmup: {warmup_epochs_remaining} epoch(s) remaining before demotion re-enabled]")
            
            # Evaluate on hardest track (final_hard) for logging
            hardest_step = curriculum[-1]  # final_hard is the last step
            hard_racer = build_racer_from_step(hardest_step, seed=args.seed)
            hard_eval = evaluate_agent(trainer, hard_racer, episodes=10, max_steps=2000)
            hard_crash_rate = hard_eval['crashes'] / float(hard_eval['episodes'])
            hard_success_rate = hard_eval['successes'] / float(hard_eval['episodes'])
            print(f"  [Hard track eval: crash_rate={hard_crash_rate:.2f} success_rate={hard_success_rate:.2f}]")
            
            # Log to CSV file for plotting
            log_file = os.path.join(args.save_dir, 'training_log.csv')
            log_exists = os.path.exists(log_file)
            with open(log_file, 'a') as f:
                if not log_exists:
                    f.write('epoch,step_idx,step_label,crash_rate,eval_success,mean_reward,mean_steps,mean_speed,hard_crash_rate,hard_success_rate\n')
                f.write(f'{epoch},{current_idx},{step_cfg.label},{crash_rate:.4f},{success_rate:.4f},{mean_reward:.4f},{np.mean(steps_list):.2f},{np.mean(speeds):.4f},{hard_crash_rate:.4f},{hard_success_rate:.4f}\n')
            
            # also dump a small JSON summary
            summary = dict(epoch=epoch, step_index=current_idx, step_label=step_cfg.label, crash_rate=crash_rate, eval=eval_stats, hard_eval=hard_eval)
            with open(os.path.join(args.save_dir, f'curriculum_epoch_{epoch:03d}.json'), 'w') as f:
                json.dump(summary, f, indent=2)

        # save best model to weights/ when eval success improves
        try:
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_actor_path = os.path.join('weights', 'curriculum_actor_best')
                best_critic_path = os.path.join('weights', 'curriculum_critic_best')
                trainer.save_actor(best_actor_path)
                trainer.save_critic(best_critic_path)
                print(f"  Saved new best actor/critic to weights/ (success_rate={best_success_rate:.3f})")
        except Exception as e:
            print('Warning: failed to save best model:', e)

        # save actor periodically
        if epoch % (report_every * 2) == 0 or promoted or demoted:
            fname = os.path.join(args.save_dir, f'actor_epoch_{epoch:03d}')
            trainer.save_actor(fname)

    # final save
    trainer.save_actor(os.path.join(args.save_dir, 'actor_final'))
    print('Curriculum training finished. Models and summaries saved to', args.save_dir)


if __name__ == '__main__':
    main()
