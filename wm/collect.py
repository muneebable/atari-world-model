"""Collect (frame, action) pairs from an Atari game with a random, action-repeating policy.

Output per split (train/, val/):
  frames.npy   uint8 (N, S, S, 3)
  actions.npy  int64 (N,)   actions[t] takes frames[t] to frames[t+1]; -1 on each episode's last frame
  ep.npy       int64 (N,)   episode id of each frame
and meta.json with the action names.
"""

import argparse
import json
import os

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

gym.register_envs(ale_py)


def resize(frames, size):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    # Max-pool, not area-average: averaging shrinks the ball to one dim pixel that a reconstruction loss ignores.
    x = F.adaptive_max_pool2d(x, (size, size))
    return x.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def run_episode(env, rng, max_steps, size):
    obs, _ = env.reset(seed=int(rng.integers(2**31)))
    frames, actions = [obs], []
    action, hold = 0, 0
    for _ in range(max_steps):
        # Hold each random action for 1-8 steps so the data has sustained movement, not jitter.
        if hold == 0:
            action, hold = int(rng.integers(env.action_space.n)), int(rng.integers(1, 9))
        hold -= 1
        obs, _, terminated, truncated, _ = env.step(action)
        actions.append(action)
        frames.append(obs)
        if terminated or truncated:
            break
    actions.append(-1)
    return resize(np.stack(frames), size), np.array(actions, dtype=np.int64)


def write_split(out, name, episodes):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/frames.npy", np.concatenate([f for f, _ in episodes]))
    np.save(f"{d}/actions.npy", np.concatenate([a for _, a in episodes]))
    np.save(f"{d}/ep.npy", np.concatenate([np.full(len(a), i) for i, (_, a) in enumerate(episodes)]))
    print(f"{name}: {len(episodes)} episodes, {sum(len(a) for _, a in episodes)} frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="ALE/Breakout-v5")
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--out", default="data")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    # Sticky actions off: with the ALE default (p=0.25) the logged action is not always the one executed,
    # which would put label noise into both the IDM and the world model.
    env = gym.make(a.game, frameskip=4, repeat_action_probability=0.0)
    rng = np.random.default_rng(a.seed)
    episodes = []
    for i in range(a.episodes):
        episodes.append(run_episode(env, rng, a.max_steps, a.size))
        if (i + 1) % 50 == 0:
            print(f"collected {i + 1}/{a.episodes} episodes", flush=True)

    n_val = max(1, int(a.episodes * a.val_frac))
    write_split(a.out, "val", episodes[:n_val])
    write_split(a.out, "train", episodes[n_val:])
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump({"game": a.game, "actions": env.unwrapped.get_action_meanings()}, f)


if __name__ == "__main__":
    main()
