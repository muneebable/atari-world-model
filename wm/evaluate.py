"""Evaluate a trained world model on held-out episodes.

  python -m wm.evaluate --data data --runs runs --dynamics dynamics

Reports, per imagined step:
  - PSNR of imagined frames vs the real future, against two references: copying the last real frame
    (a model must beat this) and the tokenizer's own reconstruction (the best a token model can do);
  - action consistency: the IDM reads the imagined video and we check it recovers the actions the world
    model was conditioned on, compared with the IDM on real video. This tests controllability, not just looks.
Writes <dynamics>_eval.json, <dynamics>_rollouts.gif (real | imagined) and <dynamics>_counterfactual.gif
(same start, each column holds one action down for the whole rollout).
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from wm.common import (
    autocast,
    device,
    foreground_mask,
    load,
    masked_psnr,
    psnr,
    seed_all,
    static_background,
    to_float,
    to_uint8,
)
from wm.data import Split, action_names
from wm.dynamics import WorldModel
from wm.idm import CTX, IDM
from wm.tokenizer import Tokenizer


@torch.no_grad()
def decode(tok, tokens, grid_hw):
    """tokens (B, T, tpf) -> images (B, T, 3, H, W)."""
    B, T, _ = tokens.shape
    with autocast(tokens.device.type):
        x = tok.decode(tokens.reshape(B * T, *grid_hw)).float()
    return x.clamp(0, 1).view(B, T, *x.shape[1:])


@torch.no_grad()
def idm_accuracy(idm, frames, actions, first):
    """IDM accuracy on transitions t >= first that have the full CTX window. frames (B, L, 3, H, W)."""
    L = frames.shape[1]
    ts = [t for t in range(first, L - 1) if t + CTX[0] >= 0 and t + CTX[-1] < L]
    pred = torch.stack([idm(frames[:, [t + c for c in CTX]]).argmax(-1) for t in ts], 1)
    return float((pred == actions[:, ts]).float().mean())


def save_gif(path, videos, scale=3):
    """videos: rows of columns of uint8 (T, H, W, 3) clips, all the same shape -> animated grid."""
    rows = [np.concatenate(row, axis=2) for row in videos]  # side by side
    clip = np.concatenate(rows, axis=1).repeat(scale, 1).repeat(scale, 2)  # stacked, upscaled
    images = [Image.fromarray(f) for f in clip]
    images[0].save(path, save_all=True, append_images=images[1:], duration=150, loop=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--runs", default="runs")
    p.add_argument("--dynamics", default="dynamics")
    p.add_argument("--context", type=int, default=4)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    seed_all(a.seed)
    dev = device()

    tok = load(os.path.join(a.runs, "tokenizer.pt"), Tokenizer, dev)
    wm = load(os.path.join(a.runs, f"{a.dynamics}.pt"), WorldModel, dev)
    idm = load(os.path.join(a.runs, "idm.pt"), IDM, dev)
    names = action_names(a.data)
    val = Split(a.data, "val", tokens=True)
    grid_hw = val.tokens.shape[1:]
    C, H, L = a.context, a.horizon, a.context + a.horizon

    candidates = val.starts(L)
    starts = np.sort(
        np.random.default_rng(a.seed).choice(candidates, min(a.n, len(candidates)), replace=False)
    )
    real = to_float(val.windows(val.frames, starts, L), dev)
    tokens = torch.as_tensor(val.windows(val.tokens, starts, L).astype(np.int64), device=dev).flatten(2)
    actions = torch.as_tensor(val.windows(val.actions, starts, L)[:, :-1], device=dev)

    with autocast(dev):
        imagined_tokens = wm.rollout(tokens[:, :C], actions, a.temperature)
    imagined = decode(tok, imagined_tokens, grid_hw)
    future = real[:, C:]
    recon = decode(tok, tokens[:, C:], grid_hw)
    copy_last = real[:, C - 1 : C].expand_as(future)

    per_step = lambda x: [round(float(v), 2) for v in psnr(x, future).mean(0)]
    background = static_background(val.frames, dev)
    foreground = lambda x: masked_psnr(
        x, future, foreground_mask(x, background) | foreground_mask(future, background)
    )
    report = {
        "dynamics": a.dynamics,
        "windows": len(starts),
        "context_frames": C,
        "temperature": a.temperature,
        "psnr_db_per_step": {
            "world_model": per_step(imagined),
            "copy_last_frame_baseline": per_step(copy_last),
            "tokenizer_upper_bound": per_step(recon),
        },
        "psnr_db_mean": {
            "world_model": float(psnr(imagined, future).mean()),
            "copy_last_frame_baseline": float(psnr(copy_last, future).mean()),
            "tokenizer_upper_bound": float(psnr(recon, future).mean()),
        },
        "foreground_psnr_db_mean": {
            "world_model": foreground(imagined),
            "copy_last_frame_baseline": foreground(copy_last),
            "tokenizer_upper_bound": foreground(recon),
        },
        "action_consistency": {
            "idm_on_imagined_video": idm_accuracy(idm, torch.cat([real[:, :C], imagined], 1), actions, C - 1),
            "idm_on_real_video": idm_accuracy(idm, real, actions, C - 1),
            "chance": 1 / len(names),
        },
    }

    # Counterfactuals: same real context, then hold one action for the whole horizon.
    k = min(4, len(starts))
    columns = []
    for action in range(len(names)):
        held = torch.cat([actions[:k, : C - 1], torch.full((k, H), action, device=dev)], 1)
        with autocast(dev):
            columns.append(decode(tok, wm.rollout(tokens[:k, :C], held, a.temperature), grid_hw))
    report["counterfactual_columns"] = ["real"] + names

    path = lambda suffix: os.path.join(a.runs, f"{a.dynamics}_{suffix}")
    save_gif(
        path("rollouts.gif"),
        [[to_uint8(future[i]), to_uint8(imagined[i])] for i in range(min(8, len(starts)))],
    )
    save_gif(
        path("counterfactual.gif"),
        [[to_uint8(future[i])] + [to_uint8(c[i]) for c in columns] for i in range(k)],
    )
    with open(path("eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({key: v for key, v in report.items() if key != "psnr_db_per_step"}, indent=2))


if __name__ == "__main__":
    main()
