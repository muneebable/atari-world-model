"""VQ-VAE that turns a 64x64 frame into an 8x8 grid of discrete tokens.

python -m wm.tokenizer train  --data data --out runs
python -m wm.tokenizer encode --data data --ckpt runs/tokenizer.pt   # writes tokens.npy for each split
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from wm.common import (
    autocast,
    device,
    foreground_mask,
    load,
    masked_psnr,
    psnr,
    save,
    seed_all,
    static_background,
    to_float,
    to_uint8,
    train_loop,
)
from wm.data import Split

# NOTE: moving objects (ball, paddle) cover <1% of pixels; unweighted L1 lets the tokenizer drop them entirely
# and still score ~28 dB. Weighting them up makes them a sizeable share of the loss.
FOREGROUND_WEIGHT = 50.0


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.f(x)


class Tokenizer(nn.Module):
    def __init__(self, n_codes=512, dim=64, ch=(64, 128, 256), beta=0.25):
        super().__init__()
        self.beta = beta
        enc, c = [nn.Conv2d(3, ch[0], 3, 1, 1)], ch[0]
        for out in ch:  # one 2x downsample per stage: 64 -> 8 with three stages
            enc += [Res(c), nn.Conv2d(c, out, 4, 2, 1)]
            c = out
        enc += [Res(c), nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, dim, 1)]
        dec = [nn.Conv2d(dim, c, 3, 1, 1), Res(c)]
        for out in reversed(ch):
            dec += [nn.Upsample(scale_factor=2), nn.Conv2d(c, out, 3, 1, 1), Res(out)]
            c = out
        dec += [nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 3, 3, 1, 1)]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)
        self.codebook = nn.Embedding(n_codes, dim)

    def _codes(self):
        # Cosine-similarity codebook (unit-norm codes and latents) keeps more codes in use than plain L2 VQ.
        return F.normalize(self.codebook.weight.float(), dim=-1)

    def quantize(self, x):
        z = F.normalize(self.enc(x).float().permute(0, 2, 3, 1), dim=-1)  # (B, h, w, D)
        idx = (z @ self._codes().T).argmax(-1)
        return z, idx

    def decode(self, idx):
        """idx: (B, h, w) token ids -> images (B, 3, H, W)."""
        return self.dec(self._codes()[idx].permute(0, 3, 1, 2))

    @torch.no_grad()
    def encode(self, x):
        return self.quantize(x)[1]

    def forward(self, x):
        z, idx = self.quantize(x)
        q = self._codes()[idx]
        vq_loss = F.mse_loss(q, z.detach()) + self.beta * F.mse_loss(z, q.detach())
        q = z + (q - z).detach()  # straight-through estimator
        recon = self.dec(q.permute(0, 3, 1, 2))
        return recon, idx, vq_loss


@torch.no_grad()
def evaluate(model, split, background, dev, n=1024, seed=0):
    i = np.sort(np.random.default_rng(seed).choice(len(split), min(n, len(split)), replace=False))
    x = to_float(split.frames[i], dev)
    with autocast(dev):
        recon = model.decode(model.encode(x)).float().clamp(0, 1)
    moving = foreground_mask(x, background) | foreground_mask(recon, background)
    return {
        "val_psnr_db": float(psnr(recon, x).mean()),
        "val_foreground_psnr_db": masked_psnr(recon, x, moving),
        "codes_used_in_sample": len(np.unique(model.encode(x).cpu().numpy())),
    }


@torch.no_grad()
def save_reconstructions(model, split, path, n=8, scale=4):
    """PNG with the n val frames that moved most (top row) above their reconstructions (bottom row)."""
    frames = np.asarray(split.frames)
    motion = np.abs(np.diff(frames.astype(np.int16), axis=0)).sum((1, 2, 3))
    picks = np.sort(np.argsort(motion)[-n:] + 1)
    real = frames[picks]
    recon = to_uint8(model.decode(model.encode(to_float(real, next(model.parameters()).device))))
    grid = np.concatenate([np.concatenate(list(real), 1), np.concatenate(list(recon), 1)], 0)
    Image.fromarray(grid.repeat(scale, 0).repeat(scale, 1)).save(path)


def train(a):
    seed_all(a.seed)
    dev = device()
    train_split, val_split = Split(a.data, "train"), Split(a.data, "val")
    rng = np.random.default_rng(a.seed)
    cfg = {"n_codes": a.n_codes, "dim": a.dim, "ch": tuple(a.ch)}
    model = Tokenizer(**cfg).to(dev)
    background = static_background(train_split.frames, dev)

    def loss_fn(step):
        x = to_float(train_split.frames[np.sort(rng.choice(len(train_split), a.batch))], dev)
        recon, _, vq_loss = model(x)
        weights = 1 + FOREGROUND_WEIGHT * foreground_mask(x, background)
        l1 = (weights * (recon.float() - x).abs()).sum() / (weights.sum() * x.shape[1])
        return l1 + vq_loss

    train_loop(model, loss_fn, a.steps, a.lr, dev)
    os.makedirs(a.out, exist_ok=True)
    save(os.path.join(a.out, "tokenizer.pt"), model, cfg)
    report = evaluate(model, val_split, background, dev)
    save_reconstructions(model, val_split, os.path.join(a.out, "tokenizer_recon.png"))
    with open(os.path.join(a.out, "tokenizer_eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(report)


def encode(a):
    dev = device()
    model = load(a.ckpt, Tokenizer, dev)
    for name in ("train", "val"):
        split = Split(a.data, name)
        with autocast(dev):
            tokens = [
                model.encode(to_float(split.frames[i : i + 1024], dev)).cpu().numpy()
                for i in range(0, len(split), 1024)
            ]
        np.save(os.path.join(split.dir, "tokens.npy"), np.concatenate(tokens).astype(np.int16))
        print(f"{name}: wrote tokens.npy for {len(split)} frames")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "encode"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--ckpt", default="runs/tokenizer.pt")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--ch", type=int, nargs=3, default=[64, 128, 256])
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a) if a.cmd == "train" else encode(a)


if __name__ == "__main__":
    main()
