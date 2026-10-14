"""Inverse dynamics model: predict the action a_t from the frames around the transition t -> t+1.

python -m wm.idm train --data data --out runs
python -m wm.idm label --data data --ckpt runs/idm.pt   # writes actions_idm.npy for each split
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.common import device, load, save, seed_all, to_float, train_loop
from wm.data import Split, action_names

CTX = (-1, 0, 1, 2)  # frame offsets fed to the IDM; the label is the action at offset 0


class IDM(nn.Module):
    def __init__(self, n_actions, ch=32):
        super().__init__()
        layers, c = [], 3 * len(CTX)
        for out in (ch, 2 * ch, 4 * ch, 4 * ch):
            layers += [nn.Conv2d(c, out, 4, 2, 1), nn.GroupNorm(8, out), nn.SiLU()]
            c = out
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, n_actions))

    def forward(self, x):
        """x: (B, len(CTX), 3, H, W) in [0, 1] -> logits (B, n_actions)."""
        return self.net(x.flatten(1, 2))


@torch.no_grad()
def predict(model, frames, dev, batch=512):
    """frames: uint8 (B, len(CTX), H, W, 3) -> predicted actions (B,)."""
    out = [model(to_float(frames[i : i + batch], dev)).argmax(-1) for i in range(0, len(frames), batch)]
    return torch.cat(out).cpu().numpy()


def evaluate(model, split, names, dev, max_windows=20000):
    starts = split.starts(len(CTX))[:max_windows]
    target = split.actions[starts - CTX[0]]
    pred = predict(model, split.windows(split.frames, starts, len(CTX)), dev)
    n = len(names)
    confusion = np.zeros((n, n), dtype=int)
    np.add.at(confusion, (target, pred), 1)
    return {
        "accuracy": float((pred == target).mean()),
        "per_action_recall": {
            names[i]: float(confusion[i, i] / max(confusion[i].sum(), 1)) for i in range(n)
        },
        "confusion_rows_true_cols_pred": confusion.tolist(),
    }


def train(a):
    seed_all(a.seed)
    dev = device()
    names = action_names(a.data)
    train_split, val_split = Split(a.data, "train"), Split(a.data, "val")
    starts = train_split.starts(len(CTX))
    rng = np.random.default_rng(a.seed)
    cfg = {"n_actions": len(names), "ch": a.ch}
    model = IDM(**cfg).to(dev)

    def loss_fn(step):
        s = rng.choice(starts, a.batch)
        x = to_float(train_split.windows(train_split.frames, s, len(CTX)), dev)
        y = torch.as_tensor(train_split.actions[s - CTX[0]], device=dev)
        return F.cross_entropy(model(x), y)

    train_loop(model, loss_fn, a.steps, a.lr, dev)
    os.makedirs(a.out, exist_ok=True)
    save(os.path.join(a.out, "idm.pt"), model, cfg)
    report = evaluate(model, val_split, names, dev)
    with open(os.path.join(a.out, "idm_eval.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "confusion_rows_true_cols_pred"}, indent=2))


def label(a):
    """Replace the true action labels with IDM predictions, VPT-style, for the pseudo-label experiment."""
    dev = device()
    model = load(a.ckpt, IDM, dev)
    for name in ("train", "val"):
        split = Split(a.data, name)
        t = np.arange(len(split))
        # Clamp the context to the episode so frames at its edges still get a label.
        first = np.searchsorted(split.ep, split.ep, side="left")
        last = np.searchsorted(split.ep, split.ep, side="right") - 1
        idx = np.clip(t[:, None] + np.array(CTX), first[:, None], last[:, None])
        shape = split.frames.shape[1:]
        labels = np.concatenate(
            [
                predict(model, split.frames[c.ravel()].reshape(len(c), len(CTX), *shape), dev)
                for c in np.array_split(idx, max(1, len(idx) // 4096))
            ]
        )
        labels[split.actions == -1] = -1
        np.save(os.path.join(split.dir, "actions_idm.npy"), labels)
        agree = (labels == split.actions)[split.actions != -1].mean()
        print(f"{name}: wrote actions_idm.npy, agreement with true labels {agree:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "label"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--ckpt", default="runs/idm.pt")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ch", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    train(a) if a.cmd == "train" else label(a)


if __name__ == "__main__":
    main()
