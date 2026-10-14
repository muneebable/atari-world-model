"""Action-conditioned world model: a causal transformer over frame tokens interleaved with actions.

A window of F frames becomes one sequence:  [frame_0 tokens] a_0 [frame_1 tokens] a_1 ... [frame_{F-1} tokens]
Action a_t is a token offset by n_codes. The model learns to predict every frame token; action tokens are
inputs only.

  python -m wm.dynamics train --data data --out runs [--actions actions_idm --name dynamics_idm]
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wm.common import device, save, seed_all, train_loop
from wm.data import Split, action_names


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class WorldModel(nn.Module):
    def __init__(self, n_codes, n_actions, tokens_per_frame=64, frames=8, d=256, layers=6, heads=8):
        super().__init__()
        self.n_codes, self.tpf, self.frames = n_codes, tokens_per_frame, frames
        self.tok = nn.Embedding(n_codes + n_actions, d)
        self.pos = nn.Embedding(frames * (tokens_per_frame + 1) - 1, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, n_codes)

    def forward(self, seq):
        """seq: (B, L) token ids, starting at a frame boundary -> logits over frame codes (B, L, n_codes)."""
        x = self.tok(seq) + self.pos(torch.arange(seq.shape[1], device=seq.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))

    def interleave(self, tokens, actions):
        """tokens (B, F, tpf), actions (B, F-1) -> (B, F*tpf + F-1)."""
        a = (actions + self.n_codes).unsqueeze(-1)
        body = torch.cat([tokens[:, :-1], a], -1).flatten(1)
        return torch.cat([body, tokens[:, -1]], 1)

    def loss(self, tokens, actions):
        seq = self.interleave(tokens, actions)
        target = seq[:, 1:].masked_fill(seq[:, 1:] >= self.n_codes, -100)  # don't predict action tokens
        logits = self(seq[:, :-1])
        return F.cross_entropy(logits.flatten(0, 1).float(), target.flatten())

    @torch.no_grad()
    def rollout(self, context, actions, temperature=1.0):
        """Imagine future frames.

        context: (B, C, tpf) tokens of real frames. actions: (B, C-1+H), actions[:, t] takes frame t to t+1.
        Returns (B, H, tpf) generated tokens. Keeps the last frames-1 frames as context (sliding window).
        """
        frames = list(context.unbind(1))
        for _ in range(actions.shape[1] - (len(frames) - 1)):
            n = len(frames)
            keep = frames[-(self.frames - 1) :]
            acts = actions[:, n - len(keep) : n] + self.n_codes
            seq = torch.cat([torch.cat([f, a[:, None]], 1) for f, a in zip(keep, acts.unbind(1))], 1)
            for _ in range(self.tpf):
                logits = self(seq)[:, -1].float()
                if temperature == 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    nxt = torch.multinomial((logits / temperature).softmax(-1), 1)
                seq = torch.cat([seq, nxt], 1)
            frames.append(seq[:, -self.tpf :])
        return torch.stack(frames[context.shape[1] :], 1)


def train(a):
    seed_all(a.seed)
    dev = device()
    split = Split(a.data, "train", actions=a.actions, tokens=True)
    starts = split.starts(a.frames)
    rng = np.random.default_rng(a.seed)
    n_actions = len(action_names(a.data))
    cfg = {
        "n_codes": a.n_codes,
        "n_actions": n_actions,
        "tokens_per_frame": int(np.prod(split.tokens.shape[1:])),
        "frames": a.frames,
        "d": a.d,
        "layers": a.layers,
        "heads": a.heads,
    }
    model = WorldModel(**cfg).to(dev)
    print(f"world model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    def loss_fn(step):
        s = rng.choice(starts, a.batch)
        tokens = torch.as_tensor(
            split.windows(split.tokens, s, a.frames).astype(np.int64), device=dev
        ).flatten(2)
        actions = torch.as_tensor(split.windows(split.actions, s, a.frames)[:, :-1], device=dev)
        return model.loss(tokens, actions)

    train_loop(model, loss_fn, a.steps, a.lr, dev)
    os.makedirs(a.out, exist_ok=True)
    save(os.path.join(a.out, f"{a.name}.pt"), model, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train"])
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="dynamics")
    p.add_argument(
        "--actions", default="actions", help="'actions' (true labels) or 'actions_idm' (IDM labels)"
    )
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-codes", type=int, default=512)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    train(p.parse_args())


if __name__ == "__main__":
    main()
