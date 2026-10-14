import contextlib
import random

import numpy as np
import torch


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast(dev):
    # fp16 on GPU: Colab T4 and Kaggle P100 have no bf16. Plain fp32 on CPU.
    if dev == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_float(frames, dev):
    """uint8 (..., H, W, 3) -> float (..., 3, H, W) in [0, 1]."""
    x = torch.as_tensor(np.asarray(frames), device=dev)
    return x.movedim(-1, -3).float() / 255.0


def to_uint8(x):
    """float (..., 3, H, W) in [0, 1] -> uint8 numpy (..., H, W, 3)."""
    return (x.clamp(0, 1) * 255).round().byte().movedim(-3, -1).cpu().numpy()


def psnr(a, b):
    """Per-image PSNR in dB for float images in [0, 1], reduced over (C, H, W)."""
    mse = ((a - b) ** 2).flatten(-3).mean(-1)
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


# NOTE: per-channel difference (in [0, 1]) from the static background that counts as a moving object.
FOREGROUND_THRESHOLD = 0.1


def static_background(frames, dev, n=2000, seed=0):
    """Median of a random sample of uint8 frames: the parts of the scene that rarely change. (3, H, W)."""
    i = np.sort(np.random.default_rng(seed).choice(len(frames), min(n, len(frames)), replace=False))
    return to_float(frames[i], dev).median(0).values


def foreground_mask(x, background):
    """(..., 1, H, W) bool: pixels that differ from the background, e.g. the ball, paddle and broken bricks."""
    return ((x - background).abs() > FOREGROUND_THRESHOLD).any(-3, keepdim=True)


def masked_psnr(a, b, mask):
    """PSNR in dB over the masked pixels only, pooled over everything. Whole-frame PSNR is dominated by the
    static background, so a model that drops every moving object still scores well on it."""
    mse = (((a - b) ** 2) * mask).sum() / (mask.sum() * a.shape[-3]).clamp_min(1)
    return float(10 * torch.log10(1.0 / mse.clamp_min(1e-10)))


def save(path, model, cfg):
    torch.save({"cfg": cfg, "state": model.state_dict()}, path)


def load(path, cls, dev):
    ck = torch.load(path, map_location=dev)
    model = cls(**ck["cfg"]).to(dev)
    model.load_state_dict(ck["state"])
    return model.eval()


def train_loop(model, loss_fn, steps, lr, dev, log_every=200, warmup=500):
    """AdamW with linear warmup then cosine decay. loss_fn(step) returns the loss of one batch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * s / steps))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    model.train()
    for step in range(steps):
        with autocast(dev):
            loss = loss_fn(step)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % log_every == 0 or step == steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)
    model.eval()
