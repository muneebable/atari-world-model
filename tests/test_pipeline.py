"""Tiny end-to-end run of every stage on real Breakout frames (CPU, about a minute)."""

import json
import subprocess
import sys

import numpy as np


def run(*args, cwd):
    subprocess.run([sys.executable, "-m", *args], cwd=cwd, check=True)


def test_pipeline(tmp_path):
    d, r = str(tmp_path / "data"), str(tmp_path / "runs")
    run("wm.collect", "--episodes", "4", "--max-steps", "60", "--out", d, "--val-frac", "0.25", cwd=tmp_path)
    frames, actions, ep = (np.load(f"{d}/train/{n}.npy") for n in ("frames", "actions", "ep"))
    assert frames.shape[1:] == (64, 64, 3) and len(frames) == len(actions) == len(ep)
    assert (actions[np.r_[ep[1:] != ep[:-1], True]] == -1).all()  # -1 exactly on each episode's last frame

    run("wm.idm", "train", "--data", d, "--out", r, "--steps", "3", "--batch", "8", cwd=tmp_path)
    run("wm.idm", "label", "--data", d, "--ckpt", f"{r}/idm.pt", cwd=tmp_path)
    run(
        "wm.tokenizer",
        "train",
        "--data",
        d,
        "--out",
        r,
        "--steps",
        "3",
        "--batch",
        "4",
        "--ch",
        "16",
        "16",
        "16",
        "--n-codes",
        "32",
        cwd=tmp_path,
    )
    run("wm.tokenizer", "encode", "--data", d, "--ckpt", f"{r}/tokenizer.pt", cwd=tmp_path)
    for name, labels in (("dynamics", "actions"), ("dynamics_idm", "actions_idm")):
        run(
            "wm.dynamics",
            "train",
            "--data",
            d,
            "--out",
            r,
            "--name",
            name,
            "--actions",
            labels,
            "--steps",
            "3",
            "--batch",
            "2",
            "--frames",
            "4",
            "--d",
            "32",
            "--layers",
            "1",
            "--heads",
            "2",
            "--n-codes",
            "32",
            cwd=tmp_path,
        )
    run("wm.evaluate", "--data", d, "--runs", r, "--context", "2", "--horizon", "3", "--n", "2", cwd=tmp_path)

    report = json.load(open(f"{r}/dynamics_eval.json"))
    assert len(report["psnr_db_per_step"]["world_model"]) == 3
    assert (tmp_path / "runs" / "dynamics_counterfactual.gif").exists()
