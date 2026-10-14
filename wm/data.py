import json
import os

import numpy as np


def action_names(root):
    with open(os.path.join(root, "meta.json")) as f:
        return json.load(f)["actions"]


class Split:
    """One split of collected data. `actions` picks the label file (true `actions` or IDM `actions_idm`)."""

    def __init__(self, root, split, actions="actions", tokens=False):
        d = os.path.join(root, split)
        self.dir = d
        self.frames = np.load(f"{d}/frames.npy", mmap_mode="r")
        self.actions = np.load(f"{d}/{actions}.npy")
        self.ep = np.load(f"{d}/ep.npy")
        self.tokens = np.load(f"{d}/tokens.npy", mmap_mode="r") if tokens else None

    def __len__(self):
        return len(self.ep)

    def starts(self, length):
        """Indices i such that frames i .. i+length-1 all belong to one episode."""
        i = np.arange(max(len(self.ep) - length + 1, 0))
        return i[self.ep[i] == self.ep[i + length - 1]]

    def windows(self, array, starts, length):
        return np.stack([array[s : s + length] for s in starts])
