import numpy as np
import torch

from wm.data import Split
from wm.dynamics import WorldModel


def test_starts_never_cross_episodes():
    split = Split.__new__(Split)
    split.ep = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    for length in (1, 2, 3, 4):
        for s in split.starts(length):
            assert len(set(split.ep[s : s + length])) == 1
    assert list(split.starts(3)) == [0, 5, 6]


def test_interleave_layout():
    wm = WorldModel(n_codes=10, n_actions=3, tokens_per_frame=2, frames=3, d=16, layers=1, heads=2)
    tokens = torch.tensor([[[1, 2], [3, 4], [5, 6]]])
    actions = torch.tensor([[0, 2]])
    assert wm.interleave(tokens, actions).tolist() == [[1, 2, 10, 3, 4, 12, 5, 6]]


class Recorder(WorldModel):
    """Records each sequence rollout() feeds the model and always predicts code 7."""

    def forward(self, seq):
        self.seen.append(seq.clone())
        logits = torch.full((*seq.shape, self.n_codes), -1e9)
        logits[..., 7] = 0
        return logits


def test_rollout_feeds_training_layout_and_slides_window():
    tpf, frames, n_codes = 2, 3, 10
    wm = Recorder(n_codes=n_codes, n_actions=3, tokens_per_frame=tpf, frames=frames, d=16, layers=1, heads=2)
    wm.seen = []
    context = torch.tensor([[[1, 2], [3, 4]]])
    actions = torch.tensor([[0, 1, 2, 0]])  # 1 context action + 3 imagined steps
    out = wm.rollout(context, actions, temperature=0)

    assert out.tolist() == [[[7, 7], [7, 7], [7, 7]]]
    # First call: the same layout as training, cut just before the frame being predicted.
    full = wm.interleave(torch.tensor([[[1, 2], [3, 4], [0, 0]]]), actions[:, :2])
    assert wm.seen[0].tolist() == full[:, :-tpf].tolist()
    for seq in wm.seen:
        assert seq.shape[1] <= frames * (tpf + 1) - 1  # fits the positional table
        assert seq[0, tpf] >= n_codes  # window starts at a frame boundary
    # Third imagined frame: window slid to [imagined_1, imagined_2] with actions 2 and 0.
    assert wm.seen[2 * tpf].tolist() == [[7, 7, 12, 7, 7, 10]]
