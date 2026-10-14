import torch

from wm.evaluate import decode
from wm.tokenizer import Tokenizer


def test_decode_builds_no_autograd_graph():
    # NOTE: a graph here keeps every decoder activation alive; at full eval size it ran a 15 GB T4 out of memory.
    tokenizer = Tokenizer(n_codes=32, ch=(16, 16, 16))
    tokens = torch.zeros(1, 2, 64, dtype=torch.long)
    assert not decode(tokenizer, tokens, (8, 8)).requires_grad
