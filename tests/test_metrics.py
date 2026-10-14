import numpy as np
import torch

from wm.common import foreground_mask, masked_psnr, psnr
from wm.tokenizer import most_moving_frames


def _frame_with_ball():
    background = torch.zeros(3, 64, 64)
    frame = background.clone()
    frame[0, 40:42, 20:22] = 1.0  # a 2x2 red ball
    return background, frame


def test_dropping_the_ball_scores_high_whole_frame_but_low_foreground_psnr():
    background, frame = _frame_with_ball()
    without_ball = background.clone()
    moving = foreground_mask(frame, background) | foreground_mask(without_ball, background)

    assert psnr(without_ball, frame) > 30
    assert masked_psnr(without_ball, frame, moving) < 5


def test_foreground_mask_marks_only_pixels_that_differ_from_background():
    background, frame = _frame_with_ball()
    assert foreground_mask(frame, background).sum() == 4


def test_most_moving_frames_skips_episode_boundaries():
    frames = np.zeros((4, 2, 2, 3), np.uint8)
    frames[1] = 50  # move inside episode 0
    frames[2:] = 255  # reset into episode 1: a bigger jump, but not motion
    ep = np.array([0, 0, 1, 1])
    assert most_moving_frames(frames, ep, 1).tolist() == [1]
