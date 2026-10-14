import torch

from wm.common import foreground_mask, masked_psnr, psnr


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
