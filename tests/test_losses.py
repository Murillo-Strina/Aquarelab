import torch

from aquarelab.losses import (
    palette_from_weights,
    reconstruct,
    reconstruction_loss,
    usage_floor_loss,
)


def _two_color_image():
    lab = torch.zeros(1, 3, 4, 4)
    lab[:, :, :, :2] = torch.tensor([30.0, 10.0, -10.0]).view(1, 3, 1, 1)
    lab[:, :, :, 2:] = torch.tensor([80.0, -5.0, 20.0]).view(1, 3, 1, 1)
    weights = torch.zeros(1, 2, 4, 4)
    weights[:, 0, :, :2] = 1.0
    weights[:, 1, :, 2:] = 1.0
    return lab, weights


def test_hard_weights_recover_exact_palette():
    lab, weights = _two_color_image()
    palette = palette_from_weights(lab, weights)
    assert torch.allclose(palette[0, 0], torch.tensor([30.0, 10.0, -10.0]), atol=1e-3)
    assert torch.allclose(palette[0, 1], torch.tensor([80.0, -5.0, 20.0]), atol=1e-3)


def test_hard_weights_reconstruct_perfectly():
    lab, weights = _two_color_image()
    recon = reconstruct(weights, palette_from_weights(lab, weights))
    assert reconstruction_loss(lab, recon) < 1e-4


def test_usage_floor_penalizes_only_dead_tones():
    weights = torch.zeros(1, 3, 10, 10)
    weights[:, 0] = 0.6
    weights[:, 1] = 0.4  # tone 2 is dead
    assert usage_floor_loss(weights, min_usage=0.02) > 0

    weights[:, 1] = 0.39
    weights[:, 2] = 0.01  # still below the floor
    assert usage_floor_loss(weights, min_usage=0.02) > 0

    weights[:, 1] = 0.3
    weights[:, 2] = 0.1  # all above the floor
    assert usage_floor_loss(weights, min_usage=0.02) == 0


def test_usage_floor_is_computed_over_the_batch():
    weights = torch.zeros(2, 2, 4, 4)
    weights[0, 0] = 1.0  # image 0 uses only tone 0
    weights[1, 1] = 1.0  # image 1 uses only tone 1
    assert usage_floor_loss(weights, min_usage=0.02) == 0
