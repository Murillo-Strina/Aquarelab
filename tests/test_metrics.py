import torch

from aquarelab.metrics import tone_ownership, tone_stats


def test_ownership_counts_only_the_strongest_tone():
    weights = torch.zeros(1, 3, 2, 2)
    weights[0, 0, 0, :] = 0.6
    weights[0, 1, 0, :] = 0.4  # top row: tone 0 wins
    weights[0, 1, 1, :] = 0.9
    weights[0, 2, 1, :] = 0.1  # bottom row: tone 1 wins
    owned = tone_ownership(weights)
    assert torch.allclose(owned, torch.tensor([[0.5, 0.5, 0.0]]))


def test_faint_weight_everywhere_owns_nothing():
    weights = torch.full((1, 2, 4, 4), 0.02)
    weights[:, 0] = 0.98  # tone 1 has 2% average weight but never wins
    owned = tone_ownership(weights)
    assert owned[0, 1] == 0


def test_stats_per_image_and_per_batch():
    weights = torch.zeros(2, 3, 2, 2)
    weights[0, 0] = 1.0  # image 0 uses tone 0
    weights[1, 1] = 1.0  # image 1 uses tone 1
    used, unused = tone_stats(weights, threshold=0.01)
    assert used == 1.0
    assert unused == 1  # tone 2 is never used
