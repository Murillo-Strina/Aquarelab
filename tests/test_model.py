import torch

from aquarelab.model import PaletteNet


def test_output_shape_and_sums_to_one():
    model = PaletteNet(num_tones=8, pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        weights = model(x)
    assert weights.shape == (2, 8, 64, 64)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 64, 64), atol=1e-5)


def test_handles_non_square_input():
    model = PaletteNet(num_tones=4, pretrained=False).eval()
    with torch.no_grad():
        weights = model(torch.randn(1, 3, 96, 160))
    assert weights.shape == (1, 4, 96, 160)
