import numpy as np
import pytest
import torch

from aquarelab.inference import (
    analyze,
    export_weights,
    load_model,
    palette_hex,
    recolor_image,
    recolor_tones,
    tone_mask,
    tones_in_use,
)
from aquarelab.model import PaletteNet
from aquarelab.recolor import hex_to_lab, lab_to_hex, rgb_to_lab


@pytest.fixture
def checkpoint(tmp_path):
    """A small untrained model saved in the same format train.py uses."""
    torch.manual_seed(0)
    model = PaletteNet(num_tones=4, pretrained=False)
    path = tmp_path / "step_000010.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": 10,
            "config": {"model": {"num_tones": 4, "backbone": "resnet34"}},
        },
        path,
    )
    return path, model


@pytest.fixture
def analysis(checkpoint):
    model = load_model(checkpoint[0], device="cpu")
    rng = np.random.default_rng(0)
    # Odd size on purpose: not a multiple of 32, like most real photos.
    image = rng.integers(0, 256, size=(70, 100, 3), dtype=np.uint8)
    return analyze(model, image)


def test_export_keeps_the_same_weights(checkpoint, tmp_path):
    path, original = checkpoint
    exported = tmp_path / "model.safetensors"
    export_weights(path, exported)
    loaded = load_model(exported, device="cpu")
    for key, value in original.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_exported_file_is_smaller_than_a_training_checkpoint(checkpoint, tmp_path):
    _, model = checkpoint
    optimizer = torch.optim.Adam(model.parameters())
    # Batch of 2 at 64 px: batch norm needs more than one value per channel in
    # training mode, and the deepest ResNet layer shrinks the image 32 times.
    model(torch.randn(2, 3, 64, 64)).sum().backward()
    optimizer.step()  # creates the optimizer state a real checkpoint carries
    full = tmp_path / "full.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": 10,
            "config": {"model": {"num_tones": 4, "backbone": "resnet34"}},
        },
        full,
    )
    slim = tmp_path / "slim.safetensors"
    export_weights(full, slim)
    assert slim.stat().st_size < full.stat().st_size / 2


def test_analysis_shapes(analysis):
    assert analysis.image.shape == (70, 100, 3)
    assert analysis.image_lab.shape == (70, 100, 3)
    assert analysis.weights.shape == (70, 100, 4)
    assert analysis.palette.shape == (4, 3)
    assert analysis.owned.shape == (4,)
    assert np.allclose(analysis.weights.sum(axis=-1), 1, atol=1e-4)


def test_cached_lab_matches_the_image(analysis):
    assert np.allclose(analysis.image_lab, rgb_to_lab(analysis.image), atol=1e-4)


def test_output_keeps_full_resolution(checkpoint):
    model = load_model(checkpoint[0], device="cpu")
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    analysis = analyze(model, image, max_side=512)
    assert analysis.image.shape == (600, 900, 3)
    assert analysis.weights.shape == (600, 900, 4)
    assert np.allclose(analysis.weights.sum(axis=-1), 1, atol=1e-4)


def test_very_large_images_are_capped(checkpoint):
    model = load_model(checkpoint[0], device="cpu")
    image = np.zeros((600, 900, 3), dtype=np.uint8)
    assert analyze(model, image, output_max_side=700).image.shape == (467, 700, 3)


def test_tones_in_use_are_sorted_dark_to_bright(analysis):
    tones = tones_in_use(analysis, threshold=0.0)
    assert sorted(tones) == [0, 1, 2, 3]
    assert np.all(np.diff(analysis.palette[tones, 0]) >= 0)


def test_recolor_with_its_own_palette_keeps_the_image(analysis):
    own = palette_hex(analysis, threshold=0.0)
    result = recolor_image(analysis, own, threshold=0.0)
    assert np.abs(result - analysis.image).max() < 0.03  # only hex rounding differs


def test_recolor_tones_without_changes_keeps_the_image(analysis):
    assert np.allclose(recolor_tones(analysis, {}), analysis.image, atol=1e-4)


def test_recolor_tones_changes_the_image(analysis):
    result = recolor_tones(analysis, {0: "#ff0000"})
    assert not np.allclose(result, analysis.image, atol=1e-3)


def test_tone_mask_is_the_tone_weights(analysis):
    assert np.array_equal(tone_mask(analysis, 2), analysis.weights[..., 2])


def test_lab_hex_round_trip():
    colors = ["#264653", "#e9c46a", "#ffffff", "#000000"]
    assert [lab_to_hex(c) for c in hex_to_lab(colors)] == colors


def test_keep_lightness_preserves_brightness(analysis):
    to_black = {t: "#000000" for t in range(4)}
    result = recolor_tones(analysis, to_black, keep_lightness=True)
    lightness_change = np.abs(rgb_to_lab(result)[..., 0] - analysis.image_lab[..., 0])
    # Small differences come only from colors clipped back into the RGB gamut.
    assert lightness_change.mean() < 2


def test_full_recolor_changes_brightness(analysis):
    to_black = {t: "#000000" for t in range(4)}
    result = recolor_tones(analysis, to_black, keep_lightness=False)
    lightness_change = np.abs(rgb_to_lab(result)[..., 0] - analysis.image_lab[..., 0])
    assert lightness_change.mean() > 10
