import numpy as np
import pytest

from aquarelab.recolor import (
    assign_by_luminance,
    normalize_hex,
    parse_hex_colors,
    recolor,
)


def _palette(luminances):
    """Palette in Lab with the given L values and neutral a, b."""
    return np.array([[L, 0.0, 0.0] for L in luminances])


def test_identical_palette_returns_identical_image():
    rng = np.random.default_rng(0)
    image = rng.random((8, 8, 3))
    weights = rng.dirichlet(np.ones(3), size=(8, 8))
    palette = _palette([20, 50, 80])
    result = recolor(image, weights, palette, palette)
    assert np.allclose(result, image, atol=1e-4)


def test_dirichlet_weights_sum_to_one():
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(3), size=(8, 8))
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_assign_matches_darkest_to_darkest():
    old = _palette([80, 20, 50])
    new = _palette([10, 90, 40])
    matched = assign_by_luminance(old, new)
    assert matched[1, 0] == 10  # darkest old gets darkest new
    assert matched[2, 0] == 40
    assert matched[0, 0] == 90  # brightest old gets brightest new


def test_dead_tones_keep_their_color():
    old = _palette([20, 50, 80])
    new = _palette([10, 40, 90])
    usage = np.array([0.5, 0.0, 0.5])  # middle tone is dead
    matched = assign_by_luminance(old, new, usage=usage)
    assert matched[1, 0] == 50  # unchanged, so zero shift
    assert matched[0, 0] == 10
    assert matched[2, 0] == 90


def test_live_tones_use_full_range_of_larger_palette():
    old = _palette([30, 70])
    new = _palette([5, 15, 25, 35, 45, 55, 65, 95])
    matched = assign_by_luminance(old, new)
    assert matched[0, 0] == 5  # darkest new color
    assert matched[1, 0] == 95  # brightest new color


def test_raises_when_new_palette_is_too_small():
    old = _palette([20, 50, 80])
    new = _palette([10, 90])
    with pytest.raises(ValueError):
        assign_by_luminance(old, new)


def test_raises_when_no_tone_is_alive():
    old = _palette([20, 50])
    new = _palette([10, 90])
    with pytest.raises(ValueError):
        assign_by_luminance(old, new, usage=np.array([0.0, 0.0]))


@pytest.mark.parametrize(
    "color, expected",
    [
        ("#2A9D8F", "#2a9d8f"),
        ("2a9d8f", "#2a9d8f"),
        ("#abc", "#aabbcc"),
        ("rgb(42, 157, 143)", "#2a9d8f"),
        ("rgba(42, 157, 143, 1)", "#2a9d8f"),
    ],
)
def test_normalize_hex_accepts_common_formats(color, expected):
    assert normalize_hex(color) == expected


def test_normalize_hex_rejects_invalid_colors():
    with pytest.raises(ValueError):
        normalize_hex("blue")


def test_parse_hex_colors_finds_colors_in_any_text():
    text = "#264653, 2a9d8f\n#E9C46A and f4a261"
    assert parse_hex_colors(text) == ["#264653", "#2a9d8f", "#e9c46a", "#f4a261"]
