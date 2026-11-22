import numpy as np
from skimage import color


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """RGB image in [0, 1] -> Lab. Works on any shape ending in 3 channels."""
    return color.rgb2lab(image_rgb)


def lab_to_rgb(image_lab: np.ndarray) -> np.ndarray:
    """Lab -> RGB, clipped to [0, 1] since shifted colors can fall out of gamut."""
    return np.clip(color.lab2rgb(image_lab), 0.0, 1.0)


def hex_to_lab(hex_colors: list[str]) -> np.ndarray:
    """List of hex colors like '#2a9d8f' -> (N, 3) palette in Lab."""
    rgb = np.array(
        [
            [int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
            for h in hex_colors
        ]
    )
    return color.rgb2lab(rgb[np.newaxis])[0]


def assign_by_luminance(
    old_palette: np.ndarray,
    new_palette: np.ndarray,
    usage: np.ndarray | None = None,
    min_usage: float = 0.01,
) -> np.ndarray:
    """
    Match new colors to the original tones by luminance: darkest to darkest.

    If usage is given, tones below min_usage are treated as dead. Dead tones keep
    their own color (zero shift), and the live tones are matched to new colors
    spread evenly across the new palette's luminance range. This way the darkest
    and brightest new colors are always used.

    old_palette: (K, 3) in Lab
    new_palette: (N, 3) in Lab, N >= number of live tones
    usage:       (K,) share of the image covered by each tone, optional
    Returns (K, 3) aligned with old_palette: matched[i] replaces old_palette[i].
    """
    old_palette = np.asarray(old_palette, dtype=float)
    new_palette = np.asarray(new_palette, dtype=float)

    if usage is None:
        live = np.ones(len(old_palette), dtype=bool)
    else:
        live = np.asarray(usage) >= min_usage

    n_live = int(live.sum())
    if n_live == 0:
        raise ValueError("No live tones to recolor.")
    if len(new_palette) < n_live:
        raise ValueError(
            f"The new palette has {len(new_palette)} colors, "
            f"but the image has {n_live} live tones."
        )

    live_idx = np.flatnonzero(live)
    old_order = live_idx[np.argsort(old_palette[live_idx, 0])]
    new_sorted = new_palette[np.argsort(new_palette[:, 0])]
    picks = np.round(np.linspace(0, len(new_palette) - 1, n_live)).astype(int)

    matched = old_palette.copy()
    matched[old_order] = new_sorted[picks]
    return matched


def recolor(
    image_rgb: np.ndarray,
    weights: np.ndarray,
    old_palette: np.ndarray,
    new_palette: np.ndarray,
) -> np.ndarray:
    """
    Recolor the image by shifting each pixel by the difference between new and old tones.

    image_rgb:   (H, W, 3) RGB, values in [0, 1]
    weights:     (H, W, K) how much each pixel belongs to each tone, sums to 1 per pixel
    old_palette: (K, 3) in Lab
    new_palette: (K, 3) in Lab, already aligned with old_palette
    """
    lab = rgb_to_lab(image_rgb)
    delta = new_palette - old_palette
    shift = weights @ delta
    return lab_to_rgb(lab + shift)
