import numpy as np
from sklearn.cluster import KMeans

from aquarelab.recolor import rgb_to_lab


def extract_palette_kmeans(
    image_rgb: np.ndarray,
    k: int = 6,
    sample_size: int = 20_000,
    temperature: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a palette with k-means in Lab space and compute soft per-pixel weights.
    Returns (palette (K, 3) in Lab, weights (H, W, K)).
    """
    lab = rgb_to_lab(image_rgb)
    pixels = lab.reshape(-1, 3)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pixels), size=min(sample_size, len(pixels)), replace=False)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(pixels[idx])
    palette = kmeans.cluster_centers_

    dist = np.linalg.norm(lab[..., None, :] - palette, axis=-1)  # (H, W, K)
    logits = -dist / temperature
    logits -= logits.max(axis=-1, keepdims=True)  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)

    return palette, weights
