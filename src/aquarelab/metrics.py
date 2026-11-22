import torch
import torch.nn.functional as F


def tone_ownership(weights: torch.Tensor) -> torch.Tensor:
    """
    Share of pixels where each tone is the strongest.

    weights: (B, K, H, W) tone memberships
    Returns (B, K). Unlike the average weight, this cannot be faked by spreading
    a faint weight everywhere: a tone only counts where it actually wins.
    """
    owner = weights.argmax(dim=1)  # (B, H, W)
    return F.one_hot(owner, weights.shape[1]).float().mean(dim=(1, 2))


def tone_stats(weights: torch.Tensor, threshold: float = 0.01) -> tuple[float, int]:
    """
    Returns (tones used per image on average, tones unused across the whole batch).
    A tone is used when it owns at least `threshold` of the pixels.
    """
    owned = tone_ownership(weights)
    used_per_image = (owned >= threshold).sum(dim=1).float().mean().item()
    unused_in_batch = int((owned.mean(dim=0) < threshold).sum())
    return used_per_image, unused_in_batch
