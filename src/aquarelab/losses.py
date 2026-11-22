import torch
import torch.nn.functional as F

LAB_SCALE = torch.tensor([1 / 100, 1 / 128, 1 / 128]).view(1, 3, 1, 1)


def palette_from_weights(
    lab: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Each tone's color is the weighted mean of the pixels that belong to it.

    lab:     (B, 3, H, W) image in Lab
    weights: (B, K, H, W) tone memberships, sum to 1 per pixel
    Returns (B, K, 3) palette in Lab.
    """
    mass = weights.sum(dim=(2, 3)) + eps
    return torch.einsum("bkhw,bchw->bkc", weights, lab) / mass[..., None]


def reconstruct(weights: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Rebuild the image from palette and memberships. Returns (B, 3, H, W) in Lab."""
    return torch.einsum("bkhw,bkc->bchw", weights, palette)


def reconstruction_loss(lab: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """L1 distance in Lab, with each channel scaled to a similar range."""
    scale = LAB_SCALE.to(lab.device)
    return F.l1_loss(lab * scale, recon * scale)


def entropy_loss(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Low entropy pushes each pixel to commit to fewer tones."""
    return -(weights * (weights + eps).log()).sum(dim=1).mean()


def usage_floor_loss(weights: torch.Tensor, min_usage: float = 0.02) -> torch.Tensor:
    """
    Penalize tones whose share of the batch falls below min_usage, so no tone dies.

    Usage is averaged over the whole batch, not per image: a tone may be absent
    from a simple image as long as other images in the batch use it. Tones above
    the floor are free to be as large as they need.
    """
    usage = weights.mean(dim=(0, 2, 3))
    return F.relu(min_usage - usage).sum()
