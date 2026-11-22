"""
Load a trained PaletteNet and use it on single images.

Analyze a photo once, then recolor it as many times as needed: recoloring only
uses the stored tone maps, so it does not run the network again.

Export slim weights from a training checkpoint, run from the project root:

    uv run python -m aquarelab.inference checkpoints/<run>/step_010000.pt models/palettenet.safetensors
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from aquarelab.data import prepare_batch
from aquarelab.losses import palette_from_weights
from aquarelab.metrics import tone_ownership
from aquarelab.model import PaletteNet
from aquarelab.recolor import (
    assign_by_luminance,
    hex_to_lab,
    lab_to_hex,
    lab_to_rgb,
    rgb_to_lab,
)

DEFAULT_THRESHOLD = 0.01  # a tone owning less than 1% of the pixels is not shown


@dataclass
class ToneAnalysis:
    """Everything needed to recolor one image without running the network again."""

    image: np.ndarray  # (H, W, 3) RGB in [0, 1], at output resolution
    image_lab: np.ndarray  # (H, W, 3) the same image in Lab, cached for fast recoloring
    weights: (
        np.ndarray
    )  # (H, W, K) tone memberships at output resolution, sum to 1 per pixel
    palette: np.ndarray  # (K, 3) tone colors in Lab
    owned: np.ndarray  # (K,) share of pixels where each tone is the strongest


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def export_weights(checkpoint_path: str | Path, output_path: str | Path) -> None:
    """
    Save only the model weights from a training checkpoint, as safetensors.

    Training checkpoints also hold the optimizer state, which roughly triples
    their size and is useless for inference. Safetensors files cannot run code
    when loaded, unlike pickled .pt files, so they are safe to publish.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_cfg = state["config"]["model"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state["model"],
        output_path,
        metadata={
            "num_tones": str(model_cfg["num_tones"]),
            "backbone": model_cfg["backbone"],
            "step": str(state["step"]),
        },
    )


def load_model(path: str | Path, device: str = "auto") -> PaletteNet:
    """Load a model from exported safetensors weights or from a training checkpoint."""
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        state_dict = load_file(path)
        num_tones, backbone = int(metadata["num_tones"]), metadata["backbone"]
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state["model"]
        num_tones = state["config"]["model"]["num_tones"]
        backbone = state["config"]["model"]["backbone"]

    # Pretrained backbone weights are not downloaded: the file already has every weight.
    model = PaletteNet(num_tones=num_tones, backbone=backbone, pretrained=False)
    model.load_state_dict(state_dict)
    return model.to(resolve_device(device)).eval()


def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = (np.clip(array, 0, 1) * 255).round().astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def _resize_to_fit(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longer side is at most max_side, keeping the aspect ratio."""
    scale = max_side / max(image.size)
    if scale >= 1:
        return image
    new_size = (round(image.width * scale), round(image.height * scale))
    return image.resize(new_size, Image.Resampling.LANCZOS)


@torch.no_grad()
def analyze(
    model: PaletteNet,
    image: Image.Image | np.ndarray,
    max_side: int = 512,
    output_max_side: int = 2048,
) -> ToneAnalysis:
    """
    Run the network once on an image and keep everything recoloring needs.

    The network runs on a copy downscaled to max_side, which keeps analysis
    fast. Tone maps are then upscaled back, so recoloring happens at full
    resolution, capped at output_max_side to keep every recolor responsive.
    Tone maps are smooth, so upscaling them does not create jagged edges.
    """
    full = _resize_to_fit(_to_pil(image), output_max_side)
    small = _resize_to_fit(full, max_side)

    rgb = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255)
    rgb = rgb.permute(2, 0, 1)[None]
    device = next(model.parameters()).device
    net_input, lab = prepare_batch(rgb.to(device))
    weights = model(net_input)
    palette = palette_from_weights(lab, weights)
    owned = tone_ownership(weights)

    if small.size != full.size:
        weights = F.interpolate(
            weights,
            size=(full.height, full.width),
            mode="bilinear",
            align_corners=False,
        )
        weights = weights / weights.sum(dim=1, keepdim=True)

    image_rgb = np.asarray(full, dtype=np.float32) / 255
    return ToneAnalysis(
        image=image_rgb,
        image_lab=rgb_to_lab(image_rgb),
        weights=weights[0].permute(1, 2, 0).cpu().numpy(),
        palette=palette[0].cpu().numpy(),
        owned=owned[0].cpu().numpy(),
    )


def tones_in_use(
    analysis: ToneAnalysis, threshold: float = DEFAULT_THRESHOLD
) -> list[int]:
    """Indices of the tones that own at least `threshold` of the pixels, darkest first."""
    live = np.flatnonzero(analysis.owned >= threshold)
    return [int(t) for t in live[np.argsort(analysis.palette[live, 0])]]


def palette_hex(
    analysis: ToneAnalysis, threshold: float = DEFAULT_THRESHOLD
) -> list[str]:
    """Colors of the tones in use, darkest first, as hex strings."""
    return [lab_to_hex(analysis.palette[t]) for t in tones_in_use(analysis, threshold)]


def tone_mask(analysis: ToneAnalysis, tone: int) -> np.ndarray:
    """How much each pixel belongs to one tone, (H, W) in [0, 1]. Used to show where a color is."""
    return analysis.weights[..., tone]


def _apply_palette(
    analysis: ToneAnalysis, new_palette: np.ndarray, keep_lightness: bool = True
) -> np.ndarray:
    """
    Lab shift with the cached Lab image, so each recolor skips the RGB to Lab step.

    By default only the a and b channels shift: hue and saturation change, while
    the brightness of every pixel stays as in the original. Dark scenes stay
    dark, and transitions between tones do not gain bright or dark rings. Set
    keep_lightness=False to also shift brightness toward the new colors.
    """
    delta = new_palette - analysis.palette
    if keep_lightness:
        delta[:, 0] = 0
    return lab_to_rgb(analysis.image_lab + analysis.weights @ delta)


def recolor_image(
    analysis: ToneAnalysis,
    new_colors: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    keep_lightness: bool = True,
) -> np.ndarray:
    """
    Recolor with a whole new palette, matched to the original tones by luminance.
    The palette needs at least as many colors as there are tones in use.
    """
    matched = assign_by_luminance(
        analysis.palette,
        hex_to_lab(new_colors),
        usage=analysis.owned,
        min_usage=threshold,
    )
    return _apply_palette(analysis, matched, keep_lightness)


def recolor_tones(
    analysis: ToneAnalysis, replacements: dict[int, str], keep_lightness: bool = True
) -> np.ndarray:
    """Recolor only the chosen tones, given as {tone index: new hex color}."""
    new_palette = analysis.palette.copy()
    for tone, color in replacements.items():
        new_palette[tone] = hex_to_lab([color])[0]
    return _apply_palette(analysis, new_palette, keep_lightness)


def main():
    parser = argparse.ArgumentParser(
        description="Export slim PaletteNet weights from a training checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Training checkpoint (.pt).")
    parser.add_argument("output", type=Path, help="Output file (.safetensors).")
    args = parser.parse_args()

    export_weights(args.checkpoint, args.output)
    print(f"Saved {args.output} ({args.output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
