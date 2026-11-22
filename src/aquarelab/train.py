"""
Train PaletteNet on a folder of images.

Run from the project root:

    uv run python -m aquarelab.train --config configs/base.toml
    uv run python -m aquarelab.train --config configs/base.toml \
        --set loss.entropy_weight=0.05 --set run.name=entropy_005

Metrics and preview images go to runs/<name>_<timestamp>/ (TensorBoard),
checkpoints and the final config go to checkpoints/<name>_<timestamp>/.
"""

import argparse
import json
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from aquarelab.data import find_images, make_loader, prepare_batch
from aquarelab.losses import (
    entropy_loss,
    palette_from_weights,
    reconstruct,
    reconstruction_loss,
    usage_floor_loss,
)
from aquarelab.metrics import tone_ownership, tone_stats
from aquarelab.model import PaletteNet
from aquarelab.recolor import assign_by_luminance, hex_to_lab, lab_to_rgb, recolor


def load_config(path: str | Path, overrides: list[str]) -> dict:
    """
    Read a TOML config and apply overrides written as "section.key=value".
    Values are parsed as TOML (numbers, booleans, arrays); anything else stays a string.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)

    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, name = key.partition(".")
        if section not in config or name not in config[section]:
            raise KeyError(f"Unknown config key: {key}")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw  # plain strings do not need quotes on the command line
        config[section][name] = value
    return config


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infinite(loader):
    """Yield batches forever, reshuffling and re-augmenting at every pass."""
    while True:
        yield from loader


def predict(model: PaletteNet, net_input: torch.Tensor, use_amp: bool) -> torch.Tensor:
    """Run the model, in bfloat16 when use_amp is on. Memberships come back in float32."""
    with torch.autocast(device_type=net_input.device.type, dtype=torch.bfloat16, enabled=use_amp):
        weights = model(net_input)
    return weights.float()


def compute_losses(lab: torch.Tensor, weights: torch.Tensor, loss_cfg: dict):
    palette = palette_from_weights(lab, weights)
    recon = reconstruct(weights, palette)
    parts = {
        "recon": reconstruction_loss(lab, recon),
        "entropy": entropy_loss(weights),
        "floor": usage_floor_loss(weights, min_usage=loss_cfg["min_usage"]),
    }
    total = (
        parts["recon"]
        + loss_cfg["entropy_weight"] * parts["entropy"]
        + loss_cfg["floor_weight"] * parts["floor"]
    )
    return total, parts


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, threshold: float) -> dict:
    """Reconstruction error and tone usage on images never used in training."""
    model.eval()
    l1_sum, count, owned_all = 0.0, 0, []
    for rgb in loader:
        net_input, lab = prepare_batch(rgb.to(device, non_blocking=True))
        weights = predict(model, net_input, use_amp)
        recon = reconstruct(weights, palette_from_weights(lab, weights))
        l1_sum += reconstruction_loss(lab, recon).item() * len(rgb)
        count += len(rgb)
        owned_all.append(tone_ownership(weights).cpu())
    model.train()

    owned = torch.cat(owned_all)
    return {
        "recon_l1": l1_sum / count,
        "tones_per_image": (owned >= threshold).sum(dim=1).float().mean().item(),
        "unused_tones": int((owned.mean(dim=0) < threshold).sum()),
    }


def load_preview_images(paths: list[Path], image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(image_size, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    return torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])


@torch.no_grad()
def render_previews(model, rgb, device, use_amp, threshold, new_palette) -> torch.Tensor:
    """One row per image: original | reconstruction | recolored. Returns (3, H*N, W*3)."""
    model.eval()
    net_input, lab = prepare_batch(rgb.to(device))
    weights = predict(model, net_input, use_amp)
    palette = palette_from_weights(lab, weights)
    recon = reconstruct(weights, palette)
    owned = tone_ownership(weights).cpu().numpy()
    model.train()

    rows = []
    for i in range(len(rgb)):
        image = rgb[i].permute(1, 2, 0).numpy()
        weights_np = weights[i].permute(1, 2, 0).cpu().numpy()
        palette_np = palette[i].cpu().numpy()
        matched = assign_by_luminance(
            palette_np, new_palette, usage=owned[i], min_usage=threshold
        )
        recon_rgb = lab_to_rgb(recon[i].permute(1, 2, 0).cpu().numpy())
        recolored = recolor(image, weights_np, palette_np, matched)
        rows.append(np.concatenate([image, recon_rgb, recolored], axis=1))

    grid = np.concatenate(rows, axis=0)
    return torch.from_numpy(grid).permute(2, 0, 1).float()


def save_checkpoint(path: Path, model, optimizer, scheduler, step: int, config: dict):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train PaletteNet.")
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Override a config value. Can be repeated.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run_cfg, data_cfg, model_cfg = cfg["run"], cfg["data"], cfg["model"]
    loss_cfg, optim_cfg, log_cfg = cfg["loss"], cfg["optim"], cfg["log"]

    # Fail fast: a palette with fewer colors than tones would only break at the
    # first evaluation, after a thousand training steps.
    palette_hex = log_cfg["preview_palette"]
    if len(palette_hex) < model_cfg["num_tones"]:
        raise ValueError(
            f"preview_palette has {len(palette_hex)} colors, but the model uses "
            f"{model_cfg['num_tones']} tones. Provide at least as many colors as tones."
        )
    new_palette = hex_to_lab(palette_hex)

    run_name = f"{run_cfg['name']}_{time.strftime('%Y%m%d-%H%M%S')}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    writer = SummaryWriter(Path("runs") / run_name)
    writer.add_text("config", f"```\n{json.dumps(cfg, indent=2)}\n```")

    device = resolve_device(run_cfg["device"])
    use_amp = optim_cfg["amp"] and device.type == "cuda"
    torch.manual_seed(run_cfg["seed"])
    torch.backends.cudnn.benchmark = True
    threshold = log_cfg["dead_threshold"]

    print(f"Run: {run_name} | device: {device} | mixed precision: {use_amp}")
    print("Scanning images...")
    train_loader = make_loader(
        data_cfg["train_root"],
        batch_size=data_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        train=True,
        num_workers=data_cfg["num_workers"],
        max_images=data_cfg["max_train_images"] or None,
        seed=run_cfg["seed"],
    )
    val_loader = make_loader(
        data_cfg["val_root"],
        batch_size=data_cfg["batch_size"],
        image_size=data_cfg["image_size"],
        train=False,
        num_workers=data_cfg["num_workers"],
        max_images=data_cfg["val_images"],
    )
    preview_paths = find_images(data_cfg["val_root"], max_images=log_cfg["preview_images"])
    preview_rgb = load_preview_images(
        preview_paths + [Path(log_cfg["test_image"])], data_cfg["image_size"]
    )
    print(
        f"Train images: {len(train_loader.dataset)} | "
        f"val images: {len(val_loader.dataset)}"
    )

    model = PaletteNet(num_tones=model_cfg["num_tones"], backbone=model_cfg["backbone"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=optim_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optim_cfg["steps"])

    batches = infinite(train_loader)
    model.train()
    interval_start = time.perf_counter()

    for step in range(1, optim_cfg["steps"] + 1):
        rgb = next(batches)
        net_input, lab = prepare_batch(rgb.to(device, non_blocking=True))
        weights = predict(model, net_input, use_amp)
        loss, parts = compute_losses(lab, weights, loss_cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % log_cfg["log_every"] == 0:
            used, unused = tone_stats(weights.detach(), threshold)
            seconds_per_step = (time.perf_counter() - interval_start) / log_cfg["log_every"]
            writer.add_scalar("train/total", loss.item(), step)
            for name, value in parts.items():
                writer.add_scalar(f"train/{name}", value.item(), step)
            writer.add_scalar("train/tones_per_image", used, step)
            writer.add_scalar("train/unused_in_batch", unused, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            writer.add_scalar("speed/seconds_per_step", seconds_per_step, step)
            print(
                f"step {step:6d} | recon {parts['recon'].item():.4f} "
                f"| tones per image {used:.1f} | unused in batch {unused} "
                f"| {seconds_per_step:.3f}s/step"
            )
            interval_start = time.perf_counter()

        if step % log_cfg["eval_every"] == 0 or step == optim_cfg["steps"]:
            metrics = evaluate(model, val_loader, device, use_amp, threshold)
            for name, value in metrics.items():
                writer.add_scalar(f"val/{name}", value, step)
            writer.add_image(
                "previews",
                render_previews(model, preview_rgb, device, use_amp, threshold, new_palette),
                step,
            )
            print(
                f"  val | recon L1 {metrics['recon_l1']:.4f} "
                f"| tones per image {metrics['tones_per_image']:.1f} "
                f"| unused tones {metrics['unused_tones']}"
            )
            interval_start = time.perf_counter()

        if step % log_cfg["save_every"] == 0 or step == optim_cfg["steps"]:
            save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, scheduler, step, cfg)
            interval_start = time.perf_counter()

    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    writer.close()
    print(f"Done. Checkpoints in {ckpt_dir}")


if __name__ == "__main__":
    main()