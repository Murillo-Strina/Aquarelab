from pathlib import Path

import kornia
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from aquarelab.constants import GRAYSCALE_MODES, IMAGE_EXTENSIONS

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def find_images(
    root: str | Path, skip_grayscale: bool = True, max_images: int | None = None
) -> list[Path]:
    """
    List image files under root, recursively, sorted so the order is reproducible.

    Grayscale and unreadable files are skipped, since v1 only handles color input.
    Only the file header is read, so scanning a large folder stays fast.
    """
    candidates = sorted(
        p for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    kept = []
    for path in candidates:
        if max_images is not None and len(kept) >= max_images:
            break
        try:
            with Image.open(path) as img:
                if skip_grayscale and img.mode in GRAYSCALE_MODES:
                    continue
        except (UnidentifiedImageError, OSError):
            continue
        kept.append(path)
    return kept


class ImageFolderDataset(Dataset):
    """Color images from a folder, cropped to a square. Returns RGB tensors in [0, 1]."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        train: bool = True,
        skip_grayscale: bool = True,
        max_images: int | None = None,
    ):
        self.paths = find_images(root, skip_grayscale=skip_grayscale, max_images=max_images)
        if not self.paths:
            raise ValueError(f"No color images found in {root}")

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))  # (3, H, W) in [0, 1]


def make_loader(
    root: str | Path,
    batch_size: int = 16,
    image_size: int = 256,
    train: bool = True,
    num_workers: int = 0,
    max_images: int | None = None,
    seed: int = 0,
) -> DataLoader:
    """
    Build a DataLoader over a folder of images.

    In training, the last incomplete batch is dropped: the usage floor is averaged
    over the batch, and a batch of one or two images would bring back the
    per-image behavior.
    """
    dataset = ImageFolderDataset(
        root, image_size=image_size, train=train, max_images=max_images
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )


def prepare_batch(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Turn an RGB batch in [0, 1] into what training needs.

    Returns (net_input, lab): the ImageNet-normalized input for the encoder,
    and the same images in Lab for the losses.
    """
    net_input = (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)
    lab = kornia.color.rgb_to_lab(rgb)
    return net_input, lab