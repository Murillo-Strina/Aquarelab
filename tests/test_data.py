import numpy as np
import pytest
import torch
from PIL import Image

from aquarelab.data import ImageFolderDataset, find_images, make_loader, prepare_batch


def _save(path, grayscale=False, size=(64, 48)):
    rng = np.random.default_rng(0)
    shape = (size[1], size[0]) if grayscale else (size[1], size[0], 3)
    Image.fromarray(rng.integers(0, 256, size=shape, dtype=np.uint8)).save(path)


@pytest.fixture
def image_folder(tmp_path):
    _save(tmp_path / "a.jpg")
    _save(tmp_path / "b.png")
    _save(tmp_path / "gray.jpg", grayscale=True)
    (tmp_path / "sub").mkdir()
    _save(tmp_path / "sub" / "c.jpg")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"not really a jpeg")
    return tmp_path


def test_find_images_keeps_only_readable_color_images(image_folder):
    names = {p.name for p in find_images(image_folder)}
    assert names == {"a.jpg", "b.png", "c.jpg"}


def test_find_images_respects_max_images(image_folder):
    assert len(find_images(image_folder, max_images=2)) == 2


@pytest.mark.parametrize("train", [True, False])
def test_dataset_returns_square_rgb_in_unit_range(image_folder, train):
    dataset = ImageFolderDataset(image_folder, image_size=32, train=train)
    image = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.min() >= 0 and image.max() <= 1


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError):
        ImageFolderDataset(tmp_path)


def test_train_loader_drops_incomplete_batch(image_folder):
    loader = make_loader(image_folder, batch_size=2, image_size=32, train=True)
    batches = list(loader)
    assert len(batches) == 1  # 3 images, batch of 2: the last one is dropped
    assert batches[0].shape == (2, 3, 32, 32)


def test_prepare_batch_shapes_and_lab_range():
    rgb = torch.rand(2, 3, 16, 16)
    net_input, lab = prepare_batch(rgb)
    assert net_input.shape == lab.shape == (2, 3, 16, 16)
    assert lab[:, 0].min() >= 0 and lab[:, 0].max() <= 100  # L channel


def test_full_scan_writes_an_index_that_is_reused(image_folder):
    first = find_images(image_folder)
    assert (image_folder / ".color_images.txt").exists()
    (image_folder / "gray.jpg").unlink()
    assert find_images(image_folder) == first


def test_partial_scan_does_not_write_an_index(image_folder):
    find_images(image_folder, max_images=2)
    assert not (image_folder / ".color_images.txt").exists()
