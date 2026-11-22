import numpy as np
from PIL import Image

from aquarelab.prepare_data import resize_one


def _save(path, size, grayscale=False):
    shape = (size[1], size[0]) if grayscale else (size[1], size[0], 3)
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)


def test_large_image_gets_short_side_resized(tmp_path):
    _save(tmp_path / "big.jpg", (640, 480))
    dst = tmp_path / "out" / "big.jpg"
    assert resize_one((tmp_path / "big.jpg", dst, 320, 92)) == "ok"
    assert Image.open(dst).size == (427, 320)


def test_small_image_is_not_upscaled(tmp_path):
    _save(tmp_path / "small.jpg", (200, 150))
    dst = tmp_path / "out" / "small.jpg"
    resize_one((tmp_path / "small.jpg", dst, 320, 92))
    assert Image.open(dst).size == (200, 150)


def test_grayscale_stays_grayscale(tmp_path):
    _save(tmp_path / "gray.jpg", (640, 480), grayscale=True)
    dst = tmp_path / "out" / "gray.jpg"
    resize_one((tmp_path / "gray.jpg", dst, 320, 92))
    assert Image.open(dst).mode == "L"


def test_existing_output_is_skipped(tmp_path):
    _save(tmp_path / "a.jpg", (640, 480))
    dst = tmp_path / "out" / "a.jpg"
    resize_one((tmp_path / "a.jpg", dst, 320, 92))
    assert resize_one((tmp_path / "a.jpg", dst, 320, 92)) == "skipped"