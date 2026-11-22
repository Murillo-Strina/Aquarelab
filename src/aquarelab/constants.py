"""
Lightweight constants shared across modules.

This module must not import torch or kornia: scripts that start many worker
processes on Windows (such as prepare_data) re-import their dependencies in
every worker, and loading torch in each one exhausts virtual memory.
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
GRAYSCALE_MODES = {"1", "L", "LA", "I", "I;16", "F"}
INDEX_NAME = ".color_images.txt"
