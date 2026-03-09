"""
Tissue detection: downsample slide -> grayscale -> threshold -> binary mask.
"""
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    from skimage.filters import threshold_otsu
except ImportError:
    threshold_otsu = None


def get_tissue_mask(
    thumbnail_rgb: np.ndarray,
    method: str = "otsu",
    quantile: float = 0.85,
) -> np.ndarray:
    """
    Compute binary tissue mask from a downsampled RGB image (H, W, 3).
    Returns (H, W) bool array: True = tissue.
    """
    gray = rgb_to_grayscale(thumbnail_rgb)
    if method == "otsu" and threshold_otsu is not None:
        thresh = threshold_otsu(gray)
        mask = gray < thresh
    else:
        thresh = np.quantile(gray, quantile)
        mask = gray < thresh
    return mask.astype(np.uint8)


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale (luminance)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB image (H, W, 3)")
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float64)


def mask_to_level0_coords(
    mask: np.ndarray,
    mask_size: Tuple[int, int],
    level0_size: Tuple[int, int],
) -> np.ndarray:
    """
    Map binary mask (H, W) at thumbnail resolution to level-0 pixel grid.
    level0_size = (width, height) of level 0 in pixels.
    Returns a boolean array of shape (level0_height, level0_width) with True where tissue.
    """
    w0, h0 = level0_size[0], level0_size[1]
    mh, mw = mask.shape[0], mask.shape[1]
    scale_x = mw / w0
    scale_y = mh / h0
    yy = (np.arange(h0) * scale_y).astype(np.int32)
    xx = (np.arange(w0) * scale_x).astype(np.int32)
    yy = np.clip(yy, 0, mh - 1)
    xx = np.clip(xx, 0, mw - 1)
    return mask[yy, :][:, xx]
