"""
Tile quality filtering: reject >80% background, low variance / blur.
"""
from typing import Tuple

import numpy as np


def tissue_fraction(tile_rgb: np.ndarray, white_threshold: int = 220) -> float:
    """
    Fraction of pixels that are not background (not nearly white).
    Returns value in [0, 1].
    """
    if tile_rgb.ndim != 3 or tile_rgb.shape[2] != 3:
        raise ValueError("Expected RGB tile (H, W, 3)")
    not_white = np.any(tile_rgb < white_threshold, axis=2)
    return float(not_white.sum() / not_white.size)


def tile_variance(tile_rgb: np.ndarray) -> float:
    """Pixel variance (grayscale) for blur/empty detection."""
    if tile_rgb.ndim != 3:
        raise ValueError("Expected RGB tile")
    gray = (0.299 * tile_rgb[..., 0] + 0.587 * tile_rgb[..., 1] + 0.114 * tile_rgb[..., 2])
    return float(np.var(gray))


def passes_quality(
    tile_rgb: np.ndarray,
    min_tissue_fraction: float = 0.2,
    min_variance: float = 200.0,
) -> Tuple[bool, dict]:
    """
    Return (pass: bool, stats: dict with tissue_fraction, variance).
    Reject if >80% background (i.e. tissue_fraction < 0.2) or variance < min_variance.
    """
    tissue = tissue_fraction(tile_rgb)
    var = tile_variance(tile_rgb)
    pass_ = tissue >= min_tissue_fraction and var >= min_variance
    return pass_, {"tissue_fraction": tissue, "variance": var, "img_he_black_ratio": 1.0 - tissue, "img_he_variance": var}
