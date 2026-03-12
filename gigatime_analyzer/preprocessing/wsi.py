"""
WSI loading via OpenSlide. Open SVS, read regions at target magnification, get thumbnails.
"""
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    import openslide
except ImportError:
    openslide = None  # type: ignore


def open_slide(path: Path) -> "openslide.OpenSlide":
    if openslide is None:
        raise ImportError("openslide-python is required. Install with: pip install openslide-python")
    return openslide.OpenSlide(str(path))


def get_level_for_magnification(slide: "openslide.OpenSlide", target_mag: float) -> int:
    """
    Return pyramid level index closest to target_mag (e.g. 20).
    Uses openslide.mpp-x if available, else assumes level 0 is 40x and doubles per level.
    """
    mpp_x = slide.properties.get("openslide.mpp-x")
    if mpp_x is not None:
        mpp_x = float(mpp_x)
        target_mpp = 0.25 * (40.0 / target_mag)
        level_dims = slide.level_dimensions
        best_level = 0
        best_diff = float("inf")
        for level, (w, h) in enumerate(level_dims):
            level_mpp = mpp_x * (2**level) if level > 0 else mpp_x
            diff = abs(level_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = level
        return best_level
    if target_mag >= 40:
        return 0
    if target_mag >= 20:
        return 1 if len(slide.level_dimensions) > 1 else 0
    return min(2, len(slide.level_dimensions) - 1) if len(slide.level_dimensions) > 2 else 0


def get_thumbnail(slide: "openslide.OpenSlide", max_size: int = 2000) -> np.ndarray:
    """Return RGB thumbnail (numpy array HWC) with longest side <= max_size."""
    thumb = slide.get_thumbnail((max_size, max_size))
    return np.array(thumb.convert("RGB"))


def read_region(
    slide: "openslide.OpenSlide",
    location: Tuple[int, int],
    level: int,
    size: Tuple[int, int],
) -> np.ndarray:
    """
    Read a region as RGB numpy array (H, W, 3).
    location: (x, y) in level-0 coordinates.
    size: (width, height) in pixels at the given level.
    """
    region = slide.read_region(location, level, size)
    return np.array(region.convert("RGB"))


def level0_size_from_level(slide: "openslide.OpenSlide", level: int, size: Tuple[int, int]) -> Tuple[int, int]:
    """Convert pixel size at given level to level-0 pixel size (for stride/overlap in level-0)."""
    w, h = size
    dims = slide.level_dimensions
    if level >= len(dims):
        level = len(dims) - 1
    l0_w, l0_h = dims[0]
    l_w, l_h = dims[level]
    scale_x = l0_w / l_w
    scale_y = l0_h / l_h
    return (int(w * scale_x), int(h * scale_y))


def get_slide_info(slide: "openslide.OpenSlide") -> dict:
    """Return basic info: dimensions per level, mpp if present."""
    dims = slide.level_dimensions
    mpp = slide.properties.get("openslide.mpp-x")
    return {
        "level_dimensions": dims,
        "mpp_x": float(mpp) if mpp is not None else None,
        "n_levels": len(dims),
    }
