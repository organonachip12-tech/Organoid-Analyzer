"""
Optional H&E stain normalization (Macenko / Reinhard). Stub: pass-through by default.
"""
from pathlib import Path
from typing import Optional

import numpy as np


def normalize_he(tile_rgb: np.ndarray, method: str = "none") -> np.ndarray:
    """
    Apply stain normalization to an RGB tile (H, W, 3). Returns uint8 RGB.
    method: "none" (pass-through), "macenko", "reinhard" (when implemented).
    """
    if method == "none" or method is None:
        return np.asarray(tile_rgb, dtype=np.uint8)
    if method in ("macenko", "reinhard"):
        return np.asarray(tile_rgb, dtype=np.uint8)
    return np.asarray(tile_rgb, dtype=np.uint8)
