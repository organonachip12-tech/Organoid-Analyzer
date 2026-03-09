"""
Tile extraction: slide over the WSI at target level with stride; yield (x, y) in level-0 coords.
"""
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


def get_tile_locations_level0(
    level0_size: Tuple[int, int],
    level_size: Tuple[int, int],
    level: int,
    tile_size: int,
    stride: int,
    tissue_mask_l0: np.ndarray,
    min_tissue_fraction: float = 0.05,
) -> List[Tuple[int, int]]:
    """
    Return list of (x_l0, y_l0) in level-0 coordinates for each tile to extract.
    Only includes positions where the tile overlaps tissue by at least min_tissue_fraction.
    """
    w0, h0 = level0_size[0], level0_size[1]
    w_l, h_l = level_size[0], level_size[1]
    downsample_x = w0 / w_l
    downsample_y = h0 / h_l
    step_x = stride
    step_y = stride
    locations = []
    for y_l in range(0, h_l - tile_size + 1, step_y):
        for x_l in range(0, w_l - tile_size + 1, step_x):
            x_l0 = int(x_l * downsample_x)
            y_l0 = int(y_l * downsample_y)
            tile_l0_w = int(tile_size * downsample_x)
            tile_l0_h = int(tile_size * downsample_y)
            x1 = min(x_l0 + tile_l0_w, w0)
            y1 = min(y_l0 + tile_l0_h, h0)
            if x_l0 >= tissue_mask_l0.shape[1] or y_l0 >= tissue_mask_l0.shape[0]:
                continue
            roi = tissue_mask_l0[y_l0:y1, x_l0:x1]
            if roi.size == 0:
                continue
            tissue_frac = roi.sum() / roi.size
            if tissue_frac >= min_tissue_fraction:
                locations.append((x_l0, y_l0))
    return locations


def iter_tiles(
    slide,
    level: int,
    tile_size: int,
    locations: List[Tuple[int, int]],
    read_region_fn,
) -> Iterator[Tuple[Tuple[int, int], np.ndarray]]:
    """
    Yield ((x_l0, y_l0), tile_rgb) for each location.
    read_region_fn(slide, (x_l0, y_l0), level, (tile_size, tile_size)) -> rgb.
    """
    for (x_l0, y_l0) in locations:
        tile = read_region_fn(slide, (x_l0, y_l0), level, (tile_size, tile_size))
        yield (x_l0, y_l0), tile
