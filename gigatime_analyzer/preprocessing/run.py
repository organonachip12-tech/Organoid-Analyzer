"""
Orchestrate SVS -> GigaTIME-compatible tiles: load, tissue mask, tile, filter, write tiles + JSON + pkl.
"""
import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from . import quality, normalization, tissue_mask, tiling, wsi

# GigaTIME comet_metadata keys expected by prov_data.update_dict_with_key_check
COMMET_METADATA_KEYS = [
    "channel_names",
    "n_channels",
    "pixel_physical_size_xyu",
    "tiling_num_tiles",
    "tiling_patch_size_um",
    "tiling_overlap",
]

COMMON_CHANNEL_LIST = [
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68",
    "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D", "PHH3-B", "Transgelin",
]
NUM_CHANNELS = len(COMMON_CHANNEL_LIST)


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _make_dummy_comet_pkl(tile_size: int, num_channels: int = NUM_CHANNELS) -> dict:
    """Build dummy comet pkl compatible with prov_data.unpack_and_load and HECOMETDataset_roi."""
    comet = np.zeros((tile_size, tile_size, num_channels), dtype=np.uint8)
    packed = np.packbits(comet, axis=-1)
    original_last_dim = num_channels
    original_shape = (tile_size, tile_size, num_channels)
    labels_dapi = np.ones((tile_size, tile_size), dtype=np.uint8)
    labels_dapi_expanded = np.ones((tile_size, tile_size), dtype=np.uint8)
    return {
        "comet_array_binary": packed,
        "original_shape": original_shape,
        "original_last_dim": original_last_dim,
        "labels_dapi": labels_dapi,
        "labels_dapi_expanded": labels_dapi_expanded,
    }


def _write_dummy_pkl(out_path: Path, tile_size: int, num_channels: int) -> None:
    data = _make_dummy_comet_pkl(tile_size, num_channels)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(data, f)


def _pair_name(x_l0: int, y_l0: int, tile_size: int) -> str:
    return f"{x_l0}_{y_l0}_{tile_size}_{tile_size}"


def process_slide(
    svs_path: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    normalize_fn: Optional[callable] = None,
) -> Tuple[int, Path]:
    """
    Process one SVS: detect tissue, extract tiles, filter, write GigaTIME-compatible output.
    Returns (num_tiles_written, slide_output_dir).
    """
    if config is None:
        config = load_config()
    tile_size = int(config.get("tile_size", 556))
    stride = int(config.get("stride", 556))
    magnification = float(config.get("magnification", 20))
    tissue_threshold = float(config.get("tissue_threshold", 0.8))
    min_tissue_frac = 1.0 - tissue_threshold
    min_variance = float(config.get("min_variance", 200))
    num_comet_channels = int(config.get("num_comet_channels", NUM_CHANNELS))
    do_normalize = config.get("normalize_stain", False)
    if normalize_fn is None and do_normalize:
        normalize_fn = lambda t: normalization.normalize_he(t, method="none")

    slide_id = svs_path.stem
    slide_dir = output_dir / f"{slide_id}_and_{slide_id}"
    slide_dir.mkdir(parents=True, exist_ok=True)

    slide = wsi.open_slide(svs_path)
    level0_size = slide.level_dimensions[0]
    level = wsi.get_level_for_magnification(slide, magnification)
    level_size = slide.level_dimensions[level]

    thumb = wsi.get_thumbnail(slide, max_size=2000)
    mask_thumb = tissue_mask.get_tissue_mask(thumb, method="otsu")
    mask_l0 = tissue_mask.mask_to_level0_coords(mask_thumb, (mask_thumb.shape[1], mask_thumb.shape[0]), level0_size)

    locations = tiling.get_tile_locations_level0(
        level0_size,
        level_size,
        level,
        tile_size,
        stride,
        mask_l0,
        min_tissue_fraction=min_tissue_frac,
    )

    img_statistics = {}
    segment_metric = {}
    written = 0

    for (x_l0, y_l0) in tqdm(locations, desc=f"Tiles {slide_id}", leave=False):
        tile = wsi.read_region(slide, (x_l0, y_l0), level, (tile_size, tile_size))
        if normalize_fn is not None:
            tile = normalize_fn(tile)

        pass_quality, stats = quality.passes_quality(
            tile,
            min_tissue_fraction=min_tissue_frac,
            min_variance=min_variance,
        )
        if not pass_quality:
            continue

        pair_name = _pair_name(x_l0, y_l0, tile_size)
        he_path = slide_dir / f"{pair_name}_he.png"
        Image.fromarray(tile).save(he_path)

        pkl_path = slide_dir / f"{pair_name}_comet_binary_thres_labels.pkl.gz"
        _write_dummy_pkl(pkl_path, tile_size, num_comet_channels)

        img_statistics[pair_name] = {
            "img_comet_black_ratio": 0.0,
            "img_comet_variance": 255.0,
            "img_he_black_ratio": min(0.29, 1.0 - float(stats.get("tissue_fraction", 0.5))),
            "img_he_variance": float(stats.get("variance", 300)),
        }
        segment_metric[pair_name] = {"dice": 0.5}

        written += 1

    slide.close()

    with open(slide_dir / "img_statistics.json", "w") as f:
        json.dump(img_statistics, f, indent=0)
    comet_metadata = {
        "channel_names": COMMON_CHANNEL_LIST,
        "n_channels": NUM_CHANNELS,
        "pixel_physical_size_xyu": 0.25,
        "tiling_num_tiles": written,
        "tiling_patch_size_um": 139.0,
        "tiling_overlap": 0,
    }
    with open(slide_dir / "comet_metadata.json", "w") as f:
        json.dump(comet_metadata, f, indent=0)
    with open(slide_dir / "segment_metric.json", "w") as f:
        json.dump(segment_metric, f, indent=0)

    return written, slide_dir


def process_slides(
    input_dir: Path,
    output_dir: Path,
    config: Optional[dict] = None,
    normalize_fn: Optional[callable] = None,
    write_metadata: bool = True,
) -> List[Tuple[str, int, Path]]:
    """Process all SVS under input_dir (recursive). Return list of (slide_id, num_tiles, slide_dir)."""
    if config is None:
        config = load_config()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for svs_path in sorted(input_dir.rglob("*.svs")):
        n, slide_dir = process_slide(svs_path, output_dir, config, normalize_fn)
        results.append((svs_path.stem, n, slide_dir))
    if write_metadata and results:
        write_metadata_csv(output_dir, [r[0] for r in results])
    return results


def write_metadata_csv(
    output_dir: Path,
    slide_ids: List[str],
    csv_path: Optional[Path] = None,
) -> Path:
    """Write metadata CSV for generate_tile_pair_df: tiff_filename, he_filename, slide_deid."""
    import csv
    if csv_path is None:
        csv_path = Path(output_dir) / "preprocessed_metadata.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tiff_filename", "he_filename", "slide_deid"])
        w.writeheader()
        for sid in slide_ids:
            w.writerow({"tiff_filename": sid, "he_filename": sid, "slide_deid": sid})
    return csv_path
