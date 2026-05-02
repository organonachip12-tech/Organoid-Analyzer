"""
GigaTIME Analyzer configuration — paths and default hyperparameters.

When ``SCRATCH`` is set (typical on HPC), preprocessed tiles default to
``$SCRATCH/gigatime_work/preprocessed_tiles`` so large tile caches do not fill ``$HOME``.
Override with ``GIGATIME_TILES_DIR``.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gigatime")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
SVS_DIR = os.path.join(DATA_DIR, "svs")


def _resolve_tiles_dir() -> str:
    override = os.environ.get("GIGATIME_TILES_DIR")
    if override:
        return override
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return os.path.join(scratch, "gigatime_work", "preprocessed_tiles")
    return os.path.join(DATA_DIR, "preprocessed_tiles")


TILES_DIR = _resolve_tiles_dir()

DEFAULT_ARCH = "gigatime"
NUM_CLASSES = 23
INPUT_CHANNELS = 3
