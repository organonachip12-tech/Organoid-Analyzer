"""
GigaTIME Analyzer configuration — paths and default hyperparameters.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "gigatime")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")
SVS_DIR = os.path.join(DATA_DIR, "svs")
TILES_DIR = os.path.join(DATA_DIR, "preprocessed_tiles")

DEFAULT_ARCH = "gigatime"
NUM_CLASSES = 23
INPUT_CHANNELS = 3
