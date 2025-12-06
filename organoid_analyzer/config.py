import os

# ============================================================
# ROOT DIRECTORIES
# ============================================================

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "Raw")       # Only NCI9 raw TIFFs
GENERATED_DIR = os.path.join(DATA_DIR, "Generated")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# FIJI / JAVA CONFIGURATION
# ============================================================

FIJI_PATH = r"/Applications/Fiji.app"
JAVA_ARGUMENTS = "-Xmx24g"

# ============================================================
# DATASET / ANNOTATION PATHS
# ============================================================

SEQ_LEN = 100

SEQ_DATASET_PATH = os.path.join(GENERATED_DIR, f"trajectory_dataset_{SEQ_LEN}.npz")
TRACK_DATASET_PATH = os.path.join(GENERATED_DIR, "track_dataset.npz")

TEST_TRAIN_SPLIT_ANNOTATION_PATH = os.path.join(DATA_DIR, "Annotations.xlsx")

# ============================================================
# FEATURES
# ============================================================

features = [
    "AREA",
    "PERIMETER",
    "CIRCULARITY",
    "ELLIPSE_ASPECTRATIO",
    "SOLIDITY",
    "SPEED",
    "MEAN_SQUARE_DISPLACEMENT",
    "RADIUS",
]

track_features = [
    "TRACK_DISPLACEMENT",
    "TRACK_STD_SPEED",
    "MEAN_DIRECTIONAL_CHANGE_RATE",
]

FEATURE_LEN = len(features)
TRACK_LEN = len(track_features)

ABLATION_CONFIGS = {
    "Specify": {
        "features": features,
        "track_features": track_features,
    }
}

# ============================================================
# PROCESSED TRACKMATE DATA (CSV)
# ============================================================

PROCESSED_DATA_FOLDERS = {
    "2ND": os.path.join(DATA_DIR, "2ND"),
    "CAF": os.path.join(DATA_DIR, "CAF"),
    "CART": os.path.join(DATA_DIR, "CART"),
    "NCI9": os.path.join(DATA_DIR, "NCI9"),
    "PDO": os.path.join(DATA_DIR, "PDO"),
}

# ============================================================
# RAW TIFF TRACKMATE CONFIG (only used by track_cells.py)
# ============================================================

CELL_TRACKING_DATASET_CONFIGS = {
    "NCI9": {
        "images_folder": RAW_DATA_DIR,
        "subcase_names": ["Round1", "Round2"],
        "case_name": "NCI9",
        "prefix": "NCI9_",
        "specific_thresholds": {},
    }
}

# ============================================================
# DATASET BUILDER CONFIG (create_dataset.py)
# ============================================================

DATASET_CONFIGS = {
    "2ND": {
        "annotation_path": TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        "data_folder": PROCESSED_DATA_FOLDERS["2ND"],
    },
    "CAF": {
        "annotation_path": TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        "data_folder": PROCESSED_DATA_FOLDERS["CAF"],
    },
    "CART": {
        "annotation_path": TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        "data_folder": PROCESSED_DATA_FOLDERS["CART"],
    },
    "NCI9": {
        "annotation_path": TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        "data_folder": PROCESSED_DATA_FOLDERS["NCI9"],
    },
    "PDO": {
        "annotation_path": TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        "data_folder": PROCESSED_DATA_FOLDERS["PDO"],
    },
}

# Output filename prefixes
SEQ_DATASET_PREFIX = ""
TRACK_DATASET_PREFIX = ""

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 256
MAX_EPOCHS = 400
DROPOUT = 0.3

