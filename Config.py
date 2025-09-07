import os

# ======= GENERATE FOLDERS =======

# Upload this folder with original data files
DATA_DIR = "./Data"
os.makedirs(DATA_DIR, exist_ok=True)
# Manually delete this folder before uploading
GENERATED_DIR = "./Generated"
os.makedirs(GENERATED_DIR, exist_ok=True)
MODEL_DIR = os.path.join(GENERATED_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
# Upload this folder with results and plots
RESULTS_DIR = "./Results"


# ======= CELL TRACKING SETTINGS =======
IMAGES_FOLDER = r"C:\Users\billy\Documents\VIP Images\20250819_Different stroma density_early stage dynamics_2 batch"
FIJI_PATH = r"C:\Users\billy\Desktop\Fiji.app"
CASE_NAME = "DiffStroma"
JAVA_ARGUMENTS = '-Xmx10g'



# ======= DATASET GENERATION SETTINGS =======
SEQ_LEN = 100 # Number of frames to use.
DATASET_CONFIGS = {
    "CART": {"annotation_path" : f"{DATA_DIR}/CART annotations.xlsx", 
                 "data_folder" : f"{DATA_DIR}/CART"},

    "2nd": {"annotation_path" : f"{DATA_DIR}/2nd batch annotations.xlsx",
                "data_folder" : f"{DATA_DIR}/2ND"},
    
    "PDO": {"annotation_path" : f"{DATA_DIR}/PDO_annotation.xlsx",
                "data_folder" : f"{DATA_DIR}/PDO"},
}

SEQ_DATASET_PREFIX = ""
TRACK_DATASET_PREFIX = ""

features = [ # Time-based Features 
    'AREA', 'PERIMETER', 'CIRCULARITY',
    'ELLIPSE_ASPECTRATIO','SOLIDITY', 
    'SPEED', "MEAN_SQUARE_DISPLACEMENT"
]


track_features = [ # Track-Level Statistics Features
    "TRACK_DISPLACEMENT", "TRACK_STD_SPEED",
    "MEAN_DIRECTIONAL_CHANGE_RATE"
]

FEATURE_LEN = len(features)
TRACK_LEN = len(track_features)


# ======= TRAINING SETTINGS =======
TEST_TRAIN_SPLIT_ANNOTATION_PATH = r"C:\Users\billy\Desktop\VIP\Tianzan\Cell-Track-Multi-Model\Data\Annotations.xlsx"
SEQ_DATASET_PATH = os.path.join(GENERATED_DIR, f"{SEQ_DATASET_PREFIX}trajectory_dataset_{SEQ_LEN}.npz")
TRACK_DATASET_PATH = os.path.join(GENERATED_DIR, f"{TRACK_DATASET_PREFIX}track_dataset.npz")

DROPOUT = 0.3
MAX_EPOCHS = 400
BATCH_SIZE = 256

MIN_POW_FUSION = 4
MAX_POW_FUSION = 12

MIN_POW_HIDDEN = 2
MAX_POW_HIDDEN = 7

ABLATION_CONFIGS = {
    "Specify" : {
        "features": features,
        "track_features" :track_features
    },
}

