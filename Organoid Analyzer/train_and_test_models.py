from train_fusion_model import train_models_and_shap
from Config import *


if __name__ == "__main__":
    train_models_and_shap(ABLATION_CONFIGS, SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH,
                 MAX_POW_HIDDEN, MAX_POW_FUSION, MIN_POW_HIDDEN, MIN_POW_FUSION, perform_SHAP_analysis = False)
