from .train_fusion_model import Test_UnifiedFusionModel
from ..config import (
    TEST_TRAIN_SPLIT_ANNOTATION_PATH,
    FEATURE_LEN,
    TRACK_LEN,
    MODEL_DIR,
    GENERATED_DIR,
)


def main():
    # Example paths – adjust as needed or wire to argparse
    seq_path = f"{GENERATED_DIR}/trajectory_dataset_100.npz"
    track_path = f"{GENERATED_DIR}/track_dataset.npz"
    model_path = f"{MODEL_DIR}/unified_fusion_model.pth"

    Test_UnifiedFusionModel(
        seq_path,
        track_path,
        model_path,
        TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        results_dir="test",
        seq_input_size=FEATURE_LEN,
        track_input_size=TRACK_LEN,
        hidden_size=32,
        fusion_size=64,
        dropout=0.3,
    )


if __name__ == "__main__":
    main()
