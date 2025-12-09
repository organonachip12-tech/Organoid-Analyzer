import argparse
import os

from ..config import (
    SEQ_LEN,
    MAX_EPOCHS,
    BATCH_SIZE,
    DROPOUT,
    ABLATION_CONFIGS,
    SEQ_DATASET_PATH,
    TRACK_DATASET_PATH,
    TEST_TRAIN_SPLIT_ANNOTATION_PATH,
)
from .train_fusion_model import train_models_and_shap


def main():
    parser = argparse.ArgumentParser(description="Train Organoid Analyzer models")

    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate")
    parser.add_argument("--hidden_sizes", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--fusion_sizes", type=int, default=128, help="Fusion layer size")
    parser.add_argument("--model_type", type=str, default="fusion", choices=["fusion", "random_forest"], help="Model type")
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "2ND", "CAF", "CART", "PDO"], help="Dataset to use")
    parser.add_argument("--output_dir", help="Output directory for results")

    args = parser.parse_args()

    # Override config values
    seq_len = args.seq_len
    max_epochs = args.epochs
    batch_size = args.batch_size
    dropout = args.dropout
    model_type = args.model_type
    dataset = args.dataset

    max_pow_hidden = args.hidden_sizes.bit_length() - 1
    min_pow_hidden = max_pow_hidden - 1

    max_pow_fusion = args.fusion_sizes.bit_length() - 1
    min_pow_fusion = max_pow_fusion - 1

    results_dir = args.output_dir
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        # We don't override global RESULTS_DIR here, we just pass output_dir
        # as the CLI-level folder; your training code still uses config.RESULTS_DIR.
        # If you want results_dir to replace RESULTS_DIR, you can update config.

    # Log selected model and dataset
    print(f"Training with model_type={model_type}, dataset={dataset}")

    train_models_and_shap(
        ABLATION_CONFIGS,
        SEQ_DATASET_PATH,
        TRACK_DATASET_PATH,
        TEST_TRAIN_SPLIT_ANNOTATION_PATH,
        max_pow_hidden,
        max_pow_fusion,
        min_pow_hidden,
        min_pow_fusion,
        perform_SHAP_analysis=False,
        model_type=model_type,
        dataset=dataset,
    )


if __name__ == "__main__":
    main()
