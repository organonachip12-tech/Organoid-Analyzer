from train_fusion_model import train_models_and_shap
from Config import *
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Organoid Analyzer models")
    
    # Experiment parameters
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout rate")
    parser.add_argument("--hidden_sizes", type=int, default=2**MAX_POW_HIDDEN, help="Hidden layer size")
    parser.add_argument("--fusion_sizes", type=int, default=2**MAX_POW_FUSION, help="Fusion layer size")
    
    # Output
    parser.add_argument("--output_dir", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Override parameters
    SEQ_LEN = args.seq_len
    MAX_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DROPOUT = args.dropout
    
    # Convert sizes to power values
    MAX_POW_HIDDEN = int(args.hidden_sizes ** 0.5)
    MIN_POW_HIDDEN = MAX_POW_HIDDEN - 1  # Create a range of at least 1
    MAX_POW_FUSION = int(args.fusion_sizes ** 0.5)
    MIN_POW_FUSION = MAX_POW_FUSION - 1  # Create a range of at least 1
    
    # Override output directory if provided
    if args.output_dir:
        RESULTS_DIR = args.output_dir
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # Update the global RESULTS_DIR in the imported modules
        import train_fusion_model
        train_fusion_model.RESULTS_DIR = RESULTS_DIR
    
    train_models_and_shap(ABLATION_CONFIGS, SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH,
                 MAX_POW_HIDDEN, MAX_POW_FUSION, MIN_POW_HIDDEN, MIN_POW_FUSION, perform_SHAP_analysis = False)
