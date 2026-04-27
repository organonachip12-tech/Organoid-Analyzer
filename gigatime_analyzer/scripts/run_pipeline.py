"""GigaTIME → Survival prediction pipeline.

Training (run on HPC with paired TCGA slides + survival annotations):
    python -m gigatime_analyzer.scripts.run_pipeline --mode train \\
        --svs_dir data/gigatime/svs \\
        --annotations_csv data/gigatime/annotations.csv \\
        --cox_model_path results/cox_model.pkl

Prediction (called from frontend or CLI for new patient slides):
    python -m gigatime_analyzer.scripts.run_pipeline --mode predict \\
        --svs_dir path/to/uploaded/slides \\
        --cox_model_path results/cox_model.pkl
"""
import argparse
import os
import sys

import numpy as np
import torch


# ---------------------------------------------------------------------------
# H&E-only inference (no paired mIF data needed)
# ---------------------------------------------------------------------------

def _load_gigatime_model(model_path, device, num_classes=23, input_channels=3):
    """Load GigaTIME model from local path or HuggingFace Hub."""
    import types
    from gigatime_analyzer.models.archs import gigatime

    model = gigatime(num_classes=num_classes, input_channels=input_channels).to(device)

    if model_path and os.path.exists(model_path):
        if "torch.utils.serialization" not in sys.modules:
            sys.modules["torch.utils.serialization"] = types.ModuleType("torch.utils.serialization")
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        print(f"Loaded GigaTIME weights from {model_path}")
    else:
        from gigatime_analyzer.training.infer import load_pretrained_model
        model = load_pretrained_model(model, model_path or "", hf_token=os.environ.get("HF_TOKEN"))

    model.eval()
    return model


def collect_tile_outputs(tile_dir, model, device, window_size=256):
    """
    Run GigaTIME inference on H&E tile PNGs in tile_dir.

    Does NOT require paired mIF data — suitable for new patient slides.
    Uses sliding-window inference (256px) matching the training setup.

    Returns
    -------
    outputs : list of torch.Tensor, each shape (1, 23, 512, 512), sigmoid probabilities
    tile_ids : list of str, basename stem of each tile (used for patient ID extraction)
    """
    import glob
    import albumentations as A
    from albumentations.augmentations import transforms as alb_transforms
    from PIL import Image

    transform = A.Compose([
        A.Resize(512, 512),
        alb_transforms.Normalize(),
    ])

    tile_paths = sorted(
        glob.glob(os.path.join(tile_dir, "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(tile_dir, "*.png"))
    )
    # deduplicate in case of overlap between recursive and flat globs
    tile_paths = list(dict.fromkeys(tile_paths))

    if not tile_paths:
        raise ValueError(f"No PNG tiles found in {tile_dir}. Run preprocessing first.")

    print(f"Found {len(tile_paths)} tiles — running inference...")

    outputs = []
    tile_ids = []

    with torch.no_grad():
        for tile_path in tile_paths:
            img = np.array(Image.open(tile_path).convert("RGB"))
            aug = transform(image=img)
            x = (
                torch.from_numpy(aug["image"])
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(device)
            )

            out = torch.zeros(1, 23, x.shape[2], x.shape[3], device=device)
            for i in range(0, x.shape[2], window_size):
                for j in range(0, x.shape[3], window_size):
                    window = x[:, :, i : i + window_size, j : j + window_size]
                    out[:, :, i : i + window_size, j : j + window_size] = model(window)

            out = torch.sigmoid(out)
            outputs.append(out.cpu())
            # Patient ID is encoded in the parent directory name (slide stem),
            # not the tile filename (which is just x_y_size_size_he).
            tile_ids.append(os.path.basename(os.path.dirname(tile_path)))

    print(f"Inference complete: {len(outputs)} tiles processed")
    return outputs, tile_ids


# ---------------------------------------------------------------------------
# Training pipeline (HPC)
# ---------------------------------------------------------------------------

def train_pipeline(args):
    """
    Full training pipeline:
      SVS slides → tiles → GigaTIME inference → patient features
      → merge TCGA annotations → train Cox model → save model + SHAP plot
    """
    from pathlib import Path

    from gigatime_analyzer.survival.feature_extraction import build_patient_dataframe
    from gigatime_analyzer.survival.shap_analysis import generate_report
    from gigatime_analyzer.survival.survival_model import (
        evaluate_model,
        preprocess_data,
        save_model,
        train_cox_model,
    )
    from gigatime_analyzer.survival.utils import (
        load_tcga_annotations_as_survival,
        merge_data,
        validate_columns,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Preprocess SVS → tiles (skip if tiles already exist)
    tiling_dir = args.tiling_dir
    if tiling_dir is None:
        if args.svs_dir is None:
            raise ValueError("Provide --tiling_dir (preprocessed tiles) or --svs_dir (raw slides).")
        from gigatime_analyzer.config import TILES_DIR
        from gigatime_analyzer.preprocessing import process_slides

        tiling_dir = str(TILES_DIR)
        print(f"\nStep 1: Preprocessing SVS slides from {args.svs_dir} → {tiling_dir}")
        process_slides(input_dir=Path(args.svs_dir), output_dir=Path(tiling_dir))
    else:
        print(f"\nStep 1: Using preprocessed tiles from {tiling_dir}")

    # Step 2: GigaTIME inference on H&E tiles
    print("\nStep 2: Running GigaTIME inference...")
    model = _load_gigatime_model(args.gigatime_model_path, device)
    outputs, tile_ids = collect_tile_outputs(tiling_dir, model, device)

    if not outputs:
        raise ValueError("No tile outputs collected — check the tiling directory.")

    # Step 3: Extract patient-level marker features
    print("\nStep 3: Extracting patient-level marker features...")
    marker_df = build_patient_dataframe(outputs, tile_ids)

    if marker_df.empty:
        raise ValueError(
            "Feature dataframe is empty — patient ID could not be parsed from tile names. "
            "Tile names must follow TCGA barcode format (e.g. TCGA-XX-XXXX-...)."
        )
    print(f"Marker dataframe: {marker_df.shape} — {marker_df['patient_id'].nunique()} patients")

    # Step 4: Load TCGA survival annotations
    print(f"\nStep 4: Loading TCGA survival annotations from {args.annotations_csv}...")
    survival_df = load_tcga_annotations_as_survival(args.annotations_csv)
    print(f"Survival dataframe: {survival_df.shape}")

    # Step 5: Merge features + survival labels
    print("\nStep 5: Merging marker features with survival data...")
    df = merge_data(marker_df, survival_df)
    validate_columns(df)

    if df.empty:
        raise ValueError(
            "Merged dataframe is empty — patient IDs in tiles don't match annotations. "
            "Check that tile names contain the same TCGA barcodes as the annotations CSV."
        )
    print(f"Merged dataframe: {df.shape} ({df['patient_id'].nunique()} patients)")

    # Step 6: Train Cox proportional hazards model
    print("\nStep 6: Training Cox model...")
    df, feature_cols, scaler = preprocess_data(df)
    cox_model = train_cox_model(df)
    evaluate_model(cox_model)

    # Step 7: Save model
    cox_out = args.cox_model_path
    out_dir = os.path.dirname(cox_out) or "."
    os.makedirs(out_dir, exist_ok=True)
    save_model(cox_model, scaler, feature_cols, cox_out)
    print(f"Cox model saved → {cox_out}")

    # Step 8: Generate analysis plots for lab report
    print("\nStep 8: Generating analysis plots...")
    shap_dir = os.path.join(out_dir, "plots")
    generate_report(cox_model, df, feature_cols, save_path=shap_dir)

    print("\nTraining pipeline complete!")
    return cox_model, scaler, feature_cols


# ---------------------------------------------------------------------------
# Prediction pipeline (frontend / new patient)
# ---------------------------------------------------------------------------

def predict_pipeline(args):
    """
    Predict survival risk for uploaded slides:
      SVS slides → tiles → GigaTIME inference → patient features → Cox risk score
    """
    from pathlib import Path

    from gigatime_analyzer.survival.feature_extraction import build_patient_dataframe
    from gigatime_analyzer.survival.survival_model import load_model, predict_patient_risk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tiling_dir = args.tiling_dir
    if tiling_dir is None:
        if args.svs_dir is None:
            raise ValueError("Provide --tiling_dir or --svs_dir.")
        from gigatime_analyzer.config import TILES_DIR
        from gigatime_analyzer.preprocessing import process_slides

        tiling_dir = str(TILES_DIR)
        print(f"Preprocessing SVS slides → {tiling_dir}")
        process_slides(input_dir=Path(args.svs_dir), output_dir=Path(tiling_dir))

    print("Running GigaTIME inference...")
    model = _load_gigatime_model(args.gigatime_model_path, device)
    outputs, tile_ids = collect_tile_outputs(tiling_dir, model, device)

    if not outputs:
        raise ValueError("No tile outputs — check preprocessing output.")

    marker_df = build_patient_dataframe(outputs, tile_ids)
    cox_model, scaler, feature_cols = load_model(args.cox_model_path)
    risk_scores = predict_patient_risk(cox_model, scaler, feature_cols, marker_df)

    print("\nRisk scores per patient:")
    print(risk_scores.to_string())
    return risk_scores


# ---------------------------------------------------------------------------
# Callable from frontend (after inference is already done)
# ---------------------------------------------------------------------------

def predict_from_channel_stats(channel_means, cox_model_path):
    """
    Predict survival risk from per-channel mean marker intensities.

    This is the lightweight entry point for the frontend: it receives the
    per-channel stats that are already computed during slide processing
    (mean probability across all tiles for each of the 21 marker channels)
    and returns a scalar risk score.

    Parameters
    ----------
    channel_means : list or array-like of length >= 21
        Mean sigmoid probability for each GigaTIME output channel.
        Only the first 21 channels (biological markers) are used.
    cox_model_path : str
        Path to the saved Cox model (.pkl) produced by train_pipeline.

    Returns
    -------
    float : partial hazard score (higher = greater risk)
    """
    import pandas as pd

    from gigatime_analyzer.survival.survival_model import load_model, predict_patient_risk

    cox_model, scaler, feature_cols = load_model(cox_model_path)

    marker_dict = {f"marker_{i + 1}": [float(v)] for i, v in enumerate(channel_means[:21])}
    marker_dict["patient_id"] = ["patient"]
    marker_df = pd.DataFrame(marker_dict)

    risk = predict_patient_risk(cox_model, scaler, feature_cols, marker_df)
    return float(risk.iloc[0])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="GigaTIME → Survival Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "predict"],
        help="train: fit Cox model on TCGA data (HPC); predict: score new patient slides",
    )
    parser.add_argument("--svs_dir", default=None,
                        help="Directory of SVS slides to preprocess (skipped if --tiling_dir given)")
    parser.add_argument("--tiling_dir", default=None,
                        help="Pre-tiled PNG directory (skips preprocessing step)")
    parser.add_argument("--annotations_csv", default="data/gigatime/annotations.csv",
                        help="TCGA survival annotations CSV (required for --mode train)")
    parser.add_argument("--gigatime_model_path", default=None,
                        help="Path to GigaTIME model.pth (defaults to HuggingFace Hub download)")
    parser.add_argument("--cox_model_path", default="results/cox_model.pkl",
                        help="Path to save (train) or load (predict) the Cox model")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train_pipeline(args)
    else:
        predict_pipeline(args)


if __name__ == "__main__":
    main()
