import os

from gigatime_analyzer.survival.feature_extraction import (
    build_patient_dataframe,
)

from gigatime_analyzer.survival.utils import (
    load_tcga_annotations_as_survival,
    merge_data,
    validate_columns
)

from gigatime_analyzer.survival.survival_model import (
    preprocess_data,
    train_cox_model,
    evaluate_model
)

from gigatime_analyzer.survival.shap_analysis import run_shap_analysis

from gigatime_analyzer.training.main import parse_args, _run_infer


def main():
    print("Running GigaTIME → Survival Pipeline (REAL TCGA DATA)")

    # -----------------------------------
    # Step 1: Run GigaTIME inference
    # -----------------------------------
    print("\nStep 1: Running GigaTIME inference...")

    args = parse_args()

    results = _run_infer(args, output_dir="Results/temp")

    outputs = results["outputs"]
    ids = results["ids"]

    print(f"Collected {len(outputs)} tile outputs")

    if len(outputs) == 0:
        raise ValueError("No outputs collected from GigaTIME.")

    # -----------------------------------
    # Step 2: Convert → patient features
    # -----------------------------------
    print("\nStep 2: Extracting marker features...")

    marker_df = build_patient_dataframe(outputs, ids)

    if marker_df.empty:
        raise ValueError("Feature dataframe is empty.")

    print(f"Marker dataframe shape: {marker_df.shape}")
    print(marker_df.head())

    # -----------------------------------
    # Step 3: Load REAL TCGA survival data
    # -----------------------------------
    print("\nStep 3: Loading TCGA survival annotations...")

    annotations_csv = "Data/gigatime/annotations.csv"  # 🔥 IMPORTANT

    survival_df = load_tcga_annotations_as_survival(annotations_csv)

    print(f"Survival dataframe shape: {survival_df.shape}")
    print(survival_df.head())

    # -----------------------------------
    # Step 4: Merge features + survival
    # -----------------------------------
    print("\nStep 4: Merging data...")

    df = merge_data(marker_df, survival_df)

    validate_columns(df)

    print(f"Merged dataframe shape: {df.shape}")

    if df.empty:
        raise ValueError("Merged dataframe is empty — ID mismatch likely.")

    # -----------------------------------
    # Step 5: Train Cox model
    # -----------------------------------
    print("\nStep 5: Training Cox model...")

    df, feature_cols, scaler = preprocess_data(df)

    model = train_cox_model(df)

    evaluate_model(model)

    # -----------------------------------
    # Step 6: SHAP analysis
    # -----------------------------------
    print("\nStep 6: Running SHAP analysis...")

    run_shap_analysis(model, df, feature_cols, save_path="Results/plots")

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()