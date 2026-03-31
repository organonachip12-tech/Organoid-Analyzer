import os

from gigatime_analyzer.survival.feature_extraction import (
    build_patient_dataframe,
    add_mock_survival_data
)
from gigatime_analyzer.survival.survival_model import (
    preprocess_data,
    train_cox_model,
    evaluate_model
)
from gigatime_analyzer.survival.shap_analysis import run_shap_analysis

from gigatime_analyzer.training.main import parse_args, _run_infer


def main():
    print(" Running GigaTIME → Survival Pipeline")

    # -----------------------------------
    # Step 1: Run GigaTIME inference
    # -----------------------------------
    print("\n Step 1: Running GigaTIME inference...")

    args = parse_args()
    results = _run_infer(args, output_dir="results/temp")

    outputs = results["outputs"]
    ids = results["ids"]

    print(f" Collected {len(outputs)} tile outputs")

    if len(outputs) == 0:
        raise ValueError("No outputs collected from GigaTIME. Check inference pipeline.")

    # -----------------------------------
    # Step 2: Convert → patient features
    # -----------------------------------
    print("\n🔹 Step 2: Extracting marker features...")

    df = build_patient_dataframe(outputs, ids)

    if df.empty:
        raise ValueError("Feature dataframe is empty. Check feature extraction.")

    print(f" Feature dataframe shape: {df.shape}")

    # -----------------------------------
    # Step 3: Add survival data (TEMP)
    # -----------------------------------
    print("\n Step 3: Adding survival data (mock for now)...")

    df = add_mock_survival_data(df)

    # -----------------------------------
    # Step 4: Train Cox model
    # -----------------------------------
    print("\n Step 4: Training Cox model...")

    df, feature_cols, scaler = preprocess_data(df)

    model = train_cox_model(df)

    evaluate_model(model)

    # -----------------------------------
    # Step 5: SHAP analysis
    # -----------------------------------
    print("\n Step 5: Running SHAP analysis...")

    run_shap_analysis(model, df, feature_cols)

    print("\n Pipeline complete!")


if __name__ == "__main__":
    main()