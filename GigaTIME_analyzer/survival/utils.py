# survival/utils.py

import pandas as pd


def load_data(marker_path, survival_path):
    """
    Load marker and survival data.
    """
    markers = pd.read_csv(marker_path)
    survival = pd.read_csv(survival_path)

    return markers, survival


def merge_data(markers, survival):
    """
    Merge marker features with survival data.
    """
    df = markers.merge(survival, on="patient_id")

    # Basic validation
    if df.isnull().sum().sum() > 0:
        print("Warning: Missing values detected.")

    return df


def validate_columns(df):
    """
    Ensure required columns exist.
    """
    required = ["patient_id", "time", "event"]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")