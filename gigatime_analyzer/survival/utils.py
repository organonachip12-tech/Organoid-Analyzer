# survival/utils.py

import pandas as pd

from gigatime_analyzer.survival.tcga_ids import tcga_barcode_from_slide_name


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


def load_tcga_annotations_as_survival(annotations_csv: str) -> pd.DataFrame:
    """
    Load tcga_pipeline-style annotations (image_path, survival_time, death_occurred)
    into one row per patient with columns patient_id, time, event for merge_data / Cox.

    If the CSV already includes patient_id (from an updated pipeline), it is used;
    otherwise patient_id is derived from image_path using the same rule as GigaTIME tiles.
    """
    df = pd.read_csv(annotations_csv)
    if "survival_time" not in df.columns or "death_occurred" not in df.columns:
        raise ValueError(
            "Expected columns survival_time and death_occurred (GDC annotations format)."
        )
    df = df.copy()
    if "patient_id" not in df.columns:
        if "image_path" not in df.columns:
            raise ValueError("Need image_path or patient_id column.")
        df["patient_id"] = df["image_path"].map(tcga_barcode_from_slide_name)
    df["time"] = pd.to_numeric(df["survival_time"], errors="coerce")
    df["event"] = pd.to_numeric(df["death_occurred"], errors="coerce").astype("Int64")
    out = df[["patient_id", "time", "event"]].drop_duplicates(subset=["patient_id"])
    if out["time"].isna().any() or out["event"].isna().any():
        bad = out[out["time"].isna() | out["event"].isna()]
        raise ValueError(f"Invalid survival values after conversion: {len(bad)} rows.")
    out = out.copy()
    out["event"] = out["event"].astype(int)
    return out.reset_index(drop=True)

