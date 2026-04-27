# survival/survival_model.py

import os
import pickle

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """Standardize marker features in-place. Returns (df, feature_cols, scaler)."""
    feature_cols = [c for c in df.columns if c not in ["patient_id", "time", "event"]]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, feature_cols, scaler


def train_cox_model(df):
    """Train Cox proportional hazards model. df must have time + event columns."""
    cph = CoxPHFitter()
    cph.fit(df.drop(columns=["patient_id"]), duration_col="time", event_col="event")
    print("\nCox Model Summary:")
    cph.print_summary()
    return cph


def evaluate_model(cph):
    """Print concordance index after training."""
    print(f"\nC-index: {cph.concordance_index_:.4f}")


def predict_risk(cph, df):
    """Return partial hazard scores for a prepared (already-scaled) DataFrame."""
    return cph.predict_partial_hazard(df)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(cph, scaler, feature_cols, path):
    """
    Save the trained Cox model, StandardScaler, and feature column list to disk.

    All three are required together to score new patients correctly.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"cph": cph, "scaler": scaler, "feature_cols": feature_cols}, f)
    print(f"Saved Cox model to {path}")


def load_model(path):
    """
    Load Cox model, scaler, and feature columns from a .pkl saved by save_model.

    Returns
    -------
    cph : CoxPHFitter (fitted)
    scaler : StandardScaler (fitted)
    feature_cols : list of str
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cox model not found at '{path}'. "
            "Run the training pipeline first (--mode train)."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["cph"], data["scaler"], data["feature_cols"]


def predict_patient_risk(cph, scaler, feature_cols, marker_df):
    """
    Predict relative risk scores for new patients.

    Parameters
    ----------
    cph : fitted CoxPHFitter
    scaler : fitted StandardScaler (from training)
    feature_cols : list of str — marker column names expected by the model
    marker_df : DataFrame with columns [patient_id, marker_1, ..., marker_N]
                Does NOT need time or event columns.

    Returns
    -------
    pd.Series : partial hazard scores indexed by patient_id (higher = greater risk)
    """
    df = marker_df.copy()

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing marker columns: {missing}")

    df[feature_cols] = scaler.transform(df[feature_cols])
    risk = cph.predict_partial_hazard(df[feature_cols])
    risk.index = df["patient_id"].values
    return risk
