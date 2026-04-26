# survival/survival_model.py

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    """
    Standardize marker features.
    """
    feature_cols = [c for c in df.columns if c not in ["patient_id", "time", "event"]]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols, scaler


def train_cox_model(df):
    """
    Train Cox proportional hazards model.
    """
    cph = CoxPHFitter()
    cph.fit(df.drop(columns=["patient_id"]), duration_col="time", event_col="event")

    print("\nCox Model Summary:")
    cph.print_summary()

    return cph


def evaluate_model(cph):
    """
    Print model performance.
    """
    print(f"\nC-index: {cph.concordance_index_:.4f}")


def predict_risk(cph, df):
    """
    Predict relative risk scores.
    """
    return cph.predict_partial_hazard(df)