# scripts/run_pipeline.py
import pandas as pd

from survival.utils import load_data, merge_data, validate_columns
from survival.survival_model import preprocess_data, train_cox_model, evaluate_model, predict_risk
from survival.shap_analysis import run_shap_analysis

# Paths (update later)
marker_path = "data/example_marker_data.csv"
survival_path = "data/example_marker_data.csv"

df = pd.read_csv("data/example_marker_data.csv")

# Validate
validate_columns(df)

# Preprocess
df, feature_cols, scaler = preprocess_data(df)

# Train Cox
model = train_cox_model(df)

# Evaluate
evaluate_model(model)

# Predict risk
risk_scores = predict_risk(model, df)

# SHAP
run_shap_analysis(model, df, feature_cols)