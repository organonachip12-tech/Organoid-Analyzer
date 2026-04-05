# survival/shap_analysis.py

import shap
import matplotlib.pyplot as plt
import os


def run_shap_analysis(model, df, feature_cols, save_path="results/plots"):
    """
    Run SHAP analysis on Cox model predictions.
    """

    X = df[feature_cols]

    # Create output directory if needed
    os.makedirs(save_path, exist_ok=True)

    print("\nRunning SHAP analysis...")

    # Use model prediction function
    explainer = shap.Explainer(model.predict_partial_hazard, X)
    shap_values = explainer(X)

    # Summary plot
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shap_summary.png"))
    plt.close()

    print(f"SHAP summary plot saved to {save_path}")

    return shap_values