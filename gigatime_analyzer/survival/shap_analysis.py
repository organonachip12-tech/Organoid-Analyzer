# survival/shap_analysis.py

import os

import matplotlib.pyplot as plt
import numpy as np
import shap

# Biological names for the first 21 GigaTIME output channels
MARKER_NAMES = [
    "DAPI", "TRITC", "Cy5", "PD-1", "CD14", "CD4", "T-bet", "CD34", "CD68",
    "CD16", "CD11c", "CD138", "CD20", "CD3", "CD8", "PD-L1", "CK", "Ki67",
    "Tryptase", "Actin-D", "Caspase3-D",
]


def _rename_map(feature_cols):
    """Map marker_N column names → biological display names."""
    m = {}
    for col in feature_cols:
        if col.startswith("marker_"):
            idx = int(col.split("_")[1]) - 1
            if 0 <= idx < len(MARKER_NAMES):
                m[col] = MARKER_NAMES[idx]
    return m


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def run_shap_analysis(cph, df, feature_cols, save_path="results/plots"):
    """
    SHAP beeswarm + bar chart showing which markers drive risk predictions.

    Saves:
        shap_beeswarm.png  — per-patient marker contributions (colour = feature value)
        shap_importance.png — mean |SHAP| bar chart (overall ranking)
    """
    os.makedirs(save_path, exist_ok=True)

    X = df[feature_cols]
    rename = _rename_map(feature_cols)

    # Use a capped background set so SHAP runs in reasonable time
    n_bg = min(100, len(X))
    background = X.sample(n_bg, random_state=42) if len(X) > n_bg else X

    print("    Computing SHAP values (may take a minute)…")
    explainer = shap.Explainer(cph.predict_partial_hazard, background)
    shap_values = explainer(X)

    # Rename feature labels for display
    shap_values.feature_names = [rename.get(n, n) for n in shap_values.feature_names]
    X_display = X.rename(columns=rename)

    # Beeswarm: each dot = one patient, x-axis = SHAP value, colour = feature level
    shap.summary_plot(shap_values, X_display, show=False, plot_type="dot")
    plt.title("SHAP Beeswarm — Marker Contributions to Risk", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bar: mean |SHAP| per marker (simpler read for a lab slide)
    shap.summary_plot(shap_values, X_display, show=False, plot_type="bar")
    plt.title("SHAP Feature Importance — Mean |SHAP Value|", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "shap_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved: shap_beeswarm.png, shap_importance.png")
    return shap_values


def plot_kaplan_meier(cph, df, feature_cols, save_path):
    """
    Kaplan-Meier survival curves split by median Cox risk score.

    Patients above the median score → High Risk (red).
    Patients at or below the median → Low Risk (blue).
    Includes 95 % confidence bands and log-rank p-value.

    Saves: kaplan_meier.png
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    os.makedirs(save_path, exist_ok=True)

    X = df[feature_cols]
    risk = cph.predict_partial_hazard(X)
    median_risk = risk.median()

    high = risk > median_risk
    low = ~high

    time = df["time"]
    event = df["event"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    for mask, label, color in [
        (high, f"High risk  (n={int(high.sum())})", "crimson"),
        (low,  f"Low risk   (n={int(low.sum())})",  "steelblue"),
    ]:
        kmf = KaplanMeierFitter(label=label)
        kmf.fit(time[mask], event_observed=event[mask])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)

    lr = logrank_test(
        time[high], time[low],
        event_observed_A=event[high],
        event_observed_B=event[low],
    )

    p_str = f"{lr.p_value:.4f}" if lr.p_value >= 0.0001 else f"{lr.p_value:.2e}"
    ax.set_title(f"Kaplan–Meier Survival Curves\nLog-rank  p = {p_str}", fontsize=13)
    ax.set_xlabel("Time (days)", fontsize=11)
    ax.set_ylabel("Survival Probability", fontsize=11)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "kaplan_meier.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved: kaplan_meier.png  (log-rank p = {p_str})")


def plot_forest(cph, feature_cols, save_path):
    """
    Forest plot of Cox hazard ratios with 95 % confidence intervals.

    Each row is one marker. Points right of the dashed line (HR > 1) indicate
    markers associated with higher risk; left (HR < 1) indicate protection.
    Red = statistically significant (p < 0.05), blue = non-significant.

    Saves: forest_plot.png
    """
    os.makedirs(save_path, exist_ok=True)

    rename = _rename_map(feature_cols)
    summary = cph.summary.copy()
    summary.index = [rename.get(str(i), str(i)) for i in summary.index]
    summary = summary.sort_values("exp(coef)")

    hr     = summary["exp(coef)"].values
    hr_low = summary["exp(coef) lower 0.95"].values
    hr_hi  = summary["exp(coef) upper 0.95"].values
    p_vals = summary["p"].values
    labels = summary.index.tolist()

    y = np.arange(len(labels))
    colors = ["crimson" if p < 0.05 else "steelblue" for p in p_vals]

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.42)))
    fig.patch.set_facecolor("white")

    ax.scatter(hr, y, color=colors, s=55, zorder=3)
    for i in range(len(labels)):
        ax.hlines(y[i], hr_low[i], hr_hi[i], colors=colors[i], linewidth=1.8, zorder=2)

    ax.axvline(x=1, color="black", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Hazard Ratio  (95 % CI)", fontsize=11)
    ax.set_title("Cox Model — Hazard Ratios per Marker\n(red = p < 0.05)", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    # Annotate HR values next to each point
    for i, (h, p) in enumerate(zip(hr, p_vals)):
        ax.text(max(hr_hi) * 1.02, y[i], f"{h:.2f}{'*' if p < 0.05 else ''}",
                va="center", fontsize=8, color=colors[i])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "forest_plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    Saved: forest_plot.png")


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def generate_report(cph, df, feature_cols, save_path="results/plots"):
    """
    Generate all lab-ready analysis plots after Cox model training.

    Output files in save_path/:
        shap_beeswarm.png   — per-patient SHAP values (which markers push risk up/down)
        shap_importance.png — mean |SHAP| bar chart (overall marker importance ranking)
        kaplan_meier.png    — survival curves for high vs low risk patient groups
        forest_plot.png     — hazard ratios with 95 % CIs, coloured by significance

    Parameters
    ----------
    cph          : fitted CoxPHFitter
    df           : merged + preprocessed DataFrame (has time, event, marker_N columns)
    feature_cols : list of str — marker column names (marker_1 … marker_21)
    save_path    : directory to write PNGs into
    """
    os.makedirs(save_path, exist_ok=True)

    print("\nGenerating analysis plots for lab report…")

    print("  [1/3] SHAP analysis…")
    run_shap_analysis(cph, df, feature_cols, save_path)

    print("  [2/3] Kaplan–Meier survival curves…")
    plot_kaplan_meier(cph, df, feature_cols, save_path)

    print("  [3/3] Forest plot (hazard ratios)…")
    plot_forest(cph, feature_cols, save_path)

    print(f"\nAll plots saved to: {save_path}/")
    print("  → shap_beeswarm.png")
    print("  → shap_importance.png")
    print("  → kaplan_meier.png")
    print("  → forest_plot.png")
