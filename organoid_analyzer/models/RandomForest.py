import os
import numpy as np
import pandas as pd
from config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    r2_score, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================================
#  Utility functions
# ============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_feature_names(seq_dim, track_dim):
    """
    Matches your fusion logic:
       seq dataset → min/max/mean features over 100 timesteps per channel
       (3 statistic types × 8 channels = 24 features)
       track dataset → already 3 static features
       
       total = 24 + 3 = 27
       actual stacked = 35 because the RF uses trajectory_window=100 & flatten.
    """
    seq_names = []
    stat_names = ["min", "max", "mean"]

    for stat in stat_names:
        for f in range(seq_dim):
            seq_names.append(f"SEQ_{stat}_f{f}")

    track_names = [f"TRACK_f{i}" for i in range(track_dim)]

    return seq_names + track_names


def plot_feature_importance(clf, feature_names, out_path):
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)

    plt.figure(figsize=(10, 8))
    plt.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df = pd.DataFrame(cm, index=labels, columns=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=True, cmap="Blues", fmt="d")
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_roc_curve(y_true, y_prob, labels, out_path):
    plt.figure(figsize=(6, 5))

    for i, class_val in enumerate(labels):
        fpr, tpr, _ = roc_curve((y_true == class_val).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{class_val} (AUC={auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title("ROC Curve (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_proportion_bar_chart(case_names, y_pred, label_classes, out_path):
    df = pd.DataFrame({"Case": case_names, "Pred": y_pred})
    proportions = df.groupby(["Case", "Pred"]).size().unstack(fill_value=0)

    proportions = proportions / proportions.sum(axis=1).values[:, None]

    proportions.plot(kind="bar", stacked=True, figsize=(10, 6))
    plt.title("Prediction Proportions per Case (RF)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================================
#  MAIN RF LOGIC
# ============================================================================

def train_rf():
    print("\nUsing:")
    print(f" SEQ_DATASET_PATH: {SEQ_DATASET_PATH}")
    print(f" TRACK_DATASET_PATH: {TRACK_DATASET_PATH}")
    print(f" TEST_TRAIN_SPLIT_ANNOTATION_PATH: {TEST_TRAIN_SPLIT_ANNOTATION_PATH}\n")

    seq = np.load(SEQ_DATASET_PATH, allow_pickle=True)
    track = np.load(TRACK_DATASET_PATH, allow_pickle=True)

    X_seq = seq["X"]          # (2988, 100, 8)
    X_track = track["X"]      # (3363, 3)
    
    # --- derive case names from track ids ---
    if "track_ids" in seq:
        track_ids_seq = seq["track_ids"]
        case_names = np.array([str(tid).split("_")[0] for tid in track_ids_seq])
    else:
        print("[WARN] track_ids not found in seq dataset — using generic case labels.")
        case_names = np.array([f"Case_{i}" for i in range(len(X_seq))])


    print(f"[INFO] seq X shape: {X_seq.shape}")
    print(f"[INFO] track X shape: {X_track.shape}")

    seq_min = np.min(X_seq, axis=1)
    seq_max = np.max(X_seq, axis=1)
    seq_mean = np.mean(X_seq, axis=1)

    X = np.hstack([seq_min, seq_max, seq_mean, X_track[:len(seq_min)]])
    y = seq["y"]

    print(f"[INFO] Built {len(X)} combined samples for RF (features dim: {X.shape[1]})")

    feature_names = extract_feature_names(seq_dim=8, track_dim=3)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_temp, y_train, y_temp, case_train, case_temp = train_test_split(
        X, y_encoded, case_names, test_size=0.30, random_state=42, stratify=y_encoded
    )

    X_val, X_test, y_val, y_test, val_case, test_case = train_test_split(
        X_temp, y_temp, case_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced_subsample",
        random_state=42
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    r2 = r2_score(y_test, clf.predict(X_test))

    # Logging
    print("\n=== Random Forest results ===")
    print(f"Train acc: {train_acc:.3f}")
    print(f"Val acc:   {val_acc:.3f}")
    print(f"Test acc:  {test_acc:.3f}")
    print(f"R² (test): {r2:.3f}")
    print("\nClassification report (test):\n")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_.astype(str)))

    # ============================================================================
    # Save results
    # ============================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS = "Results/RandomForest"
    ensure_dir(f"{RESULTS}/Graphs")
    ensure_dir(f"{RESULTS}/Accuracies")

    # Accuracies CSV
    acc_df = pd.DataFrame({
        "Train Acc": [train_acc],
        "Val Acc": [val_acc],
        "Test Acc": [test_acc],
        "R2": [r2]
    })
    acc_df.to_csv(f"{RESULTS}/Accuracies/rf_accuracies_{timestamp}.csv", index=False)

    # Confusion matrices
    save_confusion_matrix(y_test, y_test_pred, le.transform(le.classes_),
                          "Confusion Matrix (Test)",
                          f"{RESULTS}/Graphs/conf_matrix_test_{timestamp}.png")

    save_confusion_matrix(y_train, y_train_pred, le.transform(le.classes_),
                          "Confusion Matrix (Train)",
                          f"{RESULTS}/Graphs/conf_matrix_train_{timestamp}.png")

    # Feature Importance
    plot_feature_importance(clf, feature_names,
                            f"{RESULTS}/Graphs/feature_importance_{timestamp}.png")

    # ROC curve
    save_roc_curve(y_test, y_prob, le.transform(le.classes_),
                   f"{RESULTS}/Graphs/roc_curve_{timestamp}.png")

    # Proportion bar charts
    save_proportion_bar_chart(test_case, y_test_pred, le.transform(le.classes_),
                              f"{RESULTS}/Graphs/prediction_proportions_test_{timestamp}.png")

    print("\n✅ All graphs + CSV saved in Results/RandomForest/\n")


if __name__ == "__main__":
    train_rf()
