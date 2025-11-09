#!/usr/bin/env python3
"""
Random Forest baseline with extended graphs:
- Confusion matrices (train/test)
- Accuracy bar chart
- Proportion bar chart (train/test)
- R^2 calculation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score
from sklearn.model_selection import train_test_split

# Config paths
try:
    from Config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH
except Exception:
    SEQ_DATASET_PATH = "Generated/trajectory_dataset_100.npz"
    TRACK_DATASET_PATH = "Generated/track_dataset.npz"
    TEST_TRAIN_SPLIT_ANNOTATION_PATH = "Data/test_train_split_annotations.xlsx"

print("Using:")
print(" SEQ_DATASET_PATH:", SEQ_DATASET_PATH)
print(" TRACK_DATASET_PATH:", TRACK_DATASET_PATH)
print(" TEST_TRAIN_SPLIT_ANNOTATION_PATH:", TEST_TRAIN_SPLIT_ANNOTATION_PATH)
print()

# --- Helpers ------------------------------------------------------------------

def load_npz_data(seq_path, track_path):
    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)
    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, track_ids_track = track_data['X'], track_data['track_ids']

    if X_seq.ndim == 3 and X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("[INFO] Transposing seq X to (N, seq_len, features)")
        X_seq = np.transpose(X_seq, (0, 2, 1))
    return X_seq, y_seq, track_ids_seq, X_track, track_ids_track

def make_trackid_key(tid):
    if isinstance(tid, (list, tuple, np.ndarray)):
        try:
            if len(tid) > 1:
                return f"{tid[0]}_{tid[1]}"
            return str(tid[0])
        except Exception:
            return str(tid)
    return str(tid)

def aggregate_seq_features(X_seq):
    means = X_seq.mean(axis=1)
    stds = X_seq.std(axis=1)
    mins = X_seq.min(axis=1)
    maxs = X_seq.max(axis=1)
    return np.concatenate([means, stds, mins, maxs], axis=1)

def split_by_case(_, split_annotation_path):
    df = pd.read_excel(split_annotation_path)
    if 'Train or Test' not in df.columns or 'Case' not in df.columns:
        case_col = [c for c in df.columns if 'case' in c.lower()][0]
        tt_col = [c for c in df.columns if 'train' in c.lower() or 'test' in c.lower()][0]
    else:
        case_col, tt_col = 'Case', 'Train or Test'
    train_cases = df.loc[df[tt_col] == 0, case_col].astype(str).tolist()
    test_cases = df.loc[df[tt_col] == 1, case_col].astype(str).tolist()
    return set(train_cases), set(test_cases)

# --- Build features -----------------------------------------------------------

def build_Xy_for_rf(seq_path, track_path, split_annotation_path):
    X_seq, y_seq, track_ids_seq, X_track, track_ids_track = load_npz_data(seq_path, track_path)

    track_key_to_index = {make_trackid_key(tid): i for i, tid in enumerate(track_ids_track)}
    agg_seq = aggregate_seq_features(X_seq)

    combined_feats, labels, sample_case_names = [], [], []

    for i, tid in enumerate(track_ids_seq):
        key = make_trackid_key(tid)
        if key not in track_key_to_index:
            continue
        track_idx = track_key_to_index[key]
        seq_feat, track_feat = np.nan_to_num(agg_seq[i]), np.nan_to_num(X_track[track_idx])
        combined_feats.append(np.concatenate([seq_feat, track_feat]))
        labels.append(y_seq[i])
        base = str(tid[0]) if isinstance(tid, (list, tuple, np.ndarray)) else str(tid)
        sample_case_names.append("_".join(base.split("_")[:2]))

    X = np.vstack(combined_feats)
    y = np.array(labels)
    return X, y, np.array(sample_case_names)

# --- Train/Evaluate -----------------------------------------------------------

def train_eval_rf(X, y, sample_case_names, split_annotation_path):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    train_cases, test_cases = split_by_case(None, split_annotation_path)

    is_train = np.array([cn in train_cases for cn in sample_case_names])
    is_test = np.array([cn in test_cases for cn in sample_case_names])

    X_train_full, y_train_full = X[is_train], y_enc[is_train]
    X_test, y_test = X[is_test], y_enc[is_test]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        oob_score=True
    )
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train_full)  # full train set
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train_full, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # R^2 for numeric labels
    y_test_orig = le.inverse_transform(y_test)
    y_test_pred_orig = le.inverse_transform(y_test_pred)
    try:
        r2 = r2_score(y_test_orig.astype(float), y_test_pred_orig.astype(float))
    except Exception:
        r2 = None

    print(f"\nTrain acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {test_acc:.3f}")
    print("R^2 (test) between numeric label values:", r2)

    return clf, le, train_acc, val_acc, test_acc, r2, y_train_full, y_train_pred, y_val, y_val_pred, y_test, y_test_pred

# --- Save results -------------------------------------------------------------

def save_results(train_acc, val_acc, test_acc,
                 y_train, y_train_pred, y_val, y_val_pred,
                 y_test, y_test_pred, le):
    results_dir = "Results/RandomForest"
    acc_dir = os.path.join(results_dir, "Accuracies")
    graph_dir = os.path.join(results_dir, "Graphs")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Accuracies CSV
    df_acc = pd.DataFrame({
        "Train Accuracy": [train_acc],
        "Validation Accuracy": [val_acc],
        "Test Accuracy": [test_acc]
    })
    acc_path = os.path.join(acc_dir, f"rf_accuracies_{timestamp}.csv")
    df_acc.to_csv(acc_path, index=False)
    print(f"[SAVED] Accuracies → {acc_path}")

    # Confusion matrices
    def plot_cm(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, filename))
        plt.close()
        print(f"[SAVED] {title} → {filename}")

    plot_cm(y_train, y_train_pred, "Confusion Matrix (Train)", f"confusion_matrix_train_{timestamp}.png")
    plot_cm(y_test, y_test_pred, "Confusion Matrix (Test)", f"confusion_matrix_test_{timestamp}.png")

    # Accuracy bar chart
    plt.figure(figsize=(5,4))
    plt.bar(["Train","Val","Test"], [train_acc, val_acc, test_acc], color=["#4CAF50","#FFC107","#2196F3"])
    plt.title("Random Forest Accuracies")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.tight_layout()
    acc_bar_path = os.path.join(graph_dir, f"accuracy_bar_{timestamp}.png")
    plt.savefig(acc_bar_path)
    plt.close()
    print(f"[SAVED] Accuracy bar chart → {acc_bar_path}")

    # Proportion bar chart
    def plot_proportions(y_true, y_pred, le, title, filename):
        n_classes = len(le.classes_)
        prop_matrix = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            mask = y_true == i
            if mask.sum() == 0:
                continue
            counts = np.bincount(y_pred[mask], minlength=n_classes)
            prop_matrix[i] = counts / counts.sum()
        df_prop = pd.DataFrame(prop_matrix, columns=[f"Pred {c}" for c in le.classes_],
                               index=[f"True {c}" for c in le.classes_])
        df_prop.plot(kind='bar', stacked=True, figsize=(7,5), colormap="tab20")
        plt.ylabel("Proportion")
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(graph_dir, filename))
        plt.close()
        print(f"[SAVED] {title} → {filename}")

    plot_proportions(y_train, y_train_pred, le, "Class Proportions (Train)", f"proportions_train_{timestamp}.png")
    plot_proportions(y_test, y_test_pred, le, "Class Proportions (Test)", f"proportions_test_{timestamp}.png")

# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    X, y, sample_case_names = build_Xy_for_rf(
        SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH
    )
    clf, le, train_acc, val_acc, test_acc, r2, y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred = train_eval_rf(
        X, y, sample_case_names, TEST_TRAIN_SPLIT_ANNOTATION_PATH
    )
    save_results(train_acc, val_acc, test_acc,
                 y_train, y_train_pred, y_val, y_val_pred, y_test, y_test_pred, le)
    print("\n✅ Done. Results and graphs saved in Results/RandomForest/")
