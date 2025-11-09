#!/usr/bin/env python3
"""
Random Forest baseline with robust splitting and extended graphs:
- Confusion matrices (train/test)
- Accuracy bar chart
- Predicted vs Actual scatter (R²)
- Feature importance plot
- OOB error curve (incremental build)
- Multiclass ROC curves
- Proportions stacked bar chart (styled similar to provided Fusion plot)
- Saves accuracies CSV and proportions CSVs
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             r2_score, roc_curve, auc)
from sklearn.model_selection import train_test_split

# Try to reuse your config paths
try:
    from Config import SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH
except Exception:
    SEQ_DATASET_PATH = "Generated/trajectory_dataset_100.npz"
    TRACK_DATASET_PATH = "Generated/track_dataset.npz"
    TEST_TRAIN_SPLIT_ANNOTATION_PATH = "Data/test_train_split_annotations.xlsx"

plt.switch_backend("Agg")  # non-interactive backend

print("Using:")
print(" SEQ_DATASET_PATH:", SEQ_DATASET_PATH)
print(" TRACK_DATASET_PATH:", TRACK_DATASET_PATH)
print(" TEST_TRAIN_SPLIT_ANNOTATION_PATH:", TEST_TRAIN_SPLIT_ANNOTATION_PATH)
print()

# -------------------- Helpers --------------------

def load_npz_data(seq_path, track_path):
    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)
    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, track_ids_track = track_data['X'], track_data['track_ids']

    # transpose heuristics used in your other code
    if X_seq.ndim == 3 and X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("[INFO] Transposing seq X to (N, seq_len, features)")
        X_seq = np.transpose(X_seq, (0, 2, 1))
    print("[INFO] seq X shape:", X_seq.shape)
    print("[INFO] track X shape:", X_track.shape)
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

def split_by_case(split_annotation_path):
    # returns (train_cases_set, test_cases_set). If file is missing or columns unknown, returns (set(), set())
    try:
        df = pd.read_excel(split_annotation_path)
    except Exception as e:
        print(f"[WARN] Could not read split file {split_annotation_path}: {e}")
        return set(), set()
    # try canonical columns
    if 'Train or Test' in df.columns and 'Case' in df.columns:
        train_cases = df.loc[df['Train or Test'] == 0, 'Case'].astype(str).tolist()
        test_cases = df.loc[df['Train or Test'] == 1, 'Case'].astype(str).tolist()
        return set(train_cases), set(test_cases)
    # fallback heuristics
    case_col = next((c for c in df.columns if 'case' in c.lower()), df.columns[0])
    tt_col = next((c for c in df.columns if 'train' in c.lower() and 'test' in c.lower()), None)
    if tt_col is None:
        # fallback to second column if available
        tt_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    train_cases = df.loc[df[tt_col] == 0, case_col].astype(str).tolist()
    test_cases = df.loc[df[tt_col] == 1, case_col].astype(str).tolist()
    return set(train_cases), set(test_cases)

# -------------------- Build combined features --------------------

def build_Xy_for_rf(seq_path, track_path, split_annotation_path=None):
    X_seq, y_seq, track_ids_seq, X_track, track_ids_track = load_npz_data(seq_path, track_path)
    track_key_to_index = {make_trackid_key(tid): i for i, tid in enumerate(track_ids_track)}
    agg_seq = aggregate_seq_features(X_seq)

    combined_feats = []
    labels = []
    case_names = []

    for i, tid in enumerate(track_ids_seq):
        key = make_trackid_key(tid)
        # exact match
        if key not in track_key_to_index:
            # try match by prefix (e.g., "NYU318_0" -> "NYU318")
            try:
                if isinstance(tid, (list, tuple, np.ndarray)):
                    first = tid[0]
                else:
                    first = tid
                if isinstance(first, str) and "_" in first:
                    prefix = first.split("_")[0]
                    # find a track id that starts with prefix + "_"
                    found = None
                    for tk in track_key_to_index:
                        if tk.startswith(prefix + "_"):
                            found = track_key_to_index[tk]
                            break
                    if found is None:
                        continue
                    track_idx = found
                else:
                    continue
            except Exception:
                continue
        else:
            track_idx = track_key_to_index[key]

        seq_feat = np.nan_to_num(agg_seq[i], nan=0.0, posinf=0.0, neginf=0.0)
        track_feat = np.nan_to_num(X_track[track_idx], nan=0.0, posinf=0.0, neginf=0.0)
        combined_feats.append(np.concatenate([seq_feat, track_feat]))
        labels.append(y_seq[i])

        # derive case name (same logic as fusion code: first two underscore parts)
        try:
            base = tid[0] if isinstance(tid, (list, tuple, np.ndarray)) else tid
            base = base if isinstance(base, str) else str(base)
            case_names.append("_".join(base.split("_")[:2]))
        except Exception:
            case_names.append(str(tid))

    if not combined_feats:
        print("[ERROR] No combined features built. Check dataset matching logic.")
        return np.zeros((0, 0)), np.array([]), np.array([])

    X = np.vstack(combined_feats)
    y = np.array(labels)
    case_names = np.array(case_names)
    print(f"[INFO] Built {X.shape[0]} combined samples for RF (features dim: {X.shape[1]})")
    return X, y, case_names

# -------------------- Train & evaluate --------------------

def train_eval_rf(X, y, case_names, split_annotation_path):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("[INFO] label classes (original):", le.classes_)

    # Try case-level split
    train_cases, test_cases = split_by_case(split_annotation_path)

    use_case_split = bool(train_cases or test_cases)
    if use_case_split:
        is_train = np.array([cn in train_cases for cn in case_names])
        is_test = np.array([cn in test_cases for cn in case_names])
        # if no test membership or no train membership, fallback
        if not is_test.any() or not is_train.any():
            print("[WARN] Case-based split produced empty train/test. Falling back to stratified random split.")
            use_case_split = False
        else:
            X_train_all = X[is_train]
            y_train_all = y_enc[is_train]
            X_test = X[is_test]
            y_test = y_enc[is_test]
            # create validation split from training
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
                )
            except ValueError:
                # fallback to random split if stratify fails
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_all, y_train_all, test_size=0.2, random_state=42
                )
    if not use_case_split:
        # fallback to stratified random split across all samples
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_enc, test_size=0.4, random_state=42, stratify=y_enc
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
        except ValueError:
            # final fallback: simple random split without stratify
            print("[WARN] Stratified split failed due to class imbalance. Doing random split.")
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y_enc, test_size=0.4, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )

    # Train RF with class_weight balanced; also compute OOB curve later optionally
    n_estimators = 200
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                 class_weight='balanced', n_jobs=-1, oob_score=True)
    clf.fit(X_train, y_train)

    # Predictions
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    # Metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    # R^2 between numeric label values (inverse transform)
    try:
        y_test_orig = le.inverse_transform(y_test)
        y_test_pred_orig = le.inverse_transform(y_test_pred)
        r2 = r2_score(y_test_orig.astype(float), y_test_pred_orig.astype(float))
    except Exception:
        r2 = None

    print(f"\n=== Random Forest results ===")
    print("Train acc:", train_acc)
    print("Val acc:  ", val_acc)
    print("Test acc: ", test_acc)
    print("\nConfusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification report (test):")
    print(classification_report(y_test, y_test_pred, target_names=[str(c) for c in le.classes_]))
    print("R^2 (test) between numeric label values:", r2)
    return {
        "clf": clf,
        "le": le,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "r2": r2,
        "X_train": X_train, "y_train": y_train, "y_train_pred": y_train_pred,
        "X_val": X_val, "y_val": y_val, "y_val_pred": y_val_pred,
        "X_test": X_test, "y_test": y_test, "y_test_pred": y_test_pred,
        "case_names": case_names,
        "used_case_split": use_case_split
    }

# -------------------- Plots & saving --------------------

def ensure_dirs():
    results_dir = "Results/RandomForest"
    acc_dir = os.path.join(results_dir, "Accuracies")
    graph_dir = os.path.join(results_dir, "Graphs")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    return results_dir, acc_dir, graph_dir

def plot_confusion(cm, classes, outpath, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_accuracy_bar(train_acc, val_acc, test_acc, outpath):
    plt.figure(figsize=(5, 4))
    plt.bar(["Train", "Val", "Test"], [train_acc, val_acc, test_acc], color=["#90BFF9", "#FFC080", "#FFA0A0"])
    plt.title("Random Forest Accuracies")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_feature_importance(clf, n_top, outpath):
    try:
        fi = clf.feature_importances_
        idx = np.argsort(fi)[::-1][:n_top]
        plt.figure(figsize=(8, 6))
        sns.barplot(x=fi[idx], y=[f"feat_{i}" for i in idx])
        plt.title("Top Feature Importances")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        print("[WARN] Could not plot feature importances:", e)

def plot_pred_vs_actual(y_true_orig, y_pred_orig, outpath, title="Predicted vs Actual"):
    # jitter so discrete classes are visible
    jitter = (np.random.rand(len(y_true_orig)) - 0.5) * 0.1
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true_orig + jitter, y_pred_orig + jitter, alpha=0.6)
    plt.plot([min(y_true_orig)-0.1, max(y_true_orig)+0.1], [min(y_true_orig)-0.1, max(y_true_orig)+0.1], color='r', linestyle='--')
    plt.xlabel("Actual numeric label")
    plt.ylabel("Predicted numeric label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_roc_multiclass(clf, X, y_true, le, outpath, title="ROC (multiclass)"):
    try:
        probs = clf.predict_proba(X)
        classes = le.classes_
        y_bin = label_binarize(y_true, classes=np.arange(len(classes)))
        plt.figure(figsize=(6, 5))
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")
        plt.plot([0,1],[0,1],'k--')
        plt.title(title)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend(loc='lower right', fontsize='small')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
    except Exception as e:
        print("[WARN] Could not plot ROC:", e)

def compute_and_plot_oob_curve(X_train, y_train, graph_outpath, max_trees=200, step=10):
    """
    Incrementally build RandomForest with warm_start=True to record OOB error curve.
    This is somewhat slower but gives a useful curve.
    """
    try:
        # incremental classifier
        rf = RandomForestClassifier(warm_start=True, oob_score=True, n_jobs=-1, class_weight='balanced', random_state=42)
        oob_scores = []
        n_steps = list(range(step, max_trees + 1, step))
        for n in n_steps:
            rf.n_estimators = n
            rf.fit(X_train, y_train)  # warm_start=True -> add trees
            # oob_score_ available only when bootstrap=True and oob_score=True and n_estimators > 1
            if hasattr(rf, "oob_score_"):
                oob_scores.append(rf.oob_score_)
            else:
                oob_scores.append(None)
        # plot
        plt.figure(figsize=(6, 4))
        plt.plot(n_steps[:len(oob_scores)], [1 - s if s is not None else np.nan for s in oob_scores], marker='o')
        plt.xlabel("n_estimators")
        plt.ylabel("OOB Error (1 - score)")
        plt.title("OOB Error Curve")
        plt.tight_layout()
        plt.savefig(graph_outpath)
        plt.close()
    except Exception as e:
        print("[WARN] Could not compute OOB curve:", e)

def save_proportions_csv(y_true, y_pred, case_names_subset, out_csv_path, le):
    """
    Save proportions by case to CSV. case_names_subset corresponds to the entries in y_true/y_pred.
    Expects y_true/y_pred to be encoded ints (0..n_classes-1).
    """
    df_list = []
    classes = [str(c) for c in le.classes_]
    for case in np.unique(case_names_subset):
        mask = case_names_subset == case
        if mask.sum() == 0:
            continue
        counts = np.bincount(y_pred[mask], minlength=len(classes))
        props = counts / counts.sum()
        row = {"Case": case}
        for cl, p in zip(classes, props):
            row[str(cl)] = p
        df_list.append(row)
    if not df_list:
        print("[WARN] No proportions to save.")
        return
    df_out = pd.DataFrame(df_list).set_index("Case").sort_index()
    df_out.to_csv(out_csv_path)
    print(f"[SAVED] Proportions CSV → {out_csv_path}")
    return df_out

def plot_proportions_stacked(df_proportions, outpath, title="Proportions by Case"):
    # df_proportions: rows=cases, columns=class labels (strings)
    if df_proportions is None or df_proportions.empty:
        return
    # style similar to your provided Fusion plot: horizontal stacked bars
    colors = ['#90BFF9', '#FFC080', '#FFA0A0']  # progressive, stable, responsive
    ax = df_proportions.plot(kind='barh', stacked=True, figsize=(8, max(4, 0.25 * len(df_proportions))), color=colors)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Case")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()

def save_accuracies_csv(train_acc, val_acc, test_acc, r2, outpath):
    df = pd.DataFrame({
        "Train Accuracy": [train_acc],
        "Validation Accuracy": [val_acc],
        "Test Accuracy": [test_acc],
        "R2 Test": [r2]
    })
    df.to_csv(outpath, index=False)
    print(f"[SAVED] Accuracies CSV → {outpath}")

def save_classification_report(y_true, y_pred, le, outpath):
    try:
        report = classification_report(y_true, y_pred, target_names=[str(c) for c in le.classes_], output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.to_csv(outpath)
        print(f"[SAVED] Classification report → {outpath}")
    except Exception as e:
        print("[WARN] Could not save classification report:", e)

# -------------------- Main flow --------------------

if __name__ == "__main__":
    results_dir, acc_dir, graph_dir = ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    X, y, case_names = build_Xy_for_rf(SEQ_DATASET_PATH, TRACK_DATASET_PATH, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    if X.size == 0:
        raise SystemExit("[ERROR] No samples created for RF. Check matching logic and dataset paths.")

    res = train_eval_rf(X, y, case_names, TEST_TRAIN_SPLIT_ANNOTATION_PATH)
    clf = res["clf"]
    le = res["le"]

    # save accuracies CSV
    acc_csv = os.path.join(acc_dir, f"rf_accuracies_{timestamp}.csv")
    save_accuracies_csv(res["train_acc"], res["val_acc"], res["test_acc"], res["r2"], acc_csv)

    # save classification reports
    save_classification_report(res["y_test"], res["y_test_pred"], le, os.path.join(graph_dir, f"classification_report_test_{timestamp}.csv"))
    save_classification_report(res["y_train"], res["y_train_pred"], le, os.path.join(graph_dir, f"classification_report_train_{timestamp}.csv"))

    # Confusion matrices
    cm_test = confusion_matrix(res["y_test"], res["y_test_pred"])
    plot_confusion(cm_test, [str(c) for c in le.classes_], os.path.join(graph_dir, f"confusion_matrix_test_{timestamp}.png"), title="Confusion Matrix (Test)")
    cm_train = confusion_matrix(res["y_train"], res["y_train_pred"])
    plot_confusion(cm_train, [str(c) for c in le.classes_], os.path.join(graph_dir, f"confusion_matrix_train_{timestamp}.png"), title="Confusion Matrix (Train)")

    # Accuracy bar
    plot_accuracy_bar(res["train_acc"], res["val_acc"], res["test_acc"], os.path.join(graph_dir, f"accuracy_bar_{timestamp}.png"))

    # Feature importance (top 30 or all)
    plot_feature_importance(clf, n_top=min(30, X.shape[1]), outpath=os.path.join(graph_dir, f"feature_importance_{timestamp}.png"))

    # Predicted vs Actual scatter (use original numeric labels)
    try:
        y_test_orig = le.inverse_transform(res["y_test"])
        y_test_pred_orig = le.inverse_transform(res["y_test_pred"])
        plot_pred_vs_actual(y_test_orig.astype(float), y_test_pred_orig.astype(float), os.path.join(graph_dir, f"pred_vs_actual_test_{timestamp}.png"), title="Predicted vs Actual (Test)")
    except Exception as e:
        print("[WARN] Could not make predicted vs actual plot:", e)

    # ROC curves (train & test)
    try:
        plot_roc_multiclass(clf, res["X_test"], res["y_test"], le, os.path.join(graph_dir, f"roc_test_{timestamp}.png"), title="ROC (Test)")
        plot_roc_multiclass(clf, res["X_train"], res["y_train"], le, os.path.join(graph_dir, f"roc_train_{timestamp}.png"), title="ROC (Train)")
    except Exception as e:
        print("[WARN] ROC plotting failed:", e)

    # OOB curve (incremental)
    compute_and_plot_oob_curve(res["X_train"], res["y_train"], os.path.join(graph_dir, f"oob_curve_{timestamp}.png"), max_trees=200, step=20)

    # ---- PROPORTIONS (Case-based stacked bar chart + CSV) ----
    if res["used_case_split"]:
        print("[INFO] Saving proportions by case...")

        # Split case_names into train/test using *the same indices* used in splitting y/X
        case_names_arr = res["case_names"]

        # new indices based on original masks applied before splitting
        train_case_mask = np.array([cn in split_by_case(TEST_TRAIN_SPLIT_ANNOTATION_PATH)[0] for cn in case_names_arr])
        test_case_mask = np.array([cn in split_by_case(TEST_TRAIN_SPLIT_ANNOTATION_PATH)[1] for cn in case_names_arr])

        # Extract case names *aligned with predictions*
        case_names_train = case_names_arr[train_case_mask][:len(res["y_train"])]
        case_names_test = case_names_arr[test_case_mask][:len(res["y_test"])]

        # Generate CSV + plot
        df_train_prop = save_proportions_csv(
            res["y_train"], res["y_train_pred"], case_names_train,
            os.path.join(graph_dir, f"rf_train_proportions_{timestamp}.csv"), le
        )

        df_test_prop = save_proportions_csv(
            res["y_test"], res["y_test_pred"], case_names_test,
            os.path.join(graph_dir, f"rf_test_proportions_{timestamp}.csv"), le
        )

        plot_proportions_stacked(
            df_test_prop,
            os.path.join(graph_dir, f"proportions_by_case_test_{timestamp}.png"),
            title="Predicted Proportions by Case (Test Set)"
        )

        plot_proportions_stacked(
            df_train_prop,
            os.path.join(graph_dir, f"proportions_by_case_train_{timestamp}.png"),
            title="Predicted Proportions by Case (Train Set)"
        )

    else:
        print("[INFO] Case-based split was not used. Skipping case proportion plots.")

    print("\n✅ Done. Results and graphs saved in Results/RandomForest/")
