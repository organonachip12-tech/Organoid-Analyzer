import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    r2_score,
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")


def compute_metrics(y_true, preds, probs):
    # Always defined values
    acc = np.mean(preds == y_true)
    f1 = f1_score(y_true, preds, average="macro")

    # Always create 3-column binarized labels
    classes = np.array([0, 1, 2])
    y_true_bin = label_binarize(y_true, classes=classes)

    # If fewer than 3 classes exist → pad columns
    if y_true_bin.shape[1] < len(classes):
        missing = len(classes) - y_true_bin.shape[1]
        y_true_bin = np.hstack([y_true_bin, np.zeros((len(y_true_bin), missing))])

    # Safe AUC
    try:
        auc_value = roc_auc_score(
            y_true_bin,
            probs[:, :3],  # force to 3 columns
            average="macro",
            multi_class="ovo",
        )
    except Exception:
        auc_value = -1

    return acc, f1, auc_value, y_true_bin


def plot_roc(y_true_bin, probs, result_path, n_classes=3):
    if probs.shape[1] < n_classes:
        pad = n_classes - probs.shape[1]
        probs = np.hstack([probs, np.zeros((probs.shape[0], pad))])

    os.makedirs(result_path, exist_ok=True)
    plt.figure()

    if y_true_bin.sum() == 0 or y_true_bin.shape[1] == 1:
        plt.text(
            0.5,
            0.5,
            "ROC Not Applicable\n(single-class test set)",
            horizontalalignment="center",
            fontsize=14,
        )
        plt.savefig(os.path.join(result_path, "roc_curve.png"))
        plt.close()
        return

    for i in range(n_classes):
        try:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
        except Exception:
            continue

    plt.plot([0, 1], [0, 1], "k--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(result_path, "roc_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, preds, classes, result_path):
    os.makedirs(result_path, exist_ok=True)
    cm = confusion_matrix(y_true, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()
    return cm


def fusion_weight_analysis(model, test_loader, device, result_path):
    def evaluate_model(m, dataloader, alpha, device):
        m.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in dataloader:
                batch_seq = batch_seq.to(device)
                batch_track = batch_track.to(device)
                batch_y = batch_y.to(device)

                outputs = m(batch_seq, batch_track, lstm_weight=alpha)
                preds = outputs.argmax(dim=1)

                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total if total > 0 else 0.0

    os.makedirs(result_path, exist_ok=True)
    alphas = np.linspace(0, 1, 21)
    accuracies = [evaluate_model(model, test_loader, a, device) for a in alphas]

    plt.figure()
    plt.plot(alphas, accuracies, marker="o")
    plt.xlabel("LSTM Weighting")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sequence/Track Mixing Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "fusion_weights.png"))
    plt.close()


def compute_case_proportions(model, dataset, device, batch_size, result_path):
    os.makedirs(result_path, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    prefix_dict = {}

    model.eval()
    with torch.no_grad():
        for batch_seq, batch_track, batch_case_names in loader:
            batch_seq = batch_seq.to(device)
            batch_track = batch_track.to(device)

            logits = model(batch_seq, batch_track)
            preds = logits.argmax(dim=1)

            for i in range(len(batch_case_names)):
                case = str(batch_case_names[i])
                if case not in prefix_dict:
                    prefix_dict[case] = [0, 0, 0]
                prefix_dict[case][preds[i].item()] += 1

    df = pd.DataFrame(prefix_dict).T
    df.columns = ["Progressive", "Stable", "Responsive"]
    df = df.div(df.sum(axis=1), axis=0).sort_index()

    plt.figure(figsize=(8, 12))
    ax = df.plot(
        kind="barh",
        stacked=True,
        color=["#90BFF9", "#FFC080", "#FFA0A0"],
        title="T cell Proportions by Case",
    )
    for p in ax.patches:
        p.set_edgecolor("black")
        p.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "proportions_by_case.png"))
    plt.close()

    return df


def correlate_with_size_change(df, annotations_path, result_path):
    os.makedirs(result_path, exist_ok=True)

    size_df = pd.read_excel(annotations_path)
    size_dict = {
        str(k).upper(): v
        for k, v in size_df.set_index("Case")["Size Change"].to_dict().items()
    }

    x_vals, y_vals, used_cases = [], [], []

    for case in df.index:
        key = case.upper()
        if key not in size_dict:
            print(f"[WARNING] Case '{case}' missing from Annotations.xlsx — skipping.")
            continue

        x_vals.append(size_dict[key])
        y_vals.append(df.loc[case, "Combined Score"])
        used_cases.append(case)

    if not x_vals:
        print("[WARNING] No overlapping cases found.")
        return float("nan")

    x = np.array(x_vals)
    y = np.array(y_vals)

    df_out = df.copy()
    df_out.loc[used_cases, "Size Change"] = x
    df_out.to_csv(os.path.join(result_path, "proportions.csv"), index=True)

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    r2 = r2_score(y, y_pred)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(x, y_pred, color="red", label=f"Best fit (R²={r2:.2f})")
    plt.xlabel("Change in PDO size")
    plt.ylabel("Combined Score")
    plt.title("Score vs PDO Size Change")
    plt.legend()
    plt.savefig(os.path.join(result_path, "score_vs_size_change.png"))
    plt.close()

    return r2


def plot_loss_curve(train_losses, val_losses, test_losses, results_path):
    os.makedirs(results_path, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.plot(test_losses, label="Test")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(results_path, "loss_curve.png"))
    plt.close()


def plot_accuracies(train_accs, val_accs, test_accs, results_path):
    os.makedirs(results_path, exist_ok=True)
    plt.figure()
    plt.plot(np.array(train_accs) * 100, label="Train")
    plt.plot(np.array(val_accs) * 100, label="Val")
    plt.plot(np.array(test_accs) * 100, label="Test")
    plt.legend()
    plt.title("Accuracy (%)")
    plt.savefig(os.path.join(results_path, "accuracy_curve.png"))
    plt.close()
