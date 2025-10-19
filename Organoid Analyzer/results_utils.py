import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, auc, r2_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
import torch
from torch.utils.data import DataLoader
matplotlib.use('Agg')


def compute_metrics(y_true, preds, probs):
    acc = np.mean(preds == y_true)
    f1 = f1_score(y_true, preds, average="macro")

    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    try:
        auc_value = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovo")
    except:
        auc_value = -1
    return acc, f1, auc_value, y_true_bin


def plot_roc(y_true_bin, probs, result_path, n_classes=3):
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig(os.path.join(result_path,"roc_curve.png"))
    plt.close()


def plot_confusion_matrix(y_true, preds, classes, result_path):
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    os.makedirs(result_path, exist_ok=True)
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    plt.close()
    return cm


def fusion_weight_analysis(model, test_loader, device, result_path):
    def evaluate_model(model, dataloader, alpha, device):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_seq, batch_track, batch_y in dataloader:
                batch_seq, batch_track, batch_y = batch_seq.to(device), batch_track.to(device), batch_y.to(device)
                outputs = model(batch_seq, batch_track, lstm_weight=alpha)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        return correct / total
    alphas = np.linspace(0, 1, 21)
    accuracies = []
    for a in alphas:
        acc = evaluate_model(model, test_loader, a, device=device)
        accuracies.append(acc)
    plt.plot(alphas, accuracies, marker='o')
    plt.xlabel("LSTM Weighting")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Sequence/Track Mixing Ratio")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "Fusion Weights.png"))
    plt.close()


def compute_case_proportions(model, dataset, device, batch_size, result_path):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    prefix_dict = {}
    for batch_seq, batch_track, batch_prefix_tid in loader:
        batch_seq, batch_track = batch_seq.to(device), batch_track.to(device)
        logits = model(batch_seq, batch_track)
        pred = logits.argmax(dim=1)
        for i in range(len(batch_prefix_tid)):
            case_id = "_".join(batch_prefix_tid[i].split('_')[:2])
            if case_id not in prefix_dict:
                prefix_dict[case_id] = [0, 0, 0]
            prefix_dict[case_id][pred[i]] += 1

    df = pd.DataFrame(prefix_dict).transpose()
    df.columns = ['Progressive', 'Stable', 'Responsive']
    df = df.div(df.sum(axis=1), axis=0).sort_index()

    ax = df.plot(kind='barh', stacked=True,
                 title='T cell Proportions by Case',
                 color=['#90BFF9', '#FFC080', '#FFA0A0'])
    for patch in ax.patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(1)
    plt.savefig(os.path.join(result_path, "proportions_by_case.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return df

def correlate_with_size_change(df, annotations_path, result_path):
    size_df = pd.read_excel(annotations_path)
    size_dict = size_df.set_index("Case")["Size Change"].to_dict()

    x, y = [], []
    for case_name in df.index:
        y.append(df.loc[case_name, "Combined Score"])
        x.append(size_dict[case_name])

    x, y = np.array(x), np.array(y)
    df["Size Change"] = x
    df.to_csv(os.path.join(result_path, "proportions.csv"), index=True)

    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b
    r2 = r2_score(y, y_pred)

    plt.scatter(x, y)
    plt.plot(x, y_pred, color="red", label=f"Best fit (R²={r2:.2f})")
    plt.xlabel("Change in PDO size")
    plt.ylabel("Score")
    plt.title("Score by Change in PDO size")
    plt.legend()
    plt.savefig(os.path.join(result_path, "Score by Change in PDO size.png"), dpi=300, bbox_inches="tight")
    plt.close()
    return r2

def plot_loss_curve(train_losses, val_losses, test_losses, results_path):
    print("[STEP 2] Drawing Loss Graph...")
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    #plt.plot(test_losses, label="Test Loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(f"{results_path}/loss_curve.png")
    plt.close()
    print("[STEP 2] Finished Drawing Loss Graph...")

def plot_accuracies(train_accs, val_accs, test_accs, results_path):
    print("[STEP 2] Drawing Validation Accuracy Graph...")
    plt.plot(np.array(train_accs) * 100)
    plt.plot(np.array(val_accs) * 100)
    plt.plot(np.array(test_accs) * 100)
    plt.title("Validation Accuracy (%)")
    plt.savefig(f"{results_path}/val_accuracy.png")
    plt.close()
    print("[STEP 2] Finished Drawing Validation Accuracy Graph...")
