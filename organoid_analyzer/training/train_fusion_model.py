import os
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from organoid_analyzer.config import (
    RESULTS_DIR,
    BATCH_SIZE,
    DROPOUT,
    MAX_EPOCHS,
    SEQ_DATASET_PATH,
    TRACK_DATASET_PATH,
    TEST_TRAIN_SPLIT_ANNOTATION_PATH,
    ABLATION_CONFIGS,
)

from organoid_analyzer.utils.results_utils import (
    plot_loss_curve,
    plot_accuracies,
    plot_roc,
    plot_confusion_matrix,
    fusion_weight_analysis,
    compute_case_proportions,
    correlate_with_size_change,
)
from organoid_analyzer.models.unified_fusion import UnifiedFusionModel
from organoid_analyzer.utils.shap_analysis import SHAP_UnifiedFusionModel


# ============================================================
# CASE NAME EXTRACTION (FROM PREFIX ONLY)
# ============================================================
def extract_case_name_for_training(raw_prefix: str) -> str:
    """
    Version used ONLY at training time.
    PREFIX is already normalized during dataset creation,
    so we simply strip XY suffixes.
    """
    p = raw_prefix.strip()

    # Remove _tracks
    if p.endswith("_tracks"):
        p = p[:-7]

    # Remove _XY##
    if "_XY" in p:
        p = p.split("_XY")[0]

    return p


def extract_case_name(raw_prefix: str) -> str:
    """
    Map a track PREFIX (as saved in the .npz) to the 'Case' name
    used in Annotations.xlsx.

    We only see values like:
        2nd_NCI6_XY3_tracks
        NYU318_CAF7_XY3_tracks
        NCI2_XY1_tracks
        NCI9_XY2_tracks
        NCI9_Round1_Stroma7_XY2_tracks
        Device5_XY7_tracks
    etc.

    Rules to match the annotation sheet you showed:

      2ND folder:
        2nd_NCI6_XY…      -> 2ND_NCI6
        2nd_NCI8_XY…      -> 2ND_NCI8
        2nd_NCI9_XY…      -> 2ND_NCI9
        2nd_NYU352_XY…    -> 2ND_NYU352
        2nd_NYU360_XY…    -> 2ND_NYU360

      CAF folder:
        NYU318_CAF1_XY…   -> CAF_NYU318_CAF1
        ...
        NYU318_CAF8_XY…   -> CAF_NYU318_CAF8

      CART folder:
        NCI2_XY…          -> CART_NCI2
        NCI6_XY…          -> CART_NCI6
        NCI8_XY…          -> CART_NCI8
        NCI9_XY…          -> CART_NCI9   (this is the CART_NCI9 case)
        NYU318_XY…        -> CART_NYU318
        NYU352_XY…        -> CART_NYU352
        NYU358_XY…        -> CART_NYU358
        NYU360_XY…        -> CART_NYU360

      NCI9 folder:
        NCI9_Round1_Stroma7_XY… -> NCI9_Round1_Stroma7
        NCI9_Round1_Stroma8_XY… -> NCI9_Round1_Stroma8
        NCI9_Round2_Stroma7_XY… -> NCI9_Round2_Stroma7
        NCI9_Round2_Stroma8_XY… -> NCI9_Round2_Stroma8

      PDO folder:
        Device1_XY…       -> PDO_Device1
        ...
        Device8_XY…       -> PDO_Device8
    """

    p = raw_prefix.strip()

    # strip "_tracks" if present
    if p.endswith("_tracks"):
        p = p[:-7]

    # strip "_XY<something>"
    if "_XY" in p:
        p = p.split("_XY")[0]

    base = p  # e.g. "2nd_NCI6", "NYU318_CAF7", "NCI2", "NCI9_Round1_Stroma7", "Device5"

    # 1) 2ND_* cases
    if base.lower().startswith("2nd_"):
        # 2nd_NCI6 -> 2ND_NCI6
        rest = base.split("_", 1)[1]
        return f"2ND_{rest}"

    # 2) CAF (NYU318_CAF#)
    #    base: NYU318_CAF1 → CAF_NYU318_CAF1
    if re.match(r"^NYU\d+_CAF\d+$", base, flags=re.IGNORECASE):
        # preserve original capitalization of base
        return f"CAF_{base}"

    up = base.upper()

    # 3) NCI9 "Round" stromal cases (keep as-is, no CART)
    #    e.g. NCI9_Round1_Stroma7
    if up.startswith("NCI9_ROUND"):
        # Return in the same form as the filenames / sheet
        # (your sheet uses NCI9_Round1_Stroma7 style, not all-caps)
        return base

    # 4) CART_NCI9 (the one in the CART folder with just 'NCI9_XY...')
    #    base: "NCI9" → CART_NCI9
    if up == "NCI9":
        return "CART_NCI9"

    # 5) CART NCI* (2, 6, 8) — plain NCI number
    if re.match(r"^NCI\d+$", up):
        # NCI2, NCI6, NCI8
        return f"CART_{up}"

    # 6) CART NYU* numeric
    if re.match(r"^NYU\d+$", up):
        # NYU360, NYU318, NYU352, NYU358
        return f"CART_{up}"

    # 7) PDO / Device
    if re.match(r"^DEVICE\d+$", up):
        # normalize to "Device#" for the suffix, but sheet uses "PDO_Device#"
        # If the filename is "Device2" or "device2", we want "PDO_Device2"
        # Extract numeric part:
        m = re.match(r"^DEVICE(\d+)$", up)
        num = m.group(1) if m else ""
        return f"PDO_Device{num}"

    # 8) Fallback: just return base
    return base


# ---------------------------------------------------------------------
# SEEDING / DEVICE
# ---------------------------------------------------------------------
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

UNIFIED_MODEL_PATH = os.path.join(RESULTS_DIR, "unified_model_best.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# SUBSET DATASET FOR CASE-LEVEL ANALYSIS
# ---------------------------------------------------------------------
class SubsetDataset(Dataset):
    def __init__(self, seq_path, track_path, annotations_path, case_identifier, transform=None):
        X_seq, X_track, y_matched, prefix_tid = select_specific_cases(
            seq_path, track_path, annotations_path, case_identifier
        )

        self.X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
        self.X_track = torch.tensor(np.array(X_track), dtype=torch.float32)
        self.prefix_tid = prefix_tid
        self.transform = transform

    def __len__(self):
        return len(self.prefix_tid)

    def __getitem__(self, idx):
        seq = self.X_seq[idx]
        track = self.X_track[idx]
        prefix_tid = self.prefix_tid[idx]

        if self.transform:
            seq, track = self.transform((seq, track))

        return seq, track, prefix_tid


def select_specific_cases(seq_path, track_path, annotations_path, case_identifier):
    """
    Build a dataset containing ONLY the cases with Train or Test == case_identifier.
    We use the annotation's Case column and match with extract_case_name(PREFIX).
    """

    annotations_df = pd.read_excel(annotations_path)

    specific_cases = {
        str(c).strip().upper()
        for c in annotations_df.loc[
            annotations_df["Train or Test"] == case_identifier, "Case"
        ].tolist()
    }

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = (
        seq_data["X"],
        seq_data["y"],
        seq_data["track_ids"],
    )
    X_track, y_track, track_ids_track = (
        track_data["X"],
        track_data["y"],
        track_data["track_ids"],
    )

    # fix orientation if needed (11 x 20 -> 20 x 11)
    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        X_seq = np.transpose(X_seq, (0, 2, 1))

    # map (PREFIX, TRACK_ID) -> index in track dataset
    track_id_to_index = {tuple(tid): i for i, tid in enumerate(track_ids_track)}

    X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid)
        if key not in track_id_to_index:
            continue

        prefix = tid[0]
        case_name = extract_case_name(prefix)
        case_key = case_name.strip().upper()

        if case_key in specific_cases:
            idx = track_id_to_index[key]
            X_seq_matched.append(X_seq[i])
            X_track_matched.append(X_track[idx])
            y_matched.append(y_seq[i])
            prefix_tid.append(case_name)

    return X_seq_matched, X_track_matched, y_matched, prefix_tid


# ---------------------------------------------------------------------
# TRAIN/TEST SPLIT BY CASE (FROM ANNOTATION SHEET)
# ---------------------------------------------------------------------
def train_test_split_by_case(seq_path, track_path, annotation_path):
    annotations_df = pd.read_excel(annotation_path)

    train_cases = {
        str(c).strip().upper()
        for c in annotations_df.loc[annotations_df["Train or Test"] == 0, "Case"].tolist()
    }
    test_cases = {
        str(c).strip().upper()
        for c in annotations_df.loc[annotations_df["Train or Test"] == 1, "Case"].tolist()
    }

    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data["X"], seq_data["y"], seq_data["track_ids"]
    X_track, y_track, track_ids_track = (
        track_data["X"],
        track_data["y"],
        track_data["track_ids"],
    )

    # fix orientation if needed
    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {tuple(tid): i for i, tid in enumerate(track_ids_track)}

    X_seq_train, X_seq_test = [], []
    X_track_train, X_track_test = [], []
    y_train, y_test = [], []

    train_used = set()
    test_used = set()
    excluded_cases = defaultdict(int)

    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid)
        if key not in track_id_to_index:
            excluded_cases["NO_TRACK_MATCH"] += 1
            continue

        prefix = tid[0]
        case_name = extract_case_name(prefix)
        case_key = case_name.strip().upper()
        idx = track_id_to_index[key]

        if case_key in test_cases:
            X_seq_test.append(X_seq[i])
            X_track_test.append(X_track[idx])
            y_test.append(y_seq[i])
            test_used.add(case_key)

        elif case_key in train_cases:
            X_seq_train.append(X_seq[i])
            X_track_train.append(X_track[idx])
            y_train.append(y_seq[i])
            train_used.add(case_key)

        else:
            excluded_cases[case_key] += 1

    print("[DEBUG] Matched train pairs:", len(X_seq_train))
    print("[DEBUG] Matched test  pairs:", len(X_seq_test))

    print("Train cases actually used:", sorted(train_used))
    print("Test  cases actually used:", sorted(test_used))
    print("Train cases from sheet (0):", sorted(train_cases))
    print("Test  cases from sheet (1):", sorted(test_cases))

    missing_train = sorted(train_cases - train_used)
    missing_test = sorted(test_cases - test_used)
    if missing_train:
        print("[DEBUG] Annotation TRAIN cases with no matched tracks:", missing_train)
    if missing_test:
        print("[DEBUG] Annotation TEST cases with no matched tracks:", missing_test)

    # Cases seen in data but not present in the annotation sheet
    extra_cases = sorted(
        {c for c in excluded_cases.keys() if c != "NO_TRACK_MATCH"}
    )
    if extra_cases:
        print("[DEBUG] Cases in data but NOT in annotation sheet (and track counts):")
        for c in extra_cases:
            print(f"   {c}: {excluded_cases[c]} tracks")
    if excluded_cases.get("NO_TRACK_MATCH", 0) > 0:
        print(
            "[DEBUG] Tracks with no matching entry in track_ids_track:",
            excluded_cases["NO_TRACK_MATCH"],
        )

        # ======================================================================
        # EXTENDED DEBUGGING – TRACK USAGE ANALYSIS
        # ======================================================================

        print("\n" + "="*80)
        print("           EXTENDED DATASET DEBUGGING REPORT")
        print("="*80)

        # 1. Count raw tracks per case in track_ids_track
        raw_track_counts = defaultdict(int)
        for tid in track_ids_track:
            cname = extract_case_name_for_training(tid[0]).upper()
            raw_track_counts[cname] += 1

        # 2. Count raw sequence windows per case in track_ids_seq
        raw_seq_counts = defaultdict(int)
        for tid in track_ids_seq:
            cname = extract_case_name_for_training(tid[0]).upper()
            raw_seq_counts[cname] += 1

        # 3. Count matched seq-track pairs per case
        matched_counts = defaultdict(int)
        for i, tid in enumerate(track_ids_seq):
            key = tuple(tid)
            if key not in track_id_to_index:
                continue
            cname = extract_case_name_for_training(tid[0]).upper()
            matched_counts[cname] += 1

        # 4. Show table per case
        print("\nPER-CASE TRACK/SEQ/MATCH REPORT")
        print("-"*80)
        all_cases = sorted(set(list(raw_track_counts.keys()) +
                            list(raw_seq_counts.keys()) +
                            list(matched_counts.keys()) +
                            list(train_cases) +
                            list(test_cases)))

        for c in all_cases:
            print(f"{c:30s} | "
                f"Tracks(raw): {raw_track_counts.get(c,0):5d} | "
                f"Seq(raw): {raw_seq_counts.get(c,0):5d} | "
                f"Matched: {matched_counts.get(c,0):5d} | "
                f"Train? {c in train_cases} | Test? {c in test_cases}")

        # 5. Summaries
        total_tracks = sum(raw_track_counts.values())
        total_seq = sum(raw_seq_counts.values())
        total_matched = sum(matched_counts.values())

        print("\nSUMMARY:")
        print(f"  Total raw tracks:               {total_tracks}")
        print(f"  Total raw seq windows:          {total_seq}")
        print(f"  Total matched seq-track pairs:  {total_matched}")
        print(f"  → Match percentage:             {100*total_matched/max(1,total_seq):.2f}%")

        print("="*80 + "\n")


    return (
        np.array(X_seq_train),
        np.array(X_seq_test),
        np.array(X_track_train),
        np.array(X_track_test),
        np.array(y_train),
        np.array(y_test),
    )


# ---------------------------------------------------------------------
# INFERENCE
# ---------------------------------------------------------------------
def run_inference(model, X_seq, X_track, device):
    model.eval()
    with torch.no_grad():
        logits = model(X_seq.to(device), X_track.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs


# ---------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------------------------
def Train_UnifiedFusionModel(
    seq_path,
    track_path,
    result_path,
    test_train_split_annotation_path,
    seq_input_size=8,   # sequence feature count
    track_input_size=3, # track-level feature count
    hidden_size=128,
    fusion_size=128,
    dropout=0.5,
    model_save_path="",
    test_prefix="no_prefix",
):

    print("[STEP 1] Loading data using Annotation Sheet Split (Train=0, Test=1)...")
    (
        X_seq_train_total,
        X_seq_test,
        X_track_train_total,
        X_track_test,
        y_train_total,
        y_test_original,
    ) = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path)

    # Encode labels (0, 0.5, 1) → 0,1,2
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_total)

    train_classes = set(le.classes_)
    test_classes = set(y_test_original)
    unknown_labels = test_classes - train_classes
    if unknown_labels:
        print("Unknown labels found in test set but not in train:", unknown_labels)

    # Train/val split (within annotated TRAIN cases)
    (
        X_seq_train,
        X_seq_val,
        X_track_train,
        X_track_val,
        y_train,
        y_val,
    ) = train_test_split(
        X_seq_train_total,
        X_track_train_total,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # Tensor conversion
    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_val = torch.tensor(X_seq_val, dtype=torch.float32)
    X_seq_test = torch.tensor(X_seq_test, dtype=torch.float32)

    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_val = torch.tensor(X_track_val, dtype=torch.float32)
    X_track_test = torch.tensor(X_track_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    y_test_tensor = torch.tensor(le.transform(y_test_original), dtype=torch.long)

    train_dataset = TensorDataset(X_seq_train, X_track_train, y_train_tensor)
    val_dataset = TensorDataset(X_seq_val, X_track_val, y_val_tensor)
    test_dataset = TensorDataset(X_seq_test, X_track_test, y_test_tensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Class weights for imbalance
    classes, counts = np.unique(y_train, return_counts=True)
    print("Class counts in TRAIN (encoded):")
    for cls, count in zip(classes, counts):
        print(f"  Class {cls}: {count} samples")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class Weights:", weights)

    # Model
    model = UnifiedFusionModel(
        seq_input_size=seq_input_size,
        track_input_size=track_input_size,
        hidden_size=hidden_size,
        fusion_size=fusion_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )

    print("[STEP 2] Training unified fusion model...")
    early_stop = 0
    lowest_val_loss = float("inf")
    best_model_state = None

    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    test_accs, test_losses = [], []

    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    for epoch in range(MAX_EPOCHS):
        # ----------------- TRAIN -----------------
        model.train()
        correct_train, total_train, train_loss_sum = 0, 0, 0.0

        for batch_seq, batch_track, batch_y in train_loader:
            batch_seq, batch_track, batch_y = (
                batch_seq.to(device),
                batch_track.to(device),
                batch_y.to(device),
            )
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(batch_seq, batch_track)
                    loss = criterion(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            train_loss_sum += loss.item()
            preds = logits.argmax(dim=1)
            correct_train += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)

        train_loss = train_loss_sum / max(1, len(train_loader))
        train_acc = correct_train / max(1, total_train)

        # ----------------- VAL + TEST -----------------
        model.eval()
        correct_val = correct_test = 0
        val_loss_sum = test_loss_sum = 0.0
        total_val = total_test = 0

        with torch.no_grad():
            # VAL
            for batch_seq, batch_track, batch_y in val_loader:
                batch_seq, batch_track, batch_y = (
                    batch_seq.to(device),
                    batch_track.to(device),
                    batch_y.to(device),
                )
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)

                val_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                correct_val += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)

            # TEST (held-out cases from annotation sheet)
            for batch_seq, batch_track, batch_y in test_loader:
                batch_seq, batch_track, batch_y = (
                    batch_seq.to(device),
                    batch_track.to(device),
                    batch_y.to(device),
                )
                logits = model(batch_seq, batch_track)
                loss = criterion(logits, batch_y)

                test_loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                correct_test += (preds == batch_y).sum().item()
                total_test += batch_y.size(0)

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = correct_val / max(1, total_val)

        test_loss = test_loss_sum / max(1, len(test_loader))
        test_acc = correct_test / max(1, total_test)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.4f} | "
                f"Test Loss={test_loss:.4f} Acc={test_acc:.4f}"
            )
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                print("LR:", scheduler.optimizer.param_groups[0]["lr"])

        # Early stopping on validation loss
        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= 60:
                print("Early stopping triggered.")
                break

    # ------------------------------------------------------------------
    # SAVE TRAINING CURVES
    # ------------------------------------------------------------------
    train_results_path = os.path.join(
        result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results"
    )
    os.makedirs(train_results_path, exist_ok=True)

    model.load_state_dict(best_model_state)

    lstm_weight_norm = sum(
        p.norm().item()
        for n, p in model.lstm.named_parameters()
        if "weight" in n
    )
    track_weight_norm = sum(
        p.norm().item()
        for n, p in model.track_fc.named_parameters()
        if "weight" in n
    )
    print("LSTM weight norm:", lstm_weight_norm)
    print("Track weight norm:", track_weight_norm)

    np.savez(
        f"{train_results_path}/training_logs_unified.npz",
        train_losses=train_losses,
        train_accs=train_accs,
        val_losses=val_losses,
        val_accs=val_accs,
        test_losses=test_losses,
        test_accuracies=test_accs,
    )

    plot_loss_curve(train_losses, val_losses, test_losses, train_results_path)
    plot_accuracies(train_accs, val_accs, test_accs, train_results_path)

    print("[STEP 3] Final evaluation on TRAIN / VAL / TEST...")

    # ---------------- TRAIN METRICS ----------------
    preds_train, probs_train = run_inference(model, X_seq_train, X_track_train, device)
    from organoid_analyzer.utils.results_utils import compute_metrics  # local import

    best_train_acc, f1_train, auc_train, y_train_bin = compute_metrics(
        y_train, preds_train, probs_train
    )

    print(
        f"[TRAIN] Accuracy: {best_train_acc:.4f}, F1: {f1_train:.4f}, AUC: {auc_train:.4f}"
    )

    plot_roc(y_train_bin, probs_train, train_results_path, n_classes=3)

    cm_train = plot_confusion_matrix(
        y_train, preds_train, le.classes_, train_results_path
    )

    fusion_weight_analysis(model, train_loader, device, train_results_path)

    df_train = compute_case_proportions(
        model,
        SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 0),
        device,
        BATCH_SIZE,
        train_results_path,
    )
    df_train["Combined Score"] = (
        df_train["Progressive"] * 0.0
        + df_train["Stable"] * 0.5
        + df_train["Responsive"] * 1.0
    )
    r2_train = correlate_with_size_change(
        df_train, test_train_split_annotation_path, train_results_path
    )
    print(f"[TRAIN] R² correlation with size change = {r2_train:.3f}")

    # ---------------- VAL METRICS ----------------
    val_results_path = os.path.join(
        result_path,
        f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results/validation_",
    )
    os.makedirs(val_results_path, exist_ok=True)

    preds_val, probs_val = run_inference(model, X_seq_val, X_track_val, device)
    best_val_acc, f1_val, auc_val, y_val_bin = compute_metrics(
        y_val, preds_val, probs_val
    )

    print(
        f"[VAL] Accuracy: {best_val_acc:.4f}, F1: {f1_val:.4f}, AUC: {auc_val:.4f}"
    )
    plot_roc(y_val_bin, probs_val, val_results_path, n_classes=3)
    cm_val = plot_confusion_matrix(
        y_val, preds_val, le.classes_, val_results_path
    )

    # ---------------- TEST METRICS ----------------
    test_results_path = os.path.join(
        result_path, f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/test results"
    )
    os.makedirs(test_results_path, exist_ok=True)

    preds_test, probs_test = run_inference(model, X_seq_test, X_track_test, device)
    best_test_acc, f1_test, auc_test, y_test_bin = compute_metrics(
        le.transform(y_test_original), preds_test, probs_test
    )

    print(
        f"[TEST] Accuracy: {best_test_acc:.4f}, F1: {f1_test:.4f}, AUC: {auc_test:.4f}"
    )
    print(
        classification_report(
            le.transform(y_test_original),
            preds_test,
            target_names=[str(cls) for cls in le.classes_],
        )
    )

    plot_roc(y_test_bin, probs_test, test_results_path, n_classes=3)
    cm_test = plot_confusion_matrix(
        le.transform(y_test_original),
        preds_test,
        le.classes_,
        test_results_path,
    )

    fusion_weight_analysis(model, test_loader, device, test_results_path)

    df_test = compute_case_proportions(
        model,
        SubsetDataset(seq_path, track_path, test_train_split_annotation_path, 1),
        device,
        BATCH_SIZE,
        test_results_path,
    )
    df_test["Combined Score"] = (
        df_test["Progressive"] * 0.0
        + df_test["Stable"] * 0.5
        + df_test["Responsive"] * 1.0
    )
    r2_test = correlate_with_size_change(
        df_test, test_train_split_annotation_path, test_results_path
    )
    print(f"[TEST] R² correlation with size change = {r2_test:.3f}")

    torch.save(
        model.state_dict(),
        os.path.join(
            train_results_path, f"hidden{hidden_size}_fusion{fusion_size}.pth"
        ),
    )
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print("Model saved to", model_save_path)

    return {
        "f1_score": f1_test,
        "auc": auc_test,
        "confusion_matrix": cm_test.tolist(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
        "train_accuracy": train_accs,
        "val_accuracy": val_accs,
        "test_accuracy": test_accs,
        "best_train_acc": best_train_acc,
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }


# ---------------------------------------------------------------------
# Test unified model (used for external evaluation)
# ---------------------------------------------------------------------
def Test_UnifiedFusionModel(
    seq_path,
    track_path,
    model_path,
    test_train_split_annotation_path,
    results_dir="test",
    seq_input_size=8,
    track_input_size=3,
    hidden_size=128,
    fusion_size=128,
    dropout=0.5,
):
    print("[TEST] Loading external test dataset...")

    (
        X_seq_train,
        X_seq_test,
        X_track_train,
        X_track_test,
        y_train,
        y_test_original,
    ) = train_test_split_by_case(seq_path, track_path, test_train_split_annotation_path)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test_original)

    X_seq_tensor = torch.tensor(X_seq_test, dtype=torch.float32).to(device)
    X_track_tensor = torch.tensor(X_track_test, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_test_enc, dtype=torch.long).to(device)

    test_dataset = TensorDataset(X_seq_tensor, X_track_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = UnifiedFusionModel(
        seq_input_size=seq_input_size,
        track_input_size=track_input_size,
        hidden_size=hidden_size,
        fusion_size=fusion_size,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_results_path = os.path.join(
        results_dir, f"test results/hidden_{hidden_size}/fusion_{fusion_size}"
    )
    os.makedirs(test_results_path, exist_ok=True)

    from utils.results_utils import compute_metrics
    preds, probs = run_inference(model, X_seq_tensor, X_track_tensor, device)
    acc, f1, auc_value, y_test_bin = compute_metrics(y_test_enc, preds, probs)

    print(f"[RESULT] Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc_value:.4f}")

    plot_roc(y_test_bin, probs, test_results_path, n_classes=3)
    cm = plot_confusion_matrix(y_test_enc, preds, le.classes_, test_results_path)

    return {
        "f1_score": f1,
        "auc": auc_value,
        "confusion_matrix": cm.tolist(),
        "acc": acc,
    }


# ---------------------------------------------------------------------
# Full ablation loop (used by train_and_test_models.py)
# ---------------------------------------------------------------------
def train_models_and_shap(
    ablation_configs,
    seq_dataset_path,
    track_dataset_path,
    test_train_split_annotation_path,
    max_pow_hidden,
    max_pow_fusion,
    min_pow_hidden,
    min_pow_fusion,
    perform_SHAP_analysis=True,
    model_type="fusion",
    dataset="all",
):

    # Log model and dataset selection
    print(f"[CONFIG] Model type: {model_type}")
    print(f"[CONFIG] Dataset: {dataset}")
    
    # Note: model_type and dataset selection are logged but full implementation 
    # of different model types (e.g., random_forest) requires additional work.
    # Currently supports: fusion model with all datasets combined.
    if model_type == "random_forest":
        print("[WARNING] Random Forest model not yet fully implemented. Using fusion model instead.")
    if dataset != "all":
        print(f"[WARNING] Dataset filtering to '{dataset}' not yet fully implemented. Using all datasets.")

    summary_train_acc = {}
    summary_val_acc = {}
    summary_test_acc = {}
    summary_r2_test = {}
    summary_loss_diff = {}

    for name, cfg in ablation_configs.items():
        print(f"\n===== Running Ablation: {name} =====")

        prefix = f"ablation_{name}"
        result_path = os.path.join(RESULTS_DIR, prefix)
        os.makedirs(result_path, exist_ok=True)

        seq_input_size = len(cfg["features"])
        track_input_size = len(cfg["track_features"])

        for fusion_pow in range(min_pow_fusion, max_pow_fusion + 1):
            fusion_size = 2 ** fusion_pow

            for hidden_pow in range(min_pow_hidden, max_pow_hidden):
                hidden_size = 2 ** hidden_pow
                print(f"Hidden Size: {hidden_size} | Fusion Size {fusion_size}")

                model_path = os.path.join(
                    RESULTS_DIR,
                    f"{prefix}/models/{prefix}_hidden{hidden_size}_fusion{fusion_size}.pth",
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)

                metrics = Train_UnifiedFusionModel(
                    seq_dataset_path,
                    track_dataset_path,
                    result_path,
                    test_train_split_annotation_path,
                    seq_input_size,
                    track_input_size,
                    hidden_size,
                    fusion_size,
                    DROPOUT,
                    model_path,
                    prefix,
                )

                summary_train_acc[hidden_size] = metrics["best_train_acc"]
                summary_val_acc[hidden_size] = metrics["best_val_acc"]
                summary_test_acc[hidden_size] = metrics["best_test_acc"]
                summary_r2_test[hidden_size] = metrics["r2_test"]

                summary_loss_diff[hidden_size] = (
                    metrics["train_losses"][-1] - metrics["val_losses"][-1]
                )

                if perform_SHAP_analysis:
                    from utils.shap_analysis import SHAP_UnifiedFusionModel

                    shap_result_dir = os.path.join(
                        result_path,
                        f"graphs/hidden_{hidden_size}/fusion_{fusion_size}/train results",
                    )

                    SHAP_UnifiedFusionModel(
                        seq_length=100,
                        features=cfg["features"],
                        track_features=cfg["track_features"],
                        model_save_path=model_path,
                        result_path=shap_result_dir,
                        seq_path=seq_dataset_path,
                        track_path=track_dataset_path,
                        hidden_size=hidden_size,
                        fusion_size=fusion_size,
                    )

    import csv
    from datetime import datetime

    date_str = datetime.now().strftime("%m_%d")
    acc_dir = os.path.join(RESULTS_DIR, "ablation_Specify", "accuracies", date_str)
    os.makedirs(acc_dir, exist_ok=True)

    def save_summary_csv(filename, header, data_dict):
        csv_path = os.path.join(acc_dir, filename)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for hidden_size, value in sorted(data_dict.items()):
                writer.writerow([hidden_size, value])
        print(f"[SAVED] {filename}")

    save_summary_csv("train_accuracies.csv", ["Hidden Size", "Train Accuracy"], summary_train_acc)
    save_summary_csv("val_accuracies.csv", ["Hidden Size", "Val Accuracy"], summary_val_acc)
    save_summary_csv("test_accuracies.csv", ["Hidden Size", "Test Accuracy"], summary_test_acc)
    save_summary_csv("r2_test.csv", ["Hidden Size", "R2 Test"], summary_r2_test)
    save_summary_csv("loss_diff.csv", ["Hidden Size", "Train-Val Loss Difference"], summary_loss_diff)
