import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
from Config import DROPOUT

# === model and variable import ===

from UnifiedFusionModel import UnifiedFusionModel
torch.backends.cudnn.enabled = False

# === routes configuration ===

def load_and_align_data(seq_path, track_path):
    seq_data = np.load(seq_path, allow_pickle=True)
    track_data = np.load(track_path, allow_pickle=True)

    X_seq, y_seq, track_ids_seq = seq_data['X'], seq_data['y'], seq_data['track_ids']
    X_track, y_track, track_ids_track = track_data['X'], track_data['y'], track_data['track_ids']

    if X_seq.shape[1] == 11 and X_seq.shape[2] == 20:
        print("transposing...")
        X_seq = np.transpose(X_seq, (0, 2, 1))

    track_id_to_index = {
        tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,): i
        for i, tid in enumerate(track_ids_track)
    }

    X_seq_matched, X_track_matched, y_matched, prefix_tid = [], [], [], []
    for i, tid in enumerate(track_ids_seq):
        key = tuple(tid) if isinstance(tid, (list, tuple, np.ndarray)) else (tid,)
        if key in track_id_to_index:
            idx = track_id_to_index[key]
            X_seq_matched.append(X_seq[i])
            X_track_matched.append(X_track[idx])
            y_matched.append(y_seq[i])
            prefix_tid.append(tid[0]+str(tid[1]))

    print(f"[DEBUG] Matched pairs: {len(X_seq_matched)}")
    print(np.array(X_track_matched).shape)
    return np.array(X_seq_matched), np.array(X_track_matched), np.array(y_matched), prefix_tid


def SHAP_UnifiedFusionModel(seq_length, features, track_features, model_save_path, result_path, seq_path, track_path, hidden_size = 32, fusion_size = 256):
    """
    Perform SHAP analysis on the unified fusion model.
    """
    feature_length = len(features)              # e.g. 9
    track_feature_length = len(track_features)  # e.g. 12
    total_seq = seq_length * feature_length     # e.g. 180 = 20 * 9
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === load data and model  ===
    print("[STEP 1] Loading model and data...")

    model = UnifiedFusionModel(seq_input_size=feature_length, track_input_size=track_feature_length,
                               hidden_size=hidden_size, fusion_size=fusion_size, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.eval()

    X_seq, X_track, y, prefix_tid = load_and_align_data(seq_path, track_path)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_seq_train, X_seq_test, X_track_train, X_track_test, y_train, y_test = train_test_split(
        X_seq, X_track, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    X_seq_train = torch.tensor(X_seq_train, dtype=torch.float32)
    X_seq_test  = torch.tensor(X_seq_test,  dtype=torch.float32)
    X_track_train = torch.tensor(X_track_train, dtype=torch.float32)
    X_track_test  = torch.tensor(X_track_test,  dtype=torch.float32)

    # combine (X_seq, X_track) to one tensor
    X_seq_flat    = X_seq_test.reshape(X_seq_test.shape[0], -1)  # (N, T*F)
    X_test_concat = torch.cat([X_seq_flat, X_track_test], dim=1).to(device)

    X_train_seq_flat = X_seq_train.reshape(X_seq_train.shape[0], -1)
    X_train_concat   = torch.cat([X_train_seq_flat, X_track_train], dim=1).to(device)

    # define wrapped model
    class WrappedUnifiedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.seq_length = seq_length
            self.feature_length = feature_length

        def forward(self, x_concat):
            # x_concat: (N, total_seq + track_feature_length)
            batch_size = x_concat.shape[0]
            x_seq_flat = x_concat[:, :total_seq]
            x_track   = x_concat[:, total_seq:]
            x_seq = x_seq_flat.view(batch_size, self.seq_length, self.feature_length)
            return self.model(x_seq, x_track)

    # === SHAP ===
    print("[STEP 2] SHAP analysis...")
    wrapped_model = WrappedUnifiedModel(model).to(device)
    explainer = shap.GradientExplainer(wrapped_model, X_train_concat[:100])
    shap_values = explainer.shap_values(X_test_concat[:100], ranked_outputs=1)

    print("[STEP 3] SHAP drawing...")
    # shap_values_combined: (100, total_seq + track_feature_length, 1)
    shap_values_combined = shap_values[0]

    shap_value_seq   = shap_values_combined[:, :total_seq]
    shap_value_track = shap_values_combined[:, total_seq:]

    shap_value_seq = shap_value_seq.reshape(100, seq_length, feature_length)  # (N, T, F)

    # ---- Signed mean importance (original) ----
    shap_result_seq_signed   = shap_value_seq.mean(axis=(0, 1))   # (F,)
    shap_result_track_signed = shap_value_track.mean(axis=(0, 2)) # (track_F,)
    shap_result_signed = np.concatenate((shap_result_seq_signed, shap_result_track_signed))

    # ---- Absolute mean importance (NEW) ----
    shap_result_seq_abs   = np.abs(shap_value_seq).mean(axis=(0, 1))   # (F,)
    shap_result_track_abs = np.abs(shap_value_track).mean(axis=(0, 2)) # (track_F,)
    shap_result_abs = np.concatenate((shap_result_seq_abs, shap_result_track_abs))

    # Base feature names (sequence-level features collapsed over time + track features)
    base_feature_names = features + track_features

    # DataFrames
    shap_df_signed = pd.DataFrame({
        "Feature": base_feature_names,
        "Importance": shap_result_signed
    }).sort_values("Importance", ascending=False)

    shap_df_abs = pd.DataFrame({
        "Feature": base_feature_names,
        "Importance": shap_result_abs
    }).sort_values("Importance", ascending=False)

    print("[INFO] Top (signed) SHAP features:")
    print(shap_df_signed.head(10))
    print("[INFO] Top (absolute) SHAP features:")
    print(shap_df_abs.head(10))

    shap_df_signed.to_csv(f"{result_path}/Top (signed) SHAP features.csv", index=False)
    shap_df_abs.to_csv(f"{result_path}/Top (absolute) SHAP features.csv", index=False)

    # ---- Plot signed importance (unchanged file name) ----
    plt.figure(figsize=(12, 6))
    sns.barplot(data=shap_df_signed.head(30), x="Importance", y="Feature")
    plt.title("Unified Fusion SHAP Feature Importances (Signed)")
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_bar.png")
    print("SHAP signed feature importance saved to unified_shap_bar.png.")
    plt.close()

    # ---- Plot absolute importance (NEW) ----
    plt.figure(figsize=(12, 6))
    sns.barplot(data=shap_df_abs.head(30), x="Importance", y="Feature")
    plt.title("Unified Fusion SHAP Feature Importances (Absolute)")
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_bar_absolute.png")
    print("SHAP absolute feature importance saved to unified_shap_bar_absolute.png.")
    plt.close()

    # === For signed SHAP (keep direction) ===
    # Average over samples, sum over time steps
    shap_result_seq_signed = shap_value_seq.mean(axis=0).sum(axis=0)  # (features,)

    # Track features: average over samples (no time dimension)
    shap_result_track_signed = shap_value_track.mean(axis=0).mean(axis=1)  # (track_features,)

    # === For absolute SHAP (magnitude) ===
    # Take abs first, average over samples, then sum over time
    shap_result_seq_abs = np.abs(shap_value_seq).mean(axis=0).sum(axis=0)  # (features,)
    shap_result_track_abs = np.abs(shap_value_track).mean(axis=0).mean(axis=1)  # (track_features,)

    # Combine
    shap_result_signed = np.concatenate((shap_result_seq_signed, shap_result_track_signed))
    shap_result_abs = np.concatenate((shap_result_seq_abs, shap_result_track_abs))

    # Feature names
    feature_names_base = features + track_features

    # Create DataFrames
    shap_df_signed = pd.DataFrame({"Feature": feature_names_base, "Importance": shap_result_signed}).sort_values("Importance", ascending=False)
    shap_df_abs = pd.DataFrame({"Feature": feature_names_base, "Importance": shap_result_abs}).sort_values("Importance", ascending=False)
    shap_df_signed.to_csv(f"{result_path}/Top (Time-Summed, signed) SHAP features.csv", index=False)
    shap_df_abs.to_csv(f"{result_path}/Top (Time-Summed, absolute) SHAP features.csv", index=False)

    # === Plot signed ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=shap_df_signed.head(30), x="Importance", y="Feature")
    plt.title("Unified Fusion SHAP Feature Importances (Signed, Time-Summed)")
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_bar_signed_timesum.png")
    plt.close()

    # === Plot absolute ===
    plt.figure(figsize=(12, 6))
    sns.barplot(data=shap_df_abs.head(30), x="Importance", y="Feature")
    plt.title("Unified Fusion SHAP Feature Importances (Absolute, Time-Summed)")
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_bar_absolute_timesum.png")
    plt.close()


    print("[STEP 4] SHAP summary plot (beeswarm)...")

    # Convert to 2D numpy (N, total_seq + track_feature_length)
    shap_values_2d = shap_values_combined.squeeze(-1)

    # Expanded per-time-step names for summary plot
    seq_feature_names = [f"{feat}_t{t}" for t in range(seq_length) for feat in features]
    feature_names_expanded = seq_feature_names + track_features  # length = total_seq + track_feature_length

    X_test_concat_cpu = X_test_concat[:100].cpu().detach().numpy()

    # summary plot（蜂群图）
    shap.summary_plot(
        shap_values_2d,
        X_test_concat_cpu,
        feature_names=feature_names_expanded,
        plot_type="dot",
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"{result_path}/unified_shap_summary.png")
    print("SHAP summary plot saved to unified_shap_summary.png.")
    plt.close()

if __name__ == "__main__":
    from Config import UNI_RESULT_DIR, SEQ_LEN, MODEL_DIR, SEQ_RESULT_DIR, features, track_features, GENERATED_DIR
    # e.g. 180 = 20 * 9
    SEQ_DATA_PATH = f"{GENERATED_DIR}/trajectory_dataset_{SEQ_LEN}.npz"
    TRACK_DATA_PATH = f"{GENERATED_DIR}/track_dataset.npz"
    MODEL_SAVE_PATH = f"{MODEL_DIR}/unified_fusion_model.pth"
    os.makedirs(SEQ_RESULT_DIR, exist_ok=True)
    SHAP_UnifiedFusionModel(SEQ_LEN, features, track_features, MODEL_SAVE_PATH,
                            result_path=UNI_RESULT_DIR, seq_path=SEQ_DATA_PATH, track_path=TRACK_DATA_PATH)
