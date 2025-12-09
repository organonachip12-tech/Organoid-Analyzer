import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from organoid_analyzer.config import (
    GENERATED_DIR, features, track_features, SEQ_LEN,
    DATASET_CONFIGS, SEQ_DATASET_PREFIX, TRACK_DATASET_PREFIX
)

class Dataset_Batch:
    def __init__(self, annotation_path, data_folder, mapping={}):
        self.annotation_path = annotation_path
        self.data_folder = data_folder
        self.mapping = mapping


# ================================================================
# STEP 1 — LOAD ANNOTATIONS
# ================================================================
def load_annotations(path, folder):
    print(f"[INFO] Loading annotations from: {path}")
    df = pd.read_excel(path)

    if "Case" not in df.columns:
        raise Exception("Annotations.xlsx must contain a column named 'Case'.")

    df["Case"] = df["Case"].astype(str).str.replace(" ", "").str.strip().str.upper()
    df["Label"] = df["Label"].astype(float)

    mapping = dict(zip(df["Case"], df["Label"]))

    print(f"[INFO] Total annotated cases loaded: {len(mapping)}")
    return mapping


# ================================================================
# STEP 2 — LOAD TRACK + SPOT FILES 
# ================================================================
def load_tracks_and_spots(datasets):
    spots = []
    tracks = []

    for label in datasets:
        root = datasets[label].data_folder
        mapping = datasets[label].mapping

        print(f"[INFO] Scanning folder: {root}")

        # Walk the root folder (CART, PDO, CAF, 2ND, NCI9)
        for dirpath, dirnames, filenames in os.walk(root):
            folder = os.path.basename(dirpath).upper()

            for fname in filenames:
                if not fname.endswith("_tracks.csv"):
                    continue

                track_path = os.path.join(dirpath, fname)
                spot_path = track_path.replace("_tracks.csv", "_spots.csv")

                # Example: "NCI2_XY1_tracks.csv" → "NCI2_XY1"
                prefix = os.path.splitext(fname)[0]
                base = prefix.split("_XY")[0]       # "NCI2", "Device4", "NYU318_CAF7", "2nd_NCI6", etc.
                base_upper = base.upper()

                # =====================================================================
                # FOLDER-BASED CASE NAMING RULES (this is the key!)
                # =====================================================================
                if folder == "CART":
                    # CART folder → always CART_
                    # NCI2 → CART_NCI2
                    # NYU358 → CART_NYU358
                    case_name = f"CART_{base_upper}"

                elif folder == "CAF":
                    # CAF folder → CAF_
                    # NYU318_CAF7 → CAF_NYU318_CAF7
                    case_name = f"CAF_{base_upper}"

                elif folder == "2ND":
                    # Files look like: 2nd_NCI6, 2nd_NCI8, 2nd_NYU360
                    # Annotation wants: 2ND_NCI6, 2ND_NCI8, 2ND_NYU360
                    clean = base_upper
                    if clean.startswith("2ND_"):
                        case_name = clean
                    else:
                        # normalize "2nd_NCI6" → "2ND_NCI6"
                        parts = base.split("_", 1)
                        if len(parts) == 2:
                            case_name = "2ND_" + parts[1].upper()
                        else:
                            raise Exception(f"[ERROR] Cannot normalize 2ND prefix: {base}")

                elif folder == "PDO":
                    # PDO folder → PDO_DEVICE#
                    # Device4 → PDO_DEVICE4
                    case_name = f"PDO_{base_upper}"

                elif folder == "NCI9":
                    # Keep EXACT case name as in annotation sheet
                    # NCI9_Round1_Stroma7 → NCI9_ROUND1_STROMA7
                    case_name = base_upper

                else:
                    raise Exception(f"[ERROR] Unknown data folder type: {folder}")

                # =====================================================================
                # VALIDATE AGAINST ANNOTATIONS
                # =====================================================================
                if case_name not in mapping:
                    raise Exception(
                        f"[ERROR] Mapping failed: folder={folder}, orig_prefix='{prefix}', "
                        f"case_name='{case_name}' not in annotations. File: {track_path}"
                    )

                case_label = mapping[case_name]

                # =====================================================================
                # LOAD TRACK + SPOT FILES
                # =====================================================================
                try:
                    # TRACKS
                    raw_t = pd.read_csv(track_path, encoding="latin1", header=None)
                    names_t = raw_t.iloc[0].tolist()
                    df_t = pd.read_csv(track_path, encoding="latin1", skiprows=1, names=names_t)
                    df_t = df_t.apply(pd.to_numeric, errors="coerce")

                    # SPOTS
                    raw_s = pd.read_csv(spot_path, encoding="latin1", header=None)
                    names_s = raw_s.iloc[0].tolist()
                    df_s = pd.read_csv(spot_path, encoding="latin1", skiprows=1, names=names_s)
                    df_s = df_s.apply(pd.to_numeric, errors="coerce")

                    # Add prefix + label
                    df_t["PREFIX"] = prefix
                    df_s["PREFIX"] = prefix
                    df_t["LABEL"] = case_label
                    df_s["LABEL"] = case_label

                    tracks.append(df_t)
                    spots.append(df_s)

                except Exception as e:
                    print(f"[ERROR] Failed loading prefix '{prefix}': {e}")
                    raise

    # =====================================================================
    # COMBINE ALL LOADED DATA
    # =====================================================================
    spots_df = pd.concat(spots, ignore_index=True)
    tracks_df = pd.concat(tracks, ignore_index=True)

    print(f"[INFO] Loaded {len(tracks_df)} track rows, {len(spots_df)} spot rows")
    return spots_df, tracks_df

# ================================================================
# STEP 3 — FILTER VALID TRAJECTORIES
# ================================================================
def filter_valid_trajectories(spots_df, tracks_df, min_frames=10):
    valid_ids = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames][["PREFIX", "TRACK_ID"]]
    spots_f = spots_df.merge(valid_ids, on=["PREFIX", "TRACK_ID"], how="inner")
    tracks_f = tracks_df[tracks_df["NUMBER_SPOTS"] >= min_frames]
    return spots_f, tracks_f


# ================================================================
# STEP 4 — MSD + SPOT-LEVEL FEATURES
# ================================================================
def compute_msd(x, y, max_lag=None):
    N = len(x)
    if max_lag is None:
        max_lag = N // 4
    out = []
    for dt in range(1, max_lag + 1):
        dx = x[dt:] - x[:-dt]
        dy = y[dt:] - y[:-dt]
        arr = dx**2 + dy**2
        out.append(arr.mean() if len(arr) else 0)
    return out


def compute_features(spots_df):
    spots_df = spots_df.sort_values(["PREFIX", "TRACK_ID", "FRAME"])

    # per-frame velocities
    spots_df["VELOCITY_X"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_X"].diff().fillna(0)
    spots_df["VELOCITY_Y"] = spots_df.groupby(["PREFIX", "TRACK_ID"])["POSITION_Y"].diff().fillna(0)

    # speed + direction (direction in units of π)
    spots_df["SPEED"] = np.sqrt(spots_df["VELOCITY_X"]**2 + spots_df["VELOCITY_Y"]**2)
    spots_df["DIRECTION"] = np.arctan2(spots_df["VELOCITY_Y"], spots_df["VELOCITY_X"]) / np.pi

    # MSD per (PREFIX,TRACK_ID)
    msd_records = []
    for (p, tid), g in tqdm(spots_df.groupby(["PREFIX", "TRACK_ID"]), desc="MSD"):
        g = g.sort_values("FRAME")
        x, y = g["POSITION_X"].values, g["POSITION_Y"].values
        msd = compute_msd(x, y, max_lag=len(x)-1)
        frames = g["FRAME"].iloc[1:].values
        for f, val in zip(frames, msd):
            msd_records.append({
                "PREFIX": p,
                "TRACK_ID": tid,
                "FRAME": f,
                "MEAN_SQUARE_DISPLACEMENT": val
            })

    msd_df = pd.DataFrame(msd_records)
    spots_df = spots_df.merge(msd_df, on=["PREFIX", "TRACK_ID", "FRAME"], how="left")
    spots_df["MEAN_SQUARE_DISPLACEMENT"] = spots_df["MEAN_SQUARE_DISPLACEMENT"].fillna(0)

    # Drop raw intensity + positions (we already derived motion features)
    drop = [
        c for c in spots_df.columns
        if "INTENSITY" in c or c in ["POSITION_X", "POSITION_Y"]
    ]
    spots_df.drop(columns=drop, inplace=True, errors="ignore")

    # ensure the feature columns are numeric
    for f in features:
        if f in spots_df.columns:
            spots_df[f] = pd.to_numeric(spots_df[f], errors="coerce").fillna(0)

    return spots_df


# ================================================================
# NEW STEP — TRACK-LEVEL FEATURES FROM SPOT TRAJECTORIES
# ================================================================
def compute_track_level_features(spots_df, tracks_df):
    """
    Compute:
      - TRACK_DISPLACEMENT: total 2D displacement from start→end
      - TRACK_STD_SPEED: std dev of per-frame SPEED
      - MEAN_DIRECTIONAL_CHANGE_RATE: mean |Δθ| per step (wrapped to [-π, π])
    using VELOCITY_X, VELOCITY_Y, and DIRECTION that we already computed.
    """

    records = []
    grouped = spots_df.groupby(["PREFIX", "TRACK_ID"])

    for (p, tid), g in grouped:
        g = g.sort_values("FRAME")

        vx = g["VELOCITY_X"].values if "VELOCITY_X" in g.columns else np.zeros(len(g))
        vy = g["VELOCITY_Y"].values if "VELOCITY_Y" in g.columns else np.zeros(len(g))
        speed = g["SPEED"].values if "SPEED" in g.columns else np.zeros(len(g))
        dir_norm = g["DIRECTION"].values if "DIRECTION" in g.columns else np.zeros(len(g))

        # 1) TRACK_DISPLACEMENT (sum of velocity vectors over time)
        dx_total = np.nansum(vx)
        dy_total = np.nansum(vy)
        track_disp = float(np.sqrt(dx_total**2 + dy_total**2))

        # 2) TRACK_STD_SPEED
        if len(speed) > 1:
            track_std_speed = float(np.nanstd(speed))
        else:
            track_std_speed = 0.0

        # 3) MEAN_DIRECTIONAL_CHANGE_RATE (robust, wrapped)
        if len(dir_norm) > 1:
            angles = dir_norm * np.pi     # back to radians
            diffs = np.diff(angles)
            # wrap into [-π, π]
            diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
            mean_dir_change = float(np.mean(np.abs(diffs)))
        else:
            mean_dir_change = 0.0

        records.append({
            "PREFIX": p,
            "TRACK_ID": tid,
            "TRACK_DISPLACEMENT": track_disp,
            "TRACK_STD_SPEED": track_std_speed,
            "MEAN_DIRECTIONAL_CHANGE_RATE": mean_dir_change
        })

    feat_df = pd.DataFrame(records)

    # merge into tracks_df
    tracks_df = tracks_df.merge(feat_df, on=["PREFIX", "TRACK_ID"], how="left")

    for col in ["TRACK_DISPLACEMENT", "TRACK_STD_SPEED", "MEAN_DIRECTIONAL_CHANGE_RATE"]:
        if col in tracks_df.columns:
            tracks_df[col] = tracks_df[col].fillna(0.0)

    return tracks_df


# ================================================================
# ALIGN & SAVE SEQUENCE DATASET
# ================================================================

def align_and_save_dataset(spots_df, features, seq_len=20, output_prefix=""):
    X, y, ids = [], [], []
    rows = []

    for (p, tid), g in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        vals = g[features].values
        if len(vals) >= seq_len:
            vals = vals[:seq_len]
        else:
            pad = np.zeros((seq_len - len(vals), len(features)))
            vals = np.vstack([vals, pad])

        X.append(vals)
        y.append(g["LABEL"].iloc[0])
        ids.append((p, tid))

        for t in range(seq_len):
            rows.append([f"{p}_{tid}", t] + list(vals[t]))

    X = np.array(X)
    y = np.array(y)

    np.savez(
        f"{GENERATED_DIR}/{output_prefix}trajectory_dataset_{seq_len}.npz",
        X=X, y=y, track_ids=np.array(ids, dtype=object)
    )

    df = pd.DataFrame(rows, columns=["SampleID", "Frame"] + features)
    df.to_csv(
        f"{GENERATED_DIR}/{output_prefix}trajectory_dataset_{seq_len}.csv",
        index=False
    )

    print(f"[SAVE] trajectory dataset saved → {X.shape}")


# ================================================================
# BUILD TRACK-LEVEL DATASET (.csv + .npz)
# ================================================================
def build_track_level_dataset(tracks_df, datasets, output_prefix="", track_features=track_features):
    df = tracks_df.dropna(subset=["PREFIX", "TRACK_ID"])

    recs = []
    for p, g in df.groupby("PREFIX"):
        g = g.dropna(subset=track_features)
        for _, row in g.iterrows():
            rec = {
                "PREFIX": p,
                "TRACK_ID": row["TRACK_ID"],
                "LABEL": row["LABEL"],
            }
            for f in track_features:
                rec[f] = row[f]
            recs.append(rec)

    df2 = pd.DataFrame(recs)

    df2.to_csv(
        f"{GENERATED_DIR}/{output_prefix}track_dataset.csv",
        index=False
    )

    np.savez(
        f"{GENERATED_DIR}/{output_prefix}track_dataset.npz",
        X=df2[track_features].values,
        y=df2["LABEL"].values,
        track_ids=df2[["PREFIX", "TRACK_ID"]].values
    )

    print("[SAVE] track-level dataset saved.")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    # 1) Build dataset configs from Config.DATASET_CONFIGS
    datasets = {}
    for key in DATASET_CONFIGS:
        cfg = DATASET_CONFIGS[key]
        batch = Dataset_Batch(cfg["annotation_path"], cfg["data_folder"])
        batch.mapping = load_annotations(batch.annotation_path, batch.data_folder)
        datasets[key] = batch

    # 2) Load all raw tracks + spots
    spots_df, tracks_df = load_tracks_and_spots(datasets)

    # 3) Filter tracks that are too short
    spots_df, tracks_df = filter_valid_trajectories(spots_df, tracks_df)

    # 4) Spot-level features (velocity, speed, direction, MSD, etc.)
    spots_df = compute_features(spots_df)

    # 5) Track-level features derived from spots
    tracks_df = compute_track_level_features(spots_df, tracks_df)

    # 6) Scale features + track_features jointly
    all_f = sorted(set(features).union(track_features))

    # make sure both have all_f columns (missing ones filled with 0)
    spots_aligned = spots_df.reindex(columns=all_f + [c for c in spots_df.columns if c not in all_f], fill_value=0)
    tracks_aligned = tracks_df.reindex(columns=all_f + [c for c in tracks_df.columns if c not in all_f], fill_value=0)

    # =================================================
    # FIX: Remove inf / -inf / very large numbers
    # =================================================
    def clean_df(df, cols):
        df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
        df[cols] = df[cols].fillna(0)
        df[cols] = df[cols].clip(lower=-1e6, upper=1e6)  # safety clamp
        return df

    spots_aligned = clean_df(spots_aligned, all_f)
    tracks_aligned = clean_df(tracks_aligned, all_f)

    scaler = StandardScaler().fit(
        pd.concat(
            [spots_aligned[all_f], tracks_aligned[all_f]],
            axis=0,
            ignore_index=True
        ).fillna(0)
    )

    spots_scaled_all = pd.DataFrame(
        scaler.transform(spots_aligned[all_f]),
        columns=all_f,
        index=spots_aligned.index
    )
    tracks_scaled_all = pd.DataFrame(
        scaler.transform(tracks_aligned[all_f]),
        columns=all_f,
        index=tracks_aligned.index
    )

    # overwrite only the relevant subsets
    spots_df.loc[:, features] = spots_scaled_all[features]
    tracks_df.loc[:, track_features] = tracks_scaled_all[track_features]

    [{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"tracks_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 303,
	"startColumn": 5,
	"endLineNumber": 303,
	"endColumn": 14,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"tracks_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 311,
	"startColumn": 15,
	"endLineNumber": 311,
	"endColumn": 24,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"tracks_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 331,
	"startColumn": 1,
	"endLineNumber": 331,
	"endColumn": 10,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"tracks_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 331,
	"startColumn": 21,
	"endLineNumber": 331,
	"endColumn": 30,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"spots_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 332,
	"startColumn": 1,
	"endLineNumber": 332,
	"endColumn": 9,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"spots_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 332,
	"startColumn": 20,
	"endLineNumber": 332,
	"endColumn": 28,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"tracks_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 336,
	"startColumn": 5,
	"endLineNumber": 336,
	"endColumn": 14,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"spots_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 344,
	"startColumn": 5,
	"endLineNumber": 344,
	"endColumn": 13,
	"origin": "extHost1"
},{
	"resource": "/Users/christinewagner/Documents/VIP/Organoid-Analyzer/organoid_analyzer/utils/create_dataset.py",
	"owner": "Pylance3",
	"code": {
		"value": "reportUndefinedVariable",
		"target": {
			"$mid": 1,
			"path": "/microsoft/pylance-release/blob/main/docs/diagnostics/reportUndefinedVariable.md",
			"scheme": "https",
			"authority": "github.com"
		}
	},
	"severity": 4,
	"message": "\"spots_df\" is not defined",
	"source": "Pylance",
	"startLineNumber": 352,
	"startColumn": 20,
	"endLineNumber": 352,
	"endColumn": 28,
	"origin": "extHost1"
}]
    # ================================================================
    # EXTENDED DEBUGGING — TRACK / SPOT / SEQUENCE USAGE REPORT
    # ================================================================
    print("\n" + "="*90)
    print("                  EXTENDED DATASET DEBUG REPORT")
    print("="*90)

    # ---------- 1. RAW TRACK COUNTS ----------
    raw_track_counts = (
        tracks_df.groupby("PREFIX")["TRACK_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"TRACK_ID": "raw_tracks"})
    )

    # ---------- PREFIX → CASE mapping ----------
    prefix_to_case = {}
    for prefix in tracks_df["PREFIX"].unique():
        p = prefix
        base = p.split("_XY")[0].upper()

        if base.startswith("CART_") or "CART" in base:
            case = "CART_" + base.replace("CART_", "")
        elif base.startswith("NYU318_CAF"):
            case = "CAF_" + base
        elif base.startswith("2ND_"):
            case = base
        elif base.startswith("DEVICE"):
            case = "PDO_" + base
        elif base.startswith("NCI9_"):
            case = base
        else:
            case = base

        prefix_to_case[prefix] = case

    tracks_df["CASE"] = tracks_df["PREFIX"].map(prefix_to_case)
    spots_df["CASE"] = spots_df["PREFIX"].map(prefix_to_case)

    # ---------- FIX: raw_track_counts must have CASE ----------
    raw_track_counts["CASE"] = raw_track_counts["PREFIX"].map(prefix_to_case)
    raw_track_counts = raw_track_counts.drop(columns=["PREFIX"])

    # ---------- 2. VALID TRACKS ----------
    valid_track_counts = (
        tracks_df.groupby("CASE")["TRACK_ID"]
        .nunique()
        .reset_index()
        .rename(columns={"TRACK_ID": "valid_tracks"})
    )

    # ---------- 3. SPOT ROW COUNTS ----------
    spot_counts = (
        spots_df.groupby("CASE")["FRAME"]
        .count()
        .reset_index()
        .rename(columns={"FRAME": "spot_rows"})
    )

    # ---------- 4. SEQUENCE WINDOWS ----------
    seq_windows = []
    for (p, tid), g in spots_df.groupby(["PREFIX", "TRACK_ID"]):
        usable = 1 if len(g) >= SEQ_LEN else 0
        seq_windows.append((prefix_to_case[p], usable))

    seq_df = pd.DataFrame(seq_windows, columns=["CASE", "usable_sequence"])
    seq_counts = seq_df.groupby("CASE")["usable_sequence"].sum().reset_index()

    # ---------- 5. MERGE EVERYTHING ----------
    debug_df = (
        raw_track_counts
        .merge(valid_track_counts, on="CASE", how="outer")
        .merge(spot_counts, on="CASE", how="outer")
        .merge(seq_counts, on="CASE", how="outer")
        .fillna(0)
    )

    # Convert to ints
    for col in ["raw_tracks", "valid_tracks", "spot_rows", "usable_sequence"]:
        debug_df[col] = debug_df[col].astype(int)

    print("\nPER-CASE DATA REPORT")
    print("-" * 90)
    print(debug_df.sort_values("usable_sequence", ascending=False).to_string(index=False))

    # ---------- SUMMARY ----------
    total_raw = debug_df["raw_tracks"].sum()
    total_valid = debug_df["valid_tracks"].sum()
    total_seq = debug_df["usable_sequence"].sum()

    print("\nSUMMARY:")
    print(f"  Total raw tracks loaded:         {total_raw}")
    print(f"  Tracks after min_frames filter:  {total_valid}")
    print(f"  Tracks that produced sequences:  {total_seq}")
    print(f"  → Sequence retention rate:       {100*total_seq/max(1,total_valid):.2f}%")
    print("="*90 + "\n")

    # 7) Save sequence dataset
    align_and_save_dataset(
        spots_df,
        features,
        seq_len=SEQ_LEN,
        output_prefix=SEQ_DATASET_PREFIX
    )

    # 8) Save track-level dataset
    build_track_level_dataset(
        tracks_df,
        datasets,
        output_prefix=TRACK_DATASET_PREFIX
    )

    print("Dataset creation complete.")
