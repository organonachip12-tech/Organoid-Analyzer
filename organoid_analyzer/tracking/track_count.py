import os
import pandas as pd
from collections import defaultdict

# Update these to match your directory structure
OLD_DATA = "./Data/OldData"
NCI9_DATA = "./Data/NCI9"

# Helper: extract case name the same way create_dataset.py does
def get_case_name(prefix, folder_name):
    base = prefix.split("_XY")[0]

    if folder_name == "PDO":
        return f"PDO_{base}"
    elif folder_name == "CART":
        return f"CART_{base}"
    elif folder_name == "2ND":
        if base.startswith("2nd_"):
            return "2ND_" + base.split("2nd_")[1]
        return base
    elif folder_name == "CAF":
        return f"CAF_{base}"
    elif folder_name == "NCI9":
        return base
    else:
        return base

def count_tracks(data_folder):
    track_counts = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(data_folder):
        folder_name = os.path.basename(dirpath)

        for fname in filenames:
            if not fname.endswith("_tracks.csv"):
                continue

            prefix = os.path.splitext(fname)[0]    # e.g. Device2_XY6_tracks → Device2_XY6_tracks
            prefix = prefix.replace("_tracks", "") # → Device2_XY6

            case_name = get_case_name(prefix, folder_name)

            try:
                df = pd.read_csv(os.path.join(dirpath, fname), encoding="latin1", header=None)
                cols = df.iloc[0].tolist()
                df = pd.read_csv(os.path.join(dirpath, fname), encoding="latin1", skiprows=1, names=cols)

                # Count unique TRACK_IDs for this file
                count = df["TRACK_ID"].nunique()

                track_counts[case_name] += count

            except Exception as e:
                print(f"[ERROR] Failed reading {fname}: {e}")

    return track_counts


if __name__ == "__main__":
    print("\n=== Counting Tracks in OLD DATA ===")
    old_counts = count_tracks(OLD_DATA)

    print("\n=== Counting Tracks in NCI9 DATA ===")
    nci9_counts = count_tracks(NCI9_DATA)

    # Merge both
    all_cases = set(old_counts.keys()) | set(nci9_counts.keys())
    rows = []

    for case in sorted(all_cases):
        rows.append({
            "Case": case,
            "Track_Count": old_counts.get(case, 0) + nci9_counts.get(case, 0),
            "OldData": old_counts.get(case, 0),
            "NCI9": nci9_counts.get(case, 0),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv("track_counts_summary.csv", index=False)

    print("\n=== TRACK COUNT SUMMARY ===")
    print(out_df)

    print("\nSaved to track_counts_summary.csv")