"""
Run TCGA pipeline:
- Fetch cases + files from GDC
- Build annotations (image_path, patient_id, survival_time, death_occurred)
- Save to Data/gigatime/annotations.csv (used by GigaTIME pipeline)
"""

import os

from tcga_pipeline.gdc_tcga import (
    USER,
    fetch_all_cases,
    fetch_all_files,
    build_annotations,
    write_annotations_csv,
)


def main():
    print("🚀 Running TCGA data pipeline...")

    project_id = USER["project_id"]
    token = USER.get("auth_token")  # optional, usually None

    print(f"Project: {project_id}")

    # -----------------------------------
    # Step 1: Fetch TCGA case data
    # -----------------------------------
    print("\n📥 Fetching cases from GDC...")
    cases_resp = fetch_all_cases(project_id, token=token)

    # -----------------------------------
    # Step 2: Fetch TCGA slide file data
    # -----------------------------------
    print("\n🖼️ Fetching slide files from GDC...")
    files_resp = fetch_all_files(project_id, token=token)

    # -----------------------------------
    # Step 3: Build annotations
    # -----------------------------------
    print("\n🧠 Building annotations...")

    rows = build_annotations(
        cases_resp,
        files_resp,
        images_dir="Data/gigatime/images",  # 🔥 MUST match your actual image folder
        primary_tumor_only=USER.get("primary_tumor_only", True),
        min_follow_up_days=USER.get("min_follow_up_days"),
    )

    print(f"✅ Built {len(rows)} annotation rows")

    if len(rows) == 0:
        raise ValueError("No annotation rows were created. Check TCGA pipeline filters.")

    # -----------------------------------
    # Step 4: Save annotations for GigaTIME
    # -----------------------------------
    print("\n💾 Saving annotations...")

    os.makedirs("Data/gigatime", exist_ok=True)

    output_csv = "Data/gigatime/annotations.csv"

    write_annotations_csv(rows, output_csv)

    print(f"✅ Saved annotations → {output_csv}")


if __name__ == "__main__":
    main()