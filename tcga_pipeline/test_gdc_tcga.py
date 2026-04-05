"""
test_gdc_tcga.py

Quick test script for gdc_tcga.py

What this script verifies
1. We can successfully call the GDC /cases endpoint for a TCGA project
2. We can successfully call the GDC /files endpoint for slide like files
3. The responses have the expected shape: response["data"]["hits"] is a list
4. The first returned record has the fields we requested
5. We can fetch the full dataset using pagination
6. We can build joined annotation rows and extended clinical rows
7. We can write the final project specific CSV outputs
"""

from gdc_tcga import (
    fetch_all_cases,
    fetch_all_files,
    build_annotations,
    build_clinical_rows,
    write_project_annotations_csv,
    write_project_clinical_csv,
)

# Pick a TCGA project to test
PROJECT = "TCGA-BRCA"

# Fetch the full dataset using pagination
# This is necessary because the first page of /cases and the first page of /files
# are not guaranteed to overlap on submitter_id
cases_resp = fetch_all_cases(PROJECT)
files_resp = fetch_all_files(PROJECT)

# Basic structure checks
# Top level keys should include "data" for successful responses
print("cases_resp top level keys:", list(cases_resp.keys()))
print("files_resp top level keys:", list(files_resp.keys()))

# Hits are where the returned records live
cases_hits = cases_resp["data"]["hits"]
files_hits = files_resp["data"]["hits"]

print("cases hits count:", len(cases_hits))
print("files hits count:", len(files_hits))

# Show what fields are present in the first hit from each query
# Only do this if there is at least one hit to avoid IndexError
if cases_hits:
    print("cases first hit field names:", list(cases_hits[0].keys()))
    print("example case submitter_id:", cases_hits[0].get("submitter_id"))
else:
    print("cases returned 0 hits. Check your project_id or filters.")

if files_hits:
    print("files first hit field names:", list(files_hits[0].keys()))
    print("example file name:", files_hits[0].get("file_name"))
    print("example file data_format:", files_hits[0].get("data_format"))
    # cases is usually a list inside each file record
    print("example file linked cases:", files_hits[0].get("cases"))
else:
    print(
        "files returned 0 hits. This often means the slide filter is too strict.\n"
        "Try running get_files_for_project(PROJECT, size=20) and inspect data_format and data_type."
    )

# Build joined annotation rows from the full fetched data
# Section 4 cohort logic:
# - keep Primary Tumor files only
# - keep only valid Alive/Dead survival cases
# - no short follow-up exclusion yet
annotations = build_annotations(
    cases_resp,
    files_resp,
    images_dir=f"data/til/images/{PROJECT}",
    primary_tumor_only=True,
    min_follow_up_days=None,  # optional exclusion creteria
)

# Build extended clinical CSV rows using the same filtered cohort logic
clinical_rows = build_clinical_rows(
    cases_resp,
    files_resp,
    project_id=PROJECT,
    primary_tumor_only=True,
    min_follow_up_days=None,
)

print("\nannotation rows count:", len(annotations))
print("clinical rows count:", len(clinical_rows))

if annotations:
    print("\nfirst annotation row:")
    print(annotations[0])

    # Save annotations using project-specific filename (Willem spec)
    annotations_csv = write_project_annotations_csv(PROJECT, annotations)
    print(f"\nwrote {annotations_csv}")
else:
    print("\nno annotation rows were built")
    print("no annotations CSV written because no annotation rows were built")

if clinical_rows:
    print("\nfirst clinical row:")
    print(clinical_rows[0])

    # Save clinical CSV using project-specific filename
    clinical_csv = write_project_clinical_csv(PROJECT, clinical_rows)
    print(f"wrote {clinical_csv}")
else:
    print("\nno clinical rows were built")
    print("no clinical CSV written because no clinical rows were built")