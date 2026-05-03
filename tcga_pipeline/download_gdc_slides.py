"""
download_gdc_slides.py

Local test downloader for TCGA whole slide image files from GDC.

Purpose
1. Read TCGA slide metadata from clinical_gdc_TCGA-PAAD.csv
2. Use file_id values to download SVS files from the GDC data endpoint
3. Save files into data/gigatime/images so paths match the GigaTIME annotations CSV

This script is intentionally local first.
Wasabi upload should be added only after local GDC download works.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import requests


GDC_DATA_BASE_URL = "https://api.gdc.cancer.gov/data"

# Repo root is one folder above tcga_pipeline
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CLINICAL_CSV = REPO_ROOT / "tcga_pipeline" / "clinical_gdc_TCGA-PAAD.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "gigatime" / "images"


def parse_args() -> argparse.Namespace:
    """
    Parse command line options.

    Default behavior is intentionally small:
    download only 3 files unless --all is used.
    """
    parser = argparse.ArgumentParser(
        description="Download TCGA SVS slide files from GDC using clinical CSV file_id values."
    )

    parser.add_argument(
        "--clinical-csv",
        default=str(DEFAULT_CLINICAL_CSV),
        help="Path to clinical_gdc_TCGA-PAAD.csv or equivalent clinical metadata CSV.",
    )

    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Folder where downloaded SVS files should be saved.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of files to download for testing. Ignored if --all is used.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero based starting index in the de duplicated file list.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all files from the clinical CSV.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading files.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re download files even if they already exist locally.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Socket timeout for each GDC request.",
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=8,
        help="Streaming chunk size in megabytes.",
    )

    parser.add_argument(
        "--token-env",
        default="GDC_AUTH_TOKEN",
        help="Environment variable containing a GDC token if controlled access is needed.",
    )

    return parser.parse_args()


def read_clinical_csv(clinical_csv: Path) -> list[dict[str, str]]:
    """
    Read clinical CSV rows and keep only rows with file_id and file_name.

    Expected columns:
    file_id
    file_name
    patient_id

    patient_id is not required for download, but it is useful for logging.
    """
    if not clinical_csv.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {clinical_csv}")

    with clinical_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        required = {"file_id", "file_name"}
        missing = required.difference(fieldnames)

        if missing:
            raise ValueError(
                f"Clinical CSV is missing required columns: {sorted(missing)}"
            )

        rows = []
        for row in reader:
            file_id = (row.get("file_id") or "").strip()
            file_name = (row.get("file_name") or "").strip()

            if not file_id or not file_name:
                continue

            rows.append(row)

    return rows


def unique_file_records(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """
    De duplicate by file_id so the same GDC file is not downloaded twice.
    """
    seen_file_ids: set[str] = set()
    records: list[dict[str, str]] = []

    for row in rows:
        file_id = (row.get("file_id") or "").strip()

        if file_id in seen_file_ids:
            continue

        seen_file_ids.add(file_id)
        records.append(row)

    return records


def get_gdc_token(token_env: str) -> str | None:
    """
    Read optional GDC auth token from an environment variable.

    Open access TCGA slide files usually do not need a token.
    Controlled access data would need one.
    """
    token = os.environ.get(token_env)

    if token:
        return token.strip()

    return None


def mb(num_bytes: int | float) -> float:
    """
    Convert bytes to megabytes for progress messages.
    """
    return float(num_bytes) / (1024 * 1024)


def download_one_slide(
    file_id: str,
    file_name: str,
    out_dir: Path,
    token: str | None = None,
    overwrite: bool = False,
    timeout_seconds: int = 120,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> Path:
    """
    Download one GDC file by file_id.

    Saves to:
    out_dir / file_name

    Uses a .part file during download so incomplete files are obvious.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Path(file_name).name prevents accidental directory traversal if file_name ever includes paths.
    safe_file_name = Path(file_name).name
    out_path = out_dir / safe_file_name
    part_path = out_dir / f"{safe_file_name}.part"

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"SKIP existing file: {out_path}")
        return out_path

    if part_path.exists():
        part_path.unlink()

    url = f"{GDC_DATA_BASE_URL}/{file_id}"

    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    print(f"Downloading {safe_file_name}")
    print(f"  file_id: {file_id}")
    print(f"  target : {out_path}")

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()

        total_bytes = int(response.headers.get("Content-Length", 0) or 0)
        downloaded = 0
        next_report = 64 * 1024 * 1024

        with part_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if downloaded >= next_report:
                    if total_bytes:
                        print(
                            f"  progress: {mb(downloaded):.1f} MB / {mb(total_bytes):.1f} MB"
                        )
                    else:
                        print(f"  progress: {mb(downloaded):.1f} MB")

                    next_report += 64 * 1024 * 1024

    if total_bytes and downloaded != total_bytes:
        raise IOError(
            f"Incomplete download for {safe_file_name}: "
            f"got {downloaded} bytes, expected {total_bytes} bytes"
        )

    part_path.replace(out_path)
    print(f"DONE {safe_file_name} ({mb(downloaded):.1f} MB)")

    return out_path


def main() -> None:
    """
    Run the local GDC download test.
    """
    args = parse_args()

    clinical_csv = Path(args.clinical_csv).resolve()
    out_dir = Path(args.out_dir).resolve()

    rows = read_clinical_csv(clinical_csv)
    records = unique_file_records(rows)

    print(f"Clinical CSV: {clinical_csv}")
    print(f"Output dir  : {out_dir}")
    print(f"Rows read   : {len(rows)}")
    print(f"Unique files: {len(records)}")

    if args.start < 0:
        raise ValueError("--start must be 0 or greater")

    records = records[args.start :]

    if not args.all:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1 unless --all is used")
        records = records[: args.limit]

    print(f"Selected files for this run: {len(records)}")

    if not records:
        print("No files selected. Nothing to do.")
        return

    token = get_gdc_token(args.token_env)
    chunk_size_bytes = args.chunk_size_mb * 1024 * 1024

    for i, row in enumerate(records, start=1):
        file_id = row["file_id"].strip()
        file_name = row["file_name"].strip()
        patient_id = (row.get("patient_id") or "").strip()

        print("")
        print(f"[{i}/{len(records)}] patient_id={patient_id} file_name={file_name}")

        if args.dry_run:
            print("DRY RUN only. No file downloaded.")
            continue

        download_one_slide(
            file_id=file_id,
            file_name=file_name,
            out_dir=out_dir,
            token=token,
            overwrite=args.overwrite,
            timeout_seconds=args.timeout_seconds,
            chunk_size_bytes=chunk_size_bytes,
        )

    print("")
    print("Download test complete.")


if __name__ == "__main__":
    main()