"""
download_gdc_slides.py

Download TCGA whole slide image files from GDC.

Main use case
Stream TCGA SVS files from the GDC data endpoint directly into a private
Wasabi bucket without saving full slide files locally.

Modes
1. Local mode
   Used when --upload-wasabi is not provided.
   Downloads SVS files into data/gigatime/images.

2. Wasabi mode
   Used when --upload-wasabi is provided.
   Streams GDC response bytes into Wasabi multipart upload.
   No full local SVS file is written.

Credentials
Do not store access keys in code or YAML.
Use environment variables for secrets.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import requests
import yaml


GDC_DATA_BASE_URL = "https://api.gdc.cancer.gov/data"
REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = REPO_ROOT / "tcga_pipeline" / "gdc_config.yaml"
DEFAULT_CLINICAL_CSV = REPO_ROOT / "tcga_pipeline" / "clinical_gdc_TCGA-PAAD.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "gigatime" / "images"


def resolve_repo_path(path_value: str | Path) -> Path:
    """
    Resolve a path.

    Relative paths are interpreted from the repository root.
    Absolute paths are kept absolute.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (REPO_ROOT / path).resolve()


def parse_config_path() -> Path:
    """
    Parse only --config first so YAML values can become parser defaults.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to gdc_config.yaml.",
    )

    args, _ = parser.parse_known_args()
    return resolve_repo_path(args.config)


def load_config(config_path: Path) -> dict:
    """
    Load YAML config if it exists.

    Missing config is allowed so this script can still run with command line defaults.
    """
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def section(config: dict, name: str) -> dict:
    """
    Return a named YAML section as a dictionary.
    """
    value = config.get(name, {})
    if isinstance(value, dict):
        return value
    return {}


def parse_args(config_path: Path, config: dict) -> argparse.Namespace:
    """
    Parse command line options.

    YAML provides defaults.
    Command line arguments override YAML.
    """
    download_cfg = section(config, "download_settings")
    wasabi_cfg = section(config, "wasabi_settings")

    default_clinical_csv = download_cfg.get("clinical_csv", str(DEFAULT_CLINICAL_CSV))
    default_out_dir = download_cfg.get("local_images_dir", str(DEFAULT_OUT_DIR))
    default_limit = int(download_cfg.get("default_limit", 3))
    default_timeout = int(download_cfg.get("timeout_seconds", 120))
    default_chunk_size = int(download_cfg.get("chunk_size_mb", 8))
    default_part_size = int(download_cfg.get("multipart_part_size_mb", 64))
    default_gdc_token_env = download_cfg.get("gdc_token_env", "GDC_AUTH_TOKEN")

    default_upload = bool(wasabi_cfg.get("upload_default", False))
    default_bucket = wasabi_cfg.get("bucket", "")
    default_region = wasabi_cfg.get("region", "us-east-1")
    default_endpoint = wasabi_cfg.get(
        "endpoint_url",
        "https://s3.us-east-1.wasabisys.com",
    )
    default_prefix = wasabi_cfg.get("prefix", "raw_svs/TCGA-PAAD")
    default_access_key_env = wasabi_cfg.get(
        "access_key_env",
        "WASABI_ACCESS_KEY_ID",
    )
    default_secret_key_env = wasabi_cfg.get(
        "secret_key_env",
        "WASABI_SECRET_ACCESS_KEY",
    )

    parser = argparse.ArgumentParser(
        description="Download TCGA SVS slides from GDC or stream them directly to Wasabi."
    )

    parser.add_argument(
        "--config",
        default=str(config_path),
        help="Path to gdc_config.yaml.",
    )

    parser.add_argument(
        "--clinical-csv",
        default=default_clinical_csv,
        help="Path to clinical_gdc_TCGA-PAAD.csv.",
    )

    parser.add_argument(
        "--out-dir",
        default=default_out_dir,
        help="Local output folder. Used only when --upload-wasabi is not set.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="Number of files to process. Ignored if --all is used.",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero based start index in the deduplicated file list.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all files from the clinical CSV.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without downloading or uploading.",
    )

    parser.add_argument(
        "--overwrite-download",
        action="store_true",
        help="Redownload local files even if they already exist. Local mode only.",
    )

    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=default_timeout,
        help="Socket timeout for GDC requests.",
    )

    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=default_chunk_size,
        help="GDC response streaming chunk size in MB.",
    )

    parser.add_argument(
        "--multipart-part-size-mb",
        type=int,
        default=default_part_size,
        help="Wasabi multipart upload part size in MB.",
    )

    parser.add_argument(
        "--gdc-token-env",
        default=default_gdc_token_env,
        help="Environment variable containing a GDC token if needed.",
    )

    parser.add_argument(
        "--upload-wasabi",
        action="store_true",
        default=default_upload,
        help="Stream GDC slide files directly to Wasabi.",
    )

    parser.add_argument(
        "--no-upload-wasabi",
        action="store_false",
        dest="upload_wasabi",
        help="Disable Wasabi upload even if config enables it.",
    )

    parser.add_argument(
        "--wasabi-bucket",
        default=default_bucket,
        help="Wasabi bucket name.",
    )

    parser.add_argument(
        "--wasabi-region",
        default=default_region,
        help="Wasabi region.",
    )

    parser.add_argument(
        "--wasabi-endpoint-url",
        default=default_endpoint,
        help="Wasabi S3 endpoint URL.",
    )

    parser.add_argument(
        "--wasabi-prefix",
        default=default_prefix,
        help="Wasabi object prefix.",
    )

    parser.add_argument(
        "--wasabi-access-key-env",
        default=default_access_key_env,
        help="Environment variable holding the Wasabi access key.",
    )

    parser.add_argument(
        "--wasabi-secret-key-env",
        default=default_secret_key_env,
        help="Environment variable holding the Wasabi secret key.",
    )

    parser.add_argument(
        "--overwrite-wasabi",
        action="store_true",
        help="Upload even if the Wasabi object already exists.",
    )

    parser.add_argument(
        "--delete-local-after-upload",
        action="store_true",
        help="Remove any matching local SVS after Wasabi upload or skip.",
    )

    return parser.parse_args()


def read_clinical_csv(clinical_csv: Path) -> list[dict[str, str]]:
    """
    Read clinical metadata rows.

    Required columns:
    file_id
    file_name
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

            if file_id and file_name:
                rows.append(row)

    return rows


def unique_file_records(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """
    Deduplicate by file_id.
    """
    seen: set[str] = set()
    records: list[dict[str, str]] = []

    for row in rows:
        file_id = (row.get("file_id") or "").strip()

        if file_id in seen:
            continue

        seen.add(file_id)
        records.append(row)

    return records


def select_records(
    records: list[dict[str, str]],
    start: int,
    limit: int,
    process_all: bool,
) -> list[dict[str, str]]:
    """
    Apply start and limit controls.
    """
    if start < 0:
        raise ValueError("--start must be 0 or greater")

    selected = records[start:]

    if process_all:
        return selected

    if limit < 1:
        raise ValueError("--limit must be at least 1 unless --all is used")

    return selected[:limit]


def read_optional_env(env_name: str) -> str | None:
    """
    Read an optional environment variable.
    """
    value = os.environ.get(env_name)

    if value:
        return value.strip()

    return None


def read_required_env(env_name: str, purpose: str) -> str:
    """
    Read a required environment variable.
    """
    value = os.environ.get(env_name)

    if not value:
        raise ValueError(
            f"Missing environment variable {env_name}. Required for {purpose}."
        )

    return value.strip()


def mb(num_bytes: int | float) -> float:
    """
    Convert bytes to MB.
    """
    return float(num_bytes) / (1024 * 1024)


def gdc_headers(token: str | None) -> dict[str, str]:
    """
    Build optional GDC auth headers.
    """
    if token:
        return {"X-Auth-Token": token}

    return {}


def make_wasabi_client(
    endpoint_url: str,
    region: str,
    access_key_env: str,
    secret_key_env: str,
):
    """
    Create an S3 compatible Wasabi client.

    boto3 is imported only when Wasabi upload is requested.
    """
    try:
        import boto3
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "boto3 is required for Wasabi upload. Install with: python -m pip install boto3"
        ) from exc

    access_key = read_required_env(access_key_env, "Wasabi access")
    secret_key = read_required_env(secret_key_env, "Wasabi secret access")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def wasabi_object_key(prefix: str, file_name: str) -> str:
    """
    Build the Wasabi object key for one slide.
    """
    safe_file_name = Path(file_name).name
    clean_prefix = prefix.strip("/")

    if clean_prefix:
        return f"{clean_prefix}/{safe_file_name}"

    return safe_file_name


def wasabi_object_exists(client, bucket: str, key: str) -> bool:
    """
    Check if a Wasabi object exists.
    """
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:
        response = getattr(exc, "response", {})
        error = response.get("Error", {})
        code = str(error.get("Code", ""))

        if code in {"404", "NoSuchKey", "NotFound"}:
            return False

        raise


def remove_local_copy_if_requested(
    out_dir: Path,
    file_name: str,
    delete_local: bool,
) -> None:
    """
    Delete a matching local SVS file if requested.

    Useful after switching from local staging to direct Wasabi streaming.
    """
    if not delete_local:
        return

    local_path = out_dir / Path(file_name).name
    part_path = out_dir / f"{Path(file_name).name}.part"

    for path in (local_path, part_path):
        if path.exists():
            path.unlink()
            print(f"Deleted local file: {path}")


def download_one_slide_to_local(
    file_id: str,
    file_name: str,
    out_dir: Path,
    token: str | None,
    overwrite: bool,
    timeout_seconds: int,
    chunk_size_bytes: int,
) -> Path:
    """
    Download one GDC file to local disk.

    This is not used when --upload-wasabi is enabled.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_file_name = Path(file_name).name
    out_path = out_dir / safe_file_name
    part_path = out_dir / f"{safe_file_name}.part"

    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        print(f"SKIP existing local file: {out_path}")
        return out_path

    if part_path.exists():
        part_path.unlink()

    url = f"{GDC_DATA_BASE_URL}/{file_id}"

    print(f"Downloading to local file: {safe_file_name}")
    print(f"  target: {out_path}")

    with requests.get(
        url,
        headers=gdc_headers(token),
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
                    print_progress(downloaded, total_bytes)
                    next_report += 64 * 1024 * 1024

    if total_bytes and downloaded != total_bytes:
        raise IOError(
            f"Incomplete local download for {safe_file_name}: "
            f"got {downloaded} bytes, expected {total_bytes} bytes"
        )

    part_path.replace(out_path)
    print(f"DONE local download: {safe_file_name} ({mb(downloaded):.1f} MB)")

    return out_path


def print_progress(done_bytes: int, total_bytes: int) -> None:
    """
    Print transfer progress.
    """
    if total_bytes:
        print(f"  progress: {mb(done_bytes):.1f} MB / {mb(total_bytes):.1f} MB")
    else:
        print(f"  progress: {mb(done_bytes):.1f} MB")


def upload_buffer_part(
    client,
    bucket: str,
    key: str,
    upload_id: str,
    part_number: int,
    payload: bytes,
) -> dict[str, int | str]:
    """
    Upload one multipart chunk to Wasabi.
    """
    response = client.upload_part(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        PartNumber=part_number,
        Body=payload,
    )

    return {
        "PartNumber": part_number,
        "ETag": response["ETag"],
    }


def stream_gdc_slide_to_wasabi(
    file_id: str,
    file_name: str,
    client,
    bucket: str,
    key: str,
    token: str | None,
    overwrite: bool,
    timeout_seconds: int,
    chunk_size_bytes: int,
    part_size_bytes: int,
) -> None:
    """
    Stream one GDC file directly into Wasabi using multipart upload.

    No full local SVS file is written.
    Data is buffered in memory up to part_size_bytes.
    """
    if wasabi_object_exists(client, bucket, key) and not overwrite:
        print(f"SKIP existing Wasabi object: s3://{bucket}/{key}")
        return

    if part_size_bytes < 5 * 1024 * 1024:
        raise ValueError("Wasabi multipart part size must be at least 5 MB")

    url = f"{GDC_DATA_BASE_URL}/{file_id}"

    print(f"Streaming GDC to Wasabi: {Path(file_name).name}")
    print(f"  file_id: {file_id}")
    print(f"  target : s3://{bucket}/{key}")

    upload_id = None
    parts = []
    buffer = bytearray()
    part_number = 1
    read_bytes = 0
    uploaded_bytes = 0
    next_report = 64 * 1024 * 1024

    try:
        with requests.get(
            url,
            headers=gdc_headers(token),
            stream=True,
            timeout=timeout_seconds,
        ) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length", 0) or 0)

            multipart = client.create_multipart_upload(
                Bucket=bucket,
                Key=key,
                ContentType="application/octet-stream",
            )
            upload_id = multipart["UploadId"]

            for chunk in response.iter_content(chunk_size=chunk_size_bytes):
                if not chunk:
                    continue

                read_bytes += len(chunk)
                buffer.extend(chunk)

                while len(buffer) >= part_size_bytes:
                    payload = bytes(buffer[:part_size_bytes])
                    del buffer[:part_size_bytes]

                    part = upload_buffer_part(
                        client=client,
                        bucket=bucket,
                        key=key,
                        upload_id=upload_id,
                        part_number=part_number,
                        payload=payload,
                    )
                    parts.append(part)

                    uploaded_bytes += len(payload)
                    part_number += 1

                    if uploaded_bytes >= next_report:
                        print_progress(uploaded_bytes, total_bytes)
                        next_report += 64 * 1024 * 1024

            if buffer:
                payload = bytes(buffer)
                part = upload_buffer_part(
                    client=client,
                    bucket=bucket,
                    key=key,
                    upload_id=upload_id,
                    part_number=part_number,
                    payload=payload,
                )
                parts.append(part)
                uploaded_bytes += len(payload)

            if total_bytes and read_bytes != total_bytes:
                raise IOError(
                    f"Incomplete GDC stream for {Path(file_name).name}: "
                    f"got {read_bytes} bytes, expected {total_bytes} bytes"
                )

        client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )

    except Exception:
        if upload_id:
            client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )
        raise

    print(f"DONE Wasabi stream: s3://{bucket}/{key} ({mb(uploaded_bytes):.1f} MB)")


def print_run_summary(
    clinical_csv: Path,
    out_dir: Path,
    all_rows: list[dict[str, str]],
    unique_records: list[dict[str, str]],
    selected_records: list[dict[str, str]],
    upload_wasabi: bool,
    wasabi_bucket: str,
    wasabi_prefix: str,
) -> None:
    """
    Print run settings.
    """
    print(f"Clinical CSV: {clinical_csv}")
    print(f"Rows read   : {len(all_rows)}")
    print(f"Unique files: {len(unique_records)}")
    print(f"Selected files for this run: {len(selected_records)}")

    if upload_wasabi:
        print("Mode        : stream GDC directly to Wasabi")
        print(f"Wasabi bucket: {wasabi_bucket}")
        print(f"Wasabi prefix: {wasabi_prefix}")
    else:
        print("Mode        : local download")
        print(f"Output dir  : {out_dir}")


def main() -> None:
    """
    Run local download or direct Wasabi streaming workflow.
    """
    config_path = parse_config_path()
    config = load_config(config_path)
    args = parse_args(config_path, config)

    clinical_csv = resolve_repo_path(args.clinical_csv)
    out_dir = resolve_repo_path(args.out_dir)

    rows = read_clinical_csv(clinical_csv)
    unique_records = unique_file_records(rows)
    selected_records = select_records(
        records=unique_records,
        start=args.start,
        limit=args.limit,
        process_all=args.all,
    )

    print_run_summary(
        clinical_csv=clinical_csv,
        out_dir=out_dir,
        all_rows=rows,
        unique_records=unique_records,
        selected_records=selected_records,
        upload_wasabi=args.upload_wasabi,
        wasabi_bucket=args.wasabi_bucket,
        wasabi_prefix=args.wasabi_prefix,
    )

    if not selected_records:
        print("No files selected. Nothing to do.")
        return

    if args.upload_wasabi and not args.wasabi_bucket:
        raise ValueError("Wasabi upload requested, but no bucket was provided.")

    gdc_token = read_optional_env(args.gdc_token_env)
    chunk_size_bytes = args.chunk_size_mb * 1024 * 1024
    part_size_bytes = args.multipart_part_size_mb * 1024 * 1024

    wasabi_client = None
    if args.upload_wasabi and not args.dry_run:
        wasabi_client = make_wasabi_client(
            endpoint_url=args.wasabi_endpoint_url,
            region=args.wasabi_region,
            access_key_env=args.wasabi_access_key_env,
            secret_key_env=args.wasabi_secret_key_env,
        )

    for i, row in enumerate(selected_records, start=1):
        file_id = row["file_id"].strip()
        file_name = row["file_name"].strip()
        patient_id = (row.get("patient_id") or "").strip()
        wasabi_key = wasabi_object_key(args.wasabi_prefix, file_name)

        print("")
        print(f"[{i}/{len(selected_records)}] patient_id={patient_id}")
        print(f"  file_name : {file_name}")
        print(f"  file_id   : {file_id}")

        if args.upload_wasabi:
            print(f"  wasabi key: s3://{args.wasabi_bucket}/{wasabi_key}")

        if args.dry_run:
            print("DRY RUN only. No file downloaded or uploaded.")
            continue

        if args.upload_wasabi:
            stream_gdc_slide_to_wasabi(
                file_id=file_id,
                file_name=file_name,
                client=wasabi_client,
                bucket=args.wasabi_bucket,
                key=wasabi_key,
                token=gdc_token,
                overwrite=args.overwrite_wasabi,
                timeout_seconds=args.timeout_seconds,
                chunk_size_bytes=chunk_size_bytes,
                part_size_bytes=part_size_bytes,
            )
            remove_local_copy_if_requested(
                out_dir=out_dir,
                file_name=file_name,
                delete_local=args.delete_local_after_upload,
            )
        else:
            download_one_slide_to_local(
                file_id=file_id,
                file_name=file_name,
                out_dir=out_dir,
                token=gdc_token,
                overwrite=args.overwrite_download,
                timeout_seconds=args.timeout_seconds,
                chunk_size_bytes=chunk_size_bytes,
            )

    print("")
    print("Slide transfer workflow complete.")


if __name__ == "__main__":
    main()