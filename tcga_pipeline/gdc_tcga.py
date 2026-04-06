"""
GDC TCGA helper module - Reusable module for querying GDC and building TCGA datasets

Purpose
1. Query /cases for a TCGA project and pull survival and demographic fields
2. Query /files for the same project and pull slide file metadata that can be linked back to cases
3. Apply cohort filtering rules such as valid survival data and optional Primary Tumor filtering
4. Build spec compliant annotations and clinical CSV outputs

Notes
- Output schema includes patient_id and gdc_case_uuid for stable joins
- Most runtime behavior (fields, filters, pagination, outputs) is set in config doc

reusable Python wrapper around the GDC REST API plus dataset building utilities.
"""

import csv
import sys
from pathlib import Path

import yaml
import requests

#add repo root directory to import path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

#import the TCGA barcode parser (try lowercase, then capitalized)
try:
    from gigatime_analyzer.survival.tcga_ids import tcga_barcode_from_slide_name
except ModuleNotFoundError:
    from GigaTIME_analyzer.survival.tcga_ids import tcga_barcode_from_slide_name

# load and validate YAML config for pipeline settings
def load_config(config_path: str | None = None) -> dict:
    """
    Load YAML config once at startup.
    If config_path is None, look for gdc_config.yaml in the same folder.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "gdc_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def validate_config(config: dict) -> None:
    """
    Validate required structure and expected fields in YAML config.

    Ensures:
    - required top-level sections exist
    - project_id is provided
    - survival status values are valid
    - output schema matches expected pipeline format
    """
    required_top = [
        "user_settings",
        "output_settings",
        "advanced_settings",
        "internal_schema",
    ]

    for key in required_top:
        if key not in config:
            raise ValueError(f"Missing top level config section: {key}")

    project_id = config["user_settings"].get("project_id")
    if not project_id:
        raise ValueError("user_settings.project_id must not be empty")

    valid_statuses = config["user_settings"].get("valid_vital_statuses", [])
    if sorted(valid_statuses) != ["Alive", "Dead"]:
        raise ValueError(
            "user_settings.valid_vital_statuses should contain Alive and Dead"
        )

    ann_fields = config["internal_schema"].get("annotations_fields", [])
    clin_fields = config["internal_schema"].get("clinical_fields", [])

    expected_ann = ["image_path", "patient_id", "survival_time", "death_occurred"]
    expected_clin = [
        "patient_id",
        "case_id",
        "gdc_case_uuid",
        "file_name",
        "file_id",
        "project_id",
        "survival_time",
        "vital_status",
        "survival_time_alt",
        "days_to_death",
        "days_to_last_follow_up",
        "gender",
        "age_at_index",
    ]

    if ann_fields != expected_ann:
        raise ValueError(
            f"annotations_fields does not match implemented schema. Got: {ann_fields}"
        )

    if clin_fields != expected_clin:
        raise ValueError(
            f"clinical_fields does not match implemented schema. Got: {clin_fields}"
        )

# initialize config and extract key sections
CONFIG = load_config()
validate_config(CONFIG)

# shorthand access to config sections
USER = CONFIG["user_settings"]
OUTPUT = CONFIG["output_settings"]
ADVANCED = CONFIG["advanced_settings"]
SCHEMA = CONFIG["internal_schema"]

# runtime settings (from config)
DEBUG = ADVANCED.get("debug", False)
BASE = ADVANCED.get("base_url", "https://api.gdc.cancer.gov")


def gdc_post(endpoint: str, payload: dict, token: str | None = None) -> dict:
    """
    Send a POST request to a GDC search and retrieval endpoint.

    Why POST
    GET queries can become too long once filters and fields are included.
    POST lets you send the same query as JSON safely and cleanly.

    Args
    endpoint: "cases" or "files" or another search endpoint
    payload: JSON body containing keys like filters, fields, expand, format, size, from
    token: optional GDC auth token, only required for controlled access data

    Returns
    Parsed JSON as a Python dict.

    Notes about response shape
    For search endpoints like /cases and /files the actual records are typically in:
    response["data"]["hits"]
    and pagination metadata is in:
    response["data"]["pagination"]
    """
    # Build full URL like https://api.gdc.cancer.gov/cases
    url = f"{BASE}/{endpoint.lstrip('/')}"

    # Tell the server we are sending JSON in the request body
    headers = {"Content-Type": "application/json"}

    # Attach auth header if a token is provided
    if token:
        headers["X-Auth-Token"] = token

    # Send request and limit how long we wait
    # use config timeout for API requests
    r = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=ADVANCED["timeout_seconds"],
    )

    # Raise an exception for HTTP errors like 400 or 403
    r.raise_for_status()

    # Parse JSON response body into a Python dictionary
    return r.json()


def build_filter(field: str, values, op: str = "in") -> dict:
    """
    Build one GDC filter clause.

    Args
    field: a valid field path for the endpoint, for example:
      "project.project_id" when querying /cases
      "cases.project.project_id" when querying /files
    values: a single value or list of values to match
    op: GDC operator, most commonly "in" or "="

    Returns
    A dict in the GDC filter format:
    {"op": op, "content": {"field": field, "value": [values...]}}
    """
    # GDC expects a list under "value", even if you only match one thing
    if not isinstance(values, list):
        values = [values]

    return {"op": op, "content": {"field": field, "value": values}}


def and_filter(*filters: dict) -> dict:
    """
    Combine multiple filter clauses with logical AND.

    Example
    and_filter(
      build_filter("cases.project.project_id", "TCGA-BRCA", op="="),
      build_filter("data_format", "SVS", op="=")
    )
    """
    return {"op": "and", "content": list(filters)}


def get_cases_survival(project_id: str, size: int | None = None, offset: int = 0, token: str | None = None) -> dict:
    """
    Query /cases for a project and return survival relevant fields.

    Args
    project_id: TCGA project id like "TCGA-BRCA"
    size: number of records to request per call
    offset: pagination offset, passed as "from"
    token: optional auth token

    Returns
    JSON response dict from /cases.
    Records are usually in response["data"]["hits"].

    Notes
    Survival fields can appear in different nodes across releases and projects.
    This starter pulls common ones from demographic and diagnoses.
    """
    # use fields defined in config
    fields = SCHEMA["cases_fields"]

    # use config page size unless one is provided
    if size is None:
        size = ADVANCED["page_size"]

    payload = {
        # restrict to one project at the case level
        "filters": and_filter(build_filter("project.project_id", project_id, op="=")),
        # API expects a comma separated string for fields
        "fields": ",".join(fields),
        # include nested objects in the response
        "expand": ADVANCED.get("cases_expand", "diagnoses,demographic"),
        "format": "JSON",
        # pagination controls
        "size": str(size),
        "from": str(offset),
    }

    return gdc_post("cases", payload, token=token)


def get_files_for_project(project_id: str, size: int | None = None, offset: int = 0, token: str | None = None) -> dict:
    """
    Query /files for a project and return file metadata that can be linked to cases.
    """
    fields = SCHEMA["files_fields"]

    # default to config batch size if not provided
    if size is None:
        size = ADVANCED["page_size"]

    payload = {
        # /files filters project via cases.project.project_id
        "filters": and_filter(build_filter("cases.project.project_id", project_id, op="=")),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
        "from": str(offset),
    }

    return gdc_post("files", payload, token=token)


def get_slide_files(project_id: str, size: int | None = None, offset: int = 0, token: str | None = None) -> dict:
    """
    Retrieve slide-like files for a project.

    Filters files by project and data_format (e.g., SVS) so that returned files
    correspond to histology slides usable for downstream image-based analysis.
    """
    fields = SCHEMA["files_fields"]

   # default to config page size if not provided
    if size is None:
        size = ADVANCED["page_size"]

    slide_filter = and_filter(
        build_filter("cases.project.project_id", project_id, op="="),
        # restrict to slide file formats defined in config (typically SVS)
        build_filter("data_format", USER["slide_data_format"], op="="),
    )

    payload = {
        # /files requires filtering through cases.project.project_id
        "filters": slide_filter,
        "fields": ",".join(fields),
        "format": "JSON",
        "size": str(size),
        "from": str(offset),
    }

    return gdc_post("files", payload, token=token)

def get_days_to_last_follow_up(case_hit: dict):
    """
    Return the first non-empty days_to_last_follow_up found in diagnoses.
    If nothing usable is found, return None.
    """
    diagnoses = case_hit.get("diagnoses", [])

    for dx in diagnoses:
        value = dx.get("days_to_last_follow_up")
        if value not in [None, "", "--"]:
            return value

    return None

def get_days_to_death(case_hit: dict):
    """
    Return days_to_death from demographic if present,
    otherwise the first non-empty days_to_death in diagnoses.
    """
    demographic = case_hit.get("demographic", {}) or {}
    value = demographic.get("days_to_death")

    if value not in [None, "", "--"]:
        return value

    diagnoses = case_hit.get("diagnoses", [])
    for dx in diagnoses:
        value = dx.get("days_to_death")
        if value not in [None, "", "--"]:
            return value

    return None

def is_primary_tumor_file(file_hit: dict) -> bool:
    """
    Return True if any linked sample for this file is Primary Tumor.

    Why:
    Section 4 recommends using Primary Tumor (01) samples when possible.
    """
    linked_cases = file_hit.get("cases", [])

    for case in linked_cases:
        samples = case.get("samples", [])
        for sample in samples:
            if sample.get("sample_type") == "Primary Tumor":
                return True

    return False

def extract_case_survival(case_hit: dict) -> dict:
    """
    Pull survival and demographic values from one case record.

    Returns a dict with:
    - submitter_id
    - case_id
    - vital_status
    - days_to_death
    - days_to_last_follow_up
    - gender
    - days_to_birth
    """
    demographic = case_hit.get("demographic", {}) or {}

    return {
        "submitter_id": case_hit.get("submitter_id"),
        "case_id": case_hit.get("case_id") or case_hit.get("submitter_id"),
        "gdc_case_uuid": case_hit.get("case_id"),
        "vital_status": demographic.get("vital_status"),
        "days_to_death": get_days_to_death(case_hit),
        "days_to_last_follow_up": get_days_to_last_follow_up(case_hit),
        "gender": demographic.get("gender"),
        "days_to_birth": demographic.get("days_to_birth"),
    }

def build_case_map(cases_resp: dict) -> dict:
    """
    Build a lookup dictionary keyed by submitter_id.

    Example:
    case_map["TCGA-BH-A18H"] -> survival info for that case
    """
    case_map = {}
    cases_hits = cases_resp.get("data", {}).get("hits", [])

    for case_hit in cases_hits:
        case_info = extract_case_survival(case_hit)
        submitter_id = case_info.get("submitter_id")

        if DEBUG:
            print(f"[DEBUG] submitter_id: {submitter_id}")

        if submitter_id:
            case_map[submitter_id] = case_info

    return case_map

def clean_survival_value(value):
    """
    Normalize raw survival values coming from the GDC API.

    GDC fields like days_to_death or days_to_last_follow_up can be:
    - None
    - empty string ""
    - placeholder "--"
    - numeric (int or float)
    - sometimes string representations of numbers

    What this function does:
    - Filters out invalid placeholders → returns None
    - Converts valid values → float

    Why float:
    Keeps everything consistent for downstream analysis and CSV output.

    Example:
    "456" → 456.0
    "--" → None
    None → None
    """
    if value in ADVANCED["invalid_placeholders"]:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_age_at_index(days_to_birth):
    """
    Convert GDC days_to_birth into age at index in years.

    GDC commonly stores days_to_birth as a negative number.
    """
    value = clean_survival_value(days_to_birth)

    if value is None:
        return None

    return int(abs(value) / 365.25)

def is_valid_survival_case(case_info: dict, min_follow_up_days: float | None = None) -> bool:
    """
    Check whether a case has the survival information needed for downstream use.

    A valid case must have an allowed vital_status and the matching time field:
    days_to_death for Dead cases or days_to_last_follow_up for Alive cases.
    """
    vital_status = case_info.get("vital_status")
    days_to_death = clean_survival_value(case_info.get("days_to_death"))
    days_to_last_follow_up = clean_survival_value(case_info.get("days_to_last_follow_up"))

    # keep only configured survival status values
    if vital_status not in USER["valid_vital_statuses"]:
        return False

    if vital_status == "Dead" and days_to_death is None:
        return False

    if vital_status == "Alive" and days_to_last_follow_up is None:
        return False

    if min_follow_up_days is not None:
        if vital_status == "Dead" and days_to_death is not None and days_to_death < min_follow_up_days:
            return False
        if vital_status == "Alive" and days_to_last_follow_up is not None and days_to_last_follow_up < min_follow_up_days:
            return False

    return True

def compute_survival_fields(case_info: dict):
    """
    Convert raw case survival data into final model-ready fields.

    Input:
    case_info = {
        "vital_status": "Alive" or "Dead",
        "days_to_death": value,
        "days_to_last_follow_up": value
    }

    Output:
    (survival_time, death_occurred)

    Where:
    - survival_time = numeric value in days
    - death_occurred = 1 if Dead, 0 if Alive

    Logic (based on survival analysis conventions):
    - If patient is Dead:
        → event occurred → use days_to_death
    - If patient is Alive:
        → censored → use days_to_last_follow_up

    This matches Kaplan–Meier / survival modeling expectations.
    """
    vital_status = case_info.get("vital_status")
    
    days_to_death = clean_survival_value(case_info.get("days_to_death"))
    days_to_last_follow_up = clean_survival_value(case_info.get("days_to_last_follow_up"))

    # If the patient is dead, survival time is days to death.
    # Event value (usually 1) is pulled from config to support various downstream models.
    if vital_status == "Dead" and days_to_death is not None:
        return days_to_death, ADVANCED["dead_event_value"]

    # If the patient is alive, survival time is days to last follow-up (censored).
    # Event value (usually 0) is pulled from config.
    if vital_status == "Alive" and days_to_last_follow_up is not None:
        return days_to_last_follow_up, ADVANCED["alive_event_value"]

    return None, None

def build_annotations(
    cases_resp: dict,
    files_resp: dict,
    images_dir: str = "./images",
    primary_tumor_only: bool = True,
    min_follow_up_days: float | None = None,
) -> list[dict]:
    """
    Join case survival data with slide files and build annotation rows.

    Output columns:
    - image_path
    - survival_time
    - death_occurred
    """
    rows = []
    case_map = build_case_map(cases_resp)
    files_hits = files_resp.get("data", {}).get("hits", [])

    for file_hit in files_hits:
        file_name = file_hit.get("file_name")
        linked_cases = file_hit.get("cases", [])

        if not file_name or not linked_cases:
            continue

        # section 4: keep Primary Tumor files only if requested
        if primary_tumor_only and not is_primary_tumor_file(file_hit):
            if DEBUG:
                print(f"[FILTER] Skipping non-primary tumor file: {file_name}")
            continue
        
        # handle multiple linked cases
        submitter_id = None
        for c in linked_cases:
            sid = c.get("submitter_id")
            if sid:
                submitter_id = sid
                break

        # DEBUG: show what file we are processing
        if DEBUG:
            print(f"[JOIN] file={file_name} case={submitter_id}")

        case_info = case_map.get(submitter_id)
        if not case_info:
            if DEBUG:
                print(f"[DEBUG] No matching case for {submitter_id}")
            continue

        # section 4: enforce survival eligibility rules
        if not is_valid_survival_case(case_info, min_follow_up_days=min_follow_up_days):
            if DEBUG:
                print(f"[FILTER] Invalid survival case for {submitter_id}")
            continue

        survival_time, death_occurred = compute_survival_fields(case_info)

        if DEBUG:
            print(f"[SURVIVAL] time={survival_time}, event={death_occurred}")

        # Exclude unusable rows
        if survival_time in [None, "", "--"]:
            continue

        if death_occurred not in [0, 1]:
            continue

        patient_id = tcga_barcode_from_slide_name(file_name)
        row = {
            "image_path": f"{images_dir}/{file_name}",
            "patient_id": patient_id,
            "survival_time": survival_time,
            "death_occurred": death_occurred,
        }

        rows.append(row)

    return rows

def build_clinical_rows(
    cases_resp: dict,
    files_resp: dict,
    project_id: str,
    primary_tumor_only: bool = True,
    min_follow_up_days: float | None = None,
) -> list[dict]:
    """
    Build the extended clinical CSV rows.

    Output columns:
    - case_id
    - file_name
    - file_id
    - project_id
    - survival_time
    - vital_status
    - days_to_death
    - days_to_last_follow_up
    - gender
    - age_at_index
    """
    rows = []
    case_map = build_case_map(cases_resp)
    files_hits = files_resp.get("data", {}).get("hits", [])

    for file_hit in files_hits:
        file_name = file_hit.get("file_name")
        file_id = file_hit.get("file_id")
        linked_cases = file_hit.get("cases", [])

        if not file_name or not linked_cases:
            continue

        # keep Primary Tumor files only if requested
        if primary_tumor_only and not is_primary_tumor_file(file_hit):
            continue

        # choose first usable linked case
        submitter_id = None
        for c in linked_cases:
            sid = c.get("submitter_id")
            if sid:
                submitter_id = sid
                break

        if not submitter_id:
            continue

        case_info = case_map.get(submitter_id)
        if not case_info:
            continue

        # enforce section 4 survival rules
        if not is_valid_survival_case(case_info, min_follow_up_days=min_follow_up_days):
            continue

        survival_time, _ = compute_survival_fields(case_info)

        if survival_time is None:
            continue

        days_to_death = clean_survival_value(case_info.get("days_to_death"))
        days_to_last_follow_up = clean_survival_value(case_info.get("days_to_last_follow_up"))

        survival_time_alt = days_to_last_follow_up
        if survival_time_alt is None:
            survival_time_alt = days_to_death

        patient_id = tcga_barcode_from_slide_name(file_name)
        tcga_case_barcode = case_info.get("submitter_id") or patient_id

        gdc_uuid = case_info.get("gdc_case_uuid")

        row = {
            "patient_id": patient_id,
            "case_id": tcga_case_barcode,
            "gdc_case_uuid": gdc_uuid if gdc_uuid is not None else "",
            "file_name": file_name,
            "file_id": file_id,
            "project_id": project_id,
            "survival_time": int(survival_time),
            "vital_status": case_info.get("vital_status"),
            "survival_time_alt": int(survival_time_alt) if survival_time_alt is not None else None,
            "days_to_death": days_to_death,
            "days_to_last_follow_up": days_to_last_follow_up,
            "gender": case_info.get("gender"),
            "age_at_index": get_age_at_index(case_info.get("days_to_birth")),
        }

        rows.append(row)

    return rows

def write_annotations_csv(rows: list[dict], out_csv: str):
    """
    Write annotation rows to CSV.
    """
    if not rows:
        print("No annotation rows to write.")
        return

    # Write columns defined in config doc.
    fieldnames = SCHEMA["annotations_fields"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def write_clinical_csv(rows: list[dict], out_csv: str):
    """
    Write extended clinical rows to CSV.
    """
    if not rows:
        print("No clinical rows to write.")
        return

    # Write columns in the configured schema order.
    fieldnames = SCHEMA["clinical_fields"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def fetch_all_cases(project_id: str, token: str | None = None):
    """
    Retrieve all case records for a project by paginating through the API.
    """
    all_hits = []
    offset = 0
    
    # page through the API in configurable batch sizes to control request count and memory footprint
    # set in config doc
    size = ADVANCED["page_size"]

    while True:
        resp = get_cases_survival(project_id, size=size, offset=offset, token=token)
        hits = resp.get("data", {}).get("hits", [])

        # halt pagination if the API returns an empty page
        if not hits:
            break

        all_hits.extend(hits)

        if DEBUG:
            print(f"[PAGINATION][CASES] fetched {len(all_hits)}")

        # advance the starting offset to fetch the next block of records
        offset += size

        # stop if the current page returned fewer records than the maximum batch size
        if len(hits) < size:
            break

    return {"data": {"hits": all_hits}}


def fetch_all_files(project_id: str, token: str | None = None):
    """
    Retrieve all slide file records for a project by paginating through the API.
    """
    all_hits = []
    offset = 0
    
    # use configured page size to dictate /files pagination chunks
    # set in config doc
    size = ADVANCED["page_size"]

    while True:
        resp = get_slide_files(project_id, size=size, offset=offset, token=token)
        hits = resp.get("data", {}).get("hits", [])

        # halt pagination if the API returns an empty page
        if not hits:
            break

        all_hits.extend(hits)

        if DEBUG:
            print(f"[PAGINATION][FILES] fetched {len(all_hits)}")

        # advance the starting offset to fetch the next block of records
        offset += size

        # stop if the current page returned fewer records than the maximum batch size
        if len(hits) < size:
            break

    return {"data": {"hits": all_hits}}


def render_template(template: str, project_id: str) -> str:
    """
    Fill {project_id} in output filename templates.
    """
    return template.format(project_id=project_id)


# Save annotation rows to project-specific CSV file (name set in config doc)
def write_project_annotations_csv(project_id: str, rows: list[dict]):
    # set in config doc
    out_csv = render_template(OUTPUT["annotations_filename"], project_id)
    write_annotations_csv(rows, out_csv)
    return out_csv


# Save clinical rows to project-specific CSV file (name set in config doc)
def write_project_clinical_csv(project_id: str, rows: list[dict]):
    # set in config doc
    out_csv = render_template(OUTPUT["clinical_filename"], project_id)
    write_clinical_csv(rows, out_csv)
    return out_csv