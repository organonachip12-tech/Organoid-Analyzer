# TCGA Data Pipeline

A Python pipeline for querying the GDC (Genomic Data Commons) API and building structured datasets from TCGA cancer cohorts. It pulls patient survival data and H&E slide file metadata, filters for valid cohorts, and outputs two CSV files for downstream survival modeling.
The pipeline is fully config-driven - users should modify gdc_config.yaml rather than editing code.
---

## What This Pipeline Does

1. Queries the GDC `/cases` endpoint for patient survival and demographic data
2. Queries the GDC `/files` endpoint for H&E slide file metadata (SVS format)
3. Joins cases and files on `submitter_id` (the patient ID)
4. Filters to valid survival cases and Primary Tumor samples
5. Outputs two CSV files — one for model input, one for clinical reference

---

## Requirements

- Python 3.10 or higher
- Install dependencies:

```bash
pip install requests pyyaml
```

> `csv` is built into Python and does not need to be installed separately.

---

## Files

| File | Description |
|---|---|
| `gdc_tcga.py` | Core pipeline module — handles all API calls, filtering, and CSV writing |
| `run_gdc_tcga.py` | Run script — executes the pipeline |
| `gdc_config.yaml` | Config file — change project, filters, and output settings here |

---

## How to Run

1. Clone the repository and navigate to the project folder
2. Make sure your config file is set to the correct project (see Configuration section below)
3. Run the pipeline:

```bash
python run_gdc_tcga.py
```

> **Note:** confirm the exact run command with the team — this may be updated.

---

## Configuration

All pipeline settings are controlled through gdc_config.yaml.

The most important section is:

user_settings:
  project_id: TCGA-BRCA

You can also adjust filtering and runtime behavior in the config file if needed.

Do not modify internal_schema unless you are also updating the code.

### Changing the Cancer Cohort

To switch to a different TCGA project, change the `project_id`:

```yaml
user_settings:
  project_id: TCGA-PAAD   # Pancreatic Cancer
```

Common project IDs:
- `TCGA-BRCA` — Breast Cancer (used for testing)
- `TCGA-PAAD` — Pancreatic Cancer (primary target)

### Filters

```yaml
filters:
  slide_data_format: SVS        # file format for H&E slides
  primary_tumor_only: true      # set false to include other sample types
  min_follow_up_days: null      # set a number (e.g. 30) to exclude short follow-ups
```

### API Settings

```yaml
api:
  page_size: 200        # records fetched per API request
  timeout_seconds: 60   # how long to wait before a request times out
```

### Debug Mode

```yaml
debug: false  # set true to print detailed step-by-step logs while the pipeline runs
```

---

## Output Files

Running the pipeline produces two CSV files named after the project:

### `annotations_gdc_{project_id}.csv`
Input file for the survival model. Contains one row per slide.

| Column | Description |
|---|---|
| `patient_id` | TCGA patient barcode used for joins |
| `image_path` | Path to the H&E slide image file |
| `survival_time` | Days the patient was observed |
| `death_occurred` | 1 if the patient died, 0 if alive (censored) |

### `clinical_gdc_{project_id}.csv`
Full clinical reference file for human review and deeper analysis.

| Column | Description |
|---|---|
| `case_id` | Patient submitter ID (e.g. TCGA-BH-A18H) |
| `file_name` | Name of the linked slide file |
| `file_id` | GDC file UUID |
| `project_id` | TCGA project (e.g. TCGA-BRCA) |
| `survival_time` | Primary survival time in days |
| `vital_status` | Alive or Dead |
| `survival_time_alt` | Alternative survival time field |
| `days_to_death` | Raw days to death from GDC |
| `days_to_last_follow_up` | Raw days to last follow up from GDC |
| `gender` | Patient gender |
| `age_at_index` | Patient age in years at index date |

---

## Notes

- The pipeline was developed and tested on `TCGA-BRCA` (breast cancer)
- The primary target cohort is `TCGA-PAAD` (pancreatic cancer)
- No GDC authentication token is required for open access data
- Pagination is handled automatically — all records are fetched regardless of cohort size
