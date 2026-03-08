# GDC API Output Specification

This document defines the exact output format for a GDC (Genomic Data Commons) API script that fetches TCGA data for use with the TIL Analyzer pipeline. The output is designed to match our existing data formats and pipeline expectations.

---

## 1. Target Pipeline: TIL Analyzer

The TIL Analyzer expects two input formats:

1. **Annotations CSV** – used directly for training (`combined_training.csv`, `annotations_chip_*.csv`)
2. **Clinical CSV** – used by `generateAnnotations.py` to produce annotations when images are in a separate folder

---

## 2. Required Output: Annotations CSV (Primary)

This is the **direct input** for the TIL Analyzer training pipeline. One row per image/sample.

### Required Columns

| Column         | Type   | Description                                                                 |
|----------------|--------|-----------------------------------------------------------------------------|
| `image_path`   | string | Full or relative path to the image file. Must match the downloaded file.   |
| `survival_time`| float  | Survival time in **days**. Use `days_to_death` if dead, else `days_to_last_follow_up`. |
| `death_occurred` | int  | `1` = Dead (event observed), `0` = Alive (censored). Maps from `vital_status`. |

### Sample Annotations CSV

```csv
image_path,survival_time,death_occurred
./images/TCGA-A1-A0SD-01Z-00-DX1.png,456,1
./images/TCGA-A1-A0SE-01Z-00-DX1.png,1095,0
./images/TCGA-A1-A0SF-01Z-00-DX1.png,730,1
```

### Mapping from GDC Fields

| Our Column      | GDC Source                                      | Notes                                                                 |
|-----------------|--------------------------------------------------|----------------------------------------------------------------------|
| `image_path`    | `file_name` + download path                      | Construct as `{images_dir}/{file_name}` after download              |
| `survival_time` | `days_to_death` if Dead; else `days_to_last_follow_up` | If both missing, use `'--` and fall back to alternate field if available |
| `death_occurred`| `vital_status`                                  | "Dead" → 1, "Alive" → 0                                             |

---

## 3. Extended Output: Clinical CSV (Optional)

Used when running `generateAnnotations.py` to match images to clinical data. Matches the format expected by `generateAnnotations.py` (columns 1, 4, 5, 40).

### Required Columns (for generateAnnotations)

| Column Index | Column Name     | Type   | Description                                      |
|--------------|-----------------|--------|--------------------------------------------------|
| 0            | (implicit)      | -      | Row/Patient ID                                   |
| 1            | `file_name`     | string | Image filename (must match files in images dir)   |
| 4            | `survival_time` | int    | Primary survival time in days                    |
| 5            | `vital_status`  | string | "Dead" or "Alive"                                |
| 40            | `survival_time_alt` | int | Fallback survival (e.g. days_to_last_follow_up)  |

### Sample Clinical CSV

```csv
case_id,file_name,project_id,survival_time,vital_status,survival_time_alt
TCGA-A1-A0SD,TCGA-A1-A0SD-01Z-00-DX1.png,TCGA-BRCA,456,Dead,456
TCGA-A1-A0SE,TCGA-A1-A0SE-01Z-00-DX1.png,TCGA-BRCA,1095,Alive,1095
TCGA-A1-A0SF,TCGA-A1-A0SF-01Z-00-DX1.png,TCGA-BRCA,730,Dead,730
```

---

## 4. Recommended GDC Filters (Typical UI Selections)

Apply these when querying the GDC API to get a usable cohort:

| Filter              | Value / Logic                                      | Reason                                      |
|---------------------|----------------------------------------------------|---------------------------------------------|
| **Data Category**   | `Slide Image` or `Diagnostic Slide`                | TIL uses H&E whole-slide or derived images  |
| **Data Type**       | `Slide Image` (or derived feature images)          | Need image files, not raw sequencing       |
| **Project**         | e.g. `TCGA-BRCA`, `TCGA-LUAD`                      | Project-specific; user selects              |
| **Vital status**    | Include both Alive and Dead                        | Need both for survival analysis             |
| **days_to_death**   | Not null (for Dead)                                | Required for event cases                    |
| **days_to_last_follow_up** | Not null (for Alive)                      | Required for censored cases                 |
| **Sample type**     | `Primary Tumor` (01)                                | Usually want primary, not metastatic        |
| **Exclude**         | Cases with no survival data                        | Drop rows with missing survival fields      |

### Exclusion Rules

- Exclude cases where both `days_to_death` and `days_to_last_follow_up` are null or `'--`
- Exclude cases with `vital_status` not in `["Alive", "Dead"]`
- Optionally exclude very short follow-up (e.g. &lt; 30 days) if desired

---

## 5. Complete Column Reference (All Outputs)

For maximum flexibility, the GDC script can output these columns. **Bold** = required for TIL.

### Annotations CSV (minimal)

| Column          | Required | Type   |
|-----------------|----------|--------|
| **image_path**  | Yes      | string |
| **survival_time** | Yes    | float  |
| **death_occurred** | Yes   | int (0 or 1) |

### Clinical CSV (extended)

| Column                 | Required | Type   | GDC Source                          |
|------------------------|----------|--------|-------------------------------------|
| case_id                | No       | string | `case_id` or `submitter_id`         |
| file_name              | Yes      | string | `file_name`                         |
| file_id                | No       | string | `file_id` (for download)            |
| project_id             | No       | string | `project.project_id`               |
| survival_time          | Yes      | int    | `days_to_death` or `days_to_last_follow_up` |
| vital_status           | Yes      | string | `vital_status`                      |
| days_to_death          | No       | int?   | Raw GDC value                       |
| days_to_last_follow_up | No       | int?   | Raw GDC value                       |
| gender                 | No       | string | `demographic.gender`                |
| age_at_index           | No       | int?   | `demographic.days_to_birth` (convert)|

---

## 6. File Naming Conventions

- **Annotations**: `annotations_gdc_{project_id}.csv` or `combined_training_gdc.csv`
- **Clinical**: `clinical_gdc_{project_id}.csv`
- **Images**: Store in `data/til/images/{project_id}/` or similar; `image_path` should reflect this

---

## 7. Validation Checklist

Before feeding output to the TIL pipeline:

- [ ] No null/empty `survival_time` (or handle `'--` per `generateAnnotations` logic)
- [ ] `death_occurred` is 0 or 1 only
- [ ] `image_path` values match actual files on disk (after download)
- [ ] At least some Alive (0) and some Dead (1) for survival analysis
- [ ] Survival times &gt; 0

---

## 8. Packaging Options

- **GitHub repo**: Script + this spec + README; team clones and runs with config (project ID, output dir, etc.)
- **Standalone script**: Single Python file with `requests`; user sets project, runs, gets CSV + optional image download

Both should produce the same CSV format so the TIL Analyzer works without modification.
