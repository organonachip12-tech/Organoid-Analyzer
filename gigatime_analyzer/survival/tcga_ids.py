"""Map TCGA slide paths or file names to patient barcodes (e.g. TCGA-BH-A18H)."""

import os


def tcga_barcode_from_slide_name(name):
    """
    Extract TCGA case barcode from a slide file name or path.

    Uses the first three hyphen-separated segments of the basename stem, e.g.:
    TCGA-BH-A18H-01A-01-TSA.<uuid>.svs -> TCGA-BH-A18H
    """
    if isinstance(name, (list, tuple)):
        name = name[0]
    base = os.path.basename(str(name).strip())
    if not base:
        return ""
    stem = base.split(".", 1)[0]
    parts = stem.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return stem
