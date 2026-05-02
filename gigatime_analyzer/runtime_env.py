"""
HPC-friendly runtime defaults: matplotlib/font caches when $HOME quota is full.

Call ``configure_runtime_caches()`` before importing matplotlib or other GUI/cache-heavy libs.
"""
from __future__ import annotations

import os
import tempfile


def configure_runtime_caches() -> None:
    """
    If MPLCONFIGDIR / XDG_CACHE_HOME are unset, put caches under $SCRATCH,
    $TMPDIR, or the system temp directory so imports succeed when ~/.config is full.
    """
    scratch = os.environ.get("SCRATCH") or os.environ.get("TMPDIR")
    if scratch:
        base = os.path.join(scratch, ".organoid_analyzer_cache")
    else:
        base = os.path.join(tempfile.gettempdir(), "organoid_analyzer_cache")

    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        base = tempfile.mkdtemp(prefix="organoid_analyzer_cache_")

    if not os.environ.get("MPLCONFIGDIR"):
        mpl = os.path.join(base, "matplotlib")
        os.makedirs(mpl, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl

    if not os.environ.get("XDG_CACHE_HOME"):
        xdg = os.path.join(base, "xdg_cache")
        os.makedirs(xdg, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = xdg
