# src/ambientmapper/data.py
from __future__ import annotations

import os
import shutil
from pathlib import Path
from importlib import resources

_DEFAULT_LAYOUT_CANDIDATES = [    
    "96well_Tn5_bc_layout.txt",    
]

def get_default_layout_path(filename: str = "96well_Tn5_bc_layout.txt") -> Path:
    """
    Return a real filesystem path to the bundled 96-well Tn5 barcode layout file.

    Works even if ambientmapper is installed as a wheel/zip (resource not a normal file),
    by copying it into a cache directory.
    """
    pkg = "ambientmapper.data"

    # Try requested filename first, then fall back to candidate names
    candidates = [filename] + [x for x in _DEFAULT_LAYOUT_CANDIDATES if x != filename]

    cache_dir = Path(os.getenv("AMBIENTMAPPER_CACHE", Path.home() / ".cache" / "ambientmapper"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None

    for name in candidates:
        try:
            traversable = resources.files(pkg).joinpath(name)
            if not traversable.is_file():
                continue

            out_path = cache_dir / name
            if out_path.exists() and out_path.stat().st_size > 0:
                return out_path

            # Materialize to a real path then copy to cache
            with resources.as_file(traversable) as tmp_path:
                shutil.copy2(tmp_path, out_path)
            return out_path
        except Exception as e:
            last_err = e
            continue

    msg = (
        "Bundled default layout file not found in the installed package.\n"
        f"Tried: {candidates}\n"
        "Fix: ensure src/ambientmapper/data/96well_Tn5_bc_layout.txt is included in package data.\n"
        "Workaround: pass --layout-file /path/to/your/layout.txt"
    )
    raise FileNotFoundError(msg) from last_err
