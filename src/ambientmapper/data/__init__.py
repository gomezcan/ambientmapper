# ambientmapper/data/__init__.py

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Union


def get_default_layout_path() -> Path:
    """
    Return a filesystem path to the bundled 96-well Tn5 layout file.

    This works whether the package is installed from source, wheel, or zip.
    """
    with resources.path(__name__, "96well_Tn5_bc_layout.txt") as p:
        return p
