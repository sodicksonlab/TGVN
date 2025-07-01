"""
TGVN – Leveraging side information in MR image reconstruction.
Copyright (c) 2025, New York University

Exposes high-level API surfaces so client code can simply write::

    import tgvn as tgvn
    model = tgvn.models.TGVN_1S(...)
"""

from importlib.metadata import version as _version

# Public sub-modules -----------------------------------------------------
from . import models
from . import data
from . import loss
from . import distributed

# Package version (reads from pyproject / setuptools metadata)
__version__: str
try:
    __version__ = _version("tgvn")
except Exception:
    __version__ = "0.0.0"

# What gets imported with "from tgvn import *"
__all__ = [
    "__version__",
    "models",
    "data",
    "loss",
    "distributed",
]
