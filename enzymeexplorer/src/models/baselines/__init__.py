"""External-tool baseline models, exposed via PEP 562 lazy imports.

Same rationale as :mod:`enzymeexplorer.src.models`: previously this
package's eager re-exports meant that requesting any one baseline
(e.g. ``Blastp``) transitively loaded ``CLEAN`` — which imports
``gdown`` and ProFun even when the caller only wants BLASTp. Each
baseline is now imported on first attribute access only.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, str] = {
    "Blastp":     ".blastp",
    "Foldseek":   ".foldseek",
    "HMM":        ".hmm",
    "PfamSUPFAM": ".pfam_supfam",
    "CLEAN":      ".CLEAN",
}


def __getattr__(name: str) -> Any:
    submodule_path = _LAZY_IMPORTS.get(name)
    if submodule_path is None:
        raise AttributeError(
            f"module 'enzymeexplorer.src.models.baselines' has no attribute {name!r}"
        )
    submodule = importlib.import_module(submodule_path, package=__name__)
    cls = getattr(submodule, name)
    globals()[name] = cls
    return cls


def __dir__() -> list[str]:
    return sorted(set(_LAZY_IMPORTS.keys()) | set(globals().keys()))
