"""Package init — anchors silent ML caches under the repo root.

PyTorch (``TORCH_HOME``), HuggingFace transformers / hub
(``HF_HOME``, ``HF_HUB_CACHE``, ``TRANSFORMERS_CACHE``) default their
download caches to ``~/.cache/...`` if their respective env vars are
unset. fair-esm, ankh, and any future HF-based PLM the project picks
up therefore silently litter the user's home directory with multi-GB
model checkpoints — *outside* the EnzymeExplorer working tree, where
``setup_prod.sh`` can't reach to clean up and ``--workdir`` can't
constrain.

This module sets every relevant cache variable to a ``<repo>/.cache/``
sub-directory before any ML dependency gets imported. ``setdefault``
semantics preserve user overrides: if the operator already exported
e.g. ``HF_HOME=/scratch/hf`` to share a cache across runs, that wins.

Must run before:
  * ``import ankh`` / ``import transformers`` (HuggingFace caches)
  * ``import esm`` (torch hub via ``torch.hub.download_url_to_file``)
  * Any ``from torch import ...`` that triggers a hub download

Since ``enzymeexplorer/__init__.py`` is the first module Python loads
for any ``from enzymeexplorer.... import X`` (including all four CLI
entry points), being inside the package init guarantees that ordering.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROOT = _REPO_ROOT / ".cache"


def _set_default_env(name: str, value: str) -> None:
    """Pin ``name`` to ``value`` only if the operator hasn't already
    set it. Mirrors ``dict.setdefault`` semantics on ``os.environ``."""
    if not os.environ.get(name):
        os.environ[name] = value


# HuggingFace: ``HF_HOME`` controls everything (hub cache, datasets,
# tokenizer cache). ``HF_HUB_CACHE`` pins the hub sub-dir explicitly
# so older transformers versions that read it directly also see the
# repo-local path. ``TRANSFORMERS_CACHE`` is intentionally NOT set —
# it's deprecated in transformers ≥4 and emits a FutureWarning when
# present; ``HF_HOME`` covers the same surface.
_set_default_env("HF_HOME",      str(_CACHE_ROOT / "huggingface"))
_set_default_env("HF_HUB_CACHE", str(_CACHE_ROOT / "huggingface" / "hub"))

# PyTorch: ``TORCH_HOME`` controls ``torch.hub`` (fair-esm fetches
# checkpoints via this path).
_set_default_env("TORCH_HOME",         str(_CACHE_ROOT / "torch"))

# Matplotlib config / cache (avoids ``~/.config/matplotlib`` and
# ``~/.cache/matplotlib`` writes — these matter on locked-down
# read-only-home cluster nodes).
_set_default_env("MPLCONFIGDIR",       str(_CACHE_ROOT / "matplotlib"))

# Some HF libraries probe ``XDG_CACHE_HOME`` before ``$HOME`` — pin it
# too, but inside the repo so nothing leaks back to ``~/.cache``.
_set_default_env("XDG_CACHE_HOME",     str(_CACHE_ROOT / "xdg"))

# Make sure the cache root exists so the libraries don't error out on
# first write. Sub-dirs are created lazily by each library.
try:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
except OSError:
    # Read-only repo (rare, e.g. baked into a Docker image with the
    # cache mounted elsewhere via env override): tolerate silently.
    pass
