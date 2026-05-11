"""Per-invocation scratch directory for prediction & screening pipelines.

The pipelines fan out a lot of intermediate state across nested modules
(``prediction/domains.py``, ``structure_processing/{domain_detections,
get_structural_features,structural_algorithms,utils}``). Each of those
calls into ``tempfile.TemporaryDirectory()`` / ``tempfile.mkdtemp()``
*without* an explicit ``dir=``, so the directory choice falls back to
``tempfile.tempdir`` (in turn ``$TMPDIR`` / ``$TEMP`` / ``$TMP`` /
``/tmp``).

This module exposes a single context manager that:

* If the user gave a ``--workdir <parent>``: creates a fresh per-call
  subdirectory under ``parent``, points ``tempfile.tempdir`` at it,
  and removes the whole subtree on exit (unless ``keep`` is requested).
* If ``parent`` is ``None``: yields without changing anything so each
  nested ``tempfile.X`` call lands under the system tmp and is cleaned
  up by its own existing lifecycle.

The end-state guarantee is the same in both modes: no scratch left
behind after the entry-point returns.

The foldseek-DB cache at ``data/foldseek_cache/`` (configurable via
``$ENZYMEEXPLORER_FOLDSEEK_REF_DB``) is deliberately *not* touched by
this context manager. It is a content-addressed, cross-invocation
cache of the reference-domain foldseek database — rebuilding it on
every prediction would defeat its purpose. The production bundle
ships a prebuilt cache; ``managed_workdir`` leaves it in place.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@contextmanager
def managed_workdir(
    parent: str | Path | None,
    *,
    keep: bool = False,
    prefix: str = "enzyme_explorer_run_",
) -> Iterator[Path | None]:
    """Set up a scratch directory for the duration of a pipeline call.

    Parameters
    ----------
    parent:
        Parent directory under which to create the per-call scratch
        subdir. If ``None``, no swap happens — ``tempfile`` falls back
        to system tmp as usual.
    keep:
        If ``True``, leave the per-call scratch dir on disk after the
        context exits. Useful for debugging a failed run.
    prefix:
        Prefix for the per-call scratch subdir name (so concurrent
        invocations under the same parent don't collide).

    Yields
    ------
    The per-call scratch directory, or ``None`` when ``parent`` is
    ``None``.

    Notes
    -----
    ``tempfile.tempdir`` is restored to its previous value on exit, so
    this is safe to nest or to call from a long-lived process (FastAPI
    worker, screening cluster master).
    """
    if parent is None:
        yield None
        return

    parent_path = Path(parent).absolute()
    parent_path.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(parent_path)))

    prev_tempdir = tempfile.tempdir
    tempfile.tempdir = str(workdir)

    logger.info("managed_workdir: scratch root = %s", workdir)
    try:
        yield workdir
    finally:
        # Restore tempdir first so a failed rmtree doesn't strand the
        # process with a stale tempdir setting.
        tempfile.tempdir = prev_tempdir

        if keep:
            logger.info(
                "managed_workdir: --keep-intermediate given, leaving %s on disk",
                workdir,
            )
            return
        try:
            shutil.rmtree(workdir, ignore_errors=False)
            logger.info("managed_workdir: removed %s", workdir)
        except OSError as exc:
            logger.warning(
                "managed_workdir: failed to remove %s (%s); leaving for inspection",
                workdir, exc,
            )
