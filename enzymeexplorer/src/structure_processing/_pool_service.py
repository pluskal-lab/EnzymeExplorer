"""Centralized multiprocessing pool service for structure processing.

One spawn-based pool serves every parallel workload triggered inside a
single top-level run (domain detection, secondary-structure extraction,
domain storage, USalign all-vs-all). Spawn is mandatory: PyMOL is not
fork-safe once the parent has loaded a single PDB, and several entry
points (`save_file_to_all_residues`, `get_all_residues_per_file`) load
PDBs in the parent before any worker is needed. Subprocess-only
workloads (USalign batch / USalign all-vs-all) do not require spawn but
happily ride on top of one.

Usage at top-level entry points::

    with pool_session(n_jobs=20, working_dir=input_dir):
        ...all parallel work runs against the active service...

Usage inside library helpers::

    svc = require_active_service()
    results = svc.map(_my_worker_fn, items)

No helper function in this package is allowed to construct its own
``multiprocessing.Pool`` — every parallel call site must go through the
service. Top-level entry points are the only place a pool is created,
and they do so by opening a session (which is the service itself).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


_ACTIVE: "Optional[PoolService]" = None


def _spawn_worker_init(working_dir: str) -> None:
    """Spawn-pool worker initializer.

    Spawn workers boot a fresh interpreter; cwd is inherited from the
    parent at fork-then-exec time but several call sites in this package
    use relative ``f"{stem}.pdb"`` paths, so we pin cwd explicitly. PyMOL
    is initialized lazily on first ``cmd.*`` use in the worker — in a
    clean interpreter, never inheriting the parent's PyMOL state.
    """
    if working_dir:
        os.chdir(working_dir)


class PoolService:
    """A live spawn-based pool with a fixed ``n_jobs`` and ``working_dir``.

    Constructed only via :func:`pool_session`. Exposes a small subset of
    the ``multiprocessing.Pool`` API (``map``, ``starmap``,
    ``imap_unordered``) so call sites have no direct ``multiprocessing``
    dependency.
    """

    def __init__(
        self,
        n_jobs: int,
        working_dir: str,
        *,
        maxtasksperchild: int = 500,
    ) -> None:
        self._n_jobs = max(1, int(n_jobs))
        self._working_dir = working_dir
        ctx = mp.get_context("spawn")
        logger.info(
            "PoolService: spawning %d workers (cwd=%s, maxtasksperchild=%d)",
            self._n_jobs,
            self._working_dir,
            maxtasksperchild,
        )
        self._pool = ctx.Pool(
            processes=self._n_jobs,
            maxtasksperchild=maxtasksperchild,
            initializer=_spawn_worker_init,
            initargs=(self._working_dir,),
        )

    @property
    def n_jobs(self) -> int:
        return self._n_jobs

    @property
    def working_dir(self) -> str:
        return self._working_dir

    def map(
        self,
        fn: Callable[..., Any],
        iterable: Iterable[Any],
        chunksize: Optional[int] = None,
    ) -> list:
        if chunksize is None:
            return self._pool.map(fn, iterable)
        return self._pool.map(fn, iterable, chunksize=chunksize)

    def starmap(
        self,
        fn: Callable[..., Any],
        iterable: Iterable[Iterable[Any]],
        chunksize: Optional[int] = None,
    ) -> list:
        if chunksize is None:
            return self._pool.starmap(fn, iterable)
        return self._pool.starmap(fn, iterable, chunksize=chunksize)

    def imap_unordered(
        self,
        fn: Callable[..., Any],
        iterable: Iterable[Any],
        chunksize: int = 1,
    ) -> Iterator[Any]:
        return self._pool.imap_unordered(fn, iterable, chunksize=chunksize)

    def shutdown(self) -> None:
        """Graceful shutdown: refuse new tasks, then wait for in-flight
        ones to drain. Use when the caller exited the context normally."""
        self._pool.close()
        self._pool.join()

    def terminate(self) -> None:
        """Hard shutdown: SIGTERM every worker immediately, then reap.
        Use when the caller's context exited via an exception (Ctrl-C,
        SIGTERM, unhandled error) — in-flight tasks would otherwise
        keep the parent process alive long after the user expected it
        to die."""
        try:
            self._pool.terminate()
        finally:
            self._pool.join()


@contextmanager
def pool_session(
    *,
    n_jobs: int,
    working_dir: str | os.PathLike,
    maxtasksperchild: int = 500,
) -> Iterator[PoolService]:
    """Open a pool session for the duration of the ``with`` block.

    Re-entrant: a nested ``pool_session`` reuses the outer service and
    does NOT spawn a second pool — this lets composite entry points
    (e.g. a wrapper that calls both ``run_domain_detection`` and
    ``run_all_vs_all``) share one warm pool. The ``n_jobs`` and
    ``working_dir`` of nested sessions are ignored when reusing.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        yield _ACTIVE
        return

    svc = PoolService(
        n_jobs=n_jobs,
        working_dir=str(working_dir),
        maxtasksperchild=maxtasksperchild,
    )
    _ACTIVE = svc
    try:
        yield svc
    except BaseException:
        # Error / KeyboardInterrupt / SIGTERM path: terminate workers
        # immediately instead of waiting for in-flight tasks to drain.
        # Without this, ``svc.shutdown()`` would block the unwind for
        # the duration of the slowest worker (e.g. a multi-minute
        # USalign batch), defeating graceful cancellation.
        _ACTIVE = None
        svc.terminate()
        raise
    else:
        _ACTIVE = None
        svc.shutdown()


def get_active_service() -> Optional[PoolService]:
    """Return the active service, or ``None`` if no session is open."""
    return _ACTIVE


def require_active_service() -> PoolService:
    """Return the active service or raise if no session is open.

    Library helpers should call this rather than constructing pools
    themselves. If you see this error, the caller forgot to wrap its
    work in ``with pool_session(...): ...``.
    """
    if _ACTIVE is None:
        raise RuntimeError(
            "No active pool session. Top-level entry points must open one "
            "via `with pool_session(n_jobs=..., working_dir=...): ...` "
            "before invoking parallel helpers."
        )
    return _ACTIVE
