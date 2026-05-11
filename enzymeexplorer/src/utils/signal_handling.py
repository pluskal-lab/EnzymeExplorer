"""Graceful-shutdown plumbing for prediction & screening CLI entry points.

Two pieces wrapped in one context manager:

1. **SIGTERM → KeyboardInterrupt translation.** Python's default
   SIGTERM handler terminates the interpreter immediately, *skipping*
   every ``try/finally`` and context-manager ``__exit__``. That means
   ``managed_workdir`` scratch dirs leak, ``pool_session`` workers
   become orphans, and partial output files don't get rolled back.
   ``scancel`` and most cluster preemption signals are SIGTERM, so on
   SLURM this is the common case. Installing a handler that ``raise``s
   ``KeyboardInterrupt`` flips this to the same controlled-shutdown
   path as Ctrl-C: the exception unwinds through every
   ``managed_workdir`` / ``pool_session`` / ``finally`` block before
   the process exits.

2. **Top-level exception logging.** Anything that escapes ``main()`` is
   logged with full traceback before the process dies, so post-mortem
   on a failed SLURM task doesn't require digging through stderr.

Usage::

    def main() -> None:
        with graceful_shutdown(name="predict_sequences_only"):
            ... real CLI body ...

The original signal handlers are restored on exit so the helper is
safe inside long-lived servers or test runners.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)


class _GracefulExit(KeyboardInterrupt):
    """Raised by the SIGTERM handler so callers can distinguish a
    cluster-issued shutdown from a literal Ctrl-C if they want to.
    Inherits from :class:`KeyboardInterrupt` so it propagates through
    ``except Exception`` blocks unchanged."""


def _make_handler(signal_name: str):
    def _handler(signum, frame):  # noqa: ARG001 — required by signal API
        logger.warning(
            "Received %s — initiating graceful shutdown "
            "(scratch dirs, worker pools and subprocesses will be torn down)",
            signal_name,
        )
        raise _GracefulExit(f"received {signal_name}")
    return _handler


@contextmanager
def graceful_shutdown(*, name: str) -> Iterator[None]:
    """Install signal handlers and a last-resort logger for the
    duration of the ``with`` block.

    ``name`` is interpolated into the final exception log line so it's
    obvious which pipeline died when several share a single log file.
    """
    prev_term = signal.signal(signal.SIGTERM, _make_handler("SIGTERM"))
    # SIGHUP also fires when an SSH session closes; treat it the same.
    # Not available on Windows (we're Unix-only anyway).
    prev_hup = None
    if hasattr(signal, "SIGHUP"):
        prev_hup = signal.signal(signal.SIGHUP, _make_handler("SIGHUP"))
    try:
        yield
    except _GracefulExit:
        # SIGTERM/SIGHUP path: cleanup already ran in inner finally
        # blocks; emit a single line and exit with a non-zero code so
        # SLURM marks the task as cancelled/failed (so dependent tasks
        # with afterok or aftercorr correctly skip).
        logger.warning("%s: graceful shutdown complete", name)
        sys.exit(143)  # 128 + SIGTERM
    except KeyboardInterrupt:
        logger.warning("%s: interrupted by user (SIGINT) — cleanup complete", name)
        sys.exit(130)  # 128 + SIGINT
    except Exception:
        # Cleanup ran via inner finally blocks; just log the cause.
        logger.exception(
            "%s: unhandled exception — process will exit non-zero "
            "after scratch/pool teardown", name,
        )
        sys.exit(1)
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        if prev_hup is not None:
            signal.signal(signal.SIGHUP, prev_hup)


def child_exits_with_parent() -> None:
    """Best-effort: ask the OS to send SIGTERM to this process when its
    parent dies. Only effective on Linux (uses ``prctl(PR_SET_PDEATHSIG)``
    via ``ctypes``). Silently no-ops elsewhere or if ``ctypes`` isn't
    available.

    Useful in worker subprocesses spawned by the screening pipeline so
    a hard crash of the manager doesn't leave them hanging.
    """
    if sys.platform != "linux":
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # PR_SET_PDEATHSIG = 1; signal.SIGTERM resolves to int.
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)
    except Exception:
        # Non-fatal: the worker just won't auto-die if its parent dies.
        pass


def install_minimal_term_handler() -> None:
    """Standalone variant of :func:`graceful_shutdown` for code that
    can't easily wrap its entire body in a ``with`` block (e.g. test
    fixtures, REPL sessions). Idempotent — re-installing is a no-op."""
    if getattr(install_minimal_term_handler, "_installed", False):
        return
    signal.signal(signal.SIGTERM, _make_handler("SIGTERM"))
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _make_handler("SIGHUP"))
    install_minimal_term_handler._installed = True  # type: ignore[attr-defined]


# ``os`` is intentionally imported for future use (env-gated overrides);
# keep it referenced so linters don't whine.
_ = os
