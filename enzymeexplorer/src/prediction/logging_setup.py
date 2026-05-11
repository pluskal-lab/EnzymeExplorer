"""Logging setup for the prediction CLIs.

Mirrors all log output to both stderr and a timestamped file under
``outputs/logs/`` so long-running predictions (especially the structure
pipeline, which spends most of its time inside slow domain-detection and
foldseek loops) can be inspected after the fact without having to keep a
terminal open.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from enzymeexplorer.src.utils.project_info import get_output_root


# Anchored to the repo's ``outputs/`` so the console scripts can run
# from any cwd. Override via ``--log-dir`` if the prod host's outputs
# directory should live elsewhere (e.g. /scratch).
DEFAULT_LOG_DIR = get_output_root() / "logs"


def configure_logging(
    *,
    name: str,
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
) -> Path:
    """Set up root-logger handlers (file + stderr) and return the log path.

    The file path is ``<log_dir>/<name>_<YYYYMMDD-HHMMSS>.log``. Existing
    handlers on the root logger are cleared first so re-invoking this is
    idempotent across reloads.
    """
    log_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{name}_{timestamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    root.addHandler(stream_handler)

    return log_path
