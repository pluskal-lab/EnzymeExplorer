"""AlphaFold DB structure-download helper for screening.

Pulls AF-DB monomer model v4 PDBs for a list of UniProt IDs and reports
which IDs are not available. Designed for both:

* a standalone CPU-only SLURM job that pre-downloads structures into a
  staging directory before a GPU prediction job
  (:mod:`tps_download_af_structures`), and
* an inline fallback inside the GPU worker when the user runs the
  screening with ``--classifier plm_domains`` but no
  ``--structures-dir`` — convenient for one-off jobs at the cost of
  burning GPU time on HTTP.

Concurrency is via ``ThreadPoolExecutor`` — AF-DB ingress is the
bottleneck, not local CPU, so 16–32 worker threads suffices.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# AF-DB monomer v4 path. v4 is the latest stable as of 2024; if the
# upstream schema changes, bump this URL.
AF_DB_URL_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb"
)


def download_one(
    uid: str,
    out_dir: Path,
    *,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> bool:
    """Download a single AF-DB monomer PDB.

    Returns ``True`` if the file is present in ``out_dir`` after the
    call, ``False`` if AF-DB returns 404 for that UniProt ID. Network
    errors other than 404 propagate so the caller can retry.
    """
    target = out_dir / f"{uid}.pdb"
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return True

    url = AF_DB_URL_TEMPLATE.format(uid=uid)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    target.write_bytes(data)
    return True


def download_many(
    uids: Iterable[str],
    out_dir: Path,
    *,
    n_workers: int = 16,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> tuple[set[str], set[str]]:
    """Concurrent batch download.

    Returns ``(downloaded, missing)`` — ``downloaded`` is the subset of
    ``uids`` whose PDB ended up in ``out_dir``; ``missing`` is everything
    else (either 404 from AF-DB or a transient network failure that
    didn't recover). Both sets are subsets of the input ``uids``.
    """
    uids = list(dict.fromkeys(uids))  # dedupe, preserve order
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: set[str] = set()
    missing: set[str] = set()

    if not uids:
        return downloaded, missing

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(download_one, uid, out_dir, timeout=timeout, overwrite=overwrite): uid
            for uid in uids
        }
        for fut in as_completed(futures):
            uid = futures[fut]
            try:
                ok = fut.result()
            except Exception as exc:  # network error, ssl issue, etc.
                logger.warning("AF-DB download failed for %s: %s", uid, exc)
                missing.add(uid)
                continue
            (downloaded if ok else missing).add(uid)

    logger.info(
        "AF-DB batch: %d downloaded, %d missing (of %d requested)",
        len(downloaded), len(missing), len(uids),
    )
    return downloaded, missing
