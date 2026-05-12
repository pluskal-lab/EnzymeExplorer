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

For screens of millions of sequences against a public service, two
robustness properties matter:

* **404 is terminal** — if AF-DB doesn't have an ID, no number of
  retries will help, so we mark it missing immediately and move on.
* **5xx / network errors are retried** with exponential backoff so a
  transient EBI hiccup doesn't permanently misclassify thousands of
  IDs as missing. The retry budget is bounded per-ID so a sustained
  outage still fails fast rather than livelocking.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm  # type: ignore

logger = logging.getLogger(__name__)

# AF-DB serves monomer model files at predictable URLs for most
# entries — ``AF-<uid>-F1-model_v<n>.pdb`` where ``<n>`` is the latest
# version. We try the current version first (fast path); on 404 we
# fall back to the metadata API to discover the canonical pdbUrl
# (handles two real cases we hit on candidates.fasta):
#   1. AF-DB has bumped to a newer version than _CURRENT_VERSION and
#      stopped serving the older one for some entries.
#   2. The entry exists under a non-standard entryId (e.g.
#      ``AF-0000000000001806`` for A0A2P0VN22) where the
#      ``AF-<uid>-F1`` URL pattern doesn't match at all.
# The hybrid keeps the common case at one HTTP request per ID for
# screening throughput while remaining correct for edge cases.
_CURRENT_VERSION = 6
AF_DB_FAST_PATH_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v"
    f"{_CURRENT_VERSION}.pdb"
)
AF_DB_METADATA_TEMPLATE = (
    "https://alphafold.ebi.ac.uk/api/prediction/{uid}"
)

# Retry policy for transient (non-404) failures. 4 attempts spaced by
# exponential backoff with jitter — covers most EBI transient hiccups
# without livelocking on a sustained outage. The total worst-case wait
# for one ID is ~1 + 2 + 4 + jitter ≈ 8 s.
_MAX_ATTEMPTS = 4
_BASE_BACKOFF_S = 1.0


def _backoff(attempt: int) -> None:
    """Exponential backoff with 25 % jitter, used between retry attempts."""
    sleep_s = _BASE_BACKOFF_S * (2 ** (attempt - 1))
    sleep_s *= 0.75 + 0.5 * random.random()
    time.sleep(sleep_s)


_GZIP_MAGIC = b"\x1f\x8b"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` via a temp-then-rename hop.

    POSIX ``rename`` is atomic on the same filesystem, so concurrent
    readers either see the previous file or the new one — never a
    half-written file. This matters for resumability: a worker
    SIGKILLed mid-write would otherwise leave a truncated PDB on
    disk, which a rerun would mistake for a successful download
    (``download_one`` only checks ``stat().st_size > 0``).
    """
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(target)


def _maybe_gunzip(body: bytes, content_encoding: str | None) -> bytes:
    """Decompress ``body`` if it's gzipped.

    AF-DB stores every monomer PDB gzipped in its object store
    (``x-goog-stored-content-encoding: gzip``). For most entries the
    CDN decompresses on the fly and the response is plaintext PDB; but
    for some entries (e.g. the synthetic ``AF-0000000000001806`` family
    that some UniProt IDs resolve to via the metadata API) the CDN
    passes the gzip through and sets ``Content-Encoding: gzip``.

    ``urllib`` does NOT auto-decompress on ``Content-Encoding: gzip``,
    so we have to do it ourselves. We honour the header *and* sniff
    the gzip magic bytes — bare-bytes detection is the only reliable
    signal if any intermediate proxy strips the encoding header but
    leaves the body as-is.
    """
    is_gzip = (
        (content_encoding or "").lower() == "gzip"
        or body[:2] == _GZIP_MAGIC
    )
    if not is_gzip:
        return body
    return gzip.decompress(body)


def _http_get_bytes(url: str, timeout: float) -> tuple[bytes | None, bool]:
    """Fetch ``url`` and return ``(body, is_definitive_miss)``.

    * ``(bytes, False)`` — 200 OK; body is decompressed if the server
      flagged it as gzip-encoded (or if the bytes are gzip-magic).
    * ``(None, True)`` — 404 from the server. Definitive miss; caller
      should not retry the same URL.
    * raises — transient failure (5xx, network reset, timeout). Caller
      should back off and retry, or treat exhausted retries as a miss.
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            encoding = resp.headers.get("Content-Encoding")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, True
        raise
    return _maybe_gunzip(raw, encoding), False


def _resolve_canonical_pdb_url(uid: str, timeout: float) -> str | None:
    """Ask AF-DB's metadata API for the current ``pdbUrl`` of ``uid``.

    Returns the URL string, or ``None`` if AF-DB has no entry for that
    UniProt ID (HTTP 4xx, empty JSON list, or empty JSON object — AF-DB
    uses all three depending on the input shape). Network-level
    failures propagate so the caller's retry loop can handle them.
    """
    url = AF_DB_METADATA_TEMPLATE.format(uid=uid)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            return None  # invalid ID or no entry
        raise
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list) and parsed:
        return parsed[0].get("pdbUrl")
    return None


def download_one(
    uid: str,
    out_dir: Path,
    *,
    timeout: float = 30.0,
    overwrite: bool = False,
) -> bool:
    """Download a single AF-DB monomer PDB.

    Tries the predictable ``AF-<uid>-F1-model_v6.pdb`` URL first; on
    404 falls back to the AF-DB metadata API to discover the canonical
    ``pdbUrl`` (and then fetches that). Retries transient failures
    with exponential backoff.

    Returns ``True`` if the file is present in ``out_dir`` after the
    call, ``False`` for any kind of miss (AF-DB has no entry, the
    canonical URL also 404s, every retry attempt failed). Never
    raises — callers treat ``False`` as "missing for any reason".
    """
    target = out_dir / f"{uid}.pdb"
    if target.exists() and target.stat().st_size > 0 and not overwrite:
        return True

    last_exc: BaseException | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        # --- fast path: predictable v6 URL ---
        try:
            data, definitive_miss = _http_get_bytes(
                AF_DB_FAST_PATH_TEMPLATE.format(uid=uid), timeout=timeout,
            )
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                _backoff(attempt)
            continue

        if data is not None:
            _atomic_write_bytes(target, data)
            return True

        # Fast path returned a clean 404 — fall through to metadata API.
        try:
            canonical_url = _resolve_canonical_pdb_url(uid, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                _backoff(attempt)
            continue

        if canonical_url is None:
            # AF-DB has no entry for this UniProt ID. Definitive miss.
            return False

        # --- canonical path: URL discovered via metadata API ---
        try:
            data, definitive_miss = _http_get_bytes(
                canonical_url, timeout=timeout,
            )
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < _MAX_ATTEMPTS:
                _backoff(attempt)
            continue

        if data is not None:
            _atomic_write_bytes(target, data)
            return True

        if definitive_miss:
            # API listed this URL but it now 404s — race or stale
            # cache. Not retryable.
            logger.warning(
                "AF-DB API for %s returned pdbUrl %s but the URL 404s",
                uid, canonical_url,
            )
            return False

    logger.warning(
        "AF-DB download for %s failed after %d attempts: %s",
        uid, _MAX_ATTEMPTS, last_exc,
    )
    return False


def download_many(
    uids: Iterable[str],
    out_dir: Path,
    *,
    n_workers: int = 16,
    timeout: float = 30.0,
    overwrite: bool = False,
    progress: bool = True,
    progress_desc: str = "AF-DB download",
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

        # tqdm.auto picks the right backend automatically:
        #   * attached to a terminal → live ``\r``-overwriting bar
        #   * piped to a SLURM log file → simpler line-mode output
        # ``mininterval=5s`` throttles the line-mode updates to one
        # progress line every 5 seconds so the per-batch SLURM log
        # doesn't fill up with thousands of redundant lines on big
        # batches. Stats include downloaded/missing counts so a
        # human reading the log can spot a stuck batch immediately.
        pbar = tqdm(
            total=len(futures),
            desc=progress_desc,
            disable=not progress,
            mininterval=5.0,
            unit="pdb",
            file=sys.stderr,
        )

        for fut in as_completed(futures):
            uid = futures[fut]
            # ``download_one`` already swallows network errors and
            # returns False; this except is the belt-and-braces line
            # for anything truly unexpected (e.g. disk full on
            # ``target.write_bytes``).
            try:
                ok = fut.result()
            except Exception as exc:
                logger.warning("AF-DB download failed for %s: %s", uid, exc)
                missing.add(uid)
                pbar.set_postfix_str(
                    f"{len(downloaded)} ok / {len(missing)} miss", refresh=False,
                )
                pbar.update(1)
                continue
            (downloaded if ok else missing).add(uid)
            pbar.set_postfix_str(
                f"{len(downloaded)} ok / {len(missing)} miss", refresh=False,
            )
            pbar.update(1)
        pbar.close()

    logger.info(
        "AF-DB batch: %d downloaded, %d missing (of %d requested)",
        len(downloaded), len(missing), len(uids),
    )
    return downloaded, missing
