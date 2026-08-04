"""Populate ``data/enzyme_explorer_pdbs/`` for setup_dev.sh.

Strategy
--------
Iterate the unique IDs in ``data/EnzymeExplorer_Dataset.csv`` (column
``ID``). For each ID:

* ``marts_*`` — copy the matching PDB from ``data/martsDB_pdbs/``.
  (These come from the MARTS-DB bundle, so ``martsdb-pdbs`` must be
  extracted first.)
* Anything else — treat as a UniProt accession and download the
  AlphaFold-DB PDB. First try the direct v6 URL pattern
  (``https://alphafold.ebi.ac.uk/files/AF-<UID>-F1-model_v6.pdb``);
  on 404 fall back to the AFDB REST API
  (``https://alphafold.ebi.ac.uk/api/prediction/<UID>``) which returns
  a JSON record with the current ``pdbUrl`` (v4/v5/... whichever the
  entry has). Missing entries are logged and skipped.

The dir is idempotent — files already present with size > 0 are left
alone, so a partial re-run picks up from where it stopped.

Outputs (all under ``data/enzyme_explorer_pdbs/``):
    <UID>.pdb                     per-protein AlphaFold structure
    _download_manifest.csv        per-ID status
                                  (id, source, http_status, size_bytes)
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import shutil
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore

REPO = Path(__file__).resolve().parents[1]
DATASET_CSV = REPO / "data" / "EnzymeExplorer_Dataset.csv"
MARTS_PDB_DIR = REPO / "data" / "martsDB_pdbs"
OUT_DIR = REPO / "data" / "enzyme_explorer_pdbs"
MANIFEST_CSV = OUT_DIR / "_download_manifest.csv"

AFDB_URL_TMPL = "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb"
AFDB_API_TMPL = "https://alphafold.ebi.ac.uk/api/prediction/{uid}"


def _copy_marts(uid: str) -> tuple[str, str, int]:
    """Return (uid, status, size)."""
    dst = OUT_DIR / f"{uid}.pdb"
    if dst.exists() and dst.stat().st_size > 0:
        return uid, "skipped_existing", dst.stat().st_size
    src = MARTS_PDB_DIR / f"{uid}.pdb"
    if not src.exists():
        return uid, "missing_marts_source", 0
    shutil.copy2(src, dst)
    return uid, "copied_from_marts", dst.stat().st_size


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": "EnzymeExplorer-setup"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _fetch_afdb(uid: str, timeout: int = 60) -> tuple[str, str, int]:
    """Return (uid, status, size). status ∈ {ok_v6, ok_api, skipped_existing,
    missing, error:<msg>}."""
    dst = OUT_DIR / f"{uid}.pdb"
    if dst.exists() and dst.stat().st_size > 0:
        return uid, "skipped_existing", dst.stat().st_size

    # Try the v6 direct URL first.
    v6_url = AFDB_URL_TMPL.format(uid=uid)
    try:
        data = _http_get(v6_url, timeout=timeout)
        dst.write_bytes(data)
        return uid, "ok_v6", len(data)
    except HTTPError as exc:
        if exc.code != 404:
            return uid, f"error:{exc.code}", 0
    except URLError as exc:
        return uid, f"error:{str(exc)[:80]}", 0
    except Exception as exc:  # noqa: BLE001
        return uid, f"error:{str(exc)[:80]}", 0

    # v6 404 — ask the API for whatever version this entry has.
    api_url = AFDB_API_TMPL.format(uid=uid)
    try:
        payload = json.loads(_http_get(api_url, timeout=timeout))
    except HTTPError as exc:
        # 404 = no AFDB record; 400 = malformed accession (treat both as missing).
        if exc.code in (400, 404):
            return uid, "missing", 0
        return uid, f"error:api_{exc.code}", 0
    except Exception as exc:  # noqa: BLE001
        return uid, f"error:api_{str(exc)[:60]}", 0

    if not isinstance(payload, list) or not payload:
        return uid, "missing", 0
    pdb_url = payload[0].get("pdbUrl")
    if not pdb_url:
        return uid, "missing_no_pdb_url", 0
    try:
        data = _http_get(pdb_url, timeout=timeout)
        dst.write_bytes(data)
        return uid, "ok_api", len(data)
    except Exception as exc:  # noqa: BLE001
        return uid, f"error:pdb_{str(exc)[:60]}", 0


def _load_ids() -> list[str]:
    df = pd.read_csv(DATASET_CSV, usecols=["ID"])
    ids = df["ID"].dropna().astype(str).unique().tolist()
    return sorted(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=12,
                    help="Concurrent AF-DB downloads (default 12).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after N IDs. For quick smoke-testing.")
    ap.add_argument("--only-missing", action="store_true",
                    help="Skip IDs whose PDB already exists.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ids = _load_ids()
    if args.limit is not None:
        ids = ids[: args.limit]
    print(f"loaded {len(ids)} unique IDs from {DATASET_CSV}")

    marts_ids = [uid for uid in ids if uid.startswith("marts")]
    afdb_ids = [uid for uid in ids if not uid.startswith("marts")]
    print(f"  {len(marts_ids)} marts (copy from martsDB_pdbs)")
    print(f"  {len(afdb_ids)} UniProt (download from AFDB)")

    if args.only_missing:
        marts_ids = [
            u for u in marts_ids
            if not (OUT_DIR / f"{u}.pdb").exists()
        ]
        afdb_ids = [
            u for u in afdb_ids
            if not (OUT_DIR / f"{u}.pdb").exists()
        ]
        print(f"  --only-missing → {len(marts_ids)} marts + "
              f"{len(afdb_ids)} AFDB to fetch")

    rows: list[tuple[str, str, str, int]] = []

    # Stage 1 — marts copies (sequential; local FS, fast).
    t0 = time.time()
    ok_marts = 0
    for uid in marts_ids:
        u, status, size = _copy_marts(uid)
        rows.append((u, "marts", status, size))
        if status.startswith(("copied", "skipped")):
            ok_marts += 1
    print(f"[marts] {ok_marts}/{len(marts_ids)} in {time.time() - t0:.1f}s")

    # Stage 2 — AFDB downloads (concurrent).
    t0 = time.time()
    ok_afdb = 0
    missing = 0
    if afdb_ids:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, (uid, status, size) in enumerate(
                ex.map(_fetch_afdb, afdb_ids), 1
            ):
                rows.append((uid, "afdb", status, size))
                if status in ("ok_v6", "ok_api", "skipped_existing"):
                    ok_afdb += 1
                elif status.startswith("missing"):
                    missing += 1
                if i % 200 == 0:
                    print(f"[afdb] {i}/{len(afdb_ids)} "
                          f"({ok_afdb} ok, {missing} missing)")
    print(f"[afdb] {ok_afdb}/{len(afdb_ids)} in {time.time() - t0:.1f}s "
          f"({missing} genuinely missing at AFDB)")

    # Manifest.
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "source", "http_status", "size_bytes"])
        w.writerows(rows)
    print(f"wrote {MANIFEST_CSV} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
