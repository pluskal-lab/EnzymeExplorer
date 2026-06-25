"""Resolve the Kingdom of every dark-candidate UniProt accession.

Sister script to ``fetch_kingdom_for_uniprot.py`` but tailored for the
post-screening candidate set under ``data/dark_candidates/``:

* Input  : ``data/dark_candidates/all_candidates.csv`` (``id`` column).
* Output : ``data/dark_candidates/kingdom.csv`` (``id, kingdom, source``).

Two-stage resolver, since post-2020 UniProt regularly demerges or
deletes redundant entries and only keeps them as UniParc cross-refs:

1. **UniProtKB batch endpoint** (current entries; same as the legacy
   script). Returns the taxonomic lineage; we bucket it into the seven
   kingdom labels MartsDB uses (Bacteria, Archaea, Animals, Plants,
   Fungi, Protists, Viruses).
2. **UniParc fallback** for anything still ``Unknown``. We query
   ``rest.uniprot.org/uniparc/search?query=uniprotkb:<acc>`` to find the
   UniParc record that references the obsolete accession, take the
   highest-rank cross-reference's ``ncbiTaxonId``, and resolve that
   taxon ID's lineage via ``rest.uniprot.org/taxonomy/{taxId}``.
   ``lineage_to_kingdom`` from the legacy script does the bucketing.

The ``source`` column in the output marks which path resolved each row:
``uniprotkb``, ``uniparc``, or ``unresolved``.

Re-running is cheap; the script just overwrites the CSV.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd  # type: ignore

# Reuse the bucketing rule + UniProtKB batch endpoint from the
# pre-existing kingdom resolver so the seven-kingdom mapping stays in
# one place. The sibling script lives next to us under scripts/ — add
# this file's parent to sys.path so the import works regardless of cwd.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_kingdom_for_uniprot import (  # noqa: E402
    KINGDOM_ORDER,
    USER_AGENT,
    fetch_kingdoms as fetch_kingdoms_uniprotkb,
    lineage_to_kingdom,
)

logger = logging.getLogger(__name__)

UNIPARC_SEARCH_URL = "https://rest.uniprot.org/uniparc/search"
TAXONOMY_URL = "https://rest.uniprot.org/taxonomy/{tax_id}"
RETRY_LIMIT = 4
RETRY_BACKOFF_S = 5.0


def _http_get_json(url: str, *, timeout: int = 60) -> dict | None:
    """GET + JSON-decode with retry/backoff. Returns ``None`` on 404 (the
    expected outcome for accessions that genuinely don't exist anywhere)."""
    last_err: Exception | None = None
    for attempt in range(RETRY_LIMIT):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            last_err = exc
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_err = exc
        sleep_s = RETRY_BACKOFF_S * (2 ** attempt)
        logger.warning(
            "GET %s failed (attempt %d/%d): %s — sleeping %.1fs",
            url, attempt + 1, RETRY_LIMIT, last_err, sleep_s,
        )
        time.sleep(sleep_s)
    raise RuntimeError(f"GET {url} failed after {RETRY_LIMIT} attempts: {last_err}")


def _uniparc_taxonomy_for(accession: str) -> int | None:
    """Return the NCBI taxonomy ID associated with ``accession``'s UniParc
    record, or ``None`` if no UniParc entry references the accession.

    UniParc records cross-reference any database that ever held the
    sequence; for an obsolete UniProt accession we find the UniParc
    UPI by querying ``uniprotkb:<acc>`` and then pick the first cross
    reference that carries an ``organism.taxonId``. UniParc may list
    multiple source organisms (the same sequence in different species);
    we take the first one with a taxon ID since for an unstudied TPS
    candidate the kingdom is unlikely to differ across sources.
    """
    # Plain accession query rather than ``uniprotkb:<acc>``. The latter
    # only matches *active* UniProtKB cross-refs, but the accessions we
    # need to resolve here are mostly inactive (DELETED / merged); the
    # plain query hits the UniParc record regardless of whether the
    # accession is still active.
    params = urllib.parse.urlencode(
        {"query": accession, "format": "json", "size": "1"}
    )
    payload = _http_get_json(f"{UNIPARC_SEARCH_URL}?{params}")
    if not payload:
        return None
    results = payload.get("results") or []
    if not results:
        return None
    record = results[0]
    # UniParc search returns lightweight records: ``commonTaxons`` is a
    # list of ``{topLevel, commonTaxon, commonTaxonId}`` dicts.
    # ``uniParcCrossReferences`` (with per-source organism IDs) is only
    # populated by the dedicated ``/uniparc/{upi}`` endpoint, so we use
    # ``commonTaxonId`` here — it's the lowest common ancestor across
    # every source organism, which is the right granularity for our
    # kingdom bucketing anyway.
    for xref in record.get("uniParcCrossReferences", []) or []:
        org = xref.get("organism") or {}
        tax_id = org.get("taxonId")
        if tax_id:
            return int(tax_id)
    for common in record.get("commonTaxons", []) or []:
        if not isinstance(common, dict):
            continue
        tax_id = common.get("commonTaxonId") or common.get("taxonId")
        if tax_id:
            return int(tax_id)
    return None


def _taxonomy_lineage(tax_id: int) -> list[str]:
    """Fetch the lineage (root → leaf) for a taxonomy ID and return the
    list of scientific names, ready to feed into ``lineage_to_kingdom``.

    UniProt's taxonomy resource returns ``lineage`` as a list of
    ``{taxonId, scientificName, rank, ...}`` ordered most-specific →
    most-general; we reverse it so the order matches what
    ``lineage_to_kingdom`` (and the UniProtKB ``lineages`` field) expect.
    """
    payload = _http_get_json(TAXONOMY_URL.format(tax_id=tax_id))
    if not payload:
        return []
    raw = payload.get("lineage") or []
    # The /taxonomy/{id} endpoint already returns lineage root-first
    # (e.g. ``['cellular organisms', 'Eukaryota', 'Opisthokonta', ...]``),
    # matching the order ``lineage_to_kingdom`` expects.
    return [e.get("scientificName") for e in raw if e.get("scientificName")]


def fetch_kingdoms_with_uniparc_fallback(
    accessions: list[str],
) -> dict[str, tuple[str, str]]:
    """Return ``{accession: (kingdom, source)}`` for every input accession.

    ``source`` is one of ``"uniprotkb"``, ``"uniparc"``, or
    ``"unresolved"`` and records which lookup produced the kingdom (or
    failed to). Anything we can't resolve gets ``("Unknown", "unresolved")``.
    """
    primary = fetch_kingdoms_uniprotkb(accessions)
    out: dict[str, tuple[str, str]] = {}
    fallbacks: list[str] = []
    for acc, kingdom in primary.items():
        if kingdom != "Unknown":
            out[acc] = (kingdom, "uniprotkb")
        else:
            fallbacks.append(acc)

    if fallbacks:
        logger.info(
            "UniProtKB resolved %d/%d; trying UniParc fallback for %d",
            len(out), len(accessions), len(fallbacks),
        )
    for i, acc in enumerate(fallbacks, 1):
        try:
            tax_id = _uniparc_taxonomy_for(acc)
        except Exception:
            logger.exception("UniParc lookup failed for %s", acc)
            tax_id = None
        if tax_id is None:
            out[acc] = ("Unknown", "unresolved")
            continue
        try:
            lineage = _taxonomy_lineage(tax_id)
        except Exception:
            logger.exception("Taxonomy lookup failed for %s (taxId=%s)", acc, tax_id)
            lineage = []
        kingdom = lineage_to_kingdom(lineage)
        out[acc] = (kingdom, "uniparc" if kingdom != "Unknown" else "unresolved")
        if i % 25 == 0 or i == len(fallbacks):
            logger.info("UniParc fallback progress: %d/%d", i, len(fallbacks))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/dark_candidates/all_candidates.csv"),
        help="CSV with an 'id' column of UniProt accessions.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/dark_candidates/kingdom.csv"),
        help="Output CSV with columns (id, kingdom, source).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    df = pd.read_csv(args.candidates, usecols=["id"]).drop_duplicates("id")
    accessions = df["id"].astype(str).tolist()
    logger.info("Candidates: %d", len(accessions))

    # Resume support: if an earlier run already produced a kingdom.csv,
    # carry over every row whose ``source`` is *not* ``unresolved`` and
    # only retry the unresolved ones. Saves the bulk UniProtKB batch
    # call (already cached) plus any UniParc lookups that already
    # landed a kingdom.
    carryover: dict[str, tuple[str, str]] = {}
    if args.output.exists():
        prev = pd.read_csv(args.output)
        carryover = {
            r.id: (r.kingdom, r.source)
            for r in prev.itertuples(index=False)
            if r.source != "unresolved"
        }
        accessions = [a for a in accessions if a not in carryover]
        logger.info(
            "Resuming: %d already-resolved entries carried over, %d to (re)try",
            len(carryover), len(accessions),
        )

    resolved = fetch_kingdoms_with_uniparc_fallback(accessions) if accessions else {}
    resolved.update(carryover)
    rows = [
        {"id": acc, "kingdom": kingdom, "source": source}
        for acc, (kingdom, source) in resolved.items()
    ]
    out_df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Wrote %s", args.output)

    counts = out_df["kingdom"].value_counts().reindex(
        KINGDOM_ORDER + ["Unknown"], fill_value=0,
    )
    logger.info("Kingdom distribution:\n%s", counts.to_string())
    src_counts = out_df["source"].value_counts()
    logger.info("Resolver source counts:\n%s", src_counts.to_string())


if __name__ == "__main__":
    main()
