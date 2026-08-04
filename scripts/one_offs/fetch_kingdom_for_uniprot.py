"""Resolve the Kingdom of every Type=Unknown sequence in the dataset using
the UniProt REST API.

The dataset's negative pool (Type=='Unknown') has its Kingdom column also
set to 'Unknown'. The IDs are valid UniProt accessions; we hit
``rest.uniprot.org/uniprotkb/accessions`` in batches, parse the
taxonomic lineage, and map it to one of the seven Kingdom buckets used
elsewhere in the project: Bacteria, Archaea, Animals, Plants, Fungi,
Protists, Viruses. Anything we can't resolve falls back to ``Unknown``.

Run::

    python scripts/fetch_kingdom_for_uniprot.py

Writes ``data/uniprot_kingdom_cache.json`` (``{accession: kingdom}``) and
``data/uniprot_kingdom_distribution.png`` (bar chart). Designed as a
one-shot enrichment — re-running just refreshes the cache.
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

import matplotlib.pyplot as plt  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore

logger = logging.getLogger(__name__)

API_URL = "https://rest.uniprot.org/uniprotkb/accessions"
USER_AGENT = "EnzymeExplorer/1.0 (safa_mert.akmese@uochb.cas.cz)"
BATCH_SIZE = 500
RETRY_LIMIT = 4
RETRY_BACKOFF_S = 5.0

KINGDOM_ORDER = [
    "Bacteria", "Archaea", "Animals", "Plants", "Fungi", "Protists", "Viruses",
]


def lineage_to_kingdom(lineage: list[str]) -> str:
    """Map a UniProt taxonomic lineage list to one of seven kingdoms.

    The lineage is ordered from most general to most specific. UniProt
    prefixes most lineages with ``"cellular organisms"``, so we scan for
    the first domain-level clade and then look downstream for the
    eukaryotic sub-kingdom. Anything outside Animals/Plants/Fungi but
    inside Eukaryota maps to Protists (matches MartsDB convention).
    """
    if not lineage:
        return "Unknown"
    domain: str | None = None
    for clade in lineage:
        if clade in {"Bacteria", "Archaea", "Viruses", "Eukaryota"}:
            domain = clade
            break
    if domain in {"Bacteria", "Archaea", "Viruses"}:
        return {"Bacteria": "Bacteria", "Archaea": "Archaea", "Viruses": "Viruses"}[domain]
    if domain == "Eukaryota":
        for clade in lineage:
            if clade == "Metazoa":
                return "Animals"
            if clade == "Viridiplantae":
                return "Plants"
            if clade == "Fungi":
                return "Fungi"
        return "Protists"
    return "Unknown"


def _fetch_batch(accessions: list[str]) -> list[dict]:
    """Hit the UniProt batch endpoint with retry+backoff. Returns the list
    of result dicts (one per resolved accession)."""
    params = urllib.parse.urlencode(
        {
            "accessions": ",".join(accessions),
            "fields": "accession,lineage",
            "format": "json",
            "size": str(len(accessions)),
        }
    )
    url = f"{API_URL}?{params}"
    last_err: Exception | None = None
    for attempt in range(RETRY_LIMIT):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read())
            return payload.get("results", [])
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_err = exc
            sleep_s = RETRY_BACKOFF_S * (2**attempt)
            logger.warning(
                "Batch fetch failed (attempt %d/%d): %s — sleeping %.1fs",
                attempt + 1, RETRY_LIMIT, exc, sleep_s,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"UniProt batch fetch failed after {RETRY_LIMIT} attempts: {last_err}")


def fetch_kingdoms(accessions: list[str]) -> dict[str, str]:
    """Return ``{accession: kingdom}`` for every input accession. Missing
    or unresolved accessions get ``Unknown``."""
    out: dict[str, str] = {a: "Unknown" for a in accessions}
    n = len(accessions)
    for start in range(0, n, BATCH_SIZE):
        chunk = accessions[start:start + BATCH_SIZE]
        results = _fetch_batch(chunk)
        for r in results:
            acc = r.get("primaryAccession")
            # With ``fields=accession,lineage`` the response carries
            # ``lineages`` (list of {scientificName, rank, ...}), not the
            # simple string list under ``organism.lineage``.
            entries = r.get("lineages") or (r.get("organism") or {}).get("lineage") or []
            if entries and isinstance(entries[0], dict):
                lineage = [e.get("scientificName") for e in entries if e.get("scientificName")]
            else:
                lineage = list(entries)
            if acc:
                out[acc] = lineage_to_kingdom(lineage)
        logger.info(
            "Resolved %d/%d", min(start + BATCH_SIZE, n), n,
        )
    return out


def plot_distribution(kingdom_map: dict[str, str], out_path: Path) -> None:
    """Bar chart of how many distractors fall into each kingdom."""
    s = pd.Series(list(kingdom_map.values()))
    counts = s.value_counts().reindex(KINGDOM_ORDER + ["Unknown"]).fillna(0).astype(int)
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = sns.color_palette("colorblind", n_colors=len(counts))
    bars = ax.bar(counts.index, counts.values, color=palette, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# distractors")
    ax.set_xlabel("")
    ax.set_title("Kingdom distribution of distractor (Type=Unknown) sequences")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="data/EnzymeExplorer_Dataset.csv",
        help="Path to EnzymeExplorer_Dataset.csv",
    )
    parser.add_argument(
        "--cache", default="data/uniprot_kingdom_cache.json",
        help="Output JSON cache (accession -> kingdom)",
    )
    parser.add_argument(
        "--plot", default="data/uniprot_kingdom_distribution.png",
        help="Output distribution PNG",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(args.dataset, usecols=["ID", "Type"]).drop_duplicates("ID")
    neg = df[df["Type"] == "Unknown"]
    accessions = neg["ID"].astype(str).tolist()
    logger.info("Negatives in dataset: %d", len(accessions))

    kingdom_map = fetch_kingdoms(accessions)
    Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
    with open(args.cache, "w", encoding="utf-8") as fh:
        json.dump(kingdom_map, fh, indent=2, sort_keys=True)
    logger.info("Wrote cache to %s", args.cache)

    resolved = sum(1 for v in kingdom_map.values() if v != "Unknown")
    missing = len(kingdom_map) - resolved
    logger.info(
        "Resolved %d/%d (%.1f%%); missing/unresolved %d",
        resolved, len(kingdom_map), resolved / len(kingdom_map) * 100, missing,
    )

    plot_distribution(kingdom_map, Path(args.plot))
    logger.info("Wrote distribution plot to %s", args.plot)


if __name__ == "__main__":
    main()
