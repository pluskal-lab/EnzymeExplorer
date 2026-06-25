"""Build the combined FASTA + metadata for the dark-candidates phylogeny.

Pools the post-screening candidate sequences with every unique martsDB
entry into one FASTA that downstream tooling (MAFFT, IQ-TREE) consumes
as a single input. Also emits a sidecar metadata CSV that the
distant-clade extractor uses to tell candidates apart from martsDB
references and to colour the tree by kingdom.

Inputs
------
* ``data/dark_candidates/all_candidates.csv`` — output of the screening
  filter (``id``, ``sequence``, ``*_p_calibrated`` cols). Only ``id``
  and ``sequence`` are used here.
* ``data/dark_candidates/kingdom.csv`` — produced by
  ``fetch_kingdom_for_candidates.py``; ``(id, kingdom, source)``.
* ``data/martsDB_reactions_2026_02_22_preprocessed.csv`` — reaction-level
  martsDB. We deduplicate on ``Uniprot_ID`` (one sequence per protein).

Outputs (all under ``data/dark_candidates/``)
---------------------------------------------
* ``combined.fasta`` — one record per unique sequence. FASTA headers are
  the bare ``id`` (UniProt accession) so they survive MAFFT and IQ-TREE
  without escaping.
* ``metadata.csv`` — ``id, source, kingdom, kingdom_detailed, seq_len``
  where ``source ∈ {candidate, martsDB}``. Candidates inherit
  ``kingdom`` from ``kingdom.csv`` (or ``Unknown`` if missing) and have
  no ``kingdom_detailed``; martsDB rows take both columns straight from
  the source table.

Collision rule: if a candidate ID is also present in martsDB (rare but
possible — the screening pool is UniProt-wide), the martsDB entry wins
because we already know its label.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd  # type: ignore

logger = logging.getLogger(__name__)


def _write_fasta(records: list[tuple[str, str]], out_path: Path) -> None:
    """Write ``[(id, sequence), ...]`` as a 60-char-wrapped FASTA."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec_id, seq in records:
            fh.write(f">{rec_id}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path,
        default=Path("data/dark_candidates/all_candidates.csv"),
    )
    parser.add_argument(
        "--kingdom", type=Path,
        default=Path("data/dark_candidates/kingdom.csv"),
    )
    parser.add_argument(
        "--martsdb", type=Path,
        default=Path("data/martsDB_reactions_2026_02_22_preprocessed.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/dark_candidates"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    cand = pd.read_csv(args.candidates, usecols=["id", "sequence"]) \
        .dropna(subset=["id", "sequence"]).drop_duplicates("id")
    cand["sequence"] = cand["sequence"].astype(str).str.upper().str.replace(" ", "")
    logger.info("Candidates: %d", len(cand))

    king = pd.read_csv(args.kingdom)
    cand = cand.merge(king[["id", "kingdom"]], on="id", how="left")
    cand["kingdom"] = cand["kingdom"].fillna("Unknown")
    cand["source"] = "candidate"
    cand["kingdom_detailed"] = ""

    marts = pd.read_csv(
        args.martsdb,
        usecols=["Uniprot_ID", "Aminoacid_sequence", "Kingdom", "Kingdom_detailed"],
    ).dropna(subset=["Uniprot_ID", "Aminoacid_sequence"]).drop_duplicates("Uniprot_ID")
    marts = marts.rename(columns={
        "Uniprot_ID": "id",
        "Aminoacid_sequence": "sequence",
        "Kingdom": "kingdom",
        "Kingdom_detailed": "kingdom_detailed",
    })
    marts["sequence"] = marts["sequence"].astype(str).str.upper().str.replace(" ", "")
    marts["source"] = "martsDB"
    marts["kingdom_detailed"] = marts["kingdom_detailed"].fillna("")
    logger.info("martsDB unique entries: %d", len(marts))

    # Collision rule: martsDB wins. Drop any candidate row whose id is
    # already in martsDB so the FASTA stays unique on header.
    overlap = set(cand["id"]) & set(marts["id"])
    if overlap:
        logger.info(
            "Dropping %d candidate(s) that also appear in martsDB (martsDB wins)",
            len(overlap),
        )
        cand = cand[~cand["id"].isin(overlap)].reset_index(drop=True)

    combined = pd.concat(
        [cand[["id", "sequence", "source", "kingdom", "kingdom_detailed"]],
         marts[["id", "sequence", "source", "kingdom", "kingdom_detailed"]]],
        ignore_index=True,
    )
    combined["seq_len"] = combined["sequence"].str.len()
    logger.info(
        "Combined pool: %d records (cand=%d, martsDB=%d)",
        len(combined), (combined["source"] == "candidate").sum(),
        (combined["source"] == "martsDB").sum(),
    )
    logger.info(
        "Sequence-length quantiles (p50/p90/p99/max): %d / %d / %d / %d",
        int(combined["seq_len"].quantile(0.5)),
        int(combined["seq_len"].quantile(0.9)),
        int(combined["seq_len"].quantile(0.99)),
        int(combined["seq_len"].max()),
    )

    fasta_path = args.output_dir / "combined.fasta"
    meta_path = args.output_dir / "metadata.csv"
    _write_fasta(list(zip(combined["id"], combined["sequence"])), fasta_path)
    combined.drop(columns=["sequence"]).to_csv(meta_path, index=False)
    logger.info("Wrote %s and %s", fasta_path, meta_path)


if __name__ == "__main__":
    main()
