"""Combine TPS-only MARTS-DB enzymes with dark-putative candidates into a
single FASTA + metadata file for the hard-candidates selection phylogeny.

Inputs
------
* ``data/martsDB_reactions_2026_02_22.csv`` — MARTS-DB reactions;
  aggregated per enzyme to filter for at least one non-pt reaction.
* ``data/dark_proteome_screening/dark_putatives.csv``               — dark-proteome hits with
  TPS_p > 0.95.

Outputs
-------
* ``data/dark_proteome_screening/candidate_selection/phylo_tree/combined.fasta`` — MARTS-DB TPS
  enzymes labelled by their ``marts_E…`` ID, dark putatives labelled by
  ``dark_<UniProt>`` so the two sources are distinguishable at every
  downstream step.
* ``data/dark_proteome_screening/candidate_selection/phylo_tree/metadata.csv``    — one row per leaf:
  ``leaf_id, source, uniprot, kingdom, marts_id``. Kingdom is only
  populated for MARTS-DB rows (dark putatives have no kingdom label);
  the leaves are what the iTOL annotation step will consume.

The MARTS-DB Kingdom names in the reactions CSV are ``Plantae``,
``Fungi``, etc.  We normalise to the palette keys the rest of the
project uses (``Plants``, ``Fungi``, ``Animals``, ``Bacteria``,
``Archaea``, ``Protists``, ``Viruses``, ``Unknown``).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore

REPO = Path(__file__).resolve().parents[2]
REACTIONS_CSV = REPO / "data" / "martsDB_reactions_2026_02_22.csv"
DARK_CSV = REPO / "data" / "dark_proteome_screening" / "dark_putatives.csv"
OUT_DIR = REPO / "data" / "dark_proteome_screening" / "candidate_selection" / "phylo_tree"
FASTA_OUT = OUT_DIR / "combined.fasta"
META_OUT = OUT_DIR / "metadata.csv"

def _normalise_kingdom(raw: str) -> str:
    """Map MARTS-DB Kingdom field to the project's canonical palette keys."""
    if not raw or (isinstance(raw, float) and pd.isna(raw)):
        return "Unknown"
    s = str(raw).strip()
    if s.startswith("Plantae"):
        return "Plants"
    if s.startswith("Animalia"):
        return "Animals"
    if s in {"Bacteria", "Cyanobacteria"}:
        return "Bacteria"
    if s == "Fungi":
        return "Fungi"
    if s == "Archaea":
        return "Archaea"
    if s in {"Amoebozoa"}:
        return "Protists"
    if s == "Viruses":
        return "Viruses"
    return "Unknown"


def _load_martsdb_tps() -> pd.DataFrame:
    df = pd.read_csv(REACTIONS_CSV)
    per_enzyme = df.groupby("Enzyme_marts_ID").agg(
        seq=("Aminoacid_sequence", "first"),
        uniprot=("Uniprot_ID", "first"),
        kingdom=("Kingdom", "first"),
        types=("Type", lambda s: set(s.dropna().unique())),
    )
    tps_types = {"sesq", "mono", "di", "tri", "sester", "tetra", "hemi", "sesquar", "sqs", "psy"}
    per_enzyme["has_tps"] = per_enzyme["types"].map(lambda s: bool(s & tps_types))
    tps_only = per_enzyme[per_enzyme["has_tps"]].reset_index()
    tps_only["kingdom_norm"] = tps_only["kingdom"].map(_normalise_kingdom)
    print(f"MARTS-DB enzymes with ≥1 TPS reaction: {len(tps_only)}")
    print(
        "  kingdom breakdown: "
        + str(tps_only["kingdom_norm"].value_counts().to_dict())
    )
    return tps_only


def _load_dark_putatives() -> pd.DataFrame:
    df = pd.read_csv(DARK_CSV)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    print(f"Dark putatives: {len(df)}")
    return df


def _write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marts = _load_martsdb_tps()
    dark = _load_dark_putatives()

    fasta_records: list[tuple[str, str]] = []
    meta_rows: list[dict] = []

    for row in marts.itertuples(index=False):
        seq = str(row.seq or "").strip().replace("*", "")
        if not seq:
            continue
        leaf = row.Enzyme_marts_ID
        fasta_records.append((leaf, seq))
        meta_rows.append({
            "leaf_id": leaf,
            "source": "martsdb",
            "uniprot": row.uniprot if pd.notna(row.uniprot) else "",
            "kingdom": row.kingdom_norm,
            "marts_id": leaf,
        })

    for row in dark.itertuples(index=False):
        seq = str(row.sequence or "").strip().replace("*", "")
        if not seq:
            continue
        # Bare UniProt ID; MARTS-DB leaves are prefixed 'marts_E…', so the
        # 'starts-with-marts_' test is enough to distinguish sources
        # everywhere downstream.
        leaf = str(row.id)
        fasta_records.append((leaf, seq))
        meta_rows.append({
            "leaf_id": leaf,
            "source": "dark",
            "uniprot": row.id,
            "kingdom": "",
            "marts_id": "",
        })

    _write_fasta(fasta_records, FASTA_OUT)
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(META_OUT, index=False)
    print(f"Wrote {FASTA_OUT} ({len(fasta_records)} sequences)")
    print(f"Wrote {META_OUT}")


if __name__ == "__main__":
    main()
