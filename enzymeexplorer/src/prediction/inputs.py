"""Sequence loading: FASTA or CSV → DataFrame[id, sequence]."""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore
from Bio import SeqIO  # type: ignore


DEFAULT_ID_COLUMN = "ID"
DEFAULT_SEQUENCE_COLUMN = "Aminoacid_sequence"


def _load_fasta(path: Path) -> pd.DataFrame:
    records = list(SeqIO.parse(str(path), "fasta"))
    return pd.DataFrame(
        [{"id": r.id, "sequence": str(r.seq)} for r in records]
    )


def _load_csv(
    path: Path,
    id_col: str,
    seq_col: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in (id_col, seq_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {path} is missing required column(s): {missing}. "
            f"Pass --id-column / --sequence-column to override."
        )
    return df.rename(columns={id_col: "id", seq_col: "sequence"})[["id", "sequence"]]


def load_sequences(
    path: str | Path,
    *,
    id_col: str = DEFAULT_ID_COLUMN,
    seq_col: str = DEFAULT_SEQUENCE_COLUMN,
) -> pd.DataFrame:
    """Load sequences from a FASTA (.fa/.fasta/.faa) or CSV (.csv) file.

    Returns a DataFrame with two normalised columns: ``id`` and ``sequence``.
    The ``id_col`` / ``seq_col`` arguments only apply to CSV inputs.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa"}:
        df = _load_fasta(path)
    elif suffix == ".csv":
        df = _load_csv(path, id_col, seq_col)
    else:
        raise ValueError(
            f"Unsupported sequence input extension {suffix!r} (expected "
            f".fasta/.fa/.faa or .csv)"
        )
    df = df.dropna(subset=["id", "sequence"]).drop_duplicates(subset="id")
    df["id"] = df["id"].astype(str)
    df["sequence"] = df["sequence"].astype(str).str.upper().str.replace(" ", "")
    return df.reset_index(drop=True)
