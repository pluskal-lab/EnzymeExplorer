"""Load ID → cluster-representative mappings used for cluster-block bootstrap.

The eval bootstrap treats each MMseqs2 cluster (50% seq-id / 80% coverage
by default — see ``scripts/evaluation/build_eval_clusters.py``) as one
resampling unit. This module owns the small helper that loads that TSV
and turns it into a lookup table + stable content hash for cache keying.

The TSV schema is exactly what ``mmseqs easy-cluster`` writes::

    Representative<TAB>Member

with one row per member. The loader is intentionally strict — missing
IDs are the caller's problem (either add them to the TSV as singletons
or filter the eval rows).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd  # type: ignore

ID_COLUMN = "Member"
REP_COLUMN = "Representative"


def load_cluster_map(tsv_path: Path | str) -> dict[str, str]:
    """Return ``{id: cluster_representative}`` from an mmseqs cluster TSV."""
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")
    missing = {REP_COLUMN, ID_COLUMN} - set(df.columns)
    if missing:
        raise ValueError(
            f"cluster TSV {tsv_path} missing columns {sorted(missing)}"
        )
    return dict(
        zip(df[ID_COLUMN].astype(str), df[REP_COLUMN].astype(str))
    )


def cluster_map_hash(cluster_map: dict[str, str]) -> str:
    """Stable short hash of a cluster map — feeds the bootstrap cache key.

    Keys are sorted so the hash is order-independent. Truncated to 16
    hex chars for legibility (SHA1 collisions on a mapping this small
    are not a concern)."""
    payload = "\n".join(
        f"{uid}\t{cluster_map[uid]}" for uid in sorted(cluster_map)
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]
