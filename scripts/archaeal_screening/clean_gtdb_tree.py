"""Post-process the raw GTDB ar53 phylogeny so iTOL can render it.

GTDB ships ``ar53_r*.tree`` with quoted internal-node labels that carry
the taxonomic classification, e.g.::

    )'100.0:c__CAZHTU01; o__CAZHTU01':0.195063

iTOL parses those as terminating semicolons (Newick's own end-of-tree
marker) and refuses to load the file. Strip every single-quoted internal
label — the numeric bootstrap values that live outside quotes are
preserved, so branch supports remain intact.

Inputs :  data/archaeal_screening/ar53.tree       (raw from GTDB)
Outputs:  data/archaeal_screening/ar53_clean.tree (iTOL-loadable)

Meant to run once from ``setup_dev.sh`` as part of the archaeal-tree
provisioning step, right after the GTDB download + gunzip. Also
invocable directly for a one-off refresh:

    python -m scripts.archaeal_screening.clean_gtdb_tree
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "data" / "archaeal_screening" / "ar53.tree"
DEFAULT_OUT = REPO / "data" / "archaeal_screening" / "ar53_clean.tree"

# Non-greedy match on a single-quoted internal-node label; single quotes
# don't appear anywhere else in the GTDB Newick, so a global sub is safe.
_QUOTED_LABEL_RE = re.compile(r"'[^']*'")


def clean(raw: str) -> str:
    return _QUOTED_LABEL_RE.sub("", raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Missing GTDB ar53 tree at {args.input}. Run "
            f"scripts/archaeal_screening/download_gtdb_ar53_tree.sh first."
        )
    raw = args.input.read_text()
    cleaned = clean(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned)
    print(f"wrote {args.output} ({len(cleaned):,} chars, was {len(raw):,})")


if __name__ == "__main__":
    main()
