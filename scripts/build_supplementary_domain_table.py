"""Rebuild the supplementary domain-composition CSV.

Same column layout as the previously-submitted table:
    <ID>, First domain type, First domain residues,
    Second domain type, Second domain residues,
    Third domain type, Third domain residues

Differences vs. the old file:
  * Row identifier is the MARTS-DB entry ID (e.g. ``marts_E01021``)
    instead of the UniProt accession.
  * "Domain type" uses the latest canonical subtype label
    (α1A..α4B, β, γ, δ1..δ3, ε, ζ) — same taxonomy used by the current
    dendrogram / scatter figures — instead of the coarse family label.

Rows are sorted by MARTS-DB ID; detections within a row keep their
detection-order from the domain-detection pipeline.
"""
from __future__ import annotations

import csv
import pickle
from pathlib import Path


def _residue_ranges(mapping: dict[int, int]) -> str:
    """Format the residue values as ``start-end+start-end+…`` ranges.

    ``residues_mapping`` values are the query-sequence positions; -1
    means the template position was not aligned. Filter those out.
    """
    xs = sorted({v for v in mapping.values() if v > 0})
    if not xs:
        return ""
    ranges: list[str] = []
    s = p = xs[0]
    for x in xs[1:]:
        if x == p + 1:
            p = x
        else:
            ranges.append(f"{s}-{p}" if s != p else f"{s}")
            s = p = x
    ranges.append(f"{s}-{p}" if s != p else f"{s}")
    return "+".join(ranges)


def main() -> None:
    detections = pickle.load(open(
        "data/detected_domains/martsDB_detected_domains/"
        "martsDB_detected_domains.pkl", "rb"))
    subtype_map = pickle.load(open(
        "data/domain_module_id_2_domain_subtype.pkl", "rb"))

    id_to_subtype = dict(subtype_map)

    out_path = Path("outputs/domain_clustering/supplementary_domain_composition_martsDB.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "MARTS-DB ID",
            "First domain type", "First domain residues",
            "Second domain type", "Second domain residues",
            "Third domain type", "Third domain residues",
        ])

        n_rows = 0
        for marts_id in sorted(detections):
            regs = detections[marts_id]
            cells: list[str] = []
            for reg in regs[:3]:
                subtype = id_to_subtype.get(reg.module_id, reg.domain)
                cells.extend([subtype, _residue_ranges(reg.residues_mapping)])
            # Pad to three (subtype, residues) pairs.
            while len(cells) < 6:
                cells.append("")
            writer.writerow([marts_id, *cells])
            n_rows += 1

    print(f"Wrote {n_rows} rows to {out_path}")


if __name__ == "__main__":
    main()
