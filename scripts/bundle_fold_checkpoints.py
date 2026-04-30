"""Bundle per-fold trained classifiers into a single pickle.

Loads ``model_fold_{0..n}.pkl`` from a checkpoint directory in fold order and
pickles them as a single list. Used to package fold ensembles for the
prediction scripts and the API.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def bundle_fold_checkpoints(
    checkpoints_dir: Path, output_path: Path, n_folds: int = 5
) -> None:
    fold_models = []
    for fold_id in range(n_folds):
        fold_path = checkpoints_dir / f"model_fold_{fold_id}.pkl"
        if not fold_path.exists():
            raise FileNotFoundError(f"Missing fold checkpoint: {fold_path}")
        with fold_path.open("rb") as f:
            fold_models.append(pickle.load(f))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(fold_models, f)
    print(f"Wrote {len(fold_models)} fold models → {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        required=True,
        help="Directory containing model_fold_{0..n}.pkl files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/enzyme_explorer_checkpoints.pkl"),
        help="Output pickle path (a list of fold models, in fold order).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bundle_fold_checkpoints(args.checkpoints_dir, args.output_path, args.n_folds)
