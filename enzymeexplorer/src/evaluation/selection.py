"""Per-class version selection for homology-based baselines.

For models whose optimal threshold is class-dependent (Blastp, Foldseek, HMM,
Pfam, SUPFAM), the evaluation pipeline picks the version that maximises a
chosen scoring metric on each class and stitches the resulting per-class
experiment results into one virtual classifier. A YAML file may pin specific
versions; otherwise versions are auto-discovered from the configs directory
and scored on the latest run available on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

from enzymeexplorer.src.evaluation import io as eio
from enzymeexplorer.src.evaluation.classes import SHORT_TO_SMILES
from enzymeexplorer.src.utils.project_info import get_config_root, get_models_output_root

logger = logging.getLogger(__name__)

VersionSpec = str | dict[str, str]

WITH_DISTRACTORS_SUFFIX = "_with_distractors"


def discover_versions(
    model: str,
    *,
    prefix: str = "eval",
    config_root: Path | None = None,  # noqa: ARG001 — kept for API compat
    require_outputs: bool = True,  # noqa: ARG001 — kept for API compat
    output_root: Path | None = None,
    with_distractors: bool = False,
) -> list[str]:
    """Return version directories for ``model`` whose names start with
    ``prefix`` (e.g. ``"eval"`` for HMM/BLAST/Foldseek, ``"pfam_bitscore"``
    for PFAM).

    Discovery scans the **outputs** tree (``outputs/<model>/<version>/
    all_folds/all_classes/``) — the source of truth for "what experiments
    have been run". This decouples evaluation from the state of
    ``configs/<model>/``, which the user may toggle with ``.ignore``
    suffixes for unrelated reasons.

    The ``with_distractors`` flag selects between the two training-set
    universes: when ``False`` (default), versions ending in
    ``_with_distractors`` are excluded — the auto-pick stays in the
    no-distractor universe that matches the new training default. When
    ``True``, only ``_with_distractors`` versions are returned.
    """
    out_root = output_root or get_models_output_root()
    model_outputs = out_root / model
    if not model_outputs.is_dir():
        raise FileNotFoundError(f"No outputs dir for model: {model_outputs}")
    return sorted(
        p.name
        for p in model_outputs.iterdir()
        if p.is_dir()
        and p.name.startswith(prefix)
        and (p / "all_folds" / "all_classes").is_dir()
        and p.name.endswith(WITH_DISTRACTORS_SUFFIX) == with_distractors
    )


def _per_fold_metric(
    fold_dfs: dict[int, eio.FoldDfs], cls: str, metric: str
) -> float:
    scores: list[float] = []
    for _, (lab, prd) in fold_dfs.items():
        y = lab[cls].to_numpy()
        s = prd[cls].to_numpy()
        if y.sum() == 0 or y.sum() == len(y):
            continue
        if metric == "ap":
            scores.append(float(average_precision_score(y, s)))
        elif metric == "roc_auc":
            scores.append(float(roc_auc_score(y, s)))
        else:
            raise ValueError(f"Unsupported selection metric: {metric}")
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def pick_best_versions_per_class(
    model: str,
    candidate_versions: list[str],
    classes: Iterable[str],
    *,
    selection_metric: str = "ap",
    output_root: Path | None = None,
) -> dict[str, str]:
    """For each requested class, return the version with the highest mean
    per-fold metric. Versions that don't expose a class are silently skipped
    for that class. Raises if no version covers a class at all.
    """
    classes = list(classes)
    best: dict[str, str] = {}
    best_score: dict[str, float] = {c: float("-inf") for c in classes}
    for version in candidate_versions:
        try:
            exp_dir = eio.latest_experiment_dir(
                model, version, output_root=output_root
            )
            raws = eio.load_pickle_folds(exp_dir)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s/%s: %s", model, version, exc)
            continue
        common_smiles = set(raws[0][1])
        for _, names, _ in raws[1:]:
            common_smiles &= set(names)
        covered = [c for c in classes if SHORT_TO_SMILES[c] in common_smiles]
        if not covered:
            logger.warning(
                "Version %s/%s exposes none of the requested classes",
                model, version,
            )
            continue
        per_fold = eio.folds_to_dfs(raws, classes_subset=covered)
        for cls in covered:
            score = _per_fold_metric(per_fold, cls, selection_metric)
            if score > best_score[cls]:
                best_score[cls] = score
                best[cls] = version
    missing = [c for c in classes if c not in best]
    if missing:
        raise RuntimeError(
            f"No candidate versions covered classes {missing} for model {model}"
        )
    return best


def load_optimal_versions_yaml(path: Path) -> dict[str, VersionSpec]:
    """Load a YAML mapping ``{model: version_str | {class: version_str}}``."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Optimal versions YAML must be a mapping, got {type(data)}")
    return data


def resolve_classifier_version_spec(
    model: str,
    classes: Iterable[str],
    *,
    yaml_path: Path | None = None,
    discover_prefix: str | None = None,
    selection_metric: str = "ap",
    output_root: Path | None = None,
    config_root: Path | None = None,
    with_distractors: bool = False,
) -> VersionSpec:
    """Resolve the version spec a classifier should use for evaluation.

    Order of resolution:
    1. If ``yaml_path`` is given and contains an entry for ``model``, return it.
    2. If ``discover_prefix`` is given, auto-discover candidate versions and
       pick the per-class best.
    3. Otherwise raise.
    """
    if yaml_path is not None and yaml_path.exists():
        mapping = load_optimal_versions_yaml(yaml_path)
        if model in mapping:
            return mapping[model]
    if discover_prefix is not None:
        candidates = discover_versions(
            model,
            prefix=discover_prefix,
            config_root=config_root,
            output_root=output_root,
            with_distractors=with_distractors,
        )
        if not candidates:
            raise RuntimeError(
                f"No '{discover_prefix}*' versions discovered for {model}"
            )
        return pick_best_versions_per_class(
            model,
            candidates,
            classes,
            selection_metric=selection_metric,
            output_root=output_root,
        )
    raise ValueError(
        f"Cannot resolve version for {model}: provide yaml_path or discover_prefix"
    )


def write_optimal_versions_yaml(
    spec: dict[str, VersionSpec], path: Path
) -> None:
    """Persist a resolved spec to YAML for inspection or future reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False)
