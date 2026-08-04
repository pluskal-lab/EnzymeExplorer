"""CLI handlers for the bootstrap-driven evaluation pipeline.

``run_evaluate`` resolves classifier version specs (fixed or per-class auto)
into per-classifier bootstrap draws. Per-classifier results are cached under
``outputs/evaluation_results/_bootstrap_cache/<label>/<hash>/`` so repeated
runs across the four "purpose" output dirs (all-methods comparison,
ablation, confidence tiers, sweeps) share work. ``--force-bootstrap``
rebuilds every classifier; ``--force-classifiers X Y`` rebuilds only the
listed ones.

When the YAML config declares ``categories`` (e.g. Kingdom, OriginalType)
and ``id_metadata.csv``, categorical bootstrap is also run and persisted
under ``categories/<name>/`` (categorical draws are NOT cached — they
re-run each time, by design, since each output dir typically uses the
same categories and the cost is acceptable).

``run_visualize`` reads saved artefacts and renders the requested plots.
Visualize options can be supplied directly on the CLI or loaded from an
optional ``visualize:`` block inside the same eval YAML — CLI args take
precedence. After a visualize run, the effective options are dumped to
``visualize_config.yaml`` next to the plots so the run is reproducible.

Eval YAML schema::

    classifiers:
      <Label>:
        model: <Module class name in enzymeexplorer.src.models>
        classes: [TPS, IDS, FPP, ...]
        selection:
          mode: fixed | auto
          version: <str>            # required when mode=fixed
          prefix: <str>             # required when mode=auto
          metric: ap | roc_auc
          with_distractors: false   # optional; default false. When true the
                                    # auto-pick stays inside *_with_distractors
                                    # versions instead of the no-distractor
                                    # default universe.

    bootstrap:
      n_bootstraps: 2000
      seed: 42
      mode: cluster | rows
      ci: 0.95
      ci_method: percentile | bca
      metrics: [ap, roc_auc]

    aggregates:                     # optional
      Substrate_mAP: [FPP, GPP, ...]
      TPS_IDS_mAP: [TPS, IDS]
      Overall_mAP: [TPS, IDS, FPP, ...]

    categories:                     # optional; requires id_metadata.csv
      Kingdom:
        column: Kingdom
        order: [Bacteria, Fungi, Plants, Animals, Protists, Archaea, Viruses]
        negative_label: Unknown
      TPS_Type:
        column: OriginalType
        order: [mono, sesq, di, sester, tri, tetra, hemi, sqs, psy, pt]
        negative_label: Unknown

    id_metadata:                    # required when categories is set
      csv: data/EnzymeExplorer_Dataset.csv

    threshold_sweep:                # optional; pre-computes diagnostic data
      - label: BLAST
        model: Blastp
        prefix: eval
        x_axis: neglog10
        x_label: "E-value (-log10)"
        with_distractors: false     # optional; default false
      - label: PFAM
        model: PfamSUPFAM
        prefix: pfam_bitscore
        x_axis: tail_int
        x_label: "Bitscore"
        classes: [TPS]

    visualize:                      # optional; defaults for run_visualize
      plots: [tps_detection, substrate_map, ...]
      auto_zoom:
        enabled: true
        delta_pct: 2.0
      classifier_order: [...]
      pin_last: PLM_Domains         # pin this classifier to the right end
      ablation_xtick_overrides:
        tps_esm-1v-subseq: ESM-1v subseq (TPS)
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import re
import time
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import yaml  # type: ignore

from enzymeexplorer.src.evaluation import (
    aggregate as agg,
    bootstrap as bs,
    cache as boot_cache,
    calibration as cal,
    clusters as eval_clusters,
    io as eio,
    selection as sel,
)
from enzymeexplorer.src.evaluation.classes import (
    DEFAULT_PLOT_ORDER,
    SUBSTRATE_CLASSES,
)
from enzymeexplorer.src.evaluation.plotting import (
    bars,
    calibration as cal_plots,
    categorical,
    curves,
    deltas as delta_plots,
    theme,
    thresholds,
)
from enzymeexplorer.src.evaluation.significance import compute_pvalues
from enzymeexplorer.src.utils.project_info import get_evaluations_output

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _timed(step: str):
    """Log start/end + wall-clock elapsed for a named eval step.

    Usage::

        with _timed("compute CIs"):
            ...

    Emits INFO logs so an interactive user sees which stage the pipeline is
    on, and how long each stage took. Meant for stages that take more than
    ~1 second — otherwise the log noise outweighs the visibility.
    """
    t0 = time.perf_counter()
    logger.info("BEGIN %s", step)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info("END   %s (%.1fs)", step, elapsed)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts. Leaf values in ``override`` win. Lists are
    replaced (not concatenated) — matches YAML anchor semantics."""
    out = dict(base)
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_includes(node, base_dir: Path):
    """Recursively resolve ``include:`` keys inside a YAML tree.

    ``include: <path>`` OR ``include: [<p1>, <p2>, …]`` merges the listed
    files as parent defaults (later paths override earlier ones); the
    node's own keys then override the merged parent. Works at any
    nesting level — e.g. a ``visualize.include: ../_defaults.yaml``
    supplies defaults for the visualize block, and per-config overrides
    win.
    """
    if isinstance(node, dict):
        include_val = node.pop("include", None)
        if include_val is not None:
            includes = (
                [include_val] if isinstance(include_val, str)
                else list(include_val)
            )
            merged: dict = {}
            for inc_path in includes:
                inc_full = (base_dir / inc_path).resolve()
                inc_data = _load_yaml_with_includes(inc_full)
                merged = _deep_merge(merged, inc_data)
            node = _deep_merge(merged, node)
        return {k: _resolve_includes(v, base_dir) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_includes(v, base_dir) for v in node]
    return node


def _load_yaml_with_includes(path: Path) -> dict:
    """Load a YAML file with recursive ``include:`` resolution.

    See :func:`_resolve_includes` for the merge semantics.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        return raw
    return _resolve_includes(raw, path.parent)


def _resolve_version_spec(label: str, spec: dict, classes: list[str]):
    sel_cfg = spec["selection"]
    mode = sel_cfg["mode"]
    if mode == "fixed":
        return sel_cfg["version"]
    if mode == "auto":
        prefix = sel_cfg["prefix"]
        with_distractors = bool(sel_cfg.get("with_distractors", False))
        candidates = sel.discover_versions(
            spec["model"], prefix=prefix, with_distractors=with_distractors,
        )
        if not candidates:
            raise RuntimeError(
                f"No '{prefix}*' versions discovered for {label} "
                f"({spec['model']}, with_distractors={with_distractors})"
            )
        return sel.pick_best_versions_per_class(
            spec["model"],
            candidates,
            classes,
            selection_metric=sel_cfg.get("metric", "ap"),
        )
    raise ValueError(f"Unknown selection mode for {label}: {mode!r}")


def _build_classifier_dfs(
    cfg: dict,
    pinned_resolved: dict[str, dict] | None = None,
):
    """Resolve and load all classifiers in ``cfg``.

    When ``pinned_resolved`` is provided (a previously-saved
    ``resolved_versions.yaml`` payload), each classifier label found
    there bypasses ``_resolve_version_spec`` and uses the recorded
    ``resolved_versions`` and ``experiment_timestamps`` exactly. This
    is the supported way to reproduce an evaluation while a new
    training is mid-flight: the latest timestamped run might be
    incomplete or worse than the previous one.
    """
    classifier_to_dfs: dict[str, dict[str, dict[int, eio.FoldDfs]]] = {}
    classifier_to_timestamps: dict[str, dict[str, str]] = {}
    resolved: dict[str, dict] = {}
    for label, spec in cfg["classifiers"].items():
        classes = list(spec["classes"])
        pinned = pinned_resolved.get(label) if pinned_resolved else None
        if pinned is not None:
            version_spec = pinned["resolved_versions"]
            ts_per_class = dict(pinned.get("experiment_timestamps") or {})
            logger.info(
                "Pinned %s -> %s (timestamps from resolved_versions.yaml)",
                label, version_spec,
            )
        else:
            try:
                version_spec = _resolve_version_spec(label, spec, classes)
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("Skipping %s: cannot resolve version (%s)", label, exc)
                continue
            ts_per_class = None
            logger.info("Resolved %s -> %s", label, version_spec)
        try:
            dfs, timestamps = eio.load_classifier_class_fold_dfs(
                spec["model"], version_spec, classes=classes,
                timestamps_per_class=ts_per_class,
            )
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                "Skipping %s: prediction pickles incomplete or missing (%s)",
                label, exc,
            )
            continue
        classifier_to_dfs[label] = dfs
        classifier_to_timestamps[label] = timestamps
        resolved[label] = {
            "model": spec["model"],
            "classes": classes,
            "resolved_versions": version_spec,
            "experiment_timestamps": timestamps,
        }
    return classifier_to_dfs, classifier_to_timestamps, resolved


def _run_calibration_evaluate(
    cfg: dict,
    out_dir: Path,
    classifier_to_dfs: dict[str, dict[str, dict[int, eio.FoldDfs]]],
) -> None:
    """Compute calibration artefacts and persist them under ``<out_dir>/calibration/``.

    Invoked from two mutually-exclusive paths in this module:

    * :func:`run_calibrate` — the dedicated ``calibrate`` subcommand.
    * :func:`run_evaluate` — the "no ``bootstrap`` block" fast-path, kept
      for back-compat so a legacy calibration-only eval YAML still works
      when passed to ``evaluate``.

    Plots are
    NOT produced here — :func:`run_visualize` reads these CSVs/parquets
    and renders the figures.
    """
    cal_cfg = cfg.get("calibration") or {}
    if not cal_cfg:
        return

    target_classifiers = list(cal_cfg.get(
        "classifiers", list(classifier_to_dfs.keys()),
    ))
    target_classes = list(cal_cfg.get("classes", DEFAULT_PLOT_ORDER))
    min_n_pos = int(cal_cfg.get("min_n_pos", cal.DEFAULT_MIN_N_POS))
    score_eps = float(cal_cfg.get("score_eps", cal.DEFAULT_SCORE_EPS))
    n_bootstrap = int(cal_cfg.get("n_bootstrap", cal.DEFAULT_N_BOOTSTRAP))
    bootstrap_seed = int(cal_cfg.get(
        "bootstrap_seed", cal.DEFAULT_BOOTSTRAP_SEED,
    ))
    n_reliability_bins = int(cal_cfg.get(
        "n_reliability_bins", cal.DEFAULT_N_RELIABILITY_BINS,
    ))
    top_k_fp = int(cal_cfg.get("top_k_fp", cal.DEFAULT_TOP_K_FP))
    bottom_k_fn = int(cal_cfg.get("bottom_k_fn", cal.DEFAULT_BOTTOM_K_FN))
    ci = float(cal_cfg.get("ci", cal.DEFAULT_CI))
    fold_drift_threshold = float(cal_cfg.get(
        "fold_drift_threshold", cal.DEFAULT_FOLD_DRIFT_THRESHOLD,
    ))
    families = tuple(cal_cfg.get("families", cal.DEFAULT_FAMILIES))
    family_tolerance = float(cal_cfg.get(
        "family_tolerance", cal.DEFAULT_FAMILY_TOLERANCE,
    ))

    cal_dir = out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    oof_per_clf_class: dict[str, dict[str, cal.OofFrame]] = {}
    for clf in target_classifiers:
        cls_map = classifier_to_dfs.get(clf, {})
        oof_per_clf_class[clf] = {}
        for cls in target_classes:
            if cls not in cls_map:
                continue
            oof_per_clf_class[clf][cls] = cal.build_oof_frame(
                cls_map[cls], cls, clf,
            )

    # Cluster-block calibration bootstrap defaults; the config can override
    # via ``calibration.bootstrap_unit: fold|cluster|row`` and
    # ``calibration.cluster_tsv``. Defaults mirror the eval bootstrap so
    # both pipelines resample on the same 50%-seq-id groups.
    cal_bootstrap_unit = str(cal_cfg.get("bootstrap_unit", "cluster"))
    cal_cluster_tsv = cal_cfg.get(
        "cluster_tsv", "data/EnzymeExplorer_Dataset_clusters_50.tsv",
    )
    cal_cluster_map: dict[str, str] | None = None
    if cal_bootstrap_unit == "cluster":
        cal_cluster_map = eval_clusters.load_cluster_map(cal_cluster_tsv)
        logger.info(
            "Calibration cluster map loaded from %s (%d IDs, %d clusters)",
            cal_cluster_tsv, len(cal_cluster_map),
            len(set(cal_cluster_map.values())),
        )

    artefacts = cal.fit_calibration_table(
        oof_per_clf_class,
        families=families,
        family_tolerance=family_tolerance,
        min_n_pos=min_n_pos,
        eps=score_eps,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        n_reliability_bins=n_reliability_bins,
        top_k_fp=top_k_fp,
        bottom_k_fn=bottom_k_fn,
        ci=ci,
        fold_drift_threshold=fold_drift_threshold,
        bootstrap_unit=cal_bootstrap_unit,
        cluster_map=cal_cluster_map,
    )

    artefacts.fit_summary.to_csv(cal_dir / "fit_summary.csv", index=False)
    if not artefacts.selection_log_loss.empty:
        artefacts.selection_log_loss.to_csv(
            cal_dir / "selection_log_loss.csv", index=False,
        )
    if not artefacts.skipped.empty:
        artefacts.skipped.to_csv(cal_dir / "skipped.csv", index=False)
    if not artefacts.lofo_predictions.empty:
        artefacts.lofo_predictions.to_parquet(
            cal_dir / "lofo_predictions.parquet", index=False,
        )
    if not artefacts.reliability.empty:
        artefacts.reliability.to_csv(cal_dir / "reliability.csv", index=False)
    if not artefacts.reliability_per_fold.empty:
        artefacts.reliability_per_fold.to_csv(
            cal_dir / "reliability_per_fold.csv", index=False,
        )
    if not artefacts.metrics.empty:
        artefacts.metrics.to_csv(cal_dir / "metrics.csv", index=False)
    if not artefacts.metrics_per_fold.empty:
        artefacts.metrics_per_fold.to_csv(
            cal_dir / "metrics_per_fold.csv", index=False,
        )
    if not artefacts.lofo_metric_ci.empty:
        artefacts.lofo_metric_ci.to_csv(
            cal_dir / "lofo_metric_bootstrap_ci.csv", index=False,
        )
    if not artefacts.param_ci.empty:
        artefacts.param_ci.to_csv(
            cal_dir / "param_bootstrap_ci.csv", index=False,
        )
    if not artefacts.ribbon.empty:
        artefacts.ribbon.to_parquet(cal_dir / "ribbon.parquet", index=False)
    if not artefacts.ribbon_coverage.empty:
        artefacts.ribbon_coverage.to_csv(
            cal_dir / "ribbon_coverage.csv", index=False,
        )
    if not artefacts.per_fold_params.empty:
        artefacts.per_fold_params.to_csv(
            cal_dir / "per_fold_params.csv", index=False,
        )
    if not artefacts.fold_drift_summary.empty:
        artefacts.fold_drift_summary.to_csv(
            cal_dir / "fold_drift_summary.csv", index=False,
        )
    if not artefacts.hard_errors.empty:
        artefacts.hard_errors.to_csv(cal_dir / "hard_errors.csv", index=False)

    n_fits = int((artefacts.fit_summary["family"].notna()).sum()) \
        if not artefacts.fit_summary.empty else 0
    logger.info(
        "Saved calibration artefacts to %s (%d fitted (clf, class) pairs)",
        cal_dir, n_fits,
    )


def _x_position(version: str, mode: str) -> float | None:
    if mode == "neglog10":
        return thresholds.parse_eval_neglog10(version)
    if mode == "tail_int":
        m = re.search(r"(\d+)$", version)
        return float(m.group(1)) if m else None
    if mode == "categorical":
        return None
    raise ValueError(f"Unknown threshold_sweep x_axis: {mode}")


def run_evaluate(args: argparse.Namespace) -> None:
    cfg = _load_yaml_with_includes(Path(args.config))
    out_dir = get_evaluations_output() / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pinned_resolved: dict[str, dict] | None = None
    if getattr(args, "use_existing_resolved_versions", False):
        rv_path = out_dir / "resolved_versions.yaml"
        if rv_path.exists():
            loaded = yaml.safe_load(rv_path.read_text()) or {}
            pinned_resolved = loaded
            logger.info(
                "Pinning to existing resolved_versions.yaml at %s "
                "(skipping spec-based version resolution for %d labels)",
                rv_path, len(loaded),
            )
        else:
            logger.warning(
                "--use-existing-resolved-versions set but %s does not exist; "
                "falling back to spec-based resolution",
                rv_path,
            )

    with _timed(f"resolve classifier specs + load fold pickles ({len(cfg['classifiers'])} classifiers)"):
        classifier_to_dfs, classifier_to_timestamps, resolved = _build_classifier_dfs(
            cfg, pinned_resolved=pinned_resolved,
        )

    with open(out_dir / "resolved_versions.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    with open(out_dir / "eval_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    bcfg = cfg.get("bootstrap")
    # Calibration-only configs (e.g. evaluation/calibration/*.yaml) have no
    # ``bootstrap`` block. The calibration data work (family selection,
    # cluster bootstrap, reliability, hard errors) runs at the bottom of
    # this function via _run_calibration_evaluate; visualize then only
    # renders plots from the saved artefacts. Skip the paired-bootstrap
    # pipeline so calibration-only runs finish quickly.
    if not bcfg:
        logger.info(
            "No 'bootstrap' block in config — skipping paired bootstrap, "
            "summary CIs, p-values, aggregates, and categorical AP."
        )
        # Still emit the threshold_sweep artefacts if asked (they don't
        # depend on bootstrap output either).
        sweep_cfg = cfg.get("threshold_sweep")
        if sweep_cfg:
            sweep_dir = out_dir / "threshold_sweep"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            for entry in sweep_cfg:
                label = entry["label"]
                sweep_classes = entry.get("classes", DEFAULT_PLOT_ORDER)
                with_distractors = bool(entry.get("with_distractors", False))
                candidates = sel.discover_versions(
                    entry["model"], prefix=entry["prefix"],
                    with_distractors=with_distractors,
                )
                sweep_df = thresholds.compute_threshold_sweep(
                    entry["model"], candidates, sweep_classes,
                    metric=entry.get("metric", "ap"),
                )
                sweep_df["x_axis"] = entry.get("x_axis", "neglog10")
                sweep_df["x_label"] = entry.get("x_label", "")
                sweep_df.to_csv(sweep_dir / f"{label}.csv", index=False)
                logger.info("Saved threshold sweep '%s' (%d rows)", label, len(sweep_df))

        _run_calibration_evaluate(cfg, out_dir, classifier_to_dfs)
        return

    metrics = tuple(bcfg.get("metrics", ["ap", "roc_auc"]))
    n_boot = int(bcfg.get("n_bootstraps", 1000))
    seed = int(bcfg.get("seed", 42))
    # Default to pooled_oof only under the v5 cluster-block strategy —
    # fold_mean is unnecessary once resampling is cluster-level: the
    # pooled draw already respects the leakage-preventing group structure,
    # so averaging per-fold APs adds no independent information but
    # multiplies runtime. Users can still opt fold_mean back in via
    # ``ap_types: [pooled_oof, fold_mean]`` if they need paired per-fold
    # variance for a specific analysis.
    ap_types = tuple(bcfg.get("ap_types", ["pooled_oof"]))
    ci = float(bcfg.get("ci", 0.95))
    target_model = bcfg.get("target_model")
    p_adjustment = bcfg.get("p_adjustment", "holm")

    # v5 cluster-block bootstrap defaults. Legacy row bootstrap is opt-in
    # via ``bootstrap.bootstrap_unit: rows`` for reproducing pre-v5 numbers.
    # Some historical YAMLs use ``mode: cluster|rows`` — honour that as an
    # alias and normalise the value.
    bootstrap_unit = str(bcfg.get("bootstrap_unit", bcfg.get("mode", "clusters")))
    if bootstrap_unit == "cluster":
        bootstrap_unit = "clusters"
    if bootstrap_unit == "row":
        bootstrap_unit = "rows"
    cluster_tsv = bcfg.get(
        "cluster_tsv", "data/EnzymeExplorer_Dataset_clusters_50.tsv",
    )
    cluster_map: dict[str, str] | None = None
    cluster_map_hash: str | None = None
    if bootstrap_unit == "clusters":
        cluster_map = eval_clusters.load_cluster_map(cluster_tsv)
        cluster_map_hash = eval_clusters.cluster_map_hash(cluster_map)
        logger.info(
            "Loaded cluster map from %s (%d IDs, %d clusters, hash=%s)",
            cluster_tsv, len(cluster_map),
            len(set(cluster_map.values())), cluster_map_hash,
        )

    force_all = bool(getattr(args, "force_bootstrap", False))

    classifier_metadata = {
        label: {
            "model": cfg["classifiers"][label]["model"],
            "version_spec": resolved[label]["resolved_versions"],
            "timestamps": classifier_to_timestamps[label],
        }
        for label in classifier_to_dfs
    }

    with _timed(
        f"paired bootstrap ({n_boot} draws × {len(classifier_to_dfs)} classifiers, "
        f"unit={bootstrap_unit}, includes leave-one-cluster-out jackknife)"
    ):
        result, cache_hit = boot_cache.paired_bootstrap_with_cache(
            classifier_to_dfs,
            classifier_metadata,
            metrics=metrics,
            ap_types=ap_types,
            n_bootstraps=n_boot,
            seed=seed,
            target_model=target_model,
            force=force_all,
            bootstrap_unit=bootstrap_unit,
            cluster_map=cluster_map,
            cluster_map_hash=cluster_map_hash,
        )
    logger.info(
        "%s paired bootstrap (unit=%s): %s (n_classifiers=%d, n_draws=%d, "
        "ap_types=%s)",
        boot_cache.BOOTSTRAP_ALGO_VERSION, bootstrap_unit,
        "cache hit" if cache_hit else "fresh",
        len(classifier_to_dfs), n_boot, ap_types,
    )

    if aggregates := cfg.get("aggregates"):
        with _timed(f"add per-draw aggregates ({list(aggregates)})"):
            result = agg.add_aggregates(result, aggregates)
    with _timed("save bootstrap draws + point estimates + jackknife tables"):
        result.save(out_dir)

    # AP summaries: BCa only (the reported headline CI). The former
    # normal/percentile methods were dropped because BCa is the only
    # CI reported in the paper.
    ap_pieces: list[pd.DataFrame] = []
    delta_pieces: list[pd.DataFrame] = []
    with _timed("compute AP CIs (bca)"):
        try:
            ap_summary = bs.compute_cis(
                result.long_ap, result.point_ap, method="bca", ci=ci,
                jackknife=result.jackknife_ap,
            )
            ap_summary["ci_method"] = "bca"
            ap_pieces.append(ap_summary)
        except ValueError as exc:
            logger.warning("Skipping AP CI (bca): %s", exc)
    if not result.long_delta.empty:
        with _timed("compute delta CIs (bca)"):
            try:
                d_summary = bs.compute_cis(
                    result.long_delta, result.point_delta, method="bca", ci=ci,
                    jackknife=result.jackknife_delta,
                )
                d_summary["ci_method"] = "bca"
                delta_pieces.append(d_summary)
            except ValueError as exc:
                logger.warning("Skipping delta CI (bca): %s", exc)
    summary_ap = pd.concat(ap_pieces, ignore_index=True) if ap_pieces else pd.DataFrame()
    summary_delta = (
        pd.concat(delta_pieces, ignore_index=True) if delta_pieces else pd.DataFrame()
    )
    summary_ap.to_csv(out_dir / "summary_ap.csv", index=False)
    summary_delta.to_csv(out_dir / "summary_delta.csv", index=False)

    # P-values per (class, metric, ap_type). Adjustment applied within
    # each family.
    with _timed(f"compute paired-bootstrap p-values (adjustment={p_adjustment})"):
        pvalues = compute_pvalues(result.long_delta, adjustment=p_adjustment)
        pvalues.to_csv(out_dir / "pvalues.csv", index=False)

    logger.info("Saved bootstrap + CI + p-value artefacts to %s", out_dir)

    # ----------------------------------------------------------------------
    # Categorical AP (Kingdom + TPS_Type) — reuses the same draws as the
    # main bootstrap. Type-grouped categories are pure post-processing of
    # ``result.long_ap`` (no replay); masked categories replay the v4 RNG
    # with the same seed and re-evaluate AP under a row mask.
    # ----------------------------------------------------------------------
    categories_cfg = cfg.get("categories") or {}
    if categories_cfg:
        logger.info(
            "Running categorical AP for %d category axes: %s",
            len(categories_cfg), list(categories_cfg),
        )
        from enzymeexplorer.src.evaluation import categorical_bootstrap as cat_bs

        type_groupings: dict[str, dict[str, str]] = {}
        masked_categories: dict[str, dict[str, str]] = {}
        negative_labels: dict[str, str] = {}
        for cat_name, cat_spec in categories_cfg.items():
            negative_labels[cat_name] = cat_spec.get("negative_label", "Unknown")
            if cat_spec.get("type_grouping"):
                from enzymeexplorer.src.evaluation.classes import SUBSTRATE_TO_TYPE

                type_groupings[cat_name] = dict(
                    cat_spec.get("class_to_type") or SUBSTRATE_TO_TYPE
                )
            else:
                meta_csv = cfg.get("id_metadata", {}).get("csv")
                if not meta_csv:
                    raise ValueError(
                        f"category {cat_name!r} requires id_metadata.csv"
                    )
                kingdom_cache = cfg.get("id_metadata", {}).get(
                    "kingdom_cache", "data/uniprot_kingdom_cache.json"
                )
                meta = eio.load_id_metadata(
                    meta_csv, [cat_spec["column"]],
                    kingdom_cache=kingdom_cache,
                )
                masked_categories[cat_name] = meta[cat_spec["column"]]

        long_pieces: list[pd.DataFrame] = []
        point_pieces: list[pd.DataFrame] = []
        if type_groupings:
            with _timed(
                f"categorical AP — type-grouped aggregates ({list(type_groupings)})"
            ):
                l, p = cat_bs.compute_type_aggregated_ap(
                    result.long_ap, result.point_ap, type_groupings=type_groupings,
                )
                if not l.empty:
                    long_pieces.append(l)
                if not p.empty:
                    point_pieces.append(p)
        if masked_categories:
            with _timed(
                f"categorical AP — masked replay for {list(masked_categories)}"
                f" ({n_boot} draws each)"
            ):
                l, p = cat_bs.compute_masked_ap_from_seed(
                    classifier_to_dfs,
                    masked_categories=masked_categories,
                    metrics=metrics,
                    ap_types=ap_types,
                    n_bootstraps=n_boot,
                    seed=seed,
                    negative_labels=negative_labels,
                    bootstrap_unit=bootstrap_unit,
                    cluster_map=cluster_map,
                )
                if not l.empty:
                    long_pieces.append(l)
                if not p.empty:
                    point_pieces.append(p)

        if long_pieces or point_pieces:
          with _timed("categorical AP — CIs + persist artefacts"):
            cat_long = (
                pd.concat(long_pieces, ignore_index=True)
                if long_pieces else pd.DataFrame()
            )
            cat_point = (
                pd.concat(point_pieces, ignore_index=True)
                if point_pieces else pd.DataFrame()
            )
            cat_long.to_csv(out_dir / "bootstrap_long_categorical_ap.csv", index=False)
            cat_point.to_csv(out_dir / "point_estimates_categorical_ap.csv", index=False)
            # Per-category CIs — percentile only. The paired-bootstrap
            # replay used for categorical AP has no jackknife (categories
            # are masked post-hoc from the same RNG draws), so BCa
            # acceleration isn't computable here. Percentile is the
            # closest paired equivalent.
            cat_summary_pieces: list[pd.DataFrame] = []
            if not cat_long.empty:
                cat_keys = ["classifier", "class", "metric", "ap_type",
                            "category_name", "category"]
                grouped = cat_long.groupby(cat_keys)["value"]
                rows = []
                for key, draws in grouped:
                    arr = draws.dropna().to_numpy(dtype=float)
                    if arr.size == 0:
                        continue
                    rows.append({
                        **dict(zip(cat_keys, key)),
                        "point": float(np.median(arr)),
                        "ci_low": float(np.quantile(arr, (1 - ci) / 2)),
                        "ci_high": float(np.quantile(arr, 1 - (1 - ci) / 2)),
                        "ci_method": "percentile",
                    })
                cat_summary_pieces.append(pd.DataFrame.from_records(rows))
            cat_summary = (
                pd.concat(cat_summary_pieces, ignore_index=True)
                if cat_summary_pieces else pd.DataFrame()
            )
            cat_summary.to_csv(out_dir / "summary_categorical_ap.csv", index=False)
            logger.info(
                "Saved categorical AP artefacts (%d type-grouped + %d masked) to %s",
                len(type_groupings), len(masked_categories), out_dir,
            )

    sweep_cfg = cfg.get("threshold_sweep")
    if sweep_cfg:
        logger.info("Running %d threshold sweeps: %s",
                    len(sweep_cfg), [e["label"] for e in sweep_cfg])
        sweep_dir = out_dir / "threshold_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for entry in sweep_cfg:
            label = entry["label"]
            with _timed(f"threshold sweep '{label}' ({entry['model']}/{entry['prefix']}*)"):
                sweep_classes = entry.get("classes", DEFAULT_PLOT_ORDER)
                with_distractors = bool(entry.get("with_distractors", False))
                candidates = sel.discover_versions(
                    entry["model"], prefix=entry["prefix"],
                    with_distractors=with_distractors,
                )
                sweep_df = thresholds.compute_threshold_sweep(
                    entry["model"], candidates, sweep_classes,
                    metric=entry.get("metric", "ap"),
                )
                sweep_df["x_axis"] = entry.get("x_axis", "neglog10")
                sweep_df["x_label"] = entry.get("x_label", "")
                sweep_df.to_csv(sweep_dir / f"{label}.csv", index=False)
                logger.info("Saved threshold sweep '%s' (%d rows)", label, len(sweep_df))

    with _timed("calibration fitting (LOFO family selection + cluster bootstrap ribbon)"):
        _run_calibration_evaluate(cfg, out_dir, classifier_to_dfs)


def _load_classifier_dfs_for_visualize(eval_dir: Path):
    cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    resolved_path = eval_dir / "resolved_versions.yaml"
    if resolved_path.exists():
        # Self-contained re-load: use the per-class versions captured at
        # evaluate time so visualize does not depend on the original
        # configs/<model>/ dirs still being in place.
        resolved = yaml.safe_load(resolved_path.read_text())
        classifier_to_dfs: dict[str, dict[str, dict[int, eio.FoldDfs]]] = {}
        for label, info in resolved.items():
            dfs, _ = eio.load_classifier_class_fold_dfs(
                info["model"],
                info["resolved_versions"],
                classes=list(info["classes"]),
            )
            classifier_to_dfs[label] = dfs
        return classifier_to_dfs, cfg
    return _build_classifier_dfs(cfg)[0], cfg


def _resolve_classifier_order(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    vcfg: dict,
    cfg: dict | None = None,
) -> list[str]:
    summary_order = (
        sorted(summary["classifier"].unique())
        if not summary.empty and "classifier" in summary.columns
        else []
    )
    order = (
        args.classifier_order
        or vcfg.get("classifier_order")
        or summary_order
        or (list(cfg.get("classifiers", {}).keys()) if cfg else [])
    )
    return list(order)


def _apply_pin_last(order: list[str], pin: str | None) -> list[str]:
    if pin is None or pin not in order:
        return order
    return [c for c in order if c != pin] + [pin]


def _delta_pair_order(
    summary_delta: pd.DataFrame, *, classifier_rank: list[str],
) -> list[tuple[str, str]]:
    """Sort delta pairs (a, b) by performance.

    Each pair is keyed by ``(rank(a), rank(b))`` where ``rank`` is the
    index of the classifier in ``classifier_rank`` (worst→best). Lower
    rank → earlier on the x-axis. Pair orientation (which side is
    ``classifier_a``) is preserved from ``summary_delta``; ordering is
    only over the existing pairs."""
    rank = {c: i for i, c in enumerate(classifier_rank)}
    pairs = (
        summary_delta[["classifier_a", "classifier_b"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    big = len(classifier_rank) + 1
    return sorted(
        pairs,
        key=lambda p: (
            min(rank.get(p[0], big), rank.get(p[1], big)),
            max(rank.get(p[0], big), rank.get(p[1], big)),
        ),
    )


def _render_v4_scenarios(
    *,
    eval_dir: Path,
    summary_ap: pd.DataFrame,
    summary_delta: pd.DataFrame,
    long_ap: pd.DataFrame,
    long_delta: pd.DataFrame,
    pvalues: pd.DataFrame,
    cfg: dict,
    plot_set: set[str] | None = None,
    plots_subdir: str = "plots",
) -> None:
    """Write 14 plots per scenario into ``eval_dir/plots/<scenario>/``.

    Scenarios:
      * ``tps_detection``     — single class TPS
      * ``substrate_map``     — Substrate_mAP aggregate
      * ``substrate_per_class`` — every substrate class

    Per scenario:
      * ap/ ⟶ 6 bar plots (2 ap_types × 3 ci_methods)
      * delta/ ⟶ 6 delta forest plots
      * pvalues/ ⟶ 2 p-value heatmaps (one per ap_type)
    """
    plots_root = eval_dir / plots_subdir
    plots_root.mkdir(parents=True, exist_ok=True)

    # Auto-zoom knobs (all under ``visualize.auto_zoom`` — see cli docs).
    _az = dict((cfg.get("visualize") or {}).get("auto_zoom") or {})
    _zoom_padding_frac = float(_az.get("padding_frac", 0.0))
    _zoom_snap_step = _az.get("snap_step")
    _zoom_snap_step = float(_zoom_snap_step) if _zoom_snap_step else None

    vcfg = cfg.get("visualize", {}) or {}
    # Multi-metric plotting: ``visualize.metrics`` (list) wins, else fall
    # back to legacy ``visualize.metric`` (scalar). Each metric writes to
    # its own subfolder ``plots/<scenario>/<metric>/{bars,delta,pvalues}/``
    # so PR-AUC / ROC-AUC / MCC-F1 can co-exist for the same scenario.
    metrics_list = list(vcfg.get("metrics") or [vcfg.get("metric", "ap")])
    explicit_order = list(vcfg.get("classifier_order") or [])
    pin_last = vcfg.get("pin_last")
    fixed_classifier_order = bool(vcfg.get("fixed_classifier_order"))
    xtick_overrides = dict(vcfg.get("ablation_xtick_overrides") or {})
    yaml_palette = dict(vcfg.get("palette") or {}) or None

    ap_types_present = sorted(summary_ap["ap_type"].unique())
    ci_methods_present = sorted(summary_ap["ci_method"].unique())

    # Canonical sort key — picks a single (ap_type, ci_method) so the order
    # is the same across all 6 panel variants of a given scenario.
    sort_ap_type = "fold_mean" if "fold_mean" in ap_types_present else ap_types_present[0]
    sort_ci_method = "normal" if "normal" in ci_methods_present else ci_methods_present[0]

    # Effective palette resolution — three modes:
    #   * ``palette_mode: all_methods`` — Nature-Chem-Biol all-methods
    #     figure. EnzymeExplorer green, BLAST sky, Foldseek blue, rest
    #     neutral grey. Colorblind-safe (Wong).
    #   * ``palette_mode: ablation``   — Every bar sky-blue; the ablation
    #     x-axis labels carry the identity, no colour coding needed.
    #   * default / ``auto`` — legacy: master ``UNIVERSAL_PALETTE`` +
    #     YAML overrides + comparison-ramp fallback for unknown names,
    #     poster two-tone kicks in with ``pin_last``.
    classifiers_present = list(summary_ap["classifier"].unique())
    palette_mode = vcfg.get("palette_mode", "auto")
    if palette_mode == "all_methods":
        palette = theme.ncb_all_methods_palette(classifiers_present)
    elif palette_mode == "ablation":
        # Placeholder — the real Greens ramp is recomputed per scenario
        # below because it depends on ``scenario_order`` (which puts the
        # pinned "final" model on the right). Uniform light-green fill
        # here just ensures every classifier has a valid key up-front.
        palette = {c: theme.NCB_ABLATION_LIGHT_GREEN for c in classifiers_present}
    else:
        palette = dict(theme.UNIVERSAL_PALETTE)
        if yaml_palette:
            palette.update(yaml_palette)
        missing = [c for c in classifiers_present if c not in palette]
        if missing:
            fallback = theme.comparison_palette(missing)
            for c, rgb in fallback.items():
                palette.setdefault(c, rgb)
        if theme.is_poster() and pin_last:
            palette = theme.poster_two_tone_palette(classifiers_present, pin_last)
    # YAML palette overrides always take precedence per-key, so users
    # can pin one classifier's colour without redefining the whole map.
    if yaml_palette and palette_mode != "auto":
        palette.update(yaml_palette)

    # Delta forest colouring — one colour for every box across every
    # pair. Ablation mode defaults to light green (NCB_ABLATION_LIGHT_GREEN)
    # so the delta plots stay in the same green family as the bars/curves;
    # all-methods (or a user override) keeps the sky-blue default.
    uniform_delta_color = vcfg.get("uniform_delta_color")
    if uniform_delta_color is None and palette_mode == "ablation":
        uniform_delta_color = theme.NCB_ABLATION_LIGHT_GREEN

    # Clustered-bar palette (substrate_per_class scenario). The flat
    # bar palette would give every "grey" method the same colour inside
    # a cluster — indistinguishable. Same for ablation: uniform sky-
    # blue kills every method's identity when there are 5 bars in one
    # cluster. Use the shade-distinguished curve palette instead: NCB
    # identity colours for EE/BLAST/Foldseek, sequential grey shades
    # for the "other" bucket in all-methods, Blues ramp across the N
    # methods in ablation.
    if palette_mode == "all_methods":
        cluster_palette = theme.ncb_all_methods_curve_palette(classifiers_present)
    elif palette_mode == "ablation":
        # Legacy fallback; the real per-scenario ramp is recomputed below
        # so cluster ordering respects each scenario's classifier_order.
        shades = theme.ncb_curve_shades(classifiers_present, hue="green")
        cluster_palette = {c: h for c, h in zip(classifiers_present, shades)}
    else:
        cluster_palette = palette

    # Per-plot-type style overrides. The visualize schema accepts:
    #
    #   bar_plots:                    # bars.bar_classifier / bar_per_class
    #     tps_detection:      {figsize: [W,H], bar_width: 0.85}
    #     substrate_map:      {figsize: [W,H], bar_width: 0.85}
    #     substrate_per_class: {figsize: [W,H], cluster_width: 0.8,
    #                          bar_width_frac: 0.92}
    #   delta_plots:                  # deltas.plot_delta_forest
    #     figsize: [W,H]
    #     box_width: 0.30             # non-grouped panels
    #     cluster_width: 0.88         # substrate_per_class grouped panel
    #     grouped_box_frac: 0.675     # per-box width fraction inside cluster
    #   headline:                     # optional headline extras
    #     enabled: true
    #     methods: [PLM_Domains, Foldseek, BLAST]
    #     bar_plots: {…}              # any subset of the above, wins over
    #     delta_plots: {…}            # the top-level style for headline
    #
    # Legacy top-level ``bar_figsize`` is honoured as a default across
    # every bar_plots.* if the per-scenario block does not set figsize.
    bar_plot_cfg = dict(vcfg.get("bar_plots") or {})
    delta_plot_cfg = dict(vcfg.get("delta_plots") or {})
    headline_cfg = dict(vcfg.get("headline") or {})
    legacy_bar_figsize = vcfg.get("bar_figsize")

    def _bar_kwargs_for(scenario_name: str, *, overrides: dict | None = None):
        """Merge (defaults ← scenario ← headline overrides) for a bar plot."""
        cfg_here = dict(bar_plot_cfg.get(scenario_name) or {})
        if overrides:
            cfg_here.update(overrides)
        fs = cfg_here.get("figsize") or legacy_bar_figsize
        out: dict = {}
        if fs:
            out["figsize"] = tuple(fs)
        if scenario_name == "substrate_per_class":
            if "cluster_width" in cfg_here:
                out["cluster_width"] = float(cfg_here["cluster_width"])
            if "bar_width_frac" in cfg_here:
                out["bar_width_frac"] = float(cfg_here["bar_width_frac"])
        else:
            if "bar_width" in cfg_here:
                out["bar_width"] = float(cfg_here["bar_width"])
        if "xtick_rotation" in cfg_here:
            out["xtick_rotation"] = float(cfg_here["xtick_rotation"])
        if "xtick_fontsize" in cfg_here:
            out["xtick_fontsize"] = float(cfg_here["xtick_fontsize"])
        if "ytick_fontsize" in cfg_here:
            out["ytick_fontsize"] = float(cfg_here["ytick_fontsize"])
        if "title_fontsize" in cfg_here:
            out["title_fontsize"] = float(cfg_here["title_fontsize"])
        return out

    def _delta_kwargs(*, overrides: dict | None = None):
        cfg_here = dict(delta_plot_cfg)
        if overrides:
            cfg_here.update(overrides)
        out: dict = {}
        fs = cfg_here.get("figsize")
        if fs:
            out["figsize"] = tuple(fs)
        for k in ("box_width", "cluster_width", "grouped_box_frac", "title_fontsize"):
            if k in cfg_here:
                out[k] = float(cfg_here[k])
        return out

    def _order_for(class_list: list[str], metric_key: str) -> list[str]:
        """Resolve the per-scenario classifier order.

        * Explicit ``classifier_order`` (with ``fixed_classifier_order``)
          wins outright — used by ablation configs.
        * Otherwise sort classifiers by mean point value across the
          scenario's classes (worst→best, left→right) using the canonical
          (``fold_mean``, ``normal``) cell for the given metric, then
          move ``pin_last`` to the end if set — used by
          all_methods_comparison configs.
        """
        if explicit_order and fixed_classifier_order:
            return list(explicit_order)
        sub = summary_ap[
            (summary_ap["metric"] == metric_key)
            & (summary_ap["ap_type"] == sort_ap_type)
            & (summary_ap["ci_method"] == sort_ci_method)
            & (summary_ap["class"].isin(class_list))
        ].dropna(subset=["point"])
        if sub.empty:
            order = sorted(summary_ap["classifier"].unique())
        else:
            order = (
                sub.groupby("classifier")["point"].mean()
                   .sort_values(ascending=True).index.tolist()
            )
        if pin_last and pin_last in order:
            order = [c for c in order if c != pin_last] + [pin_last]
        return order

    scenarios = [
        ("tps_detection", ["TPS"], False, "TPS detection"),
        ("substrate_map", ["Substrate_mAP"], False, "Substrate prediction"),
        ("substrate_per_class",
            [c for c in DEFAULT_PLOT_ORDER if c in SUBSTRATE_CLASSES],
            True, "Substrate prediction per class"),
    ]
    if plot_set is not None:
        scenarios = [s for s in scenarios if s[0] in plot_set]

    # ``_order_for`` uses one metric to rank classifiers; we lock that
    # to the first metric in ``metrics_list`` so the same left→right
    # order is used for every metric of a given scenario.
    ranking_metric = metrics_list[0] if metrics_list else "ap"

    # Cache the substrate_map order so substrate_per_class can reuse it —
    # user wants both bar figures to share left-to-right method order.
    substrate_map_order: list[str] | None = None

    for scenario_name, class_list, multi_class, scenario_title in scenarios:
        if scenario_name == "substrate_per_class" and substrate_map_order is not None:
            scenario_order = [c for c in substrate_map_order
                              if c in classifiers_present]
        else:
            scenario_order = _order_for(class_list, ranking_metric)
        if scenario_name == "substrate_map":
            substrate_map_order = list(scenario_order)

        # Ablation palette resolution per scenario:
        #   * bar_classifier (tps_detection / substrate_map) — the
        #     rightmost (pinned "final") method takes NCB_GREEN, every
        #     other method the same lighter green
        #     (NCB_ABLATION_LIGHT_GREEN). Two-tone; ordering carries the
        #     "worst → best" story.
        #   * bar_per_class (substrate_per_class) and curves — a
        #     full light→dark green RAMP across the classifier order so
        #     within-cluster and within-panel lines can be told apart.
        if palette_mode == "ablation":
            final_method = scenario_order[-1] if scenario_order else None
            scenario_palette = {
                c: (theme.NCB_GREEN if c == final_method
                    else theme.NCB_ABLATION_LIGHT_GREEN)
                for c in scenario_order
            }
            scenario_cluster_palette = theme.ncb_ablation_palette(scenario_order)
        else:
            scenario_palette = palette
            scenario_cluster_palette = cluster_palette

        for metric in metrics_list:
            scen_dir = plots_root / scenario_name / metric
            bar_dir = scen_dir / "bars"
            dlt_dir = scen_dir / "delta"
            pv_dir = scen_dir / "pvalues"
            for d in (bar_dir, dlt_dir, pv_dir):
                d.mkdir(parents=True, exist_ok=True)

            for ap_type in ap_types_present:
                for ci_method in ci_methods_present:
                    sub = summary_ap[
                        (summary_ap["metric"] == metric)
                        & (summary_ap["ap_type"] == ap_type)
                        & (summary_ap["ci_method"] == ci_method)
                        & (summary_ap["class"].isin(class_list))
                    ].copy()
                    if sub.empty:
                        continue
                    sub = sub.rename(columns={"ci_method": "method"})

                    ci_lo_pct = (sub["ci_low"].to_numpy() * 100.0).tolist()
                    ci_hi_pct = (sub["ci_high"].to_numpy() * 100.0).tolist()
                    zoom_ylim = bars.compute_ci_zoom_ylim(
                        ci_lo_pct, ci_hi_pct,
                        padding_frac=_zoom_padding_frac,
                        snap_step=_zoom_snap_step,
                    )

                    bar_kw = _bar_kwargs_for(scenario_name)
                    if multi_class:
                        fig = bars.bar_per_class(
                            sub,
                            classes=class_list,
                            classifier_order=scenario_order,
                            metric=metric,
                            palette=scenario_cluster_palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            **bar_kw,
                        )
                    else:
                        fig = bars.bar_classifier(
                            sub, target_class=class_list[0], metric=metric,
                            classifier_order=scenario_order,
                            palette=scenario_palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            **bar_kw,
                        )
                    theme.save_figure(fig, bar_dir / f"{ap_type}_{ci_method}")

                    if multi_class:
                        fig_z = bars.bar_per_class(
                            sub,
                            classes=class_list,
                            classifier_order=scenario_order,
                            metric=metric,
                            palette=scenario_cluster_palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            ylim=zoom_ylim,
                            **bar_kw,
                        )
                    else:
                        fig_z = bars.bar_classifier(
                            sub, target_class=class_list[0], metric=metric,
                            classifier_order=scenario_order,
                            palette=scenario_palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            ylim=zoom_ylim,
                            **bar_kw,
                        )
                    theme.save_figure(
                        fig_z, bar_dir / f"{ap_type}_{ci_method}_zoomed"
                    )

                    # Delta forest plot — one per (ap_type, ci_method).
                    if not summary_delta.empty:
                        summary_delta_view = (
                            summary_delta.rename(columns={"ci_method": "method"})
                            if "method" not in summary_delta.columns
                            else summary_delta
                        )
                        delta_title = f"Δ {scenario_title}"
                        pair_order = _delta_pair_order(
                            summary_delta_view, classifier_rank=scenario_order,
                        )
                        delta_kw = _delta_kwargs()
                        fig = delta_plots.plot_delta_forest(
                            long_delta,
                            summary_delta_view,
                            classes=class_list,
                            metric=metric,
                            ap_type=ap_type,
                            ci_method=ci_method,
                            title=delta_title,
                            xtick_overrides=xtick_overrides,
                            grouped=multi_class,
                            pair_order=pair_order,
                            uniform_color=uniform_delta_color,
                            **delta_kw,
                        )
                        theme.save_figure(fig, dlt_dir / f"{ap_type}_{ci_method}")

            # P-value heatmap — one per ap_type per metric.
            for ap_type in ap_types_present:
                if pvalues.empty or summary_delta.empty:
                    continue
                fig = delta_plots.plot_pvalue_heatmap(
                    pvalues,
                    classes=class_list,
                    classifiers=scenario_order,
                    metric=metric,
                    ap_type=ap_type,
                    title=f"p-values — {scenario_title}",
                    xtick_overrides=xtick_overrides,
                )
                theme.save_figure(fig, pv_dir / f"{ap_type}")

    # ------------------------------------------------------------------
    # Headline-subset extras — narrow the panel to a hand-picked set of
    # methods (default ``PLM_Domains + Foldseek + BLAST`` for
    # all_methods_comparison, disabled everywhere else). Configuration:
    #
    #   visualize:
    #     headline:
    #       enabled: true               # false -> skip this block entirely
    #       methods: [PLM_Domains, Foldseek, BLAST]
    #       bar_plots: {...}            # optional overrides
    #       delta_plots: {...}
    # ------------------------------------------------------------------
    headline_enabled = bool(headline_cfg.get("enabled", True))
    if headline_cfg:
        headline_methods = list(headline_cfg.get("methods")
                                or ("PLM_Domains", "Foldseek", "BLAST"))
    else:
        headline_methods = ["PLM_Domains", "Foldseek", "BLAST"]
    available = set(summary_ap["classifier"].unique())
    headline_subset = (
        list(headline_methods)
        if headline_enabled and all(c in available for c in headline_methods)
        else None
    )
    headline_bar_overrides = dict(headline_cfg.get("bar_plots") or {})
    headline_delta_overrides = dict(headline_cfg.get("delta_plots") or {})
    if headline_subset is not None:
        for scenario_name, class_list, multi_class, scenario_title in scenarios:
            if multi_class:
                continue  # only single-class headline scenarios
            for metric in metrics_list:
                extras_ap_dir = plots_root / scenario_name / metric / "bars"
                extras_dlt_dir = plots_root / scenario_name / metric / "delta"
                extras_ap_dir.mkdir(parents=True, exist_ok=True)
                extras_dlt_dir.mkdir(parents=True, exist_ok=True)
                for ap_type in ap_types_present:
                    for ci_method in ci_methods_present:
                        sub = summary_ap[
                            (summary_ap["metric"] == metric)
                            & (summary_ap["ap_type"] == ap_type)
                            & (summary_ap["ci_method"] == ci_method)
                            & (summary_ap["class"].isin(class_list))
                            & (summary_ap["classifier"].isin(headline_subset))
                        ].copy()
                        if sub.empty:
                            continue
                        sub = sub.rename(columns={"ci_method": "method"})
                        # Sort the three subset classifiers by performance,
                        # pinning EnzymeExplorer (PLM_Domains*) to the right.
                        perf = (
                            sub.groupby("classifier")["point"].mean()
                               .sort_values(ascending=True).index.tolist()
                        )
                        ee_method = headline_subset[0]
                        if ee_method in perf:
                            perf = [c for c in perf if c != ee_method] + [ee_method]
                        ci_lo_pct = (sub["ci_low"].to_numpy() * 100.0).tolist()
                        ci_hi_pct = (sub["ci_high"].to_numpy() * 100.0).tolist()
                        zoom_ylim = bars.compute_ci_zoom_ylim(
                        ci_lo_pct, ci_hi_pct,
                        padding_frac=_zoom_padding_frac,
                        snap_step=_zoom_snap_step,
                    )
                        headline_bar_kw = _bar_kwargs_for(
                            scenario_name,
                            overrides=dict(headline_bar_overrides.get(scenario_name) or {}),
                        )
                        fig = bars.bar_classifier(
                            sub, target_class=class_list[0], metric=metric,
                            classifier_order=perf, palette=palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            **headline_bar_kw,
                        )
                        theme.save_figure(
                            fig, extras_ap_dir / f"{ap_type}_{ci_method}_headline",
                        )
                        fig_z = bars.bar_classifier(
                            sub, target_class=class_list[0], metric=metric,
                            classifier_order=perf, palette=palette,
                            xtick_overrides=xtick_overrides,
                            title=scenario_title,
                            ylim=zoom_ylim,
                            **headline_bar_kw,
                        )
                        theme.save_figure(
                            fig_z,
                            extras_ap_dir / f"{ap_type}_{ci_method}_headline_zoomed",
                        )

                        # Headline-subset delta forest — pairs restricted to
                        # the three subset methods, ordered by ``perf``.
                        if not summary_delta.empty:
                            sd = (
                                summary_delta.rename(columns={"ci_method": "method"})
                                if "method" not in summary_delta.columns
                                else summary_delta
                            )
                            sd_sub = sd[
                                sd["classifier_a"].isin(headline_subset)
                                & sd["classifier_b"].isin(headline_subset)
                            ].reset_index(drop=True)
                            ld_sub = long_delta[
                                long_delta["classifier_a"].isin(headline_subset)
                                & long_delta["classifier_b"].isin(headline_subset)
                            ].reset_index(drop=True)
                            if not sd_sub.empty:
                                pair_order = _delta_pair_order(
                                    sd_sub, classifier_rank=perf,
                                )
                                delta_title = f"Δ {scenario_title}"
                                headline_delta_kw = _delta_kwargs(
                                    overrides=headline_delta_overrides,
                                )
                                fig_d = delta_plots.plot_delta_forest(
                                    ld_sub, sd_sub,
                                    classes=class_list,
                                    metric=metric,
                                    ap_type=ap_type,
                                    ci_method=ci_method,
                                    title=delta_title,
                                    xtick_overrides=xtick_overrides,
                                    grouped=False,
                                    pair_order=pair_order,
                                    uniform_color=uniform_delta_color,
                                    **headline_delta_kw,
                                )
                                theme.save_figure(
                                    fig_d,
                                    extras_dlt_dir / f"{ap_type}_{ci_method}_headline",
                                )

    logger.info("Wrote v4 scenario plots to %s", plots_root)


def _resolve_visualize_cfg(eval_dir: Path, args: argparse.Namespace) -> dict:
    """Merge the YAML ``visualize:`` block with CLI args. CLI overrides YAML.

    When ``--config`` is passed on the CLI, the ``visualize:`` block is
    re-read from that source YAML instead of the ``eval_config.yaml``
    snapshot saved during ``evaluate`` — this lets users iterate on
    styling (bar widths, headline, etc.) without re-running bootstrap.
    Non-visualize sections (classifier definitions, bootstrap params)
    are always taken from the snapshot so cached draws stay valid.
    """
    cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    vcfg = dict(cfg.get("visualize", {}) or {})
    if getattr(args, "config", None):
        src = yaml.safe_load(Path(args.config).read_text())
        src_vcfg = dict((src or {}).get("visualize", {}) or {})
        if src_vcfg:
            vcfg = src_vcfg
            logger.info("Using visualize block from --config %s (bootstrap "
                        "artefacts + classifier definitions still come from "
                        "the eval_config.yaml snapshot).", args.config)
    if args.plots is not None:
        vcfg["plots"] = list(args.plots)
    if args.metric is not None:
        vcfg["metric"] = args.metric
    if getattr(args, "metrics", None):
        vcfg["metrics"] = list(args.metrics)
    if args.classifier_order:
        vcfg["classifier_order"] = list(args.classifier_order)
    if args.pin_last:
        vcfg["pin_last"] = args.pin_last
    if args.curve_classes:
        vcfg["curve_classes"] = list(args.curve_classes)
    if args.category_classes:
        vcfg["category_classes"] = list(args.category_classes)
    if args.pr_per_class_classifier:
        vcfg["pr_per_class_classifier"] = args.pr_per_class_classifier
    if args.disable_auto_zoom:
        vcfg.setdefault("auto_zoom", {})["enabled"] = False
    if args.zoom_delta_pct is not None:
        vcfg.setdefault("auto_zoom", {})["delta_pct"] = args.zoom_delta_pct
    if args.zoom_classifiers is not None:
        vcfg["zoom_classifiers"] = list(args.zoom_classifiers)
    if args.threshold_sweep_ylim is not None:
        vcfg["threshold_sweep_ylim"] = list(args.threshold_sweep_ylim)
    if args.tps_ylim is not None:
        vcfg["tps_ylim"] = list(args.tps_ylim)
    if args.substrate_map_ylim is not None:
        vcfg["substrate_map_ylim"] = list(args.substrate_map_ylim)
    if getattr(args, "poster", False):
        vcfg["poster"] = True
    return vcfg


def run_calibrate(args: argparse.Namespace) -> None:
    """Dedicated calibration entry point.

    Resolves classifier versions, loads fold prediction pickles, then
    fits the per-(classifier, class) calibrators via
    :func:`_run_calibration_evaluate`. Writes to the same output tree as
    ``evaluate`` (``outputs/evaluation_results/<output_name>/``) so
    ``visualize --eval-output-name <name>`` renders the calibration
    plots from these artefacts unchanged.
    """
    cfg = _load_yaml_with_includes(Path(args.config))
    if not cfg.get("calibration"):
        raise ValueError(
            f"Config {args.config} has no ``calibration:`` block. "
            "Use ``evaluate`` for bootstrap-only configs."
        )
    out_dir = get_evaluations_output() / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with _timed(
        f"resolve classifier specs + load fold pickles "
        f"({len(cfg['classifiers'])} classifiers)"
    ):
        classifier_to_dfs, _, resolved = _build_classifier_dfs(cfg)

    with open(out_dir / "resolved_versions.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    with open(out_dir / "eval_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    with _timed(
        "calibration fitting (LOFO family selection + cluster bootstrap ribbon)"
    ):
        _run_calibration_evaluate(cfg, out_dir, classifier_to_dfs)

    # Publish the fresh fit_summary to ``data/calibration_fit_summary.csv``
    # — the default path the prediction pipeline reads (see
    # ``prediction/pipeline.DEFAULT_CALIBRATION_CSV``). This is what makes
    # ``calibrate`` a one-shot: after it succeeds the deployed prediction
    # entry points pick up the new calibrators without any manual copy.
    # ``--no-publish`` opts out — sanity re-runs write a side-by-side
    # ``_rerun_*`` output dir and MUST NOT clobber the canonical deploy
    # table with numbers that are only meant for diff-vs-canonical.
    fit_summary_src = out_dir / "calibration" / "fit_summary.csv"
    if getattr(args, "no_publish", False):
        logger.info(
            "--no-publish set — leaving data/calibration_fit_summary.csv "
            "untouched (fit_summary still saved under %s).",
            fit_summary_src,
        )
    elif fit_summary_src.exists():
        from enzymeexplorer.src.utils.project_info import get_data_root
        published = get_data_root() / "calibration_fit_summary.csv"
        published.parent.mkdir(parents=True, exist_ok=True)
        published.write_bytes(fit_summary_src.read_bytes())
        logger.info(
            "Published fit_summary to %s (%d bytes) — prediction pipeline "
            "will now use these calibrators by default.",
            published, published.stat().st_size,
        )
    else:
        logger.warning(
            "No fit_summary.csv at %s — nothing to publish to data/.",
            fit_summary_src,
        )


def run_visualize(args: argparse.Namespace) -> None:
    eval_dir = get_evaluations_output() / args.eval_output_name
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")

    # v4 artefact set.
    summary_ap = pd.read_csv(eval_dir / "summary_ap.csv") if (eval_dir / "summary_ap.csv").exists() else pd.DataFrame()
    summary_delta = pd.read_csv(eval_dir / "summary_delta.csv") if (eval_dir / "summary_delta.csv").exists() else pd.DataFrame()
    pvalues = pd.read_csv(eval_dir / "pvalues.csv") if (eval_dir / "pvalues.csv").exists() else pd.DataFrame()
    long_ap_path = eval_dir / "bootstrap_long_ap.csv"
    long_delta_path = eval_dir / "bootstrap_long_delta.csv"
    long_ap = pd.read_csv(long_ap_path) if long_ap_path.exists() else pd.DataFrame()
    long_delta = pd.read_csv(long_delta_path) if long_delta_path.exists() else pd.DataFrame()

    # Downstream plotting expects a ``summary`` DataFrame with
    # ``classifier, class, metric, point, ci_low, ci_high`` columns.
    # Derive it from summary_ap by pinning (ap_type, ci_method) to the
    # canonical headline statistic (fold_mean, normal).
    if summary_ap.empty:
        summary = pd.DataFrame()
    else:
        summary = summary_ap[
            (summary_ap["ap_type"] == "fold_mean")
            & (summary_ap["ci_method"] == "normal")
        ].copy()

    bootstrap_long = None
    poster = bool(getattr(args, "poster", False))
    plots_subdir = "plots_poster" if poster else "plots"
    plots_dir = eval_dir / plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)
    theme.apply_theme(poster=poster)

    vcfg = _resolve_visualize_cfg(eval_dir, args)
    plot_set = set(vcfg.get("plots") or PLOTS_AVAILABLE)

    # ====================================================================
    # v4: 14 plots per scenario in subfolders
    #   plots/<scenario>/ap/         — 6 (ap_type × ci_method) bar plots
    #   plots/<scenario>/delta/      — 6 delta forest plots
    #   plots/<scenario>/pvalues/    — 2 (ap_type) p-value heatmaps
    # Only renders when the YAML ``plots:`` opt in to a v4 scenario; the
    # calibration config, for example, asks only for ``calibration`` and
    # must not be cluttered with the headline scenario subfolders.
    # ====================================================================
    v4_scenarios_in_plot_set = plot_set & {
        "tps_detection", "substrate_map", "substrate_per_class",
    }
    if (
        v4_scenarios_in_plot_set
        and not summary_ap.empty
        and not long_ap.empty
    ):
        # Feed the RESOLVED visualize block into the renderer so
        # ``--config`` overrides propagate; other cfg sections come
        # from the snapshot (unchanged by --config on purpose).
        _cfg_for_render = yaml.safe_load(
            (eval_dir / "eval_config.yaml").read_text()
        )
        _cfg_for_render["visualize"] = dict(vcfg)
        _render_v4_scenarios(
            eval_dir=eval_dir,
            summary_ap=summary_ap,
            summary_delta=summary_delta,
            long_ap=long_ap,
            long_delta=long_delta,
            pvalues=pvalues,
            cfg=_cfg_for_render,
            plot_set=v4_scenarios_in_plot_set,
            plots_subdir=plots_subdir,
        )
    metric = vcfg.get("metric", "ap")
    pin_last = vcfg.get("pin_last")
    xtick_overrides = dict(vcfg.get("ablation_xtick_overrides") or {})
    yaml_palette = dict(vcfg.get("palette") or {}) or None
    fixed_order = bool(vcfg.get("fixed_classifier_order")) or bool(vcfg.get("classifier_order"))
    # NCB curve/palette knobs (see visualize schema in module docstring).
    palette_mode = vcfg.get("palette_mode", "auto")
    curve_figsize = vcfg.get("curve_figsize")
    curve_figsize = tuple(curve_figsize) if curve_figsize else None

    # Per-curve-type overrides: figsize + linewidth. Keys:
    #   pr_curves, roc_curves, pr_per_class, pr_substrate, roc_substrate
    # Fallback: legacy ``curve_figsize`` (top-level scalar) applies to any
    # curve without an explicit figsize; linewidth default is 1.6.
    curve_plot_cfg = dict(vcfg.get("curves") or {})

    def _curve_kwargs(name: str, *, overrides: dict | None = None) -> dict:
        cfg_here = dict(curve_plot_cfg.get(name) or {})
        if overrides:
            cfg_here.update(overrides)
        fs = cfg_here.get("figsize") or curve_figsize
        out: dict = {}
        if fs:
            out["figsize"] = tuple(fs)
        if "linewidth" in cfg_here:
            out["linewidth"] = float(cfg_here["linewidth"])
        if "title_fontsize" in cfg_here:
            out["title_fontsize"] = float(cfg_here["title_fontsize"])
        return out

    # Per-category-plot overrides. Keys:
    #   category_plots:
    #     boxplot: {figsize: [W,H], box_width: 0.66, linewidth: 0.5,
    #               edge_linewidth: 0.45, flier_size: 1.5, showfliers: true}
    #     heatmap: {figsize: [W,H]}
    category_plot_cfg = dict(vcfg.get("category_plots") or {})

    def _cat_kwargs(kind: str) -> dict:
        cfg_here = dict(category_plot_cfg.get(kind) or {})
        out: dict = {}
        fs = cfg_here.get("figsize")
        if fs:
            out["figsize"] = tuple(fs)
        if kind == "boxplot":
            for k in ("box_width", "linewidth", "edge_linewidth", "flier_size",
                      "xtick_fontsize", "ytick_fontsize", "title_fontsize",
                      "intra_box_gap"):
                if k in cfg_here:
                    out[k] = float(cfg_here[k])
            if "showfliers" in cfg_here:
                out["showfliers"] = bool(cfg_here["showfliers"])
        elif kind == "heatmap":
            if "title_fontsize" in cfg_here:
                out["title_fontsize"] = float(cfg_here["title_fontsize"])
        return out

    # Headline block (defined once for the whole visualize call; reused
    # by both bars/deltas in _render_v4_scenarios and curves below).
    _headline_cfg = dict(vcfg.get("headline") or {})
    _headline_enabled = bool(_headline_cfg.get("enabled", True))
    _headline_methods = list(_headline_cfg.get("methods")
                             or ("PLM_Domains", "Foldseek", "BLAST"))
    _headline_curve_overrides = dict(_headline_cfg.get("curves") or {})

    eval_cfg_for_order = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    classifier_order = _apply_pin_last(
        _resolve_classifier_order(args, summary, vcfg, cfg=eval_cfg_for_order),
        pin_last,
    )

    needs_pooled = bool(
        plot_set & {"pr_curves", "roc_curves", "pr_per_class",
                    "pr_substrate", "roc_substrate"}
    )
    pooled = None
    if needs_pooled:
        classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
        pooled = curves.pool_fold_dfs(classifier_to_dfs)
        curves_dir = plots_dir / "curves"
        curves_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the palette actually used for curve panels. YAML wins
        # if the user pinned specific keys; otherwise a per-mode default
        # (NCB all-methods: class colours with grey shades for the
        # "everything else" bucket; NCB ablation: Blues ramp across the
        # ordered classifier list; legacy auto: model-family palette).
        curve_palette: dict[str, str] | None
        if palette_mode == "all_methods":
            curve_palette = theme.ncb_all_methods_curve_palette(classifier_order)
        elif palette_mode == "ablation":
            # Same rule as the ablation bars: lightest green on the left,
            # darkest green (== NCB_GREEN) on the right where the pinned
            # "final" model sits.
            shades = theme.ncb_curve_shades(classifier_order, hue="green")
            curve_palette = {c: h for c, h in zip(classifier_order, shades)}
        else:
            curve_palette = yaml_palette
        if yaml_palette and curve_palette is not None:
            curve_palette = {**curve_palette, **yaml_palette}

        if "pr_curves" in plot_set:
            for tgt in vcfg.get("curve_classes", ["TPS"]):
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_pr_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    palette=curve_palette,
                    title=f"{tgt} — Precision-Recall curve",
                    **_curve_kwargs("pr_curves"),
                )
                theme.save_figure(fig, curves_dir / f"pr_{tgt}")
        if "roc_curves" in plot_set:
            for tgt in vcfg.get("curve_classes", ["TPS"]):
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_roc_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    palette=curve_palette,
                    title=f"{tgt} — ROC curve",
                    **_curve_kwargs("roc_curves"),
                )
                theme.save_figure(fig, curves_dir / f"roc_{tgt}")
        if "pr_per_class" in plot_set:
            tgt_clf = vcfg.get("pr_per_class_classifier", "PLM_Domains")
            if tgt_clf and tgt_clf in pooled:
                fig = curves.plot_per_class_pr_curves(
                    pooled[tgt_clf],
                    classes=DEFAULT_PLOT_ORDER,
                    title=f"{theme.display_name(tgt_clf)} — PR curves per substrate",
                    **_curve_kwargs("pr_per_class"),
                )
                theme.save_figure(fig, curves_dir / f"pr_per_class_{tgt_clf}")
        if "pr_substrate" in plot_set:
            eligible = [
                c for c in classifier_order
                if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
            ]
            if eligible:
                fig = curves.plot_macro_pr_curves(
                    pooled, classes=SUBSTRATE_CLASSES,
                    classifier_order=eligible,
                    palette=curve_palette,
                    title="Substrate prediction — PR curve",
                    **_curve_kwargs("pr_substrate"),
                )
                theme.save_figure(fig, curves_dir / "pr_substrate")
        if "roc_substrate" in plot_set:
            eligible = [
                c for c in classifier_order
                if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
            ]
            if eligible:
                fig = curves.plot_macro_roc_curves(
                    pooled, classes=SUBSTRATE_CLASSES,
                    classifier_order=eligible,
                    palette=curve_palette,
                    title="Substrate prediction — ROC curve",
                    **_curve_kwargs("roc_substrate"),
                )
                theme.save_figure(fig, curves_dir / "roc_substrate")

        # --------------------------------------------------------------
        # Headline-restricted curve companions. Same layout as the full
        # panels but the ``classifier_order`` is filtered to
        # ``headline.methods``. Per-curve overrides come from
        # ``headline.curves.<name>`` and win over ``curves.<name>``.
        # Files land next to the full panels with a ``_headline`` suffix.
        # --------------------------------------------------------------
        headline_curve_methods = [
            m for m in _headline_methods if m in classifier_order
        ]
        if (
            _headline_enabled
            and len(headline_curve_methods) >= 2
            and pooled is not None
        ):
            def _hl_curve_kw(name: str) -> dict:
                return _curve_kwargs(
                    name, overrides=_headline_curve_overrides.get(name) or {},
                )

            hl_classifier_order = [
                c for c in classifier_order if c in headline_curve_methods
            ]
            if palette_mode == "all_methods":
                hl_palette = theme.ncb_all_methods_curve_palette(hl_classifier_order)
            elif palette_mode == "ablation":
                shades = theme.ncb_curve_shades(hl_classifier_order, hue="green")
                hl_palette = {c: h for c, h in zip(hl_classifier_order, shades)}
            else:
                hl_palette = curve_palette

            if "pr_curves" in plot_set:
                for tgt in vcfg.get("curve_classes", ["TPS"]):
                    eligible = [c for c in hl_classifier_order
                                if tgt in pooled.get(c, {})]
                    if len(eligible) < 2:
                        continue
                    fig = curves.plot_pr_curves(
                        pooled, target_class=tgt, classifier_order=eligible,
                        palette=hl_palette,
                        title=f"{tgt} — Precision-Recall curve",
                        **_hl_curve_kw("pr_curves"),
                    )
                    theme.save_figure(fig, curves_dir / f"pr_{tgt}_headline")
            if "roc_curves" in plot_set:
                for tgt in vcfg.get("curve_classes", ["TPS"]):
                    eligible = [c for c in hl_classifier_order
                                if tgt in pooled.get(c, {})]
                    if len(eligible) < 2:
                        continue
                    fig = curves.plot_roc_curves(
                        pooled, target_class=tgt, classifier_order=eligible,
                        palette=hl_palette,
                        title=f"{tgt} — ROC curve",
                        **_hl_curve_kw("roc_curves"),
                    )
                    theme.save_figure(fig, curves_dir / f"roc_{tgt}_headline")
            if "pr_substrate" in plot_set:
                eligible = [
                    c for c in hl_classifier_order
                    if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
                ]
                if len(eligible) >= 2:
                    fig = curves.plot_macro_pr_curves(
                        pooled, classes=SUBSTRATE_CLASSES,
                        classifier_order=eligible,
                        palette=hl_palette,
                        title="Substrate prediction — PR curve",
                        **_hl_curve_kw("pr_substrate"),
                    )
                    theme.save_figure(fig, curves_dir / "pr_substrate_headline")
            if "roc_substrate" in plot_set:
                eligible = [
                    c for c in hl_classifier_order
                    if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
                ]
                if len(eligible) >= 2:
                    fig = curves.plot_macro_roc_curves(
                        pooled, classes=SUBSTRATE_CLASSES,
                        classifier_order=eligible,
                        palette=hl_palette,
                        title="Substrate prediction — ROC curve",
                        **_hl_curve_kw("roc_substrate"),
                    )
                    theme.save_figure(fig, curves_dir / "roc_substrate_headline")

    # ------------------------------------------------------------------
    # v4 categorical plots (Kingdom + TPS_Type) → plots/categories/<name>/
    # If the categorical artefacts are missing (older evaluate runs) we
    # compute them on the fly here, persisting alongside the headline
    # bootstrap artefacts so subsequent visualize calls are cache hits.
    # ------------------------------------------------------------------
    cat_long_path = eval_dir / "bootstrap_long_categorical_ap.csv"
    cat_point_path = eval_dir / "point_estimates_categorical_ap.csv"
    eval_cfg_full = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    categories_cfg = eval_cfg_full.get("categories", {}) or {}
    if (
        categories_cfg
        and (plot_set & {"category_boxplot", "category_heatmap"})
        and not cat_long_path.exists()
    ):
        from enzymeexplorer.src.evaluation import categorical_bootstrap as cat_bs
        bcfg = eval_cfg_full.get("bootstrap", {}) or {}
        n_boot_v = int(bcfg.get("n_bootstraps", 1000))
        seed_v = int(bcfg.get("seed", 42))
        ap_types_v = tuple(bcfg.get("ap_types", ["pooled_oof", "fold_mean"]))
        metrics_v = tuple(bcfg.get("metrics", ["ap"]))

        type_groupings: dict[str, dict[str, str]] = {}
        masked_categories: dict[str, dict[str, str]] = {}
        negative_labels: dict[str, str] = {}
        for cat_name, cat_spec in categories_cfg.items():
            negative_labels[cat_name] = cat_spec.get("negative_label", "Unknown")
            if cat_spec.get("type_grouping"):
                from enzymeexplorer.src.evaluation.classes import SUBSTRATE_TO_TYPE
                type_groupings[cat_name] = dict(
                    cat_spec.get("class_to_type") or SUBSTRATE_TO_TYPE
                )
            else:
                meta_csv = eval_cfg_full.get("id_metadata", {}).get("csv")
                if not meta_csv:
                    logger.warning(
                        "Skipping masked category %s: id_metadata.csv missing",
                        cat_name,
                    )
                    continue
                kingdom_cache = eval_cfg_full.get("id_metadata", {}).get(
                    "kingdom_cache", "data/uniprot_kingdom_cache.json"
                )
                meta = eio.load_id_metadata(
                    meta_csv, [cat_spec["column"]],
                    kingdom_cache=kingdom_cache,
                )
                masked_categories[cat_name] = meta[cat_spec["column"]]

        long_pieces: list[pd.DataFrame] = []
        point_pieces: list[pd.DataFrame] = []
        if type_groupings and not long_ap.empty:
            point_ap_csv = eval_dir / "point_estimates_ap.csv"
            point_ap_df = (
                pd.read_csv(point_ap_csv) if point_ap_csv.exists() else pd.DataFrame()
            )
            l, p = cat_bs.compute_type_aggregated_ap(
                long_ap, point_ap_df, type_groupings=type_groupings,
            )
            if not l.empty:
                long_pieces.append(l)
            if not p.empty:
                point_pieces.append(p)
        if masked_categories:
            classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
            l, p = cat_bs.compute_masked_ap_from_seed(
                classifier_to_dfs,
                masked_categories=masked_categories,
                metrics=metrics_v,
                ap_types=ap_types_v,
                n_bootstraps=n_boot_v,
                seed=seed_v,
                negative_labels=negative_labels,
            )
            l, p = cat_bs.add_substrate_map_aggregate(
                l, p, substrate_classes=list(SUBSTRATE_CLASSES),
            )
            if not l.empty:
                long_pieces.append(l)
            if not p.empty:
                point_pieces.append(p)

        if long_pieces:
            pd.concat(long_pieces, ignore_index=True).to_csv(cat_long_path, index=False)
        if point_pieces:
            pd.concat(point_pieces, ignore_index=True).to_csv(cat_point_path, index=False)

    if (
        categories_cfg
        and (plot_set & {"category_boxplot", "category_heatmap"})
        and cat_long_path.exists()
    ):
        cat_pretty_map = {"TPS_Type": "TPS type", "Kingdom": "kingdom"}
        target_label_map = {
            "TPS": "TPS detection AP",
            "Substrate_mAP": "Substrate prediction mAP",
            "_type_grouped": "Substrate prediction AP",
        }
        cat_long_full = pd.read_csv(cat_long_path)
        cat_point_full = pd.read_csv(cat_point_path) if cat_point_path.exists() else pd.DataFrame()
        masked_targets = vcfg.get(
            "category_classes", ["TPS", "Substrate_mAP"],
        )
        for cat_label, cat_spec in categories_cfg.items():
            # Type-grouped categories (TPS_Type) only emit one target class:
            # the synthetic ``_type_grouped`` aggregate. Masked categories
            # (Kingdom, ...) emit one plot per requested headline class.
            if cat_spec.get("type_grouping"):
                cat_classes = ["_type_grouped"]
            else:
                cat_classes = list(masked_targets)
            cat_pretty = cat_pretty_map.get(cat_label, cat_label.replace("_", " "))
            cat_dir = plots_dir / "categories" / cat_label
            cat_dir.mkdir(parents=True, exist_ok=True)
            cat_long_view = cat_long_full[
                cat_long_full["category_name"] == cat_label
            ].dropna(subset=["value"])
            cat_point_view = (
                cat_point_full[cat_point_full["category_name"] == cat_label]
                if not cat_point_full.empty else pd.DataFrame()
            )
            if cat_long_view.empty:
                continue
            order_cat = list(cat_spec.get("order") or [])
            negative_label = cat_spec.get("negative_label", "Unknown")
            available_cats = [
                c for c in cat_long_view["category"].unique()
                if c != negative_label
            ]
            if order_cat:
                ordered_cats = [c for c in order_cat if c in available_cats]
                ordered_cats.extend(c for c in available_cats if c not in order_cat)
            else:
                ordered_cats = sorted(available_cats)

            for tgt in cat_classes:
                view = cat_long_view[cat_long_view["class"] == tgt]
                if view.empty:
                    continue
                target_pretty = target_label_map.get(tgt, tgt.replace("_", " "))
                if "category_boxplot" in plot_set:
                    for ap_type in sorted(view["ap_type"].unique()):
                        sub = view[view["ap_type"] == ap_type]
                        if sub.empty:
                            continue
                        fig = categorical.plot_category_boxplot(
                            sub, target_class=tgt, metric=metric,
                            categories=ordered_cats,
                            classifier_subset=classifier_order,
                            pin_last=pin_last,
                            title=f"{target_pretty} per {cat_pretty}",
                            xlabel="",
                            **_cat_kwargs("boxplot"),
                        )
                        theme.save_figure(
                            fig,
                            cat_dir / f"boxplot_{tgt.lstrip('_')}_{ap_type}",
                        )
                if "category_heatmap" in plot_set and not cat_point_view.empty:
                    pt_view = cat_point_view[cat_point_view["class"] == tgt]
                    for ap_type in sorted(pt_view["ap_type"].unique()):
                        cell = (
                            pt_view[pt_view["ap_type"] == ap_type]
                            .rename(columns={"value": "point"})
                            .copy()
                        )
                        if cell.empty:
                            continue
                        fig = categorical.plot_category_heatmap(
                            cell, target_class=tgt, metric=metric,
                            categories=ordered_cats,
                            classifier_subset=classifier_order,
                            pin_last=pin_last,
                            title=f"{target_pretty} per {cat_pretty}",
                            **_cat_kwargs("heatmap"),
                        )
                        theme.save_figure(
                            fig,
                            cat_dir / f"heatmap_{tgt.lstrip('_')}_{ap_type}",
                        )

    if "calibration" in plot_set:
        cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
        cal_cfg = cfg.get("calibration", {}) or {}
        target_classes = list(cal_cfg.get("classes", DEFAULT_PLOT_ORDER))

        cal_dir = eval_dir / "calibration"
        fit_summary_path = cal_dir / "fit_summary.csv"
        if not fit_summary_path.exists():
            logger.warning(
                "No calibration artefacts at %s — run `evaluate` first to "
                "produce them. Skipping calibration plots.",
                cal_dir,
            )
        else:
            fit_summary = pd.read_csv(fit_summary_path)

            def _read_csv(name: str) -> pd.DataFrame:
                path = cal_dir / name
                return pd.read_csv(path) if path.exists() else pd.DataFrame()

            def _read_parquet(name: str) -> pd.DataFrame:
                path = cal_dir / name
                return pd.read_parquet(path) if path.exists() else pd.DataFrame()

            metrics_df = _read_csv("metrics.csv")
            reliability_df = _read_csv("reliability.csv")
            reliability_pf_df = _read_csv("reliability_per_fold.csv")
            ribbon_coverage_df = _read_csv("ribbon_coverage.csv")
            per_fold_df = _read_csv("per_fold_params.csv")
            drift_df = _read_csv("fold_drift_summary.csv")
            hard_df = _read_csv("hard_errors.csv")
            ribbon_df = _read_parquet("ribbon.parquet")

            # Iterate over what evaluate actually produced.
            target_classifiers = list(fit_summary["classifier"].unique())

            # OOF frames are still needed for score-distribution + curve rugs.
            classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
            oof_per_clf_class: dict[str, dict[str, cal.OofFrame]] = {}
            for clf in target_classifiers:
                cls_map = classifier_to_dfs.get(clf, {})
                oof_per_clf_class[clf] = {}
                sub = fit_summary[fit_summary["classifier"] == clf]
                for cls in sub["target_class"].unique():
                    if cls not in cls_map:
                        continue
                    oof_per_clf_class[clf][cls] = cal.build_oof_frame(
                        cls_map[cls], cls, clf,
                    )

            cal_plot_dir = plots_dir / "calibration"
            cal_plot_dir.mkdir(parents=True, exist_ok=True)

            if not metrics_df.empty:
                try:
                    fig = cal_plots.plot_calibration_metrics_grid(
                        metrics_df,
                        class_order=target_classes,
                        classifier_order=target_classifiers,
                        title="Calibration improvement (raw − calibrated)",
                    )
                    theme.save_figure(fig, cal_plot_dir / "metrics_grid")
                except (ValueError, KeyError) as exc:
                    logger.warning("Skipping calibration metrics grid: %s", exc)

            drift_lookup: dict[tuple[str, str], dict] = {}
            if not drift_df.empty:
                for _, r in drift_df.iterrows():
                    drift_lookup[(r["classifier"], r["target_class"])] = r.to_dict()

            for clf in target_classifiers:
                cls_map = oof_per_clf_class.get(clf, {})
                for cls, oof in cls_map.items():
                    rib = ribbon_df[
                        (ribbon_df["classifier"] == clf)
                        & (ribbon_df["target_class"] == cls)
                    ] if not ribbon_df.empty else pd.DataFrame()
                    rel = reliability_df[
                        (reliability_df["classifier"] == clf)
                        & (reliability_df["target_class"] == cls)
                    ] if not reliability_df.empty else pd.DataFrame()
                    pfp = per_fold_df[
                        (per_fold_df["classifier"] == clf)
                        & (per_fold_df["target_class"] == cls)
                    ] if not per_fold_df.empty else pd.DataFrame()
                    hard = hard_df[
                        (hard_df.get("classifier") == clf)
                        & (hard_df.get("target_class") == cls)
                    ] if not hard_df.empty else pd.DataFrame()

                    stem = f"{clf}_{cls}"

                    try:
                        fig = cal_plots.plot_score_distribution(
                            oof.df, classifier=clf, target_class=cls,
                        )
                        theme.save_figure(
                            fig, cal_plot_dir / f"score_distribution_{stem}",
                        )
                    except (ValueError, KeyError) as exc:
                        logger.warning(
                            "Skipping score distribution for %s/%s: %s",
                            clf, cls, exc,
                        )

                    if not rib.empty:
                        try:
                            fig = cal_plots.plot_calibration_curve_with_ribbon(
                                rib, oof.df, classifier=clf, target_class=cls,
                            )
                            theme.save_figure(
                                fig,
                                cal_plot_dir / f"calibration_curve_{stem}",
                            )
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "Skipping calibration curve for %s/%s: %s",
                                clf, cls, exc,
                            )

                    if not rel.empty:
                        try:
                            n_pos = int(oof.n_pos)
                            n_total = int(oof.n_total)
                            fig = cal_plots.plot_reliability_diagram(
                                rel, classifier=clf, target_class=cls,
                                n_pos=n_pos, n_total=n_total,
                            )
                            theme.save_figure(
                                fig, cal_plot_dir / f"reliability_{stem}",
                            )
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "Skipping reliability for %s/%s: %s",
                                clf, cls, exc,
                            )

                    # Curve-overlap plot: 5 per-fold LOFO curves +
                    # deployment fit + seq-id cluster bootstrap ribbon.
                    if not rib.empty or not reliability_pf_df.empty:
                        try:
                            rel_pf = reliability_pf_df[
                                (reliability_pf_df["classifier"] == clf)
                                & (reliability_pf_df["target_class"] == cls)
                            ] if not reliability_pf_df.empty else pd.DataFrame()
                            cov_row = ribbon_coverage_df[
                                (ribbon_coverage_df["classifier"] == clf)
                                & (ribbon_coverage_df["target_class"] == cls)
                            ] if not ribbon_coverage_df.empty else pd.DataFrame()
                            pct_outside = (
                                float(cov_row.iloc[0]["pct_deployment_outside_ribbon"])
                                if not cov_row.empty else None
                            )
                            fig = cal_plots.plot_curve_overlap(
                                rib, rel_pf, rel,
                                classifier=clf, target_class=cls,
                                coverage_pct_outside=pct_outside,
                            )
                            theme.save_figure(
                                fig, cal_plot_dir / f"curve_overlap_{stem}",
                            )
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "Skipping curve-overlap for %s/%s: %s",
                                clf, cls, exc,
                            )

                    if not pfp.empty:
                        drift_row = drift_lookup.get((clf, cls), {})
                        family_for_plot = (
                            str(drift_row.get("family"))
                            if drift_row and drift_row.get("family")
                            else (str(pfp["family"].iloc[0])
                                  if "family" in pfp.columns and len(pfp)
                                  else cal.FAMILY_BETA)
                        )
                        spreads = {
                            k: float(drift_row[f"spread_{k}"])
                            for k in ("a", "b", "c", "T")
                            if drift_row and pd.notna(drift_row.get(f"spread_{k}"))
                        }
                        try:
                            fig = cal_plots.plot_per_fold_params(
                                pfp, classifier=clf, target_class=cls,
                                family=family_for_plot,
                                drift_flagged=bool(drift_row.get("drift_flagged"))
                                if drift_row else False,
                                spreads=spreads,
                            )
                            theme.save_figure(
                                fig, cal_plot_dir / f"per_fold_params_{stem}",
                            )
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "Skipping per-fold params for %s/%s: %s",
                                clf, cls, exc,
                            )

                    if not hard.empty:
                        try:
                            fig = cal_plots.plot_hard_errors(
                                hard, classifier=clf, target_class=cls,
                            )
                            theme.save_figure(
                                fig, cal_plot_dir / f"hard_errors_{stem}",
                            )
                        except (ValueError, KeyError) as exc:
                            logger.warning(
                                "Skipping hard errors for %s/%s: %s",
                                clf, cls, exc,
                            )

    if "threshold_sweep" in plot_set:
        sweep_dir = eval_dir / "threshold_sweep"
        if sweep_dir.exists():
            for csv_path in sorted(sweep_dir.glob("*.csv")):
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                x_axis = df["x_axis"].iloc[0] if "x_axis" in df.columns else "categorical"
                x_label = df["x_label"].iloc[0] if "x_label" in df.columns else csv_path.stem
                if x_axis == "neglog10":
                    v2x = {
                        v: thresholds.parse_eval_neglog10(v)
                        for v in df["version"].unique()
                    }
                    v2x = {k: x for k, x in v2x.items() if x is not None}
                elif x_axis == "tail_int":
                    v2x = {}
                    for v in df["version"].unique():
                        m = re.search(r"(\d+)$", v)
                        if m:
                            v2x[v] = float(m.group(1))
                else:
                    v2x = None
                fig = thresholds.plot_threshold_sweep(
                    df, version_to_x=v2x,
                    title=f"{csv_path.stem} — AP vs threshold",
                    xlabel=x_label,
                    ylim=tuple(vcfg["threshold_sweep_ylim"])
                    if vcfg.get("threshold_sweep_ylim") else None,
                )
                theme.save_figure(fig, plots_dir / f"threshold_sweep_{csv_path.stem}")

    # Persist effective visualize options for reproducibility.
    with open(eval_dir / "visualize_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(vcfg, fh, sort_keys=False)
    logger.info("Saved plots to %s", plots_dir)


PLOTS_AVAILABLE = (
    "tps_detection",
    "substrate_map",
    "substrate_per_class",
    "pr_curves",
    "roc_curves",
    "pr_per_class",
    "pr_substrate",
    "roc_substrate",
    "category_boxplot",
    "category_heatmap",
    "threshold_sweep",
    "calibration",
)


def add_evaluate_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="Bootstrap evaluation across classifiers"
    )
    parser.set_defaults(cmd="evaluate")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-name", required=True, type=str)
    parser.add_argument(
        "--force-bootstrap", action="store_true",
        help="Rebuild the per-classifier bootstrap cache for every classifier."
    )
    parser.add_argument(
        "--force-classifiers", nargs="+", default=None,
        help="Rebuild bootstrap cache only for the listed classifier labels."
    )
    parser.add_argument(
        "--use-existing-resolved-versions", action="store_true",
        help=(
            "Reuse the per-classifier versions and experiment timestamps "
            "recorded in <output_name>/resolved_versions.yaml, bypassing "
            "version-spec re-resolution. Useful when a fresh training is "
            "in progress and the latest timestamp would otherwise be "
            "selected even though it is incomplete or experimental."
        ),
    )


def add_calibrate_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "calibrate",
        help="Fit per-(classifier, class) calibrators from OOF predictions",
    )
    parser.set_defaults(cmd="calibrate")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-name", required=True, type=str)
    parser.add_argument(
        "--no-publish", action="store_true",
        help=(
            "Skip copying fit_summary.csv to data/calibration_fit_summary.csv. "
            "Used by sanity re-runs so the canonical deploy table stays "
            "untouched."
        ),
    )


def add_visualize_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize", help="Render plots from a saved evaluation run"
    )
    parser.set_defaults(cmd="visualize")
    parser.add_argument("--eval-output-name", required=True, type=str)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Source YAML to use instead of the eval_config.yaml snapshot "
             "saved during evaluate. Useful for iterating on visualize "
             "styling (bar widths, headline, etc.) without re-running "
             "bootstrap. Only ``visualize:`` keys are re-read from this "
             "file — classifier definitions / bootstrap params still come "
             "from the snapshot so cached draws remain valid.",
    )
    parser.add_argument(
        "--plots", nargs="+", choices=PLOTS_AVAILABLE, default=None,
        help="Plot names to render. Defaults to the set in the YAML "
             "visualize block, or all available."
    )
    parser.add_argument("--metric", choices=("ap", "roc_auc", "mcc_f1"), default=None,
                        help="Legacy single-metric override; use --metrics for multi.")
    parser.add_argument("--metrics", nargs="+",
                        choices=("ap", "roc_auc", "mcc_f1"), default=None,
                        help="Metrics to render (bars + delta + p-values per metric).")
    parser.add_argument("--classifier-order", nargs="+", default=None)
    parser.add_argument(
        "--pin-last", type=str, default=None,
        help="Classifier key pinned to the rightmost position in bar/box "
             "plots after performance sort. Defaults to YAML value."
    )
    parser.add_argument("--tps-ylim", nargs=2, type=float, default=None)
    parser.add_argument("--substrate-map-ylim", nargs=2, type=float, default=None)
    parser.add_argument(
        "--curve-classes", nargs="+", default=None,
        help="Classes for PR/ROC curves (one figure per class).",
    )
    parser.add_argument(
        "--pr-per-class-classifier", type=str, default=None,
        help="Classifier whose per-class PR curves to render.",
    )
    parser.add_argument(
        "--category-classes", nargs="+", default=None,
        help="Target classes/aggregates for category plots.",
    )
    parser.add_argument(
        "--threshold-sweep-ylim", nargs=2, type=float, default=None,
    )
    parser.add_argument(
        "--disable-auto-zoom", action="store_true",
        help="Disable the auto-zoom companion bars for headline plots.",
    )
    parser.add_argument(
        "--zoom-delta-pct", type=float, default=None,
        help="Top-group window (pct points) for auto-zoom selection.",
    )
    parser.add_argument(
        "--zoom-classifiers", nargs="+", default=None,
        help="Force the auto-zoom subset to exactly these classifiers.",
    )
    parser.add_argument(
        "--poster", action="store_true",
        help="Render plots with a poster-friendly theme (larger fonts, "
             "thicker axes/lines, bigger markers) and write them to "
             "<eval_dir>/plots_poster/ instead of <eval_dir>/plots/. "
             "Designed for ~1 m+ posters viewed from ~1–2 m. The paper-"
             "quality plots in <eval_dir>/plots/ are left untouched.",
    )
