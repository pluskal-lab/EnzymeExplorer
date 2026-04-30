"""CLI handlers for the bootstrap-driven evaluation pipeline.

``run_evaluate`` resolves classifier version specs (fixed or per-class auto),
runs bootstrap, derives aggregate classes (Substrate-mAP / TPS-IDS-mAP /
Overall-mAP) and writes long-form draws + a CI summary into
``outputs/evaluation_results/<output-name>/``.

When the YAML config declares ``categories`` (e.g. Kingdom, OriginalType) and
``id_metadata.csv`` (path to the dataset CSV), categorical bootstrap is also
run and persisted under ``categories/<name>/``. When it declares
``threshold_sweep``, per-class AP sweeps across candidate versions are
computed and saved under ``threshold_sweep/<label>.csv`` for later
``visualize`` rendering.

``run_visualize`` reads the saved artefacts and renders the requested plots.

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

    bootstrap:
      n_bootstraps: 1000
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
      - label: PFAM
        model: PfamSUPFAM
        prefix: pfam_bitscore
        x_axis: tail_int
        x_label: "Bitscore"
        classes: [TPS]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd  # type: ignore
import yaml  # type: ignore

from enzymeexplorer.src.evaluation import (
    aggregate as agg,
    bootstrap as bs,
    confidence_tiers as ct,
    io as eio,
    prediction_thresholds as pt,
    selection as sel,
)
from enzymeexplorer.src.evaluation.classes import (
    DEFAULT_PLOT_ORDER,
    SUBSTRATE_CLASSES,
)
from enzymeexplorer.src.evaluation.plotting import (
    bars,
    categorical,
    confidence_tiers as ct_plots,
    curves,
    prediction_thresholds as pt_plots,
    theme,
    thresholds,
)
from enzymeexplorer.src.utils.project_info import get_evaluations_output

logger = logging.getLogger(__name__)


def _resolve_version_spec(label: str, spec: dict, classes: list[str]):
    sel_cfg = spec["selection"]
    mode = sel_cfg["mode"]
    if mode == "fixed":
        return sel_cfg["version"]
    if mode == "auto":
        prefix = sel_cfg["prefix"]
        candidates = sel.discover_versions(spec["model"], prefix=prefix)
        if not candidates:
            raise RuntimeError(
                f"No '{prefix}*' versions discovered for {label} ({spec['model']})"
            )
        return sel.pick_best_versions_per_class(
            spec["model"],
            candidates,
            classes,
            selection_metric=sel_cfg.get("metric", "ap"),
        )
    raise ValueError(f"Unknown selection mode for {label}: {mode!r}")


def _build_classifier_dfs(cfg: dict):
    classifier_to_dfs: dict[str, dict[str, dict[int, eio.FoldDfs]]] = {}
    resolved: dict[str, dict] = {}
    for label, spec in cfg["classifiers"].items():
        classes = list(spec["classes"])
        version_spec = _resolve_version_spec(label, spec, classes)
        logger.info("Resolved %s -> %s", label, version_spec)
        classifier_to_dfs[label] = eio.load_classifier_class_fold_dfs(
            spec["model"], version_spec, classes=classes
        )
        resolved[label] = {
            "model": spec["model"],
            "classes": classes,
            "resolved_versions": version_spec,
        }
    return classifier_to_dfs, resolved


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
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = get_evaluations_output() / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier_to_dfs, resolved = _build_classifier_dfs(cfg)

    with open(out_dir / "resolved_versions.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    with open(out_dir / "eval_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    bcfg = cfg.get("bootstrap", {})
    metrics = tuple(bcfg.get("metrics", ["ap", "roc_auc"]))
    n_boot = int(bcfg.get("n_bootstraps", 1000))
    seed = int(bcfg.get("seed", 42))
    mode = bcfg.get("mode", "cluster")
    ci = float(bcfg.get("ci", 0.95))
    ci_method = bcfg.get("ci_method", "percentile")

    result = bs.bootstrap_metric_cis(
        classifier_to_dfs,
        metrics=metrics,
        n_bootstraps=n_boot,
        seed=seed,
        mode=mode,
    )
    aggregates = cfg.get("aggregates")
    if aggregates:
        result = agg.add_aggregates(result, aggregates)
    result.save(out_dir)
    bs.compute_cis(result, method=ci_method, ci=ci).to_csv(
        out_dir / "summary.csv", index=False
    )
    logger.info("Saved global evaluation artefacts to %s", out_dir)

    categories_cfg = cfg.get("categories")
    if categories_cfg:
        meta_csv = cfg.get("id_metadata", {}).get("csv")
        if not meta_csv:
            raise ValueError("categories require id_metadata.csv to be set")
        wanted_cols = sorted({c["column"] for c in categories_cfg.values()})
        meta = eio.load_id_metadata(meta_csv, wanted_cols)
        for cat_label, cat_spec in categories_cfg.items():
            id_to_cat = meta[cat_spec["column"]]
            cat_dir = out_dir / "categories" / cat_label
            cat_dir.mkdir(parents=True, exist_ok=True)
            cat_result = bs.bootstrap_categorical_metric_cis(
                classifier_to_dfs,
                id_to_category=id_to_cat,
                category_name=cat_label,
                categories=list(cat_spec.get("order")) if cat_spec.get("order") else None,
                negative_label=cat_spec.get("negative_label", "Unknown"),
                metrics=metrics,
                n_bootstraps=n_boot,
                seed=seed,
                mode=mode,
            )
            if aggregates:
                cat_result = agg.add_aggregates(cat_result, aggregates)
            cat_result.save(cat_dir)
            bs.compute_cis(cat_result, method=ci_method, ci=ci).to_csv(
                cat_dir / "summary.csv", index=False
            )
            logger.info("Saved categorical artefacts (%s) to %s", cat_label, cat_dir)

    sweep_cfg = cfg.get("threshold_sweep")
    if sweep_cfg:
        sweep_dir = out_dir / "threshold_sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for entry in sweep_cfg:
            label = entry["label"]
            sweep_classes = entry.get("classes", DEFAULT_PLOT_ORDER)
            candidates = sel.discover_versions(entry["model"], prefix=entry["prefix"])
            sweep_df = thresholds.compute_threshold_sweep(
                entry["model"], candidates, sweep_classes,
                metric=entry.get("metric", "ap"),
            )
            sweep_df["x_axis"] = entry.get("x_axis", "neglog10")
            sweep_df["x_label"] = entry.get("x_label", "")
            sweep_df.to_csv(sweep_dir / f"{label}.csv", index=False)
            logger.info("Saved threshold sweep '%s' (%d rows)", label, len(sweep_df))


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
            classifier_to_dfs[label] = eio.load_classifier_class_fold_dfs(
                info["model"],
                info["resolved_versions"],
                classes=list(info["classes"]),
            )
        return classifier_to_dfs, cfg
    return _build_classifier_dfs(cfg)[0], cfg


def _resolve_classifier_order(args: argparse.Namespace, summary: pd.DataFrame) -> list[str]:
    if args.classifier_order:
        return list(args.classifier_order)
    return sorted(summary["classifier"].unique())


def run_visualize(args: argparse.Namespace) -> None:
    eval_dir = get_evaluations_output() / args.eval_output_name
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dir not found: {eval_dir}")
    summary = pd.read_csv(eval_dir / "summary.csv")
    bootstrap_long_path = eval_dir / "bootstrap_long.csv"
    bootstrap_long = (
        pd.read_csv(bootstrap_long_path) if bootstrap_long_path.exists() else None
    )
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    theme.apply_theme()

    classifier_order = _resolve_classifier_order(args, summary)
    plot_set = set(args.plots)
    metric = args.metric

    if "tps_detection" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS", metric=metric,
            classifier_subset=classifier_order,
            title="TPS detection AP",
            ylim=tuple(args.tps_ylim) if args.tps_ylim else None,
        )
        theme.save_figure(fig, plots_dir / "tps_detection")

    if "tps_detection_zoomed" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS", metric=metric,
            classifier_subset=list(args.zoomed_tps_classifiers),
            title="TPS detection AP (zoomed)",
            ylim=tuple(args.zoomed_tps_ylim) if args.zoomed_tps_ylim else None,
        )
        theme.save_figure(fig, plots_dir / "tps_detection_zoomed")

    if "substrate_map" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="Substrate_mAP", metric=metric,
            classifier_subset=classifier_order,
            title="Substrate prediction mAP",
            ylabel="mAP (%)",
            ylim=tuple(args.substrate_map_ylim) if args.substrate_map_ylim else None,
        )
        theme.save_figure(fig, plots_dir / "substrate_map")

    if "substrate_per_class" in plot_set:
        sub = bootstrap_long[
            (bootstrap_long["metric"] == metric)
            & (bootstrap_long["class"].isin(SUBSTRATE_CLASSES))
            & (bootstrap_long["classifier"].isin(classifier_order))
        ].dropna(subset=["value"])
        if not sub.empty:
            ordered_for_sub = (
                sub.groupby("classifier")["value"].mean().sort_values().index.tolist()
            )
        else:
            ordered_for_sub = classifier_order
        fig = bars.bar_per_class(
            bootstrap_long,
            classes=[c for c in DEFAULT_PLOT_ORDER if c in SUBSTRATE_CLASSES],
            classifier_order=ordered_for_sub,
            metric=metric,
            title="Substrate prediction AP per class",
        )
        theme.save_figure(fig, plots_dir / "substrate_per_class")

    if "tps_ids_map" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS_IDS_mAP", metric=metric,
            classifier_subset=classifier_order,
            title="TPS / IDS detection mAP",
            ylabel="mAP (%)",
        )
        theme.save_figure(fig, plots_dir / "tps_ids_map")

    if "ee_comparison" in plot_set:
        ee_subset = list(args.ee_classifiers)
        ee_palette = theme.ee_ablation_palette(ee_subset)
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS", metric=metric,
            classifier_subset=ee_subset,
            palette=ee_palette,
            title="TPS detection AP — EnzymeExplorer ablation",
            ylim=tuple(args.ee_tps_ylim) if args.ee_tps_ylim else None,
        )
        theme.save_figure(fig, plots_dir / "ee_tps_detection")
        fig = bars.bar_classifier(
            bootstrap_long, target_class="Substrate_mAP", metric=metric,
            classifier_subset=ee_subset,
            palette=ee_palette,
            title="Substrate prediction mAP — EnzymeExplorer ablation",
            ylabel="mAP (%)",
            ylim=tuple(args.ee_substrate_map_ylim) if args.ee_substrate_map_ylim else None,
        )
        theme.save_figure(fig, plots_dir / "ee_substrate_map")
        sub = bootstrap_long[
            (bootstrap_long["metric"] == metric)
            & (bootstrap_long["class"].isin(SUBSTRATE_CLASSES))
            & (bootstrap_long["classifier"].isin(ee_subset))
        ].dropna(subset=["value"])
        if not sub.empty:
            ee_order_for_sub = (
                sub.groupby("classifier")["value"].mean().sort_values().index.tolist()
            )
            fig = bars.bar_per_class(
                bootstrap_long,
                classes=[c for c in DEFAULT_PLOT_ORDER if c in SUBSTRATE_CLASSES],
                classifier_order=ee_order_for_sub,
                metric=metric,
                palette=ee_palette,
                title="Substrate prediction AP per class — EnzymeExplorer ablation",
            )
            theme.save_figure(fig, plots_dir / "ee_substrate_per_class")

    needs_pooled = bool(plot_set & {"pr_curves", "roc_curves", "pr_per_class"})
    if needs_pooled:
        classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
        pooled = curves.pool_fold_dfs(classifier_to_dfs)

        if "pr_curves" in plot_set:
            for tgt in args.curve_classes:
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_pr_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    title=f"{tgt} — Precision-Recall curve",
                )
                theme.save_figure(fig, plots_dir / f"pr_{tgt}")
        if "roc_curves" in plot_set:
            for tgt in args.curve_classes:
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_roc_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    title=f"{tgt} — ROC curve",
                )
                theme.save_figure(fig, plots_dir / f"roc_{tgt}")
        if "pr_per_class" in plot_set:
            tgt_clf = args.pr_per_class_classifier
            if tgt_clf and tgt_clf in pooled:
                fig = curves.plot_per_class_pr_curves(
                    pooled[tgt_clf],
                    classes=DEFAULT_PLOT_ORDER,
                    title=f"{theme.display_name(tgt_clf)} — PR curves per substrate",
                )
                theme.save_figure(fig, plots_dir / f"pr_per_class_{tgt_clf}")

    cat_root = eval_dir / "categories"
    if cat_root.exists() and (
        plot_set & {"category_boxplot", "category_heatmap"}
    ):
        for cat_dir in sorted(p for p in cat_root.iterdir() if p.is_dir()):
            cat_label = cat_dir.name
            cat_summary_path = cat_dir / "summary.csv"
            cat_long_path = cat_dir / "bootstrap_long.csv"
            if not cat_summary_path.exists():
                continue
            cat_summary = pd.read_csv(cat_summary_path)
            cat_long = pd.read_csv(cat_long_path) if cat_long_path.exists() else None
            target_label_map = {
                "TPS": "TPS detection AP",
                "Substrate_mAP": "Substrate prediction mAP",
                "TPS_IDS_mAP": "TPS / IDS mAP",
            }
            cat_label_map = {"TPS_Type": "TPS type", "Kingdom": "kingdom"}
            cat_pretty = cat_label_map.get(cat_label, cat_label.replace("_", " "))
            for tgt in args.category_classes:
                if tgt not in set(cat_summary["class"]):
                    continue
                target_pretty = target_label_map.get(tgt, tgt.replace("_", " "))
                if "category_boxplot" in plot_set and cat_long is not None:
                    fig = categorical.plot_category_boxplot(
                        cat_long, target_class=tgt, metric=metric,
                        classifier_subset=classifier_order,
                        title=f"{target_pretty} per {cat_pretty}",
                        xlabel="",
                    )
                    theme.save_figure(fig, plots_dir / f"{cat_label}_boxplot_{tgt}")
                if "category_heatmap" in plot_set:
                    fig = categorical.plot_category_heatmap(
                        cat_summary, target_class=tgt, metric=metric,
                        classifier_subset=classifier_order,
                        title=f"{target_pretty} per {cat_pretty}",
                    )
                    theme.save_figure(fig, plots_dir / f"{cat_label}_heatmap_{tgt}")

    if "prediction_thresholds" in plot_set:
        cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
        pt_cfg = cfg.get("prediction_thresholds", {}) or {}
        precisions = (
            list(args.prediction_threshold_precisions)
            if args.prediction_threshold_precisions
            else list(pt_cfg.get("precisions", [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]))
        )
        target_classifiers = (
            list(args.prediction_threshold_classifiers)
            if args.prediction_threshold_classifiers
            else list(pt_cfg.get("classifiers", classifier_order))
        )
        target_classes = (
            list(args.prediction_threshold_classes)
            if args.prediction_threshold_classes
            else list(pt_cfg.get("classes", DEFAULT_PLOT_ORDER))
        )
        if not needs_pooled:
            classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
            pooled = curves.pool_fold_dfs(classifier_to_dfs)
        table = pt.compute_thresholds_table(
            pooled, target_classifiers, target_classes, precisions
        )
        (eval_dir / "prediction_thresholds.csv").write_text(table.to_csv(index=False))
        for clf in table["classifier"].unique():
            try:
                fig = pt_plots.plot_threshold_heatmap(
                    table, clf, classes_order=target_classes,
                )
                theme.save_figure(fig, plots_dir / f"prediction_thresholds_{clf}_score")
                fig = pt_plots.plot_recall_at_thresholds_heatmap(
                    table, clf, classes_order=target_classes,
                )
                theme.save_figure(fig, plots_dir / f"prediction_thresholds_{clf}_recall")
                fig = pt_plots.plot_n_above_threshold_heatmap(
                    table, clf, classes_order=target_classes,
                )
                theme.save_figure(fig, plots_dir / f"prediction_thresholds_{clf}_count")
            except ValueError as exc:
                logger.warning("Skipping %s prediction-threshold plots: %s", clf, exc)

    if "confidence_tiers" in plot_set:
        cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
        tiers_cfg = cfg.get("confidence_tiers", {}) or {}
        tier_definitions = ct.tier_definitions_from_config(tiers_cfg.get("tiers"))
        target_classifiers = (
            list(args.confidence_tier_classifiers)
            if args.confidence_tier_classifiers
            else list(tiers_cfg.get("classifiers", classifier_order))
        )
        target_classes = (
            list(args.confidence_tier_classes)
            if args.confidence_tier_classes
            else list(tiers_cfg.get("classes", DEFAULT_PLOT_ORDER))
        )
        if not (needs_pooled or "prediction_thresholds" in plot_set):
            classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
            pooled = curves.pool_fold_dfs(classifier_to_dfs)
        tier_table = ct.compute_tier_table(
            pooled, target_classifiers, target_classes, tier_definitions,
        )
        (eval_dir / "confidence_tiers.csv").write_text(tier_table.to_csv(index=False))
        for clf in tier_table["classifier"].unique():
            try:
                fig = ct_plots.plot_tier_ladder(
                    tier_table,
                    clf,
                    classes_order=target_classes,
                    tier_definitions=tier_definitions,
                )
                theme.save_figure(fig, plots_dir / f"confidence_tiers_{clf}")
            except ValueError as exc:
                logger.warning("Skipping %s confidence-tier plot: %s", clf, exc)

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
                    ylim=tuple(args.threshold_sweep_ylim)
                    if args.threshold_sweep_ylim else None,
                )
                theme.save_figure(fig, plots_dir / f"threshold_sweep_{csv_path.stem}")

    logger.info("Saved plots to %s", plots_dir)


PLOTS_AVAILABLE = (
    "tps_detection",
    "tps_detection_zoomed",
    "substrate_map",
    "substrate_per_class",
    "tps_ids_map",
    "ee_comparison",
    "pr_curves",
    "roc_curves",
    "pr_per_class",
    "category_boxplot",
    "category_heatmap",
    "threshold_sweep",
    "prediction_thresholds",
    "confidence_tiers",
)


def add_evaluate_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "evaluate", help="Bootstrap evaluation across classifiers"
    )
    parser.set_defaults(cmd="evaluate")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output-name", required=True, type=str)


def add_visualize_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize", help="Render plots from a saved evaluation run"
    )
    parser.set_defaults(cmd="visualize")
    parser.add_argument("--eval-output-name", required=True, type=str)
    parser.add_argument(
        "--plots", nargs="+", choices=PLOTS_AVAILABLE, default=list(PLOTS_AVAILABLE)
    )
    parser.add_argument("--metric", choices=("ap", "roc_auc"), default="ap")
    parser.add_argument("--classifier-order", nargs="+", default=None)
    parser.add_argument("--tps-ylim", nargs=2, type=float, default=None)
    parser.add_argument("--substrate-map-ylim", nargs=2, type=float, default=None)
    parser.add_argument(
        "--zoomed-tps-classifiers", nargs="+",
        default=["Foldseek", "PLM_Domains"],
        help=(
            "Subset of classifiers shown in the zoomed TPS detection plot. "
            "Defaults to FoldSeek vs the final EnzymeExplorer; the EE "
            "ablation variants live in the EE-comparison plots only."
        ),
    )
    parser.add_argument(
        "--zoomed-tps-ylim", nargs=2, type=float, default=[99.6, 100.0],
    )
    parser.add_argument(
        "--ee-classifiers", nargs="+",
        default=["Domains", "PLM", "PLM_Domains"],
        help="Classifiers compared in the EnzymeExplorer ablation plots.",
    )
    parser.add_argument(
        "--ee-tps-ylim", nargs=2, type=float, default=None,
        help="Y-axis range for the EE TPS detection bar (e.g. 99.5 100).",
    )
    parser.add_argument(
        "--ee-substrate-map-ylim", nargs=2, type=float, default=None,
    )
    parser.add_argument(
        "--curve-classes", nargs="+", default=["TPS"],
        help="Classes for PR/ROC curves (one figure per class).",
    )
    parser.add_argument(
        "--pr-per-class-classifier", type=str, default="PLM_Domains",
        help="Classifier whose per-class PR curves to render.",
    )
    parser.add_argument(
        "--category-classes",
        nargs="+",
        default=["TPS", "Substrate_mAP", "TPS_IDS_mAP"],
        help="Target classes/aggregates for category plots.",
    )
    parser.add_argument(
        "--threshold-sweep-ylim", nargs=2, type=float, default=None,
    )
    parser.add_argument(
        "--prediction-threshold-precisions", nargs="+", type=float, default=None,
        help="Precision targets (e.g. 0.99 0.95 0.9). Override YAML.",
    )
    parser.add_argument(
        "--prediction-threshold-classifiers", nargs="+", default=None,
        help="Classifiers to analyse for prediction thresholds. Override YAML.",
    )
    parser.add_argument(
        "--prediction-threshold-classes", nargs="+", default=None,
        help="Classes to analyse. Override YAML.",
    )
    parser.add_argument(
        "--confidence-tier-classifiers", nargs="+", default=None,
        help="Classifiers for the confidence-tier ladder. Override YAML.",
    )
    parser.add_argument(
        "--confidence-tier-classes", nargs="+", default=None,
        help="Classes for the confidence-tier ladder. Override YAML.",
    )
