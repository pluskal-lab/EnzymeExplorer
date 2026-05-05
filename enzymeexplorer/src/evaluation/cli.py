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
          with_distractors: true    # optional; default true. When false the
                                    # auto-pick stays inside *_no_distractors
                                    # versions.

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
        with_distractors: true      # optional; default true
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
import logging
import re
from pathlib import Path

import pandas as pd  # type: ignore
import yaml  # type: ignore

from enzymeexplorer.src.evaluation import (
    aggregate as agg,
    bootstrap as bs,
    cache as boot_cache,
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
        with_distractors = bool(sel_cfg.get("with_distractors", True))
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


def _build_classifier_dfs(cfg: dict):
    classifier_to_dfs: dict[str, dict[str, dict[int, eio.FoldDfs]]] = {}
    classifier_to_timestamps: dict[str, dict[str, str]] = {}
    resolved: dict[str, dict] = {}
    for label, spec in cfg["classifiers"].items():
        classes = list(spec["classes"])
        try:
            version_spec = _resolve_version_spec(label, spec, classes)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("Skipping %s: cannot resolve version (%s)", label, exc)
            continue
        logger.info("Resolved %s -> %s", label, version_spec)
        try:
            dfs, timestamps = eio.load_classifier_class_fold_dfs(
                spec["model"], version_spec, classes=classes
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

    classifier_to_dfs, classifier_to_timestamps, resolved = _build_classifier_dfs(cfg)

    with open(out_dir / "resolved_versions.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(resolved, fh, sort_keys=False)
    with open(out_dir / "eval_config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    bcfg = cfg.get("bootstrap", {})
    metrics = tuple(bcfg.get("metrics", ["ap", "roc_auc"]))
    n_boot = int(bcfg.get("n_bootstraps", 2000))
    seed = int(bcfg.get("seed", 42))
    mode = bcfg.get("mode", "cluster")
    ci = float(bcfg.get("ci", 0.95))
    ci_method = bcfg.get("ci_method", "percentile")

    force_all = bool(getattr(args, "force_bootstrap", False))
    force_specific = set(getattr(args, "force_classifiers", None) or [])

    parts: list[bs.BootstrapResult] = []
    for label, fold_dfs in classifier_to_dfs.items():
        spec = cfg["classifiers"][label]
        ver = resolved[label]["resolved_versions"]
        force = force_all or label in force_specific
        result, hit = boot_cache.bootstrap_with_cache(
            classifier=label,
            model=spec["model"],
            version_spec=ver,
            class_to_fold_dfs=fold_dfs,
            timestamps=classifier_to_timestamps[label],
            metrics=metrics,
            n_bootstraps=n_boot,
            seed=seed,
            mode=mode,
            force=force,
        )
        parts.append(result)
        logger.info(
            "Classifier %s: %s (n_draws=%d)",
            label, "cache hit" if hit else "fresh", n_boot,
        )
    result = boot_cache.merge_results(parts)
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
        kingdom_cache = cfg.get("id_metadata", {}).get(
            "kingdom_cache", "data/uniprot_kingdom_cache.json"
        )
        meta = eio.load_id_metadata(
            meta_csv, wanted_cols, kingdom_cache=kingdom_cache,
        )
        for cat_label, cat_spec in categories_cfg.items():
            if cat_spec.get("type_grouping"):
                logger.info(
                    "Skipping categorical bootstrap for %s (type_grouping=True; "
                    "boxplot/heatmap rendered from global bootstrap at visualize time)",
                    cat_label,
                )
                continue
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
            with_distractors = bool(entry.get("with_distractors", True))
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
) -> list[str]:
    order = (
        args.classifier_order
        or vcfg.get("classifier_order")
        or sorted(summary["classifier"].unique())
    )
    return list(order)


def _apply_pin_last(order: list[str], pin: str | None) -> list[str]:
    if pin is None or pin not in order:
        return order
    return [c for c in order if c != pin] + [pin]


def _resolve_visualize_cfg(eval_dir: Path, args: argparse.Namespace) -> dict:
    """Merge the YAML ``visualize:`` block with CLI args. CLI overrides YAML."""
    cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    vcfg = dict(cfg.get("visualize", {}) or {})
    if args.plots is not None:
        vcfg["plots"] = list(args.plots)
    if args.metric is not None:
        vcfg["metric"] = args.metric
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
    return vcfg


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

    vcfg = _resolve_visualize_cfg(eval_dir, args)
    plot_set = set(vcfg.get("plots") or PLOTS_AVAILABLE)
    metric = vcfg.get("metric", "ap")
    pin_last = vcfg.get("pin_last")
    auto_zoom_cfg = dict(vcfg.get("auto_zoom") or {})
    auto_zoom_enabled = bool(auto_zoom_cfg.get("enabled", True))
    zoom_delta_pct = float(auto_zoom_cfg.get("delta_pct", 2.0))
    xtick_overrides = dict(vcfg.get("ablation_xtick_overrides") or {})
    zoom_classifiers = vcfg.get("zoom_classifiers")
    yaml_palette = dict(vcfg.get("palette") or {}) or None
    fixed_order = bool(vcfg.get("fixed_classifier_order")) or bool(vcfg.get("classifier_order"))

    classifier_order = _apply_pin_last(
        _resolve_classifier_order(args, summary, vcfg), pin_last
    )

    def _full_zoom_ylim(target_class: str, classifier_subset: list[str]) -> tuple[float, float] | None:
        if bootstrap_long is None:
            return None
        sub = bootstrap_long[
            (bootstrap_long["metric"] == metric)
            & (bootstrap_long["class"] == target_class)
            & (bootstrap_long["classifier"].isin(classifier_subset))
        ].dropna(subset=["value"])
        if sub.empty:
            return None
        means_pct = (sub.groupby("classifier")["value"].mean() * 100.0).tolist()
        return bars.compute_full_zoom_ylim(means_pct)

    def _full_zoom_ylim_per_class(classes: list[str], classifier_subset: list[str]) -> tuple[float, float] | None:
        if bootstrap_long is None:
            return None
        sub = bootstrap_long[
            (bootstrap_long["metric"] == metric)
            & (bootstrap_long["class"].isin(classes))
            & (bootstrap_long["classifier"].isin(classifier_subset))
        ].dropna(subset=["value"])
        if sub.empty:
            return None
        # Means per (class, classifier) cell.
        cell_means = (
            sub.groupby(["class", "classifier"])["value"].mean() * 100.0
        ).tolist()
        return bars.compute_full_zoom_ylim(cell_means)

    def _maybe_zoom(fig_base_name: str, *, target_class, classifier_subset, base_title):
        if not auto_zoom_enabled or bootstrap_long is None:
            return
        try:
            zfig = bars.bar_classifier_zoom(
                bootstrap_long,
                target_class=target_class,
                metric=metric,
                classifier_subset=classifier_subset,
                forced_subset=zoom_classifiers,
                delta_pct=zoom_delta_pct,
                pin_last=pin_last,
                xtick_overrides=xtick_overrides,
                base_title=base_title,
                palette=yaml_palette,
                fixed_order=classifier_order if fixed_order else None,
            )
        except bars.NoZoomNeeded:
            return
        if zfig is None:
            return
        theme.save_figure(zfig, plots_dir / f"{fig_base_name}_zoomed")

    fixed_arg = classifier_order if fixed_order else None
    if "tps_detection" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS", metric=metric,
            classifier_subset=classifier_order,
            classifier_order=fixed_arg,
            pin_last=pin_last,
            palette=yaml_palette,
            xtick_overrides=xtick_overrides,
            title="TPS detection AP",
            ylim=tuple(vcfg["tps_ylim"]) if vcfg.get("tps_ylim") else None,
        )
        theme.save_figure(fig, plots_dir / "tps_detection")
        _maybe_zoom(
            "tps_detection", target_class="TPS",
            classifier_subset=classifier_order, base_title="TPS detection AP",
        )
        ylim = _full_zoom_ylim("TPS", classifier_order)
        if ylim is not None:
            fig = bars.bar_classifier(
                bootstrap_long, target_class="TPS", metric=metric,
                classifier_subset=classifier_order,
                classifier_order=fixed_arg,
                pin_last=pin_last,
                palette=yaml_palette,
                xtick_overrides=xtick_overrides,
                title="TPS detection AP (zoomed)",
                ylim=ylim,
            )
            theme.save_figure(fig, plots_dir / "tps_detection_zoomed_full")

    if "substrate_map" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="Substrate_mAP", metric=metric,
            classifier_subset=classifier_order,
            classifier_order=fixed_arg,
            pin_last=pin_last,
            palette=yaml_palette,
            xtick_overrides=xtick_overrides,
            title="Substrate prediction mAP",
            ylabel="mAP (%)",
            ylim=tuple(vcfg["substrate_map_ylim"]) if vcfg.get("substrate_map_ylim") else None,
        )
        theme.save_figure(fig, plots_dir / "substrate_map")
        _maybe_zoom(
            "substrate_map", target_class="Substrate_mAP",
            classifier_subset=classifier_order, base_title="Substrate prediction mAP",
        )
        ylim = _full_zoom_ylim("Substrate_mAP", classifier_order)
        if ylim is not None:
            fig = bars.bar_classifier(
                bootstrap_long, target_class="Substrate_mAP", metric=metric,
                classifier_subset=classifier_order,
                classifier_order=fixed_arg,
                pin_last=pin_last,
                palette=yaml_palette,
                xtick_overrides=xtick_overrides,
                title="Substrate prediction mAP (zoomed)",
                ylabel="mAP (%)",
                ylim=ylim,
            )
            theme.save_figure(fig, plots_dir / "substrate_map_zoomed_full")

    if "substrate_per_class" in plot_set:
        if fixed_order:
            ordered_for_sub = list(classifier_order)
        else:
            sub = bootstrap_long[
                (bootstrap_long["metric"] == metric)
                & (bootstrap_long["class"].isin(SUBSTRATE_CLASSES))
                & (bootstrap_long["classifier"].isin(classifier_order))
            ].dropna(subset=["value"])
            if not sub.empty:
                ordered_for_sub = (
                    sub.groupby("classifier")["value"].mean().sort_values().index.tolist()
                )
                ordered_for_sub = _apply_pin_last(ordered_for_sub, pin_last)
            else:
                ordered_for_sub = classifier_order
        fig = bars.bar_per_class(
            bootstrap_long,
            classes=[c for c in DEFAULT_PLOT_ORDER if c in SUBSTRATE_CLASSES],
            classifier_order=ordered_for_sub,
            metric=metric,
            palette=yaml_palette,
            xtick_overrides=xtick_overrides,
            title="Substrate prediction AP per class",
        )
        theme.save_figure(fig, plots_dir / "substrate_per_class")
        substrate_classes = [c for c in DEFAULT_PLOT_ORDER if c in SUBSTRATE_CLASSES]
        ylim = _full_zoom_ylim_per_class(substrate_classes, ordered_for_sub)
        if ylim is not None:
            fig = bars.bar_per_class(
                bootstrap_long,
                classes=substrate_classes,
                classifier_order=ordered_for_sub,
                metric=metric,
                palette=yaml_palette,
                xtick_overrides=xtick_overrides,
                title="Substrate prediction AP per class (zoomed)",
                ylim=ylim,
            )
            theme.save_figure(fig, plots_dir / "substrate_per_class_zoomed_full")

    if "tps_ids_map" in plot_set:
        fig = bars.bar_classifier(
            bootstrap_long, target_class="TPS_IDS_mAP", metric=metric,
            classifier_subset=classifier_order,
            classifier_order=fixed_arg,
            pin_last=pin_last,
            palette=yaml_palette,
            xtick_overrides=xtick_overrides,
            title="TPS / IDS detection mAP",
            ylabel="mAP (%)",
        )
        theme.save_figure(fig, plots_dir / "tps_ids_map")
        ylim = _full_zoom_ylim("TPS_IDS_mAP", classifier_order)
        if ylim is not None:
            fig = bars.bar_classifier(
                bootstrap_long, target_class="TPS_IDS_mAP", metric=metric,
                classifier_subset=classifier_order,
                classifier_order=fixed_arg,
                pin_last=pin_last,
                palette=yaml_palette,
                xtick_overrides=xtick_overrides,
                title="TPS / IDS detection mAP (zoomed)",
                ylabel="mAP (%)",
                ylim=ylim,
            )
            theme.save_figure(fig, plots_dir / "tps_ids_map_zoomed_full")

    needs_pooled = bool(
        plot_set & {"pr_curves", "roc_curves", "pr_per_class",
                    "pr_substrate", "roc_substrate"}
    )
    pooled = None
    if needs_pooled:
        classifier_to_dfs, _ = _load_classifier_dfs_for_visualize(eval_dir)
        pooled = curves.pool_fold_dfs(classifier_to_dfs)

        if "pr_curves" in plot_set:
            for tgt in vcfg.get("curve_classes", ["TPS"]):
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_pr_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    palette=yaml_palette,
                    title=f"{tgt} — Precision-Recall curve",
                )
                theme.save_figure(fig, plots_dir / f"pr_{tgt}")
        if "roc_curves" in plot_set:
            for tgt in vcfg.get("curve_classes", ["TPS"]):
                eligible = [c for c in classifier_order if tgt in pooled.get(c, {})]
                if not eligible:
                    continue
                fig = curves.plot_roc_curves(
                    pooled, target_class=tgt, classifier_order=eligible,
                    palette=yaml_palette,
                    title=f"{tgt} — ROC curve",
                )
                theme.save_figure(fig, plots_dir / f"roc_{tgt}")
        if "pr_per_class" in plot_set:
            tgt_clf = vcfg.get("pr_per_class_classifier", "PLM_Domains")
            if tgt_clf and tgt_clf in pooled:
                fig = curves.plot_per_class_pr_curves(
                    pooled[tgt_clf],
                    classes=DEFAULT_PLOT_ORDER,
                    title=f"{theme.display_name(tgt_clf)} — PR curves per substrate",
                )
                theme.save_figure(fig, plots_dir / f"pr_per_class_{tgt_clf}")
        if "pr_substrate" in plot_set:
            eligible = [
                c for c in classifier_order
                if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
            ]
            if eligible:
                fig = curves.plot_micro_pr_curves(
                    pooled, classes=SUBSTRATE_CLASSES,
                    classifier_order=eligible,
                    palette=yaml_palette,
                    title="Substrate prediction — PR curve",
                )
                theme.save_figure(fig, plots_dir / "pr_substrate")
        if "roc_substrate" in plot_set:
            eligible = [
                c for c in classifier_order
                if any(cls in pooled.get(c, {}) for cls in SUBSTRATE_CLASSES)
            ]
            if eligible:
                fig = curves.plot_micro_roc_curves(
                    pooled, classes=SUBSTRATE_CLASSES,
                    classifier_order=eligible,
                    palette=yaml_palette,
                    title="Substrate prediction — ROC curve",
                )
                theme.save_figure(fig, plots_dir / "roc_substrate")

    cat_root = eval_dir / "categories"
    eval_cfg_full = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
    categories_cfg = eval_cfg_full.get("categories", {}) or {}
    if categories_cfg and (plot_set & {"category_boxplot", "category_heatmap"}):
        target_label_map = {
            "TPS": "TPS detection AP",
            "Substrate_mAP": "Substrate prediction mAP",
            "TPS_IDS_mAP": "TPS / IDS mAP",
        }
        cat_pretty_map = {"TPS_Type": "TPS type", "Kingdom": "kingdom"}
        for cat_label, cat_spec in categories_cfg.items():
            cat_pretty = cat_pretty_map.get(cat_label, cat_label.replace("_", " "))

            if cat_spec.get("type_grouping"):
                # Render directly from the global per-substrate bootstrap.
                from enzymeexplorer.src.evaluation.classes import (
                    SUBSTRATE_TO_TYPE, DEFAULT_TYPE_ORDER,
                )
                type_map = dict(cat_spec.get("class_to_type") or SUBSTRATE_TO_TYPE)
                type_order = list(
                    cat_spec.get("type_order") or DEFAULT_TYPE_ORDER
                )
                if bootstrap_long is None:
                    continue
                sub = bootstrap_long[
                    (bootstrap_long["metric"] == metric)
                    & (bootstrap_long["class"].isin(type_map))
                    & (bootstrap_long["classifier"].isin(classifier_order))
                ].dropna(subset=["value"]).copy()
                if sub.empty:
                    continue
                sub["category"] = sub["class"].map(type_map)
                sub["class"] = "Substrate_grouped"
                target_pretty = "Substrate prediction AP"
                if "category_boxplot" in plot_set:
                    fig = categorical.plot_category_boxplot(
                        sub, target_class="Substrate_grouped", metric=metric,
                        categories=[t for t in type_order if t in set(sub["category"])],
                        classifier_subset=classifier_order,
                        pin_last=pin_last,
                        title=f"{target_pretty} grouped by {cat_pretty}",
                        xlabel="",
                    )
                    theme.save_figure(
                        fig, plots_dir / f"{cat_label}_boxplot_Substrate_mAP"
                    )
                if "category_heatmap" in plot_set:
                    point_path = eval_dir / "point_estimates.csv"
                    if point_path.exists():
                        point = pd.read_csv(point_path)
                        point = point[
                            (point["metric"] == metric)
                            & (point["class"].isin(type_map))
                            & (point["classifier"].isin(classifier_order))
                        ].copy()
                        if not point.empty:
                            point["category"] = point["class"].map(type_map)
                            cell = (
                                point.groupby(["classifier", "category", "metric"],
                                              as_index=False)["value"].mean()
                                .rename(columns={"value": "point"})
                            )
                            cell["class"] = "Substrate_grouped"
                            fig = categorical.plot_category_heatmap(
                                cell, target_class="Substrate_grouped", metric=metric,
                                categories=[t for t in type_order if t in set(cell["category"])],
                                classifier_subset=classifier_order,
                                pin_last=pin_last,
                                title=f"{target_pretty} grouped by {cat_pretty}",
                            )
                            theme.save_figure(
                                fig, plots_dir / f"{cat_label}_heatmap_Substrate_mAP"
                            )
                continue

            # Masked categorical (Kingdom, etc.) — read from disk.
            cat_dir = cat_root / cat_label
            cat_summary_path = cat_dir / "summary.csv"
            cat_long_path = cat_dir / "bootstrap_long.csv"
            if not cat_summary_path.exists():
                continue
            cat_summary = pd.read_csv(cat_summary_path)
            cat_long = pd.read_csv(cat_long_path) if cat_long_path.exists() else None
            for tgt in vcfg.get("category_classes",
                                ["TPS", "Substrate_mAP", "TPS_IDS_mAP"]):
                if tgt not in set(cat_summary["class"]):
                    continue
                target_pretty = target_label_map.get(tgt, tgt.replace("_", " "))
                if "category_boxplot" in plot_set and cat_long is not None:
                    fig = categorical.plot_category_boxplot(
                        cat_long, target_class=tgt, metric=metric,
                        classifier_subset=classifier_order,
                        pin_last=pin_last,
                        title=f"{target_pretty} per {cat_pretty}",
                        xlabel="",
                    )
                    theme.save_figure(fig, plots_dir / f"{cat_label}_boxplot_{tgt}")
                if "category_heatmap" in plot_set:
                    fig = categorical.plot_category_heatmap(
                        cat_summary, target_class=tgt, metric=metric,
                        classifier_subset=classifier_order,
                        pin_last=pin_last,
                        title=f"{target_pretty} per {cat_pretty}",
                    )
                    theme.save_figure(fig, plots_dir / f"{cat_label}_heatmap_{tgt}")

    if "prediction_thresholds" in plot_set:
        cfg = yaml.safe_load((eval_dir / "eval_config.yaml").read_text())
        pt_cfg = cfg.get("prediction_thresholds", {}) or {}
        precisions = list(pt_cfg.get("precisions",
                                     [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]))
        target_classifiers = list(pt_cfg.get("classifiers", classifier_order))
        target_classes = list(pt_cfg.get("classes", DEFAULT_PLOT_ORDER))
        if pooled is None:
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
        target_classifiers = list(tiers_cfg.get("classifiers", classifier_order))
        target_classes = list(tiers_cfg.get("classes", DEFAULT_PLOT_ORDER))
        if pooled is None:
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
        # Multi-panel grid version, one panel per classifier in the order
        # configured (or alphabetical), useful as a publication overview.
        try:
            fig = ct_plots.plot_tier_ladder_grid(
                tier_table,
                classifier_order=[
                    c for c in target_classifiers
                    if c in set(tier_table["classifier"])
                ],
                classes_order=target_classes,
                tier_definitions=tier_definitions,
            )
            theme.save_figure(fig, plots_dir / "confidence_tiers_grid")
        except ValueError as exc:
            logger.warning("Skipping confidence-tier grid plot: %s", exc)

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
    "tps_ids_map",
    "pr_curves",
    "roc_curves",
    "pr_per_class",
    "pr_substrate",
    "roc_substrate",
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
    parser.add_argument(
        "--force-bootstrap", action="store_true",
        help="Rebuild the per-classifier bootstrap cache for every classifier."
    )
    parser.add_argument(
        "--force-classifiers", nargs="+", default=None,
        help="Rebuild bootstrap cache only for the listed classifier labels."
    )


def add_visualize_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize", help="Render plots from a saved evaluation run"
    )
    parser.set_defaults(cmd="visualize")
    parser.add_argument("--eval-output-name", required=True, type=str)
    parser.add_argument(
        "--plots", nargs="+", choices=PLOTS_AVAILABLE, default=None,
        help="Plot names to render. Defaults to the set in the YAML "
             "visualize block, or all available."
    )
    parser.add_argument("--metric", choices=("ap", "roc_auc"), default=None)
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
