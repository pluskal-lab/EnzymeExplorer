"""File containing experiment evaluation"""

import argparse
import logging
import pickle
from collections import defaultdict
from typing import Optional, Union

from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


from enzymeexplorer.src.evaluation.metrics import summary_mccf1
from enzymeexplorer.src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from enzymeexplorer.src.models.ifaces import BaseConfig
from enzymeexplorer.src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_evaluations_output,
    get_output_root,
)

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

DEFAULT_SIMILARITY_BINS: list[tuple[float, float]] = [
    (0, 30),
    (30, 50),
    (50, 70),
    (70, 90),
    (90, 100),
]

_NO_HIT_LABEL = "no_hit"
_ALL_LABEL = "all"

MIN_NEGATIVES_FOR_EVAL = 3


def load_similarity_artifact(
    path: str,
) -> dict[int, dict[str, dict[str, Union[float, bool]]]]:
    """Load a similarity pickle and normalise to the rich MMseqs schema.

    Legacy BLAST format::

        {fold_i: {seq_id: max_blast_identity_float}}

    MMseqs format::

        {fold_i: {seq_id: {"pident": float, "qcov": float,
                           "evalue": float, "has_hit": bool}}}

    Returns the MMseqs-style dict in both cases.  Legacy entries are
    tagged ``is_synthetic: True`` so that downstream code can skip
    qcov-based filtering for imputed values.
    """
    with open(path, "rb") as fh:
        raw: dict = pickle.load(fh)

    normalised: dict[int, dict[str, dict[str, Union[float, bool]]]] = {}
    for fold_key, id_map in raw.items():
        fold_idx = int(fold_key)
        norm_map: dict[str, dict[str, Union[float, bool]]] = {}
        for seq_id, value in id_map.items():
            if isinstance(value, dict):
                norm_map[seq_id] = value
            else:
                norm_map[seq_id] = {
                    "pident": float(value),
                    "qcov": 1.0,
                    "evalue": 0.0,
                    "has_hit": True,
                    "is_synthetic": True,
                }
        normalised[fold_idx] = norm_map
    return normalised


def _get_pident(
    sim_record: dict[str, Union[float, bool]],
) -> float:
    return float(sim_record.get("pident", 0.0))


def _has_hit(
    sim_record: dict[str, Union[float, bool]],
) -> bool:
    return bool(sim_record.get("has_hit", True))


def _build_bin_label(lo: float, hi: float) -> str:
    return f"{lo:.0f}-{hi:.0f}"


def _compute_metrics_for_bin(
    y_true: pd.Series,
    y_pred: np.ndarray,
    mask: pd.Series,
    min_pos: int,
    min_neg: int,
) -> Optional[dict]:
    """Compute AP, ROC-AUC, MCC-F1 for a subset, returning None if ineligible."""
    y_t = y_true[mask]
    y_p = y_pred[mask]
    n_pos = int(y_t.sum())
    n_neg = int(len(y_t) - n_pos)
    if n_pos < min_pos or n_neg < min_neg:
        return None
    ap = average_precision_score(y_t, y_p)
    auc = roc_auc_score(y_t, y_p)
    mccf1 = summary_mccf1(y_t, y_p)["mccf1_metric"]
    pr = precision_recall_curve(y_t, y_p)
    return {
        "ap": ap,
        "auc": auc,
        "mccf1": mccf1,
        "pr": pr,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def eval_experiment(
    experiment_info: ExperimentInfo,
    target_col: str,
    min_sample_count_for_eval: int,
    n_folds: int,
    classes: list[str],
    id_2_category_path: Optional[str] = None,
    blast_identities_path: Optional[str] = None,
    id_col_name: Optional[str] = "Uniprot ID",
    max_allowed_blast_identity: Optional[int] = 60,
    similarity_bins: Optional[list[tuple[float, float]]] = None,
    min_negatives_for_eval: int = MIN_NEGATIVES_FOR_EVAL,
) -> tuple[list, list, list, list]:
    """
    Evaluate results of the specified experiment.

    Supports both legacy BLAST identity pickles and rich MMseqs
    similarity artifacts (auto-detected).  When *similarity_bins* is
    provided the configurable bins are used; otherwise the legacy
    10-step bucketing is applied for backward compatibility.

    :param similarity_bins: list of (lo, hi) tuples in percent-identity
        space, e.g. ``[(0, 30), (30, 50), ...]``.  An "all" and
        "no_hit" pseudo-bin are always added automatically.
    :param min_negatives_for_eval: Minimum negatives in a bin for it
        to be eligible (in addition to *min_sample_count_for_eval*
        for positives).
    """
    experiment_output_folder_root = (
        get_output_root() / experiment_info.model_type / experiment_info.model_version
    )
    assert (
        experiment_output_folder_root.exists()
    ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"

    model_version_fold_folders = {
        x.stem for x in experiment_output_folder_root.glob("*")
    }
    if (
        len(model_version_fold_folders.intersection(set(map(str, range(n_folds)))))
        == n_folds
    ):
        logger.info("Found %d fold results for %s", n_folds, str(experiment_info))
        fold_2_root_dir = {
            fold_i: experiment_output_folder_root / f"{fold_i}"
            for fold_i in range(n_folds)
        }
    elif "all_folds" in model_version_fold_folders:
        logger.info("Found all_folds results for %s", f"{experiment_info}")
        fold_2_root_dir = {
            fold_i: experiment_output_folder_root / "all_folds"
            for fold_i in range(n_folds)
        }
    else:
        raise NotImplementedError(
            f"Not all fold outputs found. Please run corresponding experiments "
            f"({experiment_info}) before evaluation"
        )

    class_2_ap_vals, class_2_rocauc_vals, class_2_mccf1_vals, class_2_pr_vals = [
        [] for _ in range(4)
    ]

    if id_2_category_path is not None:
        with open(id_2_category_path, "rb") as file:
            id_2_category = pickle.load(file)
    else:
        id_2_category = None

    fold_2_similarity: Optional[dict] = None
    if blast_identities_path is not None:
        fold_2_similarity = load_similarity_artifact(blast_identities_path)

    if similarity_bins is None:
        similarity_bins = [
            (float(lb), float(lb + 10))
            for lb in range(0, (max_allowed_blast_identity or 60) + 1, 10)
        ]

    # pylint: disable=R1702
    for fold_i, fold_root_dir in fold_2_root_dir.items():
        logger.info("Processing fold %d with root dir %s", fold_i, str(fold_root_dir))
        class_2_ap: dict = {}
        class_2_mccf1: dict = {}
        class_2_auc: dict = {}
        class_2_pr: dict = {}

        for class_name in classes:
            if class_name in {"Unknown", "other"}:
                continue

            if (fold_root_dir / f"{class_name}").exists():
                fold_class_path = fold_root_dir / f"{class_name}"
            elif (fold_root_dir / "all_classes").exists():
                fold_class_path = fold_root_dir / "all_classes"
            else:
                fold_class_path = None

            logger.info(
                "Processing class %s with path %s", class_name, str(fold_class_path)
            )
            if fold_class_path is None:
                continue

            try:
                fold_class_latest_path = sorted(fold_class_path.glob("*"))[-1]
            except IndexError as index_error:
                raise NotImplementedError(
                    f"Please run corresponding experiments "
                    f"({experiment_info}) before evaluation"
                ) from index_error
            try:
                with open(
                    fold_class_latest_path / f"fold_{fold_i}_results.pkl", "rb"
                ) as file:
                    val_proba_np, class_names_in_fold, test_df = pickle.load(file)

                if class_name not in class_names_in_fold:
                    continue
                if not isinstance(class_names_in_fold, list):
                    class_names_in_fold = list(class_names_in_fold)

                y_true = test_df[target_col].map(lambda x: class_name in x)
                y_pred = val_proba_np[:, class_names_in_fold.index(class_name)]

                current_categories = test_df[id_col_name].map(
                    lambda x: (
                        id_2_category.get(x, "Unknown")
                        if id_2_category is not None
                        else ""
                    )
                )

                fold_sim = (
                    fold_2_similarity[fold_i] if fold_2_similarity is not None else None
                )
                _no_hit_default: dict[str, Union[float, bool]] = {
                    "pident": 0.0,
                    "has_hit": False,
                }
                _full_hit_default: dict[str, Union[float, bool]] = {
                    "pident": 100.0,
                    "has_hit": True,
                }

                current_sim_records = test_df[id_col_name].map(
                    lambda x: (
                        fold_sim.get(x, _no_hit_default)
                        if fold_sim is not None
                        else _full_hit_default
                    )
                )
                current_pidents = current_sim_records.map(_get_pident)
                current_has_hit = current_sim_records.map(_has_hit)

                for category in set(current_categories).difference({"Unknown"}):
                    is_category = current_categories.isin({category, "Unknown"})

                    # ── "all" bucket (no similarity filter) ──────
                    all_key = _make_record_key(class_name, category, _ALL_LABEL)
                    res = _compute_metrics_for_bin(
                        y_true,
                        y_pred,
                        is_category,
                        min_pos=min_sample_count_for_eval,
                        min_neg=min_negatives_for_eval,
                    )
                    if res is not None:
                        class_2_ap[all_key] = res["ap"]
                        class_2_auc[all_key] = res["auc"]
                        class_2_mccf1[all_key] = res["mccf1"]
                        class_2_pr[all_key] = res["pr"]
                        logger.info(
                            "  [%s] n_pos=%d n_neg=%d AP=%.3f",
                            all_key,
                            res["n_pos"],
                            res["n_neg"],
                            res["ap"],
                        )

                    # ── "no_hit" bucket ──────────────────────────
                    if fold_2_similarity is not None:
                        no_hit_mask = is_category & ~current_has_hit
                        no_hit_key = _make_record_key(
                            class_name, category, _NO_HIT_LABEL
                        )
                        res = _compute_metrics_for_bin(
                            y_true,
                            y_pred,
                            no_hit_mask,
                            min_pos=min_sample_count_for_eval,
                            min_neg=min_negatives_for_eval,
                        )
                        if res is not None:
                            class_2_ap[no_hit_key] = res["ap"]
                            class_2_auc[no_hit_key] = res["auc"]
                            class_2_mccf1[no_hit_key] = res["mccf1"]
                            class_2_pr[no_hit_key] = res["pr"]
                            logger.info(
                                "  [%s] n_pos=%d n_neg=%d AP=%.3f",
                                no_hit_key,
                                res["n_pos"],
                                res["n_neg"],
                                res["ap"],
                            )

                    # ── similarity bins ──────────────────────────
                    for lo, hi in similarity_bins:
                        in_bin = (
                            is_category
                            & current_has_hit
                            & current_pidents.map(
                                lambda x, _lo=lo, _hi=hi: _lo <= x < _hi
                            )
                        )
                        bin_key = _make_record_key(
                            class_name,
                            category,
                            _build_bin_label(lo, hi),
                        )
                        res = _compute_metrics_for_bin(
                            y_true,
                            y_pred,
                            in_bin,
                            min_pos=min_sample_count_for_eval,
                            min_neg=min_negatives_for_eval,
                        )
                        if res is not None:
                            class_2_ap[bin_key] = res["ap"]
                            class_2_auc[bin_key] = res["auc"]
                            class_2_mccf1[bin_key] = res["mccf1"]
                            class_2_pr[bin_key] = res["pr"]
                            logger.info(
                                "  [%s] n_pos=%d n_neg=%d AP=%.3f",
                                bin_key,
                                res["n_pos"],
                                res["n_neg"],
                                res["ap"],
                            )

            except FileNotFoundError:
                logger.warning(
                    "Fold %d results were not found for (%s)",
                    fold_i,
                    str(experiment_info),
                )

        class_2_ap_vals.append(class_2_ap)
        class_2_mccf1_vals.append(class_2_mccf1)
        class_2_rocauc_vals.append(class_2_auc)
        class_2_pr_vals.append(class_2_pr)
    return class_2_ap_vals, class_2_rocauc_vals, class_2_mccf1_vals, class_2_pr_vals


def _make_record_key(class_name: str, category: str, bin_label: str) -> str:
    """Build the composite key used in per-class metric dicts.

    Backward-compatible: when *category* is empty and *bin_label* is
    "all" the key collapses to just the class name.
    """
    base = class_name if category == "" else f"{category}_|_{class_name}"
    if bin_label == _ALL_LABEL:
        return base
    return f"{bin_label}_||_{base}"


def evaluate_selected_experiments(args: argparse.Namespace):
    """
    Function for evaluating the outputs of experiments that are enabled in the configs or a selected experiment.

    :param args: Parsed argparse namespace containing the evaluation parameters

    :return: None. Outputs the evaluation results to specified files.
    """

    config_root_path = get_config_root()
    (
        model_2_class_2_ap_vals,
        model_2_class_2_rocauc_vals,
        model_2_class_2_mccf1_vals,
        model_2_class_2_pr_vals,
    ) = [{} for _ in range(4)]
    if args.select_single_experiment:
        experiment_kwargs = collect_single_experiment_arguments(config_root_path)
        experiment_info = ExperimentInfo(**experiment_kwargs)
        config_path = (
            config_root_path
            / experiment_info.model_type
            / experiment_info.model_version
            / "config.yaml"
        )
        config_dict = BaseConfig.load(config_path)
        try:
            (
                class_2_ap_vals,
                class_2_rocauc_vals,
                class_2_mccf1_vals,
                class_2_pr_vals,
            ) = eval_experiment(
                experiment_info,
                target_col=config_dict["target_col_name"],
                min_sample_count_for_eval=args.minimal_count_to_eval,
                n_folds=args.n_folds,
                classes=args.classes,
                id_2_category_path=args.id_2_category_path,
                blast_identities_path=args.blast_identities_path,
                id_col_name=args.id_col_name,
            )
        except AssertionError as error:
            raise NotImplementedError(
                f"Please run corresponding experiments ({experiment_info}) before evaluation"
            ) from error
        model_name = f"{experiment_info.model_type}__{experiment_info.model_version}"
        model_2_class_2_ap_vals[model_name] = class_2_ap_vals
        model_2_class_2_rocauc_vals[model_name] = class_2_rocauc_vals
        model_2_class_2_mccf1_vals[model_name] = class_2_mccf1_vals
        model_2_class_2_pr_vals[model_name] = class_2_pr_vals
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
            model_name = (
                f"{experiment_info.model_type}__{experiment_info.model_version}"
            )
            if model_name not in args.models:
                logger.info(
                    "Skipping %s, with all models %s",
                    model_name,
                    ", ".join(args.models),
                )
                continue
            logger.info(
                "Evaluating %s/%s",
                experiment_info.model_type,
                experiment_info.model_version,
            )
            config_path = (
                config_root_path
                / experiment_info.model_type
                / experiment_info.model_version
                / "config.yaml"
            )
            config_dict = BaseConfig.load(config_path)
            logger.info(
                "Evaluating %s/%s",
                experiment_info.model_type,
                experiment_info.model_version,
            )
            try:
                (
                    class_2_ap_vals,
                    class_2_rocauc_vals,
                    class_2_mccf1_vals,
                    class_2_pr_vals,
                ) = eval_experiment(
                    experiment_info,
                    target_col=config_dict["target_col_name"],
                    min_sample_count_for_eval=args.minimal_count_to_eval,
                    n_folds=args.n_folds,
                    classes=args.classes,
                    id_2_category_path=args.id_2_category_path,
                    blast_identities_path=args.blast_identities_path,
                    id_col_name=args.id_col_name,
                )
            except (AssertionError, NotImplementedError):
                raise NotImplementedError(
                    f"Please run corresponding experiments ({experiment_info}) before evaluation"
                )
                continue

            model_2_class_2_ap_vals[model_name] = class_2_ap_vals
            model_2_class_2_rocauc_vals[model_name] = class_2_rocauc_vals
            model_2_class_2_mccf1_vals[model_name] = class_2_mccf1_vals
            model_2_class_2_pr_vals[model_name] = class_2_pr_vals

    all_results_model = []
    all_results_map = []
    all_results_map_minus_se = []
    all_results_map_plus_se = []
    all_results_mean_rocauc = []
    all_results_mean_rocauc_minus_se = []
    all_results_mean_rocauc_plus_se = []
    all_results_mean_mcc_f1 = []
    all_results_mean_mcc_f1_minus_se = []
    all_results_mean_mcc_f1_plus_se = []

    class_results_model = []
    class_results_class_name = []
    class_results_ap = []
    class_results_rocauc = []
    class_results_mcc_f1 = []
    class_results_ap_se = []
    class_results_rocauc_se = []
    class_results_mcc_f1_se = []

    eval_output_path = get_evaluations_output()
    if not eval_output_path.exists():
        eval_output_path.mkdir(parents=True)

    for model_name, class_2_vals_list in model_2_class_2_ap_vals.items():
        # getting all present class names
        present_class_names: set = set()
        for class_2_vals in class_2_vals_list:
            present_class_names = present_class_names.union(class_2_vals.keys())
        for class_name in present_class_names:
            class_results_model.append(model_name)
            class_results_class_name.append(class_name)
            ap_values = [
                class_2_vals.get(class_name, np.nan)
                for class_2_vals in model_2_class_2_ap_vals[model_name]
            ]
            rocauc_values = [
                class_2_vals.get(class_name, np.nan)
                for class_2_vals in model_2_class_2_rocauc_vals[model_name]
            ]
            mccf1_values = [
                class_2_vals.get(class_name, np.nan)
                for class_2_vals in model_2_class_2_mccf1_vals[model_name]
            ]
            class_results_ap.append(np.nanmean(ap_values))
            class_results_rocauc.append(np.nanmean(rocauc_values))
            class_results_mcc_f1.append(np.nanmean(mccf1_values))
            class_results_ap_se.append(np.std(ap_values, ddof=1))
            class_results_rocauc_se.append(np.std(rocauc_values, ddof=1))
            class_results_mcc_f1_se.append(np.std(mccf1_values, ddof=1))

    model_2_ap_mean_se = compute_mean_and_standard_error(model_2_class_2_ap_vals)
    model_2_rocauc_mean_se = compute_mean_and_standard_error(
        model_2_class_2_rocauc_vals
    )
    model_2_mccf1_mean_se = compute_mean_and_standard_error(model_2_class_2_mccf1_vals)
    for model, (map_mean, map_sem) in model_2_ap_mean_se.items():
        all_results_model.append(model)
        all_results_map.append(map_mean)
        all_results_map_minus_se.append(map_mean - map_sem)
        all_results_map_plus_se.append(map_mean + map_sem)
        rocauc_mean, rocauc_sem = model_2_rocauc_mean_se[model]
        all_results_mean_rocauc.append(rocauc_mean)
        all_results_mean_rocauc_minus_se.append(rocauc_mean - rocauc_sem)
        all_results_mean_rocauc_plus_se.append(rocauc_mean + rocauc_sem)
        mccf1_mean, mccf1_sem = model_2_mccf1_mean_se[model]
        all_results_mean_mcc_f1.append(mccf1_mean)
        all_results_mean_mcc_f1_minus_se.append(mccf1_mean - mccf1_sem)
        all_results_mean_mcc_f1_plus_se.append(mccf1_mean + mccf1_sem)

    all_results_df = pd.DataFrame(
        {
            "Model": all_results_model,
            "Mean Average Precision (mAP)": all_results_map,
            "mAP - SEM": all_results_map_minus_se,
            "mAP + SEM": all_results_map_plus_se,
            "ROC-AUC (macro mean)": all_results_mean_rocauc,
            "Mean ROC-AUC - SEM": all_results_mean_rocauc_minus_se,
            "Mean ROC-AUC + SEM": all_results_mean_rocauc_plus_se,
            "MCC-F1 summary (macro mean)": all_results_mean_mcc_f1,
            "Mean MCC-F1 summary - SEM": all_results_mean_mcc_f1_minus_se,
            "Mean MCC-F1 summary + SEM": all_results_mean_mcc_f1_plus_se,
        }
    )
    all_results_df.to_csv(eval_output_path / f"{args.output_filename}.csv", index=False)

    per_class_results_df = pd.DataFrame(
        {
            "Model": class_results_model,
            "Class": class_results_class_name,
            "Average Precision": class_results_ap,
            "ROC-AUC": class_results_rocauc,
            "MCC-F1 summary": class_results_mcc_f1,
            "Average Precision sem": class_results_ap_se,
            "ROC-AUC sem": class_results_rocauc_se,
            "MCC-F1 summary sem": class_results_mcc_f1_se,
        }
    )
    per_class_results_df.to_csv(
        eval_output_path / f"per_class_{args.output_filename}.csv", index=False
    )

    with open(
        eval_output_path / f"model_2_class_2_pr_vals{args.output_filename}.pkl", "wb"
    ) as file:
        pickle.dump(model_2_class_2_pr_vals, file)

    with open(
        eval_output_path / f"model_2_class_2_metric_vals_{args.output_filename}.pkl",
        "wb",
    ) as file:
        pickle.dump(
            (
                model_2_class_2_ap_vals,
                model_2_class_2_rocauc_vals,
                model_2_class_2_mccf1_vals,
            ),
            file,
        )


def compute_mean_and_standard_error(
    model_2_class_2_vals: dict[str, list[dict[str, float]]],
) -> dict:
    """
    Function to compute the mean and standard error for each model's class metrics.

    :param model_2_class_2_vals: A dictionary mapping model names to a list of dictionaries, each containing class metric values.

    :return: A dictionary mapping each model name to a tuple containing the mean metric value and its standard error.
    """
    model_2_class_mean_and_variance = defaultdict(list)
    model_2_mean_se: dict = defaultdict()

    for model, class_2_vals in model_2_class_2_vals.items():
        class_2_per_fold_vals = defaultdict(list)
        for class_2_val in class_2_vals:
            for class_name, val in class_2_val.items():
                class_2_per_fold_vals[class_name].append(val)
        for class_name, vals in class_2_per_fold_vals.items():
            metric_mean = np.mean(vals)
            metric_variance = np.var(vals, ddof=1)
            model_2_class_mean_and_variance[model].append(
                (class_name, metric_mean, metric_variance)
            )

    for model, values_per_class in model_2_class_mean_and_variance.items():
        total_mean = 0.0
        total_variance = 0.0
        for class_name, mean, variance in values_per_class:
            total_mean += float(mean)
            total_variance += float(variance)
        mean_final = total_mean / len(values_per_class)
        sem = np.sqrt(total_variance) / len(values_per_class)
        model_2_mean_se[model] = (mean_final, sem)
    return model_2_mean_se
