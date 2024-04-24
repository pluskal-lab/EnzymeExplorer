"""File containing experiment evaluation"""
import argparse
import logging
import pickle
from collections import defaultdict

from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from src.evaluation.metrics import summary_mccf1
from src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from src.models.ifaces import BaseConfig
from src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_evaluations_output,
    get_output_root,
)

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def eval_experiment(
    experiment_info: ExperimentInfo,
    target_col: str,
    min_sample_count_for_eval: int,
) -> tuple[list, list, list, list]:
    """
    This function evaluates results of the specified experiment
    """
    # retrieve the model class
    experiment_output_folder_root = (
        get_output_root()
        / experiment_info.model_type
        / experiment_info.model_version
        / experiment_info.fold
    )
    assert (
        experiment_output_folder_root.exists()
    ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"
    try:
        experiment_output_folder = sorted(experiment_output_folder_root.glob("*"))[-1]
    except IndexError as index_error:
        raise NotImplementedError(
            f"Please run corresponding experiments ({experiment_info}) before evaluation"
        ) from index_error

    class_2_ap_vals, class_2_rocauc_vals, class_2_mccf1_vals, map_vals = [
        [] for _ in range(4)
    ]

    for fold_path in experiment_output_folder.glob("fold_*_results.pkl"):
        with open(fold_path, "rb") as file:
            val_proba_np, class_names, test_df = pickle.load(file)
        domain_class_2_ap = {}
        domain_class_2_mccf1 = {}
        domain_class_2_auc = {}
        for class_i, class_name in enumerate(class_names):
            if class_name not in {"Unknown", "other", "precursor substr"}:
                y_true = test_df[target_col].map(lambda x: class_name in x)
                y_pred = val_proba_np[:, class_i]

                if y_true.sum() >= min_sample_count_for_eval:
                    average_precision = average_precision_score(y_true, y_pred)
                    mccf1 = summary_mccf1(y_true, y_pred)["mccf1_metric"]
                    auc = roc_auc_score(y_true, y_pred)
                    domain_class_2_mccf1[class_name] = mccf1
                    domain_class_2_ap[class_name] = average_precision
                    domain_class_2_auc[class_name] = auc
        class_2_ap_vals.append(domain_class_2_ap)
        class_2_mccf1_vals.append(domain_class_2_mccf1)
        class_2_rocauc_vals.append(domain_class_2_auc)
        map_fold = np.mean(
            [val for key, val in class_2_ap_vals[-1].items() if key != "is_TPS"]
        )
        map_vals.append(map_fold)
    logger.info(
        "For experiment %s mAP: %.3f +/- %.3f')",
        experiment_info,
        np.mean(map_vals),
        np.std(map_vals),
    )
    return class_2_ap_vals, class_2_rocauc_vals, class_2_mccf1_vals, map_vals


def evaluate_selected_experiments(args: argparse.Namespace):
    """
    This functions evaluates outputs of the experiments which are enabled in the configs (no .ignore suffix) or the selected experiment
    :param args: parsed argparse name space
    """

    config_root_path = get_config_root()
    (
        model_2_class_2_ap_vals,
        model_2_class_2_rocauc_vals,
        model_2_class_2_mccf1_vals,
    ) = [{} for _ in range(3)]
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
                _,
            ) = eval_experiment(
                experiment_info,
                target_col=config_dict["target_col_name"],
                min_sample_count_for_eval=args.minimal_count_to_eval,
            )
        except AssertionError as error:
            raise NotImplementedError(
                f"Please run corresponding experiments ({experiment_info}) before evaluation"
            ) from error
        model_name = f"{experiment_info.model_type}__{experiment_info.model_version}"
        model_2_class_2_ap_vals[model_name] = class_2_ap_vals
        model_2_class_2_rocauc_vals[model_name] = class_2_rocauc_vals
        model_2_class_2_mccf1_vals[model_name] = class_2_mccf1_vals
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
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
                    _,
                ) = eval_experiment(
                    experiment_info,
                    target_col=config_dict["target_col_name"],
                    min_sample_count_for_eval=args.minimal_count_to_eval,
                )
            except (AssertionError, NotImplementedError):
                # raise NotImplementedError(
                #     f"Please run corresponding experiments ({experiment_info}) before evaluation"
                # )
                continue
            model_name = (
                f"{experiment_info.model_type}__{experiment_info.model_version}"
            )
            model_2_class_2_ap_vals[model_name] = class_2_ap_vals
            model_2_class_2_rocauc_vals[model_name] = class_2_rocauc_vals
            model_2_class_2_mccf1_vals[model_name] = class_2_mccf1_vals

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

    eval_output_path = get_evaluations_output()
    if not eval_output_path.exists():
        eval_output_path.mkdir(parents=True)

    for model_name in model_2_class_2_ap_vals.keys():
        for class_name in args.classes:
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
    all_results_df.to_csv(eval_output_path / "all_results.csv", index=False)

    per_class_results_df = pd.DataFrame(
        {
            "Model": class_results_model,
            "Class": class_results_class_name,
            "Average Precision ": class_results_ap,
            "ROC-AUC": class_results_rocauc,
            "MCC-F1 summary": class_results_mcc_f1,
        }
    )
    per_class_results_df.to_csv(eval_output_path / "per_class_results.csv", index=False)


def compute_mean_and_standard_error(
    model_2_class_2_vals: dict[str, list[dict[str, float]]],
) -> dict:
    """
    :param model_2_class_2_vals: computed metrics values
    :return: model_2_mean_se, mapping from model name to mean and (mean - se, mean + se) interval
    """
    model_fold_2_metric_mean_per_fold = defaultdict(list)
    model_2_metric_macro_mean = defaultdict(list)
    model_2_mean_se: dict = defaultdict()

    for model, class_2_vals in model_2_class_2_vals.items():
        for class_2_val in class_2_vals:
            model_fold_2_metric_mean_per_fold[model].append(
                np.mean(list(class_2_val.values()))
            )

    for model, fold_vals in model_fold_2_metric_mean_per_fold.items():
        model_2_metric_macro_mean[model].append(np.mean(fold_vals))

    for model, vals in model_2_metric_macro_mean.items():
        metric_mean = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
        model_2_mean_se[model] = (metric_mean, sem)
    return model_2_mean_se
