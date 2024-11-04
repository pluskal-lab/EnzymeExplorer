"""File containing experiment evaluation"""
import argparse
import logging
import pickle
from collections import defaultdict
from typing import Optional
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


from terpeneminer.src.evaluation.metrics import summary_mccf1
from terpeneminer.src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from terpeneminer.src.models.ifaces import BaseConfig
from terpeneminer.src.utils.project_info import (
    ExperimentInfo,
    get_config_root,
    get_evaluations_output,
    get_output_root,
)

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)


def eval_experiment(
    experiment_info: ExperimentInfo,
    target_col: str,
    min_sample_count_for_eval: int,
    n_folds: int,
    classes: list[str],
    id_2_category_path: Optional[str] = None,
) -> tuple[list, list, list, list]:
    """
    Function for evaluating results of the specified experiment

    :param experiment_info: Information about the experiment to evaluate
    :param target_col: The name of the target column in the dataset
    :param min_sample_count_for_eval: The minimum number of samples required for evaluation
    :param n_folds: The number of folds used in the experiment
    :param classes: A list of class names for which separate evaluations are conducted
    :param id_2_category_path: Optional path to a file mapping IDs to categories

    :return: A tuple containing lists of average precision values, ROC AUC values, MCC F1 values, and precision-recall curves
    """
    # retrieve the model class
    experiment_output_folder_root = (
        get_output_root() / experiment_info.model_type / experiment_info.model_version
    )
    assert (
        experiment_output_folder_root.exists()
    ), f"Output folder {experiment_output_folder_root} for {experiment_info} does not exist"
    # discover available fold results
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
            f"Not all fold outputs found. Please run corresponding experiments ({experiment_info}) before evaluation"
        )

    class_2_ap_vals, class_2_rocauc_vals, class_2_mccf1_vals, class_2_pr_vals = [
        [] for _ in range(4)
    ]

    if id_2_category_path is not None:
        with open(id_2_category_path, "rb") as file:
            id_2_category = pickle.load(file)
    else:
        id_2_category = None
    # pylint: disable=R1702
    for fold_i, fold_root_dir in fold_2_root_dir.items():
        logger.info("Processing fold %d with root dir %s", fold_i, str(fold_root_dir))
        class_2_ap = {}
        class_2_mccf1 = {}
        class_2_auc = {}
        class_2_pr = {}
        for class_name in classes:
            if class_name not in {"Unknown", "other"}:
                if (fold_root_dir / f"{class_name}").exists():
                    fold_class_path = fold_root_dir / f"{class_name}"
                elif (fold_root_dir / "all_classes").exists():
                    fold_class_path = fold_root_dir / "all_classes"
                else:
                    fold_class_path = None
                logger.info(
                    "Processing class %s with path %s", class_name, str(fold_class_path)
                )
                if fold_class_path is not None:
                    try:
                        fold_class_latest_path = sorted(fold_class_path.glob("*"))[-1]
                    except IndexError as index_error:
                        raise NotImplementedError(
                            f"Please run corresponding experiments ({experiment_info}) before evaluation"
                        ) from index_error
                    try:
                        with open(
                            fold_class_latest_path / f"fold_{fold_i}_results.pkl", "rb"
                        ) as file:
                            val_proba_np, class_names_in_fold, test_df = pickle.load(
                                file
                            )
                        if class_name in class_names_in_fold:
                            if not isinstance(class_names_in_fold, list):
                                class_names_in_fold = list(class_names_in_fold)
                            y_true = test_df[target_col].map(lambda x: class_name in x)
                            y_pred = val_proba_np[
                                :, class_names_in_fold.index(class_name)
                            ]
                            current_categories = test_df["Uniprot ID"].map(
                                lambda x: id_2_category.get(x, "Unknown")
                                if id_2_category is not None
                                else ""
                            )
                            for category in set(current_categories).difference(
                                {"Unknown"}
                            ):
                                is_category_bool = current_categories.isin(
                                    {category, "Unknown"}
                                )
                                y_true_category = y_true[is_category_bool]
                                y_pred_category = y_pred[is_category_bool]
                                class_name_to_record = (
                                    class_name
                                    if category == ""
                                    else f"{category}_|_{class_name}"
                                )
                                if y_true_category.sum() >= min_sample_count_for_eval:
                                    average_precision = average_precision_score(
                                        y_true_category, y_pred_category
                                    )
                                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ', class_name_to_record, y_pred_category[y_true_category == 1].mean(), y_pred_category[y_true_category == 0].mean())
                                    mccf1 = summary_mccf1(
                                        y_true_category, y_pred_category
                                    )["mccf1_metric"]
                                    auc = roc_auc_score(
                                        y_true_category, y_pred_category
                                    )
                                    class_2_mccf1[class_name_to_record] = mccf1
                                    class_2_ap[class_name_to_record] = average_precision
                                    class_2_auc[class_name_to_record] = auc
                                    class_2_pr[
                                        class_name_to_record
                                    ] = precision_recall_curve(
                                        y_true_category, y_pred_category
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
