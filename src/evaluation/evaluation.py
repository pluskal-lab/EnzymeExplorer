"""File containing experiment evaluation"""
import json
from collections import defaultdict

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
import numpy as np
import pandas as pd

# from src.evaluation.plotting import (
#     plot_avg_pr_curves_per_class,
#     compute_and_plot_map,
#     plot_ap_per_class,
# )
#
from src.experiments_orchestration.experiment_selector import (
    collect_single_experiment_arguments,
    discover_experiments_from_configs,
)
from src.utils.data import get_tps_df, get_unsplittable_targets
from src.utils.project_info import (
    ExperimentInfo,
    get_experiments_output,
    get_evaluations_output,
    get_config_root,
)
import argparse


def eval_experiment(
    experiment_info: ExperimentInfo,
    classes: list[str],
    target_col: str = "Type (mono, sesq, di, …)",
    include_other_tps: bool = True,
) -> tuple[dict, dict]:
    """
    This function evaluates results of the specified experiment
    :param experiment_info:
    :param classes: list of classes to focus on (eval only for the defined classes)
    :param target_col: name of the target column
    :returns model_class_2_ap_vals (computed average precision values)
            and model_class_2_pr (mapping model and class to precision recall curves)
    """
    # retrieve the model class
    experiment_output_folder = (
        get_experiments_output()
        / experiment_info.validation_schema
        / experiment_info.model_type
        / experiment_info.model_version
    )
    split_description, _ = experiment_info.validation_schema.rsplit("_", 1)
    unsplittable_targets = get_unsplittable_targets(split_desc=split_description)
    classes = [
        target_name
        for target_name in classes
        if target_name not in unsplittable_targets
    ]
    assert (
        experiment_output_folder.exists()
    ), f"Output folder {experiment_output_folder} for {experiment_info} does not exist"

    with open(experiment_output_folder / "model_display_name.txt", "r") as file:
        model_display_name = file.read()

    model_class_2_ap_vals = defaultdict(list)
    model_class_2_pr = defaultdict(list)

    df_ground_truth = get_tps_df(supported_classes=classes, target_col_name=target_col)
    gt_type = df_ground_truth[["Uniprot ID", target_col]].drop_duplicates()
    gt_type = gt_type.groupby("Uniprot ID")[target_col].agg(set).reset_index()

    for class_i, class_name in enumerate(
        classes + (["other"] if include_other_tps else [])
    ):
        for results_path in experiment_output_folder.glob("pred_df*"):
            assert (
                results_path.exists()
            ), f"Expected {results_path} but it does not exis"

            predictions_df = pd.read_hdf(results_path)
            assert isinstance(
                predictions_df, pd.DataFrame
            ), "Expected predictions to be stored as a pandas DataFrame"
            predictions_df = (
                predictions_df[
                    [
                        col
                        for col in predictions_df.columns
                        if "pred_" in col or col == "Uniprot ID"
                    ]
                ]
                .drop_duplicates(subset="Uniprot ID")
                .merge(gt_type, on="Uniprot ID")
            )

            if "alphafold" in experiment_info.model_type.lower():
                raise NotImplementedError

            y_true = predictions_df[target_col].map(lambda x: class_name in x)
            if y_true.sum() == 0:
                continue
            if f"pred_{class_name}" in predictions_df.columns:
                y_pred = predictions_df[f"pred_{class_name}"].fillna(0)
            else:
                y_pred = np.zeros_like(y_true)

            precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

            ap = average_precision_score(y_true, y_pred)
            rocauc = roc_auc_score(y_true, y_pred)
            print(
                "class",
                class_name,
                "ap",
                ap,
                "roc-auc",
                rocauc,
                "pred mean",
                np.mean(y_pred),
                "true mean",
                np.mean(y_true),
            )
            model_class_2_ap_vals[(model_display_name, class_name)].append((ap, rocauc))
            model_class_2_pr[(model_display_name, class_name)].append(
                (precision, recall, thresholds)
            )

    return model_class_2_ap_vals, model_class_2_pr


def get_threshold_per_class(
    model_class_2_pr: dict,
    classes: list[str],
    precision_threshold: float = 0.9,
    include_other_tps: bool = True,
):
    model_class_2_threshold = defaultdict(float)
    for class_name in classes + (["other"] if include_other_tps else []):
        for (model, class_), vals in model_class_2_pr.items():
            if class_ == class_name:
                for (precisions, _, thresholds) in vals:
                    for precision, threshold in zip(precisions, thresholds):
                        if precision > precision_threshold:
                            model_class_2_threshold[
                                f"{model}_{class_name}"
                            ] += threshold / len(vals)
                            break
    return model_class_2_threshold


def get_precision_recall_f1_at_threshold(
    model_class_2_pr: dict,
    class_name: str,
    model_name: str,
    threshold: float = 0.5,
):
    precisions_list, recalls_list, f1_scores_list = [], [], []
    for (model_, class_), vals in model_class_2_pr.items():
        if class_ == class_name and model_ == model_name:
            for (precisions, recalls, thresholds) in vals:
                for i, (precision, recall, _threshold) in enumerate(
                    zip(precisions, recalls, thresholds)
                ):
                    if _threshold >= threshold:
                        f1_score = 2 * precision * recall / (precision + recall)
                        precisions_list.append(precision)
                        recalls_list.append(recall)
                        f1_scores_list.append(f1_score)
                        break

    return (
        np.mean(precisions_list) if len(precisions_list) else np.nan,
        np.mean(recalls_list) if len(recalls_list) else np.nan,
        np.mean(f1_scores_list) if len(f1_scores_list) else np.nan,
    )


def evaluate_selected_experiments(args: argparse.Namespace):
    """
    This functions evaluates outputs of the experiments which are enabled in the configs (no .ignore suffix) or the selected experiment
    :param args: parsed argparse name space
    """

    config_root_path = get_config_root()
    if args.select_single_experiment:
        experiment_kwargs = collect_single_experiment_arguments(config_root_path)
        experiment_info = ExperimentInfo(**experiment_kwargs)
        try:
            model_class_2_ap_vals, model_class_2_pr = eval_experiment(
                experiment_info,
                classes=args.classes,
                include_other_tps=not args.exclude_other_tps,
            )
        except AssertionError:
            raise NotImplementedError(
                f"Please run corresponding experiments ({experiment_info}) before evaluation"
            )
        validation_2_model_class_2_ap_vals, validation_2_model_class_2_pr = {}, {}
        validation_2_model_class_2_ap_vals[
            experiment_info.validation_schema
        ] = model_class_2_ap_vals
        validation_2_model_class_2_pr[
            experiment_info.validation_schema
        ] = model_class_2_pr
    else:
        all_enabled_experiments_df = discover_experiments_from_configs(config_root_path)
        validation_2_model_class_2_ap_vals, validation_2_model_class_2_pr = defaultdict(
            dict
        ), defaultdict(dict)
        for _, experiment_info_row in all_enabled_experiments_df.iterrows():
            experiment_info = ExperimentInfo(**experiment_info_row.to_dict())
            print("experiment_info", experiment_info)
            try:
                model_class_2_ap_vals, model_class_2_pr = eval_experiment(
                    experiment_info,
                    classes=args.classes,
                    include_other_tps=not args.exclude_other_tps,
                )
            except AssertionError:
                raise NotImplementedError(
                    f"Please run corresponding experiments ({experiment_info}) before evaluation"
                )
            validation_2_model_class_2_ap_vals[
                experiment_info.validation_schema
            ].update(model_class_2_ap_vals)
            validation_2_model_class_2_pr[experiment_info.validation_schema].update(
                model_class_2_pr
            )

    all_results_validation = []
    all_results_model = []
    all_results_map = []
    all_results_map_minus_se = []
    all_results_map_plus_se = []
    all_results_mean_rocauc = []
    all_results_mean_rocauc_minus_se = []
    all_results_mean_rocauc_plus_se = []

    class_results_validation_schema = []
    class_results_model = []
    class_results_class_name = []
    class_results_ap = []
    class_results_rocauc = []
    class_results_precision_best_f1 = []
    class_results_recall_best_f1 = []
    class_results_f1_best_f1 = []
    class_results_precision_90 = []
    class_results_recall_90 = []
    class_results_f1_90 = []
    class_results_precision_10 = []
    class_results_recall_10 = []
    class_results_f1_10 = []

    eval_output_path = get_evaluations_output()

    for validation_schema in validation_2_model_class_2_ap_vals.keys():
        model_class_2_pr = validation_2_model_class_2_pr[validation_schema]
        supported_models = sorted(
            {model_display_name for model_display_name, _ in model_class_2_pr.keys()}
        )
        output_root = get_evaluations_output() / validation_schema
        if not output_root.exists():
            output_root.mkdir(parents=True)
        output_root = str(output_root)
        # plot_avg_pr_curves_per_class(
        #     model_class_2_pr,
        #     args.classes,
        #     validation_schema,
        #     output_root,
        #     supported_models,
        #     include_other_tps=not args.exclude_other_tps,
        # )

        model_class_2_ap_vals = validation_2_model_class_2_ap_vals[validation_schema]

        for model_name in supported_models:
            for class_name in args.classes:
                try:
                    (
                        precision_10,
                        recall_10,
                        f1_score_10,
                    ) = get_precision_recall_f1_at_threshold(
                        model_class_2_pr,
                        class_name,
                        model_name=model_name,
                        threshold=0.1,
                    )
                except TypeError:
                    print(
                        f"Issue when extracting precision/recall/f1 score @ 0.1 for model {model_name}, class {class_name}"
                    )
                try:
                    fold_results = model_class_2_pr[(model_name, class_name)]
                    best_f1_thresholds = []
                    for precision, recall, thresholds in fold_results:
                        f1_scores = 2 * recall * precision / (recall + precision)
                        best_f1_thresholds.append(thresholds[np.argmax(f1_scores)])

                    best_f1_threshold = np.mean(best_f1_thresholds)
                    (
                        precision_best_f1,
                        recall_best_f1,
                        f1_score_best_f1,
                    ) = get_precision_recall_f1_at_threshold(
                        model_class_2_pr,
                        class_name,
                        model_name=model_name,
                        threshold=best_f1_threshold,
                    )
                except TypeError:
                    print(
                        f"Issue when extracting precision/recall/f1 score @ best_f1 for model {model_name}, class {class_name}"
                    )
                try:
                    (
                        precision_90,
                        recall_90,
                        f1_score_90,
                    ) = get_precision_recall_f1_at_threshold(
                        model_class_2_pr,
                        class_name,
                        model_name=model_name,
                        threshold=0.9,
                    )
                except TypeError:
                    print(
                        f"Issue when extracting precision/recall/f1 score @ 0.9 for model {model_name}, class {class_name}"
                    )

                class_results_validation_schema.append(validation_schema)
                class_results_model.append(model_name)
                class_results_class_name.append(class_name)
                aps = [
                    ap for ap, rocauc in model_class_2_ap_vals[(model_name, class_name)]
                ]
                rocaucs = [
                    rocauc
                    for ap, rocauc in model_class_2_ap_vals[(model_name, class_name)]
                ]
                class_results_ap.append(np.mean(aps))
                class_results_rocauc.append(np.mean(rocaucs))
                class_results_precision_best_f1.append(precision_best_f1)
                class_results_recall_best_f1.append(recall_best_f1)
                class_results_f1_best_f1.append(f1_score_best_f1)
                class_results_precision_90.append(precision_90)
                class_results_recall_90.append(recall_90)
                class_results_f1_90.append(f1_score_90)
                class_results_precision_10.append(precision_10)
                class_results_recall_10.append(recall_10)
                class_results_f1_10.append(f1_score_10)

        # plot_ap_per_class(
        #     model_class_2_ap_vals,
        #     validation_schema,
        #     output_root,
        #     supported_models,
        #     args.classes,
        #     include_other_tps=not args.exclude_other_tps,
        # )

        model_class_2_threshold = get_threshold_per_class(
            model_class_2_pr,
            args.classes,
            include_other_tps=not args.exclude_other_tps,
        )
        with open(
            eval_output_path / f"{validation_schema}_detection_thresholds.json", "w"
        ) as file:
            json.dump(model_class_2_threshold, file)

    #     model_2_mean_se = compute_and_plot_map(
    #         model_class_2_ap_vals,
    #         args.classes,
    #         validation_schema,
    #         output_root,
    #         supported_models,
    #         include_other_tps=not args.exclude_other_tps,
    #     )
    #     for model, (map_mean, sem, rocauc_mean, rocauc_sem) in model_2_mean_se.items():
    #         all_results_validation.append(validation_schema)
    #         all_results_model.append(" ".join(model.split()))
    #         all_results_map.append(map_mean)
    #         all_results_map_minus_se.append(map_mean - sem)
    #         all_results_map_plus_se.append(map_mean + sem)
    #         all_results_mean_rocauc.append(rocauc_mean)
    #         all_results_mean_rocauc_minus_se.append(rocauc_mean - rocauc_sem)
    #         all_results_mean_rocauc_plus_se.append(rocauc_mean + rocauc_sem)
    #
    # all_results_df = pd.DataFrame(
    #     {
    #         "Validation schema": all_results_validation,
    #         "Model": all_results_model,
    #         "Mean Average Precision (mAP)": all_results_map,
    #         "mAP - SEM": all_results_map_minus_se,
    #         "mAP + SEM": all_results_map_plus_se,
    #         "ROC-AUC (macro mean)": all_results_mean_rocauc,
    #         "Mean ROC-AUC - SEM": all_results_mean_rocauc_minus_se,
    #         "Mean ROC-AUC + SEM": all_results_mean_rocauc_plus_se,
    #     }
    # )
    # all_results_df.to_csv(eval_output_path / "all_results.csv", index=False)
    #
    # per_class_results_df = pd.DataFrame(
    #     {
    #         "Validation schema": class_results_validation_schema,
    #         "Model": class_results_model,
    #         "Class": class_results_class_name,
    #         "Average Precision ": class_results_ap,
    #         "ROC-AUC": class_results_rocauc,
    #         "Precision @ bestF1": class_results_precision_best_f1,
    #         "Recall @ bestF1": class_results_recall_best_f1,
    #         "F1 Score @ bestF1": class_results_f1_best_f1,
    #         "Precision @ 0.1": class_results_precision_10,
    #         "Recall @ 0.1": class_results_recall_10,
    #         "F1 Score @ 0.1": class_results_f1_10,
    #         "Precision @ 0.9": class_results_precision_90,
    #         "Recall @ 0.9": class_results_recall_90,
    #         "F1 Score @ 0.9": class_results_f1_90,
    #     }
    # )
    # per_class_results_df.to_csv(eval_output_path / "per_class_results.csv", index=False)
