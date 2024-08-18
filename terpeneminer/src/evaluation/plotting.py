"""This module contains metric-related plotting"""
import argparse
import pickle
from pathlib import Path
import logging

logging.getLogger("matplotlib").setLevel(logging.INFO)
import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from seaborn import boxplot, barplot  # type: ignore

from terpeneminer.src.utils.project_info import get_evaluations_output

# set global parameters
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def plot_avg_pr_curves_per_class(
    model_class_2_pr: dict,
    class_names: list[str],
    title: str,
    output_path: str | Path,
    supported_models: list[str],
    model_names: list[str],
):
    """
    Function to plot average precision-recall curves for each class.

    :param model_class_2_pr: A dictionary mapping models and classes to precision-recall curves.
    :param class_names: A list of class names to focus on for evaluation.
    :param title: The title of the plot.
    :param output_path: The file path where the plot image will be saved.
    :param supported_models: A list of supported models for which the precision-recall curves are plotted.
    :param model_names: A list of model names corresponding to the supported models.

    :return: None. Saves the precision-recall plot to the specified output path.
    """

    base_recall = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.patch.set_facecolor("white")
    colors = plt.get_cmap("tab20" if len(supported_models) > 10 else "tab10")(
        np.linspace(0, 1, len(supported_models))
    )
    for model_i, model in enumerate(supported_models):
        class_2_vals = model_class_2_pr[model]
        precisions = []
        for class_2_vals_fold in class_2_vals:
            for class_, vals in class_2_vals_fold.items():
                if class_ in class_names:
                    precisions_fold, recalls_fold = [
                        list(val[::-1]) for val in vals[:2]
                    ]
                    precisions_fold = precisions_fold[:2] + precisions_fold[1:]
                    recalls_fold = recalls_fold[:1] + recalls_fold
                    precision_base = np.interp(
                        base_recall, recalls_fold, precisions_fold
                    )[::-1]
                    precisions.append(precision_base)

        precisions_np = np.array(precisions)
        mean_precisions = precisions_np.mean(axis=0)

        std = precisions_np.std(axis=0)

        precisions_upper = np.minimum(mean_precisions + std, 1)
        precisions_lower = mean_precisions - std
        recall_plot = base_recall[::-1]
        if mean_precisions[-1] != 1:
            mean_precisions = np.concatenate((mean_precisions, np.ones(1)))
            recall_plot = np.concatenate((recall_plot, np.zeros(1)))
        ax.plot(
            recall_plot,
            mean_precisions,
            color=colors[model_i],
            label=model_names[model_i],
        )
        ax.fill_between(
            base_recall[::-1],
            precisions_lower,
            precisions_upper,
            color=colors[model_i],
            alpha=0.1,
        )

    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)
    ax.set_title(
        title,
        fontsize=20,
    )
    ax.legend(fontsize=12, labelspacing=0.9)
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlim([-0.01, 1.05])
    plt.savefig(output_path, bbox_inches="tight")


def plot_bars(
    means, std, model_names_plots, title, metric_name, output_path, highlighted_count=1
):
    """
    Function to plot a bar chart with error bars representing standard deviation.

    :param means: A list of mean values for each model.
    :param std: A list of standard deviation values for each model.
    :param model_names_plots: A list of model names to be plotted on the x-axis.
    :param title: The title of the bar chart.
    :param metric_name: The name of the metric being plotted, used as the y-axis label.
    :param output_path: The file path where the bar chart image will be saved.
    :param highlighted_count: The number of bars to be highlighted with a different color (default is 1).

    :return: None. Saves the bar chart to the specified output path.
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.bar(
        list(range(len(model_names_plots))),
        means,
        yerr=std,
        align="center",
        alpha=0.5,
        ecolor="black",
        capsize=10,
        color=["lightskyblue"] * (len(model_names_plots) - highlighted_count)
        + ["dodgerblue"] * highlighted_count,
    )

    ax.set_xticks(list(range(len(model_names_plots))))
    ax.set_ylim([0, 1.001])
    ax.set_xticklabels(model_names_plots, rotation=90)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_title(title, fontsize=19)
    plt.savefig(output_path, bbox_inches="tight")


def plot_boxplots_per_type(
    models: list[str],
    substr_2_type: dict,
    model_2_class_2_metric_vals: dict,
    metric_name: str,
    title: str,
    output_path: str | Path,
    model_names_plots: list[str],
):
    """
    Function to plot boxplots of metric values for different types across models.

    :param models: A list of model names.
    :param substr_2_type: A dictionary mapping class names to their corresponding types.
    :param model_2_class_2_metric_vals: A dictionary mapping models to class-specific metric values.
    :param metric_name: The name of the metric to be plotted.
    :param title: The title of the boxplot.
    :param output_path: The file path where the boxplot image will be saved.
    :param model_names_plots: A list of model names to be used for plotting on the x-axis.

    :return: None. Saves the boxplot to the specified output path.
    """

    model_list = []
    class_list = []
    val_list = []

    present_type_names = set()
    for model_i, model_name in enumerate(models):
        class_dicts = model_2_class_2_metric_vals[model_name]
        for fold_dict in class_dicts:
            for class_name, val in fold_dict.items():
                if class_name in substr_2_type:
                    type_name = substr_2_type[class_name]
                    present_type_names.add(type_name)
                    model_list.append(model_names_plots[model_i])
                    class_list.append(type_name)
                    val_list.append(val)
    ap_per_class_df = pd.DataFrame(
        {"Model": model_list, "TPS Type": class_list, metric_name: val_list}
    )
    _, ax = plt.subplots(figsize=(15, 8))
    boxplot(
        x="TPS Type",
        y=metric_name,
        hue="Model",
        data=ap_per_class_df,
        width=0.4,
        ax=ax,
        order=[
            type_name
            for type_name in ["mono", "sesq", "di", "sester", "tri", "tetra"]
            if type_name in present_type_names
        ],
        fliersize=0,
    )
    ax.set_ylim([0, 1.02])
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_xlabel("TPS Type", fontsize=15)
    ax.set_title(title, fontsize=19)
    plt.savefig(output_path, bbox_inches="tight")


def plot_barplots_per_categories(
    models: list[str],
    model_2_class_2_metric_vals: dict,
    categories_order: list[str],
    category_name: str,
    metric_name: str,
    title: str,
    output_path: str | Path,
    model_names_plots: list[str],
):
    """
    Function to plot bar plots of metric values for different categories across models.

    :param models: A list of model names.
    :param model_2_class_2_metric_vals: A dictionary mapping models to class-category-specific metric values.
    :param categories_order: A list specifying the order of categories to be plotted.
    :param category_name: The name of the category for x-axis labeling.
    :param metric_name: The name of the metric to be plotted.
    :param title: The title of the bar plot.
    :param output_path: The file path where the bar plot image will be saved.
    :param model_names_plots: A list of model names to be used for plotting on the x-axis.

    :return: None. Saves the bar plot to the specified output path.
    """

    model_list = []
    category_list = []
    val_list = []

    present_categories = set()
    for model_i, model_name in enumerate(models):
        class_dicts = model_2_class_2_metric_vals[model_name]
        for fold_dict in class_dicts:
            for category_class_name, val in fold_dict.items():
                assert (
                    "_|_" in category_class_name
                ), "The evaluation results provided were not computed per different categories"
                category, _ = category_class_name.split("_|_")
                present_categories.add(category)
                model_list.append(model_names_plots[model_i])
                category_list.append(category)
                val_list.append(val)
    metric_per_category_df = pd.DataFrame(
        {"Model": model_list, category_name: category_list, metric_name: val_list}
    )
    _, ax = plt.subplots(figsize=(15, 8))
    barplot(
        x=category_list,
        y=metric_name,
        hue="Model",
        data=metric_per_category_df,
        width=0.4,
        ax=ax,
        order=[
            category_name
            for category_name in categories_order
            if category_name in present_categories
        ],
    )
    ax.set_ylim([0, 1.02])
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_xlabel(category_name, fontsize=15)
    ax.set_title(title, fontsize=19)
    plt.savefig(output_path, bbox_inches="tight")


def plot_selected_results(args: argparse.Namespace):
    """
    This functions picks experiments which are enabled in the configs (no .ignore suffix)
    and then it generates all possible hyperparameter tuning configuration, separately for each fold.
    :param args: parsed argparse name space
    """
    eval_output_path = get_evaluations_output()
    plots_name = (
        args.subset_name if args.subset_name is not None else args.eval_output_filename
    )
    with open("data/substrate_2_tps_type.pkl", "rb") as file:
        substrate_2_tps_type = pickle.load(file)
    if args.plot_tps_detection:
        detection_specification = (
            "" if args.type_detected == "isTPS" else args.type_detected
        )
        per_class_results_df = pd.read_csv(
            eval_output_path / f"per_class_{args.eval_output_filename}.csv"
        ).set_index("Model")
        for metric, column_group in zip(
            ["Average Precision", "ROC-AUC", "MCC-F1 summary"],
            [
                ["Average Precision", "Average Precision sem"],
                ["ROC-AUC", "ROC-AUC sem"],
                ["MCC-F1 summary", "MCC-F1 summary sem"],
            ],
        ):
            means, stds = [], []
            if args.type_detected == "isTPS":
                series_of_categories_to_check = per_class_results_df["Class"]
            else:
                series_of_categories_to_check = per_class_results_df["Class"].map(
                    substrate_2_tps_type
                )
            current_df = per_class_results_df[
                series_of_categories_to_check == args.type_detected
            ]
            for model in args.models:
                mean_vals = current_df.loc[model, column_group[0]]
                std_vals = current_df.loc[model, column_group[1]]
                if isinstance(mean_vals, np.float64):
                    mean_val = float(mean_vals)
                    std_val = float(std_vals)
                else:
                    total_mean, total_variance = 0, 0
                    for mean, std in zip(mean_vals, std_vals):
                        total_mean += mean
                        total_variance += std**2
                    mean_val = total_mean / len(mean_vals)
                    std_val = np.sqrt(total_variance) / len(mean_vals)
                means.append(mean_val)
                stds.append(std_val)

            plot_bars(
                means,
                stds,
                args.model_names,
                f"{detection_specification}TPS detection",
                metric,
                eval_output_path
                / f"{plots_name}_{metric}_{detection_specification}TPS.png",
            )

        with open(
            eval_output_path
            / f"model_2_class_2_pr_vals{args.eval_output_filename}.pkl",
            "rb",
        ) as file:
            model_2_class_2_pr_vals = pickle.load(file)

        if args.type_detected != "isTPS":
            classes_to_consider = [
                class_name
                for class_name, type_ in substrate_2_tps_type.items()
                if type_ == args.type_detected
            ]
        else:
            classes_to_consider = ["isTPS"]
        plot_avg_pr_curves_per_class(
            model_2_class_2_pr_vals,
            classes_to_consider,
            f"{detection_specification}TPS detection Precision-Recall curves",
            output_path=str(
                eval_output_path / f"{plots_name}_PR_{detection_specification}TPS.png"
            ),
            supported_models=args.models,
            model_names=args.model_names,
        )
    elif args.plot_boxplots_per_type:
        with open(
            eval_output_path
            / f"model_2_class_2_metric_vals_{args.eval_output_filename}.pkl",
            "rb",
        ) as file:
            (
                model_2_class_2_ap_vals,
                model_2_class_2_rocauc_vals,
                model_2_class_2_mccf1_vals,
            ) = pickle.load(file)
        for model_2_class_2_metric_vals, metric_name in zip(
            [
                model_2_class_2_ap_vals,
                model_2_class_2_rocauc_vals,
                model_2_class_2_mccf1_vals,
            ],
            ["Average Precision", "ROC AUC", "MCC-F1 summary"],
        ):
            plot_boxplots_per_type(
                models=args.models,
                substr_2_type=substrate_2_tps_type,
                model_2_class_2_metric_vals=model_2_class_2_metric_vals,
                metric_name=metric_name,
                title="TPS detection per type",
                output_path=eval_output_path
                / f"{plots_name}_{metric_name}_per_type.png",
                model_names_plots=args.model_names,
            )
    elif args.plot_barplots_per_category:
        with open(
            eval_output_path
            / f"model_2_class_2_metric_vals_{args.eval_output_filename}.pkl",
            "rb",
        ) as file:
            (
                model_2_class_2_ap_vals,
                model_2_class_2_rocauc_vals,
                model_2_class_2_mccf1_vals,
            ) = pickle.load(file)
        for model_2_class_2_metric_vals, metric_name in zip(
            [
                model_2_class_2_ap_vals,
                model_2_class_2_rocauc_vals,
                model_2_class_2_mccf1_vals,
            ],
            ["Average Precision", "ROC AUC", "MCC-F1 summary"],
        ):
            plot_barplots_per_categories(
                models=args.models,
                model_2_class_2_metric_vals=model_2_class_2_metric_vals,
                categories_order=args.categories_order,
                category_name=args.category_name,
                metric_name=metric_name,
                title=f"TPS detection per {args.category_name.lower()}",
                output_path=eval_output_path
                / f"{plots_name}_{metric_name}_per_{args.category_name.lower()}.png",
                model_names_plots=args.model_names,
            )
    else:
        results_df = pd.read_csv(
            eval_output_path / f"{args.eval_output_filename}.csv"
        ).set_index("Model")
        for metric, column_group in zip(
            ["Mean Average Precision", "ROC-AUC", "MCC-F1 summary"],
            [
                ["Mean Average Precision (mAP)", "mAP - SEM", "mAP + SEM"],
                ["ROC-AUC (macro mean)", "Mean ROC-AUC - SEM", "Mean ROC-AUC + SEM"],
                [
                    "MCC-F1 summary (macro mean)",
                    "Mean MCC-F1 summary - SEM",
                    "Mean MCC-F1 summary + SEM",
                ],
            ],
        ):
            means, stds = [], []
            for model in args.models:
                mean_val = results_df.loc[model, column_group[0]]
                std_val = mean_val - results_df.loc[model, column_group[1]]
                means.append(mean_val)
                stds.append(std_val)
            plot_bars(
                means,
                stds,
                args.model_names,
                "TPS classes detection",
                metric,
                eval_output_path / f"{plots_name}_{metric}.png",
            )
