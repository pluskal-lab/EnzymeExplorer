"""This module contains metric-related plotting"""
import argparse
import os
from collections import defaultdict
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.utils.project_info import get_evaluations_output


def plot_ap_per_class(
    model_class_2_ap_vals: Union[dict, defaultdict],
    validation_schema: str,
    output_root: str,
    supported_models: list[str],
    classes: Optional[list[str]] = None,
    include_other_tps: bool = True,
):
    """
    :param supported_models: a list of supported models
    :param model_class_2_ap_vals: computed average precision values
    :param validation_schema: name of validation schema
    :param output_root: root folder to store image
    :param classes: list of classes to focus on (eval only for the defined classes)
    """
    for class_name in classes + (["fother"] if include_other_tps else []):
        if isinstance(class_name, float):
            continue
        model_2_mean_ser = defaultdict()

        for (model, class_), vals in model_class_2_ap_vals.items():
            if class_ == class_name:
                vals = [ap for ap, _ in vals]
                ap_mean = np.mean(vals)
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals))

                model_2_mean_ser[model] = (ap_mean, sem)

        fig, ax = plt.subplots(figsize=(22, 15))
        fig.patch.set_facecolor("white")
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(supported_models)))
        colors = [
            colors[supported_models.index(model)] for model in model_2_mean_ser.keys()
        ]
        xticks = list(range(len(model_2_mean_ser)))
        means = list(map(lambda x: x[0], model_2_mean_ser.values()))
        yerr = np.empty((2, len(means)))
        for i, (_, sem) in enumerate(model_2_mean_ser.values()):
            yerr[0, i] = sem
            yerr[1, i] = sem

        ax.bar(xticks, means, yerr=yerr, color=colors)

        for i, ap in enumerate(means):
            ax.text(
                i - 0.4,
                ap + 0.02,
                f"{ap:.2f}",
                color=colors[i],
                fontweight="bold",
                fontsize=15,
            )
        ax.set_xticks(xticks)
        ax.set_xticklabels(list(model_2_mean_ser.keys()), fontsize=12, rotation=90)
        ax.set_xlabel("Model", fontsize=14)
        ax.set_ylabel("AP", fontsize=14)
        ax.set_title(f"Comparing AP for class {class_name.upper()}", fontsize=19)
        ax.set_ylim([-0.01, 1.05])
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_root, f"ap_comparison_{validation_schema}_{class_name}.svg"
            )
        )


def compute_and_plot_map(
    model_class_2_ap_vals,
    classes: list[str],
    validation_schema: str,
    output_root: str,
    supported_models: list[str],
    metric_name: str = "mAP",
    include_other_tps: bool = True,
) -> dict:
    """
    :param supported_models: a list of supported models
    :param model_class_2_ap_vals: computed average precision values
    :param validation_schema: name of validation schema
    :param output_root: root folder to store image
    :param classes: list of classes to focus on (eval only for the defined classes)
    :param metric_name:
    :return: model_2_mean_se, mapping from model name to mean and (mean - se, mean + se) interval
    """
    model_fold_2_ap = defaultdict(float)
    model_fold_2_rocauc = defaultdict(float)
    model_2_ap = defaultdict(list)
    model_2_rocauc = defaultdict(list)
    model_2_mean_se = defaultdict()

    _, vals = list(model_class_2_ap_vals.items())[0]
    n_folds = len(vals)
    for fold_i in range(n_folds):
        for (model, class_), vals in model_class_2_ap_vals.items():
            if fold_i >= len(vals):
                continue
            ap, rocauc = vals[fold_i]
            model_fold_2_ap[(model, fold_i)] += ap
            model_fold_2_rocauc[(model, fold_i)] += rocauc

    for (model, _), fold_val in model_fold_2_ap.items():
        model_2_ap[model].append(fold_val / (len(classes) + int(include_other_tps)))
    for (model, _), fold_val in model_fold_2_rocauc.items():
        model_2_rocauc[model].append(fold_val / (len(classes) + int(include_other_tps)))

    for model, vals in model_2_ap.items():
        ap_mean = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
        rocauc_vals = model_2_rocauc[model]
        rocauc_mean = np.mean(rocauc_vals)
        rocauc_sem = np.std(rocauc_vals, ddof=1) / np.sqrt(len(rocauc_vals))
        model_2_mean_se[model] = (ap_mean, sem, rocauc_mean, rocauc_sem)

    # model_2_mean_se = dict(
    #     sorted(
    #         model_2_mean_se.items(),
    #         key=lambda el: [
    #             "BLAST-based\nmatching",
    #             "Profile HMM",
    #             "BERT\n+ MLP",
    #             "BERT\n+ Random Forest",
    #         ].index(el[0]),
    #     )
    # )

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(supported_models)))
    colors = [colors[supported_models.index(model)] for model in model_2_mean_se.keys()]
    xticks = list(range(len(model_2_mean_se)))
    means = list(map(lambda x: x[0], model_2_mean_se.values()))
    yerr = np.empty((2, len(means)))
    for i, (_, sem, _, _) in enumerate(model_2_mean_se.values()):
        yerr[:, i] = sem

    ax.bar(xticks, means, yerr=yerr, color=colors)

    for i, ap in enumerate(means):
        ax.text(
            i - 0.4,
            ap + 0.02,
            f"{ap:.3f}",
            color=colors[i],
            fontweight="bold",
            fontsize=15,
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [
            tick if "HMM" not in tick else "Profile HMM"
            for tick in model_2_mean_se.keys()
        ],
        fontsize=12,
        rotation=0,
    )
    # ax.set_xticklabels(list(model_2_mean_se.keys()), fontsize=12, rotation=70)
    ax.set_xlabel("Model", fontsize=14)
    metric_name = "mAP (~Area under Precision-Recall Curve)"
    ax.set_ylabel(metric_name, fontsize=14)

    if metric_name == "mAP\n(~Area under Precision-Recall Curve)":
        ax.set_ylim([-0.01, 1.05])
    ax.set_title(f"Comparing Performance of Different models", fontsize=18)
    # ax.set_title(f"Comparing {metric_name}, data: {validation_schema}", fontsize=19)
    plt.tight_layout()
    fig.savefig(os.path.join(output_root, f"mAP_comparison_{validation_schema}.svg"))

    #####################################
    ########### ROC stats ###############
    #####################################
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(supported_models)))
    colors = [colors[supported_models.index(model)] for model in model_2_mean_se.keys()]
    xticks = list(range(len(model_2_mean_se)))
    means = list(map(lambda x: x[2], model_2_mean_se.values()))
    yerr = np.empty((2, len(means)))
    for i, (_, _, _, sem) in enumerate(model_2_mean_se.values()):
        yerr[:, i] = sem

    ax.bar(xticks, means, yerr=yerr, color=colors)

    for i, rocauc in enumerate(means):
        ax.text(
            i - 0.4,
            rocauc + 0.02,
            f"{rocauc:.3f}",
            color=colors[i],
            fontweight="bold",
            fontsize=15,
        )
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [
            tick if "HMM" not in tick else "Profile HMM"
            for tick in model_2_mean_se.keys()
        ],
        fontsize=12,
        rotation=90,
    )
    # ax.set_xticklabels(list(model_2_mean_se.keys()), fontsize=12, rotation=70)
    ax.set_xlabel("Model", fontsize=14)
    metric_name = "ROC-AUC (Area under the ROC Curve)"
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_ylim([-0.01, 1.05])
    ax.set_title(f"Comparing Performance of Different models", fontsize=19)
    # ax.set_title(f"Comparing {metric_name}, data: {validation_schema}", fontsize=19)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, f"rocauc_comparison_{validation_schema}.svg"))
    return model_2_mean_se


def plot_avg_pr_curves_per_class(
    model_class_2_pr: dict,
    classes: list[str],
    validation_schema: str,
    output_root: str,
    supported_models: list[str],
    include_other_tps: bool = True,
):
    """
    :param supported_models: a list of supported models
    :param model_class_2_pr: mapping model and class to precision recall curves
    :param validation_schema: name of validation schema
    :param output_root: root folder to store image
    :param classes: list of classes to focus on (eval only for the defined classes)
    """
    base_recall = np.linspace(0, 1, 101)[::-1]
    for class_name in classes + (["other"] if include_other_tps else []):

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        fig.patch.set_facecolor("white")
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(supported_models)))
        for (model, class_), vals in model_class_2_pr.items():
            if class_ == class_name:
                precisions = []
                for precision, recall, _ in vals:
                    f = interp1d(recall, precision)
                    precision_base = f(base_recall)
                    precisions.append(precision_base)
                precisions = np.array(precisions)
                mean_precisions = precisions.mean(axis=0)

                std = precisions.std(axis=0)

                precisions_upper = np.minimum(mean_precisions + std, 1)
                precisions_lower = mean_precisions - std
                model_i = supported_models.index(model)
                ax.plot(
                    base_recall, mean_precisions, color=colors[model_i], label=model
                )
                ax.fill_between(
                    base_recall,
                    precisions_lower,
                    precisions_upper,
                    color=colors[model_i],
                    alpha=0.05,
                )

        ax.set_xlabel("Recall", fontsize=15)
        ax.set_ylabel("Precision", fontsize=15)
        ax.set_title(
            "Comparison of PR curves for different protein language models",  # f"Comparison of PR curves\nClass: {class_name}, Data: {validation_schema}",
            fontsize=20,
        )
        ax.legend(fontsize=12, labelspacing=0.9)
        ax.set_ylim([-0.01, 1.05])
        ax.set_xlim([-0.01, 1.05])
        plt.savefig(
            os.path.join(
                output_root, f"agg_pr_curves_{class_name}_data_{validation_schema}.svg"
            )
        )


def plot_bars(means, std, model_names_plots, title, metric_name, output_path, highlighted_count = 1):
    """A function for plotting bars"""
    # set global parameters
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # prepare some data for drawing figures

    np.random.seed(42)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.bar(list(range(len(model_names_plots))), means, yerr=std, align='center',
                alpha=0.5, ecolor='black', capsize=10, color=['lightskyblue'] * (len(model_names_plots) - highlighted_count) + ['dodgerblue'] * highlighted_count)

    ax.set_xticks(list(range(len(model_names_plots))))
    ax.set_ylim([0, 1.001])
    ax.set_xticklabels(model_names_plots, rotation=90)
    ax.set_ylabel(metric_name, fontsize=15)
    ax.set_title(title, fontsize=19)
    plt.savefig(output_path, bbox_inches='tight')


def plot_selected_results(args: argparse.Namespace):
    """
    This functions picks experiments which are enabled in the configs (no .ignore suffix)
    and then it generates all possible hyperparameter tuning configuration, separately for each fold.
    :param args: parsed argparse name space
    """
    eval_output_path = get_evaluations_output()
    results_df = pd.read_csv(eval_output_path / f"{args.eval_output_filename}.csv").set_index("Model")
    plots_name = args.subset_name if args.subset_name is not None else args.eval_output_filename
    for metric, column_group in zip(['Mean Average Precision', 'ROC-AUC', 'MCC-F1 summary'],
                                    [['Mean Average Precision (mAP)', 'mAP - SEM', 'mAP + SEM'],
                                     ['ROC-AUC (macro mean)', 'Mean ROC-AUC - SEM', 'Mean ROC-AUC + SEM'],
                                     ['MCC-F1 summary (macro mean)', 'Mean MCC-F1 summary - SEM', 'Mean MCC-F1 summary + SEM']]):
        means, stds = [], []
        for model in args.models:
            mean_val = results_df.loc[model, column_group[0]]
            std_val = mean_val - results_df.loc[model, column_group[1]]
            means.append(mean_val)
            stds.append(std_val)
        plot_bars(means, stds, args.model_names, "TPS classes detection", metric, eval_output_path / f"{plots_name}_{metric}.png")



