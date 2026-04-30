"""Precision-recall and ROC curves for the evaluation pipeline.

Curves are drawn on the fold-pooled (labels, preds) pair per
(classifier, class), matching the convention of the legacy notebooks.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.metrics import precision_recall_curve, roc_curve  # type: ignore

from enzymeexplorer.src.evaluation.io import FoldDfs
from enzymeexplorer.src.evaluation.plotting import theme

ClassifierClassFoldDfs = dict[str, dict[str, dict[int, FoldDfs]]]
PooledClassifierClass = dict[str, dict[str, FoldDfs]]


def pool_fold_dfs(
    classifier_to_class_to_fold_dfs: ClassifierClassFoldDfs,
) -> PooledClassifierClass:
    """Concatenate all folds into one ``(labels_df, preds_df)`` per
    (classifier, class). Resets row indices."""
    pooled: PooledClassifierClass = {}
    for clf, cls_map in classifier_to_class_to_fold_dfs.items():
        pooled[clf] = {}
        for cls, fold_dfs in cls_map.items():
            labels = pd.concat(
                [pair[0] for pair in fold_dfs.values()], ignore_index=True
            )
            preds = pd.concat(
                [pair[1] for pair in fold_dfs.values()], ignore_index=True
            )
            pooled[clf][cls] = (labels, preds)
    return pooled


def _pr_with_f1_optimum(y_true: np.ndarray, y_score: np.ndarray):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.nanargmax(f1))
    best_threshold = float(thresholds[min(best, len(thresholds) - 1)])
    return precision, recall, f1, best, best_threshold


def plot_pr_curves(
    pooled: PooledClassifierClass,
    target_class: str,
    *,
    classifier_order: Iterable[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    mark_f1_optimum: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One PR curve per classifier for ``target_class``."""
    classifiers = (
        list(classifier_order)
        if classifier_order is not None
        else [c for c in pooled if target_class in pooled[c]]
    )
    palette = palette if palette is not None else theme.model_family_palette(classifiers)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for clf in classifiers:
        labels, preds = pooled[clf][target_class]
        y = labels[target_class].to_numpy()
        s = preds[target_class].to_numpy()
        precision, recall, f1, best, _ = _pr_with_f1_optimum(y, s)
        color = palette.get(clf, (0.4, 0.4, 0.4))
        ax.plot(
            recall, precision, label=theme.display_name(clf), color=color, linewidth=1.6
        )
        if mark_f1_optimum:
            ax.scatter(
                recall[best], precision[best], color=color, edgecolor="black",
                zorder=5, s=40,
            )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_title(title if title is not None else f"PR Curves — {target_class}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    pooled: PooledClassifierClass,
    target_class: str,
    *,
    classifier_order: Iterable[str] | None = None,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One ROC curve per classifier for ``target_class``."""
    classifiers = (
        list(classifier_order)
        if classifier_order is not None
        else [c for c in pooled if target_class in pooled[c]]
    )
    palette = palette if palette is not None else theme.model_family_palette(classifiers)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for clf in classifiers:
        labels, preds = pooled[clf][target_class]
        fpr, tpr, _ = roc_curve(
            labels[target_class].to_numpy(), preds[target_class].to_numpy()
        )
        ax.plot(
            fpr, tpr, label=theme.display_name(clf),
            color=palette.get(clf, (0.4, 0.4, 0.4)), linewidth=1.6,
        )
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_title(title if title is not None else f"ROC Curves — {target_class}")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_per_class_pr_curves(
    pooled_classifier: dict[str, FoldDfs],
    classes: list[str],
    *,
    palette: dict[str, tuple[float, float, float]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 7),
    mark_f1_optimum: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """One PR curve per class for a single classifier (already pooled)."""
    palette = palette if palette is not None else theme.class_palette(classes)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for cls in classes:
        if cls not in pooled_classifier:
            continue
        labels, preds = pooled_classifier[cls]
        y = labels[cls].to_numpy()
        s = preds[cls].to_numpy()
        if y.sum() == 0:
            continue
        precision, recall, _, best, _ = _pr_with_f1_optimum(y, s)
        color = palette.get(cls, (0.4, 0.4, 0.4))
        ax.plot(recall, precision, label=cls, color=color, linewidth=1.4)
        if mark_f1_optimum:
            ax.scatter(
                recall[best], precision[best], color=color, edgecolor="black",
                zorder=5, s=30,
            )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.set_title(title if title is not None else "PR Curves per class")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    return fig
