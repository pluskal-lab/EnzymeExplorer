"""This module implements metrics-related utils"""

import numpy as np  # type: ignore
from mcc_f1 import mcc_f1_curve  # type: ignore


def summary_mccf1(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 100):
    """
    MCC-F1 curve based metric
    """
    mcc_nor_truncated, f_truncated, thresholds = mcc_f1_curve(y_true, y_pred)
    index_of_max_mcc = np.argmax(mcc_nor_truncated)

    mcc_left = mcc_nor_truncated[: index_of_max_mcc + 1]
    f_left = f_truncated[: index_of_max_mcc + 1]

    mcc_right = mcc_nor_truncated[index_of_max_mcc + 1 :]
    f_right = f_truncated[index_of_max_mcc + 1 :]

    unit_len = (np.max(mcc_nor_truncated) - np.min(mcc_nor_truncated)) / bins

    mean_distances_left = np.zeros(bins)
    for i in range(bins):
        pos = np.where(
            (mcc_left >= np.min(mcc_nor_truncated) + (i - 1) * unit_len)
            & (mcc_left <= np.min(mcc_nor_truncated) + i * unit_len)
        )[0]
        sum_of_distance_within_subrange = np.sum(
            np.sqrt((mcc_left[pos] - 1) ** 2 + (f_left[pos] - 1) ** 2)
        )
        mean_distances_left[i] = (
            sum_of_distance_within_subrange / len(pos) if len(pos) > 0 else np.nan
        )

    num_of_na_left = np.sum(np.isnan(mean_distances_left))
    sum_of_mean_distances_left_no_na = np.nansum(mean_distances_left)

    mean_distances_right = np.zeros(bins)
    for i in range(bins):
        pos = np.where(
            (mcc_right >= np.min(mcc_nor_truncated) + (i - 1) * unit_len)
            & (mcc_right <= np.min(mcc_nor_truncated) + i * unit_len)
        )[0]
        sum_of_distance_within_subrange = np.sum(
            np.sqrt((mcc_right[pos] - 1) ** 2 + (f_right[pos] - 1) ** 2)
        )
        mean_distances_right[i] = (
            sum_of_distance_within_subrange / len(pos) if len(pos) > 0 else np.nan
        )

    num_of_na_right = np.sum(np.isnan(mean_distances_right))
    sum_of_mean_distances_right_no_na = np.nansum(mean_distances_right)

    mccf1_metric = 1 - (
        (sum_of_mean_distances_left_no_na + sum_of_mean_distances_right_no_na)
        / (bins * 2 - num_of_na_right - num_of_na_left)
    ) / np.sqrt(2)

    eu_distance = np.sqrt((1 - mcc_nor_truncated) ** 2 + (1 - f_truncated) ** 2)
    best_threshold = thresholds[np.nanargmin(eu_distance)]

    mccf1_result = {"mccf1_metric": mccf1_metric, "best_threshold": best_threshold}

    return mccf1_result
