"""This module implements metrics-related utils"""

import numpy as np  # type: ignore
from sklearn.metrics._ranking import _binary_clf_curve  # type: ignore


# from mcc_f1 package, which was broken with the latest sklearn version
def mcc_f1_curve(
    y_true, y_score, *, pos_label=None, sample_weight=None, unit_normalize_mcc=True
):
    """Compute the MCC-F1 curve

    The MCC-F1 curve combines the Matthews correlation coefficient and the
    F1-Score to clearly differentiate good and bad *binary* classifiers,
    especially with imbalanced ground truths.

    It has been recently proposed as a better alternative for the receiver
    operating characteristic (ROC) and the precision-recall (PR) curves.
    [1]

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True binary labels. If labels are not either {-1, 1} or {0, 1}, then
        pos_label should be explicitly given.

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function.

    pos_label : int or str, default=None
        The label of the positive class.
        When ``pos_label=None``, if y_true is in {-1, 1} or {0, 1},
        ``pos_label`` is set to 1, otherwise an error will be raised.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    unit_normalize_mcc : bool, default=True
        Whether to unit-normalize the MCC values, as in the original paper.

    Returns
    -------
    mcc : array, shape = [n_thresholds]
        MCC values such that element i is the MCC of predictions with
        score >= thresholds[i].

    f1 : array, shape = [n_thresholds]
        F1-Score values such that element i is the F1-Score of predictions with
        score >= thresholds[i].

    thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
        Increasing thresholds on the decision function used to compute
        MCC and F1.

    Notes
    -----
    Differently from the original MCC-F1 curve proposal, this implementation
    returns the correct limiting unit-normalized MCC value of 0.5 (or 0 for the
    non-unit-normalized) when its denominator is zero (MCC = 0/0), by
    arbitrarily setting the denominator to 1 in such cases, as suggested in
    [2].

    References
    ----------
    .. [1] `Chang Cao and Davide Chicco and Michael M. Hoffman. (2020)
            The MCC-F1 curve: a performance evaluation technique for binary
            classification.
            <https://arxiv.org/pdf/2006.11278>`_
    .. [2] `Wikipedia entry for the Matthews correlation coefficient
            <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight
    )

    positives = tps + fps  # Array of total positive predictions
    true_positives = tps[-1]  # No of positives in ground truth
    true_negatives = fps[-1]  # No of negatives in ground truth

    if true_positives == 0:
        raise ValueError("No positive samples in y_true, MCC and F1 are undefined.")
    if true_negatives == 0:
        raise ValueError("No negative samples in y_true, MCC is undefined.")

    # Compute MCC
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.sqrt(
            true_positives
            * true_negatives
            * positives
            * (true_positives + true_negatives - positives)
        )
        denominator[denominator == 0] = 1.0
        mccs = (true_negatives * tps - true_positives * fps) / denominator
    if unit_normalize_mcc:
        mccs = (mccs + 1) / 2  # Unit-normalize MCC values

    # Compute F1
    f1s = 2 * tps / (positives + true_positives)

    return mccs, f1s, thresholds


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
