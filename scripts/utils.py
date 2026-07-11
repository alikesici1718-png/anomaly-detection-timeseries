"""
Shared, pure functions extracted from detect_anomalies.py and evaluate_windows.py.
Scripts import from here; tests import from here.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def rolling_zscore(values: pd.Series, lookback: int, min_periods: int = 1) -> pd.Series:
    """Truly past-only rolling z-score (shift(1) excludes the current point).

    Bug fixed: the original implementation used pandas .rolling() directly,
    which includes the current observation in the window despite the "past-only"
    comment. shift(1) corrects this. Discovered during independent verification.
    """
    shifted = values.shift(1)
    mean = shifted.rolling(lookback, min_periods=min_periods).mean()
    std  = shifted.rolling(lookback, min_periods=min_periods).std().fillna(1).replace(0, 1)
    return (values - mean) / std


def detect_zscore(values: pd.Series, lookback: int, threshold: float,
                  min_periods: int = 1) -> pd.Series:
    """Returns a 0/1 Series; 1 = anomaly (|z| > threshold)."""
    z = rolling_zscore(values, lookback, min_periods)
    return (np.abs(z) > threshold).astype(int)


def point_metrics(pred: pd.Series, gt: np.ndarray) -> dict:
    """Point-level precision, recall, F1 plus TP/FP/FN/TN counts."""
    pred_arr = np.asarray(pred)
    tp = int(((pred_arr == 1) & (gt == 1)).sum())
    fp = int(((pred_arr == 1) & (gt == 0)).sum())
    fn = int(((pred_arr == 0) & (gt == 1)).sum())
    tn = int(((pred_arr == 0) & (gt == 0)).sum())
    p  = precision_score(gt, pred_arr, zero_division=0)
    r  = recall_score(gt, pred_arr, zero_division=0)
    f1 = f1_score(gt, pred_arr, zero_division=0)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Precision=p, Recall=r, F1=f1)


def window_eval(pred: pd.Series, windows: list, timestamps: pd.Series,
                tolerance_pct: float = 0.10) -> tuple:
    """
    Window-based evaluation.

    A window is detected if >= 1 predicted point lands within
    [start - tol, end + tol] where tol = tolerance_pct * window_length.

    Returns (details, n_detected, n_windows, fp_points, precision, recall, f1).
    """
    pred_arr = np.asarray(pred)
    all_zone = pd.Series(False, index=timestamps.index)
    details  = []

    for start, end in windows:
        tol     = (end - start) * tolerance_pct
        t_start = start - tol
        t_end   = end   + tol
        in_zone = (timestamps >= t_start) & (timestamps <= t_end)
        all_zone |= in_zone
        n_hits   = int(pred_arr[in_zone].sum())
        details.append({
            "window_start": start,
            "window_end":   end,
            "hits":         n_hits,
            "detected":     n_hits > 0,
        })

    n_detected  = sum(d["detected"] for d in details)
    n_windows   = len(windows)
    tp_points   = int((pred_arr[all_zone] == 1).sum())
    fp_points   = int((pred_arr[~all_zone] == 1).sum())
    total_pred  = int(pred_arr.sum())
    w_precision = tp_points / total_pred if total_pred > 0 else 0.0
    w_recall    = n_detected / n_windows  if n_windows  > 0 else 0.0
    denom       = w_precision + w_recall
    w_f1        = 2 * w_precision * w_recall / denom if denom > 0 else 0.0

    return details, n_detected, n_windows, fp_points, w_precision, w_recall, w_f1
