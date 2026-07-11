import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import warnings
warnings.filterwarnings("ignore")

# --- Load data ---
df = pd.read_csv(r"C:\Users\alike\Desktop\anomaly-detection-timeseries\data\raw\nyc_taxi.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

with open(r"C:\Users\alike\Desktop\anomaly-detection-timeseries\data\raw\nab_labels.json") as f:
    labels = json.load(f)

raw_windows = labels["realKnownCause/nyc_taxi.csv"]
windows = [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in raw_windows]

# --- Ground-truth point labels ---
gt = np.zeros(len(df), dtype=int)
for start, end in windows:
    gt[(df["timestamp"] >= start) & (df["timestamp"] <= end)] = 1

# --- Reproduce both detectors (unchanged from detect_anomalies.py) ---
LOOKBACK = 48 * 14
THRESH   = 2.0

roll_mean   = df["value"].rolling(LOOKBACK, min_periods=48).mean()
roll_std    = df["value"].rolling(LOOKBACK, min_periods=48).std().fillna(1).replace(0, 1)
zscore      = (df["value"] - roll_mean) / roll_std
pred_zscore = (np.abs(zscore) > THRESH).astype(int)

stl      = STL(df["value"], period=48, seasonal=7*48+1, robust=False)
result   = stl.fit()
resid    = pd.Series(result.resid, index=df.index)
resid_z  = resid / resid.std()
pred_stl = (np.abs(resid_z) > THRESH).astype(int)

# ---------------------------------------------------------------------------
# WINDOW-BASED EVALUATION
# A window is "detected" if at least one predicted anomaly falls within
# [window_start - tolerance, window_end + tolerance].
# Tolerance = 10% of window length (NAB uses a similar lead-in allowance).
# ---------------------------------------------------------------------------
TOLERANCE_PCT = 0.10   # 10% of window length

def window_eval(pred_series, windows, timestamps, tolerance_pct):
    """
    Returns per-window detection flags and summary metrics.
    pred_series : array-like of 0/1
    windows     : list of (start_ts, end_ts) tuples
    timestamps  : pd.Series of datetime
    tolerance_pct: fraction of window length added as pre/post buffer
    """
    detected = []
    details  = []
    for start, end in windows:
        length    = end - start
        tol       = length * tolerance_pct
        t_start   = start - tol
        t_end     = end   + tol
        in_zone   = (timestamps >= t_start) & (timestamps <= t_end)
        n_hits    = int(pred_series[in_zone].sum())
        hit       = n_hits > 0
        detected.append(hit)
        details.append({
            "window_start": start,
            "window_end"  : end,
            "tol_start"   : t_start,
            "tol_end"     : t_end,
            "hits"        : n_hits,
            "detected"    : hit,
        })
    n_detected = sum(detected)
    n_windows  = len(windows)
    # window-level recall = detected / total windows
    # window-level precision = detected windows / total predicted windows
    # (we count each window max once, so precision = detected / total_predicted_windows)
    # For simplicity: precision = TP_windows / (TP_windows + FP_alarms_outside_all_windows)
    # FP alarm = any predicted point outside ALL tolerance zones
    all_zone = pd.Series(False, index=timestamps.index)
    for start, end in windows:
        length  = end - start
        tol     = length * tolerance_pct
        t_start = start - tol
        t_end   = end   + tol
        all_zone |= (timestamps >= t_start) & (timestamps <= t_end)
    fp_points = int(((pred_series == 1) & (~all_zone)).sum())
    # Simplified window precision: fraction of predicted positives that fall in a zone
    tp_points = int(((pred_series == 1) & all_zone).sum())
    total_pred = int(pred_series.sum())
    w_precision = tp_points / total_pred if total_pred > 0 else 0.0
    w_recall    = n_detected / n_windows
    w_f1        = (2 * w_precision * w_recall / (w_precision + w_recall)
                   if (w_precision + w_recall) > 0 else 0.0)
    return details, n_detected, n_windows, fp_points, w_precision, w_recall, w_f1

# --- Point-based metrics helper ---
def point_metrics(pred, gt):
    tp = int(((pred==1)&(gt==1)).sum())
    fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum())
    p  = precision_score(gt, pred, zero_division=0)
    r  = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)
    return tp, fp, fn, p, r, f1

# --- Run evaluations ---
b_details, b_det, n_win, b_fp_pts, b_wp, b_wr, b_wf1 = window_eval(pred_zscore, windows, df["timestamp"], TOLERANCE_PCT)
s_details, s_det, _,     s_fp_pts, s_wp, s_wr, s_wf1 = window_eval(pred_stl,    windows, df["timestamp"], TOLERANCE_PCT)

b_tp, b_fp, b_fn, b_pp, b_pr, b_pf1 = point_metrics(pred_zscore, gt)
s_tp, s_fp, s_fn, s_pp, s_pr, s_pf1 = point_metrics(pred_stl,    gt)

tol_steps = int(TOLERANCE_PCT * 100)

# --- Per-window detection table ---
print("=" * 70)
print(f"WINDOW-BASED EVALUATION  (tolerance = {tol_steps}% of each window length)")
print("=" * 70)
print(f"\n{'#':<3} {'Window start':<22} {'Window end':<22} {'Baseline':>10} {'STL':>6}")
print("-" * 66)
for i, (bd, sd) in enumerate(zip(b_details, s_details)):
    b_mark = "YES (hits=" + str(bd["hits"]) + ")" if bd["detected"] else "NO"
    s_mark = "YES (hits=" + str(sd["hits"]) + ")" if sd["detected"] else "NO"
    print(f"{i+1:<3} {str(bd['window_start']):<22} {str(bd['window_end']):<22} {b_mark:>10} {s_mark:>10}")

print(f"\nWindows detected : Baseline = {b_det}/{n_win}   STL = {s_det}/{n_win}")

# --- Side-by-side comparison ---
print("\n" + "=" * 70)
print("POINT-BASED  vs  WINDOW-BASED  —  SIDE BY SIDE")
print("=" * 70)
print(f"\n{'Metric':<22} {'Baseline (point)':>18} {'STL (point)':>14} {'Baseline (window)':>18} {'STL (window)':>14}")
print("-" * 90)

rows = [
    ("Precision",  f"{b_pp:.4f}", f"{s_pp:.4f}", f"{b_wp:.4f}", f"{s_wp:.4f}"),
    ("Recall",     f"{b_pr:.4f}", f"{s_pr:.4f}", f"{b_wr:.4f}", f"{s_wr:.4f}"),
    ("F1",         f"{b_pf1:.4f}",f"{s_pf1:.4f}",f"{b_wf1:.4f}",f"{s_wf1:.4f}"),
    ("TP (pts/wins)",f"{b_tp}",   f"{s_tp}",     f"{b_det}/5",  f"{s_det}/5"),
    ("FP (pts/pts outside zone)", f"{b_fp}", f"{s_fp}", f"{b_fp_pts}", f"{s_fp_pts}"),
    ("FN (pts/windows missed)",   f"{b_fn}", f"{s_fn}", f"{n_win-b_det}/5", f"{n_win-s_det}/5"),
]
for r in rows:
    print(f"  {r[0]:<20} {r[1]:>18} {r[2]:>14} {r[3]:>18} {r[4]:>14}")

print(f"""
--- Key numbers ---
  Baseline predicted anomaly points : {int(pred_zscore.sum())}
  STL     predicted anomaly points  : {int(pred_stl.sum())}
  Ground-truth anomaly windows      : {n_win}
  Ground-truth anomaly points       : {int(gt.sum())}
  Tolerance applied                 : {tol_steps}% of each window length
""")
