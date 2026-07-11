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

# --- Ground-truth label vector ---
windows = labels["realKnownCause/nyc_taxi.csv"]
gt = np.zeros(len(df), dtype=int)
for win in windows:
    start = pd.Timestamp(win[0])
    end   = pd.Timestamp(win[1])
    mask  = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    gt[mask] = 1

# --- Method 1: Rolling z-score (past-only window) ---
LOOKBACK = 48 * 14
THRESH   = 2.0

roll_mean   = df["value"].rolling(LOOKBACK, min_periods=48).mean()
roll_std    = df["value"].rolling(LOOKBACK, min_periods=48).std().fillna(1).replace(0, 1)
zscore      = (df["value"] - roll_mean) / roll_std
pred_zscore = (np.abs(zscore) > THRESH).astype(int)

# --- Method 2: STL residual z-score ---
stl    = STL(df["value"], period=48, seasonal=7*48+1, robust=False)
result = stl.fit()
resid  = pd.Series(result.resid, index=df.index)
resid_z  = resid / resid.std()
pred_stl = (np.abs(resid_z) > THRESH).astype(int)

# --- Metrics ---
def get_metrics(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    p  = precision_score(gt, pred, zero_division=0)
    r  = recall_score(gt, pred, zero_division=0)
    f1 = f1_score(gt, pred, zero_division=0)
    return tp, fp, fn, tn, p, r, f1

def print_metrics(name, pred, gt):
    tp, fp, fn, tn, p, r, f1 = get_metrics(pred, gt)
    print(f"\n=== {name} ===")
    print(f"  Predicted anomaly   : {int(pred.sum())}")
    print(f"  True Positives (TP) : {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives (TN) : {tn}")
    print(f"  Precision           : {p:.4f}")
    print(f"  Recall              : {r:.4f}")
    print(f"  F1                  : {f1:.4f}")

print(f"Total observations               : {len(df)}")
print(f"Ground-truth anomaly points      : {int(gt.sum())}")
print(f"Ground-truth normal points       : {int((gt==0).sum())}")
print(f"Threshold                        : |z| > {THRESH}")
print(f"Baseline lookback                : {LOOKBACK} steps ({LOOKBACK//48} days, past-only)")

print_metrics("Baseline — Rolling z-score (past-only, 14-day window)", pred_zscore, gt)
print_metrics("Advanced  — STL residual z-score (daily+weekly seasonality removed)", pred_stl, gt)

b = get_metrics(pred_zscore, gt)
s = get_metrics(pred_stl, gt)

print("\n--- Comparison Table ---")
print(f"{'Metric':<14} {'Baseline':>10} {'STL':>10} {'Winner':>10}")
rows = [
    ("TP",        b[0], s[0], "int"),
    ("FP",        b[1], s[1], "int"),
    ("FN",        b[2], s[2], "int"),
    ("TN",        b[3], s[3], "int"),
    ("Precision", b[4], s[4], "float"),
    ("Recall",    b[5], s[5], "float"),
    ("F1",        b[6], s[6], "float"),
]
for (label, bval, sval, fmt) in rows:
    bstr = str(bval) if fmt == "int" else f"{bval:.4f}"
    sstr = str(sval) if fmt == "int" else f"{sval:.4f}"
    # For TP/Precision/Recall/F1 higher is better; for FP/FN lower is better
    if label in ("FP", "FN"):
        winner = "Baseline" if bval < sval else ("STL" if sval < bval else "Tie")
    else:
        winner = "Baseline" if bval > sval else ("STL" if sval > bval else "Tie")
    print(f"  {label:<12} {bstr:>10} {sstr:>10} {winner:>10}")
