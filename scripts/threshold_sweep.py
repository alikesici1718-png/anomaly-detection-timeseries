"""
Threshold sweep: for both methods, scan |z| thresholds from 1.0 to 4.0
and find the value that maximises window-based F1.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import warnings
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# ── data ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\alike\Desktop\anomaly-detection-timeseries"
df = pd.read_csv(BASE + r"\data\raw\nyc_taxi.csv", parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
with open(BASE + r"\data\raw\nab_labels.json") as f:
    labels = json.load(f)

windows = [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in labels["realKnownCause/nyc_taxi.csv"]]
gt = np.zeros(len(df), dtype=int)
for s, e in windows:
    gt[(df["timestamp"] >= s) & (df["timestamp"] <= e)] = 1

# ── build z-score series (once) ───────────────────────────────────────────────
LOOKBACK = 48 * 14

shifted   = df["value"].shift(1)
roll_mean = shifted.rolling(LOOKBACK, min_periods=48).mean()
roll_std  = shifted.rolling(LOOKBACK, min_periods=48).std().fillna(1).replace(0, 1)
zscore_base = (df["value"] - roll_mean) / roll_std

stl_fit    = STL(df["value"], period=48, seasonal=7*48+1, robust=False).fit()
resid      = pd.Series(stl_fit.resid, index=df.index)
zscore_stl = resid / resid.std()

# ── evaluation helpers ────────────────────────────────────────────────────────
TOL_PCT = 0.10

def window_f1(pred, windows, ts, tol=TOL_PCT):
    all_zone = pd.Series(False, index=ts.index)
    n_det = 0
    for s, e in windows:
        z0, z1 = s - (e-s)*tol, e + (e-s)*tol
        zone = (ts >= z0) & (ts <= z1)
        all_zone |= zone
        if pred[zone].sum() > 0:
            n_det += 1
    total = pred.sum()
    tp_pts = int((pred == 1)[all_zone].sum())
    wp = tp_pts / total if total > 0 else 0.0
    wr = n_det / len(windows)
    return 2*wp*wr/(wp+wr) if (wp+wr) > 0 else 0.0, wp, wr, n_det

def point_f1(pred, gt):
    p = precision_score(gt, pred, zero_division=0)
    r = recall_score(gt, pred, zero_division=0)
    f = f1_score(gt, pred, zero_division=0)
    tp = int(((pred==1)&(gt==1)).sum())
    fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum())
    return p, r, f, tp, fp, fn

# ── threshold sweep ───────────────────────────────────────────────────────────
thresholds = np.arange(1.0, 4.01, 0.1)

results = {"base": [], "stl": []}
for t in thresholds:
    for name, zs in [("base", zscore_base), ("stl", zscore_stl)]:
        pred = (np.abs(zs) > t).astype(int)
        wf1, wp, wr, nd = window_f1(pred, windows, df["timestamp"])
        results[name].append({"thresh": t, "wf1": wf1, "wp": wp, "wr": wr, "nd": nd,
                               "flags": int(pred.sum())})

base_df = pd.DataFrame(results["base"])
stl_df  = pd.DataFrame(results["stl"])

best_base = base_df.loc[base_df["wf1"].idxmax()]
best_stl  = stl_df.loc[stl_df["wf1"].idxmax()]

# ── full metrics at optimal thresholds ───────────────────────────────────────
pred_base_opt = (np.abs(zscore_base) > best_base.thresh).astype(int)
pred_stl_opt  = (np.abs(zscore_stl)  > best_stl.thresh).astype(int)

b_pp, b_pr, b_pf1, b_tp, b_fp, b_fn = point_f1(pred_base_opt, gt)
s_pp, s_pr, s_pf1, s_tp, s_fp, s_fn = point_f1(pred_stl_opt,  gt)
bw_f1, bw_p, bw_r, bw_nd = window_f1(pred_base_opt, windows, df["timestamp"])
sw_f1, sw_p, sw_r, sw_nd = window_f1(pred_stl_opt,  windows, df["timestamp"])

# ── report ────────────────────────────────────────────────────────────────────
PREV_THRESH = 2.0
prev_base = base_df[base_df["thresh"].round(1) == PREV_THRESH].iloc[0]
prev_stl  = stl_df[stl_df["thresh"].round(1)  == PREV_THRESH].iloc[0]

print("=" * 68)
print("THRESHOLD SWEEP RESULTS")
print("=" * 68)
print(f"\nThreshold range   : 1.0 – 4.0 (step 0.1)")
print(f"Optimisation target: window-based F1")

print(f"\n{'Method':<10} {'Prev thresh':>12} {'Prev win-F1':>12} {'Opt thresh':>11} {'Opt win-F1':>11} {'Change':>8}")
print("-" * 68)
print(f"  Baseline  {PREV_THRESH:>12.1f} {prev_base.wf1:>12.4f} {best_base.thresh:>11.1f} {best_base.wf1:>11.4f} {best_base.wf1-prev_base.wf1:>+8.4f}")
print(f"  STL       {PREV_THRESH:>12.1f} {prev_stl.wf1:>12.4f} {best_stl.thresh:>11.1f} {best_stl.wf1:>11.4f}  {best_stl.wf1-prev_stl.wf1:>+7.4f}")

print(f"\n--- Optimal threshold detail ---")
for label, best, pp, pr, pf1, tp, fp, fn, wf, wp, wr, nd, flags in [
    ("Baseline", best_base, b_pp, b_pr, b_pf1, b_tp, b_fp, b_fn, bw_f1, bw_p, bw_r, bw_nd, pred_base_opt.sum()),
    ("STL",      best_stl,  s_pp, s_pr, s_pf1, s_tp, s_fp, s_fn, sw_f1, sw_p, sw_r, sw_nd, pred_stl_opt.sum()),
]:
    print(f"\n  {label} — optimal threshold = {best.thresh:.1f}")
    print(f"    Flags          : {flags}")
    print(f"    Point   P/R/F1 : {pp:.4f} / {pr:.4f} / {pf1:.4f}  (TP={tp} FP={fp} FN={fn})")
    print(f"    Window  P/R/F1 : {wp:.4f} / {wr:.4f} / {wf:.4f}  ({nd}/5 windows detected)")

print("\n" + "=" * 68)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117", "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#2a2d3a", "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#c9d1d9", "grid.color": "#21262d", "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})
SURFACE = "#0f1117"
C_BASE  = "#58a6ff"
C_STL   = "#3fb950"
C_ANOM  = "#f85149"

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(base_df["thresh"], base_df["wf1"], lw=2, color=C_BASE, label="Baseline (rolling z-score)")
ax.plot(stl_df["thresh"],  stl_df["wf1"],  lw=2, color=C_STL,  label="STL residual z-score")

# vertical lines at optima
ax.axvline(best_base.thresh, color=C_BASE, lw=1.2, ls="--", alpha=0.7)
ax.axvline(best_stl.thresh,  color=C_STL,  lw=1.2, ls="--", alpha=0.7)

# previous fixed threshold
ax.axvline(PREV_THRESH, color=C_ANOM, lw=1, ls=":", alpha=0.6, label="Previous fixed threshold (2.0)")

# optimal point markers
ax.scatter([best_base.thresh], [best_base.wf1], s=70, color=C_BASE, zorder=5)
ax.scatter([best_stl.thresh],  [best_stl.wf1],  s=70, color=C_STL,  zorder=5)

# annotations
ax.annotate(f"Baseline opt\nthresh={best_base.thresh:.1f}\nF1={best_base.wf1:.3f}",
            xy=(best_base.thresh, best_base.wf1),
            xytext=(best_base.thresh + 0.25, best_base.wf1 - 0.07),
            fontsize=9, color=C_BASE,
            arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.2))
ax.annotate(f"STL opt\nthresh={best_stl.thresh:.1f}\nF1={best_stl.wf1:.3f}",
            xy=(best_stl.thresh, best_stl.wf1),
            xytext=(best_stl.thresh + 0.25, best_stl.wf1 + 0.04),
            fontsize=9, color=C_STL,
            arrowprops=dict(arrowstyle="->", color=C_STL, lw=1.2))

ax.set_xlabel("Threshold (|z| >)")
ax.set_ylabel("Window-based F1")
ax.set_title("Window-Based F1 vs. Threshold — Baseline vs. STL\n(threshold swept 1.0 → 4.0, step 0.1)",
             fontsize=12, pad=12, color="#e6edf3")
ax.legend(framealpha=0, fontsize=10)
ax.grid(axis="y")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

fig.tight_layout()
out = BASE + r"\visualizations\05_threshold_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"\nSaved: {out}")
