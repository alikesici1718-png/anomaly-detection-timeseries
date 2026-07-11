"""
Cost-sensitive threshold selection for both methods.
Three selection strategies:
  1. F1-optimal        (from threshold_sweep.py — reproduced here)
  2. Recall-priority   (recall >= 0.8, maximise precision)
  3. Cost-sensitive    (minimise FN*missed_cost + FP*alarm_cost)
     Scenario A: missed_anomaly_cost=10, false_alarm_cost=1  (critical infra)
     Scenario B: missed_anomaly_cost=1,  false_alarm_cost=5  (alarm fatigue)
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# ── data ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\alike\Desktop\anomaly-detection-timeseries"
df   = pd.read_csv(BASE + r"\data\raw\nyc_taxi.csv", parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
with open(BASE + r"\data\raw\nab_labels.json") as f:
    labels = json.load(f)

windows = [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in labels["realKnownCause/nyc_taxi.csv"]]
gt = np.zeros(len(df), dtype=int)
for s, e in windows:
    gt[(df["timestamp"] >= s) & (df["timestamp"] <= e)] = 1

# ── z-score series ────────────────────────────────────────────────────────────
LOOKBACK = 48 * 14
shifted   = df["value"].shift(1)
roll_mean = shifted.rolling(LOOKBACK, min_periods=48).mean()
roll_std  = shifted.rolling(LOOKBACK, min_periods=48).std().fillna(1).replace(0, 1)
zs_base   = (df["value"] - roll_mean) / roll_std

stl_fit  = STL(df["value"], period=48, seasonal=7*48+1, robust=False).fit()
resid    = pd.Series(stl_fit.resid, index=df.index)
zs_stl   = resid / resid.std()

# ── helpers ───────────────────────────────────────────────────────────────────
TOL = 0.10

def window_metrics(pred, windows, ts, tol=TOL):
    all_zone = pd.Series(False, index=ts.index)
    n_det = 0
    for s, e in windows:
        z0, z1 = s - (e - s) * tol, e + (e - s) * tol
        zone = (ts >= z0) & (ts <= z1); all_zone |= zone
        if pred[zone].sum() > 0: n_det += 1
    total = int(pred.sum())
    tp    = int((pred == 1)[all_zone].sum())
    fp    = int((pred == 1)[~all_zone].sum())
    wp    = tp / total if total > 0 else 0.0
    wr    = n_det / len(windows)
    wf1   = 2*wp*wr/(wp+wr) if (wp+wr) > 0 else 0.0
    return dict(win_det=n_det, win_P=wp, win_R=wr, win_F1=wf1, win_TP=tp, win_FP=fp)

def point_metrics(pred, gt):
    tp = int(((pred==1)&(gt==1)).sum()); fp = int(((pred==1)&(gt==0)).sum())
    fn = int(((pred==0)&(gt==1)).sum()); tn = int(((pred==0)&(gt==0)).sum())
    p = precision_score(gt, pred, zero_division=0)
    r = recall_score(gt, pred, zero_division=0)
    f = f1_score(gt, pred, zero_division=0)
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, P=p, R=r, F1=f)

def full_metrics(zs, t, gt, windows, ts):
    pred = (np.abs(zs) > t).astype(int)
    pm = point_metrics(pred, gt)
    wm = window_metrics(pred, windows, ts)
    return {**pm, **wm, "thresh": t, "flags": int(pred.sum())}

# ── sweep ─────────────────────────────────────────────────────────────────────
thresholds = np.arange(1.0, 4.01, 0.05)   # finer grid than before

records = {"base": [], "stl": []}
for t in thresholds:
    for key, zs in [("base", zs_base), ("stl", zs_stl)]:
        records[key].append(full_metrics(zs, t, gt, windows, df["timestamp"]))

base_df = pd.DataFrame(records["base"])
stl_df  = pd.DataFrame(records["stl"])

# ── Strategy 1: F1-optimal (window-based F1) ──────────────────────────────────
def best_f1(df_): return df_.loc[df_["win_F1"].idxmax()]

b_f1opt = best_f1(base_df)
s_f1opt = best_f1(stl_df)

# ── Strategy 2: Recall-priority  (win_R >= 0.8, then max win_P) ──────────────
RECALL_MIN = 0.80

def best_recall_priority(df_):
    eligible = df_[df_["win_R"] >= RECALL_MIN]
    if eligible.empty:
        # fallback: highest recall
        return df_.loc[df_["win_R"].idxmax()]
    return eligible.loc[eligible["win_P"].idxmax()]

b_rec = best_recall_priority(base_df)
s_rec = best_recall_priority(stl_df)

# ── Strategy 3: Cost-sensitive ────────────────────────────────────────────────
def best_cost(df_, missed_cost, alarm_cost):
    costs = df_["FN"] * missed_cost + df_["FP"] * alarm_cost
    return df_.loc[costs.idxmin()], costs.min()

b_cosA, b_cosA_val = best_cost(base_df, missed_cost=10, alarm_cost=1)
s_cosA, s_cosA_val = best_cost(stl_df,  missed_cost=10, alarm_cost=1)
b_cosB, b_cosB_val = best_cost(base_df, missed_cost=1,  alarm_cost=5)
s_cosB, s_cosB_val = best_cost(stl_df,  missed_cost=1,  alarm_cost=5)

# ── print report ──────────────────────────────────────────────────────────────
SEP = "=" * 72
print(SEP)
print("COST-SENSITIVE THRESHOLD SELECTION")
print(SEP)

strategies = [
    ("F1-optimal (win-F1 max)",        b_f1opt, s_f1opt),
    ("Recall-priority (R>=0.8, max P)", b_rec,   s_rec),
    ("Cost A: FN*10+FP*1 (min cost)",  b_cosA,  s_cosA),
    ("Cost B: FN*1+FP*5  (min cost)",  b_cosB,  s_cosB),
]

for label, br, sr in strategies:
    print(f"\n--- {label} ---")
    print(f"  {'':25} {'Baseline':>12} {'STL':>12}")
    for col, fmt in [("thresh",":.1f"),("flags","d"),("TP","d"),("FP","d"),
                     ("FN","d"),("P",":.4f"),("R",":.4f"),("F1",":.4f"),
                     ("win_det","d"),("win_P",":.4f"),("win_R",":.4f"),("win_F1",":.4f")]:
        bv = format(br[col], fmt[1:])
        sv = format(sr[col], fmt[1:])
        print(f"  {col:<25} {bv:>12} {sv:>12}")

print(f"\n{'Cost scenario details':}")
for name, mc, ac in [("A (critical infra): FN*10 + FP*1", 10, 1),
                      ("B (alarm fatigue) : FN*1  + FP*5", 1,  5)]:
    print(f"\n  Scenario {name}")
    for meth, row in [("Baseline", b_cosA if mc==10 else b_cosB),
                      ("STL",      s_cosA if mc==10 else s_cosB)]:
        cost = row["FN"]*mc + row["FP"]*ac
        print(f"    {meth:<10} thresh={row['thresh']:.1f}  FN={row['FN']}  FP={row['FP']}  cost={cost:.0f}")

print("\n" + SEP)

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":"#0f1117","axes.facecolor":"#0f1117",
    "axes.edgecolor":"#2a2d3a","axes.labelcolor":"#c9d1d9",
    "xtick.color":"#8b949e","ytick.color":"#8b949e",
    "text.color":"#c9d1d9","grid.color":"#21262d","grid.linewidth":0.6,
    "font.family":"sans-serif","font.size":11,
    "axes.spines.top":False,"axes.spines.right":False,
})
SURFACE="#0f1117"; C_BASE="#58a6ff"; C_STL="#3fb950"; C_ANOM="#f85149"
C_F1="#d2a8ff"; C_REC="#ffa657"; C_CA="#ff7b72"; C_CB="#79c0ff"

# ── figure: precision-recall trade-off for all 3 strategies ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

strat_markers = {
    "F1-opt":      (b_f1opt["win_P"],  b_f1opt["win_R"],  s_f1opt["win_P"],  s_f1opt["win_R"],  "o", C_F1,  "F1-optimal"),
    "Recall-pri":  (b_rec["win_P"],    b_rec["win_R"],    s_rec["win_P"],    s_rec["win_R"],    "s", C_REC, "Recall-priority (R>=0.8)"),
    "Cost A":      (b_cosA["win_P"],   b_cosA["win_R"],   s_cosA["win_P"],   s_cosA["win_R"],   "^", C_CA,  "Cost A: FN*10+FP*1"),
    "Cost B":      (b_cosB["win_P"],   b_cosB["win_R"],   s_cosB["win_P"],   s_cosB["win_R"],   "D", C_CB,  "Cost B: FN*1+FP*5"),
}

for ax, (name, zs_df, color, title) in zip(axes, [
    ("Baseline", base_df, C_BASE, "Baseline — Rolling Z-Score"),
    ("STL",      stl_df,  C_STL,  "STL — Residual Z-Score"),
]):
    # full precision-recall curve as threshold varies
    ax.plot(zs_df["win_R"], zs_df["win_P"], lw=1.5, color=color, alpha=0.5, label="PR curve (all thresholds)")

    # strategy operating points
    idx = 0 if name == "Baseline" else 2
    for key, vals in strat_markers.items():
        px = vals[idx];    py = vals[idx+1]
        ax.scatter([px], [py], s=90, marker=vals[4], color=vals[5], zorder=5, label=vals[6])
        # threshold label
        if name == "Baseline":
            row = b_f1opt if key=="F1-opt" else (b_rec if key=="Recall-pri" else (b_cosA if key=="Cost A" else b_cosB))
        else:
            row = s_f1opt if key=="F1-opt" else (s_rec if key=="Recall-pri" else (s_cosA if key=="Cost A" else s_cosB))
        ax.annotate(f"t={row['thresh']:.1f}", xy=(px, py),
                    xytext=(px + 0.03, py + 0.025),
                    fontsize=8.5, color=vals[5])

    ax.set_xlim(-0.05, 1.15); ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("Window Recall"); ax.set_ylabel("Window Precision")
    ax.set_title(title, fontsize=12, pad=10, color="#e6edf3")
    ax.grid(axis="both")
    ax.legend(loc="upper right", framealpha=0, fontsize=9)

fig.suptitle("Precision-Recall Trade-off — Three Threshold Selection Strategies",
             fontsize=13, color="#e6edf3", y=1.01)
fig.tight_layout()
out = BASE + r"\visualizations\06_cost_sensitive_thresholds.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
plt.show()
print(f"Saved: {out}")
