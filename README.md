# Time Series Anomaly Detection — Method Comparison

> **TL;DR:** Compared two anomaly detection methods (rolling z-score baseline vs. STL-decomposition residual) on NYC taxi demand data with 5 known anomaly windows (holidays, storms). Point-wise evaluation favors STL (F1=0.093 vs 0.035), but window-wise evaluation reverses the ranking entirely (Baseline F1=0.51 vs STL F1=0.23) — because point-wise metrics penalize slow-onset anomalies unfairly, while STL's higher false-positive rate is only correctly penalized under window-based scoring. The choice of evaluation methodology changes which method appears to "win."

---

## Data

**Source:** [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB) — `realKnownCause/nyc_taxi.csv`

| Property | Value |
|---|---|
| Total observations | 10,320 |
| Sampling frequency | 30 minutes |
| Time range | 2014-07-01 → 2015-01-31 (7 months) |
| Anomaly windows | 5 |
| Anomaly points (within windows) | 1,035 (10.0%) |
| Normal points | 9,285 (90.0%) |

### Labeled Anomaly Windows

| # | Start | End | Known Cause |
|---|---|---|---|
| 1 | 2014-10-30 15:30 | 2014-11-03 22:30 | NYC Marathon + Hurricane Sandy anniversary |
| 2 | 2014-11-25 12:00 | 2014-11-29 19:00 | Thanksgiving holiday |
| 3 | 2014-12-23 11:30 | 2014-12-27 18:30 | Christmas holiday |
| 4 | 2014-12-29 21:30 | 2015-01-03 04:30 | New Year holiday |
| 5 | 2015-01-24 20:30 | 2015-01-29 03:30 | Winter storm Juno |

---

## Methodology

### Method 1 — Rolling Z-Score (Baseline)

Computes a rolling mean and standard deviation over a **14-day past-only window** (672 steps). A point is flagged anomalous if its z-score exceeds the threshold.

- Window: 672 steps (14 days, past-only — centered window excluded because anomaly days would dilute their own mean)
- Threshold: |z| > 2.0

### Method 2 — STL Residual Z-Score

Decomposes the series into trend, seasonal, and residual components using **STL (Seasonal and Trend decomposition using Loess)**. Anomaly detection runs on the residual only, removing daily and weekly patterns before scoring.

- STL period: 48 (daily seasonality)
- Seasonal window: 337 (weekly seasonality)
- `robust=False` — robust mode suppresses holiday residuals, reducing sensitivity
- Threshold: same |z| > 2.0 applied to global residual std

### Evaluation — Point-Based vs Window-Based

**Point-based:** Each 30-minute step is independently labeled as anomaly or normal. A method must flag the exact correct points to score TP. NYC taxi anomalies are sustained multi-day demand shifts, not spikes — a method may correctly identify that *something is wrong* while scoring low on point recall.

**Window-based (NAB-style):** A window is "detected" if at least one predicted anomaly falls within the window boundaries ± 10% tolerance. Precision is computed as the fraction of predicted points that fall inside any tolerance zone. This rewards early detection and avoids penalizing methods for not flagging every individual point of a 4-day holiday.

---

## Key Findings

### Per-Window Detection (5 windows, 10% tolerance)

| # | Window | Baseline | STL |
|---|---|---|---|
| 1 | 2014-10-30 → 11-03 | **YES** (3 hits) | **YES** (20 hits) |
| 2 | 2014-11-25 → 11-29 | **NO** | **YES** (4 hits) |
| 3 | 2014-12-23 → 12-27 | **YES** (2 hits) | **YES** (1 hit) |
| 4 | 2014-12-29 → 01-03 | **YES** (6 hits) | **YES** (20 hits) |
| 5 | 2015-01-24 → 01-29 | **YES** (8 hits) | **YES** (36 hits) |

**Baseline: 4/5 windows detected — STL: 5/5 windows detected**

### Point-Based vs Window-Based — Full Comparison

| Metric | Baseline (point) | STL (point) | Baseline (window) | STL (window) |
|---|---:|---:|---:|---:|
| Precision | 0.3725 | 0.1238 | 0.3725 | 0.1302 |
| Recall | 0.0184 | 0.0744 | 0.8000 | 1.0000 |
| F1 | 0.0350 | 0.0929 | **0.5084** | 0.2304 |
| TP | 19 pts | 77 pts | 4 / 5 wins | 5 / 5 wins |
| FP | 32 pts | 545 pts | 32 pts | 541 pts |
| FN | 1,016 pts | 958 pts | 1 / 5 wins | 0 / 5 wins |

| Parameter | Value |
|---|---|
| Threshold | \|z\| > 2.0 |
| Baseline lookback | 672 steps (14 days, past-only) |
| STL period | 48 (daily), seasonal window 337 (weekly) |
| Window tolerance | 10% of each window length |
| Total predicted anomalies — Baseline | 51 points |
| Total predicted anomalies — STL | 622 points |

---

## Requirements

```
pandas
numpy
scipy
statsmodels
scikit-learn
```

Install:

```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

---

## Usage

```bash
# Step 1 — Run both detectors, compute point-based metrics
python scripts/detect_anomalies.py

# Step 2 — Run window-based evaluation and side-by-side comparison
python scripts/evaluate_windows.py
```

Both scripts read from `data/raw/nyc_taxi.csv` and `data/raw/nab_labels.json`.

---

## Project Structure

```
anomaly-detection-timeseries/
├── data/
│   └── raw/
│       ├── nyc_taxi.csv          # NAB NYC taxi demand (10,320 rows)
│       └── nab_labels.json       # NAB ground-truth anomaly windows
├── scripts/
│   ├── detect_anomalies.py       # Rolling z-score + STL, point-based metrics
│   └── evaluate_windows.py       # Window-based evaluation, side-by-side table
├── notebooks/
├── tests/
├── visualizations/
└── .gitignore
```
