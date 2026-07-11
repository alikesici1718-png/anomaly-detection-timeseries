import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import pandas as pd
import pytest
from utils import point_metrics, window_eval


# ── helpers ───────────────────────────────────────────────────────────────────

def make_timestamps(n, freq='30min', start='2020-01-01'):
    return pd.Series(pd.date_range(start=start, periods=n, freq=freq))


# ── point_metrics ─────────────────────────────────────────────────────────────

class TestPointMetrics:

    def test_perfect_prediction(self):
        gt   = np.array([0, 0, 1, 1, 0])
        pred = pd.Series([0, 0, 1, 1, 0])
        m = point_metrics(pred, gt)
        assert m['TP'] == 2
        assert m['FP'] == 0
        assert m['FN'] == 0
        assert m['Precision'] == pytest.approx(1.0)
        assert m['Recall']    == pytest.approx(1.0)
        assert m['F1']        == pytest.approx(1.0)

    def test_all_false_positives(self):
        gt   = np.array([0, 0, 0, 0])
        pred = pd.Series([1, 1, 1, 1])
        m = point_metrics(pred, gt)
        assert m['TP'] == 0
        assert m['FP'] == 4
        assert m['Precision'] == pytest.approx(0.0)
        assert m['Recall']    == pytest.approx(0.0)

    def test_all_false_negatives(self):
        gt   = np.array([1, 1, 1])
        pred = pd.Series([0, 0, 0])
        m = point_metrics(pred, gt)
        assert m['FN'] == 3
        assert m['TP'] == 0
        assert m['Precision'] == pytest.approx(0.0)
        assert m['Recall']    == pytest.approx(0.0)
        assert m['F1']        == pytest.approx(0.0)

    def test_known_values(self):
        """2 TP, 1 FP, 1 FN → P=2/3, R=2/3, F1=2/3."""
        gt   = np.array([1, 1, 0, 1])
        pred = pd.Series([1, 1, 1, 0])
        m = point_metrics(pred, gt)
        assert m['TP'] == 2
        assert m['FP'] == 1
        assert m['FN'] == 1
        assert m['Precision'] == pytest.approx(2/3)
        assert m['Recall']    == pytest.approx(2/3)
        assert m['F1']        == pytest.approx(2/3)

    # ── edge cases ────────────────────────────────────────────────────────────

    def test_no_predictions_precision_is_zero(self):
        """No predicted anomalies → precision undefined → returns 0 (zero_division=0)."""
        gt   = np.array([0, 1, 0, 1])
        pred = pd.Series([0, 0, 0, 0])
        m = point_metrics(pred, gt)
        assert m['Precision'] == pytest.approx(0.0)

    def test_no_true_anomalies_recall_is_zero(self):
        """No ground-truth anomalies → recall undefined → returns 0."""
        gt   = np.array([0, 0, 0, 0])
        pred = pd.Series([1, 0, 0, 1])
        m = point_metrics(pred, gt)
        assert m['Recall'] == pytest.approx(0.0)

    def test_all_correct_no_anomalies(self):
        """All-zero ground truth, all-zero pred → perfect TN, P/R/F1 all 0."""
        gt   = np.array([0, 0, 0])
        pred = pd.Series([0, 0, 0])
        m = point_metrics(pred, gt)
        assert m['TP'] == 0
        assert m['TN'] == 3
        assert m['F1'] == pytest.approx(0.0)


# ── window_eval ───────────────────────────────────────────────────────────────

class TestWindowEval:

    def _build(self, n=100, freq='30min'):
        timestamps = make_timestamps(n, freq=freq)
        pred       = pd.Series(np.zeros(n, dtype=int))
        return timestamps, pred

    def test_hit_inside_window_detected(self):
        """A prediction exactly inside the window must mark it detected."""
        ts, pred = self._build(100)
        win_start = ts.iloc[30]
        win_end   = ts.iloc[40]
        pred.iloc[35] = 1                          # inside
        details, n_det, n_win, *_ = window_eval(
            pred, [(win_start, win_end)], ts, tolerance_pct=0.0)
        assert n_det == 1
        assert details[0]['detected'] is True
        assert details[0]['hits'] == 1

    def test_hit_in_tolerance_zone_detected(self):
        """A prediction just before the window (within tolerance) must count."""
        ts, pred = self._build(200)
        win_start = ts.iloc[100]
        win_end   = ts.iloc[120]            # window length = 20 steps → tol = 2 steps
        pred.iloc[98] = 1                   # 2 steps before window start (within 10% tol)
        details, n_det, *_ = window_eval(
            pred, [(win_start, win_end)], ts, tolerance_pct=0.10)
        assert n_det == 1
        assert details[0]['detected'] is True

    def test_hit_outside_tolerance_not_detected(self):
        """A prediction far outside all windows must not detect any window."""
        ts, pred = self._build(200)
        win_start = ts.iloc[100]
        win_end   = ts.iloc[120]
        pred.iloc[10] = 1                  # far outside
        details, n_det, *_ = window_eval(
            pred, [(win_start, win_end)], ts, tolerance_pct=0.10)
        assert n_det == 0
        assert details[0]['detected'] is False

    def test_no_predictions_recall_is_zero(self):
        """Zero predictions → no window detected → recall = 0."""
        ts, pred = self._build(100)
        win_start = ts.iloc[40]
        win_end   = ts.iloc[60]
        _, n_det, n_win, fp, prec, rec, f1 = window_eval(
            pred, [(win_start, win_end)], ts, tolerance_pct=0.10)
        assert n_det == 0
        assert rec   == pytest.approx(0.0)
        assert prec  == pytest.approx(0.0)
        assert f1    == pytest.approx(0.0)

    def test_multiple_windows_partial_detection(self):
        """Two windows; only the first is hit → recall = 0.5."""
        ts, pred = self._build(200)
        w1 = (ts.iloc[20],  ts.iloc[40])
        w2 = (ts.iloc[100], ts.iloc[120])
        pred.iloc[30] = 1                  # hits w1 only
        _, n_det, n_win, fp, prec, rec, f1 = window_eval(
            pred, [w1, w2], ts, tolerance_pct=0.0)
        assert n_det == 1
        assert n_win == 2
        assert rec   == pytest.approx(0.5)

    def test_fp_points_counted_outside_zones(self):
        """Predictions outside all tolerance zones are FP points."""
        ts, pred = self._build(200)
        w1 = (ts.iloc[50], ts.iloc[70])
        pred.iloc[10] = 1    # FP — far outside
        pred.iloc[11] = 1    # FP — far outside
        pred.iloc[60] = 1    # TP — inside window
        _, _, _, fp_pts, *_ = window_eval(pred, [w1], ts, tolerance_pct=0.0)
        assert fp_pts == 2

    def test_precision_zero_when_all_predictions_outside(self):
        """All predictions outside zones → precision = 0."""
        ts, pred = self._build(200)
        w1 = (ts.iloc[50], ts.iloc[70])
        pred.iloc[10] = 1
        pred.iloc[20] = 1
        _, _, _, _, prec, _, _ = window_eval(pred, [w1], ts, tolerance_pct=0.0)
        assert prec == pytest.approx(0.0)
