import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import numpy as np
import pandas as pd
import pytest
from utils import rolling_zscore, detect_zscore


# ── helpers ──────────────────────────────────────────────────────────────────

def make_series(values):
    return pd.Series(values, dtype=float)


# ── rolling_zscore ────────────────────────────────────────────────────────────

class TestRollingZscore:

    def test_flat_series_zscore_is_zero(self):
        """Constant series has no deviation; z-score must be 0 everywhere
        (after the warm-up window fills)."""
        s = make_series([100.0] * 100)
        z = rolling_zscore(s, lookback=10, min_periods=10)
        assert (z.dropna().abs() == 0).all()

    def test_spike_produces_high_zscore(self):
        """A sharp spike at index 50 on an otherwise flat series should yield
        a z-score well above 3 at that point."""
        values = [100.0] * 100
        values[50] = 500.0
        s = make_series(values)
        z = rolling_zscore(s, lookback=20, min_periods=10)
        assert abs(z.iloc[50]) > 3.0

    def test_returns_same_length(self):
        s = make_series(list(range(50)))
        z = rolling_zscore(s, lookback=10)
        assert len(z) == len(s)

    def test_zero_std_does_not_raise(self):
        """A window of identical values would normally produce std=0.
        After the warm-up (min_periods), the function must not produce inf or NaN."""
        s = make_series([5.0] * 30)
        z = rolling_zscore(s, lookback=10, min_periods=5)
        after_warmup = z.dropna()
        assert len(after_warmup) > 0
        assert not np.isinf(after_warmup).any()
        assert not after_warmup.isna().any()


# ── detect_zscore ─────────────────────────────────────────────────────────────

class TestDetectZscore:

    def test_flags_spike_point(self):
        """The index of the spike must be flagged as anomalous."""
        values = [100.0] * 100
        values[60] = 900.0          # extreme spike
        s = make_series(values)
        pred = detect_zscore(s, lookback=20, threshold=2.0, min_periods=10)
        assert pred.iloc[60] == 1

    def test_no_flags_on_constant_series(self):
        """A flat series should produce zero flags regardless of threshold."""
        s = make_series([42.0] * 200)
        pred = detect_zscore(s, lookback=30, threshold=2.0, min_periods=10)
        assert pred.sum() == 0

    def test_output_is_binary(self):
        """Output must contain only 0 and 1."""
        values = list(range(100))
        s = make_series(values)
        pred = detect_zscore(s, lookback=10, threshold=2.0)
        assert set(pred.unique()).issubset({0, 1})

    def test_lower_threshold_flags_more_points(self):
        """Lowering the threshold must never decrease the flag count."""
        values = [float(v) for v in range(200)]
        s = make_series(values)
        strict = detect_zscore(s, lookback=20, threshold=3.0).sum()
        loose  = detect_zscore(s, lookback=20, threshold=1.0).sum()
        assert loose >= strict

    def test_spike_not_flagged_before_warmup(self):
        """Points before min_periods are filled with NaN → z-score falls back
        to 0 (via fillna in rolling_zscore), so they must not be flagged."""
        values = [999.0] + [1.0] * 99   # spike at index 0, before any window fills
        s = make_series(values)
        pred = detect_zscore(s, lookback=50, threshold=2.0, min_periods=50)
        assert pred.iloc[0] == 0
