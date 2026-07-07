import numpy as np
import pandas as pd
from asset_analytics_engine import normalize_to_pct, align_and_normalize


def _series(values, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype="float64")


def test_normalize_first_point_is_zero():
    s = normalize_to_pct(_series([100, 110, 90]))
    assert abs(s.iloc[0]) < 1e-9


def test_normalize_doubling_is_100pct():
    s = normalize_to_pct(_series([100, 200]))
    assert abs(s.iloc[1] - 100.0) < 1e-9


def test_normalize_halving_is_minus_50pct():
    s = normalize_to_pct(_series([100, 50]))
    assert abs(s.iloc[1] + 50.0) < 1e-9


def test_normalize_rebases_from_first_valid_point():
    # Leading NaN is dropped; the first *valid* point becomes the 0% base.
    s = normalize_to_pct(_series([np.nan, 100, 150]))
    assert len(s) == 2
    assert abs(s.iloc[0]) < 1e-9          # first valid → 0%
    assert abs(s.iloc[1] - 50.0) < 1e-9   # +50%


def test_normalize_zero_first_value_returns_empty_no_crash():
    s = normalize_to_pct(_series([0.0, 10.0]))
    assert s.empty


def test_normalize_all_nan_returns_empty():
    s = normalize_to_pct(_series([np.nan, np.nan]))
    assert s.empty


def test_align_and_normalize_uses_shared_start():
    # Asset spans Jan 1-4, benchmark spans Jan 3-6 → overlap is Jan 3-4.
    asset = _series([100, 110, 120, 130], start="2023-01-01")
    bench = pd.Series([2000, 2100, 2200, 2300],
                      index=pd.date_range("2023-01-03", periods=4, freq="D"), dtype="float64")
    a_norm, b_norm = align_and_normalize(asset, bench)
    # Both rebased to the shared first date (Jan 3).
    assert list(a_norm.index) == list(b_norm.index)
    assert a_norm.index[0] == pd.Timestamp("2023-01-03")
    assert abs(a_norm.iloc[0]) < 1e-9
    assert abs(b_norm.iloc[0]) < 1e-9
    # Asset 120 → 130 over the overlap = +8.333%
    assert abs(a_norm.iloc[-1] - (130 / 120 - 1) * 100) < 1e-9


def test_align_and_normalize_no_overlap_returns_empty_pair():
    asset = _series([100, 110], start="2023-01-01")
    bench = pd.Series([2000, 2100],
                      index=pd.date_range("2024-01-01", periods=2, freq="D"), dtype="float64")
    a_norm, b_norm = align_and_normalize(asset, bench)
    assert a_norm.empty and b_norm.empty
