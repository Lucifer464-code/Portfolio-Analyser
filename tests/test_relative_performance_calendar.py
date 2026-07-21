"""Period returns in the Relative Performance table must be anchored to
CALENDAR dates (exactly 1 month / 3 months / 1 year ago, using the nearest
trading day on or before that date), NOT a fixed count of trading rows.

Previously the code used prices.iloc[-window] with window=21 for "1M", i.e.
"21 trading rows back". A true calendar month can contain 20-22 trading days
depending on holidays, so the fixed-row method drifts from the calendar-month
figure a broker would show. These tests pin the calendar-date behavior.
"""
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from enhancement_engine import compute_portfolio_relative_performance


def _make_prices():
    """Daily business-day closes for one stock + benchmark over ~14 months.

    STOCK is engineered so the price on the exact calendar-month anchor dates
    is a known round number, letting us assert the return precisely.
    """
    idx = pd.bdate_range(end="2026-07-16", periods=300)  # business days
    # Benchmark: flat 100 so relative == stock return, keeps assertions simple.
    bm = pd.Series(100.0, index=idx)
    # Stock: linear ramp from 100 -> 200 across the range, indexed by position.
    stock = pd.Series(np.linspace(100.0, 200.0, len(idx)), index=idx)
    return pd.DataFrame({"STOCK": stock, "BM": bm})


def _nearest_on_or_before(index, target):
    prior = index[index <= target]
    return prior[-1]


def test_1m_return_anchored_to_calendar_month():
    prices = _make_prices()
    last_date = prices.index[-1]

    result = compute_portfolio_relative_performance(
        ["STOCK"], price_data=prices, benchmark="BM", period="1M"
    )
    got = float(result.loc[result["Ticker"] == "STOCK", "1M Return"].iloc[0])

    # Expected: nearest trading day on/before (last_date - 1 month).
    anchor = _nearest_on_or_before(prices.index, last_date - relativedelta(months=1))
    expected = prices["STOCK"].iloc[-1] / prices["STOCK"].loc[anchor] - 1

    assert abs(got - expected) < 1e-9, f"got {got}, expected {expected}"


def test_1y_return_anchored_to_calendar_year():
    prices = _make_prices()
    last_date = prices.index[-1]

    result = compute_portfolio_relative_performance(
        ["STOCK"], price_data=prices, benchmark="BM", period="1Y"
    )
    got = float(result.loc[result["Ticker"] == "STOCK", "1Y Return"].iloc[0])

    anchor = _nearest_on_or_before(prices.index, last_date - relativedelta(years=1))
    expected = prices["STOCK"].iloc[-1] / prices["STOCK"].loc[anchor] - 1

    assert abs(got - expected) < 1e-9, f"got {got}, expected {expected}"
