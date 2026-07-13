"""Benchmark auto-selection follows the portfolio's market."""
from data_engine import detect_market, benchmark_for_market


def test_indian_tickers_detected():
    assert detect_market(["RELIANCE.NS", "TCS.NS", "INFY.NS"]) == "IN"


def test_bse_suffix_detected():
    assert detect_market(["RELIANCE.BO", "TCS.BO"]) == "IN"


def test_us_tickers_detected():
    assert detect_market(["AAPL", "MSFT", "JNJ"]) == "US"


def test_majority_indian_wins():
    assert detect_market(["RELIANCE.NS", "TCS.NS", "AAPL"]) == "IN"


def test_majority_us_wins():
    assert detect_market(["AAPL", "MSFT", "RELIANCE.NS"]) == "US"


def test_exact_half_indian_counts_as_indian():
    """Matches the existing >= len/2 rule used elsewhere in the app."""
    assert detect_market(["RELIANCE.NS", "AAPL"]) == "IN"


def test_empty_defaults_to_us():
    assert detect_market([]) == "US"


def test_benchmark_for_india_is_nifty_50():
    assert benchmark_for_market("IN") == "^NSEI"


def test_benchmark_for_us_is_sp500():
    assert benchmark_for_market("US") == "^GSPC"
