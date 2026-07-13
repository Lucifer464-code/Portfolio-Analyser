"""Date parsing in load_and_validate_csv.

dayfirst=True applied to a whole ISO-formatted column coerces every date whose
day exceeds 12 to NaT (2023-02-15 is not the 2nd day of month 15), and silently
misreads the survivors (2023-01-10 -> 10 October). An all-ISO portfolio then
drops to zero rows and the app aborts with "contains no valid data".
"""
import io

import pandas as pd

from data_engine import load_and_validate_csv


def _load(csv_text):
    result = load_and_validate_csv(io.StringIO(csv_text))
    txns = result[0] if isinstance(result, tuple) else result
    diag = result[1] if isinstance(result, tuple) else {}
    return txns, diag


ISO_CSV = (
    "Ticker,Date,Action,Quantity,Price\n"
    "AAPL,2023-01-10,Buy,10,130.00\n"
    "MSFT,2023-02-15,Buy,5,250.00\n"
    "JNJ,2023-03-20,Buy,8,160.00\n"
    "KO,2024-11-25,Buy,12,60.00\n"
)


def test_iso_dates_all_rows_survive():
    txns, diag = _load(ISO_CSV)
    assert txns is not None
    assert "warning_dates" not in diag
    assert len(txns) == 4


def test_iso_dates_are_not_transposed():
    """2023-01-10 is 10 January, not 10 October."""
    txns, _ = _load(ISO_CSV)
    first = txns.loc[txns["Ticker"] == "AAPL", "Date"].iloc[0]
    assert first == pd.Timestamp("2023-01-10")


def test_day_first_slash_dates_still_parse():
    """Genuine day-first input must keep working: 15/02/2023 is 15 February."""
    txns, _ = _load(
        "Ticker,Date,Action,Quantity,Price\n"
        "AAPL,15/02/2023,Buy,10,130.00\n"
    )
    assert len(txns) == 1
    assert txns["Date"].iloc[0] == pd.Timestamp("2023-02-15")


def test_mixed_formats_in_one_file():
    txns, _ = _load(
        "Ticker,Date,Action,Quantity,Price\n"
        "AAPL,2023-03-20,Buy,10,130.00\n"
        "MSFT,15/02/2023,Buy,5,250.00\n"
    )
    assert len(txns) == 2
    dates = set(txns["Date"])
    assert pd.Timestamp("2023-03-20") in dates
    assert pd.Timestamp("2023-02-15") in dates


def test_ambiguous_slash_date_stays_day_first():
    """03/04/2023 is 3 April (day-first), not 4 March."""
    txns, _ = _load(
        "Ticker,Date,Action,Quantity,Price\n"
        "AAPL,03/04/2023,Buy,10,130.00\n"
    )
    assert txns["Date"].iloc[0] == pd.Timestamp("2023-04-03")


def test_genuinely_bad_dates_still_dropped():
    txns, diag = _load(
        "Ticker,Date,Action,Quantity,Price\n"
        "AAPL,2023-03-20,Buy,10,130.00\n"
        "MSFT,not-a-date,Buy,5,250.00\n"
    )
    assert len(txns) == 1
    assert "warning_dates" in diag
