import pandas as pd
import payload
from data_engine import load_and_validate_csv


def test_roundtrip_preserves_holdings_without_network():
    # All rows carry a user-provided Price, so load_and_validate_csv performs
    # no Yahoo fetch (fetch_historical_prices only fetches price-less rows).
    transactions = pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "AAPL"],
        "Date": pd.to_datetime(["2023-01-05", "2023-02-10", "2024-06-01"]),
        "Action": ["Buy", "Buy", "Sell"],
        "Quantity": [10, 5, 3],
        "Price": [130.20, 252.75, 189.50],
    })
    p = payload.build_payload(transactions, {"benchmark": "^GSPC", "max_weight_pct": 15.0})
    buf = payload.payload_to_csv_buffer(p)

    result = load_and_validate_csv(buf)
    df = result[0] if isinstance(result, tuple) else result

    assert df is not None and not df.empty
    for col in ("Ticker", "Date", "Action", "Quantity", "Price"):
        assert col in df.columns
    # Same set of tickers survived
    assert set(df["Ticker"]) == {"AAPL", "MSFT"}
    # Prices preserved exactly (proves no refetch overwrote them)
    aapl_buy = df[(df["Ticker"] == "AAPL") & (df["Action"] == "Buy")].iloc[0]
    assert abs(float(aapl_buy["Price"]) - 130.20) < 1e-6
