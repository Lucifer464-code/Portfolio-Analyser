import io
import pandas as pd
import payload


def _sample_transactions():
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT"],
        "Date": pd.to_datetime(["2023-01-05", "2023-02-10"]),
        "Action": ["Buy", "Buy"],
        "Quantity": [10, 5],
        "Price": [130.2, 252.75],
    })


def test_build_payload_shape_and_versions():
    p = payload.build_payload(_sample_transactions(), {"benchmark": "^NSEI", "max_weight_pct": 20.0})
    assert p["schema_version"] == payload.SCHEMA_VERSION == 1
    assert p["settings"] == {"benchmark": "^NSEI", "max_weight_pct": 20.0}
    assert len(p["transactions"]) == 2
    # dates serialised as ISO strings, not Timestamps
    assert p["transactions"][0]["Date"] == "2023-01-05"
    assert p["transactions"][0]["Ticker"] == "AAPL"


def test_build_payload_is_json_serialisable():
    import json
    p = payload.build_payload(_sample_transactions(), {})
    json.dumps(p)  # must not raise


def test_payload_to_csv_buffer_roundtrips_columns():
    p = payload.build_payload(_sample_transactions(), {})
    buf = payload.payload_to_csv_buffer(p)
    assert isinstance(buf, io.StringIO)
    df = pd.read_csv(buf)
    assert list(df.columns) == ["Ticker", "Date", "Action", "Quantity", "Price"]
    assert df.iloc[0]["Ticker"] == "AAPL"
    assert float(df.iloc[1]["Price"]) == 252.75


def test_payload_settings_fills_defaults():
    assert payload.payload_settings({"settings": {}}) == {"benchmark": "^GSPC", "max_weight_pct": 15.0}
    assert payload.payload_settings({"settings": {"benchmark": "^NSEI"}}) == {"benchmark": "^NSEI", "max_weight_pct": 15.0}
    assert payload.payload_settings({}) == {"benchmark": "^GSPC", "max_weight_pct": 15.0}


def test_price_column_optional():
    df = _sample_transactions().drop(columns=["Price"])
    p = payload.build_payload(df, {})
    buf = payload.payload_to_csv_buffer(p)
    out = pd.read_csv(buf)
    # Price column still emitted (empty) so downstream schema is stable
    assert "Price" in out.columns
