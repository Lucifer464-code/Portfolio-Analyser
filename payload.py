"""Serialization between the in-app transactions DataFrame + settings and a
JSON-serialisable payload stored in the browser. No Streamlit or storage here.
"""
import io
import pandas as pd

SCHEMA_VERSION = 1

# The columns a saved portfolio round-trips through load_and_validate_csv.
_CSV_COLUMNS = ["Ticker", "Date", "Action", "Quantity", "Price"]

_DEFAULT_SETTINGS = {"benchmark": "^GSPC", "max_weight_pct": 15.0}


def build_payload(transactions: pd.DataFrame, settings: dict) -> dict:
    df = transactions.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    records = df.to_dict(orient="records")
    clean_settings = {k: settings[k] for k in _DEFAULT_SETTINGS if k in settings}
    return {
        "schema_version": SCHEMA_VERSION,
        "transactions": records,
        "settings": clean_settings,
    }


def payload_to_csv_buffer(payload: dict) -> io.StringIO:
    df = pd.DataFrame(payload.get("transactions", []))
    # Ensure a stable column set (Price may be absent in older/price-less saves).
    for col in _CSV_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_CSV_COLUMNS]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf


def payload_settings(payload: dict) -> dict:
    stored = (payload or {}).get("settings", {}) or {}
    merged = dict(_DEFAULT_SETTINGS)
    for k in _DEFAULT_SETTINGS:
        if k in stored and stored[k] is not None:
            merged[k] = stored[k]
    return merged
