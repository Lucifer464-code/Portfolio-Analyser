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


def resolve_source(upload_id, pending_name, last_upload_id, active_saved_name=None):
    """Decide whether the active portfolio comes from the uploader or a saved entry.

    Streamlit's file_uploader keeps returning the same object on every rerun
    until the user clears it, so the mere presence of an upload cannot mean
    "the user just uploaded something". An upload only wins when its id differs
    from the one already consumed; otherwise a stale upload would permanently
    swallow every saved-portfolio load.

    active_saved_name keeps a loaded saved portfolio active across later reruns,
    when the pending flag has been consumed but the uploader still holds a file.

    Returns (source, source_id) where source is "upload", "saved", or None.
    """
    upload_is_fresh = upload_id is not None and upload_id != last_upload_id

    if upload_is_fresh:
        return "upload", f"upload:{upload_id}"
    if pending_name is not None:
        return "saved", f"saved:{pending_name}"
    if active_saved_name is not None:
        return "saved", f"saved:{active_saved_name}"
    if upload_id is not None:
        return "upload", f"upload:{upload_id}"
    return None, None


def payload_settings(payload: dict) -> dict:
    stored = (payload or {}).get("settings", {}) or {}
    merged = dict(_DEFAULT_SETTINGS)
    for k in _DEFAULT_SETTINGS:
        if k in stored and stored[k] is not None:
            merged[k] = stored[k]
    return merged
