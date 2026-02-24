import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, List

# ==========================================================
# OPTIONAL EXPECTED COLUMNS (NOT STRICTLY ENFORCED)
# ==========================================================

EXPECTED_COLUMNS = {
    "Asset Type",
    "Ticker (SYM)",
    "Ticker",
    "Symbol",
    "Quantity",
    "Avg. Cost",
    "Currency"
}

# ==========================================================
# CSV LOADING & VALIDATION (LOOSE + STABLE)
# ==========================================================

def load_and_validate_csv(uploaded_file) -> Tuple[pd.DataFrame, Dict]:

    diagnostics = {}

    try:
        df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as e:
        diagnostics["read_error"] = str(e)
        return None, diagnostics

    if df is None or df.empty:
        diagnostics["error"] = "CSV is empty"
        return None, diagnostics

    # ------------------------------------------------------
    # Clean Column Names
    # ------------------------------------------------------

    df.columns = (
        df.columns
            .str.strip()
            .str.replace('\ufeff', '', regex=False)
    )

    diagnostics["detected_columns"] = df.columns.tolist()

    # ------------------------------------------------------
    # Flexible Ticker Detection
    # ------------------------------------------------------

    ticker_candidates = ["Ticker (SYM)", "Ticker", "Symbol"]
    quantity_candidates = ["Quantity", "Qty", "Units", "Shares"]

    ticker_col = next((c for c in ticker_candidates if c in df.columns), None)
    quantity_col = next((c for c in quantity_candidates if c in df.columns), None)

    if ticker_col is None:
        diagnostics["error"] = "No ticker column detected."
        return None, diagnostics

    if quantity_col is None:
        diagnostics["error"] = "No quantity column detected."
        return None, diagnostics

    # ------------------------------------------------------
    # Standardize Columns
    # ------------------------------------------------------

    df = df.rename(columns={
        ticker_col: "Ticker",
        quantity_col: "Quantity"
    })

    # Clean ticker
    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Clean quantity (handle commas and currency symbols)
    df["Quantity"] = (
        df["Quantity"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

    # Drop invalid rows
    df = df.dropna(subset=["Ticker", "Quantity"])
    df = df[df["Quantity"] != 0]

    if df.empty:
        diagnostics["error"] = "No valid rows after cleaning."
        return None, diagnostics

    # ------------------------------------------------------
    # Optional Columns Handling
    # ------------------------------------------------------

    if "Avg. Cost" in df.columns:
        df["Avg. Cost"] = pd.to_numeric(df["Avg. Cost"], errors="coerce")

    if "Currency" in df.columns:
        df["Currency"] = df["Currency"].astype(str)

    if "Asset Type" in df.columns:
        df["Asset Type"] = df["Asset Type"].astype(str)

    diagnostics["rows_loaded"] = len(df)

    return df, diagnostics


# ==========================================================
# MARKET DATA
# ==========================================================

def fetch_market_data(tickers: List[str], period: str) -> pd.DataFrame:

    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    return prices.dropna(how="all")


def compute_returns(price_data: pd.DataFrame) -> pd.DataFrame:
    return price_data.pct_change().dropna()


# ==========================================================
# SECTOR MAPPING
# ==========================================================

def fetch_sector_data(tickers: List[str]) -> Dict[str, str]:

    sector_map = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector")
            sector_map[ticker] = sector if sector else "Unknown"
        except Exception:
            sector_map[ticker] = "Unknown"

    return sector_map