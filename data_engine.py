import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from typing import Tuple, Dict, List

# ==========================================================
# EXPECTED COLUMNS
# ==========================================================

TRANSACTION_COLUMNS = {"Ticker", "Date", "Action", "Quantity", "Price"}

# ==========================================================
# CSV LOADING & VALIDATION — TRANSACTION FORMAT
# Expects: Ticker, Date, Action (Buy/Sell), Quantity, Price
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

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace('\ufeff', '', regex=False)
    )
    diagnostics["detected_columns"] = df.columns.tolist()

    # ── Flexible column mapping ────────────────────────────
    col_normalised = {
        c: c.lower().strip()
             .replace(" ", "").replace("_", "").replace(".", "")
             .replace("-", "").replace("/", "")
        for c in df.columns
    }

    def _find_col(keywords):
        for col, norm in col_normalised.items():
            for kw in keywords:
                if kw in norm:
                    return col
        return None

    ticker_kw   = ["ticker", "symbol", "stock", "scrip", "instrument", "asset", "code", "sym"]
    date_kw     = ["date", "dt", "transactiondate", "tradedate", "purchasedate", "buydate", "time"]
    action_kw   = ["action", "type", "transactiontype", "buysell", "side", "buyselltype",
                   "ordertype", "txntype", "txtype", "direction"]
    quantity_kw = ["quantity", "qty", "units", "shares", "amount", "noofshares",
                   "numberofshares", "vol", "volume", "lot"]
    price_kw    = ["price", "rate", "cost", "avgcost", "averagecost", "buyprice", "sellprice",
                   "tradeprice", "transactionprice", "purchaseprice", "avgprice", "averageprice"]

    col_map = {}
    for keywords, key, label in [
        (ticker_kw,   "Ticker",   "Ticker"),
        (date_kw,     "Date",     "Date"),
        (action_kw,   "Action",   "Action (Buy/Sell)"),
        (quantity_kw, "Quantity", "Quantity"),
        (price_kw,    "Price",    "Price"),
    ]:
        match = _find_col(keywords)
        if match:
            col_map[match] = key
        else:
            diagnostics["error"] = (
                f"Could not find a '{label}' column. "
                f"Please ensure your CSV has a column for {label}."
            )
            return None, diagnostics

    df = df.rename(columns=col_map)

    # ── Clean Ticker ───────────────────────────────────────
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()

    # ── Clean Date ─────────────────────────────────────────
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    invalid_dates = df["Date"].isna().sum()
    if invalid_dates > 0:
        diagnostics["warning_dates"] = (
            f"{invalid_dates} rows had unparseable dates and were dropped"
        )
    df = df.dropna(subset=["Date"])

    # ── Clean Action ───────────────────────────────────────
    df["Action"] = df["Action"].astype(str).str.strip().str.lower()
    buy_terms  = {"buy", "b", "purchase", "purchased", "bought", "long",
                  "acquire", "acquired", "in"}
    sell_terms = {"sell", "s", "sold", "sale", "short", "dispose", "disposed", "out"}

    def _normalise_action(val):
        v = val.lower().strip()
        if v in buy_terms:  return "Buy"
        if v in sell_terms: return "Sell"
        return None

    df["Action"] = df["Action"].map(_normalise_action)
    invalid_actions = df["Action"].isna().sum()
    if invalid_actions > 0:
        diagnostics["warning_actions"] = (
            f"{invalid_actions} rows had unrecognised actions and were dropped"
        )
    df = df[df["Action"].notna()]

    # ── Clean Quantity ─────────────────────────────────────
    df["Quantity"] = (
        df["Quantity"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=["Quantity"])
    df = df[df["Quantity"] > 0]

    # ── Clean Price ────────────────────────────────────────
    df["Price"] = (
        df["Price"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.dropna(subset=["Price"])
    df = df[df["Price"] > 0]

    if df.empty:
        diagnostics["error"] = "No valid rows after cleaning."
        return None, diagnostics

    df = df.sort_values("Date").reset_index(drop=True)
    diagnostics["rows_loaded"] = len(df)

    return df, diagnostics


# ==========================================================
# AGGREGATE TRANSACTIONS → CURRENT HOLDINGS
# ==========================================================

def aggregate_holdings(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate transaction log into current open positions.
    Uses average cost tracking.
    """
    holdings = {}

    for _, row in transactions.iterrows():
        ticker = row["Ticker"]
        qty    = float(row["Quantity"])
        price  = float(row["Price"])
        action = row["Action"]

        if ticker not in holdings:
            holdings[ticker] = {"Quantity": 0.0, "Total Cost": 0.0, "Realised P/L": 0.0}

        h = holdings[ticker]

        if action == "Buy":
            h["Total Cost"] += qty * price
            h["Quantity"]   += qty

        elif action == "Sell":
            if h["Quantity"] > 0:
                avg_cost          = h["Total Cost"] / h["Quantity"]
                realised          = qty * (price - avg_cost)
                h["Realised P/L"] += realised
                h["Total Cost"]   -= qty * avg_cost
                h["Quantity"]     -= qty
                if h["Quantity"] < 1e-6:
                    h["Quantity"]   = 0.0
                    h["Total Cost"] = 0.0

    rows = []
    for ticker, h in holdings.items():
        if h["Quantity"] > 1e-6:
            avg_cost = h["Total Cost"] / h["Quantity"] if h["Quantity"] > 0 else 0
            rows.append({
                "Ticker":       ticker,
                "Quantity":     h["Quantity"],
                "Avg Cost":     avg_cost,
                "Total Cost":   h["Total Cost"],
                "Realised P/L": h["Realised P/L"],
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["Ticker", "Quantity", "Avg Cost", "Total Cost", "Realised P/L"]
    )


# ==========================================================
# XIRR CALCULATION
# ==========================================================

def compute_xirr(transactions: pd.DataFrame, current_value: float,
                 today: date = None) -> float:
    """
    Compute portfolio-level XIRR from transaction log.
    Buy  → negative cash flow (money out)
    Sell → positive cash flow (money in)
    Today's market value → positive terminal cash flow
    """
    if today is None:
        today = date.today()

    cash_flows = []
    for _, row in transactions.iterrows():
        d   = row["Date"].date() if hasattr(row["Date"], "date") else row["Date"]
        qty = float(row["Quantity"])
        px  = float(row["Price"])
        amt = qty * px
        if row["Action"] == "Buy":
            cash_flows.append((d, -amt))
        else:
            cash_flows.append((d, +amt))

    cash_flows.append((today, +current_value))

    flows   = [cf for _, cf in cash_flows]
    has_neg = any(f < 0 for f in flows)
    has_pos = any(f > 0 for f in flows)
    if not has_neg or not has_pos:
        return np.nan

    dates   = [cf[0] for cf in cash_flows]
    amounts = [cf[1] for cf in cash_flows]
    t0      = dates[0]
    years   = [(d - t0).days / 365.0 for d in dates]

    def npv(rate):
        return sum(a / (1 + rate) ** t for a, t in zip(amounts, years))

    def dnpv(rate):
        return sum(-t * a / (1 + rate) ** (t + 1) for a, t in zip(amounts, years))

    rate = 0.1
    for _ in range(200):
        try:
            f  = npv(rate)
            df = dnpv(rate)
            if abs(df) < 1e-12:
                break
            new_rate = rate - f / df
            if abs(new_rate - rate) < 1e-8:
                rate = new_rate
                break
            rate = new_rate
            if rate <= -1:
                rate = -0.999
        except (ZeroDivisionError, OverflowError):
            return np.nan

    if rate <= -1 or abs(npv(rate)) > 1.0:
        return np.nan

    return round(rate, 6)


# ==========================================================
# REALISED VS UNREALISED P/L SUMMARY
# ==========================================================

def compute_pl_summary(holdings_df: pd.DataFrame, total_market_value: float) -> Dict:
    realised   = holdings_df["Realised P/L"].sum() if "Realised P/L" in holdings_df.columns else 0.0
    total_cost = holdings_df["Total Cost"].sum()   if "Total Cost"   in holdings_df.columns else 0.0
    unrealised = total_market_value - total_cost

    return {
        "realised_pl":    realised,
        "unrealised_pl":  unrealised,
        "total_cost":     total_cost,
        "total_pl":       realised + unrealised,
        "unrealised_pct": unrealised / total_cost if total_cost > 0 else 0.0,
    }


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
    return price_data.ffill().pct_change().dropna(how='all').fillna(0.0)


# ==========================================================
# SECTOR MAPPING
# ==========================================================

def fetch_sector_data(tickers: List[str]) -> Dict[str, str]:
    sector_map = {}
    for ticker in tickers:
        try:
            info   = yf.Ticker(ticker).info
            sector = info.get("sector")
            sector_map[ticker] = sector if sector else "Unknown"
        except Exception:
            sector_map[ticker] = "Unknown"
    return sector_map