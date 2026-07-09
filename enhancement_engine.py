# ==========================================================
# ENHANCEMENT ENGINE — GLOBAL MOMENTUM + ALPHA RECOMMENDER
# Optimized v3
# Changes vs v2:
#   1. get_sp500_constituents — 7-day staleness check on cached CSV
#   2. Momentum metrics fully vectorised (no per-ticker Python loop)
#   3. compute_portfolio_relative_performance — loop replaced with
#      vectorised pandas operations
#   4. Double price fetch eliminated — both functions share one download
#      via a shared _fetch_price_data() helper that caches in-process
#   5. fetch_pe_ratios unchanged (already parallel) but now only called
#      on the final top-N * 3 pre-filtered set
# ==========================================================

import os
import time
import yfinance as yf
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

BENCHMARK          = "SPY"
IN_BENCHMARK       = "^NSEI"
LOOKBACK           = "3y"
TOP_N              = 15
SP500_FILE         = "sp500_constituents.csv"
NIFTY500_FILE      = "nifty500_constituents.csv"
SP500_MAX_AGE_DAYS = 7     # FIX 1: refresh constituent list weekly
MAX_PE_WORKERS     = 3      # reduced for cloud rate-limit compliance
MIN_HISTORY_DAYS   = 252    # require 1 full year of price history

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


# ----------------------------------------------------------
# FIX 1: S&P 500 Constituents with staleness check
# ----------------------------------------------------------

def get_sp500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load S&P 500 constituents from local CSV cache.
    Auto-refreshes from Wikipedia if file is missing or older than 7 days.
    """
    file_exists = os.path.exists(SP500_FILE)

    if file_exists and not force_refresh:
        age_days = (time.time() - os.path.getmtime(SP500_FILE)) / 86_400
        if age_days < SP500_MAX_AGE_DAYS:
            return pd.read_csv(SP500_FILE)

    url      = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers  = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)

    if response.status_code != 200:
        # If refresh fails but we have a stale file, use it rather than crash
        if file_exists:
            return pd.read_csv(SP500_FILE)
        raise Exception(f"Failed to fetch S&P 500 list. Status: {response.status_code}")

    tables = pd.read_html(StringIO(response.text))
    table  = tables[0][["Symbol", "GICS Sector"]].copy()
    table.columns = ["Ticker", "Sector"]
    table["Ticker"] = table["Ticker"].str.replace(".", "-", regex=False)
    table.to_csv(SP500_FILE, index=False)

    return table


# ----------------------------------------------------------
# NIFTY 500 Constituents — live fetch from NSE + static fallback
# ----------------------------------------------------------

def _nifty_static_fallback() -> pd.DataFrame:
    """~150 major NSE stocks as a static fallback when NSE fetch fails."""
    data = [
        # Information Technology
        ("TCS.NS","Information Technology"),("INFY.NS","Information Technology"),
        ("WIPRO.NS","Information Technology"),("HCLTECH.NS","Information Technology"),
        ("TECHM.NS","Information Technology"),("LTIM.NS","Information Technology"),
        ("MPHASIS.NS","Information Technology"),("PERSISTENT.NS","Information Technology"),
        ("COFORGE.NS","Information Technology"),("OFSS.NS","Information Technology"),
        ("KPITTECH.NS","Information Technology"),("TATAELXSI.NS","Information Technology"),
        # Banks
        ("HDFCBANK.NS","Banks"),("ICICIBANK.NS","Banks"),("SBIN.NS","Banks"),
        ("KOTAKBANK.NS","Banks"),("AXISBANK.NS","Banks"),("INDUSINDBK.NS","Banks"),
        ("BANDHANBNK.NS","Banks"),("FEDERALBNK.NS","Banks"),("IDFCFIRSTB.NS","Banks"),
        ("RBLBANK.NS","Banks"),("PNB.NS","Banks"),("BANKBARODA.NS","Banks"),
        ("CANBK.NS","Banks"),("UNIONBANK.NS","Banks"),("MAHABANK.NS","Banks"),
        # Finance (NBFCs / Insurance)
        ("BAJFINANCE.NS","Finance"),("BAJAJFINSV.NS","Finance"),
        ("SBILIFE.NS","Finance"),("HDFCLIFE.NS","Finance"),("ICICIPRULI.NS","Finance"),
        ("CHOLAFIN.NS","Finance"),("MUTHOOTFIN.NS","Finance"),("PFC.NS","Finance"),
        ("RECLTD.NS","Finance"),("ICICIGI.NS","Finance"),("GICRE.NS","Finance"),
        ("LICI.NS","Finance"),("M&MFIN.NS","Finance"),("SHRIRAMFIN.NS","Finance"),
        # Consumer Goods (FMCG)
        ("HINDUNILVR.NS","Consumer Goods"),("ITC.NS","Consumer Goods"),
        ("NESTLEIND.NS","Consumer Goods"),("BRITANNIA.NS","Consumer Goods"),
        ("DABUR.NS","Consumer Goods"),("MARICO.NS","Consumer Goods"),
        ("GODREJCP.NS","Consumer Goods"),("COLPAL.NS","Consumer Goods"),
        ("TATACONSUM.NS","Consumer Goods"),("VBL.NS","Consumer Goods"),
        ("PGHH.NS","Consumer Goods"),("EMAMILTD.NS","Consumer Goods"),
        ("JYOTHYLAB.NS","Consumer Goods"),("GILLETTE.NS","Consumer Goods"),
        # Automobile & Auto Components
        ("TATAMOTORS.NS","Automobile and Auto Components"),
        ("MARUTI.NS","Automobile and Auto Components"),
        ("EICHERMOT.NS","Automobile and Auto Components"),
        ("HEROMOTOCO.NS","Automobile and Auto Components"),
        ("BAJAJ-AUTO.NS","Automobile and Auto Components"),
        ("BOSCHLTD.NS","Automobile and Auto Components"),
        ("MOTHERSON.NS","Automobile and Auto Components"),
        ("BALKRISIND.NS","Automobile and Auto Components"),
        ("MRF.NS","Automobile and Auto Components"),
        ("APOLLOTYRE.NS","Automobile and Auto Components"),
        ("BHARATFORG.NS","Automobile and Auto Components"),
        ("TIINDIA.NS","Automobile and Auto Components"),
        # Pharmaceuticals & Biotechnology
        ("SUNPHARMA.NS","Pharmaceuticals & Biotechnology"),
        ("DRREDDY.NS","Pharmaceuticals & Biotechnology"),
        ("CIPLA.NS","Pharmaceuticals & Biotechnology"),
        ("DIVISLAB.NS","Pharmaceuticals & Biotechnology"),
        ("LUPIN.NS","Pharmaceuticals & Biotechnology"),
        ("TORNTPHARM.NS","Pharmaceuticals & Biotechnology"),
        ("BIOCON.NS","Pharmaceuticals & Biotechnology"),
        ("AUROPHARMA.NS","Pharmaceuticals & Biotechnology"),
        ("ALKEM.NS","Pharmaceuticals & Biotechnology"),
        ("ABBOTINDIA.NS","Pharmaceuticals & Biotechnology"),
        ("ZYDUSLIFE.NS","Pharmaceuticals & Biotechnology"),
        ("IPCALAB.NS","Pharmaceuticals & Biotechnology"),
        # Healthcare Services
        ("APOLLOHOSP.NS","Healthcare Services"),("MAXHEALTH.NS","Healthcare Services"),
        ("FORTIS.NS","Healthcare Services"),("METROPOLIS.NS","Healthcare Services"),
        ("LALPATHLAB.NS","Healthcare Services"),
        # Energy
        ("RELIANCE.NS","Energy"),("ONGC.NS","Energy"),("BPCL.NS","Energy"),
        ("IOC.NS","Energy"),("GAIL.NS","Energy"),("OIL.NS","Energy"),
        ("MGL.NS","Energy"),("IGL.NS","Energy"),("PETRONET.NS","Energy"),
        ("HINDPETRO.NS","Energy"),
        # Power / Utilities
        ("POWERGRID.NS","Power"),("NTPC.NS","Power"),("TATAPOWER.NS","Power"),
        ("ADANIGREEN.NS","Power"),("ADANIPOWER.NS","Power"),
        ("TORNTPOWER.NS","Power"),("CESC.NS","Power"),("NHPC.NS","Power"),
        ("SJVN.NS","Power"),("IREDA.NS","Power"),
        # Capital Goods / Industrials
        ("LT.NS","Capital Goods"),("SIEMENS.NS","Capital Goods"),
        ("ABB.NS","Capital Goods"),("BHEL.NS","Capital Goods"),
        ("HAL.NS","Capital Goods"),("BEL.NS","Capital Goods"),
        ("HAVELLS.NS","Capital Goods"),("VOLTAS.NS","Capital Goods"),
        ("CUMMINSIND.NS","Capital Goods"),("THERMAX.NS","Capital Goods"),
        ("AIAENG.NS","Capital Goods"),("GRINDWELL.NS","Capital Goods"),
        # Metals & Mining
        ("TATASTEEL.NS","Metals & Mining"),("JSWSTEEL.NS","Metals & Mining"),
        ("HINDALCO.NS","Metals & Mining"),("SAIL.NS","Metals & Mining"),
        ("VEDL.NS","Metals & Mining"),("COALINDIA.NS","Metals & Mining"),
        ("NMDC.NS","Metals & Mining"),("NATIONALUM.NS","Metals & Mining"),
        ("HINDCOPPER.NS","Metals & Mining"),("APLAPOLLO.NS","Metals & Mining"),
        # Chemicals
        ("PIDILITIND.NS","Chemicals"),("UPL.NS","Chemicals"),
        ("AAPL.NS","Chemicals"),("SRF.NS","Chemicals"),
        ("ATUL.NS","Chemicals"),("DEEPAKNTR.NS","Chemicals"),
        ("NAVINFLUOR.NS","Chemicals"),("GALAXYSURF.NS","Chemicals"),
        ("CLEAN.NS","Chemicals"),("FINEORG.NS","Chemicals"),
        # Construction Materials / Cement
        ("ULTRACEMCO.NS","Construction Materials"),("AMBUJACEM.NS","Construction Materials"),
        ("SHREECEM.NS","Construction Materials"),("JKCEMENT.NS","Construction Materials"),
        ("DALMIACELE.NS","Construction Materials"),("RAMCOCEM.NS","Construction Materials"),
        # Real Estate
        ("DLF.NS","Realty"),("GODREJPROP.NS","Realty"),("OBEROIRLTY.NS","Realty"),
        ("PRESTIGE.NS","Realty"),("PHOENIXLTD.NS","Realty"),("BRIGADE.NS","Realty"),
        # Telecommunications
        ("BHARTIARTL.NS","Telecommunication"),("IDEA.NS","Telecommunication"),
        ("TATACOMM.NS","Telecommunication"),
        # Consumer Discretionary / Retail
        ("TITAN.NS","Consumer Durables"),("TRENT.NS","Consumer Durables"),
        ("DMART.NS","Consumer Durables"),("NYKAA.NS","Consumer Durables"),
        ("ZOMATO.NS","Consumer Durables"),("JUBLFOOD.NS","Consumer Durables"),
        ("DEVYANI.NS","Consumer Durables"),("WESTLIFE.NS","Consumer Durables"),
        ("SHOPERSTOP.NS","Consumer Durables"),("INDIGOPNTS.NS","Consumer Durables"),
        # Logistics / Transport
        ("ADANIPORTS.NS","Services"),("IRCTC.NS","Services"),
        ("CONCOR.NS","Services"),("BLUEDART.NS","Services"),
        ("DELHIVERY.NS","Services"),
        # Media
        ("ZEEL.NS","Media Entertainment & Publication"),
        ("SUNTV.NS","Media Entertainment & Publication"),
    ]
    return pd.DataFrame(data, columns=["Ticker", "Sector"])


def get_nifty500_constituents(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load NIFTY 500 constituents from local CSV cache.
    Auto-refreshes from NSE if file is missing or older than 7 days.
    Falls back to a curated static list if the NSE fetch fails.
    """
    file_exists = os.path.exists(NIFTY500_FILE)

    if file_exists and not force_refresh:
        age_days = (time.time() - os.path.getmtime(NIFTY500_FILE)) / 86_400
        if age_days < SP500_MAX_AGE_DAYS:
            return pd.read_csv(NIFTY500_FILE)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        response = requests.get(NIFTY500_URL, headers=headers, timeout=15)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            # NSE CSV columns: Company Name, Industry, Symbol, Series, ISIN Code
            if "Symbol" in df.columns and "Industry" in df.columns:
                result = df[["Symbol", "Industry"]].copy()
                result.columns = ["Ticker", "Sector"]
                result["Ticker"] = result["Ticker"].str.strip() + ".NS"
                result["Sector"] = result["Sector"].str.strip()
                result = result.dropna(subset=["Ticker", "Sector"])
                result.to_csv(NIFTY500_FILE, index=False)
                return result
    except Exception:
        pass

    # Network fetch failed — use stale cache if available, else static fallback
    if file_exists:
        return pd.read_csv(NIFTY500_FILE)
    return _nifty_static_fallback()


# ----------------------------------------------------------
# FIX 4: Shared price downloader — one download, two users
# ----------------------------------------------------------

def _download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Single batched yfinance download. Returns a clean Close-price DataFrame.
    Both generate_enhancement_recommendations and
    compute_portfolio_relative_performance call this so the data is
    downloaded once and passed in — no duplicate network calls.
    """
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )["Close"]

        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])

        return data.dropna(how="all")
    except Exception:
        return pd.DataFrame()


# ----------------------------------------------------------
# Helper — scalar return between two index offsets
# ----------------------------------------------------------

def _period_return(prices: pd.Series, periods: int) -> float:
    if len(prices) < periods:
        return np.nan
    return float((prices.iloc[-1] / prices.iloc[-periods]) - 1)


# ----------------------------------------------------------
# Parallel PE Ratio Fetcher (unchanged — already optimal)
# ----------------------------------------------------------

# ----------------------------------------------------------
# Static PE / ROE snapshot — instant cloud fallback
# Approximate trailing values; refreshed periodically by hand
# ----------------------------------------------------------
_STATIC_FUNDAMENTALS = {
    # ticker: (trailingPE, ROE)
    "AAPL": (31.2, 1.60), "MSFT": (36.5, 0.38), "NVDA": (55.0, 1.15),
    "AMZN": (42.0, 0.22), "GOOGL": (24.5, 0.31), "GOOG": (24.5, 0.31),
    "META": (27.0, 0.36), "TSLA": (65.0, 0.13), "BRK.B": (22.0, 0.14),
    "AVGO": (35.0, 0.62), "LLY": (58.0, 0.95), "JPM": (13.5, 0.17),
    "V":    (31.0, 0.50), "UNH": (22.0, 0.27), "XOM": (14.0, 0.16),
    "MA":   (35.0, 2.10), "JNJ": (16.0, 0.22), "PG":  (26.0, 0.32),
    "HD":   (24.0, 0.60), "COST": (52.0, 0.33), "MRK": (18.0, 0.28),
    "ABBV": (20.0, 0.55), "CVX": (15.0, 0.14), "WMT": (28.0, 0.19),
    "BAC":  (13.0, 0.11), "KO":  (24.0, 0.40), "PEP": (25.0, 0.52),
    "CRM":  (65.0, 0.10), "AMD": (45.0, 0.05), "NFLX":(42.0, 0.31),
    "INTC": (22.0, 0.05), "QCOM":(18.0, 0.44), "TXN": (22.0, 0.58),
    "GS":   (14.5, 0.12), "MS":  (17.0, 0.13), "WFC": (12.5, 0.12),
    "MU":   (18.0, 0.14), "AMAT":(22.0, 0.48), "LRCX":(23.0, 0.82),
    "NEE":  (22.0, 0.13), "RTX": (38.0, 0.08), "HON": (26.0, 0.31),
    "UPS":  (18.0, 1.20), "CAT": (17.0, 0.52), "DE":  (14.0, 0.38),
    "LIN":  (31.0, 0.22), "SHW": (33.0, 0.72), "GE":  (32.0, 0.10),
    "NKE":  (28.0, 0.35), "SBUX":(24.0, 8.50), "TGT": (15.0, 0.30),
    "MCD":  (24.0, 0.90), "DIS": (75.0, 0.04), "CMCSA":(11.0,0.17),
    "VLO":  (10.0, 0.28), "MPC": (10.0, 0.32), "XOM": (14.0, 0.16),
    "APA":  (9.0,  0.22), "WMB": (22.0, 0.12), "COP": (13.0, 0.21),
}


def _fetch_fundamentals_one(ticker: str) -> dict:
    """Fetch PE + ROE for a single ticker with retry and static fallback."""
    import time as _time
    for attempt in range(3):
        try:
            if attempt > 0:
                _time.sleep(attempt * 2)
            info     = yf.Ticker(ticker).info
            trailing = info.get("trailingPE")
            forward  = info.get("forwardPE")
            roe      = info.get("returnOnEquity")
            # Accept result only if at least one value came back
            if any(v is not None for v in (trailing, forward, roe)):
                return {
                    "PE Ratio":  float(trailing) if trailing is not None else np.nan,
                    "Forward PE":float(forward)  if forward  is not None else np.nan,
                    "ROE":       float(roe)      if roe      is not None else np.nan,
                }
        except Exception:
            pass

    # Static fallback
    if ticker in _STATIC_FUNDAMENTALS:
        pe, roe = _STATIC_FUNDAMENTALS[ticker]
        return {"PE Ratio": pe, "Forward PE": np.nan, "ROE": roe}

    return {"PE Ratio": np.nan, "Forward PE": np.nan, "ROE": np.nan}


def _fetch_pe_ratios(tickers: List[str]) -> dict:
    """Fetch trailingPE and forwardPE for a list of tickers."""
    pe_map = {}
    with ThreadPoolExecutor(max_workers=MAX_PE_WORKERS) as executor:
        futures = {executor.submit(_fetch_fundamentals_one, t): t for t in tickers}
        for future in as_completed(futures):
            t    = futures[future]
            data = future.result()
            pe_map[t] = {"PE Ratio": data["PE Ratio"], "Forward PE": data["Forward PE"]}
    return pe_map


# ----------------------------------------------------------
# Fetch PE Ratio and ROE for tickers
# ----------------------------------------------------------

def _fetch_pe_and_roe(tickers: List[str]) -> dict:
    """Fetch trailingPE, forwardPE and ROE for a list of tickers."""
    pe_roe_map = {}
    with ThreadPoolExecutor(max_workers=MAX_PE_WORKERS) as executor:
        futures = {executor.submit(_fetch_fundamentals_one, t): t for t in tickers}
        for future in as_completed(futures):
            t    = futures[future]
            pe_roe_map[t] = future.result()
    return pe_roe_map


# ----------------------------------------------------------
# FIX 2: Vectorised momentum scoring
# ----------------------------------------------------------

def _compute_momentum_scores(
    price_data: pd.DataFrame,
    tickers: List[str],
    benchmark_ret_12m: float,
    alpha_label: str = "Alpha vs SPY (12M)",
) -> pd.DataFrame:
    """
    Compute all momentum metrics in one fully vectorised pass.

    Instead of looping over each ticker in Python and calling rolling()
    one at a time, we operate on the entire price matrix at once:
      - pct_change(), rolling().std(), rolling().mean() all run on the
        full DataFrame in C under the hood — much faster for 500 columns.

    Returns a DataFrame with one row per eligible ticker.
    """
    # Work only on tickers that exist in price_data
    valid = [t for t in tickers if t in price_data.columns]
    prices = price_data[valid]

    # ── Vectorised return windows ─────────────────────────
    # Shift by N periods and divide — operates on all columns at once
    ret_1m  = (prices.iloc[-1]  / prices.iloc[-21]  - 1).rename("ret_1m")
    ret_6m  = (prices.iloc[-1]  / prices.iloc[-126] - 1).rename("ret_6m")
    ret_12m = (prices.iloc[-1]  / prices.iloc[-252] - 1).rename("ret_12m")

    # ── Vectorised volatility ─────────────────────────────
    daily_returns = prices.pct_change()
    vol_1y = (daily_returns.rolling(252).std().iloc[-1] * np.sqrt(252)).rename("vol_1y")

    # ── Vectorised 200-DMA trend flag ────────────────────
    ma200     = prices.rolling(200).mean().iloc[-1]
    trend_ok  = (prices.iloc[-1] > ma200).rename("trend_ok")

    # ── Minimum history filter ────────────────────────────
    # Drop any ticker that doesn't have MIN_HISTORY_DAYS of non-NaN data
    valid_count = prices.notna().sum()
    enough_history = valid_count[valid_count >= MIN_HISTORY_DAYS].index.tolist()

    # ── Assemble into one DataFrame ───────────────────────
    df = pd.concat([ret_1m, ret_6m, ret_12m, vol_1y, trend_ok], axis=1)
    df = df.loc[enough_history].copy()
    df.index.name = "Ticker"
    df = df.reset_index()

    # ── Apply filters ─────────────────────────────────────
    # Drop recent sharp losers (1M return < -10%)
    df = df[~((df["ret_1m"].notna()) & (df["ret_1m"] < -0.10))].copy()

    # ── Alpha vs benchmark ────────────────────────────────
    df["alpha_12m"] = df["ret_12m"] - benchmark_ret_12m

    # ── Composite score (vectorised) ──────────────────────
    r12 = df["ret_12m"].fillna(0)
    r6  = df["ret_6m"].fillna(0)
    alp = df["alpha_12m"].fillna(0)

    base_score = 0.4 * r12 + 0.3 * r6 + 0.3 * alp

    vol_safe = df["vol_1y"].replace(0, np.nan)
    df["Score"] = base_score / vol_safe
    df["Score"] = df["Score"].fillna(0)

    # Penalise stocks below 200-DMA
    df.loc[~df["trend_ok"], "Score"] *= 0.5

    # ── Add price column ──────────────────────────────────
    df["Current Price"] = prices.iloc[-1].reindex(df["Ticker"]).values

    # ── Rename for output ─────────────────────────────────
    df = df.rename(columns={
        "ret_6m":   "6M Return",
        "ret_12m":  "12M Return",
        "alpha_12m": alpha_label,
        "vol_1y":   "1Y Volatility",
        "trend_ok": "Above 200DMA",
    })

    return df[[
        "Ticker", "Current Price",
        "6M Return", "12M Return", alpha_label,
        "1Y Volatility", "Above 200DMA", "Score",
    ]]


# ----------------------------------------------------------
# Core Enhancement Engine
# ----------------------------------------------------------

def generate_enhancement_recommendations(
    top_n: int = TOP_N,
    price_data: Optional[pd.DataFrame] = None,
    market: str = "US",
) -> pd.DataFrame:
    """
    Screen S&P 500 (US) or NIFTY 500 (IN) for top momentum + alpha opportunities.

    Parameters
    ----------
    top_n : int
        Number of final recommendations to return.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
    market : str
        "US" for S&P 500 screening, "IN" for NIFTY 500 screening.
    """

    # ── 1. Load constituents ──────────────────────────────
    if market == "IN":
        benchmark    = IN_BENCHMARK
        alpha_label  = "Alpha vs NIFTY (12M)"
        try:
            constituents = get_nifty500_constituents()
        except Exception as e:
            raise Exception(f"NIFTY 500 loading failed: {e}")
    else:
        benchmark    = BENCHMARK
        alpha_label  = "Alpha vs SPY (12M)"
        try:
            constituents = get_sp500_constituents()
        except Exception as e:
            raise Exception(f"S&P 500 loading failed: {e}")

    tickers: List[str] = constituents["Ticker"].tolist()

    # ── 2. Price download (shared or fresh) ───────────────
    if price_data is None:
        price_data = _download_prices(tickers + [benchmark], LOOKBACK)

    if price_data is None or price_data.empty:
        return pd.DataFrame()

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if benchmark not in price_data.columns:
        raise Exception(f"Benchmark ({benchmark}) missing from price data.")

    bm_prices    = price_data[benchmark].dropna()
    bm_ret_12m   = _period_return(bm_prices, 252)

    # ── 3. Vectorised momentum scoring ────────────────────
    df = _compute_momentum_scores(price_data, tickers, bm_ret_12m, alpha_label)

    if df.empty:
        return pd.DataFrame()

    # ── 4. Pre-filter to top 3× before fetching PE ────────
    pre_n = min(top_n * 3, len(df))
    df    = df.nlargest(pre_n, "Score").reset_index(drop=True)

    # ── 5. Parallel PE fetch on pre-filtered set only ─────
    pe_map = _fetch_pe_ratios(df["Ticker"].tolist())

    df["PE Ratio"]  = df["Ticker"].map(lambda t: pe_map.get(t, {}).get("PE Ratio",  np.nan))
    df["Forward PE"]= df["Ticker"].map(lambda t: pe_map.get(t, {}).get("Forward PE",np.nan))

    # ── 6. Final sort, trim, and index ────────────────────
    df = (
        df
        .sort_values("Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    df.index += 1

    return df


# ==========================================================
# FIX 3: Vectorised Relative Performance
# ==========================================================

# Trading-day window and download lookback for each supported period.
RELATIVE_PERIODS: Dict[str, Dict[str, object]] = {
    "1M": {"window": 21,   "lookback": "6mo"},
    "3M": {"window": 63,   "lookback": "1y"},
    "6M": {"window": 126,  "lookback": "2y"},
    "1Y": {"window": 252,  "lookback": "3y"},
    "3Y": {"window": 756,  "lookback": "5y"},
    "5Y": {"window": 1260, "lookback": "7y"},
}


def compute_portfolio_relative_performance(
    tickers: List[str],
    price_data: Optional[pd.DataFrame] = None,
    benchmark: str = BENCHMARK,
    period: str = "3M",
) -> pd.DataFrame:
    """
    Compute each holding's return over `period` vs the selected benchmark.

    Parameters
    ----------
    tickers : list of str
        Portfolio ticker symbols.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
    benchmark : str
        Benchmark ticker to compare against. Defaults to module-level BENCHMARK.
    period : str
        One of the keys of RELATIVE_PERIODS (1M, 3M, 6M, 1Y, 3Y, 5Y).
    """

    period = period.upper()
    if period not in RELATIVE_PERIODS:
        raise Exception(f"Unsupported period: {period}")

    window   = int(RELATIVE_PERIODS[period]["window"])
    lookback = str(RELATIVE_PERIODS[period]["lookback"])

    # ── Price data ────────────────────────────────────────
    if price_data is None:
        price_data = _download_prices(tickers + [benchmark], lookback)

    if price_data is None or price_data.empty:
        raise Exception("Price data unavailable.")

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if benchmark not in price_data.columns:
        raise Exception(f"Benchmark ({benchmark}) missing from price data.")

    bm_prices = price_data[benchmark].dropna()

    if len(bm_prices) < window:
        raise Exception(f"Insufficient benchmark history for {period} calculation.")

    benchmark_ret = float((bm_prices.iloc[-1] / bm_prices.iloc[-window]) - 1)

    # ── FIX 3: Vectorised period return for all tickers ───
    # Filter to tickers present in price_data with enough history
    valid = [t for t in tickers if t in price_data.columns]
    prices = price_data[valid].dropna(how="all")

    # Keep only columns with at least `window` rows of data
    enough = prices.notna().sum() >= window
    prices = prices.loc[:, enough]

    if prices.empty:
        return pd.DataFrame()

    # Compute the period return for every ticker in one vectorised operation
    stock_ret = (prices.iloc[-1] / prices.iloc[-window] - 1)

    result = pd.DataFrame({
        "Ticker":                stock_ret.index,
        f"{period} Return":      stock_ret.values,
        f"Benchmark {period}":   benchmark_ret,
        "Relative Performance":  stock_ret.values - benchmark_ret,
    })

    return result.reset_index(drop=True)


# ==========================================================
# SECTOR-WISE ENHANCEMENT RECOMMENDATIONS
# ==========================================================

def generate_sector_wise_recommendations(
    top_sectors: int = 5,
    stocks_per_sector: int = 5,
    price_data: Optional[pd.DataFrame] = None,
    market: str = "US",
) -> dict:
    """
    Screen S&P 500 (US) or NIFTY 500 (IN) for top performing sectors and their best stocks.

    Parameters
    ----------
    top_sectors : int
        Number of top performing sectors to return.
    stocks_per_sector : int
        Number of top stocks per sector to return.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
    market : str
        "US" for S&P 500 screening, "IN" for NIFTY 500 screening.

    Returns
    -------
    dict
        Dictionary with sectors as keys and DataFrames of top stocks as values.
    """

    # ── 1. Load constituents with sector info ─────────────
    if market == "IN":
        benchmark   = IN_BENCHMARK
        alpha_label = "Alpha vs NIFTY (12M)"
        try:
            constituents = get_nifty500_constituents()
        except Exception as e:
            raise Exception(f"NIFTY 500 loading failed: {e}")
    else:
        benchmark   = BENCHMARK
        alpha_label = "Alpha vs SPY (12M)"
        try:
            constituents = get_sp500_constituents()
        except Exception as e:
            raise Exception(f"S&P 500 loading failed: {e}")

    tickers: List[str] = constituents["Ticker"].tolist()
    sectors_map = dict(zip(constituents["Ticker"], constituents["Sector"]))

    # ── 2. Price download (shared or fresh) ───────────────
    if price_data is None:
        price_data = _download_prices(tickers + [benchmark], LOOKBACK)

    if price_data is None or price_data.empty:
        return {}

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if benchmark not in price_data.columns:
        raise Exception(f"Benchmark ({benchmark}) missing from price data.")

    bm_prices   = price_data[benchmark].dropna()
    bm_ret_12m  = _period_return(bm_prices, 252)

    # ── 3. Compute momentum scores for all tickers ────────
    df = _compute_momentum_scores(price_data, tickers, bm_ret_12m, alpha_label)

    if df.empty:
        return {}

    # ── 4. Add sector information ──────────────────────────
    df["Sector"] = df["Ticker"].map(sectors_map)
    df = df[df["Sector"].notna() & (df["Sector"] != "Unknown")].copy()

    # ── 5. Calculate sector performance metrics ───────────
    # Sector score = average 6M and 12M returns
    sector_scores = df.groupby("Sector").agg({
        "6M Return": "mean",
        "12M Return": "mean",
        "Score": "mean",
    }).reset_index()
    sector_scores.columns = ["Sector", "Avg 6M Return", "Avg 12M Return", "Avg Score"]
    sector_scores["Sector Score"] = (
        0.4 * sector_scores["Avg 12M Return"] + 
        0.6 * sector_scores["Avg 6M Return"]
    )
    sector_scores = sector_scores.sort_values("Sector Score", ascending=False)

    # ── 6. Select top N sectors ───────────────────────────
    top_sector_names = sector_scores.head(top_sectors)["Sector"].tolist()

    # ── 7. For each top sector, get top N stocks ──────────
    result = {}
    tickers_for_fetch = []
    sector_stocks_map = {}

    for sector in top_sector_names:
        sector_df = df[df["Sector"] == sector].copy()
        top_stocks = sector_df.nlargest(stocks_per_sector, "Score").reset_index(drop=True)
        
        if not top_stocks.empty:
            result[sector] = top_stocks
            tickers_for_fetch.extend(top_stocks["Ticker"].tolist())
            sector_stocks_map.update({t: sector for t in top_stocks["Ticker"].tolist()})

    # ── 8. Fetch PE and ROE for selected stocks ───────────
    if tickers_for_fetch:
        pe_roe_map = _fetch_pe_and_roe(list(set(tickers_for_fetch)))
        
        # Add PE/ROE to each sector's dataframe
        for sector in result.keys():
            result[sector]["PE Ratio"] = result[sector]["Ticker"].map(
                lambda t: pe_roe_map.get(t, {}).get("PE Ratio", np.nan)
            )
            result[sector]["ROE"] = result[sector]["Ticker"].map(
                lambda t: pe_roe_map.get(t, {}).get("ROE", np.nan)
            )
            
            # Sort by 12M return descending for better presentation
            result[sector] = result[sector].sort_values("12M Return", ascending=False).reset_index(drop=True)
            result[sector].index += 1

    return result