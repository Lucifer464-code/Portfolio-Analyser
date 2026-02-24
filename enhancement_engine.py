# ==========================================================
# ENHANCEMENT ENGINE — GLOBAL MOMENTUM + ALPHA RECOMMENDER
# Optimized v3
# Changes vs v2:
#   1. get_sp500_constituents — 7-day staleness check on cached CSV
#   2. Momentum metrics fully vectorised (no per-ticker Python loop)
#   3. compute_portfolio_3m_relative_performance — loop replaced with
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
from typing import List, Optional
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

BENCHMARK         = "SPY"
LOOKBACK          = "3y"
TOP_N             = 15
SP500_FILE        = "sp500_constituents.csv"
SP500_MAX_AGE_DAYS = 7     # FIX 1: refresh constituent list weekly
MAX_PE_WORKERS    = 20     # parallel threads for PE fetching
MIN_HISTORY_DAYS  = 252    # require 1 full year of price history


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
# FIX 4: Shared price downloader — one download, two users
# ----------------------------------------------------------

def _download_prices(tickers: List[str], period: str) -> pd.DataFrame:
    """
    Single batched yfinance download. Returns a clean Close-price DataFrame.
    Both generate_enhancement_recommendations and
    compute_portfolio_3m_relative_performance call this so the data is
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

def _fetch_pe_ratios(tickers: List[str]) -> dict:
    """
    Fetch trailingPE and forwardPE for a list of tickers concurrently.
    Only called on the pre-filtered top-N*3 candidates, not all 500.
    """
    def _fetch_one(ticker):
        try:
            info = yf.Ticker(ticker).info
            trailing = info.get("trailingPE")
            forward  = info.get("forwardPE")
            return ticker, {
                "PE Ratio":  float(trailing) if trailing is not None else np.nan,
                "Forward PE":float(forward)  if forward  is not None else np.nan,
            }
        except Exception:
            return ticker, {"PE Ratio": np.nan, "Forward PE": np.nan}

    pe_map = {}
    with ThreadPoolExecutor(max_workers=MAX_PE_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            pe_map[ticker] = data

    return pe_map


# ----------------------------------------------------------
# Fetch PE Ratio and ROE for tickers
# ----------------------------------------------------------

def _fetch_pe_and_roe(tickers: List[str]) -> dict:
    """
    Fetch trailingPE, forwardPE and ROE for a list of tickers concurrently.
    """
    def _fetch_one(ticker):
        try:
            info = yf.Ticker(ticker).info
            trailing = info.get("trailingPE")
            forward  = info.get("forwardPE")
            roe      = info.get("returnOnEquity")
            return ticker, {
                "PE Ratio":  float(trailing) if trailing is not None else np.nan,
                "Forward PE":float(forward)  if forward  is not None else np.nan,
                "ROE":       float(roe) if roe is not None else np.nan,
            }
        except Exception:
            return ticker, {"PE Ratio": np.nan, "Forward PE": np.nan, "ROE": np.nan}

    pe_roe_map = {}
    with ThreadPoolExecutor(max_workers=MAX_PE_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, data = future.result()
            pe_roe_map[ticker] = data

    return pe_roe_map


# ----------------------------------------------------------
# FIX 2: Vectorised momentum scoring
# ----------------------------------------------------------

def _compute_momentum_scores(
    price_data: pd.DataFrame,
    tickers: List[str],
    spy_ret_12m: float,
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
    df["alpha_12m"] = df["ret_12m"] - spy_ret_12m

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
        "alpha_12m":"Alpha vs SPY (12M)",
        "vol_1y":   "1Y Volatility",
        "trend_ok": "Above 200DMA",
    })

    return df[[
        "Ticker", "Current Price",
        "6M Return", "12M Return", "Alpha vs SPY (12M)",
        "1Y Volatility", "Above 200DMA", "Score",
    ]]


# ----------------------------------------------------------
# Core Enhancement Engine
# ----------------------------------------------------------

def generate_enhancement_recommendations(
    top_n: int = TOP_N,
    price_data: Optional[pd.DataFrame] = None,   # FIX 4: accept pre-fetched data
) -> pd.DataFrame:
    """
    Screen S&P 500 for top momentum + alpha opportunities.

    Parameters
    ----------
    top_n : int
        Number of final recommendations to return.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
        Pass this in from an external caller to avoid duplicate downloads.
    """

    # ── 1. Load constituents ──────────────────────────────
    try:
        sp500 = get_sp500_constituents()
    except Exception as e:
        raise Exception(f"S&P 500 loading failed: {e}")

    tickers: List[str] = sp500["Ticker"].tolist()

    # ── 2. Price download (shared or fresh) ───────────────
    if price_data is None:
        price_data = _download_prices(tickers + [BENCHMARK], LOOKBACK)

    if price_data is None or price_data.empty:
        return pd.DataFrame()

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if BENCHMARK not in price_data.columns:
        raise Exception("Benchmark (SPY) missing from price data.")

    spy_prices  = price_data[BENCHMARK].dropna()
    spy_ret_12m = _period_return(spy_prices, 252)

    # ── 3. Vectorised momentum scoring ───────────────────── FIX 2
    df = _compute_momentum_scores(price_data, tickers, spy_ret_12m)

    if df.empty:
        return pd.DataFrame()

    # ── 4. Pre-filter to top 3× before fetching PE ────────
    #    Limits PE API calls to ~45 tickers instead of 500
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
# FIX 3: Vectorised 3M Relative Performance
# ==========================================================

def compute_portfolio_3m_relative_performance(
    tickers: List[str],
    price_data: Optional[pd.DataFrame] = None,   # FIX 4: accept pre-fetched data
) -> pd.DataFrame:
    """
    Compute each holding's 3-month return vs the SPY benchmark.

    Parameters
    ----------
    tickers : list of str
        Portfolio ticker symbols.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
    """

    # ── Price data ────────────────────────────────────────
    if price_data is None:
        price_data = _download_prices(tickers + [BENCHMARK], LOOKBACK)

    if price_data is None or price_data.empty:
        raise Exception("Price data unavailable.")

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if BENCHMARK not in price_data.columns:
        raise Exception("Benchmark (SPY) missing from price data.")

    bm_prices = price_data[BENCHMARK].dropna()

    if len(bm_prices) < 63:
        raise Exception("Insufficient benchmark history for 3M calculation.")

    benchmark_3m = float((bm_prices.iloc[-1] / bm_prices.iloc[-63]) - 1)

    # ── FIX 3: Vectorised 3M return for all tickers ───────
    # Filter to tickers present in price_data with enough history
    valid = [t for t in tickers if t in price_data.columns]
    prices = price_data[valid].dropna(how="all")

    # Keep only columns with at least 63 rows of data
    enough = prices.notna().sum() >= 63
    prices = prices.loc[:, enough]

    if prices.empty:
        return pd.DataFrame()

    # Compute 3M return for every ticker in one vectorised operation
    stock_3m = (prices.iloc[-1] / prices.iloc[-63] - 1)

    result = pd.DataFrame({
        "Ticker":               stock_3m.index,
        "3M Return":            stock_3m.values,
        "Benchmark 3M":         benchmark_3m,
        "Relative Performance": stock_3m.values - benchmark_3m,
    })

    return result.reset_index(drop=True)


# ==========================================================
# SECTOR-WISE ENHANCEMENT RECOMMENDATIONS
# ==========================================================

def generate_sector_wise_recommendations(
    top_sectors: int = 5,
    stocks_per_sector: int = 5,
    price_data: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Screen S&P 500 for top performing sectors and their best stocks.
    
    Parameters
    ----------
    top_sectors : int
        Number of top performing sectors to return.
    stocks_per_sector : int
        Number of top stocks per sector to return.
    price_data : pd.DataFrame, optional
        Pre-fetched price DataFrame. If None, data is downloaded here.
    
    Returns
    -------
    dict
        Dictionary with sectors as keys and DataFrames of top stocks as values.
    """

    # ── 1. Load constituents with sector info ─────────────
    try:
        sp500 = get_sp500_constituents()
    except Exception as e:
        raise Exception(f"S&P 500 loading failed: {e}")

    tickers: List[str] = sp500["Ticker"].tolist()
    sectors_map = dict(zip(sp500["Ticker"], sp500["Sector"]))

    # ── 2. Price download (shared or fresh) ───────────────
    if price_data is None:
        price_data = _download_prices(tickers + [BENCHMARK], LOOKBACK)

    if price_data is None or price_data.empty:
        return {}

    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data["Close"]

    if BENCHMARK not in price_data.columns:
        raise Exception("Benchmark (SPY) missing from price data.")

    spy_prices  = price_data[BENCHMARK].dropna()
    spy_ret_12m = _period_return(spy_prices, 252)

    # ── 3. Compute momentum scores for all tickers ────────
    df = _compute_momentum_scores(price_data, tickers, spy_ret_12m)

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