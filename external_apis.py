# ==========================================================
# EXTERNAL API INTEGRATIONS
# ==========================================================
# Sources:
#   Finnhub      (apiKey) — earnings surprises, analyst targets,
#                            insider sentiment, recommendations
#   FRED         (apiKey) — Fed rate, CPI, unemployment, 10Y yield, VIX
#   FMP          (apiKey) — income statements, key ratios, forward estimates
#   SEC EDGAR    (no key) — insider transactions from Form 4 filings
#   WallStreetBets(no key)— Reddit mention counts and sentiment
#
# All functions return clean DataFrames or dicts and fail silently.
# ==========================================================

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET

_HEADERS = {"User-Agent": "PortfolioAnalyser/1.0 research@portfolioanalyser.app"}
_TIMEOUT = 12


# ----------------------------------------------------------
# SHARED HTTP HELPER
# ----------------------------------------------------------

def _get(url, params=None, headers=None):
    try:
        r = requests.get(
            url, params=params,
            headers=headers or _HEADERS,
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ==========================================================
# FINNHUB
# ==========================================================

_FH = "https://finnhub.io/api/v1"


def finnhub_earnings_surprises(ticker: str, api_key: str) -> pd.DataFrame:
    """Last 8 quarterly earnings vs analyst estimates."""
    data = _get(f"{_FH}/stock/earnings",
                params={"symbol": ticker, "limit": 8, "token": api_key})
    if not data:
        return pd.DataFrame()
    rows = []
    for q in data:
        actual   = q.get("actual")
        estimate = q.get("estimate")
        surprise = q.get("surprisePercent")
        rows.append({
            "Period":     q.get("period", ""),
            "Estimate":   f"{estimate:.2f}"   if estimate is not None else "—",
            "Actual EPS": f"{actual:.2f}"     if actual   is not None else "—",
            "Surprise %": f"{surprise:+.2f}%" if surprise is not None else "—",
            "Result":     "Beat" if (surprise or 0) > 0
                          else ("Miss" if (surprise or 0) < 0 else "In line"),
        })
    return pd.DataFrame(rows)


def finnhub_price_target(ticker: str, api_key: str) -> dict:
    """Analyst consensus price target."""
    data = _get(f"{_FH}/stock/price-target",
                params={"symbol": ticker, "token": api_key})
    if not data:
        return {}
    return {
        "mean":    data.get("targetMean"),
        "high":    data.get("targetHigh"),
        "low":     data.get("targetLow"),
        "median":  data.get("targetMedian"),
        "updated": data.get("lastUpdated", ""),
    }


def finnhub_recommendations(ticker: str, api_key: str) -> pd.DataFrame:
    """Analyst buy/sell/hold recommendation trend — last 6 months."""
    data = _get(f"{_FH}/stock/recommendation",
                params={"symbol": ticker, "token": api_key})
    if not data:
        return pd.DataFrame()
    rows = []
    for r in data[:6]:
        rows.append({
            "Period":      r.get("period", "")[:7],
            "Strong Buy":  r.get("strongBuy",  0),
            "Buy":         r.get("buy",        0),
            "Hold":        r.get("hold",       0),
            "Sell":        r.get("sell",       0),
            "Strong Sell": r.get("strongSell", 0),
        })
    return pd.DataFrame(rows)


def finnhub_insider_sentiment(ticker: str, api_key: str) -> dict:
    """MSPR (Monthly Share Purchase Ratio) — past 12 months."""
    start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    end   = datetime.today().strftime("%Y-%m-%d")
    data  = _get(f"{_FH}/stock/insider-sentiment",
                 params={"symbol": ticker, "from": start, "to": end,
                         "token": api_key})
    if not data or not data.get("data"):
        return {}
    df = pd.DataFrame(data["data"])
    if df.empty:
        return {}
    return {
        "mspr_latest":  float(df["mspr"].iloc[-1])       if "mspr"   in df.columns else None,
        "mspr_3m":      float(df["mspr"].tail(3).mean()) if "mspr"   in df.columns else None,
        "net_purchase": int(df["change"].sum())           if "change" in df.columns else None,
    }


def finnhub_stock_metrics(ticker: str, api_key: str) -> dict:
    """
    50+ financial metrics in one call: margins, ratios, growth rates, beta.
    Returns the flat 'metric' dict from Finnhub /stock/metric.
    """
    data = _get(f"{_FH}/stock/metric",
                params={"symbol": ticker, "metric": "all", "token": api_key})
    if not data or "metric" not in data:
        return {}
    return data["metric"]


def finnhub_company_profile(ticker: str, api_key: str) -> dict:
    """Company profile: name, exchange, country, industry, market cap, logo, website."""
    data = _get(f"{_FH}/stock/profile2",
                params={"symbol": ticker, "token": api_key})
    return data or {}


def finnhub_peers(ticker: str, api_key: str) -> list:
    """List of peer ticker symbols from Finnhub."""
    data = _get(f"{_FH}/stock/peers",
                params={"symbol": ticker, "token": api_key})
    return data if isinstance(data, list) else []


def finnhub_company_news(ticker: str, api_key: str, days: int = 30) -> pd.DataFrame:
    """Recent company news — last `days` days, up to 20 articles."""
    from_dt = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_dt   = datetime.today().strftime("%Y-%m-%d")
    data    = _get(f"{_FH}/company-news",
                   params={"symbol": ticker, "from": from_dt,
                           "to": to_dt, "token": api_key})
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for item in data[:20]:
        rows.append({
            "headline": item.get("headline", ""),
            "source":   item.get("source",   ""),
            "url":      item.get("url",       ""),
            "datetime": item.get("datetime",  0),
            "summary":  item.get("summary",   ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
        df = df.sort_values("datetime", ascending=False)
    return df


def finnhub_portfolio_consensus(tickers: list, api_key: str) -> pd.DataFrame:
    """Latest analyst consensus rating for each portfolio ticker."""
    rows = []
    for ticker in tickers:
        data = _get(f"{_FH}/stock/recommendation",
                    params={"symbol": ticker, "token": api_key})
        if not data or len(data) == 0:
            continue
        r  = data[0]
        sb = r.get("strongBuy",  0)
        b  = r.get("buy",        0)
        h  = r.get("hold",       0)
        s  = r.get("sell",       0)
        ss = r.get("strongSell", 0)
        total = sb + b + h + s + ss
        if total == 0:
            continue
        score = (sb * 2 + b - s - ss * 2) / total
        if   score >  1.2: label = "Strong Buy"
        elif score >  0.3: label = "Buy"
        elif score > -0.3: label = "Hold"
        elif score > -1.2: label = "Sell"
        else:              label = "Strong Sell"
        rows.append({
            "Ticker":      ticker,
            "Strong Buy":  sb,
            "Buy":         b,
            "Hold":        h,
            "Sell":        s,
            "Strong Sell": ss,
            "Consensus":   label,
            "Score":       round(score, 2),
            "Period":      r.get("period", "")[:7],
        })
    return pd.DataFrame(rows)


# ==========================================================
# FRED — Federal Reserve Economic Data
# ==========================================================

_FRED_OBS = "https://api.stlouisfed.org/fred/series/observations"

# (series_id, display_unit, compute_yoy)
_FRED_SERIES = {
    "Fed Funds Rate": ("FEDFUNDS",          "%",  False),
    "CPI YoY":        ("CPIAUCSL",          "%",  True),
    "Unemployment":   ("UNRATE",            "%",  False),
    "10Y Treasury":   ("GS10",              "%",  False),
    "VIX":            ("VIXCLS",            "",   False),
}


def _fred_obs(series_id: str, api_key: str, limit: int = 14) -> list:
    data = _get(_FRED_OBS, params={
        "series_id":  series_id,
        "api_key":    api_key,
        "file_type":  "json",
        "limit":      limit,
        "sort_order": "desc",
    })
    if not data or "observations" not in data:
        return []
    out = []
    for o in data["observations"]:
        try:
            v = float(o["value"]) if o["value"] not in (".", "") else None
            if v is not None:
                out.append({"date": o["date"], "value": v})
        except Exception:
            continue
    return out


def fred_macro_snapshot(api_key: str) -> dict:
    """Latest value + change for each key macro indicator."""
    result = {}
    for name, (series_id, unit, yoy) in _FRED_SERIES.items():
        obs = _fred_obs(series_id, api_key, limit=14 if yoy else 3)
        if not obs:
            continue
        latest = obs[0]["value"]
        date   = obs[0]["date"]
        if yoy and len(obs) >= 13:
            prev_year = obs[12]["value"]
            delta     = round(((latest - prev_year) / prev_year) * 100, 2) if prev_year else None
            display   = f"{delta:+.2f}%" if delta is not None else f"{latest:.2f}"
        else:
            prev    = obs[1]["value"] if len(obs) > 1 else None
            delta   = round(latest - prev, 3) if prev is not None else None
            display = f"{latest:.2f}{unit}"
        result[name] = {
            "value":   latest,
            "display": display,
            "date":    date,
            "delta":   delta,
            "unit":    unit,
        }
    return result


def fred_series_history(series_id: str, api_key: str, periods: int = 60) -> pd.Series:
    """Historical observations as a date-indexed pd.Series."""
    obs = _fred_obs(series_id, api_key, limit=periods)
    if not obs:
        return pd.Series(dtype=float)
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


# ==========================================================
# INDIA MACRO — yfinance + FRED India series
# ==========================================================

def india_macro_snapshot(fred_api_key: str = "") -> dict:
    """
    Key macro indicators for Indian portfolios.
    Uses yfinance for real-time data + FRED for CPI history.
    Returns same format as fred_macro_snapshot().
    """
    import yfinance as yf

    result = {}

    # ── India 10Y Govt Bond Yield ────────────────────────
    try:
        _bond = yf.Ticker("IN10Y=X")
        _hi   = _bond.history(period="5d")
        if not _hi.empty:
            _val  = float(_hi["Close"].dropna().iloc[-1])
            _prev = float(_hi["Close"].dropna().iloc[-2]) if len(_hi) > 1 else None
            result["India 10Y Yield"] = {
                "value":   _val,
                "display": f"{_val:.2f}%",
                "date":    str(_hi.index[-1].date()),
                "delta":   round(_val - _prev, 3) if _prev is not None else None,
                "unit":    "%",
            }
    except Exception:
        pass

    # ── India VIX ────────────────────────────────────────
    try:
        _vix  = yf.Ticker("^INDIAVIX")
        _hv   = _vix.history(period="5d")
        if not _hv.empty:
            _val  = float(_hv["Close"].dropna().iloc[-1])
            _prev = float(_hv["Close"].dropna().iloc[-2]) if len(_hv) > 1 else None
            result["India VIX"] = {
                "value":   _val,
                "display": f"{_val:.2f}",
                "date":    str(_hv.index[-1].date()),
                "delta":   round(_val - _prev, 3) if _prev is not None else None,
                "unit":    "",
            }
    except Exception:
        pass

    # ── USD/INR Exchange Rate ─────────────────────────────
    try:
        _fx   = yf.Ticker("USDINR=X")
        _hf   = _fx.history(period="5d")
        if not _hf.empty:
            _val  = float(_hf["Close"].dropna().iloc[-1])
            _prev = float(_hf["Close"].dropna().iloc[-2]) if len(_hf) > 1 else None
            result["USD/INR"] = {
                "value":   _val,
                "display": f"₹{_val:.2f}",
                "date":    str(_hf.index[-1].date()),
                "delta":   round(_val - _prev, 3) if _prev is not None else None,
                "unit":    "",
            }
    except Exception:
        pass

    # ── India CPI YoY via FRED ───────────────────────────
    if fred_api_key:
        try:
            obs = _fred_obs("INDCPIALLMINMEI", fred_api_key, limit=14)
            if obs and len(obs) >= 13:
                latest    = obs[0]["value"]
                prev_year = obs[12]["value"]
                yoy       = round(((latest - prev_year) / prev_year) * 100, 2) if prev_year else None
                result["India CPI YoY"] = {
                    "value":   yoy,
                    "display": f"{yoy:+.2f}%" if yoy is not None else "—",
                    "date":    obs[0]["date"],
                    "delta":   None,
                    "unit":    "%",
                }
        except Exception:
            pass

    # ── Nifty 50 (market pulse) ───────────────────────────
    try:
        _nifty = yf.Ticker("^NSEI")
        _hn    = _nifty.history(period="5d")
        if not _hn.empty:
            _val  = float(_hn["Close"].dropna().iloc[-1])
            _prev = float(_hn["Close"].dropna().iloc[-2]) if len(_hn) > 1 else None
            _chg  = round((_val - _prev) / _prev * 100, 2) if _prev else None
            result["NIFTY 50"] = {
                "value":   _val,
                "display": f"{_val:,.0f}",
                "date":    str(_hn.index[-1].date()),
                "delta":   _chg,
                "unit":    "",
            }
    except Exception:
        pass

    return result


def yfinance_news(ticker: str, max_items: int = 12) -> pd.DataFrame:
    """
    Fetch recent news for a ticker via yfinance.
    Handles the current nested content structure (yfinance >= 0.2.x).
    """
    import yfinance as yf
    try:
        news = yf.Ticker(ticker).news or []
        rows = []
        for item in news[:max_items]:
            # New API: data lives under item['content']
            c = item.get("content") or item
            url = (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url", "")
            rows.append({
                "headline": c.get("title", ""),
                "source":   (c.get("provider") or {}).get("displayName", ""),
                "url":      url,
                "datetime": pd.to_datetime(c.get("pubDate") or c.get("displayTime"), utc=True, errors="coerce"),
                "summary":  c.get("summary", ""),
            })
        df = pd.DataFrame(rows)
        df = df[df["headline"] != ""]
        if not df.empty:
            df = df.sort_values("datetime", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


# ==========================================================
# FINANCIAL MODELING PREP
# ==========================================================

_FMP = "https://financialmodelingprep.com/api/v3"


def fmp_income_growth(ticker: str, api_key: str) -> pd.DataFrame:
    """Annual income statement with margins — last 4 years."""
    data = _get(f"{_FMP}/income-statement/{ticker}",
                params={"limit": 4, "apikey": api_key})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        rows.append({
            "Year":         str(d.get("calendarYear", d.get("date", "")))[:4],
            "Revenue":      d.get("revenue"),
            "Gross Profit": d.get("grossProfit"),
            "Op. Income":   d.get("operatingIncome"),
            "Net Income":   d.get("netIncome"),
            "EPS":          d.get("eps"),
            "Gross Margin": d.get("grossProfitRatio"),
            "Net Margin":   d.get("netIncomeRatio"),
            "Op. Margin":   d.get("operatingIncomeRatio"),
        })
    df = pd.DataFrame(rows)
    if "Revenue" in df.columns and len(df) > 1:
        df["Rev Growth"] = df["Revenue"].pct_change(-1)
    return df


def fmp_key_metrics(ticker: str, api_key: str) -> pd.DataFrame:
    """Annual valuation & profitability ratios — last 4 years."""
    data = _get(f"{_FMP}/key-metrics/{ticker}",
                params={"limit": 4, "apikey": api_key})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        rows.append({
            "Year":        str(d.get("date", ""))[:4],
            "P/E":         d.get("peRatio"),
            "P/B":         d.get("pbRatio"),
            "P/S":         d.get("priceToSalesRatio"),
            "EV/EBITDA":   d.get("enterpriseValueOverEBITDA"),
            "ROE":         d.get("roe"),
            "ROA":         d.get("returnOnTangibleAssets"),
            "Debt/Equity": d.get("debtToEquity"),
            "FCF/Share":   d.get("freeCashFlowPerShare"),
            "Div Yield":   d.get("dividendYield"),
        })
    return pd.DataFrame(rows)


def fmp_analyst_estimates(ticker: str, api_key: str) -> pd.DataFrame:
    """Forward analyst EPS and revenue estimates — next 4 periods."""
    data = _get(f"{_FMP}/analyst-estimates/{ticker}",
                params={"limit": 4, "apikey": api_key})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        rows.append({
            "Period":       d.get("date", "")[:7],
            "Est. Revenue": d.get("estimatedRevenueAvg"),
            "Est. EPS":     d.get("estimatedEpsAvg"),
            "EPS High":     d.get("estimatedEpsHigh"),
            "EPS Low":      d.get("estimatedEpsLow"),
            "# Analysts":   d.get("numberAnalystEstimatedEps"),
        })
    return pd.DataFrame(rows)


# ==========================================================
# SEC EDGAR — Insider Transactions (Form 4)
# ==========================================================

_SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUB     = "https://data.sec.gov/submissions/CIK{cik}.json"
_SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{doc}"

_cik_cache: dict = {}


def _get_cik(ticker: str):
    global _cik_cache
    if not _cik_cache:
        data = _get(_SEC_TICKERS, headers=_HEADERS)
        if data:
            for entry in data.values():
                t = entry.get("ticker", "").upper()
                c = int(entry.get("cik_str", 0))
                _cik_cache[t] = (str(c).zfill(10), c)
    clean = ticker.upper().replace(".NS", "").replace(".BO", "")
    return _cik_cache.get(clean, (None, None))


def sec_insider_transactions(ticker: str) -> pd.DataFrame:
    """
    Recent insider buy/sell transactions from SEC EDGAR Form 4.
    Returns: Date, Insider, Title, Type, Shares, Price, Value.
    """
    cik_str, cik_int = _get_cik(ticker)
    if cik_str is None:
        return pd.DataFrame()

    sub = _get(_SEC_SUB.format(cik=cik_str), headers=_HEADERS)
    if not sub or "filings" not in sub:
        return pd.DataFrame()

    recent   = sub["filings"].get("recent", {})
    forms    = recent.get("form",            [])
    dates    = recent.get("filingDate",      [])
    accs     = recent.get("accessionNumber", [])
    pri_docs = recent.get("primaryDocument", [])

    form4s = [
        (d, a, p)
        for f, d, a, p in zip(forms, dates, accs, pri_docs)
        if f == "4"
    ][:15]

    transactions = []
    for date, acc, primary_doc in form4s:
        acc_nodash = acc.replace("-", "")
        xml_url    = _SEC_ARCHIVE.format(
            cik_int=cik_int, acc_nodash=acc_nodash, doc=primary_doc,
        )
        try:
            r = requests.get(xml_url, headers=_HEADERS, timeout=_TIMEOUT)
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)

            def _txt(node, *paths):
                for path in paths:
                    el = node.find(path)
                    if el is not None and el.text:
                        return el.text.strip()
                return None

            owner = _txt(root,
                         "reportingOwner/reportingOwnerId/rptOwnerName") or "Unknown"
            title = _txt(root,
                         "reportingOwner/reportingOwnerRelationship/officerTitle",
                         "reportingOwner/reportingOwnerRelationship/isDirector") or "Insider"

            for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
                code     = _txt(txn, "transactionCoding/transactionCode")
                txn_date = _txt(txn, "transactionDate/value") or date
                shares_v = _txt(txn, "transactionAmounts/transactionShares/value")
                price_v  = _txt(txn, "transactionAmounts/transactionPricePerShare/value")

                if code not in ("P", "S"):
                    continue
                try:
                    shares = float(shares_v) if shares_v else None
                    price  = float(price_v)  if price_v  else None
                    if not shares:
                        continue
                    transactions.append({
                        "Date":    txn_date,
                        "Insider": owner,
                        "Title":   title,
                        "Type":    "Buy" if code == "P" else "Sell",
                        "Shares":  int(shares),
                        "Price":   price,
                        "Value":   round(shares * price) if price else None,
                    })
                except (ValueError, TypeError):
                    continue
        except Exception:
            continue

    return pd.DataFrame(transactions) if transactions else pd.DataFrame()


# ==========================================================
# WALLSTREETBETS SENTIMENT
# ==========================================================

_WSB_URL = "https://dashboard.nbshare.io/api/v1/apps/reddit/get_reddit_stocks"


def wsb_sentiment(tickers: list | None = None) -> pd.DataFrame:
    """
    WallStreetBets ticker mentions and sentiment from nbshare.io.
    Filters to given tickers if provided.
    """
    data = _get(_WSB_URL)
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    rows = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        ticker = item.get("ticker", "").upper()
        rows.append({
            "Ticker":    ticker,
            "Mentions":  item.get("no_of_comments", 0),
            "Sentiment": item.get("sentiment", "Neutral").capitalize(),
            "Upvotes":   item.get("upvotes", 0),
            "WSB Rank":  i + 1,
        })
    df = pd.DataFrame(rows)
    if tickers and not df.empty:
        clean = {t.upper().replace(".NS", "").replace(".BO", "") for t in tickers}
        df = df[df["Ticker"].isin(clean)].reset_index(drop=True)
    return df
