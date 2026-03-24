import numpy as np
import pandas as pd
from typing import Dict, Optional
import yfinance as yf
from config import TRADING_DAYS, BASE_RISK_FREE_RATE


# ==========================================================
# ASSET KEY STATISTICS
# ==========================================================

def get_asset_key_stats(asset_ticker, asset_price, asset_returns, asset_weight):
    asset_vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
    asset_ret = asset_returns.mean() * TRADING_DAYS
    sharpe    = ((asset_ret - BASE_RISK_FREE_RATE) / asset_vol) if asset_vol > 0 else 0.0
    return {"ticker": asset_ticker, "weight": asset_weight,
            "annual_return": asset_ret, "volatility": asset_vol, "sharpe_ratio": sharpe}


def compute_rolling_volatility(returns, window=60):
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_rolling_correlation(asset_returns, portfolio_returns, window=60):
    return asset_returns.rolling(window).corr(portfolio_returns)


def compute_asset_drawdown(asset_returns):
    cumulative  = (1 + asset_returns).cumprod()
    running_max = cumulative.cummax()
    return (cumulative / running_max) - 1


def compute_asset_beta(asset_returns, portfolio_returns):
    aligned = pd.DataFrame({"asset": asset_returns, "portfolio": portfolio_returns}).dropna()
    if len(aligned) < 2: return 0
    cov  = aligned.cov().loc["asset", "portfolio"]
    pvar = aligned["portfolio"].var()
    return cov / pvar if pvar != 0 else 0


# ==========================================================
# FUNDAMENTAL DATA — TTM HELPERS
# ==========================================================

def _fetch_statements(ticker):
    try:
        stk   = yf.Ticker(ticker)
        fin_a = stk.financials
        fin_q = stk.quarterly_financials
        bs_a  = stk.balance_sheet
        bs_q  = stk.quarterly_balance_sheet

        def _norm(df):
            if df is None or df.empty: return df
            df.index = df.index.str.strip().str.lower().str.replace(" ","").str.replace("-","")
            return df

        fin_a, fin_q, bs_a, bs_q = _norm(fin_a), _norm(fin_q), _norm(bs_a), _norm(bs_q)
        fy_label  = f"FY {fin_a.columns[0].strftime('%b %Y')}"    if fin_a  is not None and not fin_a.empty  else "Annual"
        ttm_label = f"TTM ({fin_q.columns[0].strftime('%b %Y')})" if fin_q is not None and not fin_q.empty else "TTM"
        return dict(fin_a=fin_a, fin_q=fin_q, bs_a=bs_a, bs_q=bs_q,
                    fy_label=fy_label, ttm_label=ttm_label)
    except Exception:
        return None


def _get_col(df, key, col_idx):
    if df is None or df.empty: return None
    matches = [k for k in df.index if key in k]
    if not matches or col_idx >= len(df.columns): return None
    try:
        val = df.loc[matches[0], df.columns[col_idx]]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def _ttm(fin_a, fin_q, key):
    if fin_q is not None and not fin_q.empty and len(fin_q.columns) >= 4:
        a, q0, q3 = _get_col(fin_a, key, 0), _get_col(fin_q, key, 0), _get_col(fin_q, key, 3)
        if all(v is not None for v in (a, q0, q3)):
            return a + q0 - q3
    return _get_col(fin_a, key, 0)


def _ratio(num, den):
    if num is None or den is None or den == 0: return None
    return num / den


def _fmt(val, pct=False, mult=False):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    if pct:  return f"{val:.2%}"
    if mult: return f"{val:.2f}x"
    return f"{val:.2f}"


def _build_ratios(fin_a, fin_q, bs, info=None):
    rev   = _ttm(fin_a, fin_q, "totalrevenue")      if fin_q is not None else _get_col(fin_a, "totalrevenue", 0)
    gp    = _ttm(fin_a, fin_q, "grossprofit")        if fin_q is not None else _get_col(fin_a, "grossprofit", 0)
    op    = _ttm(fin_a, fin_q, "operatingincome")    if fin_q is not None else _get_col(fin_a, "operatingincome", 0)
    ni    = _ttm(fin_a, fin_q, "netincome")          if fin_q is not None else _get_col(fin_a, "netincome", 0)
    ebit  = _ttm(fin_a, fin_q, "ebit")               if fin_q is not None else _get_col(fin_a, "ebit", 0)
    da    = _ttm(fin_a, fin_q, "depreciationamortization") if fin_q is not None else _get_col(fin_a, "depreciationamortization", 0)
    int_  = _ttm(fin_a, fin_q, "interestexpense")    if fin_q is not None else _get_col(fin_a, "interestexpense", 0)

    ta    = _get_col(bs, "totalassets",            0)
    eq    = _get_col(bs, "stockholdersequity",     0) or _get_col(bs, "totalstockholdersequity", 0)
    cl    = _get_col(bs, "currentliabilities",     0)
    ca    = _get_col(bs, "currentassets",          0)
    inv   = _get_col(bs, "inventory",              0)
    debt  = _get_col(bs, "totaldebt",              0) or _get_col(bs, "longtermdebt", 0)
    cash  = _get_col(bs, "cashandcashequivalents", 0) or _get_col(bs, "cash",         0)

    ebitda = (ebit + da) if (ebit and da) else _ttm(fin_a, fin_q, "ebitda") if fin_q is not None else _get_col(fin_a, "ebitda", 0)

    price  = info.get("currentPrice") or info.get("regularMarketPrice") if info else None
    mktcap = info.get("marketCap")    if info else None
    shares = info.get("sharesOutstanding") if info else None

    pe     = _ratio(price,  ni / shares if (ni and shares and shares > 0) else None)
    pb     = _ratio(mktcap, eq)   if eq    else None
    ps     = _ratio(mktcap, rev)  if rev   else None
    ev     = (mktcap + (debt or 0) - (cash or 0)) if mktcap else None
    ev_ebi = _ratio(ev, ebitda)   if ebitda else None
    ev_rev = _ratio(ev, rev)      if rev    else None

    return dict(
        gross_margin  = _ratio(gp, rev),
        op_margin     = _ratio(op, rev),
        net_margin    = _ratio(ni, rev),
        roe           = _ratio(ni, eq),
        roa           = _ratio(ni, ta),
        current_ratio = _ratio(ca, cl),
        quick_ratio   = _ratio((ca - inv) if (ca and inv) else ca, cl),
        de_ratio      = _ratio(debt, eq),
        int_coverage  = _ratio(ebit, abs(int_)) if int_ else None,
        pe=pe, pb=pb, ps=ps, ev_ebitda=ev_ebi, ev_rev=ev_rev,
    )


# ==========================================================
# COMPREHENSIVE FUNDAMENTAL TABLE  (Annual + TTM dual columns)
# ==========================================================

def get_asset_fundamental_table(ticker: str) -> pd.DataFrame:
    stmts = _fetch_statements(ticker)
    if stmts is None:
        return pd.DataFrame({"Metric": ["No data available"], "Value": ["N/A"]})

    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}

    fin_a, fin_q = stmts["fin_a"], stmts["fin_q"]
    bs_a,  bs_q  = stmts["bs_a"],  stmts["bs_q"]
    bs_latest    = bs_q if (bs_q is not None and not bs_q.empty) else bs_a

    ann = _build_ratios(fin_a, None,  bs_a,        info)
    ttm = _build_ratios(fin_a, fin_q, bs_latest,   info)

    fy_col  = stmts["fy_label"]
    ttm_col = stmts["ttm_label"]

    rows = [
        ("Gross Margin",      "Profitability", ann["gross_margin"],  ttm["gross_margin"],  True,  False),
        ("Operating Margin",  "Profitability", ann["op_margin"],     ttm["op_margin"],     True,  False),
        ("Net Profit Margin", "Profitability", ann["net_margin"],    ttm["net_margin"],    True,  False),
        ("ROE",               "Profitability", ann["roe"],           ttm["roe"],           True,  False),
        ("ROA",               "Profitability", ann["roa"],           ttm["roa"],           True,  False),
        ("Current Ratio",     "Liquidity",     ann["current_ratio"], ttm["current_ratio"], False, False),
        ("Quick Ratio",       "Liquidity",     ann["quick_ratio"],   ttm["quick_ratio"],   False, False),
        ("Debt-to-Equity",    "Liquidity",     ann["de_ratio"],      ttm["de_ratio"],      False, False),
        ("Interest Coverage", "Liquidity",     ann["int_coverage"],  ttm["int_coverage"],  False, True),
        ("P/E",               "Valuation",     ann["pe"],            ttm["pe"],            False, True),
        ("P/B",               "Valuation",     ann["pb"],            ttm["pb"],            False, True),
        ("P/S",               "Valuation",     ann["ps"],            ttm["ps"],            False, True),
        ("EV/EBITDA",         "Valuation",     ann["ev_ebitda"],     ttm["ev_ebitda"],     False, True),
        ("EV/Revenue",        "Valuation",     ann["ev_rev"],        ttm["ev_rev"],        False, True),
    ]

    data = []
    for metric, category, a_val, t_val, pct, mult in rows:
        data.append({"Metric": metric, fy_col: _fmt(a_val, pct=pct, mult=mult),
                     ttm_col: _fmt(t_val, pct=pct, mult=mult), "Category": category})

    result = pd.DataFrame(data)
    result["Metric"] = result["Metric"].astype(str)
    return result


# ==========================================================
# DIVIDEND TRACKING
# ==========================================================

def get_dividend_data(ticker: str, quantity: float, current_price: float) -> dict:
    """
    Fetch dividend history for a ticker and compute yield and annual income.
    Returns a dict with keys: has_dividends, yield, annual_income, history (pd.Series).
    """
    try:
        dividends = yf.Ticker(ticker).dividends
        if dividends is None or dividends.empty:
            return {"has_dividends": False, "yield": 0.0, "annual_income": 0.0, "history": pd.Series(dtype=float)}

        # Last 12 months for yield calculation (use UTC to handle tz-aware indices)
        one_year_ago = pd.Timestamp.today(tz='UTC') - pd.DateOffset(years=1)
        last_12m = dividends[dividends.index >= one_year_ago]
        annual_divs = float(last_12m.sum())

        # Last 5 years for chart
        five_years_ago = pd.Timestamp.today(tz='UTC') - pd.DateOffset(years=5)
        history = dividends[dividends.index >= five_years_ago]

        if annual_divs == 0 or current_price <= 0:
            return {"has_dividends": False, "yield": 0.0, "annual_income": 0.0, "history": history}

        div_yield = annual_divs / current_price
        annual_income = div_yield * current_price * quantity

        return {
            "has_dividends": True,
            "yield":         div_yield,
            "annual_income": annual_income,
            "history":       history,
        }
    except Exception:
        return {"has_dividends": False, "yield": 0.0, "annual_income": 0.0, "history": pd.Series(dtype=float)}