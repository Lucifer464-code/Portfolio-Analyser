import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import yfinance as yf
from config import TRADING_DAYS


# ==========================================================
# ASSET KEY STATISTICS
# ==========================================================

def get_asset_key_stats(
    asset_ticker: str,
    asset_price: pd.Series,
    asset_returns: pd.Series,
    asset_weight: float,
) -> Dict:
    """
    Compute key statistics for a single asset.
    
    Args:
        asset_ticker: Ticker symbol
        asset_price: Price series
        asset_returns: Returns series
        asset_weight: Portfolio weight allocation
        
    Returns:
        Dictionary with weight, annual return, volatility, and Sharpe ratio
    """
    asset_vol = asset_returns.std() * np.sqrt(TRADING_DAYS)
    asset_ret = asset_returns.mean() * TRADING_DAYS
    sharpe = (asset_ret / asset_vol) if asset_vol > 0 else 0.0
    
    return {
        "ticker": asset_ticker,
        "weight": asset_weight,
        "annual_return": asset_ret,
        "volatility": asset_vol,
        "sharpe_ratio": sharpe,
    }


# ==========================================================
# ROLLING VOLATILITY
# ==========================================================

def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling volatility (annualized).
    
    Args:
        returns: Return series
        window: Rolling window size (default 60 days)
        
    Returns:
        Series of rolling volatilities
    """
    return returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


# ==========================================================
# ROLLING CORRELATION WITH PORTFOLIO
# ==========================================================

def compute_rolling_correlation(
    asset_returns: pd.Series,
    portfolio_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Compute rolling correlation between asset and portfolio.
    
    Args:
        asset_returns: Asset return series
        portfolio_returns: Portfolio return series
        window: Rolling window size (default 60 days)
        
    Returns:
        Series of rolling correlations
    """
    return asset_returns.rolling(window).corr(portfolio_returns)


# ==========================================================
# DRAWDOWN SERIES
# ==========================================================

def compute_asset_drawdown(asset_returns: pd.Series) -> pd.Series:
    """
    Compute drawdown series for an asset.
    
    Args:
        asset_returns: Return series
        
    Returns:
        Series of drawdowns (negative values)
    """
    cumulative = (1 + asset_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    return drawdown


# ==========================================================
# ASSET PERFORMANCE METRICS
# ==========================================================

def get_asset_performance_metrics(
    asset_returns: pd.Series,
) -> Dict:
    """
    Compute comprehensive performance metrics for an asset.
    
    Args:
        asset_returns: Return series
        
    Returns:
        Dictionary with performance metrics
    """
    cumulative = (1 + asset_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max) - 1
    
    max_drawdown = drawdown.min()
    
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    avg_daily = asset_returns.mean()
    daily_vol = asset_returns.std()
    
    return {
        "total_return": total_return,
        "average_daily_return": avg_daily,
        "daily_volatility": daily_vol,
        "max_drawdown": max_drawdown,
        "sharpe_daily": (avg_daily / daily_vol) if daily_vol > 0 else 0,
    }


# ==========================================================
# BETA CALCULATION
# ==========================================================

def compute_asset_beta(
    asset_returns: pd.Series,
    portfolio_returns: pd.Series,
) -> float:
    """
    Compute beta of asset relative to portfolio.
    
    Args:
        asset_returns: Asset return series
        portfolio_returns: Portfolio return series
        
    Returns:
        Beta coefficient
    """
    # Align series
    aligned = pd.DataFrame({
        "asset": asset_returns,
        "portfolio": portfolio_returns
    }).dropna()
    
    if len(aligned) < 2:
        return 0
    
    covariance = aligned.cov().loc["asset", "portfolio"]
    portfolio_variance = aligned["portfolio"].var()
    
    if portfolio_variance == 0:
        return 0
    
    return covariance / portfolio_variance


# ==========================================================
# FUNDAMENTAL DATA FETCHING
# ==========================================================

def fetch_fundamental_data(ticker: str) -> Dict:
    """
    Fetch fundamental financial data from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with fundamental metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        return {}


# ==========================================================
# PROFITABILITY RATIOS
# ==========================================================

def get_profitability_ratios(ticker: str) -> Dict:
    """
    Extract profitability metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with profitability ratios
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            "roe": info.get("returnOnEquity", None),
            "roa": info.get("returnOnAssets", None),
            "profit_margin": info.get("profitMargins", None),
            "gross_margin": info.get("grossMargins", None),
            "operating_margin": info.get("operatingMargins", None),
        }
        
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}


# ==========================================================
# LIQUIDITY RATIOS
# ==========================================================

def get_liquidity_ratios(ticker: str) -> Dict:
    """
    Extract liquidity and solvency metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with liquidity ratios
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            "current_ratio": info.get("currentRatio", None),
            "quick_ratio": info.get("quickRatio", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "total_debt_to_equity": info.get("totalDebt", None),
        }
        
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}


# ==========================================================
# VALUATION RATIOS
# ==========================================================

def get_valuation_ratios(ticker: str) -> Dict:
    """
    Extract valuation metrics for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with valuation ratios
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        ratios = {
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "pb_ratio": info.get("priceToBook", None),
            "ps_ratio": info.get("priceToSalesTrailing12Months", None),
            "peg_ratio": info.get("pegRatio", None),
            "ev_to_revenue": info.get("enterpriseToRevenue", None),
            "ev_to_ebitda": info.get("enterpriseToEbitda", None),
        }
        
        return {k: v for k, v in ratios.items() if v is not None}
    except Exception:
        return {}


# ==========================================================
# COMPREHENSIVE FUNDAMENTAL ANALYSIS TABLE
# ==========================================================

def get_asset_fundamental_table(ticker: str) -> pd.DataFrame:
    """
    Compile profitability, liquidity, and valuation ratios into a table.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with formatted ratios
    """
    prof = get_profitability_ratios(ticker)
    liq = get_liquidity_ratios(ticker)
    val = get_valuation_ratios(ticker)
    
    # Combine all ratios
    all_ratios = {**prof, **liq, **val}
    
    if not all_ratios:
        empty_df = pd.DataFrame({"Metric": ["No data available"], "Value": ["N/A"]})
        empty_df["Metric"] = empty_df["Metric"].astype(str)
        return empty_df
    
    # Create readable names
    metric_names = {
        "roe": "Return on Equity (ROE)",
        "roa": "Return on Assets (ROA)",
        "profit_margin": "Profit Margin",
        "gross_margin": "Gross Margin",
        "operating_margin": "Operating Margin",
        "current_ratio": "Current Ratio",
        "quick_ratio": "Quick Ratio",
        "debt_to_equity": "Debt-to-Equity",
        "total_debt_to_equity": "Total Debt-to-Equity",
        "pe_ratio": "P/E Ratio (Trailing)",
        "forward_pe": "P/E Ratio (Forward)",
        "pb_ratio": "Price-to-Book",
        "ps_ratio": "Price-to-Sales",
        "peg_ratio": "PEG Ratio",
        "ev_to_revenue": "EV/Revenue",
        "ev_to_ebitda": "EV/EBITDA",
    }
    
    data = []
    for key, value in all_ratios.items():
        metric_name = metric_names.get(key, key.replace("_", " ").title())
        
        # Format values appropriately
        if value is None or (isinstance(value, float) and np.isnan(value)):
            formatted = "N/A"
        elif key in ["roe", "roa", "profit_margin", "gross_margin", "operating_margin"]:
            formatted = f"{value:.2%}"
        elif key in ["pe_ratio", "forward_pe", "pb_ratio", "ps_ratio", "peg_ratio", "ev_to_revenue", "ev_to_ebitda"]:
            formatted = f"{value:.2f}x"
        else:
            formatted = f"{value:.2f}"
        
        data.append({"Metric": metric_name, "Value": formatted})
    
    df = pd.DataFrame(data)
    df["Metric"] = df["Metric"].astype(str)
    
    # Organize by category
    profitability = df[df["Metric"].str.contains("Margin|ROE|ROA", na=False)]
    liquidity = df[df["Metric"].str.contains("Ratio|Debt", na=False)]
    valuation = df[~df["Metric"].str.contains("Margin|ROE|ROA|Ratio|Debt", na=False)]
    
    result = pd.concat([profitability, liquidity, valuation], ignore_index=True)
    result["Metric"] = result["Metric"].astype(str)
    
    return result if not result.empty else pd.DataFrame({"Metric": ["No data available"], "Value": ["N/A"]})
