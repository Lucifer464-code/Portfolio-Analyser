import numpy as np
import pandas as pd
from typing import Dict
from config import TRADING_DAYS, BASE_RISK_FREE_RATE


# ==========================================================
# BASIC RISK METRICS
# ==========================================================

def annualized_volatility(returns: pd.Series) -> float:
    returns = returns.dropna()
    return returns.std(ddof=1) * np.sqrt(TRADING_DAYS)


def downside_deviation(returns: pd.Series,
                       target: float = 0.0) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0
    excess = returns - target
    semi_var = (np.minimum(excess, 0) ** 2).mean()
    return np.sqrt(semi_var) * np.sqrt(TRADING_DAYS)


def annualized_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if len(returns) == 0:
        return 0

    years = len(returns) / TRADING_DAYS
    compounded = (1 + returns).prod()

    if years == 0:
        return 0

    return compounded ** (1 / years) - 1


def sharpe_ratio(returns: pd.Series,
                 rf: float = BASE_RISK_FREE_RATE) -> float:
    excess = annualized_return(returns) - rf
    vol = annualized_volatility(returns)
    return excess / vol if vol != 0 else 0


def sortino_ratio(returns: pd.Series,
                  rf: float = BASE_RISK_FREE_RATE) -> float:
    excess = annualized_return(returns) - rf
    downside_vol = downside_deviation(returns)
    return excess / downside_vol if downside_vol != 0 else 0


# ==========================================================
# DRAWDOWN
# ==========================================================

def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    returns = returns.dropna()
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    return (cumulative - peak) / peak


def max_drawdown(returns: pd.Series) -> float:
    return compute_drawdown_series(returns).min()


# ==========================================================
# VALUE AT RISK
# ==========================================================

def historical_var(returns: pd.Series,
                   alpha: float = 0.05) -> float:
    returns = returns.dropna()
    return returns.quantile(alpha)


def conditional_var(returns: pd.Series,
                    alpha: float = 0.05) -> float:
    returns = returns.dropna()
    var = historical_var(returns, alpha)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else 0


# ==========================================================
# ACTIVE MANAGEMENT
# ==========================================================

def tracking_error(portfolio: pd.Series,
                   benchmark: pd.Series) -> float:

    aligned = pd.concat([portfolio.rename("Portfolio"),
                         benchmark.rename("Benchmark")], axis=1).dropna()

    active = aligned["Portfolio"] - aligned["Benchmark"]
    return active.std(ddof=1) * np.sqrt(TRADING_DAYS)


def information_ratio(portfolio: pd.Series,
                      benchmark: pd.Series) -> float:

    aligned = pd.concat([portfolio.rename("Portfolio"),
                         benchmark.rename("Benchmark")], axis=1).dropna()
    if len(aligned) == 0:
        return 0

    active_return = (
        annualized_return(aligned["Portfolio"])
        - annualized_return(aligned["Benchmark"])
    )

    te = tracking_error(
        aligned["Portfolio"],
        aligned["Benchmark"]
    )

    return active_return / te if te != 0 else 0


def beta(portfolio: pd.Series,
         benchmark: pd.Series) -> float:

    aligned = pd.concat([portfolio.rename("Portfolio"),
                         benchmark.rename("Benchmark")], axis=1).dropna()
    if len(aligned) == 0:
        return 0

    cov = np.cov(
        aligned["Portfolio"],
        aligned["Benchmark"],
        ddof=1
    )[0][1]

    var = np.var(
        aligned["Benchmark"],
        ddof=1
    )

    return cov / var if var != 0 else 0


def correlation(portfolio: pd.Series,
                benchmark: pd.Series) -> float:

    aligned = pd.concat([portfolio.rename("Portfolio"),
                         benchmark.rename("Benchmark")], axis=1).dropna()
    if len(aligned) == 0:
        return 0

    return aligned.corr().iloc[0, 1]


# ==========================================================
# ROLLING METRICS
# ==========================================================

def rolling_volatility(returns: pd.Series,
                       window: int = 60) -> pd.Series:
    return returns.dropna().rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_beta(portfolio: pd.Series,
                 benchmark: pd.Series,
                 window: int = 60) -> pd.Series:

    df = pd.concat([portfolio.rename("Portfolio"),
                    benchmark.rename("Benchmark")], axis=1).dropna()

    cov = df["Portfolio"].rolling(window).cov(df["Benchmark"])
    var = df["Benchmark"].rolling(window).var()

    return cov / var


def rolling_correlation(portfolio: pd.Series,
                        benchmark: pd.Series,
                        window: int = 60) -> pd.Series:

    df = pd.concat([portfolio.rename("Portfolio"),
                    benchmark.rename("Benchmark")], axis=1).dropna()

    return df["Portfolio"].rolling(window).corr(df["Benchmark"])



# ==========================================================
# PARAMETRIC VaR / CVaR
# ==========================================================

def parametric_var(returns: pd.Series, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    returns = returns.dropna()
    mu    = returns.mean()
    sigma = returns.std(ddof=1)
    return float(mu + norm.ppf(alpha) * sigma)


def parametric_cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    from scipy.stats import norm
    returns = returns.dropna()
    mu    = returns.mean()
    sigma = returns.std(ddof=1)
    z     = norm.ppf(alpha)
    return float(mu - sigma * norm.pdf(z) / alpha)


def var_cvar_summary(returns: pd.Series) -> Dict:
    returns = returns.dropna()
    return {
        "hist_var_95":   historical_var(returns, 0.05),
        "hist_cvar_95":  conditional_var(returns, 0.05),
        "hist_var_99":   historical_var(returns, 0.01),
        "hist_cvar_99":  conditional_var(returns, 0.01),
        "param_var_95":  parametric_var(returns, 0.05),
        "param_cvar_95": parametric_cvar(returns, 0.05),
        "param_var_99":  parametric_var(returns, 0.01),
        "param_cvar_99": parametric_cvar(returns, 0.01),
    }


# ==========================================================
# CONCENTRATION ANALYTICS
# ==========================================================

def sector_concentration(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("Sector")
        .agg(Weight=("Current Weight", "sum"), Holdings=("Ticker", "count"))
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    grouped["HHI Contribution"] = grouped["Weight"] ** 2
    total_hhi = grouped["HHI Contribution"].sum()
    grouped["% of HHI"] = grouped["HHI Contribution"] / total_hhi if total_hhi > 0 else 0
    return grouped


def asset_type_concentration(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("Asset Type")
        .agg(Weight=("Current Weight", "sum"), Holdings=("Ticker", "count"))
        .reset_index()
        .sort_values("Weight", ascending=False)
    )
    return grouped


def effective_n(weights: pd.Series) -> float:
    hhi = float((weights ** 2).sum())
    return 1.0 / hhi if hhi > 0 else float(len(weights))


# ==========================================================
# RISK SUMMARY
# ==========================================================

def generate_risk_summary(portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series) -> Dict:

    summary = {
        "Annual Return": annualized_return(portfolio_returns),
        "Volatility": annualized_volatility(portfolio_returns),
        "Sharpe Ratio": sharpe_ratio(portfolio_returns),
        "Sortino Ratio": sortino_ratio(portfolio_returns),
        "Max Drawdown": max_drawdown(portfolio_returns),
        "VaR 95%": historical_var(portfolio_returns),
        "CVaR 95%": conditional_var(portfolio_returns),
        "Tracking Error": tracking_error(portfolio_returns,
                                          benchmark_returns)
                          if benchmark_returns is not None else 0,
        "Information Ratio": information_ratio(portfolio_returns,
                                                benchmark_returns)
                             if benchmark_returns is not None else 0,
        "Beta": beta(portfolio_returns,
                     benchmark_returns)
                if benchmark_returns is not None else 0,
        "Correlation": correlation(portfolio_returns,
                                   benchmark_returns)
                       if benchmark_returns is not None else 0
    }

    return summary


# ==========================================================
# LIQUIDITY RISK
# ==========================================================

def compute_liquidity_risk(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute days-to-liquidate for each holding based on 3-month ADV.
    holdings_df must have columns: Ticker, Quantity, Market Value.
    Returns a DataFrame with liquidity metrics per ticker.
    """
    import yfinance as yf

    rows = []
    for _, row in holdings_df.iterrows():
        ticker = row["Ticker"]
        quantity = float(row["Quantity"])
        market_value = float(row["Market Value"])
        try:
            vol_data = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
            if vol_data.empty or "Volume" not in vol_data.columns:
                raise ValueError("No volume data")
            adv = float(vol_data["Volume"].mean())
            days_to_liquidate = quantity / adv if adv > 0 else np.nan
        except Exception:
            adv = np.nan
            days_to_liquidate = np.nan

        if np.isnan(days_to_liquidate):
            flag = "N/A"
        elif days_to_liquidate < 1:
            flag = "🟢 High"
        elif days_to_liquidate <= 5:
            flag = "🟡 Medium"
        else:
            flag = "🔴 Low"

        rows.append({
            "Ticker": ticker,
            "Market Value": market_value,
            "Avg Daily Volume (3M)": adv,
            "Days to Liquidate": days_to_liquidate,
            "Liquidity": flag,
        })

    return pd.DataFrame(rows)