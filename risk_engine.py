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
    downside = returns[returns < target]
    if len(downside) == 0:
        return 0
    return downside.std(ddof=1) * np.sqrt(TRADING_DAYS)


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

    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    aligned.columns = ["Portfolio", "Benchmark"]

    active = aligned["Portfolio"] - aligned["Benchmark"]
    return active.std(ddof=1) * np.sqrt(TRADING_DAYS)


def information_ratio(portfolio: pd.Series,
                      benchmark: pd.Series) -> float:

    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
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

    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
    if len(aligned) == 0:
        return 0

    aligned.columns = ["Portfolio", "Benchmark"]

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

    aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
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

    df = pd.concat([portfolio, benchmark], axis=1).dropna()
    df.columns = ["Portfolio", "Benchmark"]

    cov = df["Portfolio"].rolling(window).cov(df["Benchmark"])
    var = df["Benchmark"].rolling(window).var()

    return cov / var


def rolling_correlation(portfolio: pd.Series,
                        benchmark: pd.Series,
                        window: int = 60) -> pd.Series:

    df = pd.concat([portfolio, benchmark], axis=1).dropna()
    df.columns = ["Portfolio", "Benchmark"]

    return df["Portfolio"].rolling(window).corr(df["Benchmark"])


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
