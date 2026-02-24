import numpy as np
import pandas as pd
from typing import Dict

from config import TRADING_DAYS


# ==========================================================
# DIVERSIFICATION (HERFINDAHL INDEX)
# ==========================================================

def herfindahl_index(weights: pd.Series) -> float:
    return np.sum(weights ** 2)


def diversification_score(weights: pd.Series) -> float:
    """
    Converts HHI into a 0–100 diversification score.
    Lower HHI = more diversified.
    """
    hhi = herfindahl_index(weights)
    score = (1 - hhi) * 100
    return max(min(score, 100), 0)


# ==========================================================
# CONCENTRATION RISK
# ==========================================================

def concentration_metrics(weights: pd.Series) -> Dict:

    top_1 = weights.sort_values(ascending=False).iloc[0]
    top_3 = weights.sort_values(ascending=False).iloc[:3].sum()

    return {
        "Top Position Weight": top_1,
        "Top 3 Concentration": top_3
    }


# ==========================================================
# CURRENCY EXPOSURE
# ==========================================================

def currency_exposure(df: pd.DataFrame) -> pd.Series:

    exposure = (
        df.groupby("Currency")["Market Value"]
        .sum()
        / df["Market Value"].sum()
    )

    return exposure


# ==========================================================
# VOLATILITY REGIME DETECTION
# ==========================================================

def volatility_regime(returns: pd.Series,
                      window: int = 60) -> str:

    rolling_vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS)

    current_vol = rolling_vol.iloc[-1]
    historical_vol = rolling_vol.dropna()

    percentile = (
        (historical_vol < current_vol).sum()
        / len(historical_vol)
    )

    if percentile < 0.33:
        return "Low Volatility Regime"
    elif percentile < 0.66:
        return "Normal Volatility Regime"
    else:
        return "High Volatility Regime"


# ==========================================================
# PORTFOLIO HEALTH SCORE
# ==========================================================

def portfolio_health_score(
    weights: pd.Series,
    sharpe: float,
    max_drawdown: float
) -> float:
    """
    Composite score (0–100) based on:
    - Diversification
    - Sharpe ratio
    - Drawdown
    """

    div_score = diversification_score(weights)

    sharpe_score = min(max((sharpe / 2) * 100, 0), 100)

    drawdown_penalty = abs(max_drawdown) * 100
    drawdown_score = max(100 - drawdown_penalty, 0)

    total_score = (0.4 * div_score +
                   0.3 * sharpe_score +
                   0.3 * drawdown_score)

    return round(total_score, 2)
