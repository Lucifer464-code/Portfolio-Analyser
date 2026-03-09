import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional

from config import (
    TRADING_DAYS,
    BASE_RISK_FREE_RATE,
    REGULARIZATION_STRENGTH,
    MONTE_CARLO_PORTFOLIOS,
    RISK_PROFILES,
)

# ==========================================================
# PORTFOLIO PERFORMANCE
# ==========================================================

def annualized_returns(returns: pd.DataFrame) -> pd.Series:
    compounded = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    return compounded ** (1 / years) - 1


def portfolio_performance(
    weights: np.ndarray,
    returns: pd.DataFrame,
    rf_multiplier: float,
) -> Tuple[float, float, float]:
    mean_returns = annualized_returns(returns)
    cov_matrix   = returns.cov() * TRADING_DAYS
    port_return  = float(np.dot(mean_returns, weights))
    port_vol     = float(np.sqrt(weights @ cov_matrix @ weights))
    sharpe       = (port_return - BASE_RISK_FREE_RATE * rf_multiplier) / port_vol if port_vol else 0
    return port_return, port_vol, sharpe


# ==========================================================
# SHARED UTILITIES
# ==========================================================

def _base_setup(returns: pd.DataFrame, risk_profile: str, max_weight: Optional[float]):
    """Return common inputs shared by every optimizer."""
    settings    = RISK_PROFILES[risk_profile]
    max_w       = max_weight if max_weight is not None else settings["max_weight"]
    min_w       = settings["min_weight"]
    rf_mult     = settings["rf_multiplier"]
    n           = len(returns.columns)
    bounds      = tuple((min_w, max_w) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
    x0          = np.full(n, 1.0 / n)
    cov         = returns.cov() * TRADING_DAYS
    mu          = annualized_returns(returns)
    return n, bounds, constraints, x0, cov, mu, rf_mult


def _minimize(objective, x0, bounds, constraints) -> np.ndarray:
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")
    return result.x


def _port_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


# ==========================================================
# OPTIMIZATION METHODS
# ==========================================================

def _max_sharpe(returns, risk_profile, max_weight, **_) -> pd.Series:
    """Maximise the Sharpe ratio (original method)."""
    n, bounds, constraints, x0, cov, mu, rf_mult = _base_setup(returns, risk_profile, max_weight)

    def objective(w):
        ret    = float(np.dot(mu, w))
        vol    = _port_vol(w, cov)
        sharpe = (ret - BASE_RISK_FREE_RATE * rf_mult) / vol if vol else 0
        return -(sharpe - REGULARIZATION_STRENGTH * np.sum(w ** 2))

    return pd.Series(_minimize(objective, x0, bounds, constraints), index=returns.columns)


def _min_variance(returns, risk_profile, max_weight, **_) -> pd.Series:
    """Minimise portfolio variance — ignores expected returns entirely."""
    _, bounds, constraints, x0, cov, *_ = _base_setup(returns, risk_profile, max_weight)

    return pd.Series(
        _minimize(lambda w: w @ cov @ w, x0, bounds, constraints),
        index=returns.columns,
    )


def _risk_parity(returns, risk_profile, max_weight, **_) -> pd.Series:
    """
    Equal Risk Contribution: each asset contributes the same fraction
    of total portfolio volatility. Uses the squared-deviation formulation
    which is smooth and well-behaved for SLSQP.
    """
    n, bounds, constraints, x0, cov, *_ = _base_setup(returns, risk_profile, max_weight)
    target = np.full(n, 1.0 / n)

    def objective(w):
        vol = _port_vol(w, cov)
        if vol == 0:
            return 0.0
        rc_pct = (w * (cov @ w) / vol) / vol   # fractional risk contributions
        return float(np.sum((rc_pct - target) ** 2))

    # Keep weights strictly positive so marginal contributions stay defined
    rp_bounds = tuple((max(b[0], 1e-4), b[1]) for b in bounds)
    return pd.Series(_minimize(objective, x0, rp_bounds, constraints), index=returns.columns)


def _max_diversification(returns, risk_profile, max_weight, **_) -> pd.Series:
    """
    Maximise the Diversification Ratio:
        DR = (w · σ_individual) / σ_portfolio
    A DR > 1 means the portfolio is less volatile than its weighted-average
    constituents — i.e. diversification is doing real work.
    """
    _, bounds, constraints, x0, cov, *_ = _base_setup(returns, risk_profile, max_weight)
    asset_vols = np.sqrt(np.diag(cov))

    def objective(w):
        port_vol = _port_vol(w, cov)
        if port_vol == 0:
            return 0.0
        return -(np.dot(w, asset_vols) / port_vol)  # negate to maximise

    return pd.Series(_minimize(objective, x0, bounds, constraints), index=returns.columns)


def _equal_weight(returns, risk_profile, max_weight, **_) -> pd.Series:
    """1/N equal-weight baseline — no optimisation required."""
    n = len(returns.columns)
    return pd.Series(np.full(n, 1.0 / n), index=returns.columns)



# ==========================================================
# OPTIMIZER REGISTRY
# ==========================================================

OPTIMIZERS: dict = {
    "Max Sharpe":          _max_sharpe,
    "Min Variance":        _min_variance,
    "Risk Parity":         _risk_parity,
    "Max Diversification": _max_diversification,
    "Equal Weight":        _equal_weight,
}

OPTIMIZER_DESCRIPTIONS: dict = {
    "Max Sharpe":
        "Maximises return per unit of risk. Best when you have confidence in return estimates.",
    "Min Variance":
        "Minimises portfolio volatility regardless of returns. Defensive in uncertain markets.",
    "Risk Parity":
        "Each asset contributes equally to total risk. Natural diversifier across asset classes.",
    "Max Diversification":
        "Maximises the ratio of weighted-average asset vol to portfolio vol. Reduces concentration.",
    "Equal Weight":
        "Simple 1/N baseline. No assumptions required — a strong benchmark to beat.",
}


def optimize_portfolio(
    returns: pd.DataFrame,
    risk_profile: str,
    method: str = "Max Sharpe",
    max_weight: Optional[float] = None,
) -> pd.Series:
    """
    Unified entry point for all optimization methods.

    Parameters
    ----------
    returns      : Daily asset returns DataFrame.
    risk_profile : Key into RISK_PROFILES config dict.
    method       : One of OPTIMIZERS keys. Defaults to "Max Sharpe".
    max_weight   : Hard upper bound on any single asset weight (0–1).
    """
    if method not in OPTIMIZERS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(OPTIMIZERS.keys())}")

    return OPTIMIZERS[method](returns, risk_profile, max_weight)


# ==========================================================
# RISK CONTRIBUTION
# ==========================================================

def risk_contribution(weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
    cov       = returns.cov() * TRADING_DAYS
    port_vol  = _port_vol(weights, cov)
    if port_vol == 0:
        return pd.Series(weights, index=returns.columns)
    rc        = weights * (cov @ weights) / port_vol
    rc_series = pd.Series(rc, index=returns.columns)
    total     = rc_series.sum()
    return rc_series / total if total else pd.Series(weights, index=returns.columns)


# ==========================================================
# MONTE CARLO EFFICIENT FRONTIER
# ==========================================================

def simulate_efficient_frontier(returns: pd.DataFrame, risk_profile: str) -> pd.DataFrame:
    settings     = RISK_PROFILES[risk_profile]
    rf_mult      = settings["rf_multiplier"]
    n            = len(returns.columns)
    mean_returns = annualized_returns(returns)
    cov          = returns.cov() * TRADING_DAYS
    results      = []

    for _ in range(MONTE_CARLO_PORTFOLIOS):
        w      = np.random.dirichlet(np.ones(n))   # uniform on simplex
        ret    = float(np.dot(mean_returns, w))
        vol    = _port_vol(w, cov)
        sharpe = (ret - BASE_RISK_FREE_RATE * rf_mult) / vol if vol else 0
        results.append([ret, vol, sharpe])

    return pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe"])