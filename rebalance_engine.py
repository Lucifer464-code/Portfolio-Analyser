# ==========================================================
# REBALANCE ENGINE
# - Mean-Variance Allocation Comparison
# - 3M Relative Performance Rule Engine
# ==========================================================

import pandas as pd
import numpy as np


# ==========================================================
# MAIN FUNCTION
# ==========================================================

def generate_rebalance_tables(
    holdings_df: pd.DataFrame,
    optimal_weights: pd.Series,
    weights_series: pd.Series,
    portfolio_metrics_df: pd.DataFrame,
    transaction_cost: float = 0.0,
    portfolio_value: float = 0.0,
):
    """
    Generates:
    1) MVO rebalance table (current vs optimized weights)
    2) 3M relative performance rule table
    3) Portfolio turnover and transaction costs
    
    Parameters
    ----------
    holdings_df : pd.DataFrame
        Holdings data
    optimal_weights : pd.Series
        Optimal weights from optimization
    weights_series : pd.Series
        Current portfolio weights
    portfolio_metrics_df : pd.DataFrame
        3M performance metrics
    transaction_cost : float
        Transaction cost as decimal (e.g., 0.001 for 0.1%)
    portfolio_value : float
        Total portfolio value for calculating transaction cost in dollars
    """

    # ======================================================
    # VALIDATION
    # ======================================================

    if optimal_weights is None:
        raise ValueError("optimal_weights cannot be None")

    holdings_df = holdings_df.copy()
    holdings_df["Ticker"] = (
        holdings_df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # ======================================================
    # CURRENT WEIGHTS
    # ======================================================

    # Align both series to optimal_weights index for consistent comparison
    current_weights = weights_series.copy().reindex(optimal_weights.index).fillna(0)
    target_weights = optimal_weights.reindex(current_weights.index)  # Explicit alignment

    # ======================================================
    # TABLE 1 — MVO REBALANCE
    # ======================================================

    mvo_table = pd.DataFrame({
        "Ticker": current_weights.index,
        "Current Weight": current_weights.values,
        "Optimized Weight": target_weights.values
    })

    mvo_table["Allocation Change"] = (
        mvo_table["Optimized Weight"] - mvo_table["Current Weight"]
    )

    def classify_action(x):
        if x > 0.01:
            return "Increase"
        elif x < -0.01:
            return "Reduce"
        else:
            return "Minor Adjust"

    mvo_table["Action"] = mvo_table["Allocation Change"].apply(classify_action)

    mvo_turnover = (
        np.abs(mvo_table["Allocation Change"]).sum() / 2
    )
    
    # Calculate transaction cost impact
    transaction_cost_pct = mvo_turnover * transaction_cost
    transaction_cost_dollar = transaction_cost_pct * portfolio_value if portfolio_value > 0 else 0
    
    # Add transaction cost column to MVO table
    mvo_table["Transaction Cost"] = (
        np.abs(mvo_table["Allocation Change"]) * transaction_cost
    )
    
    mvo_table = (
        mvo_table
        .sort_values("Optimized Weight", ascending=False)
        .reset_index(drop=True)
    )

    if portfolio_metrics_df is None or portfolio_metrics_df.empty:
        raise ValueError("portfolio_metrics_df cannot be empty")

    rule_df = portfolio_metrics_df.copy()
    rule_df["Ticker"] = (
        rule_df["Ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    rule_table = mvo_table[["Ticker"]].merge(
        rule_df,
        on="Ticker",
        how="left"
    )

    if "Relative Performance" not in rule_table.columns:
        raise ValueError(
            "portfolio_metrics_df must contain 'Relative Performance'"
        )

    def rule_engine(x):
        if pd.isna(x):
            return "No Data"
        if x < -0.10:
            return "Sell"
        elif x > 0.20:
            return "Buy"
        else:
            return "Hold"

    rule_table["Action"] = rule_table["Relative Performance"].apply(rule_engine)

    rule_table = rule_table[[
        "Ticker",
        "3M Return",
        "Benchmark 3M",
        "Relative Performance",
        "Action"
    ]].sort_values(
        "Relative Performance",
        ascending=False
    ).reset_index(drop=True)

    # ======================================================
    # RETURN
    # ======================================================
    
    transaction_cost_pct = mvo_turnover * transaction_cost
    transaction_cost_info = {
        "turnover": mvo_turnover,
        "transaction_cost_pct": transaction_cost_pct,
        "transaction_cost_dollar": transaction_cost_pct * portfolio_value if portfolio_value > 0 else 0,
    }

    return mvo_table, rule_table, mvo_turnover, transaction_cost_info


# ==========================================================
# SCENARIO-BASED REBALANCING
# ==========================================================

def simulate_cash_injection(
    weights_series: pd.Series,
    returns: pd.DataFrame,
    optimal_weights: pd.Series,
    cash_amount: float,
    total_value: float,
) -> tuple:
    """
    Distribute cash_amount into the portfolio proportionally to optimal_weights.
    Returns (new_weights, port_return, port_vol, sharpe).
    """
    from optimizer import portfolio_performance

    new_total = total_value + cash_amount

    # Current market values per ticker
    current_values = weights_series * total_value
    # Allocate new cash proportionally to optimal weights
    cash_allocation = optimal_weights * cash_amount
    new_values = current_values.add(cash_allocation, fill_value=0)
    new_weights = (new_values / new_total).reindex(returns.columns).fillna(0)

    w_arr = new_weights.values
    port_return, port_vol, sharpe = portfolio_performance(w_arr, returns, rf_multiplier=1.0)

    return new_weights, port_return, port_vol, sharpe


def simulate_trade(
    weights_series: pd.Series,
    returns: pd.DataFrame,
    ticker: str,
    action: str,
    quantity: float,
    current_price: float,
    total_value: float,
    risk_profile: str,
) -> tuple:
    """
    Simulate buying or selling 'quantity' shares of 'ticker'.
    action: 'Buy' or 'Sell'.
    Returns (new_weights, port_return, port_vol, sharpe, warning_str_or_None).
    """
    from optimizer import portfolio_performance
    from config import RISK_PROFILES

    max_weight = RISK_PROFILES[risk_profile]["max_weight"]
    trade_value = quantity * current_price
    current_values = weights_series * total_value

    if action == "Buy":
        new_total = total_value + trade_value
        ticker_value = current_values.get(ticker, 0.0) + trade_value
    else:  # Sell
        new_total = total_value - trade_value
        ticker_value = max(current_values.get(ticker, 0.0) - trade_value, 0.0)

    if new_total <= 0:
        return weights_series, 0.0, 0.0, 0.0, "Trade would result in zero or negative portfolio value."

    new_values = current_values.copy()
    new_values[ticker] = ticker_value
    new_weights = (new_values / new_total).reindex(returns.columns).fillna(0)
    new_weights = new_weights.clip(lower=0)

    w_arr = new_weights.values
    port_return, port_vol, sharpe = portfolio_performance(w_arr, returns, rf_multiplier=1.0)

    # Warning if any position breaches max_weight
    breaches = new_weights[new_weights > max_weight]
    warning = None
    if not breaches.empty:
        breach_list = ", ".join(f"{t} ({v:.1%})" for t, v in breaches.items())
        warning = f"Position size exceeds {max_weight:.0%} limit for: {breach_list}"

    return new_weights, port_return, port_vol, sharpe, warning