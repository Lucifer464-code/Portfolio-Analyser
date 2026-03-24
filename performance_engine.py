import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from config import TRADING_DAYS


# ==========================================================
# PERFORMANCE METRICS SUMMARY
# ==========================================================

def get_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    rf_rate: float = 0.05,
) -> Dict:
    """
    Compute comprehensive performance metrics.

    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (optional)
        rf_rate: Risk-free rate (annualised, default 5%)

    Returns:
        Dictionary with performance metrics
    """
    # Cumulative returns
    portfolio_cum = (1 + portfolio_returns).cumprod() - 1
    total_return = portfolio_cum.iloc[-1] if len(portfolio_cum) > 0 else 0

    # Annualized returns
    years = len(portfolio_returns) / TRADING_DAYS
    annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0

    # Volatility
    volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)

    # Sharpe ratio (risk-free rate adjusted)
    sharpe = ((annualized_return - rf_rate) / volatility) if volatility > 0 else 0

    # Sortino ratio (downside deviation — RMS of all negative deviations from zero)
    excess = portfolio_returns - 0.0
    semi_var = (np.minimum(excess, 0) ** 2).mean()
    downside_vol = np.sqrt(semi_var) * np.sqrt(TRADING_DAYS) if semi_var > 0 else 0
    sortino = ((annualized_return - rf_rate) / downside_vol) if downside_vol > 0 else 0
    
    # Drawdown — properly calculated as (current_value - peak_value) / peak_value
    # Convert cumulative returns to portfolio values (1 + cumulative return)
    portfolio_values = 1 + portfolio_cum
    running_max_values = portfolio_values.cummax()
    drawdown = (portfolio_values - running_max_values) / running_max_values
    max_drawdown = drawdown.min()
    
    # Win rate
    win_days = (portfolio_returns > 0).sum()
    win_rate = win_days / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
    
    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }
    
    # Benchmark metrics
    if benchmark_returns is not None:
        benchmark_cum = (1 + benchmark_returns).cumprod() - 1
        bench_total = benchmark_cum.iloc[-1] if len(benchmark_cum) > 0 else 0
        bench_annualized = ((1 + bench_total) ** (1 / years) - 1) if years > 0 else 0
        bench_vol = benchmark_returns.std() * np.sqrt(TRADING_DAYS)
        
        # Alpha and beta
        aligned = pd.DataFrame({
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns
        }).dropna()
        
        if len(aligned) > 1:
            covariance = aligned.cov().loc["portfolio", "benchmark"]
            bench_variance = aligned["benchmark"].var()
            beta = covariance / bench_variance if bench_variance > 0 else 0
        else:
            beta = 0
        
        alpha = annualized_return - (beta * bench_annualized)
        
        # Information ratio
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(TRADING_DAYS)
        info_ratio = (annualized_return - bench_annualized) / tracking_error if tracking_error > 0 else 0
        
        metrics.update({
            "benchmark_total_return": bench_total,
            "benchmark_annualized_return": bench_annualized,
            "benchmark_volatility": bench_vol,
            "alpha": alpha,
            "beta": beta,
            "information_ratio": info_ratio,
            "outperformance": annualized_return - bench_annualized,
        })
    
    return metrics


# ==========================================================
# PERIOD-BASED RETURNS
# ==========================================================

def get_period_returns(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Calculate returns for multiple periods (1M, 3M, 6M, YTD, 1Y, 3Y).
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (optional)
        
    Returns:
        DataFrame with period returns
    """
    cumulative = (1 + portfolio_returns).cumprod()
    
    periods = {
        "1M": 30,      # Calendar month
        "3M": 90,      # Calendar quarter
        "6M": 180,     # Half year
        "1Y": 365,     # Full year
        "3Y": 1095,    # 3 calendar years
    }
    
    data = []
    
    # YTD calculation - from January 1 of current year to today
    if len(cumulative) > 0:
        # Get current date and start of current year
        today = pd.Timestamp.today()
        year_start = pd.Timestamp(year=today.year, month=1, day=1)
        
        # Find the first cumulative value on or after January 1st of current year
        ytd_cumulative = cumulative[cumulative.index >= year_start]
        
        if len(ytd_cumulative) > 0:
            # YTD return from year start to today
            ytd_return = ytd_cumulative.iloc[-1] / ytd_cumulative.iloc[0] - 1
            data.append({
                "Period": "YTD",
                "Portfolio Return": ytd_return,
                "Days": (today - ytd_cumulative.index[0]).days,
            })
        else:
            # If no data in current year yet, show from first available data
            ytd_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
            data.append({
                "Period": "YTD",
                "Portfolio Return": ytd_return,
                "Days": len(cumulative),
            })
    
    # Other periods
    today = pd.Timestamp.today()
    for period_name, days in periods.items():
        # Calculate the cutoff date: today minus the specified number of calendar days
        cutoff_date = today - pd.Timedelta(days=days)
        
        # Get all data points on or after the cutoff date
        period_data = cumulative[cumulative.index >= cutoff_date]
        
        if len(period_data) > 0:
            # Calculate return from oldest to newest in this period
            period_return = period_data.iloc[-1] / period_data.iloc[0] - 1
            data.append({
                "Period": period_name,
                "Portfolio Return": period_return,
                "Days": (today - period_data.index[0]).days,
            })
    
    df = pd.DataFrame(data)
    
    # Add benchmark if provided
    if benchmark_returns is not None:
        bench_cum = (1 + benchmark_returns).cumprod()
        today = pd.Timestamp.today()
        
        for idx, row in df.iterrows():
            period = row["Period"]
            
            if period == "YTD":
                # Same YTD logic for benchmark
                year_start = pd.Timestamp(year=today.year, month=1, day=1)
                ytd_bench_cum = bench_cum[bench_cum.index >= year_start]
                
                if len(ytd_bench_cum) > 0:
                    bench_return = ytd_bench_cum.iloc[-1] / ytd_bench_cum.iloc[0] - 1
                else:
                    bench_return = bench_cum.iloc[-1] / bench_cum.iloc[0] - 1 if len(bench_cum) > 0 else 0
            else:
                # Use the same calendar day cutoff
                days = row["Days"]
                cutoff_date = today - pd.Timedelta(days=days)
                period_bench_data = bench_cum[bench_cum.index >= cutoff_date]
                
                if len(period_bench_data) > 0:
                    bench_return = period_bench_data.iloc[-1] / period_bench_data.iloc[0] - 1
                else:
                    bench_return = 0
            
            df.loc[idx, "Benchmark Return"] = bench_return
            df.loc[idx, "Excess Return"] = row["Portfolio Return"] - bench_return
    
    return df


# ==========================================================
# MONTHLY RETURNS HEATMAP
# ==========================================================

def get_monthly_returns_heatmap(
    portfolio_returns: pd.Series,
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Create a calendar-style monthly returns heatmap.
    
    Args:
        portfolio_returns: Portfolio return series with datetime index
        
    Returns:
        Tuple of (data DataFrame, Plotly figure)
    """
    try:
        # Validate input
        if portfolio_returns is None or len(portfolio_returns) == 0:
            return pd.DataFrame(), go.Figure()
        
        # Ensure index is datetime
        if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
            # Try to convert if possible
            try:
                portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
            except:
                return pd.DataFrame(), go.Figure()
        
        # Calculate monthly returns
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Need at least 1 complete month
        if len(monthly_returns) < 1:
            return pd.DataFrame(), go.Figure()
        
        monthly_returns_df = pd.DataFrame(monthly_returns, columns=['Return'])
        
        # Extract year and month from index
        monthly_returns_df['Year'] = pd.to_datetime(monthly_returns_df.index).year
        monthly_returns_df['Month'] = pd.to_datetime(monthly_returns_df.index).month
        
        # Pivot for heatmap
        heatmap_data = monthly_returns_df.pivot_table(
            values='Return',
            index='Year',
            columns='Month',
            aggfunc='first'
        )
        
        if heatmap_data.empty or heatmap_data.shape[0] == 0 or heatmap_data.shape[1] == 0:
            return pd.DataFrame(), go.Figure()
        
        # Rename columns to month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        heatmap_data.columns = [month_names[i-1] if i <= len(month_names) else f"M{i}" for i in heatmap_data.columns]
        
        # Create heatmap figure
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values * 100,  # Convert to percentage
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[
                [0, '#ef4444'],      # Red for negative
                [0.5, '#f5f5f5'],    # White for zero
                [1, '#22c55e'],      # Green for positive
            ],
            zmid=0,
            text=np.round(heatmap_data.values * 100, 1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 11},
            hoverongaps=False,
            colorbar=dict(title="Return %")
        ))
        
        fig.update_layout(
            title="Monthly Returns Heatmap (%)",
            xaxis_title="Month",
            yaxis_title="Year",
            height=300,
            width=1000,
        )
        
        return heatmap_data, fig
    
    except Exception as e:
        return pd.DataFrame(), go.Figure()


# ==========================================================
# ROLLING PERFORMANCE METRICS
# ==========================================================

def get_rolling_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    window: int = 60,
) -> pd.DataFrame:
    """
    Calculate rolling Sharpe ratio and other metrics.
    
    Args:
        portfolio_returns: Portfolio return series
        benchmark_returns: Benchmark return series (optional)
        window: Rolling window in days (default 60)
        
    Returns:
        DataFrame with rolling metrics
    """
    rolling_sharpe = (
        (portfolio_returns.rolling(window).mean() * TRADING_DAYS) /
        (portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS))
    )
    
    rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)
    
    df = pd.DataFrame({
        "Sharpe Ratio": rolling_sharpe,
        "Volatility": rolling_vol,
    })
    
    if benchmark_returns is not None:
        bench_sharpe = (
            (benchmark_returns.rolling(window).mean() * TRADING_DAYS) /
            (benchmark_returns.rolling(window).std() * np.sqrt(TRADING_DAYS))
        )
        df["Benchmark Sharpe"] = bench_sharpe


# ==========================================================
# UPSIDE / DOWNSIDE CAPTURE RATIOS
# ==========================================================

def compute_capture_ratios(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Dict:
    """
    Upside capture:   portfolio compound return on benchmark-up days
                      divided by benchmark compound return on those days × 100.
    Downside capture: same for benchmark-down days.
    Values > 100 upside = good. Values < 100 downside = good.
    """
    aligned = pd.concat([
        portfolio_returns.rename("portfolio"),
        benchmark_returns.rename("benchmark"),
    ], axis=1).dropna()

    if len(aligned) == 0:
        return {"upside_capture": np.nan, "downside_capture": np.nan}

    up_days   = aligned[aligned["benchmark"] > 0]
    down_days = aligned[aligned["benchmark"] < 0]

    def _compound(s: pd.Series) -> float:
        return float((1 + s).prod())

    if len(up_days) > 0:
        bm_up   = _compound(up_days["benchmark"])
        port_up = _compound(up_days["portfolio"])
        upside_capture = port_up / bm_up * 100
    else:
        upside_capture = np.nan

    if len(down_days) > 0:
        bm_down   = _compound(down_days["benchmark"])
        port_down = _compound(down_days["portfolio"])
        downside_capture = port_down / bm_down * 100
    else:
        downside_capture = np.nan

    return {
        "upside_capture":   upside_capture,
        "downside_capture": downside_capture,
    }

    return df