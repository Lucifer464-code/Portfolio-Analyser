# ==========================================================
# PROFESSIONAL PORTFOLIO MANAGER — INSTITUTIONAL DASHBOARD v4
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from io import BytesIO

from config import RISK_PROFILES, TRADING_DAYS, DEFAULT_TRANSACTION_COST
from data_engine import (
    load_and_validate_csv, fetch_market_data, compute_returns,
    aggregate_holdings, compute_xirr, compute_pl_summary,
)
from risk_engine import (generate_risk_summary, rolling_volatility, rolling_correlation,
    compute_drawdown_series, var_cvar_summary, sector_concentration,
    asset_type_concentration, effective_n)
from optimizer import (
    optimize_portfolio, simulate_efficient_frontier, portfolio_performance,
    risk_contribution, OPTIMIZERS, OPTIMIZER_DESCRIPTIONS,
)
from analytics import portfolio_health_score
from enhancement_engine import (
    compute_portfolio_3m_relative_performance,
    generate_enhancement_recommendations,
    generate_sector_wise_recommendations,
)
from asset_analytics_engine import (
    get_asset_key_stats, compute_rolling_volatility, compute_rolling_correlation,
    compute_asset_drawdown, get_asset_fundamental_table,
)
from performance_engine import get_performance_metrics, get_period_returns, get_rolling_metrics

# ── Page config ────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Portfolio Analyser", page_icon="📈")

# ── Plotly template ────────────────────────────────────────
pio.templates["portfolio_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#0b1120", plot_bgcolor="#0b1120",
        font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
        colorway=["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#84cc16"],
        xaxis=dict(gridcolor="#1a2744", linecolor="#1a2744", zerolinecolor="#1a2744", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#1a2744", linecolor="#1a2744", zerolinecolor="#1a2744", tickfont=dict(color="#64748b")),
        legend=dict(bgcolor="rgba(11,17,32,0.8)", bordercolor="#1a2744", borderwidth=1),
        title=dict(font=dict(color="#e2e8f0", size=14)),
    )
)
px.defaults.template = "portfolio_dark"

pio.templates["portfolio_light"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
        font=dict(family="DM Sans, sans-serif", color="#334155", size=12),
        colorway=["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#f97316","#84cc16"],
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0", tickfont=dict(color="#64748b")),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
        title=dict(font=dict(color="#0f172a", size=14)),
    )
)

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* =========================================================
   THEME VARIABLES
========================================================= */

/* ---- DARK (default) ---- */
:root{
--bg-base:#080d18;
--bg-surface:#0b1120;
--bg-elevated:#111827;
--bg-card:#0f1a2e;
--border:#1a2744;
--accent:#3b82f6;
--accent-glow:rgba(59,130,246,.15);
--positive:#22c55e;
--negative:#ef4444;
--warning:#f59e0b;
--text-primary:#e2e8f0;
--text-secondary:#94a3b8;
--text-muted:#64748b;
--radius:12px;
--radius-sm:8px;
}

/* ---- LIGHT ---- */
.light-mode{
--bg-base:#f6f8fc;
--bg-surface:#ffffff;
--bg-elevated:#f1f5f9;
--bg-card:#ffffff;
--border:#e2e8f0;
--accent:#2563eb;
--accent-glow:rgba(37,99,235,.12);
--positive:#16a34a;
--negative:#dc2626;
--warning:#d97706;
--text-primary:#0f172a;
--text-secondary:#334155;
--text-muted:#64748b;
}

/* =========================================================
   GLOBAL
========================================================= */

html,body,[class*="css"]{
font-family:'DM Sans',sans-serif!important;
color:var(--text-primary);
}

.main,.block-container{
background:var(--bg-base)!important;
padding-top:1.5rem!important;
}

h1,h2,h3,h4{
font-weight:600!important;
letter-spacing:-.02em!important;
color:var(--text-primary)!important;
}

h2{font-size:1.1rem!important;margin-bottom:1rem!important}
h3{font-size:.95rem!important}

p,li,span{color:var(--text-secondary)}
code,.stMetric label{font-family:'DM Mono',monospace!important}

/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"]{
background:var(--bg-surface)!important;
border-right:1px solid var(--border)!important;
}

/* =========================================================
   METRICS
========================================================= */

div[data-testid="stMetric"]{
background:linear-gradient(145deg,var(--bg-card),var(--bg-elevated))!important;
border:1px solid var(--border)!important;
border-top:2px solid var(--accent)!important;
border-radius:var(--radius)!important;
padding:18px 20px!important;
min-height:110px!important;
display:flex!important;
flex-direction:column!important;
justify-content:space-between!important;
transition:.18s ease;
}

div[data-testid="stMetric"]:hover{
transform:translateY(-2px)!important;
box-shadow:0 8px 28px var(--accent-glow)!important;
}

div[data-testid="stMetricValue"]{
font-size:1.6rem!important;
font-weight:600!important;
color:var(--text-primary)!important;
}

/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"]{
gap:3px!important;
background:var(--bg-surface)!important;
padding:5px!important;
border-radius:var(--radius)!important;
border:1px solid var(--border)!important;
margin-bottom:1.5rem!important;
}

.stTabs [aria-selected="true"]{
background:var(--bg-elevated)!important;
color:var(--text-primary)!important;
border-bottom:2px solid var(--accent)!important;
}

/* =========================================================
   TABLES & ALERTS
========================================================= */

.stDataFrame{
border:1px solid var(--border)!important;
border-radius:var(--radius)!important;
overflow:hidden!important;
}

.stAlert{
background:var(--bg-elevated)!important;
border:1px solid var(--border)!important;
border-radius:var(--radius)!important;
color:var(--text-secondary)!important;
}

hr{
border-color:var(--border)!important;
margin:1.5rem 0!important;
}

</style>
""", unsafe_allow_html=True)
# ==========================================================
# UI HELPERS
# ==========================================================

def empty_state(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="text-align:center;padding:52px 24px;border:1px dashed var(--border);
        border-radius:var(--radius);background:var(--bg-surface);margin:8px 0;">
        <div style="font-size:36px;margin-bottom:14px;opacity:0.6;">{icon}</div>
        <div style="font-size:14px;font-weight:600;color:var(--text-secondary);margin-bottom:6px;">{title}</div>
        {"" if not subtitle else f'<div style="font-size:12px;color:var(--text-muted);">{subtitle}</div>'}
    </div>""", unsafe_allow_html=True)


def section_header(title):
    st.markdown(f"""
    <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
        color:var(--text-muted);border-bottom:1px solid var(--border);
        padding-bottom:8px;margin:24px 0 16px 0;">{title}</div>
    """, unsafe_allow_html=True)


def style_pl(df, pl_cols):
    def _colour(val):
        try:
            num = float(str(val).replace("%","").replace("$","").replace("₹","").replace(",",""))
            if num > 0: return "color:#22c55e;font-weight:600"
            if num < 0: return "color:#ef4444;font-weight:600"
        except (ValueError, TypeError): pass
        return ""
    return df.style.applymap(_colour, subset=pl_cols)


def slice_tf(data, tf):
    if not isinstance(data.index, pd.DatetimeIndex): return data
    today = pd.Timestamp.today().normalize()
    offsets = {
        "1M":  today - pd.DateOffset(months=1),
        "3M":  today - pd.DateOffset(months=3),
        "6M":  today - pd.DateOffset(months=6),
        "1Y":  today - pd.DateOffset(years=1),
        "3Y":  today - pd.DateOffset(years=3),
        "5Y":  today - pd.DateOffset(years=5),
    }
    if tf not in offsets: return data
    start = offsets[tf]
    # If start date is not a trading day, step back to the nearest prior trading day
    available = data.index[data.index <= start]
    if not available.empty:
        start = available[-1]
    return data[data.index >= start]


def quick_chart(fig, height=320):
    fig.update_layout(height=height, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# PDF HELPERS
# ==========================================================

def _pdf_table_style(header_color, alt_color="#f0f4f8"):
    return TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor(header_color)),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.whitesmoke),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE',      (0,0), (-1,0),  10),
        ('FONTSIZE',      (0,1), (-1,-1), 8),
        ('TOPPADDING',    (0,0), (-1,0),  8),
        ('BOTTOMPADDING', (0,0), (-1,0),  8),
        ('TOPPADDING',    (0,1), (-1,-1), 4),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
        ('ALIGN',         (0,0), (-1,-1), 'LEFT'),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor(alt_color)]),
    ])


def _pdf_section(story, heading_style, title, table_data, col_widths, header_color,
                 alt_color="#f0f4f8", page_break=False):
    if page_break:
        story.append(PageBreak())
    if title:
        story.append(Paragraph(title, heading_style))
    if len(table_data) > 1:
        t = Table(table_data, colWidths=col_widths)
        t.setStyle(_pdf_table_style(header_color, alt_color))
        story.append(t)
    story.append(Spacer(1, 0.15*inch))


def _fig_to_image(fig, width_px, height_px, w_inch, h_inch):
    buf = BytesIO(fig.to_image(format="png", width=width_px, height=height_px))
    buf.seek(0)
    return RLImage(buf, width=w_inch*inch, height=h_inch*inch)


def generate_portfolio_pdf(df, risk_summary, weights_series, optimal_weights,
                           curr_ret, curr_vol, opt_ret, opt_vol, health_score,
                           opt_method="Max Sharpe", portfolio_returns=None,
                           benchmark_returns=None, enhancements=None, currency="$"):
    buffer = BytesIO()
    pdf    = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.5*inch,   bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CT', parent=styles['Heading1'], fontSize=24,
        textColor=colors.HexColor('#2563eb'), spaceAfter=6, alignment=1, fontName='Helvetica-Bold')
    date_style  = ParagraphStyle('DS', parent=styles['Normal'], fontSize=10,
        textColor=colors.HexColor('#64748b'), alignment=1)
    h_style     = ParagraphStyle('CH', parent=styles['Heading2'], fontSize=13,
        textColor=colors.HexColor('#1e40af'), spaceAfter=10, spaceBefore=12, fontName='Helvetica-Bold')

    story = [
        Paragraph("PORTFOLIO ANALYSIS REPORT", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", date_style),
        Spacer(1, 0.25*inch),
    ]

    # Summary
    _pdf_section(story, h_style, "Portfolio Summary", [
        ["Metric", "Value"],
        ["Total Portfolio Value",  f"{currency}{df['Market Value'].sum():,.2f}"],
        ["Number of Holdings",     str(len(df))],
        ["Health Score",           f"{health_score:.1%}"],
        ["Optimisation Method",    opt_method],
        ["Current Return (Ann.)",  f"{curr_ret:.2%}"],
        ["Current Volatility",     f"{curr_vol:.2%}"],
    ], [3*inch, 2*inch], '#1e40af', '#f0f4f8')
    _pdf_section(story, h_style, "Risk Metrics", [
        ["Metric", "Value"],
        ["Sharpe Ratio",  f"{risk_summary.get('Sharpe Ratio',0):.3f}"],
        ["Max Drawdown",  f"{risk_summary.get('Max Drawdown',0):.2%}"],
        ["Sortino Ratio", f"{risk_summary.get('Sortino Ratio',0):.3f}"],
        ["Beta",          f"{risk_summary.get('Beta',0):.3f}"],
    ], [3*inch, 2*inch], '#dc2626', '#fee2e2')

    # Allocation pie chart
    story.append(Paragraph("Asset Allocation", h_style))
    try:
        alloc = df[["Ticker","Market Value"]].sort_values("Market Value", ascending=False).head(10)
        fig   = px.pie(alloc, names="Ticker", values="Market Value",
                       color_discrete_sequence=["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6",
                                                "#06b6d4","#f97316","#84cc16","#ec4899","#14b8a6"])
        fig.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20), font=dict(size=10))
        story.append(_fig_to_image(fig, 500, 350, 5, 3.5))
    except Exception:
        story.append(Paragraph("<i>Chart unavailable</i>", styles['Normal']))
    story.append(Spacer(1, 0.15*inch))

    # Holdings table
    story.append(PageBreak())
    try:
        hd = df[["Ticker","Quantity","Avg Cost","Current Price","Market Value","Current Weight"]].copy()
        hd["Market Value"]   = hd["Market Value"].apply(lambda x: f"{currency}{float(x):,.2f}" if pd.notna(x) else "N/A")
        hd["Current Weight"] = hd["Current Weight"].apply(lambda x: f"{float(x):.2%}" if pd.notna(x) else "N/A")
        hd["Current Price"]  = hd["Current Price"].apply(lambda x: f"{currency}{float(x):,.2f}" if pd.notna(x) else "N/A")
        hd["Avg Cost"]       = hd["Avg Cost"].apply(lambda x: f"{currency}{float(x):,.2f}" if pd.notna(x) else "N/A")
        rows = [["Ticker","Qty","Avg Cost","Price","Value","Weight"]]
        for _, r in hd.iterrows():
            try:
                rows.append([str(r["Ticker"]), f"{float(r['Quantity']):,.0f}",
                             str(r["Avg Cost"]), str(r["Current Price"]),
                             str(r["Market Value"]), str(r["Current Weight"])])
            except (ValueError, TypeError):
                continue
    except Exception:
        rows = [["Ticker","Qty","Avg Cost","Price","Value","Weight"]]
    _pdf_section(story, h_style, "Current Holdings", rows,
                 [1.0*inch,0.7*inch,1.0*inch,0.9*inch,1.0*inch,0.75*inch], '#1e40af')

    # Performance comparison chart
    if portfolio_returns is not None and benchmark_returns is not None:
        story.append(PageBreak())
        story.append(Paragraph("Performance Comparison", h_style))
        try:
            aligned = pd.concat([portfolio_returns.rename("Portfolio"),
                                  benchmark_returns.rename("Benchmark")], axis=1).dropna()
            if not aligned.empty:
                cp   = (1+aligned["Portfolio"]).cumprod()-1
                cb   = (1+aligned["Benchmark"]).cumprod()-1
                fig2 = go.Figure([
                    go.Scatter(x=cp.index, y=cp.values, name='Portfolio',
                               line=dict(color='#3b82f6', width=2.5)),
                    go.Scatter(x=cb.index, y=cb.values, name='Benchmark',
                               line=dict(color='#64748b', width=2.5, dash='dash')),
                ])
                fig2.update_layout(height=350, margin=dict(l=40,r=40,t=40,b=40),
                                   plot_bgcolor='#f9fafb', paper_bgcolor='white',
                                   font=dict(size=10), legend=dict(x=0.02,y=0.98))
                story.append(_fig_to_image(fig2, 650, 350, 6, 3))
        except Exception:
            story.append(Paragraph("<i>Chart unavailable</i>", styles['Normal']))
        story.append(Spacer(1, 0.15*inch))

    # Optimization results
    if optimal_weights is not None:
        _pdf_section(story, h_style, f"Optimisation Results — {opt_method}", [
            ["Metric",    "Current",        "Optimized",      "Change"],
            ["Return",    f"{curr_ret:.2%}", f"{opt_ret:.2%}", f"{opt_ret-curr_ret:+.2%}"],
            ["Volatility",f"{curr_vol:.2%}", f"{opt_vol:.2%}", f"{opt_vol-curr_vol:+.2%}"],
        ], [1.8*inch,1.3*inch,1.3*inch,1.3*inch], '#22c55e', '#f0fdf4')

        # Weight changes table
        try:
            wt_rows = [["Ticker","Current Weight","Optimized Weight","Change"]]
            for ticker in optimal_weights.index:
                curr_w = float(weights_series.get(ticker, 0))
                opt_w  = float(optimal_weights[ticker])
                wt_rows.append([ticker, f"{curr_w:.2%}", f"{opt_w:.2%}", f"{opt_w-curr_w:+.2%}"])
            _pdf_section(story, h_style, "Weight Changes", wt_rows,
                         [1.5*inch,1.5*inch,1.5*inch,1.5*inch], '#1e40af')
        except Exception:
            pass

    # Enhancements
    if enhancements is not None and not enhancements.empty:
        rows = [["Ticker","Price","1Y Return","Alpha","Score"]]
        for _, r in enhancements.head(10).iterrows():
            try:
                rows.append([str(r.get("Ticker","N/A")),
                             f"{currency}{float(r.get('Current Price',0)):.2f}",
                             f"{float(r.get('12M Return',0)):.2%}",
                             f"{float(r.get('Alpha vs SPY (12M)',0)):.2%}",
                             f"{float(r.get('Score',0)):.3f}"])
            except (ValueError, TypeError): continue
        _pdf_section(story, h_style, "Enhancement Recommendations", rows,
                     [0.9*inch,1.1*inch,1.2*inch,1.1*inch,1.2*inch], '#7c3aed', '#f5f3ff', page_break=True)

    pdf.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="display:flex;align-items:center;gap:16px;padding:16px 24px;border-radius:14px;
    background:linear-gradient(135deg,var(--bg-elevated) 0%,var(--bg-card) 100%);
    border:1px solid var(--border);
    margin-top:32px;margin-bottom:28px;box-shadow:0 4px 24px rgba(0,0,0,0.12);">
  <div style="width:46px;height:46px;border-radius:12px;flex-shrink:0;
      background:linear-gradient(135deg,#2563eb 0%,#60a5fa 100%);
      display:flex;align-items:center;justify-content:center;font-size:22px;
      box-shadow:0 4px 16px rgba(59,130,246,0.45);">📈</div>
  <div>
    <div style="font-size:20px;font-weight:700;color:var(--text-primary);letter-spacing:-0.03em;
        line-height:1.2;">Portfolio Analyser</div>
    <div style="font-size:11px;color:var(--accent);margin-top:3px;letter-spacing:0.1em;font-weight:500;">
        INSTITUTIONAL DASHBOARD</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# CACHED DATA FETCHERS
# ==========================================================

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_market_data(tickers, period):
    try:
        data = yf.download(list(tickers), period=period, auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(data, pd.Series): data = data.to_frame(name=tickers[0])
        return data.dropna(how="all")
    except Exception: return pd.DataFrame()


def get_latest_prices(price_data, tickers):
    if "_price_cache" not in st.session_state: st.session_state._price_cache = {}
    latest = price_data.iloc[-1] if len(price_data) > 0 else pd.Series()
    result = {}
    for t in tickers:
        price = latest.get(t) if t in latest.index else None
        if pd.notna(price) and price > 0:
            result[t] = (float(price), False)
            st.session_state._price_cache[t] = float(price)
        elif t in st.session_state._price_cache:
            result[t] = (st.session_state._price_cache[t], True)
        else:
            result[t] = (np.nan, False)
    return result


_QUOTE_TYPE_MAP = {
    "EQUITY":         "Stock",
    "ETF":            "ETF",
    "MUTUALFUND":     "Mutual Fund",
    "CRYPTOCURRENCY": "Crypto",
    "FUTURE":         "Futures",
    "INDEX":          "Index",
    "CURRENCY":       "Currency",
    "BOND":           "Bond",
}

# Static GICS sector map for common US tickers — used as cloud fallback
_US_SECTOR_MAP = {
    # Technology
    "AAPL":"Technology","MSFT":"Technology","NVDA":"Technology","AVGO":"Technology",
    "ORCL":"Technology","CRM":"Technology","AMD":"Technology","INTC":"Technology",
    "QCOM":"Technology","TXN":"Technology","MU":"Technology","AMAT":"Technology",
    "KLAC":"Technology","LRCX":"Technology","ADI":"Technology","MCHP":"Technology",
    "FTNT":"Technology","PANW":"Technology","SNPS":"Technology","CDNS":"Technology",
    "IBM":"Technology","HPQ":"Technology","DELL":"Technology","STX":"Technology",
    # Communication Services
    "GOOG":"Communication Services","GOOGL":"Communication Services",
    "META":"Communication Services","NFLX":"Communication Services",
    "DIS":"Communication Services","CMCSA":"Communication Services",
    "T":"Communication Services","VZ":"Communication Services",
    "TMUS":"Communication Services","CHTR":"Communication Services",
    "ATVI":"Communication Services","EA":"Communication Services",
    # Consumer Discretionary
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary",
    "HD":"Consumer Discretionary","MCD":"Consumer Discretionary",
    "NKE":"Consumer Discretionary","SBUX":"Consumer Discretionary",
    "TJX":"Consumer Discretionary","BKNG":"Consumer Discretionary",
    "LOW":"Consumer Discretionary","TGT":"Consumer Discretionary",
    "ABNB":"Consumer Discretionary","EBAY":"Consumer Discretionary",
    # Consumer Staples
    "WMT":"Consumer Staples","PG":"Consumer Staples","KO":"Consumer Staples",
    "PEP":"Consumer Staples","COST":"Consumer Staples","PM":"Consumer Staples",
    "MO":"Consumer Staples","CL":"Consumer Staples","MDLZ":"Consumer Staples",
    "GIS":"Consumer Staples","KHC":"Consumer Staples","STZ":"Consumer Staples",
    # Financials
    "BRK.B":"Financials","JPM":"Financials","V":"Financials","MA":"Financials",
    "BAC":"Financials","WFC":"Financials","GS":"Financials","MS":"Financials",
    "AXP":"Financials","BLK":"Financials","SCHW":"Financials","C":"Financials",
    "USB":"Financials","PNC":"Financials","TFC":"Financials","COF":"Financials",
    "CB":"Financials","MMC":"Financials","AON":"Financials","ICE":"Financials",
    # Healthcare
    "LLY":"Healthcare","UNH":"Healthcare","JNJ":"Healthcare","MRK":"Healthcare",
    "ABBV":"Healthcare","PFE":"Healthcare","TMO":"Healthcare","ABT":"Healthcare",
    "DHR":"Healthcare","BMY":"Healthcare","AMGN":"Healthcare","GILD":"Healthcare",
    "CVS":"Healthcare","MDT":"Healthcare","SYK":"Healthcare","ISRG":"Healthcare",
    "VRTX":"Healthcare","REGN":"Healthcare","BIIB":"Healthcare","MRNA":"Healthcare",
    # Industrials
    "GE":"Industrials","CAT":"Industrials","HON":"Industrials","UPS":"Industrials",
    "BA":"Industrials","RTX":"Industrials","LMT":"Industrials","DE":"Industrials",
    "MMM":"Industrials","GD":"Industrials","NOC":"Industrials","FDX":"Industrials",
    "EMR":"Industrials","ETN":"Industrials","PH":"Industrials","ROK":"Industrials",
    # Energy
    "XOM":"Energy","CVX":"Energy","COP":"Energy","EOG":"Energy","SLB":"Energy",
    "MPC":"Energy","PSX":"Energy","VLO":"Energy","OXY":"Energy","DVN":"Energy",
    "HAL":"Energy","BKR":"Energy","FANG":"Energy","APA":"Energy",
    # Utilities
    "NEE":"Utilities","DUK":"Utilities","SO":"Utilities","D":"Utilities",
    "AEP":"Utilities","EXC":"Utilities","SRE":"Utilities","XEL":"Utilities",
    "PCG":"Utilities","ED":"Utilities","ETR":"Utilities","FE":"Utilities",
    # Real Estate
    "PLD":"Real Estate","AMT":"Real Estate","EQIX":"Real Estate","CCI":"Real Estate",
    "PSA":"Real Estate","SPG":"Real Estate","WELL":"Real Estate","DLR":"Real Estate",
    "O":"Real Estate","AVB":"Real Estate","EQR":"Real Estate","VTR":"Real Estate",
    # Materials
    "LIN":"Materials","APD":"Materials","SHW":"Materials","FCX":"Materials",
    "NEM":"Materials","ECL":"Materials","DOW":"Materials","DD":"Materials",
    "NUE":"Materials","CTVA":"Materials","ALB":"Materials","MOS":"Materials",
    # Common ETFs
    "SPY":"ETF","QQQ":"ETF","IWM":"ETF","VTI":"ETF","VOO":"ETF",
    "XLK":"ETF","XLF":"ETF","XLV":"ETF","XLE":"ETF","XLI":"ETF",
    "XLY":"ETF","XLP":"ETF","XLU":"ETF","XLB":"ETF","XLRE":"ETF",
    "GLD":"ETF","SLV":"ETF","TLT":"ETF","HYG":"ETF","LQD":"ETF",
    "ARKK":"ETF","ARKG":"ETF","ARKW":"ETF","ARKF":"ETF","ARKQ":"ETF",
}

@st.cache_data(show_spinner=False)
def fetch_ticker_metadata(tickers):
    import time as _time

    def _fetch_single(ticker):
        """Try multiple methods to get sector, falling back gracefully."""
        t = ticker.upper().strip()

        # Method 1: yf.Ticker().info (most complete but rate-limited on cloud)
        for _attempt in range(2):
            try:
                if _attempt > 0:
                    _time.sleep(2)
                info     = yf.Ticker(t).info
                name     = info.get("longName") or info.get("shortName") or t
                sector   = info.get("sector") or ""
                raw_type = (info.get("quoteType") or "").upper()
                atype    = _QUOTE_TYPE_MAP.get(raw_type, raw_type.title() if raw_type else "Equity")
                if sector:
                    return t, name, sector, atype
                # Got info but no sector — use static map or ETF check
                if raw_type == "ETF":
                    return t, name, "ETF", "ETF"
                break
            except Exception:
                pass

        # Method 2: static sector map (instant, no network)
        if t in _US_SECTOR_MAP:
            _sec = _US_SECTOR_MAP[t]
            _atype = "ETF" if _sec == "ETF" else "Equity"
            return t, t, _sec, _atype

        # Method 3: yf.Ticker().fast_info for name + type only
        try:
            fi    = yf.Ticker(t).fast_info
            name  = getattr(fi, "display_name", None) or t
            # Infer type from ticker pattern
            if t.startswith("^"):
                return t, name, "Index", "Index"
            if t.endswith(".NS") or t.endswith(".BO"):
                return t, name, "Unknown", "Equity"
            return t, name, "Unknown", "Equity"
        except Exception:
            pass

        # Final fallback
        if t.startswith("^"):
            return t, t, "Index", "Index"
        if t.endswith(".NS") or t.endswith(".BO"):
            return t, t, "Unknown", "Equity"
        return t, t, "Unknown", "Equity"

    # Sequential fetch with small delay between tickers to avoid rate limiting
    rows = []
    for ticker in tickers:
        t, name, sector, atype = _fetch_single(ticker)
        rows.append({"Ticker": t, "Name": name, "Sector": sector, "Asset Type": atype})
        _time.sleep(0.3)   # small delay between tickers

    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_enhancement_recommendations(): return generate_enhancement_recommendations()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_sector_recommendations(): return generate_sector_wise_recommendations(top_sectors=5, stocks_per_sector=5)

@st.cache_data(show_spinner=False)
def cached_3m_relative_performance(tickers): return compute_portfolio_3m_relative_performance(list(tickers))


# ==========================================================
# VOLATILITY REGIME
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_vol_regime(returns, window=60):
    rv = returns.rolling(window).std() * np.sqrt(252)
    rv = rv.dropna()
    if rv.empty: return "N/A", "#64748b", 0.0
    latest = float(rv.iloc[-1])
    pct    = float(rv.rank(pct=True).iloc[-1])
    if pct < 0.33: return "LOW VOL",    "#22c55e", latest
    if pct < 0.66: return "NORMAL VOL", "#f59e0b", latest
    return "HIGH VOL", "#ef4444", latest


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("Configuration")
    uploaded_file        = st.file_uploader("Upload Portfolio CSV", type="csv")
    benchmark_map = {
        "S&P 500":   "^GSPC",
        "NIFTY 50":  "^NSEI",
        "NIFTY 500": "^CRSLDX",
        "SENSEX":    "^BSESN",
        "Dow Jones": "^DJI",
        "NASDAQ":    "^IXIC",
        "Custom…":   None,
    }
    benchmark_name = st.selectbox("Benchmark", list(benchmark_map.keys()))

    if benchmark_name == "Custom…":
        _custom_input = st.text_input(
            "Enter ticker symbol",
            value=st.session_state.get("custom_benchmark", ""),
            placeholder="e.g. QQQ, URTH, ^FTSE",
            help="Any ticker supported by Yahoo Finance — indices (^), ETFs, or stocks",
        ).strip().upper()

        if _custom_input:
            if _custom_input != st.session_state.get("_last_custom_checked"):
                with st.spinner(f"Validating {_custom_input}…"):
                    try:
                        _test  = yf.Ticker(_custom_input).fast_info
                        _valid = hasattr(_test, "last_price") and _test.last_price is not None
                    except Exception:
                        _valid = False
                st.session_state["_last_custom_checked"] = _custom_input
                st.session_state["_custom_valid"]        = _valid
                if _valid:
                    st.session_state["custom_benchmark"] = _custom_input

            if st.session_state.get("_custom_valid", False):
                st.success(f"✓ {st.session_state['custom_benchmark']}")
                benchmark = st.session_state["custom_benchmark"]
            else:
                st.error("Ticker not found on Yahoo Finance")
                benchmark = st.session_state.get("custom_benchmark", "^GSPC")
        else:
            benchmark = st.session_state.get("custom_benchmark", "^GSPC")
    else:
        benchmark = benchmark_map[benchmark_name]
        st.session_state.pop("custom_benchmark",      None)
        st.session_state.pop("_last_custom_checked",  None)
        st.session_state.pop("_custom_valid",         None)
    risk_profile         = st.selectbox("Risk Profile", list(RISK_PROFILES.keys()))

    st.markdown("<hr>", unsafe_allow_html=True)
    threshold_pct        = st.slider("Rebalance Threshold (%)",  0.0, 10.0,  2.0, 0.1)
    max_weight_pct       = st.slider("Max Position Size (%)",    5.0, 50.0, 15.0, 1.0)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.toggle("☀️ Light Theme", value=False, key="light_mode")

_light = st.session_state.get("light_mode", False)
px.defaults.template = "portfolio_light" if _light else "portfolio_dark"

if _light:
    st.markdown("""
<style>
:root {
    --bg-base:#f6f8fc;
    --bg-surface:#ffffff;
    --bg-elevated:#f1f5f9;
    --bg-card:#ffffff;
    --border:#e2e8f0;
    --accent:#2563eb;
    --accent-glow:rgba(37,99,235,.12);
    --positive:#16a34a;
    --negative:#dc2626;
    --warning:#d97706;
    --text-primary:#0f172a;
    --text-secondary:#334155;
    --text-muted:#64748b;
}
</style>
""", unsafe_allow_html=True)

threshold        = threshold_pct        / 100
max_weight       = max_weight_pct       / 100
lookback         = "max"

if uploaded_file is None:
    st.session_state.pop("data_loaded",    None)
    st.session_state.pop("selected_asset", None)
    st.markdown("""
    <div style="text-align:center;padding:80px 32px;border:1px dashed var(--border);
        border-radius:var(--radius);background:var(--bg-surface);margin-top:24px;">
        <div style="font-size:48px;margin-bottom:20px;">📂</div>
        <div style="font-size:18px;font-weight:600;color:var(--text-primary);margin-bottom:8px;">
            Upload your portfolio to begin</div>
        <div style="font-size:13px;color:var(--text-muted);max-width:320px;margin:0 auto;">
            Upload a CSV with columns: <strong style="color:var(--accent);">Ticker, Date, Action, Quantity, Price</strong>.
            Action should be <strong style="color:var(--accent);">Buy</strong> or <strong style="color:var(--accent);">Sell</strong>.
            Indian stocks: use <strong style="color:var(--accent);">.NS</strong> or
            <strong style="color:var(--accent);">.BO</strong> suffixes.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

_file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
if st.session_state.get("_last_file_id") != _file_id:
    for _k in ("data_loaded", "selected_asset", "risk_summary", "drawdown_series",
               "frontier", "optimal_weights", "_opt_key", "_portfolio_cache",
               "_benchmark_cache"):
        st.session_state.pop(_k, None)
    st.session_state["_last_file_id"] = _file_id

# Bust only the benchmark cache when benchmark ticker changes
if st.session_state.get("_last_benchmark") != benchmark:
    st.session_state.pop("_benchmark_cache", None)
    st.session_state["_last_benchmark"] = benchmark


# ==========================================================
# DATA LOAD & VALIDATION
# ==========================================================

# ── All heavy construction is cached in session_state ──────
# Only runs on first load or when a new file is uploaded.
# Tab switches, timeframe radios, and slider changes skip
# everything below and read directly from session_state.

if "_portfolio_cache" not in st.session_state:

    result       = load_and_validate_csv(uploaded_file)
    transactions = result[0] if isinstance(result, tuple) else result
    if transactions is None or transactions.empty:
        st.error("Uploaded file contains no valid data."); st.stop()
    for col in ("Ticker", "Date", "Action", "Quantity", "Price"):
        if col not in transactions.columns:
            st.error(f"'{col}' column missing."); st.stop()

    # Aggregate transactions → current open holdings
    df = aggregate_holdings(transactions)
    if df is None or df.empty:
        st.error("No open positions found after aggregating transactions."); st.stop()

    tickers       = df["Ticker"].unique().tolist()
    tickers_tuple = tuple(tickers)

    # Market detection
    _market   = "IN" if sum(1 for t in tickers if t.endswith(".NS") or t.endswith(".BO")) >= len(tickers)/2 else "US"
    _currency = "₹" if _market == "IN" else "$"
    _rf_rate  = 0.065 if _market == "IN" else 0.05

    with st.spinner("Fetching market data…"):
        price_data = cached_fetch_market_data(tickers_tuple, lookback)
    if price_data is None or price_data.empty:
        st.error("Unable to fetch price data."); st.stop()

    returns = compute_returns(price_data)
    if returns is None or returns.empty:
        st.error("Return computation failed."); st.stop()

    # Live prices & market values
    prices_info         = get_latest_prices(price_data, tickers)
    df["Current Price"] = df["Ticker"].map(lambda t: prices_info.get(t, (np.nan, False))[0])
    df["_Price Cached"] = df["Ticker"].map(lambda t: prices_info.get(t, (np.nan, False))[1])
    df["Market Value"]  = df["Current Price"] * df["Quantity"]
    total_value         = df["Market Value"].sum()
    if total_value <= 0:
        st.error("Portfolio value invalid."); st.stop()

    df["Current Weight"] = df["Market Value"] / total_value
    df["Unrealised P/L"] = df["Market Value"] - df["Total Cost"]
    df["P/L %"]          = np.where(df["Total Cost"] != 0, df["Unrealised P/L"] / df["Total Cost"], 0)

    # P/L summary
    pl_summary      = compute_pl_summary(df, total_value)
    amount_invested = pl_summary["total_cost"]
    unrealized_gain = pl_summary["unrealised_pl"]
    realised_gain   = pl_summary["realised_pl"]

    # XIRR
    portfolio_xirr = compute_xirr(transactions, total_value)

    weights_series    = df.groupby("Ticker")["Current Weight"].sum().reindex(returns.columns).fillna(0)
    portfolio_returns = returns @ weights_series.values

    # Benchmark — stored separately so it can be refreshed without full reload
    benchmark_returns = None   # placeholder; computed below outside this block

    # Risk analytics (benchmark-independent parts)
    with st.spinner("Running risk analytics…"):
        drawdown_series = compute_drawdown_series(portfolio_returns)
        try:
            frontier = simulate_efficient_frontier(returns, risk_profile)
        except Exception:
            frontier = None

    # Metadata
    with st.spinner("Loading ticker metadata…"):
        metadata = fetch_ticker_metadata(tickers_tuple)
    df["Name"]       = df["Ticker"].map(metadata["Name"])
    df["Sector"]     = df["Ticker"].map(metadata["Sector"])
    df["Asset Type"] = df["Ticker"].map(metadata["Asset Type"])

    regime, regime_color, latest_vol = detect_vol_regime(portfolio_returns)

    # ── Store everything in session_state ──────────────────
    st.session_state["_portfolio_cache"] = {
        "transactions":    transactions,
        "df":              df,
        "tickers":         tickers,
        "tickers_tuple":   tickers_tuple,
        "_market":         _market,
        "_currency":       _currency,
        "_rf_rate":        _rf_rate,
        "price_data":      price_data,
        "returns":         returns,
        "total_value":     total_value,
        "amount_invested": amount_invested,
        "unrealized_gain": unrealized_gain,
        "realised_gain":   realised_gain,
        "portfolio_xirr":  portfolio_xirr,
        "weights_series":  weights_series,
        "portfolio_returns": portfolio_returns,
        "risk_summary":    {},          # populated after benchmark load below
        "drawdown_series": drawdown_series,
        "frontier":        frontier,
        "health_score":    0,           # computed after benchmark load
        "regime":          regime,
        "regime_color":    regime_color,
        "latest_vol":      latest_vol,
    }
    st.session_state["data_loaded"] = True

# ── Read everything from cache ──────────────────────────────
_cache           = st.session_state["_portfolio_cache"]
transactions     = _cache["transactions"]
df               = _cache["df"]
tickers          = _cache["tickers"]
tickers_tuple    = _cache["tickers_tuple"]
_market          = _cache["_market"]
_currency        = _cache["_currency"]
_rf_rate         = _cache["_rf_rate"]
price_data       = _cache["price_data"]
returns          = _cache["returns"]
total_value      = _cache["total_value"]
amount_invested  = _cache["amount_invested"]
unrealized_gain  = _cache["unrealized_gain"]
realised_gain    = _cache["realised_gain"]
portfolio_xirr   = _cache["portfolio_xirr"]
weights_series   = _cache["weights_series"]
portfolio_returns= _cache["portfolio_returns"]
risk_summary     = _cache["risk_summary"]
drawdown_series  = _cache["drawdown_series"]
frontier         = _cache["frontier"]
health_score     = _cache["health_score"]
regime           = _cache["regime"]
regime_color     = _cache["regime_color"]
latest_vol       = _cache["latest_vol"]

# ── Benchmark fetch — separate cache, refreshes on ticker change ───────────
_bm_keys_required = {"benchmark_returns", "risk_summary", "health_score"}
if "_benchmark_cache" not in st.session_state or not _bm_keys_required.issubset(st.session_state.get("_benchmark_cache", {})):
    with st.spinner(f"Fetching benchmark data ({benchmark})…"):
        _bm_data = cached_fetch_market_data((benchmark,), lookback)
    _benchmark_returns = None
    if _bm_data is not None and not _bm_data.empty:
        _bm_ret = compute_returns(_bm_data)
        if _bm_ret is not None and not _bm_ret.empty:
            _benchmark_returns = _bm_ret.iloc[:, 0]
            _aligned = pd.concat([portfolio_returns.rename("Portfolio"),
                                   _benchmark_returns.rename("Benchmark")], axis=1).dropna()
            if not _aligned.empty:
                _benchmark_returns = _aligned["Benchmark"]
    _risk_summary = generate_risk_summary(portfolio_returns, _benchmark_returns)
    _health_score = portfolio_health_score(
        weights_series, _risk_summary.get("Sharpe Ratio", 0), _risk_summary.get("Max Drawdown", 0))
    st.session_state["_benchmark_cache"] = {
        "benchmark_returns": _benchmark_returns,
        "risk_summary":      _risk_summary,
        "health_score":      _health_score,
    }

benchmark_returns = st.session_state["_benchmark_cache"]["benchmark_returns"]
risk_summary      = st.session_state["_benchmark_cache"]["risk_summary"]
health_score      = st.session_state["_benchmark_cache"]["health_score"]
# Keep main cache in sync
st.session_state["_portfolio_cache"]["risk_summary"] = risk_summary
st.session_state["_portfolio_cache"]["health_score"] = health_score

# ── Sidebar market badge ────────────────────────────────────
with st.sidebar:
    _flag = "🇮🇳" if _market == "IN" else "🇺🇸"
    st.markdown(
        f"<div style='font-size:11px;color:var(--text-muted);margin-top:-4px;margin-bottom:8px;"
        f"padding:6px 10px;background:var(--bg-elevated);border-radius:var(--radius-sm);"
        f"border:1px solid var(--border);'>{_flag} Detected: <strong style='color:var(--accent);'>"
        f"{'Indian' if _market=='IN' else 'US'} Market</strong> &nbsp;·&nbsp; {_currency}</div>",
        unsafe_allow_html=True)

# ── Optimizer — re-run only when method/profile/max_weight changes ──
opt_method = st.session_state.get("opt_method", "Max Sharpe")
_opt_key   = (opt_method, risk_profile, max_weight)
if st.session_state.get("_opt_key") != _opt_key:
    with st.spinner(f"Running {opt_method} optimisation…"):
        try:
            optimal_weights = optimize_portfolio(
                returns, risk_profile=risk_profile, method=opt_method,
                max_weight=max_weight,
            )
        except Exception:
            optimal_weights = None
    st.session_state["optimal_weights"] = optimal_weights
    st.session_state["_opt_key"]        = _opt_key
else:
    optimal_weights = st.session_state.get("optimal_weights")

# ── PDF download — generated lazily only when button clicked ──
col_pdf, _ = st.columns([1, 5])
with col_pdf:
    if st.button("📥 Download PDF Report"):
        with st.spinner("Generating PDF…"):
            try:
                enhancements_pdf = cached_enhancement_recommendations()
            except Exception:
                enhancements_pdf = None
            pdf_bytes = generate_portfolio_pdf(
                df, risk_summary, weights_series, optimal_weights,
                curr_ret  = weights_series @ (returns.mean() * 252),
                curr_vol  = float(np.sqrt(weights_series @ (returns.cov() * 252) @ weights_series)),
                opt_ret   = float(optimal_weights @ (returns.mean() * 252)) if optimal_weights is not None else 0,
                opt_vol   = float(np.sqrt(optimal_weights @ (returns.cov() * 252) @ optimal_weights)) if optimal_weights is not None else 0,
                opt_method        = opt_method,
                health_score      = health_score,
                portfolio_returns = portfolio_returns,
                benchmark_returns = benchmark_returns,
                enhancements      = enhancements_pdf,
                currency          = _currency,
            )
        st.download_button(
            "📄 Click to Download",
            data=pdf_bytes,
            file_name=f"Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )


# ==========================================================
# TABS
# ==========================================================

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊  Overview","⚠️  Risk","🎯  Optimization",
    "📈  Performance","🔍  Asset Analytics","✨  Enhancement",
])


# ── TAB 1: OVERVIEW ────────────────────────────────────────
with tab1:
    st.markdown(f"""
    <div style="padding:18px 24px;border-radius:var(--radius);
        background:linear-gradient(135deg,{regime_color}18 0%,{regime_color}08 100%);
        border:1px solid {regime_color}40;border-left:4px solid {regime_color};
        display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;gap:16px;">
      <div>
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
            color:{regime_color};margin-bottom:5px;">Volatility Regime</div>
        <div style="font-size:22px;font-weight:700;color:var(--text-primary);letter-spacing:-0.02em;">{regime}</div>
      </div>
      <div style="text-align:center;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--text-muted);margin-bottom:5px;">Current Vol</div>
        <div style="font-size:22px;font-weight:700;color:var(--text-primary);">{latest_vol:.1%}</div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:10px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;
            color:var(--text-muted);margin-bottom:5px;">Health Score</div>
        <div style="font-size:22px;font-weight:700;color:var(--text-primary);">{health_score}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    section_header("Key Metrics")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Portfolio Value", f"{_currency}{total_value:,.0f}")
    c2.metric("Sharpe Ratio",    f"{risk_summary['Sharpe Ratio']:.2f}")
    c3.metric("Volatility",      f"{risk_summary['Volatility']:.2%}")
    c4.metric("Max Drawdown",    f"{risk_summary['Max Drawdown']:.2%}")
    c5.metric("Beta",            f"{risk_summary.get('Beta',0):.2f}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    section_header("Returns & P/L")
    cA, cB, cC, cD = st.columns(4)
    xirr_display = f"{portfolio_xirr:.2%}" if portfolio_xirr is not None and not np.isnan(portfolio_xirr) else "N/A"
    cA.metric("XIRR (Portfolio)", xirr_display)
    cB.metric("Amount Invested",  f"{_currency}{amount_invested:,.0f}")
    unreal_pct = (unrealized_gain / amount_invested) if amount_invested > 0 else None
    cC.metric("Unrealised P/L",   f"{_currency}{unrealized_gain:,.0f}",
              delta=f"{unreal_pct:.2%}" if unreal_pct is not None else None)
    cD.metric("Realised P/L",     f"{_currency}{realised_gain:,.0f}")

    section_header("Drawdown Trend")
    fig_dd = px.area(drawdown_series.tail(120), color_discrete_sequence=["#ef4444"])
    fig_dd.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,0.12)")
    quick_chart(fig_dd, 220)

    section_header("Allocation")
    cX, cY = st.columns(2)
    with cX:
        at = df.groupby("Asset Type")["Market Value"].sum().reset_index()
        if not at.empty:
            fig = px.pie(at, names="Asset Type", values="Market Value", hole=0.65)
            fig.update_traces(textfont_size=12, marker=dict(line=dict(color="#ffffff" if _light else "#080d18",width=2)))
            fig.update_layout(title=dict(text="Asset Allocation",x=0.5), height=380,
                              margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="v",x=1.02))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📦","Asset type data unavailable","Could not classify asset types")

    with cY:
        sa = df[df["Sector"]!="Unknown"].groupby("Sector")["Market Value"].sum().reset_index()
        if not sa.empty:
            fig = px.pie(sa, names="Sector", values="Market Value", hole=0.65)
            fig.update_traces(textfont_size=12, marker=dict(line=dict(color="#ffffff" if _light else "#080d18",width=2)))
            fig.update_layout(title=dict(text="Sector Allocation",x=0.5), height=380,
                              margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="v",x=1.02))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("🏭","Sector data unavailable","Could not classify tickers into sectors")

    section_header("Holdings")
    cached_count = df["_Price Cached"].sum() if "_Price Cached" in df.columns else 0
    if cached_count > 0:
        st.warning(f"⏱️ {cached_count} price(s) using cached values. Refresh to update.", icon="⏱️")

    holdings_display = df[["Ticker", "Quantity", "Avg Cost", "Current Price",
                            "Current Weight", "Unrealised P/L", "P/L %"]].copy()
    holdings_display = holdings_display.sort_values("Current Weight", ascending=False).reset_index(drop=True)
    holdings_display.index += 1

    def _cache_tag(t):
        row = df[df["Ticker"] == t]["_Price Cached"].values
        return f"{t} ⏱️" if len(row) > 0 and row[0] else t
    holdings_display["Ticker"] = holdings_display["Ticker"].apply(_cache_tag)

    styled = style_pl(holdings_display, ["Unrealised P/L", "P/L %"]).format({
        "Avg Cost":      f"{_currency}{{:,.2f}}",
        "Current Price": f"{_currency}{{:,.2f}}",
        "Unrealised P/L":f"{_currency}{{:,.2f}}",
        "Current Weight":"{:.2%}",
        "P/L %":         "{:.2%}",
    })
    st.dataframe(styled, use_container_width=True)


# ── TAB 2: RISK ────────────────────────────────────────────
with tab2:

    # ── Risk Summary ───────────────────────────────────────
    section_header("Risk Summary")
    pct_fields = {"Annual Return","Volatility","Max Drawdown","VaR 95%","CVaR 95%","Tracking Error","Correlation"}
    st.table(pd.DataFrame({k: f"{v:.2%}" if k in pct_fields else f"{v:.2f}"
                            for k,v in risk_summary.items()}.items(), columns=["Metric","Value"]))

    # ── Rolling charts ─────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        section_header("Rolling Volatility")
        tf = st.radio("Timeframe",["1Y","3Y","5Y"],horizontal=True,key="vol_tf",label_visibility="collapsed")
        quick_chart(px.line(slice_tf(rolling_volatility(portfolio_returns), tf),
                            color_discrete_sequence=["#3b82f6"]))
    with c2:
        if benchmark_returns is not None:
            section_header("Rolling Benchmark Correlation")
            tf = st.radio("Timeframe",["1Y","3Y","5Y"],horizontal=True,key="corr_tf",label_visibility="collapsed")
            quick_chart(px.line(slice_tf(rolling_correlation(portfolio_returns,benchmark_returns), tf),
                                color_discrete_sequence=["#f59e0b"]))
        else:
            empty_state("📉","No benchmark data","Benchmark returns could not be fetched")

    # ── VaR / CVaR ─────────────────────────────────────────
    section_header("Value at Risk & Expected Shortfall")

    _vc = var_cvar_summary(portfolio_returns)

    # Metric cards — 4 columns, 95% and 99%
    v1,v2,v3,v4 = st.columns(4)
    v1.metric("Hist. VaR 95%",   f"{_vc['hist_var_95']:.2%}",  help="Worst daily loss exceeded on 5% of trading days (historical)")
    v2.metric("Hist. CVaR 95%",  f"{_vc['hist_cvar_95']:.2%}", help="Average loss on the worst 5% of days (Expected Shortfall)")
    v3.metric("Hist. VaR 99%",   f"{_vc['hist_var_99']:.2%}",  help="Worst daily loss exceeded on 1% of trading days (historical)")
    v4.metric("Hist. CVaR 99%",  f"{_vc['hist_cvar_99']:.2%}", help="Average loss on the worst 1% of days (Expected Shortfall)")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    v5,v6,v7,v8 = st.columns(4)
    v5.metric("Param. VaR 95%",  f"{_vc['param_var_95']:.2%}",  help="Gaussian VaR 95% — assumes normally distributed returns")
    v6.metric("Param. CVaR 95%", f"{_vc['param_cvar_95']:.2%}", help="Gaussian CVaR 95%")
    v7.metric("Param. VaR 99%",  f"{_vc['param_var_99']:.2%}",  help="Gaussian VaR 99%")
    v8.metric("Param. CVaR 99%", f"{_vc['param_cvar_99']:.2%}", help="Gaussian CVaR 99%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Return distribution chart with VaR/CVaR overlays
    section_header("Return Distribution")
    _ret_clean = portfolio_returns.dropna()
    _hist_fig  = go.Figure()

    # Histogram of daily returns
    _hist_fig.add_trace(go.Histogram(
        x=_ret_clean, nbinsx=80, name="Daily Returns",
        marker_color="#3b82f6", opacity=0.6,
        hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
    ))

    # Overlay normal distribution curve
    import scipy.stats as _stats
    _mu, _sigma = _ret_clean.mean(), _ret_clean.std()
    _x_range = np.linspace(_ret_clean.min(), _ret_clean.max(), 300)
    _pdf     = _stats.norm.pdf(_x_range, _mu, _sigma)
    _scale   = len(_ret_clean) * (_ret_clean.max() - _ret_clean.min()) / 80
    _hist_fig.add_trace(go.Scatter(
        x=_x_range, y=_pdf * _scale, name="Normal Fit",
        line=dict(color="#94a3b8", width=2, dash="dot"),
        hovertemplate="Return: %{x:.2%}<extra>Normal Fit</extra>",
    ))

    # VaR lines
    for _label, _val, _color in [
        ("VaR 95%",  _vc["hist_var_95"],  "#f59e0b"),
        ("CVaR 95%", _vc["hist_cvar_95"], "#ef4444"),
        ("VaR 99%",  _vc["hist_var_99"],  "#8b5cf6"),
    ]:
        _hist_fig.add_vline(
            x=_val, line_dash="dash", line_color=_color, line_width=1.5,
            annotation_text=f"{_label}: {_val:.2%}",
            annotation_position="top left",
            annotation_font=dict(size=10, color=_color),
        )

    _hist_fig.update_layout(
        height=340, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(tickformat=".1%", title="Daily Return"),
        yaxis_title="Frequency",
        showlegend=True,
        bargap=0.05,
    )
    st.plotly_chart(_hist_fig, use_container_width=True)

    # ── Concentration Analytics ────────────────────────────
    section_header("Concentration & Diversification")

    _eff_n  = effective_n(weights_series)
    _hhi    = float((weights_series ** 2).sum())
    _top1   = float(weights_series.max())
    _top3   = float(weights_series.nlargest(3).sum())

    cx1,cx2,cx3,cx4 = st.columns(4)
    cx1.metric("Effective Positions", f"{_eff_n:.1f}",
               help="1/HHI — equivalent number of equal-weight positions with same concentration")
    cx2.metric("HHI",                 f"{_hhi:.4f}",
               help="Herfindahl-Hirschman Index. <0.15 = diversified, >0.25 = concentrated")
    cx3.metric("Largest Position",    f"{_top1:.2%}")
    cx4.metric("Top 3 Concentration", f"{_top3:.2%}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Sector breakdown — only show if we have meaningful sector data
    _has_sector_data = "Sector" in df.columns and (df["Sector"] != "Unknown").any()
    if _has_sector_data:
        _sec_df  = sector_concentration(df[df["Sector"] != "Unknown"])
        _type_df = asset_type_concentration(df)

        sc1, sc2 = st.columns(2)
        with sc1:
            section_header("Sector Breakdown")
            # Treemap
            _sec_fig = px.treemap(
                _sec_df, path=["Sector"], values="Weight",
                color="Weight", color_continuous_scale="Blues",
                custom_data=["Holdings"],
            )
            _sec_fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value:.1%}",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<br>Holdings: %{customdata[0]}<extra></extra>",
            )
            _sec_fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                                   coloraxis_showscale=False)
            st.plotly_chart(_sec_fig, use_container_width=True)

            # Table underneath
            _sec_display = _sec_df[["Sector","Weight","Holdings","% of HHI"]].copy()
            _sec_display["Weight"]    = _sec_display["Weight"].map("{:.2%}".format)
            _sec_display["% of HHI"] = _sec_display["% of HHI"].map("{:.1%}".format)
            st.dataframe(_sec_display.set_index("Sector"), use_container_width=True)

        with sc2:
            section_header("Asset Type Breakdown")
            _type_fig = px.pie(
                _type_df, names="Asset Type", values="Weight",
                color_discrete_sequence=["#3b82f6","#22c55e","#f59e0b","#ef4444","#8b5cf6","#06b6d4"],
                hole=0.45,
            )
            _type_fig.update_traces(
                texttemplate="%{label}<br>%{percent}",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>",
            )
            _type_fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                                    showlegend=True,
                                    legend=dict(orientation="h",yanchor="bottom",y=-0.15))
            st.plotly_chart(_type_fig, use_container_width=True)

            _type_display = _type_df.copy()
            _type_display["Weight"] = _type_display["Weight"].map("{:.2%}".format)
            st.dataframe(_type_display.set_index("Asset Type"), use_container_width=True)

    # ── Correlation Matrix ─────────────────────────────────
    if returns.shape[1] > 1:
        section_header("Asset Correlation Matrix")
        fig = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                        origin="lower", aspect="auto")
        fig.update_layout(height=580, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        empty_state("🔗","Correlation matrix requires multiple assets")


# ── TAB 3: OPTIMIZATION ────────────────────────────────────
with tab3:

    METHOD_INFO = {
        "Max Sharpe":          {"icon":"📈","tagline":"Best risk-adjusted return",
            "desc":"Maximises the Sharpe Ratio — the classic mean-variance optimum. "
                   "Seeks the highest return per unit of risk using your active risk profile's risk-free rate."},
        "Min Variance":        {"icon":"🛡️","tagline":"Lowest possible volatility",
            "desc":"Ignores expected returns entirely and focuses purely on the covariance structure. "
                   "Ideal when return forecasts are unreliable and capital preservation is the priority."},
        "Risk Parity":         {"icon":"⚖️","tagline":"Equal risk from every asset",
            "desc":"Sizes each position so it contributes the same marginal risk to the portfolio "
                   "(Equal Risk Contribution). Naturally diversifies risk without requiring return assumptions."},
        "Max Diversification": {"icon":"🌐","tagline":"Highest diversification ratio",
            "desc":"Maximises the ratio of weighted-average individual volatility to portfolio volatility. "
                   "Exploits low correlations to build the most structurally diversified portfolio possible."},
        "Equal Weight":        {"icon":"🔢","tagline":"Simple 1/N baseline",
            "desc":"Allocates equally to every holding. Requires no estimates and is surprisingly hard "
                   "to beat consistently — a useful benchmark against which to judge every other method."},
    }

    section_header("Optimisation Method")

    # ── Method selector: on_change busts the opt cache, no manual rerun needed ──
    def _on_method_change():
        st.session_state.pop("_opt_key", None)
        st.session_state.pop("optimal_weights", None)
        st.session_state["opt_method"] = st.session_state.get("opt_method_radio",
                                         st.session_state.get("opt_method", "Max Sharpe"))

    st.radio(
        "Method",
        list(OPTIMIZERS.keys()),
        horizontal=True,
        index=list(OPTIMIZERS.keys()).index(st.session_state.get("opt_method", "Max Sharpe")),
        key="opt_method_radio",
        label_visibility="collapsed",
        on_change=_on_method_change,
    )

    opt_method = st.session_state.get("opt_method", "Max Sharpe")

    # Method info card
    info = METHOD_INFO.get(opt_method, {})
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:16px;padding:16px 20px;margin-top:16px;
        border-radius:var(--radius);background:var(--bg-surface);
        border:1px solid var(--border);border-left:3px solid var(--accent);margin-bottom:24px;">
      <div style="font-size:28px;line-height:1;">{info.get('icon','🎯')}</div>
      <div>
        <div style="font-size:13px;font-weight:600;color:var(--text-primary);margin-bottom:4px;">
            {opt_method}
            <span style="font-size:11px;font-weight:500;color:var(--accent);
                margin-left:8px;letter-spacing:0.04em;">{info.get('tagline','')}</span>
        </div>
        <div style="font-size:12px;color:var(--text-muted);line-height:1.6;">{info.get('desc','')}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if optimal_weights is not None and frontier is not None:
        rf = RISK_PROFILES[risk_profile]["rf_multiplier"]
        curr_ret, curr_vol, curr_sharpe = portfolio_performance(weights_series.values,  returns, rf)
        opt_ret,  opt_vol,  opt_sharpe  = portfolio_performance(optimal_weights.values, returns, rf)

        # ── Comparison metrics ─────────────────────────────
        section_header("Portfolio Comparison")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Current Return",       f"{curr_ret:.2%}")
        c2.metric("Current Volatility",   f"{curr_vol:.2%}")
        c3.metric("Current Sharpe",       f"{curr_sharpe:.2f}")
        c4.metric("Optimized Return",     f"{opt_ret:.2%}",    delta=f"{opt_ret-curr_ret:.2%}")
        c5.metric("Optimized Volatility", f"{opt_vol:.2%}",    delta=f"{opt_vol-curr_vol:.2%}")
        c6.metric("Optimized Sharpe",     f"{opt_sharpe:.2f}", delta=f"{opt_sharpe-curr_sharpe:.2f}")

        # ── Weight comparison + trade instructions ────────
        section_header("Weight Comparison & Trade Instructions")

        # Build price & quantity lookup from holdings df
        _price_map = df.set_index("Ticker")["Current Price"].to_dict()
        _qty_map   = df.set_index("Ticker")["Quantity"].to_dict()

        wt_df = pd.DataFrame({
            "Current Weight":   weights_series.reindex(optimal_weights.index).fillna(0),
            "Optimized Weight": optimal_weights,
        })
        wt_df["Change"] = wt_df["Optimized Weight"] - wt_df["Current Weight"]
        wt_df = wt_df.sort_values("Optimized Weight", ascending=False)

        # Trade instruction columns
        _trades = []
        for ticker in wt_df.index:
            _price        = _price_map.get(ticker, np.nan)
            _cur_qty      = float(_qty_map.get(ticker, 0))
            _opt_weight   = float(wt_df.loc[ticker, "Optimized Weight"])
            _target_value = _opt_weight * total_value
            _target_qty   = (_target_value / _price) if (pd.notna(_price) and _price > 0) else np.nan
            _delta_qty    = (_target_qty - _cur_qty)  if pd.notna(_target_qty) else np.nan

            # Round to whole shares
            _delta_qty_r  = round(_delta_qty)  if pd.notna(_delta_qty)  else np.nan
            _target_qty_r = round(_target_qty) if pd.notna(_target_qty) else np.nan
            _cur_qty_r    = round(_cur_qty)

            if pd.isna(_delta_qty_r):
                _action = "N/A"
                _shares = "N/A"
            elif abs(_delta_qty_r) < 1:           # ignore sub-share noise
                _action = "Hold"
                _shares = "—"
            elif _delta_qty_r > 0:
                _action = "Buy"
                _shares = f"+{_delta_qty_r:,.0f}"
            else:
                _action = "Sell"
                _shares = f"{_delta_qty_r:,.0f}"

            _trades.append({
                "Action":         _action,
                "Shares":         _shares,
                "Current Qty":    f"{_cur_qty_r:,.0f}",
                "Target Qty":     f"{_target_qty_r:,.0f}" if pd.notna(_target_qty_r) else "N/A",
                "Current Price":  f"{_currency}{_price:,.2f}" if pd.notna(_price) else "N/A",
            })

        _trade_df = pd.DataFrame(_trades, index=wt_df.index)
        wt_df = pd.concat([wt_df, _trade_df], axis=1)

        # Colour-code the Action column
        def _style_action(val):
            if val == "Buy":  return "color: #22c55e; font-weight:600"
            if val == "Sell": return "color: #ef4444; font-weight:600"
            if val == "Hold": return "color: #94a3b8"
            return ""

        st.dataframe(
            style_pl(wt_df, ["Change"])
            .format({
                "Current Weight":   "{:.2%}",
                "Optimized Weight": "{:.2%}",
                "Change":           "{:+.2%}",
            })
            .applymap(_style_action, subset=["Action"]),
            use_container_width=True,
        )

        # Turnover summary
        _turnover     = float(wt_df["Change"].abs().sum() / 2)
        _cost_pct     = _turnover * DEFAULT_TRANSACTION_COST
        _cost_value   = _cost_pct * total_value
        _buy_count    = (_trade_df["Action"] == "Buy").sum()
        _sell_count   = (_trade_df["Action"] == "Sell").sum()
        _hold_count   = (_trade_df["Action"] == "Hold").sum()

        ts1,ts2,ts3,ts4,ts5 = st.columns(5)
        ts1.metric("Portfolio Turnover",   f"{_turnover:.2%}")
        ts2.metric("Est. Transaction Cost",f"{_currency}{_cost_value:,.0f}", help=f"{_cost_pct:.3%} of AUM")
        ts3.metric("Buys",                 str(_buy_count),  delta=None)
        ts4.metric("Sells",                str(_sell_count), delta=None)
        ts5.metric("Holds",                str(_hold_count), delta=None)

        # ── Weight bar chart ───────────────────────────────
        wt_bar = pd.DataFrame({
            "Current":   weights_series.reindex(optimal_weights.index).fillna(0).values,
            "Optimized": optimal_weights.values,
        }, index=optimal_weights.index)
        fig_wt = go.Figure([
            go.Bar(name="Current",   x=wt_bar.index, y=wt_bar["Current"],   marker_color="#64748b"),
            go.Bar(name="Optimized", x=wt_bar.index, y=wt_bar["Optimized"], marker_color="#22c55e"),
        ])
        fig_wt.update_layout(barmode="group", height=300, margin=dict(l=0,r=0,t=0,b=0),
                             yaxis_tickformat=".1%",
                             legend=dict(orientation="h",y=1.02,x=1,xanchor="right"))
        st.plotly_chart(fig_wt, use_container_width=True)

        # ── Risk Parity: risk contribution chart ──────────
        if opt_method == "Risk Parity":
            section_header("Risk Contributions — Equal Risk Parity Target")
            cov_ann  = returns.cov().values * 252
            w_arr    = optimal_weights.values
            pv       = float(np.sqrt(w_arr @ cov_ann @ w_arr))
            rc       = w_arr * (cov_ann @ w_arr) / pv
            rc_pct   = rc / rc.sum()
            rc_df    = pd.DataFrame({"Asset": optimal_weights.index,
                                     "Risk Contribution": rc_pct, "Weight": w_arr})
            fig_rc   = px.bar(rc_df, x="Asset", y="Risk Contribution",
                              color="Risk Contribution", color_continuous_scale="Blues",
                              hover_data={"Weight":":.2%","Risk Contribution":":.2%"})
            fig_rc.add_hline(y=1/len(optimal_weights), line_dash="dash", line_color="#f59e0b",
                             annotation_text="Equal target", annotation_position="top right")
            fig_rc.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                                 showlegend=False, coloraxis_showscale=False)
            fig_rc.update_yaxes(tickformat=".1%")
            st.plotly_chart(fig_rc, use_container_width=True)

        # ── Max Diversification: DR metrics ───────────────
        if opt_method == "Max Diversification":
            section_header("Diversification Ratio")
            cov_ann    = returns.cov().values * 252
            asset_vols = np.sqrt(np.diag(cov_ann))
            w_opt      = optimal_weights.values
            w_cur      = weights_series.reindex(optimal_weights.index).fillna(0).values
            dr_opt  = float(w_opt @ asset_vols) / float(np.sqrt(w_opt @ cov_ann @ w_opt))
            dr_cur  = float(w_cur @ asset_vols) / float(np.sqrt(w_cur @ cov_ann @ w_cur))
            d1, d2  = st.columns(2)
            d1.metric("Current DR",   f"{dr_cur:.3f}")
            d2.metric("Optimized DR", f"{dr_opt:.3f}", delta=f"{dr_opt-dr_cur:+.3f}")

        # ── Efficient frontier ─────────────────────────────
        section_header("Efficient Frontier")
        fig_f = go.Figure([
            go.Scatter(
                x=frontier["Volatility"], y=frontier["Return"], mode="markers",
                marker=dict(size=5, color=frontier["Sharpe"], colorscale="Blues", showscale=True,
                            colorbar=dict(title=dict(text="Sharpe",font=dict(color="#334155" if _light else "#94a3b8")),
                                          x=1.02,thickness=14,len=0.6,tickfont=dict(color="#334155" if _light else "#94a3b8"))),
                name="Monte Carlo Simulations",
                hovertemplate="<b>Return:</b> %{y:.2%}<br><b>Vol:</b> %{x:.2%}<extra></extra>",
            ),
            go.Scatter(
                x=[curr_vol], y=[curr_ret], mode="markers",
                marker=dict(size=24,color="#ef4444",line=dict(color="white",width=3)),
                name="Current Portfolio",
                hovertemplate="<b>Current</b><br>Return: %{y:.2%}<br>Vol: %{x:.2%}<extra></extra>",
            ),
            go.Scatter(
                x=[opt_vol], y=[opt_ret], mode="markers",
                marker=dict(size=28,color="#22c55e",line=dict(color="white",width=3)),
                name=f"Optimized ({opt_method})",
                hovertemplate=f"<b>{opt_method}</b><br>Return: %{{y:.2%}}<br>Vol: %{{x:.2%}}<extra></extra>",
            ),
        ])
        fig_f.update_layout(
            height=580, margin=dict(l=40,r=40,t=20,b=40), showlegend=True,
            legend=dict(orientation="v",yanchor="bottom",y=0.04,xanchor="left",x=0.02,
                        bgcolor="rgba(255,255,255,0.92)" if _light else "rgba(11,17,32,0.92)",
                        bordercolor="#3b82f6",borderwidth=1,
                        font=dict(color="#0f172a" if _light else "#e2e8f0",size=12)))
        st.plotly_chart(fig_f, use_container_width=True)

    else:
        empty_state("🎯","Optimization unavailable","Could not compute optimal weights for this portfolio")


# ── TAB 4: PERFORMANCE ─────────────────────────────────────
with tab4:
    section_header("Performance Metrics")
    pm = get_performance_metrics(portfolio_returns, benchmark_returns, rf_rate=_rf_rate)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Return",      f"{pm['total_return']:.2%}")
    c2.metric("Annualized Return", f"{pm['annualized_return']:.2%}")
    c3.metric("Volatility",        f"{pm['volatility']:.2%}")
    c4.metric("Sharpe Ratio",      f"{pm['sharpe_ratio']:.2f}")
    c5.metric("Sortino Ratio",     f"{pm['sortino_ratio']:.2f}")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    cA,cB,cC,cD = st.columns(4)
    cA.metric("Max Drawdown", f"{pm['max_drawdown']:.2%}")
    cB.metric("Win Rate",     f"{pm['win_rate']:.1%}")
    if "alpha" in pm:
        cC.metric("Alpha",             f"{pm['alpha']:.2%}")
        cD.metric("Information Ratio", f"{pm['information_ratio']:.2f}")

    if benchmark_returns is not None:
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        section_header("Benchmark Comparison")
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Portfolio Ann. Return", f"{pm['annualized_return']:.2%}")
        b2.metric("Benchmark Ann. Return", f"{pm['benchmark_annualized_return']:.2%}")
        b3.metric("Outperformance",        f"{pm['outperformance']:.2%}")
        b4.metric("Beta",                  f"{pm['beta']:.2f}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("Cumulative Returns")
    tf = st.radio("Timeframe",["1M","3M","6M","1Y","3Y","5Y"],horizontal=True,
                  key="perf_timeframe",label_visibility="collapsed")

    # ── Build cumulative return series, rebased to 0 at start of window ──
    # Use (1+r).cumprod() so we stay in price-relative space, then rebase:
    # rebase = series / series.iloc[0] - 1  (correct ratio rebase, not subtraction)
    cum_full = (1 + portfolio_returns.dropna()).cumprod()
    cum      = slice_tf(cum_full, tf)
    cum      = cum / cum.iloc[0] - 1          # rebase: 0% at window start

    perf_df = pd.DataFrame({"Portfolio": cum})

    if benchmark_returns is not None:
        bc_full = (1 + benchmark_returns.dropna()).cumprod()
        bc      = slice_tf(bc_full, tf)
        # Align benchmark to same start date as portfolio
        bc      = bc.reindex(cum.index, method="ffill")
        bc      = bc / bc.iloc[0] - 1
        perf_df["Benchmark"] = bc

    fig = px.line(perf_df, color_discrete_map={"Portfolio":"#3b82f6","Benchmark":"#64748b"})
    fig.update_traces(line=dict(width=2))

    # ── XIRR reference line ────────────────────────────────
    # XIRR is an annualised cash-flow-weighted rate — project it as a straight
    # compound growth curve anchored at 0% at the window start, so it serves
    # as a "target pace" reference rather than a conflated return series.
    if portfolio_xirr is not None and not np.isnan(portfolio_xirr):
        xirr_start = cum.index[0]
        xirr_end   = cum.index[-1]
        xirr_dates = pd.date_range(xirr_start, xirr_end, freq="B")
        days_from_start = (xirr_dates - xirr_start).days
        xirr_curve      = (1 + portfolio_xirr) ** (days_from_start / 365) - 1

        fig.add_scatter(
            x=xirr_dates,
            y=xirr_curve,
            name=f"XIRR Pace ({portfolio_xirr:.2%} p.a.)",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        )

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(tickformat=".0%", title="Return"),
        xaxis=dict(title="Date"),
        hovermode="x unified",
    )
    fig.update_traces(hovertemplate="%{y:.2%}")
    st.plotly_chart(fig, use_container_width=True)

    section_header("Period Returns")
    pr      = get_period_returns(portfolio_returns, benchmark_returns)
    pr.index += 1
    pl_cols = ["Portfolio Return","Benchmark Return","Excess Return"] if "Benchmark Return" in pr.columns else ["Portfolio Return"]
    st.dataframe(style_pl(pr, pl_cols).format({c:"{:.2%}" for c in pl_cols}), use_container_width=True)

    section_header("Rolling 60-Day Sharpe Ratio")
    rm  = get_rolling_metrics(portfolio_returns, benchmark_returns, window=60)
    fig = px.line(rm[["Sharpe Ratio"]], color_discrete_map={"Sharpe Ratio":"#3b82f6"})
    if "Benchmark Sharpe" in rm.columns:
        fig.add_scatter(x=rm.index, y=rm["Benchmark Sharpe"], name="Benchmark Sharpe",
                        line=dict(color="#64748b",width=2))
    fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig, use_container_width=True)

    section_header("Individual Stock Performance vs Benchmark")
    if returns is not None and not returns.empty:
        years      = len(returns) / TRADING_DAYS
        bench_ann  = 0
        if benchmark_returns is not None:
            bt        = (1+benchmark_returns).cumprod()-1
            bench_ann = ((1+bt.iloc[-1])**(1/years)-1) if years > 0 else 0

        stock_data = []
        for ticker in returns.columns:
            sr = returns[ticker].dropna()
            if len(sr) > 0:
                total = (1+sr).cumprod().iloc[-1]-1
                ann   = ((1+total)**(1/years)-1) if years > 0 else 0
                stock_data.append({"Ticker":ticker,"Annualized Return":ann,"vs Benchmark":ann-bench_ann})

        if stock_data:
            spdf = pd.DataFrame(stock_data).sort_values("Annualized Return", ascending=True)
            fig  = go.Figure([go.Bar(
                y=spdf["Ticker"], x=spdf["Annualized Return"], orientation='h',
                marker=dict(color=spdf["Annualized Return"], colorscale="RdYlGn",
                            cmin=spdf["Annualized Return"].min(), cmax=spdf["Annualized Return"].max()),
                hovertemplate="<b>%{y}</b><br>Return: %{x:.2%}<extra></extra>",
            )])
            if benchmark_returns is not None:
                fig.add_vline(x=bench_ann, line_dash="dash", line_color="#64748b",
                              annotation_text=f"Benchmark: {bench_ann:.2%}",
                              annotation_position="top right")
            fig.update_layout(height=max(300,len(spdf)*25), margin=dict(l=100,r=40,t=60,b=40),
                               xaxis_title="Annualized Return", showlegend=False)
            fig.update_xaxes(tickformat=".1%")
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 5: ASSET ANALYTICS ─────────────────────────────────
with tab5:
    if "selected_asset" not in st.session_state: st.session_state["selected_asset"] = tickers[0]
    if st.session_state["selected_asset"] not in tickers: st.session_state["selected_asset"] = tickers[0]

    selected_asset = st.selectbox("Select Asset", tickers, key="selected_asset")
    asset_price    = price_data[selected_asset]
    asset_returns  = returns[selected_asset]
    asset_weight   = float(weights_series.get(selected_asset, 0))
    asset_stats    = get_asset_key_stats(selected_asset, asset_price, asset_returns, asset_weight)

    section_header(f"{selected_asset} — Key Stats")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Portfolio Weight",  f"{asset_stats['weight']:.2%}")
    c2.metric("Annual Return",     f"{asset_stats['annual_return']:.2%}")
    c3.metric("Annual Volatility", f"{asset_stats['volatility']:.2%}")
    c4.metric("Sharpe Ratio",      f"{asset_stats['sharpe_ratio']:.2f}")

    section_header("Price History")
    tf = st.radio("Timeframe",["1M","3M","6M","1Y","3Y","5Y"],horizontal=True,
                  key="asset_price_tf",label_visibility="collapsed")
    fig = px.line(slice_tf(asset_price, tf), color_discrete_sequence=["#3b82f6"])
    fig.update_traces(line=dict(width=1.8))
    quick_chart(fig)

    # Slice returns to the same timeframe so all charts stay in sync
    asset_returns_tf = slice_tf(asset_returns, tf)

    cA, cB = st.columns(2)
    with cA:
        section_header("Rolling Volatility (60D)")
        quick_chart(px.line(slice_tf(compute_rolling_volatility(asset_returns, 60), tf),
                            color_discrete_sequence=["#f59e0b"]), 280)
    with cB:
        section_header("Rolling Correlation with Portfolio (60D)")
        quick_chart(px.line(slice_tf(compute_rolling_correlation(asset_returns, portfolio_returns, 60), tf),
                            color_discrete_sequence=["#8b5cf6"]), 280)

    section_header("Drawdown")
    fig = px.area(compute_asset_drawdown(asset_returns_tf), color_discrete_sequence=["#ef4444"])
    fig.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,0.1)")
    quick_chart(fig, 260)

    section_header("Fundamental Metrics")
    fund_df = get_asset_fundamental_table(selected_asset)
    fund_df["Metric"] = fund_df["Metric"].astype(str)
    _no_data = "Category" not in fund_df.columns or (fund_df["Metric"] == "No data available").any()
    if _no_data:
        empty_state("📊","Fundamental data unavailable",f"Could not retrieve ratios for {selected_asset}")
    else:
        prof = fund_df[fund_df["Category"]=="Profitability"].drop(columns="Category")
        liq  = fund_df[fund_df["Category"]=="Liquidity"].drop(columns="Category")
        val  = fund_df[fund_df["Category"]=="Valuation"].drop(columns="Category")

        c1, c2 = st.columns(2)
        with c1:
            section_header("Profitability")
            if not prof.empty:
                st.dataframe(prof.set_index("Metric"), use_container_width=True)
            else:
                st.caption("No data")
        with c2:
            section_header("Valuation")
            if not val.empty:
                st.dataframe(val.set_index("Metric"), use_container_width=True)
            else:
                st.caption("No data")
        section_header("Liquidity & Solvency")
        if not liq.empty:
            st.dataframe(liq.set_index("Metric"), use_container_width=True)
        else:
            st.caption("No data")

    # ── News Feed ──────────────────────────────────────────
    section_header(f"Latest News — {selected_asset}")
    try:
        _news_raw = yf.Ticker(selected_asset).news
        if _news_raw:
            # Filter to content items only (type == "STORY" or no type key)
            _stories = [
                n for n in _news_raw
                if isinstance(n, dict) and (
                    n.get("type", "STORY") == "STORY" or "title" in n or
                    (isinstance(n.get("content"), dict) and n["content"].get("title"))
                )
            ][:8]

            if _stories:
                for _item in _stories:
                    # yfinance v0.2.x returns nested content dict; older versions use flat keys
                    _content = _item.get("content", _item)
                    _title   = _content.get("title",       _item.get("title",       "No title"))
                    _source  = _content.get("provider", {}).get("displayName", "") or _item.get("publisher", "")
                    _url     = _content.get("canonicalUrl", {}).get("url", "") or _item.get("link", "#")
                    _ts      = _content.get("pubDate", "") or _item.get("providerPublishTime", "")

                    # Format timestamp
                    if isinstance(_ts, (int, float)):
                        import datetime as _dt
                        _ts = _dt.datetime.utcfromtimestamp(_ts).strftime("%d %b %Y, %H:%M UTC")
                    elif isinstance(_ts, str) and _ts:
                        try:
                            import dateutil.parser as _dp
                            _ts = _dp.parse(_ts).strftime("%d %b %Y, %H:%M UTC")
                        except Exception:
                            pass

                    st.markdown(f"""
                    <div style="padding:12px 16px;margin-bottom:8px;border-radius:var(--radius);
                        background:var(--bg-surface);border:1px solid var(--border);
                        border-left:3px solid var(--accent);">
                      <div style="font-size:13px;font-weight:600;color:var(--text-primary);
                          margin-bottom:4px;line-height:1.4;">
                        <a href="{_url}" target="_blank"
                           style="color:var(--text-primary);text-decoration:none;">
                          {_title}
                        </a>
                      </div>
                      <div style="font-size:11px;color:var(--text-muted);">
                        {_source}{' · ' + str(_ts) if _ts else ''}
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                empty_state("📰", "No recent news", f"No news stories found for {selected_asset}")
        else:
            empty_state("📰", "No recent news", f"No news stories found for {selected_asset}")
    except Exception as _e:
        empty_state("📰", "News unavailable", f"Could not fetch news for {selected_asset}")


# ── TAB 6: ENHANCEMENT ─────────────────────────────────────
with tab6:
    st.markdown("""
    <div style="padding:14px 18px;margin-bottom:24px;border-radius:var(--radius);
        background:var(--bg-surface);border:1px solid var(--border);border-left:3px solid var(--accent);">
      <div style="font-size:13px;font-weight:500;color:var(--text-primary);margin-bottom:4px;">
          Sector-Wise Enhancement Screener</div>
      <div style="font-size:12px;color:var(--text-muted);">
          Top performing sectors and their best stocks ranked by 6 &amp; 12-month returns.
          Includes PE ratios and ROE. Refreshed every hour.</div>
    </div>""", unsafe_allow_html=True)

    section_header("3-Month Relative Performance — Current Holdings")
    try:
        with st.spinner("Computing 3M relative performance…"):
            pm_df = cached_3m_relative_performance(tickers_tuple)

        def _rule_engine(x):
            if pd.isna(x): return "No Data"
            if x < -0.10:  return "Sell"
            if x > 0.20:   return "Buy"
            return "Hold"

        pm_df["Action"] = pm_df["Relative Performance"].apply(_rule_engine)
        pm_df = pm_df.sort_values("Relative Performance", ascending=False).reset_index(drop=True)
        pm_df.index += 1

        c1, c2, c3 = st.columns(3)
        c1.metric("Buy Signals",    (pm_df["Action"] == "Buy").sum())
        c2.metric("Sell Signals",   (pm_df["Action"] == "Sell").sum())
        c3.metric("Hold Positions", (pm_df["Action"] == "Hold").sum())
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.dataframe(
            style_pl(pm_df, ["Relative Performance"]).format({
                "3M Return":            "{:.2%}",
                "Benchmark 3M":         "{:.2%}",
                "Relative Performance": "{:.2%}",
            }),
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"3M relative performance computation failed: {e}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.spinner("Analyzing sectors — may take ~15s on first load…"):
        sector_recs = cached_sector_recommendations()

    if not sector_recs:
        empty_state("🔍","No sector opportunities identified","Try again later")
    else:
        section_header("Top Sectors with Best Performers")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        def _fmt(x, fn):
            try: return fn(float(x)) if pd.notna(float(x)) else "N/A"
            except (ValueError, TypeError): return "N/A"

        for sector_name, sdf in sector_recs.items():
            if sdf.empty: continue
            avg6, avg12 = sdf["6M Return"].mean(), sdf["12M Return"].mean()
            with st.expander(f"📊 {sector_name} — 6M: {avg6:+.1%} | 12M: {avg12:+.1%}", expanded=False):
                disp   = sdf[["Ticker","Current Price","6M Return","12M Return","PE Ratio","ROE","Score"]].copy()
                styled = style_pl(disp, ["6M Return","12M Return"]).format({
                    "Current Price": lambda x: _fmt(x, lambda v: f"{_currency}{v:,.2f}"),
                    "6M Return":     lambda x: _fmt(x, lambda v: f"{v:.2%}"),
                    "12M Return":    lambda x: _fmt(x, lambda v: f"{v:.2%}"),
                    "PE Ratio":      lambda x: _fmt(x, lambda v: f"{v:.2f}"),
                    "ROE":           lambda x: _fmt(x, lambda v: f"{v:.2%}"),
                    "Score":         lambda x: _fmt(x, lambda v: f"{v:.4f}"),
                })
                st.dataframe(styled, use_container_width=True)
                cA,cB,cC,cD = st.columns(4)
                cA.metric("Avg 6M Return",  f"{avg6:.2%}")
                cB.metric("Avg 12M Return", f"{avg12:.2%}")
                cC.metric("Avg PE Ratio",   f"{sdf['PE Ratio'].mean():.2f}")
                cD.metric("Avg ROE",        f"{sdf['ROE'].mean():.2%}")