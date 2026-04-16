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

from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from io import BytesIO

from config import RISK_PROFILES, TRADING_DAYS, DEFAULT_TRANSACTION_COST
from data_engine import (
    load_and_validate_csv, compute_returns,
    aggregate_holdings, compute_xirr, compute_pl_summary,
)
from risk_engine import (generate_risk_summary, rolling_volatility, rolling_correlation,
    compute_drawdown_series, var_cvar_summary, sector_concentration,
    asset_type_concentration, effective_n, compute_liquidity_risk, run_stress_tests)
from optimizer import (
    optimize_portfolio, simulate_efficient_frontier, portfolio_performance,
    OPTIMIZERS,
)
from analytics import portfolio_health_score
from enhancement_engine import (
    compute_portfolio_3m_relative_performance,
    generate_enhancement_recommendations,
    generate_sector_wise_recommendations,
)
from asset_analytics_engine import (
    get_asset_key_stats, compute_rolling_volatility, compute_rolling_correlation,
    compute_asset_drawdown, get_asset_fundamental_table, get_dividend_data,
)
from performance_engine import get_performance_metrics, get_period_returns, get_rolling_metrics, compute_capture_ratios, compute_sector_contribution, compute_brinson_attribution
from rebalance_engine import simulate_cash_injection, simulate_trade
from external_apis import (
    finnhub_earnings_surprises, finnhub_recommendations,
    finnhub_insider_sentiment, finnhub_portfolio_consensus,
    finnhub_stock_metrics, finnhub_company_profile, finnhub_peers, finnhub_company_news,
    fred_macro_snapshot, india_macro_snapshot, yfinance_news,
    sec_insider_transactions,
)

# ── Page config ────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Portfolio Analyser", page_icon="📈")

# ── Plotly template ────────────────────────────────────────
pio.templates["portfolio_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#09080f", plot_bgcolor="#09080f",
        font=dict(family="Inter, sans-serif", color="#8b7fc0", size=12),
        colorway=["#8b5cf6","#22c55e","#f59e0b","#ef4444","#06b6d4","#ec4899","#f97316","#84cc16"],
        xaxis=dict(gridcolor="#1a2744", linecolor="#1a2744", zerolinecolor="#1a2744", tickfont=dict(color="#64748b")),
        yaxis=dict(gridcolor="#1a2744", linecolor="#1a2744", zerolinecolor="#1a2744", tickfont=dict(color="#64748b")),
        legend=dict(bgcolor="rgba(11,17,32,0.8)", bordercolor="#1a2744", borderwidth=1),
        title=dict(font=dict(color="#e2e8f0", size=14)),
    )
)
px.defaults.template = "portfolio_dark"


# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* =========================================================
   DESIGN TOKENS
========================================================= */

:root {
  /* Surfaces — neutral near-black */
  --bg-base:       #09080f;
  --bg-surface:    #0e0c1a;
  --bg-elevated:   #14111f;
  --bg-card:       #0d0b14;

  /* Borders */
  --border:        #1f1c2e;
  --border-subtle: #17141f;

  /* Accent — PURPLE (replaces blue) */
  --accent:        #8b5cf6;
  --accent-dim:    #6d28d9;
  --accent-glow:   rgba(139,92,246,0.24);
  --accent-glow2:  rgba(139,92,246,0.09);

  /* Semantic */
  --positive:      #22c55e;
  --positive-dim:  #15803d;
  --positive-glow: rgba(34,197,94,0.18);
  --negative:      #ef4444;
  --negative-glow: rgba(239,68,68,0.18);
  --warning:       #f59e0b;
  --warning-glow:  rgba(245,158,11,0.18);
  --purple:        #8b5cf6;
  --purple-glow:   rgba(139,92,246,0.18);
  --cyan:          #06b6d4;
  --cyan-glow:     rgba(6,182,212,0.18);
  --pink:          #ec4899;
  --pink-glow:     rgba(236,72,153,0.18);

  /* Typography — lavender-tinted */
  --text-primary:  #ede9fe;
  --text-secondary:#8b7fc0;
  --text-muted:    #4a4066;

  /* Shape */
  --radius:        12px;
  --radius-sm:     8px;
  --radius-xs:     5px;

  /* Elevation */
  --shadow:        0 4px 24px rgba(0,0,0,0.6);
  --shadow-lg:     0 10px 48px rgba(0,0,0,0.7);
  --shadow-accent: 0 8px 32px rgba(139,92,246,0.22);

  /* Module identity colours */
  --tab-overview: #8b5cf6;
  --tab-risk:     #ef4444;
  --tab-optim:    #f59e0b;
  --tab-perf:     #22c55e;
  --tab-asset:    #06b6d4;
  --tab-enh:      #ec4899;
}

/* =========================================================
   AURORA BACKGROUND
========================================================= */

body {
  background-color: #09080f !important;
}

body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse at 15% 20%, rgba(139,92,246,0.13) 0%, transparent 55%),
    radial-gradient(ellipse at 85% 75%, rgba(109,40,217,0.09) 0%, transparent 50%),
    radial-gradient(ellipse at 65% 5%,  rgba(236,72,153,0.05)  0%, transparent 40%);
  pointer-events: none;
  z-index: -1;
  transform: translateZ(0);
  will-change: transform;
}

body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(148,163,184,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,0.025) 1px, transparent 1px);
  background-size: 28px 28px;
  pointer-events: none;
  z-index: -1;
  transform: translateZ(0);
  will-change: transform;
}

.main, .block-container {
  background: transparent !important;
  padding-top: 1.25rem !important;
  max-width: 100% !important;
}

/* =========================================================
   GLOBAL RESET
========================================================= */

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
  font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
  font-size: 13px;
  color: var(--text-primary);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

h1, h2, h3, h4 {
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: -0.025em !important;
  color: var(--text-primary) !important;
}
h2 { font-size: 1rem !important; margin-bottom: 0.6rem !important; }
h3 { font-size: 0.875rem !important; }

p, li { color: var(--text-secondary); font-size: 13px; }
code { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }
  50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* =========================================================
   SIDEBAR — ICON RAIL WITH HOVER EXPAND
========================================================= */

/* Sidebar: fixed width, page shifts naturally */
[data-testid="stSidebar"] {
  background: #0e0c16 !important;
  border-right: 1px solid rgba(139,92,246,0.14) !important;
  width: 220px !important;
  min-width: 220px !important;
  max-width: 220px !important;
}

[data-testid="stSidebarCollapseButton"],
button[data-testid="collapsedControl"] {
  display: none !important;
}

[data-testid="stSidebarContent"] {
  padding: 12px 8px !important;
}

/* Sidebar section labels */
.sidebar-group-label {
  font-size: 9px !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  padding: 14px 0 6px 4px !important;
  display: block !important;
  white-space: nowrap !important;
  overflow: hidden !important;
}

.sidebar-divider {
  border: none !important;
  border-top: 1px solid var(--border-subtle) !important;
  margin: 6px 0 10px 0 !important;
}

/* Nav radio styled as icon items */

/* Hide the group label ("Navigation") — Streamlit renders it as stWidgetLabel */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stRadio"] > label {
  display: none !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
  display: flex !important;
  flex-direction: column !important;
  gap: 2px !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
  display: flex !important;
  align-items: center !important;
  width: 100% !important;
  height: 38px !important;
  min-height: 38px !important;
  max-height: 38px !important;
  padding: 0 10px !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: background 0.15s ease, color 0.15s ease !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  font-size: 13px !important;
  color: rgba(148,163,184,0.6) !important;
  font-weight: 500 !important;
  border: 1px solid transparent !important;
  box-sizing: border-box !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background: rgba(139,92,246,0.10) !important;
  color: rgba(237,233,254,0.85) !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background: rgba(139,92,246,0.18) !important;
  color: #ede9fe !important;
  font-weight: 600 !important;
  border: 1px solid rgba(139,92,246,0.28) !important;
}

/* Hide the native input AND the custom visual radio circle (the div before the text) */
[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] {
  display: none !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
  display: none !important;
}

/* =========================================================
   METRIC CARDS
========================================================= */

div[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid rgba(148,163,184,0.09) !important;
  border-top: 2px solid var(--accent) !important;
  border-radius: var(--radius) !important;
  padding: 16px 18px 14px !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
  box-shadow: var(--shadow) !important;
  position: relative !important;
  overflow: hidden !important;
}

div[data-testid="stMetric"]::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(160deg, var(--accent-glow2) 0%, transparent 55%);
  pointer-events: none;
  border-radius: var(--radius);
}

div[data-testid="stMetric"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-accent) !important;
}

div[data-testid="stMetricValue"] {
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.04em !important;
  line-height: 1.1 !important;
}

div[data-testid="stMetricLabel"] > div {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.11em !important;
  color: var(--text-muted) !important;
  margin-bottom: 4px !important;
}

div[data-testid="stMetricDelta"] { font-size: 12px !important; font-weight: 600 !important; }

/* =========================================================
   TABLES & DATAFRAMES
========================================================= */

.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  box-shadow: var(--shadow) !important;
  font-size: 12px !important;
}

/* =========================================================
   ALERTS
========================================================= */

.stAlert {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  color: var(--text-secondary) !important;
  font-size: 12px !important;
}

/* =========================================================
   BUTTONS
========================================================= */

.stButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  padding: 10px 22px !important;
  letter-spacing: 0.03em !important;
  transition: all 0.18s ease !important;
  box-shadow: 0 4px 14px var(--accent-glow) !important;
}

.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px var(--accent-glow) !important;
}

.stDownloadButton > button {
  background: linear-gradient(135deg, var(--positive) 0%, var(--positive-dim) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 12px !important;
  padding: 10px 22px !important;
  box-shadow: 0 4px 14px var(--positive-glow) !important;
  transition: all 0.18s ease !important;
}

.stDownloadButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px var(--positive-glow) !important;
}

/* =========================================================
   INPUTS & SELECTS
========================================================= */

.stSelectbox > div > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-size: 13px !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-glow2) !important;
}

.stSlider > div > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-dim)) !important;
}

.stFileUploader { border-radius: var(--radius) !important; }

.streamlit-expanderHeader {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text-primary) !important;
  transition: all 0.15s ease !important;
  padding: 12px 16px !important;
}

.streamlit-expanderHeader:hover {
  border-color: var(--accent) !important;
  background: var(--bg-elevated) !important;
}

.streamlit-expanderContent {
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-sm) var(--radius-sm) !important;
  background: var(--bg-card) !important;
  padding: 16px !important;
}

hr {
  border: none !important;
  border-top: 1px solid var(--border-subtle) !important;
  margin: 1.25rem 0 !important;
}

/* Hide Streamlit header toolbar */
header[data-testid="stHeader"] {
  display: none !important;
}

</style>
""", unsafe_allow_html=True)
# ==========================================================
# UI HELPERS
# ==========================================================

def empty_state(icon, title, subtitle=""):
    sub_html = (f'<div style="font-size:11px;color:var(--text-muted);max-width:300px;'
                f'margin:0 auto;line-height:1.6;">{subtitle}</div>') if subtitle else ""
    st.markdown(f"""
    <div style="text-align:center;padding:60px 32px;border-radius:var(--radius);
        background:radial-gradient(ellipse at 50% 0%,var(--bg-elevated) 0%,var(--bg-surface) 70%);
        border:1px dashed var(--border);margin:8px 0;">
        <div style="font-size:42px;margin-bottom:18px;opacity:0.45;filter:grayscale(0.2);">{icon}</div>
        <div style="font-size:13px;font-weight:600;color:var(--text-secondary);
            margin-bottom:8px;letter-spacing:-0.01em;">{title}</div>
        {sub_html}
    </div>""", unsafe_allow_html=True)


def section_header(title, color="var(--accent)"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:28px 0 12px 0;">
      <div style="width:3px;height:16px;border-radius:2px;flex-shrink:0;
          background:linear-gradient(180deg,{color},{color}99);"></div>
      <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;
          text-transform:uppercase;color:var(--text-secondary);">{title}</div>
    </div>
    """, unsafe_allow_html=True)


def style_pl(df, pl_cols):
    def _colour(val):
        try:
            n = float(str(val).replace("%","").replace("$","").replace("₹","").replace(",",""))
            return ("color:#22c55e" if n>0 else "color:#ef4444") + ";font-weight:600"
        except (ValueError, TypeError): return ""
    return df.style.map(_colour, subset=pl_cols)


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


def card_header(title: str, colour: str = "var(--text-muted)") -> None:
    st.markdown(
        f'<div style="font-size:10px;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.08em;color:{colour};margin-bottom:12px;">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_chip(label: str, value: str, colour: str = "var(--text-primary)", sub: str = "") -> None:
    sub_html = (f'<div style="font-size:11px;color:var(--text-secondary);margin-top:2px;">{sub}</div>'
                if sub else "")
    st.markdown(f"""
    <div style="padding:16px 20px;background:var(--bg-card);border:1px solid var(--border);
         border-top:2px solid {colour};border-radius:var(--radius);margin-bottom:8px;">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
           letter-spacing:0.08em;color:var(--text-muted);margin-bottom:4px;">{label}</div>
      <div style="font-size:28px;font-weight:700;color:{colour};line-height:1;">{value}</div>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def stat_banner(items, accent="#3b82f6"):
    """Render a horizontal banner of (label, value) pairs.
    items: list of (label, value) or (label, value, colour) tuples.
    """
    cells = ""
    for i, item in enumerate(items):
        label, value = item[0], item[1]
        colour = item[2] if len(item) > 2 else "var(--text-primary)"
        divider = (f"<div style='width:1px;height:40px;background:rgba(255,255,255,0.07);"
                   f"flex-shrink:0;'></div>") if i > 0 else ""
        cells += (f"{divider}<div style='padding:0 28px;flex:1;text-align:center;'>"
                  f"<div style='font-size:9px;font-weight:700;letter-spacing:0.14em;"
                  f"text-transform:uppercase;color:{accent};margin-bottom:8px;"
                  f"opacity:0.9;'>{label}</div>"
                  f"<div style='font-size:26px;font-weight:700;color:{colour};"
                  f"letter-spacing:-0.04em;line-height:1;'>{value}</div>"
                  f"</div>")
    st.markdown(
        f"<div style='display:flex;align-items:center;"
        f"padding:22px 24px;border-radius:var(--radius);margin-bottom:24px;"
        f"background:linear-gradient(135deg,{accent}1a 0%,{accent}08 50%,transparent 100%);"
        f"border:1px solid {accent}35;border-left:3px solid {accent};"
        f"box-shadow:0 4px 32px {accent}14;'>{cells}</div>",
        unsafe_allow_html=True)


# ==========================================================
# PDF HELPERS
# ==========================================================

def _fig_to_image(fig, width_px, height_px, w_inch, h_inch):
    buf = BytesIO(fig.to_image(format="png", width=width_px, height=height_px))
    buf.seek(0)
    return RLImage(buf, width=w_inch*inch, height=h_inch*inch)


def generate_portfolio_pdf(df, risk_summary, weights_series, optimal_weights,
                           curr_ret, curr_vol, opt_ret, opt_vol, health_score,
                           opt_method="Max Sharpe", portfolio_returns=None,
                           benchmark_returns=None, enhancements=None, currency="$",
                           benchmark_name="Benchmark", pl_summary=None, portfolio_xirr=None):
    buffer = BytesIO()
    pdf    = SimpleDocTemplate(buffer, pagesize=letter,
                               rightMargin=0.5*inch, leftMargin=0.5*inch,
                               topMargin=0.6*inch,   bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    BLUE    = colors.HexColor('#1e40af')
    LBLUE   = colors.HexColor('#dbeafe')
    RED     = colors.HexColor('#dc2626')
    LRED    = colors.HexColor('#fee2e2')
    GREEN   = colors.HexColor('#16a34a')
    LGREEN  = colors.HexColor('#dcfce7')
    PURPLE  = colors.HexColor('#7c3aed')
    LPURPLE = colors.HexColor('#f5f3ff')
    ALTROW  = colors.HexColor('#f1f5f9')

    title_style = ParagraphStyle('CT', parent=styles['Heading1'], fontSize=22,
        textColor=BLUE, spaceAfter=4, alignment=1, fontName='Helvetica-Bold')
    sub_style   = ParagraphStyle('DS', parent=styles['Normal'], fontSize=9,
        textColor=colors.HexColor('#64748b'), alignment=1, spaceAfter=2)
    h_style     = ParagraphStyle('CH', parent=styles['Heading2'], fontSize=12,
        textColor=BLUE, spaceAfter=8, spaceBefore=14, fontName='Helvetica-Bold',
        borderPad=4, backColor=LBLUE, leading=18)
    note_style  = ParagraphStyle('NT', parent=styles['Normal'], fontSize=8,
        textColor=colors.HexColor('#64748b'), spaceAfter=6, leftIndent=4)

    def _tbl(data, col_widths, hdr_bg, alt_bg=ALTROW, num_cols_right=None):
        t = Table(data, colWidths=col_widths, repeatRows=1)
        style_cmds = [
            ('BACKGROUND',    (0,0), (-1,0),  hdr_bg),
            ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
            ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
            ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
            ('FONTSIZE',      (0,0), (-1,0),  9),
            ('FONTSIZE',      (0,1), (-1,-1), 8),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
            ('RIGHTPADDING',  (0,0), (-1,-1), 6),
            ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, alt_bg]),
            ('ALIGN',         (0,0), (0,-1),  'LEFT'),
            ('ALIGN',         (1,0), (-1,-1), 'RIGHT'),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ]
        # Colour positive/negative in Change columns
        if num_cols_right:
            for row_idx in range(1, len(data)):
                for col_idx in range(len(data[0]) - num_cols_right, len(data[0])):
                    val = str(data[row_idx][col_idx])
                    if val.startswith('+'):
                        style_cmds.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), GREEN))
                    elif val.startswith('-'):
                        style_cmds.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), RED))
        t.setStyle(TableStyle(style_cmds))
        return t

    def _h(title):
        story.append(Spacer(1, 0.08*inch))
        story.append(Paragraph(title, h_style))

    # ── Cover ─────────────────────────────────────────────────
    story = [
        Spacer(1, 0.1*inch),
        Paragraph("PORTFOLIO ANALYSIS REPORT", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", sub_style),
        Paragraph(f"Benchmark: {benchmark_name}  ·  Method: {opt_method}", sub_style),
        Spacer(1, 0.2*inch),
    ]

    # ── Portfolio Summary ─────────────────────────────────────
    total_value    = df['Market Value'].sum()
    _hs_display   = f"{min(health_score, 100.0):.1f} / 100"
    _xirr_display = f"{portfolio_xirr:.2%}" if portfolio_xirr and not np.isnan(float(portfolio_xirr or float('nan'))) else "N/A"
    _unreal       = (pl_summary or {}).get("unrealised_pl",  0)
    _real         = (pl_summary or {}).get("realised_pl",    0)
    _unreal_pct   = (pl_summary or {}).get("unrealised_pct", 0)

    summary_data = [
        ["Metric", "Value"],
        ["Total Portfolio Value",   f"{currency}{total_value:,.2f}"],
        ["Amount Invested",         f"{currency}{pl_summary.get('total_cost',0):,.2f}" if pl_summary else "N/A"],
        ["Unrealised P/L",          f"{currency}{_unreal:,.2f}  ({_unreal_pct:+.2%})"],
        ["Realised P/L",            f"{currency}{_real:,.2f}"],
        ["XIRR",                    _xirr_display],
        ["Number of Holdings",      str(len(df))],
        ["Portfolio Health Score",  _hs_display],
        ["Annual Return",           f"{curr_ret:.2%}"],
        ["Annual Volatility",       f"{curr_vol:.2%}"],
    ]
    _h("Portfolio Summary")
    story.append(_tbl(summary_data, [3.2*inch, 2.3*inch], BLUE))
    story.append(Spacer(1, 0.1*inch))

    # ── Risk Metrics ──────────────────────────────────────────
    _h("Risk Metrics")
    risk_data = [
        ["Metric", "Value"],
        ["Sharpe Ratio",     f"{risk_summary.get('Sharpe Ratio',  0):.3f}"],
        ["Sortino Ratio",    f"{risk_summary.get('Sortino Ratio', 0):.3f}"],
        ["Max Drawdown",     f"{risk_summary.get('Max Drawdown',  0):.2%}"],
        ["Beta",             f"{risk_summary.get('Beta',          0):.3f}"],
        ["VaR 95%",          f"{risk_summary.get('VaR 95%',       0):.2%}"],
        ["CVaR 95%",         f"{risk_summary.get('CVaR 95%',      0):.2%}"],
        ["Tracking Error",   f"{risk_summary.get('Tracking Error',0):.2%}"],
        ["Information Ratio",f"{risk_summary.get('Information Ratio',0):.3f}"],
    ]
    story.append(_tbl(risk_data, [3.2*inch, 2.3*inch], RED, LRED))
    story.append(Spacer(1, 0.1*inch))

    # ── Holdings Table ────────────────────────────────────────
    story.append(PageBreak())
    _h("Current Holdings")
    try:
        hd   = df[["Ticker","Quantity","Avg Cost","Current Price","Market Value",
                   "Current Weight","Unrealised P/L","P/L %"]].copy()
        rows = [["Ticker","Qty","Avg Cost","Price","Value","Weight","Unreal P/L","P/L %"]]
        for _, r in hd.iterrows():
            try:
                pl_val = float(r.get("Unrealised P/L", 0))
                pl_pct = float(r.get("P/L %", 0))
                rows.append([
                    str(r["Ticker"]),
                    f"{float(r['Quantity']):,.0f}",
                    f"{currency}{float(r['Avg Cost']):,.2f}",
                    f"{currency}{float(r['Current Price']):,.2f}",
                    f"{currency}{float(r['Market Value']):,.2f}",
                    f"{float(r['Current Weight']):.2%}",
                    f"{'+' if pl_val>=0 else ''}{currency}{pl_val:,.2f}",
                    f"{pl_pct:+.2%}",
                ])
            except (ValueError, TypeError):
                continue
    except Exception:
        rows = [["Ticker","Qty","Avg Cost","Price","Value","Weight","Unreal P/L","P/L %"]]
    story.append(_tbl(rows,
        [0.75*inch,0.55*inch,0.85*inch,0.85*inch,0.95*inch,0.65*inch,0.95*inch,0.65*inch],
        BLUE, num_cols_right=2))
    story.append(Spacer(1, 0.1*inch))

    # ── Performance Comparison (text-based, no kaleido needed) ──
    story.append(PageBreak())
    _h("Performance Comparison")
    if portfolio_returns is not None and benchmark_returns is not None:
        try:
            aligned = pd.concat([portfolio_returns.rename("Portfolio"),
                                  benchmark_returns.rename("Benchmark")], axis=1).dropna()
            if not aligned.empty:
                # Try chart first, fall back to period returns table
                chart_ok = False
                try:
                    cp   = (1+aligned["Portfolio"]).cumprod()-1
                    cb   = (1+aligned["Benchmark"]).cumprod()-1
                    fig2 = go.Figure([
                        go.Scatter(x=cp.index, y=cp.values*100, name='Portfolio',
                                   line=dict(color='#2563eb', width=2.5)),
                        go.Scatter(x=cb.index, y=cb.values*100, name=benchmark_name,
                                   line=dict(color='#94a3b8', width=2, dash='dash')),
                    ])
                    fig2.update_layout(
                        height=300, margin=dict(l=40,r=20,t=20,b=40),
                        plot_bgcolor='#f8fafc', paper_bgcolor='white',
                        font=dict(size=9, family='Helvetica'),
                        yaxis=dict(ticksuffix="%", gridcolor='#e2e8f0'),
                        xaxis=dict(gridcolor='#e2e8f0'),
                        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
                    )
                    story.append(_fig_to_image(fig2, 650, 300, 6.5, 3))
                    chart_ok = True
                except Exception:
                    pass

                # Period returns table always shown
                periods = {"1M":21,"3M":63,"6M":126,"1Y":252,"3Y":756}
                perf_rows = [["Period","Portfolio","Benchmark","Excess Return"]]
                for label, days in periods.items():
                    if len(aligned) >= days:
                        p_ret = float((1+aligned["Portfolio"].iloc[-days:]).prod()-1)
                        b_ret = float((1+aligned["Benchmark"].iloc[-days:]).prod()-1)
                        exc   = p_ret - b_ret
                        perf_rows.append([
                            label,
                            f"{p_ret:+.2%}",
                            f"{b_ret:+.2%}",
                            f"{exc:+.2%}",
                        ])
                if len(perf_rows) > 1:
                    story.append(Spacer(1, 0.1*inch))
                    story.append(_tbl(perf_rows,
                        [1.2*inch,1.5*inch,1.5*inch,1.5*inch], BLUE, num_cols_right=1))
        except Exception:
            story.append(Paragraph("<i>Performance data unavailable</i>", note_style))
    story.append(Spacer(1, 0.1*inch))

    # ── Optimisation Results ──────────────────────────────────
    if optimal_weights is not None:
        _h(f"Optimisation Results — {opt_method}")
        opt_summary = [
            ["Metric",      "Current",         "Optimised",       "Change"],
            ["Return",      f"{curr_ret:.2%}",  f"{opt_ret:.2%}",  f"{opt_ret-curr_ret:+.2%}"],
            ["Volatility",  f"{curr_vol:.2%}",  f"{opt_vol:.2%}",  f"{opt_vol-curr_vol:+.2%}"],
        ]
        story.append(_tbl(opt_summary,
            [1.8*inch,1.4*inch,1.4*inch,1.4*inch], GREEN, LGREEN, num_cols_right=1))
        story.append(Spacer(1, 0.1*inch))

        try:
            wt_rows = [["Ticker","Current","Optimised","Change"]]
            for ticker in sorted(optimal_weights.index):
                curr_w = float(weights_series.get(ticker, 0))
                opt_w  = float(optimal_weights[ticker])
                wt_rows.append([ticker, f"{curr_w:.2%}", f"{opt_w:.2%}", f"{opt_w-curr_w:+.2%}"])
            story.append(_tbl(wt_rows,
                [1.8*inch,1.4*inch,1.4*inch,1.4*inch], BLUE, num_cols_right=1))
        except Exception:
            pass

    # ── Enhancement Recommendations ───────────────────────────
    if enhancements is not None and not enhancements.empty:
        story.append(PageBreak())
        _h("Enhancement Recommendations")
        # Detect alpha column name dynamically
        alpha_col = next((c for c in enhancements.columns if "alpha" in c.lower()), None)
        rows = [["Ticker","Price","6M Return","1Y Return","Alpha","Score"]]
        for _, r in enhancements.head(10).iterrows():
            try:
                alpha_val = float(r.get(alpha_col, 0)) if alpha_col else 0.0
                rows.append([
                    str(r.get("Ticker","N/A")),
                    f"{currency}{float(r.get('Current Price',0)):.2f}",
                    f"{float(r.get('6M Return',0)):.2%}",
                    f"{float(r.get('12M Return',0)):.2%}",
                    f"{alpha_val:.2%}",
                    f"{float(r.get('Score',0)):.3f}",
                ])
            except (ValueError, TypeError):
                continue
        story.append(_tbl(rows,
            [0.9*inch,0.9*inch,0.9*inch,0.9*inch,0.9*inch,0.8*inch], PURPLE, LPURPLE))
        story.append(Paragraph(
            "* Alpha measured against selected benchmark over 12 months.", note_style))

    pdf.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ==========================================================
# PAGE HEADER
# ==========================================================

st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
    padding:22px 28px;border-radius:16px;margin:28px 0 28px 0;
    background:linear-gradient(135deg,var(--bg-elevated) 0%,var(--bg-card) 60%,var(--bg-base) 100%);
    border:1px solid var(--border);box-shadow:var(--shadow-lg);">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="width:52px;height:52px;border-radius:14px;flex-shrink:0;position:relative;
        background:linear-gradient(135deg,#6d28d9 0%,#8b5cf6 55%,#a78bfa 100%);
        display:flex;align-items:center;justify-content:center;font-size:24px;
        box-shadow:0 6px 28px rgba(139,92,246,0.55);">
      📈
      <div style="position:absolute;inset:0;border-radius:14px;
          background:linear-gradient(135deg,rgba(255,255,255,0.18),transparent);"></div>
    </div>
    <div>
      <div style="font-size:22px;font-weight:700;color:var(--text-primary);
          letter-spacing:-0.04em;line-height:1.15;">Portfolio Analyser</div>
      <div style="display:flex;align-items:center;gap:7px;margin-top:5px;">
        <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;
            box-shadow:0 0 8px #22c55e;animation:pulse-dot 2.4s ease-in-out infinite;"></div>
        <div style="font-size:10px;color:var(--accent);letter-spacing:0.14em;
            font-weight:700;text-transform:uppercase;">Institutional Dashboard</div>
      </div>
    </div>
  </div>
  <div style="text-align:right;line-height:1.7;">
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);">
      Real-time portfolio analytics</div>
    <div style="font-size:10px;color:var(--text-muted);opacity:0.6;">
      Powered by yfinance &amp; Streamlit</div>
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
        return (data.to_frame(name=tickers[0]) if isinstance(data, pd.Series) else data).dropna(how="all")
    except Exception: return pd.DataFrame()


def get_latest_prices(price_data, tickers):
    if "_price_cache" not in st.session_state: st.session_state._price_cache = {}
    latest = price_data.iloc[-1] if len(price_data) > 0 else pd.Series()
    def _get(t):
        p = latest.get(t) if t in latest.index else None
        if pd.notna(p) and p > 0:
            st.session_state._price_cache[t] = float(p); return float(p), False
        return (st.session_state._price_cache[t], True) if t in st.session_state._price_cache else (np.nan, False)
    return {t: _get(t) for t in tickers}


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

# Static sector map for popular NSE-listed Indian stocks
_IN_SECTOR_MAP = {
    # Technology / IT Services
    "TCS.NS":"Technology","INFY.NS":"Technology","WIPRO.NS":"Technology",
    "HCLTECH.NS":"Technology","TECHM.NS":"Technology","LTIM.NS":"Technology",
    "MPHASIS.NS":"Technology","PERSISTENT.NS":"Technology","COFORGE.NS":"Technology",
    "OFSS.NS":"Technology",
    # Financials — Banks
    "HDFCBANK.NS":"Financials","ICICIBANK.NS":"Financials","SBIN.NS":"Financials",
    "KOTAKBANK.NS":"Financials","AXISBANK.NS":"Financials","INDUSINDBK.NS":"Financials",
    "BANDHANBNK.NS":"Financials","FEDERALBNK.NS":"Financials","IDFCFIRSTB.NS":"Financials",
    "RBLBANK.NS":"Financials",
    # Financials — NBFCs & Insurance
    "BAJFINANCE.NS":"Financials","BAJAJFINSV.NS":"Financials","SBILIFE.NS":"Financials",
    "HDFCLIFE.NS":"Financials","ICICIPRULI.NS":"Financials","CHOLAFIN.NS":"Financials",
    "MUTHOOTFIN.NS":"Financials","PFC.NS":"Financials","RECLTD.NS":"Financials",
    # Consumer Discretionary
    "TATAMOTORS.NS":"Consumer Discretionary","M&M.NS":"Consumer Discretionary",
    "MARUTI.NS":"Consumer Discretionary","EICHERMOT.NS":"Consumer Discretionary",
    "HEROMOTOCO.NS":"Consumer Discretionary","BAJAJ-AUTO.NS":"Consumer Discretionary",
    "TITAN.NS":"Consumer Discretionary","TRENT.NS":"Consumer Discretionary",
    "NYKAA.NS":"Consumer Discretionary","ZOMATO.NS":"Consumer Discretionary",
    "PAYTM.NS":"Consumer Discretionary","DMART.NS":"Consumer Discretionary",
    # Consumer Staples
    "HINDUNILVR.NS":"Consumer Staples","ITC.NS":"Consumer Staples",
    "NESTLEIND.NS":"Consumer Staples","BRITANNIA.NS":"Consumer Staples",
    "DABUR.NS":"Consumer Staples","MARICO.NS":"Consumer Staples",
    "GODREJCP.NS":"Consumer Staples","COLPAL.NS":"Consumer Staples",
    "TATACONSUM.NS":"Consumer Staples","VBL.NS":"Consumer Staples",
    # Healthcare / Pharma
    "SUNPHARMA.NS":"Healthcare","DRREDDY.NS":"Healthcare","CIPLA.NS":"Healthcare",
    "DIVISLAB.NS":"Healthcare","APOLLOHOSP.NS":"Healthcare","LUPIN.NS":"Healthcare",
    "TORNTPHARM.NS":"Healthcare","BIOCON.NS":"Healthcare","AUROPHARMA.NS":"Healthcare",
    "MAXHEALTH.NS":"Healthcare",
    # Energy
    "RELIANCE.NS":"Energy","ONGC.NS":"Energy","BPCL.NS":"Energy",
    "IOC.NS":"Energy","GAIL.NS":"Energy","POWERGRID.NS":"Utilities",
    "NTPC.NS":"Utilities","TATAPOWER.NS":"Utilities","ADANIGREEN.NS":"Utilities",
    "ADANIPORTS.NS":"Industrials",
    # Industrials / Infrastructure
    "LT.NS":"Industrials","SIEMENS.NS":"Industrials","ABB.NS":"Industrials",
    "BHEL.NS":"Industrials","HAL.NS":"Industrials","BEL.NS":"Industrials",
    "IRFC.NS":"Industrials","IRCTC.NS":"Industrials","TIINDIA.NS":"Industrials",
    # Materials
    "TATASTEEL.NS":"Materials","JSWSTEEL.NS":"Materials","HINDALCO.NS":"Materials",
    "SAIL.NS":"Materials","VEDL.NS":"Materials","COALINDIA.NS":"Materials",
    "UPL.NS":"Materials","PIDILITIND.NS":"Materials","ASIANPAINT.NS":"Materials",
    "BERGEPAINT.NS":"Materials",
    # Conglomerates / Others
    "ADANIENT.NS":"Industrials","ULTRACEMCO.NS":"Materials","AMBUJACEM.NS":"Materials",
    "SHREECEM.NS":"Materials","GRASIM.NS":"Materials",
    # Indian ETFs / Indices
    "NIFTYBEES.NS":"ETF","JUNIORBEES.NS":"ETF","BANKBEES.NS":"ETF",
    "ITBEES.NS":"ETF","GOLDBEES.NS":"ETF","LIQUIDBEES.NS":"ETF",
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

        # Method 2: static sector maps (instant, no network)
        if t in _IN_SECTOR_MAP:
            _sec = _IN_SECTOR_MAP[t]
            _atype = "ETF" if _sec == "ETF" else "Equity"
            return t, t, _sec, _atype
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


@st.cache_data(show_spinner=False)
def cached_compute_returns(price_data):
    return compute_returns(price_data)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_enhancement_recommendations(market="US"): return generate_enhancement_recommendations(market=market)
@st.cache_data(ttl=3600, show_spinner=False)
def cached_sector_recommendations(market="US"): return generate_sector_wise_recommendations(top_sectors=5, stocks_per_sector=5, market=market)
@st.cache_data(show_spinner=False)
def cached_3m_relative_performance(tickers, benchmark): return compute_portfolio_3m_relative_performance(list(tickers), benchmark=benchmark)

# ── External API caches ─────────────────────────────────────
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_fred_macro(key):                  return fred_macro_snapshot(key)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_india_macro(key):                 return india_macro_snapshot(key)
@st.cache_data(ttl=1800,  show_spinner=False)
def cached_yf_news(ticker):                  return yfinance_news(ticker)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_finnhub_earnings(ticker, key):    return finnhub_earnings_surprises(ticker, key)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_finnhub_recs(ticker, key):        return finnhub_recommendations(ticker, key)
@st.cache_data(ttl=86400, show_spinner=False)
def cached_finnhub_insider(ticker, key):     return finnhub_insider_sentiment(ticker, key)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_finnhub_consensus(tickers, key):  return finnhub_portfolio_consensus(list(tickers), key)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_finnhub_metrics(ticker, key):     return finnhub_stock_metrics(ticker, key)
@st.cache_data(ttl=86400, show_spinner=False)
def cached_finnhub_profile(ticker, key):     return finnhub_company_profile(ticker, key)
@st.cache_data(ttl=86400, show_spinner=False)
def cached_finnhub_peers(ticker, key):       return finnhub_peers(ticker, key)
@st.cache_data(ttl=1800,  show_spinner=False)
def cached_finnhub_news(ticker, key):        return finnhub_company_news(ticker, key)
@st.cache_data(ttl=86400, show_spinner=False)
def cached_sec_insider(ticker):              return sec_insider_transactions(ticker)
@st.cache_data(ttl=3600,  show_spinner=False)
def cached_liquidity_risk(tickers, quantities, market_values):
    import pandas as pd
    holdings_df = pd.DataFrame({"Ticker": list(tickers), "Quantity": list(quantities), "Market Value": list(market_values)})
    return compute_liquidity_risk(holdings_df)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_stress_tests(tickers, weights):
    import pandas as pd
    weights_series = pd.Series(list(weights), index=list(tickers))
    return run_stress_tests(weights_series)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_dividend_data(ticker, quantity, current_price):
    return get_dividend_data(ticker, quantity, current_price)


# ==========================================================
# VOLATILITY REGIME
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_vol_regime(returns, window=60):
    rv = (returns.rolling(window).std() * np.sqrt(252)).dropna()
    if rv.empty: return "N/A", "#64748b", 0.0
    latest, pct = float(rv.iloc[-1]), float(rv.rank(pct=True).iloc[-1])
    return ("LOW VOL","#22c55e",latest) if pct<.33 else ("NORMAL VOL","#f59e0b",latest) if pct<.66 else ("HIGH VOL","#ef4444",latest)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:14px 4px 12px;border-bottom:1px solid var(--border-subtle);margin-bottom:8px;">
      <div style="font-size:20px;margin-bottom:4px;">📈</div>
      <div style="font-size:13px;font-weight:700;color:var(--text-primary);
          letter-spacing:-0.02em;white-space:nowrap;overflow:hidden;">Portfolio Analyser</div>
    </div>
    """, unsafe_allow_html=True)

    # ── NAVIGATION ─────────────────────────────────────────
    _active_module = st.radio(
        "Navigation",
        options=["overview", "risk", "optimization", "performance", "asset_analytics", "enhancement"],
        format_func=lambda x: {
            "overview":        "📊  Overview",
            "risk":            "⚠️  Risk",
            "optimization":    "🎯  Optimization",
            "performance":     "📈  Performance",
            "asset_analytics": "🔍  Asset Analytics",
            "enhancement":     "✨  Enhancement",
        }[x],
        key="_active_module",
        label_visibility="collapsed",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── DATA ───────────────────────────────────────────────
    st.markdown('<span class="sidebar-group-label">Portfolio</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", label_visibility="collapsed")

px.defaults.template = "portfolio_dark"

# ── API keys from .streamlit/secrets.toml ──────────────────
try:
    _FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") or "d6ljirpr01qrq6i31170d6ljirpr01qrq6i3117g"
    _FRED_KEY    = st.secrets.get("FRED_KEY",    "") or "1887e2a54d8ecf8a37c4df1799d4b3bb"
except Exception:
    _FINNHUB_KEY = "d6ljirpr01qrq6i31170d6ljirpr01qrq6i3117g"
    _FRED_KEY    = "1887e2a54d8ecf8a37c4df1799d4b3bb"

benchmark      = st.session_state.get("benchmark", "^GSPC")
_bm_display    = {"^GSPC": "S&P 500", "^NSEI": "NIFTY 50", "^CRSLDX": "NIFTY 500",
                   "^BSESN": "SENSEX", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}
benchmark_name = _bm_display.get(benchmark, benchmark)
risk_profile   = "Moderate"
max_weight_pct = st.session_state.get("max_weight_pct", 15.0)
max_weight     = max_weight_pct / 100
lookback       = "max"

if uploaded_file is None:
    st.session_state.pop("data_loaded",    None)
    st.session_state.pop("selected_asset", None)
    st.markdown("""
    <div style="max-width:660px;margin:48px auto 0;text-align:center;">

      <!-- Icon -->
      <div style="width:76px;height:76px;margin:0 auto 24px;border-radius:20px;
          background:linear-gradient(135deg,#6d28d9,#8b5cf6);
          display:flex;align-items:center;justify-content:center;font-size:36px;
          box-shadow:0 8px 36px rgba(139,92,246,0.45);position:relative;">
        📊
        <div style="position:absolute;inset:0;border-radius:20px;
            background:linear-gradient(135deg,rgba(255,255,255,0.18),transparent);"></div>
      </div>

      <!-- Title -->
      <div style="font-size:28px;font-weight:700;color:var(--text-primary);
          letter-spacing:-0.04em;margin-bottom:12px;line-height:1.2;">
        Institutional Portfolio Analytics</div>
      <div style="font-size:14px;color:var(--text-muted);margin-bottom:36px;
          line-height:1.75;max-width:460px;margin-left:auto;margin-right:auto;">
        Upload your trade history to unlock risk analytics, portfolio optimisation,
        performance attribution, and AI-driven enhancement recommendations.
      </div>

      <!-- CSV Format -->
      <div style="padding:20px 24px;background:var(--bg-surface);border:1px solid var(--border);
          border-radius:var(--radius);margin-bottom:24px;text-align:left;">
        <div style="font-size:10px;font-weight:700;color:var(--accent);text-transform:uppercase;
            letter-spacing:0.12em;margin-bottom:12px;">Required CSV Format</div>
        <div style="font-family:'JetBrains Mono','DM Mono',monospace;font-size:12px;
            color:var(--text-secondary);line-height:2;background:var(--bg-elevated);
            border-radius:var(--radius-sm);padding:14px 16px;border:1px solid var(--border-subtle);">
          <span style="color:var(--text-muted);">Ticker, Date, Action, Quantity, Price</span><br>
          AAPL, 2023-01-15, <span style="color:#22c55e;">Buy</span>, 10, 135.20<br>
          MSFT, 2023-02-10, <span style="color:#22c55e;">Buy</span>, 5, 252.75<br>
          AAPL, 2024-06-01, <span style="color:#ef4444;">Sell</span>, 3, 189.50
        </div>
        <div style="font-size:11px;color:var(--text-muted);margin-top:12px;line-height:1.6;">
          Indian stocks: append <strong style="color:var(--accent);">.NS</strong> (NSE) or
          <strong style="color:var(--accent);">.BO</strong> (BSE) to the ticker.
          Headers are flexible — column names are auto-detected.
        </div>
      </div>

      <!-- Feature pills -->
      <div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;">
        <div style="padding:8px 16px;background:var(--bg-surface);border:1px solid var(--border);
            border-radius:30px;font-size:12px;color:var(--text-secondary);font-weight:500;">
          ⚠️ Risk Analytics</div>
        <div style="padding:8px 16px;background:var(--bg-surface);border:1px solid var(--border);
            border-radius:30px;font-size:12px;color:var(--text-secondary);font-weight:500;">
          🎯 Portfolio Optimisation</div>
        <div style="padding:8px 16px;background:var(--bg-surface);border:1px solid var(--border);
            border-radius:30px;font-size:12px;color:var(--text-secondary);font-weight:500;">
          📈 Performance Attribution</div>
        <div style="padding:8px 16px;background:var(--bg-surface);border:1px solid var(--border);
            border-radius:30px;font-size:12px;color:var(--text-secondary);font-weight:500;">
          ✨ AI Screener</div>
        <div style="padding:8px 16px;background:var(--bg-surface);border:1px solid var(--border);
            border-radius:30px;font-size:12px;color:var(--text-secondary);font-weight:500;">
          📥 PDF Report</div>
      </div>

    </div>""", unsafe_allow_html=True)
    st.stop()

_file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
if st.session_state.get("_last_file_id") != _file_id:
    for _k in ("data_loaded","selected_asset","risk_summary","drawdown_series",
               "frontier","optimal_weights","_opt_key","_portfolio_cache","_benchmark_cache"):
        st.session_state.pop(_k, None)
    st.session_state["_last_file_id"] = _file_id
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
    for col in ("Ticker", "Date", "Action", "Quantity"):  # Price is auto-fetched by data_engine
        if col not in transactions.columns:
            st.error(f"'{col}' column missing."); st.stop()

    # Aggregate transactions → current open holdings
    df = aggregate_holdings(transactions)
    if df is None or df.empty:
        st.error("No open positions found after aggregating transactions."); st.stop()

    tickers       = df["Ticker"].unique().tolist()
    tickers_tuple = tuple(tickers)

    _market   = "IN" if sum(t.endswith((".NS",".BO")) for t in tickers) >= len(tickers)/2 else "US"
    _currency = "₹" if _market == "IN" else "$"
    _rf_rate  = 0.065 if _market == "IN" else 0.05

    with st.spinner("Fetching market data…"):
        price_data = cached_fetch_market_data(tickers_tuple, lookback)
    if price_data is None or price_data.empty:
        st.error("Unable to fetch price data."); st.stop()

    returns = cached_compute_returns(price_data)
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
        "pl_summary":      pl_summary,
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
# Safe defaults in case cache is partially populated
pl_summary, portfolio_xirr, benchmark_returns = {}, None, None
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
pl_summary       = _cache.get("pl_summary", {})
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
if not {"benchmark_returns","risk_summary","health_score"}.issubset(st.session_state.get("_benchmark_cache",{})):
    with st.spinner(f"Fetching benchmark data ({benchmark})…"):
        _bm_data = cached_fetch_market_data((benchmark,), lookback)
    _benchmark_returns = None
    if _bm_data is not None and not _bm_data.empty:
        _bm_ret = cached_compute_returns(_bm_data)
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
                enhancements_pdf = cached_enhancement_recommendations(_market)
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
                benchmark_name    = benchmark_name,
                pl_summary        = pl_summary,
                portfolio_xirr    = portfolio_xirr,
            )
        st.download_button(
            "📄 Click to Download",
            data=pdf_bytes,
            file_name=f"Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )


# ==========================================================
# MODULE ROUTER
# ==========================================================

_module = st.session_state.get("_active_module", "overview")


# ── MODULE: OVERVIEW ───────────────────────────────────────
if _module == "overview":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:var(--accent);text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Overview</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Portfolio Dashboard</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Stat row ────────────────────────────────────────────
    _unreal_pct = (unrealized_gain / amount_invested) if amount_invested > 0 else 0.0
    _xirr_display = f"{portfolio_xirr:.2%}" if portfolio_xirr and not np.isnan(portfolio_xirr) else "N/A"
    _max_dd = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    _s1, _s2, _s3, _s4 = st.columns(4)
    _s1.metric("Total Value",   f"{_currency}{total_value:,.0f}",
               f"Invested {_currency}{amount_invested:,.0f}", delta_color="off")
    _gain_delta = (f"-{_currency}{abs(unrealized_gain):,.0f}" if unrealized_gain < 0
                   else f"+{_currency}{unrealized_gain:,.0f}")
    _s2.metric("Total Return",  f"{_unreal_pct:+.2%}", _gain_delta)
    if portfolio_xirr and not np.isnan(portfolio_xirr):
        _s3.metric("Health Score", f"{health_score:.0f}/100", f"{portfolio_xirr:+.2%} XIRR")
    else:
        _s3.metric("Health Score", f"{health_score:.0f}/100", "XIRR N/A", delta_color="off")
    _s4.metric("Max Drawdown",  f"{_max_dd:.2%}")

    # ── Primary: Heatmap Grid ─────────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid var(--tab-overview);border-radius:var(--radius);'
        'padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("PORTFOLIO COMPOSITION", colour="var(--tab-overview)")

    _hm_df = df[["Ticker", "Market Value", "Current Weight", "Current Price",
                  "Unrealised P/L", "P/L %", "Sector"]].copy()
    _hm_df["Name"] = df["Name"]
    _hm_df = _hm_df.sort_values("Current Weight", ascending=False).reset_index(drop=True)

    def _pl_bg(pl_pct):
        """Return a background colour that blends from red through neutral to green."""
        _clamped = max(-0.5, min(0.5, pl_pct))
        if _clamped >= 0:
            _t = _clamped / 0.5
            _r = int(30 + (34 - 30) * _t)
            _g = int(30 + (60 - 30) * _t)
            _b = int(46 + (34 - 46) * _t)
            _border = f"rgba(34,197,94,{0.15 + _t * 0.35:.2f})"
        else:
            _t = abs(_clamped) / 0.5
            _r = int(30 + (60 - 30) * _t)
            _g = int(30 + (25 - 30) * _t)
            _b = int(46 + (35 - 46) * _t)
            _border = f"rgba(239,68,68,{0.15 + _t * 0.35:.2f})"
        return f"rgb({_r},{_g},{_b})", _border

    def _svg_spark(ticker, w=80, h=28):
        if ticker not in price_data.columns:
            return ""
        _p = price_data[ticker].dropna().tail(30)
        if len(_p) < 2:
            return ""
        _y = _p.values
        _mn, _mx = _y.min(), _y.max()
        _rng = _mx - _mn if _mx != _mn else 1
        _pts = []
        for _j, _v in enumerate(_y):
            _px = _j / (len(_y) - 1) * w
            _py = h - ((_v - _mn) / _rng) * (h - 4) - 2
            _pts.append(f"{_px:.1f},{_py:.1f}")
        _c = "#22c55e" if _y[-1] >= _y[0] else "#ef4444"
        return (
            f'<svg width="{w}" height="{h}" style="display:block;margin:6px auto 0;">'
            f'<polyline points="{" ".join(_pts)}" fill="none" '
            f'stroke="{_c}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
            f'</svg>'
        )

    # Build grid cards
    _cards = ""
    for _, _r in _hm_df.iterrows():
        _tk  = _r["Ticker"]
        _nm  = _r.get("Name") or _tk
        _pl  = _r["P/L %"]
        _plc = "#22c55e" if _pl >= 0 else "#ef4444"
        _pls = "+" if _pl >= 0 else ""
        _bg, _bdr = _pl_bg(_pl)
        _wt  = _r["Current Weight"]
        _spark = _svg_spark(_tk)

        _cards += f'''
        <div style="background:{_bg};border:1px solid {_bdr};border-radius:var(--radius-sm);
            padding:14px;display:flex;flex-direction:column;justify-content:space-between;
            min-height:140px;transition:transform 0.15s,box-shadow 0.15s;cursor:default;"
            onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 24px rgba(0,0,0,0.4)';"
            onmouseout="this.style.transform='none';this.style.boxShadow='none';">
          <div>
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
              <div style="font-size:14px;font-weight:700;color:#fff;letter-spacing:-0.01em;">{_tk}</div>
              <div style="font-size:13px;font-weight:700;color:{_plc};">{_pls}{_pl:.2%}</div>
            </div>
            <div style="font-size:10px;color:rgba(255,255,255,0.55);margin-top:2px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"
                title="{_nm}">{_nm}</div>
          </div>
          {_spark}
          <div style="display:flex;justify-content:space-between;align-items:flex-end;margin-top:8px;">
            <div>
              <div style="font-size:9px;color:rgba(255,255,255,0.4);text-transform:uppercase;
                  letter-spacing:0.06em;">Price</div>
              <div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.85);">
                {_currency}{_r["Current Price"]:,.2f}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:9px;color:rgba(255,255,255,0.4);text-transform:uppercase;
                  letter-spacing:0.06em;">Value</div>
              <div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.85);">
                {_currency}{_r["Market Value"]:,.0f}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:9px;color:rgba(255,255,255,0.4);text-transform:uppercase;
                  letter-spacing:0.06em;">Weight</div>
              <div style="font-size:12px;font-weight:600;color:rgba(255,255,255,0.85);">
                {_wt:.1%}</div>
            </div>
          </div>
        </div>'''

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));'
        f'gap:10px;">{_cards}</div>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Secondary grid: Sector Allocation | Asset Allocation ─
    _r2c1, _r2c2 = st.columns(2)

    with _r2c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid var(--tab-overview);border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("SECTOR ALLOCATION", colour="var(--tab-overview)")
        _sec_alloc = df.groupby("Sector")["Market Value"].sum().reset_index()
        if not _sec_alloc.empty:
            _sec_fig = px.pie(_sec_alloc, names="Sector", values="Market Value", hole=0.55)
            _sec_fig.update_traces(textfont_size=11, marker=dict(line=dict(color="#09080f", width=2)))
            _sec_fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                                   legend=dict(orientation="v", x=1.02))
            st.plotly_chart(_sec_fig, use_container_width=True)
        else:
            empty_state("🏭", "Sector data unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    with _r2c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid var(--tab-overview);border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("ASSET ALLOCATION", colour="var(--tab-overview)")
        _at = df.groupby("Asset Type")["Market Value"].sum().reset_index()
        if not _at.empty:
            _at_fig = px.pie(_at, names="Asset Type", values="Market Value", hole=0.55)
            _at_fig.update_traces(textfont_size=11, marker=dict(line=dict(color="#09080f", width=2)))
            _at_fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                                  legend=dict(orientation="v", x=1.02))
            st.plotly_chart(_at_fig, use_container_width=True)
        else:
            empty_state("📦", "Asset type data unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Holdings ────────────────────────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid var(--tab-overview);border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("HOLDINGS")
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
        "Quantity":      "{:,.0f}",
        "Avg Cost":      f"{_currency}{{:,.2f}}",
        "Current Price": f"{_currency}{{:,.2f}}",
        "Unrealised P/L":f"{_currency}{{:,.2f}}",
        "Current Weight":"{:.2%}",
        "P/L %":         "{:.2%}",
    })
    st.dataframe(styled, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── MODULE: RISK ───────────────────────────────────────────
elif _module == "risk":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:#ef4444;text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Risk</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Risk Management</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Timeframe selector ──────────────────────────────────
    _risk_tf = st.radio("Timeframe", ["1Y","3Y","5Y"], horizontal=True,
                        key="risk_tf_hero", label_visibility="collapsed")

    # Pre-compute values
    _vc_chip = var_cvar_summary(portfolio_returns)

    # ── Stat row ────────────────────────────────────────────
    _max_dd_risk = float(drawdown_series.min()) if not drawdown_series.empty else 0.0
    _rs1, _rs2, _rs3, _rs4 = st.columns(4)
    _rs1.metric("VaR 95% (Hist.)", f"{_vc_chip['hist_var_95']:.2%}", help="Daily loss at 95% confidence")
    _rs2.metric("Max Drawdown",    f"{_max_dd_risk:.2%}")
    _rs3.metric("Volatility (Ann.)",f"{risk_summary['Volatility']:.2%}")
    _rs4.metric("Sharpe Ratio",    f"{risk_summary['Sharpe Ratio']:.2f}")

    # ── Primary chart: full-width rolling volatility ────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("ROLLING 60-DAY VOLATILITY", colour="#ef4444")
    _vol_fig = px.line(
        slice_tf(rolling_volatility(portfolio_returns), _risk_tf),
        color_discrete_sequence=["#ef4444"],
    )
    _vol_fig.update_layout(height=360, margin=dict(l=0,r=0,t=0,b=0),
                           yaxis=dict(tickformat=".1%"), hovermode="x unified")
    st.plotly_chart(_vol_fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Risk Summary + Macro ───────────────────────
    _r2c1, _r2c2 = st.columns(2)

    with _r2c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("RISK SUMMARY", colour="#ef4444")
        pct_fields = {"Annual Return","Volatility","Max Drawdown","VaR 95%","CVaR 95%","Tracking Error","Correlation"}
        st.table(pd.DataFrame({k: f"{v:.2%}" if k in pct_fields else f"{v:.2f}"
                                for k,v in risk_summary.items()}.items(), columns=["Metric","Value"]))
        st.markdown('</div>', unsafe_allow_html=True)

    with _r2c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("MACRO ENVIRONMENT", colour="var(--warning)")
        if _market == "IN":
            _macro = cached_india_macro(_FRED_KEY)
        else:
            _macro = cached_fred_macro(_FRED_KEY) if _FRED_KEY else {}
        if _macro:
            _mcols = st.columns(len(_macro))
            for _col, (_mname, _md) in zip(_mcols, _macro.items()):
                _delta_str = f"{_md['delta']:+.3f}" if _md.get("delta") is not None else None
                _col.metric(_mname, _md["display"], delta=_delta_str,
                            help=f"As of {_md['date']}")
        else:
            empty_state("📊", "Macro data unavailable", "Could not fetch macro indicators")

        card_header("ROLLING BENCHMARK CORRELATION", colour="#ef4444")
        if benchmark_returns is not None:
            _corr_tf = st.radio("Timeframe",["1Y","3Y","5Y"],horizontal=True,
                                key="corr_tf",label_visibility="collapsed")
            quick_chart(px.line(slice_tf(rolling_correlation(portfolio_returns,benchmark_returns), _corr_tf),
                                color_discrete_sequence=["#f59e0b"]), 220)
        else:
            empty_state("📉","No benchmark data","Benchmark returns could not be fetched")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Stress Tests ────────────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("HISTORICAL STRESS TESTS", colour="#ef4444")
    with st.spinner("Running stress scenarios…"):
        _stress_df = cached_stress_tests(
            tickers_tuple,
            tuple(weights_series.reindex(tickers).fillna(0).values),
        )
    if not _stress_df.empty:
        _s_cols = st.columns(len(_stress_df))
        for _col, (_, _srow) in zip(_s_cols, _stress_df.iterrows()):
            def _fmt_pct(v):
                return f"{v:.2%}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "N/A"
            _col.markdown(f"""
            <div style="background:var(--bg-elevated);border:1px solid var(--border);
                border-top:2px solid #ef4444;border-radius:var(--radius);
                padding:16px;text-align:center;margin-bottom:8px;">
              <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:0.1em;color:var(--text-muted);margin-bottom:12px;">
                {_srow['Scenario']}</div>
              <div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;">
                {_srow['Period']}</div>
              <div style="margin-bottom:6px;">
                <div style="font-size:9px;color:var(--text-muted);">Total Return</div>
                <div style="font-size:16px;font-weight:700;color:{'var(--negative)' if _srow['Total Return'] is not None and _srow['Total Return'] < 0 else 'var(--positive)'};">
                  {_fmt_pct(_srow['Total Return'])}</div>
              </div>
              <div style="margin-bottom:6px;">
                <div style="font-size:9px;color:var(--text-muted);">Max Drawdown</div>
                <div style="font-size:14px;font-weight:600;color:var(--negative);">
                  {_fmt_pct(_srow['Max Drawdown'])}</div>
              </div>
              <div>
                <div style="font-size:9px;color:var(--text-muted);">Worst Day</div>
                <div style="font-size:14px;font-weight:600;color:var(--negative);">
                  {_fmt_pct(_srow['Worst Day'])}</div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        empty_state("📉", "Stress test data unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: VaR + Return Distribution | Liquidity ───────
    _r4c1, _r4c2 = st.columns(2)

    with _r4c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("VALUE AT RISK & EXPECTED SHORTFALL", colour="#ef4444")
        _vc = _vc_chip
        _v1,_v2,_v3,_v4 = st.columns(4)
        _v1.metric("Hist. VaR 95%",   f"{_vc['hist_var_95']:.2%}")
        _v2.metric("Hist. CVaR 95%",  f"{_vc['hist_cvar_95']:.2%}")
        _v3.metric("Hist. VaR 99%",   f"{_vc['hist_var_99']:.2%}")
        _v4.metric("Hist. CVaR 99%",  f"{_vc['hist_cvar_99']:.2%}")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        _v5,_v6,_v7,_v8 = st.columns(4)
        _v5.metric("Param. VaR 95%",  f"{_vc['param_var_95']:.2%}")
        _v6.metric("Param. CVaR 95%", f"{_vc['param_cvar_95']:.2%}")
        _v7.metric("Param. VaR 99%",  f"{_vc['param_var_99']:.2%}")
        _v8.metric("Param. CVaR 99%", f"{_vc['param_cvar_99']:.2%}")

        card_header("RETURN DISTRIBUTION")
        _ret_clean = portfolio_returns.dropna()
        _hist_fig  = go.Figure()
        _hist_fig.add_trace(go.Histogram(
            x=_ret_clean, nbinsx=80, name="Daily Returns",
            marker_color="#8b5cf6", opacity=0.6,
            hovertemplate="Return: %{x:.2%}<br>Count: %{y}<extra></extra>",
        ))
        import scipy.stats as _stats
        _mu, _sigma = _ret_clean.mean(), _ret_clean.std()
        _x_range = np.linspace(_ret_clean.min(), _ret_clean.max(), 300)
        _pdf     = _stats.norm.pdf(_x_range, _mu, _sigma)
        _scale   = len(_ret_clean) * (_ret_clean.max() - _ret_clean.min()) / 80
        _hist_fig.add_trace(go.Scatter(
            x=_x_range, y=_pdf * _scale, mode="lines", name="Normal Fit",
            line=dict(color="#f59e0b", width=2),
            hovertemplate="Return: %{x:.2%}<extra></extra>",
        ))
        for _var_lvl, _cval, _lbl in [
            (_vc["hist_var_95"],   "#ef4444", "VaR 95%"),
            (_vc["hist_cvar_95"],  "#7f1d1d", "CVaR 95%"),
        ]:
            _hist_fig.add_vline(x=_var_lvl, line=dict(color=_cval, width=2, dash="dash"),
                                annotation_text=_lbl, annotation_font_color=_cval,
                                annotation_position="top left")
        _hist_fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                                xaxis=dict(tickformat=".1%"), bargap=0.05)
        st.plotly_chart(_hist_fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with _r4c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("LIQUIDITY RISK", colour="var(--warning)")
        with st.spinner("Fetching liquidity data…"):
            _liq_df = cached_liquidity_risk(
                tickers_tuple,
                tuple(df.set_index("Ticker")["Quantity"].reindex(tickers).fillna(0).values),
                tuple(df.set_index("Ticker")["Market Value"].reindex(tickers).fillna(0).values),
            )
        if not _liq_df.empty:
            st.dataframe(_liq_df, use_container_width=True)
        else:
            empty_state("💧", "Liquidity data unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 5: Concentration & Diversification ─────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("CONCENTRATION & DIVERSIFICATION", colour="#ef4444")
    _con_c1, _con_c2, _con_c3 = st.columns(3)
    _con_c1.metric("Effective N (HHI)", f"{effective_n(weights_series):.1f}",
                   help="Higher = more diversified")
    _sc = sector_concentration(df)
    _sc_top_weight = _sc.iloc[0]["Weight"] if not _sc.empty else None
    _sc_top_sector = _sc.iloc[0]["Sector"] if not _sc.empty else "—"
    _con_c2.metric("Top Sector Weight",
                   f"{_sc_top_weight:.2%}" if _sc_top_weight is not None else "N/A",
                   help=f"Largest sector: {_sc_top_sector}")
    _atc = asset_type_concentration(df)
    _atc_top_weight = _atc.iloc[0]["Weight"] if not _atc.empty else None
    _atc_top_type = _atc.iloc[0]["Asset Type"] if not _atc.empty else "—"
    _con_c3.metric("Top Asset Type Weight",
                   f"{_atc_top_weight:.2%}" if _atc_top_weight is not None else "N/A",
                   help=f"Dominant type: {_atc_top_type}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    card_header("SECTOR BREAKDOWN")
    if not _sc.empty:
        _sfig = px.bar(_sc, x="Sector", y="Weight",
                       color_discrete_sequence=["#8b5cf6"])
        _sfig.update_layout(height=260, margin=dict(l=0,r=0,t=0,b=0),
                            yaxis_title="", xaxis_title="",
                            yaxis=dict(tickformat=".0%"))
        _sfig.update_xaxes(tickangle=-30)
        st.plotly_chart(_sfig, use_container_width=True)
    else:
        empty_state("🏭", "Sector data unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 6: Correlation Matrix (full width) ──────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ef4444;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("ASSET CORRELATION MATRIX", colour="#ef4444")
    if len(tickers) > 1:
        # Compute from raw prices with NaN preserved (pairwise) — avoids the
        # zero-fill in compute_returns() which artificially biases correlations toward 0
        _ret_for_corr = price_data[tickers].ffill().pct_change().iloc[1:]
        _corr_mat = _ret_for_corr.corr(method="pearson", min_periods=30)
        _n = len(_corr_mat)
        _cell_h = max(40, min(60, 500 // _n))
        _height  = max(360, _n * _cell_h + 80)
        _labels = _corr_mat.columns.tolist()
        _n = len(_labels)
        _font_size = max(8, min(11, 160 // _n))
        _height = min(520, max(320, _n * 24 + 100))
        _cfig = go.Figure(go.Heatmap(
            z=_corr_mat.values,
            x=_labels,
            y=_labels,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in _corr_mat.values],
            texttemplate="%{text}",
            textfont=dict(size=_font_size),
            hovertemplate="%{y} / %{x}: <b>%{z:.2f}</b><extra></extra>",
            colorbar=dict(
                title="Corr", thickness=12, len=0.5,
                tickfont=dict(color="#64748b"),
                title_font=dict(color="#64748b"),
            ),
        ))
        _cfig.update_layout(
            height=_height,
            margin=dict(l=0, r=80, t=10, b=80),
            xaxis=dict(side="bottom", tickangle=-45, tickfont=dict(size=_font_size), autorange="reversed"),
            yaxis=dict(side="right", autorange="reversed", tickfont=dict(size=_font_size)),
        )
        st.plotly_chart(_cfig, use_container_width=True)
    else:
        empty_state("📊", "Single asset", "Correlation matrix requires 2+ assets")
    st.markdown('</div>', unsafe_allow_html=True)

# ── MODULE: OPTIMIZATION ───────────────────────────────────
elif _module == "optimization":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:#f59e0b;text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Optimization</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Portfolio Optimizer</div>'
        '</div>',
        unsafe_allow_html=True,
    )

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

    # ── Method selector ────────────────────────────────────
    def _on_method_change():
        for k in ("_opt_key","optimal_weights"): st.session_state.pop(k, None)
        st.session_state["opt_method"] = st.session_state.get("opt_method_radio", st.session_state.get("opt_method","Max Sharpe"))

    st.radio(
        "Method", list(OPTIMIZERS.keys()), horizontal=True,
        index=list(OPTIMIZERS.keys()).index(st.session_state.get("opt_method", "Max Sharpe")),
        key="opt_method_radio", label_visibility="collapsed",
        on_change=_on_method_change,
    )

    opt_method = st.session_state.get("opt_method", "Max Sharpe")

    # Method info card
    info = METHOD_INFO.get(opt_method, {})
    st.markdown(f"""
    <div style="display:flex;align-items:flex-start;gap:18px;padding:18px 22px;
        border-radius:var(--radius);
        background:linear-gradient(135deg,var(--bg-surface) 0%,var(--bg-elevated) 100%);
        border:1px solid var(--border);border-left:3px solid #f59e0b;
        margin-bottom:20px;box-shadow:var(--shadow);">
      <div style="font-size:30px;line-height:1;flex-shrink:0;margin-top:2px;">{info.get('icon','🎯')}</div>
      <div>
        <div style="font-size:14px;font-weight:600;color:var(--text-primary);
            margin-bottom:5px;letter-spacing:-0.01em;">
            {opt_method}
            <span style="font-size:10px;font-weight:700;color:#f59e0b;
                margin-left:10px;letter-spacing:0.08em;text-transform:uppercase;">
                {info.get('tagline','')}</span>
        </div>
        <div style="font-size:12px;color:var(--text-muted);line-height:1.7;">{info.get('desc','')}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    if optimal_weights is not None and frontier is not None:
        rf = RISK_PROFILES[risk_profile]["rf_multiplier"]
        curr_ret, curr_vol, curr_sharpe = portfolio_performance(weights_series.values,  returns, rf)
        opt_ret,  opt_vol,  opt_sharpe  = portfolio_performance(optimal_weights.values, returns, rf)

        # ── Stat row ────────────────────────────────────────
        _os1, _os2, _os3, _os4 = st.columns(4)
        _sharpe_delta = f"{opt_sharpe - curr_sharpe:+.2f}"
        _ret_delta    = f"{opt_ret - curr_ret:+.2%}"
        _os1.metric("Current Sharpe",  f"{curr_sharpe:.2f}")
        _os2.metric("Optimal Sharpe",  f"{opt_sharpe:.2f}",  delta=_sharpe_delta)
        _os3.metric("Optimal Return",  f"{opt_ret:.2%}",     delta=_ret_delta)
        _os4.metric("Optimal Vol.",    f"{opt_vol:.2%}",     delta=f"{opt_vol - curr_vol:+.2%}")

        # ── Primary chart: Efficient Frontier ───────────────
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #f59e0b;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("EFFICIENT FRONTIER", colour="#f59e0b")
        fig_f = go.Figure([
            go.Scatter(
                x=frontier["Volatility"], y=frontier["Return"], mode="markers",
                marker=dict(size=5, color=frontier["Sharpe"], colorscale="Blues", showscale=True,
                            colorbar=dict(title=dict(text="Sharpe", font=dict(color="#94a3b8")),
                                          x=1.02, thickness=14, len=0.6, tickfont=dict(color="#94a3b8"))),
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
            height=360, margin=dict(l=40,r=40,t=10,b=40), showlegend=True,
            legend=dict(orientation="v",yanchor="bottom",y=0.04,xanchor="left",x=0.02,
                        bgcolor="rgba(11,17,32,0.92)",
                        bordercolor="#f59e0b", borderwidth=1,
                        font=dict(color="#e2e8f0", size=12)))
        st.plotly_chart(fig_f, use_container_width=True, key="efficient_frontier")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 2: Portfolio Comparison ────────────────────
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #f59e0b;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("PORTFOLIO COMPARISON", colour="#f59e0b")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Current Return",       f"{curr_ret:.2%}")
        c2.metric("Current Volatility",   f"{curr_vol:.2%}")
        c3.metric("Current Sharpe",       f"{curr_sharpe:.2f}")
        c4.metric("Optimized Return",     f"{opt_ret:.2%}",    delta=f"{opt_ret-curr_ret:.2%}")
        c5.metric("Optimized Volatility", f"{opt_vol:.2%}",    delta=f"{opt_vol-curr_vol:.2%}")
        c6.metric("Optimized Sharpe",     f"{opt_sharpe:.2f}", delta=f"{opt_sharpe-curr_sharpe:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Row 3: Weight Comparison + Trade Instructions ──
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #f59e0b;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("WEIGHT COMPARISON & TRADE INSTRUCTIONS", colour="#f59e0b")

        # Max position size slider — styled to match amber theme
        st.markdown(
            '<div style="margin-bottom:16px;padding:12px 16px;'
            'background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.2);'
            'border-radius:var(--radius-sm);">',
            unsafe_allow_html=True,
        )
        _new_max_w = st.slider(
            "Max Position Size (%)", 5.0, 50.0,
            st.session_state.get("max_weight_pct", 15.0), 1.0,
            key="opt_max_weight",
            help="Maximum weight any single asset can hold in the optimized portfolio",
        )
        st.markdown('</div>', unsafe_allow_html=True)
        if _new_max_w != st.session_state.get("max_weight_pct", 15.0):
            st.session_state["max_weight_pct"] = _new_max_w
            for k in ("_opt_key", "optimal_weights"):
                st.session_state.pop(k, None)
            st.rerun()
        max_weight_pct = st.session_state.get("max_weight_pct", 15.0)
        max_weight     = max_weight_pct / 100

        _price_map = df.set_index("Ticker")["Current Price"].to_dict()
        _qty_map   = df.set_index("Ticker")["Quantity"].to_dict()

        wt_df = pd.DataFrame({
            "Current Weight":   weights_series.reindex(optimal_weights.index).fillna(0),
            "Optimized Weight": optimal_weights,
        })
        wt_df["Change"] = wt_df["Optimized Weight"] - wt_df["Current Weight"]
        wt_df = wt_df.sort_values("Optimized Weight", ascending=False)

        _trades = []
        for ticker in wt_df.index:
            _price        = _price_map.get(ticker, np.nan)
            _cur_qty      = float(_qty_map.get(ticker, 0))
            _opt_weight   = float(wt_df.loc[ticker, "Optimized Weight"])
            _target_value = _opt_weight * total_value
            _target_qty   = (_target_value / _price) if (pd.notna(_price) and _price > 0) else np.nan
            _delta_qty    = (_target_qty - _cur_qty)  if pd.notna(_target_qty) else np.nan

            if pd.notna(_delta_qty):
                _delta_qty_rounded = int(round(_delta_qty))
            else:
                _delta_qty_rounded = None

            if _delta_qty_rounded is None:
                _action = "—"
            elif abs(_delta_qty_rounded) < 1:
                _action = "Hold"
            elif _delta_qty_rounded > 0:
                _action = f"Buy {_delta_qty_rounded:+,}"
            else:
                _action = f"Sell {abs(_delta_qty_rounded):,}"

            _trades.append({
                "Ticker":           ticker,
                "Current Weight":   f"{wt_df.loc[ticker,'Current Weight']:.2%}",
                "Optimized Weight": f"{wt_df.loc[ticker,'Optimized Weight']:.2%}",
                "Change":           f"{wt_df.loc[ticker,'Change']:+.2%}",
                "Action":           _action,
            })

        _trades_df = pd.DataFrame(_trades)
        st.dataframe(_trades_df, use_container_width=True)

        # Risk contributions / diversification ratio
        _rc_tab1, _rc_tab2 = st.tabs(["Risk Contributions", "Diversification Ratio"])
        with _rc_tab1:
            card_header("EQUAL RISK PARITY TARGET")
            try:
                from optimizer import risk_contribution
                _rc  = risk_contribution(optimal_weights.values, returns[optimal_weights.index])
                _rcd = pd.DataFrame({"Ticker": optimal_weights.index, "Risk Contribution": _rc})
                _rcd = _rcd.sort_values("Risk Contribution", ascending=False)
                _rcf = px.bar(_rcd, x="Ticker", y="Risk Contribution",
                              color="Risk Contribution", color_continuous_scale="Blues")
                _rcf.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0),
                                   showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(_rcf, use_container_width=True)
            except (ImportError, Exception):
                empty_state("⚖️","Risk contribution data unavailable")

        with _rc_tab2:
            card_header("DIVERSIFICATION RATIO")
            try:
                from optimizer import diversification_ratio
                _dr_curr = diversification_ratio(weights_series.values, returns.cov().values * 252,
                                                  returns.std().values * np.sqrt(252))
                _dr_opt  = diversification_ratio(optimal_weights.values, returns.cov().values * 252,
                                                  returns.std().values * np.sqrt(252))
                _drc1, _drc2 = st.columns(2)
                _drc1.metric("Current Diversification Ratio", f"{_dr_curr:.3f}")
                _drc2.metric("Optimized Diversification Ratio", f"{_dr_opt:.3f}",
                             delta=f"{_dr_opt - _dr_curr:+.3f}")
            except (ImportError, Exception):
                empty_state("🌐","Diversification ratio unavailable")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        empty_state("🎯","Optimization unavailable","Could not compute optimal weights for this portfolio")

    # ── Row 4: Scenario Analysis ───────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #f59e0b;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("SCENARIO ANALYSIS", colour="#f59e0b")
    with st.expander("Run a what-if scenario", expanded=False):
        _scenario_mode = st.radio(
            "Mode", ["Cash Injection", "Trade Simulation"],
            horizontal=True, key="scenario_mode"
        )

        if _scenario_mode == "Cash Injection":
            _cash_input = st.number_input(
                f"Cash to inject ({_currency})", min_value=0.0,
                value=float(total_value * 0.1), step=100.0, key="scenario_cash"
            )
            if st.button("Run Cash Injection", key="run_cash_injection") and optimal_weights is not None:
                _nw, _nr, _nv, _ns = simulate_cash_injection(
                    weights_series, returns, optimal_weights, _cash_input, total_value
                )
                _curr_ret, _curr_vol, _curr_s = portfolio_performance(
                    weights_series.values, returns, rf_multiplier=1.0
                )
                st.markdown("**Before / After**")
                _sc1, _sc2, _sc3 = st.columns(3)
                _sc1.metric("Sharpe",     f"{_ns:.3f}", delta=f"{_ns - _curr_s:+.3f}")
                _sc2.metric("Volatility", f"{_nv:.2%}", delta=f"{_nv - _curr_vol:+.2%}")
                _sc3.metric("Return",     f"{_nr:.2%}", delta=f"{_nr - _curr_ret:+.2%}")
                _nw_df = pd.DataFrame({
                    "Ticker":         _nw.index,
                    "Current Weight": weights_series.reindex(_nw.index).fillna(0).values,
                    "New Weight":     _nw.values,
                    "Change":         (_nw - weights_series.reindex(_nw.index).fillna(0)).values,
                }).reset_index(drop=True)
                st.dataframe(_nw_df.style.format({
                    "Current Weight": "{:.2%}",
                    "New Weight":     "{:.2%}",
                    "Change":         "{:+.2%}",
                }), use_container_width=True)

        else:
            _trade_ticker = st.selectbox(
                "Ticker", options=tickers + ["(new ticker)"], key="scenario_ticker"
            )
            if _trade_ticker == "(new ticker)":
                _trade_ticker = st.text_input("Enter ticker symbol", key="scenario_new_ticker").upper().strip()
            _trade_action   = st.radio("Action", ["Buy", "Sell"], horizontal=True, key="scenario_action")
            _trade_qty      = st.number_input("Quantity (shares)", min_value=0.01, value=1.0, key="scenario_qty")
            _trade_price_default = float(price_data[_trade_ticker].iloc[-1]) if _trade_ticker in price_data.columns else 0.0
            _trade_price    = st.number_input(f"Price ({_currency})", min_value=0.01,
                                              value=max(_trade_price_default, 0.01), key="scenario_price")

            if st.button("Run Trade Simulation", key="run_trade_sim") and _trade_ticker:
                _nw2, _nr2, _nv2, _ns2, _warn = simulate_trade(
                    weights_series, returns, _trade_ticker, _trade_action,
                    _trade_qty, _trade_price, total_value, risk_profile
                )
                _curr_ret2, _curr_vol2, _curr_s2 = portfolio_performance(
                    weights_series.values, returns, rf_multiplier=1.0
                )
                if _warn:
                    st.warning(_warn)
                st.markdown("**Before / After**")
                _tc1, _tc2, _tc3 = st.columns(3)
                _tc1.metric("Sharpe",     f"{_ns2:.3f}", delta=f"{_ns2 - _curr_s2:+.3f}")
                _tc2.metric("Volatility", f"{_nv2:.2%}", delta=f"{_nv2 - _curr_vol2:+.2%}")
                _tc3.metric("Return",     f"{_nr2:.2%}", delta=f"{_nr2 - _curr_ret2:+.2%}")
    st.markdown('</div>', unsafe_allow_html=True)


# ── MODULE: PERFORMANCE ────────────────────────────────────
elif _module == "performance":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:#22c55e;text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Performance</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Performance Attribution</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Benchmark selector ──────────────────────────────────
    _benchmark_map = {
        "S&P 500": "^GSPC", "NIFTY 50": "^NSEI", "NIFTY 500": "^CRSLDX",
        "SENSEX": "^BSESN", "Dow Jones": "^DJI", "NASDAQ": "^IXIC", "Custom…": None,
    }
    _perf_cols = st.columns([2, 5])
    with _perf_cols[0]:
        _bm_name = st.selectbox("Benchmark", list(_benchmark_map.keys()),
                                key="perf_benchmark_name")
    if _bm_name == "Custom…":
        with _perf_cols[1]:
            _custom_bm = st.text_input("Ticker", placeholder="e.g. QQQ, ^FTSE",
                                       key="perf_custom_bm").strip().upper()
        _new_bm = _custom_bm if _custom_bm else st.session_state.get("benchmark", "^GSPC")
    else:
        _new_bm = _benchmark_map[_bm_name]
    if _new_bm and _new_bm != st.session_state.get("benchmark"):
        st.session_state["benchmark"] = _new_bm
        st.rerun()

    # ── Timeframe selector ──────────────────────────────────
    pm_tf = st.radio("Metrics Timeframe", ["1M","3M","6M","1Y","3Y","5Y","All"],
                     horizontal=True, index=3, key="pm_timeframe", label_visibility="collapsed")
    _pr_sliced = slice_tf(portfolio_returns, pm_tf) if pm_tf != "All" else portfolio_returns
    _br_sliced = slice_tf(benchmark_returns, pm_tf) if (benchmark_returns is not None and pm_tf != "All") else benchmark_returns
    pm = get_performance_metrics(_pr_sliced, _br_sliced, rf_rate=_rf_rate)

    # ── Stat row ────────────────────────────────────────────
    _ps1, _ps2, _ps3, _ps4 = st.columns(4)
    _ps1.metric("Total Return",      f"{pm['total_return']:.2%}")
    _ps2.metric("Annualized Return", f"{pm['annualized_return']:.2%}")
    _ps3.metric("Sharpe Ratio",      f"{pm['sharpe_ratio']:.2f}")
    _ps4.metric("Max Drawdown",      f"{pm['max_drawdown']:.2%}")

    # ── Primary chart: Cumulative Returns ───────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("CUMULATIVE RETURNS", colour="#22c55e")
    tf = pm_tf if pm_tf != "All" else "5Y"
    cum_full = (1 + portfolio_returns.dropna()).cumprod()
    cum      = slice_tf(cum_full, tf)
    if cum.empty:
        cum = cum_full
    cum      = cum / cum.iloc[0] - 1
    perf_df = pd.DataFrame({"Portfolio": cum})
    if benchmark_returns is not None:
        bc_full = (1 + benchmark_returns.dropna()).cumprod()
        bc      = slice_tf(bc_full, tf).reindex(cum.index, method="ffill")
        if not bc.empty:
            bc = bc / bc.iloc[0] - 1
            perf_df["Benchmark"] = bc
    fig = px.line(perf_df, color_discrete_map={"Portfolio":"#22c55e","Benchmark":"#64748b"})
    fig.update_traces(line=dict(width=2))
    if portfolio_xirr is not None and not np.isnan(portfolio_xirr):
        xirr_start = cum.index[0]
        xirr_end   = cum.index[-1]
        xirr_dates = pd.date_range(xirr_start, xirr_end, freq="B")
        days_from_start = (xirr_dates - xirr_start).days
        xirr_curve      = (1 + portfolio_xirr) ** (days_from_start / 365) - 1
        fig.add_scatter(x=xirr_dates, y=xirr_curve,
                        name=f"XIRR Pace ({portfolio_xirr:.2%} p.a.)",
                        line=dict(color="#f59e0b", width=1.5, dash="dot"))
    fig.update_layout(height=360, margin=dict(l=0,r=0,t=0,b=0),
                      yaxis=dict(tickformat=".0%", title="Return"),
                      xaxis=dict(title="Date"), hovermode="x unified")
    fig.update_traces(hovertemplate="%{y:.2%}")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Full metrics cards ──────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("PERFORMANCE METRICS", colour="#22c55e")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Return",      f"{pm['total_return']:.2%}")
    c2.metric("Annualized Return", f"{pm['annualized_return']:.2%}")
    c3.metric("Volatility",        f"{pm['volatility']:.2%}")
    c4.metric("Sharpe Ratio",      f"{pm['sharpe_ratio']:.2f}")
    c5.metric("Sortino Ratio",     f"{pm['sortino_ratio']:.2f}")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    cA,cB,cC,cD = st.columns(4)
    cA.metric("Max Drawdown", f"{pm['max_drawdown']:.2%}")
    cB.metric("Win Rate",     f"{pm['win_rate']:.1%}")
    if "alpha" in pm:
        cC.metric("Alpha",             f"{pm['alpha']:.2%}")
        cD.metric("Information Ratio", f"{pm['information_ratio']:.2f}")
    if benchmark_returns is not None:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Portfolio Ann. Return", f"{pm['annualized_return']:.2%}")
        b2.metric("Benchmark Ann. Return", f"{pm['benchmark_annualized_return']:.2%}")
        b3.metric("Outperformance",        f"{pm['outperformance']:.2%}")
        b4.metric("Beta",                  f"{pm['beta']:.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Capture Ratios | Period Returns ─────────────
    _r3c1, _r3c2 = st.columns(2)

    with _r3c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("CAPTURE RATIOS", colour="#22c55e")
        if benchmark_returns is not None:
            _captures = compute_capture_ratios(_pr_sliced, _br_sliced)
            _uc = _captures.get("upside_capture")
            _dc = _captures.get("downside_capture")
            _cap_c1, _cap_c2 = st.columns(2)
            _cap_c1.metric(
                "Upside Capture",
                f"{_uc:.1f}%" if _uc is not None and not np.isnan(_uc) else "N/A",
                help="% of benchmark's up-day gains captured. >100% = outperforms on up days."
            )
            _cap_c2.metric(
                "Downside Capture",
                f"{_dc:.1f}%" if _dc is not None and not np.isnan(_dc) else "N/A",
                help="% of benchmark's down-day losses mirrored. <100% = better protection on down days."
            )
        else:
            empty_state("📊", "No benchmark", "Select a benchmark to see capture ratios")
        st.markdown('</div>', unsafe_allow_html=True)

    with _r3c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("PERIOD RETURNS", colour="#22c55e")
        pr      = get_period_returns(portfolio_returns, benchmark_returns)
        pr.index += 1
        pl_cols = ["Portfolio Return","Benchmark Return","Excess Return"] if "Benchmark Return" in pr.columns else ["Portfolio Return"]
        st.dataframe(style_pl(pr, pl_cols).format({c:"{:.2%}" for c in pl_cols}), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: Sector Contribution | Brinson Attribution ───
    _r4c1, _r4c2 = st.columns(2)

    _sector_map_dict     = df.set_index("Ticker")["Sector"].to_dict()
    _perf_returns_sliced = slice_tf(returns, pm_tf) if pm_tf != "All" else returns

    with _r4c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("SECTOR CONTRIBUTION", colour="#22c55e")
        _sc_df = compute_sector_contribution(weights_series, _perf_returns_sliced, _sector_map_dict)
        if not _sc_df.empty:
            _sc_fig = px.bar(
                _sc_df, x="Contribution", y="Sector", orientation="h",
                color="Contribution", color_continuous_scale=["#ef4444","#f5f5f5","#22c55e"],
                text=_sc_df["Contribution"].map(lambda x: f"{x:.2%}"),
            )
            _sc_fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0),
                                  showlegend=False, coloraxis_showscale=False)
            _sc_fig.update_traces(textposition="outside")
            st.plotly_chart(_sc_fig, use_container_width=True)
        else:
            empty_state("📊", "Sector data unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    with _r4c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("BRINSON ATTRIBUTION", colour="#22c55e")
        if benchmark in ("^GSPC", "^NSEI", "^CRSLDX") and benchmark_returns is not None:
            _br_result = compute_brinson_attribution(
                weights_series, _perf_returns_sliced, _br_sliced, _sector_map_dict, benchmark
            )
            if isinstance(_br_result, tuple):
                _br_df, _mixed = _br_result
            else:
                _br_df, _mixed = _br_result, False
            if _mixed:
                st.warning("Attribution results may be misleading for mixed-market portfolios.")
            if not _br_df.empty:
                _br_display = _br_df.copy()
                for _bcol in ["Portfolio Weight", "Benchmark Weight"]:
                    if _bcol in _br_display.columns:
                        _br_display[_bcol] = _br_display[_bcol].map(lambda x: f"{x:.2%}")
                for _bcol in ["Allocation Effect", "Selection Effect", "Interaction Effect", "Total Active"]:
                    if _bcol in _br_display.columns:
                        _br_display[_bcol] = _br_display[_bcol].map(lambda x: f"{x:+.4%}")
                st.dataframe(_br_display, use_container_width=True)
            else:
                empty_state("📊", "Attribution data unavailable")
        else:
            empty_state("📊", "Attribution unavailable",
                        "Available for S&P 500, Nifty 50, and CRSLDX benchmarks only")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 5: Rolling Sharpe | Individual Stock vs Benchmark
    _r5c1, _r5c2 = st.columns(2)

    with _r5c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("ROLLING 60-DAY SHARPE RATIO", colour="#22c55e")
        rm  = get_rolling_metrics(portfolio_returns, benchmark_returns, window=60)
        fig = px.line(rm[["Sharpe Ratio"]], color_discrete_map={"Sharpe Ratio":"#22c55e"})
        if "Benchmark Sharpe" in rm.columns:
            fig.add_scatter(x=rm.index, y=rm["Benchmark Sharpe"], name="Benchmark Sharpe",
                            line=dict(color="#64748b",width=2))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with _r5c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #22c55e;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("INDIVIDUAL STOCK VS BENCHMARK", colour="#22c55e")
        if benchmark_returns is not None:
            _indiv_perf = {}
            for _t in tickers:
                if _t in returns.columns:
                    _tret = returns[_t].dropna()
                    _tpm  = get_performance_metrics(_tret, _br_sliced, rf_rate=_rf_rate)
                    _indiv_perf[_t] = {
                        "Ann. Return":   f"{_tpm['annualized_return']:.2%}",
                        "Sharpe":        f"{_tpm['sharpe_ratio']:.2f}",
                        "Outperformance":f"{_tpm.get('outperformance', 0):.2%}",
                        "Beta":          f"{_tpm.get('beta', 0):.2f}",
                    }
            if _indiv_perf:
                st.dataframe(pd.DataFrame(_indiv_perf).T, use_container_width=True)
            else:
                empty_state("📊","No individual performance data")
        else:
            empty_state("📈","No benchmark selected","Select a benchmark for comparison")
        st.markdown('</div>', unsafe_allow_html=True)


# ── MODULE: ASSET ANALYTICS ────────────────────────────────
elif _module == "asset_analytics":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:#8b5cf6;text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Asset Analytics</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Deep-Dive Analysis</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if "selected_asset" not in st.session_state: st.session_state["selected_asset"] = tickers[0]
    if st.session_state["selected_asset"] not in tickers: st.session_state["selected_asset"] = tickers[0]

    selected_asset = st.selectbox("Select Asset", tickers, key="selected_asset")
    asset_price    = price_data[selected_asset]
    asset_returns  = returns[selected_asset]
    asset_weight   = float(weights_series.get(selected_asset, 0))
    asset_stats    = get_asset_key_stats(selected_asset, asset_price, asset_returns, asset_weight)

    # Company Profile Banner
    _is_indian = selected_asset.endswith((".NS", ".BO"))
    _profile   = {} if _is_indian else (cached_finnhub_profile(selected_asset, _FINNHUB_KEY) if _FINNHUB_KEY else {})
    if not _profile:
        try:
            _yf_info = yf.Ticker(selected_asset).info
            _profile = {
                "name":                _yf_info.get("longName") or _yf_info.get("shortName") or selected_asset,
                "exchange":            "NSE" if selected_asset.endswith(".NS") else ("BSE" if selected_asset.endswith(".BO") else _yf_info.get("exchange", "—")),
                "finnhubIndustry":     _yf_info.get("industry") or _yf_info.get("sector") or "—",
                "country":             _yf_info.get("country", "India" if _is_indian else "—"),
                "ipo":                 str(_yf_info.get("ipoExpectedDate", "—")),
                "marketCapitalization":(_yf_info.get("marketCap", 0) or 0) / 1e6,
                "weburl":              _yf_info.get("website", ""),
                "logo":                _yf_info.get("logo_url", ""),
            }
        except Exception:
            pass

    if _profile and _profile.get("name"):
        _logo_html = (f'<img src="{_profile["logo"]}" style="width:40px;height:40px;'
                      f'border-radius:8px;object-fit:contain;background:#fff;padding:3px;">'
                      ) if _profile.get("logo") else ""
        _mktcap = _profile.get("marketCapitalization", 0)
        if _is_indian and _mktcap:
            _mktcap_raw = _mktcap * 1e6
            _mktcap_str = f"₹{_mktcap_raw/1e7:.0f}Cr" if _mktcap_raw >= 1e7 else f"₹{_mktcap_raw/1e5:.1f}L"
        else:
            _mktcap_str = (
                f"${_mktcap/1_000_000:.2f}T" if _mktcap >= 1_000_000
                else f"${_mktcap/1_000:.2f}B" if _mktcap >= 1_000
                else f"${_mktcap:.0f}M"
            ) if _mktcap else "—"
        _web = _profile.get("weburl", "")
        _web_html = (f'<a href="{_web}" target="_blank" style="color:var(--accent);'
                     f'font-size:10px;text-decoration:none;">{_web.replace("https://","").rstrip("/")}'
                     f'</a>') if _web else ""
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:18px;padding:16px 20px;
            margin-bottom:16px;border-radius:var(--radius);
            background:var(--bg-surface);border:1px solid var(--border);
            border-left:3px solid #8b5cf6;box-shadow:var(--shadow);">
          {_logo_html}
          <div style="flex:1;">
            <div style="font-size:15px;font-weight:700;color:var(--text-primary);
                letter-spacing:-0.02em;">{_profile.get('name', selected_asset)}</div>
            <div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:6px;">
              <span style="font-size:11px;color:var(--text-muted);">
                <span style="color:var(--text-secondary);font-weight:600;">Exchange</span>
                &nbsp;{_profile.get('exchange','—')}</span>
              <span style="font-size:11px;color:var(--text-muted);">
                <span style="color:var(--text-secondary);font-weight:600;">Industry</span>
                &nbsp;{_profile.get('finnhubIndustry','—')}</span>
              <span style="font-size:11px;color:var(--text-muted);">
                <span style="color:var(--text-secondary);font-weight:600;">Country</span>
                &nbsp;{_profile.get('country','—')}</span>
              <span style="font-size:11px;color:var(--text-muted);">
                <span style="color:var(--text-secondary);font-weight:600;">Mkt Cap</span>
                &nbsp;{_mktcap_str}</span>
              <span style="font-size:11px;">{_web_html}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Timeframe selector ──────────────────────────────────
    tf = st.radio("Timeframe",["1M","3M","6M","1Y","3Y","5Y"],horizontal=True,
                  key="asset_price_tf",label_visibility="collapsed")

    # ── Stat row ────────────────────────────────────────────
    _cur_price = float(asset_price.iloc[-1]) if not asset_price.empty else 0.0
    _as1, _as2, _as3, _as4 = st.columns(4)
    _as1.metric("Current Price",  f"{_currency}{_cur_price:,.2f}", help=f"Weight: {asset_weight:.2%}")
    _as2.metric("Ann. Return",    f"{asset_stats['annual_return']:.2%}")
    _as3.metric("Sharpe Ratio",   f"{asset_stats['sharpe_ratio']:.2f}")
    _as4.metric("Volatility",     f"{asset_stats['volatility']:.2%}")

    # ── Primary chart: Price History ────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("PRICE HISTORY", colour="#8b5cf6")
    fig = px.line(slice_tf(asset_price, tf), color_discrete_sequence=["#8b5cf6"])
    fig.update_traces(line=dict(width=1.8))
    fig.update_layout(height=360, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Slice returns to same timeframe
    asset_returns_tf = slice_tf(asset_returns, tf)

    # ── Row 2: Rolling Vol | Rolling Correlation ───────────
    _r2c1, _r2c2 = st.columns(2)

    with _r2c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("ROLLING VOLATILITY (60D)", colour="#8b5cf6")
        quick_chart(px.line(slice_tf(compute_rolling_volatility(asset_returns, 60), tf),
                            color_discrete_sequence=["#f59e0b"]), 260)
        st.markdown('</div>', unsafe_allow_html=True)

    with _r2c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("ROLLING CORRELATION WITH PORTFOLIO (60D)", colour="#8b5cf6")
        quick_chart(px.line(slice_tf(compute_rolling_correlation(asset_returns, portfolio_returns, 60), tf),
                            color_discrete_sequence=["#8b5cf6"]), 260)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Drawdown | Dividend Tracking ───────────────
    _r3c1, _r3c2 = st.columns(2)

    with _r3c1:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("DRAWDOWN", colour="#8b5cf6")
        fig = px.area(compute_asset_drawdown(asset_returns_tf), color_discrete_sequence=["#ef4444"])
        fig.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,0.1)")
        quick_chart(fig, 260)
        st.markdown('</div>', unsafe_allow_html=True)

    with _r3c2:
        st.markdown(
            '<div style="background:var(--bg-card);border:1px solid var(--border);'
            'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
            unsafe_allow_html=True,
        )
        card_header("DIVIDEND TRACKING", colour="var(--positive)")
        _asset_price_val = float(price_data[selected_asset].iloc[-1]) if selected_asset in price_data.columns else 0.0
        _asset_qty_val   = float(df.set_index("Ticker")["Quantity"].get(selected_asset, 0))
        _div_data = cached_dividend_data(selected_asset, _asset_qty_val, _asset_price_val)
        if _div_data["has_dividends"]:
            _dc1, _dc2 = st.columns(2)
            _dc1.metric("Dividend Yield",     f"{_div_data['yield']:.2%}")
            _dc2.metric("Est. Annual Income", f"{_currency}{_div_data['annual_income']:,.2f}")
            if not _div_data["history"].empty:
                _div_fig = px.bar(
                    x=_div_data["history"].index,
                    y=_div_data["history"].values,
                    labels={"x": "Date", "y": "Dividend per Share"},
                    color_discrete_sequence=["#22c55e"],
                )
                _div_fig.update_layout(height=180, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(_div_fig, use_container_width=True)
        else:
            empty_state("💰", "No dividends", "This asset does not pay dividends")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 4: Fundamental Metrics ─────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("FUNDAMENTAL METRICS", colour="#8b5cf6")
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
            card_header("Profitability")
            if not prof.empty:
                st.dataframe(prof.set_index("Metric"), use_container_width=True)
            else:
                st.caption("No data")
        with c2:
            card_header("Valuation")
            if not val.empty:
                st.dataframe(val.set_index("Metric"), use_container_width=True)
            else:
                st.caption("No data")
        card_header("Liquidity & Solvency")
        if not liq.empty:
            st.dataframe(liq.set_index("Metric"), use_container_width=True)
        else:
            st.caption("No data")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 5: Analyst Consensus + Earnings + Peers ────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #8b5cf6;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("ANALYST CONSENSUS", colour="#8b5cf6")
    if not _is_indian and _FINNHUB_KEY:
        _fh_recs = cached_finnhub_recs(selected_asset, _FINNHUB_KEY)
        if not _fh_recs.empty:
            _rec_fig = go.Figure([
                go.Bar(name="Strong Buy",  x=_fh_recs["Period"], y=_fh_recs["Strong Buy"],  marker_color="#15803d"),
                go.Bar(name="Buy",         x=_fh_recs["Period"], y=_fh_recs["Buy"],          marker_color="#22c55e"),
                go.Bar(name="Hold",        x=_fh_recs["Period"], y=_fh_recs["Hold"],          marker_color="#f59e0b"),
                go.Bar(name="Sell",        x=_fh_recs["Period"], y=_fh_recs["Sell"],          marker_color="#ef4444"),
                go.Bar(name="Strong Sell", x=_fh_recs["Period"], y=_fh_recs["Strong Sell"],  marker_color="#7f1d1d"),
            ])
            _rec_fig.update_layout(barmode="stack", height=260, margin=dict(l=0,r=0,t=0,b=0),
                                   legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(_rec_fig, use_container_width=True)
        else:
            empty_state("📊", "Analyst ratings unavailable", f"No analyst coverage found for {selected_asset}")
    else:
        st.info("Analyst consensus not available for Indian listings or without API key.")

    card_header("EARNINGS SURPRISES", colour="#8b5cf6")
    if not _is_indian and _FINNHUB_KEY:
        _fh_earn = cached_finnhub_earnings(selected_asset, _FINNHUB_KEY)
        if not _fh_earn.empty:
            def _style_result(val):
                if val == "Beat": return "color:#22c55e;font-weight:600"
                if val == "Miss": return "color:#ef4444;font-weight:600"
                return "color:#f59e0b;font-weight:600"
            st.dataframe(_fh_earn.style.map(_style_result, subset=["Result"]), use_container_width=True)
        else:
            empty_state("📈","Earnings data unavailable")
    else:
        st.info("Earnings surprises not available for Indian listings or without API key.")

    card_header("PEER COMPANIES", colour="#8b5cf6")
    if not _is_indian and _FINNHUB_KEY:
        _peers = cached_finnhub_peers(selected_asset, _FINNHUB_KEY)
        if _peers:
            st.write(", ".join(_peers))
        else:
            empty_state("👥","Peer data unavailable")
    else:
        st.info("Peer data not available for Indian listings.")

    # ── Latest News ────────────────────────────────────────
    card_header(f"LATEST NEWS — {selected_asset}", colour="#8b5cf6")
    _news_df = cached_yf_news(selected_asset)
    if not _news_df.empty:
        for _, _nrow in _news_df.head(8).iterrows():
            _ts_str = (_nrow["datetime"].strftime("%b %d, %Y")
                       if pd.notna(_nrow["datetime"]) else "")
            st.markdown(
                f'<div style="padding:10px 0;border-bottom:1px solid var(--border-subtle);">'
                f'<a href="{_nrow["url"]}" target="_blank" style="color:var(--text-primary);'
                f'font-size:13px;font-weight:500;text-decoration:none;">{_nrow["headline"]}</a>'
                f'<div style="font-size:11px;color:var(--text-muted);margin-top:3px;">'
                f'{_nrow["source"]} · {_ts_str}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        empty_state("📰","No news available",f"No recent news found for {selected_asset}")

    st.markdown('</div>', unsafe_allow_html=True)


# ── MODULE: ENHANCEMENT ────────────────────────────────────
elif _module == "enhancement":

    # ── Page header ────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:20px;">'
        '<div style="font-size:10px;color:#ec4899;text-transform:uppercase;'
        'letter-spacing:0.12em;margin-bottom:4px;">Enhancement</div>'
        '<div style="font-size:22px;font-weight:700;color:var(--text-primary);'
        'letter-spacing:-0.02em;">Portfolio Signals</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Pre-compute 3M relative performance
    try:
        with st.spinner("Computing 3M relative performance…"):
            pm_df = cached_3m_relative_performance(tickers_tuple, benchmark)

        def _rule_engine(x):
            if pd.isna(x): return "No Data"
            return "Sell" if x < -0.10 else "Buy" if x > 0.20 else "Hold"

        pm_df["Action"] = pm_df["Relative Performance"].apply(_rule_engine)
        pm_df = pm_df.sort_values("Relative Performance", ascending=False).reset_index(drop=True)
        pm_df.index += 1
        _buy_count  = int((pm_df["Action"] == "Buy").sum())
        _sell_count = int((pm_df["Action"] == "Sell").sum())
        _hold_count = int((pm_df["Action"] == "Hold").sum())
        _top_buy_ticker = pm_df[pm_df["Action"] == "Buy"]["Ticker"].iloc[0] if _buy_count > 0 else "—"
        _top_buy_sector = df.set_index("Ticker")["Sector"].get(_top_buy_ticker, "—") if _top_buy_ticker != "—" else "—"
        _enh_data_ok = True
    except Exception as _e:
        pm_df = pd.DataFrame()
        _buy_count = _sell_count = _hold_count = 0
        _top_buy_ticker = "—"
        _top_buy_sector = "—"
        _enh_data_ok = False

    # ── Stat row ────────────────────────────────────────────
    _es1, _es2, _es3, _es4 = st.columns(4)
    _es1.metric("Buy Signals",   str(_buy_count))
    _es2.metric("Hold Signals",  str(_hold_count))
    _es3.metric("Sell Signals",  str(_sell_count))
    if _enh_data_ok and not pm_df.empty and "Relative Performance" in pm_df.columns:
        _best_rel = pm_df.iloc[0]["Relative Performance"]
        _es4.metric("Best Performer", pm_df.iloc[0]["Ticker"], delta=f"{_best_rel:+.2%}")
    else:
        _es4.metric("Best Performer", "—")

    # ── Primary chart: 3M Relative Performance ─────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ec4899;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("3-MONTH RELATIVE PERFORMANCE", colour="#ec4899")
    if _enh_data_ok and not pm_df.empty and "Relative Performance" in pm_df.columns:
        _bar_df = pm_df.reset_index(drop=True).copy()
        _bar_df["Colour"] = _bar_df["Relative Performance"].apply(
            lambda x: "#22c55e" if x > 0.20 else ("#ef4444" if x < -0.10 else "#f59e0b")
        )
        _hero_fig = go.Figure(go.Bar(
            x=_bar_df["Ticker"],
            y=_bar_df["Relative Performance"],
            marker_color=_bar_df["Colour"],
            hovertemplate="<b>%{x}</b><br>Relative Perf: %{y:.2%}<extra></extra>",
        ))
        _hero_fig.add_hline(y=0, line=dict(color="#64748b", width=1, dash="dot"))
        _hero_fig.update_layout(
            height=360, margin=dict(l=0,r=0,t=0,b=0),
            yaxis=dict(tickformat=".0%", title="Relative Performance"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(_hero_fig, use_container_width=True)
    else:
        empty_state("📊", "Relative performance data unavailable")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Analyst Consensus ───────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ec4899;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    card_header("ANALYST CONSENSUS — ALL HOLDINGS", colour="#ec4899")
    if _market == "IN":
        st.info("Portfolio analyst consensus is not available for Indian exchange listings on the free data tier.")
        _cons_df = pd.DataFrame()
    else:
        with st.spinner("Fetching analyst ratings…"):
            _cons_df = cached_finnhub_consensus(tickers_tuple, _FINNHUB_KEY) if _FINNHUB_KEY else pd.DataFrame()
    if not _cons_df.empty:
        _cA, _cB, _cC, _cD = st.columns(4)
        _cA.metric("Strong Buy", int((_cons_df["Consensus"] == "Strong Buy").sum()))
        _cB.metric("Buy",        int((_cons_df["Consensus"] == "Buy").sum()))
        _cC.metric("Hold",       int((_cons_df["Consensus"] == "Hold").sum()))
        _cD.metric("Sell / Strong Sell",
                   int(_cons_df["Consensus"].isin(["Sell", "Strong Sell"]).sum()))
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        def _style_cons(val):
            return {
                "Strong Buy":  "color:#15803d;font-weight:700",
                "Buy":         "color:#22c55e;font-weight:600",
                "Hold":        "color:#f59e0b;font-weight:600",
                "Sell":        "color:#ef4444;font-weight:600",
                "Strong Sell": "color:#7f1d1d;font-weight:700",
            }.get(val, "")
        st.dataframe(
            _cons_df.style.map(_style_cons, subset=["Consensus"]),
            use_container_width=True)
    elif _market != "IN":
        empty_state("📊", "Analyst consensus unavailable",
                    "No analyst coverage found for these tickers")

    # Holdings signal table
    card_header("HOLDINGS SIGNAL TABLE", colour="#ec4899")
    if _enh_data_ok and not pm_df.empty:
        st.dataframe(
            style_pl(pm_df, ["Relative Performance"]).format({
                "3M Return":            "{:.2%}",
                "Benchmark 3M":         "{:.2%}",
                "Relative Performance": "{:.2%}",
            }),
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Sector Screener ─────────────────────────────
    st.markdown(
        '<div style="background:var(--bg-card);border:1px solid var(--border);'
        'border-top:2px solid #ec4899;border-radius:var(--radius);padding:20px;margin-bottom:16px;">',
        unsafe_allow_html=True,
    )
    _screener_universe = "NIFTY 500" if _market == "IN" else "S&P 500"
    card_header(f"SECTOR SCREENER — {_screener_universe}", colour="#ec4899")
    st.markdown(f"""
    <div style="font-size:12px;color:var(--text-muted);line-height:1.65;margin-bottom:16px;">
        Top performing sectors and their best stocks ranked by 6 &amp; 12-month returns.
        Includes PE ratios and ROE. Refreshed every hour.</div>""", unsafe_allow_html=True)

    with st.spinner("Analyzing sectors — may take ~15s on first load…"):
        sector_recs = cached_sector_recommendations(_market)

    if not sector_recs:
        empty_state("🔍","No sector opportunities identified","Try again later")
    else:
        card_header("TOP SECTORS WITH BEST PERFORMERS", colour="#ec4899")
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
    st.markdown('</div>', unsafe_allow_html=True)