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
    asset_type_concentration, effective_n)
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
    compute_asset_drawdown, get_asset_fundamental_table,
)
from performance_engine import get_performance_metrics, get_period_returns, get_rolling_metrics
from external_apis import (
    finnhub_earnings_surprises, finnhub_recommendations,
    finnhub_insider_sentiment, finnhub_portfolio_consensus,
    finnhub_stock_metrics, finnhub_company_profile, finnhub_peers, finnhub_company_news,
    fred_macro_snapshot, india_macro_snapshot, yfinance_news,
    sec_insider_transactions,
    wsb_sentiment,
)

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


# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* =========================================================
   DESIGN TOKENS — SINGLE DARK THEME
========================================================= */

:root {
  /* Surface hierarchy — clearly distinct levels */
  --bg-base:       #080f1e;
  --bg-surface:    #0d1a2e;
  --bg-elevated:   #132338;
  --bg-card:       #0f1e33;

  /* Borders — visible but not harsh */
  --border:        #1e3a5f;
  --border-subtle: #162d4a;

  /* Brand accent */
  --accent:        #3b82f6;
  --accent-dim:    #1d4ed8;
  --accent-glow:   rgba(59,130,246,0.24);
  --accent-glow2:  rgba(59,130,246,0.09);

  /* Semantic colors */
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

  /* Typography */
  --text-primary:  #e8f0fe;
  --text-secondary:#7a98c0;
  --text-muted:    #3d5a7a;

  /* Shape */
  --radius:        12px;
  --radius-sm:     8px;
  --radius-xs:     5px;

  /* Elevation */
  --shadow:        0 4px 24px rgba(0,0,0,0.55);
  --shadow-lg:     0 10px 48px rgba(0,0,0,0.65);
  --shadow-accent: 0 8px 32px rgba(59,130,246,0.22);
}

/* =========================================================
   TYPE SCALE
   10px — badges/labels
   11px — captions/metadata
   12px — body small / table data
   13px — body default
   14px — card titles
   16px — section values
   24px — metric numbers
   28px — page title
========================================================= */

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

.main, .block-container {
  background: var(--bg-base) !important;
  padding-top: 1.25rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Headings ── */
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

/* ── Live pulse ── */
@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(34,197,94,0.6); }
  50%       { opacity: 0.7; box-shadow: 0 0 0 6px rgba(34,197,94,0); }
}

/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}

/* Sidebar section labels */
.sidebar-group-label {
  font-size: 9px !important;
  font-weight: 700 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
  padding: 14px 0 6px 2px !important;
  display: block !important;
}

.sidebar-divider {
  border: none !important;
  border-top: 1px solid var(--border-subtle) !important;
  margin: 6px 0 10px 0 !important;
}

/* =========================================================
   METRIC CARDS
========================================================= */

div[data-testid="stMetric"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border-subtle) !important;
  border-top: 2px solid var(--accent) !important;
  border-radius: var(--radius) !important;
  padding: 18px 20px 16px !important;
  min-height: 108px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
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
  transform: translateY(-3px) !important;
  box-shadow: var(--shadow-accent) !important;
}

div[data-testid="stMetricValue"] {
  font-size: 1.65rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  letter-spacing: -0.04em !important;
  font-family: 'Inter', sans-serif !important;
  line-height: 1.1 !important;
}

div[data-testid="stMetricLabel"] > div {
  font-size: 10px !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.11em !important;
  color: var(--text-muted) !important;
  margin-bottom: 6px !important;
}

div[data-testid="stMetricDelta"] {
  font-size: 12px !important;
  font-weight: 600 !important;
}

/* =========================================================
   TABS
========================================================= */

.stTabs [data-baseweb="tab-list"] {
  gap: 2px !important;
  background: var(--bg-surface) !important;
  padding: 4px !important;
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  margin-bottom: 1.75rem !important;
  box-shadow: var(--shadow) !important;
}

.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-sm) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 8px 16px !important;
  color: var(--text-muted) !important;
  transition: all 0.16s ease !important;
  letter-spacing: 0.01em !important;
  border-bottom: none !important;
}

.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-secondary) !important;
  background: var(--bg-elevated) !important;
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dim) 100%) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 16px var(--accent-glow) !important;
  border-bottom: none !important;
}

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

/* Sliders */
.stSlider > div > div > div > div {
  background: linear-gradient(90deg, var(--accent), var(--accent-dim)) !important;
}

/* File uploader */
.stFileUploader {
  border-radius: var(--radius) !important;
}

/* Expanders */
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
        background:linear-gradient(135deg,#1d4ed8 0%,#3b82f6 55%,#60a5fa 100%);
        display:flex;align-items:center;justify-content:center;font-size:24px;
        box-shadow:0 6px 28px rgba(59,130,246,0.55);">
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
@st.cache_data(ttl=1800,  show_spinner=False)
def cached_wsb(tickers):                     return wsb_sentiment(list(tickers))


# ==========================================================
# VOLATILITY REGIME
# ==========================================================

@st.cache_data(show_spinner=False)
def detect_vol_regime(returns, window=60):
    rv = (returns.rolling(window).std() * np.sqrt(TRADING_DAYS)).dropna()
    if rv.empty: return "N/A", "#64748b", 0.0
    latest, pct = float(rv.iloc[-1]), float(rv.rank(pct=True).iloc[-1])
    return ("LOW VOL","#22c55e",latest) if pct<.33 else ("NORMAL VOL","#f59e0b",latest) if pct<.66 else ("HIGH VOL","#ef4444",latest)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown("""
    <div style="padding:18px 4px 14px;border-bottom:1px solid var(--border-subtle);margin-bottom:4px;">
      <div style="font-size:15px;font-weight:700;color:var(--text-primary);
          letter-spacing:-0.02em;">Portfolio Analyser</div>
      <div style="font-size:10px;color:var(--text-muted);margin-top:3px;
          text-transform:uppercase;letter-spacing:0.1em;">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    # ── DATA ───────────────────────────────────────────────
    st.markdown('<span class="sidebar-group-label">Data</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Portfolio CSV", type="csv", label_visibility="collapsed")

    # ── BENCHMARK & PROFILE ────────────────────────────────
    st.markdown('<hr class="sidebar-divider"><span class="sidebar-group-label">Market Settings</span>', unsafe_allow_html=True)
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

    risk_profile = st.selectbox("Risk Profile", list(RISK_PROFILES.keys()))

    # ── REBALANCING ────────────────────────────────────────
    st.markdown('<hr class="sidebar-divider"><span class="sidebar-group-label">Rebalancing</span>', unsafe_allow_html=True)
    threshold_pct  = st.slider("Rebalance Threshold (%)", 0.0, 10.0,  2.0, 0.1)
    max_weight_pct = st.slider("Max Position Size (%)",   5.0, 50.0, 15.0, 1.0)

px.defaults.template = "portfolio_dark"

# ── API keys from .streamlit/secrets.toml ──────────────────
try:
    _FINNHUB_KEY = st.secrets.get("FINNHUB_KEY", "") or "d6ljirpr01qrq6i31170d6ljirpr01qrq6i3117g"
    _FRED_KEY    = st.secrets.get("FRED_KEY",    "") or "1887e2a54d8ecf8a37c4df1799d4b3bb"
except Exception:
    _FINNHUB_KEY = "d6ljirpr01qrq6i31170d6ljirpr01qrq6i3117g"
    _FRED_KEY    = "1887e2a54d8ecf8a37c4df1799d4b3bb"

threshold        = threshold_pct        / 100
max_weight       = max_weight_pct       / 100
lookback         = "max"

if uploaded_file is None:
    st.session_state.pop("data_loaded",    None)
    st.session_state.pop("selected_asset", None)
    st.markdown("""
    <div style="max-width:660px;margin:48px auto 0;text-align:center;">

      <!-- Icon -->
      <div style="width:76px;height:76px;margin:0 auto 24px;border-radius:20px;
          background:linear-gradient(135deg,#1d4ed8,#3b82f6);
          display:flex;align-items:center;justify-content:center;font-size:36px;
          box-shadow:0 8px 36px rgba(59,130,246,0.45);position:relative;">
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
                curr_ret  = weights_series @ (returns.mean() * TRADING_DAYS),
                curr_vol  = float(np.sqrt(weights_series @ (returns.cov() * TRADING_DAYS) @ weights_series)),
                opt_ret   = float(optimal_weights @ (returns.mean() * TRADING_DAYS)) if optimal_weights is not None else 0,
                opt_vol   = float(np.sqrt(optimal_weights @ (returns.cov() * TRADING_DAYS) @ optimal_weights)) if optimal_weights is not None else 0,
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
# TABS
# ==========================================================

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📊  Overview",
    "⚠️  Risk",
    "🎯  Optimization",
    "📈  Performance",
    "🔍  Asset Analytics",
    "✨  Enhancement",
])


# ── TAB 1: OVERVIEW ────────────────────────────────────────
with tab1:
    stat_banner([
        ("Volatility Regime", regime),
        ("Current Vol",       f"{latest_vol:.1%}"),
        ("Health Score",      health_score),
    ], accent=regime_color)

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
    xirr_display = f"{portfolio_xirr:.2%}" if portfolio_xirr and not np.isnan(portfolio_xirr) else "N/A"
    cA.metric("XIRR (Portfolio)", xirr_display)
    cB.metric("Amount Invested",  f"{_currency}{amount_invested:,.0f}")
    unreal_pct = (unrealized_gain / amount_invested) if amount_invested > 0 else None
    cC.metric("Unrealised P/L",   f"{_currency}{unrealized_gain:,.0f}",
              delta=f"{unreal_pct:.2%}" if unreal_pct is not None else None)
    cD.metric("Realised P/L",     f"{_currency}{realised_gain:,.0f}")

    _ov_tf = st.radio("Timeframe", ["1M","3M","6M","1Y","3Y","5Y"], horizontal=True,
                      index=3, key="overview_tf", label_visibility="collapsed")

    _ov_c1, _ov_c2 = st.columns(2)
    with _ov_c1:
        section_header("Performance History")
        _cum_full = (1 + portfolio_returns.dropna()).cumprod()
        _cum      = slice_tf(_cum_full, _ov_tf)
        if _cum.empty:
            _cum = _cum_full
        _cum      = _cum / _cum.iloc[0] - 1
        _perf_df  = pd.DataFrame({"Portfolio": _cum})
        if benchmark_returns is not None:
            _bc_full = (1 + benchmark_returns.dropna()).cumprod()
            _bc      = slice_tf(_bc_full, _ov_tf).reindex(_cum.index, method="ffill")
            if not _bc.empty:
                _bc = _bc / _bc.iloc[0] - 1
                _perf_df["Benchmark"] = _bc
        _fig_perf = px.line(_perf_df, color_discrete_map={"Portfolio":"#3b82f6","Benchmark":"#64748b"})
        _fig_perf.update_traces(line=dict(width=1.8))
        quick_chart(_fig_perf, 260)

    with _ov_c2:
        section_header("Drawdown")
        _dd_sliced = slice_tf(drawdown_series, _ov_tf)
        fig_dd = px.area(_dd_sliced, color_discrete_sequence=["#ef4444"])
        fig_dd.update_traces(fill="tozeroy", fillcolor="rgba(239,68,68,0.12)")
        quick_chart(fig_dd, 260)

    section_header("Allocation")
    cX, cY = st.columns(2)
    with cX:
        at = df.groupby("Asset Type")["Market Value"].sum().reset_index()
        if not at.empty:
            fig = px.pie(at, names="Asset Type", values="Market Value", hole=0.65)
            fig.update_traces(textfont_size=12, marker=dict(line=dict(color="#080d18", width=2)))
            fig.update_layout(title=dict(text="Asset Allocation",x=0.5), height=380,
                              margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="v",x=1.02))
            st.plotly_chart(fig, use_container_width=True)
        else:
            empty_state("📦","Asset type data unavailable","Could not classify asset types")

    with cY:
        sa = df[df["Sector"]!="Unknown"].groupby("Sector")["Market Value"].sum().reset_index()
        if not sa.empty:
            fig = px.pie(sa, names="Sector", values="Market Value", hole=0.65)
            fig.update_traces(textfont_size=12, marker=dict(line=dict(color="#080d18", width=2)))
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
        "Quantity":      "{:,.0f}",
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
    section_header("Risk Summary", color="var(--negative)")
    pct_fields = {"Annual Return","Volatility","Max Drawdown","VaR 95%","CVaR 95%","Tracking Error","Correlation"}
    st.table(pd.DataFrame({k: f"{v:.2%}" if k in pct_fields else f"{v:.2f}"
                            for k,v in risk_summary.items()}.items(), columns=["Metric","Value"]))

    # ── Macro Environment ───────────────────────────────────
    section_header("Macro Environment", color="var(--warning)")
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Rolling charts ─────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        section_header("Rolling Volatility", color="var(--warning)")
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
    section_header("Value at Risk & Expected Shortfall", color="var(--negative)")

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
            # Build per-ticker treemap data
            _treemap_df = df[df["Sector"] != "Unknown"][["Ticker","Sector","Current Weight"]].copy()
            _treemap_df = _treemap_df.rename(columns={"Current Weight": "Weight"})

            # Assign a distinct colour per sector
            _SECTOR_COLOURS = {
                "Technology":             "#3b82f6",
                "Communication Services": "#8b5cf6",
                "Consumer Discretionary": "#f59e0b",
                "Consumer Staples":       "#10b981",
                "Financials":             "#06b6d4",
                "Financial Services":     "#06b6d4",
                "Healthcare":             "#ec4899",
                "Industrials":            "#f97316",
                "Energy":                 "#eab308",
                "Utilities":              "#14b8a6",
                "Real Estate":            "#a78bfa",
                "Materials":              "#84cc16",
                "ETF":                    "#64748b",
                "Index":                  "#94a3b8",
            }
            _default_palette = ["#3b82f6","#8b5cf6","#f59e0b","#10b981",
                                 "#06b6d4","#ec4899","#f97316","#eab308",
                                 "#14b8a6","#a78bfa","#84cc16","#64748b"]
            _sectors = _treemap_df["Sector"].unique().tolist()
            _colour_map = {}
            for i, s in enumerate(_sectors):
                _colour_map[s] = _SECTOR_COLOURS.get(s, _default_palette[i % len(_default_palette)])

            # Assign colour to each row — sector-level colour, slightly dimmed for tickers
            def _tile_colour(row):
                base = _colour_map.get(row["Sector"], "#3b82f6")
                return base
            _treemap_df["Color"] = _treemap_df.apply(_tile_colour, axis=1)

            _sec_fig = px.treemap(
                _treemap_df,
                path=["Sector", "Ticker"],
                values="Weight",
                color="Sector",
                color_discrete_map=_colour_map,
            )
            _sec_fig.update_traces(
                texttemplate="<b>%{label}</b><br>%{value:.1%}",
                hovertemplate="<b>%{label}</b><br>Weight: %{value:.2%}<extra></extra>",
                textfont=dict(size=13, family="Inter, Segoe UI, sans-serif", color="white"),
                marker=dict(
                    line=dict(width=3, color="#0f172a"),
                    pad=dict(t=28, l=6, r=6, b=6),
                    cornerradius=6,
                ),
                root_color="#0f172a",
            )
            _sec_fig.update_layout(
                height=420,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, Segoe UI, sans-serif", color="white"),
            )
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

    section_header("Optimisation Method", color="var(--cyan)")

    # ── Method selector: on_change busts the opt cache, no manual rerun needed ──
    def _on_method_change():
        for k in ("_opt_key","optimal_weights"): st.session_state.pop(k, None)
        st.session_state["opt_method"] = st.session_state.get("opt_method_radio", st.session_state.get("opt_method","Max Sharpe"))

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
    <div style="display:flex;align-items:flex-start;gap:18px;padding:18px 22px;margin-top:16px;
        border-radius:var(--radius);
        background:linear-gradient(135deg,var(--bg-surface) 0%,var(--bg-elevated) 100%);
        border:1px solid var(--border);border-left:3px solid var(--accent);
        margin-bottom:24px;box-shadow:var(--shadow);">
      <div style="font-size:30px;line-height:1;flex-shrink:0;margin-top:2px;">{info.get('icon','🎯')}</div>
      <div>
        <div style="font-size:14px;font-weight:600;color:var(--text-primary);
            margin-bottom:5px;letter-spacing:-0.01em;">
            {opt_method}
            <span style="font-size:10px;font-weight:700;color:var(--accent);
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
            .map(_style_action, subset=["Action"]),
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
            cov_ann  = returns.cov().values * TRADING_DAYS
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
            cov_ann    = returns.cov().values * TRADING_DAYS
            asset_vols = np.sqrt(np.diag(cov_ann))
            w_opt      = optimal_weights.values
            w_cur      = weights_series.reindex(optimal_weights.index).fillna(0).values
            dr_opt  = float(w_opt @ asset_vols) / float(np.sqrt(w_opt @ cov_ann @ w_opt))
            dr_cur  = float(w_cur @ asset_vols) / float(np.sqrt(w_cur @ cov_ann @ w_cur))
            d1, d2  = st.columns(2)
            d1.metric("Current DR",   f"{dr_cur:.3f}")
            d2.metric("Optimized DR", f"{dr_opt:.3f}", delta=f"{dr_opt-dr_cur:+.3f}")

        # ── Efficient frontier ─────────────────────────────
        section_header("Efficient Frontier", color="var(--purple)")
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
            height=580, margin=dict(l=40,r=40,t=20,b=40), showlegend=True,
            legend=dict(orientation="v",yanchor="bottom",y=0.04,xanchor="left",x=0.02,
                        bgcolor="rgba(11,17,32,0.92)",
                        bordercolor="#3b82f6", borderwidth=1,
                        font=dict(color="#e2e8f0", size=12)))
        st.plotly_chart(fig_f, use_container_width=True, key="efficient_frontier")

    else:
        empty_state("🎯","Optimization unavailable","Could not compute optimal weights for this portfolio")


# ── TAB 4: PERFORMANCE ─────────────────────────────────────
with tab4:
    section_header("Performance Metrics", color="var(--positive)")
    pm_tf = st.radio("Metrics Timeframe", ["1M","3M","6M","1Y","3Y","5Y","All"],
                     horizontal=True, index=3, key="pm_timeframe", label_visibility="collapsed")
    _pr_sliced = slice_tf(portfolio_returns, pm_tf) if pm_tf != "All" else portfolio_returns
    _br_sliced = slice_tf(benchmark_returns, pm_tf) if (benchmark_returns is not None and pm_tf != "All") else benchmark_returns
    pm = get_performance_metrics(_pr_sliced, _br_sliced, rf_rate=_rf_rate)

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
    section_header("Cumulative Returns", color="var(--positive)")
    tf = st.radio("Timeframe",["1M","3M","6M","1Y","3Y","5Y"],horizontal=True,
                  key="perf_timeframe",label_visibility="collapsed")

    # ── Build cumulative return series, rebased to 0 at start of window ──
    # Use (1+r).cumprod() so we stay in price-relative space, then rebase:
    # rebase = series / series.iloc[0] - 1  (correct ratio rebase, not subtraction)
    cum_full = (1 + portfolio_returns.dropna()).cumprod()
    cum      = slice_tf(cum_full, tf)
    if cum.empty:
        cum = cum_full
    cum      = cum / cum.iloc[0] - 1          # rebase: 0% at window start

    perf_df = pd.DataFrame({"Portfolio": cum})

    if benchmark_returns is not None:
        bc_full = (1 + benchmark_returns.dropna()).cumprod()
        bc      = slice_tf(bc_full, tf)
        # Align benchmark to same start date as portfolio
        bc      = bc.reindex(cum.index, method="ffill")
        if not bc.empty:
            bc = bc / bc.iloc[0] - 1
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

    # ── Company Profile ─────────────────────────────────────
    _is_indian = selected_asset.endswith((".NS", ".BO"))
    _profile   = {} if _is_indian else (cached_finnhub_profile(selected_asset, _FINNHUB_KEY) if _FINNHUB_KEY else {})

    # For Indian stocks, build profile from yfinance info
    if not _profile:
        try:
            _yf_info = yf.Ticker(selected_asset).info
            _profile = {
                "name":                _yf_info.get("longName") or _yf_info.get("shortName") or selected_asset,
                "exchange":            "NSE" if selected_asset.endswith(".NS") else ("BSE" if selected_asset.endswith(".BO") else _yf_info.get("exchange", "—")),
                "finnhubIndustry":     _yf_info.get("industry") or _yf_info.get("sector") or "—",
                "country":             _yf_info.get("country", "India" if _is_indian else "—"),
                "ipo":                 str(_yf_info.get("ipoExpectedDate", "—")),
                "marketCapitalization":(_yf_info.get("marketCap", 0) or 0) / 1e6,  # convert to millions to match Finnhub unit
                "weburl":              _yf_info.get("website", ""),
                "logo":                _yf_info.get("logo_url", ""),
            }
        except Exception:
            pass

    if _profile and _profile.get("name"):
        _logo_html = (f'<img src="{_profile["logo"]}" style="width:40px;height:40px;'
                      f'border-radius:8px;object-fit:contain;background:#fff;padding:3px;">'
                      ) if _profile.get("logo") else ""
        _mktcap = _profile.get("marketCapitalization", 0)  # in millions (USD for Finnhub, INR/M for yf fallback)
        if _is_indian and _mktcap:
            # yfinance returns market cap in INR; convert millions → crores (1 crore = 10M INR)
            _mktcap_raw = (_mktcap * 1e6)   # back to INR
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
            margin-bottom:8px;border-radius:var(--radius);
            background:var(--bg-surface);border:1px solid var(--border);
            box-shadow:var(--shadow);">
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
                <span style="color:var(--text-secondary);font-weight:600;">IPO</span>
                &nbsp;{_profile.get('ipo','—')}</span>
              <span style="font-size:11px;color:var(--text-muted);">
                <span style="color:var(--text-secondary);font-weight:600;">Mkt Cap</span>
                &nbsp;{_mktcap_str}</span>
              <span style="font-size:11px;">{_web_html}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

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

    # ── Analyst Consensus ─────────────────────────────────────
    section_header("Analyst Consensus", color="var(--accent)")
    if _is_indian:
        st.info("Analyst ratings are not available for NSE/BSE-listed stocks on the free data tier.")
    else:
        _fh_recs = cached_finnhub_recs(selected_asset, _FINNHUB_KEY)
        if not _fh_recs.empty:
            _rec_fig = go.Figure([
                go.Bar(name="Strong Buy",  x=_fh_recs["Period"], y=_fh_recs["Strong Buy"],  marker_color="#15803d"),
                go.Bar(name="Buy",         x=_fh_recs["Period"], y=_fh_recs["Buy"],          marker_color="#22c55e"),
                go.Bar(name="Hold",        x=_fh_recs["Period"], y=_fh_recs["Hold"],          marker_color="#f59e0b"),
                go.Bar(name="Sell",        x=_fh_recs["Period"], y=_fh_recs["Sell"],          marker_color="#ef4444"),
                go.Bar(name="Strong Sell", x=_fh_recs["Period"], y=_fh_recs["Strong Sell"],  marker_color="#7f1d1d"),
            ])
            _rec_fig.update_layout(barmode="stack", height=260,
                                   margin=dict(l=0,r=0,t=0,b=0),
                                   legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(_rec_fig, use_container_width=True)
        else:
            empty_state("📊", "Analyst ratings unavailable",
                        f"No analyst coverage found for {selected_asset}")

    # ── Earnings Surprises ────────────────────────────────────
    section_header("Earnings Surprises", color="var(--purple)")
    if _is_indian:
        st.info("Earnings surprise history is not available for NSE/BSE-listed stocks on the free data tier.")
    else:
        _fh_earn = cached_finnhub_earnings(selected_asset, _FINNHUB_KEY) if _FINNHUB_KEY else pd.DataFrame()
        if not _fh_earn.empty:
            def _style_result(val):
                if val == "Beat":    return "color:#22c55e;font-weight:600"
                if val == "Miss":    return "color:#ef4444;font-weight:600"
                return "color:#f59e0b;font-weight:600"
            st.dataframe(
                _fh_earn.style.map(_style_result, subset=["Result"]),
                use_container_width=True)
        else:
            empty_state("📊", "Earnings data unavailable",
                        f"No earnings history found for {selected_asset}")

    # ── Peer Companies ────────────────────────────────────────
    section_header("Peer Companies", color="var(--text-secondary)")
    if _is_indian:
        st.info("Peer data is not available for NSE/BSE-listed stocks on the free data tier.")
    else:
        _peers = cached_finnhub_peers(selected_asset, _FINNHUB_KEY) if _FINNHUB_KEY else []
        if _peers:
            _peer_pills = "".join([
                f'<span style="display:inline-block;padding:5px 12px;margin:4px;'
                f'background:var(--bg-elevated);border:1px solid var(--border);'
                f'border-radius:20px;font-size:12px;font-weight:600;'
                f'color:var(--text-secondary);letter-spacing:0.02em;">{p}</span>'
                for p in _peers if p != selected_asset
            ])
            st.markdown(f'<div style="margin:4px 0 8px;">{_peer_pills}</div>',
                        unsafe_allow_html=True)
        else:
            st.caption("No peer data available.")

    # ── WallStreetBets Sentiment ───────────────────────────
    if not _is_indian:
        section_header("Reddit / WallStreetBets", color="var(--accent)")
        _wsb_df  = cached_wsb(tickers_tuple)
        _clean_t = selected_asset.upper().replace(".NS", "").replace(".BO", "")
        _wsb_row = _wsb_df[_wsb_df["Ticker"] == _clean_t] if not _wsb_df.empty else pd.DataFrame()
        if not _wsb_row.empty:
            _w = _wsb_row.iloc[0]
            _ws1, _ws2, _ws3, _ws4 = st.columns(4)
            _ws1.metric("WSB Rank",  f"#{int(_w['WSB Rank'])}")
            _ws2.metric("Mentions",  f"{int(_w['Mentions']):,}")
            _ws3.metric("Sentiment", str(_w["Sentiment"]))
            _ws4.metric("Upvotes",   f"{int(_w['Upvotes']):,}")
        else:
            empty_state("📊", f"{selected_asset} not trending on WSB",
                        "Not in today's top mentions on r/wallstreetbets")

    # ── News Feed ──────────────────────────────────────────
    section_header(f"Latest News — {selected_asset}")
    _news_df = cached_yf_news(selected_asset)
    if not _news_df.empty:
        for _, _nrow in _news_df.head(12).iterrows():
            _ts_str = (_nrow["datetime"].strftime("%d %b %Y, %H:%M UTC")
                       if pd.notna(_nrow["datetime"]) else "")
            st.markdown(f"""
            <div style="padding:14px 18px;margin-bottom:8px;border-radius:var(--radius);
                background:var(--bg-surface);border:1px solid var(--border);
                border-left:3px solid var(--accent);">
              <div style="font-size:13px;font-weight:600;color:var(--text-primary);
                  margin-bottom:6px;line-height:1.5;">
                <a href="{_nrow['url']}" target="_blank"
                   style="color:var(--text-primary);text-decoration:none;">
                  {_nrow['headline']}
                </a>
              </div>
              <div style="display:flex;align-items:center;gap:8px;">
                <div style="font-size:10px;font-weight:600;color:var(--accent);
                    text-transform:uppercase;letter-spacing:0.06em;">{_nrow['source']}</div>
                {f'<div style="width:3px;height:3px;border-radius:50%;background:var(--text-muted);"></div>'
                 f'<div style="font-size:10px;color:var(--text-muted);">{_ts_str}</div>'
                 if _ts_str else ''}
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        empty_state("📰", "No recent news", f"No news found for {selected_asset} in the last 30 days")


# ── TAB 6: ENHANCEMENT ─────────────────────────────────────
with tab6:

    # ── Portfolio-wide Analyst Consensus ─────────────────────
    section_header("Analyst Consensus — All Holdings", color="var(--accent)")
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

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    _screener_universe = "NIFTY 500" if _market == "IN" else "S&P 500"
    st.markdown(f"""
    <div style="padding:18px 22px;margin-bottom:24px;border-radius:var(--radius);
        background:linear-gradient(135deg,var(--bg-surface) 0%,var(--bg-elevated) 100%);
        border:1px solid var(--border);border-left:3px solid var(--accent);
        box-shadow:var(--shadow);">
      <div style="font-size:14px;font-weight:600;color:var(--text-primary);
          margin-bottom:6px;letter-spacing:-0.01em;">Sector-Wise Enhancement Screener — {_screener_universe}</div>
      <div style="font-size:12px;color:var(--text-muted);line-height:1.65;">
          Top performing sectors and their best stocks ranked by 6 &amp; 12-month returns.
          Includes PE ratios and ROE. Refreshed every hour.</div>
    </div>""", unsafe_allow_html=True)

    section_header("3-Month Relative Performance — Current Holdings")
    try:
        with st.spinner("Computing 3M relative performance…"):
            pm_df = cached_3m_relative_performance(tickers_tuple, benchmark)

        def _rule_engine(x):
            if pd.isna(x): return "No Data"
            return "Sell" if x < -0.10 else "Buy" if x > 0.20 else "Hold"

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
        sector_recs = cached_sector_recommendations(_market)

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