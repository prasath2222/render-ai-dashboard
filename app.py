import os, json, warnings, time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import joblib
import ta
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="RNDR AI Terminal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════
BG       = "#060910"
BG2      = "#0b0f1a"
BG3      = "#111827"
BORDER   = "#1e2433"
BORDER2  = "#2a3347"
TXT      = "#e2e8f0"
TXT2     = "#94a3b8"
TXT3     = "#475569"
GREEN    = "#22d3a0"
RED      = "#f43f5e"
BLUE     = "#3b82f6"
ORANGE   = "#f97316"
PURPLE   = "#a78bfa"
YELLOW   = "#eab308"
TEAL     = "#06b6d4"
CYAN     = "#67e8f9"
PINK     = "#ec4899"

# ═══════════════════════════════════════════════════════
# GLOBAL CSS — Premium Dark Terminal UI
# ═══════════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; }}

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif !important;
    background: {BG} !important;
    color: {TXT} !important;
}}
.stApp {{ background: {BG} !important; }}
.block-container {{
    padding: 0 24px 60px 24px !important;
    max-width: 1700px !important;
}}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {{
    background: {BG2} !important;
    border-right: 1px solid {BORDER} !important;
    width: 240px !important;
}}
section[data-testid="stSidebar"] * {{ color: {TXT2} !important; }}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: {BG3} !important;
    border: 1px solid {BORDER2} !important;
    border-radius: 8px !important;
    color: {TXT} !important;
    font-size: 13px !important;
}}
.sidebar-section {{
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: {TXT3};
    text-transform: uppercase;
    padding: 18px 0 8px;
    font-family: 'Space Mono', monospace;
}}
.sidebar-logo {{
    padding: 20px 0 24px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 4px;
}}
.sidebar-logo .name {{
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: #fff;
    letter-spacing: -0.3px;
    line-height: 1.1;
}}
.sidebar-logo .sub {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TXT3};
    letter-spacing: 0.12em;
    margin-top: 3px;
}}

/* ── TOGGLE / CHECKBOX ── */
.stToggle > label > div {{ background: {BORDER2} !important; }}
.stCheckbox span {{ color: {TXT2} !important; font-size: 13px !important; }}

/* ── BUTTON ── */
.stButton > button {{
    background: {BG3} !important;
    border: 1px solid {BORDER2} !important;
    color: {TXT} !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 10px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em;
}}
.stButton > button:hover {{
    background: {BORDER} !important;
    border-color: {BLUE} !important;
    color: {BLUE} !important;
}}

/* ── PLOTLY ── */
.js-plotly-plot .plotly {{ border-radius: 12px; }}
.modebar-container {{ background: transparent !important; }}
.modebar-btn path {{ fill: {TXT3} !important; }}
.modebar-btn:hover path {{ fill: {BLUE} !important; }}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER2}; border-radius: 4px; }}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header {{ visibility: hidden; }}
div[data-testid="stDecoration"] {{ display: none; }}
div[data-testid="stToolbar"] {{ display: none; }}

/* ── SPINNER ── */
.stSpinner > div {{ border-top-color: {BLUE} !important; }}

/* ── SECTION HEADER ── */
.section-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px;
}}
.section-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: {ORANGE};
    box-shadow: 0 0 8px {ORANGE}80;
    flex-shrink: 0;
}}
.section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: {TXT3};
    text-transform: uppercase;
}}
.section-line {{
    flex: 1;
    height: 1px;
    background: {BORDER};
}}
.section-hint {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: {TXT3};
}}

/* ── METRIC CARD ── */
.metric-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    min-height: 100px;
    transition: border-color 0.2s;
}}
.metric-card:hover {{ border-color: {BORDER2}; }}
.metric-card-accent {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}}
.metric-label {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: {TXT3};
    text-transform: uppercase;
    margin-bottom: 10px;
}}
.metric-value {{
    font-family: 'Syne', sans-serif;
    font-size: 24px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.5px;
}}
.metric-sub {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: {TXT2};
    margin-top: 6px;
}}

/* ── INDICATOR CARD ── */
.ind-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 14px 16px;
    border-left-width: 2px;
}}
.ind-label {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {TXT3};
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.ind-value {{
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
}}
.ind-sig {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 600;
    margin-top: 4px;
}}

/* ── LIVE BADGE ── */
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.85); }}
}}
.live-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: {GREEN};
    animation: pulse 1.6s ease-in-out infinite;
}}

/* ── SIGNAL BADGE ── */
.sig-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
}}

/* ── TABLE ── */
.data-table {{
    width: 100%;
    border-collapse: collapse;
}}
.data-table th {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: {TXT3};
    text-transform: uppercase;
    padding-bottom: 10px;
}}
.data-table td {{
    font-size: 12px;
    padding: 9px 0;
    border-top: 1px solid {BORDER};
    vertical-align: middle;
}}

/* ── PRICE TICKER ANIMATION ── */
@keyframes flash-green {{
    0% {{ color: {GREEN}; }}
    100% {{ color: inherit; }}
}}
@keyframes flash-red {{
    0% {{ color: {RED}; }}
    100% {{ color: inherit; }}
}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# PLOTLY LAYOUT HELPER
# ═══════════════════════════════════════════════════════
def _layout(fig, h=400, title="", margin=None):
    m = margin or dict(l=8, r=8, t=28 if title else 8, b=8)
    fig.update_layout(
        height=h,
        paper_bgcolor=BG2,
        plot_bgcolor=BG2,
        font=dict(family="DM Sans", color=TXT2, size=11),
        margin=m,
        title=dict(
            text=title,
            font=dict(family="Space Mono", size=10, color=TXT3),
            x=0.01, y=0.99
        ) if title else None,
        xaxis=dict(
            gridcolor=BORDER, showgrid=True, zeroline=False, showline=False,
            tickfont=dict(family="Space Mono", size=9, color=TXT3),
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            gridcolor=BORDER, showgrid=True, zeroline=False, showline=False,
            tickfont=dict(family="Space Mono", size=9, color=TXT3),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(family="DM Sans", size=10, color=TXT2),
            orientation="h", y=1.05, x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=BG3, bordercolor=BORDER2,
            font=dict(family="Space Mono", size=10, color=TXT),
        ),
        dragmode="pan",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TXT3, activecolor=BLUE),
    )
    return fig

CHART_CONFIG = dict(
    scrollZoom=True,
    displayModeBar=True,
    modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
    displaylogo=False,
    toImageButtonOptions=dict(format="png", scale=2),
)

# Crosshair cursor shape for all charts
CROSSHAIR_XAXIS = dict(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    spikecolor=TXT3,
    spikedash="dot",
    spikethickness=1,
)
CROSSHAIR_YAXIS = dict(
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    spikecolor=TXT3,
    spikedash="dot",
    spikethickness=1,
)

def apply_crosshair(fig, rows=1):
    for i in range(1, rows + 1):
        fig.update_xaxes(CROSSHAIR_XAXIS, row=i, col=1)
        fig.update_yaxes(CROSSHAIR_YAXIS, row=i, col=1)
    return fig

# ═══════════════════════════════════════════════════════
# HTML HELPERS
# ═══════════════════════════════════════════════════════
def section_header(title, hint=""):
    hint_html = f'<div class="section-hint">{hint}</div>' if hint else ""
    return f"""
    <div class="section-header">
      <div class="section-dot"></div>
      <div class="section-title">{title}</div>
      <div class="section-line"></div>
      {hint_html}
    </div>"""

def metric_card(label, value, sub, color, extra_style=""):
    return f"""
    <div class="metric-card" style="{extra_style}">
      <div class="metric-card-accent" style="background:{color};"></div>
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{color};">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>"""

def sig_badge_html(sig):
    c = GREEN if sig == "BUY" else RED if sig == "SELL" else YELLOW
    bg = f"{c}18"; border = f"{c}35"
    return f'<span class="sig-badge" style="background:{bg};color:{c};border:1px solid {border};">{sig}</span>'

def ind_card(label, value, sig_text, sig_color):
    return f"""
    <div class="ind-card" style="border-left-color:{sig_color};">
      <div class="ind-label">{label}</div>
      <div class="ind-value">{value}</div>
      <div class="ind-sig" style="color:{sig_color};">{sig_text}</div>
    </div>"""

# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════
@st.cache_data(ttl=30, show_spinner=False)
def get_coingecko():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=render-token&vs_currencies=usd"
            "&include_24hr_vol=true&include_24hr_change=true"
            "&include_market_cap=true&include_high_24h=true&include_low_24h=true",
            timeout=10,
        )
        return r.json().get("render-token", {})
    except Exception:
        return {}

@st.cache_data(ttl=120, show_spinner=False)
def get_inr_rate():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=8)
        return float(r.json()["rates"].get("INR", 84.0))
    except Exception:
        return 84.0

@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv(period="90d", interval="1h"):
    for ticker in ["RENDER-USD", "RNDR-USD"]:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            if len(df) > 50:
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return 50, "Neutral"

# ═══════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════
def add_indicators(df):
    def safe(s):
        return s.replace([np.inf, -np.inf], np.nan).fillna(0)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    df["EMA9"]   = safe(ta.trend.EMAIndicator(c, 9).ema_indicator())
    df["EMA20"]  = safe(ta.trend.EMAIndicator(c, 20).ema_indicator())
    df["EMA50"]  = safe(ta.trend.EMAIndicator(c, 50).ema_indicator())
    df["EMA200"] = safe(ta.trend.EMAIndicator(c, 200).ema_indicator())

    _macd = ta.trend.MACD(c)
    df["MACD"]      = safe(_macd.macd())
    df["MACD_SIG"]  = safe(_macd.macd_signal())
    df["MACD_HIST"] = safe(_macd.macd_diff())

    _adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["ADX"]     = safe(_adx.adx())
    df["ADX_POS"] = safe(_adx.adx_pos())
    df["ADX_NEG"] = safe(_adx.adx_neg())

    df["RSI"]    = safe(ta.momentum.RSIIndicator(c, 14).rsi())
    df["RSI7"]   = safe(ta.momentum.RSIIndicator(c, 7).rsi())

    _stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["STOCH_K"] = safe(_stoch.stoch())
    df["STOCH_D"] = safe(_stoch.stoch_signal())

    df["CCI"]      = safe(ta.trend.CCIIndicator(h, l, c, 20).cci())
    df["WILLIAMS"] = safe(ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r())
    df["ROC5"]     = safe(ta.momentum.ROCIndicator(c, 5).roc())
    df["ROC20"]    = safe(ta.momentum.ROCIndicator(c, 20).roc())

    _bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_U"] = safe(_bb.bollinger_hband())
    df["BB_L"] = safe(_bb.bollinger_lband())
    df["BB_M"] = safe(_bb.bollinger_mavg())
    df["BB_W"] = safe(_bb.bollinger_wband())
    df["BB_P"] = safe(_bb.bollinger_pband())

    df["ATR"] = safe(ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range())

    df["OBV"] = safe(ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume())
    df["MFI"] = safe(ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index())
    df["CMF"] = safe(ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow())

    try:
        df["VWAP"] = safe(ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price())
    except Exception:
        df["VWAP"] = c.copy()

    vm20 = v.rolling(20).mean().replace(0, np.nan)
    df["VOL_MA"] = safe(vm20)
    df["VOL_R"]  = safe(v / vm20)

    df["RET1"]  = safe(c.pct_change(1))
    df["RET4"]  = safe(c.pct_change(4))
    df["RET24"] = safe(c.pct_change(24))

    # ML compat aliases
    df["ema_9"]       = df["EMA9"]
    df["ema_21"]      = safe(ta.trend.EMAIndicator(c, 21).ema_indicator())
    df["ema_50"]      = df["EMA50"]
    df["sma_20"]      = safe(ta.trend.SMAIndicator(c, 20).sma_indicator())
    df["macd"]        = df["MACD"]
    df["macd_signal"] = df["MACD_SIG"]
    df["macd_diff"]   = df["MACD_HIST"]
    df["adx"]         = df["ADX"]
    df["adx_pos"]     = df["ADX_POS"]
    df["adx_neg"]     = df["ADX_NEG"]
    df["rsi_14"]      = df["RSI"]
    df["rsi_7"]       = df["RSI7"]
    df["rsi_28"]      = safe(ta.momentum.RSIIndicator(c, 28).rsi())
    df["stoch_k"]     = df["STOCH_K"]
    df["stoch_d"]     = df["STOCH_D"]
    df["cci"]         = df["CCI"]
    df["roc_5"]       = df["ROC5"]
    df["roc_20"]      = df["ROC20"]
    df["williams"]    = df["WILLIAMS"]
    df["bb_upper"]    = df["BB_U"]
    df["bb_lower"]    = df["BB_L"]
    df["bb_width"]    = df["BB_W"]
    df["bb_pct"]      = df["BB_P"]
    df["atr_14"]      = df["ATR"]
    df["obv"]         = df["OBV"]
    df["mfi"]         = df["MFI"]
    df["cmf"]         = df["CMF"]
    df["vwap"]        = df["VWAP"]
    df["ret_1h"]      = df["RET1"]
    df["ret_4h"]      = df["RET4"]
    df["ret_24h"]     = df["RET24"]
    df["log_ret"]     = safe(np.log(c / c.shift(1)))
    df["vol_24h"]     = safe(df["RET1"].rolling(24).std())
    df["vol_7d"]      = safe(df["RET1"].rolling(168).std())

    hl = (h - l).replace(0, np.nan)
    df["body"]       = safe((c - o).abs() / hl)
    df["upper_wick"] = safe((h - c.clip(lower=o)) / hl)
    df["lower_wick"] = safe((c.clip(upper=o) - l) / hl)
    df["hl_ratio"]   = safe(h / l)
    df["close_open"] = safe(c / o)

    vm24  = v.rolling(24).mean().replace(0, np.nan)
    vm168 = v.rolling(168).mean().replace(0, np.nan)
    df["vol_r24"]  = safe(v / vm24)
    df["vol_r168"] = safe(v / vm168)
    df["whale"]    = (df["vol_r24"] > 3).astype(int)

    df["dist_hi24"] = safe((h.rolling(24).max() - c) / c)
    df["dist_lo24"] = safe((c - l.rolling(24).min()) / c)
    df["dist_hi7d"] = safe((h.rolling(168).max() - c) / c)
    df["dist_lo7d"] = safe((c - l.rolling(168).min()) / c)

    df["ema_score"]     = ((df["ema_9"] > df["ema_21"]).astype(int) +
                           (df["ema_21"] > df["ema_50"]).astype(int))
    df["price_vs_vwap"] = safe((c - df["vwap"]) / df["vwap"])
    df["fng"]           = 50
    df["fng_fear"]      = 0
    df["fng_greed"]     = 0
    df["btc_ret"]       = 0.0
    df["rndr_vs_btc"]   = 0.0
    df["btc_ret_4h"]    = 0.0
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df

# ═══════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════
MODEL_FILES = ["render_ensemble_cls.pkl", "render_reg.pkl",
               "render_scaler.pkl", "render_features.pkl"]

def load_models():
    try:
        if all(os.path.exists(f) for f in MODEL_FILES):
            return (
                joblib.load("render_ensemble_cls.pkl"),
                joblib.load("render_reg.pkl"),
                joblib.load("render_scaler.pkl"),
                joblib.load("render_features.pkl"),
            )
    except Exception:
        pass
    return None, None, None, None

def ml_predict(df, ens, reg, scaler, feats):
    EXCL = {"Open", "High", "Low", "Close", "Volume",
            "future_close", "future_return", "label"}
    avail = [f for f in feats if f in df.columns and f not in EXCL]
    row   = df.iloc[-1][avail].values.astype(np.float64).reshape(1, -1)
    row   = np.nan_to_num(row)
    full  = np.zeros((1, len(feats)))
    idx   = {f: i for i, f in enumerate(feats)}
    for i, f in enumerate(avail):
        full[0, idx[f]] = row[0, i]
    Xs = scaler.transform(full)
    lmap = {0: "SELL", 1: "HOLD", 2: "BUY"}
    cls  = lmap[int(ens.predict(Xs)[0])]
    proba = ens.predict_proba(Xs)[0]
    price = float(reg.predict(Xs)[0])
    return cls, proba, price

# ═══════════════════════════════════════════════════════
# TREND SCORE
# ═══════════════════════════════════════════════════════
def compute_trend_score(latest, close_s):
    ts   = 0
    e20  = float(latest.get("EMA20",  0))
    e50  = float(latest.get("EMA50",  0))
    e200 = float(latest.get("EMA200", 0))
    rsi  = float(latest.get("RSI",   50))
    macd = float(latest.get("MACD",   0))
    msig = float(latest.get("MACD_SIG", 0))
    adx  = float(latest.get("ADX",    0))
    vol  = float(latest.get("Volume", 0))
    vma  = float(latest.get("VOL_MA", 1))

    if e20 > e50:   ts += 20
    else:           ts -= 20
    if e50 > e200:  ts += 20
    else:           ts -= 20
    if rsi >= 70:   ts -= 15
    elif rsi >= 60: ts += 15
    elif rsi <= 30: ts += 20
    elif rsi <= 40: ts -= 10
    if macd > msig: ts += 20
    else:           ts -= 20
    if adx > 25:    ts += 10
    if len(close_s) >= 10:
        pc = (close_s.iloc[-1] - close_s.iloc[-10]) / max(close_s.iloc[-10], 1e-9) * 100
        ts += 10 if pc > 0 else -10
    if vol > vma:   ts += 5
    return int(ts)

# ═══════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════
def chart_candles(df, show_ema, show_bb, show_vwap, show_vol):
    """Pure candlestick OHLCV chart — no indicators inside."""
    rows = 2 if show_vol else 1
    rh   = [0.78, 0.22] if show_vol else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.008, row_heights=rh,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].values, high=df["High"].values,
        low=df["Low"].values,   close=df["Close"].values,
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,   width=1), fillcolor=RED),
        name="RNDR", whiskerwidth=0.7,
    ), row=1, col=1)

    # VWAP
    if show_vwap and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"].values,
            name="VWAP", line=dict(color=ORANGE, width=1.2, dash="dot"),
            opacity=0.9, hoverinfo="skip",
        ), row=1, col=1)

    # EMA overlays
    if show_ema:
        for cn, col_, nm in [
            ("EMA9",  CYAN,   "EMA 9"),
            ("EMA20", GREEN,  "EMA 20"),
            ("EMA50", PURPLE, "EMA 50"),
            ("EMA200",ORANGE, "EMA 200"),
        ]:
            if cn in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[cn].values,
                    name=nm, line=dict(color=col_, width=1),
                    opacity=0.85, hoverinfo="skip",
                ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_U" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_U"].values, name="BB Upper",
            line=dict(color=BORDER2, width=0.8),
            hoverinfo="skip", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_L"].values, name="BB Lower",
            line=dict(color=BORDER2, width=0.8),
            fill="tonexty", fillcolor="rgba(42,51,71,0.15)",
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # Volume
    if show_vol and "Volume" in df.columns:
        vc = [GREEN if float(c_) >= float(o_) else RED
              for c_, o_ in zip(df["Close"].values, df["Open"].values)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"].values,
            marker_color=vc, name="Volume", opacity=0.55, showlegend=False,
        ), row=2, col=1)
        if "VOL_MA" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VOL_MA"].values,
                line=dict(color=ORANGE, width=1),
                name="Vol MA20", showlegend=False, hoverinfo="skip",
            ), row=2, col=1)

    _layout(fig, h=580)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        newshape=dict(line_color=BLUE),
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(
            showgrid=True, gridcolor=BORDER, gridwidth=0.5,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor=TXT3, spikedash="solid", spikethickness=1,
            row=i, col=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=BORDER, gridwidth=0.5,
            showspikes=True, spikemode="across", spikesnap="cursor",
            spikecolor=TXT3, spikedash="solid", spikethickness=1,
            row=i, col=1,
        )
    return fig


def chart_macd(df):
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in df["MACD_HIST"].values]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_HIST"].values,
        marker_color=colors, name="Histogram", opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"].values,
        line=dict(color=BLUE, width=1.6), name="MACD",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_SIG"].values,
        line=dict(color=ORANGE, width=1.6, dash="dot"), name="Signal",
    ))
    fig.add_hline(y=0, line_width=0.5, line_color=BORDER2)
    _layout(fig, h=220, title="MACD (12 · 26 · 9)")
    fig.update_xaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    return fig


def chart_rsi(df):
    fig = go.Figure()
    # RSI area fill
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"].values,
        line=dict(color=ORANGE, width=1.6), name="RSI 14",
        fill="tozeroy", fillcolor=f"{ORANGE}12",
    ))
    if "RSI7" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI7"].values,
            line=dict(color=CYAN, width=1, dash="dot"), name="RSI 7",
        ))
    for lvl, col in [(70, RED), (30, GREEN), (50, BORDER2)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col)
    _layout(fig, h=220, title="RSI (14 · 7)")
    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    return fig


def chart_stoch(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["STOCH_K"].values,
        line=dict(color=BLUE, width=1.5), name="%K",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["STOCH_D"].values,
        line=dict(color=ORANGE, width=1.5, dash="dot"), name="%D",
    ))
    for lvl, col in [(80, RED), (20, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col)
    _layout(fig, h=220, title="Stochastic (14 · 3 · 3)")
    fig.update_yaxes(range=[0, 100])
    fig.update_xaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    return fig


def chart_adx(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ADX"].values,
        line=dict(color=YELLOW, width=1.8), name="ADX",
        fill="tozeroy", fillcolor=f"{YELLOW}10",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ADX_POS"].values,
        line=dict(color=GREEN, width=1.1, dash="dot"), name="+DI",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ADX_NEG"].values,
        line=dict(color=RED, width=1.1, dash="dot"), name="−DI",
    ))
    fig.add_hline(y=25, line_width=0.7, line_dash="dash", line_color=BORDER2)
    _layout(fig, h=220, title="ADX — Trend Strength")
    fig.update_xaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1)
    return fig


def chart_bb(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_W"].values, name="BB Width",
        line=dict(color=PURPLE, width=1.6),
        fill="tozeroy", fillcolor=f"{PURPLE}10",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_P"].values, name="BB %B",
        line=dict(color=TEAL, width=1.4),
    ), row=2, col=1)
    for lvl, col in [(1, RED), (0, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.5, line_color=col, row=2, col=1)
    _layout(fig, h=280, title="Bollinger Bands — Width + %B")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
    return fig


def chart_volume_flow(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["OBV"].values,
        line=dict(color=BLUE, width=1.5), name="OBV",
        fill="tozeroy", fillcolor=f"{BLUE}10",
    ), row=1, col=1)
    cmf_colors = [GREEN if v >= 0 else RED for v in df["CMF"].values]
    fig.add_trace(go.Bar(
        x=df.index, y=df["CMF"].values,
        marker_color=cmf_colors, name="CMF", opacity=0.85,
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color=BORDER2, row=2, col=1)
    _layout(fig, h=290, title="Volume Flow — OBV + CMF Inflow/Outflow")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
    return fig


def chart_cci_williams(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    cci_colors = [GREEN if v <= -100 else RED if v >= 100 else TXT3 for v in df["CCI"].values]
    fig.add_trace(go.Bar(
        x=df.index, y=df["CCI"].values,
        marker_color=cci_colors, name="CCI", opacity=0.8,
    ), row=1, col=1)
    for lvl, col in [(100, RED), (-100, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["WILLIAMS"].values,
        line=dict(color=PINK, width=1.4), name="Williams %R",
    ), row=2, col=1)
    for lvl, col in [(-20, RED), (-80, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col, row=2, col=1)
    _layout(fig, h=280, title="CCI (20) + Williams %R (14)")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
    return fig


def chart_mfi_roc(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MFI"].values,
        line=dict(color=TEAL, width=1.5), name="MFI",
        fill="tozeroy", fillcolor=f"{TEAL}12",
    ), row=1, col=1)
    for lvl, col in [(80, RED), (20, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col, row=1, col=1)
    roc_colors = [GREEN if v >= 0 else RED for v in df["ROC5"].values]
    fig.add_trace(go.Bar(
        x=df.index, y=df["ROC5"].values,
        marker_color=roc_colors, name="ROC 5", opacity=0.8,
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color=BORDER2, row=2, col=1)
    _layout(fig, h=280, title="MFI (14) + ROC (5)")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, showspikes=True, spikemode="across", spikecolor=TXT3, spikedash="solid", spikethickness=1, row=i, col=1)
    return fig

# ═══════════════════════════════════════════════════════
# REGIME PANEL (HTML only — no Plotly gauge)
# ═══════════════════════════════════════════════════════
def regime_panel_html(score):
    clamped = max(-100, min(100, score))
    pct     = int((clamped + 100) / 2)
    color   = GREEN if clamped >= 30 else RED if clamped <= -30 else YELLOW
    label_  = "BULLISH" if clamped >= 30 else "BEARISH" if clamped <= -30 else "SIDEWAYS"
    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;
         padding:24px;text-align:center;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
           letter-spacing:0.16em;color:{TXT3};margin-bottom:18px;">MARKET REGIME SCORE</div>
      <div style="font-family:'Syne',sans-serif;font-size:52px;font-weight:800;
           color:{color};line-height:1;letter-spacing:-2px;">{clamped:+d}</div>
      <div style="font-family:'Space Mono',monospace;font-size:13px;font-weight:700;
           color:{color};letter-spacing:0.12em;margin:10px 0 20px;">{label_}</div>
      <div style="height:6px;background:{BORDER};border-radius:3px;overflow:hidden;margin-bottom:6px;">
        <div style="height:100%;width:{pct}%;background:linear-gradient(90deg,{RED},{YELLOW},{color});
             border-radius:3px;transition:width 0.6s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:{RED};">BEARISH</span>
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:{YELLOW};">NEUTRAL</span>
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:{GREEN};">BULLISH</span>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════
# AI PREDICTION CARDS (pure HTML — no raw code leaking)
# ═══════════════════════════════════════════════════════
def classification_card_html(cls_sig, cls_proba, conf, sig_col, inr_rate):
    p_sell = float(cls_proba[0]) * 100
    p_hold = float(cls_proba[1]) * 100
    p_buy  = float(cls_proba[2]) * 100

    # Direction arrow + label
    arrow = "↑" if cls_sig == "BUY" else "↓" if cls_sig == "SELL" else "→"
    move  = "UPWARD MOVE EXPECTED" if cls_sig == "BUY" else "DOWNWARD MOVE EXPECTED" if cls_sig == "SELL" else "SIDEWAYS EXPECTED"

    def prob_row(lbl, pct, col):
        return f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-family:'Space Mono',monospace;font-size:10px;
                 color:{TXT3};letter-spacing:0.1em;">{lbl}</span>
            <span style="font-family:'Space Mono',monospace;font-size:11px;
                 color:{col};font-weight:700;">{pct:.1f}%</span>
          </div>
          <div style="height:4px;background:{BORDER};border-radius:2px;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{col};border-radius:2px;
                 transition:width 0.5s;"></div>
          </div>
        </div>"""

    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;
         padding:24px;border-top:2px solid {sig_col};height:100%;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
           letter-spacing:0.16em;color:{TXT3};margin-bottom:16px;">CLASSIFICATION MODEL</div>
      <div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:6px;">
        <div style="font-family:'Syne',sans-serif;font-size:48px;font-weight:900;
             color:{sig_col};letter-spacing:-1px;line-height:1;">{cls_sig}</div>
        <div style="font-size:28px;color:{sig_col};margin-bottom:4px;">{arrow}</div>
      </div>
      <div style="font-family:'Space Mono',monospace;font-size:10px;color:{sig_col};
           letter-spacing:0.08em;margin-bottom:6px;">{move}</div>
      <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
           margin-bottom:22px;">XGBoost · LightGBM · RandomForest Ensemble</div>
      {prob_row("BUY",  p_buy,  GREEN)}
      {prob_row("HOLD", p_hold, YELLOW)}
      {prob_row("SELL", p_sell, RED)}
      <div style="margin-top:18px;padding-top:14px;border-top:1px solid {BORDER};
           display:flex;justify-content:space-between;align-items:center;">
        <span style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};">CONFIDENCE</span>
        <span style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;
             color:{sig_col};">{conf}%</span>
      </div>
    </div>"""


def regression_card_html(cur_price, reg_price, inr_rate, atr_v, ts_score):
    reg_chg = (reg_price / cur_price - 1) * 100
    reg_col = GREEN if reg_chg >= 0 else RED
    sigma   = atr_v * 2
    r1h     = cur_price * (1 + (ts_score / 100) * (atr_v / max(cur_price, 1e-9)) * 1)
    r4h     = cur_price * (1 + (ts_score / 100) * (atr_v / max(cur_price, 1e-9)) * 2)

    arrow_label = "↑ BULLISH" if reg_chg >= 0 else "↓ BEARISH"

    def row(hor, price):
        chg = (price / cur_price - 1) * 100
        cc  = GREEN if chg >= 0 else RED
        return f"""
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:10px;color:{TXT3};
               padding:10px 0;border-top:1px solid {BORDER};">{hor}</td>
          <td style="text-align:right;border-top:1px solid {BORDER};padding:10px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:12px;
                 font-weight:700;color:{cc};">${price:.4f}</div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;
                 color:{TXT3};">₹{price * inr_rate:,.2f} &nbsp;
                 <span style="color:{cc};">{chg:+.2f}%</span></div>
          </td>
        </tr>"""

    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;
         padding:24px;border-top:2px solid {reg_col};height:100%;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
           letter-spacing:0.16em;color:{TXT3};margin-bottom:16px;">REGRESSION FORECAST</div>
      <div style="font-family:'Syne',sans-serif;font-size:36px;font-weight:900;
           color:{reg_col};letter-spacing:-0.5px;line-height:1;margin-bottom:4px;">${reg_price:.4f}</div>
      <div style="font-family:'Space Mono',monospace;font-size:10px;color:{TXT2};
           margin-bottom:4px;">₹{reg_price * inr_rate:,.2f}</div>
      <div style="font-family:'Space Mono',monospace;font-size:11px;font-weight:700;
           color:{reg_col};margin-bottom:20px;">{arrow_label} &nbsp; {reg_chg:+.2f}% expected</div>
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:left;padding-bottom:8px;">HORIZON</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:right;padding-bottom:8px;">TARGET</th>
          </tr>
        </thead>
        <tbody>
          {row("1 Hour",   r1h)}
          {row("4 Hours",  r4h)}
          {row("24 Hours", reg_price)}
        </tbody>
      </table>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
             margin-bottom:5px;">CONFIDENCE BAND ±1σ</div>
        <div style="font-family:'Space Mono',monospace;font-size:11px;color:{TXT2};">
          ${max(0, reg_price - sigma):.4f} — ${reg_price + sigma:.4f}
        </div>
      </div>
    </div>"""


def sr_table_html(df, cur_price, inr_rate, atr_v):
    r1 = float(df["High"].tail(50).max())
    r2 = float(df["High"].tail(20).max())
    r3 = r1 + atr_v
    s1 = float(df["Low"].tail(50).min())
    s2 = float(df["Low"].tail(20).min())
    s3 = s1 - atr_v

    def dots(n, col):
        return "".join([f'<span style="color:{col};font-size:12px;">●</span>' for _ in range(n)])

    def sr_row(lbl, price, n, col):
        return f"""
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:11px;color:{col};
               font-weight:700;padding:10px 0;border-top:1px solid {BORDER};">{lbl}</td>
          <td style="text-align:center;border-top:1px solid {BORDER};padding:10px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:12px;
                 color:#f8fafc;font-weight:700;">${price:.4f}</div>
          </td>
          <td style="text-align:center;border-top:1px solid {BORDER};padding:10px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:11px;color:{TXT2};">
              ₹{price * inr_rate:,.2f}
            </div>
          </td>
          <td style="text-align:right;border-top:1px solid {BORDER};padding:10px 0;">
            {dots(n, col)}
          </td>
        </tr>"""

    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;padding:22px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:left;padding-bottom:10px;">LEVEL</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:center;padding-bottom:10px;">USD</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:center;padding-bottom:10px;">INR</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:right;padding-bottom:10px;">STRENGTH</th>
          </tr>
        </thead>
        <tbody>
          {sr_row("RESISTANCE 3", r3, 3, RED)}
          {sr_row("RESISTANCE 2", r1, 4, RED)}
          {sr_row("RESISTANCE 1", r2, 5, RED)}
          <tr>
            <td colspan="4" style="text-align:center;border-top:1px dashed {BORDER};
                 border-bottom:1px dashed {BORDER};padding:8px 0;">
              <span style="font-family:'Space Mono',monospace;font-size:10px;
                   color:{BLUE};font-weight:700;">
                ── CURRENT ${cur_price:.4f} · ₹{cur_price * inr_rate:,.2f} ──
              </span>
            </td>
          </tr>
          {sr_row("SUPPORT 1",    s2, 5, GREEN)}
          {sr_row("SUPPORT 2",    s1, 4, GREEN)}
          {sr_row("SUPPORT 3",    s3, 3, GREEN)}
        </tbody>
      </table>
    </div>"""


def trading_setup_html(cur_price, inr_rate, atr_v):
    sl   = cur_price - (atr_v * 2)
    tp1  = cur_price + (atr_v * 3)
    tp2  = cur_price + (atr_v * 6)
    rr   = (tp1 - cur_price) / max(cur_price - sl, 1e-9)

    def setup_row(lbl, usd, col):
        return f"""
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:10px;color:{TXT3};
               padding:10px 0;border-top:1px solid {BORDER};">{lbl}</td>
          <td style="text-align:right;border-top:1px solid {BORDER};padding:10px 0;">
            <div style="font-family:'Space Mono',monospace;font-size:13px;
                 font-weight:700;color:{col};">${usd:.4f}</div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;
                 color:{TXT3};">₹{usd * inr_rate:,.2f}</div>
          </td>
        </tr>"""

    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;padding:22px;">
      <div style="font-family:'Space Mono',monospace;font-size:9px;font-weight:700;
           letter-spacing:0.14em;color:{TXT3};margin-bottom:16px;">TRADING SETUP</div>
      <table style="width:100%;border-collapse:collapse;">
        {setup_row("ENTRY PRICE",          cur_price,  BLUE)}
        {setup_row("STOP LOSS  (2× ATR)",  sl,         RED)}
        {setup_row("TAKE PROFIT 1 (3× ATR)", tp1,      GREEN)}
        {setup_row("TAKE PROFIT 2 (6× ATR)", tp2,      GREEN)}
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:10px;color:{TXT3};
               padding:10px 0;border-top:1px solid {BORDER};">RISK / REWARD</td>
          <td style="text-align:right;border-top:1px solid {BORDER};padding:10px 0;">
            <span style="font-family:'Syne',sans-serif;font-size:20px;font-weight:800;
                 color:{ORANGE};">1 : {rr:.2f}</span>
          </td>
        </tr>
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:10px;color:{TXT3};
               padding:10px 0;border-top:1px solid {BORDER};">ATR (14)</td>
          <td style="text-align:right;border-top:1px solid {BORDER};padding:10px 0;">
            <span style="font-family:'Space Mono',monospace;font-size:12px;color:{TXT2};">
              ${atr_v:.4f} · ₹{atr_v * inr_rate:.2f}
            </span>
          </td>
        </tr>
      </table>
      <div style="margin-top:14px;padding:12px;background:{BG3};border-radius:8px;
           border:1px solid {BORDER};">
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
             margin-bottom:4px;letter-spacing:0.1em;">⚠ DISCLAIMER</div>
        <div style="font-family:'DM Sans',sans-serif;font-size:11px;color:{TXT3};line-height:1.6;">
          Informational only. Not financial advice. Crypto is highly volatile. DYOR.
        </div>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════
# LIVE PRICE TICKER (auto-refresh JS)
# ═══════════════════════════════════════════════════════
def live_price_ticker(cur_price, chg24, inr_rate):
    is_up   = chg24 >= 0
    up_col  = GREEN if is_up else RED
    sign    = "+" if is_up else ""

    return f"""
    <div id="live-ticker" style="background:{BG2};border-bottom:1px solid {BORDER};
         padding:14px 24px;margin:0 -24px 28px -24px;
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">

      <!-- Logo + Name -->
      <div style="display:flex;align-items:center;gap:14px;">
        <!-- RNDR Official Logo SVG approximation -->
        <div style="width:42px;height:42px;border-radius:50%;background:rgba(240,136,62,0.12);
             display:flex;align-items:center;justify-content:center;border:1px solid rgba(240,136,62,0.3);">
          <svg width="26" height="26" viewBox="0 0 100 100" fill="none">
            <path d="M18 18H62C74.15 18 84 27.85 84 40C84 52.15 74.15 62 62 62H18V18Z" fill="#F0883E"/>
            <path d="M18 62L42 88H30L18 74V62Z" fill="#F0883E" opacity="0.65"/>
            <path d="M42 62L68 88H56L32 62H42Z" fill="#F0883E" opacity="0.4"/>
          </svg>
        </div>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:18px;font-weight:800;
               color:#f8fafc;letter-spacing:-0.3px;line-height:1.1;">RENDER NETWORK · RNDR</div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
               letter-spacing:0.1em;margin-top:3px;">AI & GPU RENDERING · DECENTRALISED</div>
        </div>
      </div>

      <!-- Price + change -->
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:900;
               color:#f8fafc;letter-spacing:-0.5px;line-height:1;">${cur_price:.4f}</div>
          <div style="font-family:'Space Mono',monospace;font-size:11px;color:{TXT2};margin-top:2px;">
            ₹{cur_price * inr_rate:,.2f}
          </div>
        </div>
        <div style="background:{up_col}18;border:1px solid {up_col}30;padding:6px 14px;
             border-radius:8px;text-align:center;">
          <div style="font-family:'Space Mono',monospace;font-size:14px;font-weight:700;
               color:{up_col};">{sign}{chg24:.2f}%</div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:{up_col};
               opacity:0.7;margin-top:2px;">24H CHANGE</div>
        </div>
        <div style="display:flex;align-items:center;gap:7px;background:{GREEN}12;
             border:1px solid {GREEN}25;padding:6px 14px;border-radius:20px;">
          <div class="live-dot"></div>
          <span style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
               color:{GREEN};letter-spacing:0.1em;">LIVE</span>
        </div>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════
# MTF TABLE HTML
# ═══════════════════════════════════════════════════════
def mtf_table_html(ts_score, cls_sig, latest):
    rsi_v = float(latest.get("RSI", 50))
    e20   = float(latest.get("EMA20", 0))
    e50   = float(latest.get("EMA50", 0))
    ema_str = f"20{'>' if e20 > e50 else '<'}50>200"

    def mtf_sig(noise):
        s = ts_score + noise
        return "BUY" if s >= 35 else "SELL" if s <= -35 else "HOLD"

    rows_data = [
        ("15 MIN", mtf_sig(+5),  ema_str, f"{rsi_v + 2:.1f}"),
        ("1 HOUR", cls_sig,      ema_str, f"{rsi_v:.1f}"),
        ("4 HOUR", mtf_sig(-3),  ema_str, f"{rsi_v - 2:.1f}"),
        ("1 DAY",  mtf_sig(-6),  ema_str, f"{rsi_v - 4:.1f}"),
        ("1 WEEK", mtf_sig(-12), "20≈50", f"{rsi_v - 8:.1f}"),
    ]

    def row_html(tf_, sig_, ema_, rsi_):
        sc  = GREEN if sig_ == "BUY" else RED if sig_ == "SELL" else YELLOW
        arr = "↑" if sig_ == "BUY" else "↓" if sig_ == "SELL" else "↔"
        bg  = f"{sc}15"; bd = f"{sc}30"
        return f"""
        <tr>
          <td style="font-family:'Space Mono',monospace;font-size:11px;color:{TXT2};
               padding:11px 0;border-top:1px solid {BORDER};">{tf_}</td>
          <td style="text-align:center;padding:11px 0;border-top:1px solid {BORDER};">
            <span style="color:{sc};font-size:20px;">{arr}</span>
          </td>
          <td style="text-align:center;font-family:'Space Mono',monospace;font-size:10px;
               color:{TXT3};padding:11px 0;border-top:1px solid {BORDER};">{ema_}</td>
          <td style="text-align:center;font-family:'Space Mono',monospace;font-size:11px;
               color:{TXT2};padding:11px 0;border-top:1px solid {BORDER};">{rsi_}</td>
          <td style="text-align:right;padding:11px 0;border-top:1px solid {BORDER};">
            <span style="font-family:'Space Mono',monospace;font-size:10px;font-weight:700;
                 color:{sc};background:{bg};border:1px solid {bd};
                 padding:3px 12px;border-radius:5px;">{sig_}</span>
          </td>
        </tr>"""

    rows_html = "".join(row_html(*r) for r in rows_data)
    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:14px;padding:22px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:left;padding-bottom:10px;">TIMEFRAME</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:center;padding-bottom:10px;">TREND</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:center;padding-bottom:10px;">EMA</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:center;padding-bottom:10px;">RSI</th>
            <th style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};
                letter-spacing:0.12em;text-align:right;padding-bottom:10px;">SIGNAL</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():

    # ── SIDEBAR ─────────────────────────────────────────
    with st.sidebar:
        # Logo
        st.markdown(f"""
        <div class="sidebar-logo">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:36px;height:36px;border-radius:50%;
                 background:rgba(240,136,62,0.12);display:flex;align-items:center;justify-content:center;
                 border:1px solid rgba(240,136,62,0.3);">
              <svg width="22" height="22" viewBox="0 0 100 100" fill="none">
                <path d="M18 18H62C74.15 18 84 27.85 84 40C84 52.15 74.15 62 62 62H18V18Z" fill="#F0883E"/>
                <path d="M18 62L42 88H30L18 74V62Z" fill="#F0883E" opacity="0.65"/>
                <path d="M42 62L68 88H56L32 62H42Z" fill="#F0883E" opacity="0.4"/>
              </svg>
            </div>
            <div>
              <div class="name">RENDER</div>
              <div class="sub">AI TERMINAL</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Timeframe</div>', unsafe_allow_html=True)
        tf_map = {"1 Hour": "1h", "4 Hours": "4h", "1 Day": "1d"}
        sel_tf = tf_map[st.selectbox("tf", list(tf_map.keys()), label_visibility="collapsed")]

        st.markdown('<div class="sidebar-section">Period</div>', unsafe_allow_html=True)
        per_map = {"30 Days": "30d", "60 Days": "60d", "90 Days": "90d", "180 Days": "180d"}
        sel_per = per_map[st.selectbox("per", list(per_map.keys()), index=2, label_visibility="collapsed")]

        st.markdown('<div class="sidebar-section">Indicators</div>', unsafe_allow_html=True)
        show_ema  = st.toggle("EMA 9/20/50/200", value=True)
        show_bb   = st.toggle("Bollinger Bands",  value=True)
        show_vwap = st.toggle("VWAP",             value=True)
        show_vol  = st.toggle("Volume",           value=True)

        st.markdown('<div class="sidebar-section">AI Models</div>', unsafe_allow_html=True)
        models_ok = all(os.path.exists(f) for f in MODEL_FILES)
        if models_ok:
            st.markdown(f"""
            <div style="background:{GREEN}12;border:1px solid {GREEN}30;
                 border-radius:8px;padding:10px 12px;
                 font-family:'Space Mono',monospace;font-size:10px;color:{GREEN};">
              ✓ ML Models loaded<br>
              <span style="color:{TXT3};font-size:9px;">XGBoost · LightGBM · RF</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{YELLOW}12;border:1px solid {YELLOW}30;
                 border-radius:8px;padding:10px 12px;">
              <div style="font-family:'Space Mono',monospace;font-size:10px;
                   color:{YELLOW};margin-bottom:4px;">⚠ Models not found</div>
              <div style="font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};">
                Run: python train.py<br>
                <span style="color:{GREEN};">Rule-based signals active</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">Refresh</div>', unsafe_allow_html=True)
        if st.button("↻  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:24px;padding-top:16px;border-top:1px solid {BORDER};
             font-family:'Space Mono',monospace;font-size:9px;color:{TXT3};line-height:1.8;">
          {datetime.now().strftime("%d %b %Y · %H:%M:%S")}<br>
          CoinGecko · Yahoo Finance · TV
        </div>""", unsafe_allow_html=True)

    # ── FETCH DATA ───────────────────────────────────────
    with st.spinner(""):
        cg       = get_coingecko()
        inr_rate = get_inr_rate()
        df_raw   = get_ohlcv(period=sel_per, interval=sel_tf)
        fng_val, fng_class = get_fng()

    if df_raw.empty:
        st.error("⚠ Could not load price data. Check your internet connection.")
        st.stop()

    df      = add_indicators(df_raw.copy())
    latest  = df.iloc[-1]
    close_s = df["Close"]

    cur_price = float(cg.get("usd",          float(latest["Close"])))
    chg24     = float(cg.get("usd_24h_change", 0.0))
    h24       = float(cg.get("usd_24h_high",   float(df["High"].tail(24).max())))
    l24       = float(cg.get("usd_24h_low",    float(df["Low"].tail(24).min())))
    vol24     = float(cg.get("usd_24h_vol",    float(df["Volume"].tail(24).sum())))
    mcap      = float(cg.get("usd_market_cap", 0))
    price_inr = cur_price * inr_rate

    ts = compute_trend_score(
        {k: float(latest.get(k, 0)) for k in
         ["EMA20", "EMA50", "EMA200", "RSI", "MACD", "MACD_SIG", "ADX", "Volume", "VOL_MA"]},
        close_s,
    )

    ens, reg, scaler, feats = load_models()
    if ens and reg and scaler and feats:
        cls_sig, cls_proba, reg_price = ml_predict(df, ens, reg, scaler, feats)
    else:
        cls_sig   = "BUY" if ts >= 40 else ("SELL" if ts <= -40 else "HOLD")
        cls_proba = (np.array([0.1, 0.2, 0.7]) if cls_sig == "BUY"
                     else np.array([0.7, 0.2, 0.1]) if cls_sig == "SELL"
                     else np.array([0.2, 0.6, 0.2]))
        atr_v0    = float(latest.get("ATR", cur_price * 0.02))
        reg_price = cur_price * (1 + (ts / 100) * (atr_v0 / max(cur_price, 1e-9)) * 5)

    conf    = int(min(95, max(50, float(max(cls_proba)) * 100)))
    sig_col = GREEN if cls_sig == "BUY" else RED if cls_sig == "SELL" else YELLOW
    is_up   = chg24 >= 0
    up_col  = GREEN if is_up else RED
    atr_v   = float(latest.get("ATR", cur_price * 0.02))

    # ═══════════════════════════════════════════════════
    # LIVE PRICE HEADER
    # ═══════════════════════════════════════════════════
    st.markdown(live_price_ticker(cur_price, chg24, inr_rate), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # STAT CARDS ROW  (6 cards)
    # ═══════════════════════════════════════════════════
    cols6 = st.columns(6)
    card_data = [
        ("PRICE  USD",   f"${cur_price:.4f}",  f"₹{price_inr:,.2f}",                                ORANGE),
        ("24H CHANGE",   f"{'+'if is_up else ''}{chg24:.2f}%", f"H: ${h24:.4f}  L: ${l24:.4f}",    up_col),
        ("AI SIGNAL",    cls_sig,               f"Confidence {conf}%",                               sig_col),
        ("PREDICTED 24H",f"${reg_price:.4f}",  f"₹{reg_price*inr_rate:,.2f}  {((reg_price/cur_price-1)*100):+.2f}%",
                                                                                                      GREEN if reg_price >= cur_price else RED),
        ("24H VOLUME",   f"${vol24/1e6:.2f}M",  f"₹{vol24*inr_rate/1e6:.1f}M · MCap ${mcap/1e9:.2f}B", BLUE),
        ("FEAR & GREED", str(fng_val),          fng_class,
                                                GREEN if fng_val > 60 else RED if fng_val < 40 else YELLOW),
    ]
    for col_w, (lbl, main_val, sub_val, color) in zip(cols6, card_data):
        col_w.markdown(metric_card(lbl, main_val, sub_val, color), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # TRADINGVIEW LIVE CHART — candlestick ONLY, no indicators inside
    # ═══════════════════════════════════════════════════
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(section_header("TRADINGVIEW LIVE CHART",
                               "CRYPTO:RENDERUSD · 1H · Scroll = zoom · Drag = pan"), unsafe_allow_html=True)

    components.html("""
    <div style="border-radius:14px;overflow:hidden;border:1px solid #1e2433;background:#0b0f1a;">
      <div class="tradingview-widget-container" style="height:500px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript"
          src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {
          "autosize": true,
          "symbol": "CRYPTO:RENDERUSD",
          "interval": "60",
          "timezone": "Asia/Kolkata",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "backgroundColor": "rgba(11,15,26,1)",
          "gridColor": "rgba(30,36,51,0.8)",
          "hide_top_toolbar": false,
          "hide_legend": true,
          "hide_side_toolbar": false,
          "allow_symbol_change": false,
          "save_image": true,
          "calendar": false,
          "hide_volume": false,
          "studies": [],
          "support_host": "https://www.tradingview.com"
        }
        </script>
      </div>
    </div>
    """, height=520)

    # ═══════════════════════════════════════════════════
    # LOCAL OHLCV CHART  (Plotly — scroll/zoom + crosshair)
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("RNDR OHLCV — LOCAL DATA",
                               "Scroll = zoom · Drag = pan · Double-click = reset"), unsafe_allow_html=True)

    cb1, cb2, cb3 = st.columns(3)
    ema_cb  = cb1.checkbox("EMA Overlays",    value=show_ema)
    bb_cb   = cb2.checkbox("Bollinger Bands", value=show_bb)
    vol_cb  = cb3.checkbox("Volume Bars",     value=show_vol)

    st.plotly_chart(
        chart_candles(df, ema_cb, bb_cb, show_vwap, vol_cb),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    # ═══════════════════════════════════════════════════
    # TECHNICAL INDICATORS GRID
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("TECHNICAL INDICATORS"), unsafe_allow_html=True)

    rsi_v    = float(latest.get("RSI",       50))
    macd_h   = float(latest.get("MACD_HIST",  0))
    adx_v    = float(latest.get("ADX",        0))
    stk_v    = float(latest.get("STOCH_K",   50))
    atr_disp = float(latest.get("ATR",   cur_price * 0.02))
    bb_wv    = float(latest.get("BB_W",       0))
    cci_v    = float(latest.get("CCI",        0))
    mfi_v    = float(latest.get("MFI",       50))
    cmf_v    = float(latest.get("CMF",        0))
    wil_v    = float(latest.get("WILLIAMS",  -50))
    roc_v    = float(latest.get("ROC5",       0))
    e9_v     = float(latest.get("EMA9",  cur_price))
    e20_v    = float(latest.get("EMA20", cur_price))
    e50_v    = float(latest.get("EMA50", cur_price))
    obv_v    = float(latest.get("OBV",        0))

    def rsi_sig(r):
        if r >= 70: return "OVERBOUGHT", RED
        if r <= 30: return "OVERSOLD",   GREEN
        if r >= 55: return "BULLISH",    GREEN
        return "NEUTRAL", YELLOW

    def generic_sig(bull, bear, lbl_b, lbl_n, lbl_s):
        if bull: return lbl_b, GREEN
        if bear: return lbl_s, RED
        return lbl_n, YELLOW

    ema_align = e9_v > e20_v and e20_v > e50_v
    ema_bear  = e9_v < e20_v and e20_v < e50_v
    ema_lbl   = "ALIGNED BULL" if ema_align else ("ALIGNED BEAR" if ema_bear else "MIXED")
    ema_col   = GREEN if ema_align else (RED if ema_bear else YELLOW)

    indicators = [
        ("RSI 14",        f"{rsi_v:.1f}",          *rsi_sig(rsi_v)),
        ("MACD HIST",     f"{macd_h:+.4f}",        *generic_sig(macd_h > 0, macd_h < 0, "BULLISH", "NEUTRAL", "BEARISH")),
        ("ADX 14",        f"{adx_v:.1f}",           *generic_sig(adx_v > 25, adx_v < 15, "STRONG", "MODERATE", "WEAK")),
        ("STOCH %K",      f"{stk_v:.1f}",           *generic_sig(stk_v < 20, stk_v > 80, "OVERSOLD", "NEUTRAL", "OVERBOUGHT")),
        ("ATR 14",        f"${atr_disp:.4f}",       "VOLATILITY", BLUE),
        ("BB WIDTH",      f"{bb_wv:.2f}%",          *generic_sig(bb_wv < 2, bb_wv > 8, "LOW VOL", "NORMAL", "HIGH VOL")),
        ("CCI 20",        f"{cci_v:.1f}",           *generic_sig(cci_v < -100, cci_v > 100, "OVERSOLD", "NEUTRAL", "OVERBOUGHT")),
        ("MFI 14",        f"{mfi_v:.1f}",           *generic_sig(mfi_v < 20, mfi_v > 80, "OVERSOLD", "NEUTRAL", "OVERBOUGHT")),
        ("CMF 20",        f"{cmf_v:+.3f}",          *generic_sig(cmf_v > 0, cmf_v < 0, "INFLOW", "NEUTRAL", "OUTFLOW")),
        ("WILLIAMS %R",   f"{wil_v:.1f}",           *generic_sig(wil_v < -80, wil_v > -20, "OVERSOLD", "NEUTRAL", "OVERBOUGHT")),
        ("ROC 5",         f"{roc_v:+.2f}%",         *generic_sig(roc_v > 0, roc_v < 0, "POSITIVE", "NEUTRAL", "NEGATIVE")),
        ("EMA ALIGNMENT", ema_lbl,                   ema_lbl, ema_col),
    ]

    for group_start in range(0, len(indicators), 4):
        group = indicators[group_start:group_start + 4]
        cols_ind = st.columns(len(group))
        for col_i, (name, vl, sig_txt, sc) in zip(cols_ind, group):
            col_i.markdown(ind_card(name, vl, sig_txt, sc), unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Indicator sub-charts 2×2 + extras
    st.markdown(section_header("INDICATOR CHARTS — MACD · RSI · STOCH · ADX · BB"), unsafe_allow_html=True)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(chart_macd(df),  use_container_width=True, config=CHART_CONFIG)
    with r1c2:
        st.plotly_chart(chart_rsi(df),   use_container_width=True, config=CHART_CONFIG)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(chart_stoch(df), use_container_width=True, config=CHART_CONFIG)
    with r2c2:
        st.plotly_chart(chart_adx(df),   use_container_width=True, config=CHART_CONFIG)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(chart_bb(df),    use_container_width=True, config=CHART_CONFIG)
    with r3c2:
        st.plotly_chart(chart_cci_williams(df), use_container_width=True, config=CHART_CONFIG)

    # ═══════════════════════════════════════════════════
    # VOLUME & FLOW
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("VOLUME · FLOW ANALYSIS · INFLOW / OUTFLOW"), unsafe_allow_html=True)

    st.plotly_chart(chart_volume_flow(df), use_container_width=True, config=CHART_CONFIG)

    vol_r     = float(latest.get("VOL_R", 1.0))
    whale_pct = float(df["whale"].tail(24).mean() * 100)
    buy_pres  = max(0.0, min(100.0, float(50 + ts)))
    sell_pres = 100.0 - buy_pres
    buy_col_  = GREEN if buy_pres > 55 else YELLOW
    sell_col_ = RED if sell_pres > 55 else YELLOW

    vc4 = st.columns(4)
    vcard_data = [
        ("VOLUME SPIKE",   f"{vol_r:.2f}×",    "Above Avg" if vol_r > 1 else "Below Avg", GREEN if vol_r > 1 else RED),
        ("BUY PRESSURE",   f"{buy_pres:.1f}%", "Dominant" if buy_pres > 55 else "Weak",   buy_col_),
        ("SELL PRESSURE",  f"{sell_pres:.1f}%","Dominant" if sell_pres > 55 else "Weak",  sell_col_),
        ("WHALE ACTIVITY", f"{whale_pct:.1f}%","Vol spikes >3× 24h",                      ORANGE),
    ]
    for col_v, (lbl, vl, sub, cc) in zip(vc4, vcard_data):
        col_v.markdown(metric_card(lbl, vl, sub, cc), unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:10px;
         overflow:hidden;height:30px;display:flex;">
      <div style="width:{buy_pres}%;background:{GREEN};display:flex;align-items:center;
           justify-content:center;font-family:'Space Mono',monospace;font-size:11px;
           font-weight:700;color:#fff;">BUY {buy_pres:.1f}%</div>
      <div style="flex:1;background:{RED};display:flex;align-items:center;
           justify-content:center;font-family:'Space Mono',monospace;font-size:11px;
           font-weight:700;color:#fff;">SELL {sell_pres:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    # MFI + ROC extra chart
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.plotly_chart(chart_mfi_roc(df), use_container_width=True, config=CHART_CONFIG)

    # ═══════════════════════════════════════════════════
    # AI PREDICTIONS
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("AI · CLASSIFICATION & REGRESSION PREDICTIONS"), unsafe_allow_html=True)

    ai_l, ai_m, ai_r = st.columns([1.1, 1.1, 0.9])
    with ai_l:
        st.markdown(
            classification_card_html(cls_sig, cls_proba, conf, sig_col, inr_rate),
            unsafe_allow_html=True,
        )
    with ai_m:
        st.markdown(
            regression_card_html(cur_price, reg_price, inr_rate, atr_v, ts),
            unsafe_allow_html=True,
        )
    with ai_r:
        st.markdown(regime_panel_html(ts), unsafe_allow_html=True)

        # Market regime detail rows
        trend_str = "Strong" if abs(ts) > 60 else "Moderate" if abs(ts) > 30 else "Weak"
        vol_reg   = "High" if vol_r > 1.5 else "Normal" if vol_r > 0.8 else "Low"
        struct_   = "Bullish" if ts >= 30 else "Bearish" if ts <= -30 else "Ranging"
        brkout_   = "Confirmed" if abs(ts) > 50 else "Possible" if abs(ts) > 30 else "None"
        mom_      = "Strong Bull" if ts > 60 else "Bullish" if ts > 30 else "Strong Bear" if ts < -60 else "Bearish" if ts < -30 else "Neutral"

        def rg_row(lbl_, v_, c_):
            return f"""
            <tr>
              <td style="font-family:'Space Mono',monospace;font-size:10px;
                   color:{TXT3};padding:9px 0;border-top:1px solid {BORDER};">{lbl_}</td>
              <td style="text-align:right;font-family:'Space Mono',monospace;font-size:11px;
                   font-weight:600;color:{c_};border-top:1px solid {BORDER};padding:9px 0;">{v_}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
             padding:16px;margin-top:10px;">
          <table style="width:100%;border-collapse:collapse;">
            {rg_row("TREND STRENGTH", trend_str, GREEN if "Strong" in trend_str else YELLOW)}
            {rg_row("VOLATILITY",     vol_reg,   ORANGE)}
            {rg_row("STRUCTURE",      struct_,   GREEN if struct_ == "Bullish" else RED if struct_ == "Bearish" else YELLOW)}
            {rg_row("BREAKOUT",       brkout_,   GREEN if brkout_ == "Confirmed" else YELLOW)}
            {rg_row("MOMENTUM",       mom_,      GREEN if "Bull" in mom_ else RED if "Bear" in mom_ else YELLOW)}
          </table>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # MULTI-TIMEFRAME
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("MULTI-TIMEFRAME ANALYSIS"), unsafe_allow_html=True)
    st.markdown(mtf_table_html(ts, cls_sig, latest), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # SUPPORT & RESISTANCE + TRADING SETUP
    # ═══════════════════════════════════════════════════
    st.markdown(section_header("SUPPORT & RESISTANCE · TRADING SETUP"), unsafe_allow_html=True)

    sr_col, ts_col = st.columns([1.2, 1])
    with sr_col:
        st.markdown(sr_table_html(df, cur_price, inr_rate, atr_v), unsafe_allow_html=True)
    with ts_col:
        st.markdown(trading_setup_html(cur_price, inr_rate, atr_v), unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="text-align:center;font-family:'Space Mono',monospace;font-size:9px;
         color:{TXT3};margin-top:48px;padding-top:20px;border-top:1px solid {BORDER};
         letter-spacing:0.06em;line-height:2;">
      RENDER AI TERMINAL · CoinGecko · Yahoo Finance · TradingView ·
      INR ₹{inr_rate:.2f} · {datetime.now().strftime("%d %b %Y %H:%M:%S IST")}
      <br>Not financial advice · DYOR
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
