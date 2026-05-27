"""
╔══════════════════════════════════════════════════════════════════╗
║   RENDER (RNDR) — AI PREDICTION DASHBOARD                       ║
║   Streamlit app  ·  deploy on Streamlit Cloud / GitHub          ║
╚══════════════════════════════════════════════════════════════════╝

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Required files in same directory:
    render_ensemble_cls.pkl
    render_reg.pkl
    render_scaler.pkl
    render_features.pkl
    render_meta.json
    (train.py  →  generates the above)
"""

import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import joblib
import ta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RNDR AI · Render Network Prediction",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark trading terminal aesthetic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #060a10 !important;
    color: #e2e8f0 !important;
}
.stApp { background: #060a10 !important; }

/* Remove default Streamlit padding */
.block-container { padding: 0 1.5rem 2rem 1.5rem !important; max-width: 1600px !important; }
.main > div { padding-top: 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1520; }
::-webkit-scrollbar-thumb { background: #1e3050; border-radius: 3px; }

/* ─── TOP HEADER BAR ─── */
.top-header {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f3c 50%, #0a1628 100%);
    border-bottom: 1px solid #1a2d4a;
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -1.5rem 24px -1.5rem;
    flex-wrap: wrap;
    gap: 12px;
}
.header-left { display: flex; align-items: center; gap: 16px; }
.header-logo {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #f97316, #ea580c);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; font-weight: 800;
    box-shadow: 0 0 20px rgba(249,115,22,0.3);
}
.header-title { font-size: 20px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.3px; }
.header-sub { font-size: 12px; color: #64748b; letter-spacing: 0.06em; }
.live-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3);
    color: #22c55e; font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.08em;
}
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #22c55e;
    animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ─── STAT CARDS ─── */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 130px;
    background: #0d1520;
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #2a4070; }
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.stat-card.orange::before { background: linear-gradient(90deg,#f97316,#ea580c); }
.stat-card.green::before  { background: linear-gradient(90deg,#22c55e,#16a34a); }
.stat-card.red::before    { background: linear-gradient(90deg,#ef4444,#dc2626); }
.stat-card.blue::before   { background: linear-gradient(90deg,#3b82f6,#2563eb); }
.stat-card.purple::before { background: linear-gradient(90deg,#a855f7,#9333ea); }
.stat-card.teal::before   { background: linear-gradient(90deg,#14b8a6,#0d9488); }
.stat-label { font-size: 10px; color: #4a6080; letter-spacing: 0.12em; font-weight: 600; margin-bottom: 8px; }
.stat-value { font-size: 24px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #f1f5f9; line-height: 1; }
.stat-sub   { font-size: 11px; color: #64748b; margin-top: 5px; font-family: 'JetBrains Mono', monospace; }
.stat-value.up   { color: #22c55e; }
.stat-value.down { color: #ef4444; }
.stat-value.buy  { color: #22c55e; font-size: 26px; letter-spacing: 0.04em; }
.stat-value.sell { color: #ef4444; font-size: 26px; letter-spacing: 0.04em; }
.stat-value.hold { color: #eab308; font-size: 26px; letter-spacing: 0.04em; }

/* ─── SECTION HEADERS ─── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px; margin-top: 24px;
}
.section-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #f97316;
    box-shadow: 0 0 8px rgba(249,115,22,0.5);
}
.section-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.14em;
    color: #64748b; text-transform: uppercase;
}
.section-line { flex: 1; height: 1px; background: #1a2d4a; }

/* ─── INDICATOR CARDS ─── */
.ind-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}
.ind-card {
    background: #0d1520;
    border: 1px solid #1a2d4a;
    border-radius: 12px;
    padding: 14px 16px;
}
.ind-name { font-size: 10px; color: #4a6080; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 6px; }
.ind-value { font-size: 20px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #f1f5f9; }
.ind-signal { font-size: 11px; font-weight: 600; margin-top: 4px; }
.ind-signal.bull { color: #22c55e; }
.ind-signal.bear { color: #ef4444; }
.ind-signal.neut { color: #eab308; }

/* ─── MODEL CARDS ─── */
.model-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.model-card {
    flex: 1; min-width: 160px;
    background: #0d1520;
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 16px 18px;
}
.model-name { font-size: 10px; color: #4a6080; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 8px; }
.model-signal { font-size: 18px; font-weight: 700; letter-spacing: 0.06em; }
.conf-bar-bg { height: 4px; background: #1a2d4a; border-radius: 2px; margin-top: 8px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 2px; }

/* ─── PRESSURE BARS ─── */
.pressure-row { display: flex; gap: 12px; margin-bottom: 20px; }
.pressure-card {
    flex: 1; background: #0d1520; border: 1px solid #1a2d4a;
    border-radius: 14px; padding: 18px;
}

/* ─── TRADING SETUP TABLE ─── */
.setup-table { width: 100%; border-collapse: collapse; }
.setup-table tr { border-bottom: 1px solid #1a2d4a; }
.setup-table tr:last-child { border-bottom: none; }
.setup-table td { padding: 10px 0; font-size: 13px; }
.setup-table td:first-child { color: #4a6080; font-weight: 500; }
.setup-table td:last-child { text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #f1f5f9; }
.tp { color: #22c55e !important; }
.sl { color: #ef4444 !important; }
.entry-val { color: #3b82f6 !important; }

/* ─── MTF TABLE ─── */
.mtf-table { width: 100%; border-collapse: collapse; }
.mtf-table th { font-size: 10px; color: #4a6080; letter-spacing: 0.1em; font-weight: 600; padding-bottom: 10px; text-align: left; }
.mtf-table td { padding: 8px 0; font-size: 13px; border-top: 1px solid #0f1d2e; }
.mtf-table td:first-child { color: #64748b; font-family: 'JetBrains Mono', monospace; }
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 6px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
}
.badge.buy  { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.badge.sell { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.badge.hold { background: rgba(234,179,8,0.12);  color: #eab308; border: 1px solid rgba(234,179,8,0.25); }
.badge.bull { background: rgba(34,197,94,0.12); color: #22c55e; border: 1px solid rgba(34,197,94,0.25); }
.badge.bear { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ─── Plotly chart background ─── */
.js-plotly-plot { border-radius: 14px; }

/* ─── Footer ─── */
.footer {
    text-align: center; font-size: 11px; color: #2a3f5a;
    margin-top: 40px; padding-top: 16px; border-top: 1px solid #1a2d4a;
    letter-spacing: 0.04em;
}

/* Streamlit element overrides */
div[data-testid="stMetric"] { display: none; }
div[data-testid="stHorizontalBlock"] > div { gap: 12px; }
section[data-testid="stSidebar"] { background: #080e18 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════
MODEL_FILES = [
    "render_ensemble_cls.pkl",
    "render_reg.pkl",
    "render_scaler.pkl",
    "render_features.pkl",
]

CHART_BG     = "#060a10"
GRID_COLOR   = "#0f1d2e"
TEXT_COLOR   = "#4a6080"
ORANGE       = "#f97316"
GREEN        = "#22c55e"
RED          = "#ef4444"
BLUE         = "#3b82f6"
PURPLE       = "#a855f7"
YELLOW       = "#eab308"

def mono(val, decimals=4):
    return f"<span style='font-family:JetBrains Mono,monospace'>{val:.{decimals}f}</span>"

def signal_class(s):
    return s.lower() if s in ("BUY","SELL","HOLD") else "hold"

def badge(text):
    cls = signal_class(text)
    return f"<span class='badge {cls}'>{text}</span>"

def section(title):
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot"></div>
        <div class="section-title">{title}</div>
        <div class="section-line"></div>
    </div>""", unsafe_allow_html=True)

def plotly_layout(fig, height=380, title=""):
    fig.update_layout(
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Space Grotesk", color=TEXT_COLOR, size=11),
        margin=dict(l=12, r=12, t=30 if title else 12, b=12),
        title=dict(text=title, font=dict(size=12, color="#64748b"), x=0.02) if title else None,
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   showline=False, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False,
                   showline=False, tickfont=dict(size=10)),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(size=10), orientation="h", y=1.08),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0d1520", bordercolor="#1a2d4a",
                        font=dict(family="JetBrains Mono", size=11)),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=120, show_spinner=False)
def get_inr_rate():
    try:
        r = requests.get(
            "https://api.exchangerate-api.com/v4/latest/USD", timeout=8
        )
        return float(r.json()["rates"].get("INR", 84.0))
    except Exception:
        return 84.0

@st.cache_data(ttl=60, show_spinner=False)
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
            if len(df) > 100:
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=30&format=json", timeout=8)
        data = r.json()["data"][0]
        return int(data["value"]), data["value_classification"]
    except Exception:
        return 50, "Neutral"

# ══════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════
def add_indicators(df):
    c, h, l, v, o = (
        df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    )
    def safe(s):
        return s.replace([np.inf, -np.inf], np.nan).fillna(0)

    df["EMA9"]  = safe(ta.trend.EMAIndicator(c, 9).ema_indicator())
    df["EMA20"] = safe(ta.trend.EMAIndicator(c, 20).ema_indicator())
    df["EMA50"] = safe(ta.trend.EMAIndicator(c, 50).ema_indicator())
    df["EMA200"]= safe(ta.trend.EMAIndicator(c, 200).ema_indicator())
    df["SMA20"] = safe(ta.trend.SMAIndicator(c, 20).sma_indicator())

    macd_ = ta.trend.MACD(c)
    df["MACD"]        = safe(macd_.macd())
    df["MACD_SIGNAL"] = safe(macd_.macd_signal())
    df["MACD_HIST"]   = safe(macd_.macd_diff())

    adx_ = ta.trend.ADXIndicator(h, l, c, 14)
    df["ADX"]     = safe(adx_.adx())
    df["ADX_POS"] = safe(adx_.adx_pos())
    df["ADX_NEG"] = safe(adx_.adx_neg())

    df["RSI"]     = safe(ta.momentum.RSIIndicator(c, 14).rsi())
    df["RSI7"]    = safe(ta.momentum.RSIIndicator(c, 7).rsi())

    stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["STOCH_K"] = safe(stoch.stoch())
    df["STOCH_D"] = safe(stoch.stoch_signal())

    df["CCI"]      = safe(ta.trend.CCIIndicator(h, l, c, 20).cci())
    df["WILLIAMS"] = safe(ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r())
    df["ROC5"]     = safe(ta.momentum.ROCIndicator(c, 5).roc())
    df["ROC20"]    = safe(ta.momentum.ROCIndicator(c, 20).roc())

    bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_UPPER"]  = safe(bb.bollinger_hband())
    df["BB_LOWER"]  = safe(bb.bollinger_lband())
    df["BB_MIDDLE"] = safe(bb.bollinger_mavg())
    df["BB_WIDTH"]  = safe(bb.bollinger_wband())
    df["BB_PCT"]    = safe(bb.bollinger_pband())

    df["ATR"]  = safe(ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range())

    df["OBV"]  = safe(ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume())
    df["MFI"]  = safe(ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index())
    df["CMF"]  = safe(ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow())
    try:
        df["VWAP"] = safe(ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price())
    except Exception:
        df["VWAP"] = c

    vol_ma = v.rolling(20).mean().replace(0, np.nan)
    df["VOL_SMA20"] = safe(vol_ma)
    df["VOL_RATIO"] = safe(v / vol_ma)

    df["RET1"]  = safe(c.pct_change(1))
    df["RET4"]  = safe(c.pct_change(4))
    df["RET24"] = safe(c.pct_change(24))

    df["ema_9"]   = df["EMA9"]
    df["ema_21"]  = safe(ta.trend.EMAIndicator(c, 21).ema_indicator())
    df["ema_50"]  = df["EMA50"]
    df["sma_20"]  = df["SMA20"]
    df["macd"]        = df["MACD"]
    df["macd_signal"] = df["MACD_SIGNAL"]
    df["macd_diff"]   = df["MACD_HIST"]
    df["adx"]     = df["ADX"]
    df["adx_pos"] = df["ADX_POS"]
    df["adx_neg"] = df["ADX_NEG"]
    df["rsi_14"]  = df["RSI"]
    df["rsi_7"]   = df["RSI7"]
    df["rsi_28"]  = safe(ta.momentum.RSIIndicator(c, 28).rsi())
    df["stoch_k"] = df["STOCH_K"]
    df["stoch_d"] = df["STOCH_D"]
    df["cci"]     = df["CCI"]
    df["roc_5"]   = df["ROC5"]
    df["roc_20"]  = df["ROC20"]
    df["williams"]= df["WILLIAMS"]
    df["bb_upper"]= df["BB_UPPER"]
    df["bb_lower"]= df["BB_LOWER"]
    df["bb_width"]= df["BB_WIDTH"]
    df["bb_pct"]  = df["BB_PCT"]
    df["atr_14"]  = df["ATR"]
    df["obv"]     = df["OBV"]
    df["mfi"]     = df["MFI"]
    df["cmf"]     = df["CMF"]
    df["vwap"]    = df["VWAP"]
    df["ret_1h"]  = df["RET1"]
    df["ret_4h"]  = df["RET4"]
    df["ret_24h"] = df["RET24"]
    df["log_ret"] = safe(np.log(c / c.shift(1)))
    df["vol_24h"] = safe(df["RET1"].rolling(24).std())
    df["vol_7d"]  = safe(df["RET1"].rolling(168).std())

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

# ══════════════════════════════════════════════════════════════════
# MODEL PREDICTION
# ══════════════════════════════════════════════════════════════════
def load_models():
    try:
        ens    = joblib.load("render_ensemble_cls.pkl")
        reg    = joblib.load("render_reg.pkl")
        scaler = joblib.load("render_scaler.pkl")
        feats  = joblib.load("render_features.pkl")
        return ens, reg, scaler, feats
    except Exception:
        return None, None, None, None

def predict(df, ens, reg, scaler, feats):
    EXCLUDE = {"Open","High","Low","Close","Volume",
               "future_close","future_return","label"}
    available = [f for f in feats if f in df.columns and f not in EXCLUDE]
    row = df.iloc[-1][available].values.astype(np.float64).reshape(1, -1)
    row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)

    full = np.zeros((1, len(feats)), dtype=np.float64)
    idx_map = {f: i for i, f in enumerate(feats)}
    for i, f in enumerate(available):
        full[0, idx_map[f]] = row[0, i]

    X_s = scaler.transform(full)
    label_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    cls_pred  = label_map[int(ens.predict(X_s)[0])]
    cls_proba = ens.predict_proba(X_s)[0]
    reg_pred  = float(reg.predict(X_s)[0])
    return cls_pred, cls_proba, reg_pred

# ══════════════════════════════════════════════════════════════════
# TREND SCORE (rule-based fallback)
# ══════════════════════════════════════════════════════════════════
def compute_trend_score(latest, close_series):
    ts = 0
    ema20, ema50, ema200 = latest.get("EMA20",0), latest.get("EMA50",0), latest.get("EMA200",0)
    rsi  = latest.get("RSI", 50)
    macd, macd_sig = latest.get("MACD",0), latest.get("MACD_SIGNAL",0)
    adx  = latest.get("ADX", 0)
    vol, vol_sma = latest.get("Volume",0), latest.get("VOL_SMA20",0)

    if ema20 > ema50:  ts += 20
    else:              ts -= 20
    if ema50 > ema200: ts += 20
    else:              ts -= 20
    if rsi >= 70:      ts -= 15
    elif rsi >= 60:    ts += 15
    elif rsi <= 30:    ts += 20
    elif rsi <= 40:    ts -= 10
    if macd > macd_sig:ts += 20
    else:              ts -= 20
    if adx > 25:       ts += 10
    if len(close_series) >= 10:
        pc10 = (close_series.iloc[-1] - close_series.iloc[-10]) / close_series.iloc[-10] * 100
        ts += 10 if pc10 > 0 else -10
    if vol > vol_sma:  ts += 5
    return ts

# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════
def candlestick_chart(df, show_ema=True, show_bb=True, show_vol=True):
    rows = 3 if show_vol else 2
    row_heights = [0.62, 0.22, 0.16] if show_vol else [0.72, 0.28]
    subplot_titles = ["", "VOLUME", ""] if show_vol else ["", ""]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"],  close=df["Close"],
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN,  decreasing_fillcolor=RED,
        line=dict(width=1), name="Price",
        whiskerwidth=0.5,
    ), row=1, col=1)

    if show_ema:
        for col, color, name in [
            ("EMA9",  "#60a5fa", "EMA9"),
            ("EMA20", "#f97316", "EMA20"),
            ("EMA50", "#a855f7", "EMA50"),
            ("EMA200","#14b8a6", "EMA200"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=name,
                    line=dict(color=color, width=1.2),
                    opacity=0.9, hoverinfo="skip",
                ), row=1, col=1)

    if show_bb and "BB_UPPER" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_UPPER"], name="BB Upper",
            line=dict(color="#334155", width=0.8, dash="dot"),
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_LOWER"], name="BB Lower",
            line=dict(color="#334155", width=0.8, dash="dot"),
            fill="tonexty", fillcolor="rgba(51,65,85,0.08)",
            hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_MIDDLE"], name="BB Mid",
            line=dict(color="#475569", width=0.6),
            hoverinfo="skip",
        ), row=1, col=1)

    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#fb923c", width=1, dash="dash"),
            opacity=0.8, hoverinfo="skip",
        ), row=1, col=1)

    # Volume bars
    if show_vol:
        colors = [GREEN if c >= o else RED
                  for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            marker_color=colors, name="Volume",
            marker_opacity=0.55, showlegend=False,
        ), row=2, col=1)
        if "VOL_SMA20" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VOL_SMA20"], name="Vol MA20",
                line=dict(color="#f97316", width=1.2),
                showlegend=False, hoverinfo="skip",
            ), row=2, col=1)

    # RSI sub-chart
    rsi_row = 3 if show_vol else 2
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI",
            line=dict(color="#f97316", width=1.4),
            showlegend=False,
        ), row=rsi_row, col=1)
        for level, color in [(70, "#ef4444"), (30, "#22c55e"), (50, "#1a2d4a")]:
            fig.add_hline(y=level, line_width=0.6,
                          line_dash="dot", line_color=color,
                          row=rsi_row, col=1)

    plotly_layout(fig, height=620)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)

    return fig

def macd_chart(df):
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"],
                         marker_color=colors, name="Histogram", opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],
                             line=dict(color=BLUE, width=1.5), name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIGNAL"],
                             line=dict(color=ORANGE, width=1.5, dash="dot"), name="Signal"))
    fig.add_hline(y=0, line_width=0.5, line_color="#1a2d4a")
    plotly_layout(fig, height=200, title="MACD (12,26,9)")
    return fig

def stoch_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"],
                             line=dict(color=BLUE, width=1.4), name="%K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"],
                             line=dict(color=ORANGE, width=1.4, dash="dot"), name="%D"))
    for level, color in [(80, "#ef4444"), (20, "#22c55e")]:
        fig.add_hline(y=level, line_width=0.6, line_dash="dot", line_color=color)
    plotly_layout(fig, height=200, title="Stochastic (14,3,3)")
    return fig

def volume_flow_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"],
                             line=dict(color=BLUE, width=1.4), name="OBV",
                             fill="tozeroy", fillcolor="rgba(59,130,246,0.07)"),
                  row=1, col=1)
    cmf_colors = [GREEN if v >= 0 else RED for v in df["CMF"]]
    fig.add_trace(go.Bar(x=df.index, y=df["CMF"],
                         marker_color=cmf_colors, name="CMF", opacity=0.75),
                  row=2, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color="#1a2d4a", row=2, col=1)
    plotly_layout(fig, height=280, title="Volume Flow  ·  OBV  +  CMF")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)
    return fig

def adx_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"],
                             line=dict(color=ORANGE, width=1.8), name="ADX"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX_POS"],
                             line=dict(color=GREEN, width=1.2, dash="dot"), name="+DI"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX_NEG"],
                             line=dict(color=RED, width=1.2, dash="dot"), name="−DI"))
    fig.add_hline(y=25, line_width=0.7, line_dash="dash", line_color="#334155")
    plotly_layout(fig, height=200, title="ADX  ·  Trend Strength")
    return fig

def bb_width_chart(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"],
                             line=dict(color=PURPLE, width=1.5), name="BB Width",
                             fill="tozeroy", fillcolor="rgba(168,85,247,0.08)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_PCT"],
                             line=dict(color="#14b8a6", width=1.4), name="BB %B"),
                  row=2, col=1)
    fig.add_hline(y=1.0, line_width=0.5, line_color="#ef4444", row=2, col=1)
    fig.add_hline(y=0.0, line_width=0.5, line_color="#22c55e", row=2, col=1)
    plotly_layout(fig, height=260, title="Bollinger Band Width  +  %B")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=GRID_COLOR, showgrid=True, row=i, col=1)
    return fig

def regime_gauge(score):
    score_clamped = max(-100, min(100, score))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_clamped,
        number=dict(font=dict(family="JetBrains Mono", size=32, color="#f1f5f9")),
        gauge=dict(
            axis=dict(range=[-100, 100], tickwidth=0,
                      tickcolor="#1a2d4a", tickfont=dict(size=9, color="#4a6080")),
            bar=dict(color=GREEN if score_clamped >= 0 else RED,
                     thickness=0.22),
            bgcolor="#0d1520",
            borderwidth=0,
            steps=[
                dict(range=[-100, -40], color="#ef444415"),
                dict(range=[-40,   40], color="#eab30815"),
                dict(range=[40,   100], color="#22c55e15"),
            ],
            threshold=dict(line=dict(color="#f97316", width=2),
                           thickness=0.7, value=score_clamped),
        ),
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(family="Space Grotesk", color="#64748b"),
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig

# ══════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════
def main():
    # ── Top header ──────────────────────────────────────────────
    st.markdown("""
    <div class="top-header">
      <div class="header-left">
        <div class="header-logo">R</div>
        <div>
          <div class="header-title">RENDER NETWORK · RNDR/USDT</div>
          <div class="header-sub">AI &amp; GPU Rendering Infrastructure · Decentralised</div>
        </div>
      </div>
      <div class="live-chip">
        <div class="live-dot"></div>
        LIVE  ·  AUTO-REFRESH 60s
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar controls ────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️  Chart Settings")
        tf_opts = {"1 Hour": "1h", "4 Hours": "4h", "1 Day": "1d"}
        period_opts = {
            "30 Days": "30d", "60 Days": "60d",
            "90 Days": "90d", "180 Days": "180d",
        }
        selected_tf = tf_opts[st.selectbox("Timeframe", list(tf_opts.keys()), index=0)]
        selected_period = period_opts[st.selectbox("Period", list(period_opts.keys()), index=2)]
        show_ema = st.toggle("EMA Overlays", value=True)
        show_bb  = st.toggle("Bollinger Bands", value=True)
        show_vol = st.toggle("Volume Sub-chart", value=True)
        st.divider()
        st.markdown("### 🤖  ML Models")
        models_available = all(os.path.exists(f) for f in MODEL_FILES)
        if models_available:
            st.success("✅ Models loaded")
        else:
            st.warning("⚠️ Models not found\n\nRun `python train.py` first.\nRule-based signals active.")
        st.divider()
        refresh = st.button("🔄  Refresh Now", use_container_width=True)

    if refresh:
        st.cache_data.clear()
        st.rerun()

    # ── Fetch data ──────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        cg       = get_coingecko()
        inr_rate = get_inr_rate()
        df_raw   = get_ohlcv(period=selected_period, interval=selected_tf)
        fng_val, fng_class = get_fng()

    if df_raw.empty:
        st.error("⚠️ Could not load OHLCV data. Check your internet connection.")
        st.stop()

    df = add_indicators(df_raw.copy())
    latest = df.iloc[-1]
    close_series = df["Close"]

    # Current price
    current_price = float(cg.get("usd", float(latest["Close"])))
    price_inr     = current_price * inr_rate
    change_24h    = float(cg.get("usd_24h_change", 0.0))
    high_24h      = float(cg.get("usd_24h_high", float(df["High"].tail(24).max())))
    low_24h       = float(cg.get("usd_24h_low", float(df["Low"].tail(24).min())))
    vol_24h       = float(cg.get("usd_24h_vol", float(df["Volume"].tail(24).sum())))
    mcap          = float(cg.get("usd_market_cap", 0))

    # Trend score
    trend_score = compute_trend_score(
        {k: float(latest.get(k, 0)) for k in
         ["EMA20","EMA50","EMA200","RSI","MACD","MACD_SIGNAL","ADX",
          "Volume","VOL_SMA20"]},
        close_series,
    )

    # ML predictions
    ens, reg, scaler, feats = load_models()
    if ens and reg and scaler and feats:
        cls_signal, cls_proba, reg_price = predict(df, ens, reg, scaler, feats)
    else:
        label_map = {True: "BUY", False: "SELL"}
        cls_signal = "BUY" if trend_score >= 40 else ("SELL" if trend_score <= -40 else "HOLD")
        cls_proba  = np.array([0.1, 0.2, 0.7]) if cls_signal == "BUY" else \
                     np.array([0.7, 0.2, 0.1]) if cls_signal == "SELL" else \
                     np.array([0.2, 0.6, 0.2])
        atr = float(latest.get("ATR", current_price * 0.02))
        reg_price = current_price * (1 + (trend_score / 100) * (atr / current_price) * 5)

    confidence = int(min(95, max(50, max(cls_proba) * 100)))

    # ── TOP STAT CARDS ──────────────────────────────────────────
    is_up = change_24h >= 0
    chg_class = "up" if is_up else "down"
    sig_class_val = cls_signal.lower()

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card orange">
        <div class="stat-label">PRICE (USD)</div>
        <div class="stat-value">${current_price:.4f}</div>
        <div class="stat-sub">₹{price_inr:,.2f}</div>
      </div>
      <div class="stat-card {'green' if is_up else 'red'}">
        <div class="stat-label">24H CHANGE</div>
        <div class="stat-value {chg_class}">{'+'if is_up else ''}{change_24h:.2f}%</div>
        <div class="stat-sub">H: ${high_24h:.4f}  L: ${low_24h:.4f}</div>
      </div>
      <div class="stat-card {'green' if cls_signal=='BUY' else 'red' if cls_signal=='SELL' else 'blue'}">
        <div class="stat-label">AI SIGNAL</div>
        <div class="stat-value {sig_class_val}">{cls_signal}</div>
        <div class="stat-sub">Confidence {confidence}%</div>
      </div>
      <div class="stat-card purple">
        <div class="stat-label">PREDICTED PRICE</div>
        <div class="stat-value">${reg_price:.4f}</div>
        <div class="stat-sub">₹{reg_price*inr_rate:,.2f}  ·  {((reg_price/current_price-1)*100):+.2f}%</div>
      </div>
      <div class="stat-card teal">
        <div class="stat-label">24H VOLUME</div>
        <div class="stat-value">${vol_24h/1e6:.2f}M</div>
        <div class="stat-sub">₹{vol_24h*inr_rate/1e6:.1f}M  ·  MCap ${mcap/1e6:.0f}M</div>
      </div>
      <div class="stat-card blue">
        <div class="stat-label">FEAR &amp; GREED</div>
        <div class="stat-value {'up' if fng_val>60 else 'down' if fng_val<40 else ''}">{fng_val}</div>
        <div class="stat-sub">{fng_class}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TRADINGVIEW CHART (iframe embed) ────────────────────────
    section("TRADINGVIEW LIVE CHART")
    st.components.v1.html("""
    <div style="border-radius:14px; overflow:hidden; border:1px solid #1a2d4a;">
    <div class="tradingview-widget-container" style="height:520px; width:100%;">
      <div class="tradingview-widget-container__widget" style="height:100%; width:100%;"></div>
      <script type="text/javascript"
        src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
        async>
      {
        "autosize": true,
        "symbol": "BINANCE:RNDRUSDT",
        "interval": "60",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "backgroundColor": "rgba(6,10,16,1)",
        "gridColor": "rgba(15,29,46,0.8)",
        "hide_top_toolbar": false,
        "hide_legend": false,
        "allow_symbol_change": false,
        "save_image": false,
        "calendar": false,
        "hide_volume": false,
        "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies"],
        "support_host": "https://www.tradingview.com"
      }
      </script>
    </div>
    </div>
    """, height=540)

    # ── LOCAL CANDLESTICK CHART ─────────────────────────────────
    section("RNDR OHLCV CHART  ·  LOCAL DATA")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        show_ema_c = st.checkbox("EMA Lines", value=show_ema, key="ema_c")
    with c2:
        show_bb_c = st.checkbox("Bollinger Bands", value=show_bb, key="bb_c")
    with c3:
        show_vol_c = st.checkbox("Volume", value=show_vol, key="vol_c")

    st.plotly_chart(
        candlestick_chart(df, show_ema_c, show_bb_c, show_vol_c),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ══════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS GRID
    # ══════════════════════════════════════════════════════════════
    section("TECHNICAL INDICATORS")

    rsi   = float(latest.get("RSI", 50))
    macd_val  = float(latest.get("MACD", 0))
    macd_sig  = float(latest.get("MACD_SIGNAL", 0))
    adx_val   = float(latest.get("ADX", 0))
    stoch_k   = float(latest.get("STOCH_K", 50))
    atr_val   = float(latest.get("ATR", 0))
    bb_width  = float(latest.get("BB_WIDTH", 0))
    cci_val   = float(latest.get("CCI", 0))
    mfi_val   = float(latest.get("MFI", 50))
    cmf_val   = float(latest.get("CMF", 0))
    williams  = float(latest.get("WILLIAMS", -50))
    roc5      = float(latest.get("ROC5", 0))
    ema9_v    = float(latest.get("EMA9", current_price))
    ema20_v   = float(latest.get("EMA20", current_price))
    ema50_v   = float(latest.get("EMA50", current_price))

    def rsi_sig(r):
        if r >= 70: return "OVERBOUGHT", "bear"
        if r <= 30: return "OVERSOLD", "bull"
        if r >= 55: return "BULLISH", "bull"
        return "NEUTRAL", "neut"

    def macd_sig_fn(m, s): return ("BULLISH CROSS", "bull") if m > s else ("BEARISH CROSS", "bear")
    def adx_sig_fn(a): return ("STRONG TREND", "bull") if a > 25 else ("WEAK TREND", "neut")
    def stoch_sig_fn(k): return ("OVERBOUGHT", "bear") if k > 80 else ("OVERSOLD", "bull") if k < 20 else ("NEUTRAL", "neut")
    def cci_sig_fn(c): return ("OVERBOUGHT", "bear") if c > 100 else ("OVERSOLD", "bull") if c < -100 else ("NEUTRAL", "neut")
    def cmf_sig_fn(c): return ("INFLOW", "bull") if c > 0 else ("OUTFLOW", "bear")
    def mfi_sig_fn(m): return ("OVERBOUGHT", "bear") if m > 80 else ("OVERSOLD", "bull") if m < 20 else ("NEUTRAL", "neut")

    rsi_txt,   rsi_cls   = rsi_sig(rsi)
    macd_txt,  macd_cls  = macd_sig_fn(macd_val, macd_sig)
    adx_txt,   adx_cls   = adx_sig_fn(adx_val)
    stoch_txt, stoch_cls = stoch_sig_fn(stoch_k)
    cci_txt,   cci_cls   = cci_sig_fn(cci_val)
    cmf_txt,   cmf_cls   = cmf_sig_fn(cmf_val)
    mfi_txt,   mfi_cls   = mfi_sig_fn(mfi_val)

    ema_align = ema9_v > ema20_v and ema20_v > ema50_v
    ema_txt   = "ALIGNED BULL" if ema_align else ("ALIGNED BEAR" if ema9_v < ema20_v and ema20_v < ema50_v else "MIXED")
    ema_cls   = "bull" if ema_align else ("bear" if not ema_align and ema9_v < ema50_v else "neut")

    inds_html = ""
    for name, val, sig_txt, sig_cls in [
        ("RSI (14)",    f"{rsi:.1f}",          rsi_txt,   rsi_cls),
        ("MACD HIST",   f"{macd_val-macd_sig:+.4f}", macd_txt, macd_cls),
        ("ADX (14)",    f"{adx_val:.1f}",       adx_txt,   adx_cls),
        ("STOCH %K",    f"{stoch_k:.1f}",       stoch_txt, stoch_cls),
        ("ATR (14)",    f"${atr_val:.4f}",      "VOLATILITY", "neut"),
        ("BB WIDTH",    f"{bb_width:.2f}%",     "LOW" if bb_width<2 else "HIGH", "bull" if bb_width<2 else "bear"),
        ("CCI (20)",    f"{cci_val:.1f}",       cci_txt,   cci_cls),
        ("MFI (14)",    f"{mfi_val:.1f}",       mfi_txt,   mfi_cls),
        ("CMF (20)",    f"{cmf_val:+.3f}",      cmf_txt,   cmf_cls),
        ("WILLIAMS %R", f"{williams:.1f}",      "OVERSOLD" if williams<-80 else "OVERBOUGHT" if williams>-20 else "NEUTRAL", "bull" if williams<-80 else "bear" if williams>-20 else "neut"),
        ("ROC (5)",     f"{roc5:+.2f}%",        "POSITIVE" if roc5>0 else "NEGATIVE", "bull" if roc5>0 else "bear"),
        ("EMA ALIGN",   ema_txt,                ema_txt,   ema_cls),
    ]:
        inds_html += f"""
        <div class="ind-card">
          <div class="ind-name">{name}</div>
          <div class="ind-value">{val}</div>
          <div class="ind-signal {sig_cls}">{sig_txt}</div>
        </div>"""

    st.markdown(f'<div class="ind-grid">{inds_html}</div>', unsafe_allow_html=True)

    # ── Indicator sub-charts ────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(macd_chart(df), use_container_width=True,
                        config={"displayModeBar": False})
    with col_r:
        st.plotly_chart(stoch_chart(df), use_container_width=True,
                        config={"displayModeBar": False})

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.plotly_chart(adx_chart(df), use_container_width=True,
                        config={"displayModeBar": False})
    with col_r2:
        st.plotly_chart(bb_width_chart(df), use_container_width=True,
                        config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════
    # VOLUME & FLOW ANALYSIS
    # ══════════════════════════════════════════════════════════════
    section("VOLUME  ·  FLOW ANALYSIS  ·  INFLOW  /  OUTFLOW")
    st.plotly_chart(volume_flow_chart(df), use_container_width=True,
                    config={"displayModeBar": False})

    vol_ratio = float(latest.get("VOL_RATIO", 1.0))
    obv_val   = float(latest.get("OBV", 0))
    whale_pct = float(df["whale"].tail(24).mean() * 100)

    buy_pressure_pct = max(0, min(100, 50 + trend_score))
    sell_pressure_pct = 100 - buy_pressure_pct

    vc1, vc2, vc3, vc4 = st.columns(4)
    for col, label, val, sub, accent in [
        (vc1, "VOLUME SPIKE",    f"{vol_ratio:.2f}×",
         "Above Avg" if vol_ratio > 1 else "Below Avg", GREEN if vol_ratio > 1 else RED),
        (vc2, "BUY PRESSURE",    f"{buy_pressure_pct:.1f}%",
         "Dominant" if buy_pressure_pct > 55 else "Weak", GREEN),
        (vc3, "SELL PRESSURE",   f"{sell_pressure_pct:.1f}%",
         "Dominant" if sell_pressure_pct > 55 else "Weak", RED),
        (vc4, "WHALE ACTIVITY",  f"{whale_pct:.1f}%",
         "Whale Spikes 24h", ORANGE),
    ]:
        col.markdown(f"""
        <div class="stat-card blue" style="border-top-color:{accent}">
          <div class="stat-label">{label}</div>
          <div class="stat-value" style="font-size:22px;color:{accent}">{val}</div>
          <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Buy / Sell pressure bar
    bp_fig = go.Figure(go.Bar(
        x=[buy_pressure_pct, sell_pressure_pct],
        y=["Pressure", "Pressure"],
        orientation="h",
        marker_color=[GREEN, RED],
        text=[f"BUY {buy_pressure_pct:.1f}%", f"SELL {sell_pressure_pct:.1f}%"],
        textposition="inside",
        textfont=dict(color="#fff", size=12, family="JetBrains Mono"),
        insidetextanchor="middle",
    ))
    bp_fig.update_layout(
        height=60, paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(l=0, r=0, t=0, b=0),
        barmode="stack",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(bp_fig, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════════════════════
    # ML CLASSIFICATION & REGRESSION
    # ══════════════════════════════════════════════════════════════
    section("AI  ·  CLASSIFICATION  &  REGRESSION  PREDICTIONS")

    col_cls, col_reg, col_gauge = st.columns([1.3, 1.3, 1])

    with col_cls:
        proba_sell = float(cls_proba[0]) * 100
        proba_hold = float(cls_proba[1]) * 100
        proba_buy  = float(cls_proba[2]) * 100

        def bar_html(label, pct, color):
            return f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:11px;color:#4a6080;font-weight:600;letter-spacing:0.08em">{label}</span>
                <span style="font-size:12px;color:{color};font-family:JetBrains Mono,monospace;font-weight:700">{pct:.1f}%</span>
              </div>
              <div style="height:5px;background:#0f1d2e;border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{color};border-radius:3px;"></div>
              </div>
            </div>"""

        sig_color = GREEN if cls_signal == "BUY" else RED if cls_signal == "SELL" else YELLOW
        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;">
          <div style="font-size:10px;color:#4a6080;letter-spacing:0.12em;font-weight:600;margin-bottom:14px;">CLASSIFICATION MODEL</div>
          <div style="font-size:36px;font-weight:800;color:{sig_color};letter-spacing:0.06em;margin-bottom:4px;">{cls_signal}</div>
          <div style="font-size:12px;color:#4a6080;margin-bottom:20px;">Ensemble  ·  XGBoost + LightGBM + RandomForest</div>
          {bar_html("BUY", proba_buy, GREEN)}
          {bar_html("HOLD", proba_hold, YELLOW)}
          {bar_html("SELL", proba_sell, RED)}
          <div style="margin-top:16px;padding-top:14px;border-top:1px solid #1a2d4a;display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;color:#4a6080;">Confidence</span>
            <span style="font-size:16px;font-weight:700;font-family:JetBrains Mono,monospace;color:{sig_color}">{confidence}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_reg:
        reg_chg    = (reg_price / current_price - 1) * 100
        reg_color  = GREEN if reg_chg >= 0 else RED
        atr_v      = float(latest.get("ATR", current_price * 0.02))
        r1h  = current_price * (1 + (trend_score / 100) * (atr_v / current_price) * 1)
        r4h  = current_price * (1 + (trend_score / 100) * (atr_v / current_price) * 2)
        r24h = reg_price
        sigma = atr_v * 2

        def reg_row_html(horizon, price, base):
            chg = (price / base - 1) * 100
            col = GREEN if chg >= 0 else RED
            return f"""
            <tr>
              <td style="color:#4a6080;font-size:12px;padding:8px 0;">{horizon}</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;font-weight:700;font-size:13px;">
                <span style="color:{col}">${price:.4f}</span>
                <span style="color:{col};font-size:11px;margin-left:8px;">{chg:+.2f}%</span>
              </td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;">
          <div style="font-size:10px;color:#4a6080;letter-spacing:0.12em;font-weight:600;margin-bottom:14px;">REGRESSION FORECAST</div>
          <div style="font-size:36px;font-weight:800;color:{reg_color};font-family:JetBrains Mono,monospace;margin-bottom:2px;">${reg_price:.4f}</div>
          <div style="font-size:12px;color:#4a6080;margin-bottom:20px;">₹{reg_price*inr_rate:,.2f}  ·  {reg_chg:+.2f}% expected</div>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #1a2d4a;">
              <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;padding-bottom:8px;text-align:left;">HORIZON</th>
              <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;padding-bottom:8px;text-align:right;">TARGET PRICE</th>
            </tr>
            {reg_row_html("1 Hour",  r1h,  current_price)}
            {reg_row_html("4 Hours", r4h,  current_price)}
            {reg_row_html("24 Hours",r24h, current_price)}
          </table>
          <div style="margin-top:14px;padding-top:12px;border-top:1px solid #1a2d4a;">
            <div style="font-size:10px;color:#4a6080;margin-bottom:4px;">CONFIDENCE INTERVAL (±1σ)</div>
            <div style="font-size:12px;color:#64748b;font-family:JetBrains Mono,monospace;">
              ${max(0,r24h-sigma):.4f}  —  ${r24h+sigma:.4f}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        regime = ("BULLISH" if trend_score >= 40
                  else "BEARISH" if trend_score <= -40 else "SIDEWAYS")
        regime_color = GREEN if regime=="BULLISH" else RED if regime=="BEARISH" else YELLOW
        st.plotly_chart(regime_gauge(trend_score), use_container_width=True,
                        config={"displayModeBar": False})
        st.markdown(f"""
        <div style="text-align:center;margin-top:-10px;">
          <div style="font-size:18px;font-weight:700;color:{regime_color};letter-spacing:0.06em;">{regime}</div>
          <div style="font-size:11px;color:#4a6080;margin-top:4px;">Trend Score: {trend_score:+d}</div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # MULTI-TIMEFRAME + MARKET REGIME
    # ══════════════════════════════════════════════════════════════
    section("MULTI-TIMEFRAME  ·  MARKET REGIME")
    mtf_col, regime_col = st.columns([1.4, 1])

    with mtf_col:
        ema_align_str = "20>50>200" if ema20_v > ema50_v and ema50_v > float(latest.get("EMA200", 0)) else "20<50<200"

        def mtf_signal(noise):
            base = trend_score + noise
            return "BUY" if base >= 35 else "SELL" if base <= -35 else "HOLD"

        mtf_data = [
            ("15 MIN",  mtf_signal(+5),  ema_align_str, f"{rsi+2:.1f}"),
            ("1 HOUR",  cls_signal,       ema_align_str, f"{rsi:.1f}"),
            ("4 HOUR",  mtf_signal(-3),   ema_align_str, f"{rsi-2:.1f}"),
            ("1 DAY",   mtf_signal(-6),   ema_align_str, f"{rsi-4:.1f}"),
            ("1 WEEK",  mtf_signal(-12),  "20≈50",       f"{rsi-8:.1f}"),
        ]

        rows_html = ""
        for tf, sig, ema_s, rsi_s in mtf_data:
            trend_icon = "↑" if sig == "BUY" else "↓" if sig == "SELL" else "↔"
            trend_color = GREEN if sig == "BUY" else RED if sig == "SELL" else YELLOW
            rows_html += f"""
            <tr>
              <td style="color:#64748b;font-family:JetBrains Mono,monospace;font-size:12px;padding:9px 0;">{tf}</td>
              <td style="text-align:center;"><span style="color:{trend_color};font-size:16px;">{trend_icon}</span></td>
              <td style="text-align:center;color:#4a6080;font-size:11px;font-family:JetBrains Mono,monospace;">{ema_s}</td>
              <td style="text-align:center;color:#64748b;font-size:12px;font-family:JetBrains Mono,monospace;">{rsi_s}</td>
              <td style="text-align:right;">{badge(sig)}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;">
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:left;padding-bottom:10px;">TIMEFRAME</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:center;padding-bottom:10px;">TREND</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:center;padding-bottom:10px;">EMA</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:center;padding-bottom:10px;">RSI</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:right;padding-bottom:10px;">SIGNAL</th>
              </tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with regime_col:
        trend_str = "Strong" if abs(trend_score) > 60 else "Moderate" if abs(trend_score) > 30 else "Weak"
        vol_regime = "High" if vol_ratio > 1.5 else "Normal" if vol_ratio > 0.8 else "Low"
        struct = "Bullish" if regime == "BULLISH" else "Bearish" if regime == "BEARISH" else "Ranging"
        breakout = "Confirmed" if abs(trend_score) > 50 else "Possible" if abs(trend_score) > 30 else "None"
        mom = "Strong Bullish" if trend_score>60 else "Bullish" if trend_score>30 else "Strong Bearish" if trend_score<-60 else "Bearish" if trend_score<-30 else "Neutral"

        def regime_row(label, val, col="#64748b"):
            return f"""
            <tr style="border-top:1px solid #0f1d2e;">
              <td style="color:#4a6080;font-size:12px;padding:9px 0;font-weight:500;">{label}</td>
              <td style="text-align:right;font-size:12px;color:{col};font-weight:600;">{val}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;">
          <div style="font-size:10px;color:#4a6080;letter-spacing:0.12em;font-weight:600;margin-bottom:16px;">MARKET REGIME</div>
          <table style="width:100%;border-collapse:collapse;">
            {regime_row("TREND STRENGTH", trend_str, GREEN if "Strong" in trend_str else YELLOW)}
            {regime_row("VOLATILITY", vol_regime, ORANGE)}
            {regime_row("MARKET STRUCTURE", struct, GREEN if struct=="Bullish" else RED if struct=="Bearish" else YELLOW)}
            {regime_row("BREAKOUT STATUS", breakout, GREEN if breakout=="Confirmed" else YELLOW)}
            {regime_row("MOMENTUM", mom, GREEN if "Bull" in mom else RED if "Bear" in mom else YELLOW)}
          </table>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # SUPPORT & RESISTANCE + TRADING SETUP
    # ══════════════════════════════════════════════════════════════
    section("SUPPORT  &  RESISTANCE  ·  TRADING SETUP")
    sr_col, setup_col = st.columns([1.2, 1])

    with sr_col:
        atr_v2 = float(latest.get("ATR", current_price * 0.02))
        r1 = float(df["High"].tail(50).max())
        r2 = float(df["High"].tail(20).max())
        r3 = r1 + atr_v2
        s1 = float(df["Low"].tail(50).min())
        s2 = float(df["Low"].tail(20).min())
        s3 = s1 - atr_v2

        def sr_row(label, price, strength_dots, color):
            inr = price * inr_rate
            return f"""
            <tr style="border-top:1px solid #0f1d2e;">
              <td style="font-size:12px;color:{color};font-weight:600;padding:8px 0;">{label}</td>
              <td style="text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;color:#f1f5f9;">${price:.4f}</td>
              <td style="text-align:center;font-family:JetBrains Mono,monospace;font-size:11px;color:#4a6080;">₹{inr:,.2f}</td>
              <td style="text-align:right;">{strength_dots}</td>
            </tr>"""

        def dots(n, color):
            return "".join([f"<span style='color:{color};font-size:14px;'>●</span>" for _ in range(n)])

        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;">
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:left;padding-bottom:10px;">LEVEL</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:center;padding-bottom:10px;">USD</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:center;padding-bottom:10px;">INR</th>
                <th style="font-size:10px;color:#4a6080;letter-spacing:0.1em;text-align:right;padding-bottom:10px;">STRENGTH</th>
              </tr>
            </thead>
            <tbody>
              {sr_row("RESISTANCE 3", r3, dots(3,RED), RED)}
              {sr_row("RESISTANCE 2", r1, dots(4,RED), RED)}
              {sr_row("RESISTANCE 1", r2, dots(5,RED), RED)}
              <tr style="border-top:1px dashed #1a2d4a;">
                <td colspan="4" style="text-align:center;color:#3b82f6;font-size:12px;font-family:JetBrains Mono,monospace;padding:6px 0;font-weight:700;">
                  ── CURRENT  ${current_price:.4f}  ·  ₹{price_inr:,.2f} ──
                </td>
              </tr>
              {sr_row("SUPPORT 1",    s2, dots(5,GREEN), GREEN)}
              {sr_row("SUPPORT 2",    s1, dots(4,GREEN), GREEN)}
              {sr_row("SUPPORT 3",    s3, dots(3,GREEN), GREEN)}
            </tbody>
          </table>
        </div>
        """, unsafe_allow_html=True)

    with setup_col:
        entry    = current_price
        sl       = current_price - (atr_v2 * 2)
        tp1      = current_price + (atr_v2 * 3)
        tp2      = current_price + (atr_v2 * 6)
        rr       = (tp1 - entry) / max(entry - sl, 1e-9)
        entry_inr = entry * inr_rate
        sl_inr    = sl * inr_rate
        tp1_inr   = tp1 * inr_rate
        tp2_inr   = tp2 * inr_rate

        def setup_row(label, usd_val, inr_val, cls=""):
            return f"""
            <tr style="border-top:1px solid #0f1d2e;">
              <td style="font-size:12px;color:#4a6080;font-weight:500;padding:9px 0;">{label}</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;" class="{cls}">
                <span style="color:{'#3b82f6' if cls=='entry-val' else '#22c55e' if 'tp' in cls else '#ef4444' if 'sl' in cls else '#f1f5f9'}">${usd_val:.4f}</span><br>
                <span style="font-size:10px;color:#4a6080;">₹{inr_val:,.2f}</span>
              </td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1a2d4a;border-radius:14px;padding:20px;height:100%;">
          <div style="font-size:10px;color:#4a6080;letter-spacing:0.12em;font-weight:600;margin-bottom:16px;">TRADING SETUP</div>
          <table style="width:100%;border-collapse:collapse;">
            {setup_row("ENTRY PRICE", entry, entry_inr, "entry-val")}
            {setup_row("STOP LOSS  (2×ATR)", sl, sl_inr, "sl")}
            {setup_row("TAKE PROFIT 1  (3×ATR)", tp1, tp1_inr, "tp")}
            {setup_row("TAKE PROFIT 2  (6×ATR)", tp2, tp2_inr, "tp")}
            <tr style="border-top:1px solid #1a2d4a;">
              <td style="font-size:12px;color:#4a6080;font-weight:500;padding:9px 0;">RISK / REWARD</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;font-size:15px;font-weight:700;color:#f97316;">1 : {rr:.2f}</td>
            </tr>
            <tr>
              <td style="font-size:12px;color:#4a6080;font-weight:500;padding:9px 0;">ATR (14)</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;font-size:13px;color:#64748b;">${atr_v2:.4f}  ·  ₹{atr_v2*inr_rate:.2f}</td>
            </tr>
          </table>
          <div style="margin-top:16px;padding:12px;background:#0a1320;border-radius:10px;border:1px solid #0f1d2e;">
            <div style="font-size:10px;color:#4a6080;margin-bottom:6px;letter-spacing:0.08em;">RISK WARNING</div>
            <div style="font-size:11px;color:#334155;line-height:1.6;">
              This is informational only. Not financial advice. Crypto assets are highly volatile. Always do your own research.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════════════════
    st.markdown(f"""
    <div class="footer">
      RENDER AI DASHBOARD  ·  Data: CoinGecko · Yahoo Finance · TradingView  ·  
      INR Rate: ₹{inr_rate:.2f}  ·  
      Updated: {datetime.now().strftime("%d %b %Y  %H:%M:%S")}  ·  
      Not financial advice
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
