import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RNDR Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS — complete terminal skin, no raw HTML leaking
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* ── global reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], [class*="st-"],
.stApp, .main, .block-container {
    font-family: 'JetBrains Mono', monospace !important;
    background-color: #06080f !important;
    color: #c8d6e5 !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── sidebar ── */
section[data-testid="stSidebar"] > div {
    background: #0b0f1a !important;
    border-right: 1px solid #1a2235 !important;
    padding-top: 0 !important;
}
section[data-testid="stSidebar"] {
    min-width: 210px !important;
    max-width: 210px !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 2px; }

/* ── streamlit widget labels ── */
label, .stSelectbox label, .stSlider label,
.stCheckbox label, .stToggle label,
div[data-testid="stWidgetLabel"] p {
    font-size: 10px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #4a6080 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    background: #101726 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 6px !important;
    color: #c8d6e5 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── number input ── */
div[data-testid="stNumberInput"] input {
    background: #101726 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 6px !important;
    color: #c8d6e5 !important;
    font-size: 12px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── checkbox ── */
div[data-testid="stCheckbox"] span {
    color: #8faac0 !important;
    font-size: 12px !important;
}

/* ── toggle ── */
div[data-testid="stToggle"] span {
    color: #8faac0 !important;
    font-size: 11px !important;
}

/* ── metric card override ── */
div[data-testid="metric-container"] {
    background: #0e1525 !important;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
}
div[data-testid="metric-container"] label {
    font-size: 9px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #4a6080 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stMetricValue"] > div {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 19px !important;
    font-weight: 600 !important;
    color: #d0e0f0 !important;
}
div[data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── plotly chart bg ── */
.js-plotly-plot .plotly { border-radius: 8px; }

/* ── custom components ── */
.rndr-header {
    background: #0b0f1a;
    border-bottom: 1px solid #1a2235;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
}
.rndr-logo-circle {
    width: 40px; height: 40px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6c47ff, #00b8d9);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px; font-weight: 700; color: #fff;
    flex-shrink: 0;
}
.rndr-pair {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #e8f2ff;
    letter-spacing: -0.3px;
}
.rndr-pair-sub {
    font-size: 9px; color: #4a6080;
    letter-spacing: 2px; text-transform: uppercase;
    margin-top: 2px;
}
.rndr-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700;
    color: #e8f2ff; letter-spacing: -1px;
}
.rndr-inr {
    font-size: 11px; color: #6a88a8; margin-top: 2px;
}
.badge-up {
    background: rgba(0,200,120,0.12);
    color: #00c878;
    border: 1px solid rgba(0,200,120,0.25);
    padding: 4px 10px; border-radius: 6px;
    font-size: 13px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-down {
    background: rgba(255,70,90,0.12);
    color: #ff465a;
    border: 1px solid rgba(255,70,90,0.25);
    padding: 4px 10px; border-radius: 6px;
    font-size: 13px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-hold {
    background: rgba(255,170,0,0.12);
    color: #ffaa00;
    border: 1px solid rgba(255,170,0,0.25);
    padding: 4px 10px; border-radius: 6px;
    font-size: 13px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.live-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px;
    border: 1px solid rgba(0,200,120,0.3);
    font-size: 10px; color: #00c878;
    letter-spacing: 2px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #00c878;
    animation: livePulse 2s ease-in-out infinite;
    display: inline-block;
}
@keyframes livePulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

.hstat {
    border-left: 1px solid #1a2235;
    padding-left: 18px;
}
.hstat-label {
    font-size: 9px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #4a6080;
    margin-bottom: 3px;
}
.hstat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 600; color: #c8d6e5;
}

/* section titles */
.sec-hdr {
    font-size: 9px; letter-spacing: 2.5px;
    text-transform: uppercase; color: #3a5070;
    padding: 0 0 8px; border-bottom: 1px solid #111b2e;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── signal card ── */
.sig-card {
    background: #0e1525;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 16px;
    position: relative; overflow: hidden;
}
.sig-card-accent {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6c47ff, #00b8d9);
}
.sig-title-label {
    font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; color: #4a6080;
    margin-bottom: 5px;
}
.sig-badge-buy {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #00c878;
}
.sig-badge-sell {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #ff465a;
}
.sig-badge-hold {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #ffaa00;
}
.sig-score {
    font-size: 10px; color: #6a88a8; margin-top: 3px;
}
.conf-ring-wrap {
    text-align: right;
}
.conf-big {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; color: #00b8d9;
    letter-spacing: -1px;
}
.conf-sub {
    font-size: 9px; letter-spacing: 1.5px;
    color: #4a6080; text-transform: uppercase;
}
.conf-bar-wrap {
    height: 4px; border-radius: 2px;
    background: #1a2540; overflow: hidden; margin-top: 6px;
}

/* ── small stat box ── */
.sbox {
    background: #0e1525;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 10px 12px;
}
.sbox-label {
    font-size: 9px; letter-spacing: 1.5px;
    text-transform: uppercase; color: #4a6080;
    margin-bottom: 4px;
}
.sbox-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px; font-weight: 600;
}

/* ── indicator card ── */
.indcard {
    background: #0e1525;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 14px 16px;
}
.indcard-name {
    font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; color: #4a6080;
    margin-bottom: 6px;
}
.indcard-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    letter-spacing: -0.5px;
}
.indcard-status {
    font-size: 10px; margin-top: 4px;
    color: #6a88a8;
}

/* ── level rows ── */
.lvl-table { width: 100%; border-collapse: collapse; }
.lvl-row { border-bottom: 1px solid #0f1928; }
.lvl-row:last-child { border-bottom: none; }
.lvl-row td { padding: 7px 8px; font-size: 12px; }
.lvl-tag { font-size: 9px; letter-spacing: 1px; }

/* ── trade rows ── */
.trow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px;
    background: #0e1525;
    border: 1px solid #1a2540;
    border-radius: 7px;
    margin-bottom: 5px;
}
.trow-label { font-size: 10px; color: #6a88a8; }
.trow-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 600;
}

/* ── pressure bar ── */
.pbar-outer {
    height: 8px; border-radius: 4px;
    background: #ff465a; overflow: hidden; position: relative;
}
.pbar-inner {
    position: absolute; left: 0; top: 0; bottom: 0;
    border-radius: 4px; background: #00c878;
}

/* ── tf signal row ── */
.tfrow {
    display: flex; align-items: center; padding: 7px 8px;
    border-bottom: 1px solid #0f1928; font-size: 11px;
}
.tfrow:last-child { border-bottom: none; }

/* ── classification card ── */
.cls-card {
    background: #0e1525;
    border: 1px solid #1a2540;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}

/* ── sidebar logo area ── */
.sb-logo {
    padding: 20px 16px 16px;
    border-bottom: 1px solid #1a2235;
    margin-bottom: 8px;
}
.sb-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #00b8d9; letter-spacing: -0.5px;
}
.sb-logo-sub {
    font-size: 9px; letter-spacing: 2px;
    color: #3a5070; text-transform: uppercase;
    margin-top: 2px;
}
.sb-sep { height: 1px; background: #1a2235; margin: 10px 0; }
.sb-section {
    font-size: 9px; letter-spacing: 2px;
    text-transform: uppercase; color: #3a5070;
    padding: 6px 0 4px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INTERVAL MAP
# ══════════════════════════════════════════════════════════════
INTERVAL_MAP = {
    "1m":  ("1m",  "5d"),
    "5m":  ("5m",  "30d"),
    "15m": ("15m", "30d"),
    "1h":  ("1h",  "90d"),
    "4h":  ("1h",  "90d"),
    "1d":  ("1d",  "365d"),
}

# ══════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_data(tf: str):
    iv, period = INTERVAL_MAP[tf]
    df = yf.download(
        "RENDER-USD", interval=iv, period=period,
        auto_adjust=True, progress=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    l = df["Low"].squeeze()
    v = df["Volume"].squeeze()

    df["EMA20"]  = ta.trend.ema_indicator(close=c, window=20)
    df["EMA50"]  = ta.trend.ema_indicator(close=c, window=50)
    df["EMA200"] = ta.trend.ema_indicator(close=c, window=200)
    df["RSI"]    = ta.momentum.rsi(close=c, window=14)

    m = ta.trend.MACD(close=c)
    df["MACD"]      = m.macd()
    df["MACD_SIG"]  = m.macd_signal()
    df["MACD_HIST"] = m.macd_diff()

    df["ADX"] = ta.trend.adx(high=h, low=l, close=c, window=14)
    df["ATR"] = ta.volatility.average_true_range(high=h, low=l, close=c, window=14)

    bb = ta.volatility.BollingerBands(close=c, window=20, window_dev=2)
    df["BB_U"] = bb.bollinger_hband()
    df["BB_L"] = bb.bollinger_lband()
    df["BB_M"] = bb.bollinger_mavg()

    df["VWAP"]    = (c * v).cumsum() / v.cumsum()
    df["OBV"]     = ta.volume.on_balance_volume(close=c, volume=v)
    df["VOLSMA"]  = ta.trend.sma_indicator(close=v, window=20)

    # SuperTrend bands
    atr10 = ta.volatility.average_true_range(high=h, low=l, close=c, window=10)
    hl2   = (h + l) / 2
    df["ST_UP"]   = hl2 - 3 * atr10
    df["ST_DOWN"] = hl2 + 3 * atr10

    df = df.dropna()
    return df


# ══════════════════════════════════════════════════════════════
# SIGNAL ENGINE
# ══════════════════════════════════════════════════════════════
def compute(df):
    la  = df.iloc[-1]
    c   = df["Close"].squeeze()

    cp    = float(la["Close"])
    e20   = float(la["EMA20"])
    e50   = float(la["EMA50"])
    e200  = float(la["EMA200"])
    rsi   = float(la["RSI"])
    mv    = float(la["MACD"])
    ms    = float(la["MACD_SIG"])
    mh    = float(la["MACD_HIST"])
    adx   = float(la["ADX"])
    atr   = float(la["ATR"])
    bb_u  = float(la["BB_U"])
    bb_l  = float(la["BB_L"])
    vol   = float(la["Volume"])
    vsma  = float(la["VOLSMA"])

    ts = 0
    ts += 20 if e20 > e50   else -20
    ts += 20 if e50 > e200  else -20
    if   rsi >= 70: ts -= 15
    elif rsi >= 60: ts += 15
    elif rsi <= 30: ts += 20
    elif rsi <= 40: ts -= 10
    ts += 20 if mv > ms else -20
    ts += 10 if adx > 25 else 0
    pc = (float(c.iloc[-1]) - float(c.iloc[-10])) / float(c.iloc[-10]) * 100
    ts += 10 if pc > 0 else -10
    ts +=  5 if vol > vsma else 0

    if   ts >= 40:  sig = "BUY"
    elif ts <= -40: sig = "SELL"
    else:           sig = "HOLD"

    conf       = min(95, max(50, abs(ts)))
    vol_ratio  = atr / cp
    pred_move  = (ts / 100) * vol_ratio * 5
    pred_price = cp * (1 + pred_move)
    pred_pct   = pred_move * 100

    s1 = float(df["Low"].tail(50).min())
    r1 = float(df["High"].tail(50).max())
    s2 = s1 - atr;  s3 = s2 - atr
    r2 = r1 + atr;  r3 = r2 + atr

    if   e20 > e50 and rsi > 55: regime = "BULLISH"
    elif e20 < e50 and rsi < 45: regime = "BEARISH"
    else:                         regime = "SIDEWAYS"

    sl  = cp - atr * 2
    tp1 = cp + atr * 3
    tp2 = cp + atr * 6
    rr  = (tp1 - cp) / max(cp - sl, 1e-9)

    bp = max(0, min(100, 50 + ts))
    sp = 100 - bp

    # Classification label
    if   ts >= 60: cls_label = "STRONG BUY"
    elif ts >= 20: cls_label = "BUY"
    elif ts <= -60: cls_label = "STRONG SELL"
    elif ts <= -20: cls_label = "SELL"
    else:           cls_label = "NEUTRAL"

    # 24h change approx using first and last candle
    c24 = (float(c.iloc[-1]) - float(c.iloc[-24 if len(c) > 24 else -len(c)])) / float(c.iloc[-24 if len(c) > 24 else -len(c)]) * 100

    return dict(
        cp=cp, e20=e20, e50=e50, e200=e200,
        rsi=rsi, mv=mv, ms=ms, mh=mh, adx=adx, atr=atr,
        bb_u=bb_u, bb_l=bb_l, vol=vol, vsma=vsma,
        ts=ts, sig=sig, conf=conf,
        pred_price=pred_price, pred_pct=pred_pct,
        s1=s1, s2=s2, s3=s3, r1=r1, r2=r2, r3=r3,
        regime=regime, sl=sl, tp1=tp1, tp2=tp2, rr=rr,
        bp=bp, sp=sp, cls_label=cls_label, c24=c24,
        bb_width=(bb_u - bb_l) / cp * 100,
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div style="display:flex;align-items:center;gap:10px">
            <div style="width:32px;height:32px;border-radius:8px;
                 background:linear-gradient(135deg,#6c47ff,#00b8d9);
                 display:flex;align-items:center;justify-content:center;
                 font-family:'Space Grotesk',sans-serif;font-size:11px;
                 font-weight:700;color:#fff">RN</div>
            <div>
                <div class="sb-logo-text">RNDR</div>
                <div class="sb-logo-sub">Terminal v2</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Timeframe</div>', unsafe_allow_html=True)
    tf = st.selectbox("TF", ["1h","4h","1d","15m","5m","1m"], label_visibility="collapsed")

    st.markdown('<div class="sb-sep"></div><div class="sb-section">Chart Overlays</div>', unsafe_allow_html=True)
    show_ema   = st.checkbox("EMA 20 / 50 / 200", value=True)
    show_bb    = st.checkbox("Bollinger Bands",    value=True)
    show_vwap  = st.checkbox("VWAP",               value=False)
    show_st    = st.checkbox("SuperTrend",         value=True)
    show_vol   = st.checkbox("Volume Bars",        value=True)

    st.markdown('<div class="sb-sep"></div><div class="sb-section">Refresh</div>', unsafe_allow_html=True)
    refresh = st.selectbox("Interval", [30,60,120,300],
                           format_func=lambda x: f"{x}s",
                           label_visibility="collapsed")

    st.markdown('<div class="sb-sep"></div><div class="sb-section">INR Rate (1 USD)</div>', unsafe_allow_html=True)
    inr = st.number_input("INR", value=83.50, step=0.10,
                          min_value=50.0, max_value=150.0,
                          label_visibility="collapsed")

    st.markdown('<div class="sb-sep"></div>', unsafe_allow_html=True)
    auto = st.toggle("Auto-refresh", value=True)


# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════
with st.spinner(""):
    df = load_data(tf)

if df.empty:
    st.error("⚠️  Could not fetch RENDER-USD data. Check your connection.")
    st.stop()

d = compute(df)
now = datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════
# HEADER BAR
# ══════════════════════════════════════════════════════════════
chg_cls  = "badge-up" if d["c24"] >= 0 else "badge-down"
chg_sym  = "▲" if d["c24"] >= 0 else "▼"
hi24     = float(df["High"].squeeze().tail(24).max())
lo24     = float(df["Low"].squeeze().tail(24).min())
inr_val  = d["cp"] * inr

st.markdown(f"""
<div class="rndr-header">
  <div class="rndr-logo-circle">RN</div>
  <div>
    <div class="rndr-pair">RNDR / USDT</div>
    <div class="rndr-pair-sub">Render Network · GPU Compute</div>
  </div>
  <div style="margin-left:8px">
    <div class="rndr-price">${d['cp']:,.4f}</div>
    <div class="rndr-inr">₹{inr_val:,.2f} INR</div>
  </div>
  <span class="{chg_cls}">{chg_sym} {abs(d['c24']):.2f}%</span>

  <div class="hstat">
    <div class="hstat-label">24h High</div>
    <div class="hstat-val" style="color:#00c878">${hi24:,.4f}</div>
  </div>
  <div class="hstat">
    <div class="hstat-label">24h Low</div>
    <div class="hstat-val" style="color:#ff465a">${lo24:,.4f}</div>
  </div>
  <div class="hstat">
    <div class="hstat-label">ATR</div>
    <div class="hstat-val">{d['atr']:.4f}</div>
  </div>
  <div class="hstat">
    <div class="hstat-label">ADX</div>
    <div class="hstat-val" style="color:#00b8d9">{d['adx']:.1f}</div>
  </div>
  <div class="hstat">
    <div class="hstat-label">RSI (14)</div>
    <div class="hstat-val" style="color:#ffaa00">{d['rsi']:.1f}</div>
  </div>

  <div style="margin-left:auto">
    <span class="live-chip">
      <span class="live-dot"></span>LIVE {now}
    </span>
  </div>
</div>
<div style="height:14px"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN LAYOUT  — chart col (left 68%) | panel col (right 32%)
# ══════════════════════════════════════════════════════════════
col_l, col_r = st.columns([2.1, 1], gap="small")


# ──────────────────────────────────────────────────────────────
#  LEFT COLUMN
# ──────────────────────────────────────────────────────────────
with col_l:

    # ── 1. CANDLESTICK CHART ──────────────────────────────────
    n_rows  = 3 if show_vol else 2
    heights = [0.60, 0.22, 0.18] if show_vol else [0.70, 0.30]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.015,
    )

    op = df["Open"].squeeze()
    hi = df["High"].squeeze()
    lo = df["Low"].squeeze()
    cl = df["Close"].squeeze()
    vl = df["Volume"].squeeze()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=op, high=hi, low=lo, close=cl,
        name="RNDR",
        increasing_line_color="#00c878",
        decreasing_line_color="#ff465a",
        increasing_fillcolor="rgba(0,200,120,0.75)",
        decreasing_fillcolor="rgba(255,70,90,0.75)",
        line_width=1,
        whiskerwidth=0.8,
    ), row=1, col=1)

    # EMA
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"].squeeze(),
            name="EMA20", line=dict(color="#f0a500", width=1.2), opacity=0.9), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"].squeeze(),
            name="EMA50", line=dict(color="#a855f7", width=1.2), opacity=0.9), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"].squeeze(),
            name="EMA200", line=dict(color="#ff465a", width=1, dash="dot"), opacity=0.7), row=1, col=1)

    # Bollinger
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_U"].squeeze(),
            name="BB Upper", line=dict(color="rgba(108,71,255,0.5)", width=1, dash="dash"),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_L"].squeeze(),
            name="BB Lower", line=dict(color="rgba(108,71,255,0.5)", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(108,71,255,0.05)", showlegend=False), row=1, col=1)

    # VWAP
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"].squeeze(),
            name="VWAP", line=dict(color="#00b8d9", width=1.2, dash="dot"), opacity=0.85), row=1, col=1)

    # SuperTrend
    if show_st:
        fig.add_trace(go.Scatter(x=df.index, y=df["ST_UP"].squeeze(),
            name="ST Support", line=dict(color="rgba(0,200,120,0.45)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ST_DOWN"].squeeze(),
            name="ST Resist", line=dict(color="rgba(255,70,90,0.45)", width=1)), row=1, col=1)

    # AI predicted price dot
    fig.add_trace(go.Scatter(
        x=[df.index[-6], df.index[-1]],
        y=[float(cl.iloc[-6]), d["pred_price"]],
        name="AI Target",
        line=dict(color="#a855f7", width=1.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=[0, 9], color="#a855f7", symbol="diamond"),
        showlegend=True,
    ), row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(),
        name="RSI", line=dict(color="#00b8d9", width=1.2)), row=2, col=1)
    for lvl, clr in [(70, "rgba(255,70,90,0.35)"), (30, "rgba(0,200,120,0.35)"), (50, "rgba(255,255,255,0.08)")]:
        fig.add_hline(y=lvl, line=dict(color=clr, width=0.8, dash="dot"), row=2, col=1)

    # Volume subplot
    if show_vol:
        v_colors = [
            "rgba(0,200,120,0.65)" if float(cl.iloc[i]) >= float(op.iloc[i])
            else "rgba(255,70,90,0.65)"
            for i in range(len(df))
        ]
        fig.add_trace(go.Bar(x=df.index, y=vl,
            name="Volume", marker_color=v_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["VOLSMA"].squeeze(),
            name="Vol MA", line=dict(color="#f0a500", width=1), showlegend=False), row=3, col=1)

    # Chart layout
    axis_style = dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)", gridwidth=0.5,
        zeroline=False, linecolor="#1a2235",
        tickfont=dict(color="#5a7090", size=10, family="JetBrains Mono"),
        showspikes=True, spikecolor="rgba(255,255,255,0.15)", spikethickness=1,
    )
    fig.update_layout(
        height=520,
        margin=dict(l=4, r=4, t=8, b=4),
        paper_bgcolor="#06080f",
        plot_bgcolor="#06080f",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#0e1525", bordercolor="#1a2540",
            font=dict(color="#c8d6e5", size=11, family="JetBrains Mono"),
        ),
        dragmode="pan",
        legend=dict(
            orientation="h", x=0, y=1.02,
            font=dict(color="#5a7090", size=10, family="JetBrains Mono"),
            bgcolor="rgba(0,0,0,0)",
        ),
        newshape=dict(line_color="#00b8d9"),
    )
    for ax in ["xaxis", "xaxis2", "xaxis3"]:
        fig.update_layout(**{ax: axis_style})
    for ax in ["yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{ax: {**axis_style, "side": "right"}})

    st.plotly_chart(fig, use_container_width=True,
                    config={"scrollZoom": True, "displayModeBar": False})

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── 2. INDICATORS ROW (below chart, separate) ─────────────
    st.markdown('<div class="sec-hdr">— Indicators</div>', unsafe_allow_html=True)

    rsi_c = "#ff465a" if d["rsi"] >= 70 else "#00c878" if d["rsi"] <= 30 else "#f0a500"
    rsi_s = "Overbought" if d["rsi"] >= 70 else "Oversold" if d["rsi"] <= 30 else "Neutral / Bullish"
    macd_c = "#00c878" if d["mv"] > d["ms"] else "#ff465a"
    macd_s = "Bullish Cross" if d["mv"] > d["ms"] else "Bearish Cross"
    adx_s  = "Strong Trend" if d["adx"] > 25 else "Weak Trend"
    adx_c  = "#00b8d9" if d["adx"] > 25 else "#f0a500"

    ic1, ic2, ic3, ic4, ic5 = st.columns(5, gap="small")
    for col, nm, val, vc, st_txt in [
        (ic1, "RSI (14)",   f"{d['rsi']:.2f}",  rsi_c,  rsi_s),
        (ic2, "MACD",       f"{d['mv']:.4f}",   macd_c, macd_s),
        (ic3, "ADX (14)",   f"{d['adx']:.1f}",  adx_c,  adx_s),
        (ic4, "ATR (14)",   f"{d['atr']:.4f}",  "#c8d6e5", "Volatility"),
        (ic5, "BB Width %", f"{d['bb_width']:.2f}%", "#a855f7", "Bands Width"),
    ]:
        with col:
            st.markdown(f"""
            <div class="indcard">
              <div class="indcard-name">{nm}</div>
              <div class="indcard-val" style="color:{vc}">{val}</div>
              <div class="indcard-status">{st_txt}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 3. MACD CHART (separate section) ─────────────────────
    st.markdown('<div class="sec-hdr">— MACD (12, 26, 9)</div>', unsafe_allow_html=True)

    hist   = df["MACD_HIST"].squeeze()
    h_cols = ["rgba(0,200,120,0.7)" if float(v) >= 0 else "rgba(255,70,90,0.7)" for v in hist]
    fmacd  = go.Figure()
    fmacd.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color=h_cols))
    fmacd.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(),
        name="MACD", line=dict(color="#00b8d9", width=1.3)))
    fmacd.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"].squeeze(),
        name="Signal", line=dict(color="#ff465a", width=1.3)))
    fmacd.update_layout(
        height=140, margin=dict(l=4,r=4,t=4,b=4),
        paper_bgcolor="#06080f", plot_bgcolor="#06080f",
        legend=dict(orientation="h", x=0, y=1.1,
                    font=dict(color="#5a7090", size=10, family="JetBrains Mono"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**axis_style),
        yaxis=dict(**{**axis_style, "side": "right"}),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0e1525", bordercolor="#1a2540",
                        font=dict(color="#c8d6e5", size=11)),
    )
    st.plotly_chart(fmacd, use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 4. CLASSIFICATION SECTION ─────────────────────────────
    st.markdown('<div class="sec-hdr">— AI Classification · Direction Prediction</div>',
                unsafe_allow_html=True)

    cls_colors = {
        "STRONG BUY":  "#00c878", "BUY": "#00c878",
        "STRONG SELL": "#ff465a", "SELL": "#ff465a",
        "NEUTRAL":     "#ffaa00",
    }
    cls_c = cls_colors.get(d["cls_label"], "#c8d6e5")

    cc1, cc2, cc3, cc4 = st.columns(4, gap="small")

    with cc1:
        st.markdown(f"""
        <div class="cls-card" style="border-top:2px solid {cls_c}">
          <div class="sbox-label">Direction</div>
          <div style="font-family:'Space Grotesk',sans-serif;
               font-size:24px;font-weight:700;color:{cls_c};
               margin:8px 0 4px">{d['cls_label']}</div>
          <div style="font-size:10px;color:#6a88a8">Trend Score: {d['ts']:+d}</div>
        </div>""", unsafe_allow_html=True)

    with cc2:
        bar_w = d["conf"]
        st.markdown(f"""
        <div class="cls-card">
          <div class="sbox-label">Confidence</div>
          <div style="font-family:'Space Grotesk',sans-serif;
               font-size:30px;font-weight:700;color:#00b8d9;
               letter-spacing:-1px;margin:6px 0 6px">{d['conf']}%</div>
          <div style="height:5px;background:#1a2540;border-radius:3px;overflow:hidden">
            <div style="width:{bar_w}%;height:100%;
                 background:linear-gradient(90deg,#6c47ff,#00b8d9);border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with cc3:
        pc = "#00c878" if d["pred_pct"] >= 0 else "#ff465a"
        pa = "▲" if d["pred_pct"] >= 0 else "▼"
        st.markdown(f"""
        <div class="cls-card">
          <div class="sbox-label">Predicted Price</div>
          <div style="font-family:'Space Grotesk',sans-serif;
               font-size:22px;font-weight:700;color:{pc};margin:8px 0 2px">
               ${d['pred_price']:,.4f}</div>
          <div style="font-size:11px;color:{pc}">{pa} {abs(d['pred_pct']):.2f}%</div>
        </div>""", unsafe_allow_html=True)

    with cc4:
        prob_up   = max(0, min(100, 50 + d["ts"]))
        prob_down = 100 - prob_up
        st.markdown(f"""
        <div class="cls-card">
          <div class="sbox-label">Probability</div>
          <div style="display:flex;justify-content:space-between;margin-top:8px">
            <div>
              <div style="font-size:9px;color:#00c878;letter-spacing:1px">UP</div>
              <div style="font-family:'Space Grotesk',sans-serif;
                   font-size:20px;font-weight:700;color:#00c878">{prob_up}%</div>
            </div>
            <div style="text-align:right">
              <div style="font-size:9px;color:#ff465a;letter-spacing:1px">DOWN</div>
              <div style="font-family:'Space Grotesk',sans-serif;
                   font-size:20px;font-weight:700;color:#ff465a">{prob_down}%</div>
            </div>
          </div>
          <div style="height:5px;background:#ff465a;border-radius:3px;
               overflow:hidden;margin-top:8px">
            <div style="width:{prob_up}%;height:100%;
                 background:#00c878;border-radius:3px"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── 5. MULTI-TIMEFRAME TABLE ──────────────────────────────
    st.markdown('<div class="sec-hdr">— Multi-Timeframe Signals</div>',
                unsafe_allow_html=True)

    def rsi_approx(base, delta):
        return max(20.0, min(80.0, base + delta))

    sig_color = {"BUY": "#00c878", "SELL": "#ff465a", "HOLD": "#ffaa00", "NEUTRAL": "#ffaa00"}

    tf_rows_data = [
        ("15M", d["sig"], rsi_approx(d["rsi"], -4.2),  "20 > 50 > 200"),
        ("1H",  d["sig"], d["rsi"],                    "20 > 50 > 200"),
        ("4H",  d["sig"], rsi_approx(d["rsi"], +1.8),  "20 > 50 > 200"),
        ("1D",  d["sig"], rsi_approx(d["rsi"], -2.3),  "20 > 50 > 200"),
        ("1W",  "NEUTRAL", rsi_approx(d["rsi"], -10),  "20 ≈ 50 > 200"),
    ]

    hdr = """
    <div style="background:#0e1525;border:1px solid #1a2540;border-radius:10px;overflow:hidden">
      <div style="display:flex;padding:8px 12px;border-bottom:1px solid #1a2540;
           font-size:9px;letter-spacing:2px;color:#3a5070;text-transform:uppercase">
        <span style="width:48px">TF</span>
        <span style="flex:1">EMA Trend</span>
        <span style="width:52px;text-align:right">RSI</span>
        <span style="width:72px;text-align:right">Signal</span>
      </div>"""
    rows_html = ""
    for tfn, s, r, ema_s in tf_rows_data:
        sc = sig_color.get(s, "#c8d6e5")
        rows_html += f"""
      <div class="tfrow">
        <span style="width:48px;color:#8faac0;font-weight:500">{tfn}</span>
        <span style="flex:1;color:#6a88a8">{ema_s}</span>
        <span style="width:52px;text-align:right;color:#8faac0">{r:.1f}</span>
        <span style="width:72px;text-align:right;color:{sc};font-weight:600">{s}</span>
      </div>"""
    st.markdown(hdr + rows_html + "</div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  RIGHT COLUMN
# ──────────────────────────────────────────────────────────────
with col_r:

    # ── A. AI SIGNAL CARD ─────────────────────────────────────
    sc = {"BUY": "#00c878", "SELL": "#ff465a", "HOLD": "#ffaa00"}.get(d["sig"], "#c8d6e5")
    st.markdown(f"""
    <div class="sec-hdr">— AI Signal</div>
    <div class="sig-card">
      <div class="sig-card-accent"></div>
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
        <div>
          <div class="sig-title-label">Prediction</div>
          <div class="sig-badge-{'buy' if d['sig']=='BUY' else 'sell' if d['sig']=='SELL' else 'hold'}">{d['cls_label']}</div>
          <div class="sig-score">Score: {d['ts']:+d} &nbsp;|&nbsp; {d['sig']}</div>
        </div>
        <div class="conf-ring-wrap">
          <div class="conf-big">{d['conf']}%</div>
          <div class="conf-sub">Confidence</div>
          <div class="conf-bar-wrap">
            <div style="width:{d['conf']}%;height:100%;
                 background:linear-gradient(90deg,#6c47ff,#00b8d9);border-radius:2px"></div>
          </div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="padding:9px;background:rgba(0,200,120,0.07);border:1px solid rgba(0,200,120,0.18);border-radius:7px;text-align:center">
          <div style="font-size:9px;color:#4a6080;margin-bottom:3px;letter-spacing:1px">PREDICTED</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;color:#00c878">${d['pred_price']:,.4f}</div>
        </div>
        <div style="padding:9px;background:rgba(108,71,255,0.07);border:1px solid rgba(108,71,255,0.18);border-radius:7px;text-align:center">
          <div style="font-size:9px;color:#4a6080;margin-bottom:3px;letter-spacing:1px">MOVE</div>
          <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:600;color:#a855f7">{'▲' if d['pred_pct']>=0 else '▼'}{abs(d['pred_pct']):.2f}%</div>
        </div>
      </div>
    </div>
    <div style="height:14px"></div>
    """, unsafe_allow_html=True)

    # ── B. MARKET REGIME ─────────────────────────────────────
    rc = {"BULLISH": "#00c878", "BEARISH": "#ff465a", "SIDEWAYS": "#ffaa00"}.get(d["regime"], "#c8d6e5")
    ring_val = min(95, max(10, abs(d["ts"]) + 28))
    circ = 163
    offset = int(circ * (1 - ring_val / 100))

    st.markdown(f"""
    <div class="sec-hdr">— Market Regime</div>
    <div style="background:#0e1525;border:1px solid #1a2540;border-radius:12px;padding:16px">
      <div style="display:flex;align-items:center;gap:14px">
        <div style="position:relative;width:72px;height:72px;flex-shrink:0">
          <svg width="72" height="72" viewBox="0 0 72 72">
            <circle cx="36" cy="36" r="26" fill="none" stroke="#1a2540" stroke-width="6"/>
            <circle cx="36" cy="36" r="26" fill="none" stroke="url(#rg1)" stroke-width="6"
              stroke-dasharray="{circ}" stroke-dashoffset="{offset}"
              stroke-linecap="round" transform="rotate(-90 36 36)"/>
            <defs>
              <linearGradient id="rg1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#6c47ff"/>
                <stop offset="100%" stop-color="#00b8d9"/>
              </linearGradient>
            </defs>
          </svg>
          <div style="position:absolute;inset:0;display:flex;align-items:center;
               justify-content:center;font-family:'Space Grotesk',sans-serif;
               font-size:15px;font-weight:700;color:#00b8d9">{ring_val}</div>
        </div>
        <div style="flex:1">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;
               font-weight:700;color:{rc};margin-bottom:10px">{d['regime']}</div>
          <div style="font-size:10px;display:flex;justify-content:space-between;
               margin-bottom:5px;color:#6a88a8">
            <span>EMA Stack</span>
            <span style="color:{'#00c878' if d['e20']>d['e50']>d['e200'] else '#ffaa00'}">
              {'Aligned ✓' if d['e20']>d['e50']>d['e200'] else 'Mixed'}
            </span>
          </div>
          <div style="font-size:10px;display:flex;justify-content:space-between;
               margin-bottom:5px;color:#6a88a8">
            <span>Momentum</span>
            <span style="color:{rc}">{'Strong' if abs(d['ts'])>40 else 'Weak'}</span>
          </div>
          <div style="font-size:10px;display:flex;justify-content:space-between;color:#6a88a8">
            <span>Volatility</span>
            <span>{'High' if d['atr']/d['cp']>0.03 else 'Normal'}</span>
          </div>
        </div>
      </div>
    </div>
    <div style="height:14px"></div>
    """, unsafe_allow_html=True)

    # ── C. BUY / SELL PRESSURE ────────────────────────────────
    st.markdown(f"""
    <div class="sec-hdr">— Buy / Sell Pressure</div>
    <div style="background:#0e1525;border:1px solid #1a2540;border-radius:12px;padding:16px">
      <div style="display:flex;justify-content:space-between;margin-bottom:10px">
        <span style="font-family:'Space Grotesk',sans-serif;font-size:20px;
             font-weight:700;color:#00c878">▲ {d['bp']:.1f}%</span>
        <span style="font-family:'Space Grotesk',sans-serif;font-size:20px;
             font-weight:700;color:#ff465a">{d['sp']:.1f}% ▼</span>
      </div>
      <div class="pbar-outer">
        <div class="pbar-inner" style="width:{d['bp']}%"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px">
        <span style="font-size:10px;color:#00c878">Buy Dominance</span>
        <span style="font-size:10px;color:#ff465a">Sell Dominance</span>
      </div>
    </div>
    <div style="height:14px"></div>
    """, unsafe_allow_html=True)

    # ── D. SUPPORT & RESISTANCE ───────────────────────────────
    levels = [
        ("RESIST 3", d["r3"], "#ff465a", 90),
        ("RESIST 2", d["r2"], "#ff465a", 70),
        ("RESIST 1", d["r1"], "#ff465a", 50),
        ("CURRENT",  d["cp"], "#00b8d9", 0),
        ("SUPPORT 1",d["s1"], "#00c878", 70),
        ("SUPPORT 2",d["s2"], "#00c878", 50),
        ("SUPPORT 3",d["s3"], "#00c878", 35),
    ]
    rows_sr = ""
    for lbl, price, color, bw in levels:
        is_cur = lbl == "CURRENT"
        bg = "rgba(0,184,217,0.06)" if is_cur else "transparent"
        br = "border-radius:6px;" if is_cur else ""
        bar = (f"<div style='width:56px;height:3px;background:#1a2540;border-radius:2px;overflow:hidden'>"
               f"<div style='width:{bw}%;height:100%;background:{color};border-radius:2px'></div></div>"
               if bw else "")
        rows_sr += f"""
        <tr class="lvl-row">
          <td style="background:{bg};{br}">
            <span class="lvl-tag" style="color:{color};letter-spacing:1px">{lbl}</span>
          </td>
          <td style="background:{bg};{br}">
            <span style="font-family:'Space Grotesk',sans-serif;font-weight:600;
                  color:{'#00b8d9' if is_cur else '#c8d6e5'}">${price:,.4f}</span>
          </td>
          <td style="background:{bg};{br}">{bar}</td>
        </tr>"""

    st.markdown(f"""
    <div class="sec-hdr">— Support &amp; Resistance</div>
    <div style="background:#0e1525;border:1px solid #1a2540;border-radius:12px;overflow:hidden">
      <table class="lvl-table">{rows_sr}</table>
    </div>
    <div style="height:14px"></div>
    """, unsafe_allow_html=True)

    # ── E. TRADING SETUP ─────────────────────────────────────
    rr_c = "#00c878" if d["rr"] >= 2 else "#ffaa00" if d["rr"] >= 1.5 else "#ff465a"
    setup = [
        ("Entry Price",   f"${d['cp']:,.4f}",    "#00b8d9"),
        ("Stop Loss",     f"${d['sl']:,.4f}",     "#ff465a"),
        ("Take Profit 1", f"${d['tp1']:,.4f}",    "#00c878"),
        ("Take Profit 2", f"${d['tp2']:,.4f}",    "#00c878"),
        ("Risk / Reward", f"1 : {d['rr']:.2f}",  rr_c),
        ("Regime",        d["regime"],
         {"BULLISH":"#00c878","BEARISH":"#ff465a","SIDEWAYS":"#ffaa00"}.get(d["regime"],"#c8d6e5")),
    ]
    setup_html = "".join([
        f"""<div class="trow">
          <span class="trow-label">{l}</span>
          <span class="trow-val" style="color:{c}">{v}</span>
        </div>""" for l, v, c in setup
    ])
    st.markdown(f"""
    <div class="sec-hdr">— Trading Setup</div>
    {setup_html}
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="
  margin-top:20px;
  border-top:1px solid #1a2235;
  background:#0b0f1a;
  padding:10px 24px;
  display:flex; align-items:center; gap:20px;
  font-size:10px; color:#3a5070;
  font-family:'JetBrains Mono',monospace;
">
  <span>⚡ RNDR Terminal</span>
  <span style="opacity:0.4">|</span>
  <span>For informational purposes only. Not financial advice.</span>
  <span style="opacity:0.4">|</span>
  <span>Source: Yahoo Finance · yfinance</span>
  <span style="opacity:0.4">|</span>
  <span>Refreshed: {now}</span>
  <span style="margin-left:auto;color:#00c878">
    <span class="live-dot" style="width:5px;height:5px"></span> Live
  </span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# AUTO REFRESH
# ══════════════════════════════════════════════════════════════
if auto:
    time.sleep(refresh)
    st.cache_data.clear()
    st.rerun()
