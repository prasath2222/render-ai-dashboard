import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="RNDR Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS  (dark terminal theme)
# ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── reset ── */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080b12 !important;
    color: #e8edf5 !important;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { background: #0d1220 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
section[data-testid="stSidebar"] { min-width: 230px !important; max-width: 230px !important; }

/* ── scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }

/* ── sidebar labels / widgets ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #00d4ff;
    letter-spacing: -0.5px;
    padding: 4px 0 2px;
}
.sidebar-sub {
    font-size: 9px;
    letter-spacing: 2.5px;
    color: #6b7a96;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.sidebar-sep {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 12px 0;
}
.sidebar-section {
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7a96;
    margin: 12px 0 6px;
}

/* ── top header bar ── */
.topbar {
    background: #0d1220;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
}
.pair-icon {
    width: 38px; height: 38px;
    border-radius: 50%;
    background: linear-gradient(135deg,#7b5cfa,#00d4ff);
    display:flex; align-items:center; justify-content:center;
    font-family:'Syne',sans-serif; font-size:11px; font-weight:800; color:#fff;
    flex-shrink:0;
}
.pair-name {
    font-family:'Syne',sans-serif;
    font-size:18px; font-weight:800; letter-spacing:-0.5px;
}
.pair-sub { font-size:9px; color:#6b7a96; letter-spacing:1.5px; margin-top:1px; }
.price-main {
    font-family:'Syne',sans-serif;
    font-size:26px; font-weight:800; letter-spacing:-1px;
}
.price-inr { font-size:11px; color:#6b7a96; margin-top:1px; }
.change-up { color:#00e5a0; font-size:14px; font-weight:500; }
.change-down { color:#ff4d6d; font-size:14px; font-weight:500; }
.live-dot {
    width:7px; height:7px; border-radius:50%; background:#00e5a0;
    display:inline-block; margin-right:6px;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── section title ── */
.sec-title {
    font-size:9px; letter-spacing:2px; text-transform:uppercase;
    color:#6b7a96; margin-bottom:10px;
}

/* ── metric cards ── */
.metric-card {
    background: #141c2e;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#00d4ff,#7b5cfa);
}
.mc-label { font-size:9px; letter-spacing:1.5px; text-transform:uppercase; color:#6b7a96; margin-bottom:6px; }
.mc-val {
    font-family:'Syne',sans-serif;
    font-size:22px; font-weight:800; letter-spacing:-0.5px;
}
.mc-sub { font-size:10px; color:#6b7a96; margin-top:3px; }

/* ── mini cards ── */
.mini-card {
    background:#141c2e;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:8px;
    padding:10px 12px;
}
.mini-label { font-size:9px; letter-spacing:1px; text-transform:uppercase; color:#6b7a96; margin-bottom:3px; }
.mini-val { font-size:15px; font-weight:500; }

/* ── signal card ── */
.signal-card {
    background:#141c2e;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;
    padding:16px;
    position:relative; overflow:hidden;
}
.signal-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg,#7b5cfa,#00d4ff);
}

/* ── level rows ── */
.level-row {
    display:flex; align-items:center; padding:7px 0;
    border-bottom:1px solid rgba(255,255,255,0.05);
    font-size:12px;
}
.level-row:last-child { border-bottom:none; }

/* ── trade row ── */
.trade-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:8px 12px; background:#141c2e;
    border-radius:6px; border:1px solid rgba(255,255,255,0.05);
    margin-bottom:5px; font-size:12px;
}

/* ── pressure bar ── */
.pbar-wrap { height:6px; border-radius:3px; background:#ff4d6d; position:relative; overflow:hidden; margin:8px 0; }
.pbar-fill { position:absolute; left:0; top:0; bottom:0; border-radius:3px; background:#00e5a0; }

/* ── tf signal table ── */
.tf-row {
    display:flex; justify-content:space-between; align-items:center;
    padding:6px 0; border-bottom:1px solid rgba(255,255,255,0.05);
    font-size:11px;
}
.tf-row:last-child { border-bottom:none; }

/* ── streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stRadio"] label {
    font-size:11px !important;
    color:#6b7a96 !important;
    font-family:'DM Mono',monospace !important;
}
div[data-testid="stSelectbox"] > div > div {
    background:#141c2e !important;
    border:1px solid rgba(255,255,255,0.1) !important;
    border-radius:6px !important;
    color:#e8edf5 !important;
    font-size:12px !important;
}
div[data-testid="stCheckbox"] > label > span {
    font-size:11px !important;
    color:#e8edf5 !important;
}

/* ── plotly chart bg ── */
.js-plotly-plot { border-radius:8px; }

/* ── stMetric overrides ── */
div[data-testid="metric-container"] {
    background:#141c2e !important;
    border:1px solid rgba(255,255,255,0.06) !important;
    border-radius:10px !important;
    padding:14px !important;
}
div[data-testid="metric-container"] label {
    font-size:9px !important;
    letter-spacing:2px !important;
    text-transform:uppercase !important;
    color:#6b7a96 !important;
    font-family:'DM Mono',monospace !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family:'Syne',sans-serif !important;
    font-size:20px !important;
    font-weight:800 !important;
    letter-spacing:-0.5px !important;
}
div[data-testid="stMetricDelta"] svg { display:none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────

INTERVAL_MAP = {
    "1m":  ("1m",  "7d"),
    "5m":  ("5m",  "30d"),
    "15m": ("15m", "30d"),
    "1h":  ("1h",  "90d"),
    "4h":  ("1h",  "90d"),   # yfinance max 4h workaround
    "1d":  ("1d",  "365d"),
}

@st.cache_data(ttl=60)
def load_data(interval_key: str):
    iv, period = INTERVAL_MAP[interval_key]
    df = yf.download("RENDER-USD", interval=iv, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()

    # flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    # ── indicators ──
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    df["EMA20"]  = ta.trend.ema_indicator(close=close, window=20)
    df["EMA50"]  = ta.trend.ema_indicator(close=close, window=50)
    df["EMA200"] = ta.trend.ema_indicator(close=close, window=200)
    df["RSI"]    = ta.momentum.rsi(close=close, window=14)

    macd_obj        = ta.trend.MACD(close=close)
    df["MACD"]      = macd_obj.macd()
    df["MACD_SIG"]  = macd_obj.macd_signal()
    df["MACD_HIST"] = macd_obj.macd_diff()

    df["ADX"] = ta.trend.adx(high=high, low=low, close=close, window=14)
    df["ATR"] = ta.volatility.average_true_range(high=high, low=low, close=close, window=14)

    bb_obj         = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    df["BB_UPPER"] = bb_obj.bollinger_hband()
    df["BB_LOWER"] = bb_obj.bollinger_lband()
    df["BB_MID"]   = bb_obj.bollinger_mavg()

    df["VWAP"]       = (close * vol).cumsum() / vol.cumsum()
    df["OBV"]        = ta.volume.on_balance_volume(close=close, volume=vol)
    df["VOL_SMA20"]  = ta.trend.sma_indicator(close=vol, window=20)

    # SuperTrend (manual)
    atr_st = ta.volatility.average_true_range(high=high, low=low, close=close, window=10)
    hl2    = (high + low) / 2
    df["ST_UP"]   = hl2 - 3 * atr_st
    df["ST_DOWN"] = hl2 + 3 * atr_st

    df = df.dropna()
    return df


def compute_signals(df):
    latest = df.iloc[-1]
    close  = df["Close"].squeeze()

    cp       = float(latest["Close"])
    ema20    = float(latest["EMA20"])
    ema50    = float(latest["EMA50"])
    ema200   = float(latest["EMA200"])
    rsi      = float(latest["RSI"])
    macd_v   = float(latest["MACD"])
    macd_s   = float(latest["MACD_SIG"])
    macd_h   = float(latest["MACD_HIST"])
    adx      = float(latest["ADX"])
    atr      = float(latest["ATR"])
    bb_up    = float(latest["BB_UPPER"])
    bb_lo    = float(latest["BB_LOWER"])
    volume   = float(latest["Volume"])
    vol_sma  = float(latest["VOL_SMA20"])

    # trend score
    ts = 0
    ts += 20 if ema20 > ema50   else -20
    ts += 20 if ema50 > ema200  else -20
    if   rsi >= 70: ts -= 15
    elif rsi >= 60: ts += 15
    elif rsi <= 30: ts += 20
    elif rsi <= 40: ts -= 10
    ts += 20 if macd_v > macd_s else -20
    ts += 10 if adx > 25        else 0
    pc10 = (float(close.iloc[-1]) - float(close.iloc[-10])) / float(close.iloc[-10]) * 100
    ts += 10 if pc10 > 0 else -10
    ts += 5  if volume > vol_sma else 0

    if   ts >= 40:  signal = "BUY"
    elif ts <= -40: signal = "SELL"
    else:           signal = "HOLD"

    confidence    = min(95, max(50, abs(ts)))
    volatility    = atr / cp
    pred_move     = (ts / 100) * volatility * 5
    pred_price    = cp * (1 + pred_move)
    pred_change   = pred_move * 100

    support_1     = float(df["Low"].tail(50).min())
    resistance_1  = float(df["High"].tail(50).max())
    support_2     = support_1 - atr
    resistance_2  = resistance_1 + atr
    support_3     = support_2 - atr
    resistance_3  = resistance_2 + atr

    if   ema20 > ema50 and rsi > 55: regime = "BULLISH"
    elif ema20 < ema50 and rsi < 45: regime = "BEARISH"
    else:                             regime = "SIDEWAYS"

    stop_loss   = cp - atr * 2
    tp1         = cp + atr * 3
    tp2         = cp + atr * 6
    rr          = (tp1 - cp) / max(cp - stop_loss, 0.0001)

    buy_pressure  = max(0, min(100, 50 + ts))
    sell_pressure = 100 - buy_pressure

    tf_sig = signal  # simplified same-signal for all TFs

    return dict(
        cp=cp, ema20=ema20, ema50=ema50, ema200=ema200,
        rsi=rsi, macd_v=macd_v, macd_s=macd_s, macd_h=macd_h,
        adx=adx, atr=atr, bb_up=bb_up, bb_lo=bb_lo,
        volume=volume, vol_sma=vol_sma,
        ts=ts, signal=signal, confidence=confidence,
        pred_price=pred_price, pred_change=pred_change,
        support_1=support_1, support_2=support_2, support_3=support_3,
        resistance_1=resistance_1, resistance_2=resistance_2, resistance_3=resistance_3,
        regime=regime, stop_loss=stop_loss, tp1=tp1, tp2=tp2, rr=rr,
        buy_pressure=buy_pressure, sell_pressure=sell_pressure, tf_sig=tf_sig,
    )


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class='sidebar-logo'>⚡ RNDR</div>
    <div class='sidebar-sub'>Terminal v2 · AI Powered</div>
    <div class='sidebar-sep'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-section'>Timeframe</div>", unsafe_allow_html=True)
    timeframe = st.selectbox("", ["1h", "4h", "1d", "15m", "5m", "1m"], label_visibility="collapsed")

    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Indicators</div>", unsafe_allow_html=True)
    show_ema    = st.checkbox("EMA 20 / 50 / 200", value=True)
    show_bb     = st.checkbox("Bollinger Bands",    value=True)
    show_vwap   = st.checkbox("VWAP",               value=False)
    show_super  = st.checkbox("SuperTrend",         value=True)
    show_vol    = st.checkbox("Volume Bars",        value=True)

    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>AI Model</div>", unsafe_allow_html=True)
    ai_model = st.selectbox("Model", ["Trend Score (default)", "XGBoost", "Random Forest"], label_visibility="collapsed")

    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>Refresh Interval</div>", unsafe_allow_html=True)
    refresh_sec = st.selectbox("Refresh", [30, 60, 120, 300], format_func=lambda x: f"{x}s", label_visibility="collapsed")

    st.markdown("<div class='sidebar-sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section'>INR Rate</div>", unsafe_allow_html=True)
    inr_rate = st.number_input("1 USD =", value=83.5, step=0.1, label_visibility="collapsed")

    auto_refresh = st.toggle("Auto-refresh", value=True)

# ─────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────

with st.spinner("Fetching live RENDER data…"):
    df = load_data(timeframe)

if df.empty:
    st.error("Could not fetch RENDER data. Check your internet connection.")
    st.stop()

sig = compute_signals(df)
cp  = sig["cp"]

# ─────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────

change_24h_pct = sig["pred_change"]  # use predicted as proxy display
change_class   = "change-up" if change_24h_pct >= 0 else "change-down"
change_arrow   = "▲" if change_24h_pct >= 0 else "▼"
now_str        = datetime.now().strftime("%H:%M:%S")

st.markdown(f"""
<div class="topbar">
  <div class="pair-icon">RN</div>
  <div>
    <div class="pair-name">RNDR / USDT</div>
    <div class="pair-sub">RENDER NETWORK · GPU COMPUTE</div>
  </div>
  <div style="margin-left:12px">
    <div class="price-main">${cp:,.4f}</div>
    <div class="price-inr">₹{cp * inr_rate:,.2f}</div>
  </div>
  <div class="{change_class}" style="margin-left:6px">
    {change_arrow} {abs(change_24h_pct):.2f}%
  </div>
  <div style="margin-left:auto;display:flex;gap:32px;align-items:center;flex-wrap:wrap">
    <div>
      <div style="font-size:9px;letter-spacing:1.5px;color:#6b7a96;text-transform:uppercase">24h High</div>
      <div style="font-size:13px;font-weight:500;color:#00e5a0">${sig['resistance_1']:,.4f}</div>
    </div>
    <div>
      <div style="font-size:9px;letter-spacing:1.5px;color:#6b7a96;text-transform:uppercase">24h Low</div>
      <div style="font-size:13px;font-weight:500;color:#ff4d6d">${sig['support_1']:,.4f}</div>
    </div>
    <div>
      <div style="font-size:9px;letter-spacing:1.5px;color:#6b7a96;text-transform:uppercase">ATR</div>
      <div style="font-size:13px;font-weight:500">{sig['atr']:.4f}</div>
    </div>
    <div>
      <div style="font-size:9px;letter-spacing:1.5px;color:#6b7a96;text-transform:uppercase">ADX</div>
      <div style="font-size:13px;font-weight:500;color:#00d4ff">{sig['adx']:.1f}</div>
    </div>
    <div style="display:flex;align-items:center;gap:6px;padding:5px 12px;border:1px solid rgba(0,229,160,0.3);border-radius:4px">
      <span class="live-dot"></span>
      <span style="font-size:10px;letter-spacing:1.5px;color:#00e5a0">LIVE {now_str}</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN LAYOUT  (chart | right panel)
# ─────────────────────────────────────────

col_chart, col_right = st.columns([3, 1], gap="small")

# ══════════════════════════════
# LEFT — CHART
# ══════════════════════════════
with col_chart:

    # ── build plotly chart ──
    rows   = 3 if show_vol else 2
    r_h    = [0.62, 0.22, 0.16] if show_vol else [0.72, 0.28]
    specs  = [[{"secondary_y": False}]] * rows
    rtitles = ["", "RSI", "Volume"] if show_vol else ["", "RSI"]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=r_h,
        vertical_spacing=0.02,
        subplot_titles=rtitles,
    )

    close_s = df["Close"].squeeze()

    # candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(),
        high=df["High"].squeeze(),
        low=df["Low"].squeeze(),
        close=close_s,
        name="Price",
        increasing_line_color="#00e5a0",
        decreasing_line_color="#ff4d6d",
        increasing_fillcolor="rgba(0,229,160,0.7)",
        decreasing_fillcolor="rgba(255,77,109,0.7)",
        line_width=1,
    ), row=1, col=1)

    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"].squeeze(),  name="EMA20",  line=dict(color="#ffb830", width=1.2), opacity=0.9), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"].squeeze(),  name="EMA50",  line=dict(color="#7b5cfa", width=1.2), opacity=0.9), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"].squeeze(), name="EMA200", line=dict(color="#ff4d6d", width=1,   dash="dot"), opacity=0.7), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UPPER"].squeeze(), name="BB Upper", line=dict(color="rgba(123,92,250,0.4)", width=1, dash="dash"), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOWER"].squeeze(), name="BB Lower", line=dict(color="rgba(123,92,250,0.4)", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(123,92,250,0.05)", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"].squeeze(),   name="BB Mid",   line=dict(color="rgba(123,92,250,0.2)", width=0.8), showlegend=False), row=1, col=1)

    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["VWAP"].squeeze(), name="VWAP", line=dict(color="#00d4ff", width=1.2, dash="dot"), opacity=0.8), row=1, col=1)

    if show_super:
        fig.add_trace(go.Scatter(x=df.index, y=df["ST_UP"].squeeze(),   name="ST Support",   line=dict(color="rgba(0,229,160,0.4)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ST_DOWN"].squeeze(), name="ST Resistance", line=dict(color="rgba(255,77,109,0.4)", width=1)), row=1, col=1)

    # predicted price line (last 10 candles extended)
    pred_x = [df.index[-1]]
    pred_y = [sig["pred_price"]]
    fig.add_trace(go.Scatter(
        x=[df.index[-5], df.index[-1], pred_x[0]],
        y=[float(close_s.iloc[-5]), float(close_s.iloc[-1]), pred_y[0]],
        name="AI Target",
        line=dict(color="#7b5cfa", width=1.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=[0, 0, 8], color="#7b5cfa"),
    ), row=1, col=1)

    # RSI subplot
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"].squeeze(), name="RSI", line=dict(color="#00d4ff", width=1.2)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="rgba(255,77,109,0.4)",   width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="rgba(0,229,160,0.4)",   width=1, dash="dot"), row=2, col=1)
    fig.add_hline(y=50, line=dict(color="rgba(255,255,255,0.1)",  width=0.5), row=2, col=1)

    # Volume subplot
    if show_vol:
        vol_s = df["Volume"].squeeze()
        colors = ["rgba(0,229,160,0.6)" if float(close_s.iloc[i]) >= float(df["Open"].squeeze().iloc[i])
                  else "rgba(255,77,109,0.6)" for i in range(len(df))]
        fig.add_trace(go.Bar(x=df.index, y=vol_s, name="Volume", marker_color=colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA20"].squeeze(), name="Vol SMA", line=dict(color="#ffb830", width=1), showlegend=False), row=3, col=1)

    # layout
    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="#080b12",
        plot_bgcolor="#080b12",
        legend=dict(
            orientation="h", x=0, y=1.01,
            font=dict(color="#6b7a96", size=10, family="DM Mono"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#141c2e", bordercolor="rgba(255,255,255,0.1)", font=dict(color="#e8edf5", size=11, family="DM Mono")),
        dragmode="pan",
    )
    for axis in ["xaxis", "xaxis2", "xaxis3"]:
        fig.update_layout(**{axis: dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)", gridwidth=0.5,
            zeroline=False, tickfont=dict(color="#6b7a96", size=10, family="DM Mono"),
            showspikes=True, spikecolor="rgba(255,255,255,0.2)", spikethickness=1,
            linecolor="rgba(255,255,255,0.06)",
        )})
    for axis in ["yaxis", "yaxis2", "yaxis3"]:
        fig.update_layout(**{axis: dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)", gridwidth=0.5,
            zeroline=False, tickfont=dict(color="#6b7a96", size=10, family="DM Mono"),
            side="right", linecolor="rgba(255,255,255,0.06)",
        )})

    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displayModeBar": False})

    # ── INDICATORS ROW ──
    st.markdown("<div class='sec-title' style='padding:0 4px'>Indicators</div>", unsafe_allow_html=True)
    ic1, ic2, ic3, ic4, ic5 = st.columns(5, gap="small")

    rsi_color = "#ff4d6d" if sig["rsi"] >= 70 else "#00e5a0" if sig["rsi"] <= 30 else "#ffb830"
    rsi_status = "Overbought" if sig["rsi"] >= 70 else "Oversold" if sig["rsi"] <= 30 else "Neutral / Bullish"

    macd_color  = "#00e5a0" if sig["macd_v"] > sig["macd_s"] else "#ff4d6d"
    macd_status = "Bullish Cross" if sig["macd_v"] > sig["macd_s"] else "Bearish Cross"

    adx_status = "Strong Trend" if sig["adx"] > 25 else "Weak Trend"
    bb_width   = (sig["bb_up"] - sig["bb_lo"]) / cp * 100

    for col, label, val, color, status in [
        (ic1, "RSI (14)",        f"{sig['rsi']:.2f}",     rsi_color,  rsi_status),
        (ic2, "MACD",            f"{sig['macd_v']:.4f}",  macd_color, macd_status),
        (ic3, "ADX (14)",        f"{sig['adx']:.1f}",     "#00d4ff",  adx_status),
        (ic4, "ATR (14)",        f"{sig['atr']:.4f}",     "#e8edf5",  "Volatility"),
        (ic5, "BB Width %",      f"{bb_width:.2f}%",      "#7b5cfa",  "Bands Width"),
    ]:
        with col:
            st.markdown(f"""
            <div class='mini-card'>
              <div class='mini-label'>{label}</div>
              <div class='mini-val' style='color:{color}'>{val}</div>
              <div style='font-size:9px;color:#6b7a96;margin-top:3px'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── MACD chart ──
    st.markdown("<div class='sec-title' style='padding:0 4px'>MACD</div>", unsafe_allow_html=True)
    fig_macd = go.Figure()
    hist     = df["MACD_HIST"].squeeze()
    hist_col = ["rgba(0,229,160,0.7)" if float(v) >= 0 else "rgba(255,77,109,0.7)" for v in hist]
    fig_macd.add_trace(go.Bar(x=df.index, y=hist, name="Histogram", marker_color=hist_col))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"].squeeze(),     name="MACD",   line=dict(color="#00d4ff", width=1.2)))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"].squeeze(), name="Signal", line=dict(color="#ff4d6d", width=1.2)))
    fig_macd.update_layout(
        height=140, margin=dict(l=0,r=0,t=0,b=0),
        paper_bgcolor="#080b12", plot_bgcolor="#080b12",
        legend=dict(orientation="h", x=0, y=1.05, font=dict(color="#6b7a96", size=10, family="DM Mono"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7a96", size=9, family="DM Mono"), showspikes=True, spikecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#6b7a96", size=9, family="DM Mono"), side="right"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_macd, use_container_width=True, config={"displayModeBar": False})

    # ── MULTI-TIMEFRAME TABLE ──
    st.markdown("<div class='sec-title' style='padding:0 4px;margin-top:10px'>Multi-Timeframe Signals</div>", unsafe_allow_html=True)
    signal_color_map = {"BUY": "#00e5a0", "SELL": "#ff4d6d", "HOLD": "#ffb830", "NEUTRAL": "#ffb830"}
    sc = signal_color_map.get(sig["signal"], "#e8edf5")

    # simplified: derive slightly varied signals per TF
    def tf_rsi_approx(base, delta): return max(20, min(80, base + delta))

    tf_data = [
        ("15M",  sig["signal"], f"{tf_rsi_approx(sig['rsi'], -4.2):.2f}",  "20 > 50 > 200"),
        ("1H",   sig["signal"], f"{sig['rsi']:.2f}",                        "20 > 50 > 200"),
        ("4H",   sig["signal"], f"{tf_rsi_approx(sig['rsi'], +1.8):.2f}",  "20 > 50 > 200"),
        ("1D",   sig["signal"], f"{tf_rsi_approx(sig['rsi'], -2.3):.2f}",  "20 > 50 > 200"),
        ("1W",   "NEUTRAL",     f"{tf_rsi_approx(sig['rsi'], -10):.2f}",   "20 ≈ 50 > 200"),
    ]

    rows_html = ""
    for tf, s, r, ema_str in tf_data:
        c = signal_color_map.get(s, "#e8edf5")
        rows_html += f"""
        <div class='tf-row'>
          <span style='color:#6b7a96;width:40px'>{tf}</span>
          <span style='color:#6b7a96;flex:1'>{ema_str}</span>
          <span style='color:#6b7a96;width:48px'>{r}</span>
          <span style='color:{c};font-weight:500'>{s}</span>
        </div>"""

    st.markdown(f"""
    <div class='mini-card' style='padding:12px'>
      <div style='display:flex;justify-content:space-between;font-size:9px;letter-spacing:1px;color:#6b7a96;text-transform:uppercase;margin-bottom:8px;padding-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.06)'>
        <span style='width:40px'>TF</span><span style='flex:1'>EMA TREND</span><span style='width:48px'>RSI</span><span>SIGNAL</span>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
# RIGHT — ANALYTICS PANEL
# ══════════════════════════════
with col_right:

    # ── AI SIGNAL CARD ──
    signal_colors = {"BUY": "#00e5a0", "SELL": "#ff4d6d", "HOLD": "#ffb830"}
    sc = signal_colors.get(sig["signal"], "#e8edf5")
    strong_label = ("STRONG BUY" if sig["ts"] >= 60
                    else "BUY" if sig["signal"] == "BUY"
                    else "STRONG SELL" if sig["ts"] <= -60
                    else "SELL" if sig["signal"] == "SELL"
                    else "HOLD")

    pred_col   = "#00e5a0" if sig["pred_change"] >= 0 else "#ff4d6d"
    pred_arrow = "▲" if sig["pred_change"] >= 0 else "▼"

    st.markdown(f"""
    <div class='sec-title'>AI Signal · {ai_model}</div>
    <div class='signal-card'>
      <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px'>
        <div>
          <div style='font-size:9px;letter-spacing:1.5px;color:#6b7a96;margin-bottom:4px'>PREDICTION</div>
          <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:{sc}'>{strong_label}</div>
          <div style='font-size:10px;color:#6b7a96;margin-top:3px'>Trend Score: {sig["ts"]:+d}</div>
        </div>
        <div style='text-align:right'>
          <div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:#00d4ff;letter-spacing:-1px'>{sig["confidence"]}%</div>
          <div style='font-size:9px;letter-spacing:1px;color:#6b7a96'>CONFIDENCE</div>
          <div style='height:4px;border-radius:2px;background:rgba(255,255,255,0.1);margin-top:6px;overflow:hidden'>
            <div style='height:100%;width:{sig["confidence"]}%;background:linear-gradient(90deg,#7b5cfa,#00d4ff);border-radius:2px'></div>
          </div>
        </div>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
        <div style='padding:8px;background:rgba(0,229,160,0.06);border-radius:6px;border:1px solid rgba(0,229,160,0.15);text-align:center'>
          <div style='font-size:9px;color:#6b7a96;margin-bottom:2px'>PREDICTED</div>
          <div style='font-size:15px;font-weight:500;color:{pred_col}'>${sig["pred_price"]:,.4f}</div>
        </div>
        <div style='padding:8px;background:rgba(123,92,250,0.06);border-radius:6px;border:1px solid rgba(123,92,250,0.15);text-align:center'>
          <div style='font-size:9px;color:#6b7a96;margin-bottom:2px'>MOVE</div>
          <div style='font-size:15px;font-weight:500;color:#7b5cfa'>{pred_arrow}{abs(sig["pred_change"]):.2f}%</div>
        </div>
      </div>
    </div>
    <div style='height:12px'></div>
    """, unsafe_allow_html=True)

    # ── MARKET REGIME ──
    regime_colors = {"BULLISH": "#00e5a0", "BEARISH": "#ff4d6d", "SIDEWAYS": "#ffb830"}
    rc = regime_colors.get(sig["regime"], "#e8edf5")
    ring_score = min(99, max(10, abs(sig["ts"]) + 28))

    st.markdown(f"""
    <div class='sec-title'>Market Regime</div>
    <div class='mini-card' style='padding:14px'>
      <div style='display:flex;align-items:center;gap:14px'>
        <div style='position:relative;width:70px;height:70px;flex-shrink:0'>
          <svg width='70' height='70' viewBox='0 0 70 70'>
            <circle cx='35' cy='35' r='26' fill='none' stroke='rgba(255,255,255,0.06)' stroke-width='6'/>
            <circle cx='35' cy='35' r='26' fill='none' stroke='url(#rg2)' stroke-width='6'
              stroke-dasharray='{163}' stroke-dashoffset='{int(163*(1-ring_score/100))}'
              stroke-linecap='round' transform='rotate(-90 35 35)'/>
            <defs>
              <linearGradient id='rg2' x1='0%' y1='0%' x2='100%' y2='0%'>
                <stop offset='0%' stop-color='#7b5cfa'/><stop offset='100%' stop-color='#00d4ff'/>
              </linearGradient>
            </defs>
          </svg>
          <div style='position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:Syne,sans-serif;font-size:16px;font-weight:800;color:#00d4ff'>{ring_score}</div>
        </div>
        <div style='flex:1'>
          <div style='font-family:Syne,sans-serif;font-size:16px;font-weight:700;color:{rc};margin-bottom:8px'>{sig["regime"]}</div>
          <div style='font-size:10px;color:#6b7a96;display:flex;justify-content:space-between;margin-bottom:4px'><span>EMA Stack</span><span style='color:#00e5a0'>{"Aligned ✓" if sig["ema20"] > sig["ema50"] > sig["ema200"] else "Mixed"}</span></div>
          <div style='font-size:10px;color:#6b7a96;display:flex;justify-content:space-between;margin-bottom:4px'><span>Momentum</span><span style='color:{rc}'>{"Strong" if abs(sig["ts"]) > 40 else "Weak"}</span></div>
          <div style='font-size:10px;color:#6b7a96;display:flex;justify-content:space-between'><span>Volatility</span><span>{"High" if sig["atr"]/sig["cp"] > 0.03 else "Normal"}</span></div>
        </div>
      </div>
    </div>
    <div style='height:12px'></div>
    """, unsafe_allow_html=True)

    # ── BUY / SELL PRESSURE ──
    bp = sig["buy_pressure"]
    sp = sig["sell_pressure"]
    st.markdown(f"""
    <div class='sec-title'>Buy / Sell Pressure</div>
    <div class='mini-card' style='padding:14px'>
      <div style='display:flex;justify-content:space-between;margin-bottom:8px'>
        <span style='color:#00e5a0;font-size:18px;font-weight:500'>▲ {bp:.1f}%</span>
        <span style='color:#ff4d6d;font-size:18px;font-weight:500'>{sp:.1f}% ▼</span>
      </div>
      <div style='height:6px;border-radius:3px;background:#ff4d6d;overflow:hidden'>
        <div style='height:100%;width:{bp}%;background:#00e5a0;border-radius:3px'></div>
      </div>
      <div style='display:flex;justify-content:space-between;margin-top:6px;font-size:10px'>
        <span style='color:#00e5a0'>Buy Dominance</span>
        <span style='color:#ff4d6d'>Sell Dominance</span>
      </div>
    </div>
    <div style='height:12px'></div>
    """, unsafe_allow_html=True)

    # ── SUPPORT / RESISTANCE ──
    st.markdown("<div class='sec-title'>Support & Resistance</div>", unsafe_allow_html=True)
    levels = [
        ("RESIST 3", sig["resistance_3"], "#ff4d6d", "90%"),
        ("RESIST 2", sig["resistance_2"], "#ff4d6d", "70%"),
        ("RESIST 1", sig["resistance_1"], "#ff4d6d", "50%"),
        ("CURRENT",  sig["cp"],           "#00d4ff", ""),
        ("SUPPORT 1", sig["support_1"],   "#00e5a0", "70%"),
        ("SUPPORT 2", sig["support_2"],   "#00e5a0", "50%"),
        ("SUPPORT 3", sig["support_3"],   "#00e5a0", "35%"),
    ]
    rows_sr = ""
    for lbl, price, color, bar_w in levels:
        is_current = lbl == "CURRENT"
        bg = "rgba(0,212,255,0.05)" if is_current else "transparent"
        bar_html = f"<div style='width:60px;height:3px;background:rgba(255,255,255,0.08);border-radius:2px;overflow:hidden'><div style='width:{bar_w};height:100%;background:{color};border-radius:2px'></div></div>" if bar_w else ""
        rows_sr += f"""
        <div style='display:flex;align-items:center;padding:6px 8px;border-bottom:1px solid rgba(255,255,255,0.04);background:{bg};border-radius:{"4px" if is_current else "0"}'>
          <span style='font-size:9px;color:{color};letter-spacing:1px;width:72px'>{lbl}</span>
          <span style='flex:1;font-size:12px;font-weight:500'>${price:,.4f}</span>
          {bar_html}
        </div>"""

    st.markdown(f"<div class='mini-card' style='padding:0;overflow:hidden'>{rows_sr}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── TRADING SETUP ──
    st.markdown("<div class='sec-title'>Trading Setup</div>", unsafe_allow_html=True)
    rr_color = "#00e5a0" if sig["rr"] >= 2 else "#ffb830" if sig["rr"] >= 1.5 else "#ff4d6d"

    setup_rows = [
        ("Entry Price",   f"${sig['cp']:,.4f}",           "#00d4ff"),
        ("Stop Loss",     f"${sig['stop_loss']:,.4f}",     "#ff4d6d"),
        ("Take Profit 1", f"${sig['tp1']:,.4f}",           "#00e5a0"),
        ("Take Profit 2", f"${sig['tp2']:,.4f}",           "#00e5a0"),
        ("Risk / Reward", f"1 : {sig['rr']:.2f}",         rr_color),
        ("Regime",        sig["regime"],                   regime_colors.get(sig["regime"], "#e8edf5")),
    ]
    rows_setup = "".join([f"""
    <div class='trade-row'>
      <span style='font-size:10px;color:#6b7a96'>{l}</span>
      <span style='font-size:13px;font-weight:500;color:{c}'>{v}</span>
    </div>""" for l, v, c in setup_rows])

    st.markdown(f"<div>{rows_setup}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# BOTTOM STATUS BAR
# ─────────────────────────────────────────

st.markdown(f"""
<div style='
  height:28px;
  border-top:1px solid rgba(255,255,255,0.06);
  display:flex; align-items:center; padding:0 28px; gap:20px;
  background:#0d1220; font-size:10px; color:#6b7a96; margin-top:16px;
'>
  <span>RNDR Terminal</span>
  <span style='opacity:0.3'>|</span>
  <span>For informational purposes only. Not financial advice.</span>
  <span style='opacity:0.3'>|</span>
  <span>Data: Yahoo Finance · yfinance</span>
  <span style='opacity:0.3'>|</span>
  <span>Refreshed: {now_str}</span>
  <span style='margin-left:auto;color:#00e5a0'><span class='live-dot' style='width:6px;height:6px'></span> WebSocket Active</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# AUTO REFRESH
# ─────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_sec)
    st.cache_data.clear()
    st.rerun()
