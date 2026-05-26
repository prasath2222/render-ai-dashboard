# =========================================================
# ULTRA ADVANCED AI CRYPTO DASHBOARD
# =========================================================
# INSTALL:
#
# pip install streamlit plotly yfinance pandas numpy \
# ta joblib requests streamlit-autorefresh
#
# RUN:
#
# streamlit run app.py
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from streamlit_autorefresh import st_autorefresh

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import requests

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Crypto Dashboard",
    layout="wide",
    page_icon="🚀"
)

# =========================================================
# AUTO REFRESH
# =========================================================

st_autorefresh(
    interval=60000,
    key="refresh"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #050816;
    color: white;
}

.main {
    background-color: #050816;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(
        145deg,
        #0d1326,
        #111a33
    );

    padding: 20px;

    border-radius: 18px;

    border: 1px solid rgba(0,255,255,0.15);

    box-shadow:
        0 0 20px rgba(0,255,255,0.08);

    text-align: center;
}

.section-card {

    background: linear-gradient(
        145deg,
        #0c1224,
        #10182e
    );

    padding: 20px;

    border-radius: 20px;

    border: 1px solid rgba(0,255,255,0.12);

    box-shadow:
        0 0 25px rgba(0,255,255,0.05);

    margin-bottom: 20px;
}

.green {
    color: #00ff88;
}

.red {
    color: #ff4d6d;
}

.orange {
    color: #ffaa00;
}

.title-text {

    font-size: 42px;

    font-weight: 800;

    color: white;
}

.small-text {

    color: #9ca3af;

    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="title-text">
🚀 AI CRYPTO TRADING DASHBOARD
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

xgb_model = joblib.load(
    "xgb_model.pkl"
)

xgb_reg = joblib.load(
    "xgb_reg.pkl"
)

features = joblib.load(
    "features.pkl"
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙ SETTINGS")

ticker = st.sidebar.selectbox(
    "Coin",
    [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "RENDER-USD",
        "DOGE-USD"
    ]
)

interval = st.sidebar.selectbox(
    "Interval",
    [
        "15m",
        "1h",
        "4h",
        "1d"
    ]
)

period = st.sidebar.selectbox(
    "Period",
    [
        "30d",
        "90d",
        "180d",
        "365d"
    ]
)

# =========================================================
# DOWNLOAD DATA
# =========================================================

df = yf.download(
    ticker,
    interval=interval,
    period=period,
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

# =========================================================
# SERIES
# =========================================================

close = df["Close"]

high = df["High"]

low = df["Low"]

volume = df["Volume"]

# =========================================================
# INDICATORS
# =========================================================

# RSI
df["rsi14"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()

# EMA
df["ema20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()

df["ema50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()

df["ema200"] = ta.trend.EMAIndicator(
    close=close,
    window=200
).ema_indicator()

# MACD
macd = ta.trend.MACD(
    close=close
)

df["macd"] = macd.macd()

df["macd_signal"] = macd.macd_signal()

df["macd_hist"] = macd.macd_diff()

# BB
bb = ta.volatility.BollingerBands(
    close=close
)

df["bb_high"] = bb.bollinger_hband()

df["bb_low"] = bb.bollinger_lband()

df["bb_position"] = (
    close - df["bb_low"]
) / (
    df["bb_high"] - df["bb_low"] + 1e-9
)

# ATR
df["atr"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close
).average_true_range()

df["atr_pct"] = (
    df["atr"] / close
)

# ADX
adx = ta.trend.ADXIndicator(
    high=high,
    low=low,
    close=close
)

df["adx"] = adx.adx()

df["di_diff"] = (
    adx.adx_pos() - adx.adx_neg()
)

# RETURNS
df["returns_1h"] = close.pct_change()

df["returns_24h"] = close.pct_change(24)

# MOMENTUM
df["mom5"] = (
    close / close.shift(5)
) - 1

df["mom10"] = (
    close / close.shift(10)
) - 1

# VOLUME
df["vol_ma20"] = (
    volume
    .rolling(20)
    .mean()
)

df["vol_ratio"] = (
    volume / df["vol_ma20"]
)

# VOLATILITY
df["volatility"] = (
    df["returns_1h"]
    .rolling(24)
    .std()
)

# EXTRA FEATURES
df["rsi7"] = df["rsi14"]

df["ema_cross"] = (
    df["ema20"] - df["ema50"]
)

df["price_ema20"] = (
    close - df["ema20"]
) / df["ema20"]

df["price_ema50"] = (
    close - df["ema50"]
) / df["ema50"]

df["returns_4h"] = (
    close.pct_change(4)
)

df["btc_returns"] = 0

df["btc_vol_ratio"] = 1

df["coin_vs_btc"] = 0

df["fng"] = 50

# =========================================================
# CLEAN
# =========================================================

df.dropna(inplace=True)

# =========================================================
# AI PREDICTION
# =========================================================

latest = df[features].iloc[[-1]]

prob = xgb_model.predict_proba(
    latest
)[0][1]

future_price = xgb_reg.predict(
    latest
)[0]

current_price = close.iloc[-1]

change_pct = (
    (
        future_price - current_price
    )
    / current_price
) * 100

# =========================================================
# SIGNAL
# =========================================================

if prob > 0.58:

    signal = "BUY"

    signal_color = "#00ff88"

elif prob < 0.42:

    signal = "SELL"

    signal_color = "#ff4d6d"

else:

    signal = "HOLD"

    signal_color = "#ffaa00"

# =========================================================
# SUPPORT RESISTANCE
# =========================================================

support1 = df["Low"].rolling(50).min().iloc[-1]

support2 = df["Low"].rolling(100).min().iloc[-1]

resistance1 = df["High"].rolling(50).max().iloc[-1]

resistance2 = df["High"].rolling(100).max().iloc[-1]

# =========================================================
# MARKET STRENGTH
# =========================================================

market_strength = int(prob * 100)

# =========================================================
# TOP METRICS
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.markdown(f"""
    <div class="metric-card">

    <h4>PRICE</h4>

    <h1 class="green">
    ${current_price:.2f}
    </h1>

    </div>
    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""
    <div class="metric-card">

    <h4>24H CHANGE</h4>

    <h1 class="green">
    {change_pct:+.2f}%
    </h1>

    </div>
    """, unsafe_allow_html=True)

with c3:

    st.markdown(f"""
    <div class="metric-card">

    <h4>AI SIGNAL</h4>

    <h1 style="color:{signal_color}">
    {signal}
    </h1>

    </div>
    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""
    <div class="metric-card">

    <h4>CONFIDENCE</h4>

    <h1 class="green">
    {prob*100:.2f}%
    </h1>

    </div>
    """, unsafe_allow_html=True)

with c5:

    st.markdown(f"""
    <div class="metric-card">

    <h4>PREDICTED</h4>

    <h1 style="color:#b266ff">
    ${future_price:.2f}
    </h1>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN CHART
# =========================================================

st.markdown("## 📈 Price Chart")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Price",
        line=dict(
            color="#00d4ff",
            width=2
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema20"],
        name="EMA20",
        line=dict(
            color="#00ff88"
        )
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["ema50"],
        name="EMA50",
        line=dict(
            color="#ffaa00"
        )
    )
)

fig.add_hline(
    y=support1,
    line_color="green"
)

fig.add_hline(
    y=resistance1,
    line_color="red"
)

fig.update_layout(
    template="plotly_dark",
    height=650,
    paper_bgcolor="#050816",
    plot_bgcolor="#050816"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# INDICATORS
# =========================================================

st.markdown("## 📊 Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.markdown(f"""
    <div class="section-card">

    <h4>RSI</h4>

    <h1 class="green">
    {df['rsi14'].iloc[-1]:.2f}
    </h1>

    </div>
    """, unsafe_allow_html=True)

with i2:

    st.markdown(f"""
    <div class="section-card">

    <h4>ADX</h4>

    <h1 class="green">
    {df['adx'].iloc[-1]:.2f}
    </h1>

    </div>
    """, unsafe_allow_html=True)

with i3:

    st.markdown(f"""
    <div class="section-card">

    <h4>ATR %</h4>

    <h1 class="green">
    {df['atr_pct'].iloc[-1]*100:.2f}%
    </h1>

    </div>
    """, unsafe_allow_html=True)

with i4:

    st.markdown(f"""
    <div class="section-card">

    <h4>VOL RATIO</h4>

    <h1 class="green">
    {df['vol_ratio'].iloc[-1]:.2f}
    </h1>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# SUPPORT RESISTANCE
# =========================================================

st.markdown("## 🧱 Support / Resistance")

s1, s2, s3, s4 = st.columns(4)

s1.metric(
    "Support 1",
    f"${support1:.2f}"
)

s2.metric(
    "Support 2",
    f"${support2:.2f}"
)

s3.metric(
    "Resistance 1",
    f"${resistance1:.2f}"
)

s4.metric(
    "Resistance 2",
    f"${resistance2:.2f}"
)

# =========================================================
# MARKET REGIME
# =========================================================

st.markdown("## 🌍 Market Regime")

r1, r2, r3, r4 = st.columns(4)

trend = "Bullish" if signal == "BUY" else "Bearish"

r1.metric(
    "Trend",
    trend
)

r2.metric(
    "Strength",
    market_strength
)

r3.metric(
    "Volatility",
    f"{df['volatility'].iloc[-1]*100:.2f}%"
)

r4.metric(
    "Momentum",
    f"{df['mom10'].iloc[-1]*100:.2f}%"
)

# =========================================================
# TRADING SETUP
# =========================================================

entry = current_price

sl = current_price - (
    df["atr"].iloc[-1] * 2
)

tp1 = current_price + (
    df["atr"].iloc[-1] * 3
)

tp2 = current_price + (
    df["atr"].iloc[-1] * 6
)

st.markdown("## 🎯 Trading Setup")

t1, t2, t3, t4 = st.columns(4)

t1.metric(
    "Entry",
    f"${entry:.2f}"
)

t2.metric(
    "Stop Loss",
    f"${sl:.2f}"
)

t3.metric(
    "Take Profit 1",
    f"${tp1:.2f}"
)

t4.metric(
    "Take Profit 2",
    f"${tp2:.2f}"
)

# =========================================================
# RAW DATA
# =========================================================

with st.expander("📄 Raw Data"):

    st.dataframe(
        df.tail(50)
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<hr>

<center>

AI Dashboard • Real-Time Crypto Analysis

</center>
""", unsafe_allow_html=True)
