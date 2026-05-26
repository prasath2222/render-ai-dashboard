# =========================================================
# PERFECT CLEAN RNDR AI DASHBOARD
# app.py
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================

classifier = joblib.load("rf_classifier.pkl")
regressor = joblib.load("rf_regressor.pkl")
features = joblib.load("features.pkl")

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

body {
    background-color: #050816;
}

.main {
    background-color: #050816;
    color: white;
}

.big-title {
    font-size: 54px;
    font-weight: 900;
    color: white;
}

.sub {
    color: #9aa4bf;
    font-size: 18px;
}

.card {
    background: #0b1220;
    border-radius: 20px;
    padding: 25px;
    border: 1px solid #13203b;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.4);
}

.metric-title {
    color: #7f8db0;
    font-size: 18px;
}

.metric-value {
    color: white;
    font-size: 36px;
    font-weight: 800;
}

.green {
    color: #00ff88;
}

.red {
    color: #ff4b4b;
}

.yellow {
    color: #ffd700;
}

.blue {
    color: #4da6ff;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="big-title">🚀 RNDR AI Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub">Live RNDR Analysis • AI Prediction • Smart Trading Setup</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================================
# LIVE RNDR DATA
# =========================================================

df = yf.download(
    "RENDER-USD",
    period="90d",
    interval="1h"
)

df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# =========================================================
# TECHNICAL INDICATORS
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(
    close=df["Close"],
    window=14
).rsi()

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["EMA20"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=20
).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=50
).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(
    close=df["Close"],
    window=200
).ema_indicator()

df["ATR"] = ta.volatility.AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
).average_true_range()

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = df["Returns"].rolling(24).std()

df["Volume_Change"] = df["Volume"].pct_change()

df["Trend"] = np.where(
    df["EMA20"] > df["EMA50"],
    1,
    0
)

df = df.dropna()

# =========================================================
# FEATURE INPUT
# =========================================================

latest = df.iloc[-1]

X = pd.DataFrame([[
    latest["Close"],
    latest["Volume"],
    latest["RSI"],
    latest["MACD"],
    latest["MACD_SIGNAL"],
    latest["EMA20"],
    latest["EMA50"],
    latest["EMA200"],
    latest["ATR"],
    latest["Returns"],
    latest["Volatility"],
    latest["Volume_Change"],
    latest["Trend"]
]], columns=features)

# =========================================================
# PREDICTIONS
# =========================================================

class_pred = classifier.predict(X)[0]
reg_pred = regressor.predict(X)[0]

# =========================================================
# SIGNAL LOGIC
# =========================================================

price = latest["Close"]

ema20 = latest["EMA20"]
ema50 = latest["EMA50"]
ema200 = latest["EMA200"]

rsi = latest["RSI"]

volume_change = latest["Volume_Change"]

confidence = 50

if ema20 > ema50 > ema200:
    confidence += 20

if rsi > 60:
    confidence += 10

if volume_change > 0:
    confidence += 10

confidence = min(confidence, 95)

# =========================================================
# MARKET SIGNAL
# =========================================================

if ema20 > ema50 > ema200 and rsi > 58:
    signal = "BULLISH BREAKOUT"
    signal_color = "green"

elif ema20 < ema50 < ema200 and rsi < 40:
    signal = "BEARISH"
    signal_color = "red"

else:
    signal = "SIDEWAYS"
    signal_color = "yellow"

# =========================================================
# LIVE INR
# =========================================================

usd_inr = requests.get(
    "https://open.er-api.com/v6/latest/USD"
).json()["rates"]["INR"]

price_inr = price * usd_inr

# =========================================================
# TOP CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">RNDR USD</div>
        <div class="metric-value">${price:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">RNDR INR</div>
        <div class="metric-value">₹{price_inr:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">AI Signal</div>
        <div class="metric-value {signal_color}">
            {signal}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value blue">
            {confidence}%
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================================================
# CHART
# =========================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.8, 0.2]
)

# CANDLESTICK

fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="RNDR"
    ),
    row=1,
    col=1
)

# EMA 20

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA20"],
        line=dict(color="orange", width=2),
        name="EMA20"
    ),
    row=1,
    col=1
)

# EMA 50

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA50"],
        line=dict(color="cyan", width=2),
        name="EMA50"
    ),
    row=1,
    col=1
)

# EMA200

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA200"],
        line=dict(color="white", width=2),
        name="EMA200"
    ),
    row=1,
    col=1
)

# VOLUME

fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume"
    ),
    row=2,
    col=1
)

fig.update_layout(
    template="plotly_dark",
    height=800,
    xaxis_rangeslider_visible=False,
    paper_bgcolor="#050816",
    plot_bgcolor="#050816",
    font=dict(color="white")
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# INDICATORS
# =========================================================

st.write("")

st.subheader("📊 Market Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.metric(
        "RSI",
        round(rsi, 2)
    )

with i2:
    st.metric(
        "MACD",
        round(latest["MACD"], 4)
    )

with i3:
    st.metric(
        "ATR",
        round(latest["ATR"], 4)
    )

with i4:
    st.metric(
        "Volatility",
        round(latest["Volatility"], 4)
    )

# =========================================================
# PREDICTION RANGE
# =========================================================

st.write("")

st.subheader("🎯 AI Trading Setup")

upper_target = price * 1.06
lower_target = price * 0.97

st.markdown(f"""
<div class="card">

### Current Price
# ${price:.4f}

### Expected Range
# ${lower_target:.4f} → ${upper_target:.4f}

### Suggested Setup

- Buy Zone : ${ema20:.4f}
- Support : ${ema50:.4f}
- Resistance : ${upper_target:.4f}

</div>
""", unsafe_allow_html=True)
