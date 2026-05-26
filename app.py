# =========================
# RNDR AI DASHBOARD
# LIVE PREDICTION SYSTEM
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0b1220;
    color: white;
}

.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

.metric-card {
    background: linear-gradient(145deg, #161b22, #1f2937);
    border-radius: 18px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 20px rgba(0,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.05);
}

.metric-title {
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
}

.green {
    color: #22c55e;
}

.red {
    color: #ef4444;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("🚀 RNDR AI Dashboard")

# =========================
# LOAD MODELS
# =========================

rf_classifier = joblib.load("rf_classifier.pkl")
gb_classifier = joblib.load("gb_classifier.pkl")

rf_regressor = joblib.load("rf_regressor.pkl")
gb_regressor = joblib.load("gb_regressor.pkl")

features = joblib.load("features.pkl")

# =========================
# LIVE DATA
# =========================

df = yf.download(
    "RENDER-USD",
    period="90d",
    interval="1h",
    progress=False
)

# =========================
# FIX COLUMNS
# =========================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]].copy()

close = df["Close"].squeeze()
high = df["High"].squeeze()
low = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# =========================
# INDICATORS
# =========================

df["EMA20"] = close.ewm(span=20).mean()
df["EMA50"] = close.ewm(span=50).mean()
df["EMA200"] = close.ewm(span=200).mean()

# RSI

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# MACD

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

# ATR

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = tr.rolling(14).mean()

# BOLLINGER

df["BB_MIDDLE"] = close.rolling(20).mean()

std = close.rolling(20).std()

df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)

# VOLATILITY

df["Volatility"] = close.pct_change().rolling(24).std() * 100

# MOMENTUM

df["Momentum"] = close - close.shift(10)

# VOLUME CHANGE

df["VolumeChange"] = volume.pct_change()

df.dropna(inplace=True)

# =========================
# PREDICTION
# =========================

latest = df[features].iloc[[-1]]

# CLASSIFICATION

rf_direction = rf_classifier.predict(latest)[0]
gb_direction = gb_classifier.predict(latest)[0]

rf_prob = rf_classifier.predict_proba(latest)[0]
gb_prob = gb_classifier.predict_proba(latest)[0]

# REGRESSION

rf_price = rf_regressor.predict(latest)[0]
gb_price = gb_regressor.predict(latest)[0]

# ENSEMBLE

direction_votes = [
    rf_direction,
    gb_direction
]

final_direction = round(np.mean(direction_votes))

predicted_price = np.mean([
    rf_price,
    gb_price
])

confidence = np.mean([
    max(rf_prob),
    max(gb_prob)
]) * 100

# =========================
# SIGNAL
# =========================

if final_direction == 1:
    signal = "UP"
    signal_color = "green"
else:
    signal = "DOWN"
    signal_color = "red"

# =========================
# LIVE PRICE
# =========================

current_price = float(close.iloc[-1])

# =========================
# TOP CARDS
# =========================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RNDR Price</div>
        <div class="metric-value">
            ${current_price:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">AI Direction</div>
        <div class="metric-value {signal_color}">
            {signal}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Price</div>
        <div class="metric-value">
            ${predicted_price:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Confidence</div>
        <div class="metric-value">
            {confidence:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# PRICE CHART
# =========================

st.subheader("RNDR Price + EMA")

price_fig = go.Figure()

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=close,
    mode="lines",
    name="Price",
    line=dict(color="white", width=2)
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA20"],
    mode="lines",
    name="EMA20",
    line=dict(color="cyan", width=2)
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    mode="lines",
    name="EMA50",
    line=dict(color="orange", width=2)
))

price_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA200"],
    mode="lines",
    name="EMA200",
    line=dict(color="purple", width=2)
))

price_fig.update_layout(
    template="plotly_dark",
    height=650,
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(price_fig, use_container_width=True)

# =========================
# RSI + MACD
# =========================

c5, c6 = st.columns(2)

with c5:

    rsi_fig = go.Figure()

    rsi_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI"],
        line=dict(color="blue", width=3)
    ))

    rsi_fig.update_layout(
        template="plotly_dark",
        height=350,
        title="RSI"
    )

    st.plotly_chart(rsi_fig, use_container_width=True)

with c6:

    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MACD"],
        line=dict(color="pink", width=3),
        name="MACD"
    ))

    macd_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MACD_SIGNAL"],
        line=dict(color="white", width=2),
        name="Signal"
    ))

    macd_fig.update_layout(
        template="plotly_dark",
        height=350,
        title="MACD"
    )

    st.plotly_chart(macd_fig, use_container_width=True)

# =========================
# ATR + VOLATILITY
# =========================

c7, c8 = st.columns(2)

with c7:

    atr_fig = go.Figure()

    atr_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["ATR"],
        line=dict(color="yellow", width=3)
    ))

    atr_fig.update_layout(
        template="plotly_dark",
        height=350,
        title="ATR"
    )

    st.plotly_chart(atr_fig, use_container_width=True)

with c8:

    vol_fig = go.Figure()

    vol_fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Volatility"],
        line=dict(color="green", width=3)
    ))

    vol_fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Volatility"
    )

    st.plotly_chart(vol_fig, use_container_width=True)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Live RNDR AI Prediction System • Classification + Regression + Ensemble Models"
)
