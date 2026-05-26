# ==========================================
# ADVANCED RNDR AI DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #050816;
    color: white;
}

.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

.metric-card {
    background: linear-gradient(145deg,#101827,#1e293b);
    border-radius: 20px;
    padding: 22px;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.05);
}

.metric-title {
    color: #94a3b8;
    font-size: 16px;
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

.blue {
    color: #38bdf8;
}

.yellow {
    color: #facc15;
}

.purple {
    color: #c084fc;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🚀 RNDR AI Prediction System")

# ==========================================
# LOAD MODELS
# ==========================================

rf_classifier = joblib.load("rf_classifier.pkl")
gb_classifier = joblib.load("gb_classifier.pkl")

rf_regressor = joblib.load("rf_regressor.pkl")
gb_regressor = joblib.load("gb_regressor.pkl")

features = joblib.load("features.pkl")

# ==========================================
# LIVE DATA
# ==========================================

df = yf.download(
    "RENDER-USD",
    period="120d",
    interval="1h",
    progress=False
)

# ==========================================
# FIX COLUMNS
# ==========================================

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

# ==========================================
# INDICATORS
# ==========================================

df["EMA20"] = close.ewm(span=20).mean()
df["EMA50"] = close.ewm(span=50).mean()
df["EMA200"] = close.ewm(span=200).mean()

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = tr.rolling(14).mean()

df["Volatility"] = close.pct_change().rolling(24).std() * 100

df["BB_MIDDLE"] = close.rolling(20).mean()

std = close.rolling(20).std()

df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)

df["BullTrend"] = np.where(
    (
        (df["EMA20"] > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"])
    ),
    1,
    0
)

df["RecentHigh"] = high.rolling(20).max()

df["BreakoutStrength"] = (
    close - df["RecentHigh"].shift(1)
) / close

df["Momentum5"] = (
    close - close.shift(5)
) / close

df["Momentum10"] = (
    close - close.shift(10)
) / close

df["VolumeMA"] = volume.rolling(20).mean()

df["VolumeSpike"] = (
    volume / df["VolumeMA"]
)

df["BodyStrength"] = (
    abs(df["Close"] - df["Open"])
) / (
    df["High"] - df["Low"] + 0.0001
)

df.dropna(inplace=True)

# ==========================================
# PREDICTION
# ==========================================

latest = df[features].iloc[-1:]

rf_class = rf_classifier.predict(latest)[0]
gb_class = gb_classifier.predict(latest)[0]

direction_votes = [
    rf_class,
    gb_class
]

final_direction = int(round(np.mean(direction_votes)))

signal_map = {
    2: "STRONG BUY",
    1: "BUY",
    0: "SIDEWAYS",
    -1: "SELL",
    -2: "STRONG SELL"
}

signal = signal_map[final_direction]

rf_price = rf_regressor.predict(latest)[0]
gb_price = gb_regressor.predict(latest)[0]

predicted_price = (
    rf_price + gb_price
) / 2

current_price = float(close.iloc[-1])

atr = float(df["ATR"].iloc[-1])

pred_low = current_price - atr
pred_high = current_price + atr

usd_inr = 83.2

price_inr = current_price * usd_inr

# ==========================================
# BUY SELL FLOW
# ==========================================

buy_volume = volume[df["Close"] > df["Open"]].sum()
sell_volume = volume[df["Close"] <= df["Open"]].sum()

buy_ratio = (
    buy_volume / (buy_volume + sell_volume)
) * 100

sell_ratio = 100 - buy_ratio

# ==========================================
# CONFIDENCE
# ==========================================

confidence = min(
    95,
    round(
        abs(predicted_price - current_price)
        / current_price * 100 * 12,
        2
    )
)

# ==========================================
# TOP CARDS
# ==========================================

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">RNDR USD</div>
    <div class="metric-value blue">
    ${current_price:.3f}
    </div>
    </div>
    ''', unsafe_allow_html=True)

with c2:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">RNDR INR</div>
    <div class="metric-value yellow">
    ₹{price_inr:.2f}
    </div>
    </div>
    ''', unsafe_allow_html=True)

with c3:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">AI Signal</div>
    <div class="metric-value green">
    {signal}
    </div>
    </div>
    ''', unsafe_allow_html=True)

with c4:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">Predicted Range</div>
    <div class="metric-value purple">
    ${pred_low:.2f} → ${pred_high:.2f}
    </div>
    </div>
    ''', unsafe_allow_html=True)

with c5:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">Confidence</div>
    <div class="metric-value green">
    {confidence:.1f}%
    </div>
    </div>
    ''', unsafe_allow_html=True)

# ==========================================
# FLOW
# ==========================================

f1, f2 = st.columns(2)

with f1:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">Buy Inflow</div>
    <div class="metric-value green">
    {buy_ratio:.1f}%
    </div>
    </div>
    ''', unsafe_allow_html=True)

with f2:
    st.markdown(f'''
    <div class="metric-card">
    <div class="metric-title">Sell Outflow</div>
    <div class="metric-value red">
    {sell_ratio:.1f}%
    </div>
    </div>
    ''', unsafe_allow_html=True)

# ==========================================
# PRICE CHART
# ==========================================

st.subheader("RNDR EMA Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=close,
    mode="lines",
    name="Price",
    line=dict(color="white", width=2)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA20"],
    mode="lines",
    name="EMA20",
    line=dict(color="cyan", width=2)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    mode="lines",
    name="EMA50",
    line=dict(color="orange", width=2)
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA200"],
    mode="lines",
    name="EMA200",
    line=dict(color="purple", width=2)
))

fig.update_layout(
    template="plotly_dark",
    height=650,
    paper_bgcolor="#101827",
    plot_bgcolor="#101827",
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.caption(
    "Advanced RNDR AI Dashboard"
)
