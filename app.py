import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="AI Crypto Dashboard",
    layout="wide"
)

st.title("🚀 AI Crypto Prediction Dashboard")

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
# SETTINGS
# =========================================================

ticker = st.sidebar.selectbox(
    "Select Coin",
    [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "RENDER-USD"
    ]
)

interval = st.sidebar.selectbox(
    "Interval",
    [
        "15m",
        "1h",
        "4h"
    ]
)

period = "90d"

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
# FEATURES
# =========================================================

# RSI
df["rsi14"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()

df["rsi7"] = ta.momentum.RSIIndicator(
    close=close,
    window=7
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

df["ema_cross"] = (
    df["ema20"] - df["ema50"]
)

df["price_ema20"] = (
    close - df["ema20"]
) / df["ema20"]

df["price_ema50"] = (
    close - df["ema50"]
) / df["ema50"]

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
df["returns_1h"] = (
    close.pct_change()
)

df["returns_4h"] = (
    close.pct_change(4)
)

df["returns_24h"] = (
    close.pct_change(24)
)

# MOMENTUM
df["mom5"] = (
    close / close.shift(5)
) - 1

df["mom10"] = (
    close / close.shift(10)
) - 1

# VOLATILITY
df["volatility"] = (
    df["returns_1h"]
    .rolling(24)
    .std()
)

# VOLUME
df["vol_ma20"] = (
    volume
    .rolling(20)
    .mean()
)

df["vol_ratio"] = (
    volume / df["vol_ma20"]
)

# PLACEHOLDERS
df["btc_returns"] = 0

df["btc_vol_ratio"] = 1

df["coin_vs_btc"] = 0

df["fng"] = 50

# =========================================================
# CLEAN
# =========================================================

df.dropna(inplace=True)

# =========================================================
# PREDICTION
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

    color = "green"

elif prob < 0.42:

    signal = "SELL"

    color = "red"

else:

    signal = "HOLD"

    color = "orange"

# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Signal",
    signal
)

c2.metric(
    "Confidence",
    f"{prob*100:.2f}%"
)

c3.metric(
    "Current Price",
    f"${current_price:.2f}"
)

c4.metric(
    "Forecast",
    f"${future_price:.2f}"
)

# =========================================================
# CHART
# =========================================================

fig = go.Figure()

fig.add_trace(

    go.Scatter(
        x=df.index,
        y=df["Close"],
        name="Price"
    )
)

fig.add_trace(

    go.Scatter(
        x=df.index,
        y=df["ema20"],
        name="EMA20"
    )
)

fig.add_trace(

    go.Scatter(
        x=df.index,
        y=df["ema50"],
        name="EMA50"
    )
)

fig.update_layout(
    height=600,
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# EXTRA
# =========================================================

st.subheader("AI Analysis")

st.write(f"### Signal: :{color}[{signal}]")

st.write(f"Probability: {prob:.4f}")

st.write(f"Expected Change: {change_pct:+.2f}%")

# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

support = df["Low"].rolling(50).min().iloc[-1]

resistance = df["High"].rolling(50).max().iloc[-1]

st.subheader("Support / Resistance")

s1, s2 = st.columns(2)

s1.metric(
    "Support",
    f"${support:.2f}"
)

s2.metric(
    "Resistance",
    f"${resistance:.2f}"
)

# =========================================================
# INDICATORS
# =========================================================

st.subheader("Indicators")

i1, i2, i3, i4 = st.columns(4)

i1.metric(
    "RSI",
    f"{df['rsi14'].iloc[-1]:.2f}"
)

i2.metric(
    "ADX",
    f"{df['adx'].iloc[-1]:.2f}"
)

i3.metric(
    "ATR %",
    f"{df['atr_pct'].iloc[-1]*100:.2f}%"
)

i4.metric(
    "Volume Ratio",
    f"{df['vol_ratio'].iloc[-1]:.2f}"
)

# =========================================================
# DATA
# =========================================================

st.subheader("Recent Data")

st.dataframe(
    df.tail(20)
)
