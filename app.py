# =========================================================
# APP.PY
# =========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import plotly.graph_objects as go

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

classifier = joblib.load(

    "rf_classifier.pkl"

)

regressor = joblib.load(

    "rf_regressor.pkl"

)

features = joblib.load(

    "features.pkl"

)

# =========================================================
# DOWNLOAD DATA
# =========================================================

rndr = yf.download(

    "RENDER-USD",

    period="60d",

    interval="1h",

    auto_adjust=True

)

btc = yf.download(

    "BTC-USD",

    period="60d",

    interval="1h",

    auto_adjust=True

)

# =========================================================
# FIX COLUMNS
# =========================================================

if isinstance(rndr.columns, pd.MultiIndex):

    rndr.columns = rndr.columns.get_level_values(0)

if isinstance(btc.columns, pd.MultiIndex):

    btc.columns = btc.columns.get_level_values(0)

df = rndr.copy()

# =========================================================
# BTC FEATURES
# =========================================================

df["BTC_Close"] = btc["Close"]

df["BTC_Return"] = (

    df["BTC_Close"]

    .pct_change()

)

# =========================================================
# RSI
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(

    close=df["Close"],

    window=14

).rsi()

# =========================================================
# MACD
# =========================================================

macd = ta.trend.MACD(

    close=df["Close"]

)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = macd.macd_diff()

# =========================================================
# EMA
# =========================================================

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

# =========================================================
# SMA
# =========================================================

df["SMA20"] = ta.trend.SMAIndicator(

    close=df["Close"],

    window=20

).sma_indicator()

# =========================================================
# ATR
# =========================================================

df["ATR"] = ta.volatility.AverageTrueRange(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14

).average_true_range()

# =========================================================
# BOLLINGER
# =========================================================

bb = ta.volatility.BollingerBands(

    close=df["Close"],

    window=20

)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()

# =========================================================
# STOCH
# =========================================================

stoch = ta.momentum.StochasticOscillator(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14,

    smooth_window=3

)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = stoch.stoch_signal()

# =========================================================
# RETURNS
# =========================================================

df["Returns"] = (

    df["Close"]

    .pct_change()

)

# =========================================================
# VOLATILITY
# =========================================================

df["Volatility"] = (

    df["Returns"]

    .rolling(24)

    .std()

)

# =========================================================
# MOMENTUM
# =========================================================

df["Momentum5"] = (

    df["Close"]

    - df["Close"].shift(5)

)

df["Momentum10"] = (

    df["Close"]

    - df["Close"].shift(10)

)

# =========================================================
# BREAKOUT
# =========================================================

df["Breakout"] = (

    df["Close"]

    >

    df["High"]

    .rolling(20)

    .max()

    .shift(1)

).astype(int)

# =========================================================
# VOLUME
# =========================================================

df["Volume_MA"] = (

    df["Volume"]

    .rolling(20)

    .mean()

)

df["Volume_Spike"] = (

    df["Volume"]

    >

    (df["Volume_MA"] * 1.5)

).astype(int)

# =========================================================
# TREND
# =========================================================

df["Bull_Trend"] = np.where(

    (

        (df["EMA20"] > df["EMA50"])

        &

        (df["EMA50"] > df["EMA200"])

    ),

    1,

    0

)

# =========================================================
# CLEAN
# =========================================================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True

)

df.dropna(inplace=True)

# =========================================================
# LATEST
# =========================================================

latest = df[features].iloc[-1:]

# =========================================================
# PREDICTIONS
# =========================================================

prediction = classifier.predict(

    latest

)[0]

future_price = regressor.predict(

    latest

)[0]

confidence = np.max(

    classifier.predict_proba(latest)

) * 100

current_price = float(

    df["Close"].iloc[-1]

)

# =========================================================
# SIGNAL
# =========================================================

if prediction == 2:

    signal = "BULLISH 🚀"

    color = "#00ff88"

elif prediction == 0:

    signal = "BEARISH 🔻"

    color = "#ff4d4d"

else:

    signal = "SIDEWAYS ➖"

    color = "#ffaa00"

# =========================================================
# TITLE
# =========================================================

st.markdown(

    """

    <h1 style='color:white;'>

    RNDR AI Prediction Dashboard

    </h1>

    """,

    unsafe_allow_html=True

)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(

    "RNDR Price",

    f"${current_price:.4f}"

)

col2.metric(

    "Predicted Price",

    f"${future_price:.4f}"

)

col3.metric(

    "Confidence",

    f"{confidence:.2f}%"

)

col4.metric(

    "Signal",

    signal

)

# =========================================================
# AI SIGNAL BOX
# =========================================================

st.markdown(

    f"""

    <div style="

        background:#111827;

        padding:30px;

        border-radius:20px;

        border:2px solid {color};

        margin-top:20px;

    ">

    <h2 style="color:{color};">

    {signal}

    </h2>

    <h3 style="color:white;">

    AI Predicted Price :

    ${future_price:.4f}

    </h3>

    <h4 style="color:white;">

    Confidence :

    {confidence:.2f}%

    </h4>

    </div>

    """,

    unsafe_allow_html=True

)

# =========================================================
# CHART
# =========================================================

fig = go.Figure()

fig.add_trace(

    go.Candlestick(

        x=df.index,

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"],

        name="RNDR"

    )

)

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["EMA20"],

        line=dict(color="orange"),

        name="EMA20"

    )

)

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["EMA50"],

        line=dict(color="cyan"),

        name="EMA50"

    )

)

fig.update_layout(

    height=700,

    template="plotly_dark",

    xaxis_rangeslider_visible=False,

    title="RNDR Live Market Chart"

)

st.plotly_chart(

    fig,

    use_container_width=True

)

# =========================================================
# INDICATORS
# =========================================================

st.subheader(

    "Technical Indicators"

)

c1, c2, c3, c4 = st.columns(4)

c1.metric(

    "RSI",

    f"{df['RSI'].iloc[-1]:.2f}"

)

c2.metric(

    "MACD",

    f"{df['MACD'].iloc[-1]:.4f}"

)

c3.metric(

    "ATR",

    f"{df['ATR'].iloc[-1]:.4f}"

)

c4.metric(

    "Volatility",

    f"{df['Volatility'].iloc[-1]:.4f}"

)
