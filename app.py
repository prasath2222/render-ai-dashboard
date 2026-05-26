# =========================================================
# RNDR AI DASHBOARD
# FULL FIXED APP.PY
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib

from plotly import graph_objects as go

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

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp{
        background-color:#050816;
        color:white;
    }

    .main-title{
        font-size:60px;
        font-weight:900;
        color:white;
        margin-bottom:10px;
    }

    .sub{
        color:#9ca3af;
        font-size:22px;
        margin-bottom:40px;
    }

    .card{
        background:#111827;
        padding:30px;
        border-radius:20px;
        border:1px solid #1f2937;
    }

    .metric-title{
        color:#9ca3af;
        font-size:18px;
    }

    .metric-value{
        color:white;
        font-size:42px;
        font-weight:800;
    }

    .buy{
        color:#00ff99;
        font-weight:900;
    }

    .sell{
        color:#ff4d4d;
        font-weight:900;
    }

    .sideways{
        color:#ffaa00;
        font-weight:900;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DOWNLOAD DATA
# =========================================================

df = yf.download(
    "RNDR-USD",
    period="90d",
    interval="1h"
)

df.dropna(inplace=True)

# =========================================================
# INR PRICE
# =========================================================

usd_inr = yf.download(
    "INR=X",
    period="1d",
    interval="1d"
)

usd_inr_rate = float(usd_inr["Close"].iloc[-1])

# =========================================================
# INDICATORS
# =========================================================

df["EMA20"] = ta.trend.ema_indicator(
    df["Close"],
    window=20
)

df["EMA50"] = ta.trend.ema_indicator(
    df["Close"],
    window=50
)

df["EMA200"] = ta.trend.ema_indicator(
    df["Close"],
    window=200
)

df["SMA20"] = ta.trend.sma_indicator(
    df["Close"],
    window=20
)

df["RSI"] = ta.momentum.rsi(
    df["Close"],
    window=14
)

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = macd.macd_diff()

bb = ta.volatility.BollingerBands(df["Close"])

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()

stoch = ta.momentum.StochasticOscillator(
    high=df["High"],
    low=df["Low"],
    close=df["Close"]
)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = stoch.stoch_signal()

df["ATR"] = ta.volatility.average_true_range(
    df["High"],
    df["Low"],
    df["Close"]
)

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = (
    df["Returns"]
    .rolling(24)
    .std()
)

df["Volume_Change"] = (
    df["Volume"]
    .pct_change()
)

df["Trend"] = np.where(
    df["EMA20"] > df["EMA50"],
    1,
    0
)

df["Breakout"] = np.where(
    df["Close"] > df["BB_HIGH"],
    1,
    0
)

df["Volume_Spike"] = np.where(
    df["Volume"] >
    df["Volume"].rolling(20).mean() * 1.5,
    1,
    0
)

df["Bull_Trend"] = np.where(
    (df["EMA20"] > df["EMA50"]) &
    (df["EMA50"] > df["EMA200"]),
    1,
    0
)

df.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

features = [

    "Close",
    "Volume",

    "RSI",

    "MACD",
    "MACD_SIGNAL",
    "MACD_DIFF",

    "EMA20",
    "EMA50",
    "EMA200",

    "SMA20",

    "BB_HIGH",
    "BB_LOW",
    "BB_WIDTH",

    "ATR",

    "STOCH",
    "STOCH_SIGNAL",

    "Returns",
    "Volatility",
    "Volume_Change",

    "Trend",
    "Breakout",
    "Volume_Spike",
    "Bull_Trend"

]

# =========================================================
# PREDICTION
# =========================================================

latest = df[features].iloc[-1:]

predicted_price = regressor.predict(latest)[0]

signal_prob = classifier.predict_proba(latest)[0]

confidence = np.max(signal_prob) * 100

signal_class = classifier.predict(latest)[0]

# =========================================================
# SIGNAL
# =========================================================

if signal_class == 2:
    signal = "BULLISH 🚀"
    signal_classname = "buy"

elif signal_class == 1:
    signal = "SIDEWAYS ⚪"
    signal_classname = "sideways"

else:
    signal = "BEARISH 🔻"
    signal_classname = "sell"

# =========================================================
# LIVE PRICE
# =========================================================

current_price = float(df["Close"].iloc[-1])

previous_price = float(df["Close"].iloc[-2])

change_percent = (
    (current_price - previous_price)
    / previous_price
) * 100

price_inr = current_price * usd_inr_rate

# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <div class='main-title'>
    RNDR AI Trading Dashboard
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='sub'>
    Live AI Market Analysis • Trading Setup • Technical Indicators
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-title'>
            RNDR Price USD
            </div>

            <div class='metric-value'>
            ${current_price:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-title'>
            RNDR Price INR
            </div>

            <div class='metric-value'>
            ₹{price_inr:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-title'>
            AI Predicted Price
            </div>

            <div class='metric-value'>
            ${predicted_price:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:

    st.markdown(
        f"""
        <div class='card'>
            <div class='metric-title'>
            Confidence
            </div>

            <div class='metric-value'>
            {confidence:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# SIGNAL BOX
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div class='card'>

        <div class='metric-title'>
        AI Trading Signal
        </div>

        <div class='metric-value {signal_classname}'>
        {signal}
        </div>

        <br>

        <div style='font-size:22px;color:white;'>

        Live Change :
        {change_percent:.2f}%

        </div>

    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# TRADINGVIEW CHART
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("RNDR Live TradingView Chart")

tradingview_widget = """
<!-- TradingView Widget BEGIN -->
<div class="tradingview-widget-container">
  <div id="tradingview_chart"></div>

  <script type="text/javascript"
  src="https://s3.tradingview.com/tv.js"></script>

  <script type="text/javascript">

  new TradingView.widget(
  {
  "width": "100%",
  "height": 700,
  "symbol": "BINANCE:RNDRUSDT",
  "interval": "60",
  "timezone": "Asia/Kolkata",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#111827",
  "enable_publishing": false,
  "hide_side_toolbar": false,
  "allow_symbol_change": true,
  "container_id": "tradingview_chart"
}
  );

  </script>

</div>
<!-- TradingView Widget END -->
"""

st.components.v1.html(
    tradingview_widget,
    height=720
)

# =========================================================
# INDICATORS
# =========================================================

st.subheader("Technical Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "RSI",
    round(float(df["RSI"].iloc[-1]), 2)
)

c2.metric(
    "MACD",
    round(float(df["MACD"].iloc[-1]), 4)
)

c3.metric(
    "ATR",
    round(float(df["ATR"].iloc[-1]), 4)
)

c4.metric(
    "Volatility",
    round(float(df["Volatility"].iloc[-1]), 4)
)

# =========================================================
# PLOTLY CHART
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
        name="EMA20"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["EMA50"],
        name="EMA50"
    )
)

fig.update_layout(
    template="plotly_dark",
    height=700,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)
