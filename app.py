import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

# =========================================================
# PAGE
# =========================================================

st.set_page_config(

    page_title="RNDR AI Dashboard",

    layout="wide"

)

# =========================================================
# STYLE
# =========================================================

st.markdown("""

<style>

html, body, [class*="css"] {

    background-color: #050816;
    color: white;

}

.main-title {

    font-size: 52px;
    font-weight: 800;
    margin-bottom: 10px;

}

.sub-title {

    color: #94a3b8;
    font-size: 20px;
    margin-bottom: 40px;

}

.metric-card {

    background: #111827;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #1e293b;

}

.metric-title {

    font-size: 18px;
    color: #94a3b8;

}

.metric-value {

    font-size: 38px;
    font-weight: bold;
    margin-top: 10px;

}

.signal-box {

    padding: 35px;
    border-radius: 20px;
    text-align: center;
    margin-top: 30px;

}

</style>

""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
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
# DOWNLOAD RNDR
# =========================================================

df = yf.download(

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

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

if isinstance(btc.columns, pd.MultiIndex):

    btc.columns = btc.columns.get_level_values(0)

# =========================================================
# FEATURES
# =========================================================

df["BTC_Close"] = btc["Close"]

df["BTC_Return"] = df["BTC_Close"].pct_change()

df["RSI"] = ta.momentum.RSIIndicator(

    close=df["Close"],

    window=14

).rsi()

macd = ta.trend.MACD(

    close=df["Close"]

)

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

df["ATR"] = ta.volatility.AverageTrueRange(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14

).average_true_range()

df["Returns"] = df["Close"].pct_change()

df["Volatility"] = df["Returns"].rolling(24).std()

df["Momentum5"] = df["Close"] - df["Close"].shift(5)

df["Momentum10"] = df["Close"] - df["Close"].shift(10)

df.dropna(inplace=True)

# =========================================================
# PREDICT
# =========================================================

latest = df[features].iloc[-1:]

prediction = classifier.predict(latest)[0]

future_price = regressor.predict(latest)[0]

confidence = np.max(

    classifier.predict_proba(latest)

) * 100

# =========================================================
# PRICE
# =========================================================

current_price = float(

    df["Close"].iloc[-1]

)

previous_price = float(

    df["Close"].iloc[-24]

)

change_percent = (

    (current_price - previous_price)

    / previous_price

) * 100

usd_inr = 85.20

price_inr = current_price * usd_inr

# =========================================================
# SIGNAL
# =========================================================

if prediction == 2:

    signal = "BULLISH 🚀"

    signal_color = "#00ff88"

elif prediction == 0:

    signal = "BEARISH 🔻"

    signal_color = "#ff4d4d"

else:

    signal = "SIDEWAYS ➖"

    signal_color = "#facc15"

# =========================================================
# SUPPORT RESISTANCE
# =========================================================

support = df["Low"].rolling(50).min().iloc[-1]

resistance = df["High"].rolling(50).max().iloc[-1]

# =========================================================
# TITLE
# =========================================================

st.markdown(

    """

    <div class="main-title">

    RNDR AI Trading Dashboard

    </div>

    <div class="sub-title">

    Live AI Prediction • Trading Setup • Technical Analysis

    </div>

    """,

    unsafe_allow_html=True

)

# =========================================================
# METRICS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:

    st.markdown(f"""

    <div class="metric-card">

        <div class="metric-title">

        RNDR/USD

        </div>

        <div class="metric-value">

        ${current_price:.4f}

        </div>

    </div>

    """, unsafe_allow_html=True)

with c2:

    st.markdown(f"""

    <div class="metric-card">

        <div class="metric-title">

        RNDR/INR

        </div>

        <div class="metric-value">

        ₹{price_inr:.2f}

        </div>

    </div>

    """, unsafe_allow_html=True)

with c3:

    color = "#00ff88" if change_percent > 0 else "#ff4d4d"

    st.markdown(f"""

    <div class="metric-card">

        <div class="metric-title">

        24H Change

        </div>

        <div class="metric-value" style="color:{color};">

        {change_percent:.2f}%

        </div>

    </div>

    """, unsafe_allow_html=True)

with c4:

    st.markdown(f"""

    <div class="metric-card">

        <div class="metric-title">

        Confidence

        </div>

        <div class="metric-value">

        {confidence:.2f}%

        </div>

    </div>

    """, unsafe_allow_html=True)

# =========================================================
# SIGNAL BOX
# =========================================================

st.markdown(f"""

<div class="signal-box"

style="

background:#111827;

border:2px solid {signal_color};

">

<h1 style="

font-size:64px;

color:{signal_color};

">

{signal}

</h1>

<h2 style="

font-size:32px;

margin-top:20px;

">

Predicted Price :

${future_price:.4f}

</h2>

</div>

""", unsafe_allow_html=True)

# =========================================================
# TRADING SETUP
# =========================================================

st.markdown("## Trading Setup")

t1, t2, t3, t4 = st.columns(4)

t1.metric(

    "Support",

    f"${support:.4f}"

)

t2.metric(

    "Resistance",

    f"${resistance:.4f}"

)

t3.metric(

    "RSI",

    f"{df['RSI'].iloc[-1]:.2f}"

)

t4.metric(

    "MACD",

    f"{df['MACD'].iloc[-1]:.4f}"

)

# =========================================================
# TRADINGVIEW CHART
# =========================================================

st.markdown("## Live TradingView Chart")

tradingview_widget = """

<!-- TradingView Widget BEGIN -->

<div class="tradingview-widget-container">

  <div id="tradingview_chart"></div>

  <script type="text/javascript"

  src="https://s3.tradingview.com/tv.js">

  </script>

  <script type="text/javascript">

  new TradingView.widget(

  {

  "width": "100%",

  "height": 700,

  "symbol": "BINANCE:RENDERUSDT",

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
