import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import time
from streamlit_autorefresh import st_autorefresh
from streamlit.components.v1 import html

# =========================================
# AUTO REFRESH ONLY DATA
# =========================================

st_autorefresh(interval=15000, key="refresh")

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# =========================================
# CSS
# =========================================

st.markdown("""
<style>

html, body, [class*="css"] {
    background: #050816;
    color: white;
    font-family: 'Segoe UI';
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    max-width: 1500px;
}

.main-title {
    font-size: 52px;
    font-weight: bold;
    margin-bottom: 30px;
}

.glass {
    background: rgba(17,24,39,0.75);
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 22px;
}

.metric-label {
    color: #9ca3af;
    font-size: 14px;
}

.metric-value {
    font-size: 34px;
    font-weight: bold;
    margin-top: 8px;
}

.buy {
    color: #00ff99;
}

.sell {
    color: #ff3366;
}

.sideways {
    color: #ffaa00;
}

.indicator-box {
    background: #0f172a;
    padding: 16px;
    border-radius: 16px;
    text-align: center;
    border: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODELS
# =========================================

classifier = joblib.load("rf_classifier.pkl")
regressor = joblib.load("rf_regressor.pkl")
features = joblib.load("features.pkl")

# =========================================
# DOWNLOAD DATA
# =========================================

ticker = "RENDER-USD"

df = yf.download(
    ticker,
    period="90d",
    interval="1h",
    auto_adjust=True,
    progress=False
)

if df.empty:

    ticker = "RNDR-USD"

    df = yf.download(
        ticker,
        period="90d",
        interval="1h",
        auto_adjust=True,
        progress=False
    )

if df.empty:
    st.error("Failed loading RNDR data")
    st.stop()

df.columns = [
    c[0] if isinstance(c, tuple) else c
    for c in df.columns
]

df = df.dropna()

# =========================================
# INR RATE
# =========================================

usd_inr = yf.download(
    "INR=X",
    period="1d",
    interval="1m",
    progress=False
)

usd_inr.columns = [
    c[0] if isinstance(c, tuple) else c
    for c in usd_inr.columns
]

usd_inr_rate = float(
    usd_inr["Close"].dropna().iloc[-1]
)

# =========================================
# INDICATORS
# =========================================

df["RSI"] = ta.momentum.rsi(
    df["Close"],
    window=14
)

macd = ta.trend.MACD(df["Close"])

df["MACD"] = macd.macd()

df["EMA20"] = ta.trend.ema_indicator(
    df["Close"],
    window=20
)

df["EMA50"] = ta.trend.ema_indicator(
    df["Close"],
    window=50
)

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

df["Bull_Trend"] = np.where(
    df["EMA20"] > df["EMA50"],
    1,
    0
)

df["Breakout"] = np.where(
    df["Close"] > df["Close"].rolling(20).max(),
    1,
    0
)

df["Volume_Spike"] = np.where(
    df["Volume"] > df["Volume"].rolling(20).mean() * 1.5,
    1,
    0
)

df["MACD_DIFF"] = macd.macd_diff()

bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20
)

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

df = df.dropna()

# =========================================
# PREDICTION
# =========================================

latest = df[features].iloc[-1:]

signal = classifier.predict(latest)[0]

confidence = np.max(
    classifier.predict_proba(latest)
) * 100

predicted_price = regressor.predict(latest)[0]

current_price = float(df["Close"].iloc[-1])

prev_price = float(df["Close"].iloc[-2])

change_pct = (
    (current_price - prev_price)
    / prev_price
) * 100

inr_price = current_price * usd_inr_rate

# =========================================
# SIGNAL
# =========================================

if signal == 1:
    signal_text = "BUY"
    signal_class = "buy"

elif signal == -1:
    signal_text = "SELL"
    signal_class = "sell"

else:
    signal_text = "SIDEWAYS"
    signal_class = "sideways"

# =========================================
# TITLE
# =========================================

st.markdown(
    '<div class="main-title">🚀 RNDR AI Dashboard</div>',
    unsafe_allow_html=True
)

# =========================================
# TOP METRICS
# =========================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">RNDR USD</div>
        <div class="metric-value">${current_price:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">RNDR INR</div>
        <div class="metric-value">₹{inr_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:

    color = "#00ff99" if change_pct >= 0 else "#ff3366"

    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">24H Change</div>
        <div class="metric-value" style="color:{color}">
            {change_pct:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="glass">
        <div class="metric-label">AI Signal</div>
        <div class="metric-value {signal_class}">
            {signal_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================================
# PREDICTION BOX
# =========================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div class="glass">

<h2 class="{signal_class}">
{signal_text}
</h2>

<h3>
Predicted Price :
${predicted_price:.4f}
</h3>

<h4>
Confidence :
{confidence:.2f}%
</h4>

</div>
""", unsafe_allow_html=True)

# =========================================
# TRADINGVIEW CHART
# =========================================

st.markdown("## 📈 Advanced Trading Chart")

tv_chart = """
<div class="tradingview-widget-container">

<div id="tradingview_chart"></div>

<script src="https://s3.tradingview.com/tv.js"></script>

<script>

new TradingView.widget(
{
"width": "100%",
"height": 750,
"symbol": "BINANCE:RENDERUSDT",
"interval": "60",
"timezone": "Asia/Kolkata",
"theme": "dark",
"style": "1",
"locale": "en",
"toolbar_bg": "#050816",
"enable_publishing": false,
"allow_symbol_change": true,
"save_image": true,
"withdateranges": true,
"hide_side_toolbar": false,
"details": true,
"hotlist": true,
"calendar": true,
"studies": [
"MACD@tv-basicstudies",
"RSI@tv-basicstudies"
],
"container_id": "tradingview_chart"
});

</script>

</div>
"""

html(tv_chart, height=760)

# =========================================
# INDICATORS
# =========================================

st.markdown("## 📊 Indicators")

i1, i2, i3, i4 = st.columns(4)

with i1:
    st.markdown(f"""
    <div class="indicator-box">
    <h4>RSI</h4>
    <h2>{df["RSI"].iloc[-1]:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="indicator-box">
    <h4>MACD</h4>
    <h2>{df["MACD"].iloc[-1]:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class="indicator-box">
    <h4>ATR</h4>
    <h2>{df["ATR"].iloc[-1]:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown(f"""
    <div class="indicator-box">
    <h4>Volatility</h4>
    <h2>{df["Volatility"].iloc[-1]:.4f}</h2>
    </div>
    """, unsafe_allow_html=True)
