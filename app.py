# app.py
# ADVANCED RENDER AI DASHBOARD
# STREAMLIT + TRADINGVIEW STYLE UI

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide",
    page_icon="🚀"
)

# ==========================================
# CSS
# ==========================================

st.markdown("""
<style>

html, body, [class*="css"]{
    background-color:#060b16;
    color:white;
    font-family:Arial;
}

.main{
    background:#060b16;
}

.block-container{
    padding-top:1rem;
}

.metric-card{
    background:linear-gradient(145deg,#0d1529,#08101f);
    border:1px solid #14304d;
    border-radius:18px;
    padding:18px;
    box-shadow:0 0 20px rgba(0,255,255,0.08);
}

.big-title{
    font-size:48px;
    font-weight:800;
    color:white;
}

.green{
    color:#00ff88;
    font-weight:bold;
}

.red{
    color:#ff4d6d;
    font-weight:bold;
}

.purple{
    color:#bb86fc;
    font-weight:bold;
}

.section-title{
    font-size:30px;
    font-weight:700;
    margin-top:10px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙ SETTINGS")

interval = st.sidebar.selectbox(
    "Interval",
    ["15m","1h","4h","1d"],
    index=1
)

period = st.sidebar.selectbox(
    "Period",
    ["7d","30d","90d","180d"],
    index=1
)

# ==========================================
# DATA
# ==========================================

ticker = "RENDER-USD"

df = yf.download(
    ticker,
    period=period,
    interval=interval,
    auto_adjust=True
)

df.dropna(inplace=True)

# ==========================================
# INDICATORS
# ==========================================

close = df["Close"]

df["EMA20"] = EMAIndicator(close, window=20).ema_indicator()
df["EMA50"] = EMAIndicator(close, window=50).ema_indicator()

df["RSI"] = RSIIndicator(close).rsi()

macd = MACD(close)
df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

adx = ADXIndicator(df["High"], df["Low"], close)
df["ADX"] = adx.adx()

atr = AverageTrueRange(df["High"], df["Low"], close)
df["ATR"] = atr.average_true_range()

bb = BollingerBands(close)
df["BB_UPPER"] = bb.bollinger_hband()
df["BB_LOWER"] = bb.bollinger_lband()

stoch = StochasticOscillator(
    df["High"],
    df["Low"],
    close
)

df["STOCH"] = stoch.stoch()

obv = OnBalanceVolumeIndicator(close, df["Volume"])
df["OBV"] = obv.on_balance_volume()

latest = df.iloc[-1]

# ==========================================
# AI LOGIC
# ==========================================

score = 0

if latest["EMA20"] > latest["EMA50"]:
    score += 1

if latest["RSI"] > 55:
    score += 1

if latest["MACD"] > latest["MACD_SIGNAL"]:
    score += 1

if latest["ADX"] > 20:
    score += 1

confidence = int((score / 4) * 100)

if score >= 3:
    signal = "BUY"
    signal_color = "#00ff88"
elif score == 2:
    signal = "HOLD"
    signal_color = "#ffaa00"
else:
    signal = "SELL"
    signal_color = "#ff4d6d"

# ==========================================
# PREDICTION
# ==========================================

current_price = float(latest["Close"])

predicted_price = current_price * (
    1 + (
        (latest["RSI"] - 50) / 500
    )
)

change_24 = (
    (close.iloc[-1] - close.iloc[-2])
    / close.iloc[-2]
) * 100

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="big-title">
🚀 RNDR / USDT AI DASHBOARD
</div>
""", unsafe_allow_html=True)

# ==========================================
# TOP METRICS
# ==========================================

c1,c2,c3,c4,c5 = st.columns(5)

with c1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>PRICE</h3>
    <h1 class="green">${current_price:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

with c2:
    color = "#00ff88" if change_24 > 0 else "#ff4d6d"

    st.markdown(f"""
    <div class="metric-card">
    <h3>24H CHANGE</h3>
    <h1 style="color:{color}">
    {change_24:.2f}%
    </h1>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
    <h3>AI SIGNAL</h3>
    <h1 style="color:{signal_color}">
    {signal}
    </h1>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
    <h3>CONFIDENCE</h3>
    <h1 class="green">{confidence}%</h1>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
    <h3>PREDICTED</h3>
    <h1 class="purple">
    ${predicted_price:.3f}
    </h1>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# CHART
# ==========================================

st.markdown("""
<div class="section-title">
📈 TradingView Style Chart
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name="Price"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA20"],
    line=dict(color="#00ff99", width=2),
    name="EMA20"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    line=dict(color="#ffaa00", width=2),
    name="EMA50"
))

fig.update_layout(
    template="plotly_dark",
    height=700,
    paper_bgcolor="#060b16",
    plot_bgcolor="#060b16",
    xaxis_rangeslider_visible=False,
    font=dict(color="white"),
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# INDICATORS
# ==========================================

st.markdown("""
<div class="section-title">
📊 Indicators
</div>
""", unsafe_allow_html=True)

i1,i2,i3,i4 = st.columns(4)

with i1:
    st.markdown(f"""
    <div class="metric-card">
    <h3>RSI</h3>
    <h1>{latest["RSI"]:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with i2:
    st.markdown(f"""
    <div class="metric-card">
    <h3>MACD</h3>
    <h1>{latest["MACD"]:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

with i3:
    st.markdown(f"""
    <div class="metric-card">
    <h3>ADX</h3>
    <h1>{latest["ADX"]:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

with i4:
    st.markdown(f"""
    <div class="metric-card">
    <h3>ATR</h3>
    <h1>{latest["ATR"]:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SUPPORT RESISTANCE
# ==========================================

st.markdown("""
<div class="section-title">
🧱 Support / Resistance
</div>
""", unsafe_allow_html=True)

support = df["Low"].tail(50).min()
resistance = df["High"].tail(50).max()

s1,s2 = st.columns(2)

with s1:
    st.markdown(f"""
    <div class="metric-card">
    <h2 style="color:#00ff88">
    SUPPORT
    </h2>
    <h1>${support:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

with s2:
    st.markdown(f"""
    <div class="metric-card">
    <h2 style="color:#ff4d6d">
    RESISTANCE
    </h2>
    <h1>${resistance:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MARKET REGIME
# ==========================================

st.markdown("""
<div class="section-title">
🌍 Market Regime
</div>
""", unsafe_allow_html=True)

trend = "Bullish" if latest["EMA20"] > latest["EMA50"] else "Bearish"

st.markdown(f"""
<div class="metric-card">
<h1>{trend}</h1>
<h3>Momentum: {latest["RSI"]:.2f}</h3>
<h3>ADX Strength: {latest["ADX"]:.2f}</h3>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TRADING SETUP
# ==========================================

st.markdown("""
<div class="section-title">
🎯 Trading Setup
</div>
""", unsafe_allow_html=True)

entry = current_price
sl = current_price - latest["ATR"] * 2
tp1 = current_price + latest["ATR"] * 3
tp2 = current_price + latest["ATR"] * 6

t1,t2,t3,t4 = st.columns(4)

with t1:
    st.metric("ENTRY", f"${entry:.3f}")

with t2:
    st.metric("STOP LOSS", f"${sl:.3f}")

with t3:
    st.metric("TAKE PROFIT 1", f"${tp1:.3f}")

with t4:
    st.metric("TAKE PROFIT 2", f"${tp2:.3f}")

st.markdown("""
<br>
<center>
RNDR AI Dashboard • Streamlit • TradingView Style
</center>
""", unsafe_allow_html=True)
