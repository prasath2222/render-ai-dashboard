# ═══════════════════════════════════════════════════════
# RNDR AI TERMINAL — FULL app.py
# ═══════════════════════════════════════════════════════

import os
import json
import warnings
import time

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
import joblib

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════
# PAGE
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="RNDR AI TERMINAL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════

BG = "#050816"
BG2 = "#0b1023"

CARD = "#0f172a"

BORDER = "#1e293b"

GREEN = "#00ff88"
RED = "#ff4d6d"
YELLOW = "#ffaa00"
BLUE = "#00c2ff"
PURPLE = "#bb86fc"

TXT = "#ffffff"
TXT2 = "#94a3b8"
TXT3 = "#64748b"

# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════

st.markdown(f"""
<style>

html, body, [class*="css"] {{
    background:{BG};
    color:{TXT};
}}

section[data-testid="stSidebar"] {{
    background:{BG2};
    border-right:1px solid {BORDER};
}}

.block-container {{
    padding-top:1rem;
    padding-bottom:2rem;
    max-width:1600px;
}}

.card {{
    background:{CARD};
    border:1px solid {BORDER};
    border-radius:20px;
    padding:18px;
    box-shadow:0 0 20px rgba(0,0,0,0.25);
}}

.metric-title {{
    color:{TXT2};
    font-size:12px;
    font-weight:700;
    letter-spacing:0.1em;
    text-transform:uppercase;
}}

.metric-value {{
    font-size:38px;
    font-weight:900;
    color:white;
    margin-top:8px;
}}

.metric-sub {{
    color:{TXT2};
    margin-top:8px;
    font-size:13px;
}}

.section-title {{
    font-size:26px;
    font-weight:900;
    margin-top:10px;
    margin-bottom:14px;
}}

.live-dot {{
    width:10px;
    height:10px;
    border-radius:50%;
    background:{GREEN};
    box-shadow:0 0 12px {GREEN};
    display:inline-block;
}}

</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# MODEL FILES
# ═══════════════════════════════════════════════════════

MODEL_FILES = [
    "render_ensemble_cls.pkl",
    "render_reg.pkl",
    "render_scaler.pkl",
    "render_features.pkl"
]

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════

st.sidebar.markdown("## ⚙ SETTINGS")

interval = st.sidebar.selectbox(
    "Interval",
    [
        "15m",
        "1h",
        "4h",
        "1d"
    ],
    index=1
)

period = st.sidebar.selectbox(
    "Period",
    [
        "7d",
        "30d",
        "90d",
        "180d"
    ],
    index=2
)

show_ema = st.sidebar.checkbox(
    "EMA",
    True
)

show_bb = st.sidebar.checkbox(
    "Bollinger Bands",
    True
)

show_volume = st.sidebar.checkbox(
    "Volume",
    True
)

# ═══════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=120)

def get_ohlcv(period, interval):

    df = yf.download(
        "RNDR-USD",
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    df.dropna(inplace=True)

    return df


def add_indicators(df):

    close = df["Close"]

    high = df["High"]

    low = df["Low"]

    volume = df["Volume"]

    df["EMA20"] = ta.trend.ema_indicator(
        close,
        window=20
    )

    df["EMA50"] = ta.trend.ema_indicator(
        close,
        window=50
    )

    bb = ta.volatility.BollingerBands(
        close,
        window=20
    )

    df["BB_H"] = bb.bollinger_hband()

    df["BB_L"] = bb.bollinger_lband()

    df["RSI"] = ta.momentum.rsi(
        close,
        window=14
    )

    macd = ta.trend.MACD(close)

    df["MACD"] = macd.macd()

    df["MACD_SIGNAL"] = macd.macd_signal()

    df["ADX"] = ta.trend.adx(
        high,
        low,
        close,
        window=14
    )

    df["ATR"] = ta.volatility.average_true_range(
        high,
        low,
        close,
        window=14
    )

    stoch = ta.momentum.StochasticOscillator(
        high,
        low,
        close
    )

    df["STOCH"] = stoch.stoch()

    df.dropna(inplace=True)

    return df


def metric_card(title, value, sub, color):

    return f"""
    <div class="card">
        <div class="metric-title">
            {title}
        </div>

        <div class="metric-value" style="color:{color}">
            {value}
        </div>

        <div class="metric-sub">
            {sub}
        </div>
    </div>
    """


def section_header(title, sub):

    return f"""
    <div style="
        margin-top:24px;
        margin-bottom:10px;
    ">
        <div class="section-title">
            {title}
        </div>

        <div style="
            color:{TXT2};
            font-size:13px;
            margin-top:-4px;
        ">
            {sub}
        </div>
    </div>
    """


def chart_candles(df):

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

    if show_ema:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA20"],
                name="EMA20",
                line=dict(
                    color=GREEN,
                    width=1.5
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA50"],
                name="EMA50",
                line=dict(
                    color=YELLOW,
                    width=1.5
                )
            )
        )

    if show_bb:

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_H"],
                name="BB High",
                line=dict(
                    color=PURPLE,
                    width=1
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_L"],
                name="BB Low",
                line=dict(
                    color=PURPLE,
                    width=1
                )
            )
        )

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor=BG,

        plot_bgcolor=BG,

        height=650,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        xaxis_rangeslider_visible=False,

        font=dict(
            color="white"
        )
    )

    return fig


def load_models():

    ensemble = joblib.load(
        "render_ensemble_cls.pkl"
    )

    regressor = joblib.load(
        "render_reg.pkl"
    )

    scaler = joblib.load(
        "render_scaler.pkl"
    )

    features = joblib.load(
        "render_features.pkl"
    )

    return (
        ensemble,
        regressor,
        scaler,
        features
    )


def ml_predict(
    df,
    ensemble,
    regressor,
    scaler,
    features
):

    latest = df.iloc[-1:]

    X = latest[features]

    X_scaled = scaler.transform(X)

    cls = ensemble.predict(X_scaled)[0]

    proba = ensemble.predict_proba(X_scaled)[0]

    reg = regressor.predict(X_scaled)[0]

    signal_map = {
        0: "SELL",
        1: "HOLD",
        2: "BUY"
    }

    signal = signal_map.get(
        int(cls),
        "HOLD"
    )

    predicted_price = float(
        latest["Close"].iloc[0]
    ) * (
        1 + (reg / 100)
    )

    return (
        signal,
        proba,
        predicted_price
    )


# ═══════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════

st.markdown(f"""
<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:{BG2};
    border:1px solid {BORDER};
    border-radius:20px;
    padding:22px;
    margin-bottom:18px;
">

<div>

<div style="
    font-size:40px;
    font-weight:900;
    color:white;
">
    RNDR AI TERMINAL
</div>

<div style="
    color:{TXT2};
    margin-top:4px;
    letter-spacing:0.08em;
">
    ADVANCED AI CRYPTO DASHBOARD
</div>

</div>

<div style="
    display:flex;
    align-items:center;
    gap:8px;
    color:{GREEN};
    font-weight:700;
">

<span class="live-dot"></span>
LIVE

</div>

</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════

with st.spinner("Loading RNDR data..."):

    df = get_ohlcv(
        period,
        interval
    )

    df = add_indicators(df)

latest = df.iloc[-1]

current_price = float(
    latest["Close"]
)

# ═══════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════

signal = "HOLD"

confidence = 50

predicted_price = current_price

if all(
    os.path.exists(f)
    for f in MODEL_FILES
):

    try:

        ensemble, regressor, scaler, features = load_models()

        signal, proba, predicted_price = ml_predict(
            df,
            ensemble,
            regressor,
            scaler,
            features
        )

        confidence = int(
            max(proba) * 100
        )

    except Exception as e:

        st.warning(
            f"Prediction error: {e}"
        )

else:

    st.error(
        "MODEL FILES NOT FOUND"
    )

# ═══════════════════════════════════════════════════════
# SIGNAL COLOR
# ═══════════════════════════════════════════════════════

if signal == "BUY":

    sig_color = GREEN

elif signal == "SELL":

    sig_color = RED

else:

    sig_color = YELLOW

# ═══════════════════════════════════════════════════════
# PREDICTED CHANGE
# ═══════════════════════════════════════════════════════

pred_change = (
    (
        predicted_price
        - current_price
    )
    / current_price
) * 100

pred_col = (
    GREEN if pred_change >= 0
    else RED
)

# ═══════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════

c1, c2, c3, c4, c5 = st.columns(5)

with c1:

    st.markdown(
        metric_card(
            "PRICE",
            f"${current_price:.3f}",
            "RNDR / USD",
            GREEN
        ),
        unsafe_allow_html=True
    )

with c2:

    day_change = (
        (
            latest["Close"]
            - df.iloc[-24]["Close"]
        )
        /
        df.iloc[-24]["Close"]
    ) * 100

    col = GREEN if day_change >= 0 else RED

    st.markdown(
        metric_card(
            "24H CHANGE",
            f"{day_change:+.2f}%",
            "Market movement",
            col
        ),
        unsafe_allow_html=True
    )

with c3:

    st.markdown(
        metric_card(
            "AI SIGNAL",
            signal,
            "ML Ensemble",
            sig_color
        ),
        unsafe_allow_html=True
    )

with c4:

    st.markdown(
        metric_card(
            "CONFIDENCE",
            f"{confidence}%",
            "Prediction confidence",
            BLUE
        ),
        unsafe_allow_html=True
    )

with c5:

    st.markdown(
        metric_card(
            "PREDICTED",
            f"${predicted_price:.3f}",
            f"{pred_change:+.2f}%",
            pred_col
        ),
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# CHART
# ═══════════════════════════════════════════════════════

st.markdown(
    section_header(
        "MARKET STRUCTURE",
        "Candlestick + EMA + Bollinger"
    ),
    unsafe_allow_html=True
)

fig = chart_candles(df)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ═══════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════

st.markdown(
    section_header(
        "INDICATORS",
        "Momentum + Trend"
    ),
    unsafe_allow_html=True
)

i1, i2, i3, i4 = st.columns(4)

with i1:

    st.markdown(
        metric_card(
            "RSI",
            f"{latest['RSI']:.2f}",
            "Momentum",
            BLUE
        ),
        unsafe_allow_html=True
    )

with i2:

    st.markdown(
        metric_card(
            "MACD",
            f"{latest['MACD']:.3f}",
            "Trend",
            GREEN
        ),
        unsafe_allow_html=True
    )

with i3:

    st.markdown(
        metric_card(
            "ADX",
            f"{latest['ADX']:.2f}",
            "Trend strength",
            PURPLE
        ),
        unsafe_allow_html=True
    )

with i4:

    st.markdown(
        metric_card(
            "ATR",
            f"{latest['ATR']:.3f}",
            "Volatility",
            YELLOW
        ),
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# SUPPORT / RESISTANCE
# ═══════════════════════════════════════════════════════

support = float(
    df["Low"].tail(50).min()
)

resistance = float(
    df["High"].tail(50).max()
)

st.markdown(
    section_header(
        "SUPPORT & RESISTANCE",
        "Key market zones"
    ),
    unsafe_allow_html=True
)

s1, s2 = st.columns(2)

with s1:

    st.markdown(
        metric_card(
            "SUPPORT",
            f"${support:.3f}",
            "50 candle low",
            GREEN
        ),
        unsafe_allow_html=True
    )

with s2:

    st.markdown(
        metric_card(
            "RESISTANCE",
            f"${resistance:.3f}",
            "50 candle high",
            RED
        ),
        unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════

st.markdown(f"""
<div style="
    margin-top:30px;
    padding:20px;
    text-align:center;
    color:{TXT3};
    border-top:1px solid {BORDER};
    font-size:12px;
">

RNDR AI TERMINAL • STREAMLIT • MACHINE LEARNING • REALTIME DATA

</div>
""", unsafe_allow_html=True)
