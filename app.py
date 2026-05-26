import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# ---------------- CLEAN CSS ----------------

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0d1117;
    color: white;
}

.block-container {
    padding-top: 1rem;
    max-width: 100%;
}

.metric-card {
    background: #161b22;
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 0 18px rgba(0,255,255,0.08);
    margin-bottom: 10px;
}

.metric-title {
    color: #9ca3af;
    font-size: 18px;
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

# ---------------- TITLE ----------------

st.title("🚀 RNDR AI Dashboard")

# ---------------- DATA ----------------

df = yf.download(
    "RENDER-USD",
    period="90d",
    interval="1h",
    auto_adjust=True,
    progress=False
)

# FIX MULTIINDEX

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# KEEP ONLY NEEDED COLUMNS

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]].copy()

# FORCE 1D SERIES

df["Open"] = df["Open"].astype(float)
df["High"] = df["High"].astype(float)
df["Low"] = df["Low"].astype(float)
df["Close"] = df["Close"].astype(float)
df["Volume"] = df["Volume"].astype(float)

df.dropna(inplace=True)

# ---------------- INDICATORS ----------------

df["EMA20"] = ta.trend.ema_indicator(
    close=df["Close"],
    window=20
)

df["EMA50"] = ta.trend.ema_indicator(
    close=df["Close"],
    window=50
)

df["RSI"] = ta.momentum.rsi(
    close=df["Close"],
    window=14
)

macd = ta.trend.MACD(close=df["Close"])

df["MACD"] = macd.macd()
df["MACD_SIGNAL"] = macd.macd_signal()

df["ATR"] = ta.volatility.average_true_range(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["Volatility"] = (
    df["Close"]
    .pct_change()
    .rolling(24)
    .std()
    * 100
)

# ---------------- TARGETS ----------------

df["Target"] = df["Close"].shift(-1)

df["TargetClass"] = np.where(
    df["Close"].shift(-1) > df["Close"],
    1,
    0
)

df.dropna(inplace=True)

# ---------------- FEATURES ----------------

features = [
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "ATR",
    "Volatility"
]

X = df[features]

y_class = df["TargetClass"]
y_reg = df["Target"]

# ---------------- TRAIN TEST ----------------

X_train, X_test, y_train_class, y_test_class = train_test_split(
    X,
    y_class,
    test_size=0.2,
    shuffle=False
)

_, _, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    shuffle=False
)

# ---------------- MODELS ----------------

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train_class)

reg.fit(X_train, y_train_reg)

# ---------------- PREDICTIONS ----------------

latest_features = X.iloc[[-1]]

direction_pred = clf.predict(latest_features)[0]

direction_prob = clf.predict_proba(latest_features)[0]

predicted_price = reg.predict(latest_features)[0]

accuracy = accuracy_score(
    y_test_class,
    clf.predict(X_test)
)

# ---------------- SIGNAL ----------------

if direction_pred == 1:
    signal = "UP"
    signal_color = "green"
    confidence = direction_prob[1] * 100
else:
    signal = "DOWN"
    signal_color = "red"
    confidence = direction_prob[0] * 100

# ---------------- LIVE PRICE ----------------

current_price = float(df["Close"].iloc[-1])

# ---------------- CARDS ----------------

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

# ---------------- PRICE + EMA ----------------

st.subheader("EMA 20 vs EMA 50")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
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

fig.update_layout(
    template="plotly_dark",
    height=600,
    xaxis_rangeslider_visible=False,
    paper_bgcolor="#161b22",
    plot_bgcolor="#161b22"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- RSI ----------------

st.subheader("RSI")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    mode="lines",
    line=dict(color="blue", width=3)
))

rsi_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(rsi_fig, use_container_width=True)

# ---------------- MACD ----------------

st.subheader("MACD")

macd_fig = go.Figure()

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD"],
    mode="lines",
    line=dict(color="pink", width=3)
))

macd_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MACD_SIGNAL"],
    mode="lines",
    line=dict(color="white", width=2)
))

macd_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(macd_fig, use_container_width=True)

# ---------------- ATR ----------------

st.subheader("ATR")

atr_fig = go.Figure()

atr_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["ATR"],
    mode="lines",
    line=dict(color="yellow", width=3)
))

atr_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(atr_fig, use_container_width=True)

# ---------------- VOLATILITY ----------------

st.subheader("Volatility")

vol_fig = go.Figure()

vol_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Volatility"],
    mode="lines",
    line=dict(color="green", width=3)
))

vol_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(vol_fig, use_container_width=True)

# ---------------- ACCURACY ----------------

st.subheader("AI Model Accuracy")

st.write(f"Classification Accuracy: {accuracy:.2f}")

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption("RNDR AI Dashboard • RandomForest ML + Streamlit")
