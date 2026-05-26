import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ================= PAGE =================

st.set_page_config(
    page_title="RNDR AI Dashboard",
    layout="wide"
)

# ================= CSS =================

st.markdown("""
<style>

html, body, [class*="css"] {
    background-color: #0d1117;
    color: white;
}

.metric {
    background: #161b22;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}

.big {
    font-size: 32px;
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

# ================= TITLE =================

st.title("🚀 RNDR AI Dashboard")

# ================= DOWNLOAD DATA =================

df = yf.download(
    tickers="RENDER-USD",
    period="60d",
    interval="1h",
    progress=False
)

# ================= FIX COLUMNS =================

df = df.copy()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ================= KEEP NEEDED =================

df = df[["Open", "High", "Low", "Close", "Volume"]]

# ================= FORCE SERIES =================

close = df["Close"].squeeze()
high = df["High"].squeeze()
low = df["Low"].squeeze()

# ================= INDICATORS MANUAL =================

df["EMA20"] = close.ewm(span=20).mean()

df["EMA50"] = close.ewm(span=50).mean()

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

df["Volatility"] = close.pct_change().rolling(24).std() * 100

# ================= TARGETS =================

df["Target"] = close.shift(-1)

df["TargetClass"] = np.where(
    close.shift(-1) > close,
    1,
    0
)

df.dropna(inplace=True)

# ================= FEATURES =================

features = [
    "EMA20",
    "EMA50",
    "RSI",
    "Volatility"
]

X = df[features]

y_class = df["TargetClass"]

y_reg = df["Target"]

# ================= SPLIT =================

X_train, X_test, y_train_c, y_test_c = train_test_split(
    X,
    y_class,
    test_size=0.2,
    shuffle=False
)

_, _, y_train_r, y_test_r = train_test_split(
    X,
    y_reg,
    test_size=0.2,
    shuffle=False
)

# ================= MODELS =================

clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

reg = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

clf.fit(X_train, y_train_c)

reg.fit(X_train, y_train_r)

# ================= PREDICTIONS =================

latest = X.iloc[[-1]]

direction = clf.predict(latest)[0]

prob = clf.predict_proba(latest)[0]

future_price = reg.predict(latest)[0]

accuracy = accuracy_score(
    y_test_c,
    clf.predict(X_test)
)

# ================= SIGNAL =================

if direction == 1:
    signal = "UP"
    signal_color = "green"
    confidence = prob[1] * 100
else:
    signal = "DOWN"
    signal_color = "red"
    confidence = prob[0] * 100

# ================= LIVE PRICE =================

current_price = float(close.iloc[-1])

# ================= METRICS =================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric">
    <h3>RNDR Price</h3>
    <div class="big">${current_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric">
    <h3>Direction</h3>
    <div class="big {signal_color}">{signal}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric">
    <h3>Predicted Price</h3>
    <div class="big">${future_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric">
    <h3>Confidence</h3>
    <div class="big">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ================= PRICE CHART =================

st.subheader("RNDR Price + EMA")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=close,
    name="Price"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA20"],
    name="EMA20"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["EMA50"],
    name="EMA50"
))

fig.update_layout(
    template="plotly_dark",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# ================= RSI =================

st.subheader("RSI")

rsi_fig = go.Figure()

rsi_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    line=dict(color="blue")
))

rsi_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(rsi_fig, use_container_width=True)

# ================= VOLATILITY =================

st.subheader("Volatility")

vol_fig = go.Figure()

vol_fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Volatility"],
    line=dict(color="green")
))

vol_fig.update_layout(
    template="plotly_dark",
    height=300
)

st.plotly_chart(vol_fig, use_container_width=True)

# ================= ACCURACY =================

st.subheader("Model Accuracy")

st.write(f"Classification Accuracy: {accuracy:.2f}")
