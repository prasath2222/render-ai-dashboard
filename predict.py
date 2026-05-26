# ==========================================
# ADVANCED RNDR LIVE PREDICTION ENGINE
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# ==========================================
# LOAD MODELS
# ==========================================

rf_classifier = joblib.load("rf_classifier.pkl")
gb_classifier = joblib.load("gb_classifier.pkl")

rf_regressor = joblib.load("rf_regressor.pkl")
gb_regressor = joblib.load("gb_regressor.pkl")

features = joblib.load("features.pkl")

# ==========================================
# DOWNLOAD LIVE DATA
# ==========================================

df = yf.download(
    "RENDER-USD",
    period="30d",
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

# ==========================================
# SERIES
# ==========================================

close = df["Close"].squeeze()
high = df["High"].squeeze()
low = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# ==========================================
# EMA
# ==========================================

df["EMA20"] = close.ewm(span=20).mean()
df["EMA50"] = close.ewm(span=50).mean()
df["EMA200"] = close.ewm(span=200).mean()

# ==========================================
# RSI
# ==========================================

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# ==========================================
# MACD
# ==========================================

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

# ==========================================
# ATR
# ==========================================

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = tr.rolling(14).mean()

# ==========================================
# VOLATILITY
# ==========================================

df["Volatility"] = close.pct_change().rolling(24).std() * 100

# ==========================================
# BOLLINGER
# ==========================================

df["BB_MIDDLE"] = close.rolling(20).mean()

std = close.rolling(20).std()

df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)

# ==========================================
# TREND REGIME
# ==========================================

df["BullTrend"] = np.where(
    (
        (df["EMA20"] > df["EMA50"]) &
        (df["EMA50"] > df["EMA200"])
    ),
    1,
    0
)

# ==========================================
# BREAKOUT STRENGTH
# ==========================================

df["RecentHigh"] = high.rolling(20).max()

df["BreakoutStrength"] = (
    close - df["RecentHigh"].shift(1)
) / close

# ==========================================
# MOMENTUM
# ==========================================

df["Momentum5"] = (
    close - close.shift(5)
) / close

df["Momentum10"] = (
    close - close.shift(10)
) / close

# ==========================================
# VOLUME SPIKE
# ==========================================

df["VolumeMA"] = volume.rolling(20).mean()

df["VolumeSpike"] = (
    volume / df["VolumeMA"]
)

# ==========================================
# BODY STRENGTH
# ==========================================

df["BodyStrength"] = (
    abs(df["Close"] - df["Open"])
) / (
    df["High"] - df["Low"] + 0.0001
)

# ==========================================
# CLEAN
# ==========================================

df.dropna(inplace=True)

# ==========================================
# LATEST DATA
# ==========================================

latest = df[features].iloc[-1:]

# ==========================================
# CLASSIFICATION
# ==========================================

rf_class = rf_classifier.predict(latest)[0]
gb_class = gb_classifier.predict(latest)[0]

direction_votes = [
    rf_class,
    gb_class
]

final_direction = int(round(np.mean(direction_votes)))

# ==========================================
# SIGNAL MAP
# ==========================================

signal_map = {
    2: "STRONG BUY",
    1: "BUY",
    0: "SIDEWAYS",
    -1: "SELL",
    -2: "STRONG SELL"
}

signal = signal_map[final_direction]

# ==========================================
# REGIME FILTER
# ==========================================

bull_regime = (
    df["EMA20"].iloc[-1] >
    df["EMA50"].iloc[-1] >
    df["EMA200"].iloc[-1]
)

if bull_regime and final_direction < 0:
    signal = "SIDEWAYS"

# ==========================================
# MOMENTUM OVERRIDE
# ==========================================

recent_momentum = df["Momentum5"].iloc[-1]

if recent_momentum > 0.03:
    signal = "STRONG BUY"

if recent_momentum < -0.03:
    signal = "STRONG SELL"

# ==========================================
# BREAKOUT OVERRIDE
# ==========================================

breakout = df["BreakoutStrength"].iloc[-1]

if breakout > 0.02:
    signal = "BREAKOUT BUY"

# ==========================================
# REGRESSION
# ==========================================

rf_price = rf_regressor.predict(latest)[0]
gb_price = gb_regressor.predict(latest)[0]

predicted_price = (
    rf_price + gb_price
) / 2

# ==========================================
# CURRENT PRICE
# ==========================================

current_price = float(close.iloc[-1])

# ==========================================
# PRICE RANGE
# ==========================================

atr = float(df["ATR"].iloc[-1])

pred_low = current_price - atr
pred_high = current_price + atr

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
# OUTPUT
# ==========================================

result = {
    "current_price": round(current_price, 4),
    "signal": signal,
    "predicted_price": round(predicted_price, 4),
    "pred_low": round(pred_low, 4),
    "pred_high": round(pred_high, 4),
    "confidence": confidence
}

print(result)
