# =========================
# predict.py
# LIVE RNDR PREDICTIONS
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

# =========================
# LOAD MODELS
# =========================

rf_classifier = joblib.load("rf_classifier.pkl")
gb_classifier = joblib.load("gb_classifier.pkl")

rf_regressor = joblib.load("rf_regressor.pkl")
gb_regressor = joblib.load("gb_regressor.pkl")

features = joblib.load("features.pkl")

# =========================
# DOWNLOAD LIVE DATA
# =========================

df = yf.download(
    "RENDER-USD",
    period="30d",
    interval="1h",
    progress=False
)

# =========================
# FIX COLUMNS
# =========================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[[
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]].copy()

close = df["Close"].squeeze()
high = df["High"].squeeze()
low = df["Low"].squeeze()
volume = df["Volume"].squeeze()

# =========================
# INDICATORS
# =========================

df["EMA20"] = close.ewm(span=20).mean()
df["EMA50"] = close.ewm(span=50).mean()
df["EMA200"] = close.ewm(span=200).mean()

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = tr.rolling(14).mean()

df["BB_MIDDLE"] = close.rolling(20).mean()

std = close.rolling(20).std()

df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)

df["Volatility"] = close.pct_change().rolling(24).std() * 100

df["Momentum"] = close - close.shift(10)

df["VolumeChange"] = volume.pct_change()

df.dropna(inplace=True)

# =========================
# LATEST FEATURES
# =========================

latest = df[features].iloc[[-1]]

# =========================
# CLASSIFICATION
# =========================

rf_direction = rf_classifier.predict(latest)[0]
gb_direction = gb_classifier.predict(latest)[0]

rf_prob = rf_classifier.predict_proba(latest)[0]
gb_prob = gb_classifier.predict_proba(latest)[0]

# =========================
# REGRESSION
# =========================

rf_price = rf_regressor.predict(latest)[0]
gb_price = gb_regressor.predict(latest)[0]

# =========================
# ENSEMBLE LOGIC
# =========================

direction_votes = [
    rf_direction,
    gb_direction
]

final_direction = round(np.mean(direction_votes))

final_price = np.mean([
    rf_price,
    gb_price
])

confidence = np.mean([
    max(rf_prob),
    max(gb_prob)
]) * 100

# =========================
# RESULT
# =========================

prediction = {
    "current_price": float(close.iloc[-1]),
    "predicted_price": float(final_price),
    "direction": "UP" if final_direction == 1 else "DOWN",
    "confidence": float(confidence),
    "rsi": float(df["RSI"].iloc[-1]),
    "volatility": float(df["Volatility"].iloc[-1]),
}

print(prediction)
