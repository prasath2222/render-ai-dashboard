# =========================
# train.py
# ADVANCED RNDR AI SYSTEM
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================
# DOWNLOAD DATA
# =========================

df = yf.download(
    "RENDER-USD",
    period="180d",
    interval="1h",
    progress=False
)

# =========================
# FIX MULTIINDEX
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
# TECHNICAL INDICATORS
# =========================

# EMA

df["EMA20"] = close.ewm(span=20).mean()
df["EMA50"] = close.ewm(span=50).mean()
df["EMA200"] = close.ewm(span=200).mean()

# RSI

delta = close.diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss

df["RSI"] = 100 - (100 / (1 + rs))

# MACD

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()

df["MACD"] = ema12 - ema26
df["MACD_SIGNAL"] = df["MACD"].ewm(span=9).mean()

# ATR

tr1 = high - low
tr2 = abs(high - close.shift())
tr3 = abs(low - close.shift())

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["ATR"] = tr.rolling(14).mean()

# Bollinger Bands

df["BB_MIDDLE"] = close.rolling(20).mean()

std = close.rolling(20).std()

df["BB_UPPER"] = df["BB_MIDDLE"] + (std * 2)
df["BB_LOWER"] = df["BB_MIDDLE"] - (std * 2)

# Volatility

df["Volatility"] = close.pct_change().rolling(24).std() * 100

# Momentum

df["Momentum"] = close - close.shift(10)

# Volume Change

df["VolumeChange"] = volume.pct_change()

# =========================
# TARGETS
# =========================

# Regression Target

df["TargetPrice"] = close.shift(-1)

# Classification Target

df["TargetDirection"] = np.where(
    close.shift(-1) > close,
    1,
    0
)

# =========================
# CLEAN
# =========================

df.dropna(inplace=True)

# =========================
# FEATURES
# =========================

features = [
    "EMA20",
    "EMA50",
    "EMA200",
    "RSI",
    "MACD",
    "MACD_SIGNAL",
    "ATR",
    "BB_UPPER",
    "BB_LOWER",
    "Volatility",
    "Momentum",
    "VolumeChange"
]

X = df[features]

y_class = df["TargetDirection"]
y_reg = df["TargetPrice"]

# =========================
# CLASSIFICATION MODELS
# =========================

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

gb_classifier = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

log_classifier = LogisticRegression(
    max_iter=1000
)

# =========================
# REGRESSION MODELS
# =========================

rf_regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

gb_regressor = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    random_state=42
)

# =========================
# TRAIN
# =========================

rf_classifier.fit(X, y_class)
gb_classifier.fit(X, y_class)
log_classifier.fit(X, y_class)

rf_regressor.fit(X, y_reg)
gb_regressor.fit(X, y_reg)

# =========================
# SAVE MODELS
# =========================

joblib.dump(rf_classifier, "rf_classifier.pkl")
joblib.dump(gb_classifier, "gb_classifier.pkl")
joblib.dump(log_classifier, "log_classifier.pkl")

joblib.dump(rf_regressor, "rf_regressor.pkl")
joblib.dump(gb_regressor, "gb_regressor.pkl")

# =========================
# SAVE FEATURES
# =========================

joblib.dump(features, "features.pkl")

print("ALL MODELS TRAINED SUCCESSFULLY")
