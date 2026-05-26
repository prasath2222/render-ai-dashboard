# ==========================================
# ADVANCED RNDR AI TRAINING SYSTEM
# ==========================================

import pandas as pd
import numpy as np
import yfinance as yf
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

# ==========================================
# DOWNLOAD DATA
# ==========================================

df = yf.download(
    "RENDER-USD",
    period="180d",
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
# FUTURE RETURN
# ==========================================

future_period = 4

df["FutureReturn"] = (
    close.shift(-future_period) - close
) / close

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
# TARGET CLASSIFICATION
# ==========================================

df["TargetDirection"] = np.select(
    [
        df["FutureReturn"] > 0.04,
        df["FutureReturn"] > 0.015,
        df["FutureReturn"] < -0.04,
        df["FutureReturn"] < -0.015
    ],
    [
        2,
        1,
        -2,
        -1
    ],
    default=0
)

# ==========================================
# TARGET REGRESSION
# ==========================================

df["TargetPrice"] = close.shift(-future_period)

# ==========================================
# CLEAN
# ==========================================

df.dropna(inplace=True)

# ==========================================
# FEATURES
# ==========================================

features = [

    "EMA20",
    "EMA50",
    "EMA200",

    "RSI",
    "MACD",
    "MACD_SIGNAL",

    "ATR",
    "Volatility",

    "BreakoutStrength",

    "Momentum5",
    "Momentum10",

    "VolumeSpike",

    "BullTrend",

    "BodyStrength",

    "BB_UPPER",
    "BB_LOWER"
]

# ==========================================
# DATA
# ==========================================

X = df[features]

y_class = df["TargetDirection"]

y_reg = df["TargetPrice"]

# ==========================================
# MODELS
# ==========================================

rf_classifier = RandomForestClassifier(
    n_estimators=80,
    max_depth=8,
    random_state=42
)

gb_classifier = GradientBoostingClassifier(
    n_estimators=120,
    learning_rate=0.05,
    random_state=42
)

rf_regressor = RandomForestRegressor(
    n_estimators=80,
    max_depth=8,
    random_state=42
)

gb_regressor = GradientBoostingRegressor(
    n_estimators=120,
    learning_rate=0.05,
    random_state=42
)

# ==========================================
# TRAIN
# ==========================================

rf_classifier.fit(X, y_class)
gb_classifier.fit(X, y_class)

rf_regressor.fit(X, y_reg)
gb_regressor.fit(X, y_reg)

# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(rf_classifier, "rf_classifier.pkl")
joblib.dump(gb_classifier, "gb_classifier.pkl")

joblib.dump(rf_regressor, "rf_regressor.pkl")
joblib.dump(gb_regressor, "gb_regressor.pkl")

joblib.dump(features, "features.pkl")

print("ADVANCED AI MODELS TRAINED")
