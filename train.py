# =========================================================
# INSTALL LIBRARIES
# =========================================================

!pip install yfinance ta scikit-learn -q

# =========================================================
# IMPORTS
# =========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

# =========================================================
# DOWNLOAD RNDR DATA
# =========================================================

df = yf.download(

    "RENDER-USD",

    period="730d",

    interval="1h",

    auto_adjust=True

)

# =========================================================
# FIX MULTI INDEX
# =========================================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

btc = yf.download(

    "BTC-USD",

    period="730d",

    interval="1h",

    auto_adjust=True

)

# =========================================================
# FIX BTC COLUMNS
# =========================================================

if isinstance(btc.columns, pd.MultiIndex):

    btc.columns = btc.columns.get_level_values(0)

# =========================================================
# BTC FEATURES
# =========================================================

df["BTC_Close"] = btc["Close"]

df["BTC_Return"] = (

    df["BTC_Close"]

    .pct_change()

)

# =========================================================
# RSI
# =========================================================

df["RSI"] = ta.momentum.RSIIndicator(

    close=df["Close"],

    window=14

).rsi()

# =========================================================
# MACD
# =========================================================

macd = ta.trend.MACD(

    close=df["Close"]

)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_DIFF"] = macd.macd_diff()

# =========================================================
# EMA
# =========================================================

df["EMA20"] = ta.trend.EMAIndicator(

    close=df["Close"],

    window=20

).ema_indicator()

df["EMA50"] = ta.trend.EMAIndicator(

    close=df["Close"],

    window=50

).ema_indicator()

df["EMA200"] = ta.trend.EMAIndicator(

    close=df["Close"],

    window=200

).ema_indicator()

# =========================================================
# SMA
# =========================================================

df["SMA20"] = ta.trend.SMAIndicator(

    close=df["Close"],

    window=20

).sma_indicator()

# =========================================================
# ATR
# =========================================================

df["ATR"] = ta.volatility.AverageTrueRange(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14

).average_true_range()

# =========================================================
# BOLLINGER BANDS
# =========================================================

bb = ta.volatility.BollingerBands(

    close=df["Close"],

    window=20

)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()

# =========================================================
# STOCHASTIC
# =========================================================

stoch = ta.momentum.StochasticOscillator(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14,

    smooth_window=3

)

df["STOCH"] = stoch.stoch()

df["STOCH_SIGNAL"] = stoch.stoch_signal()

# =========================================================
# RETURNS
# =========================================================

df["Returns"] = (

    df["Close"]

    .pct_change()

)

# =========================================================
# VOLATILITY
# =========================================================

df["Volatility"] = (

    df["Returns"]

    .rolling(24)

    .std()

)

# =========================================================
# MOMENTUM
# =========================================================

df["Momentum5"] = (

    df["Close"]

    - df["Close"].shift(5)

)

df["Momentum10"] = (

    df["Close"]

    - df["Close"].shift(10)

)

# =========================================================
# BREAKOUT
# =========================================================

df["Breakout"] = (

    df["Close"]

    >

    df["High"]

    .rolling(20)

    .max()

    .shift(1)

).astype(int)

# =========================================================
# VOLUME
# =========================================================

df["Volume_MA"] = (

    df["Volume"]

    .rolling(20)

    .mean()

)

df["Volume_Spike"] = (

    df["Volume"]

    >

    (df["Volume_MA"] * 1.5)

).astype(int)

# =========================================================
# TREND
# =========================================================

df["Bull_Trend"] = np.where(

    (

        (df["EMA20"] > df["EMA50"])

        &

        (df["EMA50"] > df["EMA200"])

    ),

    1,

    0

)

# =========================================================
# FUTURE RETURN
# =========================================================

future_return = (

    df["Close"]

    .shift(-12)

    / df["Close"]

    - 1

)

# =========================================================
# CLASS TARGET
# =========================================================

conditions = [

    future_return > 0.04,

    future_return < -0.04

]

choices = [

    2,

    0

]

df["Target"] = np.select(

    conditions,

    choices,

    default=1

)

# =========================================================
# REGRESSION TARGET
# =========================================================

df["Future_Price"] = (

    df["Close"]

    .shift(-12)

)

# =========================================================
# CLEAN DATA
# =========================================================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True

)

df.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

features = [

    "Close",

    "Volume",

    "BTC_Return",

    "RSI",

    "MACD",

    "MACD_SIGNAL",

    "MACD_DIFF",

    "EMA20",

    "EMA50",

    "EMA200",

    "SMA20",

    "ATR",

    "BB_HIGH",

    "BB_LOW",

    "BB_WIDTH",

    "STOCH",

    "STOCH_SIGNAL",

    "Returns",

    "Volatility",

    "Momentum5",

    "Momentum10",

    "Breakout",

    "Volume_Spike",

    "Bull_Trend"

]

# =========================================================
# X AND Y
# =========================================================

X = df[features]

y_class = df["Target"]

y_reg = df["Future_Price"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(len(df) * 0.8)

X_train = X[:split]

X_test = X[split:]

y_class_train = y_class[:split]

y_class_test = y_class[split:]

y_reg_train = y_reg[:split]

y_reg_test = y_reg[split:]

# =========================================================
# CLASSIFIER MODEL
# =========================================================

classifier = RandomForestClassifier(

    n_estimators=500,

    max_depth=12,

    min_samples_split=5,

    min_samples_leaf=2,

    random_state=42,

    n_jobs=-1

)

# =========================================================
# REGRESSOR MODEL
# =========================================================

regressor = RandomForestRegressor(

    n_estimators=300,

    max_depth=10,

    random_state=42,

    n_jobs=-1

)

# =========================================================
# TRAIN CLASSIFIER
# =========================================================

classifier.fit(

    X_train,

    y_class_train

)

# =========================================================
# TRAIN REGRESSOR
# =========================================================

regressor.fit(

    X_train,

    y_reg_train

)

# =========================================================
# CLASSIFICATION PREDICTION
# =========================================================

class_pred = classifier.predict(

    X_test

)

# =========================================================
# REGRESSION PREDICTION
# =========================================================

price_pred = regressor.predict(

    X_test

)

# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(

    y_class_test,

    class_pred

)

print("\n====================================")

print("FINAL RNDR AI MODEL TRAINED")

print("====================================")

print(f"\nClassification Accuracy : {accuracy * 100:.2f}%")

print("\n====================================")

# =========================================================
# SAVE FILES
# =========================================================

joblib.dump(

    classifier,

    "rf_classifier.pkl"

)

joblib.dump(

    regressor,

    "rf_regressor.pkl"

)

joblib.dump(

    features,

    "features.pkl"

)

print("\nFILES SAVED:")

print("rf_classifier.pkl")

print("rf_regressor.pkl")

print("features.pkl")

print("\n====================================")
