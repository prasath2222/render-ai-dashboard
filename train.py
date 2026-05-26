# =========================================================
# FINAL BEST TRAIN.PY
# STABLE + CLEAN + COLAB WORKING
# =========================================================

# =========================================================
# INSTALL
# =========================================================

!pip install yfinance ta xgboost joblib -q

# =========================================================
# IMPORTS
# =========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

from xgboost import XGBClassifier
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score

# =========================================================
# DOWNLOAD RNDR
# =========================================================

df = yf.download(

    "RENDER-USD",

    period="730d",

    interval="1h",

    auto_adjust=True

)

# =========================================================
# FIX COLUMNS
# =========================================================

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

# =========================================================
# DOWNLOAD BTC
# =========================================================

btc = yf.download(

    "BTC-USD",

    period="730d",

    interval="1h",

    auto_adjust=True

)

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
# ATR
# =========================================================

df["ATR"] = ta.volatility.AverageTrueRange(

    high=df["High"],

    low=df["Low"],

    close=df["Close"],

    window=14

).average_true_range()

# =========================================================
# BOLLINGER
# =========================================================

bb = ta.volatility.BollingerBands(

    close=df["Close"],

    window=20

)

df["BB_HIGH"] = bb.bollinger_hband()

df["BB_LOW"] = bb.bollinger_lband()

df["BB_WIDTH"] = bb.bollinger_wband()

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
# MARKET STRUCTURE
# =========================================================

df["Higher_High"] = (

    df["High"]

    > df["High"].shift(1)

).astype(int)

df["Higher_Low"] = (

    df["Low"]

    > df["Low"].shift(1)

).astype(int)

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
# VOLUME SPIKE
# =========================================================

df["Volume_MA"] = (

    df["Volume"]

    .rolling(20)

    .mean()

)

df["Volume_Spike"] = (

    df["Volume"]

    >

    (df["Volume_MA"] * 2)

).astype(int)

# =========================================================
# CANDLE FEATURES
# =========================================================

df["Body"] = abs(

    df["Close"] - df["Open"]

)

df["Upper_Wick"] = (

    df["High"]

    -

    df[["Close", "Open"]]

    .max(axis=1)

)

df["Lower_Wick"] = (

    df[["Close", "Open"]]

    .min(axis=1)

    -

    df["Low"]

)

# =========================================================
# TREND REGIME
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
# FUTURE TARGET
# =========================================================

future_period = 12

df["Future_Close"] = (

    df["Close"]

    .shift(-future_period)

)

future_return = (

    (df["Future_Close"] - df["Close"])

    / df["Close"]

)

# =========================================================
# MULTI CLASS TARGET
# =========================================================

conditions = [

    future_return > 0.03,

    (

        (future_return > 0.01)

        &

        (future_return <= 0.03)

    ),

    (

        (future_return >= -0.01)

        &

        (future_return <= 0.01)

    ),

    (

        (future_return < -0.01)

        &

        (future_return >= -0.03)

    ),

    future_return < -0.03

]

choices = [

    4,

    3,

    2,

    1,

    0

]

df["Target"] = np.select(

    conditions,

    choices,

    default=2

)

# =========================================================
# CLEAN
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

    "BTC_Close",

    "BTC_Return",

    "RSI",

    "MACD",

    "MACD_SIGNAL",

    "MACD_DIFF",

    "EMA20",

    "EMA50",

    "EMA200",

    "ATR",

    "BB_HIGH",

    "BB_LOW",

    "BB_WIDTH",

    "Returns",

    "Volatility",

    "Momentum5",

    "Momentum10",

    "Higher_High",

    "Higher_Low",

    "Breakout",

    "Volume_Spike",

    "Body",

    "Upper_Wick",

    "Lower_Wick",

    "Bull_Trend"

]

X = df[features]

y_class = df["Target"]

y_reg = df["Future_Close"]

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y_class[:split]
y_test = y_class[split:]

y_reg_train = y_reg[:split]
y_reg_test = y_reg[split:]

# =========================================================
# CLASSIFIER
# =========================================================

classifier = XGBClassifier(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.9,

    colsample_bytree=0.9,

    gamma=0.2,

    min_child_weight=3,

    random_state=42,

    eval_metric="mlogloss"

)

# =========================================================
# REGRESSOR
# =========================================================

regressor = XGBRegressor(

    n_estimators=300,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.9,

    colsample_bytree=0.9,

    random_state=42

)

# =========================================================
# TRAIN
# =========================================================

classifier.fit(

    X_train,

    y_train

)

regressor.fit(

    X_train,

    y_reg_train

)

# =========================================================
# ACCURACY
# =========================================================

predictions = classifier.predict(

    X_test

)

accuracy = accuracy_score(

    y_test,

    predictions

)

print("\n===================================")
print("CLASSIFICATION ACCURACY:")
print(round(accuracy, 4))
print("===================================")

# =========================================================
# SAVE MODELS
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

print("\n===================================")
print("ADVANCED AI MODELS TRAINED")
print("===================================")

print("\nFILES SAVED:")

print("rf_classifier.pkl")
print("rf_regressor.pkl")
print("features.pkl")

print("===================================")
