# =========================================================
# PREDICT.PY
# =========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib

# =========================================================
# LOAD MODELS
# =========================================================

classifier = joblib.load(

    "rf_classifier.pkl"

)

regressor = joblib.load(

    "rf_regressor.pkl"

)

features = joblib.load(

    "features.pkl"

)

# =========================================================
# DOWNLOAD RNDR DATA
# =========================================================

df = yf.download(

    "RENDER-USD",

    period="30d",

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

    period="30d",

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
# STOCH
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
# CLEAN
# =========================================================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True

)

df.dropna(inplace=True)

# =========================================================
# LATEST ROW
# =========================================================

latest = df[features].iloc[-1:]

# =========================================================
# CLASSIFICATION PREDICTION
# =========================================================

class_prediction = classifier.predict(

    latest

)[0]

# =========================================================
# REGRESSION PREDICTION
# =========================================================

future_price = regressor.predict(

    latest

)[0]

# =========================================================
# CURRENT PRICE
# =========================================================

current_price = float(

    df["Close"].iloc[-1]

)

# =========================================================
# CONFIDENCE
# =========================================================

confidence = np.max(

    classifier.predict_proba(latest)

) * 100

# =========================================================
# LABELS
# =========================================================

if class_prediction == 2:

    movement = "BULLISH 🚀"

elif class_prediction == 0:

    movement = "BEARISH 🔻"

else:

    movement = "SIDEWAYS ➖"

# =========================================================
# OUTPUT
# =========================================================

print("\n===================================")

print("RNDR AI PREDICTION")

print("===================================")

print(f"\nCurrent Price : ${current_price:.4f}")

print(f"\nPredicted Price : ${future_price:.4f}")

print(f"\nMarket Direction : {movement}")

print(f"\nConfidence : {confidence:.2f}%")

print("\n===================================")
