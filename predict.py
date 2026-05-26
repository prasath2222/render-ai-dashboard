# =========================================================
# FINAL BEST predict.py
# LIVE RNDR AI PREDICTION
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
# DOWNLOAD RNDR
# =========================================================

df = yf.download(

    "RENDER-USD",

    period="90d",

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

    period="90d",

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
# CLEAN
# =========================================================

df.replace(

    [np.inf, -np.inf],

    np.nan,

    inplace=True

)

df.dropna(inplace=True)

# =========================================================
# LATEST FEATURES
# =========================================================

latest = df.iloc[-1]

X_live = pd.DataFrame(
    [[latest[f] for f in features]],
    columns=features
)

# =========================================================
# PREDICTIONS
# =========================================================

prediction = classifier.predict(
    X_live
)[0]

probabilities = classifier.predict_proba(
    X_live
)[0]

confidence = round(
    np.max(probabilities) * 100,
    2
)

future_price = regressor.predict(
    X_live
)[0]

current_price = latest["Close"]

# =========================================================
# LABELS
# =========================================================

labels = {

    4: "STRONG BUY 🚀",

    3: "BUY 📈",

    2: "SIDEWAYS ➖",

    1: "SELL 📉",

    0: "STRONG SELL 🔥"

}

signal = labels[prediction]

# =========================================================
# EXTRA TREND FILTER
# =========================================================

if (

    latest["EMA20"]
    >
    latest["EMA50"]

    and

    latest["EMA50"]
    >
    latest["EMA200"]

    and

    latest["RSI"] > 58

    and

    latest["Breakout"] == 1

):

    signal = "BULLISH BREAKOUT 🚀"

# =========================================================
# BEARISH FILTER
# =========================================================

elif (

    latest["EMA20"]
    <
    latest["EMA50"]

    and

    latest["EMA50"]
    <
    latest["EMA200"]

    and

    latest["RSI"] < 40

):

    signal = "BEARISH TREND 🔥"

# =========================================================
# TARGET RANGE
# =========================================================

upper_target = current_price * 1.06

lower_target = current_price * 0.97

# =========================================================
# OUTPUT
# =========================================================

print("\n===================================")
print("LIVE RNDR AI PREDICTION")
print("===================================")

print(f"\nCurrent Price : ${current_price:.4f}")

print(f"\nPredicted Price : ${future_price:.4f}")

print(f"\nAI Signal : {signal}")

print(f"\nConfidence : {confidence}%")

print(f"\nExpected Range : ${lower_target:.4f} → ${upper_target:.4f}")

print("===================================")
