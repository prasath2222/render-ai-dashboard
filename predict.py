import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import ta

from telegram_alert import send_alert

# =========================================================
# DOWNLOAD RENDER DATA
# =========================================================

df = yf.download(
    "RENDER-USD",
    interval="1h",
    period="90d",
    auto_adjust=True
)

df = df.dropna()

# =========================================================
# INDICATORS
# =========================================================

df["EMA20"] = ta.trend.ema_indicator(
    close=df["Close"],
    window=20
)

df["EMA50"] = ta.trend.ema_indicator(
    close=df["Close"],
    window=50
)

df["EMA200"] = ta.trend.ema_indicator(
    close=df["Close"],
    window=200
)

df["RSI"] = ta.momentum.rsi(
    close=df["Close"],
    window=14
)

macd = ta.trend.MACD(
    close=df["Close"]
)

df["MACD"] = macd.macd()

df["MACD_SIGNAL"] = macd.macd_signal()

df["MACD_HIST"] = macd.macd_diff()

df["ADX"] = ta.trend.adx(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["ATR"] = ta.volatility.average_true_range(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

bb = ta.volatility.BollingerBands(
    close=df["Close"],
    window=20,
    window_dev=2
)

df["BB_UPPER"] = bb.bollinger_hband()

df["BB_LOWER"] = bb.bollinger_lband()

df["BB_MIDDLE"] = bb.bollinger_mavg()

df["VOLUME_SMA"] = ta.trend.sma_indicator(
    close=df["Volume"],
    window=20
)

# =========================================================
# CLEAN
# =========================================================

df = df.dropna()

latest = df.iloc[-1]

close = df["Close"]

# =========================================================
# CURRENT VALUES
# =========================================================

current_price = float(latest["Close"])

ema20 = float(latest["EMA20"])
ema50 = float(latest["EMA50"])
ema200 = float(latest["EMA200"])

rsi = float(latest["RSI"])

macd_value = float(latest["MACD"])
macd_signal = float(latest["MACD_SIGNAL"])
macd_hist = float(latest["MACD_HIST"])

adx = float(latest["ADX"])

atr = float(latest["ATR"])

bb_upper = float(latest["BB_UPPER"])
bb_lower = float(latest["BB_LOWER"])

volume = float(latest["Volume"])
volume_sma = float(latest["VOLUME_SMA"])

# =========================================================
# TREND SCORE
# =========================================================

trend_score = 0

# EMA TREND

if ema20 > ema50:
    trend_score += 20
else:
    trend_score -= 20

if ema50 > ema200:
    trend_score += 20
else:
    trend_score -= 20

# RSI

if rsi >= 70:
    trend_score -= 15

elif rsi >= 60:
    trend_score += 15

elif rsi <= 30:
    trend_score += 20

elif rsi <= 40:
    trend_score -= 10

# MACD

if macd_value > macd_signal:
    trend_score += 20
else:
    trend_score -= 20

# ADX

if adx > 25:
    trend_score += 10

# MOMENTUM

price_change_10 = (
    (close.iloc[-1] - close.iloc[-10])
    / close.iloc[-10]
) * 100

if price_change_10 > 0:
    trend_score += 10
else:
    trend_score -= 10

# VOLUME

if volume > volume_sma:
    trend_score += 5

# =========================================================
# SIGNAL
# =========================================================

if trend_score >= 40:
    signal = "BUY"
    signal_color = "#00ff88"

elif trend_score <= -40:
    signal = "SELL"
    signal_color = "#ff4d6d"

else:
    signal = "HOLD"
    signal_color = "#ffaa00"

# =========================================================
# CONFIDENCE
# =========================================================

confidence = min(
    95,
    max(
        50,
        abs(trend_score)
    )
)

# =========================================================
# PREDICTION ENGINE
# =========================================================

volatility = atr / current_price

prediction_strength = trend_score / 100

prediction_move = (
    prediction_strength
    * volatility
    * 5
)

predicted_price = current_price * (
    1 + prediction_move
)

predicted_change = (
    (predicted_price - current_price)
    / current_price
) * 100

# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

support_1 = float(
    df["Low"].tail(50).min()
)

resistance_1 = float(
    df["High"].tail(50).max()
)

support_2 = support_1 - atr

resistance_2 = resistance_1 + atr

# =========================================================
# MARKET REGIME
# =========================================================

if ema20 > ema50 and rsi > 55:
    market_regime = "BULLISH"

elif ema20 < ema50 and rsi < 45:
    market_regime = "BEARISH"

else:
    market_regime = "SIDEWAYS"

# =========================================================
# TRADING SETUP
# =========================================================

entry_price = current_price

stop_loss = current_price - (
    atr * 2
)

take_profit_1 = current_price + (
    atr * 3
)

take_profit_2 = current_price + (
    atr * 6
)

risk_reward = (
    (take_profit_1 - entry_price)
    /
    (entry_price - stop_loss)
)

# =========================================================
# BUY / SELL PRESSURE
# =========================================================

buy_pressure = max(
    0,
    min(
        100,
        50 + trend_score
    )
)

sell_pressure = 100 - buy_pressure

# =========================================================
# MULTI TIMEFRAME SIGNALS
# =========================================================

if signal == "BUY":

    signal_15m = "BUY"
    signal_1h = "BUY"
    signal_4h = "BUY"
    signal_1d = "BUY"

elif signal == "SELL":

    signal_15m = "SELL"
    signal_1h = "SELL"
    signal_4h = "SELL"
    signal_1d = "SELL"

else:

    signal_15m = "HOLD"
    signal_1h = "HOLD"
    signal_4h = "HOLD"
    signal_1d = "HOLD"

# =========================================================
# OUTPUT
# =========================================================

print("\n=================================================")
print("RNDR AI PREDICTION")
print("=================================================\n")

print("CURRENT PRICE :", round(current_price, 4))
print("PREDICTED PRICE :", round(predicted_price, 4))
print("PREDICTED CHANGE :", round(predicted_change, 2), "%")

print("\nSIGNAL :", signal)
print("CONFIDENCE :", confidence, "%")

print("\nMARKET REGIME :", market_regime)

print("\nRSI :", round(rsi, 2))
print("MACD :", round(macd_value, 4))
print("ADX :", round(adx, 2))
print("ATR :", round(atr, 4))

print("\nSUPPORT 1 :", round(support_1, 4))
print("SUPPORT 2 :", round(support_2, 4))

print("\nRESISTANCE 1 :", round(resistance_1, 4))
print("RESISTANCE 2 :", round(resistance_2, 4))

print("\nENTRY :", round(entry_price, 4))
print("STOP LOSS :", round(stop_loss, 4))

print("TAKE PROFIT 1 :", round(take_profit_1, 4))
print("TAKE PROFIT 2 :", round(take_profit_2, 4))

print("\nRISK / REWARD :", round(risk_reward, 2))

print("\nBUY PRESSURE :", round(buy_pressure, 2), "%")
print("SELL PRESSURE :", round(sell_pressure, 2), "%")

print("\nMULTI TIMEFRAME")

print("15M :", signal_15m)
print("1H :", signal_1h)
print("4H :", signal_4h)
print("1D :", signal_1d)

print("\n=================================================\n")

# =========================================================
# TELEGRAM ALERT
# =========================================================

message = f"""
🚀 RNDR AI SIGNAL

💰 Price: ${current_price:.4f}

📈 Signal: {signal}

🎯 Confidence: {confidence}%

🔮 Predicted: ${predicted_price:.4f}

📊 Change: {predicted_change:.2f}%

📌 RSI: {rsi:.2f}

📌 ADX: {adx:.2f}

📌 Market: {market_regime}

🟢 TP1: ${take_profit_1:.4f}

🟢 TP2: ${take_profit_2:.4f}

🔴 SL: ${stop_loss:.4f}
"""

send_alert(message)
