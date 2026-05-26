import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import ta

# ==========================================
# LOAD FILES
# ==========================================

xgb_model = joblib.load("xgb_model.pkl")

xgb_reg = joblib.load("xgb_reg.pkl")

scaler = joblib.load("scaler.pkl")

features = joblib.load("features.pkl")

# ==========================================
# DOWNLOAD LATEST DATA
# ==========================================

df = yf.download(
    "BTC-USD",
    interval="1h",
    period="90d",
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

# ==========================================
# SERIES
# ==========================================

close = df["Close"]

high = df["High"]

low = df["Low"]

volume = df["Volume"]

# ==========================================
# FEATURES
# ==========================================

df["rsi14"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()

df["ema20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()

df["ema50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()

macd = ta.trend.MACD(close=close)

df["macd"] = macd.macd()

df["macd_signal"] = macd.macd_signal()

bb = ta.volatility.BollingerBands(
    close=close
)

df["bb_high"] = bb.bollinger_hband()

df["bb_low"] = bb.bollinger_lband()

df["bb_position"] = (
    close - df["bb_low"]
) / (
    df["bb_high"] - df["bb_low"] + 1e-9
)

df["atr"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close
).average_true_range()

df["returns_1h"] = close.pct_change()

df["mom5"] = (
    close / close.shift(5)
) - 1

df["vol_ma20"] = (
    volume.rolling(20).mean()
)

df["vol_ratio"] = (
    volume / df["vol_ma20"]
)

# ==========================================
# EXTRA FEATURES
# ==========================================

df["rsi7"] = ta.momentum.RSIIndicator(
    close=close,
    window=7
).rsi()

df["ema_cross"] = (
    df["ema20"] - df["ema50"]
)

df["price_ema20"] = (
    close - df["ema20"]
) / df["ema20"]

df["price_ema50"] = (
    close - df["ema50"]
) / df["ema50"]

df["macd_hist"] = macd.macd_diff()

df["atr_pct"] = (
    df["atr"] / close
)

adx = ta.trend.ADXIndicator(
    high=high,
    low=low,
    close=close
)

df["adx"] = adx.adx()

df["di_diff"] = (
    adx.adx_pos() - adx.adx_neg()
)

df["returns_4h"] = (
    close.pct_change(4)
)

df["returns_24h"] = (
    close.pct_change(24)
)

df["mom10"] = (
    close / close.shift(10)
) - 1

df["volatility"] = (
    df["returns_1h"]
    .rolling(24)
    .std()
)

# PLACEHOLDERS
df["btc_returns"] = 0
df["btc_vol_ratio"] = 1
df["coin_vs_btc"] = 0
df["fng"] = 50

# ==========================================
# CLEAN
# ==========================================

df.dropna(inplace=True)

# ==========================================
# FINAL INPUT
# ==========================================

latest = df[features].iloc[[-1]]

# ==========================================
# PREDICTIONS
# ==========================================

prob = xgb_model.predict_proba(
    latest
)[0][1]

future_price = xgb_reg.predict(
    latest
)[0]

current_price = close.iloc[-1]

change_pct = (
    (
        future_price - current_price
    )
    / current_price
) * 100

# ==========================================
# SIGNAL
# ==========================================

if prob > 0.58:

    signal = "BUY"

elif prob < 0.42:

    signal = "SELL"

else:

    signal = "HOLD"

# ==========================================
# OUTPUT
# ==========================================

print("\n==========================")
print("LIVE AI PREDICTION")
print("==========================")

print(f"Signal      : {signal}")

print(f"Probability : {prob:.4f}")

print(f"Confidence  : {prob*100:.2f}%")

print(f"Current     : {current_price:.2f}")

print(f"Forecast    : {future_price:.2f}")

print(f"Change %    : {change_pct:+.2f}%")
