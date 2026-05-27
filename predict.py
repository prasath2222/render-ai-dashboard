"""
=============================================================
RENDER (RNDR) — REAL-TIME PREDICTION ENGINE
=============================================================
Run AFTER train.py has finished.

Outputs:
  - Direction signal (UP / FLAT / DOWN) with probabilities
  - Predicted 6h return (ensemble of XGBoost + LSTM)
  - Predicted price
  - Confidence (model agreement score)
  - Telegram alert
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────
# Optional Telegram alert (safe fallback if not configured)
# ─────────────────────────────────────────────────────────
try:
    from telegram_alert import send_alert
    TELEGRAM_ENABLED = True
except ImportError:
    TELEGRAM_ENABLED = False
    def send_alert(msg):
        pass

# ─────────────────────────────────────────────────────────
# CONFIG — must match train.py
# ─────────────────────────────────────────────────────────
RENDER_TICKER = "RENDER-USD"
BTC_TICKER    = "BTC-USD"
ETH_TICKER    = "ETH-USD"
SOL_TICKER    = "SOL-USD"
NVDA_TICKER   = "NVDA"

INTERVAL  = "1h"
PERIOD    = "60d"
LOOKBACK  = 48

XGB_DIRECTION_PATH = "xgb_direction.joblib"
XGB_REGRESSOR_PATH = "xgb_regressor.joblib"
LSTM_MODEL_PATH    = "lstm_price.keras"
SCALER_X_PATH      = "scaler_x.joblib"
SCALER_Y_PATH      = "scaler_y.joblib"
FEATURE_COLS_PATH  = "feature_cols.joblib"

DIRECTION_LABELS = {0: "DOWN ↓", 1: "FLAT →", 2: "UP ↑"}
DIRECTION_EMOJIS = {0: "🔴", 1: "🟡", 2: "🟢"}

# ─────────────────────────────────────────────────────────
# Replicate feature engineering from train.py
# ─────────────────────────────────────────────────────────
def engineer_features(df):
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    df["log_return"]     = np.log(close / close.shift(1))
    df["return_3h"]      = close.pct_change(3)
    df["return_6h"]      = close.pct_change(6)
    df["return_12h"]     = close.pct_change(12)
    df["return_24h"]     = close.pct_change(24)
    df["return_48h"]     = close.pct_change(48)

    df["EMA8"]    = ta.trend.ema_indicator(close, 8)
    df["EMA20"]   = ta.trend.ema_indicator(close, 20)
    df["EMA50"]   = ta.trend.ema_indicator(close, 50)
    df["EMA100"]  = ta.trend.ema_indicator(close, 100)
    df["EMA200"]  = ta.trend.ema_indicator(close, 200)

    df["ema8_slope"]  = df["EMA8"].diff(3)
    df["ema20_slope"] = df["EMA20"].diff(3)
    df["price_vs_ema20"]  = (close - df["EMA20"]) / df["EMA20"]
    df["price_vs_ema50"]  = (close - df["EMA50"]) / df["EMA50"]
    df["price_vs_ema200"] = (close - df["EMA200"]) / df["EMA200"]
    df["ema20_vs_ema50"]  = (df["EMA20"] - df["EMA50"]) / df["EMA50"]
    df["ema50_vs_ema200"] = (df["EMA50"] - df["EMA200"]) / df["EMA200"]

    df["RSI14"]  = ta.momentum.rsi(close, 14)
    df["RSI7"]   = ta.momentum.rsi(close, 7)
    df["RSI21"]  = ta.momentum.rsi(close, 21)
    df["RSI14_slope"] = df["RSI14"].diff(3)

    stoch = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    df["ROC10"] = ta.momentum.roc(close, 10)
    df["ROC20"] = ta.momentum.roc(close, 20)

    macd = ta.trend.MACD(close, 26, 12, 9)
    df["MACD"]            = macd.macd()
    df["MACD_SIGNAL"]     = macd.macd_signal()
    df["MACD_HIST"]       = macd.macd_diff()
    df["MACD_HIST_slope"] = df["MACD_HIST"].diff(3)

    df["ADX"]     = ta.trend.adx(high, low, close, 14)
    df["ADX_POS"] = ta.trend.adx_pos(high, low, close, 14)
    df["ADX_NEG"] = ta.trend.adx_neg(high, low, close, 14)
    df["DI_DIFF"] = df["ADX_POS"] - df["ADX_NEG"]

    df["ATR14"]    = ta.volatility.average_true_range(high, low, close, 14)
    df["ATR_norm"] = df["ATR14"] / close

    bb = ta.volatility.BollingerBands(close, 20, 2)
    df["BB_UPPER"]    = bb.bollinger_hband()
    df["BB_LOWER"]    = bb.bollinger_lband()
    df["BB_WIDTH"]    = (df["BB_UPPER"] - df["BB_LOWER"]) / bb.bollinger_mavg()
    df["BB_POSITION"] = (close - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"] + 1e-9)

    df["KELT_UPPER"] = ta.volatility.keltner_channel_hband(high, low, close, 20)
    df["KELT_LOWER"] = ta.volatility.keltner_channel_lband(high, low, close, 20)
    df["KELT_POS"]   = (close - df["KELT_LOWER"]) / (df["KELT_UPPER"] - df["KELT_LOWER"] + 1e-9)

    df["SQUEEZE"] = (
        (df["BB_UPPER"] < df["KELT_UPPER"]) &
        (df["BB_LOWER"] > df["KELT_LOWER"])
    ).astype(int)

    df["RVOL_12h"] = df["log_return"].rolling(12).std()
    df["RVOL_24h"] = df["log_return"].rolling(24).std()

    df["VOL_SMA20"]  = ta.trend.sma_indicator(volume, 20)
    df["VOL_RATIO"]  = volume / (df["VOL_SMA20"] + 1e-9)
    df["VOL_CHANGE"] = volume.pct_change()

    df["OBV"]       = ta.volume.on_balance_volume(close, volume)
    df["OBV_slope"] = df["OBV"].diff(6)

    df["VWAP_24h"]      = (close * volume).rolling(24).sum() / (volume.rolling(24).sum() + 1e-9)
    df["price_vs_vwap"] = (close - df["VWAP_24h"]) / df["VWAP_24h"]

    df["high_24h"] = high.rolling(24).max()
    df["low_24h"]  = low.rolling(24).min()
    df["dist_to_resistance"] = (df["high_24h"] - close) / close
    df["dist_to_support"]    = (close - df["low_24h"]) / close

    for col, name in [("BTC_Close", "BTC"), ("ETH_Close", "ETH"), ("SOL_Close", "SOL")]:
        if col in df.columns:
            ext_ret = np.log(df[col] / df[col].shift(1))
            df[f"{name}_return_1h"]  = ext_ret
            df[f"{name}_return_6h"]  = np.log(df[col] / df[col].shift(6))
            df[f"{name}_return_24h"] = np.log(df[col] / df[col].shift(24))
            df[f"RENDER_{name}_corr48"] = df["log_return"].rolling(48).corr(ext_ret)

    if "NVDA_Close" in df.columns:
        df["NVDA_return_1d"] = np.log(df["NVDA_Close"] / df["NVDA_Close"].shift(1))

    df["hour"]        = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


# ─────────────────────────────────────────────────────────
# MAIN PREDICT
# ─────────────────────────────────────────────────────────
def main():
    # ── Load models ──────────────────────────────────────
    for path in [XGB_DIRECTION_PATH, XGB_REGRESSOR_PATH, LSTM_MODEL_PATH,
                 SCALER_X_PATH, SCALER_Y_PATH, FEATURE_COLS_PATH]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing: {path} — run train.py first")
            return

    xgb_dir    = joblib.load(XGB_DIRECTION_PATH)
    xgb_reg    = joblib.load(XGB_REGRESSOR_PATH)
    scaler_x   = joblib.load(SCALER_X_PATH)
    scaler_y   = joblib.load(SCALER_Y_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    lstm_model = load_model(LSTM_MODEL_PATH)

    # ── Download fresh data ───────────────────────────────
    render = yf.download(RENDER_TICKER, interval=INTERVAL, period=PERIOD, auto_adjust=True)
    btc    = yf.download(BTC_TICKER,    interval=INTERVAL, period=PERIOD, auto_adjust=True)
    eth    = yf.download(ETH_TICKER,    interval=INTERVAL, period=PERIOD, auto_adjust=True)
    sol    = yf.download(SOL_TICKER,    interval=INTERVAL, period=PERIOD, auto_adjust=True)

    df = render.copy()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

    for name, ext in [("BTC", btc), ("ETH", eth), ("SOL", sol)]:
        col = ext["Close"]
        col.name = name + "_Close"
        df = df.join(col, how="left")

    try:
        nvda = yf.download(NVDA_TICKER, interval="1d", period=PERIOD, auto_adjust=True)
        nvda_close = nvda["Close"].resample("1h").ffill()
        nvda_close.name = "NVDA_Close"
        df = df.join(nvda_close, how="left")
    except Exception:
        pass

    df = df.ffill().dropna()
    df = engineer_features(df)

    # ── Extract latest row ───────────────────────────────
    X_latest = df[feature_cols].values[-1:].astype(np.float32)
    X_scaled = scaler_x.transform(X_latest)

    # ── XGBoost Direction ─────────────────────────────────
    dir_probs  = xgb_dir.predict_proba(X_scaled)[0]   # [P(DOWN), P(FLAT), P(UP)]
    dir_class  = int(np.argmax(dir_probs))
    dir_conf   = float(dir_probs[dir_class])

    # ── XGBoost Return ────────────────────────────────────
    xgb_return = float(xgb_reg.predict(X_scaled)[0])

    # ── LSTM Return ───────────────────────────────────────
    X_seq = df[feature_cols].values[-LOOKBACK:].astype(np.float32)
    X_seq_scaled = scaler_x.transform(X_seq)
    X_seq_3d = X_seq_scaled.reshape(1, LOOKBACK, -1)
    lstm_return_scaled = float(lstm_model.predict(X_seq_3d, verbose=0)[0][0])
    lstm_return = float(scaler_y.inverse_transform([[lstm_return_scaled]])[0][0])

    # ── Ensemble: weight XGB more for direction, LSTM for magnitude ──
    # XGB direction agreement with return sign → weight
    xgb_weight  = 0.60
    lstm_weight = 0.40
    ensemble_return = (xgb_weight * xgb_return) + (lstm_weight * lstm_return)

    # ── Current price ─────────────────────────────────────
    current_price   = float(df["Close"].iloc[-1])
    predicted_price = current_price * (1 + ensemble_return)
    predicted_pct   = ensemble_return * 100

    # ── Confidence: model agreement score ─────────────────
    # High when XGB direction matches return sign AND dir_conf is high
    sign_agrees = (
        (dir_class == 2 and ensemble_return > 0) or
        (dir_class == 0 and ensemble_return < 0) or
        (dir_class == 1 and abs(ensemble_return) < 0.005)
    )
    confidence = dir_conf * 100 * (1.15 if sign_agrees else 0.75)
    confidence = min(95, max(40, confidence))

    # ── ATR for SL/TP ────────────────────────────────────
    atr = float(df["ATR14"].iloc[-1])
    stop_loss    = current_price - atr * 2
    take_profit1 = current_price + atr * 3
    take_profit2 = current_price + atr * 6
    risk_reward  = (take_profit1 - current_price) / (current_price - stop_loss + 1e-9)

    # ── Market regime ────────────────────────────────────
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    rsi   = float(df["RSI14"].iloc[-1])
    adx   = float(df["ADX"].iloc[-1])

    if ema20 > ema50 and rsi > 55:
        regime = "BULLISH 🐂"
    elif ema20 < ema50 and rsi < 45:
        regime = "BEARISH 🐻"
    else:
        regime = "SIDEWAYS ↔️"

    # ── Squeeze alert ─────────────────────────────────────
    squeeze = bool(df["SQUEEZE"].iloc[-1])
    btc_corr = float(df.get("RENDER_BTC_corr48", pd.Series([0])).iloc[-1])

    # ── Print output ─────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"""
╔══════════════════════════════════════════════════╗
║      RENDER ML PREDICTION ENGINE  [{now}]
╠══════════════════════════════════════════════════╣
║  Current Price    : ${current_price:.4f}
║  Predicted Price  : ${predicted_price:.4f}
║  Predicted Change : {predicted_pct:+.2f}%  (6h horizon)
╠══════════════════════════════════════════════════╣
║  DIRECTION SIGNAL : {DIRECTION_EMOJIS[dir_class]} {DIRECTION_LABELS[dir_class]}
║  Confidence       : {confidence:.1f}%
║  Model Agreement  : {'✅ YES' if sign_agrees else '⚠️  SPLIT'}
╠══════════════════════════════════════════════════╣
║  XGB Return Pred  : {xgb_return*100:+.3f}%
║  LSTM Return Pred : {lstm_return*100:+.3f}%
║  Ensemble Return  : {ensemble_return*100:+.3f}%
╠══════════════════════════════════════════════════╣
║  P(DOWN) : {dir_probs[0]*100:.1f}%
║  P(FLAT) : {dir_probs[1]*100:.1f}%
║  P(UP)   : {dir_probs[2]*100:.1f}%
╠══════════════════════════════════════════════════╣
║  RSI14  : {rsi:.1f}
║  ADX    : {adx:.1f}  ({'Trending' if adx > 25 else 'Choppy'})
║  Regime : {regime}
║  Squeeze: {'🔥 ACTIVE — breakout imminent' if squeeze else 'None'}
║  BTC Corr 48h: {btc_corr:.2f}
╠══════════════════════════════════════════════════╣
║  Entry      : ${current_price:.4f}
║  Stop Loss  : ${stop_loss:.4f}  (-{((current_price-stop_loss)/current_price)*100:.2f}%)
║  TP1        : ${take_profit1:.4f}  (+{((take_profit1-current_price)/current_price)*100:.2f}%)
║  TP2        : ${take_profit2:.4f}  (+{((take_profit2-current_price)/current_price)*100:.2f}%)
║  Risk/Reward: {risk_reward:.2f}
╚══════════════════════════════════════════════════╝
""")

    # ── Telegram alert ────────────────────────────────────
    emoji = DIRECTION_EMOJIS[dir_class]
    label = DIRECTION_LABELS[dir_class]

    msg = f"""
{emoji} RENDER ML SIGNAL

💰 Price: ${current_price:.4f}
📊 Direction: {label}
🎯 Confidence: {confidence:.1f}%
🔮 Predicted 6h: ${predicted_price:.4f} ({predicted_pct:+.2f}%)

📊 P(UP)   {dir_probs[2]*100:.0f}%
📊 P(FLAT) {dir_probs[1]*100:.0f}%
📊 P(DOWN) {dir_probs[0]*100:.0f}%

📈 RSI: {rsi:.1f} | ADX: {adx:.1f}
🌡️ Regime: {regime}
{'🔥 SQUEEZE — breakout incoming!' if squeeze else ''}

🟢 TP1: ${take_profit1:.4f}
🟢 TP2: ${take_profit2:.4f}
🔴 SL:  ${stop_loss:.4f}
⚖️ R:R = {risk_reward:.2f}
"""

    if TELEGRAM_ENABLED:
        send_alert(msg)
        print("  Telegram alert sent.")
    else:
        print("  [Telegram not configured — skipped]")


if __name__ == "__main__":
    main()
