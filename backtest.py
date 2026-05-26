# =========================================================
# BACKTEST SYSTEM
# =========================================================
# REQUIREMENTS:
#
# xgb_model.pkl
# xgb_reg.pkl
# scaler.pkl
# features.pkl
#
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

# =========================================================
# LOAD MODELS
# =========================================================

print("\nLOADING MODELS...\n")

xgb_model = joblib.load(
    "xgb_model.pkl"
)

xgb_reg = joblib.load(
    "xgb_reg.pkl"
)

features = joblib.load(
    "features.pkl"
)

# =========================================================
# SETTINGS
# =========================================================

TICKER = "BTC-USD"

INTERVAL = "1h"

PERIOD = "365d"

INITIAL_BALANCE = 1000

BUY_THRESHOLD = 0.58

SELL_THRESHOLD = 0.42

TRADE_SIZE = 1.0

STOP_LOSS = 0.03

TAKE_PROFIT = 0.06

FEE = 0.001

# =========================================================
# DOWNLOAD DATA
# =========================================================

print("\nDOWNLOADING DATA...\n")

df = yf.download(
    TICKER,
    interval=INTERVAL,
    period=PERIOD,
    auto_adjust=True,
    progress=False
)

if isinstance(df.columns, pd.MultiIndex):

    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

# =========================================================
# SERIES
# =========================================================

close = df["Close"]

high = df["High"]

low = df["Low"]

volume = df["Volume"]

# =========================================================
# FEATURES
# =========================================================

print("\nCALCULATING FEATURES...\n")

# RSI
df["rsi14"] = ta.momentum.RSIIndicator(
    close=close,
    window=14
).rsi()

df["rsi7"] = ta.momentum.RSIIndicator(
    close=close,
    window=7
).rsi()

# EMA
df["ema20"] = ta.trend.EMAIndicator(
    close=close,
    window=20
).ema_indicator()

df["ema50"] = ta.trend.EMAIndicator(
    close=close,
    window=50
).ema_indicator()

df["ema200"] = ta.trend.EMAIndicator(
    close=close,
    window=200
).ema_indicator()

# EMA FEATURES
df["ema_cross"] = (
    df["ema20"] - df["ema50"]
)

df["price_ema20"] = (
    close - df["ema20"]
) / df["ema20"]

df["price_ema50"] = (
    close - df["ema50"]
) / df["ema50"]

# MACD
macd = ta.trend.MACD(
    close=close
)

df["macd"] = macd.macd()

df["macd_signal"] = macd.macd_signal()

df["macd_hist"] = macd.macd_diff()

# BOLLINGER
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

# ATR
df["atr"] = ta.volatility.AverageTrueRange(
    high=high,
    low=low,
    close=close
).average_true_range()

df["atr_pct"] = (
    df["atr"] / close
)

# ADX
adx = ta.trend.ADXIndicator(
    high=high,
    low=low,
    close=close
)

df["adx"] = adx.adx()

df["di_diff"] = (
    adx.adx_pos() - adx.adx_neg()
)

# RETURNS
df["returns_1h"] = (
    close.pct_change()
)

df["returns_4h"] = (
    close.pct_change(4)
)

df["returns_24h"] = (
    close.pct_change(24)
)

# MOMENTUM
df["mom5"] = (
    close / close.shift(5)
) - 1

df["mom10"] = (
    close / close.shift(10)
) - 1

# VOLUME
df["vol_ma20"] = (
    volume
    .rolling(20)
    .mean()
)

df["vol_ratio"] = (
    volume / df["vol_ma20"]
)

# VOLATILITY
df["volatility"] = (
    df["returns_1h"]
    .rolling(24)
    .std()
)

# PLACEHOLDER FEATURES
df["btc_returns"] = 0

df["btc_vol_ratio"] = 1

df["coin_vs_btc"] = 0

df["fng"] = 50

# =========================================================
# CLEAN
# =========================================================

df.dropna(inplace=True)

# =========================================================
# LEAKAGE PREVENTION
# =========================================================

X = df[features].shift(1)

full = pd.concat(
    [
        X,
        df["Close"]
    ],
    axis=1
)

full.dropna(inplace=True)

X = full[features]

prices = full["Close"]

# =========================================================
# BACKTEST VARIABLES
# =========================================================

balance = INITIAL_BALANCE

position = 0

entry_price = 0

trade_count = 0

wins = 0

losses = 0

equity_curve = []

trade_history = []

# =========================================================
# BACKTEST LOOP
# =========================================================

print("\nRUNNING BACKTEST...\n")

for i in range(len(X)):

    current_features = X.iloc[[i]]

    current_price = prices.iloc[i]

    # ==========================================
    # PREDICTIONS
    # ==========================================

    prob = xgb_model.predict_proba(
        current_features
    )[0][1]

    future_price = xgb_reg.predict(
        current_features
    )[0]

    # ==========================================
    # SIGNAL
    # ==========================================

    if prob > BUY_THRESHOLD:

        signal = "BUY"

    elif prob < SELL_THRESHOLD:

        signal = "SELL"

    else:

        signal = "HOLD"

    # ==========================================
    # ENTRY
    # ==========================================

    if position == 0:

        if signal == "BUY":

            position = (
                balance * TRADE_SIZE
            ) / current_price

            entry_price = current_price

            balance = 0

            trade_count += 1

            trade_history.append(
                {
                    "type": "BUY",
                    "price": current_price
                }
            )

    # ==========================================
    # EXIT
    # ==========================================

    else:

        pnl_pct = (
            current_price - entry_price
        ) / entry_price

        stop_hit = (
            pnl_pct <= -STOP_LOSS
        )

        take_hit = (
            pnl_pct >= TAKE_PROFIT
        )

        sell_signal = (
            signal == "SELL"
        )

        if stop_hit or take_hit or sell_signal:

            final_value = (
                position * current_price
            )

            final_value *= (
                1 - FEE
            )

            if final_value > INITIAL_BALANCE:

                wins += 1

            else:

                losses += 1

            balance = final_value

            position = 0

            trade_history.append(
                {
                    "type": "SELL",
                    "price": current_price,
                    "pnl_pct": pnl_pct
                }
            )

    # ==========================================
    # EQUITY CURVE
    # ==========================================

    if position > 0:

        equity = (
            position * current_price
        )

    else:

        equity = balance

    equity_curve.append(
        equity
    )

# =========================================================
# FINAL RESULTS
# =========================================================

final_balance = equity_curve[-1]

total_return = (
    (
        final_balance - INITIAL_BALANCE
    )
    / INITIAL_BALANCE
) * 100

# =========================================================
# WINRATE
# =========================================================

total_closed = wins + losses

if total_closed > 0:

    winrate = (
        wins / total_closed
    ) * 100

else:

    winrate = 0

# =========================================================
# DRAWDOWN
# =========================================================

equity_series = pd.Series(
    equity_curve
)

rolling_max = equity_series.cummax()

drawdown = (
    equity_series - rolling_max
) / rolling_max

max_drawdown = (
    drawdown.min() * 100
)

# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("BACKTEST RESULTS")
print("=" * 60)

print(f"Initial Balance : ${INITIAL_BALANCE:.2f}")

print(f"Final Balance   : ${final_balance:.2f}")

print(f"Total Return    : {total_return:+.2f}%")

print(f"Trades          : {trade_count}")

print(f"Wins            : {wins}")

print(f"Losses          : {losses}")

print(f"Winrate         : {winrate:.2f}%")

print(f"Max Drawdown    : {max_drawdown:.2f}%")

# =========================================================
# PLOT EQUITY
# =========================================================

plt.figure(figsize=(14, 7))

plt.plot(
    equity_curve
)

plt.title(
    "Equity Curve"
)

plt.xlabel(
    "Trades"
)

plt.ylabel(
    "Balance"
)

plt.grid()

plt.tight_layout()

plt.savefig(
    "backtest_equity.png",
    dpi=150
)

plt.show()

print("\nBACKTEST COMPLETE")
