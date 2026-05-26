# =========================================================
# ADVANCED AI CRYPTO PREDICTION SYSTEM v3.0
# =========================================================
# INSTALL:
#
# pip install yfinance pandas numpy ta scikit-learn \
# xgboost lightgbm catboost tensorflow \
# joblib matplotlib requests
# =========================================================

# =========================================================
# IMPORTS
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import requests

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from xgboost import XGBRegressor

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LSTM,
    Bidirectional,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

# =========================================================
# SETTINGS
# =========================================================

TICKER = "BTC-USD"

INTERVAL = "1h"

PERIOD = "730d"

FUTURE_BARS = 12

SEQUENCE_LENGTH = 60

THRESHOLD = 0.012

BUY_PROB = 0.58

SELL_PROB = 0.42

# =========================================================
# FEAR & GREED
# =========================================================

def fetch_fng():

    try:

        url = "https://api.alternative.me/fng/?limit=730"

        r = requests.get(url, timeout=10)

        data = r.json()["data"]

        fng = pd.DataFrame(data)

        fng["timestamp"] = pd.to_datetime(
            fng["timestamp"].astype(int),
            unit="s"
        )

        fng["value"] = fng["value"].astype(float)

        fng["date"] = fng["timestamp"].dt.date

        fng = fng[["date", "value"]]

        return fng

    except:

        return None

# =========================================================
# DOWNLOAD MAIN DATA
# =========================================================

print("\nDOWNLOADING MAIN DATA...\n")

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

if "Datetime" in df.columns:

    df.rename(
        columns={"Datetime": "datetime"},
        inplace=True
    )

elif "Date" in df.columns:

    df.rename(
        columns={"Date": "datetime"},
        inplace=True
    )

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df.sort_values(
    "datetime",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

print("ROWS:", len(df))

# =========================================================
# BTC FEATURES
# =========================================================

print("\nDOWNLOADING BTC FEATURES...\n")

btc = yf.download(
    "BTC-USD",
    interval=INTERVAL,
    period=PERIOD,
    auto_adjust=True,
    progress=False
)

if isinstance(btc.columns, pd.MultiIndex):

    btc.columns = btc.columns.get_level_values(0)

btc.reset_index(inplace=True)

if "Datetime" in btc.columns:

    btc.rename(
        columns={"Datetime": "datetime"},
        inplace=True
    )

elif "Date" in btc.columns:

    btc.rename(
        columns={"Date": "datetime"},
        inplace=True
    )

btc["datetime"] = pd.to_datetime(
    btc["datetime"]
)

btc = btc[
    [
        "datetime",
        "Close",
        "Volume"
    ]
].copy()

btc.columns = [
    "datetime",
    "btc_close",
    "btc_volume"
]

btc["btc_returns"] = (
    btc["btc_close"].pct_change()
)

btc["btc_vol_ma"] = (
    btc["btc_volume"]
    .rolling(20)
    .mean()
)

btc["btc_vol_ratio"] = (
    btc["btc_volume"]
    / btc["btc_vol_ma"]
)

df = pd.merge(
    df,
    btc,
    on="datetime",
    how="left"
)

# =========================================================
# FEAR & GREED
# =========================================================

print("\nFETCHING FEAR & GREED...\n")

fng = fetch_fng()

if fng is not None:

    df["date"] = df["datetime"].dt.date

    df = pd.merge(
        df,
        fng,
        on="date",
        how="left"
    )

    df.rename(
        columns={"value": "fng"},
        inplace=True
    )

    df["fng"] = df["fng"].ffill()

else:

    df["fng"] = 50.0

# =========================================================
# SERIES
# =========================================================

close = df["Close"].squeeze()

high = df["High"].squeeze()

low = df["Low"].squeeze()

volume = df["Volume"].squeeze()

# =========================================================
# INDICATORS
# =========================================================

print("\nCALCULATING INDICATORS...\n")

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
for w in [20, 50, 200]:

    df[f"ema{w}"] = ta.trend.EMAIndicator(
        close=close,
        window=w
    ).ema_indicator()

# EMA STRUCTURE
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
macd = ta.trend.MACD(close=close)

df["macd"] = macd.macd()

df["macd_signal"] = macd.macd_signal()

df["macd_hist"] = macd.macd_diff()

# BOLLINGER
bb = ta.volatility.BollingerBands(
    close=close,
    window=20
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

# VOLUME
df["vol_ma20"] = (
    volume
    .rolling(20)
    .mean()
)

df["vol_ratio"] = (
    volume / df["vol_ma20"]
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

# VOLATILITY
df["volatility"] = (
    df["returns_1h"]
    .rolling(24)
    .std()
)

# BTC RELATIVE
df["coin_vs_btc"] = (
    df["returns_1h"]
    - df["btc_returns"]
)

# =========================================================
# TARGETS
# =========================================================

print("\nCREATING TARGETS...\n")

df["future_close"] = close.shift(
    -FUTURE_BARS
)

future_return = (
    df["future_close"] - close
) / close

# BUY / SELL ONLY
df["target_cls"] = np.where(
    future_return > THRESHOLD,
    1,
    np.where(
        future_return < -THRESHOLD,
        0,
        np.nan
    )
)

# REGRESSION
df["target_reg"] = df["future_close"]

# =========================================================
# FEATURES
# =========================================================

FEATURES = [

    "Close",
    "Volume",

    "rsi14",
    "rsi7",

    "ema_cross",

    "price_ema20",
    "price_ema50",

    "macd",
    "macd_signal",
    "macd_hist",

    "bb_position",

    "atr_pct",

    "adx",
    "di_diff",

    "vol_ratio",

    "returns_1h",
    "returns_4h",
    "returns_24h",

    "mom5",
    "mom10",

    "volatility",

    "btc_returns",
    "btc_vol_ratio",

    "coin_vs_btc",

    "fng"
]

# =========================================================
# CLASSIFICATION DATA
# =========================================================

df_cls = df.dropna(
    subset=["target_cls"]
).copy()

# REMOVE LEAKAGE
X_cls = df_cls[FEATURES].shift(1)

y_cls = pd.Series(
    df_cls["target_cls"]
).astype(int)

full_cls = pd.concat(
    [
        X_cls,
        y_cls
    ],
    axis=1
)

full_cls.dropna(inplace=True)

X_cls = full_cls[FEATURES]

y_cls = full_cls["target_cls"]

# =========================================================
# REGRESSION DATA
# =========================================================

X_reg = df[FEATURES].shift(1)

y_reg = pd.Series(
    df["target_reg"]
)

full_reg = pd.concat(
    [
        X_reg,
        y_reg
    ],
    axis=1
)

full_reg.dropna(inplace=True)

X_reg = full_reg[FEATURES]

y_reg = full_reg["target_reg"]

# =========================================================
# SPLITS
# =========================================================

split_cls = int(
    len(X_cls) * 0.8
)

X_train_cls = X_cls.iloc[:split_cls]

X_test_cls = X_cls.iloc[split_cls:]

y_train_cls = y_cls.iloc[:split_cls]

y_test_cls = y_cls.iloc[split_cls:]

split_reg = int(
    len(X_reg) * 0.8
)

X_train_reg = X_reg.iloc[:split_reg]

X_test_reg = X_reg.iloc[split_reg:]

y_train_reg = y_reg.iloc[:split_reg]

y_test_reg = y_reg.iloc[split_reg:]

# =========================================================
# CLASS WEIGHTS
# =========================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_cls),
    y=y_train_cls
)

class_weight_dict = dict(
    enumerate(weights)
)

scale_pos_weight = (
    class_weight_dict[0]
    /
    class_weight_dict[1]
)

# =========================================================
# RANDOM FOREST
# =========================================================

print("\nTRAINING RANDOM FOREST...\n")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train_cls,
    y_train_cls
)

rf_pred = rf_model.predict(
    X_test_cls
)

rf_acc = accuracy_score(
    y_test_cls,
    rf_pred
)

# =========================================================
# XGBOOST
# =========================================================

print("\nTRAINING XGBOOST...\n")

xgb_model = XGBClassifier(
    n_estimators=700,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(
    X_train_cls,
    y_train_cls
)

xgb_pred = xgb_model.predict(
    X_test_cls
)

xgb_acc = accuracy_score(
    y_test_cls,
    xgb_pred
)

# =========================================================
# LIGHTGBM
# =========================================================

print("\nTRAINING LIGHTGBM...\n")

lgb_model = LGBMClassifier(
    n_estimators=700,
    learning_rate=0.02,
    max_depth=7,
    class_weight="balanced",
    random_state=42,
    verbose=-1
)

lgb_model.fit(
    X_train_cls,
    y_train_cls
)

lgb_pred = lgb_model.predict(
    X_test_cls
)

lgb_acc = accuracy_score(
    y_test_cls,
    lgb_pred
)

# =========================================================
# CATBOOST
# =========================================================

print("\nTRAINING CATBOOST...\n")

cat_model = CatBoostClassifier(
    iterations=700,
    learning_rate=0.02,
    depth=7,
    verbose=0,
    random_seed=42
)

cat_model.fit(
    X_train_cls,
    y_train_cls
)

cat_pred = cat_model.predict(
    X_test_cls
)

cat_acc = accuracy_score(
    y_test_cls,
    cat_pred
)

# =========================================================
# REGRESSION
# =========================================================

print("\nTRAINING REGRESSION...\n")

xgb_reg = XGBRegressor(
    n_estimators=700,
    learning_rate=0.02,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb_reg.fit(
    X_train_reg,
    y_train_reg
)

reg_pred = xgb_reg.predict(
    X_test_reg
)

reg_r2 = r2_score(
    y_test_reg,
    reg_pred
)

# =========================================================
# LSTM
# =========================================================

print("\nTRAINING LSTM...\n")

scaler = RobustScaler()

scaled = scaler.fit_transform(
    X_cls
)

X_lstm = []

y_lstm = []

for i in range(
    SEQUENCE_LENGTH,
    len(scaled)
):

    X_lstm.append(
        scaled[
            i-SEQUENCE_LENGTH:i
        ]
    )

    y_lstm.append(
        y_cls.iloc[i]
    )

X_lstm = np.array(X_lstm)

y_lstm = np.array(y_lstm)

split_lstm = int(
    len(X_lstm) * 0.8
)

X_train_lstm = X_lstm[:split_lstm]

X_test_lstm = X_lstm[split_lstm:]

y_train_lstm = y_lstm[:split_lstm]

y_test_lstm = y_lstm[split_lstm:]

lstm_model = Sequential()

lstm_model.add(

    Bidirectional(

        LSTM(
            128,
            return_sequences=True
        ),

        input_shape=(
            X_train_lstm.shape[1],
            X_train_lstm.shape[2]
        )
    )
)

lstm_model.add(
    BatchNormalization()
)

lstm_model.add(
    Dropout(0.3)
)

lstm_model.add(

    Bidirectional(

        LSTM(
            64
        )
    )
)

lstm_model.add(
    BatchNormalization()
)

lstm_model.add(
    Dropout(0.3)
)

lstm_model.add(
    Dense(
        32,
        activation="relu"
    )
)

lstm_model.add(
    Dropout(0.2)
)

lstm_model.add(
    Dense(
        1,
        activation="sigmoid"
    )
)

lstm_model.compile(
    optimizer=Adam(
        learning_rate=0.0005
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3
    )
]

lstm_model.fit(
    X_train_lstm,
    y_train_lstm,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=callbacks
)

loss, lstm_acc = lstm_model.evaluate(
    X_test_lstm,
    y_test_lstm
)

# =========================================================
# ENSEMBLE
# =========================================================

print("\nBUILDING ENSEMBLE...\n")

weights = np.array([
    rf_acc,
    xgb_acc,
    lgb_acc,
    cat_acc
])

weights = weights / weights.sum()

latest = X_cls.iloc[[-1]]

rf_prob = rf_model.predict_proba(
    latest
)[0][1]

xgb_prob = xgb_model.predict_proba(
    latest
)[0][1]

lgb_prob = lgb_model.predict_proba(
    latest
)[0][1]

cat_prob = cat_model.predict_proba(
    latest
)[0][1]

avg_prob = (

    weights[0] * rf_prob +

    weights[1] * xgb_prob +

    weights[2] * lgb_prob +

    weights[3] * cat_prob
)

if avg_prob > BUY_PROB:

    signal = "BUY"

elif avg_prob < SELL_PROB:

    signal = "SELL"

else:

    signal = "HOLD"

confidence = avg_prob * 100

# =========================================================
# FORECAST
# =========================================================

future_price = xgb_reg.predict(
    X_reg.iloc[[-1]]
)[0]

current_price = df["Close"].iloc[-1]

change_pct = (
    (
        future_price - current_price
    )
    / current_price
) * 100

# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION")
print("=" * 60)

print(f"RF       : {rf_acc:.4f}")

print(f"XGB      : {xgb_acc:.4f}")

print(f"LGBM     : {lgb_acc:.4f}")

print(f"CATBOOST : {cat_acc:.4f}")

print(f"LSTM     : {lstm_acc:.4f}")

print("\n")
print("=" * 60)
print("REGRESSION")
print("=" * 60)

print(f"R² : {reg_r2:.4f}")

print("\n")
print("=" * 60)
print("LIVE SIGNAL")
print("=" * 60)

print(f"RF PROB     : {rf_prob:.4f}")

print(f"XGB PROB    : {xgb_prob:.4f}")

print(f"LGBM PROB   : {lgb_prob:.4f}")

print(f"CAT PROB    : {cat_prob:.4f}")

print(f"\nAVG PROB    : {avg_prob:.4f}")

print(f"SIGNAL      : {signal}")

print(f"CONFIDENCE  : {confidence:.2f}%")

print("\n")
print("=" * 60)
print("PRICE FORECAST")
print("=" * 60)

print(f"CURRENT PRICE : {current_price:.4f}")

print(f"FUTURE PRICE  : {future_price:.4f}")

print(f"CHANGE %      : {change_pct:+.2f}%")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance = pd.Series(
    xgb_model.feature_importances_,
    index=FEATURES
)

importance = importance.sort_values(
    ascending=False
)

print("\n")
print("=" * 60)
print("TOP FEATURES")
print("=" * 60)

print(
    importance.head(15)
)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(12, 8))

importance.head(20).plot(
    kind="barh"
)

plt.title(
    "Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=150
)

# =========================================================
# SAVE
# =========================================================

print("\nSAVING MODELS...\n")

joblib.dump(
    rf_model,
    "rf_model.pkl"
)

joblib.dump(
    xgb_model,
    "xgb_model.pkl"
)

joblib.dump(
    lgb_model,
    "lgb_model.pkl"
)

joblib.dump(
    cat_model,
    "cat_model.pkl"
)

joblib.dump(
    xgb_reg,
    "xgb_reg.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

joblib.dump(
    FEATURES,
    "features.pkl"
)

lstm_model.save(
    "lstm_model.h5"
)

print("\nMODELS SAVED")

print("\nDONE")
