# RENDER (RNDR) Coin — Full ML Training Pipeline
# ===============================================
# Models : XGBoost + LightGBM + RandomForest (ensemble)
# Tasks  : Classification (BUY/SELL/HOLD) + Regression (future price)
# Data   : Yahoo Finance + Fear & Greed Index + BTC correlation

import warnings, os, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import joblib
from datetime import datetime

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, r2_score
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, VotingClassifier
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import ta

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
TICKER        = "RENDER-USD"   # fallback → RNDR-USD
INTERVAL      = "1h"
PERIOD        = "730d"
FUTURE_HOURS  = 24             # predict 24 h ahead
BUY_THRESH    = 0.03           # +3 %
SELL_THRESH   = -0.03          # -3 %
MODEL_DIR     = "."
SEED          = 42
N_SPLITS      = 5

# ═══════════════════════════════════════════════════════════════
# 1. DOWNLOAD OHLCV
# ═══════════════════════════════════════════════════════════════
def download(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    return df

print("[1] Downloading RENDER-USD ...")
try:
    df = download(TICKER, PERIOD, INTERVAL)
    if len(df) < 500:
        raise ValueError("too few rows")
except Exception as e:
    print(f"    Fallback to RNDR-USD ({e})")
    df = download("RNDR-USD", PERIOD, INTERVAL)

print(f"    Loaded {len(df)} rows  {df.index[0].date()} → {df.index[-1].date()}")

# ═══════════════════════════════════════════════════════════════
# 2. FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════════
print("[2] Fetching Fear & Greed ...")
try:
    r = requests.get(
        "https://api.alternative.me/fng/?limit=900&format=json", timeout=10
    )
    raw = r.json()["data"]
    fng = pd.DataFrame(raw)[["value","timestamp"]]
    fng["timestamp"] = pd.to_datetime(
        fng["timestamp"].astype(int), unit="s"
    ).dt.normalize().dt.tz_localize(None)          # ← naive UTC date
    fng["fng"] = fng["value"].astype(float)
    fng = fng[["timestamp","fng"]].set_index("timestamp").sort_index()
    print(f"    {len(fng)} days fetched")

    # merge: strip tz from index → naive date
    df_date = df.index.normalize().tz_localize(None)
    df["fng"] = df_date.map(fng["fng"]).ffill().fillna(50)
except Exception as e:
    print(f"    Failed ({e}) — using default 50")
    df["fng"] = 50

# ═══════════════════════════════════════════════════════════════
# 3. BTC CORRELATION
# ═══════════════════════════════════════════════════════════════
print("[3] Downloading BTC for correlation ...")
try:
    btc = download("BTC-USD", PERIOD, INTERVAL)
    btc_ret = btc["Close"].pct_change().rename("btc_ret")
    df = df.join(btc_ret, how="left")
    df["btc_ret"] = df["btc_ret"].fillna(0)
except Exception as e:
    print(f"    Failed ({e}) — btc_ret = 0")
    df["btc_ret"] = 0.0

# ═══════════════════════════════════════════════════════════════
# 4. TECHNICAL INDICATORS  (short windows → no massive NaN loss)
# ═══════════════════════════════════════════════════════════════
print("[4] Adding indicators ...")

def safe(series):
    """Replace inf/nan with 0 for safety."""
    return series.replace([np.inf, -np.inf], np.nan).fillna(0)

c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

# ── Trend ──────────────────────────────────────────────────────
df["ema_9"]   = safe(ta.trend.EMAIndicator(c, 9).ema_indicator())
df["ema_21"]  = safe(ta.trend.EMAIndicator(c, 21).ema_indicator())
df["ema_50"]  = safe(ta.trend.EMAIndicator(c, 50).ema_indicator())
df["sma_20"]  = safe(ta.trend.SMAIndicator(c, 20).sma_indicator())

macd = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
df["macd"]        = safe(macd.macd())
df["macd_signal"] = safe(macd.macd_signal())
df["macd_diff"]   = safe(macd.macd_diff())

adx = ta.trend.ADXIndicator(h, l, c, 14)
df["adx"]     = safe(adx.adx())
df["adx_pos"] = safe(adx.adx_pos())
df["adx_neg"] = safe(adx.adx_neg())

# ── Momentum ───────────────────────────────────────────────────
df["rsi_14"]  = safe(ta.momentum.RSIIndicator(c, 14).rsi())
df["rsi_7"]   = safe(ta.momentum.RSIIndicator(c, 7).rsi())
df["rsi_28"]  = safe(ta.momentum.RSIIndicator(c, 28).rsi())

stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
df["stoch_k"] = safe(stoch.stoch())
df["stoch_d"] = safe(stoch.stoch_signal())

df["cci"]     = safe(ta.trend.CCIIndicator(h, l, c, 20).cci())
df["roc_5"]   = safe(ta.momentum.ROCIndicator(c, 5).roc())
df["roc_20"]  = safe(ta.momentum.ROCIndicator(c, 20).roc())
df["williams"]= safe(ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r())

# ── Volatility ─────────────────────────────────────────────────
bb = ta.volatility.BollingerBands(c, 20, 2)
df["bb_upper"] = safe(bb.bollinger_hband())
df["bb_lower"] = safe(bb.bollinger_lband())
df["bb_width"] = safe(bb.bollinger_wband())
df["bb_pct"]   = safe(bb.bollinger_pband())
df["atr_14"]   = safe(ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range())

# ── Volume ─────────────────────────────────────────────────────
df["obv"]  = safe(ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume())
df["mfi"]  = safe(ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index())
df["cmf"]  = safe(ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow())

try:
    df["vwap"] = safe(ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price())
except Exception:
    df["vwap"] = c

# ── Price-derived ──────────────────────────────────────────────
df["ret_1h"]  = safe(c.pct_change(1))
df["ret_4h"]  = safe(c.pct_change(4))
df["ret_24h"] = safe(c.pct_change(24))
df["log_ret"] = safe(np.log(c / c.shift(1)))

df["vol_24h"] = safe(df["ret_1h"].rolling(24).std())
df["vol_7d"]  = safe(df["ret_1h"].rolling(168).std())

hl = (h - l).replace(0, np.nan)
df["body"]        = safe((c - o).abs() / hl)
df["upper_wick"]  = safe((h - c.clip(lower=o)) / hl)
df["lower_wick"]  = safe((c.clip(upper=o) - l) / hl)
df["hl_ratio"]    = safe(h / l)
df["close_open"]  = safe(c / o)

# ── Volume / Whale signals ─────────────────────────────────────
vol_ma24       = v.rolling(24).mean().replace(0, np.nan)
vol_ma168      = v.rolling(168).mean().replace(0, np.nan)
df["vol_r24"]  = safe(v / vol_ma24)
df["vol_r168"] = safe(v / vol_ma168)
df["whale"]    = (df["vol_r24"] > 3).astype(int)   # >3× avg = whale spike

# ── Support / Resistance ───────────────────────────────────────
df["dist_hi24"] = safe((h.rolling(24).max() - c) / c)
df["dist_lo24"] = safe((c - l.rolling(24).min()) / c)
df["dist_hi7d"] = safe((h.rolling(168).max() - c) / c)
df["dist_lo7d"] = safe((c - l.rolling(168).min()) / c)

# ── EMA alignment score ────────────────────────────────────────
df["ema_score"] = (
    (df["ema_9"] > df["ema_21"]).astype(int) +
    (df["ema_21"] > df["ema_50"]).astype(int)
)
df["price_vs_vwap"] = safe((c - df["vwap"]) / df["vwap"])

# ── FNG zones ──────────────────────────────────────────────────
df["fng_fear"]  = (df["fng"] < 25).astype(int)
df["fng_greed"] = (df["fng"] > 75).astype(int)

# ── BTC features ───────────────────────────────────────────────
df["rndr_vs_btc"] = safe(df["ret_1h"] - df["btc_ret"])
df["btc_ret_4h"]  = safe(df["btc_ret"].rolling(4).sum())

# ── Seasonality ────────────────────────────────────────────────
df["hour"] = df.index.hour
df["dow"]  = df.index.dayofweek

print(f"    Shape after indicators: {df.shape}")

# ═══════════════════════════════════════════════════════════════
# 5. CLEAN ALL INF / NaN
# ═══════════════════════════════════════════════════════════════
print("[5] Cleaning data ...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)
df.fillna(0, inplace=True)

# ═══════════════════════════════════════════════════════════════
# 6. LABELS
# ═══════════════════════════════════════════════════════════════
df["future_close"]  = df["Close"].shift(-FUTURE_HOURS)
df["future_return"] = (df["future_close"] - df["Close"]) / (df["Close"] + 1e-9)
df.dropna(subset=["future_close","future_return"], inplace=True)

def make_label(ret):
    if ret >= BUY_THRESH:    return 2   # BUY
    elif ret <= SELL_THRESH: return 0   # SELL
    else:                    return 1   # HOLD

df["label"] = df["future_return"].apply(make_label)

print(f"    Rows: {len(df)}")
print(f"    Labels:\n{df['label'].value_counts().sort_index().rename({0:'SELL',1:'HOLD',2:'BUY'})}")

# ═══════════════════════════════════════════════════════════════
# 7. FEATURES / ARRAYS
# ═══════════════════════════════════════════════════════════════
EXCLUDE = {"Open","High","Low","Close","Volume",
           "future_close","future_return","label"}
features = [c for c in df.columns if c not in EXCLUDE]

X     = df[features].values.astype(np.float64)
y_cls = df["label"].values.astype(int)
y_reg = df["future_close"].values.astype(np.float64)

# Final safety net
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

print(f"    Features: {len(features)}")
print(f"    X shape : {X.shape}  |  any NaN: {np.isnan(X).any()}  |  any Inf: {np.isinf(X).any()}")

# ═══════════════════════════════════════════════════════════════
# 8. SCALE
# ═══════════════════════════════════════════════════════════════
scaler   = RobustScaler()
X_scaled = scaler.fit_transform(X)

# ═══════════════════════════════════════════════════════════════
# 9. MODELS
# ═══════════════════════════════════════════════════════════════
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

cls_models = {
    "XGBoost": XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="mlogloss", random_state=SEED,
        n_jobs=-1, verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=12,
        min_samples_leaf=5, random_state=SEED, n_jobs=-1
    ),
}

reg_models = {
    "XGBoost": XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbosity=0
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300, max_depth=12,
        min_samples_leaf=5, random_state=SEED, n_jobs=-1
    ),
}

# ── Classification CV ──────────────────────────────────────────
print("\n[9] Classification CV ...")
cls_scores = {}
for name, model in cls_models.items():
    accs = []
    for tr, te in tscv.split(X_scaled):
        model.fit(X_scaled[tr], y_cls[tr])
        accs.append(accuracy_score(y_cls[te], model.predict(X_scaled[te])))
    cls_scores[name] = np.mean(accs)
    print(f"    {name:15s}  acc={cls_scores[name]:.4f}  folds={[round(a,3) for a in accs]}")

# ── Regression CV ──────────────────────────────────────────────
print("\n[9] Regression CV ...")
reg_scores = {}
for name, model in reg_models.items():
    maes, r2s = [], []
    for tr, te in tscv.split(X_scaled):
        model.fit(X_scaled[tr], y_reg[tr])
        p = model.predict(X_scaled[te])
        maes.append(mean_absolute_error(y_reg[te], p))
        r2s.append(r2_score(y_reg[te], p))
    reg_scores[name] = np.mean(r2s)
    print(f"    {name:15s}  MAE={np.mean(maes):.4f}  R²={reg_scores[name]:.4f}")

# ═══════════════════════════════════════════════════════════════
# 10. FINAL TRAINING ON ALL DATA
# ═══════════════════════════════════════════════════════════════
print("\n[10] Final training on full data ...")

best_cls_name = max(cls_scores, key=cls_scores.get)
best_cls      = cls_models[best_cls_name]
best_cls.fit(X_scaled, y_cls)
print(f"     Best classifier : {best_cls_name}  ({cls_scores[best_cls_name]:.4f})")

ensemble_cls = VotingClassifier(
    estimators=list(cls_models.items()), voting="soft", n_jobs=-1
)
ensemble_cls.fit(X_scaled, y_cls)
ens_acc = accuracy_score(y_cls, ensemble_cls.predict(X_scaled))
print(f"     Ensemble train acc: {ens_acc:.4f}")
print(f"\n{classification_report(y_cls, ensemble_cls.predict(X_scaled), target_names=['SELL','HOLD','BUY'])}")

best_reg_name = max(reg_scores, key=reg_scores.get)
best_reg      = reg_models[best_reg_name]
best_reg.fit(X_scaled, y_reg)
print(f"     Best regressor  : {best_reg_name}  (R²={reg_scores[best_reg_name]:.4f})")

# ── Feature importance ─────────────────────────────────────────
print("\n[10] Top 15 features (XGBoost classifier):")
xgb_cls = cls_models["XGBoost"]
imp     = xgb_cls.feature_importances_
for rank, i in enumerate(np.argsort(imp)[::-1][:15], 1):
    print(f"     {rank:2d}. {features[i]:28s} {imp[i]:.4f}")

# ═══════════════════════════════════════════════════════════════
# 11. SAVE
# ═══════════════════════════════════════════════════════════════
print("\n[11] Saving models ...")
joblib.dump(best_cls,     os.path.join(MODEL_DIR, "render_model.pkl"))
joblib.dump(ensemble_cls, os.path.join(MODEL_DIR, "render_ensemble_cls.pkl"))
joblib.dump(best_reg,     os.path.join(MODEL_DIR, "render_reg.pkl"))
joblib.dump(scaler,       os.path.join(MODEL_DIR, "render_scaler.pkl"))
joblib.dump(features,     os.path.join(MODEL_DIR, "render_features.pkl"))

meta = {
    "ticker":          TICKER,
    "interval":        INTERVAL,
    "future_hours":    FUTURE_HOURS,
    "buy_thresh":      BUY_THRESH,
    "sell_thresh":     SELL_THRESH,
    "n_features":      len(features),
    "train_rows":      len(df),
    "trained_at":      datetime.utcnow().isoformat(),
    "best_classifier": best_cls_name,
    "best_regressor":  best_reg_name,
    "cls_cv_scores":   cls_scores,
    "reg_cv_r2":       reg_scores,
}
with open(os.path.join(MODEL_DIR, "render_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅ DONE — saved files:")
for fname in ["render_model.pkl","render_ensemble_cls.pkl",
              "render_reg.pkl","render_scaler.pkl",
              "render_features.pkl","render_meta.json"]:
    path = os.path.join(MODEL_DIR, fname)
    kb   = os.path.getsize(path) / 1024
    print(f"   {fname:38s} {kb:8.1f} KB")
