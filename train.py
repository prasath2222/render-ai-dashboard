#!/usr/bin/env python3
"""
predict.py — Daily signal from the champion model
Loads models/classifier.pkl, regressor.pkl, features.json, metrics.json
Outputs: BUY/SELL/HOLD, probabilities, forecast price, TP/SL
"""
import pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ── Helper (same as train) ──────────────────────────────────────────────
def _zscore_rolling(s, w, min_periods=10):
    mu = s.rolling(w, min_periods=min_periods).mean()
    std = s.rolling(w, min_periods=min_periods).std()
    return (s - mu) / std.replace(0, np.nan)

# ── Data fetching (same as train but with no save) ──────────────────────
def fetch_render():
    old = yf.download("RNDR-USD", start="2021-01-01", end="2024-07-14", progress=False)
    new = yf.download("RENDER-USD", start="2024-07-15", progress=False)
    old_close = old["Close"].squeeze()
    new_close = new["Close"].squeeze()
    old_close.index = pd.to_datetime(old_close.index).tz_localize(None)
    new_close.index = pd.to_datetime(new_close.index).tz_localize(None)
    close = pd.concat([old_close, new_close]).sort_index()
    close = close[~close.index.duplicated(keep="last")]
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["open"]  = close.shift(1).fillna(close)
    df["high"]  = close * 1.005
    df["low"]   = close * 0.995
    df["vol"]   = 1000.0
    return df.dropna(subset=["close"])

def fetch_btc_dominance_proxy():
    tickers = ["BTC-USD","ETH-USD","SOL-USD","AVAX-USD","LINK-USD"]
    prices = {}
    for t in tickers:
        try:
            data = yf.download(t, start="2021-01-01", progress=False)
            if not data.empty:
                close = data["Close"].squeeze()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                prices[t.split("-")[0]] = close.dropna()
        except: pass
    if "BTC" not in prices or len(prices) < 3: return None
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    daily = pd.date_range("2021-01-01", today, freq="D")
    aligned = {n: s.reindex(daily, method="ffill") for n, s in prices.items()}
    btc = aligned.pop("BTC")
    alt_avg = pd.DataFrame(aligned).mean(axis=1)
    btc_norm = btc / btc.iloc[0]
    alt_norm = alt_avg / alt_avg.iloc[0]
    dom = btc_norm / (btc_norm + alt_norm)
    dom = dom.replace([np.inf, -np.inf], np.nan).ffill()
    dom.name = "btc_d_proxy"
    return dom

def build_features(render_df, btc_dom_series):
    close = render_df["close"]
    features = {}
    for lag in [7, 14, 30]:
        features[f"mom_{lag}d"] = close.pct_change(lag)
    if btc_dom_series is not None:
        btc_aligned = btc_dom_series.reindex(close.index, method="ffill")
        features["btc_d_proxy"]        = btc_aligned
        features["btc_d_proxy_7d"]     = btc_aligned.pct_change(7)
        features["btc_d_proxy_30d"]    = btc_aligned.pct_change(30)
        features["btc_d_proxy_zscore"] = _zscore_rolling(btc_aligned, 30)
    return pd.DataFrame(features, index=close.index)

# ── Main prediction ────────────────────────────────────────────────────
def main():
    # Load models & config
    with open("models/classifier.pkl", "rb") as f: clf = pickle.load(f)
    with open("models/regressor.pkl", "rb") as f: reg = pickle.load(f)
    with open("models/features.json") as f: feature_cols = json.load(f)
    with open("models/metrics.json") as f: metrics = json.load(f)

    threshold = metrics["threshold"]
    risk_mult = metrics["risk_multiplier"]

    # Get latest data
    render = fetch_render()
    btc_dom = fetch_btc_dominance_proxy()
    if btc_dom is None: raise RuntimeError("BTC Dominance data unavailable")

    features_df = build_features(render, btc_dom)
    # Use the last row (today)
    latest = features_df.iloc[-1:].fillna(features_df.median()).values.astype(float)

    # Classifier probabilities
    proba = clf.predict_proba(latest)[0]
    p_down, p_up = proba[0], proba[1]

    if p_up >= threshold:
        signal = "BUY"
    elif p_down >= threshold:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Regressor forecast
    forecast_price = reg.predict(latest)[0]
    current_price = render["close"].iloc[-1]
    expected_return = (forecast_price / current_price - 1) * 100

    # TP / SL based on risk multiplier (example: 10% profit, 5% stop)
    tp_price = current_price * (1 + risk_mult * 0.10)
    sl_price = current_price * (1 - risk_mult * 0.05)

    print(f"Signal: {signal}")
    print(f"Current price: ${current_price:.4f}")
    print(f"P(UP): {p_up:.4f}   P(DOWN): {p_down:.4f}")
    print(f"Forecast 30d price: ${forecast_price:.4f}")
    print(f"Expected return: {expected_return:+.2f}%")
    print(f"Take Profit (TP): ${tp_price:.4f}")
    print(f"Stop Loss (SL): ${sl_price:.4f}")

if __name__ == "__main__":
    main()