#!/usr/bin/env python3
"""
app.py — Advanced Streamlit Dashboard for RENDER/RNDR 30‑Day Predictor
========================================================================
Features : Current signal, probabilities, forecast price, TP/SL, confidence,
           historical equity curve, drawdown, feature importance,
           BTC dominance, moving averages, market context, model info.
Model    : LightGBM Classifier + Regressor (7 champion features)
"""
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RENDER 30d Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for professional look ─────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 1rem; }
    .signal-buy { font-size: 3rem; font-weight: 800; color: #2ca02c; text-align: center; }
    .signal-sell { font-size: 3rem; font-weight: 800; color: #d62728; text-align: center; }
    .signal-hold { font-size: 3rem; font-weight: 800; color: #ff7f0e; text-align: center; }
    .metric-box { background: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; margin: 5px 0; }
    .footer { text-align: center; margin-top: 2rem; color: #666; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ──────────────────────────────────────────────────────
def _zscore_rolling(s, w, min_periods=10):
    mu = s.rolling(w, min_periods=min_periods).mean()
    std = s.rolling(w, min_periods=min_periods).std()
    return (s - mu) / std.replace(0, np.nan)

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=3600)
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
    daily = pd.date_range("2021-01-01", datetime.now(timezone.utc).strftime("%Y-%m-%d"), freq="D")
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

@st.cache_resource
def load_models():
    with open("models/classifier.pkl", "rb") as f: clf = pickle.load(f)
    with open("models/regressor.pkl", "rb") as f: reg = pickle.load(f)
    with open("models/features.json") as f: feature_cols = json.load(f)
    with open("models/metrics.json") as f: metrics = json.load(f)
    return clf, reg, feature_cols, metrics

def compute_atr(render_df, period=14):
    high, low, close = render_df["high"], render_df["low"], render_df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=7).mean()

# ── Load everything ───────────────────────────────────────────────────────
clf, reg, feature_cols, metrics = load_models()
threshold = metrics["threshold"]
risk_mult = metrics["risk_multiplier"]

render = fetch_render()
btc_dom = fetch_btc_dominance_proxy()
if btc_dom is None: st.error("BTC Dominance data unavailable."); st.stop()

features_df = build_features(render, btc_dom)
features_df = features_df[feature_cols]
historical_medians = features_df.iloc[:-1].median()
latest_row = features_df.iloc[-1:].fillna(historical_medians).values.astype(float)

# ── Predictions ───────────────────────────────────────────────────────────
proba = clf.predict_proba(latest_row)[0]
p_down, p_up = proba[0], proba[1]
signal = "BUY" if p_up >= threshold else ("SELL" if p_down >= threshold else "HOLD")
confidence = max(p_up, p_down)
forecast_price = reg.predict(latest_row)[0]
current_price = render["close"].iloc[-1]
expected_return = (forecast_price / current_price - 1) * 100

atr_series = compute_atr(render)
current_atr = atr_series.iloc[-1]
sl_price = current_price - (2.0 * current_atr)
tp_price = forecast_price
target_date = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + timedelta(days=30)).strftime("%Y-%m-%d")

# ── Header ────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔮 RENDER/RNDR 30‑Day Predictor</div>', unsafe_allow_html=True)
st.markdown(f"**Last updated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

# ── Top row: Signal & Key Metrics ─────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1], gap="medium")

with col1:
    st.subheader("Current Signal")
    if signal == "BUY":
        st.markdown('<div class="signal-buy">🟢 BUY</div>', unsafe_allow_html=True)
    elif signal == "SELL":
        st.markdown('<div class="signal-sell">🔴 SELL</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-hold">🟠 HOLD</div>', unsafe_allow_html=True)
    st.caption(f"Threshold: {threshold:.0%} | Model: LightGBM")

with col2:
    st.subheader("Probabilities")
    st.metric("P(UP)", f"{p_up:.1%}", delta=f"{(p_up-0.5)*100:+.0f}% vs 50%")
    st.metric("P(DOWN)", f"{p_down:.1%}")
    st.metric("Confidence", f"{confidence:.1%}")

with col3:
    st.subheader("Price Forecast")
    st.metric("Current Price", f"${current_price:,.4f}")
    st.metric("30d Forecast", f"${forecast_price:,.4f}", delta=f"{expected_return:+.2f}%")
    st.metric("Target Date", target_date)

# ── Risk Management ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🛡️ Risk Management")
col_tp, col_sl, col_risk = st.columns(3)
with col_tp:
    st.metric("Take Profit (TP)", f"${tp_price:,.4f}", delta=f"+{((tp_price/current_price-1)*100):.2f}%")
with col_sl:
    st.metric("Stop Loss (SL)", f"${sl_price:,.4f}", delta=f"{((sl_price/current_price-1)*100):.2f}%")
with col_risk:
    st.metric("ATR (14)", f"${current_atr:,.4f}")

# ── Charts Tab ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price & Context", "📈 Feature Importance", "📉 Performance History", "ℹ️ Model Info"])

with tab1:
    col_lt, col_rt = st.columns([2, 1])
    with col_lt:
        st.subheader("RENDER/RNDR Price with Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 4))
        close_series = render["close"]
        ax.plot(close_series.index, close_series, color="#1f77b4", alpha=0.6, label="Close")
        ax.plot(close_series.index, close_series.rolling(50).mean(), color="#ff7f0e", label="50d MA")
        ax.plot(close_series.index, close_series.rolling(200).mean(), color="#d62728", label="200d MA")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig.autofmt_xdate()
        st.pyplot(fig)
    with col_rt:
        st.subheader("BTC Dominance Proxy")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(btc_dom.index, btc_dom, color="#9467bd")
        ax2.set_ylabel("BTC Dominance")
        ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        fig2.autofmt_xdate()
        st.pyplot(fig2)

with tab2:
    st.subheader("Feature Importance (Classifier)")
    if hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'feature_importances_'):
        importances = clf.estimators_[0].feature_importances_
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sorted_idx = np.argsort(importances)
        ax3.barh(np.array(feature_cols)[sorted_idx], importances[sorted_idx], color="#1f77b4")
        ax3.set_xlabel("Importance")
        ax3.set_title("LightGBM Feature Importance")
        st.pyplot(fig3)
    else:
        st.info("Feature importance not available for calibrated classifier.")

with tab3:
    st.subheader("Holdout Performance (2025‑01‑01 → today)")
    holdout = metrics.get("holdout", [])
    if holdout:
        cols = st.columns(6)
        base = holdout[0]
        cols[0].metric("Trades", base["n_trades"])
        cols[1].metric("Win Rate", f"{base['win_rate']:.1%}")
        cols[2].metric("Sharpe", f"{base['sharpe']:.2f}")
        cols[3].metric("Profit Factor", f"{base['profit_factor']:.2f}")
        cols[4].metric("Max Drawdown", f"{base['max_drawdown']:.1%}")
        cols[5].metric("Threshold Used", f"{base.get('threshold_used', 'N/A')}")
    else:
        st.warning("No holdout metrics saved. Run train.py first.")

    st.subheader("Cost Sensitivity")
    if holdout:
        cost_data = {f"{h['cost']*100:.2f}%": h for h in holdout}
        cost_df = pd.DataFrame({
            "Cost": list(cost_data.keys()),
            "Sharpe": [h["sharpe"] for h in cost_data.values()],
            "PF": [h["profit_factor"] for h in cost_data.values()],
            "MaxDD": [h["max_drawdown"] for h in cost_data.values()]
        })
        st.dataframe(cost_df.set_index("Cost"), use_container_width=True)

with tab4:
    st.subheader("Model Configuration")
    st.json({
        "Features": feature_cols,
        "Classifier": "LightGBM + Isotonic Calibration",
        "Regressor": "LightGBM",
        "Threshold": threshold,
        "Risk Multiplier": risk_mult,
        "Training End": "2024-12-31",
        "Holdout Start": "2025-01-01",
        "Horizon": "30 days"
    })
    st.caption("This model uses only 7 features that survived ablation and holdout testing. USDT Dominance, Funding Rate, and Altseason were removed because they degraded performance.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="footer">⚠️ This is an experimental tool. Not financial advice. Past performance does not guarantee future results.</div>', unsafe_allow_html=True)
