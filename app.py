"""
================================================================================
  RENDER AI TRADING DASHBOARD  -  app.py
  Live RENDERUSDT data from Binance | Connected to predict.py PredictAPI
  Flask backend + Premium dark UI with TradingView BINANCE:RENDERUSDT widget
  USD + INR live prices | All indicators live | Crosshair cursor
================================================================================

SETUP:
    pip install flask flask-cors requests

RUN:
    python app.py

    Place predict.py and train.py in the same directory.
    After running train.py, place model output at /tmp/crypto_train_output/
================================================================================
"""

import os, sys, json, time, math, threading, traceback, logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

# ── Optional predict.py integration ─────────────────────────────────────────
PREDICT_API = None
PREDICT_READY = False

def _load_predict_api():
    global PREDICT_API, PREDICT_READY
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from predict import PredictAPI
        PREDICT_API = PredictAPI(
            train_output_dir="/tmp/crypto_train_output",
            predict_output_dir="/tmp/crypto_predict_output",
            symbol="RENDER",
            timeframe="1d",
        )
        PREDICT_API.initialize()
        PREDICT_READY = True
        print("[OK] predict.py PredictAPI loaded successfully")
    except Exception as e:
        PREDICT_READY = False
        print(f"[WARN] predict.py not loaded (train first): {e}")

threading.Thread(target=_load_predict_api, daemon=True).start()

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Live constants ───────────────────────────────────────────────────────────
BINANCE_BASE   = "https://api.binance.com/api/v3"
BINANCE_FBASE  = "https://fapi.binance.com/fapi/v1"
SYMBOL         = "RENDERUSDT"       # ← CURRENT symbol after rebrand
TV_SYMBOL      = "BINANCE:RENDERUSDT"
COIN_NAME      = "Render"
COIN_TICKER    = "RENDER"
USD_TO_INR_URL = "https://api.exchangerate-api.com/v4/latest/USD"
FNG_URL        = "https://api.alternative.me/fng/?limit=1&format=json"
CG_GLOBAL      = "https://api.coingecko.com/api/v3/global"

_cache = {}   # simple in-memory cache {key: (data, timestamp)}
CACHE_TTL = 15  # seconds

def _cached_get(url, params=None, ttl=CACHE_TTL):
    key = url + str(params)
    now = time.time()
    if key in _cache and now - _cache[key][1] < ttl:
        return _cache[key][0]
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            _cache[key] = (data, now)
            return data
    except Exception as e:
        logging.warning(f"Request failed {url}: {e}")
    return _cache.get(key, (None, 0))[0]  # return stale if available

def _usd_to_inr():
    data = _cached_get(USD_TO_INR_URL, ttl=300)
    try:
        return float(data["rates"]["INR"])
    except Exception:
        return 83.5  # fallback

# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES  – all return JSON
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/ticker")
def api_ticker():
    """Live 24h ticker from Binance for RENDERUSDT"""
    data = _cached_get(f"{BINANCE_BASE}/ticker/24hr", {"symbol": SYMBOL})
    if not data:
        return jsonify({"error": "Binance unreachable"}), 502

    inr_rate = _usd_to_inr()
    price_usd = float(data.get("lastPrice", 0))
    price_inr = round(price_usd * inr_rate, 4)

    return jsonify({
        "symbol":        COIN_TICKER,
        "name":          COIN_NAME,
        "price_usd":     price_usd,
        "price_inr":     price_inr,
        "inr_rate":      inr_rate,
        "change_pct":    float(data.get("priceChangePercent", 0)),
        "change_abs":    float(data.get("priceChange", 0)),
        "high_24h":      float(data.get("highPrice", 0)),
        "low_24h":       float(data.get("lowPrice", 0)),
        "volume":        float(data.get("volume", 0)),
        "quote_volume":  float(data.get("quoteVolume", 0)),
        "open_price":    float(data.get("openPrice", 0)),
        "prev_close":    float(data.get("prevClosePrice", 0)),
        "trades_count":  int(data.get("count", 0)),
        "bid":           float(data.get("bidPrice", 0)),
        "ask":           float(data.get("askPrice", 0)),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/klines")
def api_klines():
    """OHLCV candlestick data for RENDERUSDT"""
    interval = request.args.get("interval", "1d")
    limit    = int(request.args.get("limit", 100))
    data     = _cached_get(f"{BINANCE_BASE}/klines",
                            {"symbol": SYMBOL, "interval": interval,
                             "limit": min(limit, 500)}, ttl=30)
    if not data:
        return jsonify({"error": "No kline data"}), 502

    candles = []
    for k in data:
        candles.append({
            "time":   int(k[0]) // 1000,
            "open":   float(k[1]),
            "high":   float(k[2]),
            "low":    float(k[3]),
            "close":  float(k[4]),
            "volume": float(k[5]),
        })
    return jsonify({"symbol": SYMBOL, "interval": interval, "candles": candles})


@app.route("/api/orderbook")
def api_orderbook():
    """Order book depth for RENDERUSDT"""
    data = _cached_get(f"{BINANCE_BASE}/depth", {"symbol": SYMBOL, "limit": 20}, ttl=5)
    if not data:
        return jsonify({"error": "No depth data"}), 502

    import numpy as np
    bids = [[float(b[0]), float(b[1])] for b in data.get("bids", [])]
    asks = [[float(a[0]), float(a[1])] for a in data.get("asks", [])]
    bid_vol = sum(b[1] for b in bids)
    ask_vol = sum(a[1] for a in asks)
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    return jsonify({
        "bids":       bids[:10],
        "asks":       asks[:10],
        "bid_volume": round(bid_vol, 4),
        "ask_volume": round(ask_vol, 4),
        "imbalance":  round(imbalance, 4),
        "spread":     round(asks[0][0] - bids[0][0], 6) if bids and asks else 0,
        "best_bid":   bids[0][0] if bids else 0,
        "best_ask":   asks[0][0] if asks else 0,
    })


@app.route("/api/funding")
def api_funding():
    """Futures funding rate for RENDERUSDT"""
    data = _cached_get(f"{BINANCE_FBASE}/premiumIndex", {"symbol": SYMBOL}, ttl=60)
    oi   = _cached_get(f"{BINANCE_FBASE}/openInterest", {"symbol": SYMBOL}, ttl=60)
    ls   = _cached_get(f"{BINANCE_FBASE}/globalLongShortAccountRatio",
                        {"symbol": SYMBOL, "period": "1h", "limit": 1}, ttl=60)

    funding = 0.0
    mark_price = 0.0
    if data:
        funding    = float(data.get("lastFundingRate", 0))
        mark_price = float(data.get("markPrice", 0))

    open_interest = 0.0
    if oi:
        open_interest = float(oi.get("openInterest", 0))

    long_pct  = 0.5
    short_pct = 0.5
    ls_ratio  = 1.0
    if ls and isinstance(ls, list) and ls:
        long_pct  = float(ls[0].get("longAccount", 0.5))
        short_pct = float(ls[0].get("shortAccount", 0.5))
        ls_ratio  = float(ls[0].get("longShortRatio", 1.0))

    return jsonify({
        "funding_rate":   round(funding * 100, 6),
        "mark_price":     mark_price,
        "open_interest":  round(open_interest, 2),
        "long_pct":       round(long_pct * 100, 2),
        "short_pct":      round(short_pct * 100, 2),
        "long_short_ratio": round(ls_ratio, 4),
    })


@app.route("/api/feargreed")
def api_feargreed():
    """Fear & Greed index from alternative.me"""
    data = _cached_get(FNG_URL, ttl=3600)
    if data and "data" in data and data["data"]:
        item = data["data"][0]
        return jsonify({
            "value":             int(item["value"]),
            "classification":    item["value_classification"],
            "timestamp":         item["timestamp"],
        })
    return jsonify({"value": 50, "classification": "Neutral", "timestamp": ""})


@app.route("/api/indicators")
def api_indicators():
    """Compute live technical indicators from RENDERUSDT klines"""
    interval = request.args.get("interval", "1d")
    data = _cached_get(f"{BINANCE_BASE}/klines",
                        {"symbol": SYMBOL, "interval": interval, "limit": 200}, ttl=30)
    if not data or len(data) < 30:
        return jsonify({"error": "Insufficient data"}), 502

    closes  = [float(k[4]) for k in data]
    highs   = [float(k[2]) for k in data]
    lows    = [float(k[3]) for k in data]
    volumes = [float(k[5]) for k in data]

    def ema(prices, period):
        k_ema = 2 / (period + 1)
        e = prices[0]
        result = [e]
        for p in prices[1:]:
            e = p * k_ema + e * (1 - k_ema)
            result.append(e)
        return result

    def sma(prices, period):
        return [sum(prices[max(0,i-period+1):i+1])/min(i+1,period)
                for i in range(len(prices))]

    def rsi(prices, period=14):
        gains, losses = [], []
        for i in range(1, len(prices)):
            d = prices[i] - prices[i-1]
            gains.append(max(d, 0))
            losses.append(max(-d, 0))
        if len(gains) < period:
            return 50.0
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses[:period]) / period
        for i in range(period, len(gains)):
            avg_g = (avg_g * (period-1) + gains[i]) / period
            avg_l = (avg_l * (period-1) + losses[i]) / period
        if avg_l == 0:
            return 100.0
        rs = avg_g / avg_l
        return round(100 - (100 / (1 + rs)), 2)

    ema12  = ema(closes, 12)
    ema26  = ema(closes, 26)
    macd_line   = [ema12[i] - ema26[i] for i in range(len(closes))]
    signal_line = ema(macd_line, 9)
    macd_hist   = [macd_line[i] - signal_line[i] for i in range(len(closes))]

    ema9   = ema(closes, 9)
    ema21  = ema(closes, 21)
    ema50  = ema(closes, 50)
    ema200 = ema(closes, 200)
    sma20  = sma(closes, 20)

    # Bollinger Bands
    bb_period = 20
    bb_std    = 2.0
    bb_upper, bb_lower, bb_mid = [], [], []
    for i in range(len(closes)):
        window = closes[max(0, i-bb_period+1):i+1]
        m = sum(window) / len(window)
        s = (sum((x-m)**2 for x in window) / len(window)) ** 0.5
        bb_mid.append(m)
        bb_upper.append(m + bb_std * s)
        bb_lower.append(m - bb_std * s)

    # VWAP (approximate daily)
    vwap_vals = []
    cum_pv = cum_v = 0
    for i, k in enumerate(data):
        tp  = (float(k[2]) + float(k[3]) + float(k[4])) / 3
        vol = float(k[5])
        cum_pv += tp * vol
        cum_v  += vol
        vwap_vals.append(cum_pv / cum_v if cum_v > 0 else tp)

    # ATR
    atr_vals = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i],
                 abs(highs[i]-closes[i-1]),
                 abs(lows[i]-closes[i-1]))
        atr_vals.append(tr)
    atr14 = sum(atr_vals[-14:]) / 14 if len(atr_vals) >= 14 else 0

    # Volume analysis
    vol_sma20  = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else 0
    vol_ratio  = volumes[-1] / vol_sma20 if vol_sma20 > 0 else 1.0
    inflow     = sum(float(k[9]) for k in data[-20:])   # taker buy base
    total_vol  = sum(float(k[5]) for k in data[-20:])
    buy_ratio  = inflow / total_vol if total_vol > 0 else 0.5
    outflow    = total_vol - inflow

    cur = closes[-1]

    # Stochastic RSI (simplified)
    rsi14 = rsi(closes, 14)
    rsi_period_vals = [rsi(closes[:i], 14) for i in range(30, len(closes)+1)]
    if len(rsi_period_vals) >= 14:
        rsi_min = min(rsi_period_vals[-14:])
        rsi_max = max(rsi_period_vals[-14:])
        stoch_rsi = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-9) * 100
    else:
        stoch_rsi = 50.0

    # Williams %R
    wr_period = 14
    wr_high = max(highs[-wr_period:])
    wr_low  = min(lows[-wr_period:])
    willr   = -100 * (wr_high - cur) / (wr_high - wr_low + 1e-9)

    # CCI
    cci_period = 20
    tp_vals = [(highs[i]+lows[i]+closes[i])/3 for i in range(len(closes))]
    tp_sma  = sum(tp_vals[-cci_period:]) / cci_period
    md      = sum(abs(tp_vals[-cci_period+i]-tp_sma)
                  for i in range(cci_period)) / cci_period
    cci     = (tp_vals[-1] - tp_sma) / (0.015 * md + 1e-9)

    # Momentum
    mom10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if len(closes) > 11 else 0
    mom5  = (closes[-1] - closes[-6])  / closes[-6]  * 100 if len(closes) > 6  else 0

    # BB %B
    bb_pct_b = (cur - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-9) * 100

    # Signal determination
    signals = []
    if rsi14 < 30:          signals.append(("BUY",  "RSI Oversold"))
    elif rsi14 > 70:        signals.append(("SELL", "RSI Overbought"))
    if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
        signals.append(("BUY", "MACD Crossover"))
    elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
        signals.append(("SELL", "MACD Crossunder"))
    if cur > ema200[-1]:    signals.append(("BUY",  "Above EMA200"))
    else:                   signals.append(("SELL", "Below EMA200"))
    if cur > bb_upper[-1]:  signals.append(("SELL", "Above BB Upper"))
    elif cur < bb_lower[-1]:signals.append(("BUY",  "Below BB Lower"))
    if cur > vwap_vals[-1]: signals.append(("BUY",  "Above VWAP"))
    else:                   signals.append(("SELL", "Below VWAP"))

    buy_signals  = [s for s in signals if s[0] == "BUY"]
    sell_signals = [s for s in signals if s[0] == "SELL"]
    overall = "BUY" if len(buy_signals) > len(sell_signals) else \
              ("SELL" if len(sell_signals) > len(buy_signals) else "NEUTRAL")

    return jsonify({
        "current_price":  cur,
        "rsi":            rsi14,
        "stoch_rsi":      round(stoch_rsi, 2),
        "macd":           round(macd_line[-1], 6),
        "macd_signal":    round(signal_line[-1], 6),
        "macd_hist":      round(macd_hist[-1], 6),
        "ema9":           round(ema9[-1], 4),
        "ema21":          round(ema21[-1], 4),
        "ema50":          round(ema50[-1], 4),
        "ema200":         round(ema200[-1], 4),
        "sma20":          round(sma20[-1], 4),
        "bb_upper":       round(bb_upper[-1], 4),
        "bb_mid":         round(bb_mid[-1], 4),
        "bb_lower":       round(bb_lower[-1], 4),
        "bb_pct_b":       round(bb_pct_b, 2),
        "vwap":           round(vwap_vals[-1], 4),
        "atr14":          round(atr14, 6),
        "williams_r":     round(willr, 2),
        "cci":            round(cci, 2),
        "momentum_10":    round(mom10, 4),
        "momentum_5":     round(mom5, 4),
        "volume_current": volumes[-1],
        "volume_sma20":   round(vol_sma20, 2),
        "volume_ratio":   round(vol_ratio, 3),
        "inflow_20":      round(inflow, 2),
        "outflow_20":     round(outflow, 2),
        "buy_ratio":      round(buy_ratio * 100, 2),
        "signals":        signals,
        "buy_count":      len(buy_signals),
        "sell_count":     len(sell_signals),
        "overall_signal": overall,
    })


@app.route("/api/global")
def api_global():
    """BTC dominance, total market cap from CoinGecko"""
    data = _cached_get(CG_GLOBAL, ttl=300)
    if not data:
        return jsonify({"btc_dominance": 50.0, "total_market_cap_usd": 0})
    gd  = data.get("data", {})
    dom = gd.get("market_cap_percentage", {})
    return jsonify({
        "btc_dominance":       round(dom.get("btc", 50), 2),
        "eth_dominance":       round(dom.get("eth", 20), 2),
        "total_market_cap_usd": gd.get("total_market_cap", {}).get("usd", 0),
        "total_volume_24h_usd": gd.get("total_volume", {}).get("usd", 0),
        "active_cryptos":       gd.get("active_cryptocurrencies", 0),
    })


@app.route("/api/prediction")
def api_prediction():
    """Run live prediction via predict.py PredictAPI (if models trained)"""
    if not PREDICT_READY or PREDICT_API is None:
        # Return realistic mock so UI always shows something
        return _mock_prediction()

    try:
        pred_dict = PREDICT_API.predict_dict("RENDER", "1d")

        # Get live price for INR conversion
        inr_rate    = _usd_to_inr()
        pred_price  = pred_dict.get("predicted_price", 0)
        cur_price   = pred_dict.get("current_price", 0)

        return jsonify({
            "model_active":          True,
            "symbol":                "RENDER",
            "current_price_usd":     cur_price,
            "current_price_inr":     round(cur_price * inr_rate, 4),
            "predicted_price_usd":   pred_price,
            "predicted_price_inr":   round(pred_price * inr_rate, 4),
            "predicted_direction":   pred_dict.get("predicted_direction", "sideways"),
            "direction_confidence":  pred_dict.get("direction_confidence", 0.5),
            "prob_up":               pred_dict.get("prob_up", 0.33),
            "prob_down":             pred_dict.get("prob_down", 0.33),
            "prob_sideways":         pred_dict.get("prob_sideways", 0.34),
            "predicted_return_pct":  pred_dict.get("predicted_return_pct", 0),
            "price_lower_bound_usd": pred_dict.get("price_lower_bound", 0),
            "price_upper_bound_usd": pred_dict.get("price_upper_bound", 0),
            "price_lower_bound_inr": round(pred_dict.get("price_lower_bound", 0) * inr_rate, 4),
            "price_upper_bound_inr": round(pred_dict.get("price_upper_bound", 0) * inr_rate, 4),
            "signal":                pred_dict.get("signal", "HOLD"),
            "signal_score":          pred_dict.get("signal_score", 0),
            "entry_price":           pred_dict.get("entry_price", cur_price),
            "stop_loss":             pred_dict.get("stop_loss", 0),
            "take_profit":           pred_dict.get("take_profit", 0),
            "risk_reward":           pred_dict.get("risk_reward", 0),
            "market_regime":         pred_dict.get("market_regime", "UNKNOWN"),
            "regime_confidence":     pred_dict.get("regime_confidence", 0.5),
            "sentiment_score":       pred_dict.get("sentiment_score", 0),
            "fear_greed_index":      pred_dict.get("fear_greed_index", 50),
            "funding_rate":          pred_dict.get("funding_rate", 0),
            "btc_correlation":       pred_dict.get("btc_correlation", 0),
            "altseason_score":       pred_dict.get("altseason_score", 0),
            "n_models_used":         pred_dict.get("n_models_used", 0),
            "confidence_score":      pred_dict.get("direction_confidence", 0.5),
            "horizon_predictions":   pred_dict.get("horizon_predictions", []),
            "inr_rate":              inr_rate,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return _mock_prediction(error=str(e))


def _mock_prediction(error=None):
    """Return live price + placeholder AI fields when model not trained yet"""
    ticker = _cached_get(f"{BINANCE_BASE}/ticker/24hr", {"symbol": SYMBOL})
    inr_rate = _usd_to_inr()
    cur = float(ticker.get("lastPrice", 2.1)) if ticker else 2.1

    return jsonify({
        "model_active":          False,
        "error":                 error or "Train models first: python train.py",
        "symbol":                "RENDER",
        "current_price_usd":     cur,
        "current_price_inr":     round(cur * inr_rate, 4),
        "predicted_price_usd":   round(cur * 1.025, 4),
        "predicted_price_inr":   round(cur * 1.025 * inr_rate, 4),
        "predicted_direction":   "sideways",
        "direction_confidence":  0.52,
        "prob_up":               0.38,
        "prob_down":             0.28,
        "prob_sideways":         0.34,
        "predicted_return_pct":  2.5,
        "price_lower_bound_usd": round(cur * 0.95, 4),
        "price_upper_bound_usd": round(cur * 1.08, 4),
        "price_lower_bound_inr": round(cur * 0.95 * inr_rate, 4),
        "price_upper_bound_inr": round(cur * 1.08 * inr_rate, 4),
        "signal":                "HOLD",
        "signal_score":          0.12,
        "entry_price":           cur,
        "stop_loss":             round(cur * 0.95, 4),
        "take_profit":           round(cur * 1.10, 4),
        "risk_reward":           2.0,
        "market_regime":         "RANGING",
        "regime_confidence":     0.55,
        "sentiment_score":       0.1,
        "fear_greed_index":      50,
        "funding_rate":          0.01,
        "btc_correlation":       0.72,
        "altseason_score":       0.45,
        "n_models_used":         0,
        "confidence_score":      0.52,
        "horizon_predictions":   [],
        "inr_rate":              inr_rate,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HTML PAGE
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RENDER AI Terminal</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
/* ── RESET & BASE ──────────────────────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:        #080c14;
  --bg1:       #0d1220;
  --bg2:       #111827;
  --bg3:       #1a2235;
  --border:    #1e2d45;
  --border2:   #243350;
  --accent:    #e63946;
  --accent2:   #ff6b6b;
  --green:     #00d084;
  --green2:    #00ff9d;
  --red:       #ff3b5c;
  --red2:      #ff6b81;
  --blue:      #3b82f6;
  --blue2:     #60a5fa;
  --yellow:    #f59e0b;
  --purple:    #8b5cf6;
  --text:      #e2e8f0;
  --text2:     #94a3b8;
  --text3:     #64748b;
  --card-glow: rgba(230,57,70,0.08);
  --font:      'Space Grotesk', sans-serif;
  --mono:      'JetBrains Mono', monospace;
}
html{scroll-behavior:smooth;overflow-x:hidden}
body{
  font-family:var(--font);
  background:var(--bg);
  color:var(--text);
  cursor:crosshair;
  min-height:100vh;
  overflow-x:hidden;
}
/* scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:var(--bg1)}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* ── GRID LAYOUT ───────────────────────────────────────────────────────────── */
.layout{display:grid;grid-template-columns:56px 1fr;min-height:100vh}

/* ── SIDEBAR ────────────────────────────────────────────────────────────────── */
.sidebar{
  width:56px;background:var(--bg1);border-right:1px solid var(--border);
  display:flex;flex-direction:column;align-items:center;
  padding:12px 0;gap:4px;position:fixed;top:0;left:0;height:100vh;z-index:100;
}
.logo-box{
  width:36px;height:36px;background:var(--accent);border-radius:8px;
  display:flex;align-items:center;justify-content:center;margin-bottom:8px;
  position:relative;overflow:hidden;flex-shrink:0;
}
.logo-box svg{width:22px;height:22px}
.nav-btn{
  width:40px;height:40px;border-radius:8px;border:none;background:transparent;
  color:var(--text3);cursor:crosshair;display:flex;align-items:center;
  justify-content:center;transition:all 0.2s;font-size:16px;
}
.nav-btn:hover,.nav-btn.active{background:var(--bg3);color:var(--accent)}
.nav-btn svg{width:18px;height:18px;stroke:currentColor;fill:none;stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}

/* ── MAIN CONTENT ───────────────────────────────────────────────────────────── */
.main{
  margin-left:56px;display:flex;flex-direction:column;min-height:100vh;
}

/* ── TOPBAR ─────────────────────────────────────────────────────────────────── */
.topbar{
  height:52px;background:var(--bg1);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 20px;gap:16px;
  position:sticky;top:0;z-index:90;
}
.topbar-brand{
  font-size:14px;font-weight:600;color:var(--text);letter-spacing:.5px;
  display:flex;align-items:center;gap:8px;
}
.topbar-brand span{color:var(--accent)}
.live-dot{
  width:7px;height:7px;border-radius:50%;background:var(--green);
  animation:pulse 1.5s infinite;flex-shrink:0;
}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,208,132,.4)}
  50%{opacity:.8;box-shadow:0 0 0 5px rgba(0,208,132,0)}}
.topbar-price{
  display:flex;align-items:baseline;gap:6px;margin-left:auto;
}
.topbar-price .p-usd{font-family:var(--mono);font-size:16px;font-weight:600;color:var(--text)}
.topbar-price .p-inr{font-family:var(--mono);font-size:12px;color:var(--text2)}
.topbar-price .p-chg{font-family:var(--mono);font-size:12px;font-weight:600}
.up{color:var(--green)} .dn{color:var(--red)}
.topbar-stat{
  display:flex;flex-direction:column;align-items:flex-end;
  border-left:1px solid var(--border);padding-left:14px;
}
.topbar-stat .l{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.8px}
.topbar-stat .v{font-family:var(--mono);font-size:11px;color:var(--text2);margin-top:1px}
.timestamp{font-family:var(--mono);font-size:10px;color:var(--text3)}

/* ── CONTENT AREA ───────────────────────────────────────────────────────────── */
.content{padding:16px;display:flex;flex-direction:column;gap:14px;flex:1}

/* ── CARDS ──────────────────────────────────────────────────────────────────── */
.card{
  background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  overflow:hidden;transition:border-color .2s;
}
.card:hover{border-color:var(--border2)}
.card-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:11px 16px;border-bottom:1px solid var(--border);
}
.card-title{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.9px;color:var(--text2);display:flex;align-items:center;gap:7px;
}
.card-title .dot{width:6px;height:6px;border-radius:50%}
.dot-red{background:var(--accent)}
.dot-green{background:var(--green)}
.dot-blue{background:var(--blue)}
.dot-yellow{background:var(--yellow)}
.dot-purple{background:var(--purple)}
.card-body{padding:14px 16px}

/* ── GRID ROWS ──────────────────────────────────────────────────────────────── */
.row2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px}
.row4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}

/* ── TICKER ROW ─────────────────────────────────────────────────────────────── */
.ticker-row{
  display:grid;grid-template-columns:repeat(6,1fr);gap:1px;
  background:var(--border);border-radius:10px;overflow:hidden;
}
.ticker-cell{
  background:var(--bg2);padding:12px 14px;
  transition:background .2s;
}
.ticker-cell:hover{background:var(--bg3)}
.ticker-cell .label{font-size:9px;text-transform:uppercase;letter-spacing:.8px;color:var(--text3)}
.ticker-cell .value{font-family:var(--mono);font-size:14px;font-weight:600;color:var(--text);margin-top:3px}
.ticker-cell .sub{font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:2px}

/* ── TRADINGVIEW CHART ───────────────────────────────────────────────────────── */
.tv-container{
  width:100%;border-radius:0 0 10px 10px;overflow:hidden;
  background:#131722;
}
#tv-widget-container{height:440px}
@media(max-width:900px){#tv-widget-container{height:300px}}

/* ── PREDICTION CARDS ───────────────────────────────────────────────────────── */
.pred-direction{
  display:flex;align-items:center;justify-content:center;gap:10px;
  padding:18px 0;
}
.direction-badge{
  padding:8px 22px;border-radius:6px;font-size:16px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
  border:1.5px solid transparent;
}
.badge-bull{background:rgba(0,208,132,.12);color:var(--green);border-color:var(--green)}
.badge-bear{background:rgba(255,59,92,.12);color:var(--red);border-color:var(--red)}
.badge-side{background:rgba(245,158,11,.12);color:var(--yellow);border-color:var(--yellow)}

.prob-bars{display:flex;flex-direction:column;gap:7px;padding:0 2px}
.prob-row{display:flex;align-items:center;gap:8px;font-size:11px}
.prob-label{width:58px;color:var(--text2);font-size:10px;text-transform:uppercase;letter-spacing:.5px}
.prob-track{flex:1;height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}
.prob-fill{height:100%;border-radius:3px;transition:width .8s ease}
.prob-pct{width:36px;text-align:right;font-family:var(--mono);font-size:11px;font-weight:600}

/* ── CONFIDENCE GAUGE ───────────────────────────────────────────────────────── */
.gauge-wrap{display:flex;flex-direction:column;align-items:center;padding:8px 0}
.gauge-svg{overflow:visible}
.gauge-label{font-family:var(--mono);font-size:24px;font-weight:700;fill:var(--text)}
.gauge-sub{font-size:10px;fill:var(--text3);text-transform:uppercase;letter-spacing:.8px}

/* ── INDICATOR TABLE ─────────────────────────────────────────────────────────── */
.ind-table{display:flex;flex-direction:column;gap:0;overflow-y:auto;max-height:320px}
.ind-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 0;border-bottom:1px solid rgba(30,45,69,.5);
  transition:background .15s;
}
.ind-row:last-child{border-bottom:none}
.ind-row:hover{background:rgba(26,34,53,.4);margin:0 -4px;padding:8px 4px;border-radius:4px}
.ind-name{font-size:11px;color:var(--text2);min-width:90px}
.ind-value{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text)}
.ind-signal{
  font-size:9px;font-weight:700;letter-spacing:.6px;text-transform:uppercase;
  padding:2px 7px;border-radius:3px;
}
.sig-buy{background:rgba(0,208,132,.12);color:var(--green)}
.sig-sell{background:rgba(255,59,92,.12);color:var(--red)}
.sig-neutral{background:rgba(100,116,139,.12);color:var(--text3)}

/* ── SIGNAL CARDS ────────────────────────────────────────────────────────────── */
.signal-main{
  display:flex;flex-direction:column;align-items:center;gap:6px;padding:16px 0;
}
.signal-icon{font-size:36px;line-height:1}
.signal-text{font-size:18px;font-weight:700;letter-spacing:2px}
.signal-score-bar{
  width:100%;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden;
  margin-top:4px;
}
.signal-score-fill{height:100%;border-radius:2px;transition:width .8s}

.risk-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.risk-item{background:var(--bg3);border-radius:6px;padding:10px 12px}
.risk-label{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--text3)}
.risk-value{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--text);margin-top:3px}
.risk-sub{font-family:var(--mono);font-size:10px;color:var(--text2);margin-top:1px}

/* ── VOLUME ANALYTICS ────────────────────────────────────────────────────────── */
.flow-bars{display:flex;gap:6px;align-items:flex-end;height:60px;padding:0 2px}
.flow-bar{
  flex:1;border-radius:3px 3px 0 0;transition:height .5s ease;
  position:relative;min-height:4px;
}
.flow-bar-label{
  position:absolute;top:-16px;left:50%;transform:translateX(-50%);
  font-size:8px;color:var(--text3);white-space:nowrap;
}
.flow-meta{display:flex;justify-content:space-between;margin-top:12px}
.flow-meta-item{text-align:center}
.flow-meta-item .fm-label{font-size:9px;color:var(--text3);text-transform:uppercase;letter-spacing:.6px}
.flow-meta-item .fm-value{font-family:var(--mono);font-size:12px;font-weight:600;margin-top:2px}

/* ── MARKET SENTIMENT ────────────────────────────────────────────────────────── */
.sentiment-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.sent-card{background:var(--bg3);border-radius:6px;padding:11px 13px}
.sent-label{font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--text3)}
.sent-value{font-family:var(--mono);font-size:14px;font-weight:700;margin-top:4px}
.sent-sub{font-size:9px;color:var(--text3);margin-top:2px}

/* ── ON-CHAIN / FUTURES ──────────────────────────────────────────────────────── */
.stat-list{display:flex;flex-direction:column;gap:8px}
.stat-item{
  display:flex;justify-content:space-between;align-items:center;
  padding:8px 11px;background:var(--bg3);border-radius:6px;
}
.stat-item .si-label{font-size:10px;color:var(--text2)}
.stat-item .si-value{font-family:var(--mono);font-size:12px;font-weight:600;color:var(--text)}

/* ── CANDLESTICK MINI ────────────────────────────────────────────────────────── */
.mini-candles{
  display:flex;gap:2px;align-items:center;height:80px;
  padding:0 4px;overflow:hidden;
}
.candle-wrap{display:flex;flex-direction:column;align-items:center;flex:1;height:100%;justify-content:center;position:relative}
.candle-wick{width:1px;background:currentColor;position:absolute}
.candle-body{width:6px;border-radius:1px}

/* ── MODEL STATUS ────────────────────────────────────────────────────────────── */
.model-status{
  display:flex;align-items:center;gap:8px;padding:10px 14px;
  border-radius:6px;font-size:11px;
}
.ms-ready{background:rgba(0,208,132,.08);border:1px solid rgba(0,208,132,.2);color:var(--green)}
.ms-unready{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);color:var(--yellow)}
.ms-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.ms-green{background:var(--green);animation:pulse 2s infinite}
.ms-yellow{background:var(--yellow)}

/* ── HORIZON TABLE ───────────────────────────────────────────────────────────── */
.horizon-table{width:100%;border-collapse:collapse}
.horizon-table th{
  font-size:9px;text-transform:uppercase;letter-spacing:.7px;color:var(--text3);
  padding:6px 8px;text-align:left;border-bottom:1px solid var(--border);
}
.horizon-table td{
  padding:7px 8px;font-family:var(--mono);font-size:11px;
  border-bottom:1px solid rgba(30,45,69,.3);
}
.horizon-table tr:hover td{background:rgba(26,34,53,.5)}

/* ── LOADING SKELETON ────────────────────────────────────────────────────────── */
.skeleton{
  background:linear-gradient(90deg,var(--bg3) 25%,var(--bg2) 50%,var(--bg3) 75%);
  background-size:200% 100%;
  animation:shimmer 1.5s infinite;
  border-radius:4px;
}
@keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}

/* ── RESPONSIVE ──────────────────────────────────────────────────────────────── */
@media(max-width:1100px){.row4{grid-template-columns:1fr 1fr}.ticker-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:800px){
  .layout{grid-template-columns:1fr}
  .sidebar{display:none}
  .main{margin-left:0}
  .row2,.row3,.row4{grid-template-columns:1fr}
  .ticker-row{grid-template-columns:repeat(2,1fr)}
  .topbar-stat{display:none}
}
</style>
</head>
<body>

<!-- ── SIDEBAR ──────────────────────────────────────────────────────────────── -->
<div class="layout">
<nav class="sidebar">
  <div class="logo-box">
    <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" fill="white">
      <circle cx="50" cy="50" r="22"/>
      <path d="M50 50 m-35 0 a35 35 0 1 1 0 0.01" fill="none" stroke="white" stroke-width="7" stroke-linecap="round"/>
      <circle cx="15" cy="50" r="7" fill="white"/>
    </svg>
  </div>
  <button class="nav-btn active" title="Dashboard">
    <svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
  </button>
  <button class="nav-btn" title="Chart">
    <svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
  </button>
  <button class="nav-btn" title="Signals">
    <svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
  </button>
  <button class="nav-btn" title="Analytics">
    <svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
  </button>
  <button class="nav-btn" title="Settings" style="margin-top:auto">
    <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
  </button>
</nav>

<!-- ── MAIN ───────────────────────────────────────────────────────────────────── -->
<div class="main">

<!-- TOPBAR -->
<header class="topbar">
  <div class="topbar-brand">
    <div class="live-dot"></div>
    <span>RENDER</span> AI Terminal
  </div>
  <div style="display:flex;align-items:center;gap:6px;margin-left:16px">
    <div id="top-signal-badge" style="font-size:10px;font-weight:700;letter-spacing:.8px;padding:3px 9px;border-radius:4px;background:rgba(100,116,139,.15);color:var(--text3)">—</div>
  </div>
  <div class="topbar-price" style="margin-left:auto">
    <span class="p-usd" id="top-price-usd">$—</span>
    <span class="p-inr" id="top-price-inr">₹—</span>
    <span class="p-chg" id="top-price-chg">—%</span>
  </div>
  <div class="topbar-stat">
    <span class="l">24h Vol</span>
    <span class="v" id="top-vol">—</span>
  </div>
  <div class="topbar-stat">
    <span class="l">24h High</span>
    <span class="v" id="top-high">—</span>
  </div>
  <div class="topbar-stat">
    <span class="l">24h Low</span>
    <span class="v" id="top-low">—</span>
  </div>
  <div class="topbar-stat" style="border-left:1px solid var(--border);padding-left:14px">
    <span class="l">Updated</span>
    <span class="v timestamp" id="top-time">—</span>
  </div>
</header>

<!-- CONTENT -->
<div class="content">

  <!-- TICKER ROW -->
  <div class="ticker-row" id="ticker-row">
    <div class="ticker-cell"><div class="label">Price USD</div><div class="value" id="tk-usd">—</div><div class="sub" id="tk-chg">—</div></div>
    <div class="ticker-cell"><div class="label">Price INR</div><div class="value" id="tk-inr">—</div><div class="sub" id="tk-inrrate">—</div></div>
    <div class="ticker-cell"><div class="label">24h Volume</div><div class="value" id="tk-vol">—</div><div class="sub">RENDER</div></div>
    <div class="ticker-cell"><div class="label">Open Interest</div><div class="value" id="tk-oi">—</div><div class="sub">Futures USDT</div></div>
    <div class="ticker-cell"><div class="label">Funding Rate</div><div class="value" id="tk-fr">—</div><div class="sub">8h rate</div></div>
    <div class="ticker-cell"><div class="label">Fear & Greed</div><div class="value" id="tk-fg">—</div><div class="sub" id="tk-fg-label">—</div></div>
  </div>

  <!-- TRADINGVIEW CHART -->
  <div class="card">
    <div class="card-header">
      <div class="card-title"><span class="dot dot-red"></span>RENDER / USDT  ·  LIVE CHART</div>
      <div style="display:flex;gap:8px">
        <select id="tv-interval" style="background:var(--bg3);border:1px solid var(--border);color:var(--text2);padding:3px 8px;border-radius:4px;font-size:10px;cursor:crosshair">
          <option value="1">1m</option><option value="5">5m</option><option value="15">15m</option>
          <option value="60">1h</option><option value="240">4h</option>
          <option value="D" selected>1D</option><option value="W">1W</option>
        </select>
      </div>
    </div>
    <div class="tv-container">
      <div id="tv-widget-container"></div>
    </div>
  </div>

  <!-- PREDICTION + SIGNAL ROW -->
  <div class="row3">

    <!-- Classification -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-green"></span>Classification</div>
        <span style="font-size:9px;color:var(--text3)" id="clf-models">—</span>
      </div>
      <div class="card-body">
        <div class="pred-direction">
          <div class="direction-badge" id="clf-badge">LOADING…</div>
        </div>
        <div class="prob-bars">
          <div class="prob-row">
            <span class="prob-label">BULL</span>
            <div class="prob-track"><div class="prob-fill" id="pb-bull" style="width:33%;background:var(--green)"></div></div>
            <span class="prob-pct up" id="pv-bull">—</span>
          </div>
          <div class="prob-row">
            <span class="prob-label">BEAR</span>
            <div class="prob-track"><div class="prob-fill" id="pb-bear" style="width:33%;background:var(--red)"></div></div>
            <span class="prob-pct dn" id="pv-bear">—</span>
          </div>
          <div class="prob-row">
            <span class="prob-label">SIDE</span>
            <div class="prob-track"><div class="prob-fill" id="pb-side" style="width:34%;background:var(--yellow)"></div></div>
            <span class="prob-pct" style="color:var(--yellow)" id="pv-side">—</span>
          </div>
        </div>
        <div id="model-status-clf" style="margin-top:12px"></div>
      </div>
    </div>

    <!-- Regression -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-blue"></span>Price Prediction</div>
        <span style="font-size:9px;color:var(--text3)">Regression</span>
      </div>
      <div class="card-body">
        <div style="text-align:center;padding:8px 0 12px">
          <div style="font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.7px">Predicted Price</div>
          <div style="font-family:var(--mono);font-size:26px;font-weight:700;color:var(--text);margin:6px 0" id="pred-price-usd">$—</div>
          <div style="font-family:var(--mono);font-size:14px;color:var(--text2)" id="pred-price-inr">₹—</div>
          <div style="font-family:var(--mono);font-size:12px;margin-top:6px" id="pred-return-pct">—%</div>
        </div>
        <div style="display:flex;gap:10px">
          <div style="flex:1;background:rgba(255,59,92,.08);border:1px solid rgba(255,59,92,.2);border-radius:5px;padding:8px;text-align:center">
            <div style="font-size:9px;color:var(--text3)">Lower 95%</div>
            <div style="font-family:var(--mono);font-size:12px;color:var(--red);margin-top:3px" id="pred-lower">—</div>
          </div>
          <div style="flex:1;background:rgba(0,208,132,.08);border:1px solid rgba(0,208,132,.2);border-radius:5px;padding:8px;text-align:center">
            <div style="font-size:9px;color:var(--text3)">Upper 95%</div>
            <div style="font-family:var(--mono);font-size:12px;color:var(--green);margin-top:3px" id="pred-upper">—</div>
          </div>
        </div>
        <div style="margin-top:10px;font-family:var(--mono);font-size:10px;color:var(--text3);text-align:center" id="pred-regime">Regime: —</div>
      </div>
    </div>

    <!-- AI Confidence -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-purple"></span>AI Confidence</div>
        <span style="font-size:9px;color:var(--text3)" id="conf-models-label">Models: —</span>
      </div>
      <div class="card-body">
        <div class="gauge-wrap">
          <svg class="gauge-svg" width="160" height="100" viewBox="0 0 160 100">
            <defs>
              <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stop-color="#ff3b5c"/>
                <stop offset="50%" stop-color="#f59e0b"/>
                <stop offset="100%" stop-color="#00d084"/>
              </linearGradient>
            </defs>
            <!-- track -->
            <path d="M 20 90 A 60 60 0 0 1 140 90" fill="none" stroke="#1a2235" stroke-width="10" stroke-linecap="round"/>
            <!-- fill -->
            <path id="gauge-arc" d="M 20 90 A 60 60 0 0 1 140 90" fill="none" stroke="url(#gaugeGrad)" stroke-width="10" stroke-linecap="round"
              stroke-dasharray="188.5" stroke-dashoffset="94"/>
            <!-- needle -->
            <line id="gauge-needle" x1="80" y1="90" x2="80" y2="38" stroke="#e2e8f0" stroke-width="2" stroke-linecap="round"
              transform-origin="80 90" style="transform:rotate(-90deg);transform-box:fill-box;transition:transform .8s ease"/>
            <circle cx="80" cy="90" r="4" fill="#e2e8f0"/>
            <text id="gauge-pct" class="gauge-label" x="80" y="78" text-anchor="middle" font-family="JetBrains Mono" font-size="20" font-weight="700" fill="#e2e8f0">—</text>
            <text class="gauge-sub" x="80" y="100" text-anchor="middle" font-family="Space Grotesk" font-size="8" fill="#64748b">CONFIDENCE</text>
          </svg>
        </div>
        <div class="stat-list" style="margin-top:8px">
          <div class="stat-item"><span class="si-label">BTC Correlation</span><span class="si-value" id="conf-btccor">—</span></div>
          <div class="stat-item"><span class="si-label">Altseason Score</span><span class="si-value" id="conf-alt">—</span></div>
          <div class="stat-item"><span class="si-label">Data Quality</span><span class="si-value">LIVE</span></div>
        </div>
      </div>
    </div>
  </div>

  <!-- BUY/SELL SIGNALS + RISK -->
  <div class="row2">

    <!-- Signal -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-yellow"></span>Buy / Sell Signal</div>
        <span style="font-size:9px;color:var(--text3)" id="sig-count-label">—</span>
      </div>
      <div class="card-body">
        <div class="signal-main">
          <div class="signal-icon" id="sig-icon">⏳</div>
          <div class="signal-text" id="sig-text" style="color:var(--text2)">LOADING</div>
          <div style="font-size:11px;color:var(--text3)" id="sig-score-label">score: —</div>
          <div class="signal-score-bar" style="width:200px">
            <div class="signal-score-fill" id="sig-score-fill" style="width:50%;background:var(--text3)"></div>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px">
          <div style="background:rgba(0,208,132,.07);border:1px solid rgba(0,208,132,.15);border-radius:6px;padding:10px;text-align:center">
            <div style="font-size:9px;color:var(--text3);text-transform:uppercase">Buy Signals</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:var(--green);margin-top:2px" id="sig-buy-cnt">—</div>
          </div>
          <div style="background:rgba(255,59,92,.07);border:1px solid rgba(255,59,92,.15);border-radius:6px;padding:10px;text-align:center">
            <div style="font-size:9px;color:var(--text3);text-transform:uppercase">Sell Signals</div>
            <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:var(--red);margin-top:2px" id="sig-sell-cnt">—</div>
          </div>
        </div>
        <div style="margin-top:10px;display:flex;flex-direction:column;gap:4px;max-height:80px;overflow-y:auto" id="sig-list"></div>
      </div>
    </div>

    <!-- Risk Analysis -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-red"></span>Risk Analysis</div>
        <span style="font-size:9px;color:var(--text3)">AI Entry / Exit</span>
      </div>
      <div class="card-body">
        <div class="risk-grid">
          <div class="risk-item">
            <div class="risk-label">Entry Price</div>
            <div class="risk-value" id="risk-entry">—</div>
            <div class="risk-sub" id="risk-entry-inr">₹—</div>
          </div>
          <div class="risk-item">
            <div class="risk-label">Stop Loss</div>
            <div class="risk-value dn" id="risk-sl">—</div>
            <div class="risk-sub" id="risk-sl-inr">₹—</div>
          </div>
          <div class="risk-item">
            <div class="risk-label">Take Profit</div>
            <div class="risk-value up" id="risk-tp">—</div>
            <div class="risk-sub" id="risk-tp-inr">₹—</div>
          </div>
          <div class="risk-item">
            <div class="risk-label">Risk / Reward</div>
            <div class="risk-value" id="risk-rr">—</div>
            <div class="risk-sub">ratio</div>
          </div>
        </div>
        <div style="margin-top:10px;background:var(--bg3);border-radius:6px;padding:11px 13px">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-size:10px;color:var(--text2)">Market Regime</span>
            <span style="font-family:var(--mono);font-size:11px;font-weight:600;color:var(--blue)" id="risk-regime">—</span>
          </div>
          <div style="display:flex;justify-content:space-between;align-items:center;margin-top:7px">
            <span style="font-size:10px;color:var(--text2)">Regime Confidence</span>
            <span style="font-family:var(--mono);font-size:11px;color:var(--text)" id="risk-regime-conf">—</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TECHNICAL INDICATORS -->
  <div class="row2">
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-blue"></span>Technical Indicators</div>
        <span style="font-size:9px;color:var(--text3)">Scrollable ↕</span>
      </div>
      <div class="card-body" style="padding:8px 12px">
        <div class="ind-table" id="ind-table">
          <!-- filled by JS -->
        </div>
      </div>
    </div>

    <!-- Order Book -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-purple"></span>Order Book Depth</div>
        <span style="font-size:9px;color:var(--text3)" id="ob-spread-label">Spread: —</span>
      </div>
      <div class="card-body" style="padding:8px 12px">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0">
          <div>
            <div style="font-size:9px;color:var(--green);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px">Bids</div>
            <div id="ob-bids" style="display:flex;flex-direction:column;gap:3px"></div>
          </div>
          <div>
            <div style="font-size:9px;color:var(--red);text-transform:uppercase;letter-spacing:.7px;margin-bottom:6px;text-align:right">Asks</div>
            <div id="ob-asks" style="display:flex;flex-direction:column;gap:3px"></div>
          </div>
        </div>
        <div style="margin-top:12px;background:var(--bg3);border-radius:6px;padding:10px">
          <div style="display:flex;justify-content:space-between;margin-bottom:6px">
            <span style="font-size:10px;color:var(--green)">Buy Pressure</span>
            <span style="font-size:10px;color:var(--red)">Sell Pressure</span>
          </div>
          <div style="height:6px;background:var(--bg2);border-radius:3px;overflow:hidden;display:flex">
            <div id="ob-bid-bar" style="height:100%;background:var(--green);transition:width .5s"></div>
            <div id="ob-ask-bar" style="height:100%;background:var(--red);transition:width .5s"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:5px">
            <span style="font-family:var(--mono);font-size:10px;color:var(--green)" id="ob-bid-vol">—</span>
            <span style="font-family:var(--mono);font-size:10px;color:var(--text3)" id="ob-imbalance">—</span>
            <span style="font-family:var(--mono);font-size:10px;color:var(--red)" id="ob-ask-vol">—</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- VOLUME ANALYTICS + MARKET SENTIMENT -->
  <div class="row2">

    <!-- Volume Analytics -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-green"></span>Volume Analytics</div>
        <span style="font-size:9px;color:var(--text3)">Inflow / Outflow</span>
      </div>
      <div class="card-body">
        <div class="flow-bars" id="vol-bars">
          <!-- dynamically filled -->
        </div>
        <div class="flow-meta">
          <div class="flow-meta-item">
            <div class="fm-label">Inflow</div>
            <div class="fm-value up" id="vol-inflow">—</div>
          </div>
          <div class="flow-meta-item">
            <div class="fm-label">Outflow</div>
            <div class="fm-value dn" id="vol-outflow">—</div>
          </div>
          <div class="flow-meta-item">
            <div class="fm-label">Buy Ratio</div>
            <div class="fm-value" id="vol-buyratio">—</div>
          </div>
          <div class="flow-meta-item">
            <div class="fm-label">Vol/Avg</div>
            <div class="fm-value" id="vol-ratio">—</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Market Sentiment -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-yellow"></span>Market Sentiment</div>
        <span style="font-size:9px;color:var(--text3)">Multi-source</span>
      </div>
      <div class="card-body">
        <div class="sentiment-grid">
          <div class="sent-card">
            <div class="sent-label">Fear & Greed</div>
            <div class="sent-value" id="sent-fg">—</div>
            <div class="sent-sub" id="sent-fg-cls">—</div>
          </div>
          <div class="sent-card">
            <div class="sent-label">AI Sentiment</div>
            <div class="sent-value" id="sent-ai">—</div>
            <div class="sent-sub">model output</div>
          </div>
          <div class="sent-card">
            <div class="sent-label">Long Ratio</div>
            <div class="sent-value up" id="sent-long">—</div>
            <div class="sent-sub">futures longs</div>
          </div>
          <div class="sent-card">
            <div class="sent-label">Short Ratio</div>
            <div class="sent-value dn" id="sent-short">—</div>
            <div class="sent-sub">futures shorts</div>
          </div>
          <div class="sent-card">
            <div class="sent-label">BTC Dominance</div>
            <div class="sent-value" id="sent-btcdom">—</div>
            <div class="sent-sub">market share</div>
          </div>
          <div class="sent-card">
            <div class="sent-label">Funding Rate</div>
            <div class="sent-value" id="sent-fr">—</div>
            <div class="sent-sub">8h perpetual</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- CANDLESTICK ANALYSIS + MULTI-HORIZON -->
  <div class="row2">

    <!-- Candlestick Mini Analysis -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-red"></span>Candlestick Analysis</div>
        <span style="font-size:9px;color:var(--text3)">Last 30 candles</span>
      </div>
      <div class="card-body">
        <div class="mini-candles" id="mini-candles"></div>
        <div class="stat-list" style="margin-top:14px">
          <div class="stat-item"><span class="si-label">ATR (14)</span><span class="si-value" id="cs-atr">—</span></div>
          <div class="stat-item"><span class="si-label">BB Width</span><span class="si-value" id="cs-bbw">—</span></div>
          <div class="stat-item"><span class="si-label">Momentum (10)</span><span class="si-value" id="cs-mom">—</span></div>
          <div class="stat-item"><span class="si-label">VWAP</span><span class="si-value" id="cs-vwap">—</span></div>
          <div class="stat-item"><span class="si-label">Price vs EMA200</span><span class="si-value" id="cs-ema200delta">—</span></div>
        </div>
      </div>
    </div>

    <!-- Multi-Horizon Predictions -->
    <div class="card">
      <div class="card-header">
        <div class="card-title"><span class="dot dot-blue"></span>Multi-Horizon Forecast</div>
        <span style="font-size:9px;color:var(--text3)">Next N steps</span>
      </div>
      <div class="card-body" style="overflow-x:auto">
        <table class="horizon-table" id="horizon-table">
          <thead>
            <tr>
              <th>Step</th><th>Pred Price USD</th><th>Pred Price INR</th><th>Return %</th><th>Direction</th>
            </tr>
          </thead>
          <tbody id="horizon-tbody">
            <tr><td colspan="5" style="color:var(--text3);text-align:center;padding:20px">
              Train models to see multi-horizon forecast
            </td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- FOOTER -->
  <div style="text-align:center;padding:20px;color:var(--text3);font-size:10px;letter-spacing:.5px;border-top:1px solid var(--border)">
    RENDER AI TERMINAL · Live data: Binance RENDERUSDT · Predictions: predict.py ensemble
    · <span id="footer-time"></span>
  </div>

</div><!-- /content -->
</div><!-- /main -->
</div><!-- /layout -->

<!-- ── TRADINGVIEW WIDGET SCRIPT ────────────────────────────────────────────── -->
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
// ─── TradingView Widget ──────────────────────────────────────────────────────
let tvWidget = null;
function initTV(interval) {
  if(tvWidget){ try{ tvWidget.remove() }catch(e){} }
  tvWidget = new TradingView.widget({
    container_id: "tv-widget-container",
    autosize:     true,
    symbol:       "BINANCE:RENDERUSDT",
    interval:     interval || "D",
    timezone:     "Asia/Kolkata",
    theme:        "dark",
    style:        "1",
    locale:       "en",
    toolbar_bg:   "#080c14",
    enable_publishing: false,
    hide_top_toolbar: false,
    hide_legend:  false,
    withdateranges: true,
    allow_symbol_change: false,
    save_image:   false,
    backgroundColor: "#080c14",
    gridColor:    "rgba(30,45,69,0.4)",
    studies: [
      "RSI@tv-basicstudies",
      "MACD@tv-basicstudies",
      "BB@tv-basicstudies",
    ],
    overrides: {
      "paneProperties.background": "#080c14",
      "paneProperties.backgroundType": "solid",
      "scalesProperties.textColor": "#94a3b8",
    },
    loading_screen: { backgroundColor: "#080c14", foregroundColor: "#e63946" },
  });
}
initTV("D");
document.getElementById("tv-interval").addEventListener("change", function(){
  initTV(this.value);
});

// ─── Helpers ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
function fmt(n, d=4){ return n == null || isNaN(n) ? "—" : Number(n).toFixed(d) }
function fmtM(n){ // millions / billions
  if(!n || isNaN(n)) return "—";
  if(n >= 1e9) return (n/1e9).toFixed(2)+"B";
  if(n >= 1e6) return (n/1e6).toFixed(2)+"M";
  if(n >= 1e3) return (n/1e3).toFixed(1)+"K";
  return n.toFixed(2);
}
function fmtInr(n){ return n ? "₹"+ Number(n).toLocaleString("en-IN",{maximumFractionDigits:2}) : "₹—" }
function pct(n){ return n == null ? "—" : (n>0?"+":"")+Number(n).toFixed(2)+"%" }
function clr(n){ return n >= 0 ? "var(--green)" : "var(--red)" }

// ─── State ───────────────────────────────────────────────────────────────────
let state = { ticker:{}, indicators:{}, prediction:{}, funding:{}, fg:{}, global:{}, orderbook:{}, klines:[] };
let inrRate = 83.5;

// ─── Update Ticker ───────────────────────────────────────────────────────────
async function loadTicker(){
  try{
    const d = await fetch("/api/ticker").then(r=>r.json());
    state.ticker = d;
    inrRate = d.inr_rate || 83.5;
    const c = d.change_pct || 0;
    const chgCls = c >= 0 ? "up" : "dn";
    $("top-price-usd").textContent = "$"+ fmt(d.price_usd,4);
    $("top-price-inr").textContent = fmtInr(d.price_inr);
    $("top-price-chg").textContent = pct(c);
    $("top-price-chg").className = "p-chg " + chgCls;
    $("top-vol").textContent = fmtM(d.quote_volume) + " USDT";
    $("top-high").textContent = "$"+ fmt(d.high_24h,4);
    $("top-low").textContent  = "$"+ fmt(d.low_24h,4);
    $("top-time").textContent = new Date().toLocaleTimeString("en-IN");

    $("tk-usd").textContent   = "$"+ fmt(d.price_usd,4);
    $("tk-usd").style.color   = clr(c);
    $("tk-chg").textContent   = pct(c);
    $("tk-chg").className     = "sub " + chgCls;
    $("tk-inr").textContent   = fmtInr(d.price_inr);
    $("tk-inrrate").textContent = "1 USD = ₹"+ fmt(inrRate,2);
    $("tk-vol").textContent   = fmtM(d.volume);

    $("footer-time").textContent = "Updated " + new Date().toLocaleString("en-IN");
  }catch(e){ console.warn("ticker error",e) }
}

// ─── Update Funding ──────────────────────────────────────────────────────────
async function loadFunding(){
  try{
    const d = await fetch("/api/funding").then(r=>r.json());
    state.funding = d;
    const fr = d.funding_rate;
    $("tk-fr").textContent = (fr >= 0 ? "+" : "") + fmt(fr,4) + "%";
    $("tk-fr").style.color = fr >= 0 ? "var(--green)" : "var(--red)";
    $("tk-oi").textContent = fmtM(d.open_interest) + " RNDR";
    $("sent-long").textContent  = fmt(d.long_pct,1) + "%";
    $("sent-short").textContent = fmt(d.short_pct,1) + "%";
    $("sent-fr").textContent    = (fr>=0?"+":"")+fmt(fr,4)+"%";
    $("sent-fr").style.color    = fr>=0?"var(--green)":"var(--red)";
  }catch(e){ console.warn("funding error",e) }
}

// ─── Fear & Greed ────────────────────────────────────────────────────────────
async function loadFG(){
  try{
    const d = await fetch("/api/feargreed").then(r=>r.json());
    state.fg = d;
    const v = d.value;
    const col = v < 25 ? "var(--red)" : v < 45 ? "var(--red2)" :
                v < 55 ? "var(--yellow)" : v < 75 ? "var(--green)" : "var(--green2)";
    $("tk-fg").textContent = v;
    $("tk-fg").style.color = col;
    $("tk-fg-label").textContent = d.classification;
    $("sent-fg").textContent = v;
    $("sent-fg").style.color = col;
    $("sent-fg-cls").textContent = d.classification;
  }catch(e){ console.warn("fg error",e) }
}

// ─── Global ──────────────────────────────────────────────────────────────────
async function loadGlobal(){
  try{
    const d = await fetch("/api/global").then(r=>r.json());
    state.global = d;
    $("sent-btcdom").textContent = fmt(d.btc_dominance,2) + "%";
  }catch(e){ console.warn("global error",e) }
}

// ─── Order Book ──────────────────────────────────────────────────────────────
async function loadOrderBook(){
  try{
    const d = await fetch("/api/orderbook").then(r=>r.json());
    state.orderbook = d;
    $("ob-spread-label").textContent = "Spread: $"+ fmt(d.spread,5);

    // Bids
    const bidEl = $("ob-bids");
    bidEl.innerHTML = (d.bids||[]).slice(0,8).map(b=>
      `<div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;padding:2px 0">
        <span style="color:var(--green)">${fmt(b[0],4)}</span>
        <span style="color:var(--text3)">${fmt(b[1],2)}</span>
      </div>`).join("");

    // Asks
    const askEl = $("ob-asks");
    askEl.innerHTML = (d.asks||[]).slice(0,8).map(a=>
      `<div style="display:flex;justify-content:space-between;font-family:var(--mono);font-size:10px;padding:2px 0">
        <span style="color:var(--text3)">${fmt(a[1],2)}</span>
        <span style="color:var(--red)">${fmt(a[0],4)}</span>
      </div>`).join("");

    // Pressure bar
    const total = (d.bid_volume||0) + (d.ask_volume||0) + 0.001;
    const bidPct = (d.bid_volume / total * 100).toFixed(1);
    const askPct = (d.ask_volume / total * 100).toFixed(1);
    $("ob-bid-bar").style.width = bidPct + "%";
    $("ob-ask-bar").style.width = askPct + "%";
    $("ob-bid-vol").textContent = fmtM(d.bid_volume);
    $("ob-ask-vol").textContent = fmtM(d.ask_volume);
    $("ob-imbalance").textContent = "Imb: "+ fmt(d.imbalance,3);
    const imbColor = d.imbalance > 0 ? "var(--green)" : "var(--red)";
    $("ob-imbalance").style.color = imbColor;
  }catch(e){ console.warn("ob error",e) }
}

// ─── Indicators ──────────────────────────────────────────────────────────────
async function loadIndicators(){
  try{
    const d = await fetch("/api/indicators").then(r=>r.json());
    state.indicators = d;

    // Table
    const rows = [
      ["RSI (14)",      fmt(d.rsi,2),       rsiSignal(d.rsi)],
      ["Stoch RSI",     fmt(d.stoch_rsi,2), rsiSignal(d.stoch_rsi)],
      ["MACD",          fmt(d.macd,5),       d.macd > d.macd_signal ? "BUY" : "SELL"],
      ["MACD Signal",   fmt(d.macd_signal,5),"NEUTRAL"],
      ["MACD Hist",     fmt(d.macd_hist,5),  d.macd_hist > 0 ? "BUY" : "SELL"],
      ["EMA 9",         "$"+fmt(d.ema9,4),   d.current_price > d.ema9   ? "BUY" : "SELL"],
      ["EMA 21",        "$"+fmt(d.ema21,4),  d.current_price > d.ema21  ? "BUY" : "SELL"],
      ["EMA 50",        "$"+fmt(d.ema50,4),  d.current_price > d.ema50  ? "BUY" : "SELL"],
      ["EMA 200",       "$"+fmt(d.ema200,4), d.current_price > d.ema200 ? "BUY" : "SELL"],
      ["SMA 20",        "$"+fmt(d.sma20,4),  d.current_price > d.sma20  ? "BUY" : "SELL"],
      ["BB Upper",      "$"+fmt(d.bb_upper,4),d.current_price > d.bb_upper ? "SELL" : "NEUTRAL"],
      ["BB Mid",        "$"+fmt(d.bb_mid,4), "NEUTRAL"],
      ["BB Lower",      "$"+fmt(d.bb_lower,4),d.current_price < d.bb_lower ? "BUY" : "NEUTRAL"],
      ["BB %B",         fmt(d.bb_pct_b,1)+"%","NEUTRAL"],
      ["VWAP",          "$"+fmt(d.vwap,4),   d.current_price > d.vwap ? "BUY" : "SELL"],
      ["ATR (14)",      fmt(d.atr14,6),      "NEUTRAL"],
      ["Williams %R",   fmt(d.williams_r,2), d.williams_r < -80 ? "BUY" : d.williams_r > -20 ? "SELL" : "NEUTRAL"],
      ["CCI",           fmt(d.cci,2),         d.cci < -100 ? "BUY" : d.cci > 100 ? "SELL" : "NEUTRAL"],
      ["Momentum 10",   pct(d.momentum_10),  d.momentum_10 > 0 ? "BUY" : "SELL"],
      ["Momentum 5",    pct(d.momentum_5),   d.momentum_5 > 0 ? "BUY" : "SELL"],
      ["Vol Ratio",     fmt(d.volume_ratio,2)+"x", d.volume_ratio > 1.5 ? "BUY" : "NEUTRAL"],
      ["Buy Ratio",     fmt(d.buy_ratio,1)+"%", d.buy_ratio > 55 ? "BUY" : d.buy_ratio < 45 ? "SELL" : "NEUTRAL"],
    ];

    $("ind-table").innerHTML = rows.map(([name,val,sig])=>{
      const scls = sig === "BUY" ? "sig-buy" : sig === "SELL" ? "sig-sell" : "sig-neutral";
      return `<div class="ind-row">
        <span class="ind-name">${name}</span>
        <span class="ind-value">${val}</span>
        <span class="ind-signal ${scls}">${sig}</span>
      </div>`;
    }).join("");

    // Overall signal badge
    const os = d.overall_signal;
    const sIcon  = os === "BUY" ? "🟢" : os === "SELL" ? "🔴" : "🟡";
    const sColor = os === "BUY" ? "var(--green)" : os === "SELL" ? "var(--red)" : "var(--yellow)";
    $("sig-icon").textContent = sIcon;
    $("sig-text").textContent = os;
    $("sig-text").style.color = sColor;
    $("sig-count-label").textContent = `${d.buy_count}B / ${d.sell_count}S`;
    $("sig-buy-cnt").textContent  = d.buy_count;
    $("sig-sell-cnt").textContent = d.sell_count;

    // Score bar
    const scoreRaw = (d.buy_count - d.sell_count) / ((d.buy_count + d.sell_count) || 1);
    const scorePct  = ((scoreRaw + 1) / 2 * 100).toFixed(0);
    $("sig-score-label").textContent = "tech score: "+ scoreRaw.toFixed(2);
    $("sig-score-fill").style.width  = scorePct + "%";
    $("sig-score-fill").style.background = scoreRaw > 0 ? "var(--green)" : "var(--red)";

    // Top bar signal
    $("top-signal-badge").textContent = os;
    $("top-signal-badge").style.background = os==="BUY"?"rgba(0,208,132,.15)":os==="SELL"?"rgba(255,59,92,.15)":"rgba(245,158,11,.15)";
    $("top-signal-badge").style.color = sColor;

    // Signal list
    $("sig-list").innerHTML = (d.signals||[]).slice(0,8).map(s=>
      `<div style="display:flex;justify-content:space-between;font-size:10px;padding:3px 6px;border-radius:3px;background:var(--bg3)">
        <span style="color:${s[0]==='BUY'?'var(--green)':'var(--red)'}">${s[0]}</span>
        <span style="color:var(--text2)">${s[1]}</span>
      </div>`).join("");

    // Volume bars
    const volBars = $("vol-bars");
    volBars.innerHTML = "";
    const recentVols = [d.volume_current, d.volume_sma20];
    // Draw simple inflow/outflow bars from data
    const inMax = Math.max(d.inflow_20, d.outflow_20, 1);
    const barH_in  = Math.round((d.inflow_20  / inMax) * 55);
    const barH_out = Math.round((d.outflow_20 / inMax) * 55);
    volBars.innerHTML = `
      <div class="flow-bar" style="height:${barH_in}px;background:var(--green);position:relative">
        <span class="flow-bar-label">Inflow</span>
      </div>
      <div class="flow-bar" style="height:${barH_out}px;background:var(--red);position:relative">
        <span class="flow-bar-label">Outflow</span>
      </div>`;

    $("vol-inflow").textContent   = fmtM(d.inflow_20);
    $("vol-outflow").textContent  = fmtM(d.outflow_20);
    $("vol-buyratio").textContent = fmt(d.buy_ratio,1)+"%";
    $("vol-buyratio").style.color = d.buy_ratio > 50 ? "var(--green)" : "var(--red)";
    $("vol-ratio").textContent    = fmt(d.volume_ratio,2)+"x";
    $("vol-ratio").style.color    = d.volume_ratio > 1 ? "var(--green)" : "var(--text2)";

    // Candlestick analysis fields
    const bbw = ((d.bb_upper - d.bb_lower) / (d.bb_mid || 1) * 100).toFixed(2);
    $("cs-atr").textContent       = fmt(d.atr14,6);
    $("cs-bbw").textContent       = bbw + "%";
    $("cs-mom").textContent       = pct(d.momentum_10);
    $("cs-mom").style.color       = d.momentum_10 >= 0 ? "var(--green)" : "var(--red)";
    $("cs-vwap").textContent      = "$"+fmt(d.vwap,4);
    const delta = ((d.current_price - d.ema200) / d.ema200 * 100).toFixed(2);
    $("cs-ema200delta").textContent = (delta>=0?"+":"")+delta+"%";
    $("cs-ema200delta").style.color = delta>=0?"var(--green)":"var(--red)";

  }catch(e){ console.warn("indicator error",e) }
}

function rsiSignal(v){ return v < 30 ? "BUY" : v > 70 ? "SELL" : "NEUTRAL" }

// ─── Klines mini candles ──────────────────────────────────────────────────────
async function loadKlines(){
  try{
    const d = await fetch("/api/klines?interval=1d&limit=30").then(r=>r.json());
    const candles = d.candles || [];
    if(!candles.length) return;
    const highs  = candles.map(c=>c.high);
    const lows   = candles.map(c=>c.low);
    const gHigh  = Math.max(...highs);
    const gLow   = Math.min(...lows);
    const rng    = gHigh - gLow || 1;
    const el = $("mini-candles");
    el.innerHTML = candles.map(c=>{
      const isBull = c.close >= c.open;
      const col    = isBull ? "var(--green)" : "var(--red)";
      const bodyTop    = ((gHigh - Math.max(c.open,c.close)) / rng * 72).toFixed(1);
      const bodyHeight = (Math.abs(c.close - c.open) / rng * 72 || 2).toFixed(1);
      const wickTop    = ((gHigh - c.high) / rng * 72).toFixed(1);
      const wickHeight = ((c.high - c.low) / rng * 72).toFixed(1);
      return `<div class="candle-wrap" style="color:${col}">
        <div class="candle-wick" style="height:${wickHeight}px;top:${wickTop}px;background:${col}"></div>
        <div class="candle-body" style="height:${bodyHeight}px;margin-top:${bodyTop}px;background:${col}"></div>
      </div>`;
    }).join("");
  }catch(e){ console.warn("klines error",e) }
}

// ─── Prediction ──────────────────────────────────────────────────────────────
async function loadPrediction(){
  try{
    const d = await fetch("/api/prediction").then(r=>r.json());
    state.prediction = d;

    // Classification badge
    const dir  = (d.predicted_direction || "sideways").toLowerCase();
    const bMap = { up:"badge-bull", down:"badge-bear", sideways:"badge-side" };
    const lMap = { up:"BULLISH", down:"BEARISH", sideways:"SIDEWAYS" };
    const badge = $("clf-badge");
    badge.className = "direction-badge " + (bMap[dir] || "badge-side");
    badge.textContent = lMap[dir] || dir.toUpperCase();

    const pu = (d.prob_up||0)*100, pd2 = (d.prob_down||0)*100, ps = (d.prob_sideways||0)*100;
    $("pb-bull").style.width = pu.toFixed(1)+"%";
    $("pb-bear").style.width = pd2.toFixed(1)+"%";
    $("pb-side").style.width = ps.toFixed(1)+"%";
    $("pv-bull").textContent = pu.toFixed(1)+"%";
    $("pv-bear").textContent = pd2.toFixed(1)+"%";
    $("pv-side").textContent = ps.toFixed(1)+"%";

    const modelsLabel = d.model_active ? `${d.n_models_used} models` : "Demo mode";
    $("clf-models").textContent = modelsLabel;
    $("conf-models-label").textContent = modelsLabel;

    // Model status
    const msEl = $("model-status-clf");
    if(d.model_active){
      msEl.innerHTML = `<div class="model-status ms-ready"><div class="ms-dot ms-green"></div>Ensemble active · ${d.n_models_used} models loaded</div>`;
    } else {
      msEl.innerHTML = `<div class="model-status ms-unready"><div class="ms-dot ms-yellow"></div>Run python train.py first to activate AI</div>`;
    }

    // Regression
    $("pred-price-usd").textContent = "$"+ fmt(d.predicted_price_usd,4);
    $("pred-price-inr").textContent = fmtInr(d.predicted_price_inr);
    const rp = d.predicted_return_pct || 0;
    $("pred-return-pct").textContent = (rp>=0?"+":"")+fmt(rp,2)+"% predicted return";
    $("pred-return-pct").style.color = rp>=0?"var(--green)":"var(--red)";
    $("pred-lower").textContent = "$"+fmt(d.price_lower_bound_usd,4);
    $("pred-upper").textContent = "$"+fmt(d.price_upper_bound_usd,4);
    $("pred-regime").textContent = "Regime: "+ (d.market_regime||"—");

    // Confidence gauge
    const conf = ((d.confidence_score||0.5)*100).toFixed(0);
    $("gauge-pct").textContent = conf+"%";
    // Arc animation: full arc = 188.5px circumference for semi-circle
    const dashOff = 188.5 - (conf/100)*188.5;
    $("gauge-arc").style.strokeDashoffset = dashOff;
    // Needle: -90 to +90 deg
    const needleDeg = -90 + (conf/100)*180;
    $("gauge-needle").style.transform = `rotate(${needleDeg}deg)`;

    $("conf-btccor").textContent = fmt(d.btc_correlation,3);
    $("conf-alt").textContent    = fmt(d.altseason_score,3);

    // Risk
    const irr = d.inr_rate || inrRate;
    $("risk-entry").textContent     = "$"+fmt(d.entry_price,4);
    $("risk-entry-inr").textContent = fmtInr((d.entry_price||0)*irr);
    $("risk-sl").textContent        = "$"+fmt(d.stop_loss,4);
    $("risk-sl-inr").textContent    = fmtInr((d.stop_loss||0)*irr);
    $("risk-tp").textContent        = "$"+fmt(d.take_profit,4);
    $("risk-tp-inr").textContent    = fmtInr((d.take_profit||0)*irr);
    $("risk-rr").textContent        = fmt(d.risk_reward,2)+"x";
    $("risk-regime").textContent    = d.market_regime || "—";
    $("risk-regime-conf").textContent = fmt((d.regime_confidence||0)*100,1)+"%";

    // Sentiment
    const si = ((d.sentiment_score||0) + 1) / 2 * 100;
    $("sent-ai").textContent = fmt(si,1)+"%";
    $("sent-ai").style.color = si > 55 ? "var(--green)" : si < 45 ? "var(--red)" : "var(--yellow)";

    // Horizon table
    const horizons = d.horizon_predictions || [];
    if(horizons.length){
      $("horizon-tbody").innerHTML = horizons.map((h,i)=>{
        const hp_usd = h.predicted_price || 0;
        const hp_inr = hp_usd * irr;
        const hr_pct = h.predicted_return_pct || 0;
        const hdir   = (h.predicted_direction||"").toLowerCase();
        const hcol   = hdir==="up"?"var(--green)":hdir==="down"?"var(--red)":"var(--yellow)";
        return `<tr>
          <td style="color:var(--text3)">+${i+1}</td>
          <td>$${fmt(hp_usd,4)}</td>
          <td>${fmtInr(hp_inr)}</td>
          <td style="color:${clr(hr_pct)}">${pct(hr_pct)}</td>
          <td style="color:${hcol};text-transform:uppercase;font-size:10px">${hdir||"—"}</td>
        </tr>`;
      }).join("");
    }

  }catch(e){ console.warn("prediction error",e) }
}

// ─── Main refresh loop ────────────────────────────────────────────────────────
async function refreshAll(){
  await Promise.allSettled([
    loadTicker(),
    loadFunding(),
    loadFG(),
    loadGlobal(),
    loadOrderBook(),
    loadIndicators(),
    loadKlines(),
    loadPrediction(),
  ]);
}

// Initial load
refreshAll();

// Auto-refresh every 15 seconds for live data
setInterval(refreshAll, 15000);

// Clock
setInterval(()=>{
  const t = $("top-time");
  if(t) t.textContent = new Date().toLocaleTimeString("en-IN");
}, 1000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║         RENDER AI TRADING TERMINAL                      ║
║  Symbol  : RENDERUSDT (Binance — current post-rebrand)  ║
║  TradingView: BINANCE:RENDERUSDT                        ║
║  Dashboard: http://localhost:{port}                        ║
║                                                          ║
║  Live APIs:                                              ║
║    /api/ticker     — price USD + INR (live Binance)      ║
║    /api/indicators — RSI, MACD, BB, EMA, VWAP, ATR...   ║
║    /api/orderbook  — depth, imbalance, spread            ║
║    /api/funding    — funding rate, OI, long/short ratio  ║
║    /api/feargreed  — fear & greed index                  ║
║    /api/global     — BTC dominance, market cap           ║
║    /api/prediction — AI prediction (needs trained models)║
║    /api/klines     — OHLCV candlestick data              ║
║                                                          ║
║  To activate AI predictions:                            ║
║    python train.py --symbol RENDER --fast               ║
╚══════════════════════════════════════════════════════════╝
""")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
