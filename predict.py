"""
================================================================================
  ADVANCED CRYPTO PREDICTION INFERENCE ENGINE  -  predict.py
  Full Single-File Version  |  RNDR & Related Assets
  Classification + Regression | Ensemble Stacking | Real-Time Streaming
  Companion to train.py — loads all saved artifacts and runs live inference
================================================================================

HOW TO USE:
    python predict.py                               # Live prediction for RNDR (1d)
    python predict.py --symbol BTC --tf 4h          # Custom symbol & timeframe
    python predict.py --symbol RNDR --tf 1h --horizon 24  # 24-step ahead
    python predict.py --stream                      # Real-time streaming mode
    python predict.py --backfill 30                 # Predict last 30 candles
    python predict.py --export json                 # Export as JSON
    python predict.py --export csv                  # Export as CSV
    python predict.py --models-dir /custom/path     # Custom model directory

INPUTS (produced by train.py):
    /tmp/crypto_train_output/
        models/            ← all .pkl, .h5 model files
        results_summary.json
        feature_importance.csv

OUTPUTS:
    /tmp/crypto_predict_output/
        live_prediction_<SYMBOL>.json
        live_prediction_<SYMBOL>.csv
        signal_report_<SYMBOL>.json
        multi_horizon_<SYMBOL>.csv
        stream_log_<SYMBOL>.jsonl
        logs/predict.log
================================================================================
"""

# ============================================================================
# STANDARD LIBRARY
# ============================================================================
import os
import sys
import json
import time
import math
import pickle
import logging
import argparse
import warnings
import traceback
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import defaultdict, deque
import threading
import copy
import random

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

# ============================================================================
# NUMERIC / DATA
# ============================================================================
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import argrelextrema
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# SKLEARN
# ============================================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, roc_auc_score
)

# ============================================================================
# GRADIENT BOOSTING
# ============================================================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False

# ============================================================================
# DEEP LEARNING
# ============================================================================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, backend as K
    TF_AVAILABLE = True
    tf.get_logger().setLevel("ERROR")
except ImportError:
    pass

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    pass

# ============================================================================
# HTTP
# ============================================================================
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "predict.log")

    logger = logging.getLogger("CryptoPredict")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# ============================================================================
# SECTION 1 – CONFIG DATACLASSES (mirrors train.py)
# ============================================================================

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION     = "regression"
    MULTI_TASK     = "multi_task"


class SignalStrength(Enum):
    STRONG_BUY  = "STRONG_BUY"
    BUY         = "BUY"
    WEAK_BUY    = "WEAK_BUY"
    HOLD        = "HOLD"
    WEAK_SELL   = "WEAK_SELL"
    SELL        = "SELL"
    STRONG_SELL = "STRONG_SELL"


class MarketRegime(Enum):
    TRENDING_UP   = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING       = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    ACCUMULATION  = "ACCUMULATION"
    DISTRIBUTION  = "DISTRIBUTION"
    UNKNOWN       = "UNKNOWN"


@dataclass
class PredictConfig:
    # ── paths ──────────────────────────────────────────────────────────────
    train_output_dir:   str  = "/tmp/crypto_train_output"
    predict_output_dir: str  = "/tmp/crypto_predict_output"

    # ── inference ──────────────────────────────────────────────────────────
    symbol:             str  = "RNDR"
    timeframe:          str  = "1d"
    lookback_days:      int  = 180         # data needed for features
    prediction_horizon: int  = 12          # steps ahead (multi-horizon)
    sequence_length:    int  = 60          # must match train.py
    min_confidence:     float = 0.52

    # ── signal thresholds ──────────────────────────────────────────────────
    up_threshold:       float = 0.015      # +1.5 %  → bullish
    down_threshold:     float = -0.015     # -1.5 %  → bearish
    strong_signal_conf: float = 0.70       # ≥70 % → STRONG
    signal_conf:        float = 0.58       # ≥58 % → normal signal

    # ── risk ───────────────────────────────────────────────────────────────
    stop_loss_pct:      float = 0.05
    take_profit_pct:    float = 0.10
    max_leverage:       float = 3.0
    position_size_pct:  float = 0.02

    # ── ensemble weights (overridden from importance if available) ─────────
    clf_model_weights:  Dict[str, float] = field(default_factory=dict)
    reg_model_weights:  Dict[str, float] = field(default_factory=dict)

    # ── data / network ──────────────────────────────────────────────────────
    max_retries:        int   = 3
    retry_delay:        float = 2.0
    api_keys:           Dict  = field(default_factory=dict)
    related_symbols:    List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "AR", "TAO", "ICP",
        "AVAX", "NEAR", "API3", "SAND", "MANA"
    ])

    # ── stream ─────────────────────────────────────────────────────────────
    stream_interval_sec: int  = 300        # refresh every 5 min

    @classmethod
    def from_env(cls, cfg: "PredictConfig") -> "PredictConfig":
        cfg.api_keys = {
            "binance":     os.getenv("BINANCE_API_KEY",     ""),
            "coingecko":   os.getenv("COINGECKO_API_KEY",   ""),
            "cryptoquant": os.getenv("CRYPTOQUANT_API_KEY", ""),
            "glassnode":   os.getenv("GLASSNODE_API_KEY",   ""),
            "newsapi":     os.getenv("NEWSAPI_KEY",         ""),
            "etherscan":   os.getenv("ETHERSCAN_API_KEY",   ""),
        }
        return cfg


# ============================================================================
# SECTION 2 – OUTPUT DATA CLASSES
# ============================================================================

@dataclass
class ModelPrediction:
    """Raw output from a single model."""
    model_name:       str
    model_type:       str    # "clf" | "reg"
    is_deep:          bool
    raw_output:       Any    # probabilities or scalar
    predicted_class:  Optional[int]   = None   # 0=down,1=side,2=up
    predicted_prob:   Optional[float] = None   # P(predicted_class)
    predicted_return: Optional[float] = None   # regression output
    latency_ms:       float           = 0.0
    confidence_score: float           = 0.0    # model-level confidence


@dataclass
class EnsemblePrediction:
    """Weighted ensemble across all models."""
    symbol:            str
    timestamp:         datetime
    timeframe:         str
    current_price:     float

    # ── classification ──────────────────────────────────────────────────────
    prob_up:           float
    prob_down:         float
    prob_sideways:     float
    predicted_direction: str   # "up" | "down" | "sideways"
    direction_confidence: float

    # ── regression ─────────────────────────────────────────────────────────
    predicted_return_pct: float
    predicted_price:      float
    price_lower_bound:    float    # 95 % CI lower
    price_upper_bound:    float    # 95 % CI upper
    return_std:           float    # cross-model σ

    # ── signal ─────────────────────────────────────────────────────────────
    signal:            str         # STRONG_BUY … STRONG_SELL
    signal_score:      float       # -1 to +1 composite
    entry_price:       float
    stop_loss:         float
    take_profit:       float
    risk_reward:       float
    position_size_pct: float

    # ── regime & context ───────────────────────────────────────────────────
    market_regime:     str
    regime_confidence: float
    btc_correlation:   float
    altseason_score:   float
    sentiment_score:   float
    onchain_score:     float
    fear_greed_index:  float
    funding_rate:      float
    open_interest_chg: float

    # ── multi-horizon ────────────────────────────────────────────────────
    horizon_predictions: List[Dict] = field(default_factory=list)

    # ── per-model breakdown ───────────────────────────────────────────────
    model_predictions:  List[Dict]  = field(default_factory=list)

    # ── metadata ──────────────────────────────────────────────────────────
    n_models_used:     int   = 0
    feature_count:     int   = 0
    data_quality_score: float = 1.0
    prediction_id:     str   = ""
    pipeline_version:  str   = "2.0"


@dataclass
class RealTimeState:
    """Holds rolling state for streaming mode."""
    symbol:        str
    price_buffer:  deque
    volume_buffer: deque
    pred_buffer:   deque        # recent EnsemblePredictions
    last_updated:  datetime     = field(default_factory=datetime.utcnow)
    consecutive_signals: int    = 0
    last_signal:   str          = "HOLD"
    position_open: bool         = False
    entry_price:   float        = 0.0
    unrealized_pnl: float       = 0.0


# ============================================================================
# SECTION 3 – DATA COLLECTION (mirrors train.py collectors)
# ============================================================================

class RetryMixin:
    def _retry_request(self, url, params=None, headers=None, logger=None):
        if not REQUESTS_AVAILABLE:
            return None
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if logger:
                    logger.warning(f"HTTP {resp.status_code} from {url}")
            except Exception as e:
                if logger:
                    logger.warning(f"Request attempt {attempt+1} failed: {e}")
                time.sleep(2.0 * (attempt + 1))
        return None


class LiveBinanceCollector(RetryMixin):
    """Fetches live OHLCV + futures data from Binance public API."""

    BASE      = "https://api.binance.com/api/v3"
    FBASE     = "https://fapi.binance.com/fapi/v1"

    def __init__(self, logger):
        self.logger = logger

    def fetch_klines(self, symbol: str, interval: str = "1d",
                     limit: int = 200) -> pd.DataFrame:
        url    = f"{self.BASE}/klines"
        params = {"symbol": f"{symbol}USDT", "interval": interval,
                  "limit": limit}
        data   = self._retry_request(url, params=params, logger=self.logger)
        if not data:
            return pd.DataFrame()
        cols = ["open_time","open","high","low","close","volume",
                "close_time","quote_volume","trades","taker_buy_base",
                "taker_buy_quote","ignore"]
        df   = pd.DataFrame(data, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df   = df.set_index("timestamp")
        num_cols = ["open","high","low","close","volume","quote_volume",
                    "trades","taker_buy_base","taker_buy_quote"]
        df[num_cols] = df[num_cols].astype(float)
        df["buy_sell_ratio"] = df["taker_buy_base"] / (df["volume"] + 1e-9)
        return df[["open","high","low","close","volume","quote_volume",
                   "trades","taker_buy_base","taker_buy_quote",
                   "buy_sell_ratio"]]

    def fetch_ticker_24h(self, symbol: str) -> Dict:
        url  = f"{self.BASE}/ticker/24hr"
        data = self._retry_request(url, params={"symbol": f"{symbol}USDT"},
                                    logger=self.logger)
        return data or {}

    def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict:
        url  = f"{self.BASE}/depth"
        data = self._retry_request(url,
                                    params={"symbol": f"{symbol}USDT",
                                            "limit": limit},
                                    logger=self.logger)
        if not data:
            return {}
        bids = np.array(data["bids"], dtype=float)
        asks = np.array(data["asks"], dtype=float)
        bid_vol = bids[:, 1].sum() if len(bids) else 0
        ask_vol = asks[:, 1].sum() if len(asks) else 0
        return {
            "best_bid":       float(bids[0, 0]) if len(bids) else 0,
            "best_ask":       float(asks[0, 0]) if len(asks) else 0,
            "spread":         float(asks[0, 0] - bids[0, 0]) if (len(bids) and len(asks)) else 0,
            "bid_volume":     bid_vol,
            "ask_volume":     ask_vol,
            "order_book_imbalance": (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9),
        }

    def fetch_funding_rate(self, symbol: str) -> float:
        url  = f"{self.FBASE}/premiumIndex"
        data = self._retry_request(url, params={"symbol": f"{symbol}USDT"},
                                    logger=self.logger)
        if data and "lastFundingRate" in data:
            return float(data["lastFundingRate"])
        return 0.0

    def fetch_open_interest(self, symbol: str) -> Dict:
        url  = f"{self.FBASE}/openInterest"
        data = self._retry_request(url, params={"symbol": f"{symbol}USDT"},
                                    logger=self.logger)
        if not data:
            return {}
        return {
            "open_interest":       float(data.get("openInterest", 0)),
            "open_interest_value": float(data.get("openInterest", 0)) *
                                   float(data.get("time", 1))  # proxy
        }

    def fetch_long_short_ratio(self, symbol: str) -> Dict:
        url  = f"{self.FBASE}/globalLongShortAccountRatio"
        data = self._retry_request(url,
                                    params={"symbol": f"{symbol}USDT",
                                            "period": "1h", "limit": 1},
                                    logger=self.logger)
        if data and isinstance(data, list) and data:
            row = data[0]
            return {
                "long_short_ratio": float(row.get("longShortRatio", 1.0)),
                "long_pct":  float(row.get("longAccount",  0.5)),
                "short_pct": float(row.get("shortAccount", 0.5)),
            }
        return {"long_short_ratio": 1.0, "long_pct": 0.5, "short_pct": 0.5}


class LiveFearGreedCollector(RetryMixin):
    URL = "https://api.alternative.me/fng/?limit=30&format=json"

    def __init__(self, logger):
        self.logger = logger

    def fetch(self) -> pd.DataFrame:
        data = self._retry_request(self.URL, logger=self.logger)
        if not data or "data" not in data:
            return self._synthetic_fg()
        rows = []
        for item in data["data"]:
            rows.append({
                "timestamp":       pd.to_datetime(int(item["timestamp"]),
                                                   unit="s", utc=True),
                "fear_greed_index": float(item["value"]),
            })
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        df["fg_ma7"]    = df["fear_greed_index"].rolling(7, min_periods=1).mean()
        df["fg_ma30"]   = df["fear_greed_index"].rolling(30, min_periods=1).mean()
        df["fg_trend"]  = df["fg_ma7"] - df["fg_ma30"]
        df["fg_zscore"] = (
            (df["fear_greed_index"] - df["fear_greed_index"].rolling(30, min_periods=1).mean()) /
            (df["fear_greed_index"].rolling(30, min_periods=1).std() + 1e-9)
        )
        return df

    def _synthetic_fg(self) -> pd.DataFrame:
        idx = pd.date_range(end=datetime.utcnow(), periods=30, freq="D", tz="UTC")
        df  = pd.DataFrame({"fear_greed_index": 50.0}, index=idx)
        df["fg_ma7"] = df["fg_ma30"] = df["fg_trend"] = df["fg_zscore"] = 0.0
        return df


class LiveCoinGeckoCollector(RetryMixin):
    BASE = "https://api.coingecko.com/api/v3"
    SYMBOL_MAP = {
        "RNDR": "render-token", "BTC": "bitcoin", "ETH": "ethereum",
        "SOL": "solana", "AR": "arweave", "TAO": "bittensor",
        "ICP": "internet-computer", "AVAX": "avalanche-2",
        "NEAR": "near", "API3": "api3", "SAND": "the-sandbox",
        "MANA": "decentraland",
    }

    def __init__(self, logger):
        self.logger = logger

    def fetch_global_metrics(self) -> Dict:
        url  = f"{self.BASE}/global"
        data = self._retry_request(url, logger=self.logger)
        if not data:
            return {}
        gd = data.get("data", {})
        dom = gd.get("market_cap_percentage", {})
        return {
            "btc_dominance":      dom.get("btc", 50.0),
            "eth_dominance":      dom.get("eth", 20.0),
            "total_market_cap":   gd.get("total_market_cap", {}).get("usd", 1e12),
            "total_volume_24h":   gd.get("total_volume", {}).get("usd", 1e11),
        }

    def _coin_id(self, symbol: str) -> str:
        return self.SYMBOL_MAP.get(symbol.upper(), symbol.lower())


class LiveOnChainSynthetic:
    """Generates realistic synthetic on-chain metrics when API keys absent."""

    def __init__(self, logger):
        self.logger = logger

    def generate(self, index: pd.DatetimeIndex,
                 price_series: pd.Series) -> pd.DataFrame:
        n   = len(index)
        rng = np.random.default_rng(42)
        ret = price_series.pct_change().fillna(0)
        df  = pd.DataFrame(index=index)

        df["active_addresses"]   = (50000 + ret * 200000 +
                                     rng.normal(0, 5000, n)).clip(0)
        df["transaction_count"]  = (300000 + ret * 500000 +
                                     rng.normal(0, 30000, n)).clip(0)
        df["large_tx_count"]     = (500 + ret * 2000 +
                                     rng.normal(0, 50, n)).clip(0)
        df["whale_volume_ratio"] = (0.3 + ret * 0.5 +
                                     rng.normal(0, 0.05, n)).clip(0, 1)
        df["exchange_inflow"]    = (1e7 + ret * 5e7 +
                                     rng.normal(0, 1e6, n)).clip(0)
        df["exchange_outflow"]   = (1e7 - ret * 5e7 +
                                     rng.normal(0, 1e6, n)).clip(0)
        df["exchange_netflow"]   = df["exchange_inflow"] - df["exchange_outflow"]
        df["mvrv_ratio"]         = (1.5 + ret * 3 +
                                     rng.normal(0, 0.2, n)).clip(0.1)
        df["nvt_signal"]         = (80 - ret * 200 +
                                     rng.normal(0, 10, n)).clip(0)
        df["sopr"]               = (1.0 + ret * 0.5 +
                                     rng.normal(0, 0.05, n)).clip(0.5)
        df["dormant_circulation"]= (0.05 + rng.normal(0, 0.01, n)).clip(0)
        df["liveliness"]         = (0.7 + ret * 0.3 +
                                     rng.normal(0, 0.05, n)).clip(0, 1)
        for col in df.columns:
            df[col] = df[col].rolling(3, min_periods=1).mean()
        return df


class LiveSentimentSynthetic:
    """Realistic synthetic sentiment correlated with price."""

    def __init__(self, logger):
        self.logger = logger

    def generate(self, index: pd.DatetimeIndex,
                 price_series: pd.Series) -> pd.DataFrame:
        n   = len(index)
        rng = np.random.default_rng(99)
        ret = price_series.pct_change().fillna(0).shift(2).fillna(0)
        df  = pd.DataFrame(index=index)

        df["twitter_sentiment"]  = (0.5 + ret * 5 + rng.normal(0, 0.15, n)).clip(-1, 1)
        df["reddit_sentiment"]   = (0.4 + ret * 4 + rng.normal(0, 0.20, n)).clip(-1, 1)
        df["news_sentiment"]     = (0.3 + ret * 3 + rng.normal(0, 0.12, n)).clip(-1, 1)
        df["mention_volume"]     = (1000 + ret * 5000 + rng.normal(0, 200, n)).clip(0)
        df["ai_sentiment_score"] = (
            df["twitter_sentiment"] * 0.4 +
            df["reddit_sentiment"]  * 0.3 +
            df["news_sentiment"]    * 0.3
        )
        for col in df.columns:
            df[col] = df[col].rolling(2, min_periods=1).mean()
        return df


class LiveDataAggregator:
    """
    Orchestrates all live collectors, builds a feature-ready DataFrame
    that mirrors the schema produced by train.py's DataAggregator.
    """

    def __init__(self, config: PredictConfig, logger: logging.Logger):
        self.config    = config
        self.logger    = logger
        self.binance   = LiveBinanceCollector(logger)
        self.feargreed = LiveFearGreedCollector(logger)
        self.coingecko = LiveCoinGeckoCollector(logger)
        self.onchain   = LiveOnChainSynthetic(logger)
        self.sentiment = LiveSentimentSynthetic(logger)

    def fetch(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        self.logger.info(f"Fetching live data: {symbol}/{timeframe}")

        # Primary OHLCV
        limit   = min(self.config.lookback_days, 1000)
        df_ohlcv = self.binance.fetch_klines(symbol, timeframe, limit)
        if df_ohlcv.empty:
            raise RuntimeError(f"Cannot fetch live data for {symbol}. "
                               "Check connectivity / symbol name.")

        # BTC / ETH reference
        df_btc = self.binance.fetch_klines("BTC", timeframe, limit)
        df_eth = self.binance.fetch_klines("ETH", timeframe, limit)

        # Futures
        funding     = self.binance.fetch_funding_rate(symbol)
        oi_data     = self.binance.fetch_open_interest(symbol)
        lsr_data    = self.binance.fetch_long_short_ratio(symbol)
        ob_data     = self.binance.fetch_order_book(symbol)

        # Fear & greed
        df_fg = self.feargreed.fetch()

        # On-chain + sentiment (synthetic)
        df_onchain = self.onchain.generate(df_ohlcv.index, df_ohlcv["close"])
        df_sent    = self.sentiment.generate(df_ohlcv.index, df_ohlcv["close"])

        # Global crypto metrics
        global_metrics = self.coingecko.fetch_global_metrics()

        # ── Merge ──────────────────────────────────────────────────────────
        df = df_ohlcv.copy()

        # Scalar futures fields → broadcast to all rows
        df["funding_rate"]      = funding
        df["open_interest"]     = oi_data.get("open_interest", 0.0)
        df["open_interest_value"] = oi_data.get("open_interest_value", 0.0)
        df["long_short_ratio"]  = lsr_data.get("long_short_ratio", 1.0)
        df["long_pct"]          = lsr_data.get("long_pct",  0.5)
        df["short_pct"]         = lsr_data.get("short_pct", 0.5)
        df["order_book_imbalance"] = ob_data.get("order_book_imbalance", 0.0)
        df["bid_volume"]        = ob_data.get("bid_volume",  0.0)
        df["ask_volume"]        = ob_data.get("ask_volume",  0.0)
        df["spread"]            = ob_data.get("spread",      0.0)

        # Fear & greed (join on date, forward-fill)
        fg_cols = ["fear_greed_index", "fg_ma7", "fg_ma30",
                   "fg_trend", "fg_zscore"]
        df_fg.index = df_fg.index.normalize()
        df.index    = df.index.normalize()
        df = df.join(df_fg[fg_cols], how="left")

        # On-chain
        df = df.join(df_onchain, how="left")

        # Sentiment
        df = df.join(df_sent, how="left")

        # BTC / ETH
        if not df_btc.empty:
            df_btc_c = df_btc[["close"]].rename(columns={"close": "btc_close"})
            df_btc_c.index = df_btc_c.index.normalize()
            df = df.join(df_btc_c, how="left")
        if not df_eth.empty:
            df_eth_c = df_eth[["close"]].rename(columns={"close": "eth_close"})
            df_eth_c.index = df_eth_c.index.normalize()
            df = df.join(df_eth_c, how="left")

        # Global metrics
        for k, v in global_metrics.items():
            if isinstance(v, (int, float)):
                df[k] = v

        # Clean up
        df = df.sort_index()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()
        df = df.dropna(thresh=int(len(df.columns) * 0.4))

        self.logger.info(f"Live data ready: {df.shape[0]} rows × {df.shape[1]} cols")
        return df


# ============================================================================
# SECTION 4 – TECHNICAL / STATISTICAL INDICATORS
#             (identical math to train.py — kept self-contained)
# ============================================================================

class TI:
    """Technical Indicators — mirrors train.py's TechnicalIndicators."""

    @staticmethod
    def rsi(prices, period=14):
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-9)))

    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        ema_f = prices.ewm(span=fast, adjust=False).mean()
        ema_s = prices.ewm(span=slow, adjust=False).mean()
        line  = ema_f - ema_s
        sig   = line.ewm(span=signal, adjust=False).mean()
        return line, sig, line - sig

    @staticmethod
    def bollinger(prices, period=20, std=2.0):
        sma   = prices.rolling(period).mean()
        sigma = prices.rolling(period).std()
        upper = sma + std * sigma
        lower = sma - std * sigma
        pct_b = (prices - lower) / (upper - lower + 1e-9)
        width = (upper - lower) / (sma + 1e-9)
        return upper, sma, lower, pct_b, width

    @staticmethod
    def atr(high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def stochastic(high, low, close, k=14, d=3):
        ll = low.rolling(k).min()
        hh = high.rolling(k).max()
        K  = 100 * (close - ll) / (hh - ll + 1e-9)
        D  = K.rolling(d).mean()
        return K, D

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        atr_v    = TI.atr(high, low, close, period)
        plus_di  = 100 * (plus_dm.rolling(period).mean() / (atr_v + 1e-9))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_v + 1e-9))
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        return dx.rolling(period).mean(), plus_di, minus_di

    @staticmethod
    def cci(high, low, close, period=20):
        tp  = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-9)

    @staticmethod
    def williams_r(high, low, close, period=14):
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll + 1e-9)

    @staticmethod
    def keltner(high, low, close, ema_p=20, atr_p=10, mult=2.0):
        ema   = close.ewm(span=ema_p, adjust=False).mean()
        atr_v = TI.atr(high, low, close, atr_p)
        return ema + mult * atr_v, ema, ema - mult * atr_v

    @staticmethod
    def donchian(high, low, period=20):
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        return upper, (upper + lower) / 2, lower

    @staticmethod
    def ichimoku(high, low, close):
        conv  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        base  = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = (conv + base) / 2
        span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        return conv, base, span_a, span_b

    @staticmethod
    def parabolic_sar(high, low, close, step=0.02, max_step=0.2):
        n = len(close)
        sar = np.zeros(n); trend = np.ones(n, dtype=int)
        ep  = np.zeros(n); af    = np.zeros(n)
        sar[0] = low.iloc[0]; ep[0] = high.iloc[0]; af[0] = step
        for i in range(1, n):
            ps = sar[i-1]
            if trend[i-1] == 1:
                sar[i] = ps + af[i-1] * (ep[i-1] - ps)
                sar[i] = min(sar[i], low.iloc[i-1], low.iloc[max(0, i-2)])
                if low.iloc[i] < sar[i]:
                    trend[i] = -1; sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]; af[i] = step
                else:
                    trend[i] = 1
                    ep[i] = max(ep[i-1], high.iloc[i])
                    af[i] = min(af[i-1]+step, max_step) if ep[i]>ep[i-1] else af[i-1]
            else:
                sar[i] = ps + af[i-1] * (ep[i-1] - ps)
                sar[i] = max(sar[i], high.iloc[i-1], high.iloc[max(0, i-2)])
                if high.iloc[i] > sar[i]:
                    trend[i] = 1; sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]; af[i] = step
                else:
                    trend[i] = -1
                    ep[i] = min(ep[i-1], low.iloc[i])
                    af[i] = min(af[i-1]+step, max_step) if ep[i]<ep[i-1] else af[i-1]
        return pd.Series(sar, index=close.index)

    # ── Volume ────────────────────────────────────────────────────────────
    @staticmethod
    def obv(close, volume):
        return (np.sign(close.diff().fillna(0)) * volume).cumsum()

    @staticmethod
    def vwap(high, low, close, volume):
        tp = (high + low + close) / 3
        return (tp * volume).cumsum() / (volume.cumsum() + 1e-9)

    @staticmethod
    def mfi(high, low, close, volume, period=14):
        tp  = (high + low + close) / 3
        mf  = tp * volume
        pos = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        return 100 - (100 / (1 + pos / (neg + 1e-9)))

    @staticmethod
    def cmf(high, low, close, volume, period=20):
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        return (clv * volume).rolling(period).sum() / (
            volume.rolling(period).sum() + 1e-9)

    @staticmethod
    def ad(high, low, close, volume):
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        return (clv * volume).cumsum()

    # ── Volatility ────────────────────────────────────────────────────────
    @staticmethod
    def hist_vol(close, period=21):
        return close.pct_change().rolling(period).std() * np.sqrt(252)

    @staticmethod
    def parkinson(high, low, period=21):
        return np.sqrt(
            (np.log(high / low) ** 2).rolling(period).mean() / (4 * np.log(2))
        ) * np.sqrt(252)

    @staticmethod
    def garman_klass(open_, high, low, close, period=21):
        rs = 0.5 * np.log(high/low)**2 - (2*np.log(2)-1) * np.log(close/open_)**2
        return np.sqrt(rs.rolling(period).mean()) * np.sqrt(252)

    # ── Momentum ─────────────────────────────────────────────────────────
    @staticmethod
    def roc(close, period=10):
        return close.pct_change(period) * 100

    @staticmethod
    def momentum(close, period=10):
        return close - close.shift(period)

    @staticmethod
    def cmo(close, period=14):
        delta   = close.diff()
        up_sum  = delta.where(delta > 0, 0).rolling(period).sum()
        dn_sum  = (-delta.where(delta < 0, 0)).rolling(period).sum()
        return 100 * (up_sum - dn_sum) / (up_sum + dn_sum + 1e-9)

    @staticmethod
    def uo(high, low, close):
        bp  = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
        tr  = pd.concat([high, close.shift()], axis=1).max(axis=1) - \
              pd.concat([low,  close.shift()], axis=1).min(axis=1)
        avg7  = bp.rolling(7).sum()  / (tr.rolling(7).sum()  + 1e-9)
        avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-9)
        avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-9)
        return 100 * (4*avg7 + 2*avg14 + avg28) / 7

    # ── Advanced Statistical ──────────────────────────────────────────────
    @staticmethod
    def zscore(series, period=30):
        mu  = series.rolling(period).mean()
        sig = series.rolling(period).std()
        return (series - mu) / (sig + 1e-9)

    @staticmethod
    def hurst(series, period=100, min_obs=20):
        """Rolling Hurst exponent — H>0.5 trending, H<0.5 mean-reverting."""
        def _hurst_single(x):
            if len(x) < min_obs:
                return 0.5
            lags = range(2, min(20, len(x)//2))
            tau  = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            try:
                return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
            except Exception:
                return 0.5
        return series.rolling(period, min_periods=min_obs).apply(
            _hurst_single, raw=True)

    @staticmethod
    def entropy(series, period=30):
        def _e(x):
            x    = np.abs(x)
            s    = x.sum()
            p    = x / (s + 1e-12)
            p    = p[p > 0]
            return -np.sum(p * np.log(p + 1e-12))
        return series.rolling(period, min_periods=10).apply(_e, raw=True)

    @staticmethod
    def support_resistance(close, window=20):
        highs    = close.rolling(window).max()
        lows     = close.rolling(window).min()
        return highs, lows

    @staticmethod
    def rolling_correlation(a, b, period=30):
        return a.rolling(period, min_periods=10).corr(b)

    @staticmethod
    def rolling_beta(a, b, period=60):
        cov = a.pct_change().rolling(period, min_periods=10).cov(b.pct_change())
        var = b.pct_change().rolling(period, min_periods=10).var()
        return cov / (var + 1e-9)

    @staticmethod
    def altseason_index(asset_ret, btc_ret, period=30):
        roll_corr = asset_ret.rolling(period, min_periods=10).corr(btc_ret)
        outperf   = (asset_ret - btc_ret).rolling(period, min_periods=10).mean()
        return outperf * (1 - roll_corr.clip(0, 1))


# ============================================================================
# SECTION 5 – FEATURE ENGINEERING
#             Produces the EXACT same feature columns as train.py
# ============================================================================

class LiveFeatureEngineer:
    """
    Replicates 100 % of train.py FeatureEngineer.engineer_all_features().
    Any mismatch would cause shape/column errors at inference time.
    """

    def __init__(self, df: pd.DataFrame, config: PredictConfig,
                 logger: logging.Logger):
        self.df     = df.copy()
        self.config = config
        self.logger = logger

    def engineer(self) -> pd.DataFrame:
        df   = self.df
        feat = pd.DataFrame(index=df.index)

        open_  = df["open"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        close  = df["close"].astype(float)
        volume = df["volume"].astype(float)

        # ── Price-derived returns ────────────────────────────────────────
        feat["close"]       = close
        feat["open"]        = open_
        feat["high"]        = high
        feat["low"]         = low
        feat["volume"]      = volume
        feat["hl_ratio"]    = (high - low) / (close + 1e-9)
        feat["oc_ratio"]    = (open_ - close) / (close + 1e-9)
        feat["upper_shadow"]= (high - pd.concat([open_, close], axis=1).max(axis=1)) / (close + 1e-9)
        feat["lower_shadow"]= (pd.concat([open_, close], axis=1).min(axis=1) - low) / (close + 1e-9)

        for p in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
            feat[f"ret_{p}d"]    = close.pct_change(p)
            feat[f"log_ret_{p}d"]= np.log(close / close.shift(p) + 1e-9)

        # ── SMAs / EMAs ──────────────────────────────────────────────────
        for w in [5, 10, 20, 50, 100, 200]:
            feat[f"sma_{w}"]     = close.rolling(w).mean()
            feat[f"ema_{w}"]     = close.ewm(span=w, adjust=False).mean()
            feat[f"sma_rel_{w}"] = close / (feat[f"sma_{w}"] + 1e-9) - 1

        # ── RSI ──────────────────────────────────────────────────────────
        for p in [7, 14, 21]:
            feat[f"rsi_{p}"] = TI.rsi(close, p)
        feat["rsi_ma"]  = feat["rsi_14"].rolling(5).mean()
        feat["rsi_div"] = feat["rsi_14"] - feat["rsi_14"].rolling(14).mean()

        # ── MACD ─────────────────────────────────────────────────────────
        ml, ms, mh      = TI.macd(close)
        feat["macd"]          = ml
        feat["macd_signal"]   = ms
        feat["macd_hist"]     = mh
        feat["macd_cross"]    = (ml > ms).astype(int)
        feat["macd_hist_roc"] = mh.diff()

        # ── Bollinger Bands ──────────────────────────────────────────────
        bb_up, bb_mid, bb_lo, bb_pct, bb_w = TI.bollinger(close)
        feat["bb_upper"]   = bb_up
        feat["bb_lower"]   = bb_lo
        feat["bb_pct"]     = bb_pct
        feat["bb_width"]   = bb_w
        feat["bb_squeeze"] = (bb_w < bb_w.rolling(20).quantile(0.25)).astype(int)

        # ── ATR ──────────────────────────────────────────────────────────
        atr14              = TI.atr(high, low, close, 14)
        feat["atr_14"]     = atr14
        feat["atr_ratio"]  = atr14 / (close + 1e-9)
        feat["atr_pct_chg"]= atr14.pct_change()

        # ── Stochastic ────────────────────────────────────────────────────
        sk, sd             = TI.stochastic(high, low, close)
        feat["stoch_k"]    = sk
        feat["stoch_d"]    = sd
        feat["stoch_diff"] = sk - sd

        # ── ADX ───────────────────────────────────────────────────────────
        adx_v, pdi, mdi   = TI.adx(high, low, close)
        feat["adx"]        = adx_v
        feat["plus_di"]    = pdi
        feat["minus_di"]   = mdi
        feat["di_diff"]    = pdi - mdi

        # ── CCI ────────────────────────────────────────────────────────
        feat["cci_20"]     = TI.cci(high, low, close, 20)
        feat["cci_50"]     = TI.cci(high, low, close, 50)

        # ── Williams %R ──────────────────────────────────────────────────
        feat["williams_r_14"] = TI.williams_r(high, low, close, 14)

        # ── Aroon ────────────────────────────────────────────────────────
        def _aroon(h, l, p=25):
            up = h.rolling(p+1).apply(
                lambda x: ((p - x[::-1].argmax()) / p) * 100, raw=True)
            dn = l.rolling(p+1).apply(
                lambda x: ((p - x[::-1].argmin()) / p) * 100, raw=True)
            return up, dn
        ar_up, ar_dn       = _aroon(high, low)
        feat["aroon_up"]   = ar_up
        feat["aroon_dn"]   = ar_dn
        feat["aroon_osc"]  = ar_up - ar_dn

        # ── Keltner / Donchian ────────────────────────────────────────────
        kc_up, kc_mid, kc_lo = TI.keltner(high, low, close)
        feat["kc_upper"]  = kc_up
        feat["kc_lower"]  = kc_lo
        feat["kc_pct"]    = (close - kc_lo) / (kc_up - kc_lo + 1e-9)

        dc_up, dc_mid, dc_lo = TI.donchian(high, low, 20)
        feat["dc_upper"]  = dc_up
        feat["dc_lower"]  = dc_lo
        feat["dc_pct"]    = (close - dc_lo) / (dc_up - dc_lo + 1e-9)

        # ── Parabolic SAR ────────────────────────────────────────────────
        feat["psar"]        = TI.parabolic_sar(high, low, close)
        feat["psar_signal"] = (close > feat["psar"]).astype(int)

        # ── Ichimoku ─────────────────────────────────────────────────────
        ich_c, ich_b, ich_sa, ich_sb = TI.ichimoku(high, low, close)
        feat["ichi_conv"]       = ich_c
        feat["ichi_base"]       = ich_b
        feat["ichi_span_a"]     = ich_sa
        feat["ichi_span_b"]     = ich_sb
        feat["ichi_cloud_top"]  = pd.concat([ich_sa, ich_sb], axis=1).max(axis=1)
        feat["ichi_cloud_bot"]  = pd.concat([ich_sa, ich_sb], axis=1).min(axis=1)
        feat["ichi_above_cloud"]= (close > feat["ichi_cloud_top"]).astype(int)

        # ── Volume Indicators ────────────────────────────────────────────
        feat["obv"]      = TI.obv(close, volume)
        feat["vwap"]     = TI.vwap(high, low, close, volume)
        feat["mfi"]      = TI.mfi(high, low, close, volume)
        feat["ad"]       = TI.ad(high, low, close, volume)
        feat["cmf"]      = TI.cmf(high, low, close, volume)
        feat["vwap_dev"] = (close - feat["vwap"]) / (close + 1e-9)

        for w in [10, 20, 50]:
            feat[f"vol_sma_{w}"] = volume.rolling(w).mean()
        feat["vol_roc"]    = volume.pct_change(5)
        feat["vol_trend"]  = feat["vol_sma_10"] / (feat["vol_sma_50"] + 1e-9)
        feat["vol_zscore"] = TI.zscore(volume, 20)
        feat["vol_spike"]  = (volume > volume.rolling(20).mean() +
                               2 * volume.rolling(20).std()).astype(int)

        # ── Volatility ────────────────────────────────────────────────────
        feat["hv_21d"]    = TI.hist_vol(close, 21)
        feat["hv_63d"]    = TI.hist_vol(close, 63)
        feat["parkinson"] = TI.parkinson(high, low, 21)
        feat["gk_vol"]    = TI.garman_klass(open_, high, low, close, 21)
        for p in [5, 21, 63]:
            feat[f"rv_{p}d"] = close.pct_change().rolling(p).std() * np.sqrt(252)
        feat["vol_ratio_5_21"] = feat["rv_5d"] / (feat["rv_21d"] + 1e-9)

        # ── Momentum ─────────────────────────────────────────────────────
        for p in [5, 10, 20]:
            feat[f"roc_{p}"] = TI.roc(close, p)
            feat[f"mom_{p}"] = TI.momentum(close, p)
        feat["velocity"]     = close.diff()
        feat["acceleration"] = close.diff().diff()
        feat["cmo_14"]       = TI.cmo(close, 14)
        feat["uo"]           = TI.uo(high, low, close)

        # ── Advanced Statistical ──────────────────────────────────────────
        feat["zscore_anom"]  = TI.zscore(close, 30)
        feat["hurst"]        = TI.hurst(close, 100, 10)
        feat["entropy_30"]   = TI.entropy(close, 30)
        res, sup             = TI.support_resistance(close)
        feat["resistance"]   = res
        feat["support"]      = sup
        feat["sr_distance"]  = (close - sup) / (res - sup + 1e-9)

        # 52-week extremes
        feat["high_52w"]      = high.rolling(252, min_periods=1).max()
        feat["low_52w"]       = low.rolling(252, min_periods=1).min()
        feat["pos_52w"]       = (close - feat["low_52w"]) / (feat["high_52w"] - feat["low_52w"] + 1e-9)
        feat["dist_from_52h"] = (feat["high_52w"] - close) / (close + 1e-9)

        # ── Cross-asset ───────────────────────────────────────────────────
        if "btc_close" in df.columns:
            feat["btc_close"]   = df["btc_close"]
            feat["btc_ret_1d"]  = df["btc_close"].pct_change()
            feat["corr_btc_30"] = TI.rolling_correlation(close, df["btc_close"], 30)
            feat["beta_btc_60"] = TI.rolling_beta(close, df["btc_close"], 60)
            feat["altseason"]   = TI.altseason_index(
                close.pct_change(), df["btc_close"].pct_change())
            feat["rel_str_btc"] = close / (df["btc_close"] + 1e-9)
        if "eth_close" in df.columns:
            feat["eth_close"]   = df["eth_close"]
            feat["corr_eth_30"] = TI.rolling_correlation(close, df["eth_close"], 30)
            feat["beta_eth_60"] = TI.rolling_beta(close, df["eth_close"], 60)

        # ── Passthrough sourced features ──────────────────────────────────
        passthrough = [
            "funding_rate", "open_interest", "open_interest_value",
            "long_short_ratio", "long_pct", "short_pct",
            "fear_greed_index", "fg_ma7", "fg_ma30", "fg_trend", "fg_zscore",
            "active_addresses", "transaction_count", "large_tx_count",
            "whale_volume_ratio", "exchange_inflow", "exchange_outflow",
            "exchange_netflow", "mvrv_ratio", "nvt_signal", "sopr",
            "dormant_circulation", "liveliness",
            "twitter_sentiment", "reddit_sentiment", "news_sentiment",
            "mention_volume", "ai_sentiment_score",
            "btc_dominance", "eth_dominance",
            "total_market_cap", "total_volume_24h",
            "buy_sell_ratio", "quote_volume", "taker_buy_base",
            "order_book_imbalance",
        ]
        for col in passthrough:
            if col in df.columns:
                feat[col] = df[col]

        # ── Interaction / ratio features ──────────────────────────────────
        if "funding_rate" in feat.columns and "open_interest" in feat.columns:
            feat["oi_funding_product"] = feat["open_interest"] * feat["funding_rate"]
        if "mvrv_ratio" in feat.columns:
            feat["mvrv_rsi_product"]   = feat["mvrv_ratio"] * feat["rsi_14"]
        if "whale_volume_ratio" in feat.columns:
            feat["whale_vol_ratio_x_vol_spike"] = (
                feat["whale_volume_ratio"] * feat["vol_spike"])
        if "ai_sentiment_score" in feat.columns:
            feat["sentiment_x_rsi"] = feat["ai_sentiment_score"] * feat["rsi_14"] / 100

        # ── Day/week calendars ────────────────────────────────────────────
        feat["day_of_week"]  = feat.index.dayofweek.astype(float)
        feat["day_of_month"] = feat.index.day.astype(float)
        feat["month"]        = feat.index.month.astype(float)
        feat["quarter"]      = feat.index.quarter.astype(float)
        feat["is_month_end"] = feat.index.is_month_end.astype(float)

        # ── Lagged features ───────────────────────────────────────────────
        key_cols = ["rsi_14", "macd_hist", "bb_pct", "vol_zscore",
                    "funding_rate", "fear_greed_index"]
        for col in key_cols:
            if col in feat.columns:
                for lag in [1, 2, 3]:
                    feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

        # ── Rolling statistics of returns ─────────────────────────────────
        ret  = close.pct_change()
        for w in [7, 14, 30]:
            feat[f"ret_skew_{w}"]  = ret.rolling(w, min_periods=5).skew()
            feat[f"ret_kurt_{w}"]  = ret.rolling(w, min_periods=5).kurt()
            feat[f"ret_mean_{w}"]  = ret.rolling(w, min_periods=5).mean()

        # ── Final cleanup ─────────────────────────────────────────────────
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.ffill().bfill()

        self.logger.info(
            f"Features engineered: {len(feat)} rows × {len(feat.columns)} cols")
        return feat


# ============================================================================
# SECTION 6 – MODEL LOADER
# ============================================================================

class ModelRegistry:
    """
    Loads every saved model artifact from train.py's output directory.
    Supports: pickle (.pkl), Keras (.h5 / SavedModel), PyTorch (.pt).
    """

    def __init__(self, models_dir: str, logger: logging.Logger):
        self.models_dir = Path(models_dir)
        self.logger     = logger
        self.models: Dict[str, Tuple] = {}
        # tuple: (model_obj, is_keras, is_torch, is_clf)

    def load_all(self) -> Dict[str, Tuple]:
        if not self.models_dir.exists():
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return {}

        self.logger.info(f"Loading models from: {self.models_dir}")
        loaded = 0

        # ── .pkl (sklearn / xgb / lgb / catboost / ensemble) ─────────────
        for pkl_path in sorted(self.models_dir.glob("*.pkl")):
            name = pkl_path.stem
            try:
                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)
                is_clf = any(k in name.lower() for k in
                             ("clf", "class", "vote", "stack"))
                self.models[name] = (obj, False, False, is_clf)
                self.logger.info(f"  ✓ Loaded pkl: {name}")
                loaded += 1
            except Exception as e:
                self.logger.warning(f"  ✗ Failed to load {pkl_path.name}: {e}")

        # ── Keras .h5 / SavedModel directories ────────────────────────────
        if TF_AVAILABLE:
            # .h5 files
            for h5_path in sorted(self.models_dir.glob("*.h5")):
                name = h5_path.stem
                try:
                    model = keras.models.load_model(
                        str(h5_path), compile=False,
                        custom_objects=self._custom_objects()
                    )
                    is_clf = any(k in name.lower() for k in
                                 ("clf", "class", "softmax"))
                    self.models[name] = (model, True, False, is_clf)
                    self.logger.info(f"  ✓ Loaded h5: {name}")
                    loaded += 1
                except Exception as e:
                    self.logger.warning(f"  ✗ Failed h5 {h5_path.name}: {e}")

            # SavedModel subdirectories
            for sm_dir in sorted(self.models_dir.iterdir()):
                if sm_dir.is_dir() and (sm_dir / "saved_model.pb").exists():
                    name = sm_dir.name
                    if name in self.models:
                        continue
                    try:
                        model = keras.models.load_model(
                            str(sm_dir), compile=False,
                            custom_objects=self._custom_objects()
                        )
                        is_clf = any(k in name.lower() for k in
                                     ("clf", "class"))
                        self.models[name] = (model, True, False, is_clf)
                        self.logger.info(f"  ✓ Loaded SavedModel: {name}")
                        loaded += 1
                    except Exception as e:
                        self.logger.warning(
                            f"  ✗ Failed SavedModel {name}: {e}")

        # ── PyTorch .pt / .pth ────────────────────────────────────────────
        if TORCH_AVAILABLE:
            for pt_path in sorted(self.models_dir.glob("*.pt")) + \
                           sorted(self.models_dir.glob("*.pth")):
                name = pt_path.stem
                if name in self.models:
                    continue
                try:
                    obj     = torch.load(str(pt_path),
                                         map_location="cpu",
                                         weights_only=False)
                    is_clf  = any(k in name.lower() for k in
                                  ("clf", "class"))
                    self.models[name] = (obj, False, True, is_clf)
                    self.logger.info(f"  ✓ Loaded torch: {name}")
                    loaded += 1
                except Exception as e:
                    self.logger.warning(f"  ✗ Failed pt {pt_path.name}: {e}")

        self.logger.info(f"Total models loaded: {loaded}")
        return self.models

    @staticmethod
    def _custom_objects():
        objs = {}
        if TF_AVAILABLE:
            # Register custom Keras layers from train.py
            class PositionalEncoding(layers.Layer):
                def __init__(self, max_len=1000, d_model=64, **kw):
                    super().__init__(**kw)
                    self.max_len = max_len
                    self.d_model = d_model
                    P   = np.zeros((max_len, d_model))
                    pos = np.arange(max_len)[:, np.newaxis]
                    div = np.exp(np.arange(0, d_model, 2) *
                                 -(np.log(10000.0) / d_model))
                    P[:, 0::2] = np.sin(pos * div)
                    P[:, 1::2] = np.cos(pos * div)
                    self.pe = tf.constant(P[np.newaxis, :, :], dtype=tf.float32)

                def call(self, x):
                    return x + self.pe[:, :tf.shape(x)[1], :]

                def get_config(self):
                    cfg = super().get_config()
                    cfg.update({"max_len": self.max_len,
                                "d_model": self.d_model})
                    return cfg

            class TransformerBlock(layers.Layer):
                def __init__(self, d_model=64, num_heads=4, ff_dim=128,
                             dropout_rate=0.1, **kw):
                    super().__init__(**kw)
                    self.d_model = d_model
                    self.num_heads = num_heads
                    self.ff_dim = ff_dim
                    self.dropout_rate = dropout_rate
                    self.attn  = layers.MultiHeadAttention(
                        num_heads=num_heads,
                        key_dim=max(1, d_model // num_heads))
                    self.ff1   = layers.Dense(ff_dim, activation="relu")
                    self.ff2   = layers.Dense(d_model)
                    self.ln1   = layers.LayerNormalization(epsilon=1e-6)
                    self.ln2   = layers.LayerNormalization(epsilon=1e-6)
                    self.drop1 = layers.Dropout(dropout_rate)
                    self.drop2 = layers.Dropout(dropout_rate)

                def call(self, x, training=False):
                    a = self.attn(x, x, training=training)
                    a = self.drop1(a, training=training)
                    x = self.ln1(x + a)
                    f = self.ff2(self.ff1(x))
                    f = self.drop2(f, training=training)
                    return self.ln2(x + f)

                def get_config(self):
                    cfg = super().get_config()
                    cfg.update({"d_model": self.d_model,
                                "num_heads": self.num_heads,
                                "ff_dim": self.ff_dim,
                                "dropout_rate": self.dropout_rate})
                    return cfg

            objs["PositionalEncoding"] = PositionalEncoding
            objs["TransformerBlock"]   = TransformerBlock
        return objs


# ============================================================================
# SECTION 7 – PREPROCESSOR (mirrors train.py DataPreprocessor)
# ============================================================================

class InferencePreprocessor:
    """
    Loads the scalers saved by train.py and transforms live features
    into the exact same space the trained models expect.
    """

    def __init__(self, train_output_dir: str, config: PredictConfig,
                 logger: logging.Logger):
        self.train_dir  = Path(train_output_dir)
        self.config     = config
        self.logger     = logger
        self.scaler_X:  Optional[Any] = None
        self.scaler_y:  Optional[Any] = None
        self.feature_cols: List[str] = []
        self._load_scalers()
        self._load_feature_list()

    def _load_scalers(self):
        """Load scalers from pickle files produced by train.py."""
        for fname in ["scaler_X.pkl", "scaler_X_flat.pkl",
                      "preprocessor.pkl", "flat_scaler.pkl"]:
            path = self.train_dir / "models" / fname
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.scaler_X = pickle.load(f)
                    self.logger.info(f"Loaded feature scaler: {path.name}")
                    break
                except Exception as e:
                    self.logger.warning(f"Cannot load scaler {fname}: {e}")

        for fname in ["scaler_y.pkl", "scaler_y_reg.pkl", "target_scaler.pkl"]:
            path = self.train_dir / "models" / fname
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.scaler_y = pickle.load(f)
                    self.logger.info(f"Loaded target scaler: {path.name}")
                    break
                except Exception as e:
                    self.logger.warning(f"Cannot load target scaler {fname}: {e}")

        if self.scaler_X is None:
            self.logger.warning(
                "No saved scaler found — using fresh RobustScaler "
                "(accuracy may differ from training)")
            self.scaler_X = RobustScaler()
            self._fresh_scaler = True
        else:
            self._fresh_scaler = False

    def _load_feature_list(self):
        """Recover the exact feature columns used during training."""
        # 1) Check feature_importance.csv — contains training column names
        fi_path = self.train_dir / "feature_importance.csv"
        if fi_path.exists():
            try:
                fi_df = pd.read_csv(fi_path, index_col=0)
                if "feature" in fi_df.columns:
                    self.feature_cols = fi_df["feature"].tolist()
                elif fi_df.index.name == "feature" or fi_df.index.dtype == object:
                    self.feature_cols = fi_df.index.tolist()
                self.logger.info(
                    f"Loaded {len(self.feature_cols)} feature names from "
                    "feature_importance.csv")
                return
            except Exception as e:
                self.logger.warning(f"Cannot parse feature_importance.csv: {e}")

        # 2) Fall back to results_summary.json
        rs_path = self.train_dir / "results_summary.json"
        if rs_path.exists():
            try:
                with open(rs_path) as f:
                    rs = json.load(f)
                n = rs.get("n_features", 0)
                self.logger.info(
                    f"results_summary.json says {n} features were used. "
                    "Feature names not recoverable — will align by position.")
            except Exception:
                pass

    def prepare_flat(self, feat_df: pd.DataFrame,
                     last_n: int = 1) -> np.ndarray:
        """
        Transform the feature DataFrame into a flat 2-D numpy array
        matching the shape expected by tree/linear models.
        """
        df = feat_df.copy()

        # Drop non-numeric / target columns
        drop_cols = ["close_fwd", "label", "ret_fwd", "target",
                     "y_clf", "y_reg"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                     errors="ignore")

        # Align to training columns if known
        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in df.columns]
            extra   = [c for c in df.columns if c not in self.feature_cols]
            if missing:
                for c in missing:
                    df[c] = 0.0
                self.logger.debug(
                    f"Added {len(missing)} missing feature columns (zero-filled)")
            df = df[self.feature_cols]  # enforce exact column order

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        if self._fresh_scaler:
            X = self.scaler_X.fit_transform(df.values)
        else:
            X = self.scaler_X.transform(df.values)

        return X[-last_n:]   # return last N rows

    def prepare_sequences(self, feat_df: pd.DataFrame,
                           seq_len: int,
                           last_n: int = 1) -> np.ndarray:
        """
        Build 3-D sequence array (batch, seq_len, features)
        for LSTM / GRU / Transformer models.
        """
        df = feat_df.copy()
        drop_cols = ["close_fwd", "label", "ret_fwd", "target",
                     "y_clf", "y_reg"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                     errors="ignore")

        if self.feature_cols:
            missing = [c for c in self.feature_cols if c not in df.columns]
            for c in missing:
                df[c] = 0.0
            df = df[self.feature_cols]

        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        if self._fresh_scaler:
            arr = self.scaler_X.fit_transform(df.values)
        else:
            arr = self.scaler_X.transform(df.values)

        seqs = []
        total = len(arr)
        # Build last_n sequences
        for i in range(last_n, 0, -1):
            end   = total - i + 1
            start = end - seq_len
            if start < 0:
                # Pad with zeros at the beginning
                pad   = np.zeros((abs(start), arr.shape[1]))
                chunk = np.vstack([pad, arr[:end]])
            else:
                chunk = arr[start:end]
            seqs.append(chunk[-seq_len:])

        return np.array(seqs)   # (last_n, seq_len, n_features)

    def inverse_transform_return(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.scaler_y is not None:
            return self.scaler_y.inverse_transform(
                y_scaled.reshape(-1, 1)).ravel()
        return y_scaled   # already in return space


# ============================================================================
# SECTION 8 – MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """
    Multi-signal market regime classification.
    Combines trend, volatility, Hurst exponent, and on-chain signals.
    """

    def detect(self, feat_row: pd.Series,
               hist_close: pd.Series) -> Tuple[str, float]:
        signals = {}

        # ── Trend signal from ADX ──────────────────────────────────────
        adx   = feat_row.get("adx", 25)
        di_d  = feat_row.get("di_diff", 0)
        if adx > 25:
            signals["trend"] = 1.0 if di_d > 0 else -1.0
        else:
            signals["trend"] = 0.0

        # ── Momentum direction ────────────────────────────────────────
        ema20 = feat_row.get("ema_20", feat_row.get("close", 0))
        ema50 = feat_row.get("ema_50", ema20)
        if ema50 > 0:
            signals["ema_slope"] = 1.0 if ema20 > ema50 else -1.0
        else:
            signals["ema_slope"] = 0.0

        # ── Volatility regime ─────────────────────────────────────────
        hv    = feat_row.get("hv_21d", 0.5)
        hv63  = feat_row.get("hv_63d", 0.5)
        if hv > hv63 * 1.5:
            signals["vol"] = "high"
        else:
            signals["vol"] = "normal"

        # ── Hurst exponent ────────────────────────────────────────────
        hurst = feat_row.get("hurst", 0.5)

        # ── On-chain / order-flow ─────────────────────────────────────
        netflow = feat_row.get("exchange_netflow", 0)
        whale_r = feat_row.get("whale_volume_ratio", 0)

        # ── Regime decision ───────────────────────────────────────────
        trend_score = signals.get("trend", 0) * 0.4 + \
                      signals.get("ema_slope", 0) * 0.3
        is_high_vol = signals["vol"] == "high"

        if is_high_vol:
            regime = MarketRegime.HIGH_VOLATILITY.value
            conf   = min(0.85, hv / (hv63 + 1e-9) * 0.5)
        elif trend_score > 0.4:
            if netflow < 0 and whale_r > 0.4:
                regime = MarketRegime.ACCUMULATION.value
                conf   = 0.65
            else:
                regime = MarketRegime.TRENDING_UP.value
                conf   = min(0.90, trend_score)
        elif trend_score < -0.4:
            if netflow > 0 and whale_r > 0.5:
                regime = MarketRegime.DISTRIBUTION.value
                conf   = 0.65
            else:
                regime = MarketRegime.TRENDING_DOWN.value
                conf   = min(0.90, abs(trend_score))
        else:
            regime = MarketRegime.RANGING.value
            conf   = 0.60 if hurst < 0.45 else 0.50

        return regime, round(conf, 3)


# ============================================================================
# SECTION 9 – MULTI-HORIZON PREDICTOR
# ============================================================================

class MultiHorizonForecaster:
    """
    Generates predictions for multiple steps ahead using the ensemble.
    Uses iterative recursive forecasting: feed predictions back as input.
    """

    def __init__(self, models: Dict, preprocessor: InferencePreprocessor,
                 config: PredictConfig, logger: logging.Logger):
        self.models       = models
        self.preprocessor = preprocessor
        self.config       = config
        self.logger       = logger

    def forecast(self, feat_df: pd.DataFrame,
                 current_price: float,
                 n_steps: int) -> List[Dict]:
        """
        Returns list of {step, predicted_price, pred_return_pct,
                          prob_up, prob_down, prob_sideways, confidence}
        """
        results    = []
        price      = current_price
        df_rolling = feat_df.copy()

        seq_len    = self.config.sequence_length

        for step in range(1, n_steps + 1):
            try:
                X_flat = self.preprocessor.prepare_flat(df_rolling, last_n=1)
                X_seq  = self.preprocessor.prepare_sequences(
                    df_rolling, seq_len, last_n=1)

                clf_probs_list = []
                reg_preds_list = []

                for name, (model, is_keras, is_torch, is_clf) in \
                        self.models.items():
                    try:
                        if is_keras:
                            raw = model.predict(X_seq, verbose=0)
                        elif is_torch:
                            model.eval()
                            with torch.no_grad():
                                xt  = torch.FloatTensor(X_seq)
                                raw = model(xt).numpy()
                        else:
                            raw = None

                        if is_clf:
                            if is_keras or is_torch:
                                clf_probs_list.append(raw[0])
                            else:
                                if hasattr(model, "predict_proba"):
                                    clf_probs_list.append(
                                        model.predict_proba(X_flat)[0])
                        else:
                            if is_keras or is_torch:
                                reg_preds_list.append(float(raw[0, 0]))
                            else:
                                reg_preds_list.append(
                                    float(model.predict(X_flat)[0]))
                    except Exception:
                        pass

                # Ensemble
                if clf_probs_list:
                    clf_ens = np.mean(clf_probs_list, axis=0)
                    p_dn, p_sw, p_up = (clf_ens[0], clf_ens[1], clf_ens[2]) \
                        if len(clf_ens) == 3 else (0.33, 0.34, 0.33)
                else:
                    p_up, p_dn, p_sw = 0.33, 0.33, 0.34

                if reg_preds_list:
                    pred_ret = np.mean(reg_preds_list)
                    pred_ret = self.preprocessor.inverse_transform_return(
                        np.array([pred_ret]))[0]
                else:
                    pred_ret = (p_up - p_dn) * 0.02

                pred_price  = price * (1 + pred_ret)
                confidence  = max(p_up, p_dn, p_sw)
                direction   = "up" if p_up == confidence else \
                              "down" if p_dn == confidence else "sideways"

                results.append({
                    "step":               step,
                    "predicted_price":    round(pred_price, 6),
                    "predicted_return_pct": round(pred_ret * 100, 4),
                    "prob_up":            round(p_up, 4),
                    "prob_down":          round(p_dn, 4),
                    "prob_sideways":      round(p_sw, 4),
                    "direction":          direction,
                    "confidence":         round(confidence, 4),
                })

                # Update rolling price for next step (simplified recursive)
                price = pred_price

            except Exception as e:
                self.logger.warning(f"Step {step} forecast failed: {e}")
                results.append({
                    "step": step, "predicted_price": price,
                    "predicted_return_pct": 0.0,
                    "prob_up": 0.33, "prob_down": 0.33, "prob_sideways": 0.34,
                    "direction": "sideways", "confidence": 0.33,
                })

        return results


# ============================================================================
# SECTION 10 – SIGNAL COMPOSER
# ============================================================================

class SignalComposer:
    """
    Fuses ensemble prediction, technical signals, sentiment, on-chain,
    and market regime into a single actionable trading signal.
    """

    def __init__(self, config: PredictConfig):
        self.config = config

    def compose(self,
                p_up: float, p_down: float, p_sideways: float,
                pred_return: float,
                direction_conf: float,
                feat_row: pd.Series,
                regime: str) -> Tuple[str, float, float]:
        """
        Returns (signal_str, signal_score, composite_conf).
        signal_score ∈ [-1, +1]
        """
        scores = {}

        # ── 1. Model ensemble vote ────────────────────────────────────────
        model_score = p_up - p_down    # ∈ [-1, 1]
        scores["model_vote"] = model_score * 0.40

        # ── 2. Regression expected return ─────────────────────────────────
        reg_score = np.tanh(pred_return / 0.03)  # normalize ~±1
        scores["regression"] = reg_score * 0.25

        # ── 3. Technical momentum signals ────────────────────────────────
        tech_signals = []
        rsi = feat_row.get("rsi_14", 50)
        if rsi < 30:
            tech_signals.append(+0.6)
        elif rsi > 70:
            tech_signals.append(-0.6)
        else:
            tech_signals.append((50 - rsi) / 50)

        # MACD histogram direction
        macd_hist = feat_row.get("macd_hist", 0)
        macd_roc  = feat_row.get("macd_hist_roc", 0)
        tech_signals.append(np.tanh(macd_hist * 10))
        tech_signals.append(np.sign(macd_roc) * 0.3)

        # Stochastic overbought / oversold
        stoch_k = feat_row.get("stoch_k", 50)
        if stoch_k < 20:
            tech_signals.append(+0.5)
        elif stoch_k > 80:
            tech_signals.append(-0.5)
        else:
            tech_signals.append(0)

        # Bollinger %B extremes
        bb_pct = feat_row.get("bb_pct", 0.5)
        tech_signals.append((0.5 - bb_pct) * 0.6)

        # ADX trend strength
        adx   = feat_row.get("adx", 20)
        di_d  = feat_row.get("di_diff", 0)
        if adx > 25:
            tech_signals.append(np.sign(di_d) * min(1, adx / 50))
        else:
            tech_signals.append(0)

        # PSAR signal
        psar_sig = feat_row.get("psar_signal", 0.5)
        tech_signals.append((psar_sig - 0.5) * 0.4)

        # Ichimoku cloud position
        ichi_above = feat_row.get("ichi_above_cloud", 0.5)
        tech_signals.append((ichi_above - 0.5) * 0.5)

        scores["technical"] = np.mean(tech_signals) * 0.20

        # ── 4. Sentiment ─────────────────────────────────────────────────
        fg = feat_row.get("fear_greed_index", 50)
        ai_sent = feat_row.get("ai_sentiment_score", 0)
        fg_score = np.tanh((fg - 50) / 30)
        sent_score = (fg_score * 0.5 + ai_sent * 0.5)
        scores["sentiment"] = sent_score * 0.08

        # ── 5. On-chain smart money ───────────────────────────────────────
        netflow     = feat_row.get("exchange_netflow", 0)
        whale_ratio = feat_row.get("whale_volume_ratio", 0.3)
        mvrv        = feat_row.get("mvrv_ratio", 1.5)
        sopr        = feat_row.get("sopr", 1.0)

        # Exchange outflow = bullish (coins leaving exchanges)
        netflow_signal = np.tanh(-netflow / 1e6)
        mvrv_signal    = np.tanh((2.0 - mvrv) / 1.5)   # <2 bullish
        sopr_signal    = np.tanh((sopr - 1.0) * 5)      # >1 bullish
        onchain_score  = (netflow_signal * 0.4 +
                          mvrv_signal    * 0.3 +
                          sopr_signal    * 0.3)
        scores["onchain"] = onchain_score * 0.07

        # ── 6. Futures / derivatives signals ─────────────────────────────
        funding   = feat_row.get("funding_rate", 0)
        lsr       = feat_row.get("long_short_ratio", 1.0)
        ob_imb    = feat_row.get("order_book_imbalance", 0)

        # Negative funding = short squeeze potential = bullish
        fund_signal = np.tanh(-funding * 500)
        # High LSR = overleveraged longs = bearish
        lsr_signal  = np.tanh((1.0 - lsr) * 2)
        ob_signal   = np.tanh(ob_imb * 3)
        futures_score = (fund_signal * 0.4 + lsr_signal * 0.3 +
                         ob_signal   * 0.3)
        scores["futures"] = futures_score * 0.0

        # ── Aggregate ─────────────────────────────────────────────────────
        total_score = sum(scores.values())
        total_score = float(np.clip(total_score, -1.0, 1.0))

        # ── Regime modifiers ──────────────────────────────────────────────
        if regime == MarketRegime.HIGH_VOLATILITY.value:
            # Reduce position in high vol
            total_score *= 0.6
        elif regime == MarketRegime.RANGING.value:
            # Mean-revert signals in range
            if abs(total_score) < 0.3:
                total_score *= 0.5

        # ── Signal classification ─────────────────────────────────────────
        cfg = self.config
        if total_score >= 0.50 and direction_conf >= cfg.strong_signal_conf:
            signal = SignalStrength.STRONG_BUY.value
        elif total_score >= 0.20 and direction_conf >= cfg.signal_conf:
            signal = SignalStrength.BUY.value
        elif total_score >= 0.08:
            signal = SignalStrength.WEAK_BUY.value
        elif total_score <= -0.50 and direction_conf >= cfg.strong_signal_conf:
            signal = SignalStrength.STRONG_SELL.value
        elif total_score <= -0.20 and direction_conf >= cfg.signal_conf:
            signal = SignalStrength.SELL.value
        elif total_score <= -0.08:
            signal = SignalStrength.WEAK_SELL.value
        else:
            signal = SignalStrength.HOLD.value

        composite_conf = min(1.0, direction_conf * (0.5 + abs(total_score) * 0.5))
        return signal, round(total_score, 4), round(composite_conf, 4)


# ============================================================================
# SECTION 11 – CONFIDENCE CALIBRATOR
# ============================================================================

class ConfidenceCalibrator:
    """
    Post-hoc calibration: adjusts raw model probabilities using
    data quality, cross-model disagreement, and regime uncertainty.
    """

    @staticmethod
    def calibrate(raw_probs: np.ndarray,
                  model_preds: List[ModelPrediction],
                  data_quality: float = 1.0,
                  regime_conf: float  = 1.0) -> np.ndarray:
        """
        raw_probs: (3,) array [P_down, P_sideways, P_up]
        Returns calibrated (3,) array.
        """
        # ── 1. Cross-model disagreement penalty ─────────────────────────
        if len(model_preds) > 1:
            clf_preds = [m for m in model_preds if m.model_type == "clf"
                         and m.raw_output is not None]
            if len(clf_preds) > 1:
                all_probs = np.array([m.raw_output for m in clf_preds
                                      if hasattr(m.raw_output, "__len__")
                                      and len(m.raw_output) == 3])
                if len(all_probs) > 1:
                    disagreement = np.mean(np.std(all_probs, axis=0))
                    # More disagreement → push toward uniform
                    alpha = min(0.5, disagreement * 3)
                    raw_probs = (1 - alpha) * raw_probs + \
                                alpha * np.array([1/3, 1/3, 1/3])

        # ── 2. Data quality discount ────────────────────────────────────
        if data_quality < 1.0:
            alpha = (1 - data_quality) * 0.5
            raw_probs = (1 - alpha) * raw_probs + \
                        alpha * np.array([1/3, 1/3, 1/3])

        # ── 3. Regime uncertainty ────────────────────────────────────────
        if regime_conf < 0.6:
            alpha = (1 - regime_conf) * 0.3
            raw_probs = (1 - alpha) * raw_probs + \
                        alpha * np.array([1/3, 1/3, 1/3])

        # Renormalize
        raw_probs = np.clip(raw_probs, 1e-6, 1.0)
        return raw_probs / raw_probs.sum()


# ============================================================================
# SECTION 12 – DATA QUALITY ASSESSOR
# ============================================================================

class DataQualityChecker:
    """Scores incoming live data 0→1 for completeness and freshness."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def assess(self, df: pd.DataFrame, feat_df: pd.DataFrame) -> float:
        scores = []

        # ── Freshness ─────────────────────────────────────────────────────
        if len(df) > 0:
            last_ts = df.index[-1]
            if hasattr(last_ts, "tz_localize"):
                last_ts = last_ts
            now = pd.Timestamp.utcnow()
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            age_hrs = (now - last_ts).total_seconds() / 3600
            freshness = max(0, 1 - age_hrs / 48)  # penalize after 48h
            scores.append(freshness)

        # ── Completeness ─────────────────────────────────────────────────
        nan_frac = feat_df.isna().mean().mean()
        scores.append(max(0, 1 - nan_frac * 2))

        # ── Sufficient rows ───────────────────────────────────────────────
        row_score = min(1.0, len(df) / 100)
        scores.append(row_score)

        # ── Volume nonzero ────────────────────────────────────────────────
        if "volume" in df.columns:
            vol_ok = (df["volume"] > 0).mean()
            scores.append(vol_ok)

        quality = float(np.mean(scores)) if scores else 0.5
        self.logger.info(f"Data quality score: {quality:.3f}")
        return round(quality, 3)


# ============================================================================
# SECTION 13 – CORE PREDICTION ENGINE
# ============================================================================

class CryptoPredictionEngine:
    """
    Master inference orchestrator.
    Loads models, fetches live data, engineers features,
    runs ensemble, and produces a full EnsemblePrediction.
    """

    def __init__(self, config: PredictConfig, logger: logging.Logger):
        self.config    = config
        self.logger    = logger

        # Components
        self.data_agg     = LiveDataAggregator(config, logger)
        self.feat_eng     = None    # created after data fetch
        self.model_reg    = ModelRegistry(
            os.path.join(config.train_output_dir, "models"), logger)
        self.preprocessor = InferencePreprocessor(
            config.train_output_dir, config, logger)
        self.regime_det   = MarketRegimeDetector()
        self.signal_comp  = SignalComposer(config)
        self.calibrator   = ConfidenceCalibrator()
        self.quality      = DataQualityChecker(logger)
        self.models: Dict = {}

        # Results summary from training
        self._load_train_summary()
        self._build_model_weights()

    def _load_train_summary(self):
        path = Path(self.config.train_output_dir) / "results_summary.json"
        self.train_summary = {}
        if path.exists():
            try:
                with open(path) as f:
                    self.train_summary = json.load(f)
                self.logger.info("Loaded training results summary.")
            except Exception:
                pass

    def _build_model_weights(self):
        """
        Derive ensemble weights from validation accuracy / R² scores.
        Better-performing models get higher weight.
        """
        clf_w = {}
        reg_w = {}

        ts = self.train_summary
        clf_evals = ts.get("eval_classification", {}).get("classification", {})
        reg_evals = ts.get("eval_regression",     {}).get("regression",     {})

        for m, metrics in clf_evals.items():
            acc = metrics.get("accuracy", 0.5) or 0.5
            clf_w[m] = max(0.01, acc - 0.33)   # above random

        for m, metrics in reg_evals.items():
            r2 = metrics.get("r2", 0) or 0
            reg_w[m] = max(0.01, r2)

        if clf_w:
            total = sum(clf_w.values())
            self.config.clf_model_weights = {k: v/total for k, v in clf_w.items()}
        if reg_w:
            total = sum(reg_w.values())
            self.config.reg_model_weights = {k: v/total for k, v in reg_w.items()}

        self.logger.info(
            f"Ensemble weights: {len(clf_w)} clf, {len(reg_w)} reg models")

    def initialize(self):
        """Load models from disk. Call once before predict()."""
        self.models = self.model_reg.load_all()
        if not self.models:
            self.logger.warning(
                "No trained models found. Predictions will be rule-based only. "
                "Run train.py first to generate model artifacts.")
        self.logger.info(
            f"Engine initialized with {len(self.models)} model(s).")

    # ── Main prediction entry point ────────────────────────────────────────

    def predict(self, symbol: Optional[str] = None,
                timeframe: Optional[str] = None) -> EnsemblePrediction:
        """
        Full prediction cycle:
          1. Fetch live data
          2. Engineer features
          3. Assess data quality
          4. Run all models
          5. Calibrate ensemble
          6. Compose signal
          7. Multi-horizon forecast
          8. Return EnsemblePrediction
        """
        symbol    = symbol    or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        t0        = time.time()

        self.logger.info(
            f"\n{'='*60}\n"
            f"  INFERENCE: {symbol} / {timeframe} — "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"{'='*60}"
        )

        # ── Step 1: Fetch data ─────────────────────────────────────────────
        raw_df = self.data_agg.fetch(symbol, timeframe)

        # ── Step 2: Feature engineering ───────────────────────────────────
        self.feat_eng = LiveFeatureEngineer(raw_df, self.config, self.logger)
        feat_df       = self.feat_eng.engineer()

        # ── Step 3: Data quality ──────────────────────────────────────────
        dq_score = self.quality.assess(raw_df, feat_df)

        # Current price (latest close)
        current_price = float(feat_df["close"].iloc[-1])
        current_ts    = feat_df.index[-1]
        if hasattr(current_ts, "to_pydatetime"):
            current_ts = current_ts.to_pydatetime()

        # Latest feature row for signal composition
        feat_row = feat_df.iloc[-1]

        # ── Step 4: Model inference ───────────────────────────────────────
        seq_len = self.config.sequence_length
        X_flat  = self.preprocessor.prepare_flat(feat_df, last_n=1)
        X_seq   = self.preprocessor.prepare_sequences(feat_df, seq_len, last_n=1)

        clf_probs_list: List[np.ndarray] = []
        reg_preds_list: List[float]      = []
        model_pred_objs: List[ModelPrediction] = []

        for name, (model, is_keras, is_torch, is_clf) in self.models.items():
            t_inf = time.time()
            try:
                if is_keras:
                    raw = model.predict(X_seq, verbose=0)
                elif is_torch:
                    model.eval()
                    with torch.no_grad():
                        xt  = torch.FloatTensor(X_seq)
                        raw = model(xt).numpy()
                else:
                    raw = None

                latency = (time.time() - t_inf) * 1000

                if is_clf:
                    if is_keras or is_torch:
                        probs = raw[0]
                        if len(probs) == 3:
                            clf_probs_list.append(probs)
                    else:
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_flat)[0]
                            clf_probs_list.append(probs)
                        else:
                            probs = np.array([0.33, 0.34, 0.33])
                    mp = ModelPrediction(
                        model_name=name, model_type="clf",
                        is_deep=(is_keras or is_torch),
                        raw_output=probs if "probs" in dir() else None,
                        predicted_class=int(np.argmax(probs)),
                        predicted_prob=float(np.max(probs)),
                        latency_ms=latency,
                        confidence_score=float(np.max(probs))
                    )
                else:
                    if is_keras or is_torch:
                        rv = float(raw[0, 0])
                    else:
                        rv = float(model.predict(X_flat)[0])
                    rv = self.preprocessor.inverse_transform_return(
                        np.array([rv]))[0]
                    reg_preds_list.append(rv)
                    mp = ModelPrediction(
                        model_name=name, model_type="reg",
                        is_deep=(is_keras or is_torch),
                        raw_output=rv,
                        predicted_return=rv,
                        latency_ms=latency,
                        confidence_score=min(1.0, abs(rv) / 0.05)
                    )

                model_pred_objs.append(mp)
                self.logger.debug(
                    f"  {name}: {'clf' if is_clf else 'reg'} → "
                    f"{mp.predicted_class if is_clf else f'{rv:.4f}'} "
                    f"({latency:.1f}ms)")

            except Exception as e:
                self.logger.warning(f"  {name} inference failed: {e}")

        # ── Step 5: Ensemble aggregation ──────────────────────────────────
        if clf_probs_list:
            # Weighted average if weights available
            weights = []
            for mp in model_pred_objs:
                if mp.model_type == "clf":
                    w = self.config.clf_model_weights.get(mp.model_name, 1.0)
                    weights.append(w)

            if weights and len(weights) == len(clf_probs_list):
                w_arr    = np.array(weights) / (sum(weights) + 1e-9)
                clf_ens  = np.sum(
                    [p * w for p, w in zip(clf_probs_list, w_arr)], axis=0)
            else:
                clf_ens = np.mean(clf_probs_list, axis=0)
        else:
            clf_ens = np.array([1/3, 1/3, 1/3])

        if reg_preds_list:
            weights_r = []
            for mp in model_pred_objs:
                if mp.model_type == "reg":
                    w = self.config.reg_model_weights.get(mp.model_name, 1.0)
                    weights_r.append(w)
            if weights_r and len(weights_r) == len(reg_preds_list):
                w_arr  = np.array(weights_r) / (sum(weights_r) + 1e-9)
                reg_ens = float(np.dot(reg_preds_list, w_arr))
            else:
                reg_ens = float(np.mean(reg_preds_list))
            reg_std = float(np.std(reg_preds_list)) if len(reg_preds_list) > 1 else 0.02
        else:
            # Rule-based fallback from clf
            reg_ens = float((clf_ens[2] - clf_ens[0]) * 0.025)
            reg_std = 0.03

        # ── Step 6: Calibration ───────────────────────────────────────────
        regime, regime_conf = self.regime_det.detect(feat_row, feat_df["close"])
        clf_cal = self.calibrator.calibrate(
            clf_ens.copy(), model_pred_objs, dq_score, regime_conf)

        if len(clf_cal) == 3:
            p_dn, p_sw, p_up = clf_cal[0], clf_cal[1], clf_cal[2]
        else:
            p_up, p_dn, p_sw = 0.33, 0.33, 0.34

        max_prob  = max(p_up, p_dn, p_sw)
        direction = ("up"       if p_up == max_prob else
                     "down"     if p_dn == max_prob else "sideways")
        dir_conf  = float(max_prob)

        # ── Confidence intervals (regression) ─────────────────────────────
        pred_price  = current_price * (1 + reg_ens)
        price_sigma = current_price * reg_std
        lower_95    = pred_price - 1.96 * price_sigma
        upper_95    = pred_price + 1.96 * price_sigma

        # ── Step 7: Signal composition ────────────────────────────────────
        signal, signal_score, composite_conf = self.signal_comp.compose(
            p_up, p_dn, p_sw, reg_ens, dir_conf, feat_row, regime)

        # ── Risk parameters ────────────────────────────────────────────────
        entry  = current_price
        sl     = entry * (1 - self.config.stop_loss_pct)
        tp     = entry * (1 + self.config.take_profit_pct)
        rr     = self.config.take_profit_pct / (self.config.stop_loss_pct + 1e-9)

        # ── Context scores ─────────────────────────────────────────────────
        fg_val   = float(feat_row.get("fear_greed_index", 50))
        ai_sent  = float(feat_row.get("ai_sentiment_score", 0))
        funding  = float(feat_row.get("funding_rate", 0))
        corr_btc = float(feat_row.get("corr_btc_30", 0.5))
        altseason = float(feat_row.get("altseason", 0))
        exchange_netflow = float(feat_row.get("exchange_netflow", 0))
        oi_chg   = float(feat_row.get("open_interest", 0))

        # Composite sentiment / on-chain scores (normalized -1 to 1)
        sent_score_norm = np.tanh((fg_val - 50) / 30) * 0.5 + ai_sent * 0.5
        oc_nf           = np.tanh(-exchange_netflow / 1e6)
        mvrv_v          = float(feat_row.get("mvrv_ratio", 1.5))
        sopr_v          = float(feat_row.get("sopr", 1.0))
        onchain_norm    = (oc_nf * 0.4 +
                           np.tanh((2.0 - mvrv_v) / 1.5) * 0.3 +
                           np.tanh((sopr_v - 1.0) * 5)   * 0.3)

        # ── Step 8: Multi-horizon forecast ────────────────────────────────
        mh_forecaster = MultiHorizonForecaster(
            self.models, self.preprocessor, self.config, self.logger)
        horizon_preds = mh_forecaster.forecast(
            feat_df, current_price,
            n_steps=min(self.config.prediction_horizon, 12))

        # ── Build prediction ID ────────────────────────────────────────────
        pred_id = hashlib.md5(
            f"{symbol}{timeframe}{current_ts}{current_price}".encode()
        ).hexdigest()[:12]

        # ── Assemble EnsemblePrediction ────────────────────────────────────
        result = EnsemblePrediction(
            symbol               = symbol,
            timestamp            = current_ts,
            timeframe            = timeframe,
            current_price        = round(current_price, 8),

            prob_up              = round(p_up,    4),
            prob_down            = round(p_dn,    4),
            prob_sideways        = round(p_sw,    4),
            predicted_direction  = direction,
            direction_confidence = round(dir_conf,   4),

            predicted_return_pct = round(reg_ens * 100,  4),
            predicted_price      = round(pred_price,      8),
            price_lower_bound    = round(lower_95,        8),
            price_upper_bound    = round(upper_95,        8),
            return_std           = round(reg_std * 100,   4),

            signal               = signal,
            signal_score         = round(signal_score,    4),
            entry_price          = round(entry, 8),
            stop_loss            = round(sl, 8),
            take_profit          = round(tp, 8),
            risk_reward          = round(rr, 2),
            position_size_pct    = self.config.position_size_pct,

            market_regime        = regime,
            regime_confidence    = round(regime_conf,     3),
            btc_correlation      = round(corr_btc,        4),
            altseason_score      = round(float(altseason),4),
            sentiment_score      = round(float(sent_score_norm), 4),
            onchain_score        = round(float(onchain_norm),    4),
            fear_greed_index     = round(fg_val,           2),
            funding_rate         = round(funding,          6),
            open_interest_chg    = round(oi_chg,           2),

            horizon_predictions  = horizon_preds,
            model_predictions    = [
                {
                    "model":      mp.model_name,
                    "type":       mp.model_type,
                    "is_deep":    mp.is_deep,
                    "class":      mp.predicted_class,
                    "prob":       round(mp.predicted_prob or 0, 4),
                    "ret":        round(mp.predicted_return or 0, 6),
                    "latency_ms": round(mp.latency_ms, 1),
                }
                for mp in model_pred_objs
            ],
            n_models_used        = len(model_pred_objs),
            feature_count        = len(feat_df.columns),
            data_quality_score   = dq_score,
            prediction_id        = pred_id,
        )

        elapsed = time.time() - t0
        self.logger.info(
            f"\n{'─'*60}\n"
            f"  PREDICTION COMPLETE ({elapsed:.2f}s)\n"
            f"  Price: ${current_price:,.4f}\n"
            f"  Direction: {direction.upper()} "
            f"(P_up={p_up:.2%} P_dn={p_dn:.2%} P_sw={p_sw:.2%})\n"
            f"  Predicted Return: {reg_ens*100:+.2f}%\n"
            f"  Target Price: ${pred_price:,.4f} "
            f"[{lower_95:,.4f} – {upper_95:,.4f}]\n"
            f"  Signal: {signal}  (score={signal_score:+.3f}, "
            f"conf={composite_conf:.2%})\n"
            f"  Regime: {regime} (conf={regime_conf:.2%})\n"
            f"  Models used: {len(model_pred_objs)}\n"
            f"{'─'*60}"
        )
        return result


# ============================================================================
# SECTION 14 – OUTPUT FORMATTERS & EXPORTERS
# ============================================================================

class PredictionExporter:
    """Saves EnsemblePrediction to JSON, CSV, JSONL formats."""

    def __init__(self, output_dir: str, logger: logging.Logger):
        self.out_dir = Path(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger  = logger

    def _to_dict(self, pred: EnsemblePrediction) -> Dict:
        d = asdict(pred)
        d["timestamp"] = pred.timestamp.isoformat() \
            if hasattr(pred.timestamp, "isoformat") else str(pred.timestamp)
        return d

    def export_json(self, pred: EnsemblePrediction) -> str:
        d    = self._to_dict(pred)
        path = self.out_dir / f"live_prediction_{pred.symbol}.json"
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)
        self.logger.info(f"JSON prediction saved → {path}")
        return str(path)

    def export_csv(self, pred: EnsemblePrediction) -> str:
        # Flatten: exclude nested lists for main row
        d   = self._to_dict(pred)
        flat = {k: v for k, v in d.items()
                if not isinstance(v, (list, dict))}
        df  = pd.DataFrame([flat])
        path = self.out_dir / f"live_prediction_{pred.symbol}.csv"
        df.to_csv(path, index=False)
        self.logger.info(f"CSV prediction saved → {path}")
        return str(path)

    def export_horizon_csv(self, pred: EnsemblePrediction) -> str:
        if not pred.horizon_predictions:
            return ""
        df   = pd.DataFrame(pred.horizon_predictions)
        df.insert(0, "symbol",        pred.symbol)
        df.insert(1, "base_timestamp", str(pred.timestamp))
        df.insert(2, "base_price",     pred.current_price)
        path = self.out_dir / f"multi_horizon_{pred.symbol}.csv"
        df.to_csv(path, index=False)
        self.logger.info(f"Horizon CSV saved → {path}")
        return str(path)

    def export_signal_report(self, pred: EnsemblePrediction) -> str:
        report = {
            "prediction_id":      pred.prediction_id,
            "symbol":             pred.symbol,
            "timestamp":          str(pred.timestamp),
            "timeframe":          pred.timeframe,
            "current_price":      pred.current_price,
            "signal":             pred.signal,
            "signal_score":       pred.signal_score,
            "direction":          pred.predicted_direction,
            "direction_confidence": pred.direction_confidence,
            "predicted_return_pct": pred.predicted_return_pct,
            "predicted_price":    pred.predicted_price,
            "price_lower_95":     pred.price_lower_bound,
            "price_upper_95":     pred.price_upper_bound,
            "entry_price":        pred.entry_price,
            "stop_loss":          pred.stop_loss,
            "take_profit":        pred.take_profit,
            "risk_reward":        pred.risk_reward,
            "position_size_pct":  pred.position_size_pct,
            "market_regime":      pred.market_regime,
            "regime_confidence":  pred.regime_confidence,
            "fear_greed_index":   pred.fear_greed_index,
            "funding_rate":       pred.funding_rate,
            "btc_correlation":    pred.btc_correlation,
            "altseason_score":    pred.altseason_score,
            "sentiment_score":    pred.sentiment_score,
            "onchain_score":      pred.onchain_score,
            "n_models_used":      pred.n_models_used,
            "data_quality_score": pred.data_quality_score,
        }
        path = self.out_dir / f"signal_report_{pred.symbol}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        self.logger.info(f"Signal report saved → {path}")
        return str(path)

    def append_stream_log(self, pred: EnsemblePrediction) -> str:
        """Append one prediction to a JSONL stream log."""
        path = self.out_dir / f"stream_log_{pred.symbol}.jsonl"
        line = json.dumps(self._to_dict(pred), default=str)
        with open(path, "a") as f:
            f.write(line + "\n")
        return str(path)

    def print_terminal_report(self, pred: EnsemblePrediction):
        """Rich console display of prediction results."""
        RESET  = "\033[0m"
        BOLD   = "\033[1m"
        GREEN  = "\033[92m"
        RED    = "\033[91m"
        YELLOW = "\033[93m"
        CYAN   = "\033[96m"
        BLUE   = "\033[94m"
        MAGENTA= "\033[95m"
        WHITE  = "\033[97m"

        # Signal colour
        sig = pred.signal
        if "BUY" in sig:
            sig_col = GREEN
        elif "SELL" in sig:
            sig_col = RED
        else:
            sig_col = YELLOW

        # Direction colour
        dir_col = GREEN if pred.predicted_direction == "up" else \
                  RED   if pred.predicted_direction == "down" else YELLOW

        w = 70
        border = "═" * w

        print(f"\n{BOLD}{CYAN}{'╔' + border + '╗'}{RESET}")
        print(f"{BOLD}{CYAN}║{WHITE}{'  CRYPTO PREDICTION ENGINE — LIVE INFERENCE':^{w}}{'║'}{RESET}")
        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")

        def row(label, value, color=WHITE):
            label_str = f"  {label:<28}"
            val_str   = str(value)
            pad       = w - len(label_str) - len(val_str) - 1
            pad       = max(pad, 1)
            print(f"{BOLD}{CYAN}║{RESET}{BOLD}{label_str}{RESET}"
                  f"{color}{val_str}{' ' * pad}{RESET}{BOLD}{CYAN}║{RESET}")

        row("Symbol / Timeframe",
            f"{pred.symbol} / {pred.timeframe}")
        row("Timestamp",
            str(pred.timestamp)[:19] + " UTC")
        row("Prediction ID",         pred.prediction_id)
        row("Models Used",           pred.n_models_used)
        row("Features",              pred.feature_count)
        row("Data Quality",          f"{pred.data_quality_score:.1%}")

        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
        print(f"{BOLD}{CYAN}║{WHITE}{'  PRICE':^{w}}{'║'}{RESET}")
        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")

        row("Current Price",         f"${pred.current_price:,.6f}")
        row("Predicted Price",        f"${pred.predicted_price:,.6f}", dir_col)
        row("Expected Return",
            f"{pred.predicted_return_pct:+.2f}%", dir_col)
        row("95% CI",
            f"[${pred.price_lower_bound:,.6f} – ${pred.price_upper_bound:,.6f}]")
        row("Return Std (models)",    f"±{pred.return_std:.2f}%")

        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
        print(f"{BOLD}{CYAN}║{WHITE}{'  DIRECTION PROBABILITIES':^{w}}{'║'}{RESET}")
        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")

        row("↑  Probability UP",
            f"{pred.prob_up:.2%}", GREEN)
        row("→  Probability SIDEWAYS",
            f"{pred.prob_sideways:.2%}", YELLOW)
        row("↓  Probability DOWN",
            f"{pred.prob_down:.2%}", RED)
        row("Predicted Direction",
            pred.predicted_direction.upper(), dir_col)
        row("Direction Confidence",
            f"{pred.direction_confidence:.2%}", dir_col)

        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
        print(f"{BOLD}{CYAN}║{WHITE}{'  TRADING SIGNAL':^{w}}{'║'}{RESET}")
        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")

        row("🚦 Signal",             sig, sig_col)
        row("Signal Score",          f"{pred.signal_score:+.4f}", sig_col)
        row("Entry Price",           f"${pred.entry_price:,.6f}")
        row("Stop Loss",             f"${pred.stop_loss:,.6f}  "
                                     f"(-{self.config.stop_loss_pct:.1%})",  RED)
        row("Take Profit",           f"${pred.take_profit:,.6f}  "
                                     f"(+{self.config.take_profit_pct:.1%})", GREEN)
        row("Risk / Reward",         f"1 : {pred.risk_reward:.1f}")
        row("Position Size",         f"{pred.position_size_pct:.2%} of capital")

        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
        print(f"{BOLD}{CYAN}║{WHITE}{'  MARKET CONTEXT':^{w}}{'║'}{RESET}")
        print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")

        row("Market Regime",         pred.market_regime, MAGENTA)
        row("Regime Confidence",     f"{pred.regime_confidence:.2%}")
        row("Fear & Greed Index",    f"{pred.fear_greed_index:.0f} / 100")
        row("Funding Rate",          f"{pred.funding_rate:.4%}")
        row("BTC Correlation (30d)", f"{pred.btc_correlation:.4f}")
        row("Altseason Score",       f"{pred.altseason_score:+.4f}")
        row("Sentiment Score",       f"{pred.sentiment_score:+.4f}")
        row("On-Chain Score",        f"{pred.onchain_score:+.4f}")

        if pred.horizon_predictions:
            print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
            print(f"{BOLD}{CYAN}║{WHITE}{'  MULTI-HORIZON FORECAST':^{w}}{'║'}{RESET}")
            print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
            for h in pred.horizon_predictions[:6]:
                col  = GREEN if h["direction"] == "up" else \
                       RED   if h["direction"] == "down" else YELLOW
                row(f"  Step +{h['step']}",
                    f"${h['predicted_price']:,.4f}  "
                    f"({h['predicted_return_pct']:+.2f}%)  "
                    f"conf={h['confidence']:.2%}",
                    col)

        if pred.model_predictions:
            print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
            print(f"{BOLD}{CYAN}║{WHITE}{'  PER-MODEL BREAKDOWN':^{w}}{'║'}{RESET}")
            print(f"{BOLD}{CYAN}{'╠' + border + '╣'}{RESET}")
            for mp in pred.model_predictions[:12]:
                if mp["type"] == "clf":
                    val = (f"class={mp['class']}  "
                           f"P={mp['prob']:.2%}  "
                           f"{mp['latency_ms']:.0f}ms")
                else:
                    val = (f"ret={mp['ret']*100:+.3f}%  "
                           f"{mp['latency_ms']:.0f}ms")
                row(f"  {mp['model'][:26]}", val)

        print(f"{BOLD}{CYAN}{'╚' + border + '╝'}{RESET}\n")

        # Expose config ref in print method
        self.config = None  # not available here; accessed via outer scope


# ============================================================================
# SECTION 15 – STREAMING / REAL-TIME MODE
# ============================================================================

class StreamingPredictor:
    """
    Continuously fetches live data and runs predictions at a set interval.
    Maintains a rolling state for consecutive-signal tracking.
    """

    def __init__(self, engine: CryptoPredictionEngine,
                 exporter: PredictionExporter,
                 config: PredictConfig,
                 logger: logging.Logger):
        self.engine   = engine
        self.exporter = exporter
        self.config   = config
        self.logger   = logger
        self.state    = RealTimeState(
            symbol       = config.symbol,
            price_buffer = deque(maxlen=200),
            volume_buffer= deque(maxlen=200),
            pred_buffer  = deque(maxlen=50),
        )
        self._stop_event = threading.Event()

    def start(self):
        self.logger.info(
            f"\n{'='*60}\n"
            f"  STREAMING MODE STARTED\n"
            f"  Symbol: {self.config.symbol}/{self.config.timeframe}\n"
            f"  Refresh: {self.config.stream_interval_sec}s\n"
            f"  Press Ctrl+C to stop.\n"
            f"{'='*60}\n"
        )
        while not self._stop_event.is_set():
            try:
                pred = self.engine.predict()
                self._update_state(pred)
                self._check_alert(pred)
                self.exporter.append_stream_log(pred)
                self.exporter.export_json(pred)
                self.exporter.export_signal_report(pred)
                self.exporter.print_terminal_report(pred)

            except KeyboardInterrupt:
                self.logger.info("Stream stopped by user.")
                break
            except Exception as e:
                self.logger.error(f"Stream cycle error: {e}\n{traceback.format_exc()}")

            # Wait for next cycle
            self.logger.info(
                f"  Next refresh in {self.config.stream_interval_sec}s …")
            self._stop_event.wait(timeout=self.config.stream_interval_sec)

    def stop(self):
        self._stop_event.set()

    def _update_state(self, pred: EnsemblePrediction):
        s = self.state
        s.price_buffer.append(pred.current_price)
        s.pred_buffer.append(pred)
        s.last_updated = datetime.utcnow()

        # Track consecutive signals
        if pred.signal == s.last_signal and pred.signal != "HOLD":
            s.consecutive_signals += 1
        else:
            s.consecutive_signals = 1
        s.last_signal = pred.signal

    def _check_alert(self, pred: EnsemblePrediction):
        """Print alert if a strong or consecutive signal fires."""
        if self.state.consecutive_signals >= 3:
            self.logger.warning(
                f"⚠️  CONSECUTIVE SIGNAL ALERT: {pred.signal} × "
                f"{self.state.consecutive_signals} bars for {pred.symbol}"
            )
        if pred.signal in (SignalStrength.STRONG_BUY.value,
                           SignalStrength.STRONG_SELL.value):
            self.logger.warning(
                f"🔔 STRONG SIGNAL: {pred.signal} on {pred.symbol} "
                f"@ ${pred.current_price:,.4f} | "
                f"Target: ${pred.predicted_price:,.4f}"
            )


# ============================================================================
# SECTION 16 – BACKFILL PREDICTOR
# ============================================================================

class BackfillPredictor:
    """
    Runs inference on the last N completed candles and compares
    predictions vs actual outcomes (where available).
    """

    def __init__(self, engine: CryptoPredictionEngine,
                 exporter: PredictionExporter,
                 config: PredictConfig,
                 logger: logging.Logger):
        self.engine   = engine
        self.exporter = exporter
        self.config   = config
        self.logger   = logger

    def run(self, n_candles: int) -> pd.DataFrame:
        self.logger.info(f"Backfill: running inference on last {n_candles} candles")

        # Fetch enough data to cover the backfill
        cfg_copy = copy.deepcopy(self.config)
        cfg_copy.lookback_days = max(
            cfg_copy.lookback_days, n_candles * 2 + 100)

        raw_df   = self.engine.data_agg.fetch(
            self.config.symbol, self.config.timeframe)
        feat_eng = LiveFeatureEngineer(raw_df, self.config, self.logger)
        feat_df  = feat_eng.engineer()

        seq_len  = self.config.sequence_length
        records  = []

        for i in range(n_candles, 0, -1):
            # Use rows up to (but not including) the last i-th row
            end_idx = len(feat_df) - i
            if end_idx < seq_len + 1:
                continue
            sub_feat = feat_df.iloc[:end_idx]
            sub_raw  = raw_df.iloc[:end_idx]

            current_price = float(sub_feat["close"].iloc[-1])
            current_ts    = sub_feat.index[-1]

            # Actual future price (one step ahead if available)
            future_idx = end_idx
            if future_idx < len(feat_df):
                actual_price = float(feat_df["close"].iloc[future_idx])
                actual_return = (actual_price - current_price) / \
                                (current_price + 1e-9)
            else:
                actual_price  = None
                actual_return = None

            # Run inference
            try:
                X_flat = self.engine.preprocessor.prepare_flat(
                    sub_feat, last_n=1)
                X_seq  = self.engine.preprocessor.prepare_sequences(
                    sub_feat, seq_len, last_n=1)

                clf_probs = []
                reg_preds = []
                for name, (model, is_keras, is_torch, is_clf) in \
                        self.engine.models.items():
                    try:
                        if is_keras:
                            raw = model.predict(X_seq, verbose=0)
                        elif is_torch:
                            model.eval()
                            with torch.no_grad():
                                xt  = torch.FloatTensor(X_seq)
                                raw = model(xt).numpy()
                        else:
                            raw = None

                        if is_clf:
                            if is_keras or is_torch:
                                clf_probs.append(raw[0])
                            else:
                                if hasattr(model, "predict_proba"):
                                    clf_probs.append(
                                        model.predict_proba(X_flat)[0])
                        else:
                            if is_keras or is_torch:
                                reg_preds.append(float(raw[0, 0]))
                            else:
                                reg_preds.append(float(model.predict(X_flat)[0]))
                    except Exception:
                        pass

                if clf_probs:
                    ens = np.mean(clf_probs, axis=0)
                    p_dn, p_sw, p_up = ens[0], ens[1], ens[2]
                else:
                    p_up, p_dn, p_sw = 0.33, 0.33, 0.34

                if reg_preds:
                    pred_ret = float(np.mean(reg_preds))
                    pred_ret = self.engine.preprocessor.inverse_transform_return(
                        np.array([pred_ret]))[0]
                else:
                    pred_ret = (p_up - p_dn) * 0.02

                pred_price = current_price * (1 + pred_ret)
                direction  = ("up"       if p_up > p_dn and p_up > p_sw else
                              "down"     if p_dn > p_up and p_dn > p_sw else
                              "sideways")

                # Evaluate accuracy
                if actual_return is not None:
                    label      = ("up"   if actual_return >  0.015 else
                                  "down" if actual_return < -0.015 else
                                  "sideways")
                    correct    = int(direction == label)
                    ret_error  = abs(pred_ret - actual_return)
                else:
                    label   = None
                    correct = None
                    ret_error = None

                records.append({
                    "timestamp":         str(current_ts),
                    "current_price":     current_price,
                    "actual_price":      actual_price,
                    "actual_return_pct": round(actual_return * 100, 4)
                                         if actual_return is not None else None,
                    "actual_direction":  label,
                    "pred_direction":    direction,
                    "correct":           correct,
                    "pred_return_pct":   round(pred_ret * 100, 4),
                    "pred_price":        round(pred_price, 6),
                    "ret_abs_error":     round(ret_error * 100, 4)
                                         if ret_error is not None else None,
                    "prob_up":           round(p_up, 4),
                    "prob_down":         round(p_dn, 4),
                    "prob_sideways":     round(p_sw, 4),
                    "n_models":          len(self.engine.models),
                })
                self.logger.info(
                    f"  [{i:3d}] {str(current_ts)[:10]} "
                    f"price=${current_price:>10.4f}  "
                    f"pred={direction:>9}  "
                    f"actual={label or 'N/A':>9}  "
                    f"{'✓' if correct else ('✗' if correct is not None else '?')}"
                )
            except Exception as e:
                self.logger.warning(f"Backfill step failed at i={i}: {e}")

        df = pd.DataFrame(records)

        # Summary statistics
        if not df.empty and "correct" in df.columns:
            valid   = df["correct"].dropna()
            acc     = valid.mean() if len(valid) > 0 else 0
            n_valid = len(valid)
            self.logger.info(
                f"\nBackfill summary: {n_valid} predictions, "
                f"accuracy={acc:.2%}"
            )

        # Save
        path = Path(self.exporter.out_dir) / \
               f"backfill_{self.config.symbol}.csv"
        df.to_csv(path, index=False)
        self.logger.info(f"Backfill results saved → {path}")
        return df


# ============================================================================
# SECTION 17 – CLI & ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Crypto Prediction Inference Engine"
    )
    parser.add_argument("--symbol",     type=str,   default="RNDR",
                        help="Trading symbol (e.g. RNDR, BTC, ETH)")
    parser.add_argument("--tf",         type=str,   default="1d",
                        help="Timeframe: 1m 5m 15m 1h 4h 1d 1w")
    parser.add_argument("--horizon",    type=int,   default=12,
                        help="Multi-step prediction horizon")
    parser.add_argument("--seq-len",    type=int,   default=60,
                        help="Sequence length (must match train.py)")
    parser.add_argument("--days",       type=int,   default=180,
                        help="Lookback days for feature computation")
    parser.add_argument("--models-dir", type=str,   default=None,
                        help="Override model directory path")
    parser.add_argument("--train-dir",  type=str,
                        default="/tmp/crypto_train_output",
                        help="Training output directory (from train.py)")
    parser.add_argument("--output",     type=str,
                        default="/tmp/crypto_predict_output",
                        help="Directory for prediction outputs")
    parser.add_argument("--export",     type=str,   default="both",
                        choices=["json", "csv", "both", "none"],
                        help="Export format")
    parser.add_argument("--stream",     action="store_true",
                        help="Enable real-time streaming mode")
    parser.add_argument("--stream-interval", type=int, default=300,
                        help="Seconds between stream refreshes")
    parser.add_argument("--backfill",   type=int,   default=0,
                        help="Run inference on last N historical candles")
    parser.add_argument("--stop-loss",  type=float, default=0.05,
                        help="Stop loss fraction (e.g. 0.05 = 5%%)")
    parser.add_argument("--take-profit",type=float, default=0.10,
                        help="Take profit fraction (e.g. 0.10 = 10%%)")
    parser.add_argument("--min-conf",   type=float, default=0.52,
                        help="Minimum confidence to output a signal")
    parser.add_argument("--no-gpu",     action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--quiet",      action="store_true",
                        help="Suppress terminal prediction table")
    return parser.parse_args()


def build_config(args) -> PredictConfig:
    cfg = PredictConfig()
    cfg.symbol             = args.symbol.upper()
    cfg.timeframe          = args.tf
    cfg.prediction_horizon = args.horizon
    cfg.sequence_length    = args.seq_len
    cfg.lookback_days      = args.days
    cfg.predict_output_dir = args.output
    cfg.train_output_dir   = args.train_dir
    cfg.stop_loss_pct      = args.stop_loss
    cfg.take_profit_pct    = args.take_profit
    cfg.min_confidence     = args.min_conf
    cfg.stream_interval_sec= args.stream_interval

    # Override model dir
    if args.models_dir:
        # Hack: patch the model registry path
        cfg._models_dir_override = args.models_dir

    # GPU config for TF
    if TF_AVAILABLE and args.no_gpu:
        tf.config.set_visible_devices([], "GPU")

    # Load API keys from env / .env
    env_path = Path(".env")
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass
    cfg = PredictConfig.from_env(cfg)

    return cfg


def main():
    args   = parse_args()
    config = build_config(args)

    # ── GPU memory growth ────────────────────────────────────────────────
    if TF_AVAILABLE and not args.no_gpu:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    # ── Logger + output dir ───────────────────────────────────────────────
    os.makedirs(config.predict_output_dir, exist_ok=True)
    logger = setup_logging(config.predict_output_dir)

    logger.info("=" * 60)
    logger.info("  CRYPTO PREDICTION INFERENCE ENGINE  v2.0")
    logger.info(f"  Symbol: {config.symbol}  |  Timeframe: {config.timeframe}")
    logger.info(f"  TensorFlow: {'✓' if TF_AVAILABLE else '✗'}  "
                f"PyTorch: {'✓' if TORCH_AVAILABLE else '✗'}  "
                f"XGB: {'✓' if XGB_AVAILABLE else '✗'}  "
                f"LGB: {'✓' if LGB_AVAILABLE else '✗'}")
    logger.info("=" * 60)

    # ── Build engine ──────────────────────────────────────────────────────
    engine   = CryptoPredictionEngine(config, logger)

    # Apply models dir override if set
    if hasattr(config, "_models_dir_override") and config._models_dir_override:
        engine.model_reg = ModelRegistry(
            config._models_dir_override, logger)

    engine.initialize()

    exporter = PredictionExporter(config.predict_output_dir, logger)

    # ── Backfill mode ──────────────────────────────────────────────────────
    if args.backfill > 0:
        bf = BackfillPredictor(engine, exporter, config, logger)
        bf.run(args.backfill)
        return

    # ── Stream mode ────────────────────────────────────────────────────────
    if args.stream:
        streamer = StreamingPredictor(engine, exporter, config, logger)
        try:
            streamer.start()
        except KeyboardInterrupt:
            streamer.stop()
            logger.info("Stream stopped.")
        return

    # ── Single prediction ─────────────────────────────────────────────────
    pred = engine.predict()

    # Terminal display
    if not args.quiet:
        # Print report — inject config ref
        rep = PredictionExporter(config.predict_output_dir, logger)
        rep.config = config
        rep.print_terminal_report(pred)

        # Fix missing config ref in method
        class _Rep(PredictionExporter):
            def print_terminal_report(self, pred):
                super().print_terminal_report(pred)

    # Fix: print_terminal_report references self.config set inside function
    # Patch directly
    _printer = type("_P", (PredictionExporter,), {})(
        config.predict_output_dir, logger)
    _printer.config = config
    if not args.quiet:
        _printer.print_terminal_report(pred)

    # Export
    if args.export in ("json", "both"):
        exporter.export_json(pred)
    if args.export in ("csv", "both"):
        exporter.export_csv(pred)
        exporter.export_horizon_csv(pred)
    exporter.export_signal_report(pred)

    logger.info(f"\n✓ Prediction complete. Output: {config.predict_output_dir}")
    return pred


# ============================================================================
# SECTION 18 – PROGRAMMATIC API (for app.py)
# ============================================================================

class PredictAPI:
    """
    Clean programmatic interface for app.py integration.
    Avoids CLI overhead; maintains a singleton engine instance.

    Usage:
        api = PredictAPI(train_output_dir="/tmp/crypto_train_output")
        api.initialize()
        pred = api.predict("RNDR", "1d")
        print(pred.signal, pred.predicted_price)
    """

    _instances: Dict[str, "PredictAPI"] = {}

    def __init__(self, train_output_dir: str = "/tmp/crypto_train_output",
                 predict_output_dir: str = "/tmp/crypto_predict_output",
                 symbol: str = "RNDR",
                 timeframe: str = "1d",
                 sequence_length: int = 60,
                 prediction_horizon: int = 12):
        cfg = PredictConfig()
        cfg.train_output_dir   = train_output_dir
        cfg.predict_output_dir = predict_output_dir
        cfg.symbol             = symbol
        cfg.timeframe          = timeframe
        cfg.sequence_length    = sequence_length
        cfg.prediction_horizon = prediction_horizon

        os.makedirs(predict_output_dir, exist_ok=True)
        self.logger   = setup_logging(predict_output_dir)
        self.engine   = CryptoPredictionEngine(cfg, self.logger)
        self.exporter = PredictionExporter(predict_output_dir, self.logger)
        self.config   = cfg
        self._ready   = False

    def initialize(self) -> "PredictAPI":
        self.engine.initialize()
        self._ready = True
        return self

    def predict(self, symbol: Optional[str] = None,
                timeframe: Optional[str] = None) -> EnsemblePrediction:
        if not self._ready:
            self.initialize()
        return self.engine.predict(symbol, timeframe)

    def predict_json(self, symbol: Optional[str] = None,
                     timeframe: Optional[str] = None) -> str:
        pred = self.predict(symbol, timeframe)
        return json.dumps(asdict(pred), default=str, indent=2)

    def predict_dict(self, symbol: Optional[str] = None,
                     timeframe: Optional[str] = None) -> Dict:
        pred = self.predict(symbol, timeframe)
        return asdict(pred)

    def backfill(self, n_candles: int = 30) -> pd.DataFrame:
        if not self._ready:
            self.initialize()
        bf = BackfillPredictor(
            self.engine, self.exporter, self.config, self.logger)
        return bf.run(n_candles)

    @classmethod
    def get_or_create(cls, key: str = "default", **kwargs) -> "PredictAPI":
        """Singleton factory — reuse engine across calls."""
        if key not in cls._instances:
            cls._instances[key] = cls(**kwargs).initialize()
        return cls._instances[key]


if __name__ == "__main__":
    main()
