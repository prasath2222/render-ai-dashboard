"""
================================================================================
  ADVANCED CRYPTO PREDICTION TRAINING PIPELINE  -  train.py
  Full Single-File Version  |  RNDR & Related Assets
  Classification + Regression | All Major ML/DL Models | Ensemble Stacking
================================================================================

HOW TO USE:
    python train.py                          # Full pipeline (RNDR, 1d)
    python train.py --symbol BTC --tf 4h    # Custom symbol & timeframe
    python train.py --fast                   # Quick demo (fewer epochs/trials)
    python train.py --task classification   # Classification only
    python train.py --task regression       # Regression only

OUTPUTS:
    /tmp/crypto_train_output/
        results_summary.json
        predictions_<SYMBOL>.csv
        signals_<SYMBOL>.csv
        backtest_<SYMBOL>.csv
        feature_importance.csv
        models/ (saved .pkl and .h5 files)
        logs/train.log
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
import asyncio
import hashlib
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from collections import defaultdict
import random

warnings.filterwarnings("ignore")

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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    r2_score, mean_absolute_error, mean_squared_error,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    AdaBoostRegressor, AdaBoostClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, mutual_info_classif

# ============================================================================
# GRADIENT BOOSTING
# ============================================================================
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[WARN] xgboost not installed. Skipping XGBoost models.")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("[WARN] lightgbm not installed. Skipping LightGBM models.")

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CAT_AVAILABLE = True
except ImportError:
    CAT_AVAILABLE = False
    print("[WARN] catboost not installed. Skipping CatBoost models.")

# ============================================================================
# DEEP LEARNING - TENSORFLOW / KERAS
# ============================================================================
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, backend as K
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
        TensorBoard, LearningRateScheduler
    )
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
    TF_AVAILABLE = True
    # Suppress TF logs
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError:
    print("[WARN] TensorFlow not installed. Skipping TF/Keras models.")

# ============================================================================
# PYTORCH
# ============================================================================
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import Adam as TorchAdam
    from torch.optim.lr_scheduler import ReduceLROnPlateau as TorchReduceLR
    TORCH_AVAILABLE = True
except ImportError:
    print("[WARN] PyTorch not installed. Skipping PyTorch models.")

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    print("[WARN] optuna not installed. Skipping hyperparameter tuning.")

# ============================================================================
# HTTP / ASYNC
# ============================================================================
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

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
    log_path = os.path.join(log_dir, "train.log")

    logger = logging.getLogger("CryptoPipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ============================================================================
#  SECTION 1 – ENUMS & CONFIG DATACLASSES
# ============================================================================

class DataSource(Enum):
    BINANCE     = "binance"
    COINGECKO   = "coingecko"
    COINBASE    = "coinbase"
    KRAKEN      = "kraken"
    BYBIT       = "bybit"
    GLASSNODE   = "glassnode"
    CRYPTOQUANT = "cryptoquant"
    SANTIMENT   = "santiment"
    NEWSAPI     = "newsapi"
    ETHERSCAN   = "etherscan"
    FEARGREED   = "feargreed"


class ModelType(Enum):
    XGBOOST                    = "xgboost"
    LIGHTGBM                   = "lightgbm"
    CATBOOST                   = "catboost"
    RANDOM_FOREST              = "random_forest"
    EXTRA_TREES                = "extra_trees"
    GRADIENT_BOOSTING          = "gradient_boosting"
    ADABOOST                   = "adaboost"
    RIDGE                      = "ridge"
    LASSO                      = "lasso"
    ELASTIC_NET                = "elastic_net"
    SVR_SVC                    = "svr_svc"
    LSTM                       = "lstm"
    GRU                        = "gru"
    BIDIRECTIONAL_LSTM         = "bidirectional_lstm"
    TRANSFORMER                = "transformer"
    TEMPORAL_FUSION_TRANSFORMER = "tft"
    CNN_LSTM                   = "cnn_lstm"
    CNN_TRANSFORMER            = "cnn_transformer"
    WAVENET                    = "wavenet"
    PYTORCH_LSTM               = "pytorch_lstm"
    PYTORCH_TRANSFORMER        = "pytorch_transformer"
    ENSEMBLE_VOTING            = "ensemble_voting"
    ENSEMBLE_STACKING          = "ensemble_stacking"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION     = "regression"
    MULTI_TASK     = "multi_task"


class TimeFrame(Enum):
    ONE_MIN      = "1m"
    FIVE_MIN     = "5m"
    FIFTEEN_MIN  = "15m"
    THIRTY_MIN   = "30m"
    ONE_HOUR     = "1h"
    FOUR_HOUR    = "4h"
    DAILY        = "1d"
    WEEKLY       = "1w"


@dataclass
class DataConfig:
    primary_symbol:     str       = "RNDR"
    quote_currency:     str       = "USDT"
    related_symbols:    List[str] = field(default_factory=lambda: [
        "BTC", "ETH", "SOL", "AR", "TAO", "ICP",
        "AVAX", "NEAR", "API3", "SAND", "MANA"
    ])
    lookback_days:      int       = 730
    min_data_points:    int       = 500
    timeframes:         List[str] = field(default_factory=lambda: ["1d", "4h", "1h"])
    max_retries:        int       = 3
    retry_delay:        float     = 2.0
    rate_limit_delay:   float     = 1.0
    api_keys:           Dict      = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        cfg = cls()
        cfg.api_keys = {
            "binance":     os.getenv("BINANCE_API_KEY", ""),
            "coingecko":   os.getenv("COINGECKO_API_KEY", ""),
            "cryptoquant": os.getenv("CRYPTOQUANT_API_KEY", ""),
            "glassnode":   os.getenv("GLASSNODE_API_KEY", ""),
            "twitter":     os.getenv("TWITTER_API_KEY", ""),
            "newsapi":     os.getenv("NEWSAPI_KEY", ""),
            "etherscan":   os.getenv("ETHERSCAN_API_KEY", ""),
        }
        return cfg


@dataclass
class ModelConfig:
    task_type:                TaskType        = TaskType.MULTI_TASK
    models_to_train:          List[ModelType] = field(default_factory=lambda: [
        ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.CATBOOST,
        ModelType.RANDOM_FOREST, ModelType.EXTRA_TREES,
        ModelType.ENSEMBLE_VOTING, ModelType.ENSEMBLE_STACKING,
        # Deep-learning models (LSTM, GRU, Transformer, PyTorch) are disabled by
        # default to prevent overfitting on typical crypto tabular datasets.
        # Re-enable with:  --models xgboost,lightgbm,lstm,transformer,...
    ])
    sequence_length:          int   = 60
    prediction_horizon:       int   = 6
    test_size:                float = 0.15
    validation_size:          float = 0.10
    batch_size:               int   = 32
    epochs:                   int   = 150
    learning_rate:            float = 0.001
    dropout_rate:             float = 0.3
    l1_reg:                   float = 0.0001
    l2_reg:                   float = 0.001
    early_stopping_patience:  int   = 20
    use_gpu:                  bool  = True
    mixed_precision:          bool  = True
    use_optuna:               bool  = True
    n_trials:                 int   = 80
    optuna_timeout:           Optional[int] = None
    model_save_dir:           str   = "/tmp/crypto_train_output/models"
    checkpoint_dir:           str   = "/tmp/crypto_train_output/checkpoints"
    random_seed:              int   = 42


@dataclass
class PredictionConfig:
    classification_thresholds: Dict[str, float] = field(default_factory=lambda: {
        # For 4h tf / h=6 (= 24h ahead): ±2% separates clear moves from chop.
        # Tighter than 1d because 4h candles are noisier per bar.
        # If you run 1d tf change these back to ±0.015.
        "up":   0.020,
        "down": -0.020,
    })
    min_confidence:              float = 0.55
    ensemble_voting_threshold:   float = 0.5
    include_confidence_intervals: bool = True
    use_whale_signals:           bool  = True
    use_sentiment_signals:       bool  = True
    use_onchain_signals:         bool  = True
    use_technical_signals:       bool  = True
    max_leverage:                float = 3.0
    position_size_pct:           float = 0.02
    stop_loss_pct:               float = 0.05
    take_profit_pct:             float = 0.10


@dataclass
class BacktestConfig:
    initial_capital:        float = 10_000.0
    leverage:               float = 1.0
    slippage_pct:           float = 0.05
    commission_pct:         float = 0.10
    train_period_days:      int   = 180
    test_period_days:       int   = 30
    walk_forward_step_days: int   = 7


@dataclass
class PipelineConfig:
    data_config:       DataConfig       = field(default_factory=DataConfig)
    model_config:      ModelConfig      = field(default_factory=ModelConfig)
    prediction_config: PredictionConfig = field(default_factory=PredictionConfig)
    backtest_config:   BacktestConfig   = field(default_factory=BacktestConfig)
    output_dir:        str              = "/tmp/crypto_train_output"
    log_level:         str              = "INFO"
    save_artifacts:    bool             = True

    def validate(self):
        mc = self.model_config
        assert mc.sequence_length > 0
        assert mc.prediction_horizon > 0
        assert mc.test_size + mc.validation_size < 0.6
        return True


# ============================================================================
#  SECTION 2 – DATA COLLECTION
# ============================================================================

class DataCollectorBase:
    """Abstract base for all data collectors."""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def _retry_request(self, url: str, params: dict = None,
                       headers: dict = None) -> Optional[dict]:
        if not REQUESTS_AVAILABLE:
            return None
        for attempt in range(self.config.max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                self.logger.warning(f"HTTP {resp.status_code} from {url}")
            except Exception as e:
                self.logger.warning(f"Request attempt {attempt+1} failed: {e}")
                time.sleep(self.config.retry_delay * (attempt + 1))
        return None


class BinanceCollector(DataCollectorBase):
    """OHLCV data from Binance public REST API."""

    BASE_URL = "https://api.binance.com/api/v3"

    def fetch_klines(self, symbol: str, interval: str = "1d",
                     limit: int = 1000) -> pd.DataFrame:
        url = f"{self.BASE_URL}/klines"
        params = {
            "symbol": f"{symbol}USDT",
            "interval": interval,
            "limit": limit,
        }
        data = self._retry_request(url, params=params)
        if data is None:
            self.logger.error(f"Failed to fetch Binance klines for {symbol}")
            return pd.DataFrame()

        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(data, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "taker_buy_base", "taker_buy_quote"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        self.logger.info(f"Binance: fetched {len(df)} candles for {symbol}/{interval}")
        return df[["open", "high", "low", "close", "volume", "quote_volume",
                   "taker_buy_base", "taker_buy_quote", "trades"]]

    def fetch_funding_rates(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": f"{symbol}USDT", "limit": limit}
        data = self._retry_request(url, params=params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df["funding_rate"] = df["fundingRate"].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["funding_rate"]]

    def fetch_open_interest(self, symbol: str, period: str = "1d",
                            limit: int = 500) -> pd.DataFrame:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
        data = self._retry_request(url, params=params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["open_interest"] = df["sumOpenInterest"].astype(float)
        df["open_interest_value"] = df["sumOpenInterestValue"].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["open_interest", "open_interest_value"]]

    def fetch_long_short_ratio(self, symbol: str, period: str = "1d",
                               limit: int = 500) -> pd.DataFrame:
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        params = {"symbol": f"{symbol}USDT", "period": period, "limit": limit}
        data = self._retry_request(url, params=params)
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df["long_pct"] = df["longAccount"].astype(float)
        df["short_pct"] = df["shortAccount"].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["long_short_ratio", "long_pct", "short_pct"]]

    def fetch_liquidations_proxy(self, symbol: str, interval: str = "1d",
                                 limit: int = 500) -> pd.DataFrame:
        """
        Binance doesn't expose raw liquidation history publicly,
        so we use taker buy volume vs total volume as a proxy for
        aggressive buying/selling pressure which correlates with liquidations.
        """
        df = self.fetch_klines(symbol, interval, limit)
        if df.empty:
            return pd.DataFrame()
        df["taker_sell_volume"] = df["volume"] - df["taker_buy_base"]
        df["buy_sell_ratio"] = df["taker_buy_base"] / (df["taker_sell_volume"] + 1e-9)
        return df[["buy_sell_ratio"]]


class CoinGeckoCollector(DataCollectorBase):
    """Market data from CoinGecko (free tier)."""

    BASE_URL = "https://api.coingecko.com/api/v3"
    # Common CoinGecko IDs for crypto assets
    SYMBOL_ID_MAP = {
        "RNDR": "render",           # rebranded from render-token → render (Oct 2023)
        "BTC":  "bitcoin",
        "ETH":  "ethereum",
        "SOL":  "solana",
        "AR":   "arweave",
        "TAO":  "bittensor",
        "ICP":  "internet-computer",
        "AVAX": "avalanche-2",
        "NEAR": "near",
        "API3": "api3",
        "SAND": "the-sandbox",
        "MANA": "decentraland",
    }

    def get_coin_id(self, symbol: str) -> str:
        return self.SYMBOL_ID_MAP.get(symbol.upper(), symbol.lower())

    def fetch_market_chart(self, symbol: str, days: int = 365) -> pd.DataFrame:
        coin_id = self.get_coin_id(symbol)
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "daily"}
        data = self._retry_request(url, params=params)
        if not data:
            return pd.DataFrame()

        prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
        market_caps = pd.DataFrame(data["market_caps"], columns=["ts", "market_cap"])
        volumes = pd.DataFrame(data["total_volumes"], columns=["ts", "cg_volume"])

        df = prices.merge(market_caps, on="ts").merge(volumes, on="ts")
        df["timestamp"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df.drop(columns=["ts"], inplace=True)
        self.logger.info(f"CoinGecko: {len(df)} rows for {symbol}")
        return df

    def fetch_global_metrics(self) -> Dict:
        url = f"{self.BASE_URL}/global"
        data = self._retry_request(url)
        if not data:
            return {}
        return data.get("data", {})

    def fetch_dominance(self) -> pd.Series:
        metrics = self.fetch_global_metrics()
        return pd.Series({
            "btc_dominance": metrics.get("market_cap_percentage", {}).get("btc", 50.0),
            "eth_dominance": metrics.get("market_cap_percentage", {}).get("eth", 17.0),
            "total_market_cap": metrics.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h": metrics.get("total_volume", {}).get("usd", 0),
            "defi_market_cap": metrics.get("defi_market_cap", 0),
            "defi_to_eth_ratio": metrics.get("defi_to_eth_ratio", 0),
            "active_cryptocurrencies": metrics.get("active_cryptocurrencies", 0),
        })


class FearGreedCollector(DataCollectorBase):
    """Alternative.me Fear & Greed Index (free)."""

    URL = "https://api.alternative.me/fng/"

    def fetch(self, limit: int = 500) -> pd.DataFrame:
        data = self._retry_request(self.URL, params={"limit": limit, "format": "json"})
        if not data:
            return pd.DataFrame()
        rows = []
        for item in data.get("data", []):
            rows.append({
                "timestamp": pd.to_datetime(int(item["timestamp"]), unit="s"),
                "fear_greed_index": int(item["value"]),
                "fear_greed_class": item["value_classification"],
            })
        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        self.logger.info(f"FearGreed: {len(df)} rows fetched")
        return df


class OnChainCollector(DataCollectorBase):
    """
    On-chain metrics. Generates realistic synthetic data when API keys
    are absent so the pipeline can still train.
    """

    def generate_synthetic_onchain(self, index: pd.DatetimeIndex,
                                   price_series: pd.Series) -> pd.DataFrame:
        """Produce plausible synthetic on-chain proxies correlated with price."""
        n = len(index)
        rng = np.random.default_rng(42)
        price_norm = (price_series - price_series.min()) / (
            price_series.max() - price_series.min() + 1e-9
        )

        df = pd.DataFrame(index=index)
        df["active_addresses"]      = (5000 + price_norm * 20000 + rng.normal(0, 1000, n)).clip(0)
        df["transaction_count"]     = (10000 + price_norm * 50000 + rng.normal(0, 3000, n)).clip(0)
        df["transaction_volume_usd"]= (price_series * (df["transaction_count"] * rng.uniform(0.8, 1.2, n)))
        df["large_tx_count"]        = (50 + price_norm * 300 + rng.normal(0, 30, n)).clip(0)
        df["whale_volume_ratio"]    = (0.3 + price_norm * 0.4 + rng.normal(0, 0.05, n)).clip(0, 1)
        df["exchange_inflow"]       = (500 + rng.normal(0, 100, n)).clip(0)
        df["exchange_outflow"]      = (480 + price_norm * 200 + rng.normal(0, 100, n)).clip(0)
        df["exchange_netflow"]      = df["exchange_outflow"] - df["exchange_inflow"]
        df["mvrv_ratio"]            = (1.5 + price_norm * 3.0 + rng.normal(0, 0.3, n)).clip(0.1)
        df["nvt_signal"]            = (50 + (1 - price_norm) * 100 + rng.normal(0, 10, n)).clip(1)
        df["sopr"]                  = (0.95 + price_norm * 0.15 + rng.normal(0, 0.02, n)).clip(0.5)
        df["dormant_circulation"]   = rng.uniform(0, 1, n)
        df["liveliness"]            = (0.4 + price_norm * 0.4 + rng.normal(0, 0.05, n)).clip(0, 1)

        # Rolling smoothing
        for col in df.columns:
            df[col] = df[col].rolling(3, min_periods=1).mean()

        self.logger.info(f"Synthetic on-chain data generated: {n} rows, {len(df.columns)} cols")
        return df


class SentimentCollector(DataCollectorBase):
    """
    News/social sentiment. Falls back to synthetic correlated sentiment
    when API keys are unavailable.
    """

    def fetch_fear_greed_extended(self, fear_greed_df: pd.DataFrame) -> pd.DataFrame:
        """Expand fear & greed data with derived features."""
        df = fear_greed_df.copy()
        df["fg_ma7"]   = df["fear_greed_index"].rolling(7).mean()
        df["fg_ma30"]  = df["fear_greed_index"].rolling(30).mean()
        df["fg_trend"] = df["fg_ma7"] - df["fg_ma30"]
        df["fg_zscore"] = (
            (df["fear_greed_index"] - df["fear_greed_index"].rolling(30).mean()) /
            (df["fear_greed_index"].rolling(30).std() + 1e-9)
        )
        return df

    def generate_synthetic_sentiment(self, index: pd.DatetimeIndex,
                                     price_series: pd.Series) -> pd.DataFrame:
        n = len(index)
        rng = np.random.default_rng(123)
        ret = price_series.pct_change().fillna(0)
        # sentiment lags price by ~2 days
        ret_lag = ret.shift(2).fillna(0)

        df = pd.DataFrame(index=index)
        df["twitter_sentiment"]  = (0.5 + ret_lag * 5 + rng.normal(0, 0.15, n)).clip(-1, 1)
        df["reddit_sentiment"]   = (0.4 + ret_lag * 4 + rng.normal(0, 0.20, n)).clip(-1, 1)
        df["news_sentiment"]     = (0.3 + ret_lag * 3 + rng.normal(0, 0.12, n)).clip(-1, 1)
        df["mention_volume"]     = (1000 + ret_lag * 5000 + rng.normal(0, 200, n)).clip(0)
        df["ai_sentiment_score"] = (df["twitter_sentiment"] * 0.4 +
                                    df["reddit_sentiment"] * 0.3 +
                                    df["news_sentiment"] * 0.3)
        for col in df.columns:
            df[col] = df[col].rolling(2, min_periods=1).mean()
        self.logger.info("Synthetic sentiment data generated")
        return df


class DataAggregator:
    """Orchestrates all collectors and merges into a single DataFrame."""

    def __init__(self, config: DataConfig, logger: logging.Logger):
        self.config   = config
        self.logger   = logger
        self.binance  = BinanceCollector(config, logger)
        self.coingecko = CoinGeckoCollector(config, logger)
        self.feargreed = FearGreedCollector(config, logger)
        self.onchain  = OnChainCollector(config, logger)
        self.sentiment = SentimentCollector(config, logger)

    def fetch_all(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """Fetch and merge all data sources for a given symbol."""
        self.logger.info(f"=== Fetching all data for {symbol} / {timeframe} ===")

        # --- Primary OHLCV ---
        limit = min(self.config.lookback_days, 1000)
        df_ohlcv = self.binance.fetch_klines(symbol, timeframe, limit=limit)
        if df_ohlcv.empty:
            self.logger.warning("Binance empty – trying CoinGecko fallback")
            df_cg = self.coingecko.fetch_market_chart(symbol, days=self.config.lookback_days)
            if df_cg.empty:
                raise RuntimeError(f"No OHLCV data for {symbol}. Check connectivity.")
            # Build synthetic OHLCV from daily price
            df_ohlcv = self._synthetic_ohlcv_from_price(df_cg["price"])

        # Resample to calendar days if needed
        if timeframe in ("1d", "1w"):
            df_ohlcv = df_ohlcv.resample("D").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
                "quote_volume": "sum",
                "taker_buy_base": "sum",
                "taker_buy_quote": "sum",
                "trades": "sum",
            }).dropna(subset=["close"])

        # --- Futures data ---
        df_funding = self.binance.fetch_funding_rates(symbol)
        df_oi      = self.binance.fetch_open_interest(symbol)
        df_lsr     = self.binance.fetch_long_short_ratio(symbol)

        # --- Fear & Greed ---
        df_fg = self.feargreed.fetch(limit=limit)
        df_fg = self.sentiment.fetch_fear_greed_extended(df_fg)

        # --- On-chain (synthetic if no key) ---
        df_onchain = self.onchain.generate_synthetic_onchain(
            df_ohlcv.index, df_ohlcv["close"]
        )

        # --- Sentiment (synthetic if no key) ---
        df_sent = self.sentiment.generate_synthetic_sentiment(
            df_ohlcv.index, df_ohlcv["close"]
        )

        # --- Related symbol BTC/ETH correlation data ---
        df_btc = self.binance.fetch_klines("BTC", timeframe, limit=limit)
        df_eth = self.binance.fetch_klines("ETH", timeframe, limit=limit)

        # --- Merge everything ---
        df = df_ohlcv.copy()
        df = self._merge(df, df_funding, ["funding_rate"])
        df = self._merge(df, df_oi,      ["open_interest", "open_interest_value"])
        df = self._merge(df, df_lsr,     ["long_short_ratio", "long_pct", "short_pct"])
        df = self._merge(df, df_fg,
                         ["fear_greed_index", "fg_ma7", "fg_ma30",
                          "fg_trend", "fg_zscore"])
        df = self._merge(df, df_onchain, list(df_onchain.columns))
        df = self._merge(df, df_sent,    list(df_sent.columns))

        # BTC / ETH close as cross-asset signals
        if not df_btc.empty:
            df_btc = df_btc[["close"]].rename(columns={"close": "btc_close"})
            df = self._merge(df, df_btc, ["btc_close"])
        if not df_eth.empty:
            df_eth = df_eth[["close"]].rename(columns={"close": "eth_close"})
            df = self._merge(df, df_eth, ["eth_close"])

        # Global metrics (single-row broadcast)
        global_metrics = self.coingecko.fetch_dominance()
        for k, v in global_metrics.items():
            if isinstance(v, (int, float)):
                df[k] = v

        df = df.sort_index()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        rows_before = len(df)
        df = df.dropna(thresh=int(len(df.columns) * 0.5))
        self.logger.info(
            f"Data merged: {rows_before}→{len(df)} rows, {len(df.columns)} cols"
        )
        return df

    # -----------------------------------------------------------------------
    def _merge(self, df: pd.DataFrame, other: pd.DataFrame,
               cols: List[str]) -> pd.DataFrame:
        if other is None or other.empty:
            for c in cols:
                df[c] = np.nan
            return df
        avail_cols = [c for c in cols if c in other.columns]
        if not avail_cols:
            return df
        return df.join(other[avail_cols], how="left")

    @staticmethod
    def _synthetic_ohlcv_from_price(price: pd.Series) -> pd.DataFrame:
        """Build OHLCV stub from a price series."""
        rng = np.random.default_rng(0)
        noise = rng.uniform(0.99, 1.01, len(price))
        df = pd.DataFrame(index=price.index)
        df["close"]         = price
        df["open"]          = price.shift(1).fillna(price)
        df["high"]          = price * (1 + rng.uniform(0, 0.02, len(price)))
        df["low"]           = price * (1 - rng.uniform(0, 0.02, len(price)))
        df["volume"]        = 1e6 * noise
        df["quote_volume"]  = df["volume"] * df["close"]
        df["taker_buy_base"]  = df["volume"] * 0.5
        df["taker_buy_quote"] = df["taker_buy_base"] * df["close"]
        df["trades"]        = 10000
        return df


# ============================================================================
#  SECTION 3 – FEATURE ENGINEERING (200+ INDICATORS)
# ============================================================================

class TechnicalIndicators:
    """All technical analysis indicators."""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs    = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(prices: pd.Series, fast=12, slow=26, signal=9):
        ema_f = prices.ewm(span=fast, adjust=False).mean()
        ema_s = prices.ewm(span=slow, adjust=False).mean()
        line  = ema_f - ema_s
        sig   = line.ewm(span=signal, adjust=False).mean()
        hist  = line - sig
        return line, sig, hist

    @staticmethod
    def bollinger_bands(prices: pd.Series, period=20, std_dev=2.0):
        sma   = prices.rolling(period).mean()
        sigma = prices.rolling(period).std()
        upper = sma + std_dev * sigma
        lower = sma - std_dev * sigma
        pct_b = (prices - lower) / (upper - lower + 1e-9)
        width = (upper - lower) / (sma + 1e-9)
        return upper, sma, lower, pct_b, width

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        ll = low.rolling(k_period).min()
        hh = high.rolling(k_period).max()
        k  = 100 * (close - ll) / (hh - ll + 1e-9)
        d  = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def adx(high, low, close, period=14):
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr       = TechnicalIndicators.atr(high, low, close, period)
        plus_di  = 100 * (plus_dm.rolling(period).mean() / (tr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(period).mean() / (tr + 1e-9))
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx_val  = dx.rolling(period).mean()
        return adx_val, plus_di, minus_di

    @staticmethod
    def cci(high, low, close, period=20) -> pd.Series:
        tp  = (high + low + close) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma) / (0.015 * mad + 1e-9)

    @staticmethod
    def williams_r(high, low, close, period=14) -> pd.Series:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        return -100 * (hh - close) / (hh - ll + 1e-9)

    @staticmethod
    def aroon(high, low, period=25):
        def aroon_up(x):
            return ((period - x[::-1].argmax()) / period) * 100
        def aroon_dn(x):
            return ((period - x[::-1].argmin()) / period) * 100
        up = high.rolling(period + 1).apply(aroon_up, raw=True)
        dn = low.rolling(period + 1).apply(aroon_dn, raw=True)
        return up, dn

    @staticmethod
    def keltner_channel(high, low, close, ema_period=20, atr_period=10, mult=2.0):
        ema   = close.ewm(span=ema_period, adjust=False).mean()
        atr_v = TechnicalIndicators.atr(high, low, close, atr_period)
        upper = ema + mult * atr_v
        lower = ema - mult * atr_v
        return upper, ema, lower

    @staticmethod
    def donchian_channel(high, low, period=20):
        upper = high.rolling(period).max()
        lower = low.rolling(period).min()
        mid   = (upper + lower) / 2
        return upper, mid, lower

    @staticmethod
    def ichimoku(high, low, close):
        conv  = (high.rolling(9).max()  + low.rolling(9).min())  / 2
        base  = (high.rolling(26).max() + low.rolling(26).min()) / 2
        span_a = (conv + base) / 2
        span_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
        chikou = close.shift(-26)
        return conv, base, span_a, span_b, chikou

    @staticmethod
    def parabolic_sar(high, low, close, step=0.02, max_step=0.2) -> pd.Series:
        """Simplified Parabolic SAR."""
        n      = len(close)
        sar    = np.zeros(n)
        trend  = np.ones(n, dtype=int)
        ep     = np.zeros(n)
        af     = np.zeros(n)
        sar[0] = low.iloc[0]
        ep[0]  = high.iloc[0]
        af[0]  = step
        for i in range(1, n):
            prev_sar = sar[i-1]
            if trend[i-1] == 1:
                sar[i] = prev_sar + af[i-1] * (ep[i-1] - prev_sar)
                sar[i] = min(sar[i], low.iloc[i-1], low.iloc[max(0, i-2)])
                if low.iloc[i] < sar[i]:
                    trend[i] = -1
                    sar[i]   = ep[i-1]
                    ep[i]    = low.iloc[i]
                    af[i]    = step
                else:
                    trend[i] = 1
                    ep[i]    = max(ep[i-1], high.iloc[i])
                    af[i]    = min(af[i-1] + step, max_step) if ep[i] > ep[i-1] else af[i-1]
            else:
                sar[i] = prev_sar + af[i-1] * (ep[i-1] - prev_sar)
                sar[i] = max(sar[i], high.iloc[i-1], high.iloc[max(0, i-2)])
                if high.iloc[i] > sar[i]:
                    trend[i] = 1
                    sar[i]   = ep[i-1]
                    ep[i]    = high.iloc[i]
                    af[i]    = step
                else:
                    trend[i] = -1
                    ep[i]    = min(ep[i-1], low.iloc[i])
                    af[i]    = min(af[i-1] + step, max_step) if ep[i] < ep[i-1] else af[i-1]
        return pd.Series(sar, index=close.index)


class VolumeIndicators:
    """Volume-based indicators."""

    @staticmethod
    def obv(close, volume) -> pd.Series:
        direction = np.sign(close.diff().fillna(0))
        return (direction * volume).cumsum()

    @staticmethod
    def vwap(high, low, close, volume) -> pd.Series:
        tp   = (high + low + close) / 3
        return (tp * volume).cumsum() / (volume.cumsum() + 1e-9)

    @staticmethod
    def money_flow_index(high, low, close, volume, period=14) -> pd.Series:
        tp = (high + low + close) / 3
        mf = tp * volume
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
        neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
        mfr    = pos_mf / (neg_mf + 1e-9)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def accumulation_distribution(high, low, close, volume) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        ad  = (clv * volume).cumsum()
        return ad

    @staticmethod
    def chaikin_money_flow(high, low, close, volume, period=20) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        return (clv * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-9)

    @staticmethod
    def volume_profile_features(volume: pd.Series, period: int = 20) -> pd.DataFrame:
        """Volume moving averages, rate of change and trend."""
        df = pd.DataFrame(index=volume.index)
        df["vol_sma_10"]  = volume.rolling(10).mean()
        df["vol_sma_20"]  = volume.rolling(20).mean()
        df["vol_sma_50"]  = volume.rolling(50).mean()
        df["vol_roc"]     = volume.pct_change(5)
        df["vol_trend"]   = df["vol_sma_10"] / (df["vol_sma_50"] + 1e-9)
        df["vol_zscore"]  = (volume - df["vol_sma_20"]) / (volume.rolling(20).std() + 1e-9)
        df["vol_spike"]   = (volume > volume.rolling(20).mean() + 2 * volume.rolling(20).std()).astype(int)
        return df


class VolatilityIndicators:
    """Historical & realized volatility indicators."""

    @staticmethod
    def historical_volatility(close: pd.Series, period=21) -> pd.Series:
        log_ret = np.log(close / close.shift(1))
        return log_ret.rolling(period).std() * np.sqrt(252)

    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, period=21) -> pd.Series:
        ln_hl = np.log(high / (low + 1e-9))
        pv    = np.sqrt((ln_hl ** 2) / (4 * np.log(2)))
        return pv.rolling(period).mean() * np.sqrt(252)

    @staticmethod
    def garman_klass_volatility(open_: pd.Series, high: pd.Series,
                                 low: pd.Series, close: pd.Series,
                                 period=21) -> pd.Series:
        log_hl = (np.log(high / (low + 1e-9))) ** 2
        log_co = (np.log(close / (open_ + 1e-9))) ** 2
        gk     = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return gk.rolling(period).mean().apply(lambda x: np.sqrt(max(x, 0)) * np.sqrt(252))

    @staticmethod
    def realized_volatility(close: pd.Series, periods=[5, 10, 21, 63]) -> pd.DataFrame:
        df = pd.DataFrame(index=close.index)
        for p in periods:
            df[f"rv_{p}d"] = VolatilityIndicators.historical_volatility(close, p)
        return df

    @staticmethod
    def atr_ratio(close: pd.Series, atr_val: pd.Series) -> pd.Series:
        return atr_val / (close + 1e-9)


class MomentumIndicators:
    """Momentum-based indicators."""

    @staticmethod
    def roc(prices: pd.Series, period: int = 10) -> pd.Series:
        return prices.pct_change(period) * 100

    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        return prices - prices.shift(period)

    @staticmethod
    def velocity_acceleration(prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        velocity     = prices.diff()
        acceleration = velocity.diff()
        return velocity, acceleration

    @staticmethod
    def fisher_transform(high: pd.Series, low: pd.Series, period=9) -> pd.Series:
        ll  = low.rolling(period).min()
        hh  = high.rolling(period).max()
        val = 2 * ((high - ll) / (hh - ll + 1e-9)) - 1
        val = val.clip(-0.999, 0.999)
        fish = 0.5 * np.log((1 + val) / (1 - val + 1e-9))
        return fish

    @staticmethod
    def trix(prices: pd.Series, period=15) -> pd.Series:
        ema1 = prices.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return ema3.pct_change() * 100

    @staticmethod
    def cmo(prices: pd.Series, period=14) -> pd.Series:
        """Chande Momentum Oscillator."""
        delta = prices.diff()
        up    = delta.where(delta > 0, 0).rolling(period).sum()
        dn    = (-delta.where(delta < 0, 0)).rolling(period).sum()
        return 100 * (up - dn) / (up + dn + 1e-9)

    @staticmethod
    def detrended_price(prices: pd.Series, period=20) -> pd.Series:
        sma = prices.rolling(period).mean().shift(period // 2 + 1)
        return prices - sma

    @staticmethod
    def ultimate_oscillator(high, low, close,
                             p1=7, p2=14, p3=28) -> pd.Series:
        tr  = TechnicalIndicators.atr(high, low, close, 1)
        bp  = close - pd.concat([low, close.shift()], axis=1).min(axis=1)

        def avg(p):
            return bp.rolling(p).sum() / (tr.rolling(p).sum() + 1e-9)

        return 100 * (4 * avg(p1) + 2 * avg(p2) + avg(p3)) / 7


class AdvancedStatisticalIndicators:
    """Fractal, Hurst, entropy, anomaly detection."""

    @staticmethod
    def hurst_exponent(ts: pd.Series, lags: int = 20) -> float:
        """Estimate the Hurst exponent (H>0.5 = trending, H<0.5 = mean-reverting)."""
        try:
            tau = []
            laglist = range(2, lags)
            for lag in laglist:
                rs_vals = []
                for i in range(0, len(ts) - lag, lag):
                    chunk = ts.iloc[i:i+lag]
                    if chunk.std() == 0:
                        continue
                    rs_vals.append(np.ptp(chunk - chunk.mean()) / chunk.std())
                if rs_vals:
                    tau.append(np.mean(rs_vals))
            if len(tau) < 2:
                return 0.5
            x = np.log(list(laglist[:len(tau)]))
            y = np.log(tau)
            slope, *_ = np.polyfit(x, y, 1)
            return float(slope)
        except Exception:
            return 0.5

    @staticmethod
    def rolling_hurst(prices: pd.Series, window: int = 100, step: int = 10) -> pd.Series:
        h_vals = pd.Series(np.nan, index=prices.index)
        for i in range(window, len(prices), step):
            chunk = prices.iloc[i - window:i]
            h     = AdvancedStatisticalIndicators.hurst_exponent(chunk)
            h_vals.iloc[i] = h
        return h_vals.ffill().bfill()

    @staticmethod
    def shannon_entropy(prices: pd.Series, window: int = 30) -> pd.Series:
        def _entropy(x):
            hist, _ = np.histogram(x, bins=10, density=True)
            hist    = hist[hist > 0]
            return -np.sum(hist * np.log2(hist + 1e-9))
        return prices.rolling(window).apply(_entropy, raw=True)

    @staticmethod
    def fractal_dimension(prices: pd.Series, window: int = 30) -> pd.Series:
        def _fd(x):
            n    = len(x)
            if n < 3:
                return 1.5
            L    = np.sum(np.abs(np.diff(x)))
            D    = max(x) - min(x) + 1e-9
            return 1 + np.log(L) / (np.log(2 * n) - np.log(D) + 1e-9)
        return prices.rolling(window).apply(_fd, raw=True)

    @staticmethod
    def zscore_anomaly(prices: pd.Series, window: int = 30) -> pd.Series:
        mu  = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        return (prices - mu) / (std + 1e-9)

    @staticmethod
    def support_resistance_levels(close: pd.Series, order: int = 5) -> Tuple[pd.Series, pd.Series]:
        """Detect local support/resistance extrema."""
        arr      = close.values
        local_max = argrelextrema(arr, np.greater, order=order)[0]
        local_min = argrelextrema(arr, np.less,    order=order)[0]

        res = pd.Series(np.nan, index=close.index)
        sup = pd.Series(np.nan, index=close.index)
        for idx in local_max:
            res.iloc[idx] = close.iloc[idx]
        for idx in local_min:
            sup.iloc[idx] = close.iloc[idx]
        return res.ffill(), sup.ffill()

    @staticmethod
    def regime_detection(close: pd.Series, fast=20, slow=50) -> pd.Series:
        """0=downtrend, 1=sideways, 2=uptrend based on moving averages."""
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        regime  = pd.Series(1, index=close.index)
        regime[fast_ma > slow_ma * 1.01] = 2
        regime[fast_ma < slow_ma * 0.99] = 0
        return regime


class CrossAssetIndicators:
    """Cross-asset correlations and dominance indicators."""

    @staticmethod
    def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 30) -> pd.Series:
        return a.rolling(window).corr(b)

    @staticmethod
    def beta(asset: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
        ret_a = asset.pct_change()
        ret_m = market.pct_change()
        cov   = ret_a.rolling(window).cov(ret_m)
        var_m = ret_m.rolling(window).var()
        return cov / (var_m + 1e-9)

    @staticmethod
    def altseason_index(asset_return: pd.Series, btc_return: pd.Series,
                        window: int = 30) -> pd.Series:
        """Positive = alt outperforming BTC (altseason signal)."""
        return (asset_return - btc_return).rolling(window).mean()

    @staticmethod
    def relative_strength(asset: pd.Series, benchmark: pd.Series) -> pd.Series:
        return asset / (benchmark + 1e-9)


class FeatureEngineer:
    """
    Orchestrates all indicators and produces a flat feature DataFrame
    with 200+ engineered columns.
    """

    def __init__(self, df: pd.DataFrame, config: PipelineConfig,
                 logger: logging.Logger):
        self.df     = df.copy()
        self.config = config
        self.logger = logger
        self.features: pd.DataFrame = pd.DataFrame()

    def engineer_all_features(self) -> pd.DataFrame:
        self.logger.info("Engineering features …")
        df = self.df

        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        open_  = df["open"]
        volume = df["volume"]

        feat = pd.DataFrame(index=df.index)

        # --- Raw price ---
        feat["close"]        = close
        feat["open"]         = open_
        feat["high"]         = high
        feat["low"]          = low
        feat["hl_ratio"]     = (high - low) / (close + 1e-9)
        feat["co_ratio"]     = (close - open_) / (open_ + 1e-9)
        feat["true_range"]   = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        # --- Returns ---
        for p in [1, 2, 3, 5, 7, 10, 14, 21, 30, 60]:
            feat[f"ret_{p}d"] = close.pct_change(p)
        feat["log_ret_1d"] = np.log(close / close.shift(1))

        # --- Moving averages ---
        for w in [5, 10, 20, 50, 100, 200]:
            feat[f"sma_{w}"]    = close.rolling(w).mean()
            feat[f"ema_{w}"]    = close.ewm(span=w, adjust=False).mean()
            feat[f"sma_rel_{w}"] = close / (feat[f"sma_{w}"] + 1e-9) - 1

        # --- RSI ---
        for p in [7, 14, 21]:
            feat[f"rsi_{p}"] = TechnicalIndicators.rsi(close, p)
        feat["rsi_ma"] = feat["rsi_14"].rolling(5).mean()
        feat["rsi_div"] = feat["rsi_14"] - feat["rsi_14"].rolling(14).mean()

        # --- MACD ---
        macd_line, macd_sig, macd_hist = TechnicalIndicators.macd(close)
        feat["macd"]          = macd_line
        feat["macd_signal"]   = macd_sig
        feat["macd_hist"]     = macd_hist
        feat["macd_cross"]    = (macd_line > macd_sig).astype(int)
        feat["macd_hist_roc"] = macd_hist.diff()

        # --- Bollinger Bands ---
        bb_up, bb_mid, bb_lo, bb_pct, bb_w = TechnicalIndicators.bollinger_bands(close)
        feat["bb_upper"]   = bb_up
        feat["bb_lower"]   = bb_lo
        feat["bb_pct"]     = bb_pct
        feat["bb_width"]   = bb_w
        feat["bb_squeeze"] = (bb_w < bb_w.rolling(20).quantile(0.25)).astype(int)

        # --- ATR ---
        atr14 = TechnicalIndicators.atr(high, low, close, 14)
        feat["atr_14"]      = atr14
        feat["atr_ratio"]   = VolatilityIndicators.atr_ratio(close, atr14)
        feat["atr_pct_chg"] = atr14.pct_change()

        # --- Stochastic ---
        stoch_k, stoch_d = TechnicalIndicators.stochastic(high, low, close)
        feat["stoch_k"]    = stoch_k
        feat["stoch_d"]    = stoch_d
        feat["stoch_diff"] = stoch_k - stoch_d

        # --- ADX ---
        adx_val, plus_di, minus_di = TechnicalIndicators.adx(high, low, close)
        feat["adx"]       = adx_val
        feat["plus_di"]   = plus_di
        feat["minus_di"]  = minus_di
        feat["di_diff"]   = plus_di - minus_di

        # --- CCI ---
        feat["cci_20"] = TechnicalIndicators.cci(high, low, close, 20)
        feat["cci_50"] = TechnicalIndicators.cci(high, low, close, 50)

        # --- Williams %R ---
        feat["williams_r_14"] = TechnicalIndicators.williams_r(high, low, close, 14)

        # --- Aroon ---
        aroon_up, aroon_dn = TechnicalIndicators.aroon(high, low, 25)
        feat["aroon_up"]    = aroon_up
        feat["aroon_dn"]    = aroon_dn
        feat["aroon_osc"]   = aroon_up - aroon_dn

        # --- Keltner / Donchian ---
        kc_up, kc_mid, kc_lo = TechnicalIndicators.keltner_channel(high, low, close)
        feat["kc_upper"]  = kc_up
        feat["kc_lower"]  = kc_lo
        feat["kc_pct"]    = (close - kc_lo) / (kc_up - kc_lo + 1e-9)

        dc_up, dc_mid, dc_lo = TechnicalIndicators.donchian_channel(high, low, 20)
        feat["dc_upper"]  = dc_up
        feat["dc_lower"]  = dc_lo
        feat["dc_pct"]    = (close - dc_lo) / (dc_up - dc_lo + 1e-9)

        # --- Parabolic SAR ---
        feat["psar"] = TechnicalIndicators.parabolic_sar(high, low, close)
        feat["psar_signal"] = (close > feat["psar"]).astype(int)

        # --- Ichimoku ---
        ich_conv, ich_base, ich_sa, ich_sb, _ = TechnicalIndicators.ichimoku(high, low, close)
        feat["ichi_conv"]  = ich_conv
        feat["ichi_base"]  = ich_base
        feat["ichi_span_a"]= ich_sa
        feat["ichi_span_b"]= ich_sb
        feat["ichi_cloud_top"] = pd.concat([ich_sa, ich_sb], axis=1).max(axis=1)
        feat["ichi_cloud_bot"] = pd.concat([ich_sa, ich_sb], axis=1).min(axis=1)
        feat["ichi_above_cloud"] = (close > feat["ichi_cloud_top"]).astype(int)

        # --- Volume indicators ---
        feat["obv"]   = VolumeIndicators.obv(close, volume)
        feat["vwap"]  = VolumeIndicators.vwap(high, low, close, volume)
        feat["mfi"]   = VolumeIndicators.money_flow_index(high, low, close, volume)
        feat["ad"]    = VolumeIndicators.accumulation_distribution(high, low, close, volume)
        feat["cmf"]   = VolumeIndicators.chaikin_money_flow(high, low, close, volume)
        feat["vwap_dev"] = (close - feat["vwap"]) / (close + 1e-9)

        vol_feats = VolumeIndicators.volume_profile_features(volume)
        for c in vol_feats.columns:
            feat[c] = vol_feats[c]

        # --- Volatility ---
        feat["hv_21d"]  = VolatilityIndicators.historical_volatility(close, 21)
        feat["hv_63d"]  = VolatilityIndicators.historical_volatility(close, 63)
        feat["parkinson"] = VolatilityIndicators.parkinson_volatility(high, low, 21)
        feat["gk_vol"]  = VolatilityIndicators.garman_klass_volatility(open_, high, low, close, 21)
        rv_df = VolatilityIndicators.realized_volatility(close)
        for c in rv_df.columns:
            feat[c] = rv_df[c]
        feat["vol_ratio_5_21"] = feat["rv_5d"] / (feat["rv_21d"] + 1e-9)

        # --- Momentum ---
        for p in [5, 10, 20]:
            feat[f"roc_{p}"] = MomentumIndicators.roc(close, p)
            feat[f"mom_{p}"] = MomentumIndicators.momentum(close, p)

        vel, acc = MomentumIndicators.velocity_acceleration(close)
        feat["velocity"]     = vel
        feat["acceleration"] = acc

        feat["fisher"]       = MomentumIndicators.fisher_transform(high, low)
        feat["trix"]         = MomentumIndicators.trix(close)
        feat["cmo_14"]       = MomentumIndicators.cmo(close, 14)
        feat["dpo_20"]       = MomentumIndicators.detrended_price(close, 20)
        feat["uo"]           = MomentumIndicators.ultimate_oscillator(high, low, close)

        # --- Advanced statistical ---
        feat["entropy_30"]    = AdvancedStatisticalIndicators.shannon_entropy(close, 30)
        feat["frac_dim_30"]   = AdvancedStatisticalIndicators.fractal_dimension(close, 30)
        feat["zscore_anom"]   = AdvancedStatisticalIndicators.zscore_anomaly(close, 30)
        feat["hurst"]         = AdvancedStatisticalIndicators.rolling_hurst(close, 100, 10)
        feat["regime"]        = AdvancedStatisticalIndicators.regime_detection(close)
        feat["resistance"]    = AdvancedStatisticalIndicators.support_resistance_levels(close)[0]
        feat["support"]       = AdvancedStatisticalIndicators.support_resistance_levels(close)[1]
        feat["sr_distance"]   = (close - feat["support"]) / (feat["resistance"] - feat["support"] + 1e-9)

        # --- 52-week high/low position ---
        feat["high_52w"]      = high.rolling(252, min_periods=1).max()
        feat["low_52w"]       = low.rolling(252, min_periods=1).min()
        feat["pos_52w"]       = (close - feat["low_52w"]) / (feat["high_52w"] - feat["low_52w"] + 1e-9)
        feat["dist_from_52h"] = (feat["high_52w"] - close) / (close + 1e-9)

        # --- Cross-asset (if BTC/ETH present) ---
        if "btc_close" in df.columns:
            feat["btc_close"] = df["btc_close"]
            feat["btc_ret_1d"] = df["btc_close"].pct_change()
            feat["corr_btc_30"] = CrossAssetIndicators.rolling_correlation(close, df["btc_close"], 30)
            feat["beta_btc_60"] = CrossAssetIndicators.beta(close, df["btc_close"], 60)
            feat["altseason"]   = CrossAssetIndicators.altseason_index(
                close.pct_change(), df["btc_close"].pct_change()
            )
            feat["rel_str_btc"] = CrossAssetIndicators.relative_strength(close, df["btc_close"])

        if "eth_close" in df.columns:
            feat["eth_close"]   = df["eth_close"]
            feat["corr_eth_30"] = CrossAssetIndicators.rolling_correlation(close, df["eth_close"], 30)
            feat["beta_eth_60"] = CrossAssetIndicators.beta(close, df["eth_close"], 60)

        # --- Pass-through sourced features ---
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
        ]
        for col in passthrough:
            if col in df.columns:
                feat[col] = df[col]

        # --- Interaction / ratio features ---
        if "funding_rate" in feat.columns and "open_interest" in feat.columns:
            feat["oi_funding_product"] = feat["open_interest"] * feat["funding_rate"]

        if "mvrv_ratio" in feat.columns:
            feat["mvrv_rsi_product"] = feat["mvrv_ratio"] * feat["rsi_14"]

        if "whale_volume_ratio" in feat.columns:
            feat["whale_obv"] = feat["whale_volume_ratio"] * feat["obv"]

        if "twitter_sentiment" in feat.columns:
            feat["sentiment_composite"] = (
                feat.get("twitter_sentiment", 0) * 0.4 +
                feat.get("reddit_sentiment", 0) * 0.3 +
                feat.get("news_sentiment", 0)   * 0.3
            )

        # --- Lag features (1, 2, 3 days) ---
        lag_cols = ["close", "rsi_14", "macd_hist", "volume",
                    "fear_greed_index", "funding_rate"]
        for col in lag_cols:
            if col in feat.columns:
                for lag in [1, 2, 3]:
                    feat[f"{col}_lag{lag}"] = feat[col].shift(lag)

        # --- Rolling statistics ---
        for w in [7, 14, 30]:
            feat[f"close_mean_{w}"]   = close.rolling(w).mean()
            feat[f"close_std_{w}"]    = close.rolling(w).std()
            feat[f"close_skew_{w}"]   = close.rolling(w).skew()
            feat[f"close_kurt_{w}"]   = close.rolling(w).kurt()
            feat[f"close_min_{w}"]    = close.rolling(w).min()
            feat[f"close_max_{w}"]    = close.rolling(w).max()
            feat[f"close_range_{w}"]  = feat[f"close_max_{w}"] - feat[f"close_min_{w}"]

        # --- Clean-up ---
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.ffill().bfill()

        # Drop columns that are still all-NaN
        feat = feat.dropna(axis=1, how="all")

        self.features = feat
        self.logger.info(f"Feature engineering done: {len(feat.columns)} features, {len(feat)} rows")
        return feat


# ============================================================================
#  SECTION 4 – DATA PREPARATION FOR TRAINING
# ============================================================================

class DataPreprocessor:
    """Scale, create sequences, split (time-series safe)."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config   = config
        self.logger   = logger
        self.scaler_x = RobustScaler()
        self.scaler_y = StandardScaler()

    def build_targets(self, close: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Returns:
            y_reg   – future return over h periods (regression target)
            y_clf   – 0/1/2 (down / sideways / up) classification target

        Label logic:
            future_ret[i] = (close[i+h] - close[i]) / close[i]
            This is the clean h-period forward return with no double-shift.
            The last h rows will be NaN and are dropped during prepare().

        For 4h timeframe with h=6:  predicts 24h ahead  (6 x 4h)
        For 1d  timeframe with h=3:  predicts  3d ahead  (3 x 1d)

        Thresholds for 4h candles (±1.5% per label default):
            up    > +1.5%   →  class 2
            down  < -1.5%   →  class 0
            else  sideways  →  class 1
        """
        mc = self.config.model_config
        pc = self.config.prediction_config
        h  = mc.prediction_horizon

        # Clean single forward-return: price h bars from now vs price now
        future_ret = (close.shift(-h) - close) / (close + 1e-9)

        y_reg = future_ret.fillna(0)

        up_thresh   = pc.classification_thresholds["up"]
        down_thresh = pc.classification_thresholds["down"]

        y_clf = pd.Series(1, index=close.index)    # sideways default
        y_clf[future_ret >  up_thresh]   = 2        # up
        y_clf[future_ret <  down_thresh] = 0        # down
        # Mask last h rows as sideways (no valid future label)
        y_clf.iloc[-h:] = 1
        y_reg.iloc[-h:] = 0.0

        return y_reg.astype(np.float32), y_clf.astype(np.int32)

    def prepare(self, features: pd.DataFrame, close: pd.Series,
                task_type: str = "multi_task"):
        """
        Prepare X sequences and targets for all model types.

        Returns
        -------
        data_flat   : dict  – flat arrays for tree models (X, y)
        data_seq    : dict  – 3-D sequences for deep learning (X, y)
        split_info  : dict  – index boundaries
        """
        mc  = self.config.model_config
        sl  = mc.sequence_length
        ph  = mc.prediction_horizon

        y_reg, y_clf = self.build_targets(close)

        # Align
        feat_cols = [c for c in features.columns if c != "close"]
        X_raw     = features[feat_cols].values.astype(np.float32)

        # Scale
        X_scaled  = self.scaler_x.fit_transform(X_raw)

        n         = len(X_scaled)
        # Time-safe split boundaries
        n_test    = int(n * mc.test_size)
        n_val     = int(n * mc.validation_size)
        n_train   = n - n_test - n_val

        # --- Flat arrays for tree / linear models ---
        flat = {
            "X_train": X_scaled[:n_train],
            "X_val":   X_scaled[n_train:n_train + n_val],
            "X_test":  X_scaled[n_train + n_val:],
            "y_reg_train": y_reg.values[:n_train],
            "y_reg_val":   y_reg.values[n_train:n_train + n_val],
            "y_reg_test":  y_reg.values[n_train + n_val:],
            "y_clf_train": y_clf.values[:n_train],
            "y_clf_val":   y_clf.values[n_train:n_train + n_val],
            "y_clf_test":  y_clf.values[n_train + n_val:],
        }

        # --- Sequence arrays for deep learning ---
        X_seq, y_reg_seq, y_clf_seq = [], [], []
        for i in range(sl, n - ph):
            X_seq.append(X_scaled[i - sl:i])
            y_reg_seq.append(y_reg.values[i + ph - 1])
            y_clf_seq.append(y_clf.values[i + ph - 1])

        X_seq      = np.array(X_seq,      dtype=np.float32)
        y_reg_seq  = np.array(y_reg_seq,  dtype=np.float32)
        y_clf_seq  = np.array(y_clf_seq,  dtype=np.int32)

        ns         = len(X_seq)
        ns_test    = int(ns * mc.test_size)
        ns_val     = int(ns * mc.validation_size)
        ns_train   = ns - ns_test - ns_val

        seq = {
            "X_train":     X_seq[:ns_train],
            "X_val":       X_seq[ns_train:ns_train + ns_val],
            "X_test":      X_seq[ns_train + ns_val:],
            "y_reg_train": y_reg_seq[:ns_train],
            "y_reg_val":   y_reg_seq[ns_train:ns_train + ns_val],
            "y_reg_test":  y_reg_seq[ns_train + ns_val:],
            "y_clf_train": y_clf_seq[:ns_train],
            "y_clf_val":   y_clf_seq[ns_train:ns_train + ns_val],
            "y_clf_test":  y_clf_seq[ns_train + ns_val:],
            "n_features":  X_seq.shape[2],
            "seq_len":     sl,
        }

        self.logger.info(
            f"Data prepared | flat train={flat['X_train'].shape} | "
            f"seq train={seq['X_train'].shape} | "
            f"classes dist train: {np.bincount(flat['y_clf_train'])}"
        )
        return flat, seq

    def inverse_transform_price(self, y_scaled: np.ndarray,
                                 close_mean: float, close_std: float) -> np.ndarray:
        return y_scaled * close_std + close_mean

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"scaler_x": self.scaler_x, "scaler_y": self.scaler_y}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scaler_x = d["scaler_x"]
        self.scaler_y = d["scaler_y"]


# ============================================================================
#  SECTION 5 – ALL TREE / LINEAR / SKLEARN MODELS
# ============================================================================

class SklearnModels:
    """Wrapper for all sklearn-compatible models."""

    @staticmethod
    def build_xgboost_reg(params: dict = None) -> Optional[object]:
        if not XGB_AVAILABLE:
            return None
        p = dict(n_estimators=500, max_depth=7, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.5, reg_lambda=1.0,
                 objective="reg:squarederror", eval_metric="rmse",
                 early_stopping_rounds=30, random_state=42, n_jobs=-1)
        if params:
            p.update(params)
        return xgb.XGBRegressor(**p)

    @staticmethod
    def build_xgboost_clf(params: dict = None) -> Optional[object]:
        if not XGB_AVAILABLE:
            return None
        p = dict(n_estimators=500, max_depth=7, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 reg_alpha=0.5, reg_lambda=1.0,
                 objective="multi:softprob", num_class=3,
                 eval_metric="mlogloss",
                 early_stopping_rounds=30, random_state=42, n_jobs=-1,
                 use_label_encoder=False)
        if params:
            p.update(params)
        return xgb.XGBClassifier(**p)

    @staticmethod
    def build_lightgbm_reg(params: dict = None) -> Optional[object]:
        if not LGB_AVAILABLE:
            return None
        p = dict(n_estimators=500, max_depth=7, learning_rate=0.05,
                 num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                 lambda_l1=0.5, lambda_l2=1.0, min_child_samples=20,
                 objective="regression", metric="rmse",
                 early_stopping_round=30, random_state=42, n_jobs=-1,
                 verbose=-1)
        if params:
            p.update(params)
        return lgb.LGBMRegressor(**p)

    @staticmethod
    def build_lightgbm_clf(params: dict = None) -> Optional[object]:
        if not LGB_AVAILABLE:
            return None
        p = dict(n_estimators=500, max_depth=7, learning_rate=0.05,
                 num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                 lambda_l1=0.5, lambda_l2=1.0, min_child_samples=20,
                 objective="multiclass", num_class=3, metric="multi_logloss",
                 early_stopping_round=30, random_state=42, n_jobs=-1,
                 verbose=-1)
        if params:
            p.update(params)
        return lgb.LGBMClassifier(**p)

    @staticmethod
    def build_catboost_reg(params: dict = None) -> Optional[object]:
        if not CAT_AVAILABLE:
            return None
        p = dict(iterations=500, depth=7, learning_rate=0.05,
                 l2_leaf_reg=3.0, subsample=0.8, rsm=0.8,
                 loss_function="RMSE", eval_metric="RMSE",
                 early_stopping_rounds=30, random_state=42, verbose=False)
        if params:
            p.update(params)
        return CatBoostRegressor(**p)

    @staticmethod
    def build_catboost_clf(params: dict = None) -> Optional[object]:
        if not CAT_AVAILABLE:
            return None
        p = dict(iterations=500, depth=7, learning_rate=0.05,
                 l2_leaf_reg=3.0, subsample=0.8, rsm=0.8,
                 loss_function="MultiClass", eval_metric="MultiClass",
                 classes_count=3,
                 early_stopping_rounds=30, random_state=42, verbose=False)
        if params:
            p.update(params)
        return CatBoostClassifier(**p)

    @staticmethod
    def build_random_forest_reg(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=15, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt",
                 random_state=42, n_jobs=-1)
        if params:
            p.update(params)
        return RandomForestRegressor(**p)

    @staticmethod
    def build_random_forest_clf(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=15, min_samples_split=5,
                 min_samples_leaf=2, max_features="sqrt",
                 random_state=42, n_jobs=-1)
        if params:
            p.update(params)
        return RandomForestClassifier(**p)

    @staticmethod
    def build_extra_trees_reg(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        if params:
            p.update(params)
        return ExtraTreesRegressor(**p)

    @staticmethod
    def build_extra_trees_clf(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
        if params:
            p.update(params)
        return ExtraTreesClassifier(**p)

    @staticmethod
    def build_gradient_boosting_reg(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, random_state=42)
        if params:
            p.update(params)
        return GradientBoostingRegressor(**p)

    @staticmethod
    def build_gradient_boosting_clf(params: dict = None) -> object:
        p = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                 subsample=0.8, random_state=42)
        if params:
            p.update(params)
        return GradientBoostingClassifier(**p)

    @staticmethod
    def build_ridge(params: dict = None) -> object:
        p = dict(alpha=1.0)
        if params:
            p.update(params)
        return Ridge(**p)

    @staticmethod
    def build_lasso(params: dict = None) -> object:
        p = dict(alpha=0.001, max_iter=5000)
        if params:
            p.update(params)
        return Lasso(**p)

    @staticmethod
    def build_elastic_net(params: dict = None) -> object:
        p = dict(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        if params:
            p.update(params)
        return ElasticNet(**p)


def train_sklearn_model(model, X_train, y_train, X_val, y_val,
                        is_classification: bool = False,
                        model_name: str = "",
                        logger: logging.Logger = None) -> Tuple[object, Dict]:
    """Fit a sklearn-compatible model with eval-set support."""

    t0 = time.time()
    supports_eval = hasattr(model, "fit") and (
        "eval_set" in model.fit.__code__.co_varnames or
        isinstance(model, (xgb.XGBModel if XGB_AVAILABLE else type(None),
                           lgb.LGBMModel if LGB_AVAILABLE else type(None)))
    )

    try:
        if XGB_AVAILABLE and isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        elif LGB_AVAILABLE and isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        elif CAT_AVAILABLE and isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
    except Exception as e:
        if logger:
            logger.error(f"Error fitting {model_name}: {e}")
        raise

    duration = time.time() - t0

    if is_classification:
        tr_score = f1_score(y_train, model.predict(X_train), average="weighted", zero_division=0)
        vl_score = f1_score(y_val,   model.predict(X_val),   average="weighted", zero_division=0)
        metric   = "f1_weighted"
    else:
        tr_score = r2_score(y_train, model.predict(X_train))
        vl_score = r2_score(y_val,   model.predict(X_val))
        metric   = "r2"

    if logger:
        logger.info(f"  {model_name}: {metric} train={tr_score:.4f} val={vl_score:.4f} "
                    f"({duration:.1f}s)")

    return model, {"train_score": tr_score, "val_score": vl_score,
                   "metric": metric, "duration": duration}


# ============================================================================
#  SECTION 6 – DEEP LEARNING MODELS (TENSORFLOW / KERAS)
# ============================================================================

if TF_AVAILABLE:
    class PositionalEncoding(layers.Layer):
        """Sinusoidal positional encoding for Transformer."""

        def __init__(self, max_len=1000, d_model=64, **kwargs):
            super().__init__(**kwargs)
            self.max_len = max_len
            self.d_model = d_model
            P = np.zeros((max_len, d_model))
            pos = np.arange(max_len)[:, np.newaxis]
            div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
            P[:, 0::2] = np.sin(pos * div)
            P[:, 1::2] = np.cos(pos * div)
            self.pe = tf.constant(P[np.newaxis, :, :], dtype=tf.float32)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            return x + self.pe[:, :seq_len, :]

    class TransformerBlock(layers.Layer):
        """Single Transformer encoder block."""

        def __init__(self, d_model=64, num_heads=4, ff_dim=128,
                     dropout_rate=0.1, **kwargs):
            super().__init__(**kwargs)
            self.attn  = layers.MultiHeadAttention(num_heads=num_heads,
                                                    key_dim=d_model // num_heads)
            self.ff1   = layers.Dense(ff_dim, activation="relu")
            self.ff2   = layers.Dense(d_model)
            self.ln1   = layers.LayerNormalization(epsilon=1e-6)
            self.ln2   = layers.LayerNormalization(epsilon=1e-6)
            self.drop1 = layers.Dropout(dropout_rate)
            self.drop2 = layers.Dropout(dropout_rate)

        def call(self, x, training=False):
            attn_out = self.attn(x, x, training=training)
            attn_out = self.drop1(attn_out, training=training)
            x = self.ln1(x + attn_out)
            ff_out = self.ff2(self.ff1(x))
            ff_out = self.drop2(ff_out, training=training)
            return self.ln2(x + ff_out)

    class KerasModelFactory:
        """Builds all Keras / TF architectures."""

        @staticmethod
        def _output_layer(x, is_clf: bool, dropout_rate: float):
            x = layers.Dropout(dropout_rate)(x)
            if is_clf:
                return layers.Dense(3, activation="softmax")(x)
            else:
                return layers.Dense(1, activation="linear")(x)

        @staticmethod
        def build_lstm(input_shape, is_clf=False,
                       units=(128, 64, 32), dropout=0.3,
                       l1=0.0001, l2=0.001) -> Model:
            inp = keras.Input(shape=input_shape)
            x   = inp
            for i, u in enumerate(units):
                ret_seq = (i < len(units) - 1)
                x = layers.Bidirectional(
                    layers.LSTM(u, return_sequences=ret_seq,
                                kernel_regularizer=l1_l2(l1, l2))
                )(x)
                x = layers.Dropout(dropout)(x)
            x   = layers.Dense(64, activation="relu")(x)
            x   = layers.BatchNormalization()(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="BiLSTM")

        @staticmethod
        def build_gru(input_shape, is_clf=False,
                      units=(128, 64, 32), dropout=0.3) -> Model:
            inp = keras.Input(shape=input_shape)
            x   = inp
            for i, u in enumerate(units):
                ret_seq = (i < len(units) - 1)
                x = layers.Bidirectional(
                    layers.GRU(u, return_sequences=ret_seq)
                )(x)
                x = layers.Dropout(dropout)(x)
            x   = layers.Dense(64, activation="relu")(x)
            x   = layers.BatchNormalization()(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="BiGRU")

        @staticmethod
        def build_attention_lstm(input_shape, is_clf=False, dropout=0.3) -> Model:
            """LSTM with self-attention mechanism."""
            inp = keras.Input(shape=input_shape)
            x   = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
            x   = layers.Dropout(dropout)(x)
            # Bahdanau-style attention
            attn_scores = layers.Dense(1, activation="tanh")(x)
            attn_w      = layers.Softmax(axis=1)(attn_scores)
            x   = layers.Multiply()([x, attn_w])
            x   = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)
            x   = layers.Dense(128, activation="relu")(x)
            x   = layers.Dropout(dropout)(x)
            x   = layers.Dense(64, activation="relu")(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="AttentionLSTM")

        @staticmethod
        def build_transformer(input_shape, is_clf=False,
                               d_model=64, num_heads=4, ff_dim=128,
                               num_blocks=3, dropout=0.3) -> Model:
            inp = keras.Input(shape=input_shape)
            x   = layers.Dense(d_model)(inp)
            x   = PositionalEncoding(max_len=input_shape[0],
                                      d_model=d_model)(x)
            for _ in range(num_blocks):
                x = TransformerBlock(d_model, num_heads, ff_dim, dropout)(x)
            x   = layers.GlobalAveragePooling1D()(x)
            x   = layers.Dense(64, activation="relu")(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="Transformer")

        @staticmethod
        def build_cnn_lstm(input_shape, is_clf=False, dropout=0.3) -> Model:
            inp = keras.Input(shape=input_shape)
            # CNN block
            x   = layers.Conv1D(64, 3, activation="relu", padding="same")(inp)
            x   = layers.BatchNormalization()(x)
            x   = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
            x   = layers.MaxPooling1D(2)(x)
            x   = layers.Conv1D(128, 3, activation="relu", padding="same")(x)
            x   = layers.BatchNormalization()(x)
            x   = layers.MaxPooling1D(2)(x)
            # LSTM block
            x   = layers.LSTM(128, return_sequences=True)(x)
            x   = layers.Dropout(dropout)(x)
            x   = layers.LSTM(64)(x)
            x   = layers.Dropout(dropout)(x)
            x   = layers.Dense(64, activation="relu")(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="CNNLSTM")

        @staticmethod
        def build_cnn_transformer(input_shape, is_clf=False,
                                   d_model=64, dropout=0.3) -> Model:
            inp = keras.Input(shape=input_shape)
            x   = layers.Conv1D(64, 3, activation="relu", padding="same")(inp)
            x   = layers.BatchNormalization()(x)
            x   = layers.Conv1D(d_model, 3, activation="relu", padding="same")(x)
            x   = PositionalEncoding(max_len=input_shape[0], d_model=d_model)(x)
            x   = TransformerBlock(d_model, num_heads=4, ff_dim=128, dropout_rate=dropout)(x)
            x   = TransformerBlock(d_model, num_heads=4, ff_dim=128, dropout_rate=dropout)(x)
            x   = layers.GlobalAveragePooling1D()(x)
            x   = layers.Dense(64, activation="relu")(x)
            out = KerasModelFactory._output_layer(x, is_clf, dropout)
            return keras.Model(inp, out, name="CNNTransformer")

        @staticmethod
        def build_temporal_fusion_transformer(input_shape, is_clf=False,
                                               d_model=32, dropout=0.3) -> Model:
            """
            Simplified Temporal Fusion Transformer:
            Variable selection → LSTM encoder → Multi-head attn → Dense output
            """
            inp = keras.Input(shape=input_shape)

            # Variable selection via gating
            gate_static   = layers.Dense(d_model, activation="sigmoid")(inp)
            static_feat   = layers.Dense(d_model, activation="relu")(inp)
            static_gated  = layers.Multiply()([static_feat, gate_static])

            # LSTM encoder (past observations)
            enc_out, h, c = layers.LSTM(d_model, return_sequences=True,
                                         return_state=True)(inp)

            # Combine static and encoded
            combined = layers.Concatenate(axis=-1)([enc_out, static_gated])
            combined = layers.Dense(d_model)(combined)

            # Self-attention
            attn = layers.MultiHeadAttention(num_heads=4,
                                              key_dim=d_model // 4)(combined, combined)
            attn = layers.Add()([combined, attn])
            attn = layers.LayerNormalization()(attn)

            # Gated residual
            grn  = layers.Dense(d_model, activation="relu")(attn)
            gate = layers.Dense(d_model, activation="sigmoid")(attn)
            out  = layers.Multiply()([grn, gate])
            out  = layers.Add()([attn, out])
            out  = layers.LayerNormalization()(out)

            out  = layers.GlobalAveragePooling1D()(out)
            out  = layers.Dense(64, activation="relu")(out)
            out  = layers.Dropout(dropout)(out)
            out  = KerasModelFactory._output_layer(out, is_clf, dropout)
            return keras.Model(inp, out, name="TFT")

        @staticmethod
        def build_wavenet(input_shape, is_clf=False, dropout=0.3) -> Model:
            """WaveNet-inspired dilated causal convolutions."""
            inp  = keras.Input(shape=input_shape)
            x    = layers.Conv1D(64, 1, padding="causal")(inp)
            skip_list = []
            for dilation in [1, 2, 4, 8, 16, 32, 64]:
                residual = x
                x_tanh   = layers.Conv1D(64, 2, padding="causal",
                                          dilation_rate=dilation,
                                          activation="tanh")(x)
                x_sig    = layers.Conv1D(64, 2, padding="causal",
                                          dilation_rate=dilation,
                                          activation="sigmoid")(x)
                x_gate   = layers.Multiply()([x_tanh, x_sig])
                x_skip   = layers.Conv1D(64, 1)(x_gate)
                skip_list.append(x_skip)
                x_res    = layers.Conv1D(64, 1)(x_gate)
                # Crop residual if needed
                if residual.shape[1] != x_res.shape[1]:
                    min_len = min(residual.shape[1], x_res.shape[1])
                    residual = layers.Lambda(lambda t: t[:, -min_len:, :])(residual)
                    x_res    = layers.Lambda(lambda t: t[:, -min_len:, :])(x_res)
                x = layers.Add()([x_res, residual])
            # Sum skips
            skip = layers.Add()(skip_list)
            skip = layers.Activation("relu")(skip)
            skip = layers.GlobalAveragePooling1D()(skip)
            skip = layers.Dense(64, activation="relu")(skip)
            out  = KerasModelFactory._output_layer(skip, is_clf, dropout)
            return keras.Model(inp, out, name="WaveNet")

    def compile_and_train_keras(
        model: Model, X_train, y_train, X_val, y_val,
        config: ModelConfig,
        is_clf: bool,
        checkpoint_path: str,
        logger: logging.Logger
    ) -> Tuple[Model, Dict]:
        """Compile + fit a Keras model with callbacks."""

        lr   = config.learning_rate
        loss = "sparse_categorical_crossentropy" if is_clf else "mse"
        metrics = ["accuracy"] if is_clf else ["mae"]

        model.compile(
            optimizer=Adam(learning_rate=lr, clipnorm=1.0),
            loss=loss,
            metrics=metrics,
        )

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                              min_lr=1e-6, verbose=0),
            ModelCheckpoint(checkpoint_path, save_best_only=True,
                            monitor="val_loss", verbose=0),
        ]

        t0      = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        duration = time.time() - t0

        val_scores = history.history.get("val_accuracy", history.history.get("val_mae", [0]))
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1

        tr_eval  = model.evaluate(X_train, y_train, verbose=0)
        val_eval = model.evaluate(X_val,   y_val,   verbose=0)

        info = {
            "train_score": tr_eval[1],
            "val_score":   val_eval[1],
            "best_epoch":  best_epoch,
            "duration":    duration,
            "metric":      "accuracy" if is_clf else "mae",
        }

        logger.info(f"  {model.name}: epoch={best_epoch} train={tr_eval[1]:.4f} "
                    f"val={val_eval[1]:.4f} ({duration:.0f}s)")
        return model, info


# ============================================================================
#  SECTION 7 – PYTORCH MODELS
# ============================================================================

if TORCH_AVAILABLE:
    class PyTorchLSTM(nn.Module):
        """Stacked bidirectional LSTM in PyTorch."""

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 3, num_classes: int = 1,
                     dropout: float = 0.3):
            super().__init__()
            self.is_clf    = (num_classes > 1)
            self.lstm      = nn.LSTM(input_size, hidden_size,
                                     num_layers=num_layers,
                                     batch_first=True, bidirectional=True,
                                     dropout=dropout)
            self.attention = nn.Linear(hidden_size * 2, 1)
            self.fc1       = nn.Linear(hidden_size * 2, 64)
            self.bn1       = nn.BatchNorm1d(64)
            self.fc2       = nn.Linear(64, num_classes)
            self.dropout   = nn.Dropout(dropout)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)         # (B, T, 2H)
            attn_w      = torch.softmax(self.attention(lstm_out), dim=1)
            context     = (lstm_out * attn_w).sum(dim=1)  # (B, 2H)
            out = self.dropout(F.relu(self.bn1(self.fc1(context))))
            out = self.fc2(out)
            return out

    class PyTorchTransformer(nn.Module):
        """Transformer encoder in PyTorch."""

        def __init__(self, input_size: int, d_model: int = 64,
                     num_heads: int = 4, num_layers: int = 3,
                     ff_dim: int = 256, num_classes: int = 1,
                     dropout: float = 0.3, max_len: int = 200):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_enc    = nn.Embedding(max_len, d_model)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads,
                dim_feedforward=ff_dim, dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc1         = nn.Linear(d_model, 64)
            self.fc2         = nn.Linear(64, num_classes)
            self.dropout     = nn.Dropout(dropout)

        def forward(self, x):
            B, T, _ = x.shape
            x       = self.input_proj(x)
            pos     = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            x       = x + self.pos_enc(pos)
            x       = self.transformer(x)
            x       = x.mean(dim=1)
            out     = self.dropout(F.relu(self.fc1(x)))
            return self.fc2(out)

    def train_pytorch_model(
        model: nn.Module, X_train, y_train, X_val, y_val,
        config: ModelConfig, is_clf: bool,
        logger: logging.Logger
    ) -> Tuple[nn.Module, Dict]:
        """PyTorch training loop with early stopping."""

        device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available()
                              else "cpu")
        model  = model.to(device)

        X_tr = torch.FloatTensor(X_train).to(device)
        X_vl = torch.FloatTensor(X_val).to(device)

        if is_clf:
            y_tr = torch.LongTensor(y_train).to(device)
            y_vl = torch.LongTensor(y_val).to(device)
            criterion = nn.CrossEntropyLoss()
        else:
            y_tr = torch.FloatTensor(y_train).unsqueeze(-1).to(device)
            y_vl = torch.FloatTensor(y_val).unsqueeze(-1).to(device)
            criterion = nn.MSELoss()

        optimizer  = TorchAdam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        scheduler  = TorchReduceLR(optimizer, factor=0.5, patience=8, verbose=False)
        dataset    = TensorDataset(X_tr, y_tr)
        loader     = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                                drop_last=True)

        best_val_loss = np.inf
        patience_cnt  = 0
        best_state    = None
        t0            = time.time()

        for epoch in range(config.epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_vl)
                val_loss = criterion(val_pred, y_vl).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= config.early_stopping_patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        duration = time.time() - t0

        # Eval score
        model.eval()
        with torch.no_grad():
            tr_pred = model(X_tr)
            vl_pred = model(X_vl)
            if is_clf:
                tr_sc = accuracy_score(
                    y_train, tr_pred.argmax(dim=1).cpu().numpy())
                vl_sc = accuracy_score(
                    y_val,   vl_pred.argmax(dim=1).cpu().numpy())
                metric = "accuracy"
            else:
                tr_sc = r2_score(y_train, tr_pred.cpu().numpy().ravel())
                vl_sc = r2_score(y_val,   vl_pred.cpu().numpy().ravel())
                metric = "r2"

        logger.info(f"  PyTorch {model.__class__.__name__}: train={tr_sc:.4f} "
                    f"val={vl_sc:.4f} ({duration:.0f}s, ep={epoch+1})")
        return model.to("cpu"), {"train_score": tr_sc, "val_score": vl_sc,
                                  "metric": metric, "duration": duration}


# ============================================================================
#  SECTION 8 – HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ============================================================================

class HyperparamOptimizer:
    """Optuna-based HPO for all model types."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def optimize_xgboost(self, X_tr, y_tr, X_vl, y_vl,
                          is_clf: bool, n_trials: int = 50) -> dict:
        if not (OPTUNA_AVAILABLE and XGB_AVAILABLE):
            return {}

        def objective(trial):
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 12),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0, 5),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
            if is_clf:
                m = xgb.XGBClassifier(**params, objective="multi:softprob",
                                       num_class=3, eval_metric="mlogloss",
                                       early_stopping_rounds=20,
                                       random_state=42, n_jobs=-1,
                                       use_label_encoder=False)
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                return accuracy_score(y_vl, m.predict(X_vl))
            else:
                m = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                      eval_metric="rmse",
                                      early_stopping_rounds=20,
                                      random_state=42, n_jobs=-1)
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], verbose=False)
                return r2_score(y_vl, m.predict(X_vl))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=n_trials,
                       timeout=self.config.model_config.optuna_timeout,
                       show_progress_bar=False)
        self.logger.info(f"XGBoost HPO best params: {study.best_params}")
        return study.best_params

    def optimize_lightgbm(self, X_tr, y_tr, X_vl, y_vl,
                           is_clf: bool, n_trials: int = 50) -> dict:
        if not (OPTUNA_AVAILABLE and LGB_AVAILABLE):
            return {}

        def objective(trial):
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 12),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
                "num_leaves":       trial.suggest_int("num_leaves", 20, 150),
                "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "lambda_l1":        trial.suggest_float("lambda_l1", 0, 5),
                "lambda_l2":        trial.suggest_float("lambda_l2", 0, 5),
                "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            }
            cbs = [lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)]
            if is_clf:
                m = lgb.LGBMClassifier(**params, objective="multiclass",
                                        num_class=3, metric="multi_logloss",
                                        random_state=42, n_jobs=-1, verbose=-1)
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=cbs)
                return accuracy_score(y_vl, m.predict(X_vl))
            else:
                m = lgb.LGBMRegressor(**params, objective="regression",
                                       metric="rmse",
                                       random_state=42, n_jobs=-1, verbose=-1)
                m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], callbacks=cbs)
                return r2_score(y_vl, m.predict(X_vl))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=n_trials,
                       timeout=self.config.model_config.optuna_timeout,
                       show_progress_bar=False)
        self.logger.info(f"LightGBM HPO best params: {study.best_params}")
        return study.best_params

    def optimize_catboost(self, X_tr, y_tr, X_vl, y_vl,
                           is_clf: bool, n_trials: int = 30) -> dict:
        if not (OPTUNA_AVAILABLE and CAT_AVAILABLE):
            return {}

        def objective(trial):
            params = {
                "depth":        trial.suggest_int("depth", 4, 10),
                "learning_rate":trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "iterations":   trial.suggest_int("iterations", 200, 700),
                "l2_leaf_reg":  trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "subsample":    trial.suggest_float("subsample", 0.6, 1.0),
                "rsm":          trial.suggest_float("rsm", 0.6, 1.0),
                "random_strength": trial.suggest_float("random_strength", 0.1, 5.0),
            }
            if is_clf:
                m = CatBoostClassifier(**params, loss_function="MultiClass",
                                        classes_count=3,
                                        early_stopping_rounds=20, verbose=False,
                                        random_state=42)
                m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
                return accuracy_score(y_vl, m.predict(X_vl).ravel())
            else:
                m = CatBoostRegressor(**params, loss_function="RMSE",
                                       early_stopping_rounds=20, verbose=False,
                                       random_state=42)
                m.fit(X_tr, y_tr, eval_set=(X_vl, y_vl))
                return r2_score(y_vl, m.predict(X_vl))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials,
                       timeout=self.config.model_config.optuna_timeout,
                       show_progress_bar=False)
        self.logger.info(f"CatBoost HPO best params: {study.best_params}")
        return study.best_params


# ============================================================================
#  SECTION 9 – FEATURE IMPORTANCE & SELECTION
# ============================================================================

class FeatureImportanceAnalyzer:
    """Compute and aggregate feature importance across multiple models."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def extract_importance(self, model, feature_names: List[str],
                            model_name: str = "") -> Optional[pd.Series]:
        """Extract feature importance from a trained model."""
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                return pd.Series(imp, index=feature_names[:len(imp)]).sort_values(ascending=False)
            elif XGB_AVAILABLE and isinstance(model, (xgb.XGBRegressor, xgb.XGBClassifier)):
                imp = model.feature_importances_
                return pd.Series(imp, index=feature_names[:len(imp)]).sort_values(ascending=False)
            elif LGB_AVAILABLE and isinstance(model, (lgb.LGBMRegressor, lgb.LGBMClassifier)):
                imp = model.feature_importances_
                return pd.Series(imp, index=feature_names[:len(imp)]).sort_values(ascending=False)
            elif CAT_AVAILABLE and isinstance(model, (CatBoostRegressor, CatBoostClassifier)):
                imp = model.get_feature_importance()
                return pd.Series(imp, index=feature_names[:len(imp)]).sort_values(ascending=False)
        except Exception as e:
            self.logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
        return None

    def aggregate(self, importances: Dict[str, pd.Series]) -> pd.DataFrame:
        """Aggregate importances from multiple models (rank-based)."""
        if not importances:
            return pd.DataFrame()
        dfs = []
        for name, series in importances.items():
            if series is not None:
                ranked = (series.rank(ascending=False) / len(series)).rename(name)
                dfs.append(ranked)
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, axis=1).fillna(0)
        df["mean_rank"] = df.mean(axis=1)
        df = df.sort_values("mean_rank")
        return df

    def select_top_features(self, X: np.ndarray, y: np.ndarray,
                             feature_names: List[str], top_k: int = 100,
                             is_clf: bool = False) -> List[str]:
        """Select top features using mutual information."""
        if is_clf:
            mi = mutual_info_classif(X, y, random_state=42)
        else:
            mi = mutual_info_regression(X, y, random_state=42)
        mi_series = pd.Series(mi, index=feature_names).sort_values(ascending=False)
        selected  = mi_series.head(top_k).index.tolist()
        self.logger.info(f"Selected top {len(selected)} features by mutual info")
        return selected


# ============================================================================
#  SECTION 10 – ENSEMBLE MODELS
# ============================================================================

class VotingEnsemble:
    """Weighted average ensemble."""

    def __init__(self, models: List[Tuple[str, object]],
                 weights: Optional[List[float]] = None):
        self.models  = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for w, (name, m) in zip(self.weights, self.models):
            try:
                p = m.predict(X)
                preds.append(p * w)
            except Exception:
                pass
        return np.sum(preds, axis=0) if preds else np.zeros(len(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for w, (name, m) in zip(self.weights, self.models):
            try:
                if hasattr(m, "predict_proba"):
                    p = m.predict_proba(X)
                    preds.append(p * w)
            except Exception:
                pass
        return np.sum(preds, axis=0) if preds else np.zeros((len(X), 3))


class StackingEnsemble:
    """Stacking ensemble: base models → meta learner."""

    def __init__(self, base_models: List[Tuple[str, object]],
                 meta_model, is_clf: bool = False):
        self.base_models  = base_models
        self.meta_model   = meta_model
        self.is_clf       = is_clf
        self._fitted      = False

    def fit(self, X_train, y_train, X_val, y_val) -> "StackingEnsemble":
        # Meta-features from base models on validation set
        meta_feats = []
        for name, m in self.base_models:
            try:
                meta_feats.append(m.predict(X_val))
            except Exception:
                meta_feats.append(np.zeros(len(X_val)))
        meta_X = np.column_stack(meta_feats)
        self.meta_model.fit(meta_X, y_val)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_feats = []
        for name, m in self.base_models:
            try:
                meta_feats.append(m.predict(X))
            except Exception:
                meta_feats.append(np.zeros(len(X)))
        meta_X = np.column_stack(meta_feats)
        return self.meta_model.predict(meta_X)


# ============================================================================
#  SECTION 11 – BACKTESTING ENGINE
# ============================================================================

@dataclass
class Trade:
    entry_time:  Any
    exit_time:   Any
    entry_price: float
    exit_price:  float
    direction:   str   # "long" or "short"
    size:        float
    pnl:         float
    pnl_pct:     float
    confidence:  float


class BacktestEngine:
    """Walk-forward backtesting with realistic slippage/commission."""

    def __init__(self, config: BacktestConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def backtest_signals(
        self,
        signals: List[Dict],
        price_series: pd.Series
    ) -> Dict:
        """
        Simulate trades from a signal list.
        Each signal: {"timestamp": ts, "signal": int (0=sell,1=hold,2=buy),
                       "confidence": float, "predicted_price": float}
        """
        bc  = self.config
        cap = bc.initial_capital
        trades: List[Trade] = []
        equity  = [cap]
        in_pos  = False
        pos_dir = None
        entry_price = 0.0
        entry_time  = None
        entry_conf  = 0.0
        pos_size    = 0.0

        price_map = price_series.to_dict()
        sig_map   = {s["timestamp"]: s for s in signals}
        all_ts    = sorted(set(list(price_map.keys()) + list(sig_map.keys())))

        slippage = bc.slippage_pct / 100
        comm     = bc.commission_pct / 100

        for ts in all_ts:
            price = price_map.get(ts)
            if price is None:
                continue

            sig = sig_map.get(ts, {})
            signal     = sig.get("signal", 1)
            confidence = sig.get("confidence", 0.5)

            # Exit logic
            if in_pos:
                target_exit = False
                if pos_dir == "long":
                    tp_price = entry_price * (1 + bc.slippage_pct / 100 * 20)
                    sl_price = entry_price * (1 - 0.05)
                    if price >= tp_price or price <= sl_price or signal == 0:
                        target_exit = True
                else:
                    tp_price = entry_price * (1 - bc.slippage_pct / 100 * 20)
                    sl_price = entry_price * (1 + 0.05)
                    if price <= tp_price or price >= sl_price or signal == 2:
                        target_exit = True

                if target_exit:
                    exit_p   = price * (1 + slippage if pos_dir == "short" else 1 - slippage)
                    exit_p  *= (1 - comm)
                    pnl_pct  = ((exit_p - entry_price) / entry_price) if pos_dir == "long" \
                                else ((entry_price - exit_p) / entry_price)
                    pnl      = pnl_pct * pos_size
                    cap     += pnl
                    cap      = max(cap, 1.0)

                    trades.append(Trade(
                        entry_time=entry_time, exit_time=ts,
                        entry_price=entry_price, exit_price=exit_p,
                        direction=pos_dir, size=pos_size,
                        pnl=pnl, pnl_pct=pnl_pct * 100, confidence=entry_conf
                    ))
                    in_pos = False
                    equity.append(cap)

            # Entry logic
            if not in_pos and confidence >= 0.55:
                if signal == 2:
                    pos_dir    = "long"
                    pos_size   = cap * 0.95 * bc.leverage
                    entry_price = price * (1 + slippage)
                    entry_price *= (1 + comm)
                    entry_time = ts
                    entry_conf = confidence
                    in_pos     = True
                elif signal == 0:
                    pos_dir    = "short"
                    pos_size   = cap * 0.95 * bc.leverage
                    entry_price = price * (1 - slippage)
                    entry_price *= (1 + comm)
                    entry_time = ts
                    entry_conf = confidence
                    in_pos     = True

        return self._compute_metrics(trades, equity, bc.initial_capital)

    def _compute_metrics(self, trades: List[Trade],
                          equity: List[float],
                          initial_capital: float) -> Dict:
        if not trades:
            return {"num_trades": 0, "total_return_pct": 0.0,
                    "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0,
                    "win_rate": 0.0, "profit_factor": 0.0}

        pnls   = [t.pnl_pct for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_ret  = (equity[-1] - initial_capital) / initial_capital * 100
        win_rate   = len(wins) / len(pnls) * 100
        avg_win    = np.mean(wins)   if wins   else 0
        avg_loss   = np.mean(losses) if losses else 0
        profit_fac = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # Sharpe ratio (annualised)
        pnl_arr    = np.array(pnls)
        sharpe     = (np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-9)) * np.sqrt(252)

        # Max drawdown
        eq    = np.array(equity)
        peaks = np.maximum.accumulate(eq)
        dd    = (eq - peaks) / (peaks + 1e-9)
        max_dd = abs(dd.min()) * 100

        # Sortino
        neg_pnl  = pnl_arr[pnl_arr < 0]
        sortino  = (np.mean(pnl_arr) / (np.std(neg_pnl) + 1e-9)) * np.sqrt(252) \
                   if len(neg_pnl) > 0 else 0

        # Calmar ratio
        calmar   = (total_ret / max_dd) if max_dd > 0 else 0

        return {
            "num_trades":       len(trades),
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio":     round(sharpe, 3),
            "sortino_ratio":    round(sortino, 3),
            "calmar_ratio":     round(calmar, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "win_rate":         round(win_rate, 2),
            "profit_factor":    round(profit_fac, 3),
            "avg_win_pct":      round(avg_win, 3),
            "avg_loss_pct":     round(avg_loss, 3),
            "final_capital":    round(equity[-1], 2),
            "num_wins":         len(wins),
            "num_losses":       len(losses),
            "consecutive_wins": self._max_streak(pnls, positive=True),
            "consecutive_losses": self._max_streak(pnls, positive=False),
            "trades":           [vars(t) for t in trades[:20]],  # first 20
        }

    @staticmethod
    def _max_streak(pnls: List[float], positive: bool) -> int:
        best, curr = 0, 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p <= 0):
                curr += 1
                best  = max(best, curr)
            else:
                curr  = 0
        return best


# ============================================================================
#  SECTION 12 – SIGNAL GENERATOR
# ============================================================================

@dataclass
class PricePrediction:
    timestamp:           Any
    symbol:              str
    current_price:       float
    predicted_price:     float
    predicted_return:    float      # regression output
    predicted_direction: str        # "up" / "down" / "sideways"
    prob_up:             float
    prob_down:           float
    prob_sideways:       float
    confidence:          float
    lower_bound:         float
    upper_bound:         float
    model_votes:         Dict       = field(default_factory=dict)


@dataclass
class TradingSignal:
    timestamp:        Any
    symbol:           str
    signal_type:      str    # "buy" / "sell" / "hold"
    signal_int:       int    # 2 / 0 / 1
    confidence:       float
    entry_price:      float
    stop_loss:        float
    take_profit:      float
    risk_reward:      float
    position_size:    float  # fraction of capital
    sources:          List[str] = field(default_factory=list)


class SignalGenerator:
    """Combine model predictions with rule-based filters to emit trading signals."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pc     = config.prediction_config

    def generate(
        self,
        predictions: List[PricePrediction],
        features_df: pd.DataFrame,
    ) -> List[TradingSignal]:
        signals = []
        for pred in predictions:
            ts     = pred.timestamp
            signal = self._combine_signals(pred, features_df, ts)
            signals.append(signal)
        self.logger.info(f"Generated {len(signals)} signals")
        return signals

    def _combine_signals(
        self,
        pred: PricePrediction,
        feat: pd.DataFrame,
        ts: Any
    ) -> TradingSignal:

        scores: Dict[str, float] = {}   # +1 = bullish, -1 = bearish

        # --- Model prediction ---
        if pred.predicted_direction == "up":
            scores["model"] = pred.confidence
        elif pred.predicted_direction == "down":
            scores["model"] = -pred.confidence
        else:
            scores["model"] = 0.0

        # --- Technical signal (if features available) ---
        if ts in feat.index and self.pc.use_technical_signals:
            row = feat.loc[ts]
            # RSI
            rsi = row.get("rsi_14", 50)
            if rsi < 30:
                scores["rsi"] = 0.6
            elif rsi > 70:
                scores["rsi"] = -0.6
            else:
                scores["rsi"] = 0.0
            # MACD
            if row.get("macd_cross", 0) == 1 and row.get("macd_hist", 0) > 0:
                scores["macd"] = 0.5
            elif row.get("macd_cross", 0) == 0 and row.get("macd_hist", 0) < 0:
                scores["macd"] = -0.5
            # BB
            bb_pct = row.get("bb_pct", 0.5)
            if bb_pct < 0.1:
                scores["bb"] = 0.4
            elif bb_pct > 0.9:
                scores["bb"] = -0.4
            # ADX trend strength
            if row.get("adx", 0) > 25:
                scores["adx"] = np.sign(scores.get("model", 0)) * 0.3

        # --- Whale / on-chain ---
        if ts in feat.index and self.pc.use_whale_signals:
            row = feat.loc[ts]
            net_flow = row.get("exchange_netflow", 0)
            whale    = row.get("whale_volume_ratio", 0.3)
            if net_flow > 0 and whale > 0.5:
                scores["whale"] = 0.5
            elif net_flow < 0:
                scores["whale"] = -0.3

        # --- Sentiment ---
        if ts in feat.index and self.pc.use_sentiment_signals:
            row = feat.loc[ts]
            sent = row.get("ai_sentiment_score", 0)
            fg   = row.get("fear_greed_index", 50)
            if sent > 0.3:
                scores["sentiment"] = 0.4
            elif sent < -0.3:
                scores["sentiment"] = -0.4
            if fg < 20:
                scores["fear_greed"] = 0.5   # extreme fear = buy signal
            elif fg > 80:
                scores["fear_greed"] = -0.5  # extreme greed = caution

        # Aggregate score
        agg_score = np.mean(list(scores.values())) if scores else 0.0
        confidence = min(abs(agg_score), 1.0)

        if agg_score > 0.15:
            signal_type = "buy"
            signal_int  = 2
        elif agg_score < -0.15:
            signal_type = "sell"
            signal_int  = 0
        else:
            signal_type = "hold"
            signal_int  = 1

        ep    = pred.current_price
        sl    = ep * (1 - self.pc.stop_loss_pct)
        tp    = ep * (1 + self.pc.take_profit_pct)
        rr    = self.pc.take_profit_pct / (self.pc.stop_loss_pct + 1e-9)
        psize = self.pc.position_size_pct

        return TradingSignal(
            timestamp=ts,
            symbol=pred.symbol,
            signal_type=signal_type,
            signal_int=signal_int,
            confidence=round(confidence, 4),
            entry_price=round(ep, 4),
            stop_loss=round(sl, 4),
            take_profit=round(tp, 4),
            risk_reward=round(rr, 2),
            position_size=psize,
            sources=list(scores.keys()),
        )


# ============================================================================
#  SECTION 13 – PREDICTION GENERATOR (ENSEMBLE)
# ============================================================================

class PredictionGenerator:
    """
    Given a set of trained models and test sequences, produce ensemble
    PricePrediction objects.
    """

    def __init__(self, models: Dict, preprocessor: DataPreprocessor,
                 config: PipelineConfig, logger: logging.Logger):
        self.models       = models   # {"model_name": (model, info, is_keras, is_torch, is_clf)}
        self.preprocessor = preprocessor
        self.config       = config
        self.logger       = logger

    def predict_all(
        self,
        X_flat:  np.ndarray,
        X_seq:   np.ndarray,
        close:   pd.Series,
        timestamps: pd.DatetimeIndex,
        symbol:  str,
    ) -> List[PricePrediction]:

        clf_preds_list = []   # (n_samples, 3) prob arrays
        reg_preds_list = []   # (n_samples,) return arrays

        for name, (model, info, is_keras, is_torch, is_clf) in self.models.items():
            try:
                if is_keras:
                    X_input = X_seq
                    raw     = model.predict(X_input, verbose=0)
                elif is_torch:
                    model.eval()
                    with torch.no_grad():
                        xt  = torch.FloatTensor(X_seq)
                        raw = model(xt).numpy()
                else:
                    X_input = X_flat
                    raw     = None

                if is_clf:
                    if is_keras or is_torch:
                        # raw is (N, 3) softmax
                        clf_preds_list.append(raw)
                    else:
                        if hasattr(model, "predict_proba"):
                            clf_preds_list.append(model.predict_proba(X_input))
                else:
                    if is_keras or is_torch:
                        reg_preds_list.append(raw.ravel())
                    else:
                        reg_preds_list.append(model.predict(X_input))
            except Exception as e:
                self.logger.warning(f"Prediction failed for {name}: {e}")

        n_samples  = len(X_flat)
        # Determine which timestamps correspond to test set
        seq_len    = self.config.model_config.sequence_length
        ph         = self.config.model_config.prediction_horizon
        test_n     = min(len(X_flat), len(X_seq))
        ts_slice   = timestamps[-(test_n):]
        price_slice = close.values[-(test_n):]

        # Ensemble clf probabilities
        if clf_preds_list:
            # Align lengths
            min_len = min(len(p) for p in clf_preds_list)
            clf_ensemble = np.mean([p[-min_len:] for p in clf_preds_list], axis=0)
        else:
            clf_ensemble = None

        # Ensemble regression predictions
        if reg_preds_list:
            min_len = min(len(p) for p in reg_preds_list)
            reg_ensemble = np.mean([p[-min_len:] for p in reg_preds_list], axis=0)
        else:
            reg_ensemble = None

        predictions = []
        n = min(
            len(ts_slice),
            len(clf_ensemble) if clf_ensemble is not None else 9999,
            len(reg_ensemble) if reg_ensemble is not None else 9999,
        )

        for i in range(n):
            ts    = ts_slice[-(n-i)] if n <= len(ts_slice) else ts_slice[i]
            price = price_slice[-(n-i)] if n <= len(price_slice) else price_slice[i]

            # Classification probabilities
            if clf_ensemble is not None:
                probs = clf_ensemble[i]
                p_dn, p_sw, p_up = probs[0], probs[1], probs[2]
            else:
                p_up, p_dn, p_sw = 0.33, 0.33, 0.34

            # Determine direction
            max_prob = max(p_up, p_dn, p_sw)
            if p_up == max_prob:
                direction = "up"
                confidence = p_up
            elif p_dn == max_prob:
                direction = "down"
                confidence = p_dn
            else:
                direction  = "sideways"
                confidence = p_sw

            # Regression: expected return
            if reg_ensemble is not None:
                pred_return = float(reg_ensemble[i])
            else:
                pred_return = (p_up - p_dn) * 0.02  # proxy

            pred_price = price * (1 + pred_return)

            # Confidence interval (±2σ of regression predictions spread)
            if reg_ensemble is not None and len(reg_ensemble) > 1:
                sigma = np.std(reg_ensemble) * price
            else:
                sigma = price * 0.03

            predictions.append(PricePrediction(
                timestamp=ts,
                symbol=symbol,
                current_price=round(float(price), 4),
                predicted_price=round(pred_price, 4),
                predicted_return=round(pred_return * 100, 4),
                predicted_direction=direction,
                prob_up=round(p_up, 4),
                prob_down=round(p_dn, 4),
                prob_sideways=round(p_sw, 4),
                confidence=round(confidence, 4),
                lower_bound=round(pred_price - 2 * sigma, 4),
                upper_bound=round(pred_price + 2 * sigma, 4),
            ))

        self.logger.info(f"Generated {len(predictions)} predictions for {symbol}")
        return predictions


# ============================================================================
#  SECTION 14 – MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """Compute comprehensive metrics on the test set."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def evaluate_clf(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_prob: np.ndarray = None, model_name: str = "") -> Dict:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm     = confusion_matrix(y_true, y_pred).tolist()
        out    = {
            "model":     model_name,
            "accuracy":  accuracy_score(y_true, y_pred),
            "f1_macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
            "confusion_matrix": cm,
            "class_report": report,
        }
        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 3:
            try:
                out["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            except Exception:
                pass
        self.logger.info(
            f"  {model_name} TEST – acc={out['accuracy']:.4f} "
            f"f1={out['f1_macro']:.4f}"
        )
        return out

    def evaluate_reg(self, y_true: np.ndarray, y_pred: np.ndarray,
                      model_name: str = "") -> Dict:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        out  = {
            "model": model_name,
            "r2":    r2_score(y_true, y_pred),
            "mae":   mean_absolute_error(y_true, y_pred),
            "rmse":  rmse,
            "mape":  np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100,
        }
        self.logger.info(
            f"  {model_name} TEST – r2={out['r2']:.4f} rmse={out['rmse']:.6f}"
        )
        return out


# ============================================================================
#  SECTION 15 – MAIN TRAINING PIPELINE ORCHESTRATOR
# ============================================================================

class CryptoTrainingPipeline:
    """
    Main orchestrator.  Runs end-to-end:
      1. Data collection
      2. Feature engineering
      3. Data preparation
      4. Hyperparameter optimisation
      5. Train all models (tree + DL + PyTorch)
      6. Ensemble construction
      7. Prediction generation
      8. Signal generation
      9. Backtesting
     10. Save models + artifacts
    """

    def __init__(self, config: PipelineConfig, args=None):
        self.config      = config
        self.args        = args
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.model_config.model_save_dir, exist_ok=True)
        os.makedirs(config.model_config.checkpoint_dir, exist_ok=True)

        self.logger      = setup_logging(config.output_dir)
        self.aggregator  = DataAggregator(config.data_config, self.logger)
        self.preprocessor = DataPreprocessor(config, self.logger)
        self.evaluator   = ModelEvaluator(self.logger)
        self.hpo         = HyperparamOptimizer(config, self.logger)
        self.feat_imp    = FeatureImportanceAnalyzer(self.logger)
        self.backtester  = BacktestEngine(config.backtest_config, self.logger)
        self.sig_gen     = SignalGenerator(config, self.logger)

        # Storage
        self.raw_data:   Optional[pd.DataFrame] = None
        self.features:   Optional[pd.DataFrame] = None
        self.flat_data:  Optional[Dict]         = None
        self.seq_data:   Optional[Dict]         = None
        self.trained_models: Dict               = {}
        self.importances:    Dict               = {}
        self.eval_results:   Dict               = defaultdict(dict)
        self.predictions:    List               = []
        self.signals:        List               = []
        self.backtest_res:   Dict               = {}
        self.results_summary: Dict              = {}

        # Seed
        random.seed(config.model_config.random_seed)
        np.random.seed(config.model_config.random_seed)
        if TF_AVAILABLE:
            tf.random.set_seed(config.model_config.random_seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(config.model_config.random_seed)

    # ------------------------------------------------------------------
    # STEP 1: DATA
    # ------------------------------------------------------------------

    def step_collect_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        self.logger.info(f"\n{'='*60}\nSTEP 1: DATA COLLECTION – {symbol}/{timeframe}\n{'='*60}")
        df = self.aggregator.fetch_all(symbol, timeframe)
        self.raw_data = df
        self.logger.info(f"Raw data: {df.shape[0]} rows x {df.shape[1]} cols")
        return df

    # ------------------------------------------------------------------
    # STEP 2: FEATURES
    # ------------------------------------------------------------------

    def step_engineer_features(self) -> pd.DataFrame:
        self.logger.info(f"\n{'='*60}\nSTEP 2: FEATURE ENGINEERING\n{'='*60}")
        eng = FeatureEngineer(self.raw_data, self.config, self.logger)
        df  = eng.engineer_all_features()
        self.features = df
        return df

    # ------------------------------------------------------------------
    # STEP 3: PREPARE DATA
    # ------------------------------------------------------------------

    def step_prepare_data(self) -> Tuple[Dict, Dict]:
        self.logger.info(f"\n{'='*60}\nSTEP 3: DATA PREPARATION\n{'='*60}")
        flat, seq = self.preprocessor.prepare(
            self.features, self.features["close"],
            task_type=self.config.model_config.task_type.value
        )
        self.flat_data = flat
        self.seq_data  = seq
        return flat, seq

    # ------------------------------------------------------------------
    # STEP 4 + 5: HPO + TRAINING
    # ------------------------------------------------------------------

    def step_train_all_models(self, symbol: str):
        self.logger.info(f"\n{'='*60}\nSTEP 4-5: HYPERPARAMETER OPT + TRAINING\n{'='*60}")

        mc      = self.config.model_config
        flat    = self.flat_data
        seq     = self.seq_data
        models  = self.trained_models

        task    = mc.task_type.value
        is_clf  = (task == "classification")
        is_reg  = (task == "regression")
        is_multi = (task == "multi_task")

        model_names = [m.value for m in mc.models_to_train]
        n_feat      = flat["X_train"].shape[1]
        input_shape = (seq["seq_len"], seq["n_features"]) if seq else None

        # ---------- CLASSIFICATION models ----------
        def train_clf_models():
            Xtr, Xvl, Xte = flat["X_train"], flat["X_val"], flat["X_test"]
            ytr, yvl, yte = flat["y_clf_train"], flat["y_clf_val"], flat["y_clf_test"]

            # XGBoost CLF
            if "xgboost" in model_names and XGB_AVAILABLE:
                self.logger.info("[CLF] Training XGBoost …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_xgboost(Xtr, ytr, Xvl, yvl,
                                                        is_clf=True,
                                                        n_trials=mc.n_trials // 3)
                m = SklearnModels.build_xgboost_clf(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="XGB_CLF", logger=self.logger)
                models["xgb_clf"] = (m, info, False, False, True)
                self.importances["xgb_clf"] = self.feat_imp.extract_importance(
                    m, list(self.features.columns[1:]), "xgb_clf")

            # LightGBM CLF
            if "lightgbm" in model_names and LGB_AVAILABLE:
                self.logger.info("[CLF] Training LightGBM …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_lightgbm(Xtr, ytr, Xvl, yvl,
                                                         is_clf=True,
                                                         n_trials=mc.n_trials // 3)
                m = SklearnModels.build_lightgbm_clf(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="LGB_CLF", logger=self.logger)
                models["lgb_clf"] = (m, info, False, False, True)
                self.importances["lgb_clf"] = self.feat_imp.extract_importance(
                    m, list(self.features.columns[1:]), "lgb_clf")

            # CatBoost CLF
            if "catboost" in model_names and CAT_AVAILABLE:
                self.logger.info("[CLF] Training CatBoost …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_catboost(Xtr, ytr, Xvl, yvl,
                                                         is_clf=True,
                                                         n_trials=mc.n_trials // 4)
                m = SklearnModels.build_catboost_clf(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="CAT_CLF", logger=self.logger)
                models["cat_clf"] = (m, info, False, False, True)

            # Random Forest CLF
            if "random_forest" in model_names:
                self.logger.info("[CLF] Training RandomForest …")
                m = SklearnModels.build_random_forest_clf()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="RF_CLF", logger=self.logger)
                models["rf_clf"] = (m, info, False, False, True)

            # Extra Trees CLF
            if "extra_trees" in model_names:
                self.logger.info("[CLF] Training ExtraTrees …")
                m = SklearnModels.build_extra_trees_clf()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="ET_CLF", logger=self.logger)
                models["et_clf"] = (m, info, False, False, True)

            # Gradient Boosting CLF
            if "gradient_boosting" in model_names:
                self.logger.info("[CLF] Training GradientBoosting …")
                m = SklearnModels.build_gradient_boosting_clf()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=True,
                                              model_name="GB_CLF", logger=self.logger)
                models["gb_clf"] = (m, info, False, False, True)

        # ---------- REGRESSION models ----------
        def train_reg_models():
            Xtr, Xvl, Xte = flat["X_train"], flat["X_val"], flat["X_test"]
            ytr, yvl, yte = flat["y_reg_train"], flat["y_reg_val"], flat["y_reg_test"]

            if "xgboost" in model_names and XGB_AVAILABLE:
                self.logger.info("[REG] Training XGBoost …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_xgboost(Xtr, ytr, Xvl, yvl,
                                                        is_clf=False,
                                                        n_trials=mc.n_trials // 3)
                m = SklearnModels.build_xgboost_reg(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=False,
                                              model_name="XGB_REG", logger=self.logger)
                models["xgb_reg"] = (m, info, False, False, False)

            if "lightgbm" in model_names and LGB_AVAILABLE:
                self.logger.info("[REG] Training LightGBM …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_lightgbm(Xtr, ytr, Xvl, yvl,
                                                         is_clf=False,
                                                         n_trials=mc.n_trials // 3)
                m = SklearnModels.build_lightgbm_reg(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=False,
                                              model_name="LGB_REG", logger=self.logger)
                models["lgb_reg"] = (m, info, False, False, False)

            if "catboost" in model_names and CAT_AVAILABLE:
                self.logger.info("[REG] Training CatBoost …")
                best_p = {}
                if mc.use_optuna:
                    best_p = self.hpo.optimize_catboost(Xtr, ytr, Xvl, yvl,
                                                         is_clf=False,
                                                         n_trials=mc.n_trials // 4)
                m = SklearnModels.build_catboost_reg(best_p)
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              is_classification=False,
                                              model_name="CAT_REG", logger=self.logger)
                models["cat_reg"] = (m, info, False, False, False)

            if "random_forest" in model_names:
                m = SklearnModels.build_random_forest_reg()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              model_name="RF_REG", logger=self.logger)
                models["rf_reg"] = (m, info, False, False, False)

            if "extra_trees" in model_names:
                m = SklearnModels.build_extra_trees_reg()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              model_name="ET_REG", logger=self.logger)
                models["et_reg"] = (m, info, False, False, False)

            if "ridge" in model_names:
                m = SklearnModels.build_ridge()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              model_name="RIDGE", logger=self.logger)
                models["ridge_reg"] = (m, info, False, False, False)

            if "lasso" in model_names:
                m = SklearnModels.build_lasso()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              model_name="LASSO", logger=self.logger)
                models["lasso_reg"] = (m, info, False, False, False)

            if "elastic_net" in model_names:
                m = SklearnModels.build_elastic_net()
                m, info = train_sklearn_model(m, Xtr, ytr, Xvl, yvl,
                                              model_name="ELASTICNET", logger=self.logger)
                models["enet_reg"] = (m, info, False, False, False)

        # ---------- DEEP LEARNING (Keras) ----------
        def train_dl_models(for_clf: bool):
            if not TF_AVAILABLE or input_shape is None:
                return
            Xtr = seq["X_train"]
            Xvl = seq["X_val"]
            ytr = seq["y_clf_train"] if for_clf else seq["y_reg_train"]
            yvl = seq["y_clf_val"]   if for_clf else seq["y_reg_val"]
            suffix = "clf" if for_clf else "reg"
            ckpt   = self.config.model_config.checkpoint_dir

            dl_map = {
                "lstm":              KerasModelFactory.build_lstm,
                "gru":               KerasModelFactory.build_gru,
                "bidirectional_lstm":KerasModelFactory.build_attention_lstm,
                "transformer":       KerasModelFactory.build_transformer,
                "tft":               KerasModelFactory.build_temporal_fusion_transformer,
                "cnn_lstm":          KerasModelFactory.build_cnn_lstm,
                "cnn_transformer":   KerasModelFactory.build_cnn_transformer,
                "wavenet":           KerasModelFactory.build_wavenet,
            }

            for key, builder in dl_map.items():
                if key not in model_names:
                    continue
                self.logger.info(f"[{'CLF' if for_clf else 'REG'}] Training Keras {key} …")
                try:
                    m = builder(input_shape, is_clf=for_clf,
                                dropout=mc.dropout_rate)
                    ckpt_path = os.path.join(ckpt, f"{key}_{suffix}.h5")
                    m, info = compile_and_train_keras(
                        m, Xtr, ytr, Xvl, yvl,
                        config=mc, is_clf=for_clf,
                        checkpoint_path=ckpt_path,
                        logger=self.logger
                    )
                    models[f"{key}_{suffix}"] = (m, info, True, False, for_clf)
                except Exception as e:
                    self.logger.error(f"Keras {key} training failed: {e}")

        # ---------- PYTORCH models ----------
        def train_pytorch_models(for_clf: bool):
            if not TORCH_AVAILABLE or input_shape is None:
                return
            Xtr = seq["X_train"]
            Xvl = seq["X_val"]
            ytr = seq["y_clf_train"] if for_clf else seq["y_reg_train"]
            yvl = seq["y_clf_val"]   if for_clf else seq["y_reg_val"]
            n_classes = 3 if for_clf else 1
            suffix    = "clf" if for_clf else "reg"

            if "pytorch_lstm" in model_names:
                self.logger.info(f"[{'CLF' if for_clf else 'REG'}] Training PyTorch LSTM …")
                try:
                    m = PyTorchLSTM(input_size=seq["n_features"],
                                     num_classes=n_classes)
                    m, info = train_pytorch_model(m, Xtr, ytr, Xvl, yvl,
                                                  config=mc,
                                                  is_clf=for_clf,
                                                  logger=self.logger)
                    models[f"pt_lstm_{suffix}"] = (m, info, False, True, for_clf)
                except Exception as e:
                    self.logger.error(f"PyTorch LSTM failed: {e}")

            if "pytorch_transformer" in model_names:
                self.logger.info(f"[{'CLF' if for_clf else 'REG'}] Training PyTorch Transformer …")
                try:
                    m = PyTorchTransformer(input_size=seq["n_features"],
                                            num_classes=n_classes,
                                            max_len=seq["seq_len"])
                    m, info = train_pytorch_model(m, Xtr, ytr, Xvl, yvl,
                                                  config=mc,
                                                  is_clf=for_clf,
                                                  logger=self.logger)
                    models[f"pt_transformer_{suffix}"] = (m, info, False, True, for_clf)
                except Exception as e:
                    self.logger.error(f"PyTorch Transformer failed: {e}")

        # ---- Route training by task type ----
        if is_clf or is_multi:
            train_clf_models()
            train_dl_models(for_clf=True)
            train_pytorch_models(for_clf=True)

        if is_reg or is_multi:
            train_reg_models()
            train_dl_models(for_clf=False)
            train_pytorch_models(for_clf=False)

        self.logger.info(f"Total models trained: {len(models)}")

    # ------------------------------------------------------------------
    # STEP 6: ENSEMBLE CONSTRUCTION
    # ------------------------------------------------------------------

    def step_build_ensembles(self):
        self.logger.info(f"\n{'='*60}\nSTEP 6: ENSEMBLE CONSTRUCTION\n{'='*60}")

        clf_models = [(n, m) for n, (m, _, _, _, is_c) in self.trained_models.items()
                      if is_c and not isinstance(m, (type(None),))]
        reg_models = [(n, m) for n, (m, _, _, _, is_c) in self.trained_models.items()
                      if not is_c]

        flat = self.flat_data

        # ---- Voting ensemble (classification) ----
        if len(clf_models) >= 2 and "ensemble_voting" in [m.value for m in self.config.model_config.models_to_train]:
            # Filter to sklearn-style (have predict_proba)
            flat_clf = [(n, m) for n, m in clf_models
                        if hasattr(m, "predict_proba") and not (
                            TORCH_AVAILABLE and isinstance(m, nn.Module)
                        ) and not (TF_AVAILABLE and isinstance(m, Model))]
            if flat_clf:
                # Weight by val accuracy
                weights = []
                for n, _ in flat_clf:
                    info = self.trained_models[n][1]
                    weights.append(max(info.get("val_score", 0.5), 0.01))
                w_sum = sum(weights) + 1e-9
                weights = [w / w_sum for w in weights]
                ensemble = VotingEnsemble(flat_clf, weights)
                self.trained_models["voting_ensemble_clf"] = (ensemble, {"val_score": 0}, False, False, True)
                self.logger.info(f"Voting ensemble (clf) built with {len(flat_clf)} models")

        # ---- Stacking ensemble (classification) ----
        if len(clf_models) >= 2 and "ensemble_stacking" in [m.value for m in self.config.model_config.models_to_train]:
            flat_clf = [(n, m) for n, m in clf_models
                        if hasattr(m, "predict") and not (
                            TORCH_AVAILABLE and isinstance(m, nn.Module)
                        ) and not (TF_AVAILABLE and isinstance(m, Model))]
            if flat_clf:
                meta   = SklearnModels.build_lightgbm_clf() if LGB_AVAILABLE else SklearnModels.build_random_forest_clf()
                stk    = StackingEnsemble(flat_clf, meta, is_clf=True)
                stk.fit(flat["X_train"], flat["y_clf_train"],
                        flat["X_val"],   flat["y_clf_val"])
                self.trained_models["stacking_ensemble_clf"] = (stk, {"val_score": 0}, False, False, True)
                self.logger.info(f"Stacking ensemble (clf) built with {len(flat_clf)} bases")

        # ---- Voting ensemble (regression) ----
        if len(reg_models) >= 2 and "ensemble_voting" in [m.value for m in self.config.model_config.models_to_train]:
            flat_reg = [(n, m) for n, m in reg_models
                        if hasattr(m, "predict") and not (
                            TORCH_AVAILABLE and isinstance(m, nn.Module)
                        ) and not (TF_AVAILABLE and isinstance(m, Model))]
            if flat_reg:
                weights = []
                for n, _ in flat_reg:
                    info = self.trained_models[n][1]
                    weights.append(max(info.get("val_score", 0.1) + 1, 0.01))
                w_sum = sum(weights) + 1e-9
                weights = [w / w_sum for w in weights]
                ensemble = VotingEnsemble(flat_reg, weights)
                self.trained_models["voting_ensemble_reg"] = (ensemble, {"val_score": 0}, False, False, False)
                self.logger.info(f"Voting ensemble (reg) built with {len(flat_reg)} models")

    # ------------------------------------------------------------------
    # STEP 7: EVALUATE ON TEST SET
    # ------------------------------------------------------------------

    def step_evaluate(self, symbol: str):
        self.logger.info(f"\n{'='*60}\nSTEP 7: TEST SET EVALUATION\n{'='*60}")

        flat = self.flat_data
        seq  = self.seq_data

        for name, (model, info, is_keras, is_torch, is_clf) in self.trained_models.items():
            try:
                if is_keras:
                    X_te = seq["X_test"]
                    raw  = model.predict(X_te, verbose=0)
                elif is_torch:
                    model.eval()
                    with torch.no_grad():
                        xt  = torch.FloatTensor(seq["X_test"])
                        raw = model(xt).numpy()
                else:
                    X_te = flat["X_test"]
                    raw  = None

                if is_clf:
                    y_te = flat["y_clf_test"] if not is_keras and not is_torch \
                           else seq["y_clf_test"]
                    # Align lengths
                    n_te = min(len(y_te), len(raw) if raw is not None else len(model.predict(X_te)))

                    if is_keras or is_torch:
                        pred_proba = raw[-n_te:]
                        pred_class = pred_proba.argmax(axis=1)
                    elif hasattr(model, "predict_proba"):
                        pred_proba = model.predict_proba(X_te)[-n_te:]
                        pred_class = model.predict(X_te)[-n_te:]
                    else:
                        pred_class = model.predict(X_te)[-n_te:]
                        pred_proba = None

                    y_te_aligned = y_te[-n_te:]
                    ev = self.evaluator.evaluate_clf(
                        y_te_aligned, pred_class,
                        pred_proba if pred_proba is not None else None,
                        model_name=name
                    )
                    self.eval_results["classification"][name] = ev

                else:
                    y_te = flat["y_reg_test"] if not is_keras and not is_torch \
                           else seq["y_reg_test"]
                    if is_keras or is_torch:
                        y_pred = raw.ravel()
                    else:
                        y_pred = model.predict(X_te)
                    n_te   = min(len(y_te), len(y_pred))
                    ev = self.evaluator.evaluate_reg(
                        y_te[-n_te:], y_pred[-n_te:], model_name=name
                    )
                    self.eval_results["regression"][name] = ev

            except Exception as e:
                self.logger.warning(f"Evaluation failed for {name}: {e}")

    # ------------------------------------------------------------------
    # STEP 8: PREDICTIONS + SIGNALS + BACKTEST
    # ------------------------------------------------------------------

    def step_predict_signal_backtest(self, symbol: str):
        self.logger.info(f"\n{'='*60}\nSTEP 8: PREDICT / SIGNAL / BACKTEST\n{'='*60}")

        flat = self.flat_data
        seq  = self.seq_data
        close = self.features["close"]
        ts    = self.features.index

        pred_gen = PredictionGenerator(
            models=self.trained_models,
            preprocessor=self.preprocessor,
            config=self.config,
            logger=self.logger,
        )
        self.predictions = pred_gen.predict_all(
            X_flat=flat["X_test"],
            X_seq=seq["X_test"],
            close=close,
            timestamps=ts,
            symbol=symbol,
        )

        self.signals = self.sig_gen.generate(self.predictions, self.features)

        # Convert signals to backtest format
        bt_signals = [
            {
                "timestamp":       s.timestamp,
                "signal":          s.signal_int,
                "confidence":      s.confidence,
                "predicted_price": (p.predicted_price
                                    if i < len(self.predictions) else s.entry_price),
            }
            for i, (s, p) in enumerate(zip(
                self.signals,
                self.predictions + [self.predictions[-1]] if self.predictions else []
            ))
        ] if self.signals else []

        if bt_signals:
            price_series = close.rename("price")
            self.backtest_res = self.backtester.backtest_signals(bt_signals, price_series)
            self.logger.info(
                f"Backtest: return={self.backtest_res.get('total_return_pct', 0):.1f}% "
                f"sharpe={self.backtest_res.get('sharpe_ratio', 0):.2f} "
                f"winrate={self.backtest_res.get('win_rate', 0):.1f}%"
            )

    # ------------------------------------------------------------------
    # STEP 9: FEATURE IMPORTANCE SUMMARY
    # ------------------------------------------------------------------

    def step_feature_importance(self) -> pd.DataFrame:
        self.logger.info(f"\n{'='*60}\nSTEP 9: FEATURE IMPORTANCE\n{'='*60}")
        df = self.feat_imp.aggregate(self.importances)
        if not df.empty:
            self.logger.info(f"Top 10 features:\n{df.head(10).to_string()}")
        return df

    # ------------------------------------------------------------------
    # STEP 10: SAVE EVERYTHING
    # ------------------------------------------------------------------

    def step_save(self, symbol: str, fi_df: pd.DataFrame):
        self.logger.info(f"\n{'='*60}\nSTEP 10: SAVING ARTIFACTS\n{'='*60}")

        out = self.config.output_dir
        mdir = self.config.model_config.model_save_dir

        # ---- Save each model ----
        for name, (model, info, is_keras, is_torch, is_clf) in self.trained_models.items():
            try:
                if is_keras:
                    path = os.path.join(mdir, f"{name}.h5")
                    model.save(path, save_format="h5")
                elif is_torch:
                    path = os.path.join(mdir, f"{name}.pt")
                    torch.save(model.state_dict(), path)
                else:
                    path = os.path.join(mdir, f"{name}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(model, f)
                self.logger.info(f"  Saved {name} → {path}")
            except Exception as e:
                self.logger.warning(f"  Save failed for {name}: {e}")

        # ---- Save preprocessor scalers ----
        self.preprocessor.save(os.path.join(mdir, "preprocessor.pkl"))

        # ---- Predictions CSV ----
        if self.predictions:
            pred_rows = [vars(p) for p in self.predictions]
            pd.DataFrame(pred_rows).to_csv(
                os.path.join(out, f"predictions_{symbol}.csv"), index=False)

        # ---- Signals CSV ----
        if self.signals:
            sig_rows = [vars(s) for s in self.signals]
            pd.DataFrame(sig_rows).to_csv(
                os.path.join(out, f"signals_{symbol}.csv"), index=False)

        # ---- Backtest CSV ----
        if self.backtest_res and "trades" in self.backtest_res:
            trades = self.backtest_res.pop("trades", [])
            pd.DataFrame(trades).to_csv(
                os.path.join(out, f"backtest_trades_{symbol}.csv"), index=False)

        # ---- Feature importance CSV ----
        if not fi_df.empty:
            fi_df.to_csv(os.path.join(out, "feature_importance.csv"))

        # ---- Results summary JSON ----
        self.results_summary = {
            "symbol":     symbol,
            "timestamp":  datetime.now().isoformat(),
            "data_shape": list(self.raw_data.shape) if self.raw_data is not None else [],
            "n_features": len(self.features.columns) if self.features is not None else 0,
            "models_trained": list(self.trained_models.keys()),
            "eval_classification": {
                k: {m: {"accuracy": v.get("accuracy"), "f1_macro": v.get("f1_macro")}
                    for m, v in ev.items()}
                for k, ev in self.eval_results.items() if k == "classification"
            },
            "eval_regression": {
                k: {m: {"r2": v.get("r2"), "rmse": v.get("rmse")}
                    for m, v in ev.items()}
                for k, ev in self.eval_results.items() if k == "regression"
            },
            "backtest": {k: v for k, v in self.backtest_res.items()
                         if k != "trades"},
            "n_predictions": len(self.predictions),
            "n_signals":     len(self.signals),
        }

        summary_path = os.path.join(out, "results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.results_summary, f, indent=2, default=str)

        self.logger.info(f"Results summary saved → {summary_path}")

    # ------------------------------------------------------------------
    # FULL PIPELINE RUNNER
    # ------------------------------------------------------------------

    def run(self, symbol: str, timeframe: str = "1d"):
        t_start = time.time()
        self.logger.info(f"\n{'#'*70}")
        self.logger.info(f"#  CRYPTO PREDICTION TRAINING PIPELINE")
        self.logger.info(f"#  Symbol: {symbol}  |  Timeframe: {timeframe}")
        self.logger.info(f"#  Task: {self.config.model_config.task_type.value}")
        self.logger.info(f"{'#'*70}")

        try:
            self.step_collect_data(symbol, timeframe)
            self.step_engineer_features()
            self.step_prepare_data()
            self.step_train_all_models(symbol)
            self.step_build_ensembles()
            self.step_evaluate(symbol)
            self.step_predict_signal_backtest(symbol)
            fi_df = self.step_feature_importance()
            self.step_save(symbol, fi_df)

            elapsed = time.time() - t_start
            self.logger.info(f"\n✓ PIPELINE COMPLETE in {elapsed:.1f}s")
            self.logger.info(f"  Output directory: {self.config.output_dir}")
            self._print_summary()

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}\n{traceback.format_exc()}")
            raise

    def _print_summary(self):
        s = self.results_summary
        self.logger.info("\n" + "="*60)
        self.logger.info("RESULTS SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"  Models trained     : {len(s.get('models_trained', []))}")
        self.logger.info(f"  Features engineered: {s.get('n_features', 0)}")
        self.logger.info(f"  Predictions        : {s.get('n_predictions', 0)}")
        self.logger.info(f"  Signals            : {s.get('n_signals', 0)}")

        bt = s.get("backtest", {})
        if bt:
            self.logger.info(f"  Backtest return    : {bt.get('total_return_pct', 0):.2f}%")
            self.logger.info(f"  Sharpe ratio       : {bt.get('sharpe_ratio', 0):.3f}")
            self.logger.info(f"  Max drawdown       : {bt.get('max_drawdown_pct', 0):.2f}%")
            self.logger.info(f"  Win rate           : {bt.get('win_rate', 0):.2f}%")
            self.logger.info(f"  Trades             : {bt.get('num_trades', 0)}")


# ============================================================================
#  SECTION 16 – CLI & ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced Crypto Prediction Training Pipeline"
    )
    parser.add_argument("--symbol",  type=str,   default="RNDR",
                        help="Trading symbol (e.g. RNDR, BTC, ETH)")
    parser.add_argument("--tf",      type=str,   default="4h",
                        help="Timeframe: 1m 5m 15m 1h 4h 1d 1w")
    parser.add_argument("--task",    type=str,   default="multi_task",
                        choices=["classification", "regression", "multi_task"],
                        help="Prediction task type")
    parser.add_argument("--days",    type=int,   default=730,
                        help="Lookback days for historical data")
    parser.add_argument("--epochs",  type=int,   default=150,
                        help="Max training epochs for DL models")
    parser.add_argument("--trials",  type=int,   default=80,
                        help="Optuna hyperparameter optimization trials")
    parser.add_argument("--output",  type=str,   default="/tmp/crypto_train_output",
                        help="Output directory for all artifacts")
    parser.add_argument("--no-gpu",  action="store_true",
                        help="Disable GPU even if available")
    parser.add_argument("--no-optuna", action="store_true",
                        help="Skip hyperparameter optimization")
    parser.add_argument("--fast",    action="store_true",
                        help="Fast mode: fewer epochs/trials for quick testing")
    parser.add_argument("--models",  type=str,   default=None,
                        help="Comma-separated list of models to train "
                             "(e.g. xgboost,lightgbm,lstm,transformer)")
    return parser.parse_args()


def build_config_from_args(args) -> PipelineConfig:
    config = PipelineConfig()
    config.output_dir = args.output

    dc = config.data_config
    dc.primary_symbol = args.symbol
    dc.lookback_days  = args.days

    mc = config.model_config
    mc.task_type   = TaskType(args.task)
    mc.use_gpu     = not args.no_gpu
    mc.use_optuna  = not args.no_optuna
    mc.epochs      = args.epochs
    mc.n_trials    = args.trials

    if args.fast:
        mc.epochs   = 20
        mc.n_trials = 15
        dc.lookback_days = 180
        mc.sequence_length  = 30
        mc.prediction_horizon = 6
        mc.models_to_train = [
            ModelType.XGBOOST,
            ModelType.LIGHTGBM,
            ModelType.RANDOM_FOREST,
            ModelType.ENSEMBLE_VOTING,
            ModelType.ENSEMBLE_STACKING,
        ]

    if args.models:
        selected = [m.strip() for m in args.models.split(",")]
        mc.models_to_train = [
            ModelType(m) for m in selected
            if m in [mt.value for mt in ModelType]
        ]

    # GPU configuration for TF
    if TF_AVAILABLE and mc.use_gpu and mc.mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    if TF_AVAILABLE and mc.use_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

    return config


def main():
    args   = parse_args()
    config = build_config_from_args(args)

    # Load API keys from environment if .env file present
    env_path = Path(".env")
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        config.data_config = DataConfig.from_env()
        config.data_config.primary_symbol = args.symbol
        config.data_config.lookback_days  = args.days

    pipeline = CryptoTrainingPipeline(config, args)
    pipeline.run(symbol=args.symbol, timeframe=args.tf)


if __name__ == "__main__":
    main()
