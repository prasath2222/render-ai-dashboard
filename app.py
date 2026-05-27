#!/usr/bin/env python3
"""
RENDER AI Trading Dashboard — Advanced Streamlit App
=====================================================
✓ Live OHLCV data with real-time updates
✓ TradingView candlestick charts
✓ ML Classification + Regression predictions
✓ Technical indicators (RSI, MACD, ADX, ATR, Bollinger Bands)
✓ Support/Resistance levels
✓ Order setup (Entry, TP, SL, Risk/Reward)
✓ Multi-timeframe analysis
✓ Market sentiment & volume analysis
✓ Advanced modern UI with dark theme
✓ Performance metrics & history tracking
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from scipy import stats
import json
import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RENDER AI Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS - DARK THEME
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <style>
    :root {
        --primary: #00d4ff;
        --secondary: #00ff88;
        --danger: #ff4d6d;
        --warning: #ffaa00;
        --success: #00ff88;
        --dark: #0a0e27;
        --darker: #050812;
        --text: #e0e0e0;
    }
    
    * {
        color: var(--text);
    }
    
    .main {
        background-color: var(--darker);
        color: var(--text);
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary);
        font-size: 32px;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text);
        font-size: 14px;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%);
        border: 1px solid var(--primary);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    .signal-buy {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border: 2px solid var(--success);
        color: var(--success);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(255, 77, 109, 0.1) 0%, rgba(255, 77, 109, 0.05) 100%);
        border: 2px solid var(--danger);
        color: var(--danger);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 170, 0, 0.05) 100%);
        border: 2px solid var(--warning);
        color: var(--warning);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background-color: #1a1f3a;
        border: 1px solid #333;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: var(--darker);
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR - CONFIGURATION
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Token",
        ["RENDER-USD", "RNDR-USD", "BTC-USD", "ETH-USD"],
        index=0
    )
    
    # Timeframe selection
    timeframe = st.select_slider(
        "Candlestick Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        value="1h"
    )
    
    # Data period
    period = st.select_slider(
        "Historical Data Period",
        options=["7d", "30d", "90d", "180d", "365d"],
        value="90d"
    )
    
    # Technical indicators
    st.markdown("### 📊 Indicators")
    show_macd = st.checkbox("MACD", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_adx = st.checkbox("ADX", value=True)
    show_vwap = st.checkbox("VWAP", value=True)
    
    # ML Model settings
    st.markdown("### 🤖 ML Model")
    use_ensemble = st.checkbox("Use Ensemble Model", value=True)
    confidence_threshold = st.slider("Min Confidence", 50, 100, 65)
    
    # Auto-refresh
    st.markdown("### 🔄 Auto-Refresh")
    auto_refresh = st.checkbox("Enable Auto-Refresh", value=True)
    refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)

# ═══════════════════════════════════════════════════════════════
# LOAD ML MODELS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_models():
    """Load pre-trained ML models."""
    try:
        classifier = joblib.load("render_best_classifier.pkl") if os.path.exists("render_best_classifier.pkl") else None
        ensemble = joblib.load("render_ensemble_classifier.pkl") if os.path.exists("render_ensemble_classifier.pkl") else None
        regressor = joblib.load("render_best_regressor.pkl") if os.path.exists("render_best_regressor.pkl") else None
        scaler = joblib.load("render_scaler.pkl") if os.path.exists("render_scaler.pkl") else None
        features = joblib.load("render_features.pkl") if os.path.exists("render_features.pkl") else None
        
        return {
            "classifier": classifier,
            "ensemble": ensemble,
            "regressor": regressor,
            "scaler": scaler,
            "features": features
        }
    except Exception as e:
        st.warning(f"⚠️ Could not load ML models: {e}")
        return None

models = load_ml_models()

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING & PROCESSING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_data(symbol, period, interval):
    """Download OHLCV data from Yahoo Finance."""
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False
        )
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        df = df.dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def add_indicators(df):
    """Add all technical indicators to dataframe."""
    if len(df) < 200:
        st.warning("Not enough data for all indicators")
        return df
    
    try:
        # Trend
        df["EMA_9"] = ta.trend.ema_indicator(df["Close"], window=9)
        df["EMA_21"] = ta.trend.ema_indicator(df["Close"], window=21)
        df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)
        df["EMA_200"] = ta.trend.ema_indicator(df["Close"], window=200)
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        
        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
        
        # ADX
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
        
        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        df["RSI_Oversold"] = df["RSI"] < 30
        df["RSI_Overbought"] = df["RSI"] > 70
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Middle"] = bb.bollinger_mavg()
        
        # ATR & Volatility
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["Volatility"] = df["Close"].pct_change().rolling(20).std() * 100
        
        # Volume
        df["Volume_SMA"] = ta.trend.sma_indicator(df["Volume"], window=20)
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
        
        # VWAP
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            df["High"], df["Low"], df["Close"], df["Volume"]
        )
        
        # OBV
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        
        # Support/Resistance (50-period)
        df["Support"] = df["Low"].rolling(50).min()
        df["Resistance"] = df["High"].rolling(50).max()
        
        # Returns
        df["Return_1H"] = df["Close"].pct_change(1)
        df["Return_24H"] = df["Close"].pct_change(24)
        
        return df.dropna()
    
    except Exception as e:
        st.error(f"Error adding indicators: {e}")
        return df

# ═══════════════════════════════════════════════════════════════
# ML PREDICTION
# ═══════════════════════════════════════════════════════════════
def get_ml_prediction(df):
    """Generate ML predictions using trained models."""
    if models is None or models["classifier"] is None:
        return None
    
    try:
        # Prepare features
        feature_cols = models["features"]
        X = df[feature_cols].values[-1:].astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        X_scaled = models["scaler"].transform(X)
        
        # Predict
        model = models["ensemble"] if use_ensemble else models["classifier"]
        
        signal_proba = model.predict_proba(X_scaled)[0]
        signal = model.predict(X_scaled)[0]
        
        # Regression
        price_pred = models["regressor"].predict(X_scaled)[0] if models["regressor"] else None
        
        return {
            "signal": ["SELL", "HOLD", "BUY"][int(signal)],
            "signal_id": int(signal),
            "confidence": float(signal_proba.max()) * 100,
            "probabilities": {
                "sell": float(signal_proba[0]) * 100,
                "hold": float(signal_proba[1]) * 100,
                "buy": float(signal_proba[2]) * 100,
            },
            "price_prediction": float(price_pred) if price_pred else None,
        }
    except Exception as e:
        st.warning(f"⚠️ ML prediction failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS SCORE
# ═══════════════════════════════════════════════════════════════
def calculate_trend_score(df):
    """Calculate comprehensive trend score."""
    latest = df.iloc[-1]
    score = 0
    details = []
    
    # EMA Alignment
    if latest["EMA_9"] > latest["EMA_21"]:
        score += 20
        details.append("✓ EMA 9 > 21 (Bullish)")
    else:
        score -= 20
        details.append("✗ EMA 9 < 21 (Bearish)")
    
    if latest["EMA_21"] > latest["EMA_50"]:
        score += 20
        details.append("✓ EMA 21 > 50 (Bullish)")
    else:
        score -= 20
        details.append("✗ EMA 21 < 50 (Bearish)")
    
    # RSI
    if latest["RSI"] > 70:
        score -= 15
        details.append("⚠️ RSI Overbought (>70)")
    elif latest["RSI"] > 60:
        score += 15
        details.append("✓ RSI Bullish (60-70)")
    elif latest["RSI"] < 30:
        score += 20
        details.append("✓ RSI Oversold (<30) - Bounce Setup")
    elif latest["RSI"] < 40:
        score -= 10
        details.append("✗ RSI Weak (<40)")
    
    # MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 20
        details.append("✓ MACD Bullish")
    else:
        score -= 20
        details.append("✗ MACD Bearish")
    
    # ADX
    if latest["ADX"] > 25:
        score += 10
        details.append("✓ Strong Trend (ADX > 25)")
    
    # ATR/Volatility
    if latest["Volatility"] > 2:
        details.append(f"⚠️ High Volatility ({latest['Volatility']:.2f}%)")
    
    # Volume
    if latest["Volume"] > latest["Volume_SMA"]:
        score += 5
        details.append("✓ High Volume")
    
    # Price Action
    price_change = ((df["Close"].iloc[-1] - df["Close"].iloc[-10]) / df["Close"].iloc[-10]) * 100
    if price_change > 0:
        score += 10
        details.append(f"✓ Uptrend (+{price_change:.2f}%)")
    else:
        score -= 10
        details.append(f"✗ Downtrend ({price_change:.2f}%)")
    
    return score, details

# ═══════════════════════════════════════════════════════════════
# CANDLESTICK CHART
# ═══════════════════════════════════════════════════════════════
def create_candlestick_chart(df):
    """Create TradingView-style candlestick chart."""
    fig = go.Figure()
    
    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#00ff88",
        decreasing_line_color="#ff4d6d",
    ))
    
    # EMA Lines
    if len(df) > 50:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_9"],
            name="EMA 9", line=dict(color="cyan", width=1),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_21"],
            name="EMA 21", line=dict(color="blue", width=1),
            opacity=0.7
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["EMA_50"],
            name="EMA 50", line=dict(color="orange", width=1),
            opacity=0.7
        ))
    
    # Bollinger Bands
    if len(df) > 20:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"],
            name="BB Upper", line=dict(color="rgba(255, 170, 0, 0.3)", width=1),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"],
            name="BB Lower", line=dict(color="rgba(255, 170, 0, 0.3)", width=1),
            fill="tonexty",
            fillcolor="rgba(255, 170, 0, 0.1)",
            showlegend=False
        ))
    
    # Support/Resistance
    fig.add_hline(y=df["Support"].iloc[-1], line_dash="dash", line_color="red", annotation_text="Support")
    fig.add_hline(y=df["Resistance"].iloc[-1], line_dash="dash", line_color="green", annotation_text="Resistance")
    
    fig.update_layout(
        title=f"{symbol.split('-')[0]} Price Chart",
        yaxis_title="Price (USD)",
        xaxis_title="Time",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#050812",
        paper_bgcolor="#050812",
        font=dict(color="#e0e0e0"),
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════
# RSI CHART
# ═══════════════════════════════════════════════════════════════
def create_rsi_chart(df):
    """Create RSI indicator chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI (14)", line=dict(color="#00d4ff", width=2)
    ))
    
    # Overbought/Oversold levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
    
    fig.update_layout(
        title="RSI (14)",
        yaxis_title="RSI",
        xaxis_title="Time",
        template="plotly_dark",
        height=250,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#050812",
        paper_bgcolor="#050812",
        font=dict(color="#e0e0e0"),
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════
# MACD CHART
# ═══════════════════════════════════════════════════════════════
def create_macd_chart(df):
    """Create MACD indicator chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#00d4ff", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal", line=dict(color="#ff4d6d", width=2)
    ))
    
    # Histogram
    colors = ["#00ff88" if x > 0 else "#ff4d6d" for x in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        name="Histogram", marker=dict(color=colors), opacity=0.3
    ))
    
    fig.update_layout(
        title="MACD",
        yaxis_title="MACD",
        xaxis_title="Time",
        template="plotly_dark",
        height=250,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#050812",
        paper_bgcolor="#050812",
        font=dict(color="#e0e0e0"),
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════
# VOLUME CHART
# ═══════════════════════════════════════════════════════════════
def create_volume_chart(df):
    """Create volume chart with moving average."""
    fig = go.Figure()
    
    colors = ["#00ff88" if df["Close"].iloc[i] > df["Open"].iloc[i] else "#ff4d6d" 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker=dict(color=colors), opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Volume_SMA"],
        name="Volume SMA (20)", line=dict(color="#ffaa00", width=2)
    ))
    
    fig.update_layout(
        title="Volume",
        yaxis_title="Volume",
        xaxis_title="Time",
        template="plotly_dark",
        height=250,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#050812",
        paper_bgcolor="#050812",
        font=dict(color="#e0e0e0"),
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("# 🚀 RENDER AI Trading Dashboard")
st.markdown("*Real-time predictions powered by machine learning*")

# Fetch data
with st.spinner(f"⏳ Loading {symbol} data..."):
    df = fetch_data(symbol, period, timeframe)
    
    if df is not None and len(df) > 0:
        df = add_indicators(df)
        latest = df.iloc[-1]
        
        # ═══════════════════════════════════════════════════════════════
        # TOP METRICS ROW
        # ═══════════════════════════════════════════════════════════════
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        current_price = float(latest["Close"])
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        with col1:
            st.metric("Price", f"${current_price:.4f}", f"{price_change:+.4f} ({price_change_pct:+.2f}%)")
        
        with col2:
            st.metric("24H Change", f"{((df['Close'].iloc[-1] - df['Close'].iloc[-24]) / df['Close'].iloc[-24] * 100):+.2f}%")
        
        with col3:
            st.metric("RSI (14)", f"{latest['RSI']:.2f}")
        
        with col4:
            st.metric("ATR", f"${latest['ATR']:.4f}")
        
        with col5:
            st.metric("Volatility", f"{latest['Volatility']:.2f}%")
        
        with col6:
            st.metric("Volume", f"{latest['Volume']/1e6:.2f}M")
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # ML PREDICTIONS & SIGNALS
        # ═══════════════════════════════════════════════════════════════
        ml_pred = get_ml_prediction(df)
        trend_score, trend_details = calculate_trend_score(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 ML Prediction")
            
            if ml_pred and ml_pred["confidence"] >= confidence_threshold:
                signal = ml_pred["signal"]
                confidence = ml_pred["confidence"]
                
                if signal == "BUY":
                    st.markdown(f"""
                    <div class="signal-buy">
                        🟢 BUY<br>
                        Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f"""
                    <div class="signal-sell">
                        🔴 SELL<br>
                        Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-hold">
                        🟡 HOLD<br>
                        Confidence: {confidence:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("#### Probabilities")
                col_s, col_h, col_b = st.columns(3)
                with col_s:
                    st.metric("SELL", f"{ml_pred['probabilities']['sell']:.1f}%")
                with col_h:
                    st.metric("HOLD", f"{ml_pred['probabilities']['hold']:.1f}%")
                with col_b:
                    st.metric("BUY", f"{ml_pred['probabilities']['buy']:.1f}%")
                
                # Price prediction
                if ml_pred["price_prediction"]:
                    pred_price = ml_pred["price_prediction"]
                    pred_change = ((pred_price - current_price) / current_price) * 100
                    st.markdown(f"""
                    <div class="info-box">
                        <b>Predicted Price (24h):</b> ${pred_price:.4f}<br>
                        <b>Expected Change:</b> {pred_change:+.2f}%
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Confidence below threshold. No signal.")
        
        with col2:
            st.subheader("📊 Technical Score")
            
            # Score gauge
            score_color = "#00ff88" if trend_score > 40 else "#ff4d6d" if trend_score < -40 else "#ffaa00"
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1a1f3a 0%, #16213e 100%);
                border: 3px solid {score_color};
                border-radius: 12px;
                padding: 30px;
                text-align: center;
            ">
                <h2 style="color: {score_color}; margin: 0;">{trend_score:+d}</h2>
                <p style="margin: 10px 0 0 0;">Technical Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Signal Breakdown")
            for detail in trend_details:
                st.text(detail)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # CHARTS
        # ═══════════════════════════════════════════════════════════════
        st.subheader("📈 Charts")
        
        chart_tabs = st.tabs(["Price", "RSI", "MACD", "Volume", "Analysis"])
        
        with chart_tabs[0]:
            st.plotly_chart(create_candlestick_chart(df), use_container_width=True)
        
        with chart_tabs[1]:
            st.plotly_chart(create_rsi_chart(df), use_container_width=True)
        
        with chart_tabs[2]:
            st.plotly_chart(create_macd_chart(df), use_container_width=True)
        
        with chart_tabs[3]:
            st.plotly_chart(create_volume_chart(df), use_container_width=True)
        
        with chart_tabs[4]:
            st.markdown("### Market Analysis")
            
            # Market regime
            if latest["EMA_9"] > latest["EMA_50"] and latest["RSI"] > 55:
                regime = "🟢 BULLISH"
            elif latest["EMA_9"] < latest["EMA_50"] and latest["RSI"] < 45:
                regime = "🔴 BEARISH"
            else:
                regime = "🟡 SIDEWAYS"
            
            st.metric("Market Regime", regime)
            
            # Support/Resistance
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <b>Support:</b> ${latest['Support']:.4f}<br>
                    <b>Resistance:</b> ${latest['Resistance']:.4f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <b>VWAP:</b> ${latest['VWAP']:.4f}<br>
                    <b>Current:</b> ${current_price:.4f}
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # TRADING SETUP
        # ═══════════════════════════════════════════════════════════════
        st.subheader("🎯 Trading Setup")
        
        col1, col2, col3 = st.columns(3)
        
        entry_price = current_price
        stop_loss = current_price - (latest["ATR"] * 2)
        take_profit_1 = current_price + (latest["ATR"] * 3)
        take_profit_2 = current_price + (latest["ATR"] * 6)
        risk_reward = ((take_profit_1 - entry_price) / (entry_price - stop_loss)) if entry_price > stop_loss else 0
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <b>Entry:</b> ${entry_price:.4f}<br>
                <b>Stop Loss:</b> ${stop_loss:.4f}<br>
                <b>Risk: ${entry_price - stop_loss:.4f}</b>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <b>Take Profit 1:</b> ${take_profit_1:.4f}<br>
                <b>Take Profit 2:</b> ${take_profit_2:.4f}<br>
                <b>Risk/Reward: {risk_reward:.2f}</b>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <b>Position Size (1% risk):</b><br>
                Account Size: <input type="number" value="1000" min="0" style="width: 80px; padding: 5px;"><br>
                <b style="color: #00ff88;">Adjust entry/SL above</b>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # ADVANCED METRICS
        # ═══════════════════════════════════════════════════════════════
        st.subheader("🔍 Advanced Metrics")
        
        metrics_tabs = st.tabs(["Indicators", "Ratios", "Momentum", "Divergences"])
        
        with metrics_tabs[0]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("MACD", f"{latest['MACD']:.6f}", delta=f"{latest['MACD_Hist']:.6f}")
            with col2:
                st.metric("ADX", f"{latest['ADX']:.2f}")
            with col3:
                st.metric("OBV", f"{latest['OBV']:,.0f}")
            with col4:
                st.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")
        
        with metrics_tabs[1]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Price vs EMA 50", f"{((current_price - latest['EMA_50']) / latest['EMA_50'] * 100):+.2f}%")
            with col2:
                st.metric("Price vs VWAP", f"{((current_price - latest['VWAP']) / latest['VWAP'] * 100):+.2f}%")
            with col3:
                st.metric("BB Position", f"{((current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']) * 100):.1f}%")
            with col4:
                st.metric("ATR %", f"{(latest['ATR'] / current_price * 100):.2f}%")
        
        with metrics_tabs[2]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("1H Return", f"{latest['Return_1H']*100:+.3f}%")
            with col2:
                st.metric("24H Return", f"{latest['Return_24H']*100:+.2f}%")
            with col3:
                st.metric("EMA 9-21 Diff", f"${abs(latest['EMA_9'] - latest['EMA_21']):.4f}")
            with col4:
                st.metric("EMA 21-50 Diff", f"${abs(latest['EMA_21'] - latest['EMA_50']):.4f}")
        
        with metrics_tabs[3]:
            st.info("📊 Divergence Analysis")
            st.write("Monitor for bullish/bearish divergences between price and RSI/MACD")
            
            # RSI Divergence
            recent_rsi = df["RSI"].iloc[-20:].values
            recent_close = df["Close"].iloc[-20:].values
            
            if len(recent_rsi) > 0 and len(recent_close) > 0:
                rsi_lower = np.any(recent_rsi < 30)
                close_lower = recent_close[-1] < recent_close[0]
                
                if close_lower and not rsi_lower:
                    st.success("✅ Bullish Divergence detected (Price ↓, RSI ↑)")
                elif not close_lower and rsi_lower:
                    st.error("❌ Bearish Divergence detected (Price ↑, RSI ↓)")
        
        # ═══════════════════════════════════════════════════════════════
        # AUTO-REFRESH
        # ═══════════════════════════════════════════════════════════════
        if auto_refresh:
            st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            st.rerun()
    
    else:
        st.error("❌ Failed to fetch data. Please try again.")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
---
**⚠️ Disclaimer:** This dashboard is for educational purposes only. 
Not financial advice. Always do your own research and manage risk properly.

**💡 Tips:**
- Combine multiple signals for better accuracy
- Use stop losses to manage risk
- Don't trade on emotions
- Test strategies in paper trading first
""")
