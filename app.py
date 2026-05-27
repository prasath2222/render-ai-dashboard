#!/usr/bin/env python3
"""
RENDER AI TRADING DASHBOARD — Professional Advanced UI
======================================================
✓ TradingView-style interactive candlestick charts
✓ Clean separated sections with proper sizing
✓ Professional dark theme matching reference
✓ Proper chart zoom/pan/reset functionality
✓ Multi-timeframe analysis
✓ Advanced metrics layout
✓ Real-time data updates
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RENDER AI Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS - PROFESSIONAL DARK THEME (Like TradingView)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <style>
    :root {
        --primary: #1f2937;
        --secondary: #111827;
        --accent: #00d4ff;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --text: #f3f4f6;
        --text-secondary: #d1d5db;
    }
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0f172a;
        color: var(--text);
    }
    
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    
    .main {
        background-color: #0f172a;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent);
        font-size: 28px;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 13px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid #1f2937;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary);
        border-radius: 0;
        border: none;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid var(--accent);
        color: var(--accent);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text);
        font-weight: 700;
    }
    
    /* Input fields */
    .stSelectbox, .stSlider, .stCheckbox {
        color: var(--text);
    }
    
    /* Sections */
    .section-header {
        color: var(--text);
        font-size: 16px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #1f2937;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    
    .metric-box-title {
        color: var(--text-secondary);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    
    .metric-box-value {
        color: var(--accent);
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-box-change {
        color: var(--text-secondary);
        font-size: 12px;
        margin-top: 5px;
    }
    
    /* Signal boxes */
    .signal-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 2px solid #10b981;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    
    .signal-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 2px solid #ef4444;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.05) 100%);
        border: 2px solid #f59e0b;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    
    .signal-text {
        font-size: 24px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .confidence-text {
        font-size: 12px;
        margin-top: 8px;
    }
    
    /* Info boxes */
    .info-section {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .info-section-title {
        color: var(--accent);
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #374151;
    }
    
    .info-label {
        color: var(--text-secondary);
        font-size: 12px;
    }
    
    .info-value {
        color: var(--text);
        font-size: 14px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR - CONFIGURATION
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ RENDER SETTINGS")
    st.divider()
    
    # Timeframe selection
    timeframe = st.select_slider(
        "📊 Candlestick Timeframe",
        options=["1m", "5m", "15m", "1h", "4h", "1d"],
        value="1h"
    )
    
    # Data period
    period = st.select_slider(
        "📈 Historical Data",
        options=["7d", "30d", "90d", "180d", "365d"],
        value="90d"
    )
    
    st.divider()
    st.markdown("### 📊 INDICATORS")
    
    col1, col2 = st.columns(2)
    with col1:
        show_macd = st.checkbox("MACD", value=True)
        show_rsi = st.checkbox("RSI", value=True)
        show_bb = st.checkbox("BB Bands", value=True)
    with col2:
        show_adx = st.checkbox("ADX", value=True)
        show_vwap = st.checkbox("VWAP", value=True)
        show_obv = st.checkbox("OBV", value=True)
    
    st.divider()
    st.markdown("### 🤖 ML MODEL")
    
    use_ensemble = st.checkbox("🤖 Use Ensemble Model", value=True)
    confidence_threshold = st.slider("📍 Min Confidence", 50, 100, 65)
    
    st.divider()
    st.markdown("### 🔄 AUTO-REFRESH")
    
    auto_refresh = st.checkbox("🔄 Auto-Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.slider("⏱️ Interval (sec)", 30, 300, 60)

# ═══════════════════════════════════════════════════════════════
# LOAD ML MODELS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_models():
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
        return None

models = load_ml_models()

# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_data(period, interval):
    try:
        df = yf.download(
            "RENDER-USD",
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
    try:
        # Trend
        df["EMA_9"] = ta.trend.ema_indicator(df["Close"], window=9)
        df["EMA_21"] = ta.trend.ema_indicator(df["Close"], window=21)
        df["EMA_50"] = ta.trend.ema_indicator(df["Close"], window=50)
        
        # MACD
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
        
        # RSI
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
        df["BB_Upper"] = bb.bollinger_hband()
        df["BB_Lower"] = bb.bollinger_lband()
        df["BB_Middle"] = bb.bollinger_mavg()
        
        # ATR
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        
        # Volume
        df["Volume_SMA"] = ta.trend.sma_indicator(df["Volume"], window=20)
        
        # ADX
        df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
        
        # VWAP
        df["VWAP"] = ta.volume.volume_weighted_average_price(
            df["High"], df["Low"], df["Close"], df["Volume"]
        )
        
        # OBV
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        
        # Support/Resistance
        df["Support"] = df["Low"].rolling(50).min()
        df["Resistance"] = df["High"].rolling(50).max()
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error adding indicators: {e}")
        return df

# ═══════════════════════════════════════════════════════════════
# ML PREDICTION
# ═══════════════════════════════════════════════════════════════
def get_ml_prediction(df):
    if models is None or models["classifier"] is None:
        return None
    
    try:
        feature_cols = models["features"]
        X = df[feature_cols].values[-1:].astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = models["scaler"].transform(X)
        
        model = models["ensemble"] if use_ensemble else models["classifier"]
        signal_proba = model.predict_proba(X_scaled)[0]
        signal = model.predict(X_scaled)[0]
        
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
        return None

# ═══════════════════════════════════════════════════════════════
# TECHNICAL ANALYSIS SCORE
# ═══════════════════════════════════════════════════════════════
def calculate_trend_score(df):
    latest = df.iloc[-1]
    score = 0
    details = []
    
    # EMA
    if latest["EMA_9"] > latest["EMA_21"]:
        score += 15
        details.append("✓ EMA 9 > 21")
    else:
        score -= 15
        details.append("✗ EMA 9 < 21")
    
    if latest["EMA_21"] > latest["EMA_50"]:
        score += 15
        details.append("✓ EMA 21 > 50")
    else:
        score -= 15
        details.append("✗ EMA 21 < 50")
    
    # RSI
    if latest["RSI"] > 70:
        score -= 10
        details.append("⚠️ RSI >70 (Overbought)")
    elif latest["RSI"] > 60:
        score += 10
        details.append("✓ RSI Bullish")
    elif latest["RSI"] < 30:
        score += 15
        details.append("✓ RSI <30 (Oversold)")
    
    # MACD
    if latest["MACD"] > latest["MACD_Signal"]:
        score += 15
        details.append("✓ MACD Bullish")
    else:
        score -= 15
        details.append("✗ MACD Bearish")
    
    # ADX
    if latest["ADX"] > 25:
        score += 5
        details.append("✓ Strong Trend (ADX>25)")
    
    return score, details

# ═══════════════════════════════════════════════════════════════
# TRADINGVIEW-STYLE CANDLESTICK CHART
# ═══════════════════════════════════════════════════════════════
def create_tradingview_chart(df):
    """Create professional TradingView-style candlestick chart."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
            increasing_fillcolor="#10b981",
            decreasing_fillcolor="#ef4444",
        ),
        row=1, col=1
    )
    
    # EMA Lines
    if len(df) > 50:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_9"],
                name="EMA 9", line=dict(color="#00d4ff", width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_21"],
                name="EMA 21", line=dict(color="#3b82f6", width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["EMA_50"],
                name="EMA 50", line=dict(color="#f59e0b", width=1),
                opacity=0.8
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if len(df) > 20:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Upper"],
                name="BB Upper",
                line=dict(color="rgba(255, 170, 0, 0.2)", width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["BB_Lower"],
                name="BB Lower",
                line=dict(color="rgba(255, 170, 0, 0.2)", width=0),
                fill="tonexty",
                fillcolor="rgba(255, 170, 0, 0.1)",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Support/Resistance
    if len(df) > 50:
        fig.add_hline(
            y=df["Support"].iloc[-1],
            line_dash="dash",
            line_color="rgba(16, 185, 129, 0.5)",
            annotation_text="Support",
            row=1, col=1
        )
        fig.add_hline(
            y=df["Resistance"].iloc[-1],
            line_dash="dash",
            line_color="rgba(239, 68, 68, 0.5)",
            annotation_text="Resistance",
            row=1, col=1
        )
    
    # Volume
    colors = ["#10b981" if df["Close"].iloc[i] > df["Open"].iloc[i] else "#ef4444"
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker=dict(color=colors),
            opacity=0.3,
        ),
        row=2, col=1
    )
    
    # Volume SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Volume_SMA"],
            name="Vol SMA",
            line=dict(color="#f59e0b", width=1),
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Layout
    fig.update_layout(
        title="RENDER-USD Chart",
        yaxis_title="Price (USD)",
        xaxis_title="Time",
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f3f4f6", family="Inter"),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    
    return fig

# ═══════════════════════════════════════════════════════════════
# RSI CHART
# ═══════════════════════════════════════════════════════════════
def create_rsi_chart(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"],
        name="RSI (14)", line=dict(color="#00d4ff", width=2),
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.1)"
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", annotation_text="Oversold (30)")
    fig.add_hline(y=50, line_dash="dot", line_color="#6b7280", annotation_text="Neutral (50)")
    
    fig.update_layout(
        title="RSI (14)",
        yaxis_title="RSI",
        xaxis_title="Time",
        template="plotly_dark",
        height=300,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f3f4f6", family="Inter"),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    
    return fig

# ═══════════════════════════════════════════════════════════════
# MACD CHART
# ═══════════════════════════════════════════════════════════════
def create_macd_chart(df):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"],
        name="MACD", line=dict(color="#00d4ff", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"],
        name="Signal", line=dict(color="#ef4444", width=2)
    ))
    
    colors = ["#10b981" if x > 0 else "#ef4444" for x in df["MACD_Hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["MACD_Hist"],
        name="Histogram", marker=dict(color=colors), opacity=0.3
    ))
    
    fig.update_layout(
        title="MACD",
        yaxis_title="MACD",
        xaxis_title="Time",
        template="plotly_dark",
        height=300,
        hovermode="x unified",
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#0f172a",
        font=dict(color="#f3f4f6", family="Inter"),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1f2937")
    
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Header
st.markdown("# 🚀 RENDER AI TRADING DASHBOARD")
st.markdown("**Real-time predictions powered by machine learning**")
st.divider()

# Fetch data
with st.spinner("⏳ Loading data..."):
    df = fetch_data(period, timeframe)
    
    if df is not None and len(df) > 0:
        df = add_indicators(df)
        latest = df.iloc[-1]
        
        current_price = float(latest["Close"])
        prev_price = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        # ═══════════════════════════════════════════════════════════════
        # TOP METRICS SECTION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">📊 MARKET METRICS</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Price</div>
                <div class="metric-box-value">${current_price:.4f}</div>
                <div class="metric-box-change">{price_change:+.4f} ({price_change_pct:+.2f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            change_24h = ((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24] * 100) if len(df) > 24 else 0
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">24H Change</div>
                <div class="metric-box-value">{change_24h:+.2f}%</div>
                <div class="metric-box-change">Last 24 hours</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">RSI (14)</div>
                <div class="metric-box-value">{latest['RSI']:.1f}</div>
                <div class="metric-box-change">{'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">ATR</div>
                <div class="metric-box-value">${latest['ATR']:.4f}</div>
                <div class="metric-box-change">Volatility</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            vol = ((latest["High"] - latest["Low"]) / latest["Close"] * 100)
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Volatility</div>
                <div class="metric-box-value">{vol:.2f}%</div>
                <div class="metric-box-change">Intraday</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-box-title">Volume</div>
                <div class="metric-box-value">{latest['Volume']/1e6:.2f}M</div>
                <div class="metric-box-change">24H Volume</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # SIGNALS SECTION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">🎯 SIGNALS & PREDICTIONS</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ml_pred = get_ml_prediction(df)
            
            if ml_pred and ml_pred["confidence"] >= confidence_threshold:
                signal = ml_pred["signal"]
                confidence = ml_pred["confidence"]
                
                if signal == "BUY":
                    st.markdown(f"""
                    <div class="signal-buy">
                        <div style="font-size: 32px;">🟢</div>
                        <div class="signal-text">BUY</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif signal == "SELL":
                    st.markdown(f"""
                    <div class="signal-sell">
                        <div style="font-size: 32px;">🔴</div>
                        <div class="signal-text">SELL</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="signal-hold">
                        <div style="font-size: 32px;">🟡</div>
                        <div class="signal-text">HOLD</div>
                        <div class="confidence-text">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("**Probability Breakdown:**")
                cols = st.columns(3)
                with cols[0]:
                    st.metric("SELL", f"{ml_pred['probabilities']['sell']:.1f}%", delta=None)
                with cols[1]:
                    st.metric("HOLD", f"{ml_pred['probabilities']['hold']:.1f}%", delta=None)
                with cols[2]:
                    st.metric("BUY", f"{ml_pred['probabilities']['buy']:.1f}%", delta=None)
                
                # Price prediction
                if ml_pred["price_prediction"]:
                    pred_price = ml_pred["price_prediction"]
                    pred_change = ((pred_price - current_price) / current_price) * 100
                    
                    st.markdown(f"""
                    <div class="info-section">
                        <div class="info-section-title">📈 Price Prediction (24H)</div>
                        <div class="info-row">
                            <span class="info-label">Predicted Price:</span>
                            <span class="info-value">${pred_price:.4f}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Expected Change:</span>
                            <span class="info-value">{pred_change:+.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Confidence below threshold. No signal.")
        
        with col2:
            trend_score, trend_details = calculate_trend_score(df)
            
            score_color = "#10b981" if trend_score > 30 else "#ef4444" if trend_score < -30 else "#f59e0b"
            
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
                border: 2px solid {score_color};
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size: 40px; color: {score_color}; font-weight: 700; margin: 10px 0;">{trend_score:+d}</div>
                <div style="color: #d1d5db; font-size: 13px; text-transform: uppercase; font-weight: 600;">Technical Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("**Signal Details:**")
            for detail in trend_details[:6]:
                st.text(detail)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # CHARTS SECTION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">📈 CHARTS & INDICATORS</div>', unsafe_allow_html=True)
        
        chart_tabs = st.tabs(["PRICE", "RSI", "MACD", "DETAILS"])
        
        with chart_tabs[0]:
            st.plotly_chart(create_tradingview_chart(df), use_container_width=True, config={"displayModeBar": True})
        
        with chart_tabs[1]:
            st.plotly_chart(create_rsi_chart(df), use_container_width=True, config={"displayModeBar": True})
        
        with chart_tabs[2]:
            st.plotly_chart(create_macd_chart(df), use_container_width=True, config={"displayModeBar": True})
        
        with chart_tabs[3]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**EMA Analysis**")
                st.metric("EMA 9", f"${latest['EMA_9']:.4f}")
                st.metric("EMA 21", f"${latest['EMA_21']:.4f}")
                st.metric("EMA 50", f"${latest['EMA_50']:.4f}")
            
            with col2:
                st.markdown("**Support/Resistance**")
                st.metric("Support", f"${latest['Support']:.4f}")
                st.metric("Resistance", f"${latest['Resistance']:.4f}")
                st.metric("VWAP", f"${latest['VWAP']:.4f}")
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # TRADING SETUP SECTION
        # ═══════════════════════════════════════════════════════════════
        st.markdown('<div class="section-header">🎯 TRADING SETUP</div>', unsafe_allow_html=True)
        
        entry_price = current_price
        stop_loss = current_price - (latest["ATR"] * 2)
        take_profit_1 = current_price + (latest["ATR"] * 3)
        take_profit_2 = current_price + (latest["ATR"] * 6)
        risk_reward = ((take_profit_1 - entry_price) / (entry_price - stop_loss)) if entry_price > stop_loss else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-section">
                <div class="info-section-title">📍 ENTRY & STOP</div>
                <div class="info-row">
                    <span class="info-label">Entry Price:</span>
                    <span class="info-value">${entry_price:.4f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Stop Loss:</span>
                    <span class="info-value">${stop_loss:.4f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Risk:</span>
                    <span class="info-value">${entry_price - stop_loss:.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-section">
                <div class="info-section-title">🎯 TAKE PROFITS</div>
                <div class="info-row">
                    <span class="info-label">TP1 (3×ATR):</span>
                    <span class="info-value">${take_profit_1:.4f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">TP2 (6×ATR):</span>
                    <span class="info-value">${take_profit_2:.4f}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">R/R Ratio:</span>
                    <span class="info-value">{risk_reward:.2f}x</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-section">
                <div class="info-section-title">💰 POSITION SIZING</div>
                <div class="info-row">
                    <span class="info-label">Account Size:</span>
                    <input type="number" value="10000" style="width: 100px; padding: 5px; background: #111827; color: #f3f4f6; border: 1px solid #374151; border-radius: 4px;">
                </div>
                <div class="info-row">
                    <span class="info-label">Risk %:</span>
                    <input type="number" value="1" min="0.1" max="5" style="width: 100px; padding: 5px; background: #111827; color: #f3f4f6; border: 1px solid #374151; border-radius: 4px;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # ═══════════════════════════════════════════════════════════════
        # AUTO-REFRESH
        # ═══════════════════════════════════════════════════════════════
        if auto_refresh:
            st.info(f"🔄 Auto-refreshing every {refresh_interval} seconds...")
            time.sleep(refresh_interval)
            st.rerun()
    
    else:
        st.error("❌ Failed to fetch data. Please try again.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 12px; padding: 20px;">
⚠️ <b>Disclaimer:</b> For educational purposes only. Not financial advice. Trade at your own risk.
</div>
""", unsafe_allow_html=True)
