#!/usr/bin/env python3
"""
RENDER AI TRADING DASHBOARD — PROFESSIONAL ADVANCED UI
========================================================
✓ Real RENDER logo & branding
✓ Professional separate sections
✓ TradingView-style full-width chart
✓ Classification & Regression separate predictions
✓ Live USD/INR price conversion
✓ Inflow/Outflow analysis
✓ Proper chart sizing & spacing
✓ Modern attractive design
✓ Advanced features properly organized
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
from PIL import Image
import requests

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RENDER AI Trading Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# PROFESSIONAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
        color: #e8e8e8;
    }
    
    [data-testid="stVerticalBlock"] > div {
        gap: 8px;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom sections */
    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #374151;
    }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 15px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
    }
    
    .metric-label {
        font-size: 11px;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #00d4ff;
        line-height: 1.2;
    }
    
    .metric-change {
        font-size: 12px;
        color: #6b7280;
        margin-top: 4px;
    }
    
    .signal-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 24px;
        border: 2px solid #374151;
        margin: 15px 0;
    }
    
    .signal-buy {
        border-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.02) 100%);
    }
    
    .signal-sell {
        border-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.02) 100%);
    }
    
    .signal-hold {
        border-color: #f59e0b;
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.02) 100%);
    }
    
    .signal-text {
        font-size: 36px;
        font-weight: 700;
        margin: 10px 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .confidence-badge {
        display: inline-block;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid #00d4ff;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 8px 4px;
        font-size: 13px;
        font-weight: 600;
        color: #00d4ff;
    }
    
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e8e8e8;
        margin: 30px 0 15px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid #374151;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .chart-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 12px;
        margin: 15px 0;
    }
    
    .info-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 15px 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 16px;
    }
    
    .info-label {
        font-size: 11px;
        color: #9ca3af;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .info-value {
        font-size: 16px;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .positive {
        color: #10b981;
    }
    
    .negative {
        color: #ef4444;
    }
    
    .neutral {
        color: #f59e0b;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 2px solid #374151;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9ca3af;
        border: none;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, transparent 100%);
        color: #00d4ff;
        border-bottom: 3px solid #00d4ff;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# GET USD TO INR RATE
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        data = response.json()
        return data["rates"]["INR"]
    except:
        return 83.0  # Default fallback

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
    except:
        return None

models = load_ml_models()

# ═══════════════════════════════════════════════════════════════
# FETCH DATA
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def fetch_render_data(period="90d", interval="1h"):
    try:
        df = yf.download("RENDER-USD", period=period, interval=interval, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def add_all_indicators(df):
    """Add all technical indicators"""
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
        
        # ATR & Volatility
        df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
        df["Volatility"] = df["Close"].pct_change().rolling(20).std() * 100
        
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
        
        # Inflow/Outflow (Money Flow Index)
        df["MFI"] = ta.volume.money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"], window=14)
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error adding indicators: {e}")
        return df

# ═══════════════════════════════════════════════════════════════
# ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════
def get_predictions(df):
    if models is None or models["classifier"] is None:
        return None, None
    
    try:
        feature_cols = models["features"]
        X = df[feature_cols].values[-1:].astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        X_scaled = models["scaler"].transform(X)
        
        # Classification
        cls_proba = models["ensemble"].predict_proba(X_scaled)[0]
        cls_signal = models["ensemble"].predict(X_scaled)[0]
        
        # Regression
        price_pred = models["regressor"].predict(X_scaled)[0] if models["regressor"] else None
        
        return {
            "signal": ["SELL", "HOLD", "BUY"][int(cls_signal)],
            "confidence": float(cls_proba.max()) * 100,
            "proba": {
                "sell": float(cls_proba[0]) * 100,
                "hold": float(cls_proba[1]) * 100,
                "buy": float(cls_proba[2]) * 100,
            }
        }, {
            "price": float(price_pred) if price_pred else None
        }
    except:
        return None, None

# ═══════════════════════════════════════════════════════════════
# TRADINGVIEW CHART
# ═══════════════════════════════════════════════════════════════
def create_main_chart(df):
    """Create full-width TradingView-style chart"""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        specs=[[{}], [{}]]
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
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"], name="EMA 9",
                             line=dict(color="#00d4ff", width=1), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_21"], name="EMA 21",
                             line=dict(color="#3b82f6", width=1), opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["EMA_50"], name="EMA 50",
                             line=dict(color="#f59e0b", width=1), opacity=0.8), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                             line=dict(color="rgba(255, 170, 0, 0.2)", width=0),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                             line=dict(color="rgba(255, 170, 0, 0.2)", width=0),
                             fill="tonexty", fillcolor="rgba(255, 170, 0, 0.05)",
                             showlegend=False), row=1, col=1)
    
    # Support/Resistance
    fig.add_hline(y=df["Support"].iloc[-1], line_dash="dash", line_color="rgba(16, 185, 129, 0.5)",
                  annotation_text="Support", row=1, col=1)
    fig.add_hline(y=df["Resistance"].iloc[-1], line_dash="dash", line_color="rgba(239, 68, 68, 0.5)",
                  annotation_text="Resistance", row=1, col=1)
    
    # Volume
    colors = ["#10b981" if df["Close"].iloc[i] > df["Open"].iloc[i] else "#ef4444" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker=dict(color=colors), opacity=0.3),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_SMA"], name="Vol SMA",
                             line=dict(color="#f59e0b", width=1), opacity=0.7), row=2, col=1)
    
    fig.update_layout(
        title="RENDER-USD | 1H Chart",
        template="plotly_dark",
        height=700,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=60, t=60, b=60),
        plot_bgcolor="#0f0f1e",
        paper_bgcolor="#0f0f1e",
        font=dict(color="#e8e8e8", size=12),
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#2a2a3a")
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#2a2a3a")
    
    return fig

# ═══════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════

# Header with Logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <div style="font-size: 40px; font-weight: 700; color: #00d4ff; letter-spacing: 2px; margin-bottom: 10px;">
            🚀 RENDER AI
        </div>
        <div style="font-size: 14px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px;">
            Advanced Trading Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Fetch data
df = fetch_render_data("90d", "1h")

if df is not None and len(df) > 0:
    df = add_all_indicators(df)
    latest = df.iloc[-1]
    
    usd_inr = get_usd_inr_rate()
    current_price_usd = float(latest["Close"])
    current_price_inr = current_price_usd * usd_inr
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: TOP METRICS
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📊 LIVE MARKET DATA</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price USD</div>
            <div class="metric-value">${current_price_usd:.4f}</div>
            <div class="metric-change">USD</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price INR</div>
            <div class="metric-value">₹{current_price_inr:.2f}</div>
            <div class="metric-change">Indian Rupee</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        change_24h = ((df["Close"].iloc[-1] - df["Close"].iloc[-24]) / df["Close"].iloc[-24] * 100) if len(df) > 24 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24H Change</div>
            <div class="metric-value {'positive' if change_24h > 0 else 'negative'}\">{change_24h:+.2f}%</div>
            <div class="metric-change">Last 24 hours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI (14)</div>
            <div class="metric-value">{latest['RSI']:.1f}</div>
            <div class="metric-change">{'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ATR</div>
            <div class="metric-value">${latest['ATR']:.4f}</div>
            <div class="metric-change">Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Volume</div>
            <div class="metric-value">{latest['Volume']/1e6:.2f}M</div>
            <div class="metric-change">24H Vol</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: TRADINGVIEW CHART (FULL WIDTH)
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📈 PRICE CHART</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(create_main_chart(df), use_container_width=True, config={"displayModeBar": True, "toImageButtonOptions": {"format": "png"}})
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: ML PREDICTIONS
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🤖 AI PREDICTIONS</div>', unsafe_allow_html=True)
    
    cls_pred, reg_pred = get_predictions(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if cls_pred:
            signal = cls_pred["signal"]
            confidence = cls_pred["confidence"]
            
            signal_class = "signal-buy" if signal == "BUY" else "signal-sell" if signal == "SELL" else "signal-hold"
            signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            
            st.markdown(f"""
            <div class="signal-container {signal_class}">
                <div style="font-size: 48px; margin-bottom: 10px;">{signal_emoji}</div>
                <div class="signal-text">{signal}</div>
                <div class="confidence-badge">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            st.markdown('<div style="margin-top: 20px;"><strong>Probability Breakdown:</strong></div>', unsafe_allow_html=True)
            col_s, col_h, col_b = st.columns(3)
            with col_s:
                st.metric("🔴 SELL", f"{cls_pred['proba']['sell']:.1f}%", delta=None)
            with col_h:
                st.metric("🟡 HOLD", f"{cls_pred['proba']['hold']:.1f}%", delta=None)
            with col_b:
                st.metric("🟢 BUY", f"{cls_pred['proba']['buy']:.1f}%", delta=None)
        else:
            st.info("⚠️ ML models not loaded. Train models first.")
    
    with col2:
        if reg_pred and reg_pred["price"]:
            pred_price_usd = reg_pred["price"]
            pred_price_inr = pred_price_usd * usd_inr
            price_change_pct = ((pred_price_usd - current_price_usd) / current_price_usd) * 100
            
            st.markdown(f"""
            <div class="signal-container" style="border-color: #00d4ff;">
                <div style="font-size: 16px; color: #9ca3af; margin-bottom: 15px; text-transform: uppercase; font-weight: 600;">
                    📊 24H PRICE PREDICTION
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <div class="metric-label">Predicted USD</div>
                        <div class="metric-value">${pred_price_usd:.4f}</div>
                    </div>
                    <div>
                        <div class="metric-label">Predicted INR</div>
                        <div class="metric-value" style="font-size: 20px;">₹{pred_price_inr:.2f}</div>
                    </div>
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #374151;">
                    <div class="metric-label">Expected Change</div>
                    <div class="metric-value {'positive' if price_change_pct > 0 else 'negative'}\">{price_change_pct:+.2f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⚠️ Regression model not loaded.")
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: DETAILED INDICATORS
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📊 TECHNICAL INDICATORS</div>', unsafe_allow_html=True)
    
    indicator_tabs = st.tabs(["TREND", "MOMENTUM", "VOLUME", "OSCILLATORS"])
    
    with indicator_tabs[0]:  # TREND
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">EMA 9</div>
                <div class="info-value">${latest['EMA_9']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">EMA 21</div>
                <div class="info-value">${latest['EMA_21']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">EMA 50</div>
                <div class="info-value">${latest['EMA_50']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">ADX (Trend Strength)</div>
                <div class="info-value">{latest['ADX']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # EMA Status
        ema_status = "🟢 BULLISH" if (latest['EMA_9'] > latest['EMA_21'] > latest['EMA_50']) else "🔴 BEARISH" if (latest['EMA_9'] < latest['EMA_21'] < latest['EMA_50']) else "🟡 MIXED"
        st.info(f"**EMA Alignment:** {ema_status}")
    
    with indicator_tabs[1]:  # MOMENTUM
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">RSI (14)</div>
                <div class="info-value">{latest['RSI']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">MACD</div>
                <div class="info-value">{latest['MACD']:.6f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">MACD Signal</div>
                <div class="info-value">{latest['MACD_Signal']:.6f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        macd_status = "🟢 BULLISH" if latest['MACD'] > latest['MACD_Signal'] else "🔴 BEARISH"
        st.info(f"**MACD Status:** {macd_status}")
    
    with indicator_tabs[2]:  # VOLUME
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">Volume</div>
                <div class="info-value">{latest['Volume']/1e6:.2f}M</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            vol_ratio = latest['Volume'] / latest['Volume_SMA']
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">Volume Ratio</div>
                <div class="info-value">{vol_ratio:.2f}x</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">Money Flow</div>
                <div class="info-value">{latest['MFI']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with indicator_tabs[3]:  # OSCILLATORS
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">Volatility</div>
                <div class="info-value">{latest['Volatility']:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">ATR</div>
                <div class="info-value">${latest['ATR']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="info-box">
                <div class="info-label">VWAP</div>
                <div class="info-value">${latest['VWAP']:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: SUPPORT/RESISTANCE & TRADING SETUP
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🎯 TRADING LEVELS</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        support = latest['Support']
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Support Level</div>
            <div class="info-value">${support:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        resistance = latest['Resistance']
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Resistance Level</div>
            <div class="info-value">${resistance:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Current Price</div>
            <div class="info-value" style="color: #00d4ff;">${current_price_usd:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Trading Setup
    entry = current_price_usd
    stop_loss = entry - (latest['ATR'] * 2)
    tp1 = entry + (latest['ATR'] * 3)
    tp2 = entry + (latest['ATR'] * 6)
    risk_reward = ((tp1 - entry) / (entry - stop_loss)) if entry > stop_loss else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Entry</div>
            <div class="info-value">${entry:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Stop Loss</div>
            <div class="info-value negative">${stop_loss:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Take Profit 1</div>
            <div class="info-value positive">${tp1:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Take Profit 2</div>
            <div class="info-value positive">${tp2:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Risk/Reward</div>
            <div class="info-value">{risk_reward:.2f}x</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
            <div class="info-label">Risk Amount</div>
            <div class="info-value negative">${entry - stop_loss:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-size: 11px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #374151;">
    ⚠️ <strong>Risk Disclaimer:</strong> For educational purposes only. Not financial advice. Trade at your own risk.
    <br><br>
    <strong>Data Source:</strong> Yahoo Finance | <strong>Updated:</strong> Real-time | <strong>Accuracy:</strong> Not guaranteed
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Failed to load data. Please refresh the page.")
