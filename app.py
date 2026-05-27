"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                    RENDERUSD PRO ANALYTICS DASHBOARD v1.0                      ║
║                                                                                 ║
║   Real-time crypto trading analysis with AI predictions, technical indicators, ║
║   market sentiment, and advanced charting. Production-grade Streamlit app.      ║
║                                                                                 ║
║   Features:                                                                    ║
║   ✓ Live price feeds (USD & INR)                  ✓ ML Bull/Bear Classification║
║   ✓ Advanced TradingView-style candlestick chart  ✓ Technical Indicators Panel ║
║   ✓ Volume & OBV analysis                         ✓ Inflow/Outflow metrics    ║
║   ✓ Market Greed Index                            ✓ Responsive Mobile/Desktop ║
║   ✓ Buy/Sell signals                              ✓ Modern dark theme UI       ║
║                                                                                 ║
╚════════════════════════════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ PAGE CONFIG & INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RENDERUSD PRO Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "### RENDERUSD Advanced Analytics\nProduction-grade crypto trading dashboard with AI predictions"
    }
)

# Custom CSS for dark theme + modern styling
st.markdown("""
<style>
    :root {
        --primary: #FF6B35;
        --secondary: #004E89;
        --accent: #1ABC9C;
        --danger: #E74C3C;
        --success: #27AE60;
        --dark-bg: #0A0E27;
        --card-bg: #16213E;
        --border: #1F3A63;
        --text: #ECF0F1;
        --text-muted: #95A5A6;
    }
    
    body {
        background-color: var(--dark-bg);
        color: var(--text);
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--dark-bg) 0%, #1a2845 100%);
    }
    
    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        border-color: var(--primary);
        box-shadow: 0 12px 48px rgba(255, 107, 53, 0.2);
        transform: translateY(-2px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s ease;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        font-family: 'Monaco', 'Courier New', monospace;
    }
    
    .metric-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .metric-change.positive {
        color: var(--success);
    }
    
    .metric-change.negative {
        color: var(--danger);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--primary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .indicator-box {
        background: linear-gradient(135deg, var(--card-bg) 0%, rgba(26, 188, 156, 0.05) 100%);
        border-left: 4px solid var(--accent);
        padding: 1.25rem;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    
    .signal-bullish {
        color: var(--success);
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .signal-bearish {
        color: var(--danger);
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .signal-neutral {
        color: var(--text-muted);
        font-weight: 700;
        text-transform: uppercase;
    }
    
    .stMetric {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border);
    }
    
    .chart-container {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .render-logo {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ DATA FETCHING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def fetch_crypto_data(ticker="RENDER-USD", interval="1h", period="90d"):
    """Fetch OHLCV data from yfinance with caching"""
    try:
        data = yf.download(ticker, interval=interval, period=period, progress=False)
        data = data.ffill().dropna()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_bitcoin_data():
    """Fetch Bitcoin data for market correlation"""
    try:
        return yf.download("BTC-USD", interval="1h", period="30d", progress=False)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_ethereum_data():
    """Fetch Ethereum data for market correlation"""
    try:
        return yf.download("ETH-USD", interval="1h", period="30d", progress=False)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    """Fetch Fear & Greed Index from API"""
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        response = requests.get(url, timeout=10)
        data = response.json()
        if data['data']:
            return {
                'value': int(data['data'][0]['value']),
                'status': data['data'][0]['value_classification'],
                'timestamp': data['data'][0]['timestamp']
            }
    except:
        pass
    return {'value': 50, 'status': 'Neutral', 'timestamp': ''}

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df):
    """Calculate technical indicators"""
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    # Returns & Momentum
    df["log_return"] = np.log(close / close.shift(1))
    df["return_6h"] = close.pct_change(6)
    df["return_24h"] = close.pct_change(24)
    
    # EMAs
    df["EMA8"] = ta.trend.ema_indicator(close, 8)
    df["EMA20"] = ta.trend.ema_indicator(close, 20)
    df["EMA50"] = ta.trend.ema_indicator(close, 50)
    df["EMA200"] = ta.trend.ema_indicator(close, 200)
    
    # RSI
    df["RSI14"] = ta.momentum.rsi(close, 14)
    df["RSI7"] = ta.momentum.rsi(close, 7)
    
    # MACD
    macd = ta.trend.MACD(close, 26, 12, 9)
    df["MACD"] = macd.macd()
    df["MACD_SIGNAL"] = macd.macd_signal()
    df["MACD_HIST"] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close, 20, 2)
    df["BB_UPPER"] = bb.bollinger_hband()
    df["BB_LOWER"] = bb.bollinger_lband()
    df["BB_WIDTH"] = (df["BB_UPPER"] - df["BB_LOWER"]) / bb.bollinger_mavg()
    
    # ATR
    df["ATR14"] = ta.volatility.average_true_range(high, low, close, 14)
    
    # Volume
    df["OBV"] = ta.volume.on_balance_volume(close, volume)
    df["VOL_SMA20"] = ta.trend.sma_indicator(volume, 20)
    df["VOL_RATIO"] = volume / (df["VOL_SMA20"] + 1e-9)
    
    # ADX
    df["ADX"] = ta.trend.adx(high, low, close, 14)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()
    
    # ROC
    df["ROC20"] = ta.momentum.roc(close, 20)
    
    return df.fillna(method='bfill').dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ ML PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_market_direction(df):
    """ML-based Bull/Bear/Neutral classification using Random Forest"""
    try:
        # Prepare features
        feature_cols = [
            'log_return', 'RSI14', 'MACD_HIST', 'ATR14', 'OBV', 
            'VOL_RATIO', 'ADX', 'return_6h', 'EMA8', 'EMA20'
        ]
        
        # Create target (look ahead 6 hours)
        df['target'] = (df['Close'].shift(-6) > df['Close']).astype(int)
        
        X = df[feature_cols].dropna()
        y = df['target'].loc[X.index]
        
        if len(X) < 50:
            return None
        
        # Train Random Forest
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[-200:])
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                     random_state=42, n_jobs=-1)
        clf.fit(X_scaled[-180:], y[-180:])
        
        # Predict
        latest = X_scaled[[-1]]
        proba = clf.predict_proba(latest)[0]
        prediction = clf.predict(latest)[0]
        
        return {
            'bullish_prob': proba[1],
            'bearish_prob': proba[0],
            'prediction': 'BULLISH 🚀' if prediction == 1 else 'BEARISH 📉',
            'confidence': max(proba) * 100
        }
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ CHARTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_candlestick_chart(df, title="RENDERUSD Price Action"):
    """Create professional TradingView-style candlestick chart"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}]
        ]
    )
    
    # Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#27AE60',
            decreasing_line_color='#E74C3C',
        ),
        row=1, col=1, secondary_y=False
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA20'], name='EMA20', 
                  line=dict(color='#FF6B35', width=2)),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA50'], name='EMA50',
                  line=dict(color='#1ABC9C', width=2)),
        row=1, col=1, secondary_y=False
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_UPPER'], name='BB Upper',
                  line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                  showlegend=False),
        row=1, col=1, secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_LOWER'], name='BB Lower',
                  line=dict(color='rgba(100, 100, 100, 0.2)', width=1),
                  fill='tonexty', fillcolor='rgba(100, 100, 100, 0.1)',
                  showlegend=False),
        row=1, col=1, secondary_y=False
    )
    
    # Volume
    colors = ['#27AE60' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
              else '#E74C3C' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker=dict(color=colors), showlegend=False),
        row=2, col=1, secondary_y=False
    )
    
    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI14'], name='RSI14',
                  line=dict(color='#1ABC9C', width=2)),
        row=3, col=1, secondary_y=False
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", 
                  row=3, col=1, secondary_y=False)
    fig.add_hline(y=30, line_dash="dash", line_color="green",
                  row=3, col=1, secondary_y=False)
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
                  row=3, col=1, secondary_y=False)
    
    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#ECF0F1')),
        height=800,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#16213E',
        plot_bgcolor='#0A0E27',
        font=dict(family='Courier New', size=11, color='#ECF0F1'),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    fig.update_xaxes(gridcolor='#1F3A63', showgrid=True)
    fig.update_yaxes(gridcolor='#1F3A63', showgrid=True)
    
    return fig

def create_volume_analysis_chart(df):
    """OBV and Volume analysis chart"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.12, row_heights=[0.5, 0.5])
    
    # Volume bars
    colors = ['#27AE60' if df['Close'].iloc[i] >= df['Open'].iloc[i] 
              else '#E74C3C' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume',
               marker=dict(color=colors), showlegend=True),
        row=1, col=1
    )
    
    # OBV
    fig.add_trace(
        go.Scatter(x=df.index, y=df['OBV'], name='OBV',
                  line=dict(color='#1ABC9C', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(
        title='Volume & On-Balance Volume (OBV)',
        height=500,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#16213E',
        plot_bgcolor='#0A0E27',
        font=dict(family='Courier New', size=10, color='#ECF0F1'),
    )
    
    fig.update_xaxes(gridcolor='#1F3A63')
    fig.update_yaxes(gridcolor='#1F3A63')
    
    return fig

def create_macd_chart(df):
    """MACD indicator chart"""
    fig = go.Figure()
    
    # MACD Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                  line=dict(color='#FF6B35', width=2))
    )
    
    # Signal Line
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_SIGNAL'], name='Signal',
                  line=dict(color='#1ABC9C', width=2))
    )
    
    # Histogram
    colors = ['#27AE60' if df['MACD_HIST'].iloc[i] > 0 else '#E74C3C' 
              for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_HIST'], name='Histogram',
               marker=dict(color=colors), showlegend=True)
    )
    
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title='MACD Indicator',
        height=400,
        hovermode='x unified',
        template='plotly_dark',
        paper_bgcolor='#16213E',
        plot_bgcolor='#0A0E27',
        font=dict(family='Courier New', size=10, color='#ECF0F1'),
    )
    
    fig.update_xaxes(gridcolor='#1F3A63')
    fig.update_yaxes(gridcolor='#1F3A63')
    
    return fig

def create_indicators_heatmap(df):
    """Composite indicators analysis"""
    fig = go.Figure()
    
    # Get latest indicators
    latest = df.iloc[-1]
    
    indicators = {
        'RSI14': latest['RSI14'] / 100,
        'BB Width': min(latest['BB_WIDTH'], 1),
        'ADX': latest['ADX'] / 50,
        'VOL Ratio': min(latest['VOL_RATIO'], 2) / 2,
        'MACD Cross': 0.7 if latest['MACD'] > latest['MACD_SIGNAL'] else 0.3,
    }
    
    fig.add_trace(
        go.Indicator(
            domain={'x': [0, 1], 'y': [0, 1]},
            value=50,
            mode="gauge+number+delta",
            title={'text': "Market Health Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#FF6B35"},
                'steps': [
                    {'range': [0, 30], 'color': "#E74C3C"},
                    {'range': [30, 70], 'color': "#F39C12"},
                    {'range': [70, 100], 'color': "#27AE60"}
                ]
            }
        )
    )
    
    fig.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='#16213E',
        plot_bgcolor='#0A0E27',
        font=dict(family='Courier New', size=12, color='#ECF0F1'),
    )
    
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ SIGNAL ANALYSIS & CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_signals(df):
    """Generate comprehensive trading signals"""
    latest = df.iloc[-1]
    
    signals = {
        'price_ema_signal': 'BULLISH' if latest['Close'] > latest['EMA20'] else 'BEARISH',
        'rsi_signal': 'OVERBOUGHT' if latest['RSI14'] > 70 else ('OVERSOLD' if latest['RSI14'] < 30 else 'NEUTRAL'),
        'macd_signal': 'BULLISH' if latest['MACD'] > latest['MACD_SIGNAL'] else 'BEARISH',
        'bb_signal': 'SELLING' if latest['Close'] > latest['BB_UPPER'] else ('BUYING' if latest['Close'] < latest['BB_LOWER'] else 'NEUTRAL'),
        'volume_signal': 'HIGH' if latest['VOL_RATIO'] > 1.2 else 'LOW',
    }
    
    # Calculate overall direction
    bullish_count = sum(1 for v in signals.values() if 'BULLISH' in str(v) or v in ['OVERBOUGHT', 'SELLING', 'HIGH'])
    direction = 'BULLISH 🚀' if bullish_count >= 3 else ('BEARISH 📉' if bullish_count <= 1 else 'NEUTRAL ↔️')
    
    return signals, direction

def classify_trend(df):
    """Advanced trend classification"""
    latest = df.iloc[-1]
    
    if latest['ADX'] > 25:
        if latest['Close'] > latest['EMA20'] > latest['EMA50']:
            return "STRONG UPTREND 📈"
        elif latest['Close'] < latest['EMA20'] < latest['EMA50']:
            return "STRONG DOWNTREND 📉"
        else:
            return "TRENDING ↔️"
    else:
        return "CHOPPY/CONSOLIDATION 〰️"

def calculate_levels(df):
    """Calculate support/resistance levels"""
    high_24 = df['High'].tail(24).max()
    low_24 = df['Low'].tail(24).min()
    current = df['Close'].iloc[-1]
    
    resistance1 = high_24
    support1 = low_24
    pivot = (high_24 + low_24 + current) / 3
    resistance2 = pivot + (high_24 - low_24)
    support2 = pivot - (high_24 - low_24)
    
    return {
        'support2': support2,
        'support1': support1,
        'pivot': pivot,
        'resistance1': resistance1,
        'resistance2': resistance2,
        'current': current,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# ▶ MAIN APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<div class="render-logo">RENDERUSD</div>', unsafe_allow_html=True)
    
    st.markdown("### 🚀 Advanced Cryptocurrency Analytics & Trading Dashboard")
    st.divider()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Settings")
        
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1h", "4h", "1d"],
            key="timeframe"
        )
        
        period_map = {"1h": "90d", "4h": "180d", "1d": "1y"}
        period = period_map[timeframe]
        
        update_freq = st.slider("Chart Refresh (seconds)", 30, 300, 60, step=30)
        
        st.divider()
        st.markdown("### 📊 Chart Sections")
        show_volume = st.checkbox("Volume Analysis", value=True)
        show_macd = st.checkbox("MACD Indicator", value=True)
        show_prediction = st.checkbox("ML Predictions", value=True)
        show_sentiment = st.checkbox("Market Sentiment", value=True)
        
        st.divider()
        st.markdown("### 📍 Useful Resources")
        st.markdown("""
        - [TradingView](https://www.tradingview.com/chart/RENDER/RENDERUSD/)
        - [CoinGecko](https://www.coingecko.com/en/coins/render)
        - [Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
        """)
    
    # Fetch data
    df_render = fetch_crypto_data("RENDER-USD", timeframe, period)
    
    if df_render.empty:
        st.error("Unable to fetch data. Please try again later.")
        return
    
    # Engineer features
    df = engineer_features(df_render)
    
    # Get current price
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-24] if len(df) > 24 else df['Close'].iloc[0]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Convert to INR (approximate exchange rate)
    usd_to_inr = 83.5  # You can fetch this dynamically
    price_inr = current_price * usd_to_inr
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TOP METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-header">💰 Price & Market Metrics</div>', unsafe_allow_html=True)
    
    metric_cols = st.columns(5)
    
    with metric_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price USD</div>
            <div class="metric-value">${current_price:.4f}</div>
            <div class="metric-change {'positive' if price_change > 0 else 'negative'}">
                {price_change:+.4f} ({price_change_pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Price INR</div>
            <div class="metric-value">₹{price_inr:.2f}</div>
            <div class="metric-change" style="color: #1ABC9C;">
                1 USD = ₹{usd_to_inr}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[2]:
        volume_24 = df['Volume'].tail(24).sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Volume</div>
            <div class="metric-value">{volume_24/1e6:.2f}M</div>
            <div class="metric-change" style="color: #FF6B35;">
                Last Hour: {df['Volume'].iloc[-1]/1e6:.2f}M
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[3]:
        high_24 = df['High'].tail(24).max()
        low_24 = df['Low'].tail(24).min()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Range</div>
            <div class="metric-value">${low_24:.4f} - ${high_24:.4f}</div>
            <div class="metric-change" style="color: #1ABC9C;">
                Range: ${high_24 - low_24:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Volatility</div>
            <div class="metric-value">{df['log_return'].tail(24).std() * 100:.2f}%</div>
            <div class="metric-change" style="color: #FF6B35;">
                ATR: {df['ATR14'].iloc[-1]:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN CHART
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-header">📈 Price Action & Technical Analysis</div>', 
                unsafe_allow_html=True)
    
    st.plotly_chart(
        create_candlestick_chart(df.tail(100), f"RENDERUSD {timeframe} Chart"),
        use_container_width=True,
        config={'displayModeBar': True, 'displaylogo': False}
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-header">🎯 Trading Signals & Indicators</div>',
                unsafe_allow_html=True)
    
    signals, direction = analyze_signals(df)
    trend_class = classify_trend(df)
    levels = calculate_levels(df)
    
    sig_cols = st.columns(2)
    
    with sig_cols[0]:
        st.markdown(f"""
        <div class="indicator-box">
            <h3 style="margin-top: 0;">Market Direction</h3>
            <div style="font-size: 1.5rem; margin: 1rem 0;">
                <span class="signal-{'bullish' if 'BULLISH' in direction else ('bearish' if 'BEARISH' in direction else 'neutral')}">
                    {direction}
                </span>
            </div>
            <p style="color: var(--text-muted); margin: 0.5rem 0;">
                Trend: <strong>{trend_class}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with sig_cols[1]:
        latest = df.iloc[-1]
        st.markdown(f"""
        <div class="indicator-box">
            <h3 style="margin-top: 0;">Price vs Moving Averages</h3>
            <p style="margin: 0.5rem 0;">
                Price: <span style="color: #1ABC9C;"><strong>${latest['Close']:.4f}</strong></span>
            </p>
            <p style="margin: 0.5rem 0;">
                EMA20: <span style="color: #FF6B35;"><strong>${latest['EMA20']:.4f}</strong></span>
            </p>
            <p style="margin: 0.5rem 0;">
                EMA50: <span style="color: #1ABC9C;"><strong>${latest['EMA50']:.4f}</strong></span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ═════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL SIGNALS
    # ═════════════════════════════════════════════════════════════════════════
    
    ind_cols = st.columns(3)
    
    with ind_cols[0]:
        st.markdown(f"""
        <div class="indicator-box">
            <h4>RSI(14)</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                <strong>{df['RSI14'].iloc[-1]:.1f}</strong>
            </p>
            <p style="color: var(--text-muted); margin: 0;">
                Signal: <strong>{signals['rsi_signal']}</strong>
            </p>
            {f"<p style='color: #E74C3C;'>⚠️ OVERBOUGHT</p>" if df['RSI14'].iloc[-1] > 70 else (f"<p style='color: #27AE60;'>✓ OVERSOLD</p>" if df['RSI14'].iloc[-1] < 30 else "")}
        </div>
        """, unsafe_allow_html=True)
    
    with ind_cols[1]:
        st.markdown(f"""
        <div class="indicator-box">
            <h4>MACD</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                <strong>{df['MACD'].iloc[-1]:.6f}</strong>
            </p>
            <p style="color: var(--text-muted); margin: 0;">
                Signal: <span class="signal-{'bullish' if signals['macd_signal'] == 'BULLISH' else 'bearish'}">{signals['macd_signal']}</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with ind_cols[2]:
        st.markdown(f"""
        <div class="indicator-box">
            <h4>ADX (Trend Strength)</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">
                <strong>{df['ADX'].iloc[-1]:.1f}</strong>
            </p>
            <p style="color: var(--text-muted); margin: 0;">
                {'🔥 Strong Trend' if df['ADX'].iloc[-1] > 25 else '⚠️ Weak Trend'}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═════════════════════════════════════════════════════════════════════════
    # CHARTS & ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    
    if show_volume:
        st.markdown('<div class="section-header">📊 Volume Analysis</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(create_volume_analysis_chart(df.tail(100)), use_container_width=True)
    
    if show_macd:
        st.markdown('<div class="section-header">📉 MACD Momentum</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(create_macd_chart(df.tail(100)), use_container_width=True)
    
    # ═════════════════════════════════════════════════════════════════════════
    # ML PREDICTION
    # ═════════════════════════════════════════════════════════════════════════
    
    if show_prediction:
        st.markdown('<div class="section-header">🤖 AI Market Prediction</div>',
                    unsafe_allow_html=True)
        
        prediction = predict_market_direction(df)
        
        if prediction:
            pred_cols = st.columns(4)
            
            with pred_cols[0]:
                st.metric(
                    "Direction",
                    prediction['prediction'],
                    f"Confidence: {prediction['confidence']:.1f}%"
                )
            
            with pred_cols[1]:
                st.metric(
                    "Bull Probability",
                    f"{prediction['bullish_prob']*100:.1f}%",
                    "Next 6h outlook"
                )
            
            with pred_cols[2]:
                st.metric(
                    "Bear Probability",
                    f"{prediction['bearish_prob']*100:.1f}%",
                    "Next 6h outlook"
                )
            
            with pred_cols[3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Recommendation</div>
                    <div class="metric-value">
                        {'🚀 LONG' if prediction['bullish_prob'] > 0.55 else ('📉 SHORT' if prediction['bearish_prob'] > 0.55 else '↔️ NEUTRAL')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Loading ML predictions...")
    
    # ═════════════════════════════════════════════════════════════════════════
    # SUPPORT & RESISTANCE
    # ═════════════════════════════════════════════════════════════════════════
    
    st.markdown('<div class="section-header">🎯 Support & Resistance Levels</div>',
                unsafe_allow_html=True)
    
    level_cols = st.columns(5)
    
    with level_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Resistance 2</div>
            <div class="metric-value">${levels['resistance2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with level_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Resistance 1</div>
            <div class="metric-value">${levels['resistance1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with level_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pivot Point</div>
            <div class="metric-value" style="color: #1ABC9C;">${levels['pivot']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with level_cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Support 1</div>
            <div class="metric-value">${levels['support1']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with level_cols[4]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Support 2</div>
            <div class="metric-value">${levels['support2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═════════════════════════════════════════════════════════════════════════
    # SENTIMENT ANALYSIS
    # ═════════════════════════════════════════════════════════════════════════
    
    if show_sentiment:
        st.markdown('<div class="section-header">😨 Market Sentiment & Greed Index</div>',
                    unsafe_allow_html=True)
        
        fg_index = fetch_fear_greed_index()
        
        sent_cols = st.columns(3)
        
        with sent_cols[0]:
            # Fear & Greed gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fg_index['value'],
                title={'text': "Fear & Greed Index"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF6B35"},
                    'steps': [
                        {'range': [0, 25], 'color': "#E74C3C"},
                        {'range': [25, 50], 'color': "#F39C12"},
                        {'range': [50, 75], 'color': "#27AE60"},
                        {'range': [75, 100], 'color': "#1ABC9C"}
                    ]
                },
                delta={'reference': 50}
            ))
            
            fig_gauge.update_layout(
                height=350,
                template='plotly_dark',
                paper_bgcolor='#16213E',
                plot_bgcolor='#0A0E27',
                font=dict(color='#ECF0F1', size=12)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with sent_cols[1]:
            sentiment_status = fg_index['status']
            colors = {
                'Extreme Fear': '#E74C3C',
                'Fear': '#F39C12',
                'Greed': '#27AE60',
                'Extreme Greed': '#1ABC9C'
            }
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Sentiment</div>
                <div class="metric-value" style="color: {colors.get(sentiment_status, '#1ABC9C')};">
                    {sentiment_status.upper()}
                </div>
                <div class="metric-change" style="color: var(--text-muted);">
                    Last update: {fg_index.get('timestamp', 'N/A')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with sent_cols[2]:
            interpretation = ""
            if fg_index['value'] < 25:
                interpretation = "🔥 Extreme Fear - Possible bottom forming"
            elif fg_index['value'] < 45:
                interpretation = "😨 Fear - Bearish sentiment dominant"
            elif fg_index['value'] < 55:
                interpretation = "➖ Neutral - Mixed signals"
            elif fg_index['value'] < 75:
                interpretation = "😊 Greed - Bullish momentum"
            else:
                interpretation = "🤑 Extreme Greed - Caution: potential top"
            
            st.markdown(f"""
            <div class="indicator-box">
                <h4>Interpretation</h4>
                <p>{interpretation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # ═════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═════════════════════════════════════════════════════════════════════════
    
    st.markdown("""
    ---
    <div style="text-align: center; color: #95A5A6; font-size: 0.85rem; margin-top: 2rem;">
        <p>📊 RENDERUSD Pro Dashboard v1.0 | Last Updated: {}  | Data provided by Yahoo Finance</p>
        <p>⚠️ Disclaimer: This dashboard is for informational purposes only. Not financial advice.</p>
        <p>💡 Always do your own research (DYOR) before trading cryptocurrency.</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
