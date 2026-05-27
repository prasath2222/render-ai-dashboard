import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIGURATION & CSS OVERRIDES
# ==========================================
st.set_page_config(page_title="RNDR Nexus Terminal", page_icon="🌌", layout="wide", initial_sidebar_state="expanded")

def apply_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Global resets and background */
        .stApp {
            background-color: #0b0e14;
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        
        /* Hide Streamlit chrome */
        header, footer {visibility: hidden;}
        .block-container {padding-top: 1rem !important; max-width: 98% !important;}
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(18, 24, 38, 0.65);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 15px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            border: 1px solid rgba(0, 255, 136, 0.2);
        }
        
        /* Typography */
        h1, h2, h3 { font-weight: 700; letter-spacing: -0.5px; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        /* Colored Text */
        .text-bull { color: #00ff88; font-weight: 600; }
        .text-bear { color: #ff4d4d; font-weight: 600; }
        .text-neutral { color: #ffb800; font-weight: 600; }
        .text-neon { color: #00d2ff; }
        
        /* Metrics Header */
        .header-metric {
            display: flex; justify-content: space-between; align-items: center;
            background: #121826; border-radius: 8px; padding: 15px 25px;
            border-left: 4px solid #00d2ff;
        }
        .price-large { font-size: 2.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #0d1117 !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
            color: #8b949e;
        }
        .stTabs [aria-selected="true"] {
            color: #00d2ff !important;
            border-bottom: 2px solid #00d2ff !important;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ==========================================
# 2. DATA ACQUISITION & FEATURE ENGINEERING
# ==========================================
INTERVAL_MAP = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d", "1d": "max"}

@st.cache_data(ttl=60)
def fetch_market_data(ticker="RENDER-USD", interval="1h"):
    period = INTERVAL_MAP.get(interval, "60d")
    df = yf.download(ticker, interval=interval, period=period, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    if df.empty:
        return df

    # Technical Indicators
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], 20)
    df['EMA_50'] = ta.trend.ema_indicator(df['Close'], 50)
    df['SMA_200'] = ta.trend.sma_indicator(df['Close'], 200)
    df['RSI'] = ta.momentum.rsi(df['Close'], 14)
    
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], 14)
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    bb = ta.volatility.BollingerBands(df['Close'], 20, 2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    # VWAP (Approximation since we lack tick data)
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    # Whale Proxy (Volume > 3 Standard Deviations from 20-period mean)
    vol_mean = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['Whale_Spike'] = np.where(df['Volume'] > (vol_mean + (3 * vol_std)), 1, 0)
    
    # Returns for ML
    df['Return_Next'] = df['Close'].shift(-1) / df['Close'] - 1
    
    return df.dropna()

def fetch_inr_rate():
    try:
        # Fallback to hardcoded if Yahoo fails on forex
        inr = yf.download("USDINR=X", period="1d", progress=False)
        return float(inr['Close'].iloc[-1])
    except:
        return 83.50 

# ==========================================
# 3. MACHINE LEARNING ENGINE
# ==========================================
@st.cache_resource
def train_classification_model(df):
    # Predict direction of next candle
    features = ['EMA_20', 'RSI', 'MACD', 'ATR', 'OBV', 'Whale_Spike']
    X = df[features].copy()
    
    # 0 = Bearish, 1 = Neutral, 2 = Bullish
    conditions = [
        (df['Return_Next'] > 0.002),
        (df['Return_Next'] < -0.002)
    ]
    choices = [2, 0]
    y = np.select(conditions, choices, default=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    probs = model.predict_proba(X.iloc[[-1]])[0]
    
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='weighted'),
        'cm': confusion_matrix(y_test, preds)
    }
    return model, probs, metrics

@st.cache_resource
def train_regression_model(df):
    # Predict exact price of next candle
    features = ['Close', 'EMA_20', 'RSI', 'MACD', 'ATR', 'OBV']
    X = df[features].copy()
    y = df['Close'].shift(-1).fillna(method='ffill')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    next_price = model.predict(X.iloc[[-1]])[0]
    
    metrics = {
        'r2': r2_score(y_test, preds)
    }
    return model, next_price, metrics

# ==========================================
# 4. SIDEBAR & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    selected_tf = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    
    st.markdown("### 📊 Chart Overlays")
    show_ema = st.toggle("EMA 20/50", value=True)
    show_bb = st.toggle("Bollinger Bands", value=False)
    show_vwap = st.toggle("VWAP", value=True)
    
    st.markdown("### 📈 Sub-Indicators")
    show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD", value=True)
    show_vol = st.checkbox("Volume & Order Flow", value=True)
    
    st.markdown("---")
    st.markdown("<span style='font-size:0.8rem; color:#8b949e;'>AUTO REFRESH</span>", unsafe_allow_html=True)
    auto_refresh = st.toggle("Live Data Stream", value=False)
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# ==========================================
# 5. MAIN EXECUTION & HEADER
# ==========================================
df = fetch_market_data(interval=selected_tf)
inr_rate = fetch_inr_rate()

if df.empty:
    st.error("No data fetched from Oracle. Check network connection.")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]
current_price = latest['Close']
price_change = ((current_price - prev['Close']) / prev['Close']) * 100
color_class = "text-bull" if price_change >= 0 else "text-bear"
arrow = "▲" if price_change >= 0 else "▼"

# Header Panel
st.markdown(f"""
<div class="header-metric">
    <div>
        <h2 style="margin:0; color:#8b949e; font-size:1rem; letter-spacing:2px;">RENDER / USDT</h2>
        <div class="price-large">${current_price:,.4f}</div>
        <div class="mono" style="color:#8b949e; font-size:0.9rem;">₹{(current_price * inr_rate):,.2f} INR</div>
    </div>
    <div style="text-align: right;">
        <div class="{color_class} mono" style="font-size:1.5rem;">{arrow} {abs(price_change):.2f}%</div>
        <div style="font-size:0.8rem; color:#8b949e; margin-top:5px;">Vol: {latest['Volume']:,.0f} RNDR</div>
        <div style="font-size:0.7rem; color:#00d2ff; margin-top:8px;">● LIVE STREAM ACTIVE</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.write("") # Spacer

# ==========================================
# 6. TAB ARCHITECTURE
# ==========================================
tab_chart, tab_cls, tab_reg, tab_flow = st.tabs([
    "📈 Professional Chart", 
    "🤖 AI Classification", 
    "🎯 AI Regression", 
    "🐋 Order Flow & Liquidity"
])

# ------------------------------------------
# TAB 1: PROFESSIONAL CHARTING
# ------------------------------------------
with tab_chart:
    # Determine rows dynamically based on selected indicators
    active_subs = sum([show_rsi, show_macd, show_vol])
    row_heights = [0.6] + [0.4 / active_subs] * active_subs if active_subs > 0 else [1.0]
    
    fig = make_subplots(
        rows=1 + active_subs, cols=1, 
        shared_xaxes=True, vertical_spacing=0.02,
        row_heights=row_heights
    )
    
    # 1. Main Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='RNDR', increasing_line_color='#00ff88', decreasing_line_color='#ff4d4d'
    ), row=1, col=1)
    
    if show_ema:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='#00d2ff', width=1.5), name='EMA 20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], line=dict(color='#ffb800', width=1.5), name='EMA 50'), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB High'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], fill='tonexty', fillcolor='rgba(255,255,255,0.05)', line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB Low'), row=1, col=1)
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#b142ff', width=2, dash='dot'), name='VWAP'), row=1, col=1)

    curr_row = 2
    
    # 2. Volume
    if show_vol:
        colors = ['#00ff88' if row['Close'] >= row['Open'] else '#ff4d4d' for idx, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=curr_row, col=1)
        curr_row += 1

    # 3. MACD
    if show_macd:
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='#00d2ff', width=1.5), name='MACD'), row=curr_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#ffb800', width=1.5), name='Signal'), row=curr_row, col=1)
        macd_colors = ['#00ff88' if val >= 0 else '#ff4d4d' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], marker_color=macd_colors, name='Histogram'), row=curr_row, col=1)
        curr_row += 1

    # 4. RSI
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#b142ff', width=1.5), name='RSI'), row=curr_row, col=1)
        fig.add_hline(y=70, line=dict(color='#ff4d4d', width=1, dash='dash'), row=curr_row, col=1)
        fig.add_hline(y=30, line=dict(color='#00ff88', width=1, dash='dash'), row=curr_row, col=1)

    # Layout mimicking TradingView
    fig.update_layout(
        height=800,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='#0b0e14',
        paper_bgcolor='#0b0e14',
        hovermode='x unified',
        dragmode='pan',
        showlegend=False,
        xaxis_rangeslider_visible=False,
        font=dict(color='#8b949e', family='JetBrains Mono')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.05)', zeroline=False, side='right')
    
    # st.plotly_chart with config for scroll zoom and hiding the default toolbar
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})

# ------------------------------------------
# TAB 2: AI CLASSIFICATION
# ------------------------------------------
with tab_cls:
    cls_model, cls_probs, cls_metrics = train_classification_model(df)
    
    prob_bear, prob_neutral, prob_bull = cls_probs[0], cls_probs[1], cls_probs[2]
    
    if prob_bull > 0.6: signal, sig_color = "STRONG BUY", "#00ff88"
    elif prob_bull > 0.45: signal, sig_color = "BUY", "#00d2ff"
    elif prob_bear > 0.6: signal, sig_color = "STRONG SELL", "#ff4d4d"
    elif prob_bear > 0.45: signal, sig_color = "SELL", "#ff8800"
    else: signal, sig_color = "NEUTRAL", "#ffb800"

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>AI SIGNAL ({selected_tf})</p><h1 style='color:{sig_color}; font-size:2.5rem; margin:0;'>{signal}</h1>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>BULL CONFIDENCE</p><h2 class='mono'>{prob_bull*100:.1f}%</h2>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>MODEL ACCURACY</p><h2 class='mono'>{cls_metrics['accuracy']*100:.1f}%</h2>", unsafe_allow_html=True)
    
    st.markdown("<br><p style='color:#8b949e;'>PREDICTION PROBABILITY DISTRIBUTION</p>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="width:100%; height:8px; display:flex; border-radius:4px; overflow:hidden;">
        <div style="width:{prob_bear*100}%; background:#ff4d4d;"></div>
        <div style="width:{prob_neutral*100}%; background:#ffb800;"></div>
        <div style="width:{prob_bull*100}%; background:#00ff88;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:0.8rem; color:#8b949e; font-family:'JetBrains Mono';">
        <span>BEAR ({prob_bear*100:.1f}%)</span>
        <span>NEUTRAL ({prob_neutral*100:.1f}%)</span>
        <span>BULL ({prob_bull*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: AI REGRESSION
# ------------------------------------------
with tab_reg:
    reg_model, next_price, reg_metrics = train_regression_model(df)
    proj_change = ((next_price - current_price) / current_price) * 100
    
    rc_color = "#00ff88" if proj_change >= 0 else "#ff4d4d"
    rc_arrow = "▲" if proj_change >= 0 else "▼"

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>PREDICTED NEXT CANDLE ({selected_tf})</p><h1 class='mono' style='color:#ffffff; font-size:2.5rem; margin:0;'>${next_price:.4f}</h1>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>EXPECTED MOVE</p><h2 class='mono' style='color:{rc_color};'>{rc_arrow} {abs(proj_change):.2f}%</h2>", unsafe_allow_html=True)
    with r3:
        st.markdown(f"<p style='color:#8b949e; margin:0;'>MODEL R² SCORE</p><h2 class='mono'>{reg_metrics['r2']:.4f}</h2>", unsafe_allow_html=True)
    
    # Simple projection visualization
    st.markdown("<br><p style='color:#8b949e;'>PRICE PATH SIMULATION</p>", unsafe_allow_html=True)
    proj_df = df.tail(20).copy()
    
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(x=proj_df.index, y=proj_df['Close'], line=dict(color='#00d2ff', width=2), name='Historical'))
    
    # Add predicted point
    next_time = proj_df.index[-1] + (proj_df.index[-1] - proj_df.index[-2])
    fig_proj.add_trace(go.Scatter(x=[proj_df.index[-1], next_time], y=[proj_df['Close'].iloc[-1], next_price], 
                                  line=dict(color=rc_color, width=3, dash='dot'), name='Forecast'))
    
    fig_proj.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
    fig_proj.update_xaxes(showgrid=False, zeroline=False)
    fig_proj.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', side='right')
    st.plotly_chart(fig_proj, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------
# TAB 4: ORDER FLOW & LIQUIDITY (Proxy)
# ------------------------------------------
with tab_flow:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Institutional Activity Proxies")
    st.markdown("<span style='color:#8b949e; font-size:0.8rem;'>*Note: True Level 2 order flow requires paid APIs. These metrics are derived using quantitative volumetric algorithms.*</span>", unsafe_allow_html=True)
    
    f1, f2, f3 = st.columns(3)
    
    # Smart Money Concepts: Support & Resistance via Pivot Points
    high = df['High'].tail(20).max()
    low = df['Low'].tail(20).min()
    pivot = (high + low + current_price) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    
    with f1:
        st.markdown(f"**Immediate Resistance (R1):** <span class='text-bear'>${r1:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"**Immediate Support (S1):** <span class='text-bull'>${s1:.4f}</span>", unsafe_allow_html=True)
    
    with f2:
        whale_count = df['Whale_Spike'].tail(24).sum()
        st.markdown(f"**Whale Spikes (Last 24 candles):** <span class='text-neon'>{whale_count}</span>", unsafe_allow_html=True)
        st.markdown(f"**Current Volatility (ATR):** <span class='mono'>{latest['ATR']:.4f}</span>", unsafe_allow_html=True)
        
    with f3:
        buy_vol = df[df['Close'] > df['Open']]['Volume'].tail(14).sum()
        sell_vol = df[df['Close'] < df['Open']]['Volume'].tail(14).sum()
        total_vol = buy_vol + sell_vol
        buy_pressure = (buy_vol / total_vol) * 100 if total_vol > 0 else 50
        
        st.markdown(f"**Buy Pressure (14 periods):** <span class='text-bull'>{buy_pressure:.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"**Sell Pressure (14 periods):** <span class='text-bear'>{100-buy_pressure:.1f}%</span>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
