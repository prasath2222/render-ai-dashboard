import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION & EDGE-TO-EDGE CSS
# ==========================================
st.set_page_config(
    page_title="RENDER Pro Terminal", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Deep CSS override for premium institutional UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global resets for edge-to-edge layout */
    html, body, [class*="css"], [class*="st-"], .stApp {
        background-color: #0b0e11 !important; /* Binance Dark */
        color: #eaecef !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Remove all Streamlit padding and headers */
    .block-container { padding: 0rem 1rem 1rem 1rem !important; max-width: 100% !important; }
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Premium Metric Cards */
    .metric-card {
        background-color: #181a20;
        border: 1px solid #2b3139;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.4);
    }
    
    /* Top Ticker Bar */
    .ticker-bar {
        background-color: #181a20;
        border-bottom: 1px solid #2b3139;
        padding: 12px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-left: -1rem;
        margin-right: -1rem;
    }
    
    /* Typography overrides */
    .mono { font-family: 'JetBrains Mono', monospace; }
    .text-bull { color: #0ecb81 !important; } /* Crypto Green */
    .text-bear { color: #f6465d !important; } /* Crypto Red */
    .text-muted { color: #848e9c !important; }
    
    .label { font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #848e9c; margin-bottom: 4px; }
    .val-large { font-size: 28px; font-weight: 700; }
    .val-med { font-size: 20px; font-weight: 600; }
    .val-small { font-size: 14px; font-weight: 500; }
    
    hr { border-color: #2b3139; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ORCHESTRATION (USD & INR)
# ==========================================
@st.cache_data(ttl=60)
def fetch_market_data():
    try:
        # Fetch RENDER data
        df = yf.download("RENDER-USD", period="60d", interval="1h", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        
        # Fetch live INR rate
        inr_data = yf.download("USDINR=X", period="1d", progress=False)
        inr_rate = float(inr_data['Close'].iloc[-1]) if not inr_data.empty else 83.50
        
        return df, inr_rate
    except Exception as e:
        return pd.DataFrame(), 83.50

@st.cache_resource
def run_quant_models(df):
    """XGBoost models for directional classification and price regression."""
    df = df.copy()
    
    # Feature Engineering
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Vol_Mean'] = df['Volume'].rolling(20).mean()
    df['Price_Mom'] = df['Close'].pct_change(4)
    
    df['Target_Reg'] = df['Close'].shift(-1)
    df['Target_Cls'] = np.where(df['Target_Reg'] > df['Close'], 1, 0)
    df = df.dropna()
    
    features = ['Close', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'Vol_Mean', 'Price_Mom']
    X = df[features]
    
    # Classification Model (Direction)
    y_cls = df['Target_Cls']
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.1, shuffle=False)
    cls_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    cls_model.fit(X_train_c, y_train_c)
    cls_probs = cls_model.predict_proba(X.iloc[[-1]])[0]
    
    # Regression Model (Price Target)
    y_reg = df['Target_Reg']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.1, shuffle=False)
    reg_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg_model.fit(X_train_r, y_train_r)
    next_price = reg_model.predict(X.iloc[[-1]])[0]
    
    return cls_probs, next_price

# Helper to format dual currency
def fmt_price(usd, inr_rate, dec=4):
    inr = usd * inr_rate
    return f"${usd:,.{dec}f} <span class='text-muted val-small mono'>(₹{inr:,.2f})</span>"

# ==========================================
# 3. INITIALIZATION & HEADER
# ==========================================
df, inr_rate = fetch_market_data()

if df.empty:
    st.error("Market data feed unavailable. Retrying connection...")
    st.stop()

current_price = float(df['Close'].iloc[-1])
prev_price = float(df['Close'].iloc[-24]) # 24h ago approximation (using 1h candles)
pct_24h = ((current_price - prev_price) / prev_price) * 100

high_24h = float(df['High'].tail(24).max())
low_24h = float(df['Low'].tail(24).min())
vol_24h = float(df['Volume'].tail(24).sum())

color_class = "text-bull" if pct_24h >= 0 else "text-bear"
sign = "+" if pct_24h >= 0 else ""

# Ticker Bar HTML
st.markdown(f"""
<div class="ticker-bar">
    <div style="display: flex; align-items: center; gap: 20px;">
        <div>
            <h2 style="margin: 0; font-size: 22px; font-weight: 700;">RENDER/USDT</h2>
            <div class="label" style="color: #0ecb81;">● Real-Time Feed</div>
        </div>
        <div>
            <div class="val-large mono {color_class}">${current_price:,.4f}</div>
            <div class="mono text-muted">₹{(current_price * inr_rate):,.2f} INR</div>
        </div>
    </div>
    <div style="display: flex; gap: 40px; text-align: right;">
        <div>
            <div class="label">24h Change</div>
            <div class="val-small mono {color_class}">{sign}{pct_24h:.2f}%</div>
        </div>
        <div>
            <div class="label">24h High</div>
            <div class="val-small mono">${high_24h:,.4f}</div>
        </div>
        <div>
            <div class="label">24h Low</div>
            <div class="val-small mono">${low_24h:,.4f}</div>
        </div>
        <div>
            <div class="label">24h Volume (RNDR)</div>
            <div class="val-small mono">{vol_24h:,.0f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. MAIN LAYOUT: CHART + AI DASHBOARD
# ==========================================
# Embed TradingView Advanced Chart Engine
# This handles responsive sizing, smooth zooming, and split-pane indicators natively.
tv_widget = """
<div class="tradingview-widget-container" style="height: 600px; width: 100%; margin-bottom: 20px;">
  <div id="tradingview_rndr" style="height: 100%; width: 100%;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({
      "autosize": true,
      "symbol": "BINANCE:RENDERUSDT",
      "interval": "60",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "backgroundColor": "#0b0e11",
      "gridColor": "#1f2937",
      "hide_top_toolbar": false,
      "hide_legend": false,
      "save_image": false,
      "container_id": "tradingview_rndr",
      "toolbar_bg": "#0b0e11",
      "studies": [
        "Volume@tv-basicstudies",
        "MACD@tv-basicstudies",
        "RSI@tv-basicstudies"
      ]
  });
  </script>
</div>
"""
components.html(tv_widget, height=600)

# ==========================================
# 5. AI ENGINE & QUANT METRICS
# ==========================================
cls_probs, next_price = run_quant_models(df)
prob_bear, prob_bull = cls_probs[0], cls_probs[1]

# Signal Logic
if prob_bull > 0.65: ai_signal, sig_color = "STRONG BUY", "text-bull"
elif prob_bull > 0.52: ai_signal, sig_color = "BUY", "text-bull"
elif prob_bear > 0.65: ai_signal, sig_color = "STRONG SELL", "text-bear"
elif prob_bear > 0.52: ai_signal, sig_color = "SELL", "text-bear"
else: ai_signal, sig_color = "NEUTRAL", "text-muted"

# Support / Resistance via Pivot Points
pivot = (high_24h + low_24h + current_price) / 3
r1 = (2 * pivot) - low_24h
s1 = (2 * pivot) - high_24h

# Dashboard Grid
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">AI Directional Classification (1H)</div>
        <div class="val-large {sig_color}" style="margin-top: 5px;">{ai_signal}</div>
        <hr>
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span class="label">Bull Probability</span>
            <span class="mono val-small text-bull">{prob_bull*100:.1f}%</span>
        </div>
        <div style="width:100%; background-color:#2b3139; border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{prob_bull*100}%; background-color:#0ecb81; height:100%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 15px; margin-bottom: 5px;">
            <span class="label">Bear Probability</span>
            <span class="mono val-small text-bear">{prob_bear*100:.1f}%</span>
        </div>
        <div style="width:100%; background-color:#2b3139; border-radius:4px; height:6px; overflow:hidden;">
            <div style="width:{prob_bear*100}%; background-color:#f6465d; height:100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    proj_move = ((next_price - current_price) / current_price) * 100
    move_color = "text-bull" if proj_move >= 0 else "text-bear"
    move_sign = "+" if proj_move >= 0 else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">AI Target Regression (Next Candle)</div>
        <div class="val-med mono" style="margin-top: 5px;">{fmt_price(next_price, inr_rate)}</div>
        <hr>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span class="label">Projected Move</span>
            <span class="mono val-small {move_color}">{move_sign}{proj_move:.2f}%</span>
        </div>
        <div class="label">Order Execution Context</div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px;">
            <span class="text-muted mono val-small">Target TP</span>
            <span class="mono val-small text-bull">{fmt_price(current_price * 1.03, inr_rate)}</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 8px;">
            <span class="text-muted mono val-small">Dynamic SL</span>
            <span class="mono val-small text-bear">{fmt_price(current_price * 0.98, inr_rate)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Market Structure & Liquidity Zones</div>
        <div style="margin-top: 15px;">
            <div class="label" style="color: #f6465d;">Resistance 1 (R1)</div>
            <div class="val-med mono">{fmt_price(r1, inr_rate)}</div>
        </div>
        <hr>
        <div>
            <div class="label" style="color: #0ecb81;">Support 1 (S1)</div>
            <div class="val-med mono">{fmt_price(s1, inr_rate)}</div>
        </div>
        <hr>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <div class="label">VWAP Proxy</div>
                <div class="val-small mono text-muted">{fmt_price(pivot, inr_rate)}</div>
            </div>
            <div style="text-align: right;">
                <div class="label">Model Base</div>
                <div class="val-small text-muted mono">XGBoost DMatrix</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import yfinance as yf
import time
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIGURATION & SAFE CSS
# ==========================================
st.set_page_config(
    page_title="RENDER Nexus Terminal", 
    page_icon="🌌", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Safe, non-breaking CSS for dark mode and clean UI
st.markdown("""
<style>
    .block-container { padding-top: 2rem !important; max-width: 95% !important; }
    
    /* Clean Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Top Header Styling */
    .header-box {
        background: linear-gradient(145deg, #1f2937, #111827);
        border-left: 4px solid #00d2ff;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title { font-size: 1.2rem; color: #9ca3af; font-weight: 600; letter-spacing: 2px;}
    .header-price { font-size: 2.5rem; color: #ffffff; font-weight: bold; font-family: monospace;}
    .header-inr { font-size: 1rem; color: #6b7280; font-family: monospace;}
    
    .bull { color: #10b981 !important; font-weight: bold;}
    .bear { color: #ef4444 !important; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ACQUISITION & AI MODELS
# ==========================================
@st.cache_data(ttl=60)
def fetch_data():
    df = yf.download("RENDER-USD", period="90d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    if df.empty:
        return df

    # Basic feature engineering for the ML models
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['Vol_Mean'] = df['Volume'].rolling(20).mean()
    df['Target_Reg'] = df['Close'].shift(-1)
    df['Target_Cls'] = np.where(df['Target_Reg'] > df['Close'], 1, 0)
    
    return df.dropna()

@st.cache_resource
def run_ai_models(df):
    features = ['Close', 'High', 'Low', 'Volume', 'SMA_20', 'Vol_Mean']
    X = df[features]
    
    # Classification Model
    y_cls = df['Target_Cls']
    X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, shuffle=False)
    cls_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    cls_model.fit(X_train, y_train_c)
    cls_acc = accuracy_score(y_test_c, cls_model.predict(X_test))
    cls_prob = cls_model.predict_proba(X.iloc[[-1]])[0]
    
    # Regression Model
    y_reg = df['Target_Reg']
    X_train, X_test, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
    reg_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    reg_model.fit(X_train, y_train_r)
    reg_r2 = r2_score(y_test_r, reg_model.predict(X_test))
    next_price = reg_model.predict(X.iloc[[-1]])[0]
    
    return cls_prob, cls_acc, next_price, reg_r2

# ==========================================
# 3. HEADER UI
# ==========================================
df = fetch_data()
if df.empty:
    st.error("Error fetching data from Yahoo Finance. Check your network or ticker.")
    st.stop()

current_price = float(df['Close'].iloc[-1])
prev_price = float(df['Close'].iloc[-2])
pct_change = ((current_price - prev_price) / prev_price) * 100

color_class = "bull" if pct_change >= 0 else "bear"
arrow = "▲" if pct_change >= 0 else "▼"
inr_price = current_price * 83.50 # Static fallback for speed, dynamic forex takes too long

st.markdown(f"""
<div class="header-box">
    <div>
        <div class="header-title">RENDER / USDT</div>
        <div class="header-price">${current_price:,.4f}</div>
        <div class="header-inr">₹{inr_price:,.2f} INR</div>
    </div>
    <div style="text-align: right;">
        <h2 class="{color_class}" style="margin: 0; font-family: monospace;">{arrow} {abs(pct_change):.2f}%</h2>
        <p style="color: #6b7280; font-size: 0.9rem; margin: 5px 0 0 0;">Vol: {df['Volume'].iloc[-1]:,.0f}</p>
        <p style="color: #00d2ff; font-size: 0.8rem; margin: 5px 0 0 0; font-weight: bold;">● LIVE STREAM ACTIVE</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Professional Chart", "🤖 AI Classification", "🎯 AI Regression"])

# --- TAB 1: REAL TRADINGVIEW WIDGET ---
with tab1:
    # We use the actual TradingView iframe. This guarantees perfect zoom, pan, and indicators.
    tradingview_html = """
    <div class="tradingview-widget-container" style="height: 700px; width: 100%;">
      <div id="tradingview_rndr" style="height: calc(100% - 32px); width: 100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget(
      {
      "autosize": true,
      "symbol": "BINANCE:RENDERUSDT",
      "interval": "60",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "enable_publishing": false,
      "backgroundColor": "#0b0e14",
      "gridColor": "#1f2937",
      "hide_top_toolbar": false,
      "hide_legend": false,
      "save_image": false,
      "container_id": "tradingview_rndr",
      "studies": [
        "MACD@tv-basicstudies",
        "RSI@tv-basicstudies",
        "Volume@tv-basicstudies"
      ]
    }
      );
      </script>
    </div>
    """
    components.html(tradingview_html, height=700)

# --- TAB 2: AI CLASSIFICATION ---
with tab2:
    cls_prob, cls_acc, next_price, reg_r2 = run_ai_models(df)
    prob_bear, prob_bull = cls_prob[0], cls_prob[1]
    
    if prob_bull > 0.6: 
        signal, sig_color = "STRONG BUY", "#10b981"
    elif prob_bull > 0.5: 
        signal, sig_color = "BUY", "#00d2ff"
    else: 
        signal, sig_color = "SELL", "#ef4444"

    c1, c2, c3 = st.columns(3)
    c1.metric("AI Prediction Direction", signal)
    c2.metric("Bullish Confidence", f"{prob_bull*100:.1f}%")
    c3.metric("Model Historical Accuracy", f"{cls_acc*100:.1f}%")
    
    st.markdown("### Signal Strength Meter")
    st.progress(float(prob_bull))
    st.caption("👈 Bearish | Bullish 👉")

# --- TAB 3: AI REGRESSION ---
with tab3:
    proj_change = ((next_price - current_price) / current_price) * 100
    
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted Next Candle (1H)", f"${next_price:.4f}")
    r2.metric("Expected Move", f"{proj_change:.2f}%")
    r3.metric("Model R² Score", f"{reg_r2:.3f}")
    
    st.info("Regression model uses XGBoost to forecast the exact closing price of the next 1-hour candle based on volume profiles and moving averages.")
