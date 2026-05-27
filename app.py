import streamlit as st
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
