# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
import os
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="RNDR/RENDER AI Trading Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for Professional Dark UI
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #0f172a 100%);
    }
    
    /* Card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateY(-2px);
    }
    
    /* Signal badges */
    .signal-buy {
        background: linear-gradient(135deg, #10b981, #059669);
        padding: 0.75rem 2rem;
        border-radius: 40px;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }
    .signal-sell {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        padding: 0.75rem 2rem;
        border-radius: 40px;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(239,68,68,0.3);
    }
    .signal-hold {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        padding: 0.75rem 2rem;
        border-radius: 40px;
        font-weight: bold;
        font-size: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(245,158,11,0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(135deg, #e2e8f0, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(56, 189, 248, 0.1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: rgba(30, 41, 59, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #64748b;
        font-size: 0.75rem;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        margin-top: 2rem;
    }
    
    /* Price ticker animation */
    @keyframes pulse {
        0% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .price-ticker {
        font-size: 2rem;
        font-weight: bold;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================
def _zscore_rolling(s, w, min_periods=10):
    """Compute rolling z-score"""
    mu = s.rolling(w, min_periods=min_periods).mean()
    std = s.rolling(w, min_periods=min_periods).std()
    return (s - mu) / std.replace(0, np.nan)

@st.cache_data(ttl=3600)
def fetch_render_data():
    """Fetch RENDER/RNDR price data"""
    try:
        old = yf.download("RNDR-USD", start="2021-01-01", end="2024-07-14", progress=False)
        new = yf.download("RENDER-USD", start="2024-07-15", progress=False)
        
        if old.empty or new.empty:
            # Fallback: try RENDER-USD only
            data = yf.download("RENDER-USD", start="2021-01-01", progress=False)
            close = data["Close"].squeeze()
            close.index = pd.to_datetime(close.index).tz_localize(None)
        else:
            old_close = old["Close"].squeeze()
            new_close = new["Close"].squeeze()
            old_close.index = pd.to_datetime(old_close.index).tz_localize(None)
            new_close.index = pd.to_datetime(new_close.index).tz_localize(None)
            close = pd.concat([old_close, new_close]).sort_index()
            close = close[~close.index.duplicated(keep="last")]
        
        df = pd.DataFrame(index=close.index)
        df["close"] = close
        df["open"] = close.shift(1).fillna(close)
        df["high"] = close * 1.005
        df["low"] = close * 0.995
        df["volume"] = 1000000.0
        return df.dropna(subset=["close"])
    except Exception as e:
        st.error(f"Error fetching RENDER data: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_btc_dominance_proxy():
    """Compute BTC dominance proxy using top altcoins"""
    tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "LINK-USD"]
    prices = {}
    for t in tickers:
        try:
            data = yf.download(t, start="2021-01-01", progress=False)
            if not data.empty:
                close = data["Close"].squeeze()
                close.index = pd.to_datetime(close.index).tz_localize(None)
                prices[t.split("-")[0]] = close.dropna()
        except:
            pass
    
    if "BTC" not in prices or len(prices) < 3:
        return None
    
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    daily = pd.date_range("2021-01-01", today, freq="D")
    aligned = {n: s.reindex(daily, method="ffill") for n, s in prices.items()}
    btc = aligned.pop("BTC")
    alt_avg = pd.DataFrame(aligned).mean(axis=1)
    btc_norm = btc / btc.iloc[0]
    alt_norm = alt_avg / alt_avg.iloc[0]
    dom = btc_norm / (btc_norm + alt_norm)
    dom = dom.replace([np.inf, -np.inf], np.nan).ffill()
    dom.name = "btc_d_proxy"
    return dom

@st.cache_data(ttl=300)
def get_inr_rate():
    """Get USD to INR exchange rate"""
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data["rates"].get("INR", 83.0)
    except:
        pass
    return 83.0  # fallback rate

def compute_features(render_df, btc_dom_series):
    """Build feature dataframe"""
    if render_df is None or render_df.empty:
        return None
    
    close = render_df["close"]
    features = {}
    
    # Momentum features
    for lag in [7, 14, 30]:
        features[f"mom_{lag}d"] = close.pct_change(lag)
    
    # BTC dominance features
    if btc_dom_series is not None:
        btc_aligned = btc_dom_series.reindex(close.index, method="ffill")
        features["btc_d_proxy"] = btc_aligned
        features["btc_d_proxy_7d"] = btc_aligned.pct_change(7)
        features["btc_d_proxy_30d"] = btc_aligned.pct_change(30)
        features["btc_d_proxy_zscore"] = _zscore_rolling(btc_aligned, 30)
    
    return pd.DataFrame(features, index=close.index)

def compute_atr(df, period=14):
    """Compute Average True Range"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=7).mean()

@st.cache_resource
def load_models():
    """Load trained models and configs"""
    model_dir = Path("models")
    
    # Check if models exist
    required_files = ["classifier.pkl", "regressor.pkl", "features.json", "metrics.json"]
    missing = [f for f in required_files if not (model_dir / f).exists()]
    
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}")
        st.info("Please ensure models are in the 'models/' directory")
        return None, None, None, None
    
    try:
        with open(model_dir / "classifier.pkl", "rb") as f:
            clf = joblib.load(f)
        with open(model_dir / "regressor.pkl", "rb") as f:
            reg = joblib.load(f)
        with open(model_dir / "features.json", "r") as f:
            feature_cols = json.load(f)
        with open(model_dir / "metrics.json", "r") as f:
            metrics = json.load(f)
        return clf, reg, feature_cols, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def create_gauge(probability, title, color="green"):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        title={"text": title, "font": {"size": 14, "color": "#e2e8f0"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(30, 41, 59, 0.3)",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 30], "color": "rgba(239, 68, 68, 0.3)"},
                {"range": [30, 70], "color": "rgba(245, 158, 11, 0.3)"},
                {"range": [70, 100], "color": "rgba(16, 185, 129, 0.3)"}
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "thickness": 0.75,
                "value": probability * 100
            }
        },
        number={"font": {"size": 28, "color": "#e2e8f0"}, "suffix": "%"},
        delta={"reference": 50, "increasing": {"color": "#10b981"}, "decreasing": {"color": "#ef4444"}}
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"}
    )
    return fig

def create_feature_importance_chart(importances, feature_names):
    """Create horizontal bar chart for feature importance"""
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df["Importance"],
        y=df["Feature"],
        orientation="h",
        marker=dict(
            color=df["Importance"],
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Importance", tickfont=dict(color="#e2e8f0"))
        ),
        text=df["Importance"].round(3),
        textposition="outside"
    ))
    fig.update_layout(
        title="Feature Importance (Classifier)",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.3)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155")
    )
    return fig

def save_prediction_to_csv(data, filename="predictions_history.csv"):
    """Save prediction to CSV history"""
    filepath = Path(filename)
    df_new = pd.DataFrame([data])
    
    if filepath.exists():
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Keep last 1000 records
        if len(df_combined) > 1000:
            df_combined = df_combined.tail(1000)
        df_combined.to_csv(filepath, index=False)
    else:
        df_new.to_csv(filepath, index=False)
    
    return filepath

def load_prediction_history(filename="predictions_history.csv"):
    """Load prediction history from CSV"""
    filepath = Path(filename)
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()

def tradingview_html(symbol="RENDERUSDT", theme="dark"):
    """Generate TradingView widget HTML"""
    return f"""
    <div style="border-radius: 16px; overflow: hidden; border: 1px solid rgba(56,189,248,0.2);">
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container">
            <div id="tradingview_chart"></div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget({{
                    "width": "100%",
                    "height": 500,
                    "symbol": "{symbol}",
                    "interval": "D",
                    "timezone": "Etc/UTC",
                    "theme": "{theme}",
                    "style": "1",
                    "locale": "en",
                    "toolbar_bg": "#f1f3f6",
                    "enable_publishing": false,
                    "allow_symbol_change": true,
                    "container_id": "tradingview_chart"
                }});
            </script>
        </div>
        <!-- TradingView Widget END -->
    </div>
    """

# ============================================================================
# Main Application
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/render-render-logo.png", width=60)
        st.title("RENDER AI Dashboard")
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        show_tradingview = st.checkbox("Show TradingView Chart", value=True)
        inr_enabled = st.checkbox("Show INR Values", value=True)
        theme = st.selectbox("Chart Theme", ["dark", "light"], index=0)
        
        st.markdown("---")
        st.subheader("📊 Model Info")
        
        # Load models
        clf, reg, feature_cols, metrics = load_models()
        
        if clf is None:
            st.error("Models not loaded. Please check models directory.")
            st.stop()
        
        # Display model info
        st.info(f"""
        - **Features:** {len(feature_cols)}
        - **Classifier:** LightGBM + Calibration
        - **Regressor:** LightGBM
        - **Threshold:** {metrics.get('threshold', 0.7):.0%}
        - **Holdout Sharpe:** {metrics.get('holdout', [{}])[0].get('sharpe', 0):.2f}
        - **Win Rate:** {metrics.get('holdout', [{}])[0].get('win_rate', 0):.1%}
        """)
        
        st.markdown("---")
        st.caption("⚠️ Not financial advice. For educational purposes only.")
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>RNDR / RENDER AI Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Fetch data
    with st.spinner("Fetching market data..."):
        render_df = fetch_render_data()
        btc_dom = fetch_btc_dominance_proxy()
        
        if render_df is None or render_df.empty:
            st.error("Failed to fetch RENDER price data.")
            st.stop()
        
        # Get current price
        current_price = render_df["close"].iloc[-1]
        prev_price = render_df["close"].iloc[-2] if len(render_df) > 1 else current_price
        daily_change = (current_price - prev_price) / prev_price * 100
        
        # Get INR rate
        inr_rate = get_inr_rate() if inr_enabled else 1
        current_price_inr = current_price * inr_rate
        
        # Build features
        features_df = compute_features(render_df, btc_dom)
        if features_df is None or features_df.empty:
            st.error("Failed to compute features.")
            st.stop()
        
        # Prepare latest feature vector
        latest_features = features_df[feature_cols].iloc[-1:].fillna(features_df[feature_cols].median())
        latest_values = latest_features.values.astype(float)
    
    # Make predictions
    proba = clf.predict_proba(latest_values)[0]
    p_down, p_up = proba[0], proba[1]
    
    # Determine signal
    threshold = metrics.get("threshold", 0.7)
    if p_up >= threshold:
        signal = "BUY"
        signal_color = "buy"
    elif p_down >= threshold:
        signal = "SELL"
        signal_color = "sell"
    else:
        signal = "HOLD"
        signal_color = "hold"
    
    confidence = max(p_up, p_down)
    
    # Regressor forecast
    forecast_price = reg.predict(latest_values)[0]
    expected_return = (forecast_price / current_price - 1) * 100
    forecast_price_inr = forecast_price * inr_rate
    
    # Risk metrics
    atr = compute_atr(render_df).iloc[-1]
    risk_mult = metrics.get("risk_multiplier", 1.0)
    tp_price = current_price + (2.0 * atr)
    sl_price = current_price - (1.5 * atr)
    tp_price_inr = tp_price * inr_rate
    sl_price_inr = sl_price * inr_rate
    risk_reward = ((tp_price - current_price) / (current_price - sl_price)) if (current_price - sl_price) > 0 else 0
    
    # Timestamp
    last_update = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    # ========================================================================
    # MARKET OVERVIEW Row
    # ========================================================================
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💰 Current Price (USD)", f"${current_price:,.4f}", delta=f"{daily_change:+.2f}%")
        if inr_enabled:
            st.metric("💰 Current Price (INR)", f"₹{current_price_inr:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📊 BTC Dominance", f"{btc_dom.iloc[-1]:.2%}" if btc_dom is not None else "N/A")
        st.metric("🕐 Last Update", last_update[:16])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📈 24h Volume", f"${render_df['volume'].iloc[-1]:,.0f}")
        st.metric("📉 ATR (14)", f"${atr:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🎯 30d High/Low", f"${render_df['high'].rolling(30).max().iloc[-1]:.2f} / ${render_df['low'].rolling(30).min().iloc[-1]:.2f}")
        st.metric("📅 Target Date", (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # AI SIGNAL PANEL
    # ========================================================================
    st.markdown("---")
    st.subheader("🤖 AI Signal Panel")
    
    sig_col1, sig_col2, sig_col3 = st.columns([1, 1, 1])
    
    with sig_col1:
        st.markdown(f'<div class="signal-{signal_color}" style="text-align: center;">{signal}</div>', unsafe_allow_html=True)
        st.caption(f"Threshold: {threshold:.0%} | Confidence: {confidence:.1%}")
        
        # Confidence bar
        st.markdown(f"""
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Signal strength
        if confidence >= 0.8:
            strength = "Strong"
        elif confidence >= 0.6:
            strength = "Medium"
        else:
            strength = "Weak"
        st.info(f"Signal Strength: **{strength}**")
    
    with sig_col2:
        # Bull/Bear probabilities
        st.markdown("#### Bull vs Bear")
        bull_color = "#10b981" if p_up > p_down else "#f59e0b"
        bear_color = "#ef4444" if p_down > p_up else "#f59e0b"
        
        fig_prob = go.Figure(data=[
            go.Bar(name="Bull", x=["Probability"], y=[p_up * 100], marker_color=bull_color, text=f"{p_up*100:.1f}%", textposition="auto"),
            go.Bar(name="Bear", x=["Probability"], y=[p_down * 100], marker_color=bear_color, text=f"{p_down*100:.1f}%", textposition="auto")
        ])
        fig_prob.update_layout(
            barmode="group",
            height=200,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30,41,59,0.3)",
            font=dict(color="#e2e8f0"),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with sig_col3:
        st.markdown("#### Model Metrics")
        st.metric("Holdout Sharpe", f"{metrics.get('holdout', [{}])[0].get('sharpe', 0):.2f}")
        st.metric("Profit Factor", f"{metrics.get('holdout', [{}])[0].get('profit_factor', 0):.2f}")
        st.metric("Max Drawdown", f"{metrics.get('holdout', [{}])[0].get('max_drawdown', 0):.1%}")
    
    # ========================================================================
    # PRICE FORECAST & RISK MANAGEMENT
    # ========================================================================
    st.markdown("---")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📈 Price Forecast")
        st.metric("Current Price", f"${current_price:,.4f}", delta=f"{daily_change:+.2f}%")
        st.metric("Forecast (30d)", f"${forecast_price:,.4f}", delta=f"{expected_return:+.2f}%")
        st.metric("Forecast Direction", "🟢 UP" if expected_return > 0 else "🔴 DOWN")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_f2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("🛡️ Risk Management")
        st.metric("Take Profit (TP)", f"${tp_price:,.4f}", delta=f"+{((tp_price/current_price-1)*100):.2f}%")
        st.metric("Stop Loss (SL)", f"${sl_price:,.4f}", delta=f"{((sl_price/current_price-1)*100):.2f}%")
        st.metric("Risk/Reward", f"{risk_reward:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_f3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("💱 INR Conversion")
        st.metric("USD/INR Rate", f"₹{inr_rate:.2f}")
        st.metric("Current Price (INR)", f"₹{current_price_inr:,.2f}")
        st.metric("Forecast (INR)", f"₹{forecast_price_inr:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # PROBABILITY GAUGES
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Probability Visualization")
    
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        fig_bull = create_gauge(p_up, "Bull Probability", "#10b981")
        st.plotly_chart(fig_bull, use_container_width=True)
    with gauge_col2:
        fig_bear = create_gauge(p_down, "Bear Probability", "#ef4444")
        st.plotly_chart(fig_bear, use_container_width=True)
    
    # ========================================================================
    # TRADINGVIEW CHART
    # ========================================================================
    if show_tradingview:
        st.markdown("---")
        st.subheader("📉 Live Chart (TradingView)")
        st.components.v1.html(tradingview_html("RENDERUSDT", theme), height=550)
    
    # ========================================================================
    # FEATURE IMPORTANCE
    # ========================================================================
    st.markdown("---")
    st.subheader("📊 Model Analysis")
    
    tab_imp, tab_perf, tab_hist = st.tabs(["Feature Importance", "Performance Metrics", "Prediction History"])
    
    with tab_imp:
        # Extract feature importance from classifier
        if hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'feature_importances_'):
            importances = clf.estimators_[0].feature_importances_
            fig_imp = create_feature_importance_chart(importances, feature_cols)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Feature importance not available for calibrated classifier. Training raw model required.")
    
    with tab_perf:
        st.subheader("Holdout Performance")
        holdout = metrics.get("holdout", [])
        if holdout:
            perf_df = pd.DataFrame(holdout)
            perf_df["cost_pct"] = perf_df["cost"] * 100
            perf_df = perf_df.rename(columns={
                "n_trades": "Trades",
                "win_rate": "Win Rate",
                "profit_factor": "Profit Factor",
                "sharpe": "Sharpe",
                "max_drawdown": "Max DD",
                "cost_pct": "Cost (%)"
            })
            st.dataframe(perf_df[["Trades", "Win Rate", "Profit Factor", "Sharpe", "Max DD", "Cost (%)"]].style.format({
                "Win Rate": "{:.1%}",
                "Max DD": "{:.1%}",
                "Cost (%)": "{:.3f}%"
            }), use_container_width=True)
            
            # Cost sensitivity chart
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Scatter(x=perf_df["cost"], y=perf_df["sharpe"], mode="lines+markers", name="Sharpe"))
            fig_cost.add_trace(go.Scatter(x=perf_df["cost"], y=perf_df["profit_factor"], mode="lines+markers", name="Profit Factor", yaxis="y2"))
            fig_cost.update_layout(
                title="Cost Sensitivity Analysis",
                xaxis_title="Transaction Cost",
                yaxis_title="Sharpe Ratio",
                yaxis2=dict(title="Profit Factor", overlaying="y", side="right"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(30,41,59,0.3)",
                font=dict(color="#e2e8f0")
            )
            st.plotly_chart(fig_cost, use_container_width=True)
        else:
            st.warning("No holdout metrics available.")
        
        st.subheader("Regression Metrics")
        st.metric("MAE", f"{metrics.get('regression_mae', 0):.4f}")
        st.metric("RMSE", f"{metrics.get('regression_rmse', 0):.4f}")
    
    with tab_hist:
        st.subheader("📜 Prediction History")
        
        # Save current prediction to history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "p_up": round(p_up, 4),
            "p_down": round(p_down, 4),
            "current_price": round(current_price, 4),
            "forecast_price": round(forecast_price, 4),
            "expected_return": round(expected_return, 2),
            "tp": round(tp_price, 4),
            "sl": round(sl_price, 4),
            "confidence": round(confidence, 4)
        }
        
        # Save button
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            if st.button("💾 Save Current Prediction", use_container_width=True):
                save_prediction_to_csv(history_entry)
                st.success("Prediction saved to history!")
                st.balloons()
        
        with col_save2:
            # Download full history
            hist_df = load_prediction_history()
            if not hist_df.empty:
                csv_data = hist_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All History (CSV)",
                    data=csv_data,
                    file_name=f"render_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Display history table
        hist_df = load_prediction_history()
        if not hist_df.empty:
            display_df = hist_df.tail(30).copy()
            display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
            display_df = display_df[["timestamp", "signal", "p_up", "p_down", "current_price", "forecast_price", "expected_return", "confidence"]]
            display_df.columns = ["Date", "Signal", "P(UP)", "P(DOWN)", "Price", "Forecast", "Return %", "Confidence"]
            st.dataframe(display_df.style.format({
                "P(UP)": "{:.1%}",
                "P(DOWN)": "{:.1%}",
                "Return %": "{:+.2f}%",
                "Confidence": "{:.1%}"
            }), use_container_width=True)
        else:
            st.info("No predictions saved yet. Click 'Save Current Prediction' to start history.")
    
    # ========================================================================
    # EXPORT SECTION
    # ========================================================================
    st.markdown("---")
    st.subheader("📎 Export")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    # Prepare report data
    report_data = {
        "timestamp": last_update,
        "signal": signal,
        "confidence": confidence,
        "p_up": p_up,
        "p_down": p_down,
        "current_price_usd": current_price,
        "current_price_inr": current_price_inr,
        "forecast_price_usd": forecast_price,
        "forecast_price_inr": forecast_price_inr,
        "expected_return": expected_return,
        "tp_usd": tp_price,
        "sl_usd": sl_price,
        "risk_reward": risk_reward,
        "btc_dominance": btc_dom.iloc[-1] if btc_dom is not None else None,
        "atr": atr,
        "threshold": threshold,
        "model_sharpe": metrics.get('holdout', [{}])[0].get('sharpe', 0),
        "model_win_rate": metrics.get('holdout', [{}])[0].get('win_rate', 0)
    }
    
    with export_col1:
        # Export as JSON
        json_str = json.dumps(report_data, indent=2, default=str)
        st.download_button(
            label="📄 Export as JSON",
            data=json_str,
            file_name=f"render_signal_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with export_col2:
        # Export as CSV (single prediction)
        df_single = pd.DataFrame([report_data])
        csv_single = df_single.to_csv(index=False)
        st.download_button(
            label="📊 Export as CSV",
            data=csv_single,
            file_name=f"render_signal_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with export_col3:
        # Generate and download report
        report_html = f"""
        <html>
        <head><title>RENDER AI Signal Report</title></head>
        <body style="font-family: Arial; padding: 20px;">
            <h1>RENDER AI Trading Signal Report</h1>
            <p><strong>Generated:</strong> {last_update}</p>
            <hr>
            <h2>Signal: {signal}</h2>
            <p><strong>Confidence:</strong> {confidence:.1%}</p>
            <p><strong>Bull Probability:</strong> {p_up:.1%}</p>
            <p><strong>Bear Probability:</strong> {p_down:.1%}</p>
            <hr>
            <h3>Price Forecast</h3>
            <p>Current: ${current_price:.4f} (₹{current_price_inr:.2f})</p>
            <p>30d Forecast: ${forecast_price:.4f} (₹{forecast_price_inr:.2f})</p>
            <p>Expected Return: {expected_return:+.2f}%</p>
            <hr>
            <h3>Risk Management</h3>
            <p>Take Profit: ${tp_price:.4f}</p>
            <p>Stop Loss: ${sl_price:.4f}</p>
            <p>Risk/Reward: {risk_reward:.2f}</p>
            <hr>
            <h3>Model Metrics</h3>
            <p>Holdout Sharpe: {report_data['model_sharpe']:.2f}</p>
            <p>Win Rate: {report_data['model_win_rate']:.1%}</p>
            <p>Threshold: {threshold:.0%}</p>
            <hr>
            <p style="color: gray;">This report is auto-generated. Not financial advice.</p>
        </body>
        </html>
        """
        st.download_button(
            label="📑 Export Report (HTML)",
            data=report_html,
            file_name=f"render_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
    
    # ========================================================================
    # ERROR HANDLING DISPLAY (if any issues)
    # ========================================================================
    st.markdown("---")
    with st.expander("⚠️ System Status & Diagnostics"):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**Data Sources:**")
            st.write(f"- RENDER Data: {'✅' if render_df is not None else '❌'}")
            st.write(f"- BTC Dominance: {'✅' if btc_dom is not None else '❌'}")
            st.write(f"- INR Rate: {'✅' if inr_rate else '❌'}")
        with col_d2:
            st.write("**Models:**")
            st.write(f"- Classifier: {'✅' if clf else '❌'}")
            st.write(f"- Regressor: {'✅' if reg else '❌'}")
            st.write(f"- Features: {len(feature_cols) if feature_cols else 0}")
    
    # Footer
    st.markdown('<div class="footer">⚠️ This is an experimental AI prediction tool. Not financial advice. Past performance does not guarantee future results. Use at your own risk.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
