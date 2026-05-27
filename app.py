"""
╔══════════════════════════════════════════════════════════╗
║   RENDER (RNDR) — INSTITUTIONAL PREDICTION DASHBOARD     ║
║   Production-grade Streamlit app                         ║
║   Fully modular, responsive, premium fintech UI          ║
╚══════════════════════════════════════════════════════════╝
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import ta
import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

# ── Optional model imports (graceful fallback) ──
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ──────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RENDER · ML Prediction Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────
RENDER_TICKER = "RENDER-USD"
BTC_TICKER    = "BTC-USD"
ETH_TICKER    = "ETH-USD"
SOL_TICKER    = "SOL-USD"
INTERVAL      = "1h"
PERIOD        = "60d"
LOOKBACK      = 48

MODEL_FILES = {
    "xgb_direction": "xgb_direction.joblib",
    "xgb_regressor": "xgb_regressor.joblib",
    "lstm_model":    "lstm_price.keras",
    "scaler_x":      "scaler_x.joblib",
    "scaler_y":      "scaler_y.joblib",
    "feature_cols":  "feature_cols.joblib",
}

DIRECTION_LABELS = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}
DIRECTION_COLORS = {0: "#ef4444", 1: "#f59e0b", 2: "#22c55e"}
DIRECTION_ICONS  = {0: "↓", 1: "→", 2: "↑"}

# ──────────────────────────────────────────────────────────
# GLOBAL CSS — Premium dark trading terminal UI
# ──────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary:    #060b14;
        --bg-secondary:  #0d1526;
        --bg-card:       #0f1c33;
        --bg-hover:      #162240;
        --border:        #1e3054;
        --border-glow:   #2563eb44;
        --accent-blue:   #3b82f6;
        --accent-cyan:   #06b6d4;
        --accent-green:  #22c55e;
        --accent-red:    #ef4444;
        --accent-yellow: #f59e0b;
        --accent-purple: #a855f7;
        --text-primary:  #e2e8f0;
        --text-secondary:#94a3b8;
        --text-muted:    #475569;
        --font-mono:     'Space Mono', monospace;
        --font-sans:     'DM Sans', sans-serif;
        --font-brand:    'Orbitron', sans-serif;
        --radius-sm:     6px;
        --radius-md:     12px;
        --radius-lg:     18px;
        --shadow-glow:   0 0 30px #3b82f620;
        --shadow-card:   0 4px 24px #00000060;
        --transition:    all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ── Global Reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stSidebar"] { background: var(--bg-secondary) !important; border-right: 1px solid var(--border); }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section.main > div { padding: 0 !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
    .stColumns { gap: 0 !important; }
    .element-container { margin: 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-blue); }

    /* ── Navbar ── */
    .rn-navbar {
        background: linear-gradient(180deg, #0a1220 0%, #060b14 100%);
        border-bottom: 1px solid var(--border);
        padding: 14px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 1000;
        backdrop-filter: blur(20px);
    }
    .rn-brand {
        font-family: var(--font-brand);
        font-size: 18px;
        font-weight: 900;
        letter-spacing: 4px;
        background: linear-gradient(135deg, #3b82f6, #06b6d4, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
    }
    .rn-live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #22c55e18;
        border: 1px solid #22c55e40;
        color: #22c55e;
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        padding: 4px 12px;
        border-radius: 999px;
        text-transform: uppercase;
    }
    .rn-live-dot {
        width: 6px; height: 6px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse-green 1.5s ease-in-out infinite;
    }
    @keyframes pulse-green {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.8); }
    }
    .rn-nav-time {
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--text-muted);
        letter-spacing: 1px;
    }

    /* ── Section Wrapper ── */
    .rn-section {
        padding: 24px 32px;
    }
    .rn-section-sm { padding: 16px 32px; }

    /* ── Section Headers ── */
    .rn-section-title {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 3px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .rn-section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }

    /* ── Price Hero ── */
    .rn-price-hero {
        padding: 32px 32px 24px;
        background: linear-gradient(135deg, #0d1526 0%, #060b14 60%, #0d1526 100%);
        border-bottom: 1px solid var(--border);
        position: relative;
        overflow: hidden;
    }
    .rn-price-hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 60% 80% at 80% 50%, #3b82f608 0%, transparent 70%);
        pointer-events: none;
    }
    .rn-price-ticker {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 3px;
        color: var(--accent-cyan);
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .rn-price-value {
        font-family: var(--font-brand);
        font-size: clamp(32px, 5vw, 52px);
        font-weight: 900;
        color: var(--text-primary);
        letter-spacing: -1px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .rn-price-change {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: var(--font-mono);
        font-size: 13px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 999px;
    }
    .rn-price-change.up   { background: #22c55e18; color: #22c55e; border: 1px solid #22c55e30; }
    .rn-price-change.down { background: #ef444418; color: #ef4444; border: 1px solid #ef444430; }
    .rn-price-change.flat { background: #f59e0b18; color: #f59e0b; border: 1px solid #f59e0b30; }

    /* ── Metric Cards ── */
    .rn-metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
    }
    .rn-metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        position: relative;
        overflow: hidden;
        transition: var(--transition);
        cursor: default;
    }
    .rn-metric-card:hover {
        border-color: var(--accent-blue);
        background: var(--bg-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow-glow);
    }
    .rn-metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
        opacity: 0;
        transition: var(--transition);
    }
    .rn-metric-card:hover::before { opacity: 1; }
    .rn-metric-label {
        font-family: var(--font-mono);
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 2px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .rn-metric-value {
        font-family: var(--font-mono);
        font-size: 20px;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    .rn-metric-sub {
        font-family: var(--font-sans);
        font-size: 11px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* ── Signal Card (Direction) ── */
    .rn-signal-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 28px 32px;
        position: relative;
        overflow: hidden;
    }
    .rn-signal-direction {
        font-family: var(--font-brand);
        font-size: clamp(24px, 4vw, 40px);
        font-weight: 900;
        letter-spacing: 4px;
        line-height: 1;
        margin-bottom: 8px;
    }
    .rn-signal-icon {
        font-size: 48px;
        line-height: 1;
        margin-bottom: 12px;
        filter: drop-shadow(0 0 20px currentColor);
    }

    /* ── Probability Bar ── */
    .rn-prob-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
    }
    .rn-prob-label {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        width: 50px;
        text-transform: uppercase;
    }
    .rn-prob-bar-bg {
        flex: 1;
        height: 6px;
        background: var(--bg-secondary);
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .rn-prob-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .rn-prob-bar-fill::after {
        content: '';
        position: absolute;
        right: 0; top: 0; bottom: 0;
        width: 4px;
        background: white;
        opacity: 0.6;
        border-radius: 999px;
        animation: shimmer 2s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 0.2; }
    }
    .rn-prob-pct {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        width: 36px;
        text-align: right;
    }

    /* ── Gauge / Confidence ── */
    .rn-confidence-ring {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 120px;
        height: 120px;
        margin: 0 auto 16px;
    }
    .rn-confidence-label {
        position: absolute;
        text-align: center;
    }
    .rn-confidence-pct {
        font-family: var(--font-brand);
        font-size: 22px;
        font-weight: 900;
        color: var(--text-primary);
        line-height: 1;
    }
    .rn-confidence-sub {
        font-family: var(--font-mono);
        font-size: 8px;
        letter-spacing: 1px;
        color: var(--text-muted);
        text-transform: uppercase;
    }

    /* ── Trade Levels ── */
    .rn-levels-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    .rn-level-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 20px;
        border-bottom: 1px solid var(--border);
        transition: var(--transition);
    }
    .rn-level-row:last-child { border-bottom: none; }
    .rn-level-row:hover { background: var(--bg-hover); }
    .rn-level-name {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .rn-level-price {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 700;
    }
    .rn-level-pct {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 400;
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* ── Regime Badge ── */
    .rn-regime-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 18px;
        border-radius: 999px;
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: 1px solid;
    }
    .rn-regime-badge.bullish {
        background: #22c55e10;
        border-color: #22c55e40;
        color: #22c55e;
    }
    .rn-regime-badge.bearish {
        background: #ef444410;
        border-color: #ef444440;
        color: #ef4444;
    }
    .rn-regime-badge.sideways {
        background: #f59e0b10;
        border-color: #f59e0b40;
        color: #f59e0b;
    }

    /* ── Squeeze Alert ── */
    .rn-squeeze-alert {
        display: flex;
        align-items: center;
        gap: 12px;
        background: linear-gradient(135deg, #f59e0b0a, #ef44440a);
        border: 1px solid #f59e0b40;
        border-radius: var(--radius-md);
        padding: 14px 20px;
        animation: squeeze-pulse 2s ease-in-out infinite;
    }
    @keyframes squeeze-pulse {
        0%, 100% { border-color: #f59e0b40; box-shadow: 0 0 0 0 #f59e0b20; }
        50% { border-color: #f59e0b80; box-shadow: 0 0 12px #f59e0b20; }
    }
    .rn-squeeze-icon {
        font-size: 20px;
        animation: bounce 1s ease-in-out infinite;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }
    .rn-squeeze-text {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        color: #f59e0b;
    }

    /* ── Indicator Pills ── */
    .rn-indicator-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 10px;
    }
    .rn-indicator-pill {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 12px 14px;
        text-align: center;
        transition: var(--transition);
    }
    .rn-indicator-pill:hover {
        border-color: var(--accent-blue);
        background: var(--bg-hover);
    }
    .rn-indicator-name {
        font-family: var(--font-mono);
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 2px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .rn-indicator-val {
        font-family: var(--font-mono);
        font-size: 16px;
        font-weight: 700;
    }
    .rn-indicator-badge {
        display: inline-block;
        font-family: var(--font-mono);
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 2px 6px;
        border-radius: 3px;
        margin-top: 4px;
        text-transform: uppercase;
    }

    /* ── Disclaimer / footer ── */
    .rn-footer {
        padding: 20px 32px;
        border-top: 1px solid var(--border);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }
    .rn-footer-text {
        font-family: var(--font-mono);
        font-size: 9px;
        color: var(--text-muted);
        letter-spacing: 1px;
    }

    /* ── Error / Warning Cards ── */
    .rn-error-card {
        background: #ef444408;
        border: 1px solid #ef444440;
        border-radius: var(--radius-md);
        padding: 20px 24px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
    }
    .rn-error-icon { font-size: 24px; line-height: 1; margin-top: 2px; }
    .rn-error-title {
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        color: #ef4444;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .rn-error-body {
        font-family: var(--font-sans);
        font-size: 13px;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .rn-warn-card {
        background: #f59e0b08;
        border: 1px solid #f59e0b40;
        border-radius: var(--radius-md);
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: var(--font-mono);
        font-size: 11px;
        color: #f59e0b;
        letter-spacing: 1px;
    }
    .rn-info-card {
        background: #3b82f608;
        border: 1px solid #3b82f640;
        border-radius: var(--radius-md);
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: var(--font-mono);
        font-size: 11px;
        color: var(--accent-blue);
        letter-spacing: 1px;
    }

    /* ── Skeleton Loading ── */
    .rn-skeleton {
        background: linear-gradient(90deg, var(--bg-card) 0%, var(--bg-hover) 50%, var(--bg-card) 100%);
        background-size: 200% 100%;
        animation: skeleton-wave 1.5s ease-in-out infinite;
        border-radius: var(--radius-sm);
    }
    @keyframes skeleton-wave {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ── Tab overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary) !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 0 32px !important;
        gap: 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-mono) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        color: var(--text-muted) !important;
        background: transparent !important;
        border: none !important;
        padding: 14px 20px !important;
        text-transform: uppercase !important;
        transition: var(--transition) !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom: 2px solid var(--accent-blue) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding: 0 !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-mono) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 10px 24px !important;
        transition: var(--transition) !important;
        cursor: pointer !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px #3b82f640 !important;
    }

    /* ── Selectbox ── */
    .stSelectbox > div > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-mono) !important;
        font-size: 11px !important;
    }

    /* ── Divider ── */
    .rn-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 4px 0;
    }

    /* ── Ticker tape ── */
    .rn-tape-wrapper {
        overflow: hidden;
        background: var(--bg-secondary);
        border-bottom: 1px solid var(--border);
        padding: 8px 0;
    }
    .rn-tape-inner {
        display: flex;
        gap: 40px;
        animation: tape-scroll 30s linear infinite;
        white-space: nowrap;
    }
    @keyframes tape-scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .rn-tape-item {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: var(--font-mono);
        font-size: 11px;
        font-weight: 700;
        color: var(--text-secondary);
    }
    .rn-tape-price { color: var(--text-primary); }
    .rn-tape-up { color: var(--accent-green); }
    .rn-tape-down { color: var(--accent-red); }

    /* ── Correlation badge ── */
    .rn-corr-bar {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
    .rn-corr-label {
        font-family: var(--font-mono);
        font-size: 10px;
        color: var(--text-muted);
        width: 40px;
        text-transform: uppercase;
    }
    .rn-corr-track {
        flex: 1;
        height: 4px;
        background: var(--bg-secondary);
        border-radius: 999px;
        overflow: hidden;
        position: relative;
    }
    .rn-corr-fill {
        height: 100%;
        border-radius: 999px;
        position: absolute;
    }
    .rn-corr-val {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        width: 36px;
        text-align: right;
    }

    /* ── Model ensemble row ── */
    .rn-model-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid var(--border);
    }
    .rn-model-row:last-child { border-bottom: none; }
    .rn-model-name {
        font-family: var(--font-mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
    .rn-model-val {
        font-family: var(--font-mono);
        font-size: 14px;
        font-weight: 700;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .rn-section { padding: 16px 16px; }
        .rn-navbar  { padding: 12px 16px; }
        .rn-price-hero { padding: 24px 16px 20px; }
        .rn-price-value { font-size: 32px; }
        .rn-metric-grid { grid-template-columns: repeat(2, 1fr); }
        .rn-footer { padding: 16px; }
        .stTabs [data-baseweb="tab-list"] { padding: 0 16px !important; }
    }
    </style>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# DATA FETCHING (cached)
# ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data():
    """Download OHLCV data for all tickers with robust error handling."""
    errors = []
    data = {}

    tickers = {
        "render": RENDER_TICKER,
        "btc": BTC_TICKER,
        "eth": ETH_TICKER,
        "sol": SOL_TICKER,
    }
    for key, ticker in tickers.items():
        try:
            df = yf.download(ticker, interval=INTERVAL, period=PERIOD,
                             auto_adjust=True, progress=False)
            if df.empty:
                errors.append(f"No data returned for {ticker}")
            else:
                data[key] = df
        except Exception as e:
            errors.append(f"{ticker}: {str(e)[:80]}")

    return data, errors


@st.cache_data(ttl=60, show_spinner=False)
def fetch_current_prices():
    """Fast current-price fetch for ticker tape."""
    prices = {}
    tickers = {"RNDR": RENDER_TICKER, "BTC": BTC_TICKER, "ETH": ETH_TICKER, "SOL": SOL_TICKER}
    for sym, t in tickers.items():
        try:
            tk = yf.Ticker(t)
            hist = tk.history(period="2d", interval="1h")
            if not hist.empty:
                latest = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest
                prices[sym] = {
                    "price": latest,
                    "change_pct": ((latest - prev) / prev) * 100 if prev else 0,
                }
        except Exception:
            prices[sym] = {"price": 0, "change_pct": 0}
    return prices


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate feature engineering from train.py."""
    try:
        close  = df["Close"].squeeze()
        high   = df["High"].squeeze()
        low    = df["Low"].squeeze()
        volume = df["Volume"].squeeze()

        df["log_return"] = np.log(close / close.shift(1))
        for n in [3, 6, 12, 24, 48]:
            df[f"return_{n}h"] = close.pct_change(n)

        for w, name in [(8,"EMA8"),(20,"EMA20"),(50,"EMA50"),(100,"EMA100"),(200,"EMA200")]:
            df[name] = ta.trend.ema_indicator(close, w)

        df["ema8_slope"]  = df["EMA8"].diff(3)
        df["ema20_slope"] = df["EMA20"].diff(3)
        df["price_vs_ema20"]  = (close - df["EMA20"]) / df["EMA20"]
        df["price_vs_ema50"]  = (close - df["EMA50"]) / df["EMA50"]
        df["price_vs_ema200"] = (close - df["EMA200"]) / df["EMA200"]
        df["ema20_vs_ema50"]  = (df["EMA20"] - df["EMA50"]) / df["EMA50"]
        df["ema50_vs_ema200"] = (df["EMA50"] - df["EMA200"]) / df["EMA200"]

        df["RSI14"] = ta.momentum.rsi(close, 14)
        df["RSI7"]  = ta.momentum.rsi(close, 7)
        df["RSI21"] = ta.momentum.rsi(close, 21)
        df["RSI14_slope"] = df["RSI14"].diff(3)

        stoch = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
        df["STOCH_K"] = stoch.stoch()
        df["STOCH_D"] = stoch.stoch_signal()

        df["ROC10"] = ta.momentum.roc(close, 10)
        df["ROC20"] = ta.momentum.roc(close, 20)

        macd = ta.trend.MACD(close, 26, 12, 9)
        df["MACD"]            = macd.macd()
        df["MACD_SIGNAL"]     = macd.macd_signal()
        df["MACD_HIST"]       = macd.macd_diff()
        df["MACD_HIST_slope"] = df["MACD_HIST"].diff(3)

        df["ADX"]     = ta.trend.adx(high, low, close, 14)
        df["ADX_POS"] = ta.trend.adx_pos(high, low, close, 14)
        df["ADX_NEG"] = ta.trend.adx_neg(high, low, close, 14)
        df["DI_DIFF"] = df["ADX_POS"] - df["ADX_NEG"]

        df["ATR14"]    = ta.volatility.average_true_range(high, low, close, 14)
        df["ATR_norm"] = df["ATR14"] / close

        bb = ta.volatility.BollingerBands(close, 20, 2)
        df["BB_UPPER"]    = bb.bollinger_hband()
        df["BB_LOWER"]    = bb.bollinger_lband()
        df["BB_WIDTH"]    = (df["BB_UPPER"] - df["BB_LOWER"]) / bb.bollinger_mavg()
        df["BB_POSITION"] = (close - df["BB_LOWER"]) / (df["BB_UPPER"] - df["BB_LOWER"] + 1e-9)

        df["KELT_UPPER"] = ta.volatility.keltner_channel_hband(high, low, close, 20)
        df["KELT_LOWER"] = ta.volatility.keltner_channel_lband(high, low, close, 20)
        df["KELT_POS"]   = (close - df["KELT_LOWER"]) / (df["KELT_UPPER"] - df["KELT_LOWER"] + 1e-9)

        df["SQUEEZE"] = (
            (df["BB_UPPER"] < df["KELT_UPPER"]) &
            (df["BB_LOWER"] > df["KELT_LOWER"])
        ).astype(int)

        df["RVOL_12h"] = df["log_return"].rolling(12).std()
        df["RVOL_24h"] = df["log_return"].rolling(24).std()

        df["VOL_SMA20"]  = ta.trend.sma_indicator(volume, 20)
        df["VOL_RATIO"]  = volume / (df["VOL_SMA20"] + 1e-9)
        df["VOL_CHANGE"] = volume.pct_change()

        df["OBV"]       = ta.volume.on_balance_volume(close, volume)
        df["OBV_slope"] = df["OBV"].diff(6)

        df["VWAP_24h"]      = (close * volume).rolling(24).sum() / (volume.rolling(24).sum() + 1e-9)
        df["price_vs_vwap"] = (close - df["VWAP_24h"]) / df["VWAP_24h"]

        df["high_24h"] = high.rolling(24).max()
        df["low_24h"]  = low.rolling(24).min()
        df["dist_to_resistance"] = (df["high_24h"] - close) / close
        df["dist_to_support"]    = (close - df["low_24h"]) / close

        for col, name in [("BTC_Close","BTC"),("ETH_Close","ETH"),("SOL_Close","SOL")]:
            if col in df.columns:
                ext_ret = np.log(df[col] / df[col].shift(1))
                df[f"{name}_return_1h"]  = ext_ret
                df[f"{name}_return_6h"]  = np.log(df[col] / df[col].shift(6))
                df[f"{name}_return_24h"] = np.log(df[col] / df[col].shift(24))
                df[f"RENDER_{name}_corr48"] = df["log_return"].rolling(48).corr(ext_ret)

        df["hour"]        = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df
    except Exception as e:
        raise RuntimeError(f"Feature engineering failed: {e}")


def load_models():
    """Load all saved model files. Returns (models_dict, missing_list)."""
    missing = []
    for key, path in MODEL_FILES.items():
        if not Path(path).exists():
            missing.append(path)

    if missing:
        return None, missing

    if not JOBLIB_AVAILABLE:
        return None, ["joblib package not installed"]
    if not TF_AVAILABLE:
        return None, ["tensorflow package not installed"]

    try:
        models = {
            "xgb_dir":      joblib.load(MODEL_FILES["xgb_direction"]),
            "xgb_reg":      joblib.load(MODEL_FILES["xgb_regressor"]),
            "lstm":         load_model(MODEL_FILES["lstm_model"]),
            "scaler_x":     joblib.load(MODEL_FILES["scaler_x"]),
            "scaler_y":     joblib.load(MODEL_FILES["scaler_y"]),
            "feature_cols": joblib.load(MODEL_FILES["feature_cols"]),
        }
        return models, []
    except Exception as e:
        return None, [str(e)]


def run_prediction(models: dict, df: pd.DataFrame) -> dict:
    """Run full ML inference and return results dict."""
    feature_cols = models["feature_cols"]
    scaler_x     = models["scaler_x"]
    scaler_y     = models["scaler_y"]

    X_latest  = df[feature_cols].values[-1:].astype(np.float32)
    X_scaled  = scaler_x.transform(X_latest)

    # XGBoost Direction
    dir_probs = models["xgb_dir"].predict_proba(X_scaled)[0]
    dir_class = int(np.argmax(dir_probs))
    dir_conf  = float(dir_probs[dir_class])

    # XGBoost Return
    xgb_return = float(models["xgb_reg"].predict(X_scaled)[0])

    # LSTM Return
    X_seq        = df[feature_cols].values[-LOOKBACK:].astype(np.float32)
    X_seq_scaled = scaler_x.transform(X_seq)
    X_seq_3d     = X_seq_scaled.reshape(1, LOOKBACK, -1)
    lstm_raw     = float(models["lstm"].predict(X_seq_3d, verbose=0)[0][0])
    lstm_return  = float(scaler_y.inverse_transform([[lstm_raw]])[0][0])

    # Ensemble
    ensemble_return = 0.60 * xgb_return + 0.40 * lstm_return
    current_price   = float(df["Close"].iloc[-1])
    predicted_price = current_price * (1 + ensemble_return)

    # Confidence
    sign_agrees = (
        (dir_class == 2 and ensemble_return > 0) or
        (dir_class == 0 and ensemble_return < 0) or
        (dir_class == 1 and abs(ensemble_return) < 0.005)
    )
    confidence = dir_conf * 100 * (1.15 if sign_agrees else 0.75)
    confidence = min(95.0, max(40.0, confidence))

    # ATR-based levels
    atr          = float(df["ATR14"].iloc[-1])
    stop_loss    = current_price - atr * 2
    take_profit1 = current_price + atr * 3
    take_profit2 = current_price + atr * 6
    risk_reward  = (take_profit1 - current_price) / (current_price - stop_loss + 1e-9)

    # Regime
    ema20 = float(df["EMA20"].iloc[-1])
    ema50 = float(df["EMA50"].iloc[-1])
    rsi   = float(df["RSI14"].iloc[-1])
    adx   = float(df["ADX"].iloc[-1])

    if ema20 > ema50 and rsi > 55:
        regime = "BULLISH"
    elif ema20 < ema50 and rsi < 45:
        regime = "BEARISH"
    else:
        regime = "SIDEWAYS"

    # Correlations
    btc_corr = float(df.get("RENDER_BTC_corr48", pd.Series([0])).iloc[-1]) if "RENDER_BTC_corr48" in df.columns else 0.0
    eth_corr = float(df.get("RENDER_ETH_corr48", pd.Series([0])).iloc[-1]) if "RENDER_ETH_corr48" in df.columns else 0.0

    return {
        "current_price":   current_price,
        "predicted_price": predicted_price,
        "predicted_pct":   ensemble_return * 100,
        "dir_class":       dir_class,
        "dir_probs":       dir_probs,
        "confidence":      confidence,
        "sign_agrees":     sign_agrees,
        "xgb_return":      xgb_return * 100,
        "lstm_return":     lstm_return * 100,
        "ensemble_return": ensemble_return * 100,
        "stop_loss":       stop_loss,
        "take_profit1":    take_profit1,
        "take_profit2":    take_profit2,
        "risk_reward":     risk_reward,
        "rsi":             rsi,
        "adx":             adx,
        "ema20":           ema20,
        "ema50":           ema50,
        "atr":             atr,
        "regime":          regime,
        "squeeze":         bool(df["SQUEEZE"].iloc[-1]),
        "btc_corr":        btc_corr,
        "eth_corr":        eth_corr,
        "bb_position":     float(df["BB_POSITION"].iloc[-1]),
        "stoch_k":         float(df["STOCH_K"].iloc[-1]),
        "macd_hist":       float(df["MACD_HIST"].iloc[-1]),
        "vol_ratio":       float(df["VOL_RATIO"].iloc[-1]),
        "rvol_24h":        float(df["RVOL_24h"].iloc[-1]),
    }

# ──────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Mono, monospace", color="#94a3b8", size=10),
    margin=dict(l=0, r=0, t=8, b=0),
    showlegend=True,
    legend=dict(
        bgcolor="rgba(13,21,38,0.8)",
        bordercolor="#1e3054",
        borderwidth=1,
        font=dict(size=9),
    ),
    xaxis=dict(
        gridcolor="#1e3054",
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=9),
    ),
    yaxis=dict(
        gridcolor="#1e3054",
        showgrid=True,
        zeroline=False,
        tickfont=dict(size=9),
    ),
)


def build_candlestick_chart(df: pd.DataFrame, pred: dict | None = None) -> go.Figure:
    """Full OHLCV candlestick with EMAs, Volume, and optional prediction line."""
    close  = df["Close"].squeeze()
    recent = df.tail(168)  # last 7 days at 1h

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.60, 0.20, 0.20],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent["Open"].squeeze(),
        high=recent["High"].squeeze(),
        low=recent["Low"].squeeze(),
        close=recent["Close"].squeeze(),
        name="RNDR",
        increasing_line_color="#22c55e",
        decreasing_line_color="#ef4444",
        increasing_fillcolor="#22c55e",
        decreasing_fillcolor="#ef4444",
        line=dict(width=1),
    ), row=1, col=1)

    # EMAs
    for col, color, name in [
        ("EMA8",  "#06b6d4", "EMA 8"),
        ("EMA20", "#3b82f6", "EMA 20"),
        ("EMA50", "#a855f7", "EMA 50"),
    ]:
        if col in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent.index,
                y=recent[col].squeeze(),
                line=dict(color=color, width=1.2),
                name=name,
                opacity=0.85,
            ), row=1, col=1)

    # Bollinger Bands
    if "BB_UPPER" in recent.columns and "BB_LOWER" in recent.columns:
        fig.add_trace(go.Scatter(
            x=list(recent.index) + list(recent.index[::-1]),
            y=list(recent["BB_UPPER"].squeeze()) + list(recent["BB_LOWER"].squeeze()[::-1]),
            fill="toself",
            fillcolor="rgba(59,130,246,0.04)",
            line=dict(color="rgba(59,130,246,0.2)", width=0.5),
            name="BB",
            showlegend=True,
        ), row=1, col=1)

    # Prediction line
    if pred:
        last_time = recent.index[-1]
        pred_time = last_time + timedelta(hours=6)
        fig.add_trace(go.Scatter(
            x=[last_time, pred_time],
            y=[pred["current_price"], pred["predicted_price"]],
            mode="lines+markers",
            line=dict(
                color="#22c55e" if pred["predicted_pct"] >= 0 else "#ef4444",
                width=2, dash="dot"
            ),
            marker=dict(size=[0, 8]),
            name="6h Prediction",
        ), row=1, col=1)

    # RSI
    if "RSI14" in recent.columns:
        rsi_vals = recent["RSI14"].squeeze()
        fig.add_trace(go.Scatter(
            x=recent.index, y=rsi_vals,
            line=dict(color="#f59e0b", width=1.5),
            name="RSI 14",
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.05)",
        ), row=2, col=1)
        for lvl, clr in [(70, "#ef4444"), (30, "#22c55e"), (50, "#475569")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr,
                          line_width=0.8, row=2, col=1)

    # Volume
    vol = recent["Volume"].squeeze()
    vol_colors = ["#22c55e" if c >= o else "#ef4444"
                  for c, o in zip(recent["Close"].squeeze(), recent["Open"].squeeze())]
    fig.add_trace(go.Bar(
        x=recent.index, y=vol,
        marker_color=vol_colors,
        marker_opacity=0.6,
        name="Volume",
        showlegend=False,
    ), row=3, col=1)

    fig.update_layout(**CHART_LAYOUT, height=520)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(row=2, range=[0, 100])
    return fig


def build_correlation_chart(df: pd.DataFrame, btc_df: pd.DataFrame,
                            eth_df: pd.DataFrame) -> go.Figure:
    """Rolling 48h correlation of RNDR vs BTC/ETH."""
    close = df["Close"].squeeze()
    log_r = np.log(close / close.shift(1))

    fig = go.Figure()
    for ext_df, name, color in [
        (btc_df, "BTC", "#f59e0b"),
        (eth_df, "ETH", "#6366f1"),
    ]:
        try:
            ext_close = ext_df["Close"].squeeze().reindex(df.index, method="nearest")
            ext_ret   = np.log(ext_close / ext_close.shift(1))
            corr      = log_r.rolling(48).corr(ext_ret).dropna()
            fig.add_trace(go.Scatter(
                x=corr.index, y=corr,
                name=f"RNDR/{name} 48h Corr",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.07,)}",
            ))
        except Exception:
            pass

    fig.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=0.8)
    fig.update_layout(**CHART_LAYOUT, height=240, title=dict(text="", x=0))
    return fig


def build_probability_gauge(probs: list) -> go.Figure:
    """Donut / polar chart for direction probabilities."""
    labels = ["DOWN", "FLAT", "UP"]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]
    fig = go.Figure(go.Pie(
        values=probs,
        labels=labels,
        hole=0.7,
        marker=dict(colors=colors, line=dict(color="#060b14", width=2)),
        textinfo="none",
        hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        height=200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        annotations=[dict(
            text=f"{max(probs)*100:.0f}%",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=22, color="#e2e8f0", family="Orbitron"),
        )],
        margin=dict(l=0, r=0, t=0, b=30),
    )
    return fig


def build_return_history_chart(df: pd.DataFrame) -> go.Figure:
    """6h rolling returns histogram."""
    close  = df["Close"].squeeze()
    ret_6h = close.pct_change(6).dropna() * 100

    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in ret_6h]
    fig = go.Figure(go.Histogram(
        x=ret_6h,
        nbinsx=60,
        marker=dict(
            color=ret_6h.apply(lambda v: "#22c55e" if v >= 0 else "#ef4444"),
            line=dict(width=0),
        ),
        opacity=0.8,
        name="6h Return Distribution",
    ))
    fig.update_layout(**CHART_LAYOUT, height=240,
                      xaxis_title="6h Return (%)",
                      yaxis_title="Frequency",
                      bargap=0.02)
    return fig


# ──────────────────────────────────────────────────────────
# UI COMPONENTS
# ──────────────────────────────────────────────────────────
def render_navbar(prices: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    rndr_price  = prices.get("RNDR", {}).get("price", 0)
    rndr_change = prices.get("RNDR", {}).get("change_pct", 0)

    tape_items = ""
    for sym, data in prices.items():
        p = data.get("price", 0)
        c = data.get("change_pct", 0)
        cls = "rn-tape-up" if c >= 0 else "rn-tape-down"
        sign = "+" if c >= 0 else ""
        tape_items += f"""
        <span class="rn-tape-item">
            <span style="color:#475569">{sym}</span>
            <span class="rn-tape-price">${p:,.4f}</span>
            <span class="{cls}">{sign}{c:.2f}%</span>
        </span>"""

    # Duplicate for seamless scroll
    tape_all = tape_items * 4

    st.markdown(f"""
    <div class="rn-navbar">
        <div class="rn-brand">⚡ RENDER ML</div>
        <div class="rn-live-badge">
            <div class="rn-live-dot"></div>LIVE
        </div>
        <div class="rn-nav-time">{now}</div>
    </div>
    <div class="rn-tape-wrapper">
        <div class="rn-tape-inner">{tape_all}</div>
    </div>
    """, unsafe_allow_html=True)


def render_price_hero(price: float, pred_price: float, pred_pct: float, df: pd.DataFrame):
    change_24h = float(df["Close"].squeeze().pct_change(24).iloc[-1]) * 100
    high_24h   = float(df["High"].squeeze().rolling(24).max().iloc[-1])
    low_24h    = float(df["Low"].squeeze().rolling(24).min().iloc[-1])
    vol_24h    = float(df["Volume"].squeeze().rolling(24).sum().iloc[-1])

    change_cls  = "up" if change_24h >= 0 else "down"
    change_sign = "+" if change_24h >= 0 else ""
    pred_cls    = "up" if pred_pct >= 0 else "down"
    pred_sign   = "+" if pred_pct >= 0 else ""

    vol_str = f"{vol_24h/1e6:.2f}M" if vol_24h >= 1e6 else f"{vol_24h/1e3:.1f}K"

    st.markdown(f"""
    <div class="rn-price-hero">
        <div class="rn-price-ticker">RENDER · RNDR/USD · 1H</div>
        <div class="rn-price-value">${price:,.4f}</div>
        <span class="rn-price-change {change_cls}">
            {change_sign}{change_24h:.2f}%  24h
        </span>
        &nbsp;&nbsp;
        <span class="rn-price-change {pred_cls}" style="opacity:0.85">
            {pred_sign}{pred_pct:.2f}%  6h pred
        </span>
        <div style="margin-top:20px">
            <div class="rn-metric-grid">
                <div class="rn-metric-card">
                    <div class="rn-metric-label">Predicted Price</div>
                    <div class="rn-metric-value" style="color:{'#22c55e' if pred_pct >= 0 else '#ef4444'}">${pred_price:,.4f}</div>
                    <div class="rn-metric-sub">6h horizon</div>
                </div>
                <div class="rn-metric-card">
                    <div class="rn-metric-label">24h High</div>
                    <div class="rn-metric-value">${high_24h:,.4f}</div>
                    <div class="rn-metric-sub">rolling window</div>
                </div>
                <div class="rn-metric-card">
                    <div class="rn-metric-label">24h Low</div>
                    <div class="rn-metric-value">${low_24h:,.4f}</div>
                    <div class="rn-metric-sub">rolling window</div>
                </div>
                <div class="rn-metric-card">
                    <div class="rn-metric-label">24h Volume</div>
                    <div class="rn-metric-value">{vol_str}</div>
                    <div class="rn-metric-sub">RNDR</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_signal_block(pred: dict):
    dc   = pred["dir_class"]
    col  = DIRECTION_COLORS[dc]
    lbl  = DIRECTION_LABELS[dc]
    icon = DIRECTION_ICONS[dc]
    conf = pred["confidence"]
    probs = pred["dir_probs"]
    agree = pred["sign_agrees"]

    # Regime badge class
    regime_map = {"BULLISH": "bullish", "BEARISH": "bearish", "SIDEWAYS": "sideways"}
    regime_cls = regime_map.get(pred["regime"], "sideways")
    regime_icons = {"BULLISH": "🐂", "BEARISH": "🐻", "SIDEWAYS": "↔️"}
    regime_icon = regime_icons.get(pred["regime"], "")

    agree_html = (
        '<span style="color:#22c55e;font-weight:700">✅ MODELS AGREE</span>'
        if agree else
        '<span style="color:#f59e0b;font-weight:700">⚠️ SPLIT SIGNAL</span>'
    )

    # Squeeze alert
    squeeze_html = ""
    if pred["squeeze"]:
        squeeze_html = """
        <div class="rn-squeeze-alert" style="margin-top:16px">
            <div class="rn-squeeze-icon">🔥</div>
            <div>
                <div class="rn-squeeze-text">VOLATILITY SQUEEZE ACTIVE</div>
                <div style="font-family:var(--font-sans);font-size:12px;color:#94a3b8;margin-top:2px">
                    Bollinger inside Keltner — breakout imminent
                </div>
            </div>
        </div>"""

    st.markdown(f"""
    <div class="rn-signal-card">
        <div class="rn-section-title">ML DIRECTION SIGNAL</div>
        <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;margin-bottom:20px">
            <div>
                <div class="rn-signal-icon" style="color:{col}">{icon}</div>
                <div class="rn-signal-direction" style="color:{col}">{lbl}</div>
                <div style="font-family:var(--font-mono);font-size:11px;color:#94a3b8;margin-top:4px">
                    {agree_html}
                </div>
            </div>
            <div style="flex:1;min-width:200px">
                <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:6px">Confidence</div>
                <div style="font-family:'Orbitron',sans-serif;font-size:36px;font-weight:900;color:{col}">{conf:.1f}<span style="font-size:18px">%</span></div>
            </div>
            <div>
                <div class="rn-regime-badge {regime_cls}">{regime_icon} {pred['regime']}</div>
            </div>
        </div>

        <div class="rn-section-title" style="margin-top:4px">DIRECTION PROBABILITIES</div>
        <div class="rn-prob-row">
            <span class="rn-prob-label" style="color:#22c55e">UP</span>
            <div class="rn-prob-bar-bg"><div class="rn-prob-bar-fill" style="width:{probs[2]*100:.1f}%;background:linear-gradient(90deg,#16a34a,#22c55e)"></div></div>
            <span class="rn-prob-pct" style="color:#22c55e">{probs[2]*100:.1f}%</span>
        </div>
        <div class="rn-prob-row">
            <span class="rn-prob-label" style="color:#f59e0b">FLAT</span>
            <div class="rn-prob-bar-bg"><div class="rn-prob-bar-fill" style="width:{probs[1]*100:.1f}%;background:linear-gradient(90deg,#b45309,#f59e0b)"></div></div>
            <span class="rn-prob-pct" style="color:#f59e0b">{probs[1]*100:.1f}%</span>
        </div>
        <div class="rn-prob-row">
            <span class="rn-prob-label" style="color:#ef4444">DOWN</span>
            <div class="rn-prob-bar-bg"><div class="rn-prob-bar-fill" style="width:{probs[0]*100:.1f}%;background:linear-gradient(90deg,#b91c1c,#ef4444)"></div></div>
            <span class="rn-prob-pct" style="color:#ef4444">{probs[0]*100:.1f}%</span>
        </div>

        {squeeze_html}
    </div>
    """, unsafe_allow_html=True)


def render_trade_levels(pred: dict):
    cp  = pred["current_price"]
    sl  = pred["stop_loss"]
    tp1 = pred["take_profit1"]
    tp2 = pred["take_profit2"]
    rr  = pred["risk_reward"]

    sl_pct  = abs((cp - sl) / cp) * 100
    tp1_pct = abs((tp1 - cp) / cp) * 100
    tp2_pct = abs((tp2 - cp) / cp) * 100

    st.markdown(f"""
    <div>
        <div class="rn-section-title">TRADE LEVELS (ATR-BASED)</div>
        <div class="rn-levels-card">
            <div class="rn-level-row">
                <span class="rn-level-name" style="color:#94a3b8">Entry</span>
                <span class="rn-level-price">${cp:,.4f}</span>
                <span class="rn-level-pct" style="background:#3b82f618;color:#3b82f6">Current</span>
            </div>
            <div class="rn-level-row">
                <span class="rn-level-name" style="color:#22c55e">TP1</span>
                <span class="rn-level-price" style="color:#22c55e">${tp1:,.4f}</span>
                <span class="rn-level-pct" style="background:#22c55e18;color:#22c55e">+{tp1_pct:.2f}%</span>
            </div>
            <div class="rn-level-row">
                <span class="rn-level-name" style="color:#4ade80">TP2</span>
                <span class="rn-level-price" style="color:#4ade80">${tp2:,.4f}</span>
                <span class="rn-level-pct" style="background:#22c55e18;color:#4ade80">+{tp2_pct:.2f}%</span>
            </div>
            <div class="rn-level-row">
                <span class="rn-level-name" style="color:#ef4444">Stop Loss</span>
                <span class="rn-level-price" style="color:#ef4444">${sl:,.4f}</span>
                <span class="rn-level-pct" style="background:#ef444418;color:#ef4444">-{sl_pct:.2f}%</span>
            </div>
            <div class="rn-level-row" style="background:rgba(59,130,246,0.04)">
                <span class="rn-level-name" style="color:#3b82f6">Risk / Reward</span>
                <span class="rn-level-price" style="color:#3b82f6">{rr:.2f}</span>
                <span class="rn-level-pct" style="background:#3b82f618;color:#3b82f6">
                    {'✅ Good' if rr >= 2 else '⚠️ Low'}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_indicators(pred: dict):
    rsi  = pred["rsi"]
    adx  = pred["adx"]
    stk  = pred["stoch_k"]
    bbp  = pred["bb_position"] * 100
    macd = pred["macd_hist"]
    volr = pred["vol_ratio"]
    rvol = pred["rvol_24h"] * 100

    def rsi_badge(v):
        if v > 70: return ("OVERBOUGHT", "#ef4444")
        if v < 30: return ("OVERSOLD", "#22c55e")
        return ("NEUTRAL", "#f59e0b")

    def adx_badge(v):
        if v > 40: return ("STRONG TREND", "#22c55e")
        if v > 25: return ("TRENDING", "#3b82f6")
        return ("CHOPPY", "#f59e0b")

    rsi_lbl, rsi_col = rsi_badge(rsi)
    adx_lbl, adx_col = adx_badge(adx)
    macd_col = "#22c55e" if macd > 0 else "#ef4444"
    macd_lbl = "BULLISH" if macd > 0 else "BEARISH"

    st.markdown(f"""
    <div class="rn-section-title" style="margin-top:20px">TECHNICAL INDICATORS</div>
    <div class="rn-indicator-grid">
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">RSI 14</div>
            <div class="rn-indicator-val" style="color:{rsi_col}">{rsi:.1f}</div>
            <div class="rn-indicator-badge" style="background:{rsi_col}20;color:{rsi_col}">{rsi_lbl}</div>
        </div>
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">ADX 14</div>
            <div class="rn-indicator-val" style="color:{adx_col}">{adx:.1f}</div>
            <div class="rn-indicator-badge" style="background:{adx_col}20;color:{adx_col}">{adx_lbl}</div>
        </div>
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">STOCH %K</div>
            <div class="rn-indicator-val" style="color:#a855f7">{stk:.1f}</div>
            <div class="rn-indicator-badge" style="background:#a855f720;color:#a855f7">
                {'O/B' if stk > 80 else 'O/S' if stk < 20 else 'MID'}
            </div>
        </div>
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">BB Position</div>
            <div class="rn-indicator-val" style="color:#06b6d4">{bbp:.1f}%</div>
            <div class="rn-indicator-badge" style="background:#06b6d420;color:#06b6d4">
                {'UPPER' if bbp > 75 else 'LOWER' if bbp < 25 else 'MID'}
            </div>
        </div>
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">MACD Hist</div>
            <div class="rn-indicator-val" style="color:{macd_col}">{macd:.4f}</div>
            <div class="rn-indicator-badge" style="background:{macd_col}20;color:{macd_col}">{macd_lbl}</div>
        </div>
        <div class="rn-indicator-pill">
            <div class="rn-indicator-name">Vol Ratio</div>
            <div class="rn-indicator-val" style="color:#f59e0b">{volr:.2f}x</div>
            <div class="rn-indicator-badge" style="background:#f59e0b20;color:#f59e0b">
                {'HIGH VOL' if volr > 1.5 else 'LOW VOL' if volr < 0.7 else 'NORMAL'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_model_ensemble(pred: dict):
    xgb_r  = pred["xgb_return"]
    lstm_r = pred["lstm_return"]
    ens_r  = pred["ensemble_return"]

    def clr(v): return "#22c55e" if v >= 0 else "#ef4444"
    def fmt(v): return f"{'+' if v >= 0 else ''}{v:.3f}%"

    st.markdown(f"""
    <div>
        <div class="rn-section-title">MODEL ENSEMBLE · 6H RETURN PREDICTION</div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md);padding:4px 20px">
            <div class="rn-model-row">
                <div>
                    <div class="rn-model-name">XGBoost Regressor</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;font-family:var(--font-sans)">Weight: 60%  ·  Gradient-boosted trees</div>
                </div>
                <div class="rn-model-val" style="color:{clr(xgb_r)}">{fmt(xgb_r)}</div>
            </div>
            <div class="rn-model-row">
                <div>
                    <div class="rn-model-name">LSTM Neural Network</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;font-family:var(--font-sans)">Weight: 40%  ·  48-step sequence</div>
                </div>
                <div class="rn-model-val" style="color:{clr(lstm_r)}">{fmt(lstm_r)}</div>
            </div>
            <div class="rn-model-row" style="background:rgba(59,130,246,0.04);margin:0 -20px;padding:12px 20px;border-radius:0 0 var(--radius-md) var(--radius-md)">
                <div>
                    <div class="rn-model-name" style="color:#3b82f6">Ensemble (Weighted)</div>
                    <div style="font-size:10px;color:var(--text-muted);margin-top:2px;font-family:var(--font-sans)">Final prediction signal</div>
                </div>
                <div class="rn-model-val" style="color:{clr(ens_r)};font-size:18px">{fmt(ens_r)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_correlations(pred: dict):
    btc_c = pred.get("btc_corr", 0)
    eth_c = pred.get("eth_corr", 0)

    def corr_color(v):
        if v > 0.6: return "#22c55e"
        if v > 0.3: return "#06b6d4"
        if v < -0.3: return "#ef4444"
        return "#f59e0b"

    def pct_pos(v, max_=1.0):
        """Normalize to 0-100 for bar width, centered at 50."""
        return 50 + (v / max_) * 50

    btc_col = corr_color(btc_c)
    eth_col = corr_color(eth_c)

    st.markdown(f"""
    <div>
        <div class="rn-section-title">MARKET CORRELATIONS · 48H ROLLING</div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-md);padding:16px 20px">
            <div class="rn-corr-bar">
                <span class="rn-corr-label" style="color:#f59e0b">BTC</span>
                <div class="rn-corr-track">
                    <div class="rn-corr-fill" style="background:{btc_col};left:50%;width:{abs(btc_c)*50:.1f}%;{'right:50%;left:auto' if btc_c < 0 else ''}"></div>
                </div>
                <span class="rn-corr-val" style="color:{btc_col}">{btc_c:+.2f}</span>
            </div>
            <div class="rn-corr-bar">
                <span class="rn-corr-label" style="color:#6366f1">ETH</span>
                <div class="rn-corr-track">
                    <div class="rn-corr-fill" style="background:{eth_col};left:50%;width:{abs(eth_c)*50:.1f}%"></div>
                </div>
                <span class="rn-corr-val" style="color:{eth_col}">{eth_c:+.2f}</span>
            </div>
            <div style="font-family:var(--font-mono);font-size:9px;color:var(--text-muted);letter-spacing:1px;margin-top:8px;text-align:center">
                CORRELATION SCALE  −1 ←────────────── 0 ──────────────→ +1
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_error(title: str, body: str):
    st.markdown(f"""
    <div class="rn-error-card">
        <div class="rn-error-icon">⚠️</div>
        <div>
            <div class="rn-error-title">{title}</div>
            <div class="rn-error-body">{body}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_info(msg: str):
    st.markdown(f"""
    <div class="rn-info-card">
        <span>ℹ️</span> {msg}
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    st.markdown(f"""
    <div class="rn-footer">
        <div class="rn-footer-text">⚡ RENDER ML PREDICTION ENGINE  ·  POWERED BY XGBOOST + LSTM</div>
        <div class="rn-footer-text">{now}</div>
        <div class="rn-footer-text" style="color:#334155">⚠️ NOT FINANCIAL ADVICE · FOR RESEARCH PURPOSES ONLY</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# DEMO MODE — realistic simulated predictions
# ──────────────────────────────────────────────────────────
def make_demo_prediction(df: pd.DataFrame) -> dict:
    """Generate realistic demo signals when models aren't loaded."""
    import random
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    volume = df["Volume"].squeeze()

    rsi14 = float(ta.momentum.rsi(close, 14).iloc[-1])
    adx14 = float(ta.trend.adx(high, low, close, 14).iloc[-1])
    ema20 = float(ta.trend.ema_indicator(close, 20).iloc[-1])
    ema50 = float(ta.trend.ema_indicator(close, 50).iloc[-1])
    atr14 = float(ta.volatility.average_true_range(high, low, close, 14).iloc[-1])
    bb    = ta.volatility.BollingerBands(close, 20, 2)
    bb_pos= float(((close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)).iloc[-1])
    stoch = ta.momentum.StochasticOscillator(high, low, close, 14, 3)
    stk   = float(stoch.stoch().iloc[-1])
    macd  = ta.trend.MACD(close, 26, 12, 9)
    mhist = float(macd.macd_diff().iloc[-1])
    kelt_u= float(ta.volatility.keltner_channel_hband(high, low, close, 20).iloc[-1])
    kelt_l= float(ta.volatility.keltner_channel_lband(high, low, close, 20).iloc[-1])
    squeeze = float(bb.bollinger_hband().iloc[-1]) < kelt_u and float(bb.bollinger_lband().iloc[-1]) > kelt_l
    vol_sma = float(ta.trend.sma_indicator(volume, 20).iloc[-1])
    vol_r   = float(volume.iloc[-1]) / (vol_sma + 1e-9)
    log_r   = np.log(close / close.shift(1))
    rvol    = float(log_r.rolling(24).std().iloc[-1])

    current_price = float(close.iloc[-1])

    # Derive direction from indicators
    score = 0
    score += 1 if rsi14 > 55 else (-1 if rsi14 < 45 else 0)
    score += 1 if ema20 > ema50 else -1
    score += 1 if mhist > 0 else -1
    score += 0.5 if bb_pos > 0.6 else (-0.5 if bb_pos < 0.4 else 0)

    if score > 1:
        dir_class = 2  # UP
        probs = [max(0.05, 0.5 - score*0.08), max(0.05, 0.2 - score*0.02), min(0.90, 0.3 + score*0.10)]
    elif score < -1:
        dir_class = 0  # DOWN
        probs = [min(0.90, 0.3 + abs(score)*0.10), max(0.05, 0.2), max(0.05, 0.5 - abs(score)*0.08)]
    else:
        dir_class = 1  # FLAT
        probs = [0.28, 0.44, 0.28]

    # Normalize
    total = sum(probs)
    probs = [p/total for p in probs]

    xgb_return = (score * 0.0015) + random.gauss(0, 0.001)
    lstm_return = xgb_return + random.gauss(0, 0.0008)
    ensemble_return = 0.60 * xgb_return + 0.40 * lstm_return

    dir_conf = max(probs)
    sign_agrees = (
        (dir_class == 2 and ensemble_return > 0) or
        (dir_class == 0 and ensemble_return < 0) or
        (dir_class == 1 and abs(ensemble_return) < 0.005)
    )
    confidence = dir_conf * 100 * (1.15 if sign_agrees else 0.75)
    confidence = min(95.0, max(40.0, confidence))

    predicted_price = current_price * (1 + ensemble_return)

    stop_loss    = current_price - atr14 * 2
    take_profit1 = current_price + atr14 * 3
    take_profit2 = current_price + atr14 * 6
    risk_reward  = (take_profit1 - current_price) / (current_price - stop_loss + 1e-9)

    if ema20 > ema50 and rsi14 > 55:
        regime = "BULLISH"
    elif ema20 < ema50 and rsi14 < 45:
        regime = "BEARISH"
    else:
        regime = "SIDEWAYS"

    return {
        "current_price":   current_price,
        "predicted_price": predicted_price,
        "predicted_pct":   ensemble_return * 100,
        "dir_class":       dir_class,
        "dir_probs":       probs,
        "confidence":      confidence,
        "sign_agrees":     sign_agrees,
        "xgb_return":      xgb_return * 100,
        "lstm_return":     lstm_return * 100,
        "ensemble_return": ensemble_return * 100,
        "stop_loss":       stop_loss,
        "take_profit1":    take_profit1,
        "take_profit2":    take_profit2,
        "risk_reward":     risk_reward,
        "rsi":             rsi14,
        "adx":             adx14,
        "ema20":           ema20,
        "ema50":           ema50,
        "atr":             atr14,
        "regime":          regime,
        "squeeze":         squeeze,
        "btc_corr":        0.0,
        "eth_corr":        0.0,
        "bb_position":     bb_pos,
        "stoch_k":         stk,
        "macd_hist":       mhist,
        "vol_ratio":       vol_r,
        "rvol_24h":        rvol,
    }


# ──────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Session state ──
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None

    # ── Ticker tape prices (fast) ──
    with st.spinner(""):
        try:
            prices = fetch_current_prices()
        except Exception:
            prices = {}

    render_navbar(prices)

    # ── Control bar ──
    st.markdown('<div class="rn-section-sm" style="border-bottom:1px solid #1e3054">', unsafe_allow_html=True)
    ctrl_cols = st.columns([3, 1, 1, 1])
    with ctrl_cols[0]:
        st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:11px;color:#475569;letter-spacing:2px;padding-top:8px">RENDER (RNDR) · MACHINE LEARNING PREDICTION DASHBOARD · v2.0</div>', unsafe_allow_html=True)
    with ctrl_cols[2]:
        demo_mode = st.checkbox("Demo Mode", value=True,
            help="Use indicator-derived signals without trained model files")
    with ctrl_cols[3]:
        refresh = st.button("⟳  Refresh Data")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Load data ──
    if refresh:
        st.cache_data.clear()

    with st.spinner("Fetching market data…"):
        market_data, data_errors = fetch_market_data()

    if data_errors:
        for err in data_errors:
            render_error("Data Fetch Warning", err)

    if "render" not in market_data or market_data["render"].empty:
        render_error("No Market Data", "Could not fetch RENDER (RNDR) OHLCV data. Check your connection and try again.")
        render_footer()
        return

    # ── Build main dataframe ──
    try:
        df = market_data["render"].copy()
        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

        for key, col_name in [("btc", "BTC_Close"), ("eth", "ETH_Close"), ("sol", "SOL_Close")]:
            if key in market_data and not market_data[key].empty:
                ext = market_data[key]["Close"].squeeze()
                ext.name = col_name
                df = df.join(ext, how="left")

        df = df.ffill().dropna()
        df = engineer_features(df)
    except Exception as e:
        render_error("Feature Engineering Failed",
                     f"Could not compute technical indicators. Error: {str(e)[:200]}")
        render_footer()
        return

    # ── Run prediction ──
    pred = None
    model_status = "DEMO"

    if not demo_mode:
        models, missing = load_models()
        if missing:
            render_info(f"Model files not found ({', '.join(missing)}). Switching to Demo Mode.")
            demo_mode = True
        else:
            try:
                pred = run_prediction(models, df)
                model_status = "ML"
            except Exception as e:
                render_error("Model Inference Error",
                             f"Prediction failed: {str(e)[:200]}")
                demo_mode = True

    if demo_mode or pred is None:
        try:
            pred = make_demo_prediction(df)
            model_status = "DEMO"
        except Exception as e:
            render_error("Demo Mode Error", f"Could not generate demo prediction: {str(e)[:200]}")
            render_footer()
            return

    # ── Mode badge ──
    mode_html = (
        '<span style="background:#22c55e18;border:1px solid #22c55e40;color:#22c55e;font-family:\'Space Mono\',monospace;font-size:9px;font-weight:700;letter-spacing:2px;padding:3px 10px;border-radius:4px">● ML MODEL ACTIVE</span>'
        if model_status == "ML" else
        '<span style="background:#f59e0b18;border:1px solid #f59e0b40;color:#f59e0b;font-family:\'Space Mono\',monospace;font-size:9px;font-weight:700;letter-spacing:2px;padding:3px 10px;border-radius:4px">◎ DEMO MODE · INDICATOR-DERIVED</span>'
    )
    st.markdown(f'<div style="padding:8px 32px;border-bottom:1px solid #1e3054">{mode_html}</div>', unsafe_allow_html=True)

    # ── Price Hero ──
    render_price_hero(
        pred["current_price"],
        pred["predicted_price"],
        pred["predicted_pct"],
        df,
    )

    # ── TABS ──
    tab_overview, tab_chart, tab_analysis, tab_about = st.tabs([
        "  OVERVIEW  ",
        "  CHART  ",
        "  DEEP ANALYSIS  ",
        "  ABOUT  ",
    ])

    # ── TAB: OVERVIEW ──
    with tab_overview:
        st.markdown('<div class="rn-section">', unsafe_allow_html=True)

        left_col, right_col = st.columns([1.1, 0.9], gap="medium")

        with left_col:
            render_signal_block(pred)
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            render_model_ensemble(pred)
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            render_indicators(pred)

        with right_col:
            render_trade_levels(pred)
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            render_correlations(pred)

            # Probability donut
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="rn-section-title">PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
            fig_donut = build_probability_gauge(list(pred["dir_probs"]))
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB: CHART ──
    with tab_chart:
        st.markdown('<div class="rn-section">', unsafe_allow_html=True)
        st.markdown('<div class="rn-section-title">PRICE ACTION · RNDR/USD · 1H (LAST 7D)</div>', unsafe_allow_html=True)

        try:
            fig_candle = build_candlestick_chart(df, pred)
            st.plotly_chart(fig_candle, use_container_width=True,
                            config={"displayModeBar": True, "displaylogo": False,
                                    "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
        except Exception as e:
            render_error("Chart Error", f"Could not render candlestick chart: {str(e)[:150]}")

        st.markdown('<div class="rn-section-title" style="margin-top:24px">ROLLING CORRELATIONS (48H) · RNDR vs BTC/ETH</div>', unsafe_allow_html=True)
        try:
            if "btc" in market_data and "eth" in market_data:
                fig_corr = build_correlation_chart(df, market_data["btc"], market_data["eth"])
                st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})
            else:
                render_info("BTC/ETH data unavailable — correlation chart skipped.")
        except Exception as e:
            render_error("Correlation Chart Error", str(e)[:150])

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB: DEEP ANALYSIS ──
    with tab_analysis:
        st.markdown('<div class="rn-section">', unsafe_allow_html=True)

        a1, a2 = st.columns(2, gap="medium")

        with a1:
            st.markdown('<div class="rn-section-title">6H RETURN DISTRIBUTION (60D)</div>', unsafe_allow_html=True)
            try:
                fig_hist = build_return_history_chart(df)
                st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                render_error("Chart Error", str(e)[:100])

            # ATR heatmap over time
            st.markdown('<div class="rn-section-title" style="margin-top:24px">ATR VOLATILITY OVER TIME</div>', unsafe_allow_html=True)
            try:
                atr_s = df["ATR14"].squeeze().tail(168)
                atr_norm = df["ATR_norm"].squeeze().tail(168) * 100
                fig_atr = go.Figure()
                fig_atr.add_trace(go.Scatter(
                    x=atr_s.index, y=atr_norm,
                    name="ATR Norm %",
                    fill="tozeroy",
                    line=dict(color="#a855f7", width=1.5),
                    fillcolor="rgba(168,85,247,0.08)",
                ))
                fig_atr.update_layout(**CHART_LAYOUT, height=200)
                st.plotly_chart(fig_atr, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                render_error("ATR Chart Error", str(e)[:100])

        with a2:
            st.markdown('<div class="rn-section-title">VOLUME PROFILE (LAST 7D)</div>', unsafe_allow_html=True)
            try:
                vol_s = df["Volume"].squeeze().tail(168)
                vol_r = df["VOL_RATIO"].squeeze().tail(168)
                fig_vol = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                        row_heights=[0.5, 0.5], vertical_spacing=0.06)
                close_s = df["Close"].squeeze().tail(168)
                colors = ["#22c55e" if c >= o else "#ef4444"
                          for c, o in zip(close_s, df["Open"].squeeze().tail(168))]
                fig_vol.add_trace(go.Bar(x=vol_s.index, y=vol_s,
                                         marker_color=colors, marker_opacity=0.7,
                                         name="Volume"), row=1, col=1)
                fig_vol.add_trace(go.Scatter(x=vol_r.index, y=vol_r,
                                              line=dict(color="#06b6d4", width=1.5),
                                              name="Vol Ratio",
                                              fill="tozeroy",
                                              fillcolor="rgba(6,182,212,0.07)"),
                                   row=2, col=1)
                fig_vol.add_hline(y=1, line_dash="dash", line_color="#475569",
                                   line_width=0.8, row=2, col=1)
                fig_vol.update_layout(**CHART_LAYOUT, height=400)
                st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})
            except Exception as e:
                render_error("Volume Chart Error", str(e)[:100])

            # Feature snapshot table
            st.markdown('<div class="rn-section-title" style="margin-top:8px">SIGNAL SNAPSHOT · LATEST BAR</div>', unsafe_allow_html=True)
            try:
                snapshot = {
                    "RSI 14": f"{pred['rsi']:.2f}",
                    "RSI 7":  f"{float(df['RSI7'].iloc[-1]):.2f}",
                    "RSI 21": f"{float(df['RSI21'].iloc[-1]):.2f}",
                    "ADX":    f"{pred['adx']:.2f}",
                    "DI+":    f"{float(df['ADX_POS'].iloc[-1]):.2f}",
                    "DI−":    f"{float(df['ADX_NEG'].iloc[-1]):.2f}",
                    "MACD":   f"{float(df['MACD'].iloc[-1]):.4f}",
                    "MACD Sig": f"{float(df['MACD_SIGNAL'].iloc[-1]):.4f}",
                    "MACD Hist": f"{pred['macd_hist']:.4f}",
                    "BB Width": f"{float(df['BB_WIDTH'].iloc[-1]):.4f}",
                    "BB Pos":   f"{pred['bb_position']*100:.1f}%",
                    "Stoch %K": f"{pred['stoch_k']:.1f}",
                    "Vol Ratio": f"{pred['vol_ratio']:.2f}x",
                    "RVOL 24h": f"{pred['rvol_24h']*100:.3f}%",
                }
                snap_rows = "".join(
                    f'<div class="rn-level-row"><span class="rn-level-name" style="color:#94a3b8">{k}</span><span class="rn-level-price" style="font-size:13px">{v}</span></div>'
                    for k, v in snapshot.items()
                )
                st.markdown(f'<div class="rn-levels-card">{snap_rows}</div>', unsafe_allow_html=True)
            except Exception as e:
                render_error("Snapshot Error", str(e)[:100])

        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB: ABOUT ──
    with tab_about:
        st.markdown("""
        <div class="rn-section">
        <div class="rn-section-title">ABOUT THIS DASHBOARD</div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius-lg);padding:32px;max-width:860px">

            <div style="font-family:'Orbitron',sans-serif;font-size:20px;font-weight:900;background:linear-gradient(135deg,#3b82f6,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:16px">
                RENDER ML PREDICTION ENGINE
            </div>

            <div style="font-family:'DM Sans',sans-serif;font-size:14px;color:#94a3b8;line-height:1.8;margin-bottom:24px">
                Institutional-grade machine learning prediction system for the RENDER (RNDR) token.
                Combines XGBoost gradient-boosted trees and LSTM neural networks trained on 60+ technical features
                derived from multi-asset OHLCV data.
            </div>

            <div class="rn-section-title">MODEL ARCHITECTURE</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:24px">
                <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:var(--radius-md);padding:16px">
                    <div style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:#3b82f6;text-transform:uppercase;margin-bottom:8px">XGBoost Classifier</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.6">Direction signal (UP/FLAT/DOWN) with class probabilities. Weight: 60% in ensemble.</div>
                </div>
                <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:var(--radius-md);padding:16px">
                    <div style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:#06b6d4;text-transform:uppercase;margin-bottom:8px">XGBoost Regressor</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.6">6h return prediction. Gradient-boosted on engineered feature set.</div>
                </div>
                <div style="background:var(--bg-secondary);border:1px solid var(--border);border-radius:var(--radius-md);padding:16px">
                    <div style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:2px;color:#a855f7;text-transform:uppercase;margin-bottom:8px">LSTM Network</div>
                    <div style="font-size:12px;color:#94a3b8;line-height:1.6">48-step temporal sequence model. Captures non-linear time dependencies. Weight: 40%.</div>
                </div>
            </div>

            <div class="rn-section-title">FEATURE GROUPS (60+ FEATURES)</div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:8px;margin-bottom:24px">
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Price Momentum</div><div style="font-size:12px;color:#94a3b8">Returns 3h–48h</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Trend</div><div style="font-size:12px;color:#94a3b8">EMA 8/20/50/100/200</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Oscillators</div><div style="font-size:12px;color:#94a3b8">RSI, Stoch, ROC</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">MACD</div><div style="font-size:12px;color:#94a3b8">Signal + Histogram</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Volatility</div><div style="font-size:12px;color:#94a3b8">ATR, BB, Keltner</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Volume</div><div style="font-size:12px;color:#94a3b8">OBV, VWAP, Ratio</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Market Corr</div><div style="font-size:12px;color:#94a3b8">BTC/ETH/SOL/NVDA</div></div>
                <div class="rn-indicator-pill"><div class="rn-indicator-name">Time</div><div style="font-size:12px;color:#94a3b8">Hour, DOW cyclical</div></div>
            </div>

            <div class="rn-warn-card">
                ⚠️ DISCLAIMER: This dashboard is for research and educational purposes only. Nothing here constitutes financial advice. Crypto trading involves significant risk of loss.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    render_footer()


if __name__ == "__main__":
    main()
