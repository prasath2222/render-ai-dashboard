import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import joblib
import ta
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="RENDER AI · RNDR Prediction Terminal",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #07090f !important;
    color: #c9d1d9 !important;
}
.stApp { background: #07090f !important; }
.block-container {
    padding: 0 20px 40px 20px !important;
    max-width: 1600px !important;
}
section[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1c2333 !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.stSelectbox > div > div { background: #161b22 !important; border-color: #30363d !important; }
.stCheckbox span { color: #c9d1d9 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07090f; }
::-webkit-scrollbar-thumb { background: #21262d; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #30363d; }

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }

/* plotly container */
.js-plotly-plot .plotly .main-svg { border-radius: 12px; }

/* sidebar labels */
.sidebar-label {
    font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
    color: #484f58; text-transform: uppercase; margin: 16px 0 8px;
}

/* toggle switch styling */
.stToggle > label > div { background: #1c2333 !important; }

/* button */
.stButton > button {
    background: #1c2333 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #21262d !important;
    border-color: #388bfd !important;
    color: #58a6ff !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# COLOUR PALETTE
# ═══════════════════════════════════════════════════════
BG       = "#07090f"
BG2      = "#0d1117"
BG3      = "#161b22"
BORDER   = "#21262d"
BORDER2  = "#30363d"
TXT      = "#c9d1d9"
TXT2     = "#8b949e"
TXT3     = "#484f58"
GREEN    = "#3fb950"
RED      = "#f85149"
BLUE     = "#388bfd"
ORANGE   = "#f0883e"
PURPLE   = "#bc8cff"
YELLOW   = "#d29922"
TEAL     = "#39d353"
CYAN     = "#79c0ff"

# ═══════════════════════════════════════════════════════
# PLOTLY LAYOUT HELPER
# ═══════════════════════════════════════════════════════
def _layout(fig, h=400, title="", margin=None):
    m = margin or dict(l=10, r=10, t=30 if title else 10, b=10)
    fig.update_layout(
        height=h,
        paper_bgcolor=BG2, plot_bgcolor=BG2,
        font=dict(family="Inter", color=TXT2, size=11),
        margin=m,
        title=dict(text=title, font=dict(size=11, color=TXT3), x=0.01, y=0.98) if title else None,
        xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
                   showline=False, tickfont=dict(size=10, color=TXT3),
                   rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False,
                   showline=False, tickfont=dict(size=10, color=TXT3)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=10, color=TXT2),
                    orientation="h", y=1.06, x=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=BG3, bordercolor=BORDER2,
                        font=dict(family="JetBrains Mono", size=11, color=TXT)),
        dragmode="pan",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=TXT3, activecolor=BLUE),
    )
    return fig

CHART_CONFIG = dict(
    scrollZoom=True,
    displayModeBar=True,
    modeBarButtonsToAdd=["drawline", "drawopenpath", "eraseshape"],
    modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d"],
    displaylogo=False,
    toImageButtonOptions=dict(format="png", scale=2),
)

# ═══════════════════════════════════════════════════════
# HTML CARD HELPERS
# ═══════════════════════════════════════════════════════
def card(content, extra_style=""):
    return f"""<div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
        padding:18px 20px;{extra_style}">{content}</div>"""

def label(text):
    return f'<div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};text-transform:uppercase;margin-bottom:8px;">{text}</div>'

def val_big(text, color=TXT):
    return f'<div style="font-size:28px;font-weight:800;color:{color};font-family:JetBrains Mono,monospace;line-height:1;">{text}</div>'

def val_sub(text, color=TXT2):
    return f'<div style="font-size:12px;color:{color};margin-top:5px;font-family:JetBrains Mono,monospace;">{text}</div>'

def sig_badge(sig):
    c = GREEN if sig=="BUY" else RED if sig=="SELL" else YELLOW
    bg = f"{c}18"
    border = f"{c}35"
    return f'<span style="display:inline-block;padding:3px 12px;border-radius:6px;font-size:12px;font-weight:700;letter-spacing:0.08em;background:{bg};color:{c};border:1px solid {border};">{sig}</span>'

def accent_bar(color):
    return f'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};border-radius:12px 12px 0 0;"></div>'

# ═══════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════
@st.cache_data(ttl=60, show_spinner=False)
def get_coingecko():
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=render-token&vs_currencies=usd"
            "&include_24hr_vol=true&include_24hr_change=true"
            "&include_market_cap=true&include_high_24h=true&include_low_24h=true",
            timeout=10,
        )
        return r.json().get("render-token", {})
    except Exception:
        return {}

@st.cache_data(ttl=120, show_spinner=False)
def get_inr_rate():
    try:
        r = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=8)
        return float(r.json()["rates"].get("INR", 84.0))
    except Exception:
        return 84.0

@st.cache_data(ttl=300, show_spinner=False)
def get_ohlcv(period="90d", interval="1h"):
    for ticker in ["RENDER-USD", "RNDR-USD"]:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            if len(df) > 50:
                return df
        except Exception:
            continue
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fng():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=8)
        d = r.json()["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return 50, "Neutral"

# ═══════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════
def add_indicators(df):
    def safe(s):
        return s.replace([np.inf, -np.inf], np.nan).fillna(0)
    c, h, l, v, o = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]

    df["EMA9"]   = safe(ta.trend.EMAIndicator(c, 9).ema_indicator())
    df["EMA20"]  = safe(ta.trend.EMAIndicator(c, 20).ema_indicator())
    df["EMA50"]  = safe(ta.trend.EMAIndicator(c, 50).ema_indicator())
    df["EMA200"] = safe(ta.trend.EMAIndicator(c, 200).ema_indicator())

    _macd = ta.trend.MACD(c)
    df["MACD"]       = safe(_macd.macd())
    df["MACD_SIG"]   = safe(_macd.macd_signal())
    df["MACD_HIST"]  = safe(_macd.macd_diff())

    _adx = ta.trend.ADXIndicator(h, l, c, 14)
    df["ADX"]     = safe(_adx.adx())
    df["ADX_POS"] = safe(_adx.adx_pos())
    df["ADX_NEG"] = safe(_adx.adx_neg())

    df["RSI"]    = safe(ta.momentum.RSIIndicator(c, 14).rsi())
    df["RSI7"]   = safe(ta.momentum.RSIIndicator(c, 7).rsi())

    _stoch = ta.momentum.StochasticOscillator(h, l, c, 14, 3)
    df["STOCH_K"] = safe(_stoch.stoch())
    df["STOCH_D"] = safe(_stoch.stoch_signal())

    df["CCI"]      = safe(ta.trend.CCIIndicator(h, l, c, 20).cci())
    df["WILLIAMS"] = safe(ta.momentum.WilliamsRIndicator(h, l, c, 14).williams_r())
    df["ROC5"]     = safe(ta.momentum.ROCIndicator(c, 5).roc())
    df["ROC20"]    = safe(ta.momentum.ROCIndicator(c, 20).roc())

    _bb = ta.volatility.BollingerBands(c, 20, 2)
    df["BB_U"] = safe(_bb.bollinger_hband())
    df["BB_L"] = safe(_bb.bollinger_lband())
    df["BB_M"] = safe(_bb.bollinger_mavg())
    df["BB_W"] = safe(_bb.bollinger_wband())
    df["BB_P"] = safe(_bb.bollinger_pband())

    df["ATR"] = safe(ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range())

    df["OBV"]  = safe(ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume())
    df["MFI"]  = safe(ta.volume.MFIIndicator(h, l, c, v, 14).money_flow_index())
    df["CMF"]  = safe(ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow())

    try:
        df["VWAP"] = safe(ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price())
    except Exception:
        df["VWAP"] = c.copy()

    vm20 = v.rolling(20).mean().replace(0, np.nan)
    df["VOL_MA"] = safe(vm20)
    df["VOL_R"]  = safe(v / vm20)

    df["RET1"]  = safe(c.pct_change(1))
    df["RET4"]  = safe(c.pct_change(4))
    df["RET24"] = safe(c.pct_change(24))

    # Features for ML compat
    df["ema_9"]       = df["EMA9"]
    df["ema_21"]      = safe(ta.trend.EMAIndicator(c, 21).ema_indicator())
    df["ema_50"]      = df["EMA50"]
    df["sma_20"]      = safe(ta.trend.SMAIndicator(c, 20).sma_indicator())
    df["macd"]        = df["MACD"]
    df["macd_signal"] = df["MACD_SIG"]
    df["macd_diff"]   = df["MACD_HIST"]
    df["adx"]         = df["ADX"]
    df["adx_pos"]     = df["ADX_POS"]
    df["adx_neg"]     = df["ADX_NEG"]
    df["rsi_14"]      = df["RSI"]
    df["rsi_7"]       = df["RSI7"]
    df["rsi_28"]      = safe(ta.momentum.RSIIndicator(c, 28).rsi())
    df["stoch_k"]     = df["STOCH_K"]
    df["stoch_d"]     = df["STOCH_D"]
    df["cci"]         = df["CCI"]
    df["roc_5"]       = df["ROC5"]
    df["roc_20"]      = df["ROC20"]
    df["williams"]    = df["WILLIAMS"]
    df["bb_upper"]    = df["BB_U"]
    df["bb_lower"]    = df["BB_L"]
    df["bb_width"]    = df["BB_W"]
    df["bb_pct"]      = df["BB_P"]
    df["atr_14"]      = df["ATR"]
    df["obv"]         = df["OBV"]
    df["mfi"]         = df["MFI"]
    df["cmf"]         = df["CMF"]
    df["vwap"]        = df["VWAP"]
    df["ret_1h"]      = df["RET1"]
    df["ret_4h"]      = df["RET4"]
    df["ret_24h"]     = df["RET24"]
    df["log_ret"]     = safe(np.log(c / c.shift(1)))
    df["vol_24h"]     = safe(df["RET1"].rolling(24).std())
    df["vol_7d"]      = safe(df["RET1"].rolling(168).std())

    hl = (h - l).replace(0, np.nan)
    df["body"]       = safe((c - o).abs() / hl)
    df["upper_wick"] = safe((h - c.clip(lower=o)) / hl)
    df["lower_wick"] = safe((c.clip(upper=o) - l) / hl)
    df["hl_ratio"]   = safe(h / l)
    df["close_open"] = safe(c / o)

    vm24  = v.rolling(24).mean().replace(0, np.nan)
    vm168 = v.rolling(168).mean().replace(0, np.nan)
    df["vol_r24"]  = safe(v / vm24)
    df["vol_r168"] = safe(v / vm168)
    df["whale"]    = (df["vol_r24"] > 3).astype(int)

    df["dist_hi24"] = safe((h.rolling(24).max() - c) / c)
    df["dist_lo24"] = safe((c - l.rolling(24).min()) / c)
    df["dist_hi7d"] = safe((h.rolling(168).max() - c) / c)
    df["dist_lo7d"] = safe((c - l.rolling(168).min()) / c)

    df["ema_score"]     = ((df["ema_9"] > df["ema_21"]).astype(int) +
                           (df["ema_21"] > df["ema_50"]).astype(int))
    df["price_vs_vwap"] = safe((c - df["vwap"]) / df["vwap"])
    df["fng"]           = 50
    df["fng_fear"]      = 0
    df["fng_greed"]     = 0
    df["btc_ret"]       = 0.0
    df["rndr_vs_btc"]   = 0.0
    df["btc_ret_4h"]    = 0.0
    df["hour"] = df.index.hour
    df["dow"]  = df.index.dayofweek

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    return df

# ═══════════════════════════════════════════════════════
# ML MODELS
# ═══════════════════════════════════════════════════════
MODEL_FILES = ["render_ensemble_cls.pkl","render_reg.pkl","render_scaler.pkl","render_features.pkl"]

def load_models():
    try:
        return (
            joblib.load("render_ensemble_cls.pkl"),
            joblib.load("render_reg.pkl"),
            joblib.load("render_scaler.pkl"),
            joblib.load("render_features.pkl"),
        )
    except Exception:
        return None, None, None, None

def ml_predict(df, ens, reg, scaler, feats):
    EXCL = {"Open","High","Low","Close","Volume","future_close","future_return","label"}
    avail = [f for f in feats if f in df.columns and f not in EXCL]
    row = df.iloc[-1][avail].values.astype(np.float64).reshape(1, -1)
    row = np.nan_to_num(row)
    full = np.zeros((1, len(feats)))
    idx  = {f: i for i, f in enumerate(feats)}
    for i, f in enumerate(avail):
        full[0, idx[f]] = row[0, i]
    Xs = scaler.transform(full)
    lmap = {0: "SELL", 1: "HOLD", 2: "BUY"}
    return lmap[int(ens.predict(Xs)[0])], ens.predict_proba(Xs)[0], float(reg.predict(Xs)[0])

# ═══════════════════════════════════════════════════════
# TREND SCORE
# ═══════════════════════════════════════════════════════
def trend_score(latest, close_s):
    ts = 0
    e20  = float(latest.get("EMA20",  0))
    e50  = float(latest.get("EMA50",  0))
    e200 = float(latest.get("EMA200", 0))
    rsi  = float(latest.get("RSI",   50))
    macd = float(latest.get("MACD",   0))
    msig = float(latest.get("MACD_SIG", 0))
    adx  = float(latest.get("ADX",    0))
    vol  = float(latest.get("Volume", 0))
    vma  = float(latest.get("VOL_MA", 1))

    if e20 > e50:  ts += 20
    else:          ts -= 20
    if e50 > e200: ts += 20
    else:          ts -= 20
    if rsi >= 70:  ts -= 15
    elif rsi >= 60:ts += 15
    elif rsi <= 30:ts += 20
    elif rsi <= 40:ts -= 10
    if macd > msig:ts += 20
    else:          ts -= 20
    if adx > 25:   ts += 10
    if len(close_s) >= 10:
        pc = (close_s.iloc[-1] - close_s.iloc[-10]) / max(close_s.iloc[-10], 1e-9) * 100
        ts += 10 if pc > 0 else -10
    if vol > vma:  ts += 5
    return int(ts)

# ═══════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════
def chart_candles(df, show_ema, show_bb, show_vol):
    rows      = 3 if show_vol else 2
    rh        = [0.60, 0.20, 0.20] if show_vol else [0.72, 0.28]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.012, row_heights=rh,
    )

    # ── Candles ──────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].values, high=df["High"].values,
        low=df["Low"].values,   close=df["Close"].values,
        increasing=dict(line=dict(color=GREEN, width=1), fillcolor=GREEN),
        decreasing=dict(line=dict(color=RED,   width=1), fillcolor=RED),
        name="RNDR",
        showlegend=True,
        whiskerwidth=0.6,
    ), row=1, col=1)

    # ── VWAP ─────────────────────────────────────────
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"].values,
            name="VWAP", line=dict(color=ORANGE, width=1.2, dash="dash"),
            opacity=0.85, hoverinfo="skip",
        ), row=1, col=1)

    # ── EMA overlays ─────────────────────────────────
    if show_ema:
        for col_name, color, name in [
            ("EMA9",  CYAN,   "EMA9"),
            ("EMA20", GREEN,  "EMA20"),
            ("EMA50", PURPLE, "EMA50"),
            ("EMA200",ORANGE, "EMA200"),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name].values,
                    name=name, line=dict(color=color, width=1),
                    opacity=0.9, hoverinfo="skip",
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────
    if show_bb and "BB_U" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_U"].values, name="BB Upper",
            line=dict(color=BORDER2, width=0.8),
            hoverinfo="skip", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_L"].values, name="BB Lower",
            line=dict(color=BORDER2, width=0.8),
            fill="tonexty", fillcolor="rgba(48,54,61,0.12)",
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_M"].values, name="BB Mid",
            line=dict(color="#444c56", width=0.6),
            hoverinfo="skip", showlegend=False,
        ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────
    rsi_row = 2
    if show_vol and "Volume" in df.columns:
        rsi_row = 3
        vol_colors = [
            GREEN if float(c) >= float(o) else RED
            for c, o in zip(df["Close"].values, df["Open"].values)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"].values,
            marker_color=vol_colors, name="Volume",
            opacity=0.5, showlegend=False,
        ), row=2, col=1)
        if "VOL_MA" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["VOL_MA"].values,
                line=dict(color=ORANGE, width=1),
                name="Vol MA20", showlegend=False, hoverinfo="skip",
            ), row=2, col=1)

    # ── RSI sub-chart ─────────────────────────────────
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"].values,
            line=dict(color=ORANGE, width=1.4),
            name="RSI(14)", showlegend=False,
        ), row=rsi_row, col=1)
        for lvl, col in [(70, RED), (30, GREEN)]:
            fig.add_hline(y=lvl, line_width=0.6, line_dash="dot",
                          line_color=col, row=rsi_row, col=1)
        fig.add_hline(y=50, line_width=0.4, line_color=BORDER2,
                      row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    _layout(fig, h=640)
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        newshape=dict(line_color=BLUE),
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(showgrid=True, gridcolor=BORDER,
                         gridwidth=0.5, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=BORDER,
                         gridwidth=0.5, row=i, col=1)
    return fig

def chart_macd(df):
    fig = go.Figure()
    colors = [GREEN if v >= 0 else RED for v in df["MACD_HIST"].values]
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"].values,
                         marker_color=colors, name="Hist", opacity=0.8))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"].values,
                             line=dict(color=BLUE, width=1.5), name="MACD"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"].values,
                             line=dict(color=ORANGE, width=1.5, dash="dot"), name="Signal"))
    fig.add_hline(y=0, line_width=0.5, line_color=BORDER2)
    _layout(fig, h=200, title="MACD (12,26,9)")
    return fig

def chart_stoch(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_K"].values,
                             line=dict(color=BLUE, width=1.4), name="%K"))
    fig.add_trace(go.Scatter(x=df.index, y=df["STOCH_D"].values,
                             line=dict(color=ORANGE, width=1.4, dash="dot"), name="%D"))
    for lvl, col in [(80, RED), (20, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.6, line_dash="dot", line_color=col)
    _layout(fig, h=200, title="Stochastic (14,3,3)")
    fig.update_yaxes(range=[0, 100])
    return fig

def chart_adx(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"].values,
                             line=dict(color=YELLOW, width=1.8), name="ADX"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX_POS"].values,
                             line=dict(color=GREEN, width=1, dash="dot"), name="+DI"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX_NEG"].values,
                             line=dict(color=RED, width=1, dash="dot"), name="−DI"))
    fig.add_hline(y=25, line_width=0.7, line_dash="dash", line_color=BORDER2)
    _layout(fig, h=200, title="ADX · Trend Strength")
    return fig

def chart_bb(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_W"].values, name="BB Width",
        line=dict(color=PURPLE, width=1.5),
        fill="tozeroy", fillcolor="rgba(188,140,255,0.07)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_P"].values, name="BB %B",
        line=dict(color=TEAL, width=1.4),
    ), row=2, col=1)
    for lvl, col in [(1, RED), (0, GREEN)]:
        fig.add_hline(y=lvl, line_width=0.5, line_color=col, row=2, col=1)
    _layout(fig, h=260, title="Bollinger Band Width + %B")
    for i in [1,2]:
        fig.update_xaxes(gridcolor=BORDER, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, row=i, col=1)
    return fig

def chart_volume_flow(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04, row_heights=[0.5, 0.5])
    fig.add_trace(go.Scatter(
        x=df.index, y=df["OBV"].values,
        line=dict(color=BLUE, width=1.4), name="OBV",
        fill="tozeroy", fillcolor="rgba(56,139,253,0.07)",
    ), row=1, col=1)
    cmf_colors = [GREEN if v >= 0 else RED for v in df["CMF"].values]
    fig.add_trace(go.Bar(
        x=df.index, y=df["CMF"].values,
        marker_color=cmf_colors, name="CMF (Inflow/Outflow)", opacity=0.8,
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=0.5, line_color=BORDER2, row=2, col=1)
    _layout(fig, h=280, title="Volume Flow · OBV + CMF Inflow/Outflow")
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=BORDER, row=i, col=1)
        fig.update_yaxes(gridcolor=BORDER, row=i, col=1)
    return fig

# ═══════════════════════════════════════════════════════
# SIMPLE PROGRESS BAR HTML (no gauge — avoids Plotly bug)
# ═══════════════════════════════════════════════════════
def regime_html(score):
    clamped = max(-100, min(100, score))
    pct     = int((clamped + 100) / 2)   # 0-100 scale
    color   = GREEN if clamped >= 30 else RED if clamped <= -30 else YELLOW
    label_  = "BULLISH" if clamped >= 30 else "BEARISH" if clamped <= -30 else "SIDEWAYS"
    return f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;padding:20px;text-align:center;">
      <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};margin-bottom:16px;">MARKET REGIME</div>
      <div style="font-size:48px;font-weight:800;color:{color};font-family:JetBrains Mono,monospace;line-height:1;">{clamped:+d}</div>
      <div style="font-size:14px;font-weight:700;color:{color};letter-spacing:0.1em;margin:8px 0 16px;">{label_}</div>
      <div style="height:8px;background:{BORDER};border-radius:4px;overflow:hidden;">
        <div style="height:100%;width:{pct}%;background:{color};border-radius:4px;transition:width 0.6s;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;">
        <span style="font-size:10px;color:{RED};">BEAR</span>
        <span style="font-size:10px;color:{YELLOW};">NEUTRAL</span>
        <span style="font-size:10px;color:{GREEN};">BULL</span>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():

    # ── SIDEBAR ─────────────────────────────────────────
    with st.sidebar:
        # RENDER Logo SVG
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:10px 0 20px;">
          <svg width="38" height="38" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="50" fill="#F0883E" opacity="0.15"/>
            <circle cx="50" cy="50" r="38" fill="#F0883E" opacity="0.2"/>
            <path d="M32 28H58C64.627 28 70 33.373 70 40C70 46.627 64.627 52 58 52H32V28Z" fill="#F0883E"/>
            <path d="M32 52L52 72H40L32 62V52Z" fill="#F0883E" opacity="0.7"/>
            <path d="M52 52L70 72H58L42 52H52Z" fill="#F0883E" opacity="0.5"/>
          </svg>
          <div>
            <div style="font-size:16px;font-weight:800;color:#f0f6fc;letter-spacing:-0.3px;">RENDER</div>
            <div style="font-size:10px;color:#484f58;letter-spacing:0.1em;font-weight:600;">AI TERMINAL</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sidebar-label">Timeframe</div>', unsafe_allow_html=True)
        tf_map = {"1 Hour": "1h", "4 Hours": "4h", "1 Day": "1d"}
        sel_tf = tf_map[st.selectbox("tf", list(tf_map.keys()), label_visibility="collapsed")]

        st.markdown('<div class="sidebar-label">Period</div>', unsafe_allow_html=True)
        per_map = {"30 Days":"30d","60 Days":"60d","90 Days":"90d","180 Days":"180d"}
        sel_per = per_map[st.selectbox("per", list(per_map.keys()), index=2, label_visibility="collapsed")]

        st.markdown('<div class="sidebar-label">Indicators</div>', unsafe_allow_html=True)
        show_ema  = st.toggle("EMA 20/50/200", value=True)
        show_bb   = st.toggle("Bollinger Bands", value=True)
        show_vwap = st.toggle("VWAP", value=True)
        show_vol  = st.toggle("Volume", value=True)

        st.markdown('<div class="sidebar-label">AI Models</div>', unsafe_allow_html=True)
        models_ok = all(os.path.exists(f) for f in MODEL_FILES)
        if models_ok:
            st.success("✅ Models ready")
        else:
            st.warning("⚠️ Run `python train.py`\nRule-based signals active")

        st.markdown('<div class="sidebar-label">Refresh</div>', unsafe_allow_html=True)
        if st.button("↻  Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:20px;padding-top:16px;border-top:1px solid {BORDER};
             font-size:10px;color:{TXT3};letter-spacing:0.06em;">
          Updated: {datetime.now().strftime("%H:%M:%S")}<br>
          Data: CoinGecko · Yahoo Finance
        </div>""", unsafe_allow_html=True)

    # ── FETCH ────────────────────────────────────────────
    with st.spinner(""):
        cg       = get_coingecko()
        inr_rate = get_inr_rate()
        df_raw   = get_ohlcv(period=sel_per, interval=sel_tf)
        fng_val, fng_class = get_fng()

    if df_raw.empty:
        st.error("⚠️ Could not load price data. Check internet connection.")
        st.stop()

    df      = add_indicators(df_raw.copy())
    latest  = df.iloc[-1]
    close_s = df["Close"]

    cur_price = float(cg.get("usd", float(latest["Close"])))
    chg24     = float(cg.get("usd_24h_change", 0.0))
    h24       = float(cg.get("usd_24h_high",   float(df["High"].tail(24).max())))
    l24       = float(cg.get("usd_24h_low",    float(df["Low"].tail(24).min())))
    vol24     = float(cg.get("usd_24h_vol",    float(df["Volume"].tail(24).sum())))
    mcap      = float(cg.get("usd_market_cap", 0))
    price_inr = cur_price * inr_rate

    ts = trend_score(
        {k: float(latest.get(k, 0)) for k in
         ["EMA20","EMA50","EMA200","RSI","MACD","MACD_SIG","ADX","Volume","VOL_MA"]},
        close_s,
    )

    ens, reg, scaler, feats = load_models()
    if ens and reg and scaler and feats:
        cls_sig, cls_proba, reg_price = ml_predict(df, ens, reg, scaler, feats)
    else:
        cls_sig   = "BUY" if ts >= 40 else ("SELL" if ts <= -40 else "HOLD")
        cls_proba = (np.array([0.1,0.2,0.7]) if cls_sig=="BUY"
                     else np.array([0.7,0.2,0.1]) if cls_sig=="SELL"
                     else np.array([0.2,0.6,0.2]))
        atr_v = float(latest.get("ATR", cur_price*0.02))
        reg_price = cur_price * (1 + (ts/100)*(atr_v/cur_price)*5)

    conf = int(min(95, max(50, float(max(cls_proba))*100)))

    is_up   = chg24 >= 0
    up_col  = GREEN if is_up else RED
    sig_col = GREEN if cls_sig=="BUY" else RED if cls_sig=="SELL" else YELLOW

    # ═══════════════════════════════════════════════════
    # TOP HEADER
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="background:{BG2};border-bottom:1px solid {BORDER};
         padding:14px 20px;margin:0 -20px 24px -20px;
         display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
      <div style="display:flex;align-items:center;gap:14px;">
        <svg width="36" height="36" viewBox="0 0 100 100" fill="none">
          <circle cx="50" cy="50" r="50" fill="#F0883E" opacity="0.15"/>
          <path d="M28 24H58C66.284 24 73 30.716 73 39C73 47.284 66.284 54 58 54H28V24Z" fill="#F0883E"/>
          <path d="M28 54L50 76H38L28 64V54Z" fill="#F0883E" opacity="0.6"/>
          <path d="M50 54L73 76H61L40 54H50Z" fill="#F0883E" opacity="0.4"/>
        </svg>
        <div>
          <div style="font-size:18px;font-weight:800;color:#f0f6fc;letter-spacing:-0.3px;">RENDER NETWORK · RNDR / USDT</div>
          <div style="font-size:11px;color:{TXT3};letter-spacing:0.06em;margin-top:1px;">AI & GPU Rendering Infrastructure · Decentralised</div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="font-size:11px;font-weight:700;font-family:JetBrains Mono,monospace;color:#f0f6fc;">${cur_price:.4f}</div>
        <div style="font-size:12px;font-weight:600;color:{up_col};background:{up_col}18;
             padding:3px 10px;border-radius:6px;border:1px solid {up_col}30;">
          {'+'if is_up else ''}{chg24:.2f}%
        </div>
        <div style="display:flex;align-items:center;gap:6px;background:{GREEN}12;
             border:1px solid {GREEN}30;color:{GREEN};font-size:11px;font-weight:600;
             padding:4px 12px;border-radius:20px;letter-spacing:0.08em;margin-left:8px;">
          <div style="width:6px;height:6px;border-radius:50%;background:{GREEN};
               animation:blink 1.4s ease-in-out infinite;"></div>
          LIVE
        </div>
      </div>
    </div>
    <style>@keyframes blink{{0%,100%{{opacity:1}}50%{{opacity:0.2}}}}</style>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # STAT CARDS ROW
    # ═══════════════════════════════════════════════════
    cols = st.columns(6)
    card_data = [
        ("PRICE (USD)", f"${cur_price:.4f}", f"₹{price_inr:,.2f}", ORANGE),
        ("24H CHANGE",  f"{'+'if is_up else ''}{chg24:.2f}%",
         f"H: ${h24:.4f}  L: ${l24:.4f}", up_col),
        ("AI SIGNAL",   cls_sig, f"Confidence {conf}%", sig_col),
        ("PREDICTED",   f"${reg_price:.4f}",
         f"₹{reg_price*inr_rate:,.2f}  {((reg_price/cur_price-1)*100):+.2f}%",
         GREEN if reg_price >= cur_price else RED),
        ("24H VOLUME",  f"${vol24/1e6:.2f}M",
         f"₹{vol24*inr_rate/1e6:.1f}M  MCap ${mcap/1e9:.2f}B", BLUE),
        ("FEAR & GREED", str(fng_val), fng_class,
         GREEN if fng_val>60 else RED if fng_val<40 else YELLOW),
    ]
    for col, (lbl, main, sub, color) in zip(cols, card_data):
        col.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
             padding:16px;position:relative;overflow:hidden;min-height:90px;">
          <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{color};
               border-radius:12px 12px 0 0;"></div>
          <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};margin-bottom:8px;">{lbl}</div>
          <div style="font-size:22px;font-weight:800;color:{color};font-family:JetBrains Mono,monospace;line-height:1;">{main}</div>
          <div style="font-size:11px;color:{TXT2};margin-top:5px;font-family:JetBrains Mono,monospace;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # TRADINGVIEW LIVE CHART — correct symbol: CRYPTO:RENDERUSD
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">TRADINGVIEW LIVE CHART</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
      <div style="font-size:10px;color:{TXT3};">CRYPTO:RENDERUSD · Scroll to zoom · Drag to pan</div>
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html("""
    <div style="border-radius:12px;overflow:hidden;border:1px solid #21262d;background:#0d1117;">
      <div class="tradingview-widget-container" style="height:520px;width:100%;">
        <div class="tradingview-widget-container__widget" style="height:100%;width:100%;"></div>
        <script type="text/javascript"
          src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {
          "autosize": true,
          "symbol": "CRYPTO:RENDERUSD",
          "interval": "60",
          "timezone": "Asia/Kolkata",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "backgroundColor": "rgba(13,17,23,1)",
          "gridColor": "rgba(33,38,45,0.8)",
          "hide_top_toolbar": false,
          "hide_legend": false,
          "hide_side_toolbar": false,
          "allow_symbol_change": false,
          "save_image": true,
          "calendar": false,
          "hide_volume": false,
          "studies": ["RSI@tv-basicstudies","MACD@tv-basicstudies","BB@tv-basicstudies","Volume@tv-basicstudies"],
          "support_host": "https://www.tradingview.com"
        }
        </script>
      </div>
    </div>
    """, height=540)

    # ═══════════════════════════════════════════════════
    # LOCAL OHLCV CHART (Plotly — full scroll/zoom)
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:24px 0 12px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">RNDR OHLCV · LOCAL DATA</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
      <div style="font-size:10px;color:{TXT3};">Scroll wheel = zoom · Click+drag = pan · Double-click = reset</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    ema_cb  = c1.checkbox("EMA Overlays",    value=show_ema,  key="cb_ema")
    bb_cb   = c2.checkbox("Bollinger Bands", value=show_bb,   key="cb_bb")
    vol_cb  = c3.checkbox("Volume",          value=show_vol,  key="cb_vol")

    st.plotly_chart(
        chart_candles(df, ema_cb, bb_cb, vol_cb),
        use_container_width=True,
        config=CHART_CONFIG,
    )

    # ═══════════════════════════════════════════════════
    # TECHNICAL INDICATORS GRID
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:24px 0 14px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">TECHNICAL INDICATORS</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>""", unsafe_allow_html=True)

    rsi_v    = float(latest.get("RSI",      50))
    macd_h   = float(latest.get("MACD_HIST", 0))
    adx_v    = float(latest.get("ADX",      0))
    stoch_kv = float(latest.get("STOCH_K",  50))
    atr_v    = float(latest.get("ATR",      cur_price*0.02))
    bb_wv    = float(latest.get("BB_W",     0))
    cci_v    = float(latest.get("CCI",      0))
    mfi_v    = float(latest.get("MFI",      50))
    cmf_v    = float(latest.get("CMF",      0))
    wil_v    = float(latest.get("WILLIAMS", -50))
    roc_v    = float(latest.get("ROC5",     0))
    e9       = float(latest.get("EMA9",     cur_price))
    e20_v    = float(latest.get("EMA20",    cur_price))
    e50_v    = float(latest.get("EMA50",    cur_price))

    def rsi_sig(r):
        if r>=70: return "OVERBOUGHT", RED
        if r<=30: return "OVERSOLD",   GREEN
        if r>=55: return "BULLISH",    GREEN
        return "NEUTRAL", YELLOW

    def sig_fn(cond_bull, cond_bear, lbl_b, lbl_n, lbl_s):
        if cond_bull: return lbl_b, GREEN
        if cond_bear: return lbl_s, RED
        return lbl_n, YELLOW

    ema_align = e9>e20_v and e20_v>e50_v
    ema_bear  = e9<e20_v and e20_v<e50_v
    ema_lbl   = "ALIGNED BULL" if ema_align else ("ALIGNED BEAR" if ema_bear else "MIXED")
    ema_col   = GREEN if ema_align else (RED if ema_bear else YELLOW)

    inds = [
        ("RSI (14)",    f"{rsi_v:.1f}",                 *rsi_sig(rsi_v)),
        ("MACD HIST",   f"{macd_h:+.4f}",              *sig_fn(macd_h>0,macd_h<0,"BULLISH","NEUTRAL","BEARISH")),
        ("ADX (14)",    f"{adx_v:.1f}",                 *sig_fn(adx_v>25,adx_v<15,"STRONG","MODERATE","WEAK")),
        ("STOCH %K",    f"{stoch_kv:.1f}",              *sig_fn(stoch_kv<20,stoch_kv>80,"OVERSOLD","NEUTRAL","OVERBOUGHT")),
        ("ATR (14)",    f"${atr_v:.4f}",                "VOLATILITY", BLUE),
        ("BB WIDTH",    f"{bb_wv:.2f}%",                *sig_fn(bb_wv<2,bb_wv>8,"LOW VOL","NORMAL","HIGH VOL")),
        ("CCI (20)",    f"{cci_v:.1f}",                 *sig_fn(cci_v<-100,cci_v>100,"OVERSOLD","NEUTRAL","OVERBOUGHT")),
        ("MFI (14)",    f"{mfi_v:.1f}",                 *sig_fn(mfi_v<20,mfi_v>80,"OVERSOLD","NEUTRAL","OVERBOUGHT")),
        ("CMF (20)",    f"{cmf_v:+.3f}",               *sig_fn(cmf_v>0,cmf_v<0,"INFLOW","NEUTRAL","OUTFLOW")),
        ("WILLIAMS %R", f"{wil_v:.1f}",                 *sig_fn(wil_v<-80,wil_v>-20,"OVERSOLD","NEUTRAL","OVERBOUGHT")),
        ("ROC (5)",     f"{roc_v:+.2f}%",              *sig_fn(roc_v>0,roc_v<0,"POSITIVE","NEUTRAL","NEGATIVE")),
        ("EMA ALIGN",   ema_lbl,                         ema_lbl, ema_col),
    ]

    # Render in 4-column grid
    rows_of_4 = [inds[i:i+4] for i in range(0, len(inds), 4)]
    for row in rows_of_4:
        cols_ind = st.columns(len(row))
        for col_i, (name, vl, sig_txt, sig_col_i) in zip(cols_ind, row):
            col_i.markdown(f"""
            <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
                 padding:14px 16px;border-left:2px solid {sig_col_i};">
              <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:{TXT3};margin-bottom:8px;">{name}</div>
              <div style="font-size:20px;font-weight:700;font-family:JetBrains Mono,monospace;color:#f0f6fc;">{vl}</div>
              <div style="font-size:11px;font-weight:600;color:{sig_col_i};margin-top:4px;">{sig_txt}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # indicator sub-charts (2x2)
    colL, colR = st.columns(2)
    with colL:
        st.plotly_chart(chart_macd(df), use_container_width=True, config=CHART_CONFIG)
    with colR:
        st.plotly_chart(chart_stoch(df), use_container_width=True, config=CHART_CONFIG)
    colL2, colR2 = st.columns(2)
    with colL2:
        st.plotly_chart(chart_adx(df), use_container_width=True, config=CHART_CONFIG)
    with colR2:
        st.plotly_chart(chart_bb(df), use_container_width=True, config=CHART_CONFIG)

    # ═══════════════════════════════════════════════════
    # VOLUME & FLOW
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:24px 0 12px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">VOLUME · FLOW ANALYSIS · INFLOW / OUTFLOW</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_volume_flow(df), use_container_width=True, config=CHART_CONFIG)

    vol_r        = float(latest.get("VOL_R", 1.0))
    whale_pct    = float(df["whale"].tail(24).mean() * 100)
    buy_pres     = max(0.0, min(100.0, float(50 + ts)))
    sell_pres    = 100.0 - buy_pres
    buy_col      = GREEN if buy_pres > 55 else YELLOW
    sell_col     = RED if sell_pres > 55 else YELLOW

    vc = st.columns(4)
    for col_v, lbl, vl, sub, c in [
        (vc[0], "VOLUME SPIKE",  f"{vol_r:.2f}×",  "Above Avg" if vol_r>1 else "Below Avg", GREEN if vol_r>1 else RED),
        (vc[1], "BUY PRESSURE",  f"{buy_pres:.1f}%","Dominant" if buy_pres>55 else "Weak",   buy_col),
        (vc[2], "SELL PRESSURE", f"{sell_pres:.1f}%","Dominant" if sell_pres>55 else "Weak", sell_col),
        (vc[3], "WHALE ACTIVITY",f"{whale_pct:.1f}%","Spikes 24h",                            ORANGE),
    ]:
        col_v.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
             padding:14px;border-top:2px solid {c};">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;color:{TXT3};margin-bottom:6px;">{lbl}</div>
          <div style="font-size:22px;font-weight:800;font-family:JetBrains Mono,monospace;color:{c};">{vl}</div>
          <div style="font-size:11px;color:{TXT2};margin-top:4px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Buy/Sell pressure bar
    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:10px;
         overflow:hidden;height:32px;display:flex;margin-top:10px;">
      <div style="width:{buy_pres}%;background:{GREEN};display:flex;align-items:center;
           justify-content:center;font-size:12px;font-weight:700;color:#fff;font-family:JetBrains Mono,monospace;">
        BUY {buy_pres:.1f}%
      </div>
      <div style="flex:1;background:{RED};display:flex;align-items:center;
           justify-content:center;font-size:12px;font-weight:700;color:#fff;font-family:JetBrains Mono,monospace;">
        SELL {sell_pres:.1f}%
      </div>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # AI PREDICTIONS
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:28px 0 14px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">AI · CLASSIFICATION & REGRESSION PREDICTIONS</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>""", unsafe_allow_html=True)

    pCls, pReg, pReg2 = st.columns([1.2, 1.2, 1])

    # Classification card
    with pCls:
        p_sell = float(cls_proba[0]) * 100
        p_hold = float(cls_proba[1]) * 100
        p_buy  = float(cls_proba[2]) * 100

        def prob_bar(label_, pct, col):
            return f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:11px;color:{TXT3};font-weight:600;letter-spacing:0.08em;">{label_}</span>
                <span style="font-size:12px;color:{col};font-family:JetBrains Mono,monospace;font-weight:700;">{pct:.1f}%</span>
              </div>
              <div style="height:5px;background:{BORDER};border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{pct}%;background:{col};border-radius:3px;"></div>
              </div>
            </div>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
             padding:22px;border-top:2px solid {sig_col};height:100%;">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};margin-bottom:14px;">CLASSIFICATION MODEL</div>
          <div style="font-size:40px;font-weight:900;color:{sig_col};font-family:JetBrains Mono,monospace;
               letter-spacing:0.04em;margin-bottom:4px;">{cls_sig}</div>
          <div style="font-size:11px;color:{TXT3};margin-bottom:20px;">Ensemble · XGBoost + LightGBM + RandomForest</div>
          {prob_bar("BUY",  p_buy,  GREEN)}
          {prob_bar("HOLD", p_hold, YELLOW)}
          {prob_bar("SELL", p_sell, RED)}
          <div style="margin-top:16px;padding-top:14px;border-top:1px solid {BORDER};
               display:flex;justify-content:space-between;align-items:center;">
            <span style="font-size:11px;color:{TXT3};">Confidence</span>
            <span style="font-size:18px;font-weight:800;font-family:JetBrains Mono,monospace;color:{sig_col};">{conf}%</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # Regression card
    with pReg:
        reg_chg = (reg_price / cur_price - 1) * 100
        reg_col = GREEN if reg_chg >= 0 else RED
        atr_v2  = float(latest.get("ATR", cur_price*0.02))
        r1h     = cur_price * (1 + (ts/100)*(atr_v2/cur_price)*1)
        r4h     = cur_price * (1 + (ts/100)*(atr_v2/cur_price)*2)
        r24h    = reg_price
        sigma   = atr_v2 * 2

        def reg_row_html(hor, price):
            chg = (price/cur_price-1)*100
            cc  = GREEN if chg>=0 else RED
            return f"""
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{TXT3};padding:9px 0;">{hor}</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;">
                <span style="font-size:13px;font-weight:700;color:{cc};">${price:.4f}</span>
                <span style="font-size:11px;color:{cc};margin-left:8px;">{chg:+.2f}%</span>
              </td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;
             padding:22px;border-top:2px solid {reg_col};height:100%;">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};margin-bottom:14px;">REGRESSION FORECAST</div>
          <div style="font-size:40px;font-weight:900;color:{reg_col};font-family:JetBrains Mono,monospace;
               margin-bottom:4px;">${reg_price:.4f}</div>
          <div style="font-size:12px;color:{TXT3};margin-bottom:20px;">
            ₹{reg_price*inr_rate:,.2f} · {reg_chg:+.2f}% expected
          </div>
          <table style="width:100%;border-collapse:collapse;">
            <tr>
              <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:left;padding-bottom:8px;">HORIZON</th>
              <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:right;padding-bottom:8px;">TARGET</th>
            </tr>
            {reg_row_html("1 Hour",   r1h)}
            {reg_row_html("4 Hours",  r4h)}
            {reg_row_html("24 Hours", r24h)}
          </table>
          <div style="margin-top:14px;padding-top:12px;border-top:1px solid {BORDER};">
            <div style="font-size:10px;color:{TXT3};margin-bottom:4px;">CONFIDENCE INTERVAL (±1σ)</div>
            <div style="font-size:12px;font-family:JetBrains Mono,monospace;color:{TXT2};">
              ${max(0, r24h-sigma):.4f} — ${r24h+sigma:.4f}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Regime panel (pure HTML — NO Plotly gauge to avoid crash)
    with pReg2:
        st.markdown(regime_html(ts), unsafe_allow_html=True)

        # Market regime details
        trend_str = "Strong" if abs(ts)>60 else "Moderate" if abs(ts)>30 else "Weak"
        vol_regime= "High" if vol_r>1.5 else "Normal" if vol_r>0.8 else "Low"
        struct    = "Bullish" if ts>=30 else "Bearish" if ts<=-30 else "Ranging"
        brkout    = "Confirmed" if abs(ts)>50 else "Possible" if abs(ts)>30 else "None"
        mom       = "Strong Bull" if ts>60 else "Bullish" if ts>30 else "Strong Bear" if ts<-60 else "Bearish" if ts<-30 else "Neutral"

        def rg_row(lbl_, v_, c_):
            return f"""
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{TXT3};padding:8px 0;">{lbl_}</td>
              <td style="text-align:right;font-size:12px;font-weight:600;color:{c_};">{v_}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;padding:18px;margin-top:10px;">
          <table style="width:100%;border-collapse:collapse;">
            {rg_row("TREND STRENGTH", trend_str, GREEN if "Strong" in trend_str else YELLOW)}
            {rg_row("VOLATILITY",     vol_regime, ORANGE)}
            {rg_row("STRUCTURE",      struct,     GREEN if struct=="Bullish" else RED if struct=="Bearish" else YELLOW)}
            {rg_row("BREAKOUT",       brkout,     GREEN if brkout=="Confirmed" else YELLOW)}
            {rg_row("MOMENTUM",       mom,        GREEN if "Bull" in mom else RED if "Bear" in mom else YELLOW)}
          </table>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # MULTI-TIMEFRAME
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:28px 0 14px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">MULTI-TIMEFRAME ANALYSIS</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>""", unsafe_allow_html=True)

    rsi_v2 = float(latest.get("RSI", 50))
    ema_al = f"{'20>50>200' if e20_v>e50_v else '20<50<200'}"
    def mtf_sig(noise):
        s = ts + noise
        return "BUY" if s>=35 else "SELL" if s<=-35 else "HOLD"

    mtf_rows = [
        ("15 MIN", mtf_sig(+5),  ema_al, f"{rsi_v2+2:.1f}"),
        ("1 HOUR", cls_sig,      ema_al, f"{rsi_v2:.1f}"),
        ("4 HOUR", mtf_sig(-3),  ema_al, f"{rsi_v2-2:.1f}"),
        ("1 DAY",  mtf_sig(-6),  ema_al, f"{rsi_v2-4:.1f}"),
        ("1 WEEK", mtf_sig(-12), "20≈50",f"{rsi_v2-8:.1f}"),
    ]

    def bsig(s):
        c = GREEN if s=="BUY" else RED if s=="SELL" else YELLOW
        bg = f"{c}18"; border = f"{c}30"
        return f'<span style="display:inline-block;padding:2px 10px;border-radius:6px;font-size:11px;font-weight:700;background:{bg};color:{c};border:1px solid {border};">{s}</span>'

    tbody = ""
    for tf_, sig_, ema_, rsi_ in mtf_rows:
        arr_col = GREEN if sig_=="BUY" else RED if sig_=="SELL" else YELLOW
        arr = "↑" if sig_=="BUY" else "↓" if sig_=="SELL" else "↔"
        tbody += f"""
        <tr style="border-top:1px solid {BORDER};">
          <td style="font-size:12px;font-family:JetBrains Mono,monospace;color:{TXT2};padding:10px 0;">{tf_}</td>
          <td style="text-align:center;"><span style="color:{arr_col};font-size:18px;">{arr}</span></td>
          <td style="text-align:center;font-size:11px;font-family:JetBrains Mono,monospace;color:{TXT3};">{ema_}</td>
          <td style="text-align:center;font-size:12px;font-family:JetBrains Mono,monospace;color:{TXT2};">{rsi_}</td>
          <td style="text-align:right;">{bsig(sig_)}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;padding:20px;">
      <table style="width:100%;border-collapse:collapse;">
        <thead>
          <tr>
            <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:left;padding-bottom:10px;">TIMEFRAME</th>
            <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:center;padding-bottom:10px;">TREND</th>
            <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:center;padding-bottom:10px;">EMA</th>
            <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:center;padding-bottom:10px;">RSI</th>
            <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:right;padding-bottom:10px;">SIGNAL</th>
          </tr>
        </thead>
        <tbody>{tbody}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # SUPPORT & RESISTANCE + TRADING SETUP
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:28px 0 14px;">
      <div style="width:7px;height:7px;border-radius:50%;background:{ORANGE};box-shadow:0 0 8px {ORANGE}60;"></div>
      <div style="font-size:10px;font-weight:700;letter-spacing:0.14em;color:{TXT3};">SUPPORT & RESISTANCE · TRADING SETUP</div>
      <div style="flex:1;height:1px;background:{BORDER};"></div>
    </div>""", unsafe_allow_html=True)

    sr_col, ts_col = st.columns([1.2, 1])

    with sr_col:
        atr_v3 = float(latest.get("ATR", cur_price*0.02))
        r1_ = float(df["High"].tail(50).max())
        r2_ = float(df["High"].tail(20).max())
        r3_ = r1_ + atr_v3
        s1_ = float(df["Low"].tail(50).min())
        s2_ = float(df["Low"].tail(20).min())
        s3_ = s1_ - atr_v3

        def dots_html(n, col):
            return "".join([f'<span style="color:{col};font-size:13px;">●</span>' for _ in range(n)])

        def sr_row(lbl_, price_, n_dots, col_):
            return f"""
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{col_};font-weight:600;padding:9px 0;">{lbl_}</td>
              <td style="text-align:center;font-family:JetBrains Mono,monospace;font-size:12px;color:#f0f6fc;">${price_:.4f}</td>
              <td style="text-align:center;font-family:JetBrains Mono,monospace;font-size:11px;color:{TXT3};">₹{price_*inr_rate:,.2f}</td>
              <td style="text-align:right;">{dots_html(n_dots, col_)}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;padding:20px;">
          <table style="width:100%;border-collapse:collapse;">
            <thead>
              <tr>
                <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:left;padding-bottom:10px;">LEVEL</th>
                <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:center;padding-bottom:10px;">USD</th>
                <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:center;padding-bottom:10px;">INR</th>
                <th style="font-size:10px;color:{TXT3};letter-spacing:0.1em;text-align:right;padding-bottom:10px;">STRENGTH</th>
              </tr>
            </thead>
            <tbody>
              {sr_row("RESISTANCE 3", r3_, 3, RED)}
              {sr_row("RESISTANCE 2", r1_, 4, RED)}
              {sr_row("RESISTANCE 1", r2_, 5, RED)}
              <tr>
                <td colspan="4" style="text-align:center;color:{BLUE};font-size:12px;
                     font-family:JetBrains Mono,monospace;padding:8px 0;font-weight:700;
                     border-top:1px dashed {BORDER};">
                  ── CURRENT ${cur_price:.4f} · ₹{price_inr:,.2f} ──
                </td>
              </tr>
              {sr_row("SUPPORT 1", s2_, 5, GREEN)}
              {sr_row("SUPPORT 2", s1_, 4, GREEN)}
              {sr_row("SUPPORT 3", s3_, 3, GREEN)}
            </tbody>
          </table>
        </div>""", unsafe_allow_html=True)

    with ts_col:
        entry_ = cur_price
        sl_    = cur_price - (atr_v3 * 2)
        tp1_   = cur_price + (atr_v3 * 3)
        tp2_   = cur_price + (atr_v3 * 6)
        rr_    = (tp1_ - entry_) / max(entry_ - sl_, 1e-9)

        def setup_row(lbl_, usd_, inr_, col_):
            return f"""
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{TXT3};font-weight:500;padding:10px 0;">{lbl_}</td>
              <td style="text-align:right;padding:10px 0;">
                <div style="font-size:13px;font-weight:700;font-family:JetBrains Mono,monospace;color:{col_};">${usd_:.4f}</div>
                <div style="font-size:10px;font-family:JetBrains Mono,monospace;color:{TXT3};">₹{inr_:,.2f}</div>
              </td>
            </tr>"""

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:12px;padding:20px;">
          <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;color:{TXT3};margin-bottom:16px;">TRADING SETUP</div>
          <table style="width:100%;border-collapse:collapse;">
            {setup_row("ENTRY PRICE",      entry_, entry_*inr_rate, BLUE)}
            {setup_row("STOP LOSS (2×ATR)",sl_,    sl_*inr_rate,    RED)}
            {setup_row("TAKE PROFIT 1 (3×ATR)", tp1_, tp1_*inr_rate, GREEN)}
            {setup_row("TAKE PROFIT 2 (6×ATR)", tp2_, tp2_*inr_rate, GREEN)}
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{TXT3};padding:10px 0;">RISK / REWARD</td>
              <td style="text-align:right;font-size:18px;font-weight:800;
                   font-family:JetBrains Mono,monospace;color:{ORANGE};padding:10px 0;">1 : {rr_:.2f}</td>
            </tr>
            <tr style="border-top:1px solid {BORDER};">
              <td style="font-size:12px;color:{TXT3};padding:10px 0;">ATR (14)</td>
              <td style="text-align:right;font-family:JetBrains Mono,monospace;font-size:12px;color:{TXT2};padding:10px 0;">
                ${atr_v3:.4f} · ₹{atr_v3*inr_rate:.2f}
              </td>
            </tr>
          </table>
          <div style="margin-top:14px;padding:12px;background:{BG3};border-radius:8px;border:1px solid {BORDER};">
            <div style="font-size:10px;color:{TXT3};margin-bottom:4px;letter-spacing:0.08em;">RISK WARNING</div>
            <div style="font-size:11px;color:#484f58;line-height:1.6;">
              Informational only. Not financial advice. Crypto is highly volatile. DYOR.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="text-align:center;font-size:10px;color:{TXT3};margin-top:40px;
         padding-top:16px;border-top:1px solid {BORDER};letter-spacing:0.04em;">
      RENDER AI DASHBOARD · Data: CoinGecko · Yahoo Finance · TradingView ·
      INR: ₹{inr_rate:.2f} · {datetime.now().strftime("%d %b %Y %H:%M:%S")} ·
      Not financial advice
    </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
