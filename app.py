# app.py
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import cvxpy as cp
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------
# 0) Streamlit page config
# ------------------------------
st.set_page_config(page_title="Crypto Sentiment Portfolio", layout="wide")

# ------------------------------
# 1) Config & constants
# ------------------------------
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COINS = ["BTC", "ADA", "XRP"]
VS_CURRENCY = "USD"
LOOKBACK_DAYS = 180  # 历史窗口大一点，更稳定
SENTIMENT_SMOOTH_DAYS = 7  # 与报告一致，7 日平滑
MAX_WEIGHT = 0.7

# ------------------------------
# 2) Helpers (cached)
# ------------------------------
@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """CryptoCompare 日线 OHLCV"""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": VS_CURRENCY,
        "limit": days,
        "api_key": CRYPTOCOMPARE_API_KEY
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("Data", {}).get("Data", [])
    if not data:
        raise RuntimeError(f"No OHLCV for {symbol}")
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("UTC")
    df.set_index("time", inplace=True)
    df.rename(columns={"open":"open","high":"high","low":"low","close":symbol,"volumeto":"volume"}, inplace=True)
    return df[[symbol, "volume"]]

@st.cache_data(ttl=60*30, show_spinner=False)
def fetch_all_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    price_cols = []
    vol_cols = []
    for c in COINS:
        df = fetch_ohlcv(c)
        price_cols.append(df[[c]])
        vol_cols.append(df[["volume"]].rename(columns={"volume": f"{c}_volume"}))
    price_df = pd.concat(price_cols, axis=1)
    vol_df = pd.concat(vol_cols, axis=1)
    return price_df, vol_df

@st.cache_data(ttl=60*15, show_spinner=False)
def fetch_news() -> pd.DataFrame:
    """CryptoCompare News（英文），简单拉取最新若干条"""
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "api_key": CRYPTOCOMPARE_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("Data", [])
    if not data:
        return pd.DataFrame(columns=["title", "body", "published_on", "url"])
    df = pd.DataFrame(data)
    # 统一列
    keep = ["title", "body", "published_on", "url", "tags", "source"]
    for k in keep:
        if k not in df.columns:
            df[k] = None
    df = df[keep].copy()
    # 时间处理
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s", utc=True).dt.tz_convert("UTC")
    df.sort_values("published_on", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def score_sentiment(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    对新闻做 VADER，生成逐日 coin CSSI 和 MSI。
    - 对包含 coin 名称或 ticker 的新闻 body/title 打分
    - 当天无新闻 -> CSSI=0
    - 7 日平滑
    """
    if news_df.empty:
        # 空返回，全 0
        idx = []
        cssi_df = pd.DataFrame(columns=COINS, index=idx).fillna(0.0)
        msi = pd.Series(0.0, index=cssi_df.index)
        return cssi_df, msi

    analyzer = SentimentIntensityAnalyzer()
    news_df["text"] = (news_df["title"].fillna("") + ". " + news_df["body"].fillna("")).str.lower()
    news_df["date"] = news_df["published_on"].dt.floor("D")

    # 针对每个币筛选包含 coin 名称/常见别称 的新闻行
    cssi_daily = {}
    all_dates = sorted(news_df["date"].unique())
    for coin in COINS:
        # 简单匹配（可以扩展：btc|bitcoin 等）
        if coin == "BTC":
            patt = r"\b(btc|bitcoin|btc/usd)\b"
        elif coin == "ADA":
            patt = r"\b(ada|cardano|ada/usd)\b"
        elif coin == "XRP":
            patt = r"\b(xrp|ripple|xrp/usd)\b"
        sub = news_df[news_df["text"].str.contains(patt, na=False, regex=True)]
        # 按天聚合平均 compound 分数
        rows = []
        for d, grp in sub.groupby("date"):
            scores = [analyzer.polarity_scores(t)["compound"] for t in grp["text"].tolist()]
            rows.append((d, np.mean(scores) if scores else 0.0))
        s = pd.Series({d:v for d,v in rows})
        s = s.reindex(all_dates).fillna(0.0)      # 无新闻=0
        s = s.rolling(SENTIMENT_SMOOTH_DAYS, min_periods=1).mean()  # 7日平滑
        cssi_daily[coin] = s

    cssi_df = pd.DataFrame(cssi_daily).fillna(0.0)
    msi = cssi_df.mean(axis=1)
    return cssi_df, msi

def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df[COINS] / price_df[COINS].shift(1)).dropna()

def optimize_portfolio(returns_df: pd.DataFrame,
                       cssi_latest: dict[str, float],
                       model: str = "MVO",
                       sentiment_tilt: bool = True) -> dict[str, float]:
    """
    简化版优化器：
    - MVO: maximize mu^T w - gamma w^T Σ w
    - MinVar: minimize w^T Σ w
    - ERC: 退化为等权（演示用）
    - 约束: sum w = 1, 0<=w<=MAX_WEIGHT
    - 情绪倾斜：mu += alpha * CSSI
    """
    cols = [c for c in COINS if c in returns_df.columns]
    if len(cols) < 2:
        # 数据不足，等权
        return {c: 1/len(COINS) for c in COINS}

    mu = returns_df[cols].mean().values
    Sigma = returns_df[cols].cov().values

    # 情绪倾斜：与报告一致思路（Return Tilt）
    if sentiment_tilt:
        alpha = 0.02  # 可在侧栏调整
        mu = mu + alpha * np.array([cssi_latest[c] for c in cols])

    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]

    if model == "MVO":
        gamma = 1.0
        objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
    elif model == "MinVar":
        objective = cp.Minimize(cp.quad_form(w, Sigma))
    else:  # "ERC" 简化为等权
        eq = {c: 1/len(cols) for c in cols}
        # map back to all COINS
        out = {c: eq.get(c, 0.0) for c in COINS}
        return out

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            raise RuntimeError("Optimization failed")
        weights_core = {cols[i]: float(w.value[i]) for i in range(len(cols))}
    except Exception:
        # 失败则等权兜底
        weights_core = {c: 1/len(cols) for c in cols}

    # 扩展到全币种
    weights_all = {c: weights_core.get(c, 0.0) for c in COINS}
    # 归一 + 边界清理
    s = sum(weights_all.values())
    if s <= 0:
        weights_all = {c: 1/len(COINS) for c in COINS}
    else:
        weights_all = {k: max(0.0, min(MAX_WEIGHT, v/s)) for k, v in weights_all.items()}
        # 再归一
        s2 = sum(weights_all.values())
        weights_all = {k: v/s2 for k, v in weights_all.items()}
    return weights_all

def call_gemini(portfolio: dict, cssi_latest: dict, msi_latest: float) -> str:
    """
    调用 Google Gemini（REST）。返回自然语言解释。
    需在环境变量配置 GEMINI_API_KEY。
    """
    if not GEMINI_API_KEY:
        return "Gemini API key not set. 请在 Streamlit Cloud 的 Advanced settings → Environment variables 添加 GEMINI_API_KEY。"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    prompt = f"""
You are an investment assistant. 
Explain briefly and clearly why the weekly portfolio weights are {json.dumps(portfolio)}
given the latest coin-level CSSI {json.dumps(cssi_latest)} and market MSI {round(float(msi_latest), 4)}.
Highlight how sentiment contributed to tilts (positive → overweight, negative → underweight) 
and mention any risk considerations. Keep it under 120 words.
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {e}"

# ------------------------------
# 3) Sidebar controls
# ------------------------------
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Optimization Model", ["MVO", "MinVar", "ERC"], index=0)
sentiment_tilt = st.sidebar.checkbox("Apply Sentiment Tilt", True)

st.sidebar.caption("Constraints")
st.sidebar.write(f"• Max weight per asset: {MAX_WEIGHT:.0%}")
alpha_user = st.sidebar.slider("Sentiment tilt strength (alpha)", 0.00, 0.05, 0.02, 0.005)

# ------------------------------
# 4) Main Flow
# ------------------------------
st.title("📈 Sentiment-Enhanced Crypto Portfolio (BTC / ADA / XRP)")

# Market data
with st.spinner("Fetching market data…"):
    price_df, vol_df = fetch_all_prices()
returns_df = compute_returns(price_df)

# News & sentiment
with st.spinner("Fetching news and computing sentiment…"):
    news_df = fetch_news()
    cssi_df, msi_ser = score_sentiment(news_df)

# 对齐时间索引（以价格为基准）
common_idx = price_df.index.intersection(cssi_df.index)
cssi_df = cssi_df.reindex(common_idx).fillna(0.0)
msi_ser = msi_ser.reindex(common_idx).fillna(0.0)
price_df = price_df.reindex(common_idx)
returns_df = returns_df.reindex(common_idx).dropna()

# 取最新 CSSI & MSI（用于优化与 Gemini 解释）
cssi_latest = {}
if not cssi_df.empty:
    last_row = cssi_df.iloc[-1]
    cssi_latest = {c: float(last_row.get(c, 0.0)) for c in COINS}
else:
    cssi_latest = {c: 0.0 for c in COINS}
msi_latest = float(msi_ser.iloc[-1]) if not msi_ser.empty else 0.0

# 使用用户侧栏 alpha
def _opt_with_alpha(ret_df, cssi_dict, model, tilt, alpha_override):
    # 使用一个本地拷贝，替换函数内的 alpha
    cols = [c for c in COINS if c in ret_df.columns]
    mu = ret_df[cols].mean().values
    Sigma = ret_df[cols].cov().values
    if tilt:
        mu = mu + alpha_override * np.array([cssi_dict[c] for c in cols])
    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]
    if model == "MVO":
        objective = cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, Sigma))
    elif model == "MinVar":
        objective = cp.Minimize(cp.quad_form(w, Sigma))
    else:
        eq = {c: 1/len(cols) for c in cols}
        return {c: eq.get(c, 0.0) for c in COINS}
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None:
            raise RuntimeError("Optimization failed")
        out_core = {cols[i]: float(w.value[i]) for i in range(len(cols))}
    except Exception:
        out_core = {c: 1/len(cols) for c in cols}
    out_all = {c: out_core.get(c, 0.0) for c in COINS}
    s = sum(out_all.values())
    out_all = {k: (v/s if s>0 else 1/len(COINS)) for k,v in out_all.items()}
    return out_all

portfolio_weights = _opt_with_alpha(returns_df, cssi_latest, model_choice, sentiment_tilt, alpha_user)

# ------------------------------
# 5) UI Tabs
# ------------------------------
tab1, tab2, tab3 = st.tabs(["Portfolio", "Sentiment", "Market"])

with tab1:
    st.subheader("Current Portfolio Weights")
    st.table(pd.DataFrame([portfolio_weights], index=["Weight"]).T.style.format("{:.2%}"))

    st.divider()
    colA, colB = st.columns([1,1])
    with colA:
        st.metric("Latest MSI (7D-smoothed)", f"{msi_latest:.3f}")
    with colB:
        st.write("Latest CSSI")
        st.table(pd.DataFrame([cssi_latest], index=["CSSI"]).T.style.format("{:.3f}"))

    st.divider()
    st.subheader("Gemini Insight")
    with st.spinner("Asking Gemini…"):
        explanation = call_gemini(portfolio_weights, cssi_latest, msi_latest)
    st.write(explanation)

    st.download_button(
        "Download Weights CSV",
        pd.DataFrame(
