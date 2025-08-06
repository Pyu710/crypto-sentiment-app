# app.py
import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import cvxpy as cp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------ #
# Page config
# ------------------------------ #
st.set_page_config(page_title="Crypto Sentiment Portfolio", layout="wide")

# ------------------------------ #
# Config
# ------------------------------ #
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COINS = ["BTC", "ADA", "XRP"]
VS_CCY = "USD"
LOOKBACK_DAYS = 180
SMOOTH_DAYS = 7
MAX_WEIGHT = 0.7

# ------------------------------ #
# Data helpers
# ------------------------------ #
@st.cache_data(ttl=1800)
def fetch_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol, "tsym": VS_CCY, "limit": days, "api_key": CRYPTOCOMPARE_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    arr = r.json().get("Data", {}).get("Data", [])
    if not arr:
        raise RuntimeError(f"No OHLCV for {symbol}")
    df = pd.DataFrame(arr)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    # 仅保留收盘价列，命名为币种名；同时保留成交量以备后用
    df.rename(columns={"close": symbol, "volumeto": f"{symbol}_volume"}, inplace=True)
    return df[[symbol, f"{symbol}_volume"]]

@st.cache_data(ttl=1800)
def fetch_all_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = []
    vols = []
    for c in COINS:
        df = fetch_ohlcv(c)
        prices.append(df[[c]])
        vols.append(df[[f"{c}_volume"]])
    price_df = pd.concat(prices, axis=1)
    vol_df = pd.concat(vols, axis=1)
    return price_df, vol_df

@st.cache_data(ttl=900)
def fetch_news() -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "api_key": CRYPTOCOMPARE_API_KEY}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("Data", [])
    if not data:
        return pd.DataFrame(columns=["title", "body", "published_on", "url", "source", "tags"])
    df = pd.DataFrame(data)
    for col in ["title", "body", "published_on", "url", "source", "tags"]:
        if col not in df.columns:
            df[col] = None
    df = df[["title", "body", "published_on", "url", "source", "tags"]].copy()
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s", utc=True)
    df.sort_values("published_on", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df[COINS] / price_df[COINS].shift(1)).dropna()

def score_sentiment(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """VADER 对新闻打分，产出逐日 CSSI（7日平滑）与 MSI。"""
    if news_df.empty:
        empty_idx = pd.Index([], dtype="datetime64[ns, UTC]")
        return pd.DataFrame(index=empty_idx, columns=COINS).fillna(0.0), pd.Series(0.0, index=empty_idx)

    a = SentimentIntensityAnalyzer()
    news_df["text"] = (news_df["title"].fillna("") + ". " + news_df["body"].fillna("")).str.lower()
    news_df["date"] = news_df["published_on"].dt.floor("D")
    all_dates = sorted(news_df["date"].unique())

    cssi = {}
    patterns = {
        "BTC": r"\b(btc|bitcoin|btc/usd)\b",
        "ADA": r"\b(ada|cardano|ada/usd)\b",
        "XRP": r"\b(xrp|ripple|xrp/usd)\b",
    }
    for coin in COINS:
        sub = news_df[news_df["text"].str.contains(patterns[coin], regex=True, na=False)]
        daily = {}
        for d, grp in sub.groupby("date"):
            comp = [a.polarity_scores(t)["compound"] for t in grp["text"].tolist()]
            daily[d] = float(np.mean(comp)) if comp else 0.0
        s = pd.Series(daily)
        s = s.reindex(all_dates).fillna(0.0)
        s = s.rolling(SMOOTH_DAYS, min_periods=1).mean()
        cssi[coin] = s

    cssi_df = pd.DataFrame(cssi).fillna(0.0)
    msi = cssi_df.mean(axis=1)
    return cssi_df, msi

def optimize_portfolio(
    returns_df: pd.DataFrame,
    cssi_now: dict[str, float],
    model: str = "MVO",
    sentiment_tilt: bool = True,
    alpha: float = 0.02,
) -> dict[str, float]:
    """MVO / MinVar /（ERC 简化为等权）。约束：sum w=1，0<=w<=MAX_WEIGHT"""
    cols = [c for c in COINS if c in returns_df.columns]
    if len(cols) < 2:
        return {c: 1/len(COINS) for c in COINS}

    mu = returns_df[cols].mean().values
    Sigma = returns_df[cols].cov().values

    if sentiment_tilt:
        mu = mu + alpha * np.array([cssi_now.get(c, 0.0) for c in cols])

    # 变量与约束
    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]

    if model == "MVO":
        objective = cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, Sigma))
    elif model == "MinVar":
        objective = cp.Minimize(cp.quad_form(w, Sigma))
    else:
        # ERC 简化：等权
        eq = {c: 1/len(cols) for c in cols}
        return {c: eq.get(c, 0.0) for c in COINS}

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()  # 让 cvxpy 自动选择可用求解器
        if w.value is None:
            raise RuntimeError("Optimization failed")
        core = {cols[i]: float(w.value[i]) for i in range(len(cols))}
    except Exception:
        core = {c: 1/len(cols) for c in cols}

    out = {c: core.get(c, 0.0) for c in COINS}
    s = sum(out.values())
    if s <= 0:
        out = {c: 1/len(COINS) for c in COINS}
    else:
        out = {k: max(0.0, min(MAX_WEIGHT, v/s)) for k, v in out.items()}
        s2 = sum(out.values())
        out = {k: v/s2 for k, v in out.items()}
    return out

def call_gemini(portfolio: dict, cssi_now: dict, msi_now: float) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key not set. 请在 Streamlit Cloud 的 Advanced settings → Environment variables 添加 GEMINI_API_KEY。"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "You are an investment assistant.\n"
        f"Explain briefly why the weekly portfolio weights are {json.dumps(portfolio)} "
        f"given CSSI {json.dumps(cssi_now)} and MSI {round(float(msi_now), 4)}. "
        "Highlight how sentiment tilted weights (positive → overweight, negative → underweight) "
        "and mention risk considerations. Keep it under 120 words."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {e}"

# ------------------------------ #
# Sidebar
# ------------------------------ #
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Optimization Model", ["MVO", "MinVar", "ERC"], index=0)
sentiment_tilt = st.sidebar.checkbox("Apply Sentiment Tilt", True)
alpha_user = st.sidebar.slider("Sentiment tilt strength (alpha)", 0.00, 0.05, 0.02, 0.005)
st.sidebar.caption(f"Max weight per asset: {int(MAX_WEIGHT*100)}%")

# ------------------------------ #
# Main flow
# ------------------------------ #
st.title("📈 Sentiment-Enhanced Crypto Portfolio (BTC / ADA / XRP)")

with st.spinner("Fetching market data…"):
    price_df, vol_df = fetch_all_prices()
returns_df = compute_returns(price_df)

with st.spinner("Fetching news and computing sentiment…"):
    news_df = fetch_news()
    cssi_df, msi_ser = score_sentiment(news_df)

# 对齐索引（以价格为准）
common_idx = price_df.index
if not cssi_df.empty:
    common_idx = common_idx.intersection(cssi_df.index)
if not msi_ser.empty:
    common_idx = common_idx.intersection(msi_ser.index)

price_df = price_df.reindex(common_idx)
returns_df = returns_df.reindex(common_idx).dropna()
cssi_df = cssi_df.reindex(common_idx).fillna(0.0)
msi_ser = msi_ser.reindex(common_idx).fillna(0.0)

# 取最新 CSSI / MSI
if not cssi_df.empty:
    cssi_latest_series = cssi_df.iloc[-1]
    cssi_latest = {c: float(cssi_latest_series.get(c, 0.0)) for c in COINS}
else:
    cssi_latest = {c: 0.0 for c in COINS}
msi_latest = float(msi_ser.iloc[-1]) if not msi_ser.empty else 0.0

# 组合优化
weights = optimize_portfolio(
    returns_df=returns_df,
    cssi_now=cssi_latest,
    model=model_choice,
    sentiment_tilt=sentiment_tilt,
    alpha=alpha_user,
)

# ------------------------------ #
# Tabs
# ------------------------------ #
tab1, tab2, tab3 = st.tabs(["Portfolio", "Sentiment", "Market"])

with tab1:
    st.subheader("Current Portfolio Weights")
    weights_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"]).sort_index()
    # 显示百分比（用新列避免链式格式化带来的括号嵌套）
    weights_view = (weights_df * 100).round(2).astype(str) + "%"
    st.dataframe(weights_view, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest MSI (7D-smoothed)", f"{msi_latest:.3f}")
    with c2:
        cssi_view = pd.DataFrame.from_dict(cssi_latest, orient="index", columns=["CSSI"]).round(3)
        st.dataframe(cssi_view, use_container_width=True)

    st.divider()
    st.subheader("Gemini Insight")
    with st.spinner("Asking Gemini…"):
        explanation = call_gemini(weights, cssi_latest, msi_latest)
    st.write(explanation)

    st.download_button(
        label="Download Weights CSV",
        data=pd.DataFrame([weights]).to_csv(index=False),
        file_name="weights.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Sentiment Indices")
    if not cssi_df.empty:
        # 清理时区方便作图
        cssi_plot = cssi_df.copy()
        try:
            cssi_plot.index = cssi_plot.index.tz_convert(None)
        except Exception:
            pass
        fig_cssi = px.line(cssi_plot, title="Coin-Specific Sentiment Index (7D-smoothed)")
        st.plotly_chart(fig_cssi, use_container_width=True)

        msi_plot = msi_ser.copy()
        try:
            msi_plot.index = msi_plot.index.tz_convert(None)
        except Exception:
            pass
        fig_msi = px.line(msi_plot, title="Market Sentiment Index (MSI)")
        st.plotly_chart(fig_msi, use_container_width=True)
    else:
        st.info("No sentiment data yet.")

    st.subheader("Recent News")
    if not news_df.empty:
        show_cols = ["published_on", "title", "url", "source", "tags"]
        st.dataframe(news_df[show_cols].sort_values("published_on", ascending=False).head(50), use_container_width=True)
    else:
        st.info("No news pulled from API.")

with tab3:
    st.subheader("Market Prices")
    if not price_df.empty:
        pplot = price_df[COINS].copy()
        try:
            pplot.index = pplot.index.tz_convert(None)
        except Exception:
            pass
        fig_px = px.line(pplot, title="Close Prices")
        st.plotly_chart(fig_px, use_container_width=True)
    else:
        st.info("Price data not available.")
