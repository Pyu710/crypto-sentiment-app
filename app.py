import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import cvxpy as cp
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========================
# Config
# ========================
st.set_page_config(page_title="Crypto Sentiment Portfolio", layout="wide")

CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COINS = ["BTC", "ADA", "XRP"]
VS_CCY = "USD"
LOOKBACK_DAYS = 200  # 增加请求天数
SMOOTH_DAYS = 7
MAX_WEIGHT = 0.7

# ========================
# Helpers
# ========================
def _add_api_key(params):
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    return params

@st.cache_data(ttl=1800)
def fetch_ohlcv(symbol, days=LOOKBACK_DAYS):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = _add_api_key({"fsym": symbol, "tsym": VS_CCY, "limit": days})
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    arr = r.json().get("Data", {}).get("Data", [])
    if not arr or len(arr) < 2:
        # 如果数据不足，尝试降级到 30 天
        params["limit"] = 30
        r = requests.get(url, params=params, timeout=30)
        arr = r.json().get("Data", {}).get("Data", [])
    if not arr:
        raise RuntimeError(f"No OHLCV data for {symbol}")
    df = pd.DataFrame(arr)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"close": symbol, "volumeto": f"{symbol}_volume"}, inplace=True)
    return df[[symbol, f"{symbol}_volume"]]

@st.cache_data(ttl=900)
def fetch_all_prices():
    prices, vols = [], []
    for c in COINS:
        df = fetch_ohlcv(c)
        prices.append(df[[c]])
        vols.append(df[[f"{c}_volume"]])
    return pd.concat(prices, axis=1), pd.concat(vols, axis=1)

@st.cache_data(ttl=900)
def fetch_news():
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = _add_api_key({"lang": "EN"})
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("Data", [])
    if not data:
        return pd.DataFrame(columns=["title", "body", "published_on", "url", "source", "tags"])
    df = pd.DataFrame(data)
    for col in ["title", "body", "published_on", "url", "source", "tags"]:
        if col not in df.columns:
            df[col] = None
    df = df[["title", "body", "published_on", "url", "source", "tags"]]
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s", utc=True)
    return df.sort_values("published_on").reset_index(drop=True)

def compute_returns(price_df):
    return np.log(price_df[COINS] / price_df[COINS].shift(1)).dropna()

def score_sentiment(news_df):
    if news_df.empty:
        idx = pd.Index([], dtype="datetime64[ns, UTC]")
        return pd.DataFrame(index=idx, columns=COINS).fillna(0.0), pd.Series(0.0, index=idx)

    a = SentimentIntensityAnalyzer()
    news_df["text"] = (news_df["title"].fillna("") + ". " + news_df["body"].fillna("")).str.lower()
    news_df["date"] = news_df["published_on"].dt.floor("D")
    all_dates = sorted(news_df["date"].unique())

    patterns = {
        "BTC": r"\b(btc|bitcoin|btc/usd)\b",
        "ADA": r"\b(ada|cardano|ada/usd)\b",
        "XRP": r"\b(xrp|ripple|xrp/usd)\b",
    }
    cssi = {}
    for coin in COINS:
        sub = news_df[news_df["text"].str.contains(patterns[coin], regex=True, na=False)]
        daily = {d: np.mean([a.polarity_scores(t)["compound"] for t in grp["text"]]) for d, grp in sub.groupby("date")}
        s = pd.Series(daily).reindex(all_dates).fillna(0.0)
        s = s.rolling(SMOOTH_DAYS, min_periods=1).mean()
        cssi[coin] = s

    cssi_df = pd.DataFrame(cssi).fillna(0.0)
    msi = cssi_df.mean(axis=1)
    return cssi_df, msi

def optimize_portfolio(returns_df, cssi_now, model="MVO", sentiment_tilt=True, alpha=0.02):
    cols = [c for c in COINS if c in returns_df.columns]
    if len(cols) < 2:
        return {c: 1/len(COINS) for c in COINS}

    mu = returns_df[cols].mean().to_numpy()
    Sigma = returns_df[cols].cov().to_numpy()
    Sigma = np.nan_to_num(Sigma)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6
    P = cp.psd_wrap(Sigma)

    if sentiment_tilt:
        mu = mu + alpha * np.array([cssi_now.get(c, 0.0) for c in cols])

    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]

    if model == "MVO":
        obj = cp.Maximize(mu @ w - cp.quad_form(w, P))
    elif model == "MinVar":
        obj = cp.Minimize(cp.quad_form(w, P))
    else:
        return {c: 1/len(COINS) for c in COINS}

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if w.value is None or not np.all(np.isfinite(w.value)):
            raise RuntimeError
        core = {cols[i]: float(w.value[i]) for i in range(len(cols))}
    except Exception:
        core = {c: 1/len(COINS) for c in cols}

    out = {c: core.get(c, 0.0) for c in COINS}
    s = sum(out.values())
    out = {k: v/s for k, v in out.items()} if s > 0 else {c: 1/len(COINS) for c in COINS}
    return out

def call_gemini(portfolio, cssi_now, msi_now):
    if not GEMINI_API_KEY:
        return "❌ GEMINI_API_KEY 未设置，请在 Advanced settings → Environment variables 添加。"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "You are an investment assistant.\n"
        f"Explain briefly why the weekly portfolio weights are {json.dumps(portfolio)} "
        f"given CSSI {json.dumps(cssi_now)} and MSI {round(float(msi_now), 4)}."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {e}"

# ========================
# Sidebar
# ========================
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Optimization Model", ["MVO", "MinVar", "ERC"])
sentiment_tilt = st.sidebar.checkbox("Apply Sentiment Tilt", True)
alpha_user = st.sidebar.slider("Sentiment tilt strength (alpha)", 0.00, 0.05, 0.02, 0.005)

# ========================
# Main Flow
# ========================
st.title("📈 Sentiment-Enhanced Crypto Portfolio (BTC / ADA / XRP)")

try:
    price_df, vol_df = fetch_all_prices()
except Exception as e:
    st.error(f"价格数据获取失败: {e}")
    st.stop()

returns_df = compute_returns(price_df)

try:
    news_df = fetch_news()
    cssi_df, msi_ser = score_sentiment(news_df)
except Exception as e:
    st.warning(f"情绪数据获取失败: {e}")
    cssi_df = pd.DataFrame(index=returns_df.index, columns=COINS).fillna(0.0)
    msi_ser = pd.Series(0.0, index=returns_df.index)

common_idx = price_df.index
common_idx = common_idx.intersection(cssi_df.index).intersection(msi_ser.index)

price_df = price_df.reindex(common_idx)
returns_df = returns_df.reindex(common_idx).dropna()
cssi_df = cssi_df.reindex(common_idx).fillna(0.0)
msi_ser = msi_ser.reindex(common_idx).fillna(0.0)

cssi_latest = cssi_df.iloc[-1].to_dict() if not cssi_df.empty else {c: 0.0 for c in COINS}
msi_latest = float(msi_ser.iloc[-1]) if not msi_ser.empty else 0.0

weights = optimize_portfolio(returns_df, cssi_latest, model_choice, sentiment_tilt, alpha_user)

# ========================
# Tabs
# ========================
tab1, tab2, tab3 = st.tabs(["Portfolio", "Sentiment", "Market"])

with tab1:
    st.subheader("Current Portfolio Weights")
    st.dataframe((pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"])*100).round(2).astype(str)+"%")
    st.subheader("Gemini Insight")
    st.write(call_gemini(weights, cssi_latest, msi_latest))
    st.download_button("Download Weights CSV", pd.DataFrame([weights]).to_csv(index=False), "weights.csv")

with tab2:
    st.subheader("Sentiment Indices")
    if len(cssi_df) > 1:
        st.plotly_chart(px.line(cssi_df, title="CSSI (7D-smoothed)"), use_container_width=True)
        st.plotly_chart(px.line(msi_ser, title="MSI"), use_container_width=True)
    else:
        st.info("情绪数据不足，无法绘制曲线。")
    st.subheader("Recent News")
    if not news_df.empty:
        st.dataframe(news_df[["published_on", "title", "url", "source", "tags"]].sort_values("published_on", ascending=False).head(50))
    else:
        st.info("暂无新闻数据。")

with tab3:
    st.subheader("Market Prices")
    if len(price_df) > 1:
        st.plotly_chart(px.line(price_df[COINS], title="Close Prices"), use_container_width=True)
    else:
        st.info("价格数据不足，无法绘制曲线。")
