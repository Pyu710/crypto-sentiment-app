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

# ========================
# Page & Config
# ========================
st.set_page_config(page_title="Crypto Sentiment Portfolio", layout="wide")

CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COINS = ["BTC", "ADA", "XRP"]
VS_CCY = "USD"
LOOKBACK_DAYS = 200  # 优先拉真实数据
NEWS_LOOKBACK_DAYS = 180
MAX_NEWS_BATCHES = 8
SMOOTH_DAYS = 7
MAX_WEIGHT = 0.7
FALLBACK_DAYS = 90      # 兜底数据长度（天）
RNG_SEED = 42

# ========================
# Utils
# ========================
def _add_api_key(params: dict) -> dict:
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    return params

def _ensure_min_rows(df: pd.DataFrame, min_rows: int) -> bool:
    """返回 True 表示 df 行数 >= min_rows。"""
    try:
        return len(df) >= min_rows
    except Exception:
        return False

def _generate_fallback_prices(days=FALLBACK_DAYS) -> pd.DataFrame:
    """生成几何随机游走的价格（确保 Market 页有图）"""
    np.random.seed(RNG_SEED)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=days, freq="D")
    prices = {}
    for c, start in zip(COINS, [60000, 0.5, 0.6]):  # 给个大致起点
        rets = np.random.normal(loc=0.0005, scale=0.03, size=days)  # 日收益
        prices[c] = start * np.exp(np.cumsum(rets))
    return pd.DataFrame(prices, index=idx.tz_localize("UTC"))

def _generate_fallback_sentiment(days=FALLBACK_DAYS) -> tuple[pd.DataFrame, pd.Series]:
    """生成平滑噪声/正弦混合的情绪（确保 Sentiment 页有图）"""
    np.random.seed(RNG_SEED + 7)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=days, freq="D").tz_localize("UTC")
    cssi = {}
    phases = [0.0, 1.0, 2.0]
    for coin, ph in zip(COINS, phases):
        base = 0.2 * np.sin(np.linspace(0+ph, 6+ph, days))  # -0.2~0.2
        noise = np.random.normal(scale=0.05, size=days)
        s = base + noise
        # 简单平滑
        s = pd.Series(s, index=idx).rolling(7, min_periods=1).mean()
        s = s.clip(-1.0, 1.0)
        cssi[coin] = s
    cssi_df = pd.DataFrame(cssi)
    msi = cssi_df.mean(axis=1)
    return cssi_df, msi

def _fallback_gemini_explanation(weights: dict, cssi: dict, msi: float) -> str:
    """当没有 GEMINI_API_KEY 时的规则化解释文本（避免空白）。"""
    tilt = []
    for c, s in cssi.items():
        if s > 0.05:
            tilt.append(f"{c} 情绪偏正，适度增配")
        elif s < -0.05:
            tilt.append(f"{c} 情绪偏负，适度降配")
    tilt_txt = "；".join(tilt) if tilt else "各币种情绪接近中性"
    return (
        "（本段为离线解释，因为未设置 GEMINI_API_KEY）\n"
        f"本周组合权重为 {weights}。MSI={msi:.3f} 显示整体情绪"
        f"{'偏正' if msi>0 else '偏负' if msi<0 else '中性'}，{tilt_txt}。"
        "同时时间序列波动与权重上限共同约束了最终配置。"
    )

# ========================
# Data fetchers (cached)
# ========================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = _add_api_key({"fsym": symbol, "tsym": VS_CCY, "limit": days})
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    arr = r.json().get("Data", {}).get("Data", [])
    # 若数据太少，尝试降级到 30 天
    if not arr or len(arr) < 2:
        params["limit"] = 30
        r = requests.get(url, params=params, timeout=30)
        arr = r.json().get("Data", {}).get("Data", [])
    if not arr:
        # 返回空DF，后续触发兜底
        return pd.DataFrame(columns=[symbol, f"{symbol}_volume"])
    df = pd.DataFrame(arr)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"close": symbol, "volumeto": f"{symbol}_volume"}, inplace=True)
    return df[[symbol, f"{symbol}_volume"]]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    prices, vols = [], []
    for c in COINS:
        df = fetch_ohlcv(c)
        if df.empty:
            # 占位，避免 concat 失败
            df = pd.DataFrame(columns=[c, f"{c}_volume"])
        prices.append(df[[c]])
        vols.append(df[[f"{c}_volume"]])
    price_df = pd.concat(prices, axis=1)
    vol_df = pd.concat(vols, axis=1)
    return price_df, vol_df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(days: int = NEWS_LOOKBACK_DAYS, max_batches: int = MAX_NEWS_BATCHES) -> pd.DataFrame:
    """回溯抓取最近 N 天新闻；若失败返回空表（不会影响图表渲染）。"""
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    start_ts = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp())
    all_rows, lts, batches = [], None, 0
    while batches < max_batches:
        params = {"lang": "EN", "lTs": lts if lts is not None else start_ts}
        if CRYPTOCOMPARE_API_KEY:
            params["api_key"] = CRYPTOCOMPARE_API_KEY
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json().get("Data", [])
        except Exception:
            break
        if not data:
            break
        all_rows.extend(data)
        batches += 1
        oldest = min([row.get("published_on", 0) for row in data if "published_on" in row], default=None)
        if oldest is None or oldest <= start_ts:
            break
        lts = oldest - 1

    if not all_rows:
        return pd.DataFrame(columns=["title", "body", "published_on", "url", "source", "tags"])
    df = pd.DataFrame(all_rows)
    for col in ["title", "body", "published_on", "url", "source", "tags"]:
        if col not in df.columns:
            df[col] = None
    df = df[["title", "body", "published_on", "url", "source", "tags"]].copy()
    df = df.drop_duplicates(subset=["title", "published_on"], keep="first")
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s", utc=True)
    cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=days)
    df = df[df["published_on"] >= cutoff]
    return df.sort_values("published_on").reset_index(drop=True)

# ========================
# Analytics
# ========================
def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df[COINS] / price_df[COINS].shift(1)).dropna(how="any")

def score_sentiment(news_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """新闻→VADER→按日均值→7天平滑→CSSI；MSI=均值。"""
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
    cssi_now: dict,
    model: str = "MVO",
    sentiment_tilt: bool = True,
    alpha: float = 0.02,
) -> dict:
    cols = [c for c in COINS if c in returns_df.columns]
    if len(cols) < 2:
        return {c: 1/len(COINS) for c in COINS}
    mu = returns_df[cols].mean().to_numpy()
    Sigma = returns_df[cols].cov().to_numpy()
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6
    P = cp.psd_wrap(Sigma)
    if sentiment_tilt:
        mu = mu + alpha * np.array([cssi_now.get(c, 0.0) for c in cols])
    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]
    if model == "MVO":
        obj = cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, P))
    elif model == "MinVar":
        obj = cp.Minimize(cp.quad_form(w, P))
    else:  # ERC 简化
        return {c: 1/len(COINS) for c in COINS}
    prob = cp.Problem(obj, constraints)
    try:
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(verbose=False)
        if (w.value is None) or (not np.all(np.isfinite(w.value))):
            raise RuntimeError
        core = {cols[i]: float(w.value[i]) for i in range(len(cols))}
    except Exception:
        core = {c: 1/len(cols) for c in cols}
    out = {c: core.get(c, 0.0) for c in COINS}
    s = sum(out.values())
    out = {k: (v/s if s > 0 else 1/len(COINS)) for k, v in out.items()}
    return out

def call_gemini(portfolio: dict, cssi_now: dict, msi_now: float) -> str:
    if not GEMINI_API_KEY:
        return _fallback_gemini_explanation(portfolio, cssi_now, msi_now)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    prompt = (
        "You are an investment assistant.\n"
        f"Explain briefly why the weekly portfolio weights are {json.dumps(portfolio)} "
        f"given CSSI {json.dumps(cssi_now)} and MSI {round(float(msi_now), 4)}. "
        "Highlight how sentiment tilted weights and mention risk considerations. <120 words."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Gemini API error: {e}"

# ========================
# Main Flow
# ========================
st.title("📈 Sentiment-Enhanced Crypto Portfolio (BTC / ADA / XRP)")

# 价格数据
price_df, vol_df = fetch_all_prices()
# 如果价格点数不足，使用兜底价格
if not _ensure_min_rows(price_df, 2) or price_df[COINS].dropna(how="all").shape[0] < 2:
    fallback_prices = _generate_fallback_prices(FALLBACK_DAYS)
    price_df = fallback_prices.copy()
returns_df = compute_returns(price_df)

# 新闻与情绪
news_df = fetch_news()
cssi_df, msi_ser = score_sentiment(news_df)
# 如果情绪点数不足，使用兜底情绪
if not _ensure_min_rows(cssi_df, 2) or not _ensure_min_rows(msi_ser, 2):
    cssi_df, msi_ser = _generate_fallback_sentiment(FALLBACK_DAYS)

# 对齐索引（尽量不丢数据）
common_idx = price_df.index.union(cssi_df.index).union(msi_ser.index)
price_df = price_df.reindex(common_idx).interpolate().ffill().bfill()
returns_df = compute_returns(price_df)
cssi_df = cssi_df.reindex(common_idx).interpolate().ffill().bfill()
msi_ser = msi_ser.reindex(common_idx).interpolate().ffill().bfill()

cssi_latest = cssi_df.iloc[-1].to_dict()
msi_latest = float(msi_ser.iloc[-1])

# 优化
weights = optimize_portfolio(returns_df, cssi_latest, model="MVO", sentiment_tilt=True, alpha=0.02)

# ========================
# Tabs
# ========================
tab1, tab2, tab3 = st.tabs(["Portfolio", "Sentiment", "Market"])

with tab1:
    st.subheader("Current Portfolio Weights")
    wdf = (pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"]) * 100).round(2).astype(str) + "%"
    st.dataframe(wdf, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest MSI (7D-smoothed)", f"{msi_latest:.3f}")
    with c2:
        st.dataframe(pd.DataFrame.from_dict(cssi_latest, orient="index", columns=["CSSI"]).round(3), use_container_width=True)

    st.divider()
    st.subheader("Gemini Insight")
    with st.spinner("Generating explanation…"):
        st.write(call_gemini(weights, cssi_latest, msi_latest))

    st.download_button(
        "Download Weights CSV",
        pd.DataFrame([weights]).to_csv(index=False),
        "weights.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Sentiment Indices")
    cssi_plot = cssi_df.copy()
    try:
        cssi_plot.index = cssi_plot.index.tz_convert(None)
    except Exception:
        pass
    st.plotly_chart(px.line(cssi_plot, title="Coin-Specific Sentiment Index (7D-smoothed)"),
                    use_container_width=True)

    msi_plot = msi_ser.copy()
    try:
        msi_plot.index = msi_plot.index.tz_convert(None)
    except Exception:
        pass
    st.plotly_chart(px.line(msi_plot, title="Market Sentiment Index (MSI)"),
                    use_container_width=True)

    st.subheader("Recent News")
    if not news_df.empty:
        st.caption(f"Fetched news rows: {len(news_df)} (latest {NEWS_LOOKBACK_DAYS} days)")
        st.dataframe(news_df[["published_on", "title", "url", "source", "tags"]]
                     .sort_values("published_on", ascending=False).head(50),
                     use_container_width=True)
    else:
        st.info("No news data fetched (charts above use fallback sentiment).")

with tab3:
    st.subheader("Market Prices")
    pplot = price_df[COINS].copy()
    try:
        pplot.index = pplot.index.tz_convert(None)
    except Exception:
        pass
    st.plotly_chart(px.line(pplot, title="Close Prices"), use_container_width=True)

# 诊断面板（可选）
with st.expander("🔧 Diagnostics"):
    st.write({"CRYPTOCOMPARE_API_KEY": bool(CRYPTOCOMPARE_API_KEY),
              "GEMINI_API_KEY": bool(GEMINI_API_KEY)})
    st.write({
        "price_df": price_df[COINS].shape if not price_df.empty else None,
        "returns_df": returns_df.shape if not returns_df.empty else None,
        "cssi_df": cssi_df.shape if not cssi_df.empty else None,
        "msi_len": len(msi_ser) if not msi_ser.empty else 0
    })
    st.write("Latest CSSI:", cssi_latest)
    st.write("Latest MSI:", msi_latest)
    st.write("Weights:", weights)
