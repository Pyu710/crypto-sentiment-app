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
# Helpers
# ------------------------------ #
def _add_api_key(params: dict) -> dict:
    """Attach CryptoCompare API key only if provided."""
    if CRYPTOCOMPARE_API_KEY:
        params = {**params, "api_key": CRYPTOCOMPARE_API_KEY}
    return params

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch daily OHLCV (we keep close price as symbol column and volume)."""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = _add_api_key({"fsym": symbol, "tsym": VS_CCY, "limit": days})
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    arr = r.json().get("Data", {}).get("Data", [])
    if not arr:
        raise RuntimeError(f"No OHLCV for {symbol}")
    df = pd.DataFrame(arr)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"close": symbol, "volumeto": f"{symbol}_volume"}, inplace=True)
    return df[[symbol, f"{symbol}_volume"]]

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_all_prices() -> tuple:
    prices, vols = [], []
    for c in COINS:
        df = fetch_ohlcv(c)
        prices.append(df[[c]])
        vols.append(df[[f"{c}_volume"]])
    price_df = pd.concat(prices, axis=1)
    vol_df = pd.concat(vols, axis=1)
    return price_df, vol_df

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news() -> pd.DataFrame:
    """Fetch recent crypto news (EN) from CryptoCompare."""
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
    df = df[["title", "body", "published_on", "url", "source", "tags"]].copy()
    df["published_on"] = pd.to_datetime(df["published_on"], unit="s", utc=True)
    df.sort_values("published_on", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return np.log(price_df[COINS] / price_df[COINS].shift(1)).dropna(how="any")

def score_sentiment(news_df: pd.DataFrame) -> tuple:
    """VADER on news -> daily CSSI (7D smooth) and MSI."""
    if news_df.empty:
        empty_idx = pd.Index([], dtype="datetime64[ns, UTC]")
        return pd.DataFrame(index=empty_idx, columns=COINS).fillna(0.0), pd.Series(0.0, index=empty_idx)

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
        s = s.rolling(SMOOTH_DAYS, min_periods=1).mean()  # 7-day smoothing
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
    """MVO / MinVar / (ERC simplified to EW). Constraints: sum w=1, 0<=w<=MAX_WEIGHT."""
    cols = [c for c in COINS if c in returns_df.columns]
    if len(cols) < 2:
        return {c: 1/len(COINS) for c in COINS}

    # Mean & covariance (robust)
    mu = returns_df[cols].mean().to_numpy()
    Sigma = returns_df[cols].cov().to_numpy()
    Sigma = np.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma += np.eye(Sigma.shape[0]) * 1e-6
    P = cp.psd_wrap(Sigma)

    # Sentiment return tilt
    if sentiment_tilt:
        mu = mu + alpha * np.array([cssi_now.get(c, 0.0) for c in cols])

    # Variable & constraints
    w = cp.Variable(len(cols))
    constraints = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT]

    # Objective
    if model == "MVO":
        objective = cp.Maximize(mu @ w - 1.0 * cp.quad_form(w, P))
    elif model == "MinVar":
        objective = cp.Minimize(cp.quad_form(w, P))
    else:
        eq = {c: 1/len(cols) for c in cols}
        return {c: eq.get(c, 0.0) for c in COINS}

    prob = cp.Problem(objective, constraints)
    try:
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except Exception:
            prob.solve(verbose=False)

        if (w.value is None) or (not np.all(np.isfinite(w.value))):
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

# Fetch market data
with st.spinner("Fetching market data…"):
    try:
        price_df, vol_df = fetch_all_prices()
    except Exception as e:
        st.error(f"Failed to fetch OHLCV data: {e}")
        st.stop()

returns_df = compute_returns(price_df)

# Fetch news & build sentiment
with st.spinner("Fetching news and computing sentiment…"):
    try:
        news_df = fetch_news()
        cssi_df, msi_ser = score_sentiment(news_df)
    except Exception as e:
        st.warning(f"News sentiment unavailable: {e}")
        # Use zero sentiment if news fails
        cssi_df = pd.DataFrame(index=returns_df.index, columns=COINS).fillna(0.0)
        msi_ser = pd.Series(0.0, index=returns_df.index)

# Align indices
common_idx = price_df.index
if not cssi_df.empty:
    common_idx = common_idx.intersection(cssi_df.index)
if not msi_ser.empty:
    common_idx = common_idx.intersection(msi_ser.index)

price_df = price_df.reindex(common_idx)
returns_df = returns_df.reindex(common_idx).dropna(how="any")
cssi_df = cssi_df.reindex(common_idx).fillna(0.0)
msi_ser = msi_ser.reindex(common_idx).fillna(0.0)

# Latest CSSI/MSI
if not cssi_df.empty:
    last_cssi = cssi_df.iloc[-1]
    cssi_latest = {c: float(last_cssi.get(c, 0.0)) for c in COINS}
else:
    cssi_latest = {c: 0.0 for c in COINS}
msi_latest = float(msi_ser.iloc[-1]) if not msi_ser.empty else 0.0

# Optimize
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
        st.dataframe(
            news_df[show_cols].sort_values("published_on", ascending=False).head(50),
            use_container_width=True
        )
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
# ========== Diagnostics Tab ==========
with st.expander("🔧 Diagnostics (for debugging)"):
    st.write("Environment keys present?", {
        "CRYPTOCOMPARE_API_KEY": bool(CRYPTOCOMPARE_API_KEY),
        "GEMINI_API_KEY": bool(GEMINI_API_KEY)
    })
    st.write("Data shapes:", {
        "price_df": price_df[COINS].shape if not price_df.empty else None,
        "returns_df": returns_df.shape if not returns_df.empty else None,
        "cssi_df": cssi_df.shape if not cssi_df.empty else None,
        "msi_len": len(msi_ser) if not msi_ser.empty else 0
    })
    st.write("Latest CSSI:", cssi_latest)
    st.write("Latest MSI:", msi_latest)
    st.write("Weights:", weights)
    st.write("News rows (head 3):")
    if not news_df.empty:
        st.dataframe(news_df.head(3))
    else:
        st.write("No news data fetched.")

