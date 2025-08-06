# app.py
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import cvxpy as cp

# ------------------------------
# 1. Config
# ------------------------------
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

COINS = ["BTC", "ADA", "XRP"]
VS_CURRENCY = "USD"
LOOKBACK_DAYS = 90

# ------------------------------
# 2. Helper Functions
# ------------------------------

def fetch_ohlcv(symbol, days=LOOKBACK_DAYS):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday"
    params = {
        "fsym": symbol,
        "tsym": VS_CURRENCY,
        "limit": days,
        "api_key": CRYPTOCOMPARE_API_KEY
    }
    r = requests.get(url, params=params)
    data = r.json()["Data"]["Data"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"close": symbol}, inplace=True)
    return df[[symbol]]

def fetch_news():
    url = "https://min-api.cryptocompare.com/data/v2/news/"
    params = {"lang": "EN", "api_key": CRYPTOCOMPARE_API_KEY}
    r = requests.get(url, params=params)
    articles = r.json()["Data"]
    return pd.DataFrame(articles)

def clean_and_score_news(news_df):
    analyzer = SentimentIntensityAnalyzer()
    coin_scores = {c: [] for c in COINS}
    for coin in COINS:
        df_coin = news_df[news_df["body"].str.contains(coin, case=False, na=False)]
        scores = []
        for text in df_coin["body"]:
            vs = analyzer.polarity_scores(text)
            scores.append(vs["compound"])
        if scores:
            coin_scores[coin] = np.mean(scores)
        else:
            coin_scores[coin] = 0.0
    return coin_scores

def compute_cssi(sentiment_dict):
    return sentiment_dict  # dict: {BTC: score, ADA: score, XRP: score}

def compute_msi(cssi):
    return np.mean(list(cssi.values()))

def optimize_portfolio(returns_df, cssi, model="MVO", sentiment_tilt=True):
    mu = returns_df.mean().values
    Sigma = returns_df.cov().values
    w = cp.Variable(len(COINS))
    gamma = cp.Parameter(nonneg=True)
    gamma.value = 1.0

    constraints = [cp.sum(w) == 1, w >= 0, w <= 0.7]

    if sentiment_tilt:
        mu = mu + 0.02 * np.array([cssi[c] for c in COINS])

    if model == "MVO":
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Maximize(ret - gamma*risk), constraints)
    elif model == "MinVar":
        risk = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Minimize(risk), constraints)
    elif model == "ERC":
        # Simple equal weight fallback for demo
        return {COINS[i]: 1/len(COINS) for i in range(len(COINS))}
    prob.solve()
    weights = {COINS[i]: w.value[i] for i in range(len(COINS))}
    return weights

def call_gemini(portfolio, cssi, msi):
    prompt = f"""Explain in plain English why the portfolio weights are {portfolio}
    given that CSSI is {cssi} and MSI is {msi}. Mention sentiment effects."""
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    r = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
        headers=headers,
        json={"contents": [{"parts": [{"text": prompt}]}]}
    )
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Gemini explanation not available."

# ------------------------------
# 3. Streamlit App Layout
# ------------------------------
st.set_page_config(page_title="Crypto Sentiment Portfolio", layout="wide")

st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Optimization Model", ["MVO", "MinVar", "ERC"])
sentiment_tilt = st.sidebar.checkbox("Apply Sentiment Tilt", True)

st.title("📈 Sentiment-Enhanced Crypto Portfolio")

# Fetch Data
st.info("Fetching market data...")
price_data = pd.concat([fetch_ohlcv(c) for c in COINS], axis=1)
returns_df = np.log(price_data / price_data.shift(1)).dropna()

st.info("Fetching news data...")
news_df = fetch_news()
cssi = compute_cssi(clean_and_score_news(news_df))
msi = compute_msi(cssi)

# Optimize
portfolio_weights = optimize_portfolio(returns_df, cssi, model=model_choice, sentiment_tilt=sentiment_tilt)

# Tabs
tab1, tab2, tab3 = st.tabs(["Portfolio", "Sentiment", "Market"])

with tab1:
    st.subheader("Current Portfolio Weights")
    st.table(pd.DataFrame([portfolio_weights], index=["Weight"]))
    st.subheader("Gemini Insight")
    st.write(call_gemini(portfolio_weights, cssi, msi))

with tab2:
    st.subheader("Sentiment Indices")
    st.metric("Market Sentiment Index (MSI)", round(msi, 3))
    st.table(pd.DataFrame.from_dict(cssi, orient="index", columns=["CSSI"]))

with tab3:
    st.subheader("Price Charts")
    fig = px.line(price_data)
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.download_button(
    "Download Weights CSV",
    pd.DataFrame([portfolio_weights]).to_csv(index=False),
    "weights.csv"
)
