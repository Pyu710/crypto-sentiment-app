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
LOOKBACK_DAYS = 200          # 优先拉真实价格
NEWS_LOOKBACK_DAYS = 180     # 新闻回溯天数
MAX_NEWS_BATCHES = 8         # 新闻最多回溯批次
SMOOTH_DAYS = 7              # CSSI 平滑
MAX_WEIGHT = 0.7
FALLBACK_DAYS = 90           # 兜底数据长度
RNG_SEED = 42

# ========================
# Utils
# ========================
def _add_api_key(params: dict) -> dict:
    if CRYPTOCOMPARE_API_KEY:
        params["api_key"] = CRYPTOCOMPARE_API_KEY
    return params

def _ensure_min_rows(df: pd.DataFrame, min_rows: int) -> bool:
    try:
        return len(df) >= min_rows
    except Exception:
        return False

def _generate_fallback_prices(days=FALLBACK_DAYS) -> pd.DataFrame:
    """几何随机游走价格，保证 Market 页有图"""
    np.random.seed(RNG_SEED)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=days, freq="D")
    prices = {}
    for c, start in zip(COINS, [60000, 0.5, 0.6]):
        rets = np.random.normal(loc=0.0005, scale=0.03, size=days)  # 日收益
        prices[c] = start * np.exp(np.cumsum(rets))
    return pd.DataFrame(prices, index=idx.tz_localize("UTC"))

def _generate_fallback_sentiment(days=FALLBACK_DAYS) -> tuple[pd.DataFrame, pd.Series]:
    """平滑噪声/正弦混合的 CSSI/MSI，保证 Sentiment 页有图"""
    np.random.seed(RNG_SEED + 7)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=days, freq="D").tz_localize("UTC")
    cssi = {}
    phases = [0.0, 1.0, 2.0]
    for coin, ph in zip(COINS, phases):
        base = 0.2 * np.sin(np.linspace(0+ph, 6+ph, days))
        noise = np.random.normal(scale=0.05, size=days)
        s = base + noise
        s = pd.Series(s, index=idx).rolling(7, min_periods=1).mean().clip(-1.0, 1.0)
        cssi[coin] = s
    cssi_df = pd.DataFrame(cssi)
    msi = cssi_df.mean(axis=1)
    return cssi_df, msi

def _fallback_gemini_explanation(weights: dict, cssi: dict, msi: float) -> str:
    tilt = []
    for c, s in cssi.items():
        if s > 0.05:
            tilt.append(f"{c} 情绪偏正，适度增配")
        elif s < -0.05:
            tilt.append(f"{c} 情绪偏负，适度降配")
    tilt_txt = "；".join(tilt) if tilt else "各币种情绪接近中性"
    return (
        "（离线解释：未设置 GEMINI_API_KEY）\n"
        f"本周权重 {weights}。MSI={msi:.3f} 显示整体情绪"
        f"{'偏正' if msi>0 else '偏负' if msi<0 else '中性'}，{tilt_txt}。"
        "同时波动与权重上限共同约束最终配置。"
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
    if not arr or len(arr) < 2:
        params["limit"] = 30
        r = requests.get(url, params=params, timeout=30)
        arr = r.json().get("Data", {}).get("Data", [])
    if not arr:
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
            df = pd.DataFrame(columns=[c, f"{c}_volume"])
        prices.append(df[[c]])
        vols.append(df[[f"{c}_volume"]])
    return pd.concat(prices, axis=1), pd.concat(vols, axis=1)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(days: int = NEWS_LOOKBACK_DAYS, max_batches: int = MAX_NEWS_BATCHES) -> pd.DataFrame:
    """回溯抓取最近 N 天新闻；失败返回空表（不影响图表渲染）"""
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
# Backtest (weekly)
# ========================
def resample_weekly_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    wk = price_df[COINS].resample("W-FRI").last()
    # 若有缺口，线性插值并前后填充
    wk = wk.interpolate().ffill().bfill()
    return wk

def weekly_backtest(
    price_df: pd.DataFrame,
    cssi_df: pd.DataFrame,
    msi_ser: pd.Series,
    lookback_weeks: int = 12,
    model: str = "MVO",
    alpha: float = 0.02,
    commission: float = 0.0010,
    slippage: float = 0.0005,
    sentiment_tilt: bool = True,
) -> dict:
    """
    周频回测：
    - 每周末（W-FRI）用过去 lookback_weeks 的周收益估计 mu/Sigma
    - CSSI/MSI 取该周末的值（CSSI 已日内7日平滑）
    - 本周开盘调仓 -> 本周净值用新权重，扣除换手成本
    """
    # 周价格与收益
    wk_px = resample_weekly_prices(price_df)
    wk_ret = wk_px.pct_change().dropna()
    if wk_ret.shape[0] < max(lookback_weeks + 2, 4):
        # 数据太少，返回空
        return {"curve": pd.Series(dtype=float), "table": pd.DataFrame(), "turnover": [], "costs": []}

    # 周 CSSI/MSI 对齐到周频
    wk_cssi = cssi_df.resample("W-FRI").last().reindex(wk_ret.index).ffill().fillna(0.0)
    wk_msi = msi_ser.resample("W-FRI").last().reindex(wk_ret.index).ffill().fillna(0.0)

    # 回测主循环
    idx = wk_ret.index
    n = len(idx)
    ew = np.array([1.0/len(COINS)]*len(COINS))
    w_prev = ew.copy()
    curve = [1.0]
    rets = []
    turnovers, costs = [], []
    rows = []

    cost_rate = commission + slippage

    for i in range(n):
        date_i = idx[i]
        # 用过去 lookback_weeks 的周收益来估计参数（避免前瞻）
        if i < lookback_weeks:
            w_now = ew.copy()
        else:
            hist = wk_ret.iloc[i - lookback_weeks:i]
            cssi_now = wk_cssi.iloc[i].to_dict()
            w_now = optimize_portfolio(hist, cssi_now, model=model, sentiment_tilt=sentiment_tilt, alpha=alpha)
            w_now = np.array([w_now.get(c, 0.0) for c in COINS])

        # 本周收益用“本周资产收益 * 本周权重”，并在调仓时收取成本
        r_vec = wk_ret.iloc[i].to_numpy()
        turnover = float(np.sum(np.abs(w_now - w_prev)))
        cost = turnover * cost_rate
        r_gross = float(np.dot(w_now, r_vec))
        r_net = r_gross - cost

        rets.append(r_net)
        turnovers.append(turnover)
        costs.append(cost)

        curve.append(curve[-1] * (1.0 + r_net))
        w_prev = w_now.copy()

        rows.append({
            "date": date_i,
            "gross_return": r_gross,
            "net_return": r_net,
            "turnover": turnover,
            "cost": cost,
            "MSI": float(wk_msi.loc[date_i]),
            **{f"w_{c}": float(w_now[j]) for j, c in enumerate(COINS)}
        })

    curve = pd.Series(curve[1:], index=idx, name="Portfolio")
    rets = pd.Series(rets, index=idx, name="weekly_net_return")
    trades = pd.DataFrame(rows).set_index("date")

    # 计算指标
    weeks = len(rets)
    if weeks > 1:
        ann_return = (1 + rets).prod() ** (52/weeks) - 1
        ann_vol = rets.std(ddof=0) * np.sqrt(52)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        cum = (1 + rets).cumprod()
        peak = cum.cummax()
        mdd = (cum/peak - 1).min()
        avg_turnover = np.mean(turnovers)
        avg_cost = np.mean(costs)
    else:
        ann_return = ann_vol = sharpe = mdd = avg_turnover = avg_cost = np.nan

    table = pd.DataFrame({
        "Annualized Return": [ann_return],
        "Annualized Volatility": [ann_vol],
        "Sharpe": [sharpe],
        "Max Drawdown": [mdd],
        "Avg Weekly Turnover": [avg_turnover],
        "Avg Weekly Cost Drag": [avg_cost],
    })
    return {
        "curve": curve,
        "stats": table,
        "rets": rets,
        "trades": trades,
        "turnover": turnovers,
        "costs": costs,
    }

# ========================
# Sidebar
# ========================
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Optimization Model", ["MVO", "MinVar", "ERC"], index=0)
sentiment_tilt = st.sidebar.checkbox("Apply Sentiment Tilt", True)
alpha_user = st.sidebar.slider("Sentiment tilt strength (alpha)", 0.00, 0.05, 0.02, 0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Backtest")
lookback_weeks = st.sidebar.slider("Lookback window (weeks)", 4, 26, 12, 1)
commission = st.sidebar.number_input("Commission per trade", value=0.0010, step=0.0005, format="%.4f")
slippage = st.sidebar.number_input("Slippage per trade", value=0.0005, step=0.0005, format="%.4f")

# ========================
# Main Flow
# ========================
st.title("📈 Sentiment-Enhanced Crypto Portfolio (BTC / ADA / XRP)")

# 价格数据
price_df, vol_df = fetch_all_prices()
if not _ensure_min_rows(price_df, 2) or price_df[COINS].dropna(how="all").shape[0] < 2:
    price_df = _generate_fallback_prices(FALLBACK_DAYS)
returns_df = compute_returns(price_df)

# 新闻与情绪
news_df = fetch_news()
cssi_df, msi_ser = score_sentiment(news_df)
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

# 即时优化（展示）
weights = optimize_portfolio(returns_df, cssi_latest, model=model_choice, sentiment_tilt=sentiment_tilt, alpha=alpha_user)

# 周频回测
bt = weekly_backtest(
    price_df=price_df,
    cssi_df=cssi_df,
    msi_ser=msi_ser,
    lookback_weeks=lookback_weeks,
    model=model_choice,
    alpha=alpha_user,
    commission=commission,
    slippage=slippage,
    sentiment_tilt=sentiment_tilt,
)

# ========================
# Tabs
# ========================
tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Sentiment", "Market", "Backtest"])

with tab1:
    st.subheader("Current Portfolio Weights")
    wdf = (pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"]) * 100).round(2).astype(str) + "%"
    st.dataframe(wdf, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Latest MSI (7D-smoothed)", f"{msi_latest:.3f}")
    with c2:
        st.dataframe(pd.DataFrame.from_dict(cssi_latest, orient="index", columns=["CSSI"]).round(3),
                     use_container_width=True)

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

    st.subheader("ΔCSSI Heatmap (Daily change)")
    delta_cssi = cssi_df.diff().tail(180)  # 近 180 天
    if not delta_cssi.empty:
        delta_plot = delta_cssi.copy()
        try:
            delta_plot.index = delta_plot.index.tz_convert(None)
        except Exception:
            pass
        # 为了更好的色带中心，限制到 [-0.5, 0.5]
        zmin, zmax = -0.5, 0.5
        hm = px.imshow(
            delta_plot.T,
            aspect="auto",
            origin="lower",
            title="ΔCSSI Heatmap (last 180 days)",
            zmin=zmin, zmax=zmax,
        )
        st.plotly_chart(hm, use_container_width=True)
    else:
        st.info("ΔCSSI 数据不足。")

    st.subheader("Recent News")
    if not news_df.empty:
        st.caption(f"Fetched news rows: {len(news_df)} (latest {NEWS_LOOKBACK_DAYS} days)")
        st.dataframe(
            news_df[["published_on", "title", "url", "source", "tags"]]
            .sort_values("published_on", ascending=False).head(50),
            use_container_width=True
        )
    else:
        st.info("No news data fetched（charts may use fallback sentiment）.")

with tab3:
    st.subheader("Market Prices")
    pplot = price_df[COINS].copy()
    try:
        pplot.index = pplot.index.tz_convert(None)
    except Exception:
        pass
    st.plotly_chart(px.line(pplot, title="Close Prices"), use_container_width=True)

with tab4:
    st.subheader("Weekly Backtest (Net of costs)")
    curve = bt["curve"]
    stats = bt["stats"]
    rets = bt["rets"]
    trades = bt["trades"]

    if curve.empty:
        st.info("回测数据不足，请尝试降低 lookback 或等待更多历史。")
    else:
        # 累计收益曲线
        curve_plot = curve.copy()
        try:
            curve_plot.index = curve_plot.index.tz_convert(None)
        except Exception:
            pass
        st.plotly_chart(px.line(curve_plot, title="Cumulative Return (Weekly, net)"),
                        use_container_width=True)

        # 指标表
        if not stats.empty:
            st.subheader("Performance Metrics")
            fmt = {
                "Annualized Return": "{:.2%}".format,
                "Annualized Volatility": "{:.2%}".format,
                "Sharpe": "{:.2f}".format,
                "Max Drawdown": "{:.2%}".format,
                "Avg Weekly Turnover": "{:.2%}".format,
                "Avg Weekly Cost Drag": "{:.2%}".format,
            }
            stats_view = stats.copy()
            for c, f in fmt.items():
                if c in stats_view.columns and pd.notnull(stats_view[c].iloc[0]):
                    stats_view[c] = stats_view[c].apply(lambda x: f(x) if pd.notnull(x) else "NA")
            st.table(stats_view)

        # 交易/权重与成本概览（近 10 周）
        if not trades.empty:
            st.subheader("Recent Rebalances (last 10 weeks)")
            show_cols = ["gross_return", "net_return", "turnover", "cost", "MSI"] + [f"w_{c}" for c in COINS]
            tail = trades[show_cols].tail(10).copy()
            try:
                tail.index = tail.index.tz_convert(None)
            except Exception:
                pass
            st.dataframe(tail.style.format({
                "gross_return": "{:.2%}",
                "net_return": "{:.2%}",
                "turnover": "{:.2%}",
                "cost": "{:.2%}",
                **{f"w_{c}": "{:.2%}" for c in COINS}
            }), use_container_width=True)

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
