"""
Crypto Forecast Lab: probabilistic forecasts for any coin on Coinbase.

Pick a coin and a timeframe. The app forecasts:
  1. Direction   - the chance the price ends higher (ensemble of two machine-learning models)
  2. Price range - a middle estimate plus 50% and 80% ranges (quantile models, calibrated
                   against the model's own past errors)
  3. Volatility  - how big the swings should be (a GARCH(1,1) model fit by maximum likelihood)

Then it grades itself: it replays the past ("walk-forward" testing, never peeking at the
future) and shows how often its forecasts were right, so you know how much to trust them.

Data (all free, no API keys):
  Coinbase (price, volume), FRED (S&P 500, Nasdaq, VIX, 10-year yield, US dollar),
  Alternative.me (Crypto Fear & Greed Index), GDELT (crypto news tone),
  Wikipedia (daily page views as a public-attention signal).

Setup (once):  pip install streamlit coinbase-advanced-py pandas numpy scikit-learn scipy requests
Run:           streamlit run streamlit_app.py   (this file is one page of the combined app)
"""

import io
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from coinbase.rest import RESTClient
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---- Settings -------------------------------------------------------------
HISTORY_DAYS = 1825            # up to 5 years of daily data
MIN_HISTORY_DAYS = 400         # coins with less history can't be forecast reliably
MIN_TRAIN_ROWS = 150
TEST_FRACTION = 0.40           # last 40% of history is used to grade the model
REFIT_EVERY = 45               # walk-forward: retrain every 45 days
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
LOW_VOLUME_WARNING = 1_000_000
HORIZONS = {"1 day": 1, "3 days": 3, "1 week": 7, "2 weeks": 14, "1 month": 30, "3 months": 90}
HEADERS = {"User-Agent": "CryptoForecastLab/1.0 (personal research app)"}
STABLECOINS = {"USDT", "USDC", "DAI", "PYUSD", "EURC", "GUSD", "PAX", "USDS",
               "FDUSD", "TUSD", "RLUSD", "USD1"}
FRED_SERIES = {
    "SP500": "S&P 500",
    "NASDAQCOM": "Nasdaq",
    "VIXCLS": "VIX (stock market fear gauge)",
    "DGS10": "10-year Treasury yield",
    "DTWEXBGS": "US dollar index",
}
FEATURE_NAMES = {
    "ret_1d": "Yesterday's move", "ret_3d": "3-day momentum", "ret_7d": "1-week momentum",
    "ret_14d": "2-week momentum", "ret_30d": "1-month momentum",
    "vol_7d": "Swings over the past week", "vol_30d": "Swings over the past month",
    "vol_ratio": "Swings speeding up or calming down", "ewma_vol": "Recent swings (weighted)",
    "range_7d": "Daily high-to-low range", "vs_ma20": "Price vs 20-day average",
    "vs_ma50": "Price vs 50-day average", "vs_ma200": "Price vs 200-day average",
    "rsi_14": "Overbought or oversold (RSI)", "volume_spike": "Trading volume vs normal",
    "drawdown_90d": "Distance below 90-day high",
    "btc_ret_7d": "Bitcoin's past week", "btc_vol_30d": "Bitcoin's recent swings",
    "btc_corr_30d": "How closely it tracks Bitcoin",
    "sp500_ret_5d": "S&P 500 past week", "nasdaq_ret_5d": "Nasdaq past week",
    "vix_level": "Stock market fear (VIX)", "vix_chg_5d": "Change in stock market fear",
    "yield10_chg_20d": "Change in 10-year yield", "dollar_ret_20d": "US dollar strength",
    "fear_greed": "Crypto Fear & Greed", "fear_greed_chg_7d": "Change in Fear & Greed",
    "news_tone_7d": "Crypto news tone", "news_tone_chg": "News tone vs its usual level",
    "attention_spike": "Wikipedia attention spike", "attention_trend": "Wikipedia attention trend",
}
# ---------------------------------------------------------------------------



# ======================= Data loading =====================================

def _d(resp):
    return resp.to_dict() if hasattr(resp, "to_dict") else resp


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _daily(s):
    """Clean a series into one value per calendar day, timezone-free."""
    s = s.dropna()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    s.index = idx.normalize()
    return s.groupby(level=0).mean().sort_index()


def _align(s, index, limit=7):
    """Line a series up with the coin's dates, carrying values over weekends and holidays."""
    return s.reindex(s.index.union(index)).ffill(limit=limit).reindex(index)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_universe():
    data = _d(RESTClient().get_public_products(product_type="SPOT", limit=1000))
    rows = []
    for p in data.get("products", []):
        base = p.get("base_currency_id", "")
        if (p.get("quote_currency_id") != "USD" or base in STABLECOINS
                or p.get("trading_disabled") or p.get("is_disabled")
                or p.get("status", "online") != "online"):
            continue
        vol = _f(p.get("approximate_quote_24h_volume")) or _f(p.get("volume_24h")) * _f(p.get("price"))
        name = p.get("base_name") or base
        rows.append({"product_id": p.get("product_id"), "symbol": base, "name": name,
                     "label": f"{name} ({base})" if name != base else base, "usd_volume_24h": vol})
    return pd.DataFrame(rows).sort_values("usd_volume_24h", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_candles(product_id):
    client = RESTClient()
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - HISTORY_DAYS * 86_400
    rows, t = [], start
    while t < end:
        t_end = min(t + 300 * 86_400, end)
        resp = _d(client.get_public_candles(product_id=product_id, start=str(t),
                                            end=str(t_end), granularity="ONE_DAY"))
        rows.extend(resp.get("candles", []))
        t = t_end
        time.sleep(0.1)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["start"].astype(int), unit="s").dt.normalize()
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
    return df[~df.index.duplicated()].sort_index()


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_fred(series_id):
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": series_id}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0]))
    return _daily(s)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_fear_greed():
    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"},
                     headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    s = pd.Series([_f(d["value"]) for d in data],
                  index=pd.to_datetime([int(d["timestamp"]) for d in data], unit="s"))
    return _daily(s)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_news_tone():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=HISTORY_DAYS)
    params = {"query": "(bitcoin OR cryptocurrency OR crypto)", "mode": "timelinetone",
              "format": "json", "startdatetime": start.strftime("%Y%m%d%H%M%S"),
              "enddatetime": end.strftime("%Y%m%d%H%M%S")}
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params,
                     headers=HEADERS, timeout=40)
    r.raise_for_status()
    points = r.json()["timeline"][0]["data"]
    s = pd.Series([_f(p["value"]) for p in points],
                  index=pd.to_datetime([p["date"] for p in points], format="%Y%m%dT%H%M%SZ"))
    return _daily(s)


@st.cache_data(ttl=7 * 86_400, show_spinner=False)
def find_wiki_article(name):
    r = requests.get("https://en.wikipedia.org/w/api.php", headers=HEADERS, timeout=20,
                     params={"action": "query", "list": "search", "format": "json",
                             "srsearch": f"{name} cryptocurrency", "srlimit": 1})
    r.raise_for_status()
    hits = r.json().get("query", {}).get("search", [])
    if not hits:
        return None
    title = hits[0]["title"]
    key_word = name.lower().split()[0]
    return title if key_word in title.lower() else None   # guard against a wrong match


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_wiki_views(title):
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=HISTORY_DAYS)
    article = quote(title.replace(" ", "_"), safe="")
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
           f"all-access/user/{article}/daily/{start:%Y%m%d}/{end:%Y%m%d}")
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    items = r.json()["items"]
    s = pd.Series([float(i["views"]) for i in items],
                  index=pd.to_datetime([i["timestamp"][:8] for i in items], format="%Y%m%d"))
    return _daily(s)


def gather_data(product_id, name, status):
    """Load everything for one coin. Optional sources fail quietly and are reported."""
    sources = []
    status.write("Price and volume from Coinbase…")
    candles = load_candles(product_id)
    if candles is None or len(candles) < MIN_HISTORY_DAYS:
        return candles, None, {}, sources
    sources.append(("Coinbase", "Price and volume", True, f"{len(candles):,} days"))

    btc = None
    if product_id != "BTC-USD":
        try:
            btc = load_candles("BTC-USD")
            sources.append(("Coinbase", "Bitcoin, as the market leader", True, ""))
        except Exception:
            sources.append(("Coinbase", "Bitcoin, as the market leader", False, "unavailable"))

    extras = {}
    status.write("Stock market and economic data from FRED…")
    for sid, label in FRED_SERIES.items():
        try:
            extras[sid] = load_fred(sid)
            sources.append(("FRED", label, True, ""))
        except Exception:
            sources.append(("FRED", label, False, "unavailable right now"))

    status.write("Crypto sentiment and news tone…")
    for key, label, src, fn in [("fear_greed", "Crypto Fear & Greed Index", "Alternative.me", load_fear_greed),
                                ("news_tone", "Tone of crypto news coverage", "GDELT", load_news_tone)]:
        try:
            extras[key] = fn()
            sources.append((src, label, True, ""))
        except Exception:
            sources.append((src, label, False, "unavailable right now"))

    status.write("Public attention from Wikipedia…")
    try:
        title = find_wiki_article(name)
        if title:
            extras["wiki"] = load_wiki_views(title)
            sources.append(("Wikipedia", f"Daily views of “{title}”", True, ""))
        else:
            sources.append(("Wikipedia", "Daily page views", False, "no matching article found"))
    except Exception:
        sources.append(("Wikipedia", "Daily page views", False, "unavailable right now"))
    return candles, btc, extras, sources


# ======================= Features =========================================

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_features(c, btc, extras):
    """Everything is computed from data available at each day's close: no peeking ahead."""
    close, idx = c["close"], c.index
    lr = np.log(close).diff()
    yr = np.sqrt(365)
    f = pd.DataFrame(index=idx)
    for n in [1, 3, 7, 14, 30]:
        f[f"ret_{n}d"] = np.log(close / close.shift(n))
    f["vol_7d"] = lr.rolling(7).std() * yr
    f["vol_30d"] = lr.rolling(30).std() * yr
    f["vol_ratio"] = f["vol_7d"] / f["vol_30d"]
    f["ewma_vol"] = np.sqrt((lr ** 2).ewm(alpha=0.06).mean()) * yr
    f["range_7d"] = np.log(c["high"] / c["low"]).rolling(7).mean()
    for n in [20, 50, 200]:
        f[f"vs_ma{n}"] = close / close.rolling(n).mean() - 1
    f["rsi_14"] = rsi(close)
    f["volume_spike"] = np.log(c["volume"] / c["volume"].rolling(30).mean())
    f["drawdown_90d"] = close / close.rolling(90).max() - 1

    if btc is not None:
        b = _align(btc["close"], idx, limit=2)
        blr = np.log(b).diff()
        f["btc_ret_7d"] = np.log(b / b.shift(7))
        f["btc_vol_30d"] = blr.rolling(30).std() * yr
        f["btc_corr_30d"] = lr.rolling(30).corr(blr)

    def series(key):
        return _align(extras[key], idx) if key in extras else None

    if (s := series("SP500")) is not None:
        f["sp500_ret_5d"] = np.log(s / s.shift(5))
    if (s := series("NASDAQCOM")) is not None:
        f["nasdaq_ret_5d"] = np.log(s / s.shift(5))
    if (s := series("VIXCLS")) is not None:
        f["vix_level"] = s
        f["vix_chg_5d"] = s - s.shift(5)
    if (s := series("DGS10")) is not None:
        f["yield10_chg_20d"] = s - s.shift(20)
    if (s := series("DTWEXBGS")) is not None:
        f["dollar_ret_20d"] = np.log(s / s.shift(20))
    if (s := series("fear_greed")) is not None:
        f["fear_greed"] = s
        f["fear_greed_chg_7d"] = s - s.shift(7)
    if (s := series("news_tone")) is not None:
        tone7 = s.rolling(7, min_periods=3).mean()
        f["news_tone_7d"] = tone7
        f["news_tone_chg"] = tone7 - s.rolling(60, min_periods=20).mean()
    if (s := series("wiki")) is not None:
        s = s.clip(lower=1)
        f["attention_spike"] = np.log(s.rolling(3).mean() / s.rolling(30).mean())
        wk = s.rolling(7).mean()
        f["attention_trend"] = np.log(wk / wk.shift(7))
    return f.replace([np.inf, -np.inf], np.nan)


def prepare(features, close, h):
    y = np.log(close.shift(-h) / close)          # the h-day future return we try to predict
    start = features[["ret_30d", "vol_30d"]].dropna().index.min()
    X = features.loc[start:]
    keep = [col for col in X.columns if X[col].notna().mean() >= 0.7]
    return X[keep], y.loc[start:]


# ======================= Models ===========================================

def fit_models(X, y):
    up = (y > 0).astype(int)
    logit = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    boost = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                           l2_regularization=1.0, random_state=0)
    if len(np.unique(up)) < 2:
        logit = boost = None
    else:
        logit.fit(X, up)
        boost.fit(X, up)
    quant = {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3,
                                              learning_rate=0.05, max_iter=150,
                                              random_state=0).fit(X, y)
             for q in QUANTILES}
    return {"logit": logit, "boost": boost, "quant": quant, "base_up": float(up.mean())}


def predict(models, X):
    if models["logit"] is None:
        p_logit = p_boost = np.full(len(X), models["base_up"])
    else:
        p_logit = models["logit"].predict_proba(X)[:, 1]
        p_boost = models["boost"].predict_proba(X)[:, 1]
    q = np.column_stack([models["quant"][qq].predict(X) for qq in QUANTILES])
    q.sort(axis=1)                                  # keep quantiles in order
    return (p_logit + p_boost) / 2, p_logit, p_boost, q


def walk_forward(X, y, h, on_progress=None):
    """Grade the model honestly: at each point, train only on outcomes already known."""
    Xv, yv = X.to_numpy(), y.to_numpy()
    n = len(Xv)
    known = ~np.isnan(yv)
    last = n - 1 - h
    first = max(int(n * (1 - TEST_FRACTION)), MIN_TRAIN_ROWS + h)
    if last - first < 30:
        return None
    starts = list(range(first, last + 1, REFIT_EVERY))
    rows = []
    for k, s in enumerate(starts):
        train = np.arange(0, s - h + 1)
        train = train[known[train]]
        if len(train) >= MIN_TRAIN_ROWS:
            m = fit_models(Xv[train], yv[train])
            e = min(s + REFIT_EVERY, last + 1)
            p, _, _, q = predict(m, Xv[s:e])
            for j, i in enumerate(range(s, e)):
                rows.append({"date": X.index[i], "p_up": p[j], "actual": yv[i],
                             **{f"q{int(qq * 100)}": q[j, c] for c, qq in enumerate(QUANTILES)}})
        if on_progress:
            on_progress((k + 1) / len(starts))
    return pd.DataFrame(rows).set_index("date") if rows else None


def score(bt):
    up = (bt["actual"] > 0).astype(int)
    base = up.mean()
    brier = ((bt["p_up"] - up) ** 2).mean()
    brier_ref = base * (1 - base)
    return {
        "n": len(bt),
        "accuracy": ((bt["p_up"] > 0.5).astype(int) == up).mean(),
        "always_up": base,
        "skill": 1 - brier / brier_ref if brier_ref > 0 else 0.0,
        "cov80": ((bt["actual"] >= bt["q10"]) & (bt["actual"] <= bt["q90"])).mean(),
        "cov50": ((bt["actual"] >= bt["q25"]) & (bt["actual"] <= bt["q75"])).mean(),
    }


def conformal_corrections(bt):
    """Conformalized quantile regression: widen or narrow ranges using past misses."""
    n = len(bt)
    e80 = np.maximum(bt["q10"] - bt["actual"], bt["actual"] - bt["q90"])
    e50 = np.maximum(bt["q25"] - bt["actual"], bt["actual"] - bt["q75"])
    return (float(np.quantile(e80, min(1.0, 0.80 * (1 + 1 / n)))),
            float(np.quantile(e50, min(1.0, 0.50 * (1 + 1 / n)))))


def fit_garch(returns):
    """GARCH(1,1) by maximum likelihood. Returns in percent for numerical stability."""
    r = (returns.dropna().to_numpy() * 100)[-1095:]
    r = r - r.mean()
    v0 = r.var()

    def variances(omega, alpha, beta):
        s2 = np.empty_like(r)
        s2[0] = v0
        for t in range(1, len(r)):
            s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        return s2

    def neg_log_lik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e12
        s2 = variances(omega, alpha, beta)
        return 0.5 * np.sum(np.log(s2) + r ** 2 / s2)

    res = minimize(neg_log_lik, x0=[v0 * 0.05, 0.08, 0.88], method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    omega, alpha, beta = res.x
    if not res.success and neg_log_lik(res.x) >= 1e12:
        return None
    s2 = variances(omega, alpha, beta)
    next_var = omega + alpha * r[-1] ** 2 + beta * s2[-1]
    long_run = omega / (1 - alpha - beta)
    return {"omega": omega, "alpha": alpha, "beta": beta, "next_var": next_var, "long_run": long_run}


def garch_horizon_sd(g, h):
    """Total expected swing over h days: sum of each day's forecast variance."""
    k = np.arange(h)
    path = g["long_run"] + (g["alpha"] + g["beta"]) ** k * (g["next_var"] - g["long_run"])
    return float(np.sqrt(path.sum())) / 100


def run_forecast(candles, btc, extras, h, on_progress):
    features = build_features(candles, btc, extras)
    X, y = prepare(features, candles["close"], h)
    bt = walk_forward(X, y, h, on_progress)
    if bt is None or len(bt) < 30:
        return None
    known = y.notna().to_numpy()
    full = fit_models(X.to_numpy()[known], y.to_numpy()[known])
    p, p_logit, p_boost, q = predict(full, X.to_numpy()[-1:])
    corr80, corr50 = conformal_corrections(bt)
    q10, q25, q50, q75, q90 = q[0]
    lo80, hi80 = min(q10 - corr80, q50), max(q90 + corr80, q50)
    lo50, hi50 = min(q25 - corr50, q50), max(q75 + corr50, q50)
    lo80, hi80 = min(lo80, lo50), max(hi80, hi50)

    # What pushed the forecast: the simpler model's contribution from each input
    logit = full["logit"]
    drivers = None
    if logit is not None:
        scaled = logit[:-1].transform(X.to_numpy()[-1:])[0]
        contrib = logit[-1].coef_[0] * scaled
        drivers = (pd.DataFrame({"feature": X.columns, "contribution": contrib})
                   .assign(name=lambda d: d["feature"].map(FEATURE_NAMES).fillna(d["feature"]),
                           size=lambda d: d["contribution"].abs())
                   .sort_values("size", ascending=False).head(8))

    log_ret = np.log(candles["close"]).diff()
    g = fit_garch(log_ret)
    if g:
        sd_h = garch_horizon_sd(g, h)
        daily_now, daily_usual = np.sqrt(g["next_var"]) / 100, np.sqrt(g["long_run"]) / 100
    else:   # fallback: exponentially weighted volatility
        daily_now = float(np.sqrt((log_ret ** 2).ewm(alpha=0.06).mean().iloc[-1]))
        daily_usual = float(log_ret.std())
        sd_h = daily_now * np.sqrt(h)

    return {"p_up": float(p[0]), "p_logit": float(p_logit[0]), "p_boost": float(p_boost[0]),
            "q50": q50, "lo80": lo80, "hi80": hi80, "lo50": lo50, "hi50": hi50,
            "sd_h": sd_h, "daily_now": daily_now, "daily_usual": daily_usual, "garch": g,
            "drivers": drivers, "bt": bt, "score": score(bt), "n_features": X.shape[1],
            "n_train": int(known.sum()), "corr80": corr80}


# ======================= Display helpers ==================================

def usd(x):
    if x >= 100:
        return f"${x:,.0f}"
    if x >= 1:
        return f"${x:,.2f}"
    return f"${x:,.4g}"


def md(text):
    """Escape dollar signs so Streamlit shows them as money, not as math formulas."""
    return text.replace("$", r"\$")


NEXT = {"1 day": "the next day", "3 days": "the next 3 days", "1 week": "the next week",
        "2 weeks": "the next 2 weeks", "1 month": "the next month", "3 months": "the next 3 months"}


def pct(x, signed=True):
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def trust_verdict(s):
    edge = s["accuracy"] - max(0.5, s["always_up"])
    if s["skill"] > 0.02 and edge > 0.03:
        return "good", "The direction call has shown some real skill on past data."
    if s["skill"] > 0:
        return "weak", "The direction call has been only slightly better than guessing. Treat it as a weak hint."
    return "none", ("The direction call has not beaten simple guessing on past data. "
                    "Lean on the range and volatility forecasts instead.")


def range_verdict(cov80):
    if cov80 < 0.72:
        return "Before calibration, its 80% ranges were too narrow (reality landed outside more often than claimed). The ranges shown are widened to correct for this."
    if cov80 > 0.88:
        return "Before calibration, its 80% ranges were too cautious (wider than needed). The ranges shown are tightened to correct for this."
    return "Its 80% ranges have been well calibrated: reality landed inside them about 8 times out of 10."


# ======================= Page =============================================

st.title("🔭 Crypto Forecast Lab")
st.write("Pick a coin and a timeframe. The model forecasts direction, a likely price range, "
         "and expected swings, then shows how well its past forecasts held up.")

try:
    universe = load_universe()
except Exception as e:
    st.error(f"Couldn't reach Coinbase to list coins. Check your connection and refresh. ({e})")
    st.stop()

ids = universe["product_id"].tolist()
labels = dict(zip(universe["product_id"], universe["label"]))
with st.form("inputs"):
    c1, c2 = st.columns([2, 1])
    coin = c1.selectbox(f"🪙 Coin ({len(ids)} on Coinbase, most traded first)", ids,
                        index=ids.index("BTC-USD") if "BTC-USD" in ids else 0,
                        format_func=lambda p: labels[p])
    horizon_label = c2.select_slider("⏱️ Forecast how far ahead", list(HORIZONS), value="1 week")
    go = st.form_submit_button("Run forecast", type="primary")

if go:
    st.session_state["run"] = (coin, horizon_label)

if "run" not in st.session_state:
    st.info("Choose a coin and timeframe, then press **Run forecast**. The first run for each coin "
            "downloads several years of data and tests the model, which takes about a minute.")
    with st.expander("How should I read a forecast?"):
        st.write(
            "Think of a weather forecast. \"70% chance of rain\" doesn't mean it will rain; it means "
            "that on days like this, it rained about 7 times out of 10. A good forecaster's 70% days "
            "really do see rain about 70% of the time.\n\n"
            "This app works the same way for crypto, and it checks itself. Before showing you "
            "anything, it replays the past few years day by day, making forecasts using only what it "
            "would have known at the time, and keeps score. You'll see that score next to every "
            "forecast, so you know whether to take it seriously."
        )
    st.stop()

coin, horizon_label = st.session_state["run"]
h = HORIZONS[horizon_label]
row = universe.set_index("product_id").loc[coin]
name = row["name"]

span = horizon_label.replace(" ", "-")   # "1-day", "2-week" ...
span = span[:-1] if span.endswith("s") else span
with st.status(f"Building the {span} forecast for {labels[coin]}…", expanded=True) as status:
    try:
        candles, btc, extras, sources = gather_data(coin, name, status)
    except Exception as e:
        status.update(label="Couldn't load price data", state="error")
        st.error(f"Coinbase didn't return price history for {labels[coin]}. Refresh to try again. ({e})")
        st.stop()
    if candles is None or len(candles) < MIN_HISTORY_DAYS:
        have = 0 if candles is None else len(candles)
        status.update(label="Not enough history", state="error")
        st.error(f"{labels[coin]} has {have} days of price history on Coinbase. The model needs at "
                 f"least {MIN_HISTORY_DAYS} days to learn from and test itself. Try an older coin.")
        st.stop()
    status.write("Testing the model on past data (walk-forward)…")
    bar = st.progress(0.0)
    try:
        result = run_forecast(candles, btc, extras, h, lambda x: bar.progress(min(x, 1.0)))
    except Exception as e:
        status.update(label="The model hit an error", state="error")
        st.error(f"Something went wrong while fitting the model: {e}")
        st.stop()
    bar.empty()
    if result is None:
        status.update(label="Not enough history for this timeframe", state="error")
        st.error("There isn't enough history to test a forecast this far ahead. Try a shorter timeframe.")
        st.stop()
    status.update(label=f"The {span} forecast for {labels[coin]} is ready", state="complete",
                  expanded=False)

r = result
s = r["score"]
price = float(candles["close"].iloc[-1])
as_of = candles.index[-1]
mid = price * np.exp(r["q50"])
lo80, hi80 = price * np.exp(r["lo80"]), price * np.exp(r["hi80"])
lo50, hi50 = price * np.exp(r["lo50"]), price * np.exp(r["hi50"])
swing_ratio = r["daily_now"] / r["daily_usual"] if r["daily_usual"] > 0 else 1.0
mood = ("much wilder than usual" if swing_ratio > 1.3 else "wilder than usual" if swing_ratio > 1.1
        else "calmer than usual" if swing_ratio < 0.9 else "about normal")
level, verdict = trust_verdict(s)

if row["usd_volume_24h"] < LOW_VOLUME_WARNING:
    st.warning(md(f"{labels[coin]} trades only about {usd(row['usd_volume_24h'])} per day on Coinbase. "
                  "Thinly traded coins move erratically and are expensive to trade, so forecasts are "
                  "less reliable."))

# ---- The forecast ----
st.subheader(f"{labels[coin]}: {NEXT[horizon_label]}")
m1, m2, m3 = st.columns(3)
m1.metric("Chance it ends higher", f"{r['p_up']:.0%}",
          help="Average of two models: a logistic regression and a gradient-boosted tree ensemble.")
m1.caption({"good": "Some proven skill on past data", "weak": "Weak signal on past data",
            "none": "No proven skill on past data"}[level])
m2.metric("Likely range (8 in 10)", f"{usd(lo80)} – {usd(hi80)}",
          help="80% range from quantile models, calibrated against the model's past misses.")
m3.metric("Expected swing", f"±{r['sd_h']:.1%}", help="One standard deviation over the period, from a GARCH(1,1) volatility model.")

st.write(md(
    f"Today's price is **{usd(price)}** (as of {as_of:%b %d, %Y}). Over {NEXT[horizon_label]}, "
    f"the model gives it a **{r['p_up']:.0%} chance of finishing higher**. Its middle estimate is "
    f"**{usd(mid)}** ({pct(r['q50'])}); half the time it expects the price to land between "
    f"{usd(lo50)} and {usd(hi50)}, and 8 times in 10 between **{usd(lo80)} and {usd(hi80)}**. "
    f"Swings right now are **{mood}**."
))
{"good": st.success, "weak": st.warning, "none": st.error}[level](f"**How much to trust this:** {verdict}")

# ---- Fan chart ----
hist = candles["close"].iloc[-max(120, 3 * h):]
future = pd.date_range(as_of, periods=h + 1, freq="D")
frac = np.arange(h + 1) / h
band = lambda edge: price * np.exp(r["q50"] * frac + (edge - r["q50"]) * np.sqrt(frac))
fan = pd.DataFrame({"date": future, "mid": band(r["q50"]), "lo80": band(r["lo80"]),
                    "hi80": band(r["hi80"]), "lo50": band(r["lo50"]), "hi50": band(r["hi50"])})
hist_df = pd.DataFrame({"date": hist.index, "price": hist.values})
y_enc = alt.Y("lo80:Q", title="Price ($)", scale=alt.Scale(zero=False))
chart = (
    alt.Chart(fan).mark_area(opacity=0.15, color="#4c78a8").encode(x=alt.X("date:T", title=None), y=y_enc, y2="hi80:Q")
    + alt.Chart(fan).mark_area(opacity=0.3, color="#4c78a8").encode(x="date:T", y="lo50:Q", y2="hi50:Q")
    + alt.Chart(hist_df).mark_line(color="#4c78a8").encode(x="date:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
    + alt.Chart(fan).mark_line(strokeDash=[5, 4], color="#4c78a8").encode(x="date:T", y="mid:Q")
).properties(height=380, width="container")
st.altair_chart(chart)
st.caption("Solid line: actual price. Dashed: middle estimate. Dark band: 50% range. Light band: 80% range.")

# ---- Under the hood ----
st.subheader("Under the hood")
t1, t2, t3, t4 = st.tabs(["What's driving it", "Track record", "The math", "Data used"])

with t1:
    if r["drivers"] is not None:
        d = r["drivers"].assign(direction=lambda x: np.where(x["contribution"] > 0, "Pushes toward higher", "Pushes toward lower"))
        bars = alt.Chart(d).mark_bar().encode(
            x=alt.X("contribution:Q", title="Push on the forecast (log-odds)"),
            y=alt.Y("name:N", sort=alt.EncodingSortField("size", order="descending"), title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["Pushes toward higher", "Pushes toward lower"],
                                                           range=["#54a24b", "#e45756"]), title=None),
        ).properties(height=280, width="container")
        st.altair_chart(bars)
        st.caption("The eight inputs pushing hardest on today's forecast, from the logistic regression "
                   "(the easier of the two models to explain). The tree model sees interactions this chart can't show.")
    else:
        st.write("Driver breakdown isn't available for this coin and timeframe.")

with t2:
    a, b, c = st.columns(3)
    a.metric("Direction calls right", f"{s['accuracy']:.0%}",
             help=f"Always guessing 'up' would have scored {s['always_up']:.0%}; a coin flip scores 50%.")
    b.metric("Forecast skill score", f"{s['skill']:+.2f}",
             help="Brier skill score. Above 0 = better probabilities than just using how often it went up. Below 0 = worse.")
    c.metric("Landed in the 80% range", f"{s['cov80']:.0%}", help="Target: 80%. Measured before calibration.")
    st.write(f"{verdict} {range_verdict(s['cov80'])}")
    st.caption(f"Graded on {s['n']:,} past forecasts, made using only data available at the time. "
               + (f"Forecasts made {horizon_label} ahead overlap, so this covers roughly {max(1, s['n'] // h)} truly separate periods. Treat the score as rough."
                  if h >= 14 else ""))

    bt = r["bt"].reset_index()
    bt_chart = (
        alt.Chart(bt).mark_area(opacity=0.2, color="#4c78a8").encode(
            x=alt.X("date:T", title=None), y=alt.Y("q10:Q", title=f"{horizon_label} return", axis=alt.Axis(format="%")), y2="q90:Q")
        + alt.Chart(bt).mark_circle(size=12, color="#333").encode(x="date:T", y="actual:Q")
    ).properties(height=260, width="container")
    st.altair_chart(bt_chart)
    st.caption("Band: what the model predicted (80% range, before calibration). Dots: what actually happened.")

    calib = bt.assign(bucket=pd.cut(bt["p_up"], [0, 0.4, 0.45, 0.5, 0.55, 0.6, 1],
                                    labels=["Under 40%", "40–45%", "45–50%", "50–55%", "55–60%", "Over 60%"]))
    table = (calib.groupby("bucket", observed=True)
             .agg(forecasts=("p_up", "size"), said=("p_up", "mean"), happened=("actual", lambda x: (x > 0).mean()))
             .reset_index())
    table["said"] = (table["said"] * 100).round().astype(int).astype(str) + "%"
    table["happened"] = (table["happened"] * 100).round().astype(int).astype(str) + "%"
    table.columns = ["When the model said", "Forecasts", "Average chance it gave", "How often it actually rose"]
    st.markdown("**Did its percentages mean what they said?**")
    st.dataframe(table, hide_index=True)

with t3:
    st.markdown("**1. Direction: an ensemble of two classifiers.** A logistic regression and a "
                "gradient-boosted decision-tree model each estimate the chance of a rise; the app averages them.")
    st.latex(r"P(\text{up}) = \tfrac{1}{2}\Big[\sigma(\beta_0 + \boldsymbol{\beta}^\top \mathbf{x}) + f_{\text{GBT}}(\mathbf{x})\Big], \quad \sigma(z) = \frac{1}{1+e^{-z}}")
    st.write(f"Today: logistic regression {r['p_logit']:.0%}, tree model {r['p_boost']:.0%}. "
             f"Trained on {r['n_train']:,} days using {r['n_features']} inputs.")
    st.markdown("**2. Range: quantile regression.** Five gradient-boosted models each target one "
                "percentile of the future return by minimizing the pinball loss:")
    st.latex(r"L_\tau(y, \hat{q}) = \max\big(\tau\,(y-\hat{q}),\ (\tau-1)\,(y-\hat{q})\big), \quad \tau \in \{0.10, 0.25, 0.50, 0.75, 0.90\}")
    st.markdown("Then **conformal calibration** adjusts the ranges using the model's past misses, "
                "so an 80% range really covers about 80% of outcomes:")
    st.latex(r"E_i = \max(\hat{q}_{0.10,i} - y_i,\ y_i - \hat{q}_{0.90,i}), \qquad [\hat{q}_{0.10} - \hat{c},\ \hat{q}_{0.90} + \hat{c}], \ \ \hat{c} = Q_{0.8}(E)")
    st.write(f"Calibration adjustment applied to the 80% range: {r['corr80']:+.2%} on each side.")
    st.markdown("**3. Volatility: GARCH(1,1),** fit by maximum likelihood. Today's swing size depends on "
                "yesterday's surprise and yesterday's swing size, and drifts back toward a long-run level:")
    st.latex(r"\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2, \qquad \sigma^2_{\text{total}}(h) = \sum_{k=1}^{h} \Big[\bar{\sigma}^2 + (\alpha+\beta)^{k-1}(\sigma^2_{t+1} - \bar{\sigma}^2)\Big]")
    if r["garch"]:
        g = r["garch"]
        st.write(f"Estimated: α = {g['alpha']:.3f}, β = {g['beta']:.3f} (persistence α+β = "
                 f"{g['alpha'] + g['beta']:.3f}). Daily swing now ±{r['daily_now']:.1%} vs a long-run "
                 f"±{r['daily_usual']:.1%}.")
    else:
        st.write("GARCH didn't converge for this coin, so an exponentially weighted volatility estimate was used.")
    st.markdown("**4. Walk-forward testing.** The model is retrained every "
                f"{REFIT_EVERY} days over the last {TEST_FRACTION:.0%} of history, each time using only "
                "outcomes that were already known, then graded on what happened next.")

with t4:
    st.dataframe(pd.DataFrame(sources, columns=["Source", "What", "Used", "Note"]), hide_index=True)
    st.caption("Optional sources that are temporarily unavailable are skipped; the forecast uses whatever loaded. "
               "Inputs with too little history for this coin are left out automatically.")

st.divider()
st.caption("Forecasts are probabilities learned from the past, not promises. Crypto can move in ways no "
           "past data contains. This is not financial advice. Data refreshes every 6–12 hours.")
