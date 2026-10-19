"""
Crypto Reality Check: a browser app for exploring crypto against YOUR money, risk, and goal.

You enter:  how much to invest, the most you're OK losing, your goal, and a timeframe.
It shows:   for the most-traded coins on Coinbase, how much you could put in while
            staying inside your loss limit, and how often your goal was actually
            reached across every past stretch of that length (fees included).

It replays real history. It does not predict the future or tell you what to buy.

Setup (once):   pip install streamlit coinbase-advanced-py pandas numpy
Run:            streamlit run streamlit_app.py   (this file is one page of the combined app)
"""

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from numpy.lib.stride_tricks import sliding_window_view
from coinbase.rest import RESTClient

# ---- Settings -------------------------------------------------------------
TOP_N = 25                      # most-traded coins to analyze
MIN_24H_USD_VOLUME = 1_000_000  # skip coins trading less than this per day
HISTORY_DAYS = 1095             # about 3 years of daily prices
FEE_PER_SIDE = 0.006            # 0.6% to buy + 0.6% to sell (adjust to your Coinbase tier)
BAD_CASE_PERCENTILE = 10        # "bad case" = the worst 1 in 10 past periods
MIN_PERIODS = 60
BASKET_SIZE = 10
HORIZONS = {"1 week": 7, "2 weeks": 14, "1 month": 30, "3 months": 90, "6 months": 180}
STABLECOINS = {"USDT", "USDC", "DAI", "PYUSD", "EURC", "GUSD", "PAX", "USDS",
               "FDUSD", "TUSD", "RLUSD", "USD1"}
# ---------------------------------------------------------------------------



# ---------- Data ----------------------------------------------------------

def _d(resp):
    return resp.to_dict() if hasattr(resp, "to_dict") else resp


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_universe():
    client = RESTClient()
    data = _d(client.get_public_products(product_type="SPOT", limit=1000))
    rows = []
    for p in data.get("products", []):
        base = p.get("base_currency_id", "")
        if (p.get("quote_currency_id") != "USD" or base in STABLECOINS
                or p.get("trading_disabled") or p.get("is_disabled")
                or p.get("status", "online") != "online"):
            continue
        vol = _f(p.get("approximate_quote_24h_volume")) or _f(p.get("volume_24h")) * _f(p.get("price"))
        name = p.get("base_name") or base
        rows.append({"product_id": p.get("product_id"), "symbol": base,
                     "label": f"{name} ({base})" if name != base else base,
                     "usd_volume_24h": vol})
    df = pd.DataFrame(rows).sort_values("usd_volume_24h", ascending=False).reset_index(drop=True)
    return df


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_daily_closes(product_id):
    client = RESTClient()
    end = int(datetime.now(timezone.utc).timestamp())
    start = end - HISTORY_DAYS * 86_400
    step = 349 * 86_400
    rows, t = [], start
    while t < end:
        t_end = min(t + step, end)
        resp = _d(client.get_public_candles(product_id=product_id, start=str(t),
                                            end=str(t_end), granularity="ONE_DAY"))
        rows.extend(resp.get("candles", []))
        t = t_end
        time.sleep(0.1)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    s = pd.Series(pd.to_numeric(df["close"]).to_numpy(),
                  index=pd.to_datetime(df["start"].astype(int), unit="s", utc=True))
    return s[~s.index.duplicated()].sort_index()


def build_basket(closes):
    df = pd.DataFrame(closes)
    daily = df.pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0)
    return (1 + daily).cumprod()


# ---------- Analysis ------------------------------------------------------

def analyze(close, horizon, amount, max_loss_pct, goal_pct):
    """Replay every past stretch of `horizon` days for one asset."""
    c = close.dropna().to_numpy(dtype=float)
    n = len(c) - horizon
    if n < MIN_PERIODS:
        return None

    entry = c[:n]
    end_ret = c[horizon:horizon + n] / entry - 1
    lowest_after = sliding_window_view(c[1:], horizon)[:n].min(axis=1)
    dip = np.minimum(lowest_after / entry - 1, 0)        # deepest point below where you bought

    keep = (1 - FEE_PER_SIDE) ** 2                        # buy fee + sell fee
    end_net = (1 + end_ret) * keep - 1
    dip_net = (1 + dip) * keep - 1

    # Size the position so a bad-case dip stays inside the loss limit; the rest stays in cash.
    bad_dip = -np.percentile(dip_net, BAD_CASE_PERCENTILE)
    share = 1.0 if bad_dip <= max_loss_pct else max_loss_pct / bad_dip
    result = share * end_net                              # whole-portfolio return

    return {
        "share": share,
        "invest": amount * share,
        "cash": amount * (1 - share),
        "must_rise": (goal_pct / share + 1) / keep - 1,
        "hit_rate": float((result >= goal_pct).mean()),
        "typical": amount * float(np.median(result)),
        "bad": amount * float(np.percentile(result, BAD_CASE_PERCENTILE)),
        "worst": amount * share * float(dip_net.min()),
        "half_time_goal": float(np.median(result)),
        "all_in_hit": float((end_net >= goal_pct).mean()),
        "all_in_bad_dip": bad_dip,
        "outcomes": amount * result,
        "periods": n,
    }


def money(x, sign=True):
    s = f"${abs(x):,.0f}"
    if not sign:
        return s
    return f"+{s}" if x >= 0 else f"−{s}"


def md(text):
    """Escape dollar signs so Streamlit shows them as money, not as math formulas."""
    return text.replace("$", r"\$")


# ---------- Page ----------------------------------------------------------

st.title("🧭 Crypto Reality Check")
st.write("Tell it your money, your limit, and your goal. It replays real Coinbase price "
         "history to show which coins would have fit, and how often your goal was actually reached.")

with st.form("inputs"):
    c1, c2, c3, c4 = st.columns(4)
    amount = c1.number_input("💵 Amount to invest ($)", min_value=10.0, value=1000.0,
                             step=100.0, format="%.0f")
    max_loss = c2.number_input("🛡️ Most I'm OK losing ($)", min_value=1.0, value=200.0,
                               step=50.0, format="%.0f")
    goal = c3.number_input("🎯 Goal: grow it by (%)", min_value=0.5, value=10.0,
                           step=1.0, format="%.1f")
    horizon_label = c4.selectbox("⏱️ Over", list(HORIZONS), index=2)
    submitted = st.form_submit_button("Show my options", type="primary")

if submitted:
    st.session_state["ran"] = True

if not st.session_state.get("ran"):
    st.info("Fill in the four boxes and press **Show my options**. "
            "The first run downloads about 3 years of prices and takes around a minute.")
    with st.expander("How does this work?"):
        st.write(
            "Think of it like planning a beach trip. Nobody can promise sunshine next week, "
            "but you can check how often it rained that week over past years. This app does the "
            "same for crypto: it takes every past stretch of your chosen length, pretends you "
            "invested at the start, and counts how often you'd have hit your goal and how far "
            "down you'd have been along the way.\n\n"
            "It also sizes each option to your loss limit. If a coin has often dropped 40% in a "
            "month and you only want to risk 20% of your money, you'd put about half in that coin "
            "and keep the rest in cash. Smaller bets are safer, but the coin then has to rise "
            "more to reach your goal. That trade-off is the whole game."
        )
    st.stop()

if max_loss > amount:
    st.error(md("Your loss limit is bigger than the amount you're investing. Lower it to at most "
                f"{money(amount, sign=False)}."))
    st.stop()

horizon = HORIZONS[horizon_label]
span = horizon_label.replace(" ", "-")   # "1-month", for use before "stretches"
max_loss_pct = max_loss / amount
goal_pct = goal / 100
st.caption(md(f"You're asking for {money(amount * goal_pct)} on {money(amount, sign=False)} over "
              f"{horizon_label}, while never being down more than {money(max_loss, sign=False)} "
              f"({max_loss_pct:.0%})."))

# Load data
try:
    universe = load_universe()
except Exception as e:
    st.error(f"Couldn't reach Coinbase to list coins. Check your internet connection and refresh. ({e})")
    st.stop()

coins = universe[universe["usd_volume_24h"] >= MIN_24H_USD_VOLUME].head(TOP_N)
closes, labels = {}, {}
bar = st.progress(0.0, text="Loading price history from Coinbase…")
for i, row in enumerate(coins.itertuples()):
    try:
        s = load_daily_closes(row.product_id)
        if s is not None and len(s) > horizon + MIN_PERIODS:
            closes[row.product_id] = s
            labels[row.product_id] = row.label
    except Exception:
        pass
    bar.progress((i + 1) / len(coins), text=f"Loading price history… {row.symbol}")
bar.empty()

if not closes:
    st.error("No price history came back from Coinbase. Refresh the page to try again.")
    st.stop()

basket_ids = [pid for pid in coins["product_id"] if pid in closes][:BASKET_SIZE]
assets = dict(closes)
assets["BASKET"] = build_basket({pid: closes[pid] for pid in basket_ids})
labels["BASKET"] = f"Top-{len(basket_ids)} mix (equal amounts)"

results = {}
for pid, series in assets.items():
    r = analyze(series, horizon, amount, max_loss_pct, goal_pct)
    if r:
        results[pid] = r

if not results:
    st.warning("Not enough history for that timeframe. Try a shorter one.")
    st.stop()

ranked = sorted(results.items(), key=lambda kv: (kv[1]["hit_rate"], kv[1]["typical"]), reverse=True)
best_id, best = ranked[0]

# ---------- Reality check ----------
st.subheader("Reality check")
hit = best["hit_rate"]
if hit >= 0.5:
    st.success(f"**Your goal looks reachable at your risk level.** {labels[best_id]} reached it "
               f"in {hit:.0%} of past {span} stretches while staying inside your loss limit.")
elif hit >= 0.2:
    st.warning(f"**Your goal is possible but not likely.** The best fit, {labels[best_id]}, "
               f"reached it in {hit:.0%} of past {span} stretches at your risk level.")
else:
    st.error(f"**Your goal is a stretch for this much risk.** Even the best fit, {labels[best_id]}, "
             f"reached it in only {hit:.0%} of past {span} stretches.")

if hit < 0.5:
    realistic_id, realistic = max(results.items(), key=lambda kv: kv[1]["half_time_goal"])
    bold_id, bold = max(results.items(), key=lambda kv: kv[1]["all_in_hit"])
    st.write("You have two levers:")
    if realistic["half_time_goal"] > 0:
        st.write(md(f"- **Lower the goal.** At your current risk, a goal of about "
                 f"{realistic['half_time_goal']:.1%} ({money(amount * realistic['half_time_goal'])}) "
                 f"was reached about half the time with {labels[realistic_id]}."))
    else:
        st.write("- **Lower the goal.** At your current risk, no option came out ahead in most "
                 "past stretches of this length. A longer timeframe may help.")
    if bold["all_in_hit"] > hit:
        st.write(md(f"- **Accept more risk.** Putting the full {money(amount, sign=False)} in "
                 f"{labels[bold_id]} reached your goal in {bold['all_in_hit']:.0%} of past "
                 f"stretches, but in a bad case you'd have been down about "
                 f"{bold['all_in_bad_dip']:.0%} ({money(-amount * bold['all_in_bad_dip'])}) at some point."))

# ---------- Options table ----------
st.subheader("Your options")
st.caption("Each option is sized so that in 9 of 10 past stretches you'd never have been down "
           "more than your limit. Whatever isn't invested stays in cash. Fees are included.")

table = pd.DataFrame([{
    "Option": labels[pid],
    "Put in": money(r["invest"], sign=False),
    "Keep in cash": money(r["cash"], sign=False),
    "Coin must rise": f"{r['must_rise']:.0%}",
    "Goal reached": round(r["hit_rate"] * 100),
    "Typical result": money(r["typical"]),
    "Bad case (1 in 10)": money(r["bad"]),
    "Lowest point seen": money(r["worst"]),
} for pid, r in ranked])

groups = [("Reached your goal in most past stretches", table["Goal reached"] >= 50),
          ("Reached it sometimes", (table["Goal reached"] >= 20) & (table["Goal reached"] < 50)),
          ("Rarely reached it", table["Goal reached"] < 20)]
for title, mask in groups:
    part = table[mask]
    if part.empty:
        continue
    st.markdown(f"**{title}** ({len(part)})")
    st.dataframe(part, hide_index=True, column_config={
        "Goal reached": st.column_config.ProgressColumn(
            "Goal reached", help="Share of past stretches where you'd have hit your goal",
            min_value=0, max_value=100, format="%.0f%%"),
        "Coin must rise": st.column_config.TextColumn(
            help="How much the coin itself has to gain for your whole amount to hit the goal"),
        "Lowest point seen": st.column_config.TextColumn(
            help="The deepest you'd have been down at any point, in the worst past stretch"),
    })

if horizon >= 90:
    st.caption("Longer timeframes overlap heavily in 3 years of data, so treat these "
               "percentages as rough.")

# ---------- Detail view ----------
st.subheader("Look closer at one option")
pick = st.selectbox("Option", [pid for pid, _ in ranked], format_func=lambda p: labels[p])
r = results[pick]
st.write(md(
    f"If you'd put **{money(r['invest'], sign=False)}** in {labels[pick]} and kept "
    f"**{money(r['cash'], sign=False)}** in cash, then across {r['periods']:,} past "
    f"{span} stretches you'd have reached {money(amount * goal_pct)} in "
    f"**{r['hit_rate']:.0%}** of them. A typical stretch ended at **{money(r['typical'])}**; "
    f"the worst 1 in 10 ended at **{money(r['bad'])}** or lower."
))
left, right = st.columns(2)
with left:
    st.markdown("**Price history**")
    st.line_chart(assets[pick])
with right:
    st.markdown(f"**How past {span} stretches ended for you ($)**")
    counts, edges = np.histogram(r["outcomes"], bins=30)
    hist = pd.DataFrame({"Result ($)": (edges[:-1] + edges[1:]) / 2, "Stretches": counts})
    st.bar_chart(hist, x="Result ($)", y="Stretches")

st.divider()
st.caption(
    "This replays history; it can't predict the future, and crypto can fall further than it "
    "ever has before. It is not financial advice. Includes a 0.6% fee to buy and to sell, but not "
    "spreads, taxes, or coins that were delisted (only coins listed today are shown, which makes "
    "the past look a bit better than it was). Data: Coinbase public prices, refreshed every 6 hours."
)
