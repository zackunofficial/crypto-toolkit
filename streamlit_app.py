"""
Crypto Toolkit: one app, two tools, built on live Coinbase data.

  Reality Check  - is my goal realistic for the risk I'm willing to take?
  Forecast Lab   - where might a coin go next, and how much should I trust that forecast?

Setup (once):  pip install -r requirements.txt
Run:           streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="Crypto Toolkit", page_icon="🪙", layout="wide")


def home():
    st.title("🪙 Crypto Toolkit")
    st.write("Two tools that use real Coinbase price history to help you think clearly about crypto. "
             "Neither one promises returns; both show you the odds and how much to trust them.")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("🧭 Reality Check")
        st.write("Enter how much you'd invest, the most you're OK losing, and your goal. It replays "
                 "about 3 years of history for the 25 most-traded coins and shows how often your goal "
                 "was actually reached, sized to your loss limit.")
        st.page_link(reality_page, label="Open Reality Check", icon="🧭")
    with right, st.container(border=True):
        st.subheader("🔭 Forecast Lab")
        st.write("Pick any coin on Coinbase and a timeframe. It forecasts direction, a likely price "
                 "range, and expected swings, then grades its own past forecasts so you know how much "
                 "to trust it.")
        st.page_link(forecast_page, label="Open Forecast Lab", icon="🔭")

    st.divider()
    st.caption("Both tools replay the past and can't predict the future with certainty. Crypto can fall "
               "further than it ever has before. This is not financial advice. Data: Coinbase, FRED, "
               "Alternative.me, GDELT, and Wikipedia (all free and public).")


home_page = st.Page(home, title="Home", icon="🏠", default=True)
reality_page = st.Page("reality_check.py", title="Reality Check", icon="🧭", url_path="reality-check")
forecast_page = st.Page("forecast_lab.py", title="Forecast Lab", icon="🔭", url_path="forecast-lab")

st.navigation([home_page, reality_page, forecast_page], position="top").run()
