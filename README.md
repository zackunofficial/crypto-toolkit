# 🪙 Crypto Toolkit

Two browser tools that use real Coinbase price history to answer the questions retail crypto investors should ask first. Neither tool promises returns; both show the odds and how much to trust them.

## Try it online (no install)

**👉 [Open Crypto Toolkit](https://add-your-streamlit-link-here)**

Works in any browser. If nobody has used it in the last 12 hours, it takes about a minute to wake up.

## How to use it

Pick a tool from the menu at the top of the page.

**🧭 Reality Check: is my goal realistic for my risk?**

1. Enter how much you'd invest, the most you're OK losing, your goal (%), and a timeframe.
2. Press **Show my options**. The first run downloads about 3 years of prices and takes about a minute.
3. Read the **Reality check** box first: it says whether your goal is reachable, possible, or a stretch at your risk level, and what to change if it's a stretch.
4. Scan **Your options** to see how much to put in each coin, how often your goal was reached in the past, and typical and bad-case results in dollars.

**🔭 Forecast Lab: where might a coin go, and should I trust the forecast?**

1. Pick any coin on Coinbase and drag the slider to a timeframe from 1 day to 3 months.
2. Press **Run forecast**. The first run for each coin takes about a minute while it downloads data and tests itself.
3. Read the three results: the chance it ends higher, the likely price range, and the expected swing.
4. Check the **How much to trust this** box before relying on the direction call:

| Box color | Meaning | How to use the forecast |
| --- | --- | --- |
| Green | The direction call has shown some skill on past data | One input among others, never a sure thing |
| Yellow | Only slightly better than guessing | A weak hint; lean on the range and swing |
| Red | Has not beaten simple guessing | Ignore the direction; use only the range and swing |

Open **Under the hood** to see what's driving the forecast, its track record, the math, and which data sources loaded.

## Run it on your own computer

You need **Python 3.11 or newer** ([download](https://www.python.org/downloads/)).

**1. Get the code.** Pick one:

- **No Git:** click the green **Code** button at the top of this page, choose **Download ZIP**, and unzip it. The folder will be called `crypto-toolkit-main`.
- **With Git:**

  ```bash
  git clone https://github.com/YOUR-USERNAME/crypto-toolkit.git
  ```

**2. Open a terminal inside the project folder.** This step matters: the commands below only work from inside the folder.

```bash
cd ~/Downloads/crypto-toolkit-main   # if you downloaded the ZIP
cd crypto-toolkit                    # if you used git clone
```

**3. Install the packages** (first time only):

```bash
python -m pip install -r requirements.txt
```

**4. Start the app:**

```bash
python -m streamlit run streamlit_app.py
```

Your browser opens to `http://localhost:8501`. Keep the terminal open while you use the app, and press **Control + C** in it to stop.

**Troubleshooting**

| What you see | Fix |
| --- | --- |
| `File does not exist: streamlit_app.py` or `Could not open requirements file` | You're not inside the project folder. Repeat step 2. |
| `No module named streamlit` | Run step 3 again. |
| Streamlit asks for an email the first time | Press Return to skip it. |
| A data source shows "unavailable" in Forecast Lab | A free source is briefly down; the forecast runs without it. |

## Methods

| Component | Technique |
| --- | --- |
| Direction | Ensemble of logistic regression and gradient-boosted trees (scikit-learn) |
| Price range | Quantile regression (pinball loss) with conformalized calibration |
| Volatility | GARCH(1,1) fit by maximum likelihood (SciPy) |
| Validation | Walk-forward testing that trains only on outcomes known at each point in time |
| Position sizing | Bad-case (10th percentile) drawdown scaled to the user's loss limit |
| Costs | 0.6% fee on every buy and sell |

## Data (all free, no API keys)

Coinbase (prices and volume), FRED (S&P 500, Nasdaq, VIX, 10-year yield, US dollar), Alternative.me (Crypto Fear & Greed Index), GDELT (crypto news tone), and Wikipedia (daily page views as a public-attention signal).

## Project structure

| File | Purpose |
| --- | --- |
| `streamlit_app.py` | Entry point: Home page and navigation |
| `reality_check.py` | Reality Check tool |
| `forecast_lab.py` | Forecast Lab tool |
| `requirements.txt` | Package versions |

## Limitations

Results replay history and cannot predict the future. Only coins listed today are included (survivorship bias), spreads and taxes are not modeled, and longer horizons rely on few truly independent periods. This project is for education and is not financial advice.

## About

Built by Zach Sanders with AI-assisted development (Claude). Released under the MIT License.
