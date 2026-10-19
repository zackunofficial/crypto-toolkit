# 🪙 Crypto Toolkit

Two browser tools that use real Coinbase price history to answer the questions retail crypto investors should ask first. Neither tool promises returns; both show the odds and how much to trust them.

**Live app:** _add your Streamlit link here after deploying_

## The tools

**🧭 Reality Check: is my goal realistic for my risk?**
Enter an amount, the most you're willing to lose, a target gain, and a timeframe. The app replays about three years of daily prices for the 25 most-traded coins, sizes each position so a bad-case dip stays inside your loss limit, and reports how often your goal was actually reached, with fees included.

**🔭 Forecast Lab: where might a coin go, and should I trust the forecast?**
Pick any coin on Coinbase and a horizon from 1 day to 3 months. The app forecasts the chance of a rise, a calibrated price range, and expected volatility, then grades its own past forecasts with walk-forward testing and tells you plainly when it has no proven skill.

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

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

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
