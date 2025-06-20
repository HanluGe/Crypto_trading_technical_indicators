# High Frequency Crypto Trading Strategy

**FINM 33150 - Quant Trading Strategies (Winter 2024)**  
**Team Members:**  
- Hanlu Ge  
- Junyuan Liu  
- Xianjie Zhang  
- Yong Li  

---

## Project Overview

This project explores high-frequency quantitative trading in the cryptocurrency market using 5-minute interval data. Leveraging the volatility, liquidity, and 24/7 structure of crypto assets, we developed a multi-strategy ensemble model targeting consistent risk-adjusted returns.

---

## Table of Contents

- [Introduction](#-project-overview)  
- [Data Overview](#-data-overview)  
- [Single Asset Analysis (BTC)](#-btc-analysis)  
- [Multi-Asset Comparison](#-multi-crypto-analysis)  
- [Sub-strategy Design](#-strategy-methodology)  
- [Ensemble Backtesting](#-integrated-strategy-backtest)  
- [Results & Insights](#-conclusion)  
- [How to Run](#-getting-started)  

---

## Data Overview

- **Frequency:** 5-minute interval OHLCV  
- **Assets:** 11 major cryptocurrencies  
- **Fields:** timestamp, open, high, low, close, volume  
- **Preprocessing:** handled missing values, resampling, feature engineering with `TA-Lib` and custom factors

---

## BTC Analysis

- Performed exploratory analysis on BTC:
  - Volatility clustering
  - Seasonality (hour-of-day)
  - Momentum and mean-reversion behavior
  - Rolling Sharpe/Sortino visualization

---

## Multi-Crypto Analysis

- Cross-sectional analysis across coins:
  - Signal consistency
  - Correlation matrices
  - Liquidity-adjusted performance

---

## Strategy Methodology

We implemented and compared multiple strategies:

### 1. Momentum / Mean-Reversion  
- Signal generation using custom z-score, RSI, volatility breakout  

### 2. Volatility-Adjusted Long-Short  
- Dynamic thresholds based on market volatility  
- Entry depends on signal strength × volatility band  
- Early exit if volatility spike triggers stop

### 3. Labeling + Tree-Based Ensemble  
- Market regime labeling using technical/meta labels  
- Train ensemble model (e.g. XGBoost) for directional prediction  
- Signal smoothing and aggregation for ensemble decision

---

## Integrated Strategy Backtest

- **Backtesting framework:** Custom implementation  
- **Metrics evaluated:**  
  - Annualized return  
  - Sharpe ratio  
  - Max drawdown  
  - Win rate  
- **Comparison:** individual sub-strategies vs ensemble

---

## Run the Notebook

jupyter notebook Final_Project_Submission_v4.ipynb

