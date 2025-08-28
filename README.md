# deep-learning-option-pricing

Author: Kyle Bistrain

Date: August 29th, 2025

This repository contains the full implementation of a master's thesis project titled **"Deep Learning Framework for Option Pricing"**. The goal is to evaluate whether a multilayer perceptron (MLP) model can outperform classical Black-Scholes pricing, using end-of-day data.

## 🔍 Project Overview

- Models options using both the **Black-Scholes model** and **deep learning techniques**.
- Applies a **walkforward training strategy** across different market regimes.
- Includes **data preprocessing**, **feature engineering**, and **model evaluation metrics** segmented by moneyness.
- Uses implied volatility proxies like VIX and compares against realized volatility.

## 📁 Repository Structure

├── README.md                                            # This documentation file

├── Deep Learning Framework for Option Pricing.pdf       # Final thesis document

├── Thesis_Presentation_Kyle_Bistrain.pptx               # Defense slide deck

├── optionpricing_cleaning_final.py             # Data preprocessing notebook

└── optionpricing_mlp_walkforward_final.py           # Walkforward MLP training and evaluation


## 🧪 Methodology

- **American Option Pricing:** While Black-Scholes-Merton assumes European-style exercise, this study approximates early exercise behavior heuristically using ex-dividend flags and days to ex-dividend date.
- **Volatility Inputs:**
  - Realized volatility calculated using rolling log-return windows.
  - Implied volatility approximated with VIX3M, and VIX6M for matched horizons.
- **Training Strategy:** Rolling window walkforward CV with monthly step-forward retraining and evaluation.
- **Constraints:** The MLP model fails to enforce no-arbitrage constraints such as:
  - Monotonicity in strike
  - Monotonicity in price

## 📊 Key Results

- Volatility estimates make a bigger difference than architecture. 
- The constrained MLP achieves lower RMSE and MAE than Black-Scholes-Merton in several test periods.
- Neural networks demonstrate flexibility across moneyness regimes, however violates no aribitrage conditions.


## Citation

Bistrain, Kyle (2025). Deep Learning Framework for Option Pricing: A Walkforward Study of SPY Options. 
Master’s Thesis, Cal Poly San Luis Obispo.


