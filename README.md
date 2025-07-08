# PricingOptionsMastersThesis


Author: Kyle Bistrain

Date: August 29th, 2025

This repository contains the full implementation of a master's thesis project titled **"Deep Learning Framework for Option Pricing: A Walkforward Study of SPY Options"**. The goal is to evaluate whether a constrained multilayer perceptron (MLP) model can outperform classical Black-Scholes pricing, using end-of-day data.

## 🔍 Project Overview

- Models options using both the **Black-Scholes model** and **deep learning techniques**.
- Applies a **walkforward training strategy** across different market regimes.
- Includes **data preprocessing**, **feature engineering**, and **model evaluation metrics** segmented by moneyness.
- Uses implied volatility proxies like VIX and compares against realized volatility.

## 📁 Repository Structure

├── Deep Learning Framework for Option Pricing.pdf # Full thesis document

├── Thesis_Presentation_Kyle_Bistrain.pptx # Final defense slide deck

├── PricingOptionsMastersThesis.Rproj # RStudio project file

├── README.md # Project documentation

├── option-pricing-via-machine-learning_KyleB...ipynb # Core modeling notebook

├── optionpricing_data_cleaning_may28.ipynb # Data cleaning and preprocessing

├── optionpricing_mlp_walkforward_may28.ipynb # Walkforward training and evaluation


## 🧪 Methodology

- **American Option Pricing:** While Black-Scholes assumes European-style exercise, this study approximates early exercise behavior heuristically using ex-dividend flags.
- **Volatility Inputs:**
  - Realized volatility calculated using rolling log-return windows.
  - Implied volatility approximated with VIX, VIX3M, and VIX6M for matched horizons.
- **Training Strategy:** Rolling window walkforward CV with monthly step-forward retraining and evaluation.
- **Constraints:** The MLP model enforces no-arbitrage constraints such as:
  - Monotonicity in strike
  - Convexity of the price surface
  - Time-consistency and intrinsic value adherence

---

## 📊 Key Results

- The constrained MLP achieves lower RMSE and MAE than Black-Scholes in many test periods.
- Neural networks demonstrate flexibility across moneyness regimes while maintaining arbitrage-free conditions.



