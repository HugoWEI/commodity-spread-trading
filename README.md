# Cross-Market Commodity Spread Trading

A systematic study of statistical arbitrage strategies across energy markets, focusing on **Henry Hub–TTF gas** and **Brent–WTI crude oil** spreads.

---

## 🧠 Overview

This project builds and evaluates a full pipeline for spread trading:
Market Data -> Hedge Ratio Estimation -> Signal Generation -> Risk Filtering -> Execution -> Evaluation


Key techniques:
- Cointegration / rolling regression
- State-space (Kalman) hedge ratio
- Z-score mean-reversion
- Volatility scaling
- Regime filtering
- Walk-forward backtesting

---

## 📊 Key Results

| Model | Sharpe | Max DD | Notes |
|------|--------|--------|------|
| HH–TTF Rolling Beta | ~0.5 → ~0.3 | High | Unstable hedge ratio |
| HH–TTF + ML Filter | ~1.5 (unstable) | High | Not robust |
| Brent–WTI Baseline | ~0.33 | Very high | Stable but weak |
| Brent–WTI Production OLS | ~0.7 | Low | Controlled risk |
| **Brent–WTI Kalman** | **~1.43** | **~-2** | Adaptive + robust |

---

## 🔍 Models & Methodologies

---

### Henry Hub – TTF (Rolling Beta Baseline)

#### Methodology
- Cross-market gas spread (US vs Europe)
- FX + unit normalization
- Rolling OLS hedge ratio
- Z-score mean-reversion
- Walk-forward evaluation

#### Results
- Strong long-term relationship
- **Highly unstable hedge ratio**
- Poor out-of-sample performance
- Large drawdowns

---

### Henry Hub – TTF (ML Filtered Strategy)

#### Methodology
- Logistic regression filters trades
- Features:
  - Z-score lags
  - residual momentum
  - volatility ratios
  - hedge ratio dynamics
- Train-only calibration

#### Results
- Improves selectivity in-sample
- **Fails under walk-forward**
- Sensitive to regime shifts

---

### Brent – WTI (Rolling Beta Baseline)

#### Methodology
- Crude oil spread (global vs US benchmark)
- Direct spread / beta ≈ 1
- Z-score signal

#### Results
- Stable hedge ratio
- Weak mean-reversion signal
- High drawdown without controls

---

### Brent – WTI (Production Walk-Forward, OLS)

#### Methodology
- Train-only hedge ratio estimation
- Lagged signal execution (no look-ahead)
- Volatility-based position sizing
- Regime filtering
- Transaction costs included

#### Results
- Sharpe ~0.7
- Strong drawdown control
- Low-frequency, selective trading

---

### ⭐ Brent – WTI (Kalman Production Walk-Forward)

#### Methodology
- **State-space hedge ratio (Kalman filter)**
- Adaptive alpha and beta estimation
- Grid search of Kalman parameters (delta, R) on train only
- Volatility scaling + regime gating
- Walk-forward:
  - train → calibrate → freeze → test
- Fully lagged execution

#### Results
- Sharpe ~1.43  
- Max drawdown ~-2  
- Low trade frequency (~8 trades per split)  
- ~13% of periods with no trades  

---

> The strategy exhibits strong performance in periods where sufficient trading opportunities exist (median split Sharpe ~2.8), but activity is sparse, with ~13% of periods producing no trades. This highlights the importance of opportunity filtering and capacity constraints in spread trading.

---

## 🚀 How to Run

```powershell
# Download data
python -m src.data.download_data

# Prepare datasets
python -m src.data.prepare_data
python -m src.data.prepare_brent_wti

# Run production Kalman model
python -m src.backtest.brent_wti_kalman_production_walkforward