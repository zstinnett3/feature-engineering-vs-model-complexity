# Feature Engineering vs Model Complexity in Time Series Forecasting

This repository contains the code and notebook that support the article:

> **“Stop Blaming Your Model: How Feature Engineering Wins More Than You Think”**

The goal is to show, on a synthetic but realistic daily time series, how much improvement you can get from **systematic feature engineering** before changing the model class. We keep the model family fixed (XGBoost) and vary only the feature set.

---

## 1. Project Overview

We construct a univariate daily time series that mimics an operational metric such as demand, traffic, or usage. The series has:

- A **slow upward trend**
- **Weekly seasonality** (weekday vs weekend pattern)
- **Annual seasonality** (some months systematically higher)
- **Holiday shocks** on a few fixed dates (e.g. Jan 1, Jul 4, Dec 25):contentReference[oaicite:0]{index=0}  

On top of this series, we define a **14-day-ahead forecasting task** and compare three approaches:

1. **Naive 14-day persistence baseline**  
   \(\hat{y}_{t+14} = y_t\)

2. **XGBoost with weak features**  
   - `lag_1`, `lag_7`,  
   - a simple `day_index`

3. **XGBoost with engineered features**  
   - All weak features, plus  
   - Calendar indicators: `day_of_week`, `month`, `is_weekend`, `is_holiday`  
   - Cyclic encodings: sine/cosine transforms for day-of-week and month  
   - Additional lag: `lag_14`  
   - Rolling statistics: 7-day and 30-day rolling mean and standard deviation (leakage-safe)

The XGBoost hyperparameters are held **constant** across models. Any performance difference comes purely from the feature space, not from changing architectures or tuning.

---

## 2. Repository Structure

- `TDS Feature Engineering.ipynb`  
  Main Jupyter notebook used for the article. Contains:
  - Synthetic data generation
  - Exploratory data analysis and plots
  - Forecasting task definition (14-day horizon)
  - Feature engineering
  - Baseline and XGBoost experiments
  - Results and relative improvement calculations

- `FeatureEngineeringArticle.py`  
  Standalone Python script with the core experiment logic (data generation, feature engineering, and model training) without the narrative/EDA cells.

You can run either the notebook or the script to reproduce the main results.

---

## 3. Experimental Setup

**Frequency:** Daily  
**Horizon:** 14 days ahead  
**History:** 5 years of synthetic data  
**Split:** Time-based  
- 60% train  
- 20% validation  
- 20% test  

**Models:**

- Naive baseline: 14-day persistence  
- XGBoost regressor (`reg:squarederror`) with fixed hyperparameters:
  - `n_estimators = 300`
  - `learning_rate = 0.05`
  - `max_depth = 4`
  - `subsample = 0.9`
  - `colsample_bytree = 0.9`
  - `random_state = 42`

---

## 4. Results

Test-set performance (values will be exactly reproducible from the notebook):

| Model        | MAE    | RMSE   |
|-------------|--------|--------|
| baseline    | 0.3612 | 0.4798 |
| xgb_weak    | 0.3144 | 0.4107 |
| xgb_rich    | 0.3012 | 0.3865 |

Relative improvements (MAE):

- **Engineered vs weak features:**  
  ~**4.2%** reduction (0.314 → 0.301)

- **Engineered vs naive baseline:**  
  ~**16.6%** reduction (0.361 → 0.301)

All three models use the **same XGBoost configuration**. The only change is the feature set.

---

## 5. How to Run

### 5.1. Requirements

Tested with:

- Python 3.10+
- `numpy`
- `pandas`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `statsmodels`
- `jupyter` / `notebook` (for running the `.ipynb`)

You can install dependencies with:

```bash
pip install numpy pandas scikit-learn xgboost matplotlib statsmodels notebook

5.2. Running the notebook

jupyter notebook "TDS Feature Engineering.ipynb"

Then run all cells in order.

6. Reproducibility and Usage

The synthetic data are generated from a fixed random seed (np.random.seed(42)), so results are deterministic.
You are free to adapt the notebook for your own experiments (e.g., different horizons, model classes, or feature sets).
The repository is intended as a teaching/demo resource for:
	How to construct realistic synthetic time series
	How to set up leak-free forecasting experiments
	How to compare feature sets while holding the model family constant
	
7. License
MIT License

Copyright (c) 2025 Zack Stinnett

8. Citation / Reference
Example and code structure adapted from Zack Stinnett’s feature-engineering time series demo (GitHub link).
