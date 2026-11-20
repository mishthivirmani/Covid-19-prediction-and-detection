# Covid-19-prediction and analysis
# COVID-19 Prediction and Analysis Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-1.6+-orange.svg)
![Status](https://img.shields.io/badge/Academic%20Project-Final%20Year-green.svg)

A comprehensive data science and machine learning college project designed to analyze global COVID-19 transmission patterns, explore epidemiological dynamics, and forecast future daily infection rates using time-series machine learning models without future lookahead data leakage.

---

## 📌 Project Abstract
The COVID-19 pandemic caused unprecedented public health challenges worldwide. Accurately analyzing spread patterns and forecasting short-term case trajectories is essential for resource planning and epidemiological understanding. 

This project implements an end-to-end data pipeline that:
1. **Ingests** historical epidemic data from the curated Johns Hopkins University CSSE archive.
2. **Cleans and standardizes** time-series observations across 198 countries.
3. **Engineers leak-free temporal features**, including autoregressive lags ($t-1, t-2, t-3, t-7, t-14$) and historical rolling moving averages.
4. **Benchmarks candidate regression models** (Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor) using strict chronological train/test splitting.
5. **Deploys an interactive Streamlit web dashboard** featuring real-time KPI tracking, exploratory visualizations, country-level drilldowns, and multi-day forecasting.

---

## 🎯 Key Objectives
- **Data Ingestion & Hygiene**: Validate, clean, and chronologically sort multi-country epidemic data.
- **Exploratory Data Analysis (EDA)**: Uncover global case dynamics, fatality trajectories, and international rankings using interactive Plotly charts and Seaborn heatmaps.
- **Leakage Prevention**: Strictly split data chronologically (no randomized shuffling) and construct rolling statistics based exclusively on historical observations ($t-1$).
- **Machine Learning Benchmarking**: Train and compare multiple regressors using standard statistical metrics ($MAE$, $RMSE$, and $R^2$).
- **Web Dashboard**: Provide a clean, user-friendly interface for stakeholders and evaluators to explore metrics and generate forecasts.

---

## 📂 Project Architecture

```text
covid19_prediction_analysis/
│
├── data/
│   ├── .gitkeep
│   └── covid_data.csv                 # Johns Hopkins CSSE dataset (161,568 rows)
│
├── models/
│   ├── .gitkeep
│   ├── covidpredictionmodel.pkl       # Best-performing trained model artifact
│   ├── covid_prediction_model.pkl      # Alias artifact
│   └── feature_config.json            # Model parameters and feature column metadata
│
├── notebooks/
│   ├── .gitkeep
│   └── covid_eda_and_modeling.ipynb   # Interactive analysis & experimental notebook
│
├── reports/
│   └── figures/                       # Generated interactive & static charts
│       ├── actual_vs_predicted.html
│       ├── actual_vs_predicted.png
│       ├── cases_over_time.html
│       ├── correlation_heatmap.png
│       ├── daily_cases_ma.html
│       ├── deaths_over_time.html
│       ├── recoveries_over_time.html
│       └── top_countries.html
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_loader.py                 # Safe dataset ingestion & verification
│   ├── data_cleaning.py               # Cleaning pipeline & derived feature engineering
│   ├── analysis.py                    # Global & country-level epidemiological analytics
│   ├── visualization.py               # Plotly interactive & Seaborn visualizers
│   └── prediction.py                  # Lag feature engineering, training, & forecasting
│
├── app.py                             # Streamlit interactive web application
├── train_model.py                     # Standalone ML training and evaluation script
├── requirements.txt                   # Project Python dependencies
├── README.md                          # Comprehensive project documentation
└── .gitignore                         # Git exclusion rules
```

---

## 🔬 Methodology & Feature Engineering

### 1. Data Cleaning
- **Datetime Casting**: Converted raw dates into `pd.to_datetime` format and sorted chronologically by `[country, date]`.
- **Deduplication**: Removed any redundant `(country, date)` observations.
- **Sanitization**: Removed negative adjustments resulting from clerical reporting revisions by applying zero-clipping (`.clip(lower=0)`).

### 2. Time-Series Feature Engineering (No Lookahead Leakage)
To forecast daily new cases ($y_t$), predictors are constructed strictly from historical observations:
- **Autoregressive Lags**: $y_{t-1}, y_{t-2}, y_{t-3}, y_{t-7}, y_{t-14}$
- **Historical Rolling Means**: 7-day and 14-day rolling averages computed on $y_{t-1}$ backwards.
- **Historical Rolling Standard Deviations**: 7-day and 14-day volatility measures.
- **Calendar Indicators**: Day-of-week (capturing weekend reporting lags) and Day-of-year.

### 3. Chronological Train-Test Split
- **Training Window**: `2020-02-05` to `2021-11-06` (641 days / 80% split)
- **Testing Window**: `2021-11-07` to `2022-04-16` (161 days / 20% split)
- **Zero Shuffling**: Random shuffling in time series introduces catastrophic data leakage; chronological splitting ensures validation mimics real-world forward testing.

---

## 📊 Model Benchmark Results

Evaluated on the unseen chronological test set (covering the global Omicron wave):

| Model | MAE | RMSE | $R^2$ Score | Performance Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **153,028.9** | **227,294.9** | **0.9432** | ⭐ **Selected Best Model** |
| **Random Forest Regressor** | 833,483.3 | 1,193,694.2 | -0.5660 | Bounded by training ceiling |
| **Gradient Boosting Regressor** | 863,587.3 | 1,224,839.4 | -0.6488 | Bounded by training ceiling |

### Academic Discussion: Why Did Linear Regression Outperform Decision Trees?
Tree-based ensemble models (Random Forest and Gradient Boosted Trees) make predictions by averaging training samples in terminal leaf nodes. Consequently, they **cannot extrapolate values greater than the maximum target observed during training** ($\approx 900,000$ cases/day). When the unseen test period experienced the unprecedented Omicron surge (peaking at $> 4,000,000$ cases/day), tree models under-predicted severely. 

In contrast, **Linear Regression** learns continuous proportional lag relationships ($y_t \propto \beta_1 y_{t-1} + \dots$), enabling predictions to scale upward dynamically and achieve an $R^2$ score of **0.9432**.

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PowerShell, Bash, or Command Prompt

### 2. Environment Setup
```powershell
# Navigate into the project folder
cd covid19_prediction_analysis

# Create virtual environment
py -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
# source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Running Data & Training Scripts
```powershell
# 1. Verify dataset loading
python src/data_loader.py

# 2. Test cleaning pipeline
python src/data_cleaning.py

# 3. Run exploratory data analysis
python src/analysis.py

# 4. Generate and export visualization charts
python src/visualization.py

# 5. Train models and save artifacts
python train_model.py
```

### 4. Launching the Interactive Web Dashboard
```powershell
.\.venv\Scripts\streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

---

## 💻 Dashboard Capabilities
- **Overview Dashboard**: Global KPI summary cards (Total Cases, Deaths, Peak Recoveries, Active Cases, Latest Daily Cases) and transmission trajectory charts.
- **Data Analysis**: Country-specific time-series explorer with 7-day and 14-day moving averages, mortality curves, and recovery histories.
- **Prediction**: Interactive forecasting engine for 3 to 30 days ahead with projected totals, average daily rates, and expected trend percentages.
- **Model Performance**: Full benchmark leaderboard, metric cards ($MAE$, $RMSE$, $R^2$), and evaluation methodology discussion.
- **Dataset Explorer**: Upload custom CSV datasets, inspect schema, missing value percentages, and export cleaned CSVs.

---

## ⚠️ Academic & Medical Disclaimer
> **Important Notice**: This project is developed solely for **academic, educational, and computational demonstration purposes** as a college computer science/data science project. 
> 
> The predictive outputs are mathematical time-series approximations and **must NOT be construed as medical advice, clinical diagnoses, epidemiological guidance, or public health policy directives**. No clinical validation has been performed.
