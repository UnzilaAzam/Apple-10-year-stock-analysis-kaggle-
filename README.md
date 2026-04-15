# <b>Data Analysis of Apple's 10-Year Stock Market Behavior</b>

## Project Overview

This project demonstrates the application of three fundamental machine learning paradigms on Apple (AAPL) stock market data spanning 10 years:

1. **Regression:** Predict next-day closing price
2. **Classification:** Predict next-day movement direction (Up/Down)
3. **Clustering:** Discover underlying market regimes

## Features

- **Comprehensive Data Analysis:** Complete exploratory data analysis with visualizations
- **Feature Engineering:** Technical indicators including Moving Averages, RSI, Volatility, etc.
- **Multiple ML Models:** Linear Regression, Random Forest, Logistic Regression, Decision Trees, K-Means
- **Interactive Flask Web App:** User-friendly interface for making predictions
- **Detailed Visualizations:** Charts, confusion matrices, ROC curves, and cluster analysis

## Web Application Features

### 1. Price Prediction (Regression)
- Input today's OHLC (Open, High, Low, Close) and Volume
- Get predicted next-day closing price
- View expected price change and percentage

### 2. Movement Prediction (Classification)
- Input technical indicators (Daily Return, Moving Averages, RSI, etc.)
- Get prediction: UP or DOWN
- View confidence levels and probabilities

### 3. Market Regime Identification (Clustering)
- Input monthly aggregated metrics
- Identify which market regime cluster the period belongs to
- Understand market conditions (volatile, stable, corrective)

## Model Performance

### Regression Results:
- **Linear Regression:** Baseline model
- **Random Forest:** Superior performance with lower RMSE and higher R²

### Classification Results:
- **Accuracy:** ~52-60% 
- **Best Model:** Random Forest Classifier
- **Evaluation:** Confusion matrices, ROC curves, precision/recall

### Clustering Results:
- **Optimal Clusters:** 3 (determined by elbow method and silhouette score)
- **Regimes Identified:**
  - Cluster 0: Low Volatility - Stable Growth
  - Cluster 1: High Volatility - Turbulent Conditions
  - Cluster 2: Moderate - Correction/Consolidation

## Technical Details

### Feature Engineering:
- Daily Return
- Moving Averages (5-day, 10-day, 20-day)
- Volatility (10-day rolling standard deviation)
- High-Low Spread
- Volume Change
- RSI (Relative Strength Index)

### Data Preprocessing:
- Date parsing and indexing
- Missing value handling
- Feature scaling using StandardScaler
- Train-test split (80-20)

### Evaluation Metrics:
- **Regression:** RMSE, MAE, R² Score
- **Classification:** Accuracy, Precision, Recall, ROC-AUC
- **Clustering:** Silhouette Score, Elbow Method

## Key Insights

1. Regression models capture price trends but perfect accuracy is impossible in stock prediction
2. Random Forest consistently outperforms simpler models across all tasks
3. Classification accuracy of 52-60% is normal for stock market direction prediction
4. Three distinct market regimes can be identified through unsupervised clustering
5. Technical indicators like RSI and moving averages are useful features

## Limitations

- Stock market prediction is inherently uncertain
- Past performance doesn't guarantee future results
- Models should be used for educational purposes, not financial advice
- External factors (news, global events) not captured in technical data
