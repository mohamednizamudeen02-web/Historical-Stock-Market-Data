Analyzing Historical Stock Market Data

This project analyzes historical stock market data to uncover trends, patterns, and correlations. Using ARIMA and LSTM models, it predicts future stock prices while demonstrating the practical applications of data science in financial markets.

Features

1.Data Preprocessing
  Load and clean historical stock data.
  Perform time series stationarity checks using the Augmented Dickey-Fuller (ADF) test.
  
2.Visualization
  Plot stock closing prices over time.
  Compare actual vs predicted values in ARIMA and LSTM models.
  
3.Time Series Modeling
  ARIMA Model for short-term stock price forecasting.
  LSTM Model for capturing sequential dependencies in stock price data.

4.Performance Evaluation
  Visualize model predictions compared to actual stock prices.
  
Technologies Used
  Programming Language: Python
  
Libraries:
  Data Analysis: Pandas, NumPy
  Visualization: Matplotlib
  Time Series Modeling: Statsmodels (ARIMA), TensorFlow (LSTM)
  Data Scaling: Scikit-learn
  
How to Run the Project

Prerequisites:
Install Python (>=3.10).
Install dependencies using the following command:
pip install pandas numpy matplotlib statsmodels scikit-learn tensorflow  

Steps:
Clone the repository:
git clone https://github.com/your-username/stock-market-analysis.git  
cd stock-market-analysis  
Replace the Stock_data.csv path in the code with your dataset file path.

Run the script:
python stock_analysis.py  

Outputs
ARIMA Forecast: Short-term future predictions of stock prices.
LSTM Predictions: Sequential data predictions for test datasets.
Visualization: Compare actual vs predicted stock prices.

Project Highlights
Combines statistical and machine learning models for financial data analysis.
Demonstrates real-world applications of ARIMA and LSTM for time series forecasting.

Disclaimer
This project is for educational purposes only and should not be used for financial decision-making.
