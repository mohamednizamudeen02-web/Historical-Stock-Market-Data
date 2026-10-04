import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Load historical stock data
df = pd.read_csv("C:\data analytics task\Stock_data.csv")  # Replace with your dataset
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Visualize closing prices
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# Augmented Dickey-Fuller Test
def check_stationarity(data):
    result = adfuller(data)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is non-stationary.")

check_stationarity(df['Close'])

# ARIMA Model
model = ARIMA(df['Close'], order=(5, 1, 0))  # Order (p, d, q) based on ACF/PACF
arima_result = model.fit()
print(arima_result.summary())

# Forecasting
forecast = arima_result.forecast(steps=30)
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Actual')
plt.plot(pd.date_range(df.index[-1], periods=30, freq='B'), forecast, label='Forecast', color='red')
plt.legend()
plt.show()

# Scaling data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

# Split data
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Visualization
test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
plt.figure(figsize=(10, 6))
plt.plot(test_actual, label="Actual Prices")
plt.plot(predictions, label="Predicted Prices")
plt.title("LSTM Predictions vs Actual")
plt.legend()
plt.show()
