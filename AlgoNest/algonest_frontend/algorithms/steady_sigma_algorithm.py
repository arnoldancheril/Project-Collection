# steady_sigma_algorithm.py

import pandas as pd
import numpy as np
import yfinance as yf

def calculate_SMA(data, window_short=50, window_long=200):
    data['SMA_short'] = data['Adj Close'].rolling(window=window_short).mean()
    data['SMA_long'] = data['Adj Close'].rolling(window=window_long).mean()
    return data

def calculate_RSI(data, period=14):
    delta = data['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()
    rs = average_gain / average_loss
    RSI = 100 - (100 / (1 + rs))
    data['RSI'] = RSI
    return data

def generate_signals(data):
    data['Buy'] = np.where(
        (data['SMA_short'] > data['SMA_long']) &
        (data['SMA_short'].shift(1) <= data['SMA_long'].shift(1)) &
        (data['RSI'] > 40),
        1,
        0
    )

    data['Sell'] = np.where(
        (data['SMA_short'] < data['SMA_long']) &
        (data['SMA_short'].shift(1) >= data['SMA_long'].shift(1)) &
        (data['RSI'] < 60),
        -1,
        0
    )
    return data

def steady_sigma_strategy(ticker):
    # Fetch historical data
    data = yf.download(ticker, period='5y', interval='1d')

    # Calculate indicators
    data = calculate_SMA(data)
    data = calculate_RSI(data)

    # Generate trading signals
    data = generate_signals(data)

    # Filter buy and sell signals
    buy_signals = data[data['Buy'] == 1]
    sell_signals = data[data['Sell'] == -1]

    # Print the signals
    print("Buy Signals:")
    print(buy_signals[['Adj Close', 'SMA_short', 'SMA_long', 'RSI']])
    print("\nSell Signals:")
    print(sell_signals[['Adj Close', 'SMA_short', 'SMA_long', 'RSI']])

if __name__ == "__main__":
    steady_sigma_strategy('SPY')  # Example ticker symbol
