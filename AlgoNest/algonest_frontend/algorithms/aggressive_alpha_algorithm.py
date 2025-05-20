# aggressive_alpha_algorithm.py

import pandas as pd
import numpy as np
import yfinance as yf

def calculate_RSI(data, period=14):
    delta = data['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    average_gain = gain.rolling(window=period).mean()
    average_loss = loss.rolling(window=period).mean()
    rs = average_gain / average_loss
    RSI = 100 - (100 / (1 + rs))
    data['RSI'] = RSI
    return data

def calculate_EMA(data, span_short=12, span_long=26):
    data['EMA_short'] = data['Adj Close'].ewm(span=span_short, adjust=False).mean()
    data['EMA_long'] = data['Adj Close'].ewm(span=span_long, adjust=False).mean()
    return data

def generate_signals(data):
    data['Buy'] = np.where(
        (data['RSI'] < 30) &
        (data['RSI'].shift(1) >= 30) &
        (data['EMA_short'] > data['EMA_long']) &
        (data['EMA_short'].shift(1) <= data['EMA_long'].shift(1)),
        1,
        0
    )

    data['Sell'] = np.where(
        (data['RSI'] > 70) &
        (data['RSI'].shift(1) <= 70) &
        (data['EMA_short'] < data['EMA_long']) &
        (data['EMA_short'].shift(1) >= data['EMA_long'].shift(1)),
        -1,
        0
    )
    return data

def aggressive_alpha_strategy(ticker):
    # Fetch historical data
    data = yf.download(ticker, period='6mo', interval='1h')

    # Calculate indicators
    data = calculate_RSI(data)
    data = calculate_EMA(data)

    # Generate trading signals
    data = generate_signals(data)

    # Filter buy and sell signals
    buy_signals = data[data['Buy'] == 1]
    sell_signals = data[data['Sell'] == -1]

    # Print the signals
    print("Buy Signals:")
    print(buy_signals[['Adj Close', 'RSI', 'EMA_short', 'EMA_long']])
    print("\nSell Signals:")
    print(sell_signals[['Adj Close', 'RSI', 'EMA_short', 'EMA_long']])

if __name__ == "__main__":
    aggressive_alpha_strategy('TSLA')  # Example ticker symbol
