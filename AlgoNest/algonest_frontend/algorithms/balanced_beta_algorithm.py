# balanced_beta_algorithm.py

import pandas as pd
import numpy as np
import yfinance as yf

def calculate_MACD(data, span_short=12, span_long=26, span_signal=9):
    data['EMA_short'] = data['Adj Close'].ewm(span=span_short, adjust=False).mean()
    data['EMA_long'] = data['Adj Close'].ewm(span=span_long, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal_Line'] = data['MACD'].ewm(span=span_signal, adjust=False).mean()
    return data

def calculate_Bollinger_Bands(data, window=20):
    data['BB_Middle'] = data['Adj Close'].rolling(window=window).mean()
    data['BB_Std'] = data['Adj Close'].rolling(window=window).std()
    data['BB_Upper'] = data['BB_Middle'] + (2 * data['BB_Std'])
    data['BB_Lower'] = data['BB_Middle'] - (2 * data['BB_Std'])
    return data

def generate_signals(data):
    data['Buy'] = np.where(
        (data['MACD'] > data['Signal_Line']) &
        (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)) &
        (data['Adj Close'] > data['BB_Lower']) &
        (data['Adj Close'].shift(1) <= data['BB_Lower'].shift(1)),
        1,
        0
    )

    data['Sell'] = np.where(
        (data['MACD'] < data['Signal_Line']) &
        (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)) &
        (data['Adj Close'] < data['BB_Upper']) &
        (data['Adj Close'].shift(1) >= data['BB_Upper'].shift(1)),
        -1,
        0
    )
    return data

def balanced_beta_strategy(ticker):
    # Fetch historical data
    data = yf.download(ticker, period='1y', interval='1d')

    # Calculate indicators
    data = calculate_MACD(data)
    data = calculate_Bollinger_Bands(data)

    # Generate trading signals
    data = generate_signals(data)

    # Filter buy and sell signals
    buy_signals = data[data['Buy'] == 1]
    sell_signals = data[data['Sell'] == -1]

    # Print the signals
    print("Buy Signals:")
    print(buy_signals[['Adj Close', 'MACD', 'Signal_Line', 'BB_Lower']])
    print("\nSell Signals:")
    print(sell_signals[['Adj Close', 'MACD', 'Signal_Line', 'BB_Upper']])

if __name__ == "__main__":
    balanced_beta_strategy('AAPL')  # Example ticker symbol
