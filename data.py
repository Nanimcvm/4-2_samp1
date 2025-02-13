import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period='2y')
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close']].values)
    return df_scaled, scaler

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps])
    return np.array(X), np.array(y)
