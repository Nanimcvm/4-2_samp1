import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_future_price(model, data, scaler, days_ahead):
    input_data = data[-60:].reshape(1, 60, 1)
    future_price = model.predict(input_data)
    return scaler.inverse_transform(future_price)[0][0]
