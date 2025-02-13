import streamlit as st
from data_loader import get_stock_data, preprocess_data, create_sequences
from model import build_lstm_model, predict_future_price
import datetime
from datetime import timedelta
import numpy as np

def main():
    st.title("Stock Price Prediction & Investment Recommendation")
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    selected_stock = st.selectbox("Select a Stock", stocks)
    
    df = get_stock_data(selected_stock)
    df_scaled, scaler = preprocess_data(df)
    
    X, y = create_sequences(df_scaled)
    
    lstm_model = build_lstm_model((X.shape[1], 1))
    lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=1)
    
    st.line_chart(df['Close'].tail(365))
    
    st.subheader("Predict Future Stock Price")
    prediction_dates = {
        "Tomorrow": 1,
        "Day After Tomorrow": 2,
        "After 1 Week": 7,
        "After 2 Weeks": 14,
        "After 1 Month": 30
    }
    selected_date = st.selectbox("Select Future Date", list(prediction_dates.keys()))
    
    if st.button("Predict Price"):
        days_ahead = prediction_dates[selected_date]
        predicted_price = predict_future_price(lstm_model, df_scaled, scaler, days_ahead)
        st.write(f"Predicted price for {selected_stock} on {datetime.date.today() + timedelta(days=days_ahead)}: ${predicted_price:.2f}")
    
    st.write("Investment Recommendation: HOLD/BUY/SELL")

if __name__ == "__main__":
    main()
