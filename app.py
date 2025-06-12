import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Set up Streamlit
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")
st.markdown("Fetch stock data from Yahoo Finance and predict using a pre-trained LSTM model.")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY.NS):", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

if st.button("Fetch & Predict"):
    try:
        # Download data from yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.warning("No data fetched. Please check the ticker and date range.")
        else:
            st.success(f"Downloaded {len(df)} records for {ticker}")
            df = df[['Close']]  # Use only 'Close' column

            # Display raw data
            st.subheader("📉 Closing Price")
            st.line_chart(df['Close'])

            # Scaling
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)

            # Prepare test data from last 100 timesteps
            X_test = []
            for i in range(100, len(scaled_data)):
                X_test.append(scaled_data[i-100:i])
            X_test = np.array(X_test)

            # Load model
            model = load_model("stock_model.h5")

            # Predict
            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)
            actual = scaler.inverse_transform(scaled_data[100:])

            # Plot
            st.subheader("📈 Predicted vs Actual Price")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(actual, label="Actual Price", color="red")
            ax.plot(predictions, label="Predicted Price", color="blue")
            ax.set_title(f"Stock Price Prediction vs Actual for {ticker}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Metrics
            mae = np.mean(np.abs(actual - predictions))
            mse = np.mean(np.square(actual - predictions))
            rmse = np.sqrt(mse)

            st.subheader("📊 Model Evaluation")
            st.write(f"**MAE**: {mae:.2f}")
            st.write(f"**MSE**: {mse:.2f}")
            st.write(f"**RMSE**: {rmse:.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
