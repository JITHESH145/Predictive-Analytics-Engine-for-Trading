import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


try:
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Scaler file not found. Please run data_preparation.py first.")
    st.stop()

st.title("Stock Price Prediction App")
st.write("This app predicts the next day's closing price. Select a model, enter a stock symbol, and a date range to get started.")


st.sidebar.header("User Inputs")


model_files = {
    'Linear Regression': 'Linear_Regression.pkl',
    'Polynomial Regression': 'Polynomial_Regression.pkl',
    'Ridge': 'Ridge.pkl',
    'Lasso': 'Lasso.pkl',
    'Decision Tree Regressor': 'Decision_Tree_Regressor.pkl',
    'Random Forest Regressor': 'Random_Forest_Regressor.pkl',
    'Support Vector Regression': 'Support_Vector_Regression.pkl',
    'Tuned Ridge': 'Tuned_Ridge.pkl',
    'Tuned Lasso': 'Tuned_Lasso.pkl',
    'Tuned Random Forest': 'Tuned_Random_Forest.pkl'
}
selected_model = st.sidebar.selectbox("Choose a Model", list(model_files.keys()))

stock_symbol = st.sidebar.text_input("Stock Symbol (e.g., TSLA)", value="TSLA")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
predict_button = st.sidebar.button("Predict")


def prepare_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        st.error("No data found for the given symbol and dates.")
        return None, None
    
    data['Next_Close'] = data['Close'].shift(-1)
    # Drop the last row with NaN in 'Next_Close'
    data = data.iloc[:-1]
    
    features = ['Open', 'High', 'Low', 'Volume', 'Close']
    X = data[features]
    
    # Use the loaded scaler
    X_scaled = scaler.transform(X)
    
    return X_scaled, data

# Load the selected model
try:
    model = joblib.load(model_files[selected_model])
except FileNotFoundError:
    st.error(f"Model file '{model_files[selected_model]}' not found. Please run model_training.py first.")
    st.stop()

# Handle prediction on button click
if predict_button:
    X_scaled, data = prepare_data(stock_symbol, start_date, end_date)
    if X_scaled is not None and data is not None:
        
        predictions = model.predict(X_scaled)
        
       
        st.subheader(f"Prediction Results using {selected_model}")
        results_df = pd.DataFrame({
            'Date': data.index,
            'Actual Next Close': data['Next_Close'],
            'Predicted Next Close': predictions
        })
        st.dataframe(results_df)
        
       
        st.subheader("Actual vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index, data['Next_Close'], label='Actual')
        ax.plot(data.index, predictions, label='Predicted')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.legend()
        st.pyplot(fig)


st.sidebar.header("Hypothetical Prediction")
open_price = st.sidebar.number_input("Open Price", value=0.0, format="%.2f")
high_price = st.sidebar.number_input("High Price", value=0.0, format="%.2f")
low_price = st.sidebar.number_input("Low Price", value=0.0, format="%.2f")
volume = st.sidebar.number_input("Volume", value=0)
prev_close = st.sidebar.number_input("Previous Day's Close", value=0.0, format="%.2f")

if st.sidebar.button("Predict Hypothetical"):
    if any(v == 0.0 for v in [open_price, high_price, low_price, prev_close]):
        st.sidebar.warning("Please fill in all price and volume fields for a hypothetical prediction.")
    else:
        custom_input = np.array([[open_price, high_price, low_price, volume, prev_close]])
        custom_scaled = scaler.transform(custom_input)
        custom_pred = model.predict(custom_scaled)
        st.sidebar.success(f"Predicted Next Close: ${custom_pred[0]:.2f}")
