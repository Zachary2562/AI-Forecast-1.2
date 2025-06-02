import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
import ta
import sys

# App config
st.set_page_config(page_title="AI Forecast App", layout="wide")
st.title("AI Forecast App By Zachary2562")

# Sidebar settings
import os

# Load ticker list from file or use a default list
if os.path.exists("tickers.txt"):
    ticker_list = open("tickers.txt").read().splitlines()
else:
    ticker_list = ["AAPL", "GOOG", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
selected_ticker = st.sidebar.selectbox("Select an option", ticker_list)
custom_ticker = st.sidebar.text_input("🔎 Search Yahoo Finance (e.g., AAPL, BTC-USD)")
ticker = custom_ticker.upper() if custom_ticker else selected_ticker

high_accuracy = st.sidebar.toggle("Enable High Accuracy Forecast")

enable_lstm = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    enable_lstm = sys.version_info < (3, 13)
except:
    enable_lstm = False

# Load data
@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, start="2010-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
        df.dropna(inplace=True)
        return df
    except:
        return pd.read_csv("sample_data.csv")

df = load_data(ticker)

# Show data
st.subheader(f"Price History for {ticker}")
st.line_chart(df["Close"])

# Technical indicators
df_clean = df[["Close"]].dropna().copy()

# Ensure Close is a 1D Series
close_series = df_clean["Close"]
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.iloc[:, 0]

rsi = ta.momentum.RSIIndicator(close=close_series).rsi()
df.loc[df_clean.index, "RSI"] = rsi

macd = ta.trend.MACD(close=close_series).macd()
df.loc[df_clean.index, "MACD"] = macd

ema12 = ta.trend.EMAIndicator(close=close_series, window=12).ema_indicator()
df.loc[df_clean.index, "EMA12"] = ema12

ema26 = ta.trend.EMAIndicator(close=close_series, window=26).ema_indicator()
df.loc[df_clean.index, "EMA26"] = ema26

bb = ta.volatility.BollingerBands(close=close_series)
boll_high = bb.bollinger_hband()
df.loc[df_clean.index, "Bollinger High"] = boll_high
boll_low = bb.bollinger_lband()
df.loc[df_clean.index, "Bollinger Low"] = boll_low


# Additional technical indicators
if set(["High", "Low", "Close", "Volume"]).issubset(df.columns):

    # VWAP
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low']) / 2).cumsum() / df['Volume'].cumsum()

    # ADX
    adx = ta.trend.ADXIndicator(high=df["High"], low=df["Low"], close=df["Close"])
    df["ADX"] = adx.adx()

    # OBV
    obv = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"])
    df["OBV"] = obv.on_balance_volume()

    st.subheader("📊 Additional Indicators")
    st.line_chart(df[["VWAP", "ADX", "OBV"]])

# Prophet Forecast
st.subheader("📅 Prophet Forecast")
df_prophet = df.reset_index()[["Date", "Close"]]
df_prophet.columns = ["ds", "y"]
model = Prophet()
if high_accuracy:
    model.add_seasonality(name='monthly', period=30.5, fourier_order=10)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=15)
    model.changepoint_prior_scale = 0.5

model.fit(df_prophet)
future = model.make_future_dataframe(periods=365*5)
forecast = model.predict(future)
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Optional LSTM Forecast
if enable_lstm:
    st.subheader("🔁 LSTM Forecast (Experimental)")
    from sklearn.preprocessing import MinMaxScaler

    data = df.filter(["Close"])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model_lstm.add(LSTM(50, return_sequences=False))
    model_lstm.add(Dense(25))
    model_lstm.add(Dense(1))
    
model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    
epochs_to_use = 50 if high_accuracy else 20
history = model_lstm.fit(x_train, y_train, batch_size=32, epochs=epochs_to_use, verbose=1)


    test_data = scaled_data[training_data_len - 60:, :]
    x_test = []
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model_lstm.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[training_data_len:]
    valid["Predictions"] = predictions
    st.line_chart(valid[["Close", "Predictions"]])
    # Evaluate model with RMSE
    from sklearn.metrics import mean_squared_error
    predictions = model_lstm.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    true_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    st.write(f"🔍 LSTM Forecast RMSE: {rmse:.2f}")

