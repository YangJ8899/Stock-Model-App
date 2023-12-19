import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# set start date and end date of information retreival 
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# title of the app
st.title("Stock Model App")

# input stock symbol to query
stocks = st.text_input('Enter Stock Symbol', "GOOG")

# Range of prediction years
slider_years = st.slider("Years of Prediction:", 1, 4)
period = slider_years * 365

# Function to load in the data
@st.cache_data # Caches the data
def load_data(stock):
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load Data...")
data = load_data(stocks)
data_load_state.text("Loading Data...Done!")

st.subheader("Data")
st.write(data.tail())

# Plot the data
def plot_data():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open Price'))
    figure.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    figure.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

plot_data()

# Forcasting the data
training_data = data[['Date', 'Close']]
training_data = training_data.rename(columns={"Date": "ds", "Close":"y"})

model = Prophet()
model.fit(training_data)

future_data = model.make_future_dataframe(periods=period)

forecast = model.predict(future_data)

st.subheader("Future Trends Prediction")
st.write(forecast.tail())

st.subheader("Future Trends Data Graph")
figure_future = plot_plotly(model, forecast)
st.plotly_chart(figure_future)

st.subheader("Future Trends Data Components")

figure_future1 = model.plot_components(forecast)
st.write(figure_future1)