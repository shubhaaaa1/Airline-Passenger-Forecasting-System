import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import pickle


#loading Model
model = pickle.load(open('prophet.pkl','rb'))


# Load data
data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data = data.rename(columns={'Month': 'ds', 'Passengers': 'y'})

# Streamlit UI
st.title('Airline Passengers Forecasting App')
st.write('This app forecasts monthly airline passengers using Facebook Prophet.')

# Input: Select forecast period
periods_input = st.slider('How many months would you like to forecast?', 1, 60)


# Future dataframe
future = model.make_future_dataframe(periods=periods_input, freq='M')
forecast = model.predict(future)

# Plot forecast
st.subheader('Forecasted Data')
fig = plot_plotly(model, forecast)
st.plotly_chart(fig)

# Show forecast components
st.subheader('Forecast Components')
components_fig = model.plot_components(forecast)
st.write(components_fig)
