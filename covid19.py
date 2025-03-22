
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Title of the app
st.title('COVID-19 Time Series Analysis')

# Load the COVID-19 dataset
@st.cache_data
def load_data():
    data = pd.read_csv('owid-covid-data.csv')
    # Convert date column to datetime
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Select country
countries = data['location'].unique()
selected_country = st.sidebar.selectbox("Select Country", countries)

# Filter data for the selected country
country_data = data[data['location'] == selected_country]

# Display the data
st.subheader(f'COVID-19 Data for {selected_country}')
st.write(country_data)

# Plotting the data
st.subheader('Time Series Plot')

# Using Matplotlib
st.write("### Daily New Cases (Matplotlib)")
plt.figure(figsize=(10, 4))
plt.plot(country_data['date'], country_data['new_cases'], label='New Cases')
plt.title(f'Daily New COVID-19 Cases in {selected_country}')
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.legend()
st.pyplot(plt)

# Using Plotly for interactive plot
st.write("### Daily New Cases (Plotly)")
fig = px.line(country_data, x='date', y='new_cases', title=f'Daily New COVID-19 Cases in {selected_country}')
st.plotly_chart(fig)

# Rolling Average of New Cases
st.write("### Rolling Average of New Cases")
rolling_window = st.slider("Select Rolling Window Size (Days)", 1, 30, 7)  # Default window of 7 days
country_data['Rolling Average'] = country_data['new_cases'].rolling(window=rolling_window).mean()
fig_rolling = px.line(country_data, x='date', y='Rolling Average', title=f'{rolling_window}-Day Rolling Average of New Cases in {selected_country}')
st.plotly_chart(fig_rolling)

# Daily Deaths
st.write("### Daily Deaths")
fig_deaths = px.line(country_data, x='date', y='new_deaths', title=f'Daily COVID-19 Deaths in {selected_country}')
st.plotly_chart(fig_deaths)

# Summary Statistics
st.subheader('Summary Statistics')
st.write(country_data['new_cases'].describe())

# Correlation Analysis
st.subheader('Correlation Analysis')
st.write(country_data[['new_cases', 'new_deaths']].corr())

# Conclusion
st.write("### Conclusion")
st.write(f"This app provides a basic time series analysis of COVID-19 data for {selected_country}. You can explore daily new cases, rolling averages, and daily deaths by selecting a country and adjusting the rolling window size.")