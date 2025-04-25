import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

#Custom CSS for background color and styling 
st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    }
    /* Style header text */
    h1 {
        color: #ffffff;
        text-align: center;
    }
    /* Style subheader text */
    h2, h3 {
        color: #ffffff;
    }
    /* Optional: style expander content */
    .streamlit-expanderHeader {
        font-weight: bold;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

header_image_url = "https://images.unsplash.com/photo-1518837695005-2083093ee35b?auto=format&fit=crop&w=1350&q=80"
st.image(header_image_url,  use_container_width =True)
st.title("🌍 US Air Pollution Analysis Dashboard")
st.markdown("""
Welcome to an interactive dashboard for exploring and forecasting air pollutant levels across US cities.
""")


@st.cache_data
def load_data():
    # Update the file path to your dataset file.
    df = pd.read_csv('pollution_us_2000_2016.csv', parse_dates=['Date Local'])
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🎛️ Filter Options")
state = st.sidebar.selectbox("Select State", sorted(df['State'].unique()))
city = st.sidebar.selectbox("Select City", sorted(df[df['State'] == state]['City'].unique()))
pollutant = st.sidebar.selectbox("Select Pollutant", ['NO2 Mean', 'SO2 Mean', 'CO Mean', 'O3 Mean'])

# --- Filtered Data ---
df_filtered = df[(df['State'] == state) & (df['City'] == city)][['Date Local', pollutant]].dropna()
df_filtered = df_filtered.rename(columns={pollutant: 'Concentration'})

# --- Show Raw Data Preview ---
with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered.head(50))

# --- Summary Statistics ---
with st.expander("📊 Summary Statistics"):
    st.write(df_filtered.describe())

# --- Line Chart with Optional Rolling Average ---
st.subheader(f"📈 {pollutant} Levels in {city}, {state}")
use_rolling = st.checkbox("Apply 30-day Rolling Average", value=True)

if use_rolling:
    df_filtered['Smoothed'] = df_filtered['Concentration'].rolling(window=30).mean()
    fig = px.line(df_filtered, x='Date Local', y='Smoothed', title=f"30-Day Smoothed {pollutant} in {city}")
else:
    fig = px.line(df_filtered, x='Date Local', y='Concentration', title=f"{pollutant} in {city}")
st.plotly_chart(fig, use_container_width=True)

# --- Forecasting Section ---
st.subheader("📅 Forecasting for Next Year")
df_forecast = df_filtered.rename(columns={'Date Local': 'ds', 'Concentration': 'y'})
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("Dataset: US EPA via Kaggle")
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# --- Forecasting Section with Model Selection ---
st.subheader("🗕️ Forecasting for Next Year")

model_type = st.radio("Select Forecasting Method", ['Prophet', 'Linear Regression'])
show_compare = st.checkbox("🔁 Compare Prophet & Linear Regression on same plot", value=False)

# Prepare data
df_forecast_base = df_filtered.rename(columns={'Date Local': 'ds', 'Concentration': 'y'})

# Prophet Forecast
prophet_model = Prophet()
prophet_model.fit(df_forecast_base)
future_prophet = prophet_model.make_future_dataframe(periods=365)
forecast_prophet = prophet_model.predict(future_prophet)

# Linear Regression Forecast
df_lr = df_forecast_base.copy()
df_lr['Day'] = (df_lr['ds'] - df_lr['ds'].min()).dt.days
X = df_lr[['Day']]
y = df_lr['y']
lr_model = LinearRegression()
lr_model.fit(X, y)

last_day = df_lr['Day'].max()
future_days = np.arange(last_day + 1, last_day + 366).reshape(-1, 1)
predictions_lr = lr_model.predict(future_days)
future_dates_lr = pd.date_range(start=df_lr['ds'].max() + pd.Timedelta(days=1), periods=365)

df_future_lr = pd.DataFrame({
    'ds': future_dates_lr,
    'yhat': predictions_lr
})

# Combine historical and prediction
df_combined_lr = pd.concat([
    df_lr[['ds', 'y']].rename(columns={'y': 'Observed'}),
    df_future_lr.rename(columns={'yhat': 'Observed'})
])

# Metrics (only on training data)
y_pred_train = lr_model.predict(X)
mse = mean_squared_error(y, y_pred_train)
r2 = r2_score(y, y_pred_train)

# Forecast Display
if show_compare:
    st.subheader("🔁 Combined Forecast (Prophet vs Regression)")
    merged = forecast_prophet[['ds', 'yhat']].copy()
    merged['Model'] = 'Prophet'
    lr_copy = df_future_lr.copy()
    lr_copy['Model'] = 'Linear Regression'
    lr_copy.rename(columns={'yhat': 'yhat'}, inplace=True)
    combined_forecast = pd.concat([merged[['ds', 'yhat', 'Model']], lr_copy[['ds', 'yhat', 'Model']]])
    fig_compare = px.line(combined_forecast, x='ds', y='yhat', color='Model',
                          title=f"Forecast Comparison for {pollutant}")
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    if model_type == 'Prophet':
        st.subheader("📈 Prophet Forecast")
        fig_prophet = plot_plotly(prophet_model, forecast_prophet)
        st.plotly_chart(fig_prophet, use_container_width=True)

    elif model_type == 'Linear Regression':
        st.subheader("📉 Linear Regression Forecast")
        fig_lr = px.line(df_combined_lr, x='ds', y='Observed',
                         title=f"Linear Regression Forecast - {pollutant}")
        st.plotly_chart(fig_lr, use_container_width=True)

        st.markdown("### 📊 Model Performance on Training Data")
        st.write(f"**R² Score:** {r2:.4f}")

# Download button
st.markdown("### ⬇️ Download Forecast Data")
if model_type == 'Prophet':
    forecast_out = forecast_prophet[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
else:
    forecast_out = df_future_lr.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})

st.download_button(
    label="Download Forecast as CSV",
    data=forecast_out.to_csv(index=False).encode('utf-8'),
    file_name=f'{model_type}_forecast_{pollutant}.csv',
    mime='text/csv'
)



