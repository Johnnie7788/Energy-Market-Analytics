#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# CWE Power Price Forecaster Pro
# Updated Streamlit app to load external real-world CSV file only, with no internal sample data

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Forecasting Model Builder
# ----------------------------
def build_forecasting_model(df, region):
    region_df = df[df['Region'] == region].copy()
    region_df['Date'] = pd.to_datetime(region_df['Date'])
    region_df['DayOfYear'] = region_df['Date'].dt.dayofyear

    X = region_df[['Demand', 'Solar', 'Wind', 'Gas_Price', 'DayOfYear']]
    y = region_df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    return model, error

# ----------------------------
# Forecast Function with Scenario
# ----------------------------
def forecast_prices(model, df, region, gas_scenario):
    future_days = pd.date_range(start='2025-01-01', periods=180, freq='D')
    base_df = df[df['Region'] == region].copy()
    latest_demand = base_df['Demand'].iloc[-30:].mean()
    latest_solar = base_df['Solar'].iloc[-30:].mean()
    latest_wind = base_df['Wind'].iloc[-30:].mean()

    scenario_data = []
    for date in future_days:
        day = date.dayofyear
        gas_price = base_df['Gas_Price'].iloc[-1] * (1 + gas_scenario / 100)
        scenario_data.append([latest_demand, latest_solar, latest_wind, gas_price, day])

    scenario_df = pd.DataFrame(scenario_data, columns=['Demand', 'Solar', 'Wind', 'Gas_Price', 'DayOfYear'])
    forecasted_prices = model.predict(scenario_df)
    result = pd.DataFrame({'Date': future_days, 'Forecasted_Price': forecasted_prices})
    return result

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="CWE Power Price Forecaster Pro", layout="wide")
st.title("🔋 CWE Power Price Forecaster Pro")
st.markdown("AI-driven electricity price forecasting with gas price scenario simulation.")

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV File", type="csv")
required_columns = {'Date', 'Region', 'Demand', 'Solar', 'Wind', 'Gas_Price', 'Price'}

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not required_columns.issubset(set(df.columns)):
            st.error(f"Uploaded CSV is missing one or more required columns: {required_columns}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])

            region = st.sidebar.selectbox("Select Region", df['Region'].unique())
            gas_scenario = st.sidebar.slider("Gas Price Scenario (% Change)", -50, 100, 0)

            model, mae = build_forecasting_model(df, region)
            st.sidebar.markdown(f"**Model MAE:** {mae:.2f} €/MWh")

            forecast_df = forecast_prices(model, df, region, gas_scenario)

            # Horizon selection
            horizon = st.radio("Select Forecast Horizon", ["Short-term (0–14 days)", "Mid-term (15–90 days)", "Long-term (91–180 days)"])
            if horizon == "Short-term (0–14 days)":
                plot_df = forecast_df.iloc[:14]
            elif horizon == "Mid-term (15–90 days)":
                plot_df = forecast_df.iloc[14:90]
            else:
                plot_df = forecast_df.iloc[90:]

            st.subheader(f"Price Forecast for {region} ({horizon}, Gas Scenario: {gas_scenario:+}%)")
            fig = px.line(plot_df, x='Date', y='Forecasted_Price', title='Forecasted Electricity Prices', labels={'Forecasted_Price': '€/MWh'})
            st.plotly_chart(fig, use_container_width=True)

            st.download_button(
                label="📥 Download Forecast Data",
                data=forecast_df.to_csv(index=False).encode('utf-8'),
                file_name=f'{region}_forecast_scenario_{gas_scenario}pct.csv',
                mime='text/csv'
            )

            # Historical trends and volatility
            st.subheader(f"📊 Historical Market Trends – {region}")
            hist_df = df[df['Region'] == region].copy()
            hist_fig = px.line(hist_df, x='Date', y='Price', title='Historical Electricity Prices', labels={'Price': '€/MWh'})
            st.plotly_chart(hist_fig, use_container_width=True)

            st.subheader(f"📈 Historical Price Volatility – {region}")
            hist_df['Rolling_Std'] = hist_df['Price'].rolling(window=14).std()
            vol_fig = px.line(hist_df, x='Date', y='Rolling_Std', title='14-Day Rolling Volatility', labels={'Rolling_Std': 'Volatility (€/MWh)'})
            st.plotly_chart(vol_fig, use_container_width=True)

            st.caption("Powered by machine learning & CWE market intelligence")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a CSV file with the required columns to begin.")

