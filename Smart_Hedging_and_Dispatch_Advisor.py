#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Smart Hedging & Dispatch Advisor (SHDA)
# Streamlit app for optimal dispatch and hedging strategy support

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Config and Title
# ----------------------------
st.set_page_config(page_title="Smart Hedging & Dispatch Advisor (SHDA)", layout="wide")
st.title("⚙️ Smart Hedging & Dispatch Advisor (SHDA)")
st.markdown("AI-backed operational intelligence for production dispatch and energy market hedging decisions.")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload Dispatch & Market Data CSV", type="csv")
required_columns = {'Date', 'Region', 'Electricity_Price', 'Gas_Price', 'Carbon_Price', 'Forecasted_Demand', 'Wind_Forecast', 'Solar_Forecast'}

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if not required_columns.issubset(df.columns):
            st.error(f"Uploaded file must include: {required_columns}")
        else:
            df['Date'] = pd.to_datetime(df['Date'])
            region = st.sidebar.selectbox("Select Region", df['Region'].unique())
            region_df = df[df['Region'] == region].copy()

            # ----------------------------
            # Constants (user adjustable)
            # ----------------------------
            st.sidebar.markdown("### ⚙️ Dispatch Settings")
            gas_efficiency = st.sidebar.slider("Gas Plant Efficiency (%)", 40, 60, 52)
            co2_factor = st.sidebar.number_input("CO₂ Emission Factor (ton/MWh)", value=0.202, step=0.001)
            margin_threshold = st.sidebar.slider("Minimum Operational Margin (€/MWh)", 0, 20, 5)

            # ----------------------------
            # Calculate Generation Cost and Margin
            # ----------------------------
            region_df['Gas_Cost'] = region_df['Gas_Price'] / (gas_efficiency / 100)
            region_df['Carbon_Cost'] = region_df['Carbon_Price'] * co2_factor
            region_df['Total_Generation_Cost'] = region_df['Gas_Cost'] + region_df['Carbon_Cost']
            region_df['Net_Margin'] = region_df['Electricity_Price'] - region_df['Total_Generation_Cost']
            region_df['Dispatch_Recommended'] = region_df['Net_Margin'] > margin_threshold

            # ----------------------------
            # Plot Margin & Dispatch Trend
            # ----------------------------
            st.subheader(f"📊 Dispatch Decision Support – {region}")
            fig_margin = px.line(region_df, x='Date', y='Net_Margin', title='Net Margin Over Time', labels={'Net_Margin': '€/MWh'})
            st.plotly_chart(fig_margin, use_container_width=True)

            fig_dispatch = px.scatter(region_df, x='Date', y='Electricity_Price', color='Dispatch_Recommended',
                                      title='Electricity Price with Dispatch Signals',
                                      labels={'Dispatch_Recommended': 'Dispatch?', 'Electricity_Price': '€/MWh'})
            st.plotly_chart(fig_dispatch, use_container_width=True)

            # ----------------------------
            # Hedging Strategy Panel
            # ----------------------------
            st.subheader("📈 Hedging Opportunity Index")
            region_df['Rolling_Avg'] = region_df['Electricity_Price'].rolling(window=30).mean()
            region_df['Volatility'] = region_df['Electricity_Price'].rolling(window=14).std()
            latest_price = region_df['Electricity_Price'].iloc[-1]
            rolling_avg = region_df['Rolling_Avg'].iloc[-1]
            volatility = region_df['Volatility'].iloc[-1]

            if latest_price > rolling_avg and volatility > 5:
                st.warning("⚠️ Market is trending upward and volatile – consider hedging now.")
            elif latest_price < rolling_avg and volatility < 3:
                st.success("✅ Prices are low and stable – optional hedge opportunity.")
            else:
                st.info("ℹ️ Mixed signals – monitor closely.")

            # ----------------------------
            # Exportable Recommendation Card
            # ----------------------------
            st.subheader("📥 Export Recommendation Card")
            export_df = region_df[['Date', 'Electricity_Price', 'Total_Generation_Cost', 'Net_Margin', 'Dispatch_Recommended']].copy()
            st.download_button(
                label="📄 Download Dispatch & Hedging Summary",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name=f'{region}_dispatch_hedging_summary.csv',
                mime='text/csv'
            )

            st.caption("Built for production planners, trading desks, and market strategists.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a valid CSV file to begin.")

