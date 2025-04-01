#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Energy Markets Insight Dashboard (EMID)
# Streamlit app delivering AI-powered insights into gas, carbon, and electricity markets

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ----------------------------
# Load CSV Data (user upload)
# ----------------------------
st.set_page_config(page_title="Energy Markets Insight Dashboard (EMID)", layout="wide")
st.title("🌐 Energy Markets Insight Dashboard (EMID)")
st.markdown("Strategic insights into electricity, gas, and carbon price dynamics for hedging and trading decisions.")

uploaded_file = st.sidebar.file_uploader("📂 Upload Market Data CSV", type="csv")
required_columns = {'Date', 'Region', 'Electricity_Price', 'Gas_Price', 'Carbon_Price'}

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
            # Live Commodity Trend Detection
            # ----------------------------
            st.subheader(f"📈 Market Price Trends – {region}")
            fig = px.line(region_df, x='Date', y=['Electricity_Price', 'Gas_Price', 'Carbon_Price'],
                         labels={'value': 'Price (€/MWh or €/ton)', 'variable': 'Commodity'},
                         title='Live Commodity Market Trends')
            st.plotly_chart(fig, use_container_width=True)

            # ----------------------------
            # Historical Impact Analysis
            # ----------------------------
            st.subheader("🔍 Carbon & Gas Impact on Electricity Price")
            corr_matrix = region_df[['Electricity_Price', 'Gas_Price', 'Carbon_Price']].corr()
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', axis=None))

            # ----------------------------
            # AI Summary Generator (Simple Rule-Based)
            # ----------------------------
            st.subheader("🤖 AI Market Insight Summary")
            latest_gas = region_df['Gas_Price'].iloc[-1]
            baseline_gas = region_df['Gas_Price'].rolling(window=30).mean().iloc[-1]
            gas_change = (latest_gas - baseline_gas) / baseline_gas * 100

            latest_carbon = region_df['Carbon_Price'].iloc[-1]
            baseline_carbon = region_df['Carbon_Price'].rolling(window=30).mean().iloc[-1]
            carbon_change = (latest_carbon - baseline_carbon) / baseline_carbon * 100

            ai_summary = ""
            if gas_change > 10:
                ai_summary += f"🔺 Gas prices have risen by {gas_change:.1f}% – likely to increase electricity production costs.\n"
            elif gas_change < -10:
                ai_summary += f"🔻 Gas prices dropped by {abs(gas_change):.1f}% – potential easing on baseload prices.\n"

            if carbon_change > 10:
                ai_summary += f"🔺 Carbon price up by {carbon_change:.1f}% – emissions costs may pressure generation margins.\n"
            elif carbon_change < -10:
                ai_summary += f"🔻 Carbon price down {abs(carbon_change):.1f}% – may reduce regulatory cost impact.\n"

            if ai_summary == "":
                ai_summary = "Stable commodity prices observed – no major impacts forecasted this week."

            st.code(ai_summary, language='markdown')

            # ----------------------------
            # Signal Generator Panel
            # ----------------------------
            st.subheader("📊 Production & Trading Signals")
            if gas_change > 10 and carbon_change > 10:
                st.warning("⚠️ ALERT: Combined rise in gas and carbon prices may raise electricity costs – consider adjusting hedges.")
            elif gas_change < -10 and carbon_change < -10:
                st.success("✅ Market relief: Lower gas and carbon may offer favorable procurement windows.")
            else:
                st.info("ℹ️ Mixed market signals – monitor closely for volatility triggers.")

            # ----------------------------
            # Exportable Briefing
            # ----------------------------
            st.download_button(
                label="📥 Download Market Snapshot (CSV)",
                data=region_df.to_csv(index=False).encode('utf-8'),
                file_name=f'{region}_market_snapshot.csv',
                mime='text/csv'
            )

            st.caption("Built for energy market analysts, traders, and strategy teams.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
else:
    st.warning("Please upload a valid CSV file to begin.")

