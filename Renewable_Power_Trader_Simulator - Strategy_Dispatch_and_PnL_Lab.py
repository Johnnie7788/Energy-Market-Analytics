#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
from fpdf import FPDF
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Renewable Power Trader Simulator – Strategy, Dispatch & PnL Lab")

# File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

def load_data_from_upload(upload):
    df = pd.read_csv(upload)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

if uploaded_file is not None:
    df = load_data_from_upload(uploaded_file)
else:
    st.warning("Please upload a dataset (.csv) to start the simulation.")
    st.stop()

# Sidebar Controls
st.sidebar.header("Trading Strategy Parameters")
strategy = st.sidebar.selectbox("Select Trading Strategy", ["Threshold-Based", "High Volatility Buy", "Demand-Driven", "ML-Based Predictor"])
buy_threshold = st.sidebar.slider("Buy if Forecasted < Actual by (%)", 0.0, 10.0, 2.0, 0.1)
sell_threshold = st.sidebar.slider("Sell if Forecasted > Actual by (%)", 0.0, 10.0, 2.0, 0.1)
gas_cost_efficiency = st.sidebar.slider("Gas Efficiency Factor (€/MWh)", 0.5, 2.0, 1.2, 0.1)

# Simulate Trades
def simulate_trading(data, strategy, buy_thresh, sell_thresh, gas_eff):
    trades = []
    pnl = []
    dispatch_margin = []
    wins = 0
    losses = 0

    # ML-Based Strategy Setup
    if strategy == "ML-Based Predictor":
        df_model = data.copy()
        df_model["Price_Diff"] = df_model["Actual_Price_EUR_MWh"] - df_model["Forecasted_Price_EUR_MWh"]
        df_model["Target"] = df_model["Price_Diff"].apply(lambda x: "Buy" if x > 1 else ("Sell" if x < -1 else "Hold"))
        features = ["Forecasted_Price_EUR_MWh", "Forecast_Demand_MW", "Gas_Price_EUR_MWh", "Wind_Forecast_MW", "Solar_Forecast_MW"]
        X = df_model[features]
        y = df_model["Target"]
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X)
        predicted_actions = le.inverse_transform(y_pred)

    for i, row in data.iterrows():
        price_diff_pct = ((row["Actual_Price_EUR_MWh"] - row["Forecasted_Price_EUR_MWh"]) / row["Forecasted_Price_EUR_MWh"]) * 100
        gas_cost = row["Gas_Price_EUR_MWh"] * gas_eff
        margin = row["Actual_Price_EUR_MWh"] - gas_cost
        dispatch_margin.append(round(margin, 2))

        action = "Hold"
        profit = 0.0

        if strategy == "Threshold-Based":
            if price_diff_pct > buy_thresh:
                action = "Buy"
                profit = row["Actual_Price_EUR_MWh"] - row["Forecasted_Price_EUR_MWh"]
            elif price_diff_pct < -sell_thresh:
                action = "Sell"
                profit = row["Forecasted_Price_EUR_MWh"] - row["Actual_Price_EUR_MWh"]

        elif strategy == "High Volatility Buy":
            if abs(price_diff_pct) > 3 and row["Actual_Price_EUR_MWh"] > row["Forecasted_Price_EUR_MWh"]:
                action = "Buy"
                profit = row["Actual_Price_EUR_MWh"] - row["Forecasted_Price_EUR_MWh"]

        elif strategy == "Demand-Driven":
            if row["Forecast_Demand_MW"] > 42000 and row["Forecasted_Price_EUR_MWh"] < row["Actual_Price_EUR_MWh"]:
                action = "Buy"
                profit = row["Actual_Price_EUR_MWh"] - row["Forecasted_Price_EUR_MWh"]

        elif strategy == "ML-Based Predictor":
            action = predicted_actions[i]
            if action == "Buy":
                profit = row["Actual_Price_EUR_MWh"] - row["Forecasted_Price_EUR_MWh"]
            elif action == "Sell":
                profit = row["Forecasted_Price_EUR_MWh"] - row["Actual_Price_EUR_MWh"]

        trades.append(action)
        pnl.append(round(profit, 2))
        if profit > 0:
            wins += 1
        elif profit < 0:
            losses += 1

    data["Trade_Action"] = trades
    data["PnL_EUR"] = pnl
    data["Cumulative_PnL_EUR"] = data["PnL_EUR"].cumsum()
    data["Dispatch_Margin_EUR"] = dispatch_margin

    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    max_drawdown = data["Cumulative_PnL_EUR"].cummax() - data["Cumulative_PnL_EUR"]
    sharpe_ratio = np.mean(data["PnL_EUR"]) / np.std(data["PnL_EUR"]) if np.std(data["PnL_EUR"]) > 0 else 0

    performance = {
        "Selected Strategy": strategy,
        "Total Trades": total_trades,
        "Winning Trades": wins,
        "Losing Trades": losses,
        "Win Rate (%)": round(win_rate, 2),
        "Max Drawdown (EUR)": round(max_drawdown.max(), 2),
        "Sharpe Ratio": round(sharpe_ratio, 2)
    }

    return data, performance

# Simulate and Display
result_df, metrics = simulate_trading(df.copy(), strategy, buy_threshold, sell_threshold, gas_cost_efficiency)

st.subheader("Simulated Trading Outcomes")
st.dataframe(result_df[["Date", "Forecasted_Price_EUR_MWh", "Actual_Price_EUR_MWh", "Trade_Action", "PnL_EUR", "Cumulative_PnL_EUR", "Dispatch_Margin_EUR"]])

# Charts
st.subheader("Cumulative Profit & Loss")
st.line_chart(result_df.set_index("Date")["Cumulative_PnL_EUR"])

st.subheader("Daily Trading Actions")
st.bar_chart(result_df["PnL_EUR"])

st.subheader("Dispatch Margin vs. Actual Price")
st.line_chart(result_df.set_index("Date")[["Actual_Price_EUR_MWh", "Dispatch_Margin_EUR"]])

# Performance Metrics
st.subheader("Performance Metrics")
st.json(metrics)

# Export CSV
st.download_button("Download Trade Log (CSV)", result_df.to_csv(index=False), file_name="simulated_trades.csv")

# PDF Report Generator
def generate_pdf(metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Trading Strategy Performance Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    for key, value in metrics.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True)
    return pdf

if st.button("Download Strategy Report (PDF)"):
    pdf = generate_pdf(metrics)
    buffer = BytesIO()
    pdf.output(buffer)
    st.download_button(label="Download PDF", data=buffer.getvalue(), file_name="strategy_report.pdf", mime="application/pdf")

