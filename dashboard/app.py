import streamlit as st
import pandas as pd
import numpy as np

from scripts.utils.visualizations import *
from scripts.data_utils.loaders import load_csv
from scripts.data_utils.cleaning import clean_data, handle_missing_values

# Set Streamlit page layout
st.set_page_config(page_title='Sales Dashboard', layout='wide')

# Sidebar configuration
st.sidebar.title("Sales Dashboard")
st.sidebar.markdown("Explore different visualizations related to sales data, including trends around holidays, competitor openings, and more.")

train = load_csv("../resources/data/train.csv")
store = load_csv("../resources/data/store.csv")
train = clean_data(train)
store = clean_data(store)

missing_strategies = {"CompetitionDistance": "median", "StateHoliday": "None", "PromoInterval": "None"}
store = handle_missing_values(store, missing_strategies)

data = train.merge(store, on="Store", how='left')
st.info(data.info())

# # Upload file and data processing
# file_path = st.sidebar.file_uploader("Upload CSV", type=["csv"])
# if uploaded_file is not None:
#     data = pd.read_csv(uploaded_file)
#     data['Date'] = pd.to_datetime(data['Date'])
# else:
#     st.warning("Please upload a CSV file to view the visualizations.")

# Sections to group visualizations
st.header("Visualizations")
st.markdown("Select one of the sections below to explore different visualizations.")

# Define a sidebar dropdown for navigation
selection = st.sidebar.selectbox(
    "Choose a Section",
    [
        "Competitor Openings Analysis",
        "Sales and Holidays",
        "Store Performance",
        "Growth and Prediction Analysis",
        "Feature Importance"
    ]
)

# Competitor Openings Section
if selection == "Competitor Openings Analysis":
    st.subheader("Sales Trends Around Competitor Openings")
    st.markdown("Analyze the sales behavior before and after the opening of a competitor store.")
    plot_sales_around_competitor_openings(data)

# Sales and Holidays Section
elif selection == "Sales and Holidays":
    st.subheader("Sales Trend Around Holidays")
    st.markdown("Explore sales trends before, during, and after specific holidays.")
    
    # Holidays Selection
    country = st.selectbox("Select a country for holiday data:", ['US', 'DE', 'UK'], index=0)
    sales_trend_holidays(data, holiday_dates=None, country=country)
    
    st.subheader("Sales on Holidays vs Non-Holidays")
    plot_sales_holidays(data, country=country)

    st.subheader("Sales Before, During and After Holidays")
    sales_before_during_after_holidays(data, holiday_dates=None, country=country)

    st.subheader("Sales Per Specific Holidays")
    plot_sales_by_holiday(data, country=country)

# Store Performance Section
elif selection == "Store Performance":
    st.subheader("Store Type Performance")
    st.markdown("Analyze performance by store type with monthly average sales.")
    plot_store_type_performance(data)

    st.subheader("Sales vs Number of Customers")
    st.markdown("Plot a scatter plot between the number of customers and sales.")
    plot_sales_vs_customers(data)

    st.subheader("Cumulative Sales Analysis")
    st.markdown("Track cumulative sales over time.")
    plot_cumulative_sales(data)

# Growth and Prediction Analysis Section
elif selection == "Growth and Prediction Analysis":
    st.subheader("Sales Growth Rate")
    st.markdown("Analyze the daily sales growth rate over time.")
    plot_sales_growth_rate(data)

    st.subheader("Actual vs Predicted Sales")
    st.markdown("Plot the actual vs predicted sales for your model.")
    y_true = st.text_input('Enter actual sales data (comma-separated)', '')
    y_pred = st.text_input('Enter predicted sales data (comma-separated)', '')

    if y_true and y_pred:
        y_true = np.array([float(x) for x in y_true.split(',')])
        y_pred = np.array([float(x) for x in y_pred.split(',')])
        plot_actual_vs_predicted(y_true, y_pred, dataset_type="Sales Dataset")
    
    st.subheader("Training History")
    st.markdown("Visualize the training and validation loss of an LSTM model.")
    history = st.file_uploader("Upload Training History (CSV)", type=["csv"])

    if history is not None:
        history_data = pd.read_csv(history)
        plot_training_history(history_data)

# Feature Importance Section
elif selection == "Feature Importance":
    st.subheader("Correlation and Feature Importances")
    st.markdown("Analyze the correlation of features with sales and the importance of those features.")
    correlation_analysis(data)

    st.subheader("Feature Importance Visualization")
    importances = np.random.rand(data.shape[1])  # Replace this with actual feature importances
    plot_feature_importances(data, importances)

# Footer section
st.sidebar.markdown("### Footer")
st.sidebar.markdown("Developed by [Your Name].")
st.sidebar.markdown("Streamlit Dashboard | Sales Analysis")
