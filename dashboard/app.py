import os
import sys
import logging
import holidays
import numpy as np
import pandas as pd
import streamlit as st
from utils import *

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Set up logging
try:
    from scripts.utils.logger import setup_logger
except ImportError as e:
    logging.error(f"Import error: {e}. Please check the module path.")

# Configure logger for the dashboard
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
logger = setup_logger("dashboard", log_dir)
logger.info("Starting dashboard...")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-right: 1px solid #ddd;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    .stHeader {
        color: #007BFF;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("📊 Sales Analytics Dashboard")

# Load and preprocess data
@st.cache_data
def load_data():
    """Load and clean the sales data."""
    data = pd.read_csv("data/train_preprocessed.csv")
    return clean_data(data)

# Function to fetch holidays for a country and specific years
@st.cache_data
def get_country_holidays(country, years):
    """Fetch holidays for a specific country and years."""
    return holidays.CountryHoliday(country, years=years)

data = load_data()

# Cache plotting functions for better performance
@st.cache_data
def cached_plot_sales_holidays(data, country):
    return plot_sales_holidays(data, country=country)

@st.cache_data
def cached_plot_sales_by_holiday(data, country):
    return plot_sales_by_holiday(data, country=country)

@st.cache_data
def cached_plot_sales_trend_holidays(data, country, holiday_dates=None):
    return plot_sales_trend_holidays(data, country=country, holiday_dates=holiday_dates)

@st.cache_data
def cached_plot_sales_before_during_after_holidays(data, country, holiday_dates=None):
    return plot_sales_before_during_after_holidays(data, country=country, holiday_dates=holiday_dates)

@st.cache_data
def cached_plot_store_type_performance(data):
    return plot_store_type_performance(data)

@st.cache_data
def cached_plot_sales_vs_customers(data):
    return plot_sales_vs_customers(data)

@st.cache_data
def cached_plot_cumulative_sales(data):
    return plot_cumulative_sales(data)

@st.cache_data
def cached_plot_sales_vs_competitor_distance(data):
    return plot_sales_vs_competitor_distance(data)

@st.cache_data
def cached_plot_sales_around_competitor_openings(data):
    return plot_sales_around_competitor_openings(data)

@st.cache_data
def cached_plot_sales_growth_rate(data):
    return plot_sales_growth_rate(data)

@st.cache_data
def cached_plot_actual_vs_predicted(y_true, y_pred):
    return plot_actual_vs_predicted(y_true, y_pred, dataset_type="Sales Dataset")

@st.cache_data
def cached_plot_training_history(history_data):
    return plot_training_history(history_data)

@st.cache_data
def cached_correlation_analysis(data):
    return plot_correlation_analysis(data)

@st.cache_data
def cached_plot_feature_importances(data, importances):
    return plot_feature_importances(data, importances)

@st.cache_data
def cached_plot_sales_various_trends(data, freq='Month'):
    return plot_sales_various_trends(data, freq=freq)

@st.cache_data
def cached_plot_distribution_comparison(data, column, compare_column=None, kind='numerical'):
    return plot_distribution_comparison_dataset(data, column, compare_column, kind)

@st.cache_data
def cached_plot_sales_correlations(data, columns, trends=False, title=""):
    return plot_sales_correlations_analysis(data, columns, trends, title)

@st.cache_data
def cached_plot_sales_analysis(data, analysis_type, **kwargs):
    return plot_sales_analysis(data, analysis_type, **kwargs)

@st.cache_data
def cached_plot_event_effect(data, event_column, effect_type='promotion'):
    return plot_event_effect(data, event_column, effect_type)

@st.cache_data
def cached_plot_time_series_analysis(data, column, analysis_type, **kwargs):
    return plot_time_series_analysis(data, column, analysis_type, **kwargs)

@st.cache_data
def cached_plot_distribution(data, column, title=""):
    return plot_distribution(data, column, title)

@st.cache_data
def cached_plot_sales_trends(data, sales_column="Sales"):
    return plot_sales_trends(data, sales_column)

@st.cache_data
def cached_plot_sales_heatmap(data, freq='M'):
    return plot_sales_heatmap(data, freq)

@st.cache_data
def cached_plot_sales_month_week_heatmap(data):
    return plot_sales_month_week_heatmap(data)

@st.cache_data
def cached_plot_weekday_sales(data):
    return plot_weekday_sales(data)

@st.cache_data
def cached_plot_promotion_effect(data, freq='M'):
    return plot_promotion_effect(data, freq)

@st.cache_data
def cached_plot_holiday_effect(data):
    return plot_holiday_effect(data)

@st.cache_data
def cached_plot_sales_during_holidays(data, holiday_column, sales_column):
    return plot_sales_during_holidays(data, holiday_column, sales_column)

@st.cache_data
def cached_plot_seasonal_decomposition(data, column_name, freq='M'):
    return plot_seasonal_decomposition(data, column_name, freq=freq)

@st.cache_data
def cached_plot_time_series_diagnostics(data, column, diagnostics_type='ACF_PACF', **kwargs):
    return plot_time_series_diagnostics(data, column, diagnostics_type, **kwargs)

@st.cache_data
def cached_plot_ACF_PACF(data, column_name, lags=40):
    return plot_ACF_PACF(data, column_name, lags)

@st.cache_data
def cached_plot_sales_comparison(data, status_column, category_column, title='', xlabel='', ylabel=''):
    return plot_sales_comparison(data, status_column, category_column, title, xlabel, ylabel)

@st.cache_data
def cached_plot_weekly_sales_heatmap(data, date_column='Date', sales_column='Sales'):
    return plot_weekly_sales_heatmap(data, date_column, sales_column)

# Main content
st.title("📈 Sales Analytics Dashboard")

st.markdown("""
Welcome to the Sales Analytics Dashboard! Explore various visualizations related to sales data, including trends around holidays, competitor openings, and more.
""")

st.markdown("""
Select a section from the sidebar to explore different visualizations and insights from the sales data.
""")

# Sidebar navigation
selection = st.sidebar.selectbox(
    "Navigate to a Section",
    [
        "Sales Trends and Distributions",
        "Sales and Holidays",
        "Store Performance",
        "Promotion and Holiday Effects",
        "Competitor Openings Analysis",
        "Time Series Analysis",
        "Growth and Prediction Analysis",
        "Feature Importance"
    ]
)

# Sales Trends and Distributions Section
if selection == "Sales Trends and Distributions":
    st.header("📊 Sales Trends and Distributions")
    st.markdown("Explore various sales trends and distributions over time.")

    st.subheader("Sales Trends by Frequency")
    freq = st.selectbox("Select a frequency for sales trends:", ['Month', 'Week', 'Weekday', 'Year', 'Quarter', 'Day'])
    st.pyplot(cached_plot_sales_various_trends(data, freq=freq))

    st.subheader("Distribution Comparison")
    column = st.selectbox("Select a column for distribution comparison:", data.columns)
    compare_column = st.selectbox("Select a column to compare (optional):", [None] + list(data.columns))
    kind = st.selectbox("Select the kind of comparison:", ['numerical', 'categorical'])
    st.pyplot(cached_plot_distribution_comparison(data, column, compare_column, kind))

    st.subheader("Sales Correlation Analysis")
    columns = st.multiselect("Select columns for correlation analysis:", data.columns, default=data.columns[:5])
    trends = st.checkbox("Show trends over time")
    st.pyplot(cached_plot_sales_correlations(data, columns, trends))

# Sales and Holidays Section
elif selection == "Sales and Holidays":
    st.header("🎉 Sales and Holidays Analysis")
    st.markdown("Explore sales trends before, during, and after specific holidays.")
    
    country = st.selectbox("Select a country for holiday data:", ['US', 'DE', 'UK', 'ET'], index=0)    

    st.subheader("Sales on Holidays vs Non-Holidays")
    st.pyplot(cached_plot_sales_holidays(data, country))

    st.subheader("Sales Per Specific Holidays")
    st.pyplot(cached_plot_sales_by_holiday(data, country))
    
    # Select years for holiday analysis
    unique_years = sorted(data['Date'].dt.year.unique()) 
    selected_years = st.multiselect("Select years for holiday analysis:", options=unique_years, default=unique_years)
    country_holidays = get_country_holidays(country, selected_years)
    holiday_dict = {name: date.strftime('%Y-%m-%d') for date, name in country_holidays.items()}
    
    # Select specific holidays
    selected_holidays = st.multiselect("Select specific holidays to analyze:", options=list(holiday_dict.keys()),default=list(holiday_dict.keys()))
    holiday_dates = [holiday_dict[holiday] for holiday in selected_holidays]

    st.subheader("Sales trend On Holidays")
    st.pyplot(cached_plot_sales_trend_holidays(data, country=country, holiday_dates=holiday_dates))

    st.subheader("Sales Before, During, and After Holidays")
    st.pyplot(cached_plot_sales_before_during_after_holidays(data, country, holiday_dates=holiday_dates))

# Store Performance Section
elif selection == "Store Performance":
    st.header("🏪 Store Performance Analysis")
    st.markdown("Analyze performance by store type with monthly average sales.")
    st.pyplot(cached_plot_store_type_performance(data))

    st.subheader("Sales vs Number of Customers")
    st.markdown("Plot a scatter plot between the number of customers and sales.")
    st.pyplot(cached_plot_sales_vs_customers(data))

    st.subheader("Cumulative Sales Analysis")
    st.markdown("Track cumulative sales over time.")
    st.pyplot(cached_plot_cumulative_sales(data))

# Promotion and Holiday Effects Section
elif selection == "Promotion and Holiday Effects":
    st.header("🎉 Promotion and Holiday Effects")
    st.markdown("Analyze the effects of promotions and holidays on sales.")

    st.subheader("Promotion Effect on Sales")
    st.pyplot(cached_plot_promotion_effect(data))

    st.subheader("Holiday Effect on Sales")
    st.pyplot(cached_plot_holiday_effect(data))

    st.subheader("Sales During Holidays")
    holiday_column = st.selectbox("Select a holiday column:", data.columns)
    sales_column = st.selectbox("Select a sales column:", data.columns)
    st.pyplot(cached_plot_sales_during_holidays(data, holiday_column, sales_column))

# Competitor Openings Section
elif selection == "Competitor Openings Analysis":
    st.header("📊 Competitor Openings Analysis")
    st.markdown("Analyze the sales behavior before and after the opening of a competitor store.")

    # Sales vs Competitor Distance
    st.subheader("Sales vs. Competitor Distance")
    st.markdown("Explore the relationship between sales and the distance to the nearest competitor.")
    st.pyplot(cached_plot_sales_vs_competitor_distance(data))

    # Sales Trends Around Competitor Openings
    st.subheader("Sales Trends Around Competitor Openings")
    st.markdown("Analyze the sales behavior before and after the opening of a competitor store.")
    st.pyplot(cached_plot_sales_around_competitor_openings(data))

# Time Series Analysis Section
elif selection == "Time Series Analysis":
    st.header("⏳ Time Series Analysis")
    st.markdown("Perform advanced time series analysis on sales data.")

    st.subheader("Seasonal Decomposition")
    column_name = st.selectbox("Select a column for seasonal decomposition:", data.columns)
    freq = st.selectbox("Select a frequency for decomposition:", ['M', 'W', 'D'])
    st.pyplot(cached_plot_seasonal_decomposition(data, column_name, freq=freq))

    st.subheader("Time Series Diagnostics (ACF/PACF)")
    column = st.selectbox("Select a column for ACF/PACF analysis:", data.columns)
    lags = st.slider("Select the number of lags:", 10, 100, 40)
    st.pyplot(cached_plot_ACF_PACF(data, column, lags=lags))

# Growth and Prediction Analysis Section
elif selection == "Growth and Prediction Analysis":
    st.header("📈 Growth and Prediction Analysis")
    st.markdown("Analyze the daily sales growth rate and model predictions.")
    
    st.subheader("Sales Growth Rate")
    st.markdown("Analyze the daily sales growth rate over time.")
    st.pyplot(cached_plot_sales_growth_rate(data))

    st.subheader("Actual vs Predicted Sales")
    st.markdown("Plot the actual vs predicted sales for your model.")
    y_true = st.text_input('Enter actual sales data (comma-separated)', '')
    y_pred = st.text_input('Enter predicted sales data (comma-separated)', '')

    if y_true and y_pred:
        y_true = np.array([float(x) for x in y_true.split(',')])
        y_pred = np.array([float(x) for x in y_pred.split(',')])
        st.pyplot(cached_plot_actual_vs_predicted(y_true, y_pred))
    
    st.subheader("Training History")
    st.markdown("Visualize the training and validation loss of a model.")
    history = st.file_uploader("Upload Training History (CSV)", type=["csv"])

    if history is not None:
        history_data = pd.read_csv(history)
        st.pyplot(cached_plot_training_history(history_data))

# Feature Importance Section
elif selection == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    st.markdown("Analyze the correlation of features with sales and the importance of those features.")
    
    st.subheader("Correlation Analysis")
    st.pyplot(cached_correlation_analysis(data))

    st.subheader("Feature Importance Visualization")
    importances = np.random.rand(data.shape[1])  # Replace with actual feature importances
    st.pyplot(cached_plot_feature_importances(data, importances))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Developed by KM")
st.sidebar.markdown("Streamlit Dashboard | Sales Analysis")