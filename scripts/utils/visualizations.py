import holidays
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scripts.data_utils.feature_engineering import generate_holiday_data
from scripts.utils.logger import setup_logger

# Setup logger for visualizations.py
logger = setup_logger("Visualizations")

# Set a unified style for all plots
sns.set_theme(style="whitegrid")
def plot_distribution_comparison_dataset(data, column, compare_column=None, kind='numerical', bins=30, figsize=(15, 7)):
    """
    Plot a single column's distribution or compare distributions based on another column.
    """
    logger.info("Plotting distribution comparison dataset...")
    fig, ax = plt.subplots(figsize=figsize)
    
    if compare_column:
        if kind == 'numerical':
            sns.histplot(data, x=column, hue=compare_column, kde=True, color='blue', bins=bins, ax=ax)
        elif kind == 'categorical':
            sns.countplot(data=data, x=column, hue=compare_column, color='blue', ax=ax)
    else:
        sns.histplot(data[column], kde=True, color='blue', bins=bins, ax=ax)

    ax.set_title(f"{'Comparison of ' + kind if compare_column else 'Distribution of'} '{column}'")
    ax.set_ylabel(column)
    plt.tight_layout()
    return fig

def plot_distribution_comparision_datasets(train, test, column, kind='numerical', title=""):
    """
    Compare distributions of a column in the train and test datasets.
    """
    logger.info("Plotting distribution comparison datasets...")
    fig, ax = plt.subplots(figsize=(15, 7))
    if kind == 'categorical':
        sns.countplot(x=column, data=train, label="Train", color='blue', alpha=0.5, ax=ax)
        sns.countplot(x=column, data=test, label="Test", color='red', alpha=0.5, ax=ax)
    else:
        sns.kdeplot(train[column], label="Train", fill=True, color="blue", ax=ax)
        sns.kdeplot(test[column], label="Test", fill=True, color="red", ax=ax)

    ax.set_title(title or f"Comparison of {column} in Train and Test Sets")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_sales_correlations(data, columns, trends=False, title=""):
    """
    Analyze correlations or plot trends based on the provided columns.
    """
    logger.info("Analyzing sales correlations...")
    fig, ax = plt.subplots(figsize=(15, 7))
    if trends:
        for col in columns[1:]:
            sns.lineplot(data=data, x=columns[0], y=col, label=col, ax=ax)
        ax.set_title("Trends Over Time")
        ax.legend()
        return fig
    else:
        correlation = data[columns].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title(title if title else "Correlation Matrix")
        return fig

def plot_sales_analysis(data, analysis_type, **kwargs):
    """
    Perform different sales analysis tasks based on the given type.
    """
    logger.info("Analyzing sales...")
    data = data.copy()
    if analysis_type == 'by_store_type':
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=data, x='StoreType', y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales by Store Type')
    elif analysis_type == 'sales_vs_customers':
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.scatterplot(data=data, x='Customers', y='Sales', hue='StoreType', ax=ax)
        ax.set_title('Sales vs. Customers')
    elif analysis_type == 'heatmap':
        freq = kwargs.get('freq', 'M')
        data['Month'] = data['Date'].dt.to_period(freq).dt.to_timestamp()
        pivot = data.groupby('Month')['Sales'].mean().reset_index()
        pivot['Month'] = pivot['Month'].dt.strftime('%Y-%m')
        pivot = pivot.pivot_table(index='Month', values='Sales')
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt='.0f', ax=ax)
        ax.set_title('Sales Heatmap')
    elif analysis_type == 'cumulative':
        cumulative_sales = data['Sales'].cumsum()
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.plot(data['Date'], cumulative_sales)
        ax.set_title('Cumulative Sales Over Time')
    elif analysis_type == 'weekday':
        data['Weekday'] = data['Date'].dt.day_name()
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.barplot(data=data, x='Weekday', y='Sales', order=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
        ax.set_title('Sales by Weekday')
    elif analysis_type == 'holiday_effect':
        data = generate_holiday_data(data)
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=data, x='IsHoliday', y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales During Holidays' if kwargs.get('detailed', False) else 'Sales and Holidays')
    else:
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.boxplot(data=data, y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales')
    
    ax.set_ylabel('Sales')
    plt.tight_layout()
    return fig

def plot_event_effect(data, event_column, effect_type='promotion'):
    """
    Analyze and visualize the effects of events (e.g., promotions, holidays) on sales.
    """
    logger.info("Plotting event effect...")
    if effect_type == 'holiday':
        data.copy()
        data = generate_holiday_data(data)
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(data=data, x=event_column, y='Sales', palette="Set2", ax=ax)
    ax.set_title(f"Effect of {effect_type.capitalize()} on Sales")
    return fig

def plot_time_series_analysis(data, column, analysis_type, **kwargs):
    """
    Perform rolling statistics or decomposition on a time series column.
    """
    logger.info("Performing time series analysis...")
    data[column] = pd.to_numeric(data[column], errors='coerce')


    if analysis_type == 'rolling':
        window = kwargs.get('window', 12)
        rolling_mean = data[column].rolling(window=window).mean()
        rolling_std = data[column].rolling(window=window).std()

        fig, ax = plt.subplots(figsize=(15, 7))
        plt.plot(data[column], color='blue', label='Original')
        plt.plot(rolling_mean, color='red', label='Rolling Mean')
        plt.plot(rolling_std, color='black', label='Rolling Std')
        ax.set_title(f"Rolling Mean & Standard Deviation (Window={window})")
        ax.legend()
        return fig
    elif analysis_type == 'decomposition':
        model = kwargs.get('model', 'additive')
        result = seasonal_decompose(data[column], model=model, period=kwargs.get('period', 12))
        fig, ax = plt.subplots(figsize=(25, 15))
        result.plot()
        return fig
    else:
        raise ValueError("Invalid analysis type.")

def plot_distribution(data, column, title=""):
    """
    Plot the distribution of a single column using seaborn.
    """
    logger.info("Plotting distribution...")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.histplot(data[column], kde=True, color="blue", ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_sales_trends(data, sales_column="Sales"):
    """
    Plot sales trends over time.
    Assumes the dataset has a 'Date' column in datetime format.
    """
    logger.info("Plotting sales trends...")
    data = data.copy()
    data = data.dropna().sort_values(by="Date")
    fig, ax = plt.subplots(figsize=(15, 7))
    plt.plot(data["Date"], data[sales_column], label="Sales", color="blue")
    ax.set_title("Sales Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(sales_column)
    plt.grid(True)
    plt.tight_layout()
    return fig

def plot_sales_heatmap(data, freq='M', figsize=(15, 10)):
    logger.info("Plotting sales heatmap...")
    data = data.copy()
    data['Month'] = data['Date'].dt.to_period(freq)
    # monthly_data = data.pivot_table(index='Month', values='Sales', aggfunc='sum')
    monthly_data = data.groupby('Month')['Sales'].sum().values.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(monthly_data, annot=True, fmt='.0f', cmap='coolwarm', ax=ax)
    ax.set_title('Monthly Sales Heatmap')
    ax.set_ylabel('Month')
    ax.set_xlabel('Sales')
    return fig

def plot_sales_month_week_heatmap(data):
    logger.info("Plotting sales heatmap by day of week and month...")
    data = data.copy()
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Month'] = data['Date'].dt.month
    sales_heatmap = data.pivot_table(values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sales_heatmap, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax)
    ax.set_title('Average Sales by Day of Week and Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Day of Week (0=Monday, 6=Sunday)')
    return fig
    
def plot_sales(data, column_name='Sales', title='Sales Trend Over Time', xlabel='Date', ylabel='Sales', freq='w', figsize=(15, 7)):
    """
    Plots the sales data over time.

    Args:
    - data (pd.DataFrame): DataFrame containing the sales data.
    - column_name (str): Name of the column containing the sales data.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - figsize (tuple): Size of the figure.
    """
    logger.info("Plotting sales data...")

    aggregated_sales = data[column_name].resample(freq).sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(aggregated_sales.index, aggregated_sales)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_weekday_sales(data):
    logger.info("Plotting weekday sales...")
    data = data.copy()
    data['Weekday'] = data['Date'].dt.day_name()
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x='Weekday', y='Sales', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
    ax.set_title('Weekday Sales Distribution')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Sales')
    return fig

def plot_promotion_effect(data, freq='M', figsize=(15, 7)):
    logger.info("Plotting promotion effect...")
    data = data.copy()
    data['Month'] = data['Date'].dt.to_period(freq)
    promo_effect = data.groupby('Promo')['Sales'].mean()
    fig, ax = plt.subplots(figsize=figsize)
    promo_effect.plot(kind='bar', color='skyblue', rot=0, ax=ax)
    ax.set_title('Average Sales by Promotion')
    ax.set_xlabel('Promotion Status')
    ax.set_ylabel('Average Sales')
    plt.tight_layout()
    return fig

def plot_promotion_effect_2(data, freq='M', figsize=(15, 7)):
    logger.info("Plotting promotion effect 2...")
    data = data.copy()
    sales = data.groupby([data['Date'].dt.to_period(freq), 'Promo'])['Sales'].mean().unstack()
    sales.columns = ['No Promo', 'Promo']
    
    fig, ax = plt.subplots(figsize=figsize)
    sales[['No Promo', 'Promo']].plot(ax=ax)
    ax.set_title('Average Sales by Promotion')
    ax.set_xlabel('Promotion Status')
    ax.set_ylabel('Average Sales')
    return fig

def plot_holiday_effect(data, figsize=(15, 7)):
    logger.info("Plotting holiday effect...")
    data.copy()
    data = generate_holiday_data(data)
    holiday_effect = data.groupby('IsHoliday')['Sales'].mean()

    fig, ax = plt.subplots(figsize=figsize)
    holiday_effect.plot(kind='bar', ax=ax)
    ax.set_title('Holiday Effect on Average Sales')
    ax.set_xlabel('Holiday')
    ax.set_ylabel('Sales')
    return fig

def plot_sales_during_holidays(data, holiday_column, sales_column):
    """
    Compare sales during holidays vs. non-holidays.
    """
    logger.info("Comparing sales during holidays vs. non-holidays...")
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x=data[holiday_column], y=data[sales_column], palette="Set2", ax=ax)
    ax.set_title(f"Sales Distribution by {holiday_column}")
    ax.set_xlabel(holiday_column)
    ax.set_ylabel(sales_column)
    plt.tight_layout()
    return fig

def plot_seasonal_decomposition(data, column_name, title="Seasonal Decomposition", xlabel="Date", ylabel="Sales", freq='M', figsize=(15, 7)):
    """
    Decompose the time series data into trend, seasonal, and residual components.
    Handles exceptions such as periodicity mismatches.
    """
    logger.info("Performing seasonal decomposition...")
    try:
        sales = data[column_name].resample(freq).sum()
        period = data[column_name].resample(freq).size
        decomposition = seasonal_decompose(sales, model='additive', period=period)
        
        fig, ax = plt.subplots(figsize=figsize)
        decomposition.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        return fig
    except ValueError:
        print(f"Error: Periodicity mismatch when decomposing {column_name}. Please verify the frequency or the data periodicity.")

def plot_time_series_diagnostics(data, column, diagnostics_type='ACF_PACF', **kwargs):
    """
    Plot ACF and PACF for a time series.
    """
    logger.info("Plotting time series diagnostics...")
    if diagnostics_type == 'ACF_PACF':
        lags = kwargs.get('lags', 40)

        fig, ax = plt.subplots(figsize=(15, 7))
        plt.subplot(211)
        plot_acf(data[column].dropna(), ax=plt.gca(), lags=lags)
        plt.subplot(212)
        plot_pacf(data[column].dropna(), ax=plt.gca(), lags=lags)
        plt.tight_layout()
        return fig
    else:
        raise ValueError("Invalid diagnostics type.")

def plot_ACF_PACF(data, column_name, lags=40, figsize=(15, 10)):
    logger.info("Plotting ACF and PACF...")
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplot(121)
    plot_acf(data[column_name], lags=lags, ax=ax)
    plt.title('Autocorrelation (ACF)')
    plt.subplot(122)
    plot_pacf(data[column_name], lags=lags, ax=ax)
    plt.title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    return fig
    
def plot_ACF_PACF_2(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 10)):
    logger.info("Plotting ACF and PACF 2...")
    sales = data[column_name].resample(freq).sum()
    n_lags = len(sales) // 3
    acf_values = acf(sales.dropna(), nlags=n_lags)
    pacf_values = pacf(sales.dropna(), nlags=n_lags)

    fig, ax = plt.subplots(figsize=figsize)

    plt.subplot(121)
    plt.stem(range(len(acf_values)), acf_values, use_line_collection=True)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.subplot(122)
    plt.stem(range(len(pacf_values)), pacf_values, use_line_collection=True)
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    plt.axhline(y=1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    plt.title("Partial " + title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    return fig

def plot_sales_various_trends(data, date_column='Date', sales_column='Sales', freq='Month'):
    logger.info("Plotting sales trends...")
    data = data.copy()
    if freq == 'Month':
        data['freq'] = data[date_column].dt.month
    elif freq == 'Week':
        data['freq'] = data[date_column].dt.isocalendar().week
    elif freq == 'Weekday':
        data['freq'] = data[date_column].dt.dayofweek
    elif freq == 'Year':
        data['freq'] = data[date_column].dt.year
    elif freq == 'Quarter':
        data['freq'] = data[date_column].dt.quarter
    elif freq == 'Day':
        data['freq'] = data[date_column].dt.day
    elif freq == 'Hour':
        data['freq'] = data[date_column].dt.hour
    elif freq == 'Minute':
        data['freq'] = data[date_column].dt.minute
    elif freq == 'Second':
        data['freq'] = data[date_column].dt.second
    else:
        raise ValueError("Invalid frequency")

    monthly_sales = data.groupby('freq')[sales_column].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=monthly_sales, x='freq', y=sales_column, ax=ax)
    ax.set_title(f'{freq}ly Sales Trends')
    ax.set_xlabel(freq)
    ax.set_ylabel('Sales')
    ax.set_xticklabels(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_comparison(data, status_column, category_column, title='', xlabel='', ylabel=''):
    logger.info("Plotting sales comparison...")
    data = data.copy()
    data['Week'] = data['Date'].dt.isocalendar().week
    data['Year'] = data['Date'].dt.year
    data['Weekday'] = data['Date'].dt.dayofweek
    grouped_data = data[status_column].groupby(data[category_column]).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_data.plot(
        kind='bar', title=title, xlabel=xlabel, ylabel=ylabel, ax=ax
    )
    plt.tight_layout()
    return fig

def plot_weekly_sales_heatmap(data, date_column='Date', sales_column='Sales'):
    logger.info("Plotting weekly sales heatmap...")
    data = data.copy()
    data['Week'] = data[date_column].dt.isocalendar().week
    data['Year'] = data[date_column].dt.year
    data['Weekday'] = data[date_column].dt.dayofweek

    weekly_sales = data.groupby(['Year', 'Week', 'Weekday'])[sales_column].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(weekly_sales, cmap='Blues', annot=True, fmt='g', ax=ax)
    ax.set_title('Weekly Sales Heatmap')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Week')
    plt.tight_layout()
    return fig

def plot_sales_vs_competitor_distance(data, distance_column='CompetitionDistance', sales_column='Sales'):
    logger.info("Plotting sales vs competitor distance...")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=data, x=distance_column, y=sales_column, hue='StoreType', ax=ax)
    ax.set_title('Sales vs. Competitor Distance')
    ax.set_xlabel('Distance to Competitor')
    ax.set_ylabel('Sales')
    plt.tight_layout()
    return fig

def plot_sales_around_competitor_openings(data, date_column='Date', sales_column='Sales'):
    logger.info("Plotting sales around competitor openings...")
    
    # Ensure the date columns are in datetime format
    data[date_column] = pd.to_datetime(data[date_column])

    # Create additional columns for analysis
    data = data.copy()
    
    # Create a 'Competitor_Open_Date' column based on CompetitionOpenSinceMonth and CompetitionOpenSinceYear
    data['Competitor_Open_Date'] = pd.to_datetime(
        data['CompetitionOpenSinceYear'].astype(str) + '-' + 
        data['CompetitionOpenSinceMonth'].astype(str) + '-01', 
        errors='coerce'
    )

    # Create additional column for analysis
    data['Before_After_Competitor_Opening'] = data[date_column].apply(
        lambda x: 'Before' if x < data['Competitor_Open_Date'].max() else 'After'
    )

    # Group by the new column and date, summing the sales
    competitor_sales = data.groupby(['Before_After_Competitor_Opening', date_column])[sales_column].sum().reset_index()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=competitor_sales, x=date_column, y=sales_column, hue='Before_After_Competitor_Opening', ax=ax)
    ax.set_title('Sales Trends Around Competitor Openings')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(title='Competitor Opening')
    ax.set_xticklabels(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_trend_holidays(data, holiday_dates=None, country='US', date_column='Date', sales_column='Sales', holidays_only=True, offset=7):
    logger.info("Plotting sales trend during holidays...")
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.copy()
    
    # Create a holiday calendar for the specified country
    holiday_calendar = holidays.CountryHoliday(country)
    
    # Retrieve holidays for each unique year in the data if none are provided
    if not holiday_dates:
        holiday_dates = []
        for year in data['Date'].dt.year.unique():
            holiday_calendar.get(str(year))
            holiday_dates.extend(list(holiday_calendar.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 7))

    for holiday_date in holiday_dates:
        holiday = pd.Timestamp(holiday_date)
        
        # Check if the date is an actual holiday
        if holiday not in holiday_calendar:
            print(f"Warning: {holiday.strftime('%Y-%m-%d')} is not a recognized holiday in {country}. Skipping.")
            continue
            
        holiday_name = holiday_calendar.get(holiday)
        filtered_data = data.copy()

        if holidays_only:
            offset_days = pd.DateOffset(days=offset)
            
            # Filter sales data for the relevant period
            filtered_data = data[(data[date_column] >= holiday - offset_days) & (data[date_column] <= holiday + offset_days)]
        
        # Classify holiday status
        filtered_data['HolidayStatus'] = filtered_data[date_column].apply(
            lambda x: 'During' if pd.Timestamp(x) == holiday else ('Before' if pd.Timestamp(x) < holiday else 'After')
        )

        # Grouping by holiday for color coding
        filtered_data['Holiday'] = filtered_data[date_column].apply(lambda x: next((h for h in holiday_dates if pd.Timestamp(h) == x), None))

        # Plotting
        sns.scatterplot(data=filtered_data, x=date_column, y=sales_column, hue='Holiday', style='HolidayStatus', palette='tab10', ax=ax)

        # Highlight each holiday with a vertical line and its name
        ax.axvline(holiday, color='red', linestyle='--', label=f'Holiday: {holiday_name} ({holiday.strftime("%Y-%m-%d")})')

    # Final plot adjustments
    ax.set_title('Sales Before, During, and After Multiple Holidays', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    ax.set_xticklabels(rotation=45)
    # plt.legend(title='Holidays', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig

def plot_sales_holidays(data, country='US', date_column='Date', sales_column='Sales'):
    logger.info("Plotting sales on holidays...")
    data = data.copy()
    # holiday_calendar = holidays.US()
    holiday_calendar = holidays.CountryHoliday(country)
    data['IsHoliday'] = data[date_column].apply(lambda x: 1 if x in holiday_calendar else 0)    
    
    holiday_sales = data[data['IsHoliday'] == 1].groupby(date_column)[sales_column].sum()
    non_holiday_sales = data[data['IsHoliday'] == 0].groupby(date_column)[sales_column].sum()

    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot holiday sales as a bar chart
    # plt.bar(holiday_sales.index, holiday_sales.values, label='Holiday Sales', color='red', alpha=0.6)
    plt.plot(holiday_sales.index, holiday_sales.values, label='Holiday Sales', color='red', linestyle='-', alpha=0.7)
    
    # Plot non-holiday sales as a line plot
    plt.plot(non_holiday_sales.index, non_holiday_sales.values, label='Non-Holiday Sales', color='blue', marker='o', alpha=0.7)
    
    ax.set_title('Sales on Holidays vs Non-Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    ax.set_xticklabels(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig

def plot_sales_before_during_after_holidays(data, holiday_dates=None, country='US', date_column='Date', sales_column='Sales', holidays_only=True, offset=7):
    logger.info("Plotting sales before, during, and after holidays...")
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.copy()
    
    # Create a holiday calendar for the specified country
    holiday_calendar = holidays.CountryHoliday(country)
    
    # Retrieve holidays for each unique year in the data if none are provided
    if not holiday_dates:
        holiday_dates = []
        for year in data['Date'].dt.year.unique():
            holiday_calendar.get(str(year))
            holiday_dates.extend(list(holiday_calendar.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 7))

    for holiday_date in holiday_dates:
    
        holiday = pd.Timestamp(holiday_date)
        
        # Check if the date is an actual holiday
        if holiday not in holiday_calendar:
            print(f"Warning: {holiday.strftime('%Y-%m-%d')} is not a recognized holiday in {country}. Skipping.")
            continue
            
        holiday_name = holiday_calendar.get(holiday)
        filtered_data = data.copy()

        if holidays_only:
            offset_days = pd.DateOffset(days=offset)
            
            # Filter sales data for the relevant period
            filtered_data = data[(data[date_column] >= holiday - offset_days) & (data[date_column] <= holiday + offset_days)]
            
        # Plot the sales data
        plt.plot(filtered_data[date_column], filtered_data[sales_column], marker='o', label=f'Sales around {holiday.strftime("%Y-%m-%d")}', alpha=0.7)

        # Highlight each holiday with a vertical line and its name
        plt.axvline(holiday, color='red', linestyle='--', label=f'Holiday: {holiday_name} ({holiday.strftime("%Y-%m-%d")})')
        
        # Shading areas
        plt.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] < holiday), color='blue', alpha=0.1)
        plt.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] == holiday), color='red', alpha=0.3)
        plt.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] > holiday), color='green', alpha=0.1)

    # Final touches for the plot
    ax.set_title('Sales Trends Before, During, and After Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    ax.set_xticklabels(rotation=45)
    # plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig

def plot_sales_by_holiday(data, country='US', date_column='Date', sales_column='Sales'):
    logger.info("Plotting sales by holiday...")
    data = data.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    holiday_calendar = holidays.CountryHoliday(country)

    # Classify sales
    def holiday_status(row):
        date = pd.Timestamp(row[date_column])
        if date in holiday_calendar:
            return 'During'
        closest_holiday = min(
            (pd.Timestamp(holiday) for holiday in holiday_calendar.keys() if pd.Timestamp(holiday) > date),
            default=None
        )
        return 'Before' if closest_holiday is not None and date < closest_holiday else 'After'

    data['HolidayStatus'] = data.apply(holiday_status, axis=1)

    # Grouping by holiday for color coding
    data['Holiday'] = data[date_column].apply(lambda x: next((h for h in holiday_calendar.keys() if pd.Timestamp(h) == x), None))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(data=data, x=date_column, y=sales_column, hue='Holiday', palette='tab10', style='HolidayStatus', ax=ax)    
    ax.set_title('Sales Before, During, and After Specific Holidays')
    ax.set_title('Sales on Holidays vs Non-Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    ax.set_xticklabels(rotation=45)
    # plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig

def plot_store_type_performance(data):
    logger.info("Plotting store type performance over time...")
    data = data.copy()
    store_type_sales = data.groupby([data['Date'].dt.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
    store_type_sales.plot(figsize=(15, 7))
    plt.title('Monthly Average Sales by Store Type')
    plt.xlabel('Date')
    plt.ylabel('Average Sales')
    plt.legend(title='Store Type')
    plt.show()

def plot_sales_vs_customers(data):
    logger.info("Plotting sales vs customers scatter plot...")
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(data['Customers'], data['Sales'], c=data['Date'], cmap='viridis')
    plt.colorbar(scatter, label='Date')
    plt.title('Sales vs Customers Over Time')
    plt.xlabel('Number of Customers')
    plt.ylabel('Sales')
    plt.show()

def plot_cumulative_sales(data):
    logger.info("Plotting cumulative sales over time...")
    data = data.copy()
    plt.figure(figsize=(15, 7))
    plt.plot(data['Date'], data['Sales'].cumsum())
    plt.title('Cumulative Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Sales')
    plt.show()

def plot_sales_growth_rate(data):
    logger.info("Plotting daily sales growth rate...")
    data = data.copy()
    data['SalesGrowthRate'] = data['Sales'].pct_change()
    plt.figure(figsize=(15, 7))
    plt.plot(data['Date'], data['SalesGrowthRate'])
    plt.title('Daily Sales Growth Rate')
    plt.xlabel('Date')
    plt.ylabel('Growth Rate')
    plt.show()

def plot_correlation_analysis(data):
    logger.info("Performing correlation analysis...")
    data = data.copy()
    data = data.drop(columns=['SalesGrowthRate', 'IsHoliday'], errors='ignore')
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    correlations = data[numeric_columns].corr()['Sales'].abs().sort_values(ascending=False)

    top_features = correlations[1:11].index.tolist()
    f_correlation = data[top_features].corr()
    f_mask = np.triu(np.ones_like(f_correlation, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(f_correlation, mask=f_mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title('Top 10 Features Correlated with Sales', fontsize=16)
    plt.tight_layout()
    plt.show()

    logger.info("Correlations with Sales:")
    logger.info(correlations[top_features])

# Plot Actual vs Predicted
def plot_actual_vs_predicted(y_true, y_pred, dataset_type="Dataset"):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(y_true, y_pred, alpha=0.5, color='blue')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_title(f'{dataset_type}: Actual vs Predicted')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.grid(True)
    return fig

def plot_training_history(history):
    """
    Plot training and validation loss/metrics.

    Args:
    - history: Keras training history object.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('LSTM Training vs. Validation Loss')
    ax.legend()
    return fig

def plot_feature_importances(data, importances):
        
    indices = np.argsort(importances)
    features_ranked=[data.columns[indices[f]] for f in range(data.shape[1])]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title("Feature importances")
    ax.barh(range(data.shape[1]), importances[indices],
                color=[next(itertools.cycle(sns.color_palette()))], align="center")
    ax.set_yticks(range(data.shape[1]), features_ranked)
    ax.set_ylabel('Features')
    ax.set_ylim([-1, data.shape[1]])
    return fig