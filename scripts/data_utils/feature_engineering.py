import holidays
import numpy as np
import pandas as pd

from ast import Add
from pandas.tseries.offsets import MonthBegin
from statsmodels.tsa.seasonal import seasonal_decompose
from scripts.utils.logger import setup_logger

# Setup logger for feature_engineering.py
logger = setup_logger("feature_engineering")

def extract_date_features(data, date_column='Date'):
    """Extract useful features from the Date column."""
    data = data.copy()
    logger.info("Starting to extract date features...")
    data[date_column] = pd.to_datetime(data[date_column])
    data['Year'] = data[date_column].dt.year
    data['Month'] = data[date_column].dt.month
    data['Day'] = data[date_column].dt.day
    data['Week'] = data[date_column].dt.isocalendar().week.astype(int)
    data['WeekOfYear'] = data[date_column].dt.isocalendar().week
    data['Weekday'] = data[date_column].dt.dayofweek  # 0 = Monday, 6 = Sunday
    data['IsWeekend'] = (data[date_column].dt.weekday >= 5).astype(int)

    logger.info("Date features extracted successfully.")
    return data

def generate_holiday_data(data, date_column='Date', country='ET'):
    """
    This function generates a new column 'IsHoliday' and 'Holiday' in the dataframe based on the country's holidays.
    And add a 'Holiday_Status' column with values 'Before', 'During', or 'After' based on proximity to holidays.

    Args:
        data (pd.DataFrame): The dataframe to add the 'IsHoliday' column to.
        country (str, optional): The country code for which to generate the holidays. Defaults to 'ET'.

    Returns:
        None
    """
    data = data.copy()
    logger.info(f"Starting to generate holiday data for {country}...")
    holiday_calendar = holidays.country_holidays(country)
    # holiday_calendar = holidays.CountryHoliday(country)
    data['IsHoliday'] = data[date_column].apply(lambda date: date in holiday_calendar).astype(int)
    # data['IsHoliday'] = data.index.to_series().apply(lambda date: date in holiday_calendar).astype(bool)


    # Use the get method to retrieve holiday names
    # data['Holiday'] = data[date_column].apply(lambda date: holiday_calendar.get(date, None))
    # data['Holiday'] = data[date_column].apply(
    #     lambda x: next((holiday for holiday in holiday_calendar.keys() if pd.Timestamp(holiday) == x), None)
    # )
    # data = data.merge(
    #     pd.DataFrame(list(holiday_calendar.items()), columns=['Holiday_Date', 'Holiday']), 
    #     left_on=date_column, right_on='Holiday_Date', how='left').drop(columns=['Holiday_Date']
    # )

    # def holiday_status(row):
    #     date = pd.Timestamp(row[date_column])
    #     if date in holiday_calendar:
    #         return 'During'
    #     closest_holiday = min(
    #         (pd.Timestamp(holiday) for holiday in holiday_calendar.keys() if pd.Timestamp(holiday) > date),
    #         default=None
    #     )
    #     return 'Before' if closest_holiday is not None and date < closest_holiday else 'After'

    # data['Holiday_Status'] = data.apply(holiday_status, axis=1)

    logger.info("Holiday data generated successfully.")
    return data

def generate_sales_features(data):
# def generate_sales_features(data, training=True, sales_agg=None, group_by='Store'):
    """
    Generate sales-related features. Handles cases where `Sales` is not available in the dataset.

    Args:
        data (pd.DataFrame): Input dataframe with sales data.
        for_training (bool): Whether the data is for training. If False, proxies will be used.
        sales_agg (pd.DataFrame): Aggregated statistics for `SalesPerCustomer` or similar features.

    Returns:
        pd.DataFrame: Dataframe with sales features.
    """
    data = data.copy()
    logger.info("Generating sales-related features...")
    
    # if training:
    data['SalesPerCustomer'] = np.where(data['Customers'] > 0, 
                                            data['Sales'] / data['Customers'], 
                                            0)
    data['PromoEffectiveness'] = np.where(data['Promo'] == 1, data['Sales'], 0)
    data['SalesGrowthRate'] = data['Sales'].pct_change() * 100
    data['SalesGrowthRate'] = data['SalesGrowthRate'].replace([np.inf, -np.inf], 0)
    # else:
    #     data['SalesPerCustomer'] = data[group_by].map(sales_agg['SalesPerCustomer'])
    #     data['PromoEffectiveness'] = data[group_by].map(sales_agg['PromoEffectiveness'])
    #     data['SalesGrowthRate'] = data[group_by].map(sales_agg['SalesGrowthRate'])
    #     # data['SalesGrowthRate'] = np.nan
    
    logger.info("Sales features generated successfully.")
    return data

def calculate_sales_aggregates(data, group_by='Store'):
    """
    Calculate aggregated sales features for use in test data preprocessing.

    Args:
        data (pd.DataFrame): Training data containing `Sales`, `Customers`, and `Promo`.
        group_by (str): Column to group data by (e.g., 'Store').

    Returns:
        pd.DataFrame: Aggregated sales metrics as a lookup table.
    """
    logger.info("Calculating sales aggregates...")

    # Filter out rows with missing or zero values to ensure clean aggregates
    data = data.copy()
    data = data[data['Customers'] > 0]

    # Group by and aggregate sales-related metrics
    sales_agg = data.groupby(group_by).agg(
        # SalesPerCustomer=('Sales', lambda x: (x / data['Customers']).mean()),
        SalesPerCustomer=('Sales', lambda x: (x / data.loc[x.index, 'Customers']).mean()),
        # PromoEffectiveness=('Sales', lambda x: x[data['Promo'] == 1].mean()),
        PromoEffectiveness=('Sales', lambda x: x[data.loc[x.index, 'Promo'] == 1].mean()),
    ).reset_index()

    # Calculate SalesGrowthRate
    sales_agg['SalesGrowthRate'] = data.groupby(group_by)['Sales'].mean().pct_change() * 100

    logger.info("Sales aggregates calculated successfully.")
    return sales_agg

def generate_competitor_open_date(data):
    """Creates a 'CompetitionOpenSince' column from month and year columns."""
    
    data = data.copy()
    logger.info("Starting to generate competitor opening date...")

    data['CompetitionOpenSince'] = pd.to_datetime(
        data['CompetitionOpenSinceYear'].fillna(0).astype(int).astype(str) + '-' +
        data['CompetitionOpenSinceMonth'].fillna(1).astype(int).astype(str) + '-01', 
        format='%Y-%m-%d', 
        errors='coerce'
    )
    
    logger.info("Competitor opening date generated successfully.")
    return data

def generate_competitor_features(data, date_column='Date'):
    """Adds a 'BeforeAfterCompetitorOpening' column based on CompetitionOpenSince."""
    data = data.copy()
    logger.info("Starting to generate competitor features...")

    # Create a boolean mask for compariso
    data['BeforeAfterCompetitorOpening'] = data.apply(
        lambda x: 'Before' if x[date_column] < x['CompetitionOpenSince'] else 'After', axis=1
    )
    logger.info("Competitor features generated successfully.")
    return data

def decompose_time_series(data, column_name, freq, model='additive'):
    """Performs time series decomposition for the given column."""
    data = data.copy()
    logger.info(f"Starting to decompose time series for {column_name}...")
    sales = data[column_name].resample(freq).sum()
    period = data[column_name].resample(freq).size

    # decomposition = seasonal_decompose(sales, model=model)
    # decomposition = seasonal_decompose(data[column_name], model=model, period=freq)
    decomposition = seasonal_decompose(sales, model=model, period=period)

    logger.info(f"Time series decomposition for {column_name} completed.")
    return decomposition

def create_interaction_features(data):
    """Generate meaningful interaction features."""

    data = data.copy()
    logger.info("Starting to create interaction features...")
    data['CompetitionActive'] = ((data['Year'] > data['CompetitionOpenSinceYear']) |
                                 ((data['Year'] == data['CompetitionOpenSinceYear']) & 
                                  (data['Month'] >= data['CompetitionOpenSinceMonth']))).astype(int)

    data['Promo2Active'] = ((data['Year'] > data['Promo2SinceYear']) |
                            ((data['Year'] == data['Promo2SinceYear']) & 
                             (data['WeekOfYear'] >= data['Promo2SinceWeek']))).astype(int)

    data['CompetitionMonthsOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + \
                                     (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['CompetitionMonthsOpen'] = data['CompetitionMonthsOpen'].apply(lambda x: x if x > 0 else 0)

    data['Promo2WeeksActive'] = 52 * (data['Year'] - data['Promo2SinceYear']) + \
                                 (data['WeekOfYear'] - data['Promo2SinceWeek'])
    data['Promo2WeeksActive'] = data['Promo2WeeksActive'].apply(lambda x: x if x > 0 else 0)

    logger.info("Interaction features created successfully.")
    return data