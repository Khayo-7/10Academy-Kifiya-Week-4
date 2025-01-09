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
    """
    This function calculates the sales growth rate.

    Args:
        data (pd.DataFrame): The dataframe containing the sales data.

    Returns:
        pd.DataFrame: The dataframe with the sales growth rate calculated.
    """
    data = data.copy()
    logger.info("Starting to calculate sales growth rate...")
    data['Sales_per_Customer'] = data['Sales'] / data['Customers']
    data['PromoEffectiveness'] = np.where(data['Promo'] == 1, data['Sales'], 0)
    data['SalesGrowthRate'] = data['Sales'].pct_change() * 100
    # data = data.dropna()
    logger.info("Sales growth rate calculated successfully.")
    return data

def generate_competitor_features(data, date_column='Date'):
    """Adds a 'Before_After_Competitor_Opening' column based on Competitor_Open_Date."""
    """Creates a 'Competitor_Open_Date' column from month and year columns."""
    
    data = data.copy()
    logger.info("Starting to generate competitor features...")
    data['Competitor_Open_Date'] = pd.to_datetime(
        data['CompetitionOpenSinceYear'].astype(str) + '-' + data['CompetitionOpenSinceMonth'].astype(str) + '-01', 
        format='%Y-%m-%d', errors='coerce'
    )
    
    # Create a boolean mask for comparison
    competitor_open_date_max = data['Competitor_Open_Date'].max()
    data['Before_After_Competitor_Opening'] = data[date_column].apply(
        lambda x: 'Before' if x < competitor_open_date_max else 'After'
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
