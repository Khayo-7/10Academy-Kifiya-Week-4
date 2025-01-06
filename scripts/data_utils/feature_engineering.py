import holidays
import numpy as np
import pandas as pd

from ast import Add
from pandas.tseries.offsets import MonthBegin
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scripts.utils.logger import setup_logger

# Setup logger for feature_engineering.py
logger = setup_logger("feature_engineering")

def handle_missing_data(data):
    """Handle missing values in the data."""
    data = data.copy()
    logger.info("Starting to handle missing data...")
    # Replace missing CompetitionDistance with a large value (assumed no competition nearby)
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].max() * 2)

    # Replace missing competition open dates with default values
    data['CompetitionOpenSinceYear'] = data['CompetitionOpenSinceYear'].fillna(data['CompetitionOpenSinceYear'].median())
    data['CompetitionOpenSinceMonth'] = data['CompetitionOpenSinceMonth'].fillna(1)

    # Replace missing Promo2 information with defaults
    data['Promo2SinceYear'] = data['Promo2SinceYear'].fillna(0)
    data['Promo2SinceWeek'] = data['Promo2SinceWeek'].fillna(0)
    data['PromoInterval'] = data['PromoInterval'].fillna('')

    logger.info("Missing data handled successfully.")
    return data

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
    data['IsWeekend'] = data[date_column].dt.weekday >= 5

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
    data['IsHoliday'] = data[date_column].apply(lambda date: date in holiday_calendar)#.astype(bool)
    # data['IsHoliday'] = data.index.to_series().apply(lambda date: date in holiday_calendar).astype(bool)


    # Use the get method to retrieve holiday names
    data['Holiday'] = data[date_column].apply(lambda date: holiday_calendar.get(date, None))
    # data['Holiday'] = data[date_column].apply(
    #     lambda x: next((holiday for holiday in holiday_calendar.keys() if pd.Timestamp(holiday) == x), None)
    # )
    # data = data.merge(
    #     pd.DataFrame(list(holiday_calendar.items()), columns=['Holiday_Date', 'Holiday']), 
    #     left_on=date_column, right_on='Holiday_Date', how='left').drop(columns=['Holiday_Date']
    # )

    def holiday_status(row):
        date = pd.Timestamp(row[date_column])
        if date in holiday_calendar:
            return 'During'
        closest_holiday = min(
            (pd.Timestamp(holiday) for holiday in holiday_calendar.keys() if pd.Timestamp(holiday) > date),
            default=None
        )
        return 'Before' if closest_holiday is not None and date < closest_holiday else 'After'

    data['Holiday_Status'] = data.apply(holiday_status, axis=1)

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
                                  (data['Month'] >= data['CompetitionOpenSinceMonth'])))

    data['Promo2Active'] = ((data['Year'] > data['Promo2SinceYear']) |
                            ((data['Year'] == data['Promo2SinceYear']) & 
                             (data['WeekOfYear'] >= data['Promo2SinceWeek'])))

    data['CompetitionMonthsOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) + \
                                     (data['Month'] - data['CompetitionOpenSinceMonth'])
    data['CompetitionMonthsOpen'] = data['CompetitionMonthsOpen'].apply(lambda x: x if x > 0 else 0)

    data['Promo2WeeksActive'] = 52 * (data['Year'] - data['Promo2SinceYear']) + \
                                 (data['WeekOfYear'] - data['Promo2SinceWeek'])
    data['Promo2WeeksActive'] = data['Promo2WeeksActive'].apply(lambda x: x if x > 0 else 0)

    logger.info("Interaction features created successfully.")
    return data

def encode_categorical_features(data, label_encoders=None):
    """Encode categorical variables into numerical values."""

    data = data.copy()
    logger.info("Starting to encode categorical features...")
    if label_encoders is None:
        label_encoders = {}

    for column in ['StoreType', 'Assortment', 'PromoInterval', 'StateHoliday']:
        le = label_encoders.get(column, LabelEncoder())
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    logger.info("Categorical features encoded successfully.")
    return data, label_encoders

def scale_numerical_features(data, scaler=None):
    """Scale numerical columns for consistency."""
    
    data = data.copy()
    logger.info("Starting to scale numerical features...")
    numerical_columns = ['CompetitionDistance', 'CompetitionMonthsOpen', 'Promo2WeeksActive']

    if scaler is None:
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    else:
        data[numerical_columns] = scaler.transform(data[numerical_columns])

    logger.info("Numerical features scaled successfully.")
    return data, scaler

def preprocess_data(data, store, label_encoders=None, scaler=None):
    """Preprocess the data by merging and applying feature engineering."""
    
    data = data.copy()
    logger.info("Starting to preprocess data...")
    # Merge train/test with store data
    data = data.merge(store, on='Store', how='left')

    # Handle Missing Data
    data = handle_missing_data(data)

    # Generate Holidays Features
    data = generate_holiday_data(data)

    # Extract Date Features
    data = extract_date_features(data)

    # Generate Competitor Features
    data = generate_competitor_features(data)

    # Generate Interaction Features
    data = create_interaction_features(data)

    # Encode Categorical Features
    data, label_encoders = encode_categorical_features(data, label_encoders)

    # Scale Numerical Features
    data, scaler = scale_numerical_features(data, scaler)

    logger.info("Data preprocessing completed successfully.")
    return data, label_encoders, scaler

# Main Preprocessing Function
def preprocess_datasets(train, test, store):
    """Preprocess training, test, and store data, applying feature engineering."""

    store_cleaned = handle_missing_data(store)

    # Preprocess train and test for unique cases
    train = generate_sales_features(train)

    # Preprocess training data
    train_preprocessed, label_encoders, scaler = preprocess_data(train, store_cleaned)

    # Preprocess test data using the fitted encoders and scaler from training data
    test_preprocessed, _, _ = preprocess_data(test, store_cleaned, label_encoders, scaler)

    return train_preprocessed, test_preprocessed, store_cleaned