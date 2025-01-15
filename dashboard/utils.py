import holidays
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set a unified style for all plots
sns.set_theme(style="whitegrid")

def clean_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['CompetitionOpenSince'] = pd.to_datetime(data['CompetitionOpenSince']) 
    data ['StateHoliday'] = data['StateHoliday'].astype(str) 
    data = data.replace([np.inf, -np.inf], np.nan)
    return data

def plot_distribution_comparison_dataset(data, column, compare_column=None, kind='numerical', bins=30, figsize=(15, 7)):
    fig, ax = plt.subplots(figsize=figsize)
    
    if compare_column:
        if kind == 'numerical':
            sns.histplot(data, x=column, hue=compare_column, kde=True, color='blue', ax=ax)#, bins=bins)
        elif kind == 'categorical':
            sns.countplot(data=data, x=column, hue=compare_column, color='blue', ax=ax)
    else:
        sns.histplot(data[column], kde=True, color='blue', ax=ax)#, bins=bins)

    ax.set_title(f"{'Comparison of ' + kind if compare_column else 'Distribution of'} '{column}'")
    ax.set_ylabel(column)
    plt.tight_layout()
    return fig

def plot_distribution_comparision_datasets(train, test, column, kind='numerical', title=""):
    fig, ax = plt.subplots(figsize=(15, 7))
    if kind == 'categorical':
        sns.countplot(x=column, data=train, label="Train", color='blue', alpha=0.5, ax=ax)#, hue=compare_column)
        sns.countplot(x=column, data=test, label="Test", color='red', alpha=0.5, ax=ax)
    else:
        sns.kdeplot(train[column], label="Train", fill=True, color="blue", ax=ax)#, hue=compare_column)
        sns.kdeplot(test[column], label="Test", fill=True, color="red", ax=ax)

    ax.set_title(title or f"Comparison of {column} in Train and Test Sets")
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_sales_correlations_analysis(data, columns, trends=False, title=""):
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
    if analysis_type == 'by_store_type':
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='StoreType', y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales by Store Type')
    elif analysis_type == 'sales_vs_customers':
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Customers', y='Sales', hue='StoreType', ax=ax)
        ax.set_title('Sales vs. Customers')
    elif analysis_type == 'heatmap':
        freq = kwargs.get('freq', 'M')
        pivot = data.groupby('Month')['Sales'].mean().reset_index()
        pivot['Month'] = pivot['Month'].dt.strftime('%Y-%m')
        pivot = pivot.pivot_table(index='Month', values='Sales')
        fig, ax = plt.subplots()
        sns.heatmap(pivot, cmap='coolwarm', annot=True, fmt='.0f', ax=ax)
        ax.set_title('Sales Heatmap')
    elif analysis_type == 'cumulative':
        cumulative_sales = data['Sales'].cumsum()
        fig, ax = plt.subplots()
        plt.plot(data['Date'], cumulative_sales)
        ax.set_title('Cumulative Sales Over Time')
    elif analysis_type == 'weekday':
        fig, ax = plt.subplots()
        sns.barplot(data=data, x='Weekday', y='Sales', order=[
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
        ax.set_title('Sales by Weekday')
    elif analysis_type == 'holiday_effect':
        fig, ax = plt.subplots()
        sns.boxplot(data=data, x='IsHoliday', y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales During Holidays' if kwargs.get('detailed', False) else 'Sales and Holidays')
    else:
        fig, ax = plt.subplots()
        sns.boxplot(data=data, y='Sales', palette="Set2", ax=ax)
        ax.set_title('Sales')
    
    ax.set_ylabel('Sales')
    plt.tight_layout()
    return fig

def plot_event_effect(data, event_column, effect_type='promotion'):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(data=data, x=event_column, y='Sales', palette="Set2", ax=ax)
    ax.set_title(f"Effect of {effect_type.capitalize()} on Sales")
    return fig

def plot_time_series_analysis(data, column, analysis_type, **kwargs):
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
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.histplot(data[column], kde=True, color="blue", ax=ax)
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_sales_trends(data, sales_column="Sales"):
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
    monthly_data = data.groupby('Month')['Sales'].sum().values.reshape(-1, 1)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(monthly_data, annot=True, fmt='.0f', cmap='coolwarm', ax=ax)
    ax.set_title('Monthly Sales Heatmap')
    ax.set_ylabel('Month')
    ax.set_xlabel('Sales')
    return fig

def plot_sales_month_week_heatmap(data):
    sales_heatmap = data.pivot_table(values='Sales', index='DayOfWeek', columns='Month', aggfunc='mean')    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sales_heatmap, cmap='YlOrRd', annot=True, fmt='.0f', ax=ax)
    ax.set_title('Average Sales by Day of Week and Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Day of Week (0=Monday, 6=Sunday)')
    return fig
    
def plot_sales(data, column_name='Sales', title='Sales Trend Over Time', xlabel='Date', ylabel='Sales', freq='w', figsize=(15, 7)):

    aggregated_sales = data[column_name].resample(freq).sum()
    
    fig, ax = plt.subplots(figsize=figsize)
    plt.plot(aggregated_sales.index, aggregated_sales)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    return fig

def plot_weekday_sales(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x='Weekday', y='Sales', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ax=ax)
    ax.set_title('Weekday Sales Distribution')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Sales')
    return fig

def plot_promotion_effect(data, freq='M', figsize=(15, 7)):
    promo_effect = data.groupby('Promo')['Sales'].mean()
    fig, ax = plt.subplots(figsize=figsize)
    promo_effect.plot(kind='bar', color='skyblue', rot=0, ax=ax)
    ax.set_title('Average Sales by Promotion')
    ax.set_xlabel('Promotion Status')
    ax.set_ylabel('Average Sales')
    plt.tight_layout()
    return fig

def plot_promotion_effect_2(data, freq='M', figsize=(15, 7)):
    sales = data.groupby([data['Date'].dt.to_period(freq), 'Promo'])['Sales'].mean().unstack()
    sales.columns = ['No Promo', 'Promo']
    
    fig, ax = plt.subplots(figsize=figsize)
    sales[['No Promo', 'Promo']].plot(ax=ax)
    ax.set_title('Average Sales by Promotion')
    ax.set_xlabel('Promotion Status')
    ax.set_ylabel('Average Sales')
    return fig

def plot_holiday_effect(data, figsize=(15, 7)):
    holiday_effect = data.groupby('IsHoliday')['Sales'].mean()

    fig, ax = plt.subplots(figsize=figsize)
    holiday_effect.plot(kind='bar', ax=ax)
    ax.set_title('Holiday Effect on Average Sales')
    ax.set_xlabel('Holiday')
    ax.set_ylabel('Sales')
    return fig

def plot_sales_during_holidays(data, holiday_column, sales_column):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(x=data[holiday_column], y=data[sales_column], palette="Set2", ax=ax)
    ax.set_title(f"Sales Distribution by {holiday_column}")
    ax.set_xlabel(holiday_column)
    ax.set_ylabel(sales_column)
    plt.tight_layout()
    return fig

def plot_seasonal_decomposition(data, column_name, title="Seasonal Decomposition", xlabel="Date", ylabel="Sales", freq='M', figsize=(15, 7)):
    try:
        sales = data[column_name].resample(freq).sum()
        period = data[column_name].resample(freq).size
        decomposition = seasonal_decompose(sales, model='additive', period=period)
        
        fig, ax = plt.subplots(figsize=figsize)
        decomposition.plot(ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig
    except ValueError:
        print(f"Error: Periodicity mismatch when decomposing {column_name}. Please verify the frequency or the data periodicity.")

def plot_time_series_diagnostics(data, column, diagnostics_type='ACF_PACF', **kwargs):
    if diagnostics_type == 'ACF_PACF':
        lags = kwargs.get('lags', 40)

        fig, ax = plt.subplots(figsize=(15, 7))
        ax1 = plt.subplot(211)
        plot_acf(data[column].dropna(), ax=ax1, lags=lags)
        ax1.axhline(y=0, linestyle='--', color='gray')
        ax1.axhline(y=-1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
        ax1.axhline(y=1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
        ax1.set_title(kwargs.get('title', 'title1'))
        ax1.set_xlabel(kwargs.get('xlabel', 'xlabel1'))
        ax1.set_ylabel(kwargs.get('ylabel', 'ylabel1'))

        ax2 = plt.subplot(212)
        plot_pacf(data[column].dropna(), ax=ax2, lags=lags)
        ax2.axhline(y=0, linestyle='--', color='gray')
        ax2.axhline(y=-1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
        ax2.axhline(y=1.96/np.sqrt(len(data[column])), linestyle='--', color='gray')
        ax2.set_title("Partial " + kwargs.get('title', 'title2'))
        ax2.set_xlabel(kwargs.get('xlabel', 'xlabel2'))
        ax2.set_ylabel(kwargs.get('ylabel', 'ylabel2'))

        plt.tight_layout()
        return fig
    else:
        raise ValueError("Invalid diagnostics type.")

def plot_ACF_PACF(data, column_name, lags=40, figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    ax1 = plt.subplot(121)
    plot_acf(data[column_name], lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation (ACF)')
    ax2 = plt.subplot(122)
    plot_pacf(data[column_name], lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation (PACF)')
    plt.tight_layout()
    return fig

def plot_ACF_PACF_2(data, column_name, title, xlabel, ylabel, freq='M', figsize=(15, 10)):
    sales = data[column_name].resample(freq).sum()
    n_lags = len(sales) // 3
    acf_values = acf(sales.dropna(), nlags=n_lags)
    pacf_values = pacf(sales.dropna(), nlags=n_lags)

    fig, ax = plt.subplots(figsize=figsize)

    ax1 = fig.add_subplot(121)
    ax1.stem(range(len(acf_values)), acf_values, use_line_collection=True)
    ax1.axhline(y=0, linestyle='--', color='gray')
    ax1.axhline(y=-1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax2 = fig.add_subplot(122)
    ax2.stem(range(len(pacf_values)), pacf_values, use_line_collection=True)
    ax2.axhline(y=0, linestyle='--', color='gray')
    ax2.axhline(y=-1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    ax2.axhline(y=1.96/np.sqrt(len(sales)), linestyle='--', color='gray')
    ax2.set_title("Partial " + title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    plt.tight_layout()
    return fig

def plot_sales_various_trends(data, date_column='Date', sales_column='Sales', freq='Month'):
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
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_comparison(data, status_column, category_column, title='', xlabel='', ylabel=''):
    grouped_data = data[status_column].groupby(data[category_column]).mean()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    grouped_data.plot(
        kind='bar', title=title, xlabel=xlabel, ylabel=ylabel, ax=ax
    )
    plt.tight_layout()
    return fig

def plot_weekly_sales_heatmap(data, date_column='Date', sales_column='Sales'):

    weekly_sales = data.groupby(['Year', 'Week', 'Weekday'])[sales_column].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(weekly_sales, cmap='Blues', annot=True, fmt='g', ax=ax)
    ax.set_title('Weekly Sales Heatmap')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Week')
    plt.tight_layout()
    return fig

def plot_sales_vs_competitor_distance(data, distance_column='CompetitionDistance', sales_column='Sales'):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=data, x=distance_column, y=sales_column, hue='StoreType', ax=ax)
    ax.set_title('Sales vs. Competitor Distance')
    ax.set_xlabel('Distance to Competitor')
    ax.set_ylabel('Sales')
    plt.tight_layout()
    return fig

def plot_sales_around_competitor_openings(data, date_column='Date', sales_column='Sales'):
    
    competitor_sales = data.groupby(['BeforeAfterCompetitorOpening', date_column])[sales_column].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=competitor_sales, x=date_column, y=sales_column, hue='BeforeAfterCompetitorOpening', ax=ax)
    ax.set_title('Sales Trends Around Competitor Openings')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.legend(title='Competitor Opening')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_trend_holidays(data, holiday_dates=None, country='US', date_column='Date', sales_column='Sales', holidays_only=True, offset=7):
    
    holiday_calendar = holidays.CountryHoliday(country)
    
    if not holiday_dates:
        holiday_dates = []
        for year in data['Date'].dt.year.unique():
            holiday_calendar.get(str(year))
            holiday_dates.extend(list(holiday_calendar.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 7))

    for holiday_date in holiday_dates:
        holiday = pd.Timestamp(holiday_date)
        
        if holiday not in holiday_calendar:
            print(f"Warning: {holiday.strftime('%Y-%m-%d')} is not a recognized holiday in {country}. Skipping.")
            continue
            
        holiday_name = holiday_calendar.get(holiday)
        filtered_data = data.copy()

        if holidays_only:
            offset_days = pd.DateOffset(days=offset)
            
            filtered_data = data[(data[date_column] >= holiday - offset_days) & (data[date_column] <= holiday + offset_days)]
        
        filtered_data['HolidayStatus'] = filtered_data[date_column].apply(
            lambda x: 'During' if pd.Timestamp(x) == holiday else ('Before' if pd.Timestamp(x) < holiday else 'After')
        )

        filtered_data['Holiday'] = filtered_data[date_column].apply(lambda x: next((h for h in holiday_dates if pd.Timestamp(h) == x), None))

        sns.scatterplot(data=filtered_data, x=date_column, y=sales_column, hue='Holiday', style='HolidayStatus', palette='tab10', ax=ax)

        ax.axvline(holiday, color='red', linestyle='--', label=f'Holiday: {holiday_name} ({holiday.strftime("%Y-%m-%d")})')

    ax.set_title('Sales Before, During, and After Multiple Holidays', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_sales_holidays(data, country='US', date_column='Date', sales_column='Sales'):

    holiday_sales = data[data['IsHoliday'] == 1].groupby(date_column)[sales_column].sum()
    non_holiday_sales = data[data['IsHoliday'] == 0].groupby(date_column)[sales_column].sum()

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(holiday_sales.index, holiday_sales.values, label='Holiday Sales', color='red', linestyle='-', alpha=0.7)
    ax.plot(non_holiday_sales.index, non_holiday_sales.values, label='Non-Holiday Sales', color='blue', marker='o', alpha=0.7)
    
    ax.set_title('Sales on Holidays vs Non-Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    plt.xticks(rotation=45)
    ax.legend()
    plt.grid()
    plt.tight_layout()
    return fig

def plot_sales_before_during_after_holidays(data, holiday_dates=None, country='US', date_column='Date', sales_column='Sales', holidays_only=True, offset=7):

    holiday_calendar = holidays.CountryHoliday(country)
    
    if not holiday_dates:
        holiday_dates = []
        for year in data['Date'].dt.year.unique():
            holiday_calendar.get(str(year))
            holiday_dates.extend(list(holiday_calendar.keys()))
    
    fig, ax = plt.subplots(figsize=(14, 7))

    for holiday_date in holiday_dates:
    
        holiday = pd.Timestamp(holiday_date)
        
        if holiday not in holiday_calendar:
            print(f"Warning: {holiday.strftime('%Y-%m-%d')} is not a recognized holiday in {country}. Skipping.")
            continue
            
        holiday_name = holiday_calendar.get(holiday)
        filtered_data = data.copy()

        if holidays_only:
            offset_days = pd.DateOffset(days=offset)
            
            filtered_data = data[(data[date_column] >= holiday - offset_days) & (data[date_column] <= holiday + offset_days)]
            
        ax.plot(filtered_data[date_column], filtered_data[sales_column], marker='o', label=f'Sales around {holiday.strftime("%Y-%m-%d")}', alpha=0.7)

        ax.axvline(holiday, color='red', linestyle='--', label=f'Holiday: {holiday_name} ({holiday.strftime("%Y-%m-%d")})')
        
        ax.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] < holiday), color='blue', alpha=0.1)
        ax.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] == holiday), color='red', alpha=0.3)
        ax.fill_between(data[date_column], 0, data[sales_column], where=(data[date_column] > holiday), color='green', alpha=0.1)

    ax.set_title('Sales Trends Before, During, and After Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    return fig

def plot_sales_by_holiday(data, country='US', date_column='Date', sales_column='Sales'):

    data = data.copy()
    holiday_calendar = holidays.CountryHoliday(country)

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
    data['Holiday'] = data[date_column].apply(lambda x: next((h for h in holiday_calendar.keys() if pd.Timestamp(h) == x), None))

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.scatterplot(data=data, x=date_column, y=sales_column, hue='Holiday', palette='tab10', style='HolidayStatus', ax=ax)
    ax.set_title('Sales Before, During, and After Specific Holidays')
    ax.set_title('Sales on Holidays vs Non-Holidays', fontsize=16)
    ax.set_xlabel(date_column, fontsize=12)
    ax.set_ylabel(sales_column, fontsize=12)
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    return fig

def plot_store_type_performance(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    store_type_sales = data.groupby([data['Date'].dt.to_period('M'), 'StoreType'])['Sales'].mean().unstack()
    store_type_sales.plot(ax=ax)
    ax.set_title('Monthly Average Sales by Store Type')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average Sales')
    ax.legend(title='Store Type')
    return fig

def plot_sales_vs_customers(data):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(data['Customers'], data['Sales'], c=data['Date'], cmap='viridis')
    plt.colorbar(scatter, ax=ax, label='Date')
    ax.set_title('Sales vs Customers Over Time')
    ax.set_xlabel('Number of Customers')
    ax.set_ylabel('Sales')
    return fig

def plot_cumulative_sales(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data['Date'], data['Sales'].cumsum())
    ax.set_title('Cumulative Sales Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Sales')
    return fig

def plot_sales_growth_rate(data):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(data['Date'], data['SalesGrowthRate'])
    ax.set_title('Daily Sales Growth Rate')
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth Rate')
    return fig

def plot_correlation_analysis(data):
    data = data.copy()
    data = data.drop(columns=['SalesGrowthRate', 'IsHoliday'], errors='ignore')
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    correlations = data[numeric_columns].corr()['Sales'].abs().sort_values(ascending=False)

    top_features = correlations[1:11].index.tolist()
    f_correlation = data[top_features].corr()
    f_mask = np.triu(np.ones_like(f_correlation, dtype=bool))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(f_correlation, mask=f_mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    ax.set_title('Top 10 Features Correlated with Sales', fontsize=16)
    plt.tight_layout()
    return fig

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
    ax.set_yticks(range(data.shape[1]))
    ax.set_yticklabels(features_ranked)
    ax.set_ylabel('Features')
    ax.set_ylim([-1, data.shape[1]])
    return fig