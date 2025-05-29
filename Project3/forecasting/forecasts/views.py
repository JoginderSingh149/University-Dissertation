from django.shortcuts import render, redirect
from django.conf import settings
from django.db import transaction
from django.utils import timezone

import pandas as pd
import numpy as np
import os
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

from plotly.offline import plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.graph_objs import Scatter, Figure

from .forms import UploadFileForm
from .models import BookingData, OptimizationParameters, ProphetOptimizationParameters
from .tasks import optimize_parameters, optimize_prophet_parameters

from prophet import Prophet

warnings.filterwarnings("ignore")


# Define a function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.FILES['file'], request)
            return redirect('dashboard')
        
    else:
        form = UploadFileForm()
    return render(request, 'forecasts/upload.html', {'form': form})


def handle_uploaded_file(f, request):


    df = pd.read_csv(f, delimiter=',', skipinitialspace=True, on_bad_lines='skip', low_memory=False, header=0) # pd.read_csv(f)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    print("Columns:", df.columns.tolist())

    required_columns = ['Check In', 'Check Out', 'Total', 'Nights', 'Guest']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df['Check In'] = pd.to_datetime(df['Check In'], format='%d/%m/%Y %H:%M')
    df['Check In'] = df['Check In'].apply(lambda x: timezone.make_aware(x, timezone.get_default_timezone()))
    df['Check Out'] = pd.to_datetime(df['Check Out'], format='%d/%m/%Y %H:%M')
    df['Check Out'] = df['Check Out'].apply(lambda x: timezone.make_aware(x, timezone.get_default_timezone()))
    df.sort_values(by='Check In', inplace=True)
   
    # Ensure 'Nights' only contains numeric values
    df['Nights'] = df['Nights'].str.extract('(\\d+)').astype(float)

    # Correctly identifying bookings with a standard identifier (each row represents one booking)
    df['Number_of_Bookings'] = 1


    # Find the unique number of rooms
    total_rooms = df['Room'].nunique()

    # Calculate the total number of booked room-nights
    total_booked_nights = df['Nights'].sum()

    # Determine the period covered by the dataset
    start_date = df['Check In'].min()
    end_date = df['Check Out'].max()
    total_days = (end_date - start_date).days + 1

    # ALOS Calculation
    total_bookings = len(df)
    ALOS = total_booked_nights / total_bookings

    total_available_room_nights = total_rooms * total_days

# Calculate Occupancy Rate
    occupancy_rate = total_booked_nights / total_available_room_nights

    # Calculate Total Revenue
    total_revenue = df['Total'].sum()

    # RevPAR Calculation
    #RevPAR = total_revenue / total_available_room_nights

    # Adjusting Daily RevPAR Calculation
    Daily_RevPAR = total_revenue / (total_rooms * total_days)
    year = df 
    year['Year'] = df['Check In'].dt.year
    years = year['Year'].unique()

    # Initialize a variable to accumulate RevPARs and a counter for years with data
    accumulated_revpar = 0
    years_with_data = 0

    # Iterate over each year to calculate RevPAR
    for year in years:
        # Filter the DataFrame for the current year
        df_year = df[df['Year'] == year]
        
        # Calculate total revenue for the year
        total_revenue_year = df_year['Total'].sum()
        
        # Calculate the number of days in the current year (accounting for leap years)
        days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
        
        # Calculate total available room-nights for the year
        total_available_room_nights_year = total_rooms * days_in_year
        
        # Calculate RevPAR for the year
        revpar_year = total_revenue_year / total_available_room_nights_year
        
        # Accumulate the RevPAR and increment the counter
        accumulated_revpar += revpar_year
        years_with_data += 1

    # Calculate the average yearly RevPAR over the entire period
    if years_with_data > 0:
        average_yearly_revpar = accumulated_revpar / years_with_data
    else:
        # Fallback value if the dataset covers less than a year or no complete data for any year
        average_yearly_revpar = "N/A"


    number_of_rooms_sold = df['Nights'].sum()  # This assumes each row represents a booking for one room.
    cumulative_annual_revpar = (total_revenue / total_rooms) / years_with_data if years_with_data else "N/A"

    print(f"Cumulative Annual RevPAR: £{cumulative_annual_revpar}")
# Calculate ADR
    total_room_revenue = df['Total'].sum()
    ADR = total_room_revenue / number_of_rooms_sold

    print(f"Average Daily Rate (ADR): ${ADR:.2f}")
    # Output the results
    print(f"Occupancy Rate: {occupancy_rate:.2%}")
    print(f"Average Length of Stay (ALOS): {ALOS:.2f} nights")
    print(f"Daily RevPAR: £{Daily_RevPAR:.2f}")
    print(f"RevPAR: £{average_yearly_revpar:.2f}")

    request.session['KPIs'] = {
        'ADR': ADR,
        'Occupancy Rate': occupancy_rate,
        'ALOS': ALOS,
        'Daily RevPAR': Daily_RevPAR,
        'Cumulative Annual RevPAR': cumulative_annual_revpar,
    }
    
    # Ensure we  only keep columns necessary for the aggregation step.
    df = df[['Check In', 'Nights', 'Guest', 'Total', 'Number_of_Bookings']]
    df['Guest'] = df['Guest'].astype(str)
    
    # Aggregating data
    daily_records = df.groupby('Check In').agg({'Nights': 'sum', 'Guest': lambda x: ', '.join(x), 'Total': 'sum', 'Number_of_Bookings': 'sum'})
    daily_records['Guest'] = daily_records['Guest'].astype(str)

    print("Daily records shape after processing:", daily_records.shape)
    print("Daily records columns after processing:", daily_records.columns.tolist())

    # Proceed with forecasting logic...
    df = df[df['Total'] != 0]
    daily_records = daily_records[daily_records['Total'] != 0]
    daily_records.reset_index(inplace=True)
    daily_records.columns = ['Check In', 'Nights', 'Guest', 'Total', 'Number_of_Bookings']
    daily_records.set_index('Check In', inplace=True)
    daily_records = daily_records.resample('W').sum()
    daily_records.reset_index(inplace=True)
    daily_records.set_index("Check In", inplace=True)
    train_size = int(0.8 * len(daily_records))
    train_data = daily_records.iloc[:train_size]
    test_data = daily_records.iloc[train_size:]
    
    bookings = [BookingData(check_in=row['Check In'], nights=row['Nights'], guest=row['Guest'], total=row['Total'], number_of_bookings=row['Number_of_Bookings']) for index, row in df.iterrows()]
    # Bulk create to insert new records, ignoring conflicts to avoid duplicates
    BookingData.objects.bulk_create(bookings, ignore_conflicts=True)

    actual_revenue = df.groupby('Check In')['Total'].sum().reset_index()
    # Ensure 'Check In' column is in the right format
    actual_revenue['Check In'] = pd.to_datetime(actual_revenue['Check In'])
    actual_revenue.set_index('Check In', inplace=True)  # Set 'Check In' as the index
    actual_revenue = actual_revenue.resample('W').sum()
    # Save to CSV in a similar format to forecasts.csv
    actual_csv_path = os.path.join(settings.BASE_DIR, 'actual.csv')
    actual_revenue.to_csv(actual_csv_path, index=True)

    actual_bookings = df.groupby('Check In')['Number_of_Bookings'].sum().reset_index()
    # Ensure 'Check In' column is in the right format
    actual_bookings['Check In'] = pd.to_datetime(actual_bookings['Check In'])
    actual_bookings.set_index('Check In', inplace=True)  # Set 'Check In' as the index
    actual_bookings = actual_bookings.resample('W').sum()
    # Save to CSV in a similar format to forecasts.csv
    actual_bookings_csv_path = os.path.join(settings.BASE_DIR, 'actual_bookings.csv')
    actual_bookings.to_csv(actual_bookings_csv_path, index=True)
    

    with transaction.atomic():
        if not OptimizationParameters.objects.exists():
            OptimizationParameters.objects.create(p=1, d=0, q=1, seasonal_p=2, seasonal_d=1, seasonal_q=0, s=12)
    
    with transaction.atomic():
        if not ProphetOptimizationParameters.objects.exists():
            ProphetOptimizationParameters.objects.get_or_create(defaults={'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 15, 'yearly_seasonality': True, 'rmse': 1400})

    latest_params = OptimizationParameters.objects.latest('last_updated')
    order = (latest_params.p, latest_params.d, latest_params.q)
    seasonal_order = (latest_params.seasonal_p, latest_params.seasonal_d, latest_params.seasonal_q, latest_params.s)

    model = SARIMAX(train_data['Total'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()

    start_idx = len(train_data)
    end_idx = len(train_data) + len(test_data) - 1
    forecast1 = model_fit.predict(start=start_idx, end=end_idx, dynamic=False)
    forecast1.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts.csv'), index=False)

    train_data = train_data.copy()  # Create a copy to avoid modifying the original
    train_data['ds'] = train_data.index  # Use the index which contains Check In dates
    train_data['y'] = train_data['Total']

    # Ensure timezone-naive datetime
    train_data['ds'] = train_data['ds'].dt.tz_localize(None)

    # Sort by date
    train_data.sort_values(by='ds', inplace=True)
    # Fetch the latest Prophet optimization parameters
    latest_prophet_params = ProphetOptimizationParameters.objects.latest('last_updated')
    changepoint_prior_scale = latest_prophet_params.changepoint_prior_scale
    seasonality_prior_scale = latest_prophet_params.seasonality_prior_scale
    yearly_seasonality = latest_prophet_params.yearly_seasonality #True

    # Initialize and fit the Prophet model
    model_prophet = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality= yearly_seasonality
    )
    model_prophet.fit(train_data)

    # Forecast with the Prophet model
    future_prophet = model_prophet.make_future_dataframe(periods=len(test_data), freq='W')
    forecast_prophet = model_prophet.predict(future_prophet)

    # Save the Prophet forecast to CSV
    forecast_prophet[['ds', 'yhat']].rename(columns={'ds': 'Check In', 'yhat': 'Forecasted_Total'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_prophet.csv'), index=False)

    # Forecast for 54 weeks after January 2024 (weekly)
    future_forecast_weekly = model_fit.forecast(steps=54)
    future_forecast_monthly = future_forecast_weekly.resample('M').sum()
    future_forecast_weekly.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_weekly.csv'), index=False)
    future_forecast_monthly.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_monthly.csv'), index=False)


    weekly_prophet = model_prophet.make_future_dataframe(periods=84+15+50, include_history=False, freq='W')
    forecast_prophet_weekly = model_prophet.predict(weekly_prophet)
    forecast_prophet_weekly = forecast_prophet_weekly.tail(67)
    forecast_prophet_weekly = forecast_prophet_weekly[
    (forecast_prophet_weekly['ds'] >= '2024-01-01') & (forecast_prophet_weekly['ds'] <= '2025-01-01')
    ].copy()
    forecast_prophet_weekly[['ds', 'yhat']].rename(columns={'ds': 'Check In', 'yhat': 'Forecasted_Total'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_prophet_weekly.csv'), index=False)
    
    forecast_prophet_weekly.set_index("ds", inplace=True)
    forecast_prophet_monthly = forecast_prophet_weekly['yhat'].resample('M').sum().reset_index()

    forecast_prophet_monthly[['ds', 'yhat']].rename(columns={'ds': 'Check In', 'yhat': 'Forecasted_Total'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_prophet_monthly.csv'), index=False)

    
    future_forecast_weekly2 = model_fit.forecast(steps=52 * 6)
    future_forecast_monthly2 = future_forecast_weekly2.resample('M').sum()
    future_forecast_yearly = future_forecast_monthly2.resample('Y').sum()
    future_forecast_yearly.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_yearly.csv'), index=False)

    weekly_prophet2 = model_prophet.make_future_dataframe(periods=84+(52*6), include_history=False, freq='W')
    forecast_prophet_weekly2 = model_prophet.predict(weekly_prophet2)
    # Remove the first 81 rows
    forecast_prophet_weekly2 = forecast_prophet_weekly2.iloc[81:]
    forecast_prophet_weekly2.set_index("ds", inplace=True)
    forecast_prophet_monthly2 = forecast_prophet_weekly2['yhat'].resample('M').sum().reset_index()
    # Resample the monthly aggregated DataFrame to yearly frequency and sum the values
    forecast_prophet_monthly2.set_index("ds", inplace=True)
    forecast_prophet_yearly = forecast_prophet_monthly2['yhat'].resample('Y').sum().reset_index()
    forecast_prophet_yearly[['ds', 'yhat']].rename(columns={'ds': 'Check In', 'yhat': 'Forecasted_Total'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_prophet_yearly.csv'), index=False)

    train_data['Check In'] = train_data['ds'] # Convert to timezone-naive
    train_data['Total'] = train_data['y']
    train_data['Check In'] = pd.to_datetime(train_data['Check In'])
    # Set 'Check In' as the index
    train_data.set_index('Check In', inplace=True)

    model_bookings = SARIMAX(train_data['Number_of_Bookings'], order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit_bookings = model_bookings.fit()

    forecast_bookings = model_fit_bookings.forecast(steps=52)
    forecast_bookings.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_bookings.csv'), index=False)


    # Forecast for 54 weeks after January 2024 (weekly)
    future_forecast_weekly_bookings = model_fit_bookings.forecast(steps=52)
    future_forecast_monthly_bookings = future_forecast_weekly_bookings.resample('M').sum()
    future_forecast_weekly_bookings.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_weekly_bookings.csv'), index=False)
    future_forecast_monthly_bookings.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_monthly_bookings.csv'), index=False)

    future_forecast_weekly2_bookings = model_fit_bookings.forecast(steps=52 * 5)
    future_forecast_monthly2_bookings = future_forecast_weekly2_bookings.resample('M').sum()
    future_forecast_yearly_bookings = future_forecast_monthly2_bookings.resample('Y').sum()
    future_forecast_yearly_bookings.to_frame(name='Forecasted_Total').reset_index().rename(columns={'index': 'Check In'}).to_csv(os.path.join(settings.BASE_DIR, 'forecasts_yearly_bookings.csv'), index=False)

    optimize_prophet_parameters.delay()
    optimize_parameters.delay()
    
  


def dashboard(request):
 
      # Load SARIMAX forecasts
    forecast_sarimax_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'forecasts.csv'))
    forecast_sarimax_df['Check In'] = pd.to_datetime(forecast_sarimax_df['Check In'])

    # Load Prophet forecasts
    forecast_prophet_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'forecasts_prophet.csv'))
    forecast_prophet_df['Check In'] = pd.to_datetime(forecast_prophet_df['Check In'])

    # Assuming actual_df is loaded similarly for test data period
    actual_df_path = os.path.join(settings.BASE_DIR, 'actual.csv')
    actual_df = pd.read_csv(actual_df_path)
    actual_df['Check In'] = pd.to_datetime(actual_df['Check In'])

    # Ensure test_data range is defined
    test_data_start_date = '2022-06-12'  
    test_data_end_date = '2023-06-04'  

    # Filter the forecasts and actual data for the test data period
    forecast_sarimax_df_1 = forecast_sarimax_df[(forecast_sarimax_df['Check In'] >= test_data_start_date) & (forecast_sarimax_df['Check In'] <= test_data_end_date)]
    forecast_prophet_df_1 = forecast_prophet_df[(forecast_prophet_df['Check In'] >= test_data_start_date) & (forecast_prophet_df['Check In'] <= test_data_end_date)]
    actual_df = actual_df[(actual_df['Check In'] >= test_data_start_date) & (actual_df['Check In'] <= test_data_end_date)]

    # Print to check if there are data points in the test period
    print(forecast_sarimax_df_1.head())
    print(forecast_prophet_df_1.head())
    print(actual_df.head())
    # Ensure combined forecast calculation is performed correctly
    # Only perform calculation if both forecasts are not empty
    if not forecast_sarimax_df_1.empty and not forecast_prophet_df_1.empty:
        combined_forecast = (forecast_sarimax_df_1['Forecasted_Total'].reset_index(drop=True) + forecast_prophet_df_1['Forecasted_Total'].reset_index(drop=True)) / 2
        # Create a new DataFrame for combined forecast with 'Check In' and 'Forecasted_Total' columns
        combined_forecast_df = pd.DataFrame({
            'Check In': forecast_sarimax_df_1['Check In'].reset_index(drop=True),
            'Forecasted_Total': combined_forecast
        })
    else:
        print("One of the forecasts is empty after filtering for the test period.")

   
    # Plot configuration
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=forecast_sarimax_df_1['Check In'], y=actual_df['Total'], mode='lines', name='Actual Revenue'))
    fig.add_trace(go.Scatter(x=forecast_sarimax_df_1['Check In'], y=forecast_sarimax_df_1['Forecasted_Total'], mode='lines', name='SARIMAX Forecast'))
    fig.add_trace(go.Scatter(x=forecast_sarimax_df_1['Check In'], y=forecast_prophet_df_1['Forecasted_Total'], mode='lines', name='Prophet Forecast'))
    #fig.add_trace(go.Scatter(x=forecast_sarimax_df_1['Check In'], y=combined_forecast, mode='lines', name='Combined Forecast'))
    if not combined_forecast_df.empty:
        fig.add_trace(go.Scatter(x=combined_forecast_df['Check In'], y=combined_forecast_df['Forecasted_Total'], mode='lines', name='Combined Forecast', line=dict(color='purple')))
    else:
        print("Combined forecast DataFrame is empty.")

    # Update plot layout
    fig.update_layout(title="Forecast vs Actual Comparison for Test Period", xaxis_title="Date", yaxis_title="Total (£)")

    plot_div = plot(fig, output_type='div')


    weekly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_weekly.csv')
    weekly_forecast_sarimax_df = pd.read_csv(weekly_forecast_sarimax_df_path)


    weekly_prophet_df_path = os.path.join(settings.BASE_DIR, 'forecasts_prophet_weekly.csv')
    weekly_prophet_df = pd.read_csv(weekly_prophet_df_path)

    if 'Check In' not in weekly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {forecast_sarimax_df.columns.tolist()}")

    weekly_prophet_df['Check In'] = pd.to_datetime(weekly_prophet_df['Check In']).dt.tz_localize(None)

   
    future_start_date = '2024-01-01' 
    
    weekly_forecast_sarimax_df = pd.DataFrame({
        'Check In': pd.date_range(start=future_start_date, periods=len(weekly_forecast_sarimax_df), freq='W'),
        'SARIMAX_Forecast': weekly_forecast_sarimax_df['Forecasted_Total']  
    })

    future_end_date_weekly = pd.date_range(start=future_start_date, periods=54, freq='W')
   
    print(weekly_forecast_sarimax_df.columns)
    print(weekly_prophet_df.columns)
    print(weekly_forecast_sarimax_df.head())
    print(weekly_prophet_df.head())


    combined_forecast_df = pd.merge(weekly_forecast_sarimax_df, weekly_prophet_df, on='Check In', how='inner')

    if not combined_forecast_df.empty:

        combined_forecast_df['Hybrid_Forecast'] = combined_forecast_df[['SARIMAX_Forecast', 'Forecasted_Total']].mean(axis=1)

        print("SARIMAX Forecast Dates:")
        print(weekly_forecast_sarimax_df['Check In'])

        print("Prophet Forecast Dates:")
        print(weekly_prophet_df['Check In'])

        # Check the length of both DataFrames to ensure they are expected
        print(f"Length of SARIMAX DataFrame: {len(weekly_forecast_sarimax_df)}")
        print(f"Length of Prophet DataFrame: {len(weekly_prophet_df)}")

        trace_sarimax = go.Scatter(x=combined_forecast_df['Check In'], y=combined_forecast_df['SARIMAX_Forecast'], mode='lines', name='SARIMAX Forecast')
        trace_prophet = go.Scatter(x=combined_forecast_df['Check In'], y=combined_forecast_df['Forecasted_Total'], mode='lines', name='Prophet Forecast')
        trace_hybrid = go.Scatter(x=combined_forecast_df['Check In'], y=combined_forecast_df['Hybrid_Forecast'], mode='lines', name='Hybrid Forecast')

        layout = go.Layout(title='Weekly Forecast Comparison: SARIMAX vs. Prophet vs. Hybrid (2024-01-01 to 2025-01-01)', xaxis=dict(title='Date'), yaxis=dict(title='Forecast Value'))

        fig2 = go.Figure(data=[trace_sarimax, trace_prophet, trace_hybrid], layout=layout)

        fig2.update_layout(title='Forecasted weekly Revenue', xaxis_title='Date', yaxis_title='Revenue (£)')

        plot_div_weekly = plot(fig2, output_type='div')

    else:
        print("The merged forecast DataFrame is empty. Check the alignment of dates.")
        plot_div_weekly = None
    

    # Load monthly SARIMAX forecasts
    monthly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_monthly.csv')
    monthly_forecast_sarimax_df = pd.read_csv(monthly_forecast_sarimax_df_path)

    # Ensure 'Check In' column is present
    if 'Check In' not in monthly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {monthly_forecast_sarimax_df.columns.tolist()}")

    # Convert 'Check In' to datetime
   
    monthly_forecast_sarimax_df = pd.DataFrame({
        'Check In': pd.date_range(start=future_start_date, periods=len(monthly_forecast_sarimax_df), freq='M'),
        'SARIMAX_Forecast': monthly_forecast_sarimax_df['Forecasted_Total'] 
    })
    # Load monthly Prophet forecasts
    monthly_prophet_df_path = os.path.join(settings.BASE_DIR, 'forecasts_prophet_monthly.csv')
    monthly_prophet_df = pd.read_csv(monthly_prophet_df_path)

    # Convert 'Check In' to datetime and standardize format
    monthly_prophet_df['Check In'] = pd.to_datetime(monthly_prophet_df['Check In']).dt.tz_localize(None)
    
    # Merge SARIMAX and Prophet forecasts

    print(monthly_forecast_sarimax_df.columns)
    print(monthly_prophet_df.columns)
    print(monthly_forecast_sarimax_df.head())
    print(monthly_prophet_df.head())

    combined_monthly_forecast_df = pd.merge(monthly_forecast_sarimax_df, monthly_prophet_df, on='Check In', how='inner')

    if not combined_monthly_forecast_df.empty:
        # Calculate hybrid forecast
        combined_monthly_forecast_df['Hybrid_Forecast'] = combined_monthly_forecast_df[['SARIMAX_Forecast', 'Forecasted_Total']].mean(axis=1)

        print("SARIMAX Forecast Dates:")
        print(monthly_forecast_sarimax_df['Check In'])

        print("Prophet Forecast Dates:")
        print(monthly_prophet_df['Check In'])
        # Check the length of both DataFrames to ensure they are expected
        print(f"Length of SARIMAX DataFrame: {len(monthly_forecast_sarimax_df)}")
        print(f"Length of Prophet DataFrame: {len(monthly_prophet_df)}")

        # Visualization
        trace_sarimax = go.Scatter(x=combined_monthly_forecast_df['Check In'], y=combined_monthly_forecast_df['SARIMAX_Forecast'], mode='lines', name='SARIMAX Forecast')
        trace_prophet = go.Scatter(x=combined_monthly_forecast_df['Check In'], y=combined_monthly_forecast_df['Forecasted_Total'], mode='lines', name='Prophet Forecast')
        trace_hybrid = go.Scatter(x=combined_monthly_forecast_df['Check In'], y=combined_monthly_forecast_df['Hybrid_Forecast'], mode='lines', name='Hybrid Forecast')

        layout = go.Layout(title='Monthly Forecast Comparison: SARIMAX vs. Prophet vs. Hybrid', xaxis=dict(title='Date'), yaxis=dict(title='Forecasted Revenue'))

        fig = go.Figure(data=[trace_sarimax, trace_prophet, trace_hybrid], layout=layout)
        fig.update_layout(title='Forecasted Monthly Revenue', xaxis_title='Date', yaxis_title='Revenue (£)')

        plot_div_monthly = plot(fig, output_type='div')
    else:
        print("The merged forecast DataFrame is empty. Check the alignment of dates.")
        plot_div_monthly = None



    yearly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_yearly.csv')
    yearly_forecast_sarimax_df = pd.read_csv(yearly_forecast_sarimax_df_path)

    # Ensure 'Check In' column is present
    if 'Check In' not in yearly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {yearly_forecast_sarimax_df.columns.tolist()}")

    # Convert 'Check In' to datetime
    yearly_forecast_sarimax_df = pd.DataFrame({
        'Check In': pd.date_range(start=future_start_date, periods=len(yearly_forecast_sarimax_df), freq='Y'),
        'SARIMAX_Forecast': yearly_forecast_sarimax_df['Forecasted_Total'] 
    })

    yearly_forecast_sarimax_df = yearly_forecast_sarimax_df[:-1]
    # Load monthly Prophet forecasts
    yearly_prophet_df_path = os.path.join(settings.BASE_DIR, 'forecasts_prophet_yearly.csv')
    yearly_prophet_df = pd.read_csv(yearly_prophet_df_path)

    # Convert 'Check In' to datetime and standardize format
    yearly_prophet_df = yearly_prophet_df[:-1]
    yearly_prophet_df['Check In'] = pd.to_datetime(yearly_prophet_df['Check In']).dt.tz_localize(None)
    
    # Merge SARIMAX and Prophet forecasts

    print(yearly_forecast_sarimax_df.columns)
    print(yearly_prophet_df.columns)
    print(yearly_forecast_sarimax_df.head())
    print(yearly_prophet_df.head())

    combined_yearly_forecast_df = pd.merge(yearly_forecast_sarimax_df, yearly_prophet_df, on='Check In', how='inner')

    if not combined_yearly_forecast_df.empty:
        # Calculate hybrid forecast
        combined_yearly_forecast_df['Hybrid_Forecast'] = combined_yearly_forecast_df[['SARIMAX_Forecast', 'Forecasted_Total']].mean(axis=1)

        print("SARIMAX Forecast Dates:")
        print(yearly_forecast_sarimax_df['Check In'])

        print("Prophet Forecast Dates:")
        print(yearly_prophet_df['Check In'])
        # Check the length of both DataFrames to ensure they are expected
        print(f"Length of SARIMAX DataFrame: {len(yearly_forecast_sarimax_df)}")
        print(f"Length of Prophet DataFrame: {len(yearly_prophet_df)}")

        # Visualization
        trace_sarimax = go.Scatter(x=combined_yearly_forecast_df['Check In'], y=combined_yearly_forecast_df['SARIMAX_Forecast'], mode='lines', name='SARIMAX Forecast')
        trace_prophet = go.Scatter(x=combined_yearly_forecast_df['Check In'], y=combined_yearly_forecast_df['Forecasted_Total'], mode='lines', name='Prophet Forecast')
        trace_hybrid = go.Scatter(x=combined_yearly_forecast_df['Check In'], y=combined_yearly_forecast_df['Hybrid_Forecast'], mode='lines', name='Hybrid Forecast')

        layout = go.Layout(title='Yearly Forecast Comparison: SARIMAX vs. Prophet vs. Hybrid', xaxis=dict(title='Date'), yaxis=dict(title='Forecasted Revenue'))

        fig = go.Figure(data=[trace_sarimax, trace_prophet, trace_hybrid], layout=layout)
        fig.update_layout(title='Forecasted Yearly Revenue', xaxis_title='Date', yaxis_title='Revenue (£)')

        plot_div_yearly = plot(fig, output_type='div')
    else:
        print("The merged forecast DataFrame is empty. Check the alignment of dates.")
        plot_div_yearly = None
        
   
    forecast_sarimax_df_bookings_path = os.path.join(settings.BASE_DIR, 'forecasts_bookings.csv')
    forecast_sarimax_df_bookings = pd.read_csv(forecast_sarimax_df_bookings_path)

    actual_bookings_df_path = os.path.join(settings.BASE_DIR, 'actual_bookings.csv')
    actual_bookings_df = pd.read_csv(actual_bookings_df_path)
    actual_bookings_df['Check In'] = pd.to_datetime(actual_bookings_df['Check In'])

    # Ensure test_data range is defined
    test_data_start_date = '2022-06-12'  
    test_data_end_date = '2023-06-04'    

    # Filter the forecasts and actual data for the test data period
    forecast_sarimax_df_bookings = forecast_sarimax_df_bookings[(forecast_sarimax_df_bookings['Check In'] >= test_data_start_date) & (forecast_sarimax_df_bookings['Check In'] <= test_data_end_date)]
    actual_bookings_df = actual_bookings_df[(actual_bookings_df['Check In'] >= test_data_start_date) & (actual_bookings_df['Check In'] <= test_data_end_date)]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=forecast_sarimax_df_bookings['Check In'], y=actual_bookings_df['Number_of_Bookings'], mode='lines', name='Actual Bookings'))
    fig.add_trace(go.Scatter(x=forecast_sarimax_df_bookings['Check In'], y=forecast_sarimax_df_bookings['Forecasted_Total'], mode='lines', name='SARIMAX Prediction'))
    fig.update_layout(title="Forecast vs Actual Comparison for Test Period", xaxis_title="Date", yaxis_title="Total Bookings")

    plot_div_bookings = plot(fig, output_type='div')

    
    weekly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_weekly_bookings.csv')
    weekly_forecast_sarimax_df = pd.read_csv(weekly_forecast_sarimax_df_path)

    if 'Check In' not in weekly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {forecast_sarimax_df.columns.tolist()}")

   
    weekly_forecast_sarimax_df['Check In'] = pd.to_datetime(weekly_forecast_sarimax_df['Check In'])
    weekly_forecast_sarimax_df = weekly_forecast_sarimax_df['Forecasted_Total']  # Assuming the column name is 'Forecast'
    future_start_date = '2024-01-01' #change to final date in csv
    future_end_date_weekly = pd.date_range(start=future_start_date, periods=54, freq='W')
    weekly_forecast_sarimax_df = weekly_forecast_sarimax_df.round()
 
  
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(Scatter(x=future_end_date_weekly, y=weekly_forecast_sarimax_df, mode='lines', name='Weekly Forecast'))
    fig2.update_layout(title='Forecasted Weekly Bookings', xaxis_title='Date', yaxis_title='Total Bookings')
    plot_div_weekly_bookings = plot(fig2, output_type='div')

    monthly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_monthly_bookings.csv')
    monthly_forecast_sarimax_df = pd.read_csv(monthly_forecast_sarimax_df_path)

    if 'Check In' not in monthly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {forecast_sarimax_df.columns.tolist()}")

   
    monthly_forecast_sarimax_df['Check In'] = pd.to_datetime(monthly_forecast_sarimax_df['Check In'])
    monthly_forecast_sarimax_df = monthly_forecast_sarimax_df['Forecasted_Total']  
    future_start_date = '2024-01-01'
    future_end_date_monthly = pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS')
    monthly_forecast_sarimax_df = monthly_forecast_sarimax_df.round()
    
    fig3 = Figure()
    # Example plot for future forecasts
    fig3.add_trace(Scatter(x=future_end_date_monthly, y=monthly_forecast_sarimax_df, mode='lines', name='monthly Forecast'))
    fig3.update_layout(title='Forecasted monthly Bookings', xaxis_title='Date', yaxis_title='Total Bookings')
    plot_div_monthly_bookings = plot(fig3, output_type='div')

    yearly_forecast_sarimax_df_path = os.path.join(settings.BASE_DIR, 'forecasts_yearly_bookings.csv')
    yearly_forecast_sarimax_df = pd.read_csv(yearly_forecast_sarimax_df_path)
    yearly_forecast_sarimax_df = yearly_forecast_sarimax_df.round()

    if 'Check In' not in yearly_forecast_sarimax_df.columns:
        raise ValueError(f"'Check In' column is missing in the loaded DataFrame. Available columns: {forecast_sarimax_df.columns.tolist()}")

   
    yearly_forecast_sarimax_df['Check In'] = pd.to_datetime(yearly_forecast_sarimax_df['Check In'])
    yearly_forecast_sarimax_df = yearly_forecast_sarimax_df['Forecasted_Total']  # Assuming the column name is 'Forecast'
    future_start_date = '2024-01-01' #change to final date in csv
    future_end_date_yearly = pd.date_range(start='2024-01-01', periods=5, freq='Y')
    
    fig4 = Figure()
    # Example plot for future forecasts
    fig4.add_trace(Scatter(x=future_end_date_yearly, y=yearly_forecast_sarimax_df, mode='lines', name='yearly Forecast'))
    fig4.update_layout(title='Forecasted yearly Bookings', xaxis_title='Date', yaxis_title='Total Bookings')
    plot_div_yearly_bookings = plot(fig4, output_type='div')

    kpis = request.session.get('KPIs', {})

    formatted_kpis = {
        'Daily RevPAR': f"£{kpis['Daily RevPAR']:.2f}",
        'ADR': f"£{kpis['ADR']:.2f}",
        'Cumulative Annual RevPAR': f"£{kpis['Cumulative Annual RevPAR']:.2f}",
        'ALOS': f"{kpis['ALOS']:.2f} nights",
        'Occupancy Rate': f"{kpis['Occupancy Rate']*100:.2f}%",  # Occupancy rate as percentage
    }

   # return render(request, 'forecasts/dashboard.html', {'plot_div': plot_div})
    return render(request, 'forecasts/dashboard.html', {
        'plot_div': plot_div,
        'plot_div_weekly': plot_div_weekly,
        'plot_div_monthly': plot_div_monthly,
        'plot_div_yearly': plot_div_yearly,
        'plot_div_bookings':plot_div_bookings,
        'plot_div_weekly_bookings':plot_div_weekly_bookings,
        'plot_div_monthly_bookings':plot_div_monthly_bookings,
        'plot_div_yearly_bookings':plot_div_yearly_bookings,
        'kpis': formatted_kpis,
    })

