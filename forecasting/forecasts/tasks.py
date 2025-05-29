from __future__ import absolute_import, unicode_literals

# Standard library imports
import logging
import itertools
import warnings

# Third-party library imports
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Django imports
from django.db import transaction
from django.db.models import Sum
from django.db.models.functions import ExtractWeek, ExtractYear, ExtractMonth, ExtractDay

# Celery imports
from celery import shared_task

# Local application imports
from forecasts.models import BookingData, OptimizationParameters, ProphetOptimizationParameters

warnings.filterwarnings("ignore")


logger = logging.getLogger(__name__)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def ts_cross_val_rmse(y, order, seasonal_order, cv=3):
    n_records = len(y)
    fold_size = n_records // cv
    errors = []

    for i in range(cv):
        start_test = i * fold_size
        if i == cv - 1:
            end_test = n_records
        else:
            end_test = (i + 1) * fold_size

        train = y[:start_test]
        test = y[start_test:end_test]

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))

        rmse = np.sqrt(mean_squared_error(test, forecast))
        errors.append(rmse)

    return np.mean(errors)

    
@shared_task
def optimize_parameters():
    # Fetch the aggregated data from the BookingData model
    logger.info("Starting optimization of parameters.")
    try:

        data = BookingData.objects.annotate(
            week=ExtractWeek('check_in'),
            year=ExtractYear('check_in')
        ).values('year', 'week').annotate(total=Sum('total')).order_by('year', 'week')

        # Convert to DataFrame for processing
        df = pd.DataFrame(list(data))
        df['date'] = pd.to_datetime(df['year'].astype(str), format='%Y') + pd.to_timedelta(df['week'].mul(7).astype(str) + ' days')

        # 'total' is the column we want to forecast
        train_data = df.set_index('date')['total']

        # Optimization logic from..
        # Define the parameter space
        p = [0, 1]  # Possible p values
        d = [0, 1]  # Possible d values
        q = [0, 1, 3]  # Possible q values
        seasonal_p = [0, 2]  #removed 3
        seasonal_d = [0, 1]
        seasonal_q = [0, 1]
        s = 12
        pdq = list(itertools.product(p, d, q))
        
        seasonal_pdq = []
        for sp in seasonal_p:
            for sd in seasonal_d:
                for sq in seasonal_q:
                    if (sd == 0 and sq == 0) or (sd == 0 and sq == 1):  # Exclude (x, 0, 0, 12) and (x, 0, 1, 12)
                        continue
                    if sd == 0 and sq != 0:  # Allow (x, 0, x!=0, 12)
                        seasonal_pdq.append((sp, sd, sq, s))
                    elif sd != 0 and sq == 0:  # Allow (x, x!=0, 0, 12)
                        seasonal_pdq.append((sp, sd, sq, s))
        
        best_score, best_cfg = float("inf"), None
        for param in pdq:
            if param == (0, 1, 0) and any(sd == 0 for (_, sd, _, _) in seasonal_pdq):
                continue
            for param_seasonal in seasonal_pdq:
                try:
                    logger.info("Optimization completed successfully.")
                    rmse = ts_cross_val_rmse(train_data, order=param, seasonal_order=param_seasonal, cv=3)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, (param, param_seasonal)
                    print('SARIMAX{}x{} - RMSE:{}'.format(param, param_seasonal, rmse))
                    logger.info('SARIMAX{}x{} - RMSE:{}'.format(param, param_seasonal, rmse))

                except Exception as e:
                    logger.error(f"Optimization failed with an exception: {e}", exc_info=True)
                    continue

        print(f"Saving optimal parameters to database: {best_cfg}")
        logger.info(f"Saving optimal parameters to database: {best_cfg}")

        # Saving the best parameters to the database
        OptimizationParameters.objects.create(
            p=best_cfg[0][0],
            d=best_cfg[0][1],
            q=best_cfg[0][2],
            seasonal_p=best_cfg[1][0],
            seasonal_d=best_cfg[1][1],
            seasonal_q=best_cfg[1][2],
            s=best_cfg[1][3]
        )
    except Exception as e:
        logger.error(f"Failed to complete optimize_parameters: {e}", exc_info=True)
        raise e
    





def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

@shared_task
def optimize_prophet_parameters():
    logger.info("Starting Prophet parameters optimization task.")
    print("Starting Prophet parameters optimization task.")
    
    data = BookingData.objects.annotate(
        year=ExtractYear('check_in'),
        month=ExtractMonth('check_in'),
        day=ExtractDay('check_in')
    ).values('year', 'month', 'day', 'total').filter(total__gt=0).order_by('year', 'month', 'day')
    
    df = pd.DataFrame(list(data))
    df['ds'] = pd.to_datetime(df[['year', 'month', 'day']])
    prophet_data = df[['ds', 'total']].rename(columns={'total': 'y'})

    # Now, 'prophet_data' is ready to be used with Prophet
    print(prophet_data.head())

    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 1.0], #removed 0.5
        'seasonality_prior_scale': [12.0, 15.0], #0.1, 1.0, 10.0, removed - low seasonality when data like this shows high
        'yearly_seasonality': [True, False]
    }
    
    best_rmse = float('inf')
    best_params = {}
    
    # Train-test split
    train_size = int(0.8 * len(prophet_data))
    train_df = prophet_data.iloc[:train_size]
    test_df = prophet_data.iloc[train_size:]
    
    for cps in param_grid['changepoint_prior_scale']:
        for sps in param_grid['seasonality_prior_scale']:
            for ys in param_grid['yearly_seasonality']:
                model = Prophet(daily_seasonality=False, changepoint_prior_scale=cps, seasonality_prior_scale=sps, yearly_seasonality=ys)
                model.fit(train_df)
                
                future = model.make_future_dataframe(periods=len(test_df), freq='D')  # Ensure frequency matches Colab
                forecast = model.predict(future)
                
                # Calculate RMSE for the test set
                actual = test_df['y'].values
                predicted = forecast['yhat'][-len(test_df):].values
                rmse = calculate_rmse(actual, predicted)
                current_params =  {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps, 'yearly_seasonality': ys}
                logger.info(f"Current parameters: {current_params}, RMSE: {rmse}")
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'changepoint_prior_scale': cps, 'seasonality_prior_scale': sps, 'yearly_seasonality': ys}

    try:
        with transaction.atomic():
            obj, created = ProphetOptimizationParameters.objects.update_or_create(
                defaults={
                    'changepoint_prior_scale': best_params['changepoint_prior_scale'],
                    'seasonality_prior_scale': best_params['seasonality_prior_scale'],
                    'yearly_seasonality': best_params['yearly_seasonality'],
                    'rmse': best_rmse
                }
            )
        logger.info(f"{'Created' if created else 'Updated'} ProphetOptimizationParameters: {obj}")
        print(f"{'Created' if created else 'Updated'} ProphetOptimizationParameters: {obj}")
    except Exception as e:
        logger.info(f"Error updating/creating ProphetOptimizationParameters within a transaction: {e}")
        print(f"An exception occurred: {e}")

    
    logger.info(f"Optimization completed. Best parameters: {best_params}, Best RMSE: {best_rmse}")
    print(f"Optimization completed. Best parameters: {best_params}, Best RMSE: {best_rmse}")
    # Save or update the model parameters in the database
