import os
import pytest
import pandas as pd
import numpy as np
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.conf import settings
from unittest.mock import patch, MagicMock
from ..views import handle_uploaded_file

class TestHandleUploadedFile(TestCase):
    def setUp(self):
        # Create sample CSV data
        self.csv_data = (
            "Check In,Check Out,Room,Nights,Guest,Total,Number_of_Bookings\n"
            "01/01/2023 12:00,02/01/2023 10:00,101,1,John Doe,100.00,1\n"
            "02/01/2023 14:00,04/01/2023 10:00,102,2,Jane Smith,200.00,1\n"
        )
        self.uploaded_file = SimpleUploadedFile(
            "test.csv",
            self.csv_data.encode('utf-8'),
            content_type='text/csv'
        )

        # Mock request object
        self.request = MagicMock()
        self.request.session = {}

    def tearDown(self):
        # Clean up generated files after tests
        test_files = [
            'actual.csv', 'forecasts.csv', 'forecasts_prophet.csv',
            'forecasts_weekly.csv', 'forecasts_monthly.csv',
            'forecasts_prophet_weekly.csv', 'forecasts_prophet_monthly.csv'
        ]
        for file in test_files:
            file_path = os.path.join(settings.BASE_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)

    @patch('forecasts.views.OptimizationParameters')
    @patch('forecasts.views.ProphetOptimizationParameters')
    @patch('forecasts.views.BookingData')
    def test_handle_valid_file(self, mock_booking_data, mock_prophet_params, mock_opt_params):
        # Setup mock database objects
        mock_opt_params.objects.exists.return_value = False
        mock_prophet_params.objects.exists.return_value = False
        mock_opt_params.objects.create.return_value = MagicMock()
        mock_prophet_params.objects.get_or_create.return_value = (MagicMock(), True)

        # Execute function
        handle_uploaded_file(self.uploaded_file, self.request)

        # Verify KPIs were calculated and stored in session
        self.assertTrue('KPIs' in self.request.session)
        kpis = self.request.session['KPIs']
        self.assertIn('ADR', kpis)
        self.assertIn('Occupancy Rate', kpis)
        self.assertIn('ALOS', kpis)

    def test_missing_required_columns(self):
        # Create CSV with missing columns
        invalid_csv = "Date,Room,Total\n01/01/2023,101,100.00"
        invalid_file = SimpleUploadedFile(
            "invalid.csv",
            invalid_csv.encode('utf-8'),
            content_type='text/csv'
        )

        with pytest.raises(ValueError) as exc_info:
            handle_uploaded_file(invalid_file, self.request)
        
        assert "Missing required columns" in str(exc_info.value)

    @patch('forecasts.views.SARIMAX')
    @patch('forecasts.views.Prophet')
    def test_forecast_generation(self, mock_prophet, mock_sarimax):
        # Setup mock forecasting models
        mock_model = MagicMock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = pd.Series(np.random.randn(10))
        mock_model.forecast.return_value = pd.Series(np.random.randn(10))
        mock_sarimax.return_value = mock_model
        mock_prophet.return_value = mock_model

        # Execute function
        handle_uploaded_file(self.uploaded_file, self.request)

        # Verify forecast files were created
        forecast_files = [
            'forecasts.csv',
            'forecasts_prophet.csv',
            'forecasts_weekly.csv',
            'forecasts_monthly.csv'
        ]
        for file in forecast_files:
            file_path = os.path.join(settings.BASE_DIR, file)
            self.assertTrue(os.path.exists(file_path))

    def test_data_processing(self):
        # Execute function
        handle_uploaded_file(self.uploaded_file, self.request)

        # Read processed data files
        actual_df = pd.read_csv(os.path.join(settings.BASE_DIR, 'actual.csv'))
        
        # Verify data processing
        self.assertGreater(len(actual_df), 0)
        self.assertIn('Total', actual_df.columns)

    @patch('forecasts.views.optimize_parameters')
    @patch('forecasts.views.optimize_prophet_parameters')
    def test_optimization_tasks(self, mock_prophet_opt, mock_opt):
        # Execute function
        handle_uploaded_file(self.uploaded_file, self.request)

        # Verify optimization tasks were called
        mock_prophet_opt.delay.assert_called_once()
        mock_opt.delay.assert_called_once()
