
# Forecasting App README

This is a Django-based forecasting application that allows users to upload CSV files generated via Integrum and view analytics data on the dashboard.

USE THE CSV FILE TITLED Adfan_Integrum_Insights.csv TO TEST THIS SYSTEM

## Prerequisites

- Python 3.x
- pip (Python package installer)
- Virtualenv (optional but recommended for creating virtual environments)
- PostgreSQL database
- Redis server

## Setup Instructions

### 1. Virtual Environment

It is recommended to set up a Python virtual environment to isolate project dependencies. Run the following command to create a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Unix or MacOS
env\Scripts\activate  # On Windows
```

### 2. Install Dependencies

Install the required Python packages using pip:

```bash
pip install django
pip install pandas
pip install numpy
pip install celery
pip install redis
pip install psycopg2  # Or psycopg2-binary
pip install prophet
pip install statsmodels
pip install sklearn
pip install plotly
```

### 3. Database and Redis Server

Ensure that you have a PostgreSQL database set up and accessible. 
You will also need to have a Redis server running. For setting up Redis on WSL (Windows Subsystem for Linux) as I did, you can follow these steps:

1. Open your WSL terminal.
2. Update your package lists: `sudo apt update`
3. Install Redis Server: `sudo apt install redis-server`
4. Start Redis Server: `sudo service redis-server start`
5. Check if Redis is running using the Redis CLI:

```bash
redis-cli ping
```

If Redis is running, it will return `PONG`.

### 4. Database Configuration

Update the database settings in `forecasting/settings.py` with your PostgreSQL database details.

```python
# forecasting/settings.py
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'your_db_host',
        'PORT': 'your_db_port',
    }
}
```

### 5. Django Migrations

Navigate to the directory that contains `forecasting` (i.e everything that has been uploaded to this git) and run the following commands to set up your database schema:

```bash 
cd path/to/forecasting
python manage.py makemigrations
python manage.py migrate
```

### 6. Run Celery Worker

Open a new terminal window and run the following command to start the Celery worker:

```bash
celery -A forecasting worker -l info --pool=solo
```

Ensure the Celery worker is running and connected to the Redis broker.

### 7. Run Django Development Server

In another terminal, start the Django development server:

```bash
python manage.py runserver
```

### 8. Access the Application

With the server running, open a web browser and navigate to [http://127.0.0.1:8000/forecasts/upload/](http://127.0.0.1:8000/forecasts/upload/) to upload the CSV file provided with this README.

### 9. Viewing the Dashboard

After uploading the CSV file, you should be redirected to the dashboard where you can view the forecasting results.

---

Make sure all services (Django, Celery, Redis) are running before trying to upload the CSV file for forecasting.

For any issues, please refer to the documentation of each technology stack or open an issue on this project's repository.
