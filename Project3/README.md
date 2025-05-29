# Forecasting App README

This Django application allows users to upload CSV files, generate forecasts, and view analytics. For deployments, this setup uses Nginx as a reverse proxy directly to Django's development server.

## Prerequisites

- Python 3.x
- pip (Python package installer)
- Virtualenv (recommended)
- PostgreSQL database
- Redis server
- Nginx
- Domain name with DNS configured (optional)

## Development Setup

### 1. Virtual Environment
```bash
python -m venv env
source env/bin/activate  # Unix/MacOS
env\Scripts\activate    # Windows
2. Install Dependencies
bash
pip install django pandas numpy celery redis psycopg2 prophet statsmodels sklearn plotly
3. Database & Redis
Set up PostgreSQL database

Start Redis server:

bash
sudo service redis-server start
redis-cli ping  # Should return "PONG"
4. Configure Settings
Update forecasting/settings.py with your database credentials:

python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'your_db_name',
        'USER': 'your_db_user',
        'PASSWORD': 'your_db_password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
5. Database Migrations
bash
python manage.py makemigrations
python manage.py migrate
6. Run Services
In separate terminals:

bash
# Start Celery worker
celery -A forecasting worker -l info --pool=solo

# Start development server (bind to all interfaces)
python manage.py runserver 0.0.0.0:8000
Access the app at: http://localhost:8000

Production Setup with Nginx & HTTPS (Without Gunicorn)
1. Install and Configure Nginx
bash
sudo apt install nginx
Create /etc/nginx/sites-available/forecasting:

nginx
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;  # Direct to Django dev server
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static/ {
        alias /path/to/your/project/staticfiles/;
        expires 30d;
    }

    location /media/ {
        alias /path/to/your/project/media/;
        expires 30d;
    }
}
Enable the site:

bash
sudo ln -s /etc/nginx/sites-available/forecasting /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
2. Configure HTTPS with Let's Encrypt
bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
Follow prompts to configure HTTPS and certificate renewal.

3. Update Django Settings
In settings.py:

python
# Security settings
ALLOWED_HOSTS = ['yourdomain.com', 'www.yourdomain.com']
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
Collect static files:

bash
python manage.py collectstatic
4. Start Production Services
bash
# Start Django development server (on port 8000)
python manage.py runserver 0.0.0.0:8000

# Start Redis
sudo service redis-server start

# Start Celery
celery -A forecasting worker -l info --pool=solo
Security Considerations
While this setup uses Nginx for HTTPS termination, be aware of these limitations:

Django Development Server Limitations:

Single-threaded (handles one request at a time)

No production-grade security protections

Not optimized for performance

Vulnerable to denial-of-service attacks

Nginx Reverse Proxy Benefits:

SSL/TLS termination for HTTPS encryption

Serves static files directly (bypassing Django)

Hides Django server details from external clients

Basic request filtering

HTTPS Encryption:

Encrypts traffic between client and server

Prevents eavesdropping on sensitive data

Enabled via Let's Encrypt certificates

Automatic certificate renewal

Maintenance
Renew SSL certificates:

bash
sudo certbot renew --dry-run
Check Nginx status:

bash
sudo systemctl status nginx
sudo tail -f /var/log/nginx/error.log
Monitor Django server (restart manually if crashes)

When to Use This Configuration
This setup is appropriate for:

Testing environments

Internal dashboards with limited users

Short-term demonstrations

Development/staging servers

For any public-facing or production application:

Always use a production WSGI server (Gunicorn/uWSGI)

Implement proper process monitoring (systemd/supervisor)

Use a dedicated application server for Django

Implement additional security hardening

Access the application at: https://yourdomain.com/forecasts/upload/