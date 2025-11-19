# gunicorn.conf.py

# --- FIX: Import the OS module to read environment variables ---
import os 

# Use Gunicorn's sync worker class for Flask/standard apps
worker_class = 'sync'

# Set the number of worker processes (e.g., 2 * CPU_CORES + 1)
workers = 5 

# Set the host and port (uses the PORT environment variable provided by Railway)
bind = '0.0.0.0:' + os.environ.get('PORT', '8080') 

# Timeout for long Celery tasks
timeout = 120
