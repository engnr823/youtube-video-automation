# gunicorn.conf.py
# Use Gunicorn's sync worker class for Flask/standard apps
worker_class = 'sync'

# Use gevent workers for better handling of concurrent I/O (since you installed gevent)
# This requires installing gevent, which you did in requirements.txt
# Alternatively, you can use 'gthread' for standard multithreading.
# worker_class = 'gevent' 

# Set the number of worker processes (Recommended: 2 * CPU_CORES + 1)
# For a typical basic Railway CPU (1 or 2 cores), 3 or 5 workers is safe.
workers = 5 

# Set the host and port (Railway uses PORT environment variable)
bind = '0.0.0.0:' + os.environ.get('PORT', '8080')

# Timeout for long Celery tasks (optional, but good practice)
# Set to something high since your Celery tasks take minutes.
timeout = 120
