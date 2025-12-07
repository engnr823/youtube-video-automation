# Procfile

web: gunicorn -c gunicorn.conf.py app:app
worker: celery -A celery_init worker --loglevel=info -c 1


