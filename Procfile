# Procfile

web: gunicorn -c gunicorn.conf.py app:app
worker: celery -A celery_worker.celery worker --loglevel=info -c 2
