import os
from celery import Celery


def make_celery():

    # 1. Smart Redis URL Handling
    redis_url = (
        os.getenv('CELERY_BROKER_URL') or
        os.getenv('REDIS_URL') or
        'redis://localhost:6379/0'
    )

    app = Celery(
        'video_automation_tasks',
        broker=redis_url,
        backend=redis_url,
        include=['celery_worker']  # auto-load task file
    )

    # 2. PRODUCTION-GRADE CONFIG
    app.conf.update(

        # --- SERIALIZATION (No Pickle = No Crashes)
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        task_default_queue='video_tasks',

        # --- RELIABILITY
        task_acks_late=True,      # If worker dies, task returns to queue
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,   # Handle heavy video tasks safely

        # --- TIMEZONE
        timezone='UTC',
        enable_utc=True,

        # --- BROKER STABILITY (Redis Friendly)
        broker_connection_retry_on_startup=True,
        broker_pool_limit=10,
        broker_transport_options={
            'visibility_timeout': 7200, # 2 hours video job timeout
            'max_retries': 5,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.5,
        },

        # --- RESULT SETTINGS
        result_expires=86400,  # Keep results 24 hours
        task_track_started=True,
        result_extended=True,
    )

    return app


celery = make_celery()

