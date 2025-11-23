import os
from celery import Celery

def make_celery():
    # 1. Smart URL Handling
    # Checks CELERY_BROKER_URL first, then REDIS_URL (Railway default), then localhost fallback
    redis_url = os.getenv('CELERY_BROKER_URL') or os.getenv('REDIS_URL') or 'redis://localhost:6379/0'

    app = Celery(
        'video_automation_tasks',
        broker=redis_url,
        backend=redis_url,
        include=['celery_worker'] # Ensures the worker tasks are registered
    )

    # 2. CRITICAL CONFIGURATION
    app.conf.update(
        # Fix for "Exception information must include..." error:
        # We force everything to be JSON. This prevents Python objects from crashing Redis.
        task_serializer='json',
        accept_content=['json'],  
        result_serializer='json',
        
        # Timezones
        timezone='UTC',
        enable_utc=True,

        # Connection Settings
        broker_connection_retry_on_startup=True,
        broker_pool_limit=10, # Limit connections to prevent Redis crashes

        # Video Processing Optimizations
        # Prefetch=1 means: "Don't grab a new video task until the current one is 100% done"
        # This prevents the worker from choking on RAM.
        worker_prefetch_multiplier=1, 
        
        # Result Settings
        result_expires=86400, # Keep results for 24 hours
        task_track_started=True, # Helps the UI show "Processing"
        result_extended=True,

        # Timeouts (Video generation is slow!)
        broker_transport_options={
            'visibility_timeout': 7200, # 2 Hours (Plenty of time for long renders)
            'max_retries': 3,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.5,
        }
    )

    return app

celery = make_celery()
