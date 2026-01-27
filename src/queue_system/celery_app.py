"""
Celery application for managing the video processing queue.
"""

import os
from celery import Celery
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Celery app
redis_password = os.getenv('REDIS_PASSWORD', '')
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', '6379')
redis_db = os.getenv('REDIS_DB', '0')

# Create Redis URL with password if provided
if redis_password:
    broker_url = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
    result_backend = f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
else:
    broker_url = f"redis://{redis_host}:{redis_port}/{redis_db}"
    result_backend = f"redis://{redis_host}:{redis_port}/{redis_db}"

celery_app = Celery(
    'video_processor',
    broker=os.getenv('CELERY_BROKER_URL', broker_url),
    backend=os.getenv('CELERY_RESULT_BACKEND', result_backend)
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour timeout for tasks
    worker_max_tasks_per_child=10,  # Restart worker after 10 tasks
    worker_prefetch_multiplier=1,  # Don't prefetch tasks
)

# Set up task routes
from kombu import Queue
celery_app.conf.task_queues = (
    Queue('celery', routing_key='celery'),
    Queue('video_processing', routing_key='video_processing'),
)

celery_app.conf.task_default_queue = 'video_processing'
celery_app.conf.task_default_exchange = 'video_processing'
celery_app.conf.task_default_routing_key = 'video_processing'

# Load tasks from all task modules
celery_app.autodiscover_tasks(['src.queue_system.tasks'], force=True)
