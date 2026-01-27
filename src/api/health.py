"""
Health check endpoints for the API service.
"""

from fastapi import APIRouter
from src.api.app import redis_client, celery_app
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check Redis connection
    redis_status = "ok"
    try:
        redis_client.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = f"error: {str(e)}"
    
    # Check Celery connection (broker only)
    celery_status = "ok"
    try:
        celery_app.control.ping(timeout=1.0)
    except Exception as e:
        logger.error(f"Celery broker health check failed: {e}")
        celery_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "service": "video-processing-api",
        "components": {
            "redis": redis_status,
            "celery_broker": celery_status
        }
    }
