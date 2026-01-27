#!/usr/bin/env python
"""
Standalone Celery worker service for video processing.
This service runs independently from the API.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/celery_worker_{os.getpid()}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start the Celery worker service."""
    try:
        # Load environment variables
        load_dotenv()
        
        logger.info("Starting Celery worker service...")
        
        # Import the Celery app
        from src.queue_system.celery_app import celery_app
        
        # Start worker using Celery's API directly
        argv = [
            'worker',
            '--loglevel=INFO',
            '--concurrency=2',
            '--hostname=worker@%h',
            '--pool=prefork',
        ]
        
        logger.info(f"Worker initialized with app: {celery_app}")
        celery_app.worker_main(argv)
        
    except Exception as e:
        logger.error(f"Failed to start Celery worker service: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
