#!/usr/bin/env python
"""
Standalone API service for video processing.
This service runs independently from the Celery worker.
"""

import os
import sys
import logging
from pathlib import Path
import uvicorn
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/api_service_{os.getpid()}.log")
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start the API service."""
    try:
        # Load environment variables
        load_dotenv()
        
        logger.info("Starting API service...")
        
        # Import the FastAPI app
        from src.api.app import app
        
        # Get host and port from environment variables
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "6000"))
        
        # Start uvicorn server
        logger.info(f"API service listening on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
        
    except Exception as e:
        logger.error(f"Failed to start API service: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
