#!/usr/bin/env python3
"""
Example script demonstrating how to use the API with custom min_duration and max_duration parameters.
"""

import os
import sys
import json
import requests
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# API configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_URL = f"http://{API_HOST}:{API_PORT}"

def submit_video_processing_request(
    video_id: str,
    bucket_path: str,
    pubid: str,
    final_video_ids: Optional[List[int]] = None,
    min_duration: int = 30,
    max_duration: int = 180,
    no_of_videos: int = 5,
    canvas_type: str = "shorts",
    aspect_ratio: Optional[str] = None,
    storage_bucket: str = "videos"
) -> Dict:
    """
    Submit a video processing request to the API with custom duration parameters.
    
    Args:
        video_id: Unique identifier for the video job
        bucket_path: Path to the input video in the storage bucket
        pubid: Publisher ID
        final_video_ids: List of integer IDs for the output videos
        min_duration: Minimum duration of each output video in seconds (default: 30)
        max_duration: Maximum duration of each output video in seconds (default: 180)
        no_of_videos: Number of output videos to generate (default: 5)
        canvas_type: Canvas type: 'shorts' (9:16) or 'clips' (16:9)
        aspect_ratio: Content aspect ratio within the canvas ('9:16', '16:9', '1:1', etc.)
        storage_bucket: Name of the storage bucket
        
    Returns:
        Response from the API
    """
    # Prepare request data
    request_data = {
        "video_id": video_id,
        "bucket_path": bucket_path,
        "pubid": pubid,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "no_of_videos": no_of_videos,
        "canvas_type": canvas_type,
        "storage_bucket": storage_bucket,
    }
    
    # Add optional parameters if provided
    if final_video_ids:
        request_data["final_video_ids"] = final_video_ids
    
    if aspect_ratio:
        request_data["aspect_ratio"] = aspect_ratio
    
    # Make request to API
    logger.info(f"Submitting request to {API_URL}/api/queue/video with min_duration={min_duration}, max_duration={max_duration}")
    
    try:
        response = requests.post(
            f"{API_URL}/api/queue/video",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Request successful: {result}")
            return result
        else:
            logger.error(f"Request failed with status code {response.status_code}: {response.text}")
            return {"error": f"Request failed with status code {response.status_code}", "details": response.text}
    
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}

def main():
    """Main function demonstrating API usage."""
    # Example parameters
    video_id = "test-duration-example"
    bucket_path = "test/example_video.mp4"
    pubid = "test-publisher"
    final_video_ids = [4001, 4002, 4003, 4004, 4005]
    
    # Custom duration parameters
    min_duration = 30  # 30 seconds
    max_duration = 180  # 3 minutes
    
    print(f"=== Video Processing API Demo: Custom Duration Parameters ===")
    print(f"Creating request with min_duration={min_duration}s, max_duration={max_duration}s")
    
    # Submit the request
    result = submit_video_processing_request(
        video_id=video_id,
        bucket_path=bucket_path,
        pubid=pubid,
        final_video_ids=final_video_ids,
        min_duration=min_duration,
        max_duration=max_duration
    )
    
    # Pretty print the result
    print("\nAPI Response:")
    print(json.dumps(result, indent=2))
    
    if "task_id" in result:
        task_id = result["task_id"]
        print(f"\nTask submitted successfully with ID: {task_id}")
        print(f"You can check the status at: {API_URL}/api/queue/status/{task_id}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
