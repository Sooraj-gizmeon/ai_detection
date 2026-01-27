"""
Download Manager for handling video downloads from Wasabi S3 storage.
"""

import os
import sys
import json
import logging
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add project root to path to avoid import issues
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from download_from_wasabi import (
    WasabiClient, 
    APIClient, 
    process_queue_item_with_semaphore,
    cleanup_input_video
)

# Configure logging
logger = logging.getLogger('download_manager')

class DownloadManager:
    """
    Manages downloading videos from Wasabi S3 based on queue information
    and cleanup after processing.
    """
    
    def __init__(
        self, 
        input_dir: str = "input",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        wasabi_access_key: Optional[str] = None,
        wasabi_secret_key: Optional[str] = None,
        max_concurrent: int = 2
    ):
        """
        Initialize the download manager.
        
        Args:
            input_dir: Directory for downloaded videos
            api_url: API endpoint URL
            api_key: Optional API authentication key (not required if API doesn't use auth)
            wasabi_access_key: Wasabi S3 access key
            wasabi_secret_key: Wasabi S3 secret key
            max_concurrent: Maximum concurrent downloads
        """
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(exist_ok=True)
        
        # Initialize API client
        self.api_client = APIClient(api_url=api_url)
        
        # Initialize Wasabi client
        self.wasabi_client = WasabiClient(
            access_key=wasabi_access_key,
            secret_key=wasabi_secret_key
        )
        
        # Concurrency control
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"Download Manager initialized with input_dir: {input_dir}, max_concurrent: {max_concurrent}")
    
    async def download_video_from_queue(self, queue_item_id: str) -> Tuple[bool, Optional[str]]:
        """
        Download a video from a specific queue item.
        
        Args:
            queue_item_id: ID of the queue item to process
            
        Returns:
            Tuple of (success, local_path)
        """
        try:
            # Get queue item details
            item = await self.api_client.get_queue_item(queue_item_id)
            if not item:
                logger.error(f"Failed to get details for queue item {queue_item_id}")
                return False, None
            
            # Process the item
            success = await process_queue_item_with_semaphore(
                self.semaphore,
                self.wasabi_client,
                self.api_client,
                item,
                str(self.input_dir)
            )
            
            if success:
                # Get the local path from the bucket_path
                video_id = item.get("video_id")
                bucket_path = item.get("bucket_path")
                local_path = str(self.input_dir / bucket_path)
                
                logger.info(f"Successfully downloaded video {video_id} to {local_path}")
                return True, local_path
            else:
                logger.error(f"Failed to download video for queue item {queue_item_id}")
                return False, None
                
        except Exception as e:
            logger.error(f"Error downloading video from queue {queue_item_id}: {e}")
            return False, None
    
    async def process_pending_videos(self, limit: int = 5) -> List[Dict]:
        """
        Get and process pending videos from the queue.
        
        Args:
            limit: Maximum number of videos to process
            
        Returns:
            List of processed video info dictionaries
        """
        try:
            # Get pending queue items
            pending_items = await self.api_client.get_queue_items(status="queued", limit=limit)
            
            if not pending_items:
                logger.info("No pending videos found in queue")
                return []
            
            logger.info(f"Found {len(pending_items)} pending videos in queue")
            
            # Filter valid items
            valid_items = []
            for item in pending_items:
                video_id = item.get("video_id")
                bucket_path = item.get("bucket_path")
                storage_bucket = item.get("storage_bucket")
                
                if all([video_id, bucket_path, storage_bucket]):
                    valid_items.append(item)
                else:
                    logger.warning(f"Skipping item {item.get('id')} due to missing required fields")
                    # Update status to failed for invalid items
                    if item.get("id"):
                        await self.api_client.update_queue_item(
                            item.get("id"), 
                            "failed", 
                            message="Missing required fields (video_id, bucket_path, or storage_bucket)"
                        )
            
            if not valid_items:
                logger.info("No valid items to process")
                return []
            
            # Process valid items up to concurrent limit
            download_tasks = []
            for item in valid_items[:self.max_concurrent]:
                item_id = item.get("id")
                if item_id:
                    task = asyncio.create_task(
                        self.download_video_from_queue(item_id)
                    )
                    download_tasks.append(task)
            
            # Wait for downloads to complete
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            # Process results
            processed_videos = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Download task failed with exception: {result}")
                elif isinstance(result, tuple) and result[0]:
                    # Successful download
                    item = valid_items[i]
                    video_id = item.get("video_id")
                    local_path = result[1]
                    
                    processed_videos.append({
                        "video_id": video_id,
                        "local_path": local_path,
                        "item_id": item.get("id"),
                        "pubid": item.get("pubid"),
                        "no_of_videos": item.get("no_of_videos"),
                        "final_video_ids": item.get("final_video_ids"),
                        "canvas_type": item.get("canvas_type", "shorts"),
                        "aspect_ratio": item.get("aspect_ratio", (9, 16))
                    })
            
            logger.info(f"Successfully processed {len(processed_videos)} videos")
            return processed_videos
            
        except Exception as e:
            logger.error(f"Error processing pending videos: {e}")
            return []
    
    def cleanup_video(self, video_path: str) -> bool:
        """
        Clean up a video file after processing.
        
        Args:
            video_path: Path to the video file to clean up
            
        Returns:
            Boolean indicating success
        """
        return cleanup_input_video(video_path)
    
    def get_metadata_for_video(self, video_path: str) -> Optional[Dict]:
        """
        Get metadata for a video from its accompanying .meta.json file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing metadata or None if not found
        """
        metadata_path = f"{video_path}.meta.json"
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"No metadata file found for {video_path}")
                return None
        except Exception as e:
            logger.error(f"Error reading metadata for {video_path}: {e}")
            return None
    
    async def update_processing_status(
        self, 
        video_id: str, 
        status: str, 
        message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update processing status for a video.
        
        Args:
            video_id: ID of the video
            status: New status value
            message: Optional message
            metadata: Optional metadata dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            # Find the queue item ID for this video ID
            pending_items = await self.api_client.get_queue_items(limit=50)
            item_id = None
            
            for item in pending_items:
                if item.get("video_id") == video_id:
                    item_id = item.get("id")
                    break
            
            if not item_id:
                logger.warning(f"Could not find queue item for video {video_id}")
                return False
            
            # Update the queue item status
            success = await self.api_client.update_queue_item(
                item_id,
                status,
                message=message,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Updated status of video {video_id} to {status}")
            else:
                logger.error(f"Failed to update status of video {video_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating processing status for video {video_id}: {e}")
            return False
