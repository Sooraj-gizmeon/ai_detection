"""
Integrated Video Processing Pipeline that handles downloading, processing, and cleanup
of videos from the queue system.
"""

import os
import sys
import asyncio
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add root directory to sys.path to avoid import issues
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.download_manager import DownloadManager
from main import VideoToShortsProcessor

# Configure logging
logger = logging.getLogger('integrated_processor')

class IntegratedVideoProcessor:
    """
    Integrates video download, processing, and cleanup in a complete pipeline.
    Manages the workflow from queue to finished shorts.
    """
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        input_dir: str = "input",
        output_dir: str = "output",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent: int = 2
    ):
        """
        Initialize the integrated processor.
        
        Args:
            config_path: Path to configuration file
            input_dir: Directory for input videos
            output_dir: Directory for output videos
            api_url: API endpoint URL
            api_key: Optional API authentication key (not required if API doesn't use auth)
            max_concurrent: Maximum concurrent downloads
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Ensure directories exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize download manager
        self.download_manager = DownloadManager(
            input_dir=input_dir,
            api_url=api_url,
            max_concurrent=max_concurrent
        )
        
        # Initialize video processor
        self.processor = VideoToShortsProcessor(config_path=config_path)
        
        # Keep track of processed videos
        self.processed_videos = []
        
        logger.info(f"Integrated Video Processor initialized with input_dir: {input_dir}, output_dir: {output_dir}")
    
    async def initialize_components(self):
        """Initialize all processing components."""
        await self.processor.initialize_components()
        logger.info("All processing components initialized")
    
    async def process_single_video(self, video_path: str, video_id: str = None) -> Dict:
        """
        Process a single video file.
        
        Args:
            video_path: Path to the video file
            video_id: Optional video ID for tracking
            
        Returns:
            Processing results dictionary
        """
        try:
            # Check if metadata exists for this video
            metadata = self.download_manager.get_metadata_for_video(video_path)
            
            # If video_id not provided, try to get from metadata
            if not video_id and metadata and "video_id" in metadata:
                video_id = metadata["video_id"]
            
            # Update configuration based on metadata
            if metadata:
                # Set canvas type (shorts, clips, etc.)
                if "canvas_type" in metadata:
                    self.processor.config["canvas_type"] = metadata["canvas_type"]
                
                # Set aspect ratio
                if "aspect_ratio" in metadata:
                    aspect_ratio = metadata["aspect_ratio"]
                    # Convert from string format if needed
                    if isinstance(aspect_ratio, str) and ":" in aspect_ratio:
                        width, height = map(int, aspect_ratio.split(":"))
                        self.processor.config["target_aspect_ratio"] = (width, height)
                    elif isinstance(aspect_ratio, (list, tuple)) and len(aspect_ratio) == 2:
                        self.processor.config["target_aspect_ratio"] = tuple(aspect_ratio)
                
                # Set max number of shorts to generate
                if "no_of_videos" in metadata:
                    self.processor.config["max_shorts_per_video"] = int(metadata["no_of_videos"])
                
                # Set duration limits
                if "min_duration" in metadata and "max_duration" in metadata:
                    min_duration = int(metadata.get("min_duration", 15))
                    max_duration = int(metadata.get("max_duration", 60))
                    self.processor.config["target_short_duration"] = (min_duration, max_duration)
            
            # If we have a video_id, update status to processing
            if video_id:
                await self.download_manager.update_processing_status(
                    video_id, 
                    "processing",
                    message=f"Processing video: {Path(video_path).name}"
                )
            
            # Process the video
            logger.info(f"Processing video: {video_path}")
            start_time = time.time()
            results = await self.processor.process_video(video_path)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {video_path} in {processing_time:.2f} seconds")
            
            # Add video_id to results if we have it
            if video_id:
                results["video_id"] = video_id
            
            # Add metadata information
            if metadata:
                results["metadata"] = metadata
            
            # Update status to completed
            if video_id:
                # Get details about output videos
                output_files = results.get("output_files", [])
                
                # Map output files to final_video_ids if available
                output_mapping = {}
                if metadata and "final_video_ids" in metadata:
                    final_video_ids = metadata["final_video_ids"]
                    
                    # Ensure we don't exceed the number of final_video_ids
                    for i, output_file in enumerate(output_files):
                        if i < len(final_video_ids):
                            output_mapping[final_video_ids[i]] = output_file
                
                await self.download_manager.update_processing_status(
                    video_id,
                    "completed",
                    message=f"Generated {len(output_files)} shorts",
                    metadata={
                        "output_files": output_files,
                        "output_mapping": output_mapping,
                        "processing_time": processing_time,
                        "shorts_count": len(output_files)
                    }
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            
            # Update status to failed if we have a video_id
            if video_id:
                await self.download_manager.update_processing_status(
                    video_id,
                    "failed",
                    message=f"Processing failed: {str(e)}"
                )
            
            return {
                "video_path": video_path,
                "video_id": video_id,
                "success": False,
                "error": str(e)
            }
    
    async def process_pending_videos(self, limit: int = 5) -> List[Dict]:
        """
        Process pending videos from the queue.
        
        Args:
            limit: Maximum number of videos to process
            
        Returns:
            List of processing results
        """
        try:
            # Download pending videos from queue
            downloaded_videos = await self.download_manager.process_pending_videos(limit=limit)
            
            if not downloaded_videos:
                logger.info("No pending videos to process")
                return []
            
            logger.info(f"Processing {len(downloaded_videos)} downloaded videos")
            
            # Process each downloaded video
            results = []
            for video_info in downloaded_videos:
                video_path = video_info["local_path"]
                video_id = video_info["video_id"]
                
                try:
                    # Process the video
                    result = await self.process_single_video(video_path, video_id)
                    results.append(result)
                    
                    # Add to processed videos list
                    self.processed_videos.append({
                        "video_id": video_id,
                        "video_path": video_path,
                        "success": True,
                        "output_files": result.get("output_files", []),
                        "shorts_generated": result.get("shorts_generated", 0)
                    })
                    
                    # Clean up input video after successful processing
                    if self.processor.config.get("cleanup_input_files", True):
                        logger.info(f"Cleaning up input video: {video_path}")
                        self.download_manager.cleanup_video(video_path)
                    
                except Exception as e:
                    logger.error(f"Failed to process video {video_id}: {e}")
                    results.append({
                        "video_id": video_id,
                        "video_path": video_path,
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Completed processing {len(results)} videos")
            return results
            
        except Exception as e:
            logger.error(f"Error in process_pending_videos: {e}")
            return []
    
    async def run_continuous_processing(self, interval: int = 60):
        """
        Run continuous processing of queue videos.
        
        Args:
            interval: Polling interval in seconds
        """
        logger.info(f"Starting continuous processing with interval {interval}s")
        
        # Initialize components
        await self.initialize_components()
        
        while True:
            try:
                # Process pending videos
                results = await self.process_pending_videos()
                
                if not results:
                    logger.info(f"No videos processed. Waiting {interval} seconds...")
                else:
                    logger.info(f"Processed {len(results)} videos. Waiting {interval} seconds...")
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous processing loop: {e}")
                # Wait before retrying
                await asyncio.sleep(interval)
    
    async def cleanup(self):
        """Clean up resources."""
        if self.processor:
            await self.processor.cleanup()
        logger.info("Cleaned up all resources")

async def main():
    """Main function for running the integrated processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrated Video Processing Pipeline")
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")
    parser.add_argument("--input-dir", "-i", type=str, default="input", help="Input directory")
    parser.add_argument("--output-dir", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--api-url", type=str, help="API URL")
    parser.add_argument("--api-key", type=str, help="API key (optional)")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Maximum concurrent downloads")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--single-run", "-s", action="store_true", help="Process pending videos once and exit")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"logs/video_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Initialize processor
    processor = IntegratedVideoProcessor(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_url=args.api_url,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent
    )
    
    try:
        # Initialize components
        await processor.initialize_components()
        
        if args.single_run:
            # Process pending videos once
            results = await processor.process_pending_videos()
            print(f"Processed {len(results)} videos")
        else:
            # Run continuous processing
            await processor.run_continuous_processing(interval=args.interval)
    
    finally:
        # Clean up resources
        await processor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
