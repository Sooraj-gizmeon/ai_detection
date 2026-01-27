#!/usr/bin/env python3
"""
Intro/Outro Handler for video processing.
This module handles adding intro and outro videos to generated clips.
"""

import os
import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class IntroOutroHandler:
    """Handler for adding intro and outro videos to clips."""
    
    def __init__(self, storage_bucket: str = None):
        """
        Initialize the IntroOutroHandler.
        
        Args:
            storage_bucket: Name of the storage bucket for downloading intro/outro videos
        """
        self.storage_bucket = storage_bucket
        self.temp_files = []
        
    def get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing video info (duration, width, height, fps) or None if failed
        """
        try:
            # Get duration
            duration_cmd = [
                'ffprobe', '-v', 'quiet',
                '-show_entries', 'format=duration',
                '-of', 'csv=p=0',
                video_path
            ]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
            duration = float(duration_result.stdout.strip())
            
            # Get resolution and fps
            video_cmd = [
                'ffprobe', '-v', 'quiet',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'csv=p=0',
                video_path
            ]
            video_result = subprocess.run(video_cmd, capture_output=True, text=True, check=True)
            video_info = video_result.stdout.strip().split(',')
            
            width = int(video_info[0])
            height = int(video_info[1])
            
            # Parse frame rate
            fps_str = video_info[2] if len(video_info) > 2 else "30/1"
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info for {video_path}: {e}")
            return None
    
    def download_intro_outro_videos(
        self, 
        intro_url: Optional[str] = None, 
        outro_url: Optional[str] = None,
        storage_bucket: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Download intro and outro videos from storage.
        
        Args:
            intro_url: URL/path to intro video in storage
            outro_url: URL/path to outro video in storage
            storage_bucket: Storage bucket name (overrides instance bucket)
            
        Returns:
            Tuple of (intro_path, outro_path) - paths to downloaded files or None
        """
        bucket = storage_bucket or self.storage_bucket
        intro_path = None
        outro_path = None
        
        if not bucket:
            logger.warning("No storage bucket provided - cannot download intro/outro videos")
            return None, None
        
        try:
            # Initialize Wasabi S3 client (intro/outro always uses Wasabi)
            import boto3
            
            s3_client = boto3.client(
                's3',
                endpoint_url=os.environ.get('WASABI_ENDPOINT'),
                aws_access_key_id=os.environ.get('WASABI_ACCESS_KEY'),
                aws_secret_access_key=os.environ.get('WASABI_SECRET_KEY')
            )
            logger.info(f"Using Wasabi storage for intro/outro downloads")
            
            # Create temp directory for downloads
            import tempfile
            temp_dir = tempfile.gettempdir()
            
            # Download intro video
            if intro_url:
                logger.info(f"Downloading intro video from: {intro_url}")
                try:
                    # Create output path for intro
                    intro_filename = os.path.basename(intro_url)
                    intro_output_path = os.path.join(temp_dir, f"intro_{int(time.time())}_{intro_filename}")
                    
                    # Download file using boto3
                    s3_client.download_file(bucket, intro_url, intro_output_path)
                    
                    if os.path.exists(intro_output_path):
                        size = os.path.getsize(intro_output_path)
                        logger.info(f"✅ Downloaded intro video: {intro_output_path} ({size} bytes)")
                        intro_path = intro_output_path
                        self.temp_files.append(intro_path)
                    else:
                        logger.warning(f"Failed to download intro video from {intro_url}")
                        intro_path = None
                except Exception as e:
                    logger.error(f"Error downloading intro video: {e}")
                    intro_path = None
            
            # Download outro video
            if outro_url:
                logger.info(f"Downloading outro video from: {outro_url}")
                try:
                    # Create output path for outro
                    outro_filename = os.path.basename(outro_url)
                    outro_output_path = os.path.join(temp_dir, f"outro_{int(time.time())}_{outro_filename}")
                    
                    # Download file using boto3
                    s3_client.download_file(bucket, outro_url, outro_output_path)
                    
                    if os.path.exists(outro_output_path):
                        size = os.path.getsize(outro_output_path)
                        logger.info(f"✅ Downloaded outro video: {outro_output_path} ({size} bytes)")
                        outro_path = outro_output_path
                        self.temp_files.append(outro_path)
                    else:
                        logger.warning(f"Failed to download outro video from {outro_url}")
                        outro_path = None
                except Exception as e:
                    logger.error(f"Error downloading outro video: {e}")
                    outro_path = None
            
            return intro_path, outro_path
            
        except Exception as e:
            logger.error(f"Error in download_intro_outro_videos: {e}")
            return None, None
    
    def concatenate_videos(
        self,
        main_video_path: str,
        intro_path: Optional[str] = None,
        outro_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> bool:
        """
        Concatenate intro, main video, and outro using FFmpeg.
        
        Args:
            main_video_path: Path to the main video file
            intro_path: Optional path to intro video
            outro_path: Optional path to outro video
            output_path: Output path for concatenated video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # If no intro/outro, just copy the file
            if not intro_path and not outro_path:
                if output_path and output_path != main_video_path:
                    import shutil
                    shutil.copy2(main_video_path, output_path)
                    logger.info(f"No intro/outro - copied main video to: {output_path}")
                return True
            
            # Determine output path
            if not output_path:
                output_path = main_video_path.replace('.mp4', '_with_intro_outro.mp4')
            
            # Get main video info
            main_info = self.get_video_info(main_video_path)
            if not main_info:
                logger.error(f"Failed to get info for main video: {main_video_path}")
                return False
            
            target_width = main_info['width']
            target_height = main_info['height']
            target_fps = main_info['fps']
            
            logger.info(f"Target dimensions: {target_width}x{target_height} @ {target_fps:.2f}fps")
            
            # Create concat list file
            concat_list = []
            
            # Add intro if provided
            if intro_path and os.path.exists(intro_path):
                # Verify intro video
                intro_info = self.get_video_info(intro_path)
                if intro_info:
                    logger.info(f"Intro video: {intro_info['width']}x{intro_info['height']}, {intro_info['duration']:.2f}s")
                    concat_list.append(intro_path)
                else:
                    logger.warning(f"Could not get info for intro video: {intro_path}")
            
            # Add main video
            concat_list.append(main_video_path)
            
            # Add outro if provided
            if outro_path and os.path.exists(outro_path):
                # Verify outro video
                outro_info = self.get_video_info(outro_path)
                if outro_info:
                    logger.info(f"Outro video: {outro_info['width']}x{outro_info['height']}, {outro_info['duration']:.2f}s")
                    concat_list.append(outro_path)
                else:
                    logger.warning(f"Could not get info for outro video: {outro_path}")
            
            if len(concat_list) == 1:
                # Only main video, just copy
                import shutil
                shutil.copy2(main_video_path, output_path)
                logger.info(f"Only main video available - copied to: {output_path}")
                return True
            
            # Method 1: Use FFmpeg filter_complex for better quality
            # This method scales and pads all videos to match dimensions
            logger.info(f"Concatenating {len(concat_list)} videos using filter_complex...")
            
            # Build filter_complex command
            filter_parts = []
            for i, video_path in enumerate(concat_list):
                # Scale and pad each video to match target dimensions
                filter_parts.append(
                    f"[{i}:v]scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                    f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                    f"setsar=1,fps={target_fps}[v{i}]"
                )
                filter_parts.append(f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo[a{i}]")
            
            # Concatenate all processed streams - MUST be interleaved: [v0][a0][v1][a1][v2][a2]
            concat_streams = ''.join(f"[v{i}][a{i}]" for i in range(len(concat_list)))
            filter_parts.append(f"{concat_streams}concat=n={len(concat_list)}:v=1:a=1[outv][outa]")
            
            filter_complex = ';'.join(filter_parts)
            
            # Build FFmpeg command
            ffmpeg_cmd = ['ffmpeg', '-y']
            
            # Add input files
            for video_path in concat_list:
                ffmpeg_cmd.extend(['-i', video_path])
            
            # Add filter complex
            ffmpeg_cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '[outv]',
                '-map', '[outa]',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                output_path
            ])
            
            logger.info(f"Running FFmpeg concatenation...")
            logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                if os.path.exists(output_path):
                    output_size = os.path.getsize(output_path)
                    output_info = self.get_video_info(output_path)
                    if output_info:
                        logger.info(
                            f"✅ Successfully concatenated videos: {output_path} "
                            f"({output_size} bytes, {output_info['duration']:.2f}s)"
                        )
                        return True
                    else:
                        logger.warning("Concatenation completed but could not verify output")
                        return True
                else:
                    logger.error(f"FFmpeg completed but output file not found: {output_path}")
                    return False
            else:
                logger.error(f"FFmpeg concatenation failed with return code {result.returncode}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg concatenation timed out")
            return False
        except Exception as e:
            logger.error(f"Error concatenating videos: {e}", exc_info=True)
            return False
    
    def process_clip_with_intro_outro(
        self,
        clip_path: str,
        intro_url: Optional[str] = None,
        outro_url: Optional[str] = None,
        storage_bucket: Optional[str] = None
    ) -> str:
        """
        Process a single clip by adding intro and outro videos.
        
        Args:
            clip_path: Path to the clip to process
            intro_url: URL/path to intro video in storage
            outro_url: URL/path to outro video in storage
            storage_bucket: Storage bucket name
            
        Returns:
            Path to the processed clip (same as input if no intro/outro or if failed)
        """
        try:
            if not intro_url and not outro_url:
                logger.info("No intro/outro URLs provided - skipping")
                return clip_path
            
            logger.info(f"Processing clip with intro/outro: {clip_path}")
            
            # Download intro/outro videos
            intro_path, outro_path = self.download_intro_outro_videos(
                intro_url=intro_url,
                outro_url=outro_url,
                storage_bucket=storage_bucket or self.storage_bucket
            )
            
            if not intro_path and not outro_path:
                logger.warning("Failed to download intro/outro videos - using original clip")
                return clip_path
            
            # Verify the clip exists and has content
            if not os.path.exists(clip_path):
                logger.error(f"Clip file does not exist: {clip_path}")
                return clip_path
            
            clip_size = os.path.getsize(clip_path)
            if clip_size == 0:
                logger.error(f"Clip file is empty: {clip_path}")
                return clip_path
            
            logger.info(f"Original clip size: {clip_size} bytes")
            
            # Create temporary output path
            clip_dir = os.path.dirname(clip_path)
            clip_name = os.path.basename(clip_path)
            temp_output = os.path.join(clip_dir, f"temp_concat_{clip_name}")
            
            # Concatenate videos
            success = self.concatenate_videos(
                main_video_path=clip_path,
                intro_path=intro_path,
                outro_path=outro_path,
                output_path=temp_output
            )
            
            if success and os.path.exists(temp_output):
                # Replace original with concatenated version
                import shutil
                shutil.move(temp_output, clip_path)
                
                new_size = os.path.getsize(clip_path)
                logger.info(f"✅ Replaced clip with intro/outro version: {new_size} bytes")
                return clip_path
            else:
                logger.error("Failed to concatenate videos - using original clip")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return clip_path
                
        except Exception as e:
            logger.error(f"Error processing clip with intro/outro: {e}", exc_info=True)
            return clip_path
    
    def cleanup(self):
        """Clean up temporary downloaded files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
        
        self.temp_files = []


def process_shorts_with_intro_outro(
    shorts_details: List[Dict],
    intro_url: Optional[str] = None,
    outro_url: Optional[str] = None,
    storage_bucket: Optional[str] = None
) -> List[Dict]:
    """
    Process multiple shorts by adding intro and outro videos.
    
    Args:
        shorts_details: List of shorts dictionaries with 'output_path' keys
        intro_url: URL/path to intro video in storage
        outro_url: URL/path to outro video in storage
        storage_bucket: Storage bucket name
        
    Returns:
        Updated list of shorts details with processed videos
    """
    if not intro_url and not outro_url:
        logger.info("No intro/outro URLs provided - returning original shorts")
        return shorts_details
    
    logger.info(f"Processing {len(shorts_details)} shorts with intro/outro")
    
    handler = IntroOutroHandler(storage_bucket=storage_bucket)
    
    try:
        for short in shorts_details:
            clip_path = short.get('output_path')
            if not clip_path:
                logger.warning(f"Short missing output_path: {short}")
                continue
            
            # Process the clip
            processed_path = handler.process_clip_with_intro_outro(
                clip_path=clip_path,
                intro_url=intro_url,
                outro_url=outro_url,
                storage_bucket=storage_bucket
            )
            
            # Update the output path if changed
            if processed_path != clip_path:
                short['output_path'] = processed_path
        
        return shorts_details
        
    finally:
        handler.cleanup()
