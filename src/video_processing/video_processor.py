# src/video_processing/video_processor.py
"""Video processing utilities for format conversion and segment extraction"""

import cv2
import ffmpeg
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import subprocess
import json


class VideoProcessor:
    """
    Handles video processing operations including format conversion,
    segment extraction, and aspect ratio changes.
    """
    
    def __init__(self, temp_dir: str = "temp", output_dir: str = "output"):
        """
        Initialize video processor.
        
        Args:
            temp_dir: Directory for temporary files
            output_dir: Directory for output files
        """
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
            
            info = {
                'duration': float(probe['format']['duration']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),
                'codec': video_stream['codec_name'],
                'bitrate': int(probe['format'].get('bit_rate', 0)),
                'aspect_ratio': video_stream['width'] / video_stream['height'],
                'total_frames': int(video_stream.get('nb_frames', 0))
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            return {}
    
    def extract_segment(self, 
                       input_path: str,
                       output_path: str,
                       start_time: float,
                       end_time: float,
                       copy_streams: bool = True) -> bool:
        """
        Extract a segment from video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output segment
            start_time: Start time in seconds
            end_time: End time in seconds
            copy_streams: Whether to copy streams without re-encoding
            
        Returns:
            True if successful, False otherwise
        """
        try:
            duration = end_time - start_time
            
            # Log extraction details
            self.logger.info(f"🔄 FFMPEG SEGMENT EXTRACTION:")
            self.logger.info(f"   📂 Input: {input_path}")
            self.logger.info(f"   📂 Output: {output_path}")
            self.logger.info(f"   ⏰ Time Range: {start_time:.3f}s → {end_time:.3f}s")
            self.logger.info(f"   ⏱️  Duration: {duration:.3f}s")
            self.logger.info(f"   ⚙️  Copy Streams: {copy_streams}")

            if copy_streams:
                # FIXED: Simplified copy mode without timestamp manipulation that can cause A/V sync issues
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),     # Place BEFORE -i for fast seeking
                    '-i', input_path,
                    '-t', str(duration),
                    '-map', '0:v:0',            # Map only first video stream
                    '-map', '0:a:0',            # Map only first audio stream
                    '-c', 'copy',               # Copy without re-encoding
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-y',
                    output_path
                ]
                
                self.logger.info(f"🔧 FFmpeg command (copy mode): {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"❌ FFmpeg extraction failed (copy mode): {result.stderr}")
                    # Fallback to re-encode mode if copy mode fails
                    self.logger.info("🔄 Attempting fallback to re-encode mode...")
                    copy_streams = False
                else:
                    self.logger.info(f"✅ FFmpeg extraction successful (copy mode)")
            
            if not copy_streams:
                # Re-encode mode with proper A/V sync - more reliable but slower
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-ss', str(start_time),     # Place before -i for fast seeking in re-encode mode
                    '-i', input_path,
                    '-t', str(duration),
                    '-map', '0:v:0',            # Map only first video stream
                    '-map', '0:a:0',            # Map only first audio stream
                    '-c:v', 'libx264',          # Re-encode video
                    '-c:a', 'aac',              # Re-encode audio
                    '-crf', '23',               # Good quality setting
                    '-preset', 'medium',        # Balance speed vs compression
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-vsync', '1',              # Ensure proper video frame sync
                    '-async', '1',              # Fix audio sync to video
                    '-copytb', '1',             # Copy input timebase to output
                    '-y',
                    output_path
                ]
                
                self.logger.info(f"🔧 FFmpeg command (re-encode mode): {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.logger.error(f"❌ FFmpeg extraction failed (re-encode mode): {result.stderr}")
                    return False
                else:
                    self.logger.info(f"✅ FFmpeg extraction successful (re-encode mode)")
            
            # Verify output file was created and has content
            import os
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                self.logger.info(f"📁 Output file created: {output_path} ({file_size} bytes)")
                
                # Additional verification: try to get video info to ensure it's valid
                try:
                    info = self.get_video_info(output_path)
                    if info and info.get('duration', 0) > 0:
                        self.logger.info(f"✅ Video validation successful: {info.get('duration', 0):.2f}s duration")
                    else:
                        self.logger.warning(f"⚠️  Video validation failed: invalid duration or metadata")
                        return False
                except Exception as e:
                    self.logger.warning(f"⚠️  Video validation failed: {e}")
                    return False
            else:
                self.logger.error(f"❌ Output file not created: {output_path}")
                return False
            
            self.logger.info(f"✅ Successfully extracted segment: {start_time:.2f}s - {end_time:.2f}s → {duration:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract segment: {e}")
            return False
    
    def convert_to_vertical(self, 
                          input_path: str,
                          output_path: str,
                          target_aspect_ratio: Tuple[int, int] = (9, 16),
                          crop_strategy: str = "center") -> bool:
        """
        Convert video to vertical format.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_aspect_ratio: Target aspect ratio (width, height)
            crop_strategy: Cropping strategy ('center', 'top', 'bottom')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get input video info
            info = self.get_video_info(input_path)
            if not info:
                return False
            
            input_width = info['width']
            input_height = info['height']
            target_width, target_height = target_aspect_ratio
            
            # Calculate target dimensions
            target_aspect = target_width / target_height
            
            # Calculate crop dimensions
            if input_width / input_height > target_aspect:
                # Input is wider, crop width
                crop_height = input_height
                crop_width = int(input_height * target_aspect)
            else:
                # Input is taller, crop height
                crop_width = input_width
                crop_height = int(input_width / target_aspect)
            
            # Calculate crop position
            if crop_strategy == "center":
                crop_x = (input_width - crop_width) // 2
                crop_y = (input_height - crop_height) // 2
            elif crop_strategy == "top":
                crop_x = (input_width - crop_width) // 2
                crop_y = 0
            elif crop_strategy == "bottom":
                crop_x = (input_width - crop_width) // 2
                crop_y = input_height - crop_height
            else:
                crop_x = (input_width - crop_width) // 2
                crop_y = (input_height - crop_height) // 2
            
            # Apply crop and resize with explicit stream mapping for vertical conversion
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-filter:v', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width * 100}:{target_height * 100}',
                '-map', '0:v:0',              # Map only first video stream
                '-map', '0:a:0',              # Map only first audio stream  
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-crf', '23',
                '-preset', 'medium',
                '-y',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg vertical conversion failed: {result.stderr}")
                return False
            
            self.logger.info(f"Converted to vertical format {target_width}:{target_height}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert to vertical: {e}")
            return False
            
    def convert_to_horizontal(self, 
                            input_path: str,
                            output_path: str,
                            target_aspect_ratio: Tuple[int, int] = (16, 9),
                            crop_strategy: str = "center") -> bool:
        """
        Convert video to horizontal format (for clips).
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_aspect_ratio: Target aspect ratio (width, height)
            crop_strategy: Cropping strategy ('center', 'top', 'bottom')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get input video info
            info = self.get_video_info(input_path)
            if not info:
                return False
            
            input_width = info['width']
            input_height = info['height']
            target_width, target_height = target_aspect_ratio
            
            # Calculate target dimensions
            target_aspect = target_width / target_height
            
            # Calculate crop dimensions
            if input_width / input_height > target_aspect:
                # Input is wider, crop width
                crop_height = input_height
                crop_width = int(input_height * target_aspect)
            else:
                # Input is taller, crop height
                crop_width = input_width
                crop_height = int(input_width / target_aspect)
            
            # Calculate crop position
            if crop_strategy == "center":
                crop_x = (input_width - crop_width) // 2
                crop_y = (input_height - crop_height) // 2
            elif crop_strategy == "top":
                crop_x = (input_width - crop_width) // 2
                crop_y = 0
            elif crop_strategy == "bottom":
                crop_x = (input_width - crop_width) // 2
                crop_y = input_height - crop_height
            else:
                crop_x = (input_width - crop_width) // 2
                crop_y = (input_height - crop_height) // 2
            
            # Apply crop and resize with explicit stream mapping for horizontal conversion
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-filter:v', f'crop={crop_width}:{crop_height}:{crop_x}:{crop_y},scale={target_width * 100}:{target_height * 100}',
                '-map', '0:v:0',              # Map only first video stream
                '-map', '0:a:0',              # Map only first audio stream  
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-crf', '23',
                '-preset', 'medium',
                '-y',
                output_path
            ]
            
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(f"FFmpeg horizontal conversion failed: {result.stderr}")
                return False
            
            self.logger.info(f"Converted to horizontal format {target_width}:{target_height}: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to convert to horizontal: {e}")
            return False
    
    def resize_video(self, 
                    input_path: str,
                    output_path: str,
                    target_width: int,
                    target_height: int,
                    maintain_aspect: bool = True) -> bool:
        """
        Resize video to target dimensions.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_width: Target width in pixels
            target_height: Target height in pixels
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if maintain_aspect:
                # Use scale filter with force_original_aspect_ratio
                (
                    ffmpeg
                    .input(input_path)
                    .filter('scale', target_width, target_height, force_original_aspect_ratio='decrease')
                    .filter('pad', target_width, target_height, -1, -1, 'black')
                    .output(output_path, vcodec='libx264', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
            else:
                # Direct resize
                (
                    ffmpeg
                    .input(input_path)
                    .filter('scale', target_width, target_height)
                    .output(output_path, vcodec='libx264', acodec='aac')
                    .overwrite_output()
                    .run(quiet=True)
                )
            
            self.logger.info(f"Resized video: {target_width}x{target_height}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resize video: {e}")
            return False
    
    def compress_video(self, 
                      input_path: str,
                      output_path: str,
                      target_bitrate: str = "2M",
                      quality: str = "medium") -> bool:
        """
        Compress video to reduce file size.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_bitrate: Target bitrate (e.g., "2M", "1M")
            quality: Compression quality preset
            
        Returns:
            True if successful, False otherwise
        """
        try:
            preset_map = {
                'fast': 'ultrafast',
                'medium': 'medium',
                'high': 'slow',
                'best': 'veryslow'
            }
            
            preset = preset_map.get(quality, 'medium')
            
            (
                ffmpeg
                .input(input_path)
                .output(output_path, 
                       vcodec='libx264', 
                       acodec='aac',
                       **{'b:v': target_bitrate, 'preset': preset, 'crf': '23'})
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Compressed video: {target_bitrate} bitrate")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to compress video: {e}")
            return False
    
    def add_fade_effects(self, 
                        input_path: str,
                        output_path: str,
                        fade_in_duration: float = 0.5,
                        fade_out_duration: float = 0.5) -> bool:
        """
        Add fade in/out effects to video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            fade_in_duration: Fade in duration in seconds
            fade_out_duration: Fade out duration in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            info = self.get_video_info(input_path)
            duration = info['duration']
            
            fade_out_start = duration - fade_out_duration
            
            (
                ffmpeg
                .input(input_path)
                .filter('fade', type='in', duration=fade_in_duration)
                .filter('fade', type='out', start_time=fade_out_start, duration=fade_out_duration)
                .output(output_path, vcodec='libx264', acodec='aac')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Added fade effects: {fade_in_duration}s in, {fade_out_duration}s out")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add fade effects: {e}")
            return False
    
    def normalize_audio(self, 
                       input_path: str,
                       output_path: str,
                       target_lufs: float = -16.0) -> bool:
        """
        Normalize audio levels in video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            target_lufs: Target loudness in LUFS
            
        Returns:
            True if successful, False otherwise
        """
        try:
            (
                ffmpeg
                .input(input_path)
                .filter('loudnorm', I=target_lufs, LRA=11, TP=-1.5)
                .output(output_path, vcodec='copy')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Normalized audio: {target_lufs} LUFS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to normalize audio: {e}")
            return False
    
    def create_preview_thumbnail(self, 
                               input_path: str,
                               output_path: str,
                               timestamp: float = None) -> bool:
        """
        Create a thumbnail image from video.
        
        Args:
            input_path: Path to input video
            output_path: Path for output thumbnail
            timestamp: Time position for thumbnail (None for middle)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if timestamp is None:
                info = self.get_video_info(input_path)
                timestamp = info['duration'] / 2
            
            (
                ffmpeg
                .input(input_path, ss=timestamp)
                .output(output_path, vframes=1, format='image2')
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Created thumbnail at {timestamp:.2f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create thumbnail: {e}")
            return False
    
    def validate_video_file(self, video_path: str) -> bool:
        """
        Validate that a video file is playable.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            info = self.get_video_info(video_path)
            return bool(info and info.get('duration', 0) > 0)
            
        except Exception as e:
            self.logger.error(f"Video validation failed: {e}")
            return False
    
    def get_processing_stats(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            'temp_dir': str(self.temp_dir),
            'output_dir': str(self.output_dir),
            'temp_files': len(list(self.temp_dir.glob('*'))),
            'output_files': len(list(self.output_dir.glob('*.mp4')))
        }
    
    def cleanup_temp_files(self) -> int:
        """
        Clean up temporary files.
        
        Returns:
            Number of files cleaned up
        """
        count = 0
        try:
            for temp_file in self.temp_dir.glob('*'):
                if temp_file.is_file():
                    temp_file.unlink()
                    count += 1
            
            self.logger.info(f"Cleaned up {count} temporary files")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {e}")
            return 0
