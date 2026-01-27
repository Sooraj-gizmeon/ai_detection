# src/video_processing/video_converter.py
"""Video conversion utilities for format and aspect ratio changes"""

import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, Optional
import subprocess
import tempfile


class VideoConverter:
    """
    Handles video format conversion and aspect ratio changes.
    """
    
    def __init__(self, temp_dir: str = "temp"):
        """
        Initialize VideoConverter.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def convert_to_vertical(self, 
                          input_path: str,
                          output_path: str,
                          target_aspect_ratio: Tuple[int, int] = (9, 16),
                          crop_method: str = "center") -> Dict:
        """
        Convert horizontal video to vertical format.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            target_aspect_ratio: Target aspect ratio (width, height)
            crop_method: Cropping method ("center", "smart", "top", "bottom")
            
        Returns:
            Conversion results
        """
        try:
            # Get video info
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Calculate target dimensions
            target_width, target_height = self._calculate_target_dimensions(
                original_width, original_height, target_aspect_ratio
            )
            
            # Use ffmpeg for conversion
            cmd = [
                'ffmpeg', '-i', input_path,
                '-vf', self._build_filter_string(
                    original_width, original_height,
                    target_width, target_height,
                    crop_method
                ),
                '-c:a', 'copy',  # Copy audio without re-encoding
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'input_dimensions': (original_width, original_height),
                    'output_dimensions': (target_width, target_height),
                    'fps': fps,
                    'frame_count': frame_count,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Video conversion failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error converting video: {e}")
            raise
    
    def _calculate_target_dimensions(self, 
                                   original_width: int,
                                   original_height: int,
                                   target_aspect_ratio: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculate target dimensions based on aspect ratio.
        
        Args:
            original_width: Original video width
            original_height: Original video height
            target_aspect_ratio: Target aspect ratio (width, height)
            
        Returns:
            Target dimensions (width, height)
        """
        target_ratio = target_aspect_ratio[0] / target_aspect_ratio[1]
        
        # Use original height as base, calculate width
        target_height = original_height
        target_width = int(target_height * target_ratio)
        
        # Ensure even dimensions for video encoding
        target_width = target_width - (target_width % 2)
        target_height = target_height - (target_height % 2)
        
        return target_width, target_height
    
    def _build_filter_string(self,
                           original_width: int,
                           original_height: int,
                           target_width: int,
                           target_height: int,
                           crop_method: str) -> str:
        """
        Build FFmpeg filter string for video conversion.
        
        Args:
            original_width: Original video width
            original_height: Original video height
            target_width: Target video width
            target_height: Target video height
            crop_method: Cropping method
            
        Returns:
            FFmpeg filter string
        """
        if crop_method == "center":
            # Center crop
            crop_x = (original_width - target_width) // 2
            crop_y = (original_height - target_height) // 2
            return f"crop={target_width}:{target_height}:{crop_x}:{crop_y}"
        
        elif crop_method == "top":
            # Crop from top
            crop_x = (original_width - target_width) // 2
            return f"crop={target_width}:{target_height}:{crop_x}:0"
        
        elif crop_method == "bottom":
            # Crop from bottom
            crop_x = (original_width - target_width) // 2
            crop_y = original_height - target_height
            return f"crop={target_width}:{target_height}:{crop_x}:{crop_y}"
        
        else:  # Default to center
            crop_x = (original_width - target_width) // 2
            crop_y = (original_height - target_height) // 2
            return f"crop={target_width}:{target_height}:{crop_x}:{crop_y}"
    
    def extract_segment(self,
                       input_path: str,
                       output_path: str,
                       start_time: float,
                       end_time: float) -> Dict:
        """
        Extract a segment from video.
        
        Args:
            input_path: Path to input video
            output_path: Path to output segment
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Extraction results
        """
        try:
            duration = end_time - start_time
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Copy without re-encoding
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Segment extraction failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error extracting segment: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information dictionary
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'codec': cap.get(cv2.CAP_PROP_FOURCC),
                'path': video_path
            }
            
            cap.release()
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting video info: {e}")
            raise
    
    def resize_video(self,
                    input_path: str,
                    output_path: str,
                    width: int,
                    height: int,
                    maintain_aspect: bool = True) -> Dict:
        """
        Resize video to specific dimensions.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            width: Target width
            height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resize results
        """
        try:
            if maintain_aspect:
                filter_str = f"scale={width}:{height}:force_original_aspect_ratio=decrease"
            else:
                filter_str = f"scale={width}:{height}"
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', filter_str,
                '-c:a', 'copy',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'target_dimensions': (width, height),
                    'maintain_aspect': maintain_aspect,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Video resize failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error resizing video: {e}")
            raise
    
    def compress_video(self,
                      input_path: str,
                      output_path: str,
                      quality: str = "medium") -> Dict:
        """
        Compress video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            quality: Compression quality ("low", "medium", "high")
            
        Returns:
            Compression results
        """
        try:
            # Quality settings
            quality_settings = {
                "low": {"crf": "28", "preset": "fast"},
                "medium": {"crf": "23", "preset": "medium"},
                "high": {"crf": "18", "preset": "slow"}
            }
            
            settings = quality_settings.get(quality, quality_settings["medium"])
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',
                '-crf', settings["crf"],
                '-preset', settings["preset"],
                '-c:a', 'aac',
                '-b:a', '128k',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Get file sizes for compression ratio
                input_size = Path(input_path).stat().st_size
                output_size = Path(output_path).stat().st_size
                compression_ratio = output_size / input_size
                
                return {
                    'success': True,
                    'quality': quality,
                    'input_size': input_size,
                    'output_size': output_size,
                    'compression_ratio': compression_ratio,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Video compression failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error compressing video: {e}")
            raise
    
    def add_audio_to_video(self,
                          video_path: str,
                          audio_path: str,
                          output_path: str,
                          audio_offset: float = 0.0) -> Dict:
        """
        Add audio track to video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            output_path: Path to output file
            audio_offset: Audio offset in seconds
            
        Returns:
            Results of audio addition
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest'  # Use shortest stream
            ]
            
            if audio_offset != 0.0:
                cmd.extend(['-itsoffset', str(audio_offset)])
            
            cmd.extend(['-y', output_path])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'video_path': video_path,
                    'audio_path': audio_path,
                    'audio_offset': audio_offset,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Audio addition failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error adding audio to video: {e}")
            raise
    
    def create_video_from_images(self,
                               image_paths: list,
                               output_path: str,
                               fps: float = 30.0,
                               duration_per_image: float = 1.0) -> Dict:
        """
        Create video from a sequence of images.
        
        Args:
            image_paths: List of image file paths
            output_path: Path to output video
            fps: Frames per second
            duration_per_image: Duration each image should be shown
            
        Returns:
            Video creation results
        """
        try:
            # Create temporary file list for ffmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for image_path in image_paths:
                    f.write(f"file '{image_path}'\n")
                    f.write(f"duration {duration_per_image}\n")
                file_list_path = f.name
            
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-vf', f'fps={fps}',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temporary file
            Path(file_list_path).unlink()
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'image_count': len(image_paths),
                    'fps': fps,
                    'duration_per_image': duration_per_image,
                    'total_duration': len(image_paths) * duration_per_image,
                    'output_path': output_path
                }
            else:
                self.logger.error(f"FFmpeg error: {result.stderr}")
                raise Exception(f"Video creation failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error creating video from images: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            import shutil
            temp_files = list(self.temp_dir.glob("*.mp4"))
            temp_files.extend(list(self.temp_dir.glob("*.avi")))
            temp_files.extend(list(self.temp_dir.glob("*.mov")))
            
            for temp_file in temp_files:
                temp_file.unlink()
                
            self.logger.info(f"Cleaned up {len(temp_files)} temporary video files")
            
        except Exception as e:
            self.logger.warning(f"Error cleaning up temporary files: {e}")
