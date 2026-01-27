# src/utils/video_utils.py
"""Video processing utilities and helper functions"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import ffmpeg
import subprocess
import tempfile
import os
from .brand_logo_overlay import BrandLogoOverlay


class VideoReader:
    """
    Video reader with frame-by-frame processing capabilities.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        self.current_frame = 0
        
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from video.
        
        Returns:
            Frame as numpy array or None if end of video
        """
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
            return frame
        return None
    
    def seek_to_frame(self, frame_number: int) -> bool:
        """
        Seek to specific frame number.
        
        Args:
            frame_number: Frame number to seek to
            
        Returns:
            True if successful, False otherwise
        """
        if 0 <= frame_number < self.frame_count:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False
    
    def seek_to_time(self, timestamp: float) -> bool:
        """
        Seek to specific timestamp.
        
        Args:
            timestamp: Timestamp in seconds
            
        Returns:
            True if successful, False otherwise
        """
        frame_number = int(timestamp * self.fps)
        return self.seek_to_frame(frame_number)
    
    def get_info(self) -> Dict:
        """
        Get video information.
        
        Returns:
            Dictionary with video metadata
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'width': self.width,
            'height': self.height,
            'duration': self.duration,
            'current_frame': self.current_frame
        }
    
    def close(self):
        """Close video reader."""
        if self.cap:
            self.cap.release()


class VideoWriter:
    """
    Video writer with audio preservation capabilities.
    """
    
    def __init__(self, output_path: str, fps: float = 30.0, target_aspect_ratio: Tuple[int, int] = (16, 9),
                 brand_logo: Optional[str] = None, logo_position: Optional[str] = None):
        """
        Initialize video writer.
        
        Args:
            output_path: Path for output video
            fps: Frames per second
            target_aspect_ratio: Target aspect ratio (width, height)
            brand_logo: Brand logo URL (optional)
            logo_position: Logo position ('tl', 'tr', 'bl', 'br', 'tc', 'bc') (optional)
        """
        self.output_path = output_path
        self.fps = fps
        self.target_aspect_ratio = target_aspect_ratio
        self.brand_logo = brand_logo
        self.logo_position = logo_position
        self.logger = logging.getLogger(__name__)
        
        # Calculate target dimensions
        self.target_width = 1080
        self.target_height = int(self.target_width * target_aspect_ratio[1] / target_aspect_ratio[0])
        
        # Create temporary video file (without audio)
        self.temp_video_path = str(Path(output_path).with_suffix('.temp.mp4'))
        
        # Initialize video writer for frames only
        # Use H264 codec for better FFmpeg compatibility
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        except:
            # Fallback to XVID if H264 not available
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        self.writer = cv2.VideoWriter(
            self.temp_video_path,
            fourcc,
            fps,
            (self.target_width, self.target_height)
        )
        
        # Verify writer was initialized successfully
        if not self.writer.isOpened():
            self.logger.error(f"Failed to initialize VideoWriter for {self.temp_video_path}")
            raise RuntimeError("VideoWriter initialization failed")
        
        self.frames_written = 0
        self.input_video_path = None  # Will be set later
    
    def set_input_video_path(self, input_path: str):
        """
        Set input video path for audio extraction.
        
        Args:
            input_path: Path to original input video with audio
        """
        self.input_video_path = input_path
    
    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video.
        
        Args:
            frame: Frame to write (should already be in target aspect ratio)
        """
        # Check if frame is already in target dimensions
        frame_h, frame_w = frame.shape[:2]
        
        if frame_w == self.target_width and frame_h == self.target_height:
            # Frame is already correct size
            self.writer.write(frame)
        else:
            # Frame needs resizing - but maintain aspect ratio
            target_aspect = self.target_width / self.target_height
            frame_aspect = frame_w / frame_h
            
            if abs(frame_aspect - target_aspect) < 0.01:  # Aspect ratios match
                # Simple resize to exact dimensions
                resized_frame = cv2.resize(frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LANCZOS4)
            else:
                # Aspect ratios don't match - need to pad or crop
                if frame_aspect > target_aspect:
                    # Frame is wider - crop width
                    new_width = int(frame_h * target_aspect)
                    start_x = (frame_w - new_width) // 2
                    cropped_frame = frame[:, start_x:start_x + new_width]
                else:
                    # Frame is taller - crop height
                    new_height = int(frame_w / target_aspect)
                    start_y = (frame_h - new_height) // 2
                    cropped_frame = frame[start_y:start_y + new_height, :]
                
                resized_frame = cv2.resize(cropped_frame, (self.target_width, self.target_height), interpolation=cv2.INTER_LANCZOS4)
            
            self.writer.write(resized_frame)
        
        self.frames_written += 1
    
    def close(self):
        """Close video writer and merge with audio."""
        if self.writer:
            self.writer.release()
        
        # Check if frames were actually written
        if self.frames_written == 0:
            self.logger.error("No frames were written to video")
            return
            
        # Validate the temp video file was created properly
        if not os.path.exists(self.temp_video_path):
            self.logger.error(f"Temp video file was not created: {self.temp_video_path}")
            return
        
        # Check temp file size
        temp_size = os.path.getsize(self.temp_video_path)
        if temp_size < 1000:  # Less than 1KB
            self.logger.error(f"Temp video file too small ({temp_size} bytes): {self.temp_video_path}")
            return
            
        self.logger.info(f"VideoWriter completed: {self.frames_written} frames, {temp_size} bytes")
            
        # Apply brand logo overlay if specified
        video_to_merge = self.temp_video_path
        if self.brand_logo and self.logo_position and self._validate_video_file(self.temp_video_path):
            try:
                logo_overlay = BrandLogoOverlay()
                temp_with_logo = str(Path(self.output_path).with_suffix('.logo.mp4'))
                
                # Determine canvas type based on aspect ratio
                canvas_type = "shorts" if self.target_aspect_ratio[1] > self.target_aspect_ratio[0] else "clips"
                
                logo_overlay.add_logo_to_video(
                    input_video_path=self.temp_video_path,
                    output_video_path=temp_with_logo,
                    logo_url=self.brand_logo,
                    logo_position=self.logo_position,  # Use legacy positioning for VideoWriter
                    canvas_type=canvas_type
                )
                # Update the video to merge to the one with logo
                if os.path.exists(temp_with_logo) and self._validate_video_file(temp_with_logo):
                    video_to_merge = temp_with_logo
                    self.logger.info(f"Applied brand logo overlay at position: {self.logo_position}")
            except Exception as e:
                self.logger.error(f"Failed to apply brand logo overlay: {e}")
                # Continue with original video
                video_to_merge = self.temp_video_path
            
        # Merge video with audio from original source
        if self.input_video_path and os.path.exists(video_to_merge):
            try:
                self._merge_with_audio(video_to_merge)
            except Exception as e:
                self.logger.error(f"Failed to merge audio: {e}")
                # Fallback: copy temp video to final output if valid
                if os.path.exists(video_to_merge) and self._validate_video_file(video_to_merge):
                    import shutil
                    shutil.copy2(video_to_merge, self.output_path)
                    self.logger.info(f"Fallback: copied video without audio")
        else:
            # No audio to merge, just copy temp file if valid
            if os.path.exists(video_to_merge) and self._validate_video_file(video_to_merge):
                import shutil
                shutil.copy2(video_to_merge, self.output_path)
                self.logger.info(f"Copied video without audio merge: {self.output_path}")
                
        # Clean up temporary files
        for temp_file in [self.temp_video_path, str(Path(self.output_path).with_suffix('.logo.mp4'))]:
            if os.path.exists(temp_file) and temp_file != self.output_path:
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def _validate_video_file(self, video_path: str) -> bool:
        """Validate that a video file is readable and has streams."""
        try:
            probe = ffmpeg.probe(video_path)
            video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
            if not video_streams:
                self.logger.error(f"No video streams found in {video_path}")
                return False
            
            # Check file size
            import os
            file_size = os.path.getsize(video_path)
            if file_size < 1000:  # Less than 1KB indicates corruption
                self.logger.error(f"Video file too small ({file_size} bytes): {video_path}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Video validation failed for {video_path}: {e}")
            return False
    
    def _merge_with_audio(self, video_file_path: str):
        """
        Merge processed video frames with original audio.
        
        Args:
            video_file_path: Path to the video file to merge with audio
        """
        try:
            # Validate the processed video file first
            if not self._validate_video_file(video_file_path):
                self.logger.error(f"Processed video file is invalid: {video_file_path}")
                # Don't attempt merge, just fail gracefully
                return
            
            # Check if input video has audio
            probe = ffmpeg.probe(self.input_video_path)
            has_audio = any(stream['codec_type'] == 'audio' for stream in probe['streams'])
            
            if has_audio:
                # Merge video and audio using subprocess with safer stream mapping
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', video_file_path,         # Video input
                    '-i', self.input_video_path,   # Audio input
                    '-map', '0:v:0?',              # Optional video stream mapping
                    '-map', '1:a:0?',              # Optional audio stream mapping
                    '-c:v', 'libx264',             # Re-encode video for compatibility
                    '-c:a', 'aac',                 # Re-encode audio for compatibility
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-preset', 'fast',
                    '-crf', '23',
                    '-shortest',                   # Match shortest stream duration
                    '-y',                          # Overwrite output
                    self.output_path
                ]
                
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"Successfully merged video with audio: {self.output_path}")
                    # Validate the final output
                    if not self._validate_video_file(self.output_path):
                        self.logger.warning(f"Merged output may be corrupted: {self.output_path}")
                else:
                    self.logger.error(f"FFmpeg merge failed: {result.stderr}")
                    # Fallback: copy video to final output if it's valid
                    if os.path.exists(video_file_path) and self._validate_video_file(video_file_path):
                        import shutil
                        shutil.copy2(video_file_path, self.output_path)
                        self.logger.info(f"Fallback: copied video without audio merge")
            else:
                # No audio in source, just copy processed video
                if self._validate_video_file(video_file_path):
                    import shutil
                    shutil.copy2(video_file_path, self.output_path)
                    self.logger.info(f"No audio in source, copied video only: {self.output_path}")
                else:
                    self.logger.error(f"Cannot copy invalid video file: {video_file_path}")
            
        except Exception as e:
            self.logger.error(f"Audio merging failed: {e}")
            # Fallback: rename video file to output
            if os.path.exists(video_file_path):
                os.rename(video_file_path, self.output_path)


def get_video_info(video_path: str) -> Dict:
    """
    Get video information without opening a reader.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {}
        
        info = {
            'path': video_path,
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': 0.0
        }
        
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        cap.release()
        return info
        
    except Exception as e:
        logging.error(f"Failed to get video info: {e}")
        return {}


def validate_video_files(directory: str) -> List[Dict]:
    """
    Validate all video files in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of validation results
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    results = []
    
    directory_path = Path(directory)
    
    for ext in video_extensions:
        for video_file in directory_path.glob(f"*{ext}"):
            info = get_video_info(str(video_file))
            
            result = {
                'file': str(video_file),
                'valid': bool(info),
                'info': info,
                'issues': []
            }
            
            if info:
                # Check for potential issues
                if info['duration'] < 1.0:
                    result['issues'].append('Video too short')
                
                if info['fps'] < 10:
                    result['issues'].append('Low frame rate')
                
                if info['width'] < 640 or info['height'] < 480:
                    result['issues'].append('Low resolution')
            else:
                result['issues'].append('Could not read video file')
            
            results.append(result)
    
    return results


def create_video_preview(video_path: str, output_path: str, num_frames: int = 9) -> bool:
    """
    Create a preview grid from video frames.
    
    Args:
        video_path: Path to input video
        output_path: Path for output preview image
        num_frames: Number of frames to include in preview
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame positions
        frame_positions = [int(i * frame_count / num_frames) for i in range(num_frames)]
        
        frames = []
        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                # Resize frame for preview
                frame = cv2.resize(frame, (200, 150))
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            return False
        
        # Create preview grid
        rows = int(np.sqrt(num_frames))
        cols = int(np.ceil(num_frames / rows))
        
        grid_height = rows * 150
        grid_width = cols * 200
        
        preview_grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        for i, frame in enumerate(frames):
            row = i // cols
            col = i % cols
            
            y1 = row * 150
            y2 = y1 + 150
            x1 = col * 200
            x2 = x1 + 200
            
            preview_grid[y1:y2, x1:x2] = frame
        
        # Save preview
        cv2.imwrite(output_path, preview_grid)
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to create video preview: {e}")
        return False
