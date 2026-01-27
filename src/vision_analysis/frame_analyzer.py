# src/vision_analysis/frame_analyzer.py
"""Frame analysis for extracting key frames and visual information from videos"""

import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time


class FrameAnalyzer:
    """
    Analyzes video frames to extract key visual information for Ollama vision analysis.
    Implements efficient frame sampling and preprocessing.
    """
    
    def __init__(self, 
                 target_width: int = 512, 
                 target_height: int = 288,
                 sample_rate: float = 0.5):
        """
        Initialize frame analyzer.
        
        Args:
            target_width: Target width for processed frames
            target_height: Target height for processed frames  
            sample_rate: Frame sampling rate (frames per second)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.sample_rate = sample_rate
        
        self.logger = logging.getLogger(__name__)
        
    def extract_key_frames(self, video_path: str, timestamps: List[float] = None) -> List[Dict]:
        """
        Extract key frames from video at specified timestamps or with regular sampling.
        
        Args:
            video_path: Path to video file
            timestamps: Specific timestamps to extract frames from (optional)
            
        Returns:
            List of frame data dictionaries
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            self.logger.info(f"Video info: {fps:.2f} fps, {total_frames} frames, {duration:.2f}s duration")
            
            frames = []
            
            if timestamps:
                # Extract frames at specific timestamps
                for timestamp in timestamps:
                    frame_data = self._extract_frame_at_timestamp(cap, timestamp, fps)
                    if frame_data:
                        frames.append(frame_data)
            else:
                # Regular sampling based on sample_rate
                frame_interval = int(fps / self.sample_rate) if fps > 0 else 30
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_interval == 0:
                        timestamp = frame_count / fps if fps > 0 else frame_count
                        processed_frame = self._process_frame(frame, timestamp)
                        if processed_frame:
                            frames.append(processed_frame)
                    
                    frame_count += 1
            
            cap.release()
            
            self.logger.info(f"Extracted {len(frames)} key frames from {video_path}")
            return frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames from {video_path}: {e}")
            return []
    
    def _extract_frame_at_timestamp(self, cap: cv2.VideoCapture, timestamp: float, fps: float) -> Optional[Dict]:
        """Extract a single frame at a specific timestamp."""
        try:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                return self._process_frame(frame, timestamp)
            else:
                self.logger.warning(f"Could not extract frame at timestamp {timestamp}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting frame at timestamp {timestamp}: {e}")
            return None
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Process a single frame for analysis with enhanced quality scoring."""
        try:
            # Resize frame for efficiency while maintaining aspect ratio
            height, width = frame.shape[:2]
            if width > self.target_width or height > self.target_height:
                # Calculate optimal resize ratio
                width_ratio = self.target_width / width
                height_ratio = self.target_height / height
                resize_ratio = min(width_ratio, height_ratio)
                
                new_width = int(width * resize_ratio)
                new_height = int(height * resize_ratio)
                
                resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized_frame = frame.copy()
            
            # Convert to base64 for Ollama API with optimized quality
            _, buffer = cv2.imencode('.jpg', resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Enhanced visual quality analysis
            gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            contrast = np.std(gray_frame)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            
            # Detect if frame has significant visual content
            non_zero_pixels = np.count_nonzero(gray_frame)
            total_pixels = gray_frame.size
            content_ratio = non_zero_pixels / total_pixels
            
            # Enhanced quality score calculation
            quality_score = self._calculate_enhanced_quality(
                brightness, contrast, laplacian_var, content_ratio
            )
            
            return {
                'timestamp': timestamp,
                'base64': frame_base64,
                'width': resized_frame.shape[1],
                'height': resized_frame.shape[0],
                'brightness': float(brightness),
                'contrast': float(contrast),
                'sharpness': float(laplacian_var),
                'content_ratio': float(content_ratio),
                'quality_score': quality_score
            }
            
        except Exception as e:
            self.logger.error(f"Error processing frame at timestamp {timestamp}: {e}")
            return None
    
    def _calculate_enhanced_quality(self, brightness: float, contrast: float, sharpness: float, content_ratio: float) -> float:
        """
        Calculate enhanced frame quality score based on multiple factors.
        
        Args:
            brightness: Average brightness (0-255)
            contrast: Standard deviation of pixel values
            sharpness: Laplacian variance (focus measure)
            content_ratio: Ratio of non-zero pixels
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            # Normalize brightness score (penalize over/under exposure)
            normalized_brightness = brightness / 255.0
            brightness_score = 1.0 - abs(normalized_brightness - 0.5) * 2
            
            # Normalize contrast score (higher contrast generally better)
            contrast_score = min(1.0, contrast / 100.0)  # Assume good contrast around 100
            
            # Normalize sharpness score (higher variance = sharper image)
            sharpness_score = min(1.0, sharpness / 1000.0)  # Normalize based on typical values
            
            # Content ratio score (should be close to 1 for good frames)
            content_score = content_ratio
            
            # Weighted combination of scores
            quality_score = (
                brightness_score * 0.25 +
                contrast_score * 0.25 + 
                sharpness_score * 0.3 +  # Sharpness is important for vision models
                content_score * 0.2
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced quality: {e}")
            return 0.5  # Default medium quality
    
    def _calculate_frame_quality(self, brightness: float, contrast: float, content_ratio: float) -> float:
        """Calculate a quality score for the frame."""
        # Normalize metrics to 0-1 range and combine
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # Prefer mid-range brightness
        contrast_score = min(contrast / 50.0, 1.0)  # Higher contrast is better, cap at 50
        content_score = content_ratio  # Higher content ratio is better
        
        # Weighted combination
        quality = (brightness_score * 0.3 + contrast_score * 0.4 + content_score * 0.3)
        return max(0.0, min(1.0, quality))
    
    def extract_frames_for_segments(self, video_path: str, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Extract representative frames for video segments.
        
        Args:
            video_path: Path to video file
            segments: List of segment dictionaries with start_time and end_time
            
        Returns:
            Dictionary mapping segment IDs to frame lists
        """
        segment_frames = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return segment_frames
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            for i, segment in enumerate(segments):
                segment_id = f"segment_{i}"
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                
                # Extract 2-3 representative frames per segment
                duration = end_time - start_time
                
                if duration <= 0:
                    continue
                
                # Choose frame extraction strategy based on segment duration
                if duration <= 15:
                    # Short segment: extract 1-2 frames
                    timestamps = [start_time + duration * 0.3, start_time + duration * 0.7]
                elif duration <= 60:
                    # Medium segment: extract 2-3 frames
                    timestamps = [start_time + duration * 0.25, start_time + duration * 0.5, start_time + duration * 0.75]
                else:
                    # Long segment: extract 3-4 frames
                    timestamps = [start_time + duration * 0.2, start_time + duration * 0.4, 
                                start_time + duration * 0.6, start_time + duration * 0.8]
                
                frames = []
                for timestamp in timestamps:
                    frame_data = self._extract_frame_at_timestamp(cap, timestamp, fps)
                    if frame_data:
                        frames.append(frame_data)
                
                if frames:
                    segment_frames[segment_id] = frames
                    self.logger.debug(f"Extracted {len(frames)} frames for {segment_id} ({start_time:.1f}s-{end_time:.1f}s)")
            
            cap.release()
            
            self.logger.info(f"Extracted frames for {len(segment_frames)} segments")
            return segment_frames
            
        except Exception as e:
            self.logger.error(f"Error extracting segment frames from {video_path}: {e}")
            return segment_frames
    
    def create_segment_thumbnail(self, frames: List[Dict]) -> Optional[str]:
        """
        Create a single thumbnail image representing a segment.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            Base64 encoded thumbnail image
        """
        if not frames:
            return None
        
        try:
            # Select the highest quality frame
            best_frame = max(frames, key=lambda f: f.get('quality_score', 0))
            return best_frame['base64']
            
        except Exception as e:
            self.logger.error(f"Error creating segment thumbnail: {e}")
            return None
