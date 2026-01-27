"""
Podcast Template Processor for speaker-based video cropping.

This module handles the detection of speakers in video frames and creates
split-screen vertical videos with left speaker on top and right speaker on bottom.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import subprocess
import tempfile
import os

# Import detection components
import sys
sys.path.append(str(Path(__file__).parents[1]))
from subject_detection import SubjectDetector


class PodcastTemplateProcessor:
    """
    Processes videos using podcast template layout:
    - Detects 2 speakers in frames
    - Crops left speaker to top half of vertical video
    - Crops right speaker to bottom half of vertical video
    """
    
    def __init__(self, 
                 output_width: int = 1080, 
                 output_height: int = 1920,
                 speaker_detection_threshold: float = 0.6,
                 force_split_when_two_present: bool = True):
        """
        Initialize podcast template processor.
        
        Args:
            output_width: Width of output vertical video
            output_height: Height of output vertical video  
            speaker_detection_threshold: Confidence threshold for speaker detection
            force_split_when_two_present: Whether to force split-screen when two speakers are present
        """
        self.output_width = output_width
        self.output_height = output_height
        self.speaker_threshold = speaker_detection_threshold
        self.force_split_when_two_present = force_split_when_two_present
        
        # Half height for each speaker
        self.speaker_height = output_height // 2
        
        # Initialize subject detector for speaker detection with CUDA preference
        self.subject_detector = SubjectDetector(device="cuda")
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PodcastTemplateProcessor: {output_width}x{output_height}")
    
    def _iou(self, a: List[int], b: List[int]) -> float:
        """Compute IoU between two bboxes [x1,y1,x2,y2]."""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
    
    def _dedupe_speakers(self, speakers: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """Remove duplicate detections of the same person by IoU and keep higher confidence."""
        if len(speakers) <= 1:
            return speakers
        speakers_sorted = sorted(speakers, key=lambda s: s.get('confidence', 0), reverse=True)
        kept: List[Dict] = []
        for s in speakers_sorted:
            if not any(self._iou(s['bbox'], k['bbox']) > iou_thresh for k in kept):
                kept.append(s)
        return kept
    
    def detect_speakers_in_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect speakers in a video frame and determine layout type.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with speaker information and layout type, or None if no speakers found
        """
        try:
            # Detect all subjects in frame
            subjects = self.subject_detector.detect_subjects(frame)
            
            # Normalize subjects: treat YOLO 'person' objects as 'person' subjects
            normalized: List[Dict] = []
            for subj in subjects:
                stype = subj.get('type')
                if stype == 'object' and subj.get('class') == 'person':
                    normalized.append({
                        'type': 'person',
                        'bbox': subj['bbox'],
                        'confidence': subj.get('confidence', 0.0),
                        'tracking_id': subj.get('tracking_id')
                    })
                else:
                    normalized.append(subj)
            
            # Filter for faces/people with sufficient confidence
            speakers = []
            for subject in normalized:
                if (subject.get('type') in ['face', 'person'] and 
                    subject.get('confidence', 0.0) >= self.speaker_threshold):
                    speakers.append(subject)
            
            # De-duplicate overlapping detections (e.g., face+person for same individual)
            speakers = self._dedupe_speakers(speakers)
            
            # Return different layouts based on speaker count
            if len(speakers) == 0:
                return None
            elif len(speakers) == 1:
                return {
                    'layout_type': 'single_speaker',
                    'primary_speaker': speakers[0],
                    'speaker_count': 1,
                    'confidence': speakers[0]['confidence']
                }
            else:
                # Prefer the two most confident distinct speakers, left-to-right
                speakers = sorted(speakers, key=lambda s: s.get('confidence', 0.0), reverse=True)
                top_two = speakers[:2]
                top_two.sort(key=lambda s: s['bbox'][0])
                left_speaker, right_speaker = top_two[0], top_two[1]
                
                # After dedupe
                if len(speakers) >= 2 and self.force_split_when_two_present:
                    speakers = sorted(speakers, key=lambda s: s.get('confidence', 0.0), reverse=True)[:2]
                    speakers.sort(key=lambda s: s['bbox'][0])
                    left_speaker, right_speaker = speakers[0], speakers[1]
                    return {
                        'layout_type': 'split_screen',
                        'left_speaker': left_speaker,
                        'right_speaker': right_speaker,
                        'speaker_count': 2,
                        'confidence_avg': (left_speaker['confidence'] + right_speaker['confidence']) / 2
                    }
                
                return {
                    'layout_type': 'split_screen',
                    'left_speaker': left_speaker,
                    'right_speaker': right_speaker,
                    'speaker_count': 2,
                    'confidence_avg': (left_speaker['confidence'] + right_speaker['confidence']) / 2
                }
            
        except Exception as e:
            self.logger.error(f"Speaker detection failed: {e}")
            return None
    
    def calculate_speaker_crop_regions(self, 
                                     frame_width: int, 
                                     frame_height: int,
                                     speaker_info: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Calculate crop regions for each speaker to fit in vertical layout.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            speaker_info: Speaker detection information
            
        Returns:
            Tuple of (left_speaker_crop, right_speaker_crop) dictionaries
        """
        left_speaker = speaker_info['left_speaker']
        right_speaker = speaker_info['right_speaker']
        
        # Get bounding boxes
        left_bbox = left_speaker['bbox']  # [x1, y1, x2, y2]
        right_bbox = right_speaker['bbox']
        
        # Calculate crop regions with padding
        padding = 50  # pixels
        
        # Left speaker crop (goes to top half)
        left_crop = {
            'x1': max(0, left_bbox[0] - padding),
            'y1': max(0, left_bbox[1] - padding),
            'x2': min(frame_width, left_bbox[2] + padding),
            'y2': min(frame_height, left_bbox[3] + padding),
            'target_position': 'top',  # Top half of vertical video
            'target_y': 0,
            'target_height': self.speaker_height
        }
        
        # Right speaker crop (goes to bottom half)
        right_crop = {
            'x1': max(0, right_bbox[0] - padding),
            'y1': max(0, right_bbox[1] - padding),
            'x2': min(frame_width, right_bbox[2] + padding),
            'y2': min(frame_height, right_bbox[3] + padding),
            'target_position': 'bottom',  # Bottom half of vertical video
            'target_y': self.speaker_height,
            'target_height': self.speaker_height
        }
        
        return left_crop, right_crop
    
    def calculate_single_speaker_crop(self, 
                                    frame_width: int, 
                                    frame_height: int,
                                    speaker_info: Dict[str, Any]) -> Dict:
        """
        Calculate crop region for single speaker to fill entire vertical video.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            speaker_info: Single speaker detection information
            
        Returns:
            Dictionary with crop information for the single speaker
        """
        speaker = speaker_info['primary_speaker']
        bbox = speaker['bbox']  # [x1, y1, x2, y2]
        
        # Calculate crop region with padding - centered on speaker
        padding = 100  # More padding for single speaker to show more context
        
        # Calculate speaker center
        speaker_center_x = (bbox[0] + bbox[2]) // 2
        speaker_center_y = (bbox[1] + bbox[3]) // 2
        
        # Calculate crop dimensions to maintain vertical aspect ratio
        target_aspect = self.output_width / self.output_height  # 9:16 = 0.5625
        
        # Try to center crop around speaker
        crop_height = frame_height
        crop_width = int(crop_height * target_aspect)
        
        # If calculated width is too wide, adjust based on width
        if crop_width > frame_width:
            crop_width = frame_width
            crop_height = int(crop_width / target_aspect)
        
        # Center the crop around the speaker
        crop_x1 = max(0, speaker_center_x - crop_width // 2)
        crop_x2 = min(frame_width, crop_x1 + crop_width)
        
        # Adjust if crop goes out of bounds
        if crop_x2 > frame_width:
            crop_x2 = frame_width
            crop_x1 = max(0, crop_x2 - crop_width)
        
        crop_y1 = max(0, speaker_center_y - crop_height // 2)
        crop_y2 = min(frame_height, crop_y1 + crop_height)
        
        # Adjust if crop goes out of bounds
        if crop_y2 > frame_height:
            crop_y2 = frame_height
            crop_y1 = max(0, crop_y2 - crop_height)
        
        return {
            'x1': crop_x1,
            'y1': crop_y1,
            'x2': crop_x2,
            'y2': crop_y2,
            'target_position': 'full_screen',
            'target_y': 0,
            'target_height': self.output_height
        }
    
    def create_podcast_layout_frame(self, 
                                  frame: np.ndarray,
                                  speaker_info: Dict[str, Any]) -> np.ndarray:
        """
        Create a frame with appropriate podcast layout based on detected speakers.
        
        Args:
            frame: Input video frame
            speaker_info: Speaker detection information with layout type
            
        Returns:
            Processed frame with appropriate layout
        """
        frame_height, frame_width = frame.shape[:2]
        layout_type = speaker_info['layout_type']
        
        if layout_type == 'single_speaker':
            # Single speaker - crop and center around speaker
            return self._create_single_speaker_frame(frame, speaker_info)
        elif layout_type == 'split_screen':
            # Two speakers - create split screen layout
            return self._create_split_screen_frame(frame, speaker_info)
        else:
            # Fallback to center crop
            return self._center_crop_fallback(frame)
    
    def _create_single_speaker_frame(self, 
                                   frame: np.ndarray,
                                   speaker_info: Dict[str, Any]) -> np.ndarray:
        """
        Create a frame focused on a single speaker.
        
        Args:
            frame: Input video frame
            speaker_info: Single speaker information
            
        Returns:
            Processed frame with single speaker layout
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate crop region for single speaker
        crop_info = self.calculate_single_speaker_crop(
            frame_width, frame_height, speaker_info
        )
        
        # Extract and resize the crop region
        crop_x1, crop_y1 = crop_info['x1'], crop_info['y1']
        crop_x2, crop_y2 = crop_info['x2'], crop_info['y2']
        
        cropped_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if cropped_region.size == 0:
            return self._center_crop_fallback(frame)
        
        # Resize to target dimensions
        resized_frame = cv2.resize(cropped_region, (self.output_width, self.output_height))
        
        return resized_frame
    
    def _create_split_screen_frame(self, 
                                 frame: np.ndarray,
                                 speaker_info: Dict[str, Any]) -> np.ndarray:
        """
        Create a split-screen frame with two speakers.
        
        Args:
            frame: Input video frame
            speaker_info: Two-speaker information
            
        Returns:
            Processed frame with split-screen layout
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate crop regions for both speakers
        left_crop, right_crop = self.calculate_speaker_crop_regions(
            frame_width, frame_height, speaker_info
        )
        
        # Create output frame
        output_frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # Crop and resize speakers
        for crop_info in [left_crop, right_crop]:
            # Extract crop region
            crop_x1, crop_y1 = crop_info['x1'], crop_info['y1']
            crop_x2, crop_y2 = crop_info['x2'], crop_info['y2']
            
            cropped_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if cropped_region.size == 0:
                continue
            
            # Resize to fit target area (maintaining aspect ratio)
            target_height = crop_info['target_height']
            target_y = crop_info['target_y']
            
            # Calculate scaling to fit width while maintaining aspect ratio
            crop_height, crop_width = cropped_region.shape[:2]
            scale_x = self.output_width / crop_width
            scale_y = target_height / crop_height
            scale = min(scale_x, scale_y)  # Maintain aspect ratio
            
            new_width = int(crop_width * scale)
            new_height = int(crop_height * scale)
            
            # Resize the cropped region
            resized_region = cv2.resize(cropped_region, (new_width, new_height))
            
            # Center the resized region in target area
            x_offset = (self.output_width - new_width) // 2
            y_offset = target_y + (target_height - new_height) // 2
            
            # Place in output frame
            if (y_offset + new_height <= self.output_height and 
                x_offset + new_width <= self.output_width):
                output_frame[y_offset:y_offset + new_height, 
                           x_offset:x_offset + new_width] = resized_region
        
        return output_frame
    
    def process_video_with_podcast_template(self, 
                                          input_video_path: str,
                                          output_video_path: str) -> bool:
        """
        Process entire video with podcast template layout.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Processing video with podcast template: {input_video_path}")
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video: {input_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer with reliable codec
            try:
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            except:
                # Fallback to XVID if H264 not available
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                except:
                    # Final fallback to mp4v (though less reliable)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_video_path,
                fourcc, 
                fps,
                (self.output_width, self.output_height)
            )
            
            frames_processed = 0
            frames_with_speakers = 0
            frames_with_split_screen = 0
            frames_with_single_speaker = 0
            
            self.logger.info(f"Processing {total_frames} frames at {fps} FPS")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect speakers in frame
                speaker_info = self.detect_speakers_in_frame(frame)
                
                if speaker_info:
                    # Create appropriate layout based on speaker detection
                    layout_type = speaker_info['layout_type']
                    processed_frame = self.create_podcast_layout_frame(frame, speaker_info)
                    
                    frames_with_speakers += 1
                    if layout_type == 'split_screen':
                        frames_with_split_screen += 1
                    elif layout_type == 'single_speaker':
                        frames_with_single_speaker += 1
                else:
                    # No speakers detected, use center crop as fallback
                    processed_frame = self._center_crop_fallback(frame)
                
                # Write frame
                out.write(processed_frame)
                frames_processed += 1
                
                # Log progress
                if frames_processed % 100 == 0:
                    self.logger.info(f"Processed {frames_processed}/{total_frames} frames "
                                   f"({frames_with_speakers} with speakers: "
                                   f"{frames_with_split_screen} split-screen, "
                                   f"{frames_with_single_speaker} single-speaker)")
            
            # Cleanup
            cap.release()
            out.release()
            
            # Convert to final format with audio using FFmpeg
            self._finalize_video_with_audio(output_video_path, input_video_path)
            
            self.logger.info(f"Podcast template processing complete:")
            self.logger.info(f"  Total frames: {frames_processed}")
            self.logger.info(f"  Frames with speakers: {frames_with_speakers}")
            self.logger.info(f"  Split-screen frames: {frames_with_split_screen}")
            self.logger.info(f"  Single-speaker frames: {frames_with_single_speaker}")
            self.logger.info(f"  Fallback frames: {frames_processed - frames_with_speakers}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Podcast template processing failed: {e}")
            return False
    
    def _center_crop_fallback(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback method when speakers are not detected - use center crop.
        
        Args:
            frame: Input frame
            
        Returns:
            Center-cropped frame
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate center crop for vertical aspect ratio
        target_aspect = self.output_width / self.output_height  # 9:16 = 0.5625
        frame_aspect = frame_width / frame_height
        
        if frame_aspect > target_aspect:
            # Frame is wider, crop width
            new_width = int(frame_height * target_aspect)
            x_offset = (frame_width - new_width) // 2
            cropped = frame[:, x_offset:x_offset + new_width]
        else:
            # Frame is taller, crop height
            new_height = int(frame_width / target_aspect)
            y_offset = (frame_height - new_height) // 2
            cropped = frame[y_offset:y_offset + new_height, :]
        
        # Resize to target dimensions
        resized = cv2.resize(cropped, (self.output_width, self.output_height))
        return resized
    
    def _finalize_video_with_audio(self, temp_video_path: str, original_video_path: str):
        """
        Use FFmpeg to add audio from original video to processed video.
        
        Args:
            temp_video_path: Path to processed video (no audio)
            original_video_path: Path to original video (with audio)
        """
        try:
            # Create final output path
            final_path = temp_video_path.replace('.mp4', '_final.mp4')
            
            # FFmpeg command to combine video and audio
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', temp_video_path,  # Processed video (no audio)
                '-i', original_video_path,  # Original video (with audio)
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Encode audio to AAC
                '-map', '0:v:0', # Use video from first input
                '-map', '1:a:0', # Use audio from second input
                '-shortest',     # Match shortest stream duration
                final_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Replace temp file with final file
                os.replace(final_path, temp_video_path)
                self.logger.info("Audio successfully added to podcast video")
            else:
                self.logger.error(f"FFmpeg failed: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Failed to add audio: {e}")


def apply_podcast_template_if_needed(input_video_path: str, 
                                   output_video_path: str,
                                   template: Optional[str] = None) -> str:
    """
    Apply podcast template if specified in task configuration.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        template: Template type ('podcast' or None)
        
    Returns:
        Path to final video (processed or original)
    """
    if template != 'podcast':
        # No podcast template, return original path
        return input_video_path
    
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Applying podcast template to video: {input_video_path}")
        
        # Create podcast processor
        processor = PodcastTemplateProcessor()
        
        # Process video with podcast template
        success = processor.process_video_with_podcast_template(
            input_video_path,
            output_video_path
        )
        
        if success and os.path.exists(output_video_path):
            logger.info(f"Podcast template applied successfully: {output_video_path}")
            return output_video_path
        else:
            logger.warning("Podcast template processing failed, using original video")
            return input_video_path
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error applying podcast template: {e}")
        return input_video_path
