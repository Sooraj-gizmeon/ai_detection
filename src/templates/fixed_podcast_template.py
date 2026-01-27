"""
Fixed Podcast Template Processor with Enhanced Person Splitting Logic

This module provides a robust implementation of the podcast template with proper:
- 2-person video splitting (left speaker → top, right speaker → bottom)
- Comprehensive error handling and logging
- Fallback mechanisms for edge cases
- Proper video file creation and audio muxing
"""

import cv2
import numpy as np
import logging
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Import detection components
import sys
sys.path.append(str(Path(__file__).parents[1]))
from subject_detection import SubjectDetector
from subject_detection.mediapipe_face_detector import MediaPipeFaceDetector


class FixedPodcastTemplateProcessor:
    """
    Fixed podcast template processor with robust 2-person splitting logic.
    
    Key improvements:
    1. Enhanced person detection and tracking
    2. Robust left/right to top/bottom mapping
    3. Comprehensive error handling
    4. Detailed logging for debugging
    5. Proper video file creation
    """
    
    def __init__(self, 
                 output_width: int = 1080, 
                 output_height: int = 1920,
                 speaker_detection_threshold: float = 0.6,
                 force_split_when_two_present: bool = True):
        """
        Initialize fixed podcast template processor.
        
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
        
        # Half height for each speaker in split mode
        self.speaker_height = output_height // 2
        
        # Frame smoothing for stability
        self.frame_history = []
        self.max_history = 5  # Keep track of last 5 frames
        self.speaker_positions = {}  # Track speaker positions for smoothing
        self.layout_consistency_buffer = []  # Track layout decisions
        self.consistency_threshold = 3  # Require 3 consistent frames before switching layout
        
        # Initialize detection components
        self.subject_detector = SubjectDetector(device="cuda")
        self.face_detector = MediaPipeFaceDetector()
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized FixedPodcastTemplateProcessor: {output_width}x{output_height}")
        self.logger.info(f"Speaker threshold: {speaker_detection_threshold}")
        self.logger.info(f"Force split when two present: {force_split_when_two_present}")
    
    def detect_speakers_in_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Enhanced speaker detection with comprehensive logging.
        
        Args:
            frame: Input video frame
            
        Returns:
            Dictionary with speaker information and layout type, or None if no speakers found
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            self.logger.debug(f"Detecting speakers in frame: {frame_width}x{frame_height}")
            
            # Step 1: Detect faces using MediaPipe (more reliable for faces)
            faces = []
            try:
                detected_faces = self.face_detector.detect_faces(frame)
                for face in detected_faces:
                    if face.get('confidence', 0.0) >= self.speaker_threshold:
                        faces.append({
                            'type': 'face',
                            'bbox': face['bbox'],
                            'confidence': face['confidence'],
                            'center_x': (face['bbox'][0] + face['bbox'][2]) // 2,
                            'center_y': (face['bbox'][1] + face['bbox'][3]) // 2
                        })
                self.logger.debug(f"MediaPipe detected {len(detected_faces)} faces, {len(faces)} above threshold")
            except Exception as e:
                self.logger.warning(f"MediaPipe face detection failed: {e}")
            
            # Step 2: Detect people using YOLO (fallback)
            people = []
            try:
                subjects = self.subject_detector.detect_subjects(frame)
                for subject in subjects:
                    if (subject.get('type') == 'object' and 
                        subject.get('class') == 'person' and
                        subject.get('confidence', 0.0) >= self.speaker_threshold):
                        bbox = subject['bbox']
                        people.append({
                            'type': 'person',
                            'bbox': bbox,
                            'confidence': subject['confidence'],
                            'center_x': (bbox[0] + bbox[2]) // 2,
                            'center_y': (bbox[1] + bbox[3]) // 2
                        })
                self.logger.debug(f"YOLO detected {len(people)} people above threshold")
            except Exception as e:
                self.logger.warning(f"YOLO person detection failed: {e}")
            
            # Step 3: Combine and deduplicate detections
            all_speakers = faces + people
            if not all_speakers:
                self.logger.debug("No speakers detected in frame")
                return None
            
            # Remove overlapping detections (IoU > 0.5)
            deduplicated_speakers = self._dedupe_speakers(all_speakers)
            self.logger.debug(f"After deduplication: {len(deduplicated_speakers)} speakers")
            
            # Apply position smoothing to reduce shakiness
            smoothed_speakers = self._smooth_speaker_positions(deduplicated_speakers)
            
            # Step 4: Determine stable layout based on speaker count
            speaker_count = len(smoothed_speakers)
            stable_layout = self._determine_stable_layout(speaker_count)
            
            self.logger.debug(f"Layout decision: {stable_layout} (speakers: {speaker_count})")
            
            if stable_layout == 'single_speaker' and smoothed_speakers:
                # Use the most confident speaker for single-speaker layout
                speaker = max(smoothed_speakers, key=lambda s: s['confidence'])
                
                self.logger.info(f"Single speaker mode: confidence={speaker['confidence']:.3f}, "
                               f"bbox={speaker['bbox']}")
                
                return {
                    'layout_type': 'single_speaker',
                    'primary_speaker': speaker,
                    'speaker_count': 1,
                    'confidence': speaker['confidence']
                }
            
            elif stable_layout == 'split_screen' and speaker_count >= 2:
                # For split-screen, use the two most confident speakers
                sorted_speakers = sorted(smoothed_speakers, key=lambda s: s['confidence'], reverse=True)[:2]
                
                # Sort by horizontal position (left to right)
                sorted_speakers.sort(key=lambda s: s['center_x'])
                left_speaker, right_speaker = sorted_speakers[0], sorted_speakers[1]
                
                self.logger.info(f"Split-screen mode:")
                self.logger.info(f"  Left speaker (to top): confidence={left_speaker['confidence']:.3f}, "
                               f"bbox={left_speaker['bbox']}, center_x={left_speaker['center_x']}")
                self.logger.info(f"  Right speaker (to bottom): confidence={right_speaker['confidence']:.3f}, "
                               f"bbox={right_speaker['bbox']}, center_x={right_speaker['center_x']}")
                
                return {
                    'layout_type': 'split_screen',
                    'left_speaker': left_speaker,   # Will become top in vertical video
                    'right_speaker': right_speaker, # Will become bottom in vertical video
                    'speaker_count': 2,
                    'confidence_avg': (left_speaker['confidence'] + right_speaker['confidence']) / 2
                }
            
            # Default to single speaker if we have at least one
            elif smoothed_speakers:
                speaker = max(smoothed_speakers, key=lambda s: s['confidence'])
                
                self.logger.info(f"Fallback to single speaker: confidence={speaker['confidence']:.3f}")
                
                return {
                    'layout_type': 'single_speaker',
                    'primary_speaker': speaker,
                    'speaker_count': 1,
                    'confidence': speaker['confidence']
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Speaker detection failed: {e}", exc_info=True)
            return None
    
    def _smooth_speaker_positions(self, speakers: List[Dict]) -> List[Dict]:
        """
        Smooth speaker positions to reduce shakiness.
        
        Args:
            speakers: List of detected speakers
            
        Returns:
            List of speakers with smoothed positions
        """
        if not speakers:
            return speakers
            
        smoothed_speakers = []
        
        for speaker in speakers:
            speaker_id = f"speaker_{speaker['center_x']//100}_{speaker['center_y']//100}"
            bbox = speaker['bbox']
            
            # Initialize position tracking for new speakers
            if speaker_id not in self.speaker_positions:
                self.speaker_positions[speaker_id] = {
                    'history': [bbox],
                    'last_seen': 0
                }
            else:
                # Add current position to history
                self.speaker_positions[speaker_id]['history'].append(bbox)
                self.speaker_positions[speaker_id]['last_seen'] = 0
                
                # Keep only recent history
                if len(self.speaker_positions[speaker_id]['history']) > self.max_history:
                    self.speaker_positions[speaker_id]['history'].pop(0)
            
            # Calculate smoothed position using weighted average
            history = self.speaker_positions[speaker_id]['history']
            if len(history) > 1:
                # More weight to recent positions
                weights = [i + 1 for i in range(len(history))]
                total_weight = sum(weights)
                
                smoothed_bbox = [0, 0, 0, 0]
                for i, bbox_hist in enumerate(history):
                    weight = weights[i] / total_weight
                    for j in range(4):
                        smoothed_bbox[j] += bbox_hist[j] * weight
                
                # Round to integers
                smoothed_bbox = [int(x) for x in smoothed_bbox]
                
                # Update speaker with smoothed position
                smoothed_speaker = speaker.copy()
                smoothed_speaker['bbox'] = smoothed_bbox
                smoothed_speaker['center_x'] = (smoothed_bbox[0] + smoothed_bbox[2]) // 2
                smoothed_speaker['center_y'] = (smoothed_bbox[1] + smoothed_bbox[3]) // 2
                
                smoothed_speakers.append(smoothed_speaker)
            else:
                smoothed_speakers.append(speaker)
        
        # Clean up old speaker positions
        for speaker_id in list(self.speaker_positions.keys()):
            self.speaker_positions[speaker_id]['last_seen'] += 1
            if self.speaker_positions[speaker_id]['last_seen'] > 10:
                del self.speaker_positions[speaker_id]
        
        return smoothed_speakers
    
    def _determine_stable_layout(self, speaker_count: int) -> str:
        """
        Determine layout with consistency checking to prevent bouncing.
        
        Args:
            speaker_count: Number of detected speakers
            
        Returns:
            Layout type ('single_speaker' or 'split_screen')
        """
        # Determine current frame's preferred layout
        if speaker_count >= 2:
            current_layout = 'split_screen'
        else:
            current_layout = 'single_speaker'
        
        # Add to consistency buffer
        self.layout_consistency_buffer.append(current_layout)
        
        # Keep only recent decisions
        if len(self.layout_consistency_buffer) > self.consistency_threshold * 2:
            self.layout_consistency_buffer.pop(0)
        
        # Check for consistent layout preference
        if len(self.layout_consistency_buffer) >= self.consistency_threshold:
            recent_layouts = self.layout_consistency_buffer[-self.consistency_threshold:]
            
            # If all recent frames agree, use that layout
            if all(layout == current_layout for layout in recent_layouts):
                return current_layout
            
            # If there's disagreement, stick with previous stable layout
            if len(self.layout_consistency_buffer) > self.consistency_threshold:
                stable_layouts = self.layout_consistency_buffer[-self.consistency_threshold*2:-self.consistency_threshold]
                if stable_layouts and all(layout == stable_layouts[0] for layout in stable_layouts):
                    return stable_layouts[0]
        
        # Default to current layout if no history
        return current_layout
    
    def _dedupe_speakers(self, speakers: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
        """Remove duplicate detections based on IoU overlap."""
        if len(speakers) <= 1:
            return speakers
        
        # Sort by confidence descending
        speakers = sorted(speakers, key=lambda s: s['confidence'], reverse=True)
        
        keep = []
        for i, speaker in enumerate(speakers):
            should_keep = True
            for kept_speaker in keep:
                iou = self._calculate_iou(speaker['bbox'], kept_speaker['bbox'])
                if iou > iou_thresh:
                    self.logger.debug(f"Removing overlapping detection: IoU={iou:.3f}")
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(speaker)
        
        return keep
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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
        try:
            frame_height, frame_width = frame.shape[:2]
            layout_type = speaker_info['layout_type']
            
            self.logger.debug(f"Creating {layout_type} layout for {frame_width}x{frame_height} frame")
            
            if layout_type == 'single_speaker':
                return self._create_single_speaker_frame(frame, speaker_info)
            elif layout_type == 'split_screen':
                return self._create_split_screen_frame(frame, speaker_info)
            else:
                self.logger.warning(f"Unknown layout type: {layout_type}, using fallback")
                return self._center_crop_fallback(frame)
                
        except Exception as e:
            self.logger.error(f"Layout creation failed: {e}", exc_info=True)
            return self._center_crop_fallback(frame)
    
    def _create_single_speaker_frame(self, 
                                   frame: np.ndarray,
                                   speaker_info: Dict[str, Any]) -> np.ndarray:
        """
        Create frame focused on single speaker.
        
        Args:
            frame: Input video frame
            speaker_info: Single speaker information
            
        Returns:
            Processed frame with single speaker layout
        """
        try:
            speaker = speaker_info['primary_speaker']
            bbox = speaker['bbox']
            frame_height, frame_width = frame.shape[:2]
            
            self.logger.debug(f"Creating single speaker frame: bbox={bbox}")
            
            # Simple center crop for single speaker
            target_aspect = self.output_width / self.output_height
            frame_aspect = frame_width / frame_height
            
            if frame_aspect > target_aspect:
                new_width = int(frame_height * target_aspect)
                x_offset = (frame_width - new_width) // 2
                cropped = frame[:, x_offset:x_offset + new_width]
            else:
                new_height = int(frame_width / target_aspect)
                y_offset = (frame_height - new_height) // 2
                cropped = frame[y_offset:y_offset + new_height, :]
            
            return cv2.resize(cropped, (self.output_width, self.output_height))
            
        except Exception as e:
            self.logger.error(f"Single speaker frame creation failed: {e}")
            return self._center_crop_fallback(frame)
    
    def _create_split_screen_frame(self, 
                                 frame: np.ndarray,
                                 speaker_info: Dict[str, Any]) -> np.ndarray:
        """
        Create a split-screen frame with two speakers (left→top, right→bottom).
        Enhanced with better alignment and stability.
        
        Args:
            frame: Input video frame
            speaker_info: Two-speaker information
            
        Returns:
            Processed frame with split-screen layout
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            left_speaker = speaker_info['left_speaker']   # Will go to top
            right_speaker = speaker_info['right_speaker'] # Will go to bottom
            
            self.logger.debug(f"Creating split-screen frame: {frame_width}x{frame_height}")
            self.logger.debug(f"  Left speaker (to top): {left_speaker['bbox']}")
            self.logger.debug(f"  Right speaker (to bottom): {right_speaker['bbox']}")
            
            # Create output frame with black background
            output_frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
            
            # Process each speaker with improved positioning
            speakers_config = [
                (left_speaker, 0, "top"),                    # Left speaker goes to top half
                (right_speaker, self.speaker_height, "bottom") # Right speaker goes to bottom half
            ]
            
            for speaker, target_y, position in speakers_config:
                try:
                    bbox = speaker['bbox']
                    
                    # Use adaptive padding based on bounding box size
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    padding_x = int(bbox_width * 0.1)  # 10% padding
                    padding_y = int(bbox_height * 0.15)  # 15% padding for head room
                    
                    # Calculate crop region with bounds checking
                    crop_x1 = max(0, bbox[0] - padding_x)
                    crop_y1 = max(0, bbox[1] - padding_y)
                    crop_x2 = min(frame_width, bbox[2] + padding_x)
                    crop_y2 = min(frame_height, bbox[3] + padding_y)
                    
                    self.logger.debug(f"  {position} speaker crop: ({crop_x1},{crop_y1}) to ({crop_x2},{crop_y2})")
                    
                    # Extract speaker region
                    speaker_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    if speaker_region.size == 0:
                        self.logger.warning(f"Empty {position} speaker region, skipping")
                        continue
                    
                    # Calculate scaling to fit target area while maintaining aspect ratio
                    crop_height, crop_width = speaker_region.shape[:2]
                    
                    # Target dimensions for each speaker region
                    target_width = self.output_width
                    target_height = self.speaker_height
                    
                    # Calculate scale to fit within target area
                    scale_x = target_width / crop_width
                    scale_y = target_height / crop_height
                    scale = min(scale_x, scale_y)  # Maintain aspect ratio
                    
                    # Apply a slight zoom (crop) to make speakers more prominent
                    scale *= 1.1  # 10% zoom
                    
                    new_width = int(crop_width * scale)
                    new_height = int(crop_height * scale)
                    
                    # Ensure we don't exceed target dimensions
                    if new_width > target_width:
                        scale_adjust = target_width / new_width
                        new_width = target_width
                        new_height = int(new_height * scale_adjust)
                    
                    if new_height > target_height:
                        scale_adjust = target_height / new_height
                        new_height = target_height
                        new_width = int(new_width * scale_adjust)
                    
                    self.logger.debug(f"  {position} speaker resize: {crop_width}x{crop_height} → {new_width}x{new_height} (scale={scale:.3f})")
                    
                    # Resize the speaker region with high quality interpolation
                    resized_speaker = cv2.resize(speaker_region, (new_width, new_height),
                                                interpolation=cv2.INTER_LANCZOS4)
                    
                    # Center the resized region in target area
                    x_offset = (target_width - new_width) // 2
                    y_offset = target_y + (target_height - new_height) // 2
                    
                    self.logger.debug(f"  {position} speaker placement: offset=({x_offset},{y_offset})")
                    
                    # Ensure placement is within bounds
                    x_start = max(0, x_offset)
                    y_start = max(0, y_offset)
                    x_end = min(self.output_width, x_offset + new_width)
                    y_end = min(self.output_height, y_offset + new_height)
                    
                    # Adjust resized speaker if needed to fit bounds
                    if x_offset < 0 or y_offset < 0 or x_end != x_offset + new_width or y_end != y_offset + new_height:
                        # Crop the resized speaker to fit
                        src_x_start = max(0, -x_offset)
                        src_y_start = max(0, -y_offset)
                        src_x_end = src_x_start + (x_end - x_start)
                        src_y_end = src_y_start + (y_end - y_start)
                        
                        cropped_speaker = resized_speaker[src_y_start:src_y_end, src_x_start:src_x_end]
                        output_frame[y_start:y_end, x_start:x_end] = cropped_speaker
                    else:
                        # Normal placement
                        output_frame[y_start:y_end, x_start:x_end] = resized_speaker
                    
                    self.logger.debug(f"  {position} speaker placed successfully")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {position} speaker: {e}")
                    continue
            
            return output_frame
            
        except Exception as e:
            self.logger.error(f"Split-screen frame creation failed: {e}", exc_info=True)
            return self._center_crop_fallback(frame)
    
    def _center_crop_fallback(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback method when speakers are not detected - use center crop.
        
        Args:
            frame: Input frame
            
        Returns:
            Center-cropped frame
        """
        try:
            frame_height, frame_width = frame.shape[:2]
            self.logger.debug(f"Using center crop fallback for {frame_width}x{frame_height} frame")
            
            # Calculate center crop for vertical aspect ratio
            target_aspect = self.output_width / self.output_height  # 9:16 = 0.5625
            frame_aspect = frame_width / frame_height
            
            if frame_aspect > target_aspect:
                # Frame is wider, crop width
                new_width = int(frame_height * target_aspect)
                x_offset = (frame_width - new_width) // 2
                cropped = frame[:, x_offset:x_offset + new_width]
                self.logger.debug(f"Width crop: {new_width}px centered at x={x_offset}")
            else:
                # Frame is taller, crop height
                new_height = int(frame_width / target_aspect)
                y_offset = (frame_height - new_height) // 2
                cropped = frame[y_offset:y_offset + new_height, :]
                self.logger.debug(f"Height crop: {new_height}px centered at y={y_offset}")
            
            # Resize to target dimensions
            resized = cv2.resize(cropped, (self.output_width, self.output_height),
                               interpolation=cv2.INTER_LINEAR)
            
            return resized
            
        except Exception as e:
            self.logger.error(f"Center crop fallback failed: {e}", exc_info=True)
            # Ultimate fallback - just resize the entire frame
            return cv2.resize(frame, (self.output_width, self.output_height))
    
    def process_video(self, input_video_path: str, output_video_path: str) -> bool:
        """
        Process a video using the fixed podcast template.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video
            
        Returns:
            True if processing succeeded, False otherwise
        """
        cap = None
        out = None
        temp_video_path = None
        
        try:
            self.logger.info(f"Processing video: {input_video_path} -> {output_video_path}")
            
            # Create temporary video file for processing
            temp_video_path = output_video_path.replace('.mp4', '_temp.mp4')
            
            # Open input video
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open input video: {input_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.logger.info(f"Video properties: {total_frames} frames at {fps:.2f} fps")
            
            # Setup video writer for temp file (no audio)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (self.output_width, self.output_height))
            
            if not out.isOpened():
                self.logger.error(f"Failed to create temp video writer: {temp_video_path}")
                return False
            
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                try:
                    # Detect speakers in this frame
                    speaker_info = self.detect_speakers_in_frame(frame)
                    
                    # Create appropriate layout
                    if speaker_info:
                        processed_frame = self.create_podcast_layout_frame(frame, speaker_info)
                        processed_frames += 1
                    else:
                        # Fallback to center crop
                        processed_frame = self._center_crop_fallback(frame)
                    
                    # Write the frame
                    out.write(processed_frame)
                    
                    # Progress logging every 100 frames
                    if frame_count % 100 == 0:
                        self.logger.info(f"Processed {frame_count}/{total_frames} frames "
                                       f"({processed_frames} with speakers)")
                
                except Exception as e:
                    self.logger.warning(f"Failed to process frame {frame_count}: {e}")
                    # Write fallback frame
                    fallback_frame = self._center_crop_fallback(frame)
                    out.write(fallback_frame)
            
            # Release video resources
            cap.release()
            out.release()
            cap = None
            out = None
            
            self.logger.info(f"Video processing completed: {frame_count} total frames, "
                           f"{processed_frames} frames with speaker detection")
            
            # Now add audio from original video
            if self._finalize_video_with_audio(temp_video_path, input_video_path, output_video_path):
                self.logger.info("Audio muxing completed successfully")
                
                # Clean up temp file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                
                return True
            else:
                self.logger.error("Audio muxing failed")
                return False
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {e}", exc_info=True)
            return False
            
        finally:
            if cap:
                cap.release()
            if out:
                out.release()
            # Clean up temp file on failure
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass
    
    def _finalize_video_with_audio(self, temp_video_path: str, input_video_path: str, output_video_path: str) -> bool:
        """
        Add audio from original video to processed video.
        
        Args:
            temp_video_path: Path to processed video (no audio)
            input_video_path: Path to original video (with audio)
            output_video_path: Path to final output video
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Muxing audio: {temp_video_path} + {input_video_path} -> {output_video_path}")
            
            # Check if temp video exists
            if not os.path.exists(temp_video_path) or os.path.getsize(temp_video_path) == 0:
                self.logger.error("Temp video file missing or empty")
                return False
            
            # Strategy 1: Hardware-accelerated method
            try:
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-hwaccel', 'cuda',
                    '-i', temp_video_path,
                    '-i', input_video_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    output_video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(output_video_path):
                    self.logger.info("Hardware-accelerated audio muxing successful")
                    return True
                else:
                    self.logger.warning(f"Hardware acceleration failed: {result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Hardware acceleration failed: {e}")
            
            # Strategy 2: Standard FFmpeg method  
            try:
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-i', temp_video_path,
                    '-i', input_video_path,
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    output_video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(output_video_path):
                    self.logger.info("Standard audio muxing successful")
                    return True
                else:
                    self.logger.warning(f"Standard muxing failed: {result.stderr}")
                    
            except Exception as e:
                self.logger.warning(f"Standard muxing failed: {e}")
            
            # Strategy 3: Basic copy method (no re-encoding)
            try:
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
                    '-i', temp_video_path,
                    '-i', input_video_path,
                    '-c', 'copy',
                    '-map', '0:v:0',
                    '-map', '1:a:0?',
                    '-shortest',
                    output_video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and os.path.exists(output_video_path):
                    self.logger.info("Basic copy audio muxing successful") 
                    return True
                else:
                    self.logger.error(f"Basic copy failed: {result.stderr}")
                    
            except Exception as e:
                self.logger.error(f"Basic copy failed: {e}")
            
            return False
            
        except Exception as e:
            self.logger.error(f"Audio muxing failed: {e}", exc_info=True)
            return False


def apply_fixed_podcast_template(input_video_path: str, 
                                 output_video_path: str,
                                 template: Optional[str] = None) -> str:
    """
    Apply fixed podcast template processing.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video  
        template: Template type ('podcast' or None)
        
    Returns:
        Path to final video (processed or original)
    """
    if template != 'podcast':
        return input_video_path
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Starting fixed podcast template processing: {input_video_path} -> {output_video_path}")
        
        # Initialize the fixed processor
        processor = FixedPodcastTemplateProcessor(
            output_width=1080,
            output_height=1920,
            speaker_detection_threshold=0.6,
            force_split_when_two_present=True
        )
        
        # Process the video
        success = processor.process_video(input_video_path, output_video_path)
        
        if success and os.path.exists(output_video_path):
            logger.info(f"Fixed podcast template processing completed successfully: {output_video_path}")
            return output_video_path
        else:
            logger.warning("Fixed podcast template processing failed or output not created")
            return input_video_path
        
    except Exception as e:
        logger.error(f"Fixed podcast template failed: {e}", exc_info=True)
        return input_video_path