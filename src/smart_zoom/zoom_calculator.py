# src/smart_zoom/zoom_calculator.py
"""Zoom calculation and positioning logic for smart framing"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from config.smart_zoom_settings import SMART_ZOOM_CONFIG


class ZoomCalculator:
    """
    Calculate optimal zoom levels and crop positions for vertical format conversion.
    """
    
    def __init__(self, target_aspect_ratio: Tuple[int, int] = (9, 16)):
        """
        Initialize zoom calculator.
        
        Args:
            target_aspect_ratio: Target aspect ratio (width, height)
        """
        self.target_aspect_ratio = target_aspect_ratio
        self.target_ratio = target_aspect_ratio[0] / target_aspect_ratio[1]
        
        # Load configuration
        self.config = SMART_ZOOM_CONFIG.get('framing', {})
        self.zoom_config = SMART_ZOOM_CONFIG.get('zoom_behavior', {})
        
        # Zoom parameters
        self.min_zoom = self.zoom_config.get('min_zoom', 1.0)
        self.max_zoom = self.zoom_config.get('max_zoom', 3.0)
        self.zoom_speed = self.zoom_config.get('zoom_speed', 0.05)
        self.smoothing_factor = self.zoom_config.get('smoothing_factor', 0.7)
        
        # Subject parameters
        self.subject_padding = self.config.get('subject_padding_ratio', 0.15)
        self.min_subject_size = self.config.get('min_subject_size', 0.3)
        self.max_subject_size = self.config.get('max_subject_size', 0.8)
        
        self.logger = logging.getLogger(__name__)
        
        # Previous zoom state for smoothing
        self.previous_zoom = 1.0
        self.previous_position = (0.5, 0.5)  # Center position
    
    def calculate_optimal_zoom(self, 
                             frame_shape: Tuple[int, int],
                             subjects: List[Dict],
                             content_context: Dict = None) -> Dict:
        """
        Calculate optimal zoom level and position for given subjects.
        
        Args:
            frame_shape: (height, width) of the frame
            subjects: List of detected subjects with bounding boxes
            content_context: Optional context from content analysis
            
        Returns:
            Dictionary with zoom parameters
        """
        if not subjects:
            return self._get_default_zoom(frame_shape)
        
        # Calculate composite bounding box for all important subjects
        composite_bbox = self._calculate_composite_bbox(subjects, frame_shape, content_context)
        
        # Calculate base zoom level
        base_zoom = self._calculate_base_zoom(composite_bbox, frame_shape)
        
        # Apply content-based adjustments
        if content_context:
            base_zoom = self._apply_content_adjustments(base_zoom, content_context)
        
        # Calculate optimal position
        position = self._calculate_optimal_position(composite_bbox, frame_shape, base_zoom)
        
        # Apply smoothing
        smoothed_zoom = self._apply_smoothing(base_zoom, self.previous_zoom)
        smoothed_position = self._apply_position_smoothing(position, self.previous_position)
        
        # Calculate crop rectangle
        target_ar = content_context.get('target_aspect_ratio', (9, 16)) if content_context else (9, 16)
        crop_rect = self._calculate_crop_rectangle(
            frame_shape, smoothed_zoom, smoothed_position, target_ar
        )
        
        # Update previous state
        self.previous_zoom = smoothed_zoom
        self.previous_position = smoothed_position
        
        return {
            'zoom_level': smoothed_zoom,
            'position': smoothed_position,  # (x, y) center position (0-1 normalized)
            'crop_position': (crop_rect['x'], crop_rect['y'], crop_rect['width'], crop_rect['height']),  # 4-value tuple for smart zoom processor
            'crop_rect': crop_rect,  # (x, y, width, height) in pixels
            'subjects_bbox': composite_bbox,
            'confidence': self._calculate_confidence(subjects, content_context),
            'strategy': 'optimal_subject_tracking',
            'reasoning': f"Optimizing for {len(subjects)} subjects with confidence {self._calculate_confidence(subjects, content_context):.2f}",
            'frame_shape': frame_shape
        }
    
    def _calculate_composite_bbox(self, 
                                subjects: List[Dict], 
                                frame_shape: Tuple[int, int],
                                content_context: Dict = None) -> Dict:
        """
        Calculate bounding box that encompasses all important subjects.
        
        Args:
            subjects: List of subject detections
            frame_shape: Frame dimensions
            content_context: Optional content context for weight adjustment
            
        Returns:
            Composite bounding box dictionary
        """
        if not subjects:
            return self._get_center_bbox(frame_shape)
        
        # Extract bounding boxes with weights
        weighted_boxes = []
        total_weight = 0
        
        for subject in subjects:
            bbox_raw = subject.get('bbox', [])
            confidence = subject.get('confidence', 0.5)
            subject_type = subject.get('type', 'unknown')
            
            # Convert bbox to normalized format
            bbox = self._normalize_bbox(bbox_raw, frame_shape)
            if not bbox:
                continue
            
            # Weight subjects by type and confidence with content awareness
            weight = self._get_subject_weight(subject_type, confidence, content_context)
            
            if weight > 0:
                weighted_boxes.append({
                    'bbox': bbox,
                    'weight': weight,
                    'type': subject_type
                })
                total_weight += weight
        
        if not weighted_boxes:
            return self._get_center_bbox(frame_shape)
        
        # Calculate weighted center and bounds
        min_x = min(box['bbox']['x'] for box in weighted_boxes)
        min_y = min(box['bbox']['y'] for box in weighted_boxes)
        max_x = max(box['bbox']['x'] + box['bbox']['width'] for box in weighted_boxes)
        max_y = max(box['bbox']['y'] + box['bbox']['height'] for box in weighted_boxes)
        
        # Add padding
        padding_x = (max_x - min_x) * self.subject_padding
        padding_y = (max_y - min_y) * self.subject_padding
        
        composite_bbox = {
            'x': max(0, min_x - padding_x),
            'y': max(0, min_y - padding_y),
            'width': min(frame_shape[1], max_x - min_x + 2 * padding_x),
            'height': min(frame_shape[0], max_y - min_y + 2 * padding_y),
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2,
            'subject_count': len(weighted_boxes),
            'total_weight': total_weight
        }
        
        return composite_bbox
    
    def _calculate_base_zoom(self, 
                           bbox: Dict, 
                           frame_shape: Tuple[int, int]) -> float:
        """
        Calculate base zoom level for the given bounding box.
        
        Args:
            bbox: Subject bounding box
            frame_shape: Frame dimensions
            
        Returns:
            Base zoom level
        """
        frame_height, frame_width = frame_shape
        
        # Calculate subject size relative to frame
        subject_width_ratio = bbox['width'] / frame_width
        subject_height_ratio = bbox['height'] / frame_height
        
        # Determine zoom based on subject size
        if subject_height_ratio < self.min_subject_size:
            # Subject too small, zoom in
            zoom = self.min_subject_size / subject_height_ratio
        elif subject_height_ratio > self.max_subject_size:
            # Subject too large, zoom out
            zoom = self.max_subject_size / subject_height_ratio
        else:
            # Subject size is good
            zoom = 1.0
        
        # Consider aspect ratio requirements
        target_width = frame_height * self.target_ratio
        if target_width > frame_width:
            # Need to zoom in to maintain aspect ratio
            aspect_zoom = target_width / frame_width
            zoom = max(zoom, aspect_zoom)
        
        # Apply zoom limits
        zoom = max(self.min_zoom, min(self.max_zoom, zoom))
        
        return zoom
    
    def _apply_content_adjustments(self, 
                                 base_zoom: float, 
                                 content_context: Dict) -> float:
        """
        Apply content-based zoom adjustments.
        
        Args:
            base_zoom: Base zoom level
            content_context: Content analysis context
            
        Returns:
            Adjusted zoom level
        """
        adjustment_factor = 1.0
        
        # Get content type
        content_type = content_context.get('content_type', 'general')
        
        # Apply type-specific adjustments
        if content_type == 'presentation':
            adjustment_factor *= 0.8  # Zoom out for presentations
        elif content_type == 'conversation':
            adjustment_factor *= 1.1  # Slight zoom in for conversations
        elif content_type == 'action':
            adjustment_factor *= 0.9  # Zoom out for action scenes
        
        # Apply emotional adjustments
        emotional_intensity = content_context.get('emotional_intensity', 0.5)
        if emotional_intensity > 0.7:
            adjustment_factor *= 1.2  # Zoom in for high emotion
        elif emotional_intensity < 0.3:
            adjustment_factor *= 0.9  # Zoom out for low emotion
        
        # Apply engagement adjustments
        engagement_score = content_context.get('engagement_score', 0.5)
        if engagement_score > 0.8:
            adjustment_factor *= 1.1  # Zoom in for high engagement
        
        adjusted_zoom = base_zoom * adjustment_factor
        return max(self.min_zoom, min(self.max_zoom, adjusted_zoom))
    
    def _calculate_optimal_position(self, 
                                  bbox: Dict, 
                                  frame_shape: Tuple[int, int],
                                  zoom_level: float) -> Tuple[float, float]:
        """
        Calculate optimal center position for the crop.
        
        Args:
            bbox: Subject bounding box
            frame_shape: Frame dimensions
            zoom_level: Current zoom level
            
        Returns:
            (x, y) center position normalized to 0-1
        """
        frame_height, frame_width = frame_shape
        
        # Start with subject center
        center_x = bbox['center_x'] / frame_width
        center_y = bbox['center_y'] / frame_height
        
        # Calculate crop dimensions
        crop_width = frame_width / zoom_level
        crop_height = frame_height / zoom_level
        
        # Ensure crop doesn't go outside frame bounds
        half_crop_width = crop_width / (2 * frame_width)
        half_crop_height = crop_height / (2 * frame_height)
        
        # Constrain position to keep crop within frame
        center_x = max(half_crop_width, min(1 - half_crop_width, center_x))
        center_y = max(half_crop_height, min(1 - half_crop_height, center_y))
        
        # Apply rule of thirds for better composition
        center_x, center_y = self._apply_rule_of_thirds(center_x, center_y, bbox)
        
        return (center_x, center_y)
    
    def _apply_rule_of_thirds(self, 
                            center_x: float, 
                            center_y: float, 
                            bbox: Dict) -> Tuple[float, float]:
        """
        Apply rule of thirds for better composition.
        
        Args:
            center_x: Current x position
            center_y: Current y position
            bbox: Subject bounding box
            
        Returns:
            Adjusted (x, y) position
        """
        # Rule of thirds lines
        third_lines_x = [1/3, 2/3]
        third_lines_y = [1/3, 2/3]
        
        # Find closest third line for x
        closest_x = min(third_lines_x, key=lambda x: abs(x - center_x))
        if abs(closest_x - center_x) < 0.1:  # If close to third line
            center_x = center_x * 0.7 + closest_x * 0.3  # Blend towards third line
        
        # For y, prefer upper third for faces/people
        subject_types = bbox.get('types', [])
        if 'face' in subject_types or 'person' in subject_types:
            # Prefer upper third for people
            center_y = max(0.2, min(0.4, center_y))
        
        return (center_x, center_y)
    
    def _calculate_crop_rectangle(self, 
                                frame_shape: Tuple[int, int],
                                zoom_level: float,
                                position: Tuple[float, float],
                                target_aspect_ratio: Tuple[int, int] = (9, 16)) -> Dict:
        """
        Calculate crop rectangle in pixel coordinates for target aspect ratio.
        
        Args:
            frame_shape: Frame dimensions
            zoom_level: Zoom level
            position: Center position (normalized)
            target_aspect_ratio: Target aspect ratio (width, height)
            
        Returns:
            Crop rectangle dictionary
        """
        frame_height, frame_width = frame_shape
        center_x, center_y = position
        
        # Calculate target aspect ratio
        target_w, target_h = target_aspect_ratio
        target_aspect = target_w / target_h
        
        # Calculate crop dimensions maintaining target aspect ratio
        frame_aspect = frame_width / frame_height
        
        if frame_aspect > target_aspect:
            # Frame is wider than target - limit by height
            crop_height = frame_height / zoom_level
            crop_width = crop_height * target_aspect
        else:
            # Frame is taller than target - limit by width
            crop_width = frame_width / zoom_level
            crop_height = crop_width / target_aspect
        
        # Ensure crop doesn't exceed frame boundaries
        crop_width = min(crop_width, frame_width)
        crop_height = min(crop_height, frame_height)
        
        # Convert to pixel coordinates
        center_x_px = center_x * frame_width
        center_y_px = center_y * frame_height
        
        # Calculate crop rectangle
        x = int(center_x_px - crop_width / 2)
        y = int(center_y_px - crop_height / 2)
        width = int(crop_width)
        height = int(crop_height)
        
        # Ensure crop is within frame bounds
        x = max(0, min(frame_width - width, x))
        y = max(0, min(frame_height - height, y))
        
        return {
            'x': x,
            'y': y,
            'width': width,
            'height': height
        }
    
    def _apply_smoothing(self, current_zoom: float, previous_zoom: float) -> float:
        """
        Apply smoothing to zoom transitions.
        
        Args:
            current_zoom: Target zoom level
            previous_zoom: Previous zoom level
            
        Returns:
            Smoothed zoom level
        """
        # Calculate maximum allowed change
        max_change = self.zoom_speed
        
        # Calculate difference
        diff = current_zoom - previous_zoom
        
        # Limit change
        if abs(diff) > max_change:
            if diff > 0:
                smoothed_zoom = previous_zoom + max_change
            else:
                smoothed_zoom = previous_zoom - max_change
        else:
            # Apply smoothing factor
            smoothed_zoom = (
                previous_zoom * self.smoothing_factor + 
                current_zoom * (1 - self.smoothing_factor)
            )
        
        return smoothed_zoom
    
    def _apply_position_smoothing(self, 
                                current_pos: Tuple[float, float], 
                                previous_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        Apply smoothing to position transitions.
        
        Args:
            current_pos: Target position
            previous_pos: Previous position
            
        Returns:
            Smoothed position
        """
        smoothing = self.smoothing_factor * 0.8  # Slightly less smoothing for position
        
        smoothed_x = (
            previous_pos[0] * smoothing + 
            current_pos[0] * (1 - smoothing)
        )
        smoothed_y = (
            previous_pos[1] * smoothing + 
            current_pos[1] * (1 - smoothing)
        )
        
        return (smoothed_x, smoothed_y)
    
    def _normalize_bbox(self, bbox_raw, frame_shape: Tuple[int, int]) -> Optional[Dict]:
        """
        Normalize bounding box to standard dictionary format.
        
        Args:
            bbox_raw: Raw bounding box (can be list or dict)
            frame_shape: Frame dimensions
            
        Returns:
            Normalized bbox dictionary or None if invalid
        """
        if not bbox_raw:
            return None
        
        try:
            if isinstance(bbox_raw, list):
                # Handle list format [x1, y1, x2, y2]
                if len(bbox_raw) >= 4:
                    x1, y1, x2, y2 = bbox_raw[:4]
                    return {
                        'x': float(x1),
                        'y': float(y1),
                        'width': float(x2 - x1),
                        'height': float(y2 - y1)
                    }
            elif isinstance(bbox_raw, dict):
                # Handle dictionary format - check for different key formats
                if 'x' in bbox_raw and 'y' in bbox_raw and 'width' in bbox_raw and 'height' in bbox_raw:
                    return {
                        'x': float(bbox_raw['x']),
                        'y': float(bbox_raw['y']),
                        'width': float(bbox_raw['width']),
                        'height': float(bbox_raw['height'])
                    }
                elif 'x1' in bbox_raw and 'y1' in bbox_raw and 'x2' in bbox_raw and 'y2' in bbox_raw:
                    return {
                        'x': float(bbox_raw['x1']),
                        'y': float(bbox_raw['y1']),
                        'width': float(bbox_raw['x2'] - bbox_raw['x1']),
                        'height': float(bbox_raw['y2'] - bbox_raw['y1'])
                    }
        except (TypeError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to normalize bbox {bbox_raw}: {e}")
        
        return None

    def _get_subject_weight(self, subject_type: str, confidence: float, content_context: Dict = None) -> float:
        """
        Get weight for subject type and confidence with content awareness.
        
        Args:
            subject_type: Type of subject
            confidence: Detection confidence
            content_context: Optional content context for weight adjustment
            
        Returns:
            Subject weight
        """
        # Base weights for different content types
        default_weights = {
            'face': 1.0,
            'person': 0.8,
            'object': 0.3,
            'unknown': 0.1
        }
        
        # Content-aware weight adjustments
        if content_context:
            content_type = content_context.get('content_type', '').lower()
            
            # Yoga pose content - prioritize full person over face
            if content_type == 'yoga_pose':
                yoga_pose_weights = {
                    'face': 0.4,      # Lower priority for faces during poses
                    'person': 1.0,    # Highest priority for full body
                    'object': 0.1,    # Minimal priority for objects
                    'unknown': 0.1
                }
                type_weights = yoga_pose_weights
            
            # Yoga movement/flow - balance person and face
            elif content_type == 'yoga_movement':
                movement_weights = {
                    'face': 0.7,      # Moderate priority for face
                    'person': 1.0,    # High priority for body movement
                    'object': 0.2,    # Low priority for objects
                    'unknown': 0.1
                }
                type_weights = movement_weights
                
            # Breathing content - focus more on face for expressions
            elif content_type == 'yoga_breathing':
                breath_weights = {
                    'face': 1.0,      # Highest priority for face during breathing
                    'person': 0.8,    # Still important to see body
                    'object': 0.1,    # Minimal priority for objects
                    'unknown': 0.1
                }
                type_weights = breath_weights
                
            # Instruction content - balance face and person
            elif content_type == 'instruction':
                instruction_weights = {
                    'face': 0.9,      # High priority for face during instruction
                    'person': 0.8,    # Good priority for body
                    'object': 0.3,    # Some priority for props/equipment
                    'unknown': 0.1
                }
                type_weights = instruction_weights
                
            # Demonstration content - prioritize what's being shown
            elif content_type == 'demonstration':
                demo_weights = {
                    'face': 0.6,      # Lower priority for face
                    'person': 1.0,    # Highest priority for body demonstration
                    'object': 0.4,    # Higher priority for objects being used
                    'unknown': 0.1
                }
                type_weights = demo_weights
                
            # Hold/rest positions - balanced view
            elif content_type == 'yoga_hold':
                hold_weights = {
                    'face': 0.8,      # Good priority for face
                    'person': 0.9,    # High priority for body position
                    'object': 0.2,    # Low priority for objects
                    'unknown': 0.1
                }
                type_weights = hold_weights
            else:
                type_weights = default_weights
        else:
            type_weights = default_weights
        
        base_weight = type_weights.get(subject_type, 0.1)
        return base_weight * confidence
    
    def _calculate_confidence(self, subjects: List[Dict], content_context: Dict = None) -> float:
        """
        Calculate overall confidence for the zoom decision.
        
        Args:
            subjects: List of subjects
            content_context: Optional content context for weight adjustment
            
        Returns:
            Confidence score (0-1)
        """
        if not subjects:
            return 0.1
        
        # Average confidence weighted by subject importance
        total_confidence = 0
        total_weight = 0
        
        for subject in subjects:
            confidence = subject.get('confidence', 0.5)
            subject_type = subject.get('type', 'unknown')
            weight = self._get_subject_weight(subject_type, confidence, content_context)
            
            total_confidence += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.1
        
        return min(1.0, total_confidence / total_weight)
    
    def _get_default_zoom(self, frame_shape: Tuple[int, int]) -> Dict:
        """
        Get default zoom parameters when no subjects are detected.
        
        Args:
            frame_shape: Frame dimensions
            
        Returns:
            Default zoom parameters
        """
        return {
            'zoom_level': 1.0,
            'position': (0.5, 0.4),  # Slightly above center
            'crop_position': (0, 0, frame_shape[1], frame_shape[0]),  # Full frame as 4-value tuple
            'crop_rect': {
                'x': 0,
                'y': 0,
                'width': frame_shape[1],
                'height': frame_shape[0]
            },
            'subjects_bbox': self._get_center_bbox(frame_shape),
            'confidence': 0.1,
            'strategy': 'default_center',
            'reasoning': 'No subjects detected, using default center framing',
            'frame_shape': frame_shape
        }
    
    def _get_center_bbox(self, frame_shape: Tuple[int, int]) -> Dict:
        """
        Get center bounding box for default framing.
        
        Args:
            frame_shape: Frame dimensions
            
        Returns:
            Center bounding box
        """
        frame_height, frame_width = frame_shape
        
        return {
            'x': frame_width * 0.25,
            'y': frame_height * 0.25,
            'width': frame_width * 0.5,
            'height': frame_height * 0.5,
            'center_x': frame_width * 0.5,
            'center_y': frame_height * 0.5,
            'subject_count': 0,
            'total_weight': 0
        }
    
    def visualize_zoom_decision(self, 
                              frame: np.ndarray, 
                              zoom_params: Dict,
                              subjects: List[Dict] = None) -> np.ndarray:
        """
        Visualize zoom decision on frame for debugging.
        
        Args:
            frame: Input frame
            zoom_params: Zoom parameters
            subjects: Optional subjects to draw
            
        Returns:
            Frame with visualization overlay
        """
        vis_frame = frame.copy()
        
        # Draw crop rectangle
        crop_rect = zoom_params['crop_rect']
        cv2.rectangle(vis_frame, 
                     (crop_rect['x'], crop_rect['y']),
                     (crop_rect['x'] + crop_rect['width'], 
                      crop_rect['y'] + crop_rect['height']),
                     (0, 255, 0), 2)
        
        # Draw subjects
        if subjects:
            for subject in subjects:
                bbox_raw = subject.get('bbox', [])
                bbox = self._normalize_bbox(bbox_raw, frame.shape[:2])
                if bbox:
                    cv2.rectangle(vis_frame,
                                 (int(bbox['x']), int(bbox['y'])),
                                 (int(bbox['x'] + bbox['width']), 
                                  int(bbox['y'] + bbox['height'])),
                                 (255, 0, 0), 1)
        
        # Draw center point
        center_x = int(zoom_params['position'][0] * frame.shape[1])
        center_y = int(zoom_params['position'][1] * frame.shape[0])
        cv2.circle(vis_frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Add text overlay
        text = f"Zoom: {zoom_params['zoom_level']:.2f}, Conf: {zoom_params['confidence']:.2f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset_state(self):
        """Reset zoom calculator state."""
        self.previous_zoom = 1.0
        self.previous_position = (0.5, 0.5)
        self.logger.info("Zoom calculator state reset")

    def calculate_smart_zoom(self, 
                           frame: np.ndarray,
                           subjects: List[Dict],
                           audio_analysis: Dict,
                           timestamp: float,
                           target_aspect_ratio: Tuple[int, int] = (9, 16)) -> Dict:
        """
        Alias for calculate_optimal_zoom to maintain compatibility.
        
        Args:
            frame: The video frame
            subjects: List of detected subjects
            audio_analysis: Audio analysis data
            timestamp: Current timestamp
            target_aspect_ratio: Target aspect ratio (width, height)
            
        Returns:
            Dict containing zoom parameters
        """
        frame_shape = frame.shape[:2]  # Get height, width from frame
        
        # Extract content context from audio analysis
        content_context = {
            'audio_analysis': audio_analysis,
            'timestamp': timestamp,
            'target_aspect_ratio': target_aspect_ratio
        }
        
        # Extract content type from audio analysis if available
        if audio_analysis and isinstance(audio_analysis, dict):
            # Try to get content type from segments or transcription
            segments = audio_analysis.get('segments', [])
            transcription = audio_analysis.get('text', '').lower()
            
            # Determine content type based on transcription
            content_type = self._classify_content_from_audio(transcription, segments, timestamp)
            content_context['content_type'] = content_type
        
        return self.calculate_optimal_zoom(frame_shape, subjects, content_context)
    
    def _classify_content_from_audio(self, transcription: str, segments: List, timestamp: float) -> str:
        """
        Classify content type from audio transcription.
        
        Args:
            transcription: Full transcription text
            segments: List of transcription segments
            timestamp: Current timestamp
            
        Returns:
            Content type classification
        """
        # Find the segment that contains the current timestamp
        current_segment_text = ""
        for segment in segments:
            if isinstance(segment, dict):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                if start <= timestamp <= end:
                    current_segment_text = segment.get('text', '').lower()
                    break
        
        # Use current segment if available, otherwise use full transcription
        text_to_analyze = current_segment_text if current_segment_text else transcription
        
        # Yoga/fitness content classification
        if any(keyword in text_to_analyze for keyword in ['pose', 'position', 'stretch', 'bend', 'twist', 'warrior', 'downward', 'upward', 'plank', 'balance']):
            return 'yoga_pose'
        elif any(keyword in text_to_analyze for keyword in ['breathe', 'breathing', 'inhale', 'exhale', 'breath']):
            return 'yoga_breathing'
        elif any(keyword in text_to_analyze for keyword in ['move', 'movement', 'flow', 'transition', 'lift', 'lower']):
            return 'yoga_movement'
        elif any(keyword in text_to_analyze for keyword in ['relax', 'rest', 'pause', 'hold', 'stay']):
            return 'yoga_hold'
        elif any(keyword in text_to_analyze for keyword in ['demonstrate', 'show', 'example', 'watch', 'see']):
            return 'demonstration'
        elif any(keyword in text_to_analyze for keyword in ['instruction', 'now', 'next', 'then', 'remember']):
            return 'instruction'
        else:
            return 'general'


def create_zoom_calculator(config: Dict = None) -> ZoomCalculator:
    """
    Factory function to create zoom calculator with configuration.
    
    Args:
        config: Optional configuration override
        
    Returns:
        Configured ZoomCalculator instance
    """
    if config:
        # Merge with default config
        merged_config = SMART_ZOOM_CONFIG.copy()
        merged_config.update(config)
        
        # Create calculator with custom target ratio if specified
        target_ratio = config.get('target_aspect_ratio', (9, 16))
        calculator = ZoomCalculator(target_ratio)
        
        # Override internal config
        calculator.config = merged_config.get('framing', {})
        calculator.zoom_config = merged_config.get('zoom_behavior', {})
        
        return calculator
    
    return ZoomCalculator()
