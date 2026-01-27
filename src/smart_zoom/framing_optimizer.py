# src/smart_zoom/framing_optimizer.py
"""Framing optimization logic for smart zoom functionality"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from config.smart_zoom_settings import SMART_ZOOM_CONFIG


@dataclass
class FramingParameters:
    """Parameters for optimal framing."""
    crop_x: int
    crop_y: int
    crop_width: int
    crop_height: int
    zoom_level: float
    confidence: float
    subjects: List[Dict]
    frame_number: int


class FramingOptimizer:
    """
    Optimize framing decisions for vertical video format.
    """
    
    def __init__(self, target_aspect_ratio: Tuple[int, int] = (9, 16)):
        """
        Initialize framing optimizer.
        
        Args:
            target_aspect_ratio: Target aspect ratio (width, height)
        """
        self.target_aspect_ratio = target_aspect_ratio
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = SMART_ZOOM_CONFIG
        self.framing_config = self.config.get('framing', {})
        self.detection_config = self.config.get('detection', {})
        
        # Framing history for smooth transitions
        self.framing_history: List[FramingParameters] = []
        self.max_history = 30  # frames
        
        # Quality metrics
        self.quality_metrics = {
            'subject_visibility_score': 0.0,
            'composition_score': 0.0,
            'stability_score': 0.0,
            'overall_score': 0.0
        }
    
    def optimize_framing(self, 
                        frame: np.ndarray,
                        subjects: List[Dict],
                        audio_context: Dict = None,
                        previous_framing: FramingParameters = None) -> FramingParameters:
        """
        Optimize framing for the given frame and subjects.
        
        Args:
            frame: Input frame
            subjects: Detected subjects with bounding boxes
            audio_context: Audio analysis context
            previous_framing: Previous frame's framing parameters
            
        Returns:
            Optimal framing parameters
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate base framing from subjects
        base_framing = self._calculate_base_framing(
            frame_width, frame_height, subjects
        )
        
        # Apply composition rules
        composition_framing = self._apply_composition_rules(
            base_framing, frame_width, frame_height, subjects
        )
        
        # Apply audio-based adjustments
        if audio_context:
            audio_framing = self._apply_audio_context(
                composition_framing, audio_context
            )
        else:
            audio_framing = composition_framing
        
        # Apply smooth transitions
        if previous_framing:
            smooth_framing = self._apply_smooth_transitions(
                audio_framing, previous_framing
            )
        else:
            smooth_framing = audio_framing
        
        # Validate and constrain framing
        final_framing = self._validate_framing(
            smooth_framing, frame_width, frame_height
        )
        
        # Calculate quality metrics
        final_framing.confidence = self._calculate_framing_confidence(
            final_framing, subjects, frame_width, frame_height
        )
        
        # Add to history
        self._add_to_history(final_framing)
        
        return final_framing
    
    def _calculate_base_framing(self, 
                               frame_width: int, 
                               frame_height: int,
                               subjects: List[Dict]) -> FramingParameters:
        """Calculate base framing from subject positions."""
        if not subjects:
            # Default center framing
            target_width = int(frame_height * self.target_aspect_ratio[0] / self.target_aspect_ratio[1])
            crop_x = max(0, (frame_width - target_width) // 2)
            crop_y = 0
            crop_width = min(target_width, frame_width)
            crop_height = frame_height
            
            return FramingParameters(
                crop_x=crop_x,
                crop_y=crop_y,
                crop_width=crop_width,
                crop_height=crop_height,
                zoom_level=1.0,
                confidence=0.5,
                subjects=[],
                frame_number=0
            )
        
        # Calculate bounding box for all important subjects
        subject_bbox = self._calculate_composite_bbox(subjects)
        
        # Calculate optimal crop position
        crop_params = self._calculate_crop_from_bbox(
            subject_bbox, frame_width, frame_height
        )
        
        return FramingParameters(
            crop_x=crop_params['x'],
            crop_y=crop_params['y'],
            crop_width=crop_params['width'],
            crop_height=crop_params['height'],
            zoom_level=crop_params['zoom_level'],
            confidence=0.8,
            subjects=subjects,
            frame_number=0
        )
    
    def _calculate_composite_bbox(self, subjects: List[Dict]) -> Dict:
        """Calculate composite bounding box for all subjects."""
        if not subjects:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        
        # Find extremes
        min_x = min(subject['bbox']['x'] for subject in subjects)
        min_y = min(subject['bbox']['y'] for subject in subjects)
        max_x = max(subject['bbox']['x'] + subject['bbox']['width'] for subject in subjects)
        max_y = max(subject['bbox']['y'] + subject['bbox']['height'] for subject in subjects)
        
        # Add padding
        padding_ratio = self.framing_config.get('subject_padding_ratio', 0.15)
        width = max_x - min_x
        height = max_y - min_y
        
        padding_x = width * padding_ratio
        padding_y = height * padding_ratio
        
        return {
            'x': max(0, min_x - padding_x),
            'y': max(0, min_y - padding_y),
            'width': width + 2 * padding_x,
            'height': height + 2 * padding_y
        }
    
    def _calculate_crop_from_bbox(self, 
                                 bbox: Dict, 
                                 frame_width: int, 
                                 frame_height: int) -> Dict:
        """Calculate crop parameters from bounding box."""
        # Target dimensions for vertical format
        target_aspect = self.target_aspect_ratio[0] / self.target_aspect_ratio[1]
        target_width = int(frame_height * target_aspect)
        
        # Center the crop on the subject bbox
        bbox_center_x = bbox['x'] + bbox['width'] / 2
        bbox_center_y = bbox['y'] + bbox['height'] / 2
        
        # Calculate crop position
        crop_x = int(bbox_center_x - target_width / 2)
        crop_y = 0  # Usually keep full height for vertical video
        crop_width = target_width
        crop_height = frame_height
        
        # Constrain to frame boundaries
        if crop_x < 0:
            crop_x = 0
        elif crop_x + crop_width > frame_width:
            crop_x = frame_width - crop_width
        
        # Calculate zoom level
        zoom_level = frame_width / crop_width if crop_width > 0 else 1.0
        
        return {
            'x': crop_x,
            'y': crop_y,
            'width': crop_width,
            'height': crop_height,
            'zoom_level': zoom_level
        }
    
    def _apply_composition_rules(self, 
                               framing: FramingParameters,
                               frame_width: int,
                               frame_height: int,
                               subjects: List[Dict]) -> FramingParameters:
        """Apply composition rules like rule of thirds."""
        # Rule of thirds for vertical format
        thirds_x = framing.crop_width / 3
        thirds_y = framing.crop_height / 3
        
        # Find primary subject (usually largest or most central)
        if subjects:
            primary_subject = max(subjects, key=lambda s: s['bbox']['width'] * s['bbox']['height'])
            
            # Adjust crop to place primary subject on rule of thirds
            subject_center_x = primary_subject['bbox']['x'] + primary_subject['bbox']['width'] / 2
            subject_center_y = primary_subject['bbox']['y'] + primary_subject['bbox']['height'] / 2
            
            # Choose closest third line
            target_x_positions = [thirds_x, 2 * thirds_x]
            target_x = min(target_x_positions, key=lambda x: abs(x - (subject_center_x - framing.crop_x)))
            
            # Adjust crop position
            new_crop_x = int(subject_center_x - target_x)
            new_crop_x = max(0, min(new_crop_x, frame_width - framing.crop_width))
            
            framing.crop_x = new_crop_x
        
        return framing
    
    def _apply_audio_context(self, 
                           framing: FramingParameters,
                           audio_context: Dict) -> FramingParameters:
        """Apply audio-based framing adjustments."""
        # Get framing strategy from audio context
        strategy = audio_context.get('framing_strategy', 'medium_shot')
        
        zoom_adjustments = {
            'close_up': 1.5,
            'medium_shot': 1.0,
            'wide_shot': 0.8,
            'dynamic': 1.2
        }
        
        zoom_multiplier = zoom_adjustments.get(strategy, 1.0)
        
        # Adjust zoom level
        new_zoom = framing.zoom_level * zoom_multiplier
        
        # Recalculate crop dimensions
        if new_zoom != framing.zoom_level:
            original_width = framing.crop_width
            new_width = int(original_width / zoom_multiplier)
            
            # Center the new crop
            width_diff = original_width - new_width
            framing.crop_x += width_diff // 2
            framing.crop_width = new_width
            framing.zoom_level = new_zoom
        
        return framing
    
    def _apply_smooth_transitions(self, 
                                framing: FramingParameters,
                                previous_framing: FramingParameters) -> FramingParameters:
        """Apply smooth transitions between frames."""
        if not previous_framing:
            return framing
        
        # Smoothing factors
        smoothing_factor = self.config.get('zoom_behavior', {}).get('smoothing_factor', 0.7)
        
        # Smooth position changes
        framing.crop_x = int(
            previous_framing.crop_x * smoothing_factor + 
            framing.crop_x * (1 - smoothing_factor)
        )
        
        framing.crop_y = int(
            previous_framing.crop_y * smoothing_factor + 
            framing.crop_y * (1 - smoothing_factor)
        )
        
        # Smooth zoom changes
        framing.zoom_level = (
            previous_framing.zoom_level * smoothing_factor + 
            framing.zoom_level * (1 - smoothing_factor)
        )
        
        # Recalculate crop width based on smoothed zoom
        base_width = int(framing.crop_height * self.target_aspect_ratio[0] / self.target_aspect_ratio[1])
        framing.crop_width = int(base_width / framing.zoom_level)
        
        return framing
    
    def _validate_framing(self, 
                         framing: FramingParameters,
                         frame_width: int,
                         frame_height: int) -> FramingParameters:
        """Validate and constrain framing parameters."""
        # Ensure crop stays within frame boundaries
        framing.crop_x = max(0, min(framing.crop_x, frame_width - framing.crop_width))
        framing.crop_y = max(0, min(framing.crop_y, frame_height - framing.crop_height))
        
        # Ensure crop dimensions are valid
        framing.crop_width = max(1, min(framing.crop_width, frame_width - framing.crop_x))
        framing.crop_height = max(1, min(framing.crop_height, frame_height - framing.crop_y))
        
        # Ensure zoom level is within bounds
        zoom_config = self.config.get('zoom_parameters', {})
        min_zoom = zoom_config.get('min_zoom', 1.0)
        max_zoom = zoom_config.get('max_zoom', 3.0)
        
        framing.zoom_level = max(min_zoom, min(framing.zoom_level, max_zoom))
        
        return framing
    
    def _calculate_framing_confidence(self, 
                                    framing: FramingParameters,
                                    subjects: List[Dict],
                                    frame_width: int,
                                    frame_height: int) -> float:
        """Calculate confidence score for framing decision."""
        confidence_factors = []
        
        # Subject visibility factor
        if subjects:
            visible_subjects = 0
            for subject in subjects:
                if self._is_subject_visible_in_crop(subject, framing):
                    visible_subjects += 1
            
            visibility_score = visible_subjects / len(subjects)
            confidence_factors.append(visibility_score)
        else:
            confidence_factors.append(0.5)  # Neutral for no subjects
        
        # Composition factor (rule of thirds alignment)
        composition_score = self._calculate_composition_score(framing, subjects)
        confidence_factors.append(composition_score)
        
        # Stability factor (compared to history)
        stability_score = self._calculate_stability_score(framing)
        confidence_factors.append(stability_score)
        
        # Overall confidence
        return sum(confidence_factors) / len(confidence_factors)
    
    def _is_subject_visible_in_crop(self, subject: Dict, framing: FramingParameters) -> bool:
        """Check if subject is visible in the crop area."""
        bbox = subject['bbox']
        
        # Subject boundaries
        subject_left = bbox['x']
        subject_right = bbox['x'] + bbox['width']
        subject_top = bbox['y']
        subject_bottom = bbox['y'] + bbox['height']
        
        # Crop boundaries
        crop_left = framing.crop_x
        crop_right = framing.crop_x + framing.crop_width
        crop_top = framing.crop_y
        crop_bottom = framing.crop_y + framing.crop_height
        
        # Check overlap
        overlap_width = max(0, min(subject_right, crop_right) - max(subject_left, crop_left))
        overlap_height = max(0, min(subject_bottom, crop_bottom) - max(subject_top, crop_top))
        
        subject_area = bbox['width'] * bbox['height']
        overlap_area = overlap_width * overlap_height
        
        # Subject is visible if at least 70% is in the crop
        return overlap_area / subject_area >= 0.7 if subject_area > 0 else False
    
    def _calculate_composition_score(self, framing: FramingParameters, subjects: List[Dict]) -> float:
        """Calculate composition quality score."""
        if not subjects:
            return 0.5
        
        # Find primary subject
        primary_subject = max(subjects, key=lambda s: s['bbox']['width'] * s['bbox']['height'])
        
        # Calculate subject position relative to crop
        subject_center_x = primary_subject['bbox']['x'] + primary_subject['bbox']['width'] / 2
        subject_center_y = primary_subject['bbox']['y'] + primary_subject['bbox']['height'] / 2
        
        crop_center_x = framing.crop_x + framing.crop_width / 2
        crop_center_y = framing.crop_y + framing.crop_height / 2
        
        # Distance from center
        distance_x = abs(subject_center_x - crop_center_x) / (framing.crop_width / 2)
        distance_y = abs(subject_center_y - crop_center_y) / (framing.crop_height / 2)
        
        # Rule of thirds positions (ideal positions are at 1/3 and 2/3)
        thirds_score_x = 1.0 - min(
            abs(distance_x - 1/3), 
            abs(distance_x - 2/3),
            distance_x  # Center is also acceptable
        )
        
        # For vertical video, center Y is often preferred
        thirds_score_y = 1.0 - distance_y
        
        return (thirds_score_x + thirds_score_y) / 2
    
    def _calculate_stability_score(self, framing: FramingParameters) -> float:
        """Calculate stability score based on history."""
        if not self.framing_history:
            return 1.0
        
        # Compare with recent frames
        recent_frames = self.framing_history[-5:]  # Last 5 frames
        
        position_changes = []
        zoom_changes = []
        
        for prev_frame in recent_frames:
            # Position change
            pos_change = abs(framing.crop_x - prev_frame.crop_x) + abs(framing.crop_y - prev_frame.crop_y)
            position_changes.append(pos_change)
            
            # Zoom change
            zoom_change = abs(framing.zoom_level - prev_frame.zoom_level)
            zoom_changes.append(zoom_change)
        
        # Calculate stability (lower changes = higher stability)
        avg_position_change = sum(position_changes) / len(position_changes)
        avg_zoom_change = sum(zoom_changes) / len(zoom_changes)
        
        # Normalize to 0-1 scale (assuming max acceptable change)
        position_stability = max(0, 1.0 - avg_position_change / 100)  # 100px max change
        zoom_stability = max(0, 1.0 - avg_zoom_change / 0.5)  # 0.5 max zoom change
        
        return (position_stability + zoom_stability) / 2
    
    def _add_to_history(self, framing: FramingParameters):
        """Add framing to history."""
        self.framing_history.append(framing)
        
        # Keep only recent history
        if len(self.framing_history) > self.max_history:
            self.framing_history.pop(0)
    
    def get_quality_metrics(self) -> Dict:
        """Get current quality metrics."""
        return self.quality_metrics.copy()
    
    def reset_history(self):
        """Reset framing history."""
        self.framing_history.clear()
    
    def optimize_for_content_type(self, content_type: str) -> Dict:
        """
        Get optimization parameters for specific content types.
        
        Args:
            content_type: Type of content (e.g., 'presentation', 'conversation', 'action')
            
        Returns:
            Optimization parameters
        """
        optimization_profiles = {
            'presentation': {
                'prefer_wide_shots': True,
                'zoom_sensitivity': 0.3,
                'stability_priority': 0.9,
                'composition_priority': 0.7
            },
            'conversation': {
                'prefer_medium_shots': True,
                'zoom_sensitivity': 0.5,
                'stability_priority': 0.8,
                'composition_priority': 0.8
            },
            'action': {
                'prefer_dynamic_framing': True,
                'zoom_sensitivity': 0.8,
                'stability_priority': 0.5,
                'composition_priority': 0.6
            },
            'interview': {
                'prefer_close_ups': True,
                'zoom_sensitivity': 0.4,
                'stability_priority': 0.9,
                'composition_priority': 0.9
            }
        }
        
        return optimization_profiles.get(content_type, optimization_profiles['conversation'])
