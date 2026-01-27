# src/smart_zoom/enhanced_frame_processor.py
"""Enhanced frame-by-frame processor with improved face-centric and multi-subject intelligence"""

import cv2
import numpy as np
import torch
import copy
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time
from ..subject_detection import SubjectDetector
from .zoom_calculator import ZoomCalculator


@dataclass
class FrameAnalysis:
    """Comprehensive frame analysis result."""
    faces: List[Dict]
    people: List[Dict]
    objects: List[Dict]
    face_orientations: List[Dict]
    active_speaker: Optional[Dict]
    composition_quality: float
    face_visibility_ratio: float
    awkward_crop_risk: float
    primary_subjects: List[Dict]
    frame_score: float


@dataclass
class EnhancedZoomDecision:
    """Enhanced zoom decision with detailed reasoning."""
    zoom_level: float
    crop_position: Tuple[int, int, int, int]
    primary_subject: Dict
    face_focus_score: float
    orientation_score: float
    multi_subject_handling: str
    gesture_context_score: float
    composition_score: float
    confidence: float
    strategy: str
    reasoning: str
    quality_flags: List[str]


class EnhancedFrameProcessor:
    """
    Enhanced frame-by-frame processor that implements all smart zoom improvements
    with face-centric, multi-subject intelligence, and composition quality control.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize enhanced frame processor.
        
        Args:
            device: Device to use for processing
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.subject_detector = SubjectDetector(device=device)
        self.zoom_calculator = ZoomCalculator()
        
        # Enhanced detection parameters
        self.face_priority_weight = 0.9  # High priority for faces
        self.min_face_visibility = 0.15  # Minimum face area ratio
        self.max_face_visibility = 0.8   # Maximum face area ratio
        self.composition_target = 0.8    # Target composition quality
        
        # Multi-subject handling
        self.speaker_detection_window = 5  # Frames to track speaker
        self.subject_switching_cooldown = 30  # Frames before switching primary subject
        
        # Quality control
        self.face_focus_target = 0.9  # 90% of frames should be face-focused
        self.awkward_crop_threshold = 0.3  # Threshold for awkward crop detection
        
        # State tracking
        self.previous_primary_subject = None
        self.subject_switch_counter = 0
        self.frame_quality_history = []
        self.speaker_history = []
        
        self.logger.info("EnhancedFrameProcessor initialized")
    
    def process_frame(self, 
                     frame: np.ndarray,
                     audio_context: Dict = None,
                     timestamp: float = None) -> Tuple[np.ndarray, EnhancedZoomDecision]:
        """
        Process a single frame with enhanced smart zoom.
        
        Args:
            frame: Input frame
            audio_context: Audio analysis context
            timestamp: Current timestamp
            
        Returns:
            Tuple of (processed_frame, zoom_decision)
        """
        # Step 1: Comprehensive frame analysis
        analysis = self._analyze_frame_comprehensive(frame, audio_context, timestamp)
        
        # Step 2: Enhanced subject selection with multi-person intelligence
        primary_subjects = self._select_primary_subjects_enhanced(analysis, audio_context)
        
        # Step 3: Face-centric zoom calculation
        zoom_decision = self._calculate_enhanced_zoom(frame, analysis, primary_subjects, audio_context)
        
        # Step 4: Quality validation and correction
        validated_decision = self._validate_and_correct_decision(frame, zoom_decision, analysis)
        
        # Step 5: Apply enhanced zoom with composition optimization
        processed_frame = self._apply_enhanced_zoom(frame, validated_decision)
        
        # Step 6: Update tracking state
        self._update_tracking_state(validated_decision, analysis, timestamp)
        
        return processed_frame, validated_decision
    
    def _analyze_frame_comprehensive(self, 
                                   frame: np.ndarray,
                                   audio_context: Dict = None,
                                   timestamp: float = None) -> FrameAnalysis:
        """
        Perform comprehensive frame analysis including all detection types.
        
        Args:
            frame: Input frame
            audio_context: Audio analysis context
            timestamp: Current timestamp
            
        Returns:
            FrameAnalysis object with comprehensive results
        """
        # Detect all subjects
        all_subjects = self.subject_detector.detect_subjects(frame, timestamp)
        
        # Separate by type
        faces = [s for s in all_subjects if s['type'] == 'face']
        people = [s for s in all_subjects if s['type'] == 'person']
        objects = [s for s in all_subjects if s['type'] == 'object']
        
        # Enhanced face analysis with orientation detection
        face_orientations = self._analyze_face_orientations(faces, frame)
        
        # Active speaker detection
        active_speaker = self._detect_active_speaker(faces, people, audio_context, timestamp)
        
        # Composition quality analysis
        composition_quality = self._analyze_composition_quality(frame, all_subjects)
        
        # Face visibility ratio calculation
        face_visibility_ratio = self._calculate_face_visibility_ratio(faces, frame.shape)
        
        # Awkward crop risk assessment
        awkward_crop_risk = self._assess_awkward_crop_risk(all_subjects, frame.shape)
        
        # Primary subject selection
        primary_subjects = self._determine_primary_subjects(all_subjects, face_orientations, active_speaker)
        
        # Overall frame quality score
        frame_score = self._calculate_frame_score(composition_quality, face_visibility_ratio, awkward_crop_risk)
        
        return FrameAnalysis(
            faces=faces,
            people=people,
            objects=objects,
            face_orientations=face_orientations,
            active_speaker=active_speaker,
            composition_quality=composition_quality,
            face_visibility_ratio=face_visibility_ratio,
            awkward_crop_risk=awkward_crop_risk,
            primary_subjects=primary_subjects,
            frame_score=frame_score
        )
    
    def _analyze_face_orientations(self, faces: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Analyze face orientations to prioritize front-facing subjects.
        
        Args:
            faces: List of detected faces
            frame: Input frame
            
        Returns:
            List of face orientation data
        """
        orientations = []
        
        for face in faces:
            bbox = face['bbox']
            
            # Extract face region
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                continue
            
            # Simple orientation detection based on face symmetry and features
            orientation_score = self._detect_face_orientation(face_region)
            
            # Calculate visibility score based on face completeness
            visibility_score = self._calculate_face_visibility_score(bbox, frame.shape)
            
            # Get face landmarks if available
            landmarks = face.get('landmarks', [])
            
            orientation_data = {
                'face_id': face.get('tracking_id', 'unknown'),
                'bbox': bbox,
                'orientation': orientation_score['orientation'],  # 'frontal', 'profile', 'back'
                'orientation_confidence': orientation_score['confidence'],
                'visibility_score': visibility_score,
                'landmarks': landmarks,
                'priority_score': self._calculate_face_priority_score(orientation_score, visibility_score)
            }
            
            orientations.append(orientation_data)
        
        return orientations
    
    def _detect_face_orientation(self, face_region: np.ndarray) -> Dict:
        """
        Detect face orientation using simple heuristics.
        
        Args:
            face_region: Cropped face region
            
        Returns:
            Orientation analysis result
        """
        if face_region.size == 0:
            return {'orientation': 'unknown', 'confidence': 0.0}
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Split face into left and right halves
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = gray[:, w//2:]
        
        # Calculate symmetry score
        right_flipped = cv2.flip(right_half, 1)
        
        # Resize to match if needed
        if left_half.shape != right_flipped.shape:
            min_w = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_w]
            right_flipped = right_flipped[:, :min_w]
        
        # Calculate similarity between halves
        try:
            diff = cv2.absdiff(left_half, right_flipped)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        except:
            symmetry_score = 0.5
        
        # Detect edges to assess face completeness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Classify orientation
        if symmetry_score > 0.7 and edge_density > 0.05:
            orientation = 'frontal'
            confidence = min(1.0, symmetry_score + edge_density)
        elif symmetry_score > 0.4 and edge_density > 0.03:
            orientation = 'profile'
            confidence = symmetry_score * 0.8
        else:
            orientation = 'back'
            confidence = max(0.1, 1.0 - symmetry_score)
        
        return {
            'orientation': orientation,
            'confidence': confidence,
            'symmetry_score': symmetry_score,
            'edge_density': edge_density
        }
    
    def _calculate_face_visibility_score(self, bbox: List[int], frame_shape: Tuple[int, int]) -> float:
        """
        Calculate how well a face is visible in the frame.
        
        Args:
            bbox: Face bounding box
            frame_shape: Frame dimensions
            
        Returns:
            Visibility score (0-1)
        """
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        
        # Check if face is completely within frame
        if x1 < 0 or y1 < 0 or x2 > frame_w or y2 > frame_h:
            # Face is partially cut off
            visible_area = max(0, min(x2, frame_w) - max(x1, 0)) * max(0, min(y2, frame_h) - max(y1, 0))
            total_area = (x2 - x1) * (y2 - y1)
            visibility = visible_area / total_area if total_area > 0 else 0
        else:
            visibility = 1.0
        
        # Calculate relative size
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame_h * frame_w
        size_ratio = face_area / frame_area
        
        # Optimal size range (15% to 80% of frame)
        if self.min_face_visibility <= size_ratio <= self.max_face_visibility:
            size_score = 1.0
        elif size_ratio < self.min_face_visibility:
            size_score = size_ratio / self.min_face_visibility
        else:
            size_score = self.max_face_visibility / size_ratio
        
        return visibility * size_score
    
    def _calculate_face_priority_score(self, orientation_data: Dict, visibility_score: float) -> float:
        """
        Calculate priority score for face selection.
        
        Args:
            orientation_data: Face orientation analysis
            visibility_score: Face visibility score
            
        Returns:
            Priority score (0-1)
        """
        orientation = orientation_data['orientation']
        orientation_confidence = orientation_data['confidence']
        
        # Orientation weights
        orientation_weights = {
            'frontal': 1.0,
            'profile': 0.7,
            'back': 0.2,
            'unknown': 0.3
        }
        
        orientation_score = orientation_weights.get(orientation, 0.3) * orientation_confidence
        
        # Combine with visibility
        priority = (orientation_score * 0.6 + visibility_score * 0.4)
        
        return min(1.0, priority)
    
    def _detect_active_speaker(self, 
                              faces: List[Dict], 
                              people: List[Dict],
                              audio_context: Dict = None,
                              timestamp: float = None) -> Optional[Dict]:
        """
        Detect active speaker using audio-visual correlation.
        
        Args:
            faces: Detected faces
            people: Detected people
            audio_context: Audio analysis context
            timestamp: Current timestamp
            
        Returns:
            Active speaker data or None
        """
        if not audio_context or not faces:
            return None
        
        # Get current audio segment
        current_segment = self._get_current_audio_segment(audio_context, timestamp)
        if not current_segment:
            return None
        
        # Simple heuristic: face closest to center is likely the speaker
        # In practice, you'd use more sophisticated audio-visual correlation
        frame_center_x = 0.5  # Normalized
        frame_center_y = 0.4  # Slightly above center
        
        best_speaker = None
        best_score = 0
        
        for face in faces:
            bbox = face['bbox']
            face_center_x = (bbox[0] + bbox[2]) / 2
            face_center_y = (bbox[1] + bbox[3]) / 2
            
            # Normalize to frame size would be needed here
            # For now, use simple distance to center
            distance_score = 1.0 / (1.0 + abs(face_center_x - frame_center_x) + abs(face_center_y - frame_center_y))
            
            # Consider face orientation
            orientation_bonus = 1.0  # Would be calculated from orientation analysis
            
            # Combine scores
            speaker_score = distance_score * orientation_bonus * face.get('confidence', 0.5)
            
            if speaker_score > best_score:
                best_score = speaker_score
                best_speaker = {
                    'subject': face,
                    'confidence': speaker_score,
                    'reasoning': 'center_proximity_heuristic'
                }
        
        return best_speaker
    
    def _get_current_audio_segment(self, audio_context: Dict, timestamp: float) -> Optional[Dict]:
        """Get current audio segment for timestamp."""
        if not audio_context or not timestamp:
            return None
        
        segments = audio_context.get('segments', [])
        for segment in segments:
            if isinstance(segment, dict):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                if start <= timestamp <= end:
                    return segment
        
        return None
    
    def _analyze_composition_quality(self, frame: np.ndarray, subjects: List[Dict]) -> float:
        """
        Analyze composition quality of current frame.
        
        Args:
            frame: Input frame
            subjects: All detected subjects
            
        Returns:
            Composition quality score (0-1)
        """
        if not subjects:
            return 0.2
        
        frame_h, frame_w = frame.shape[:2]
        
        # Rule of thirds analysis
        third_lines_x = [frame_w / 3, 2 * frame_w / 3]
        third_lines_y = [frame_h / 3, 2 * frame_h / 3]
        
        composition_scores = []
        
        for subject in subjects:
            bbox = subject['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Distance to nearest third line
            min_dist_x = min(abs(center_x - line) for line in third_lines_x)
            min_dist_y = min(abs(center_y - line) for line in third_lines_y)
            
            # Normalize distances
            dist_score_x = 1.0 - min(1.0, min_dist_x / (frame_w / 6))
            dist_score_y = 1.0 - min(1.0, min_dist_y / (frame_h / 6))
            
            # Subject size scoring
            subject_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            frame_area = frame_h * frame_w
            size_ratio = subject_area / frame_area
            
            # Optimal size range
            if 0.1 <= size_ratio <= 0.6:
                size_score = 1.0
            else:
                size_score = max(0.2, 1.0 - abs(size_ratio - 0.35) / 0.35)
            
            # Combine scores
            subject_score = (dist_score_x + dist_score_y) * 0.3 + size_score * 0.7
            composition_scores.append(subject_score)
        
        return np.mean(composition_scores) if composition_scores else 0.2
    
    def _calculate_face_visibility_ratio(self, faces: List[Dict], frame_shape: Tuple[int, int]) -> float:
        """
        Calculate ratio of frame area occupied by visible faces.
        
        Args:
            faces: Detected faces
            frame_shape: Frame dimensions
            
        Returns:
            Face visibility ratio (0-1)
        """
        if not faces:
            return 0.0
        
        total_face_area = 0
        frame_area = frame_shape[0] * frame_shape[1]
        
        for face in faces:
            bbox = face['bbox']
            face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_face_area += face_area
        
        return min(1.0, total_face_area / frame_area)
    
    def _assess_awkward_crop_risk(self, subjects: List[Dict], frame_shape: Tuple[int, int]) -> float:
        """
        Assess risk of creating awkward crops.
        
        Args:
            subjects: All detected subjects
            frame_shape: Frame dimensions
            
        Returns:
            Awkward crop risk score (0-1, higher = more risk)
        """
        if not subjects:
            return 0.5
        
        risk_factors = []
        frame_h, frame_w = frame_shape[:2]
        
        for subject in subjects:
            bbox = subject['bbox']
            x1, y1, x2, y2 = bbox
            
            # Check for edge proximity (partial cut-off risk)
            edge_distances = [x1, y1, frame_w - x2, frame_h - y2]
            min_edge_distance = min(edge_distances)
            edge_risk = 1.0 if min_edge_distance < 10 else max(0, 1.0 - min_edge_distance / 50)
            
            # Check for very small subjects that might get lost
            subject_area = (x2 - x1) * (y2 - y1)
            frame_area = frame_h * frame_w
            size_ratio = subject_area / frame_area
            size_risk = 1.0 if size_ratio < 0.05 else 0.0
            
            # Check for subjects at frame edges
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Risk if subject center is too close to edges
            edge_proximity_risk = 0.0
            if center_x < frame_w * 0.1 or center_x > frame_w * 0.9:
                edge_proximity_risk += 0.5
            if center_y < frame_h * 0.1 or center_y > frame_h * 0.9:
                edge_proximity_risk += 0.5
            
            subject_risk = max(edge_risk, size_risk, edge_proximity_risk)
            risk_factors.append(subject_risk)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    def _determine_primary_subjects(self, 
                                  all_subjects: List[Dict],
                                  face_orientations: List[Dict],
                                  active_speaker: Optional[Dict]) -> List[Dict]:
        """
        Determine primary subjects for framing with enhanced logic.
        
        Args:
            all_subjects: All detected subjects
            face_orientations: Face orientation analysis
            active_speaker: Active speaker if detected
            
        Returns:
            List of primary subjects
        """
        if not all_subjects:
            return []
        
        # Create enhanced subject scoring
        enhanced_subjects = []
        
        for subject in all_subjects:
            subject_id = subject.get('tracking_id', 'unknown')
            subject_type = subject['type']
            base_confidence = subject['confidence']
            
            # Find corresponding face orientation data
            orientation_data = None
            for face_orient in face_orientations:
                if face_orient['face_id'] == subject_id:
                    orientation_data = face_orient
                    break
            
            # Calculate enhanced score
            enhanced_score = self._calculate_enhanced_subject_score(
                subject, orientation_data, active_speaker
            )
            
            enhanced_subject = subject.copy()
            enhanced_subject['enhanced_score'] = enhanced_score
            enhanced_subject['orientation_data'] = orientation_data
            enhanced_subjects.append(enhanced_subject)
        
        # Sort by enhanced score
        enhanced_subjects.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        # Select top subjects with smart multi-subject handling
        return self._smart_multi_subject_selection(enhanced_subjects)
    
    def _calculate_enhanced_subject_score(self, 
                                        subject: Dict,
                                        orientation_data: Optional[Dict],
                                        active_speaker: Optional[Dict]) -> float:
        """
        Calculate enhanced subject score with all factors.
        
        Args:
            subject: Subject data
            orientation_data: Face orientation data
            active_speaker: Active speaker data
            
        Returns:
            Enhanced score (0-1)
        """
        base_score = subject['confidence'] * subject.get('priority', 0.5)
        
        # Orientation bonus
        orientation_bonus = 1.0
        if orientation_data:
            orientation_bonus = orientation_data['priority_score']
        
        # Speaker bonus
        speaker_bonus = 1.0
        if active_speaker and subject.get('tracking_id') == active_speaker['subject'].get('tracking_id'):
            speaker_bonus = 1.5
        
        # Face type bonus
        face_bonus = 1.2 if subject['type'] == 'face' else 1.0
        
        # Size appropriateness bonus
        bbox = subject['bbox']
        subject_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Assume reasonable frame size for calculation
        estimated_frame_area = 1920 * 1080  # Will be corrected with actual frame
        size_ratio = subject_area / estimated_frame_area
        
        if 0.15 <= size_ratio <= 0.8:
            size_bonus = 1.0
        else:
            size_bonus = 0.7
        
        enhanced_score = base_score * orientation_bonus * speaker_bonus * face_bonus * size_bonus
        
        return min(1.0, enhanced_score)
    
    def _smart_multi_subject_selection(self, enhanced_subjects: List[Dict]) -> List[Dict]:
        """
        Smart selection of multiple subjects avoiding common issues.
        
        Args:
            enhanced_subjects: Subjects with enhanced scores
            
        Returns:
            Selected primary subjects
        """
        if not enhanced_subjects:
            return []
        
        selected = []
        
        # Always include the highest scoring subject
        primary = enhanced_subjects[0]
        selected.append(primary)
        
        # Check if we should include additional subjects
        if len(enhanced_subjects) > 1:
            secondary = enhanced_subjects[1]
            
            # Only include secondary if it's close in score and spatially appropriate
            score_ratio = secondary['enhanced_score'] / primary['enhanced_score']
            
            if score_ratio > 0.7:  # Secondary is reasonably close in score
                # Check spatial relationship
                spatial_ok = self._check_spatial_compatibility(primary, secondary)
                
                if spatial_ok:
                    selected.append(secondary)
        
        return selected[:2]  # Maximum 2 primary subjects
    
    def _check_spatial_compatibility(self, subject1: Dict, subject2: Dict) -> bool:
        """
        Check if two subjects can be framed together without awkward composition.
        
        Args:
            subject1: First subject
            subject2: Second subject
            
        Returns:
            True if spatially compatible
        """
        bbox1 = subject1['bbox']
        bbox2 = subject2['bbox']
        
        # Calculate centers
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate distance
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        
        # Check for overlap
        overlap = self._calculate_bbox_overlap(bbox1, bbox2)
        
        # Spatial compatibility rules
        if overlap > 0.3:  # Too much overlap
            return False
        
        if distance < 50:  # Too close together
            return False
        
        # Check if they're positioned to avoid "space between" framing
        # Subjects should be either clearly separate or clearly together
        bbox1_width = bbox1[2] - bbox1[0]
        bbox2_width = bbox2[2] - bbox2[0]
        
        # If horizontal gap is larger than subject widths, might create awkward framing
        horizontal_gap = abs(center1_x - center2_x) - (bbox1_width + bbox2_width) / 2
        if horizontal_gap > max(bbox1_width, bbox2_width):
            return False
        
        return True
    
    def _calculate_bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y1_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        
        overlap_area = x1_overlap * y1_overlap
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = area1 + area2 - overlap_area
        
        return overlap_area / union_area if union_area > 0 else 0.0
    
    def _calculate_frame_score(self, 
                             composition_quality: float,
                             face_visibility_ratio: float,
                             awkward_crop_risk: float) -> float:
        """
        Calculate overall frame quality score.
        
        Args:
            composition_quality: Composition quality score
            face_visibility_ratio: Face visibility ratio
            awkward_crop_risk: Awkward crop risk score
            
        Returns:
            Overall frame quality score (0-1)
        """
        # Weight the factors
        composition_weight = 0.3
        face_visibility_weight = 0.4
        awkward_risk_weight = 0.3
        
        # Invert awkward crop risk (higher risk = lower score)
        awkward_score = 1.0 - awkward_crop_risk
        
        frame_score = (
            composition_quality * composition_weight +
            face_visibility_ratio * face_visibility_weight +
            awkward_score * awkward_risk_weight
        )
        
        return min(1.0, frame_score)
    
    def _select_primary_subjects_enhanced(self, 
                                        analysis: FrameAnalysis,
                                        audio_context: Dict = None) -> List[Dict]:
        """
        Enhanced primary subject selection with multi-person intelligence.
        
        Args:
            analysis: Comprehensive frame analysis
            audio_context: Audio context
            
        Returns:
            Selected primary subjects
        """
        # Use the primary subjects from analysis
        primary_subjects = analysis.primary_subjects
        
        # Apply speaker focus if available
        if analysis.active_speaker and audio_context:
            speaker_subject = analysis.active_speaker['subject']
            
            # Boost speaker priority if not already primary
            speaker_in_primary = any(
                s.get('tracking_id') == speaker_subject.get('tracking_id')
                for s in primary_subjects
            )
            
            if not speaker_in_primary and len(primary_subjects) > 0:
                # Replace lowest scoring primary with speaker
                primary_subjects[-1] = speaker_subject
        
        return primary_subjects[:2]  # Maximum 2 subjects
    
    def _calculate_enhanced_zoom(self, 
                               frame: np.ndarray,
                               analysis: FrameAnalysis,
                               primary_subjects: List[Dict],
                               audio_context: Dict = None) -> EnhancedZoomDecision:
        """
        Calculate enhanced zoom decision with detailed analysis.
        
        Args:
            frame: Input frame
            analysis: Frame analysis
            primary_subjects: Primary subjects for framing
            audio_context: Audio context
            
        Returns:
            Enhanced zoom decision
        """
        frame_shape = frame.shape[:2]
        
        if not primary_subjects:
            return self._get_default_enhanced_decision(frame_shape)
        
        # Use zoom calculator for base calculation
        zoom_result = self.zoom_calculator.calculate_optimal_zoom(
            frame_shape, primary_subjects, audio_context
        )
        
        # Enhanced analysis
        face_focus_score = self._calculate_face_focus_score(primary_subjects, zoom_result)
        orientation_score = self._calculate_orientation_score(primary_subjects, analysis.face_orientations)
        multi_subject_handling = self._determine_multi_subject_strategy(primary_subjects)
        gesture_context_score = self._calculate_gesture_context_score(primary_subjects, frame)
        composition_score = analysis.composition_quality
        
        # Overall confidence with enhanced factors
        enhanced_confidence = self._calculate_enhanced_confidence(
            zoom_result['confidence'], face_focus_score, orientation_score, composition_score
        )
        
        # Quality flags
        quality_flags = self._generate_quality_flags(analysis, zoom_result)
        
        return EnhancedZoomDecision(
            zoom_level=zoom_result['zoom_level'],
            crop_position=zoom_result['crop_position'],
            primary_subject=primary_subjects[0],
            face_focus_score=face_focus_score,
            orientation_score=orientation_score,
            multi_subject_handling=multi_subject_handling,
            gesture_context_score=gesture_context_score,
            composition_score=composition_score,
            confidence=enhanced_confidence,
            strategy=zoom_result['strategy'],
            reasoning=zoom_result['reasoning'],
            quality_flags=quality_flags
        )
    
    def _calculate_face_focus_score(self, subjects: List[Dict], zoom_result: Dict) -> float:
        """Calculate how well the zoom focuses on faces."""
        face_subjects = [s for s in subjects if s['type'] == 'face']
        
        if not face_subjects:
            return 0.1
        
        # Calculate face coverage in crop
        crop_x, crop_y, crop_w, crop_h = zoom_result['crop_position']
        crop_area = crop_w * crop_h
        
        total_face_area_in_crop = 0
        for face in face_subjects:
            bbox = face['bbox']
            # Calculate intersection with crop
            x1 = max(bbox[0], crop_x)
            y1 = max(bbox[1], crop_y)
            x2 = min(bbox[2], crop_x + crop_w)
            y2 = min(bbox[3], crop_y + crop_h)
            
            if x2 > x1 and y2 > y1:
                face_area_in_crop = (x2 - x1) * (y2 - y1)
                total_face_area_in_crop += face_area_in_crop
        
        face_ratio = total_face_area_in_crop / crop_area if crop_area > 0 else 0
        
        # Score based on target ratio (15% to 80%)
        if 0.15 <= face_ratio <= 0.8:
            return 1.0
        elif face_ratio < 0.15:
            return face_ratio / 0.15
        else:
            return 0.8 / face_ratio
    
    def _calculate_orientation_score(self, subjects: List[Dict], face_orientations: List[Dict]) -> float:
        """Calculate orientation quality score."""
        if not face_orientations:
            return 0.5
        
        orientation_scores = []
        for face_orient in face_orientations:
            orientation_scores.append(face_orient['priority_score'])
        
        return np.mean(orientation_scores) if orientation_scores else 0.5
    
    def _determine_multi_subject_strategy(self, subjects: List[Dict]) -> str:
        """Determine strategy for handling multiple subjects."""
        if len(subjects) <= 1:
            return "single_subject"
        elif len(subjects) == 2:
            return "dual_subject_balanced"
        else:
            return "group_subject_prioritized"
    
    def _calculate_gesture_context_score(self, subjects: List[Dict], frame: np.ndarray) -> float:
        """Calculate how well gestures are contextualized with faces."""
        # Simplified gesture context scoring
        # In practice, you'd detect hand/gesture regions and check face inclusion
        
        gesture_score = 0.8  # Default good score
        
        for subject in subjects:
            if subject['type'] == 'person':
                # Check if person bounding box includes both face region and gesture region
                bbox = subject['bbox']
                bbox_height = bbox[3] - bbox[1]
                bbox_width = bbox[2] - bbox[0]
                
                # If person bbox is very wide, might be showing gestures
                aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1
                
                if aspect_ratio > 1.5:  # Wide bbox suggests hand gestures
                    # Check if face is likely included (upper portion)
                    # This is a simplified heuristic
                    gesture_score = 0.9
                else:
                    gesture_score = 0.8
        
        return gesture_score
    
    def _calculate_enhanced_confidence(self, 
                                     base_confidence: float,
                                     face_focus_score: float,
                                     orientation_score: float,
                                     composition_score: float) -> float:
        """Calculate enhanced confidence score."""
        weights = {
            'base': 0.3,
            'face_focus': 0.3,
            'orientation': 0.2,
            'composition': 0.2
        }
        
        enhanced_confidence = (
            base_confidence * weights['base'] +
            face_focus_score * weights['face_focus'] +
            orientation_score * weights['orientation'] +
            composition_score * weights['composition']
        )
        
        return min(1.0, enhanced_confidence)
    
    def _generate_quality_flags(self, analysis: FrameAnalysis, zoom_result: Dict) -> List[str]:
        """Generate quality flags for the decision."""
        flags = []
        
        if analysis.face_visibility_ratio < 0.1:
            flags.append("low_face_visibility")
        
        if analysis.awkward_crop_risk > self.awkward_crop_threshold:
            flags.append("awkward_crop_risk")
        
        if analysis.composition_quality < 0.5:
            flags.append("poor_composition")
        
        if zoom_result['zoom_level'] > 2.5:
            flags.append("high_zoom_level")
        
        if len(analysis.primary_subjects) == 0:
            flags.append("no_primary_subjects")
        
        return flags
    
    def _get_default_enhanced_decision(self, frame_shape: Tuple[int, int]) -> EnhancedZoomDecision:
        """Get default enhanced decision when no subjects detected."""
        return EnhancedZoomDecision(
            zoom_level=1.0,
            crop_position=(0, 0, frame_shape[1], frame_shape[0]),
            primary_subject={},
            face_focus_score=0.0,
            orientation_score=0.0,
            multi_subject_handling="no_subjects",
            gesture_context_score=0.0,
            composition_score=0.2,
            confidence=0.1,
            strategy="default_center",
            reasoning="No subjects detected, using default center framing",
            quality_flags=["no_subjects_detected"]
        )
    
    def _validate_and_correct_decision(self, 
                                     frame: np.ndarray,
                                     decision: EnhancedZoomDecision,
                                     analysis: FrameAnalysis) -> EnhancedZoomDecision:
        """
        Validate and correct zoom decision to prevent common issues.
        
        Args:
            frame: Input frame
            decision: Initial zoom decision
            analysis: Frame analysis
            
        Returns:
            Validated and corrected decision
        """
        corrected_decision = decision
        
        # Check for awkward crop risks
        if "awkward_crop_risk" in decision.quality_flags:
            corrected_decision = self._correct_awkward_crop(frame, decision, analysis)
        
        # Check face visibility
        if decision.face_focus_score < 0.5:
            corrected_decision = self._correct_face_visibility(frame, corrected_decision, analysis)
        
        # Check multi-subject handling
        if len(analysis.primary_subjects) > 1:
            corrected_decision = self._correct_multi_subject_framing(frame, corrected_decision, analysis)
        
        return corrected_decision
    
    def _correct_awkward_crop(self, 
                            frame: np.ndarray,
                            decision: EnhancedZoomDecision,
                            analysis: FrameAnalysis) -> EnhancedZoomDecision:
        """Adjust crop to focus on tracking subjects properly."""
        # Create a copy of the decision to modify
        corrected_decision = copy.deepcopy(decision)
        
        # For tracking, we want to ensure we're not zooming in too aggressively
        corrected_zoom = max(1.0, decision.zoom_level * 0.9)
        
        # Recalculate crop with focus on full subject visibility
        frame_shape = frame.shape[:2]
        
        # Get subject position (typically center-weighted but depends on content)
        center_position = (0.5, 0.5)  # Center tracking by default
        
        # If we have primary subjects, adjust to track them
        if analysis.primary_subjects:
            # Calculate center of all primary subjects
            centers_x = []
            centers_y = []
            
            for subject in analysis.primary_subjects:
                bbox = subject['bbox']
                centers_x.append((bbox[0] + bbox[2]) / 2)
                centers_y.append((bbox[1] + bbox[3]) / 2)
            
            # Use average center as tracking point
            if centers_x and centers_y:
                center_position = (
                    np.mean(centers_x) / frame_shape[1], 
                    np.mean(centers_y) / frame_shape[0]
                )
        
        crop_rect = self.zoom_calculator._calculate_crop_rectangle(
            frame_shape, corrected_zoom, center_position
        )
        
        corrected_decision.zoom_level = corrected_zoom
        corrected_decision.crop_position = (crop_rect['x'], crop_rect['y'], crop_rect['width'], crop_rect['height'])
        corrected_decision.reasoning += " | Adjusted for tracking"
        
        # Remove the awkward crop flag
        if "awkward_crop_risk" in corrected_decision.quality_flags:
            corrected_decision.quality_flags.remove("awkward_crop_risk")
        
        return corrected_decision
    
    def _correct_face_visibility(self, 
                               frame: np.ndarray,
                               decision: EnhancedZoomDecision,
                               analysis: FrameAnalysis) -> EnhancedZoomDecision:
        """Correct face visibility issues."""
        # Adjust zoom to better track faces
        if analysis.faces:
            # Find best face
            best_face = max(analysis.faces, key=lambda f: f.get('confidence', 0))
            
            # Calculate optimal zoom for this face
            bbox = best_face['bbox']
            face_height = bbox[3] - bbox[1]
            frame_height = frame.shape[0]
            
            # For tracking, we want to ensure the face is visible but not necessarily 
            # filling a specific percentage of the frame
            target_ratio = 0.35  # Reduced ratio to show more context
            optimal_zoom = target_ratio * frame_height / face_height
            
            # Limit zoom but prioritize tracking the whole subject
            optimal_zoom = max(1.0, min(2.5, optimal_zoom))
            
            decision.zoom_level = optimal_zoom
            decision.face_focus_score = 0.8
            decision.reasoning += " | Tracking face"
        
        return decision
    
    def _correct_multi_subject_framing(self, 
                                     frame: np.ndarray,
                                     decision: EnhancedZoomDecision,
                                     analysis: FrameAnalysis) -> EnhancedZoomDecision:
        """Correct multi-subject framing issues."""
        # Ensure we're not focusing on space between subjects
        primary_subjects = analysis.primary_subjects[:2]  # Max 2 subjects
        
        if len(primary_subjects) == 2:
            # Check if current crop captures both subjects well
            crop_x, crop_y, crop_w, crop_h = decision.crop_position
            
            subjects_in_crop = 0
            for subject in primary_subjects:
                bbox = subject['bbox']
                # Check if subject center is in crop
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                if crop_x <= center_x <= crop_x + crop_w and crop_y <= center_y <= crop_y + crop_h:
                    subjects_in_crop += 1
            
            if subjects_in_crop < 2:
                # Adjust crop to include both subjects
                all_x = [s['bbox'][0] for s in primary_subjects] + [s['bbox'][2] for s in primary_subjects]
                all_y = [s['bbox'][1] for s in primary_subjects] + [s['bbox'][3] for s in primary_subjects]
                
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                # Tracking-focused approach: ensure we capture the full subject area
                # without additional padding that might dilute subject focus
                decision.crop_position = (min_x, min_y, max_x - min_x, max_y - min_y)
                decision.reasoning += " | Tracking multiple subjects"
        
        return decision
    
    def _apply_enhanced_zoom(self, frame: np.ndarray, decision: EnhancedZoomDecision) -> np.ndarray:
        """
        Apply enhanced zoom with dynamic tracking instead of static resizing.
        
        Args:
            frame: Input frame
            decision: Enhanced zoom decision
            
        Returns:
            Processed frame
        """
        # Extract crop region
        crop_x, crop_y, crop_w, crop_h = decision.crop_position
        
        # Ensure crop is within bounds
        frame_h, frame_w = frame.shape[:2]
        crop_x = max(0, min(crop_x, frame_w - crop_w))
        crop_y = max(0, min(crop_y, frame_h - crop_h))
        crop_w = min(crop_w, frame_w - crop_x)
        crop_h = min(crop_h, frame_h - crop_y)
        
        # Extract the cropped region
        cropped_frame = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        
        # Target vertical format dimensions (9:16 aspect ratio)
        target_width = 1080
        target_height = 1920
        
        # Create a target frame of the desired dimensions
        output_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Track the subject by centering the cropped region
        # Resize without maintaining aspect ratio to fill the entire vertical frame
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Use the resized frame directly without applying additional effects
        return resized_frame
    
    def _apply_visual_enhancements(self, frame: np.ndarray) -> np.ndarray:
        """
        No visual enhancements applied as per user request.
        Simply return the original frame.
        """
        # Return the frame without any modifications
        return frame
    
    def _update_tracking_state(self, 
                             decision: EnhancedZoomDecision,
                             analysis: FrameAnalysis,
                             timestamp: float = None):
        """Update tracking state for temporal consistency."""
        # Update frame quality history
        self.frame_quality_history.append(analysis.frame_score)
        if len(self.frame_quality_history) > 30:  # Keep last 30 frames
            self.frame_quality_history.pop(0)
        
        # Update speaker history
        if analysis.active_speaker:
            self.speaker_history.append({
                'timestamp': timestamp,
                'speaker': analysis.active_speaker['subject'].get('tracking_id'),
                'confidence': analysis.active_speaker['confidence']
            })
            if len(self.speaker_history) > 10:  # Keep last 10 speaker detections
                self.speaker_history.pop(0)
        
        # Update primary subject tracking
        current_primary = decision.primary_subject.get('tracking_id')
        if current_primary != self.previous_primary_subject:
            self.subject_switch_counter += 1
            self.previous_primary_subject = current_primary
    
    def get_processing_stats(self) -> Dict:
        """Get comprehensive processing statistics."""
        avg_frame_quality = np.mean(self.frame_quality_history) if self.frame_quality_history else 0.0
        
        return {
            'average_frame_quality': avg_frame_quality,
            'subject_switches': self.subject_switch_counter,
            'recent_speaker_changes': len(set(s['speaker'] for s in self.speaker_history[-5:])) if self.speaker_history else 0,
            'face_priority_weight': self.face_priority_weight,
            'device': self.device,
            'quality_target': self.composition_target
        }
    
    def reset_state(self):
        """Reset processor state for new video."""
        self.previous_primary_subject = None
        self.subject_switch_counter = 0
        self.frame_quality_history.clear()
        self.speaker_history.clear()
        self.zoom_calculator.reset_state()
        
        self.logger.info("EnhancedFrameProcessor state reset")
