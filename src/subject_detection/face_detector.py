# src/subject_detection/face_detector.py
"""Face detection using MediaPipe"""

import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class FaceDetector:
    """
    Face detection using MediaPipe Face Detection.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize face detector.
        
        Args:
            device: Device to use (not used by MediaPipe)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        # Try to initialize MediaPipe face detection; gracefully disable if not available
        if mp is None or not hasattr(mp, 'solutions'):
            self.logger.warning("MediaPipe 'solutions' not available; face detection disabled.")
            self.mp_face_detection = None
            self.mp_drawing = None
            self.face_detection = None
            self.loaded = False
            return

        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            # Initialize MediaPipe Face Detection with static image mode to avoid timestamp issues
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,              # Full-range model improves small/far faces
                min_detection_confidence=0.4    # Lower threshold to reduce misses
            )
            self.loaded = True
        except Exception as e:
            self.logger.warning("Failed to initialize MediaPipe FaceDetection: %s", e)
            self.mp_face_detection = None
            self.mp_drawing = None
            self.face_detection = None
            self.loaded = False
            return
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected faces with metadata
        """
        if not self.loaded:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make frame contiguous to avoid MediaPipe issues
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Detect faces
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    # Get bounding box
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to absolute coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Ensure bounding box is within frame
                    x = max(0, x)
                    y = max(0, y)
                    width = min(width, w - x)
                    height = min(height, h - y)
                    
                    face = {
                        'bbox': [x, y, x + width, y + height],
                        'confidence': detection.score[0],
                        'landmarks': self._extract_landmarks(detection, w, h),
                        'attributes': self._analyze_face_attributes(frame, [x, y, x + width, y + height])
                    }
                    
                    faces.append(face)
            
            return faces
            
        except Exception as e:
            self.logger.error(f"Face detection failed: {e}")
            return []
    
    def _extract_landmarks(self, detection, frame_width: int, frame_height: int) -> List[Tuple[int, int]]:
        """Extract facial landmarks."""
        landmarks = []
        
        if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_keypoints'):
            for keypoint in detection.location_data.relative_keypoints:
                x = int(keypoint.x * frame_width)
                y = int(keypoint.y * frame_height)
                landmarks.append((x, y))
        
        return landmarks
    
    def _analyze_face_attributes(self, frame: np.ndarray, bbox: List[int]) -> Dict:
        """Analyze face attributes (simplified)."""
        x1, y1, x2, y2 = bbox
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            return {}
        
        # Simple analysis based on face region
        face_area = (x2 - x1) * (y2 - y1)
        frame_area = frame.shape[0] * frame.shape[1]
        
        return {
            'face_area_ratio': face_area / frame_area,
            'face_size': 'large' if face_area > 10000 else 'medium' if face_area > 5000 else 'small',
            'position': self._get_face_position(bbox, frame.shape)
        }
    
    def _get_face_position(self, bbox: List[int], frame_shape: Tuple[int, int, int]) -> str:
        """Determine face position in frame."""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine position
        if center_x < w / 3:
            h_pos = 'left'
        elif center_x > 2 * w / 3:
            h_pos = 'right'
        else:
            h_pos = 'center'
        
        if center_y < h / 3:
            v_pos = 'top'
        elif center_y > 2 * h / 3:
            v_pos = 'bottom'
        else:
            v_pos = 'middle'
        
        return f"{v_pos}_{h_pos}"
    
    def is_loaded(self) -> bool:
        """Check if detector is loaded."""
        return self.loaded
    
    def cleanup(self):
        """Clean up resources."""
        if self.face_detection:
            self.face_detection.close()
        self.loaded = False
