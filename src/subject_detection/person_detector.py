# src/subject_detection/person_detector.py
"""Person detection using MediaPipe Pose"""

import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class PersonDetector:
    """
    Person detection using MediaPipe Pose estimation.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize person detector.
        
        Args:
            device: Device to use (not used by MediaPipe)
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        if mp is None or not hasattr(mp, 'solutions'):
            self.logger.warning("MediaPipe 'solutions' not available; person detection disabled.")
            self.mp_pose = None
            self.mp_drawing = None
            self.pose = None
            self.loaded = False
            return

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            # Initialize MediaPipe Pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.loaded = True
        except Exception as e:
            self.logger.warning("Failed to initialize MediaPipe Pose: %s", e)
            self.mp_pose = None
            self.mp_drawing = None
            self.pose = None
            self.loaded = False
            return
    
    def detect_people(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in frame using pose estimation.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected people with metadata
        """
        if not self.loaded:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Make frame contiguous to avoid MediaPipe issues
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Process frame
            results = self.pose.process(rgb_frame)
            
            people = []
            if results.pose_landmarks:
                h, w, _ = frame.shape
                
                # Extract keypoints
                keypoints = self._extract_keypoints(results.pose_landmarks, w, h)
                
                # Calculate bounding box from keypoints
                bbox = self._calculate_bbox_from_keypoints(keypoints)
                
                # Analyze pose
                pose_analysis = self._analyze_pose(keypoints)
                
                person = {
                    'bbox': bbox,
                    'confidence': self._calculate_confidence(keypoints),
                    'keypoints': keypoints,
                    'pose': pose_analysis,
                    'visibility': self._calculate_visibility(keypoints)
                }
                
                people.append(person)
            
            return people
            
        except Exception as e:
            self.logger.error(f"Person detection failed: {e}")
            return []
    
    def _extract_keypoints(self, landmarks, width: int, height: int) -> List[Dict]:
        """Extract keypoints from MediaPipe landmarks."""
        keypoints = []
        
        for i, landmark in enumerate(landmarks.landmark):
            keypoint = {
                'id': i,
                'x': int(landmark.x * width),
                'y': int(landmark.y * height),
                'z': landmark.z,
                'visibility': landmark.visibility,
                'name': self._get_keypoint_name(i)
            }
            keypoints.append(keypoint)
        
        return keypoints
    
    def _get_keypoint_name(self, keypoint_id: int) -> str:
        """Get name for keypoint ID."""
        keypoint_names = {
            0: 'nose',
            1: 'left_eye_inner',
            2: 'left_eye',
            3: 'left_eye_outer',
            4: 'right_eye_inner',
            5: 'right_eye',
            6: 'right_eye_outer',
            7: 'left_ear',
            8: 'right_ear',
            9: 'mouth_left',
            10: 'mouth_right',
            11: 'left_shoulder',
            12: 'right_shoulder',
            13: 'left_elbow',
            14: 'right_elbow',
            15: 'left_wrist',
            16: 'right_wrist',
            17: 'left_pinky',
            18: 'right_pinky',
            19: 'left_index',
            20: 'right_index',
            21: 'left_thumb',
            22: 'right_thumb',
            23: 'left_hip',
            24: 'right_hip',
            25: 'left_knee',
            26: 'right_knee',
            27: 'left_ankle',
            28: 'right_ankle',
            29: 'left_heel',
            30: 'right_heel',
            31: 'left_foot_index',
            32: 'right_foot_index'
        }
        
        return keypoint_names.get(keypoint_id, f'keypoint_{keypoint_id}')
    
    def _calculate_bbox_from_keypoints(self, keypoints: List[Dict]) -> List[int]:
        """Calculate bounding box from visible keypoints."""
        visible_keypoints = [kp for kp in keypoints if kp['visibility'] > 0.35]  # lower threshold
        
        if not visible_keypoints:
            return [0, 0, 0, 0]
        
        x_coords = [kp['x'] for kp in visible_keypoints]
        y_coords = [kp['y'] for kp in visible_keypoints]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # Add more padding to avoid tight crops
        padding = 30
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = max_x + padding
        max_y = max_y + padding
        
        return [min_x, min_y, max_x, max_y]
    
    def _calculate_confidence(self, keypoints: List[Dict]) -> float:
        """Calculate overall confidence based on keypoint visibility."""
        if not keypoints:
            return 0.0
        
        visible_keypoints = [kp for kp in keypoints if kp['visibility'] > 0.35]
        confidence = len(visible_keypoints) / max(len(keypoints), 1)
        
        return confidence
    
    def _analyze_pose(self, keypoints: List[Dict]) -> Dict:
        """Analyze pose from keypoints."""
        if not keypoints:
            return {}
        
        # Get key body parts
        nose = next((kp for kp in keypoints if kp['name'] == 'nose'), None)
        left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
        right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
        left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
        right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)
        
        pose_analysis = {
            'body_orientation': self._calculate_body_orientation(left_shoulder, right_shoulder),
            'head_position': self._analyze_head_position(nose, left_shoulder, right_shoulder),
            'body_posture': self._analyze_body_posture(keypoints),
            'is_facing_camera': self._is_facing_camera(keypoints)
        }
        
        return pose_analysis
    
    def _calculate_body_orientation(self, left_shoulder: Dict, right_shoulder: Dict) -> str:
        """Calculate body orientation."""
        if not left_shoulder or not right_shoulder:
            return 'unknown'
        
        # Calculate shoulder angle
        dx = right_shoulder['x'] - left_shoulder['x']
        dy = right_shoulder['y'] - left_shoulder['y']
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        if abs(angle) < 15:
            return 'straight'
        elif angle > 15:
            return 'tilted_right'
        else:
            return 'tilted_left'
    
    def _analyze_head_position(self, nose: Dict, left_shoulder: Dict, right_shoulder: Dict) -> str:
        """Analyze head position relative to shoulders."""
        if not all([nose, left_shoulder, right_shoulder]):
            return 'unknown'
        
        shoulder_center_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        
        if abs(nose['x'] - shoulder_center_x) < 20:
            return 'centered'
        elif nose['x'] < shoulder_center_x:
            return 'left'
        else:
            return 'right'
    
    def _analyze_body_posture(self, keypoints: List[Dict]) -> str:
        """Analyze overall body posture."""
        # Simple posture analysis
        visible_upper_body = sum(1 for kp in keypoints 
                               if kp['name'] in ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow']
                               and kp['visibility'] > 0.5)
        
        visible_lower_body = sum(1 for kp in keypoints
                               if kp['name'] in ['left_hip', 'right_hip', 'left_knee', 'right_knee']
                               and kp['visibility'] > 0.5)
        
        if visible_upper_body >= 3 and visible_lower_body >= 3:
            return 'full_body'
        elif visible_upper_body >= 2:
            return 'upper_body'
        else:
            return 'partial'
    
    def _is_facing_camera(self, keypoints: List[Dict]) -> bool:
        """Determine if person is facing the camera."""
        # Check if both eyes are visible
        left_eye = next((kp for kp in keypoints if kp['name'] == 'left_eye'), None)
        right_eye = next((kp for kp in keypoints if kp['name'] == 'right_eye'), None)
        
        if left_eye and right_eye:
            return left_eye['visibility'] > 0.5 and right_eye['visibility'] > 0.5
        
        return False
    
    def _calculate_visibility(self, keypoints: List[Dict]) -> float:
        """Calculate overall visibility score."""
        if not keypoints:
            return 0.0
        
        visibility_scores = [kp['visibility'] for kp in keypoints]
        return sum(visibility_scores) / len(visibility_scores)
    
    def is_loaded(self) -> bool:
        """Check if detector is loaded."""
        return self.loaded
    
    def cleanup(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()
        self.loaded = False
