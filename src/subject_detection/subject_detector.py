# src/subject_detection/subject_detector.py
"""Main subject detection coordinator using YOLO, MediaPipe, and custom detectors"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
try:
    import mediapipe as mp
except Exception:
    mp = None
import threading
from ultralytics import YOLO
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .object_tracker import ObjectTracker
from config.smart_zoom_settings import MODEL_CONFIG


class SubjectDetector:
    """
    Main subject detection coordinator that combines multiple detection methods
    for comprehensive subject identification and tracking.
    """
    
    def __init__(self, device: str = "cuda", model_dir: str = "models"):
        """
        Initialize Subject Detector with multiple detection models.
        
        Args:
            device: Device to use for processing ('cuda' or 'cpu')
            model_dir: Directory containing model files
        """
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Detection configuration
        self.config = MODEL_CONFIG
        # Ensure device preference if CUDA is available
        if torch.cuda.is_available():
            self.device = "cuda"
            self.config['yolo']['device'] = 'cuda'
        
        # Thread safety for YOLO
        self._yolo_lock = threading.Lock()
        
        # Initialize components
        self.face_detector = FaceDetector(device=self.device)
        self.person_detector = PersonDetector(device=self.device)
        self.object_tracker = ObjectTracker()
        
        # Load YOLO model
        self.yolo_model = None
        if self.config['yolo'].get('enabled', True):
            self._load_yolo_model()
        else:
            self.logger.info("YOLO object detection disabled in configuration")
        
        # Tracking state
        self.tracking_history = {}
        self.frame_count = 0
        
        self.logger.info(f"SubjectDetector initialized on {self.device}")
    
    def _load_yolo_model(self):
        """Load YOLO model for object detection."""
        try:
            model_path = self.config['yolo']['model_path']
            
            self.logger.info(f"Loading YOLO model: {model_path}")
            
            # Try multiple loading approaches
            loading_attempts = [
                # Method 1: Direct loading
                lambda: YOLO(model_path),
                # Method 2: Explicit device specification
                lambda: YOLO(model_path).to(self.device) if self.device == 'cuda' else YOLO(model_path),
                # Method 3: CPU fallback
                lambda: YOLO(model_path).to('cpu') if self.device == 'cuda' else YOLO(model_path)
            ]
            
            for i, attempt in enumerate(loading_attempts):
                try:
                    self.logger.debug(f"YOLO loading attempt {i+1}")
                    self.yolo_model = attempt()
                    
                    # Test the model with a dummy frame
                    import numpy as np
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    test_results = self.yolo_model.predict(dummy_frame, verbose=False, save=False)
                    
                    self.logger.info("YOLO model loaded and tested successfully")
                    return
                    
                except Exception as e:
                    self.logger.warning(f"YOLO loading attempt {i+1} failed: {e}")
                    continue
            
            # If all attempts failed
            self.logger.error("All YOLO loading attempts failed, disabling object detection")
            self.yolo_model = None
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            self.yolo_model = None
    
    def detect_subjects(self, frame: np.ndarray, timestamp: float = None) -> List[Dict]:
        """
        Detect all subjects in a frame using multiple detection methods.
        
        Args:
            frame: Input frame
            timestamp: Optional timestamp for tracking
            
        Returns:
            List of detected subjects with metadata
        """
        self.frame_count += 1
        detected_subjects = []
        
        try:
            # 1. Detect faces
            faces = self.face_detector.detect_faces(frame)
            for face in faces:
                subject = {
                    'type': 'face',
                    'bbox': face['bbox'],
                    'confidence': face['confidence'],
                    'landmarks': face.get('landmarks', []),
                    'attributes': face.get('attributes', {}),
                    'priority': 1.0,  # High priority for faces
                    'tracking_id': self._assign_tracking_id(face, 'face')
                }
                detected_subjects.append(subject)
            
            # 2. Detect people/persons
            people = self.person_detector.detect_people(frame)
            for person in people:
                subject = {
                    'type': 'person',
                    'bbox': person['bbox'],
                    'confidence': person['confidence'],
                    'keypoints': person.get('keypoints', []),
                    'pose': person.get('pose', {}),
                    'priority': 0.8,  # High priority for people
                    'tracking_id': self._assign_tracking_id(person, 'person')
                }
                detected_subjects.append(subject)
            
            # 3. Detect objects with YOLO
            objects = self._detect_objects_yolo(frame)
            for obj in objects:
                subject = {
                    'type': 'object',
                    'class': obj['class'],
                    'bbox': obj['bbox'],
                    'confidence': obj['confidence'],
                    'priority': self._calculate_object_priority(obj['class']),
                    'tracking_id': self._assign_tracking_id(obj, 'object')
                }
                detected_subjects.append(subject)
            
            # 4. Filter and prioritize subjects
            filtered_subjects = self._filter_and_prioritize(detected_subjects)
            
            # 5. Update tracking
            if timestamp is not None:
                self._update_tracking(filtered_subjects, timestamp)
            
            return filtered_subjects
            
        except Exception as e:
            self.logger.error(f"Subject detection failed: {e}")
            return []
    
    def _detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects using YOLO model.
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected objects
        """
        if self.yolo_model is None:
            return []
        
        # Use thread lock to prevent concurrent YOLO calls
        with self._yolo_lock:
            try:
                # Run YOLO inference with explicit parameters to avoid issues
                results = self.yolo_model.predict(
                    frame,
                    conf=self.config['yolo']['conf_threshold'],
                    iou=self.config['yolo'].get('iou_threshold', 0.45),
                    verbose=False,
                    save=False,
                    save_txt=False,
                    save_conf=False,
                    save_crop=False,
                    show=False,
                    stream=False
                )
                
                objects = []
                for result in results:
                    # Guard: no boxes
                    if not hasattr(result, 'boxes') or result.boxes is None:
                        continue
                        
                    boxes = result.boxes
                    if len(boxes) == 0:
                        continue
                    
                    # Extract data safely
                    try:
                        xyxy_tensor = boxes.xyxy
                        conf_tensor = boxes.conf
                        cls_tensor = boxes.cls
                        
                        if xyxy_tensor is None or conf_tensor is None or cls_tensor is None:
                            continue
                        
                        # Convert to CPU numpy arrays
                        xyxy_np = xyxy_tensor.detach().cpu().numpy() if hasattr(xyxy_tensor, 'detach') else xyxy_tensor.cpu().numpy()
                        conf_np = conf_tensor.detach().cpu().numpy() if hasattr(conf_tensor, 'detach') else conf_tensor.cpu().numpy()
                        cls_np = cls_tensor.detach().cpu().numpy() if hasattr(cls_tensor, 'detach') else cls_tensor.cpu().numpy()
                        
                        for i in range(len(xyxy_np)):
                            x1, y1, x2, y2 = xyxy_np[i]
                            confidence = float(conf_np[i])
                            class_id = int(cls_np[i])
                            
                            # Double-check confidence threshold
                            if confidence < self.config['yolo']['conf_threshold']:
                                continue
                            
                            # Get class name safely
                            if class_id < len(self.yolo_model.names):
                                class_name = self.yolo_model.names[class_id]
                            else:
                                self.logger.warning(f"Invalid class_id {class_id}, skipping")
                                continue
                                
                            obj = {
                                'class': class_name,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id
                            }
                            objects.append(obj)
                            
                    except Exception as inner_e:
                        self.logger.warning(f"Error processing YOLO result: {inner_e}")
                        continue
                
                return objects
                
            except Exception as e:
                # More detailed error logging
                import traceback
                error_detail = traceback.format_exc()
                self.logger.error(f"YOLO detection failed: {type(e).__name__}: {e}")
                self.logger.debug(f"Full traceback: {error_detail}")
                
                # If YOLO keeps failing, disable it temporarily
                if not hasattr(self, '_yolo_error_count'):
                    self._yolo_error_count = 0
                self._yolo_error_count += 1
                
                if self._yolo_error_count > 10:
                    self.logger.warning("YOLO failed too many times, disabling object detection")
                    self.yolo_model = None
                
                return []
    
    def _calculate_object_priority(self, class_name: str) -> float:
        """
        Calculate priority score for different object classes.
        
        Args:
            class_name: Name of the object class
            
        Returns:
            Priority score (0.0 to 1.0)
        """
        # Define priority mapping for different object classes
        priority_map = {
            'person': 1.0,
            'face': 1.0,
            'car': 0.6,
            'bicycle': 0.5,
            'motorcycle': 0.5,
            'airplane': 0.7,
            'bus': 0.6,
            'train': 0.6,
            'truck': 0.6,
            'boat': 0.5,
            'dog': 0.7,
            'cat': 0.7,
            'bird': 0.6,
            'horse': 0.7,
            'elephant': 0.8,
            'giraffe': 0.8,
            'sports ball': 0.4,
            'book': 0.3,
            'laptop': 0.5,
            'cell phone': 0.4,
            'tv': 0.5,
        }
        
        return priority_map.get(class_name.lower(), 0.3)
    
    def _filter_and_prioritize(self, subjects: List[Dict]) -> List[Dict]:
        """
        Filter and prioritize detected subjects.
        
        Args:
            subjects: List of detected subjects
            
        Returns:
            Filtered and prioritized subjects
        """
        # Remove duplicates (same subject detected by multiple methods)
        filtered_subjects = self._remove_duplicate_subjects(subjects)
        
        # Sort by priority and confidence
        filtered_subjects.sort(
            key=lambda x: (x['priority'], x['confidence']), 
            reverse=True
        )
        
        # Apply confidence thresholds
        final_subjects = []
        for subject in filtered_subjects:
            threshold = self._get_confidence_threshold(subject['type'])
            if subject['confidence'] >= threshold:
                final_subjects.append(subject)
        
        return final_subjects
    
    def _remove_duplicate_subjects(self, subjects: List[Dict]) -> List[Dict]:
        """
        Remove duplicate subjects detected by multiple methods.
        
        Args:
            subjects: List of subjects with potential duplicates
            
        Returns:
            List of unique subjects
        """
        unique_subjects = []
        
        for subject in subjects:
            is_duplicate = False
            
            for existing in unique_subjects:
                if self._are_subjects_duplicate(subject, existing):
                    # Keep the one with higher confidence
                    if subject['confidence'] > existing['confidence']:
                        unique_subjects.remove(existing)
                        unique_subjects.append(subject)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_subjects.append(subject)
        
        return unique_subjects
    
    def _are_subjects_duplicate(self, subject1: Dict, subject2: Dict) -> bool:
        """
        Check if two subjects are duplicates based on overlap.
        
        Args:
            subject1: First subject
            subject2: Second subject
            
        Returns:
            True if subjects are duplicates
        """
        # Calculate IoU (Intersection over Union)
        iou = self._calculate_iou(subject1['bbox'], subject2['bbox'])
        
        # Consider duplicates if IoU > 0.5
        return iou > 0.5
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_confidence_threshold(self, subject_type: str) -> float:
        """
        Get confidence threshold for different subject types.
        
        Args:
            subject_type: Type of subject
            
        Returns:
            Confidence threshold
        """
        thresholds = {
            'face': 0.55,   # lower to reduce misses
            'person': 0.6,  # lower to reduce misses
            'object': 0.5
        }
        
        return thresholds.get(subject_type, 0.5)
    
    def _assign_tracking_id(self, subject: Dict, subject_type: str) -> str:
        """
        Assign tracking ID to a subject for temporal consistency.
        
        Args:
            subject: Subject dictionary
            subject_type: Type of subject
            
        Returns:
            Tracking ID
        """
        # Use object tracker for ID assignment
        return self.object_tracker.assign_id(subject, subject_type, self.frame_count)
    
    def _update_tracking(self, subjects: List[Dict], timestamp: float):
        """
        Update tracking information for subjects.
        
        Args:
            subjects: List of detected subjects
            timestamp: Current timestamp
        """
        for subject in subjects:
            tracking_id = subject['tracking_id']
            
            if tracking_id not in self.tracking_history:
                self.tracking_history[tracking_id] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'positions': [],
                    'confidence_history': [],
                    'subject_type': subject['type']
                }
            
            # Update tracking history
            self.tracking_history[tracking_id]['last_seen'] = timestamp
            self.tracking_history[tracking_id]['positions'].append(subject['bbox'])
            self.tracking_history[tracking_id]['confidence_history'].append(subject['confidence'])
            
            # Keep only recent history (last 30 frames)
            if len(self.tracking_history[tracking_id]['positions']) > 30:
                self.tracking_history[tracking_id]['positions'].pop(0)
                self.tracking_history[tracking_id]['confidence_history'].pop(0)
    
    def get_primary_subjects(self, subjects: List[Dict], max_subjects: int = 3) -> List[Dict]:
        """
        Get primary subjects for framing decisions.
        
        Args:
            subjects: List of detected subjects
            max_subjects: Maximum number of subjects to return
            
        Returns:
            List of primary subjects
        """
        if not subjects:
            return []
        
        # Sort by priority and confidence
        sorted_subjects = sorted(
            subjects,
            key=lambda x: (x['priority'], x['confidence'], self._calculate_subject_area(x)),
            reverse=True
        )
        
        return sorted_subjects[:max_subjects]
    
    def _calculate_subject_area(self, subject: Dict) -> float:
        """
        Calculate the area of a subject's bounding box.
        
        Args:
            subject: Subject dictionary
            
        Returns:
            Area of the bounding box
        """
        bbox = subject['bbox']
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def calculate_subject_center(self, subject: Dict) -> Tuple[int, int]:
        """
        Calculate the center point of a subject.
        
        Args:
            subject: Subject dictionary
            
        Returns:
            Center point (x, y)
        """
        bbox = subject['bbox']
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        return (center_x, center_y)
    
    def get_subjects_bounding_box(self, subjects: List[Dict]) -> Optional[List[int]]:
        """
        Get bounding box that encompasses all subjects.
        
        Args:
            subjects: List of subjects
            
        Returns:
            Combined bounding box [x1, y1, x2, y2] or None
        """
        if not subjects:
            return None
        
        # Find min/max coordinates
        min_x = min(subject['bbox'][0] for subject in subjects)
        min_y = min(subject['bbox'][1] for subject in subjects)
        max_x = max(subject['bbox'][2] for subject in subjects)
        max_y = max(subject['bbox'][3] for subject in subjects)
        
        return [min_x, min_y, max_x, max_y]
    
    def predict_subject_movement(self, tracking_id: str, frames_ahead: int = 5) -> Optional[Tuple[int, int]]:
        """
        Predict future position of a subject based on tracking history.
        
        Args:
            tracking_id: ID of the subject to predict
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted center position (x, y) or None
        """
        if tracking_id not in self.tracking_history:
            return None
        
        positions = self.tracking_history[tracking_id]['positions']
        if len(positions) < 2:
            return None
        
        # Calculate recent movement vector
        recent_positions = positions[-5:]  # Use last 5 positions
        
        # Simple linear extrapolation
        if len(recent_positions) >= 2:
            # Calculate average velocity
            velocities = []
            for i in range(1, len(recent_positions)):
                prev_center = self._bbox_center(recent_positions[i-1])
                curr_center = self._bbox_center(recent_positions[i])
                
                velocity = (
                    curr_center[0] - prev_center[0],
                    curr_center[1] - prev_center[1]
                )
                velocities.append(velocity)
            
            # Average velocity
            avg_velocity = (
                sum(v[0] for v in velocities) / len(velocities),
                sum(v[1] for v in velocities) / len(velocities)
            )
            
            # Predict future position
            current_center = self._bbox_center(positions[-1])
            predicted_center = (
                int(current_center[0] + avg_velocity[0] * frames_ahead),
                int(current_center[1] + avg_velocity[1] * frames_ahead)
            )
            
            return predicted_center
        
        return None
    
    def _bbox_center(self, bbox: List[int]) -> Tuple[int, int]:
        """Calculate center of bounding box."""
        return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    
    def get_detection_stats(self) -> Dict:
        """
        Get detection statistics and performance metrics.
        
        Returns:
            Dictionary with detection statistics
        """
        total_subjects = len(self.tracking_history)
        active_subjects = sum(
            1 for track in self.tracking_history.values()
            if self.frame_count - track.get('last_frame', 0) < 30
        )
        
        return {
            'total_subjects_detected': total_subjects,
            'active_subjects': active_subjects,
            'frames_processed': self.frame_count,
            'tracking_history_size': len(self.tracking_history),
            'device': self.device,
            'models_loaded': {
                'yolo': self.yolo_model is not None,
                'face_detector': self.face_detector.is_loaded(),
                'person_detector': self.person_detector.is_loaded()
            }
        }
    
    def cleanup(self):
        """Clean up resources and temporary files."""
        try:
            # Clear tracking history
            self.tracking_history.clear()
            
            # Cleanup individual detectors
            self.face_detector.cleanup()
            self.person_detector.cleanup()
            self.object_tracker.cleanup()
            
            self.logger.info("SubjectDetector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def visualize_detections(self, frame: np.ndarray, subjects: List[Dict]) -> np.ndarray:
        """
        Visualize detected subjects on frame for debugging.
        
        Args:
            frame: Input frame
            subjects: List of detected subjects
            
        Returns:
            Frame with visualized detections
        """
        vis_frame = frame.copy()
        
        for subject in subjects:
            bbox = subject['bbox']
            subject_type = subject['type']
            confidence = subject['confidence']
            
            # Choose color based on subject type
            color_map = {
                'face': (0, 255, 0),      # Green
                'person': (255, 0, 0),    # Blue
                'object': (0, 0, 255)     # Red
            }
            color = color_map.get(subject_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label
            label = f"{subject_type}: {confidence:.2f}"
            cv2.putText(vis_frame, label, (bbox[0], bbox[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
