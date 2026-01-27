# src/subject_detection/object_tracker.py
"""Object tracking for maintaining subject IDs across frames"""

import uuid
import numpy as np
from typing import Dict, List, Optional, Set, Tuple
import logging


class ObjectTracker:
    """
    Simple object tracker for maintaining subject IDs across frames.
    """
    
    def __init__(self, max_distance: float = 100.0, max_frames_lost: int = 10):
        """
        Initialize object tracker.
        
        Args:
            max_distance: Maximum distance for matching objects
            max_frames_lost: Maximum frames an object can be lost before removal
        """
        self.max_distance = max_distance
        self.max_frames_lost = max_frames_lost
        
        # Tracking state
        self.tracked_objects = {}
        self.next_id = 1
        self.frame_count = 0
        
        self.logger = logging.getLogger(__name__)
    
    def assign_id(self, detection: Dict, detection_type: str, frame_count: int) -> str:
        """
        Assign tracking ID to a detection.
        
        Args:
            detection: Detection dictionary with bbox
            detection_type: Type of detection ('face', 'person', 'object')
            frame_count: Current frame number
            
        Returns:
            Tracking ID string
        """
        self.frame_count = frame_count
        
        # Get detection center
        detection_center = self._get_center(detection['bbox'])
        
        # Find best match among existing tracked objects
        best_match = None
        best_distance = float('inf')
        
        for track_id, track_info in self.tracked_objects.items():
            if track_info['type'] == detection_type:
                # Calculate distance
                distance = self._calculate_distance(detection_center, track_info['last_position'])
                
                if distance < self.max_distance and distance < best_distance:
                    best_distance = distance
                    best_match = track_id
        
        if best_match:
            # Update existing track
            self.tracked_objects[best_match].update({
                'last_position': detection_center,
                'last_seen_frame': frame_count,
                'bbox': detection['bbox'],
                'confidence': detection['confidence']
            })
            return best_match
        else:
            # Create new track
            track_id = f"{detection_type}_{self.next_id}"
            self.next_id += 1
            
            self.tracked_objects[track_id] = {
                'type': detection_type,
                'first_seen_frame': frame_count,
                'last_seen_frame': frame_count,
                'last_position': detection_center,
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'track_history': [detection_center]
            }
            
            return track_id
    
    def _get_center(self, bbox: List[int]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update_frame(self, frame_count: int):
        """
        Update tracker for new frame and remove lost objects.
        
        Args:
            frame_count: Current frame number
        """
        self.frame_count = frame_count
        
        # Remove objects that have been lost for too long
        lost_tracks = []
        for track_id, track_info in self.tracked_objects.items():
            frames_lost = frame_count - track_info['last_seen_frame']
            if frames_lost > self.max_frames_lost:
                lost_tracks.append(track_id)
        
        for track_id in lost_tracks:
            del self.tracked_objects[track_id]
    
    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """
        Get information about a tracked object.
        
        Args:
            track_id: Tracking ID
            
        Returns:
            Track information dictionary or None
        """
        return self.tracked_objects.get(track_id)
    
    def get_active_tracks(self) -> List[str]:
        """
        Get list of active track IDs.
        
        Returns:
            List of active track IDs
        """
        return list(self.tracked_objects.keys())
    
    def get_track_statistics(self) -> Dict:
        """
        Get tracking statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.tracked_objects:
            return {
                'total_tracks': 0,
                'active_tracks': 0,
                'track_types': {}
            }
        
        # Count track types
        track_types = {}
        for track_info in self.tracked_objects.values():
            track_type = track_info['type']
            track_types[track_type] = track_types.get(track_type, 0) + 1
        
        return {
            'total_tracks': self.next_id - 1,
            'active_tracks': len(self.tracked_objects),
            'track_types': track_types,
            'frame_count': self.frame_count
        }
    
    def cleanup(self):
        """Clean up tracking data."""
        self.tracked_objects.clear()
        self.next_id = 1
        self.frame_count = 0
