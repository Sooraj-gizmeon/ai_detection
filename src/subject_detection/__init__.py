# src/subject_detection/__init__.py
"""Subject Detection Module - AI-powered subject detection and tracking"""

from .subject_detector import SubjectDetector
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .object_tracker import ObjectTracker

__all__ = ["SubjectDetector", "FaceDetector", "PersonDetector", "ObjectTracker"]
