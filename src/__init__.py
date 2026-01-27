# src/__init__.py
"""Video-to-Shorts Pipeline Package"""

__version__ = "1.0.0"
__author__ = "ARJUN-GIZ"
__description__ = "AI-powered video processing pipeline for creating short-form content"

from .audio_analysis import WhisperAnalyzer
from .smart_zoom import SmartZoomProcessor
from .subject_detection import SubjectDetector
from .scene_detection import SceneDetector
from .content_analysis import ContentAnalyzer
from .video_processing import VideoProcessor
from .ai_integration import OllamaClient

__all__ = [
    "WhisperAnalyzer",
    "SmartZoomProcessor", 
    "SubjectDetector",
    "SceneDetector",
    "ContentAnalyzer",
    "VideoProcessor",
    "OllamaClient",
]
