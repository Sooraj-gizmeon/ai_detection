"""
Intelligent Object Detector with YOLO integration and prompt-based object selection.

This module provides advanced object detection capabilities that can understand
user prompts and identify relevant objects for video processing.
"""

import cv2
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO
import json
import asyncio
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """Represents a detected object with all its properties."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: float
    frame_timestamp: float
    relevance_score: float = 0.0
    prompt_match_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert DetectedObject to dictionary for JSON serialization."""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': list(self.bbox),
            'center': list(self.center),
            'area': self.area,
            'frame_timestamp': self.frame_timestamp,
            'relevance_score': self.relevance_score,
            'prompt_match_score': self.prompt_match_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DetectedObject':
        """Create DetectedObject from dictionary."""
        return cls(
            class_name=data['class_name'],
            confidence=data['confidence'],
            bbox=tuple(data['bbox']),
            center=tuple(data['center']),
            area=data['area'],
            frame_timestamp=data['frame_timestamp'],
            relevance_score=data.get('relevance_score', 0.0),
            prompt_match_score=data.get('prompt_match_score', 0.0)
        )


@dataclass
class ObjectTrackingResult:
    """Results of object tracking across video frames."""
    object_id: str
    class_name: str
    appearances: List[DetectedObject]
    total_frames: int
    avg_confidence: float
    max_confidence: float
    temporal_consistency: float
    prompt_relevance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ObjectTrackingResult to dictionary for JSON serialization."""
        return {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'appearances': [obj.to_dict() for obj in self.appearances],
            'total_frames': self.total_frames,
            'avg_confidence': self.avg_confidence,
            'max_confidence': self.max_confidence,
            'temporal_consistency': self.temporal_consistency,
            'prompt_relevance': self.prompt_relevance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectTrackingResult':
        """Create ObjectTrackingResult from dictionary."""
        return cls(
            object_id=data['object_id'],
            class_name=data['class_name'],
            appearances=[DetectedObject.from_dict(obj) for obj in data['appearances']],
            total_frames=data['total_frames'],
            avg_confidence=data['avg_confidence'],
            max_confidence=data['max_confidence'],
            temporal_consistency=data['temporal_consistency'],
            prompt_relevance=data['prompt_relevance']
        )


class IntelligentObjectDetector:
    """
    Advanced object detector that combines YOLO with prompt understanding
    to identify and track objects relevant to user requests.
    """
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 device: str = "auto",
                 cache_dir: str = "cache/object_detection"):
        """
        Initialize the intelligent object detector.
        
        Args:
            model_path: Path to YOLO model
            device: Device to run inference on
            cache_dir: Directory for caching results
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize YOLO model
        self.device = self._setup_device(device)
        self.model = self._load_yolo_model(model_path)
        
        # Object class mappings and priorities
        self.class_names = self.model.names
        self.sport_objects = self._initialize_sport_objects()
        self.general_objects = self._initialize_general_objects()
        
        # Prompt analysis cache
        self.prompt_analysis_cache = {}
        
        self.logger.info(f"IntelligentObjectDetector initialized with {len(self.model.names)} object classes")
    
    def _draw_detection_on_frame(self, frame, detection, prompt_analysis):
        """Draw bounding box and labels on frame for debugging."""
        import cv2
        
        x1, y1, x2, y2 = [int(coord) for coord in detection.bbox]
        class_name = detection.class_name
        confidence = detection.confidence
        relevance = detection.relevance_score
        
        # Choose color based on relevance (green for relevant, red for irrelevant, yellow for medium)
        if relevance > 0.7:
            color = (0, 255, 0)  # Green - highly relevant
        elif relevance > 0.3:
            color = (0, 255, 255)  # Yellow - medium relevance
        else:
            color = (0, 0, 255)  # Red - low relevance
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        label = f"{class_name} {confidence:.2f} (rel:{relevance:.2f})"
        
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1
        )
        
        # Add prompt info at top of frame
        if hasattr(self, '_debug_frame_counter'):
            self._debug_frame_counter += 1
        else:
            self._debug_frame_counter = 1
            
        if self._debug_frame_counter <= 1:  # Only on first frame
            prompt_text = f"Prompt: {prompt_analysis.get('original_prompt', 'N/A')}"
            sport_text = f"Sport: {prompt_analysis.get('sport_type', 'N/A')}"
            
            cv2.putText(frame, prompt_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, sport_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_yolo_model(self, model_path: str) -> YOLO:
        """Load YOLO model."""
        try:
            model = YOLO(model_path)
            self.logger.info(f"Loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to default model
            model = YOLO("yolov8n.pt")
            self.logger.info("Using default YOLOv8n model")
            return model
    
    def _initialize_sport_objects(self) -> Dict[str, List[str]]:
        """Initialize sport-specific object mappings."""
        return {
            "basketball": {
                "primary": ["sports ball", "person"],
                "targets": ["basketball hoop", "backboard"],
                "keywords": ["goal", "basket", "hoop", "shot", "dunk", "score"]
            },
            "football": {
                "primary": ["sports ball", "person"],
                "targets": ["goal post", "goalpost"],
                "keywords": ["goal", "touchdown", "field goal", "kick", "score"]
            },
            "soccer": {
                "primary": ["sports ball", "person"],
                "targets": ["goal", "goalpost", "net"],
                "keywords": ["goal", "penalty", "shot", "score", "kick"]
            },
            "tennis": {
                "primary": ["sports ball", "person", "tennis racket"],
                "targets": ["net", "court"],
                "keywords": ["serve", "rally", "ace", "winner", "shot"]
            },
            "baseball": {
                "primary": ["sports ball", "person", "baseball bat"],
                "targets": ["base", "home plate"],
                "keywords": ["hit", "home run", "base", "pitch", "catch"]
            },
            "hockey": {
                "primary": ["sports ball", "person"],
                "targets": ["goal", "net"],
                "keywords": ["goal", "puck", "shot", "save", "score"]
            }
        }
    
    def _initialize_general_objects(self) -> Dict[str, List[str]]:
        """Initialize general object categories for various scenarios."""
        return {
            "vehicles": ["car", "truck", "bus", "motorcycle", "bicycle", "train", "boat"],
            "animals": ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
            "food": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut", "cake"],
            "electronics": ["laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster"],
            "furniture": ["chair", "couch", "bed", "dining table", "toilet"],
            "tools": ["scissors", "hair drier", "toothbrush"],
            "sports_equipment": ["sports ball", "tennis racket", "baseball bat", "skateboard", "surfboard", "ski"],
            "people_activities": ["person"]
        }
    
    async def analyze_prompt_for_objects(self, 
                                       user_prompt: str, 
                                       ollama_client=None) -> Dict[str, Any]:
        """
        Analyze user prompt to determine what objects to detect and track.
        
        Args:
            user_prompt: User's prompt text
            ollama_client: Optional Ollama client for advanced analysis
            
        Returns:
            Dictionary with object detection strategy
        """
        # Check cache first
        if user_prompt in self.prompt_analysis_cache:
            return self.prompt_analysis_cache[user_prompt]
        
        analysis = {
            "sport_type": None,
            "target_objects": [],
            "priority_objects": [],
            "keywords": [],
            "detection_strategy": "general",
            "focus_areas": [],
            "temporal_requirements": {}
        }
        
        # Basic keyword analysis
        prompt_lower = user_prompt.lower()
        
        # Detect sport type
        for sport, config in self.sport_objects.items():
            if any(keyword in prompt_lower for keyword in config["keywords"]):
                analysis["sport_type"] = sport
                analysis["target_objects"].extend(config["targets"])
                analysis["priority_objects"].extend(config["primary"])
                analysis["keywords"].extend(config["keywords"])
                analysis["detection_strategy"] = "sport_specific"
                break
        
        # Detect general object categories
        for category, objects in self.general_objects.items():
            if any(obj_name in prompt_lower for obj_name in objects):
                analysis["target_objects"].extend(objects)
                analysis["detection_strategy"] = "category_specific"
        
        # Advanced LLM analysis if available
        if ollama_client:
            try:
                llm_analysis = await self._analyze_prompt_with_llm(user_prompt, ollama_client)
                analysis.update(llm_analysis)
            except Exception as e:
                self.logger.warning(f"LLM analysis failed: {e}")
        
        # Determine focus areas based on prompt
        analysis["focus_areas"] = self._determine_focus_areas(prompt_lower)
        
        # Cache the result
        self.prompt_analysis_cache[user_prompt] = analysis
        
        return analysis
    
    async def _analyze_prompt_with_llm(self, 
                                     user_prompt: str, 
                                     ollama_client) -> Dict[str, Any]:
        """Use LLM to analyze prompt for object detection strategy."""
        
        prompt_template = f"""
        Analyze this user prompt for video processing: "{user_prompt}"
        
        Determine:
        1. What specific objects should be detected and tracked?
        2. What sport or activity is being referenced?
        3. What are the key moments to focus on?
        4. What framing strategy would work best?
        5. Are there temporal requirements (e.g., "goals" happen at specific moments)?
        
        Respond in JSON format:
        {{
            "detected_activity": "sport/activity name or general",
            "key_objects": ["list", "of", "objects"],
            "priority_objects": ["most", "important", "objects"],
            "temporal_events": ["goal", "score", "action"],
            "framing_strategy": "close_up/wide_shot/tracking",
            "detection_confidence_threshold": 0.7,
            "focus_requirements": "description of what to focus on"
        }}
        """
        
        response = await ollama_client._make_request(
            prompt=prompt_template,
            model=ollama_client.get_best_model("analysis"),
            cache_key=f"object_prompt_analysis_{hash(user_prompt)}"
        )
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                llm_result = json.loads(json_match.group())
                return {
                    "llm_analysis": llm_result,
                    "detected_activity": llm_result.get("detected_activity"),
                    "framing_strategy": llm_result.get("framing_strategy", "general"),
                    "confidence_threshold": llm_result.get("detection_confidence_threshold", 0.5),
                    "temporal_events": llm_result.get("temporal_events", [])
                }
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
        
        return {}
    
    def _determine_focus_areas(self, prompt_lower: str) -> List[str]:
        """Determine what areas of the frame to focus on based on prompt."""
        focus_areas = []
        
        # Movement and action focus
        if any(word in prompt_lower for word in ["goal", "score", "shot", "kick", "throw"]):
            focus_areas.append("action_zone")
        
        # Person-centric focus
        if any(word in prompt_lower for word in ["player", "person", "character", "people"]):
            focus_areas.append("person_tracking")
        
        # Object-centric focus
        if any(word in prompt_lower for word in ["ball", "object", "item", "equipment"]):
            focus_areas.append("object_tracking")
        
        # Environmental focus
        if any(word in prompt_lower for word in ["background", "scenery", "environment", "setting"]):
            focus_areas.append("environmental")
        
        return focus_areas
    
    async def detect_objects_in_video(self, 
                                    video_path: str,
                                    prompt_analysis: Dict[str, Any],
                                    sample_rate: float = 1.0,
                                    max_frames: int = 300,
                                    save_debug_frames: bool = True) -> Dict[str, Any]:
        """
        Detect objects in video based on prompt analysis with visual debugging.
        
        Args:
            video_path: Path to video file
            prompt_analysis: Analysis of user prompt
            sample_rate: Frames per second to sample
            max_frames: Maximum frames to process
            save_debug_frames: Whether to save frames with bounding boxes
            
        Returns:
            Object detection results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Determine sampling strategy
            frame_skip = max(1, int(fps / sample_rate))
            frames_to_process = min(max_frames, total_frames // frame_skip)
            
            detected_objects = []
            objects_by_frame = {}
            object_tracks = {}
            frame_count = 0
            processed_frames = 0
            
            # Create debug output directory
            debug_dir = None
            if save_debug_frames:
                from pathlib import Path
                video_name = Path(video_path).stem
                debug_dir = Path("cache/object_detection_debug") / video_name
                debug_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Debug frames will be saved to: {debug_dir}")
                
                # Reset debug counter
                self._debug_frame_counter = 0
            
            self.logger.info(f"Processing {frames_to_process} frames from {video_path}")
            self.logger.info(f"Prompt analysis: {prompt_analysis}")
            
            while cap.isOpened() and processed_frames < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on sampling rate
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                timestamp = frame_count / fps
                
                # Run YOLO detection
                results = self.model(frame, verbose=False)
                
                # Create annotated frame for debugging
                annotated_frame = frame.copy() if save_debug_frames else None
                
                # Process detections
                frame_objects = self._process_frame_detections(
                    results[0], 
                    frame, 
                    timestamp, 
                    prompt_analysis
                )
                
                # Draw debug annotations
                if save_debug_frames and frame_objects and annotated_frame is not None:
                    for detection in frame_objects:
                        self._draw_detection_on_frame(annotated_frame, detection, prompt_analysis)
                    
                    # Save debug frame
                    if debug_dir:
                        debug_filename = f"frame_{frame_count:06d}_t{timestamp:.2f}s_det{len(frame_objects)}.jpg"
                        debug_path = debug_dir / debug_filename
                        cv2.imwrite(str(debug_path), annotated_frame)
                
                detected_objects.extend(frame_objects)
                objects_by_frame[frame_count] = frame_objects
                
                # Update object tracking
                self._update_object_tracking(frame_objects, object_tracks, timestamp)
                
                processed_frames += 1
                frame_count += 1
                
                # Log progress
                if processed_frames % 50 == 0:
                    self.logger.info(f"Processed {processed_frames}/{frames_to_process} frames")
            
            cap.release()
            
            # Analyze tracking results
            tracking_results = self._analyze_tracking_results(object_tracks, prompt_analysis)
            
            # Generate detection summary
            summary = self._generate_detection_summary(
                detected_objects, 
                tracking_results, 
                prompt_analysis,
                duration
            )
            
            return {
                "status": "success",
                "video_duration": duration,
                "frames_processed": processed_frames,
                "total_detections": len(detected_objects),
                "detected_objects": [obj.to_dict() for obj in detected_objects],
                "object_tracks": tracking_results,
                "summary": summary,
                "prompt_analysis": prompt_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return {
                "status": "error",
                "error": str(e),
                "detected_objects": [],
                "object_tracks": {},
                "summary": {}
            }
    
    def _process_frame_detections(self, 
                                results, 
                                frame: np.ndarray,
                                timestamp: float,
                                prompt_analysis: Dict[str, Any]) -> List[DetectedObject]:
        """Process YOLO detections for a single frame."""
        frame_objects = []
        
        if results.boxes is None:
            return frame_objects
        
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        target_objects = prompt_analysis.get("target_objects", [])
        priority_objects = prompt_analysis.get("priority_objects", [])
        confidence_threshold = prompt_analysis.get("confidence_threshold", 0.5)
        
        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
            if conf < confidence_threshold:
                continue
            
            class_name = self.class_names[class_id]
            x1, y1, x2, y2 = box.astype(int)
            
            # Calculate object properties
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            # Calculate relevance score based on prompt
            relevance_score = self._calculate_object_relevance(
                class_name, 
                prompt_analysis
            )
            
            # Calculate prompt match score
            prompt_match_score = self._calculate_prompt_match_score(
                class_name, 
                prompt_analysis
            )
            
            detected_obj = DetectedObject(
                class_name=class_name,
                confidence=float(conf),
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                area=area,
                frame_timestamp=timestamp,
                relevance_score=relevance_score,
                prompt_match_score=prompt_match_score
            )
            
            # Add compatibility fields for debugging (using setattr to avoid keyword issues)
            setattr(detected_obj, 'class', class_name)  # For debugging compatibility
            setattr(detected_obj, 'relevance', relevance_score)  # For debugging compatibility
            
            frame_objects.append(detected_obj)
        
        return frame_objects
    
    def _calculate_object_relevance(self, 
                                  class_name: str, 
                                  prompt_analysis: Dict[str, Any]) -> float:
        """Calculate how relevant an object is to the prompt."""
        relevance = 0.0
        
        # Check if it's a target object
        if class_name in prompt_analysis.get("target_objects", []):
            relevance += 0.8
        
        # Check if it's a priority object
        if class_name in prompt_analysis.get("priority_objects", []):
            relevance += 0.6
        
        # Check sport-specific relevance
        sport_type = prompt_analysis.get("sport_type")
        if sport_type and sport_type in self.sport_objects:
            sport_config = self.sport_objects[sport_type]
            if class_name in sport_config.get("primary", []):
                relevance += 0.7
            if class_name in sport_config.get("targets", []):
                relevance += 0.9
        
        # General object relevance
        for category, objects in self.general_objects.items():
            if class_name in objects:
                relevance += 0.3
                break
        
        return min(relevance, 1.0)
    
    def _calculate_prompt_match_score(self, 
                                    class_name: str, 
                                    prompt_analysis: Dict[str, Any]) -> float:
        """Calculate how well object matches the specific prompt."""
        score = 0.0
        
        # Direct keyword matching
        keywords = prompt_analysis.get("keywords", [])
        for keyword in keywords:
            if keyword in class_name or class_name in keyword:
                score += 0.5
        
        # LLM analysis matching
        llm_analysis = prompt_analysis.get("llm_analysis", {})
        if llm_analysis:
            key_objects = llm_analysis.get("key_objects", [])
            if class_name in key_objects:
                score += 0.8
            
            priority_objects = llm_analysis.get("priority_objects", [])
            if class_name in priority_objects:
                score += 0.9
        
        return min(score, 1.0)
    
    def _update_object_tracking(self, 
                              frame_objects: List[DetectedObject],
                              object_tracks: Dict[str, List[DetectedObject]],
                              timestamp: float):
        """Update object tracking across frames."""
        for obj in frame_objects:
            # Create tracking key based on class and rough position
            track_key = f"{obj.class_name}_{obj.center[0]//100}_{obj.center[1]//100}"
            
            if track_key not in object_tracks:
                object_tracks[track_key] = []
            
            object_tracks[track_key].append(obj)
    
    def _analyze_tracking_results(self, 
                                object_tracks: Dict[str, List[DetectedObject]],
                                prompt_analysis: Dict[str, Any]) -> List[ObjectTrackingResult]:
        """Analyze object tracking results."""
        tracking_results = []
        
        for track_id, detections in object_tracks.items():
            if len(detections) < 2:  # Skip single detections
                continue
            
            class_name = detections[0].class_name
            total_frames = len(detections)
            
            # Calculate metrics
            confidences = [obj.confidence for obj in detections]
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
            
            # Calculate temporal consistency
            timestamps = [obj.frame_timestamp for obj in detections]
            time_span = max(timestamps) - min(timestamps)
            temporal_consistency = min(1.0, total_frames / max(1, time_span))
            
            # Calculate prompt relevance
            relevance_scores = [obj.relevance_score for obj in detections]
            prompt_relevance = np.mean(relevance_scores)
            
            tracking_result = ObjectTrackingResult(
                object_id=track_id,
                class_name=class_name,
                appearances=detections,
                total_frames=total_frames,
                avg_confidence=avg_confidence,
                max_confidence=max_confidence,
                temporal_consistency=temporal_consistency,
                prompt_relevance=prompt_relevance
            )
            
            tracking_results.append(tracking_result)
        
        # Sort by relevance and consistency
        tracking_results.sort(
            key=lambda x: (x.prompt_relevance, x.temporal_consistency, x.avg_confidence),
            reverse=True
        )
        
        return tracking_results
    
    def _generate_detection_summary(self, 
                                  detected_objects: List[DetectedObject],
                                  tracking_results: List[ObjectTrackingResult],
                                  prompt_analysis: Dict[str, Any],
                                  video_duration: float) -> Dict[str, Any]:
        """Generate a summary of detection results."""
        
        # Count object classes
        class_counts = {}
        for obj in detected_objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1
        
        # Find most relevant objects
        relevant_objects = [obj for obj in detected_objects if obj.relevance_score > 0.5]
        
        # Temporal analysis
        if detected_objects:
            timestamps = [obj.frame_timestamp for obj in detected_objects]
            detection_density = len(detected_objects) / video_duration
        else:
            detection_density = 0
        
        # Top tracking results
        top_tracks = tracking_results[:5] if tracking_results else []
        
        return {
            "total_objects_detected": len(detected_objects),
            "unique_classes": len(class_counts),
            "class_distribution": class_counts,
            "relevant_objects_count": len(relevant_objects),
            "detection_density": detection_density,
            "top_tracked_objects": [
                {
                    "class": track.class_name,
                    "frames": track.total_frames,
                    "relevance": track.prompt_relevance,
                    "confidence": track.avg_confidence
                }
                for track in top_tracks
            ],
            "prompt_match_quality": np.mean([obj.prompt_match_score for obj in relevant_objects]) if relevant_objects else 0.0
        }
    
    def get_optimal_crop_regions(self, 
                               tracking_results: List[ObjectTrackingResult],
                               frame_size: Tuple[int, int],
                               target_aspect_ratio: Tuple[int, int] = (9, 16)) -> List[Dict[str, Any]]:
        """
        Calculate optimal crop regions based on object tracking results.
        
        Args:
            tracking_results: Object tracking results
            frame_size: Original frame size (width, height)
            target_aspect_ratio: Target aspect ratio for cropping
            
        Returns:
            List of crop regions with timestamps
        """
        crop_regions = []
        frame_width, frame_height = frame_size
        target_width_ratio = target_aspect_ratio[0] / target_aspect_ratio[1]
        
        # Calculate target crop dimensions
        if frame_width / frame_height > target_width_ratio:
            # Frame is wider than target, crop horizontally
            crop_height = frame_height
            crop_width = int(crop_height * target_width_ratio)
        else:
            # Frame is taller than target, crop vertically
            crop_width = frame_width
            crop_height = int(crop_width / target_width_ratio)
        
        # Group detections by timestamp
        timestamp_groups = {}
        for track in tracking_results:
            for obj in track.appearances:
                timestamp = obj.frame_timestamp
                if timestamp not in timestamp_groups:
                    timestamp_groups[timestamp] = []
                timestamp_groups[timestamp].append(obj)
        
        # Calculate crop region for each timestamp
        for timestamp, objects in timestamp_groups.items():
            if not objects:
                continue
            
            # Find center of mass of all relevant objects
            weighted_x = 0
            weighted_y = 0
            total_weight = 0
            
            for obj in objects:
                weight = obj.relevance_score * obj.confidence
                center_x, center_y = obj.center
                weighted_x += center_x * weight
                weighted_y += center_y * weight
                total_weight += weight
            
            if total_weight > 0:
                center_x = int(weighted_x / total_weight)
                center_y = int(weighted_y / total_weight)
            else:
                center_x = frame_width // 2
                center_y = frame_height // 2
            
            # Calculate crop region centered on objects
            crop_x1 = max(0, center_x - crop_width // 2)
            crop_y1 = max(0, center_y - crop_height // 2)
            crop_x2 = min(frame_width, crop_x1 + crop_width)
            crop_y2 = min(frame_height, crop_y1 + crop_height)
            
            # Adjust if crop region goes out of bounds
            if crop_x2 > frame_width:
                crop_x1 = frame_width - crop_width
                crop_x2 = frame_width
            if crop_y2 > frame_height:
                crop_y1 = frame_height - crop_height
                crop_y2 = frame_height
            
            crop_region = {
                "timestamp": timestamp,
                "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
                "center": (center_x, center_y),
                "objects_count": len(objects),
                "total_relevance": sum(obj.relevance_score for obj in objects),
                "confidence": sum(obj.confidence for obj in objects) / len(objects)
            }
            
            crop_regions.append(crop_region)
        
        # Sort by timestamp
        crop_regions.sort(key=lambda x: x["timestamp"])
        
        return crop_regions
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clear caches
            self.prompt_analysis_cache.clear()
            
            # If using GPU, clear cache
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("IntelligentObjectDetector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
