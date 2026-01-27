"""
AI-Powered Reframing Processor for intelligent video cropping and framing.

This module provides advanced reframing capabilities that combine object detection
results with user prompts to create optimal video crops and frames.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import asyncio
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


@dataclass
class ReframingParameters:
    """Parameters for AI-powered reframing."""
    target_aspect_ratio: Tuple[int, int] = (9, 16)
    enable_object_tracking: bool = True
    enable_smooth_transitions: bool = True
    smoothing_window: int = 30
    crop_padding: float = 0.1
    min_object_size: float = 0.02
    max_crop_movement: float = 0.3
    focus_strength: float = 0.8
    temporal_consistency_weight: float = 0.4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ReframingParameters to dictionary for JSON serialization."""
        return {
            'target_aspect_ratio': list(self.target_aspect_ratio),
            'enable_object_tracking': self.enable_object_tracking,
            'enable_smooth_transitions': self.enable_smooth_transitions,
            'smoothing_window': self.smoothing_window,
            'crop_padding': self.crop_padding,
            'min_object_size': self.min_object_size,
            'max_crop_movement': self.max_crop_movement,
            'focus_strength': self.focus_strength,
            'temporal_consistency_weight': self.temporal_consistency_weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReframingParameters':
        """Create ReframingParameters from dictionary."""
        return cls(
            target_aspect_ratio=tuple(data['target_aspect_ratio']),
            enable_object_tracking=data['enable_object_tracking'],
            enable_smooth_transitions=data['enable_smooth_transitions'],
            smoothing_window=data['smoothing_window'],
            crop_padding=data['crop_padding'],
            min_object_size=data['min_object_size'],
            max_crop_movement=data['max_crop_movement'],
            focus_strength=data['focus_strength'],
            temporal_consistency_weight=data['temporal_consistency_weight']
        )


@dataclass
class FramingDecision:
    """Represents a framing decision for a specific timestamp."""
    timestamp: float
    crop_region: Tuple[int, int, int, int]  # x1, y1, x2, y2
    focus_objects: List[str]
    confidence: float
    reasoning: str
    zoom_level: float
    pan_speed: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FramingDecision to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'crop_region': list(self.crop_region),
            'focus_objects': self.focus_objects,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'zoom_level': self.zoom_level,
            'pan_speed': self.pan_speed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FramingDecision':
        """Create FramingDecision from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            crop_region=tuple(data['crop_region']),
            focus_objects=data['focus_objects'],
            confidence=data['confidence'],
            reasoning=data['reasoning'],
            zoom_level=data['zoom_level'],
            pan_speed=data['pan_speed']
        )


class AIReframingProcessor:
    """
    Advanced reframing processor that uses AI to make intelligent cropping decisions
    based on object detection and user prompts.
    """
    
    def __init__(self, 
                 cache_dir: str = "cache/reframing",
                 default_params: Optional[ReframingParameters] = None):
        """
        Initialize the AI reframing processor.
        
        Args:
            cache_dir: Directory for caching results
            default_params: Default reframing parameters
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.params = default_params or ReframingParameters()
        
        # Reframing strategy cache
        self.strategy_cache = {}
        
        self.logger.info("AIReframingProcessor initialized")
    
    async def analyze_reframing_strategy(self, 
                                       user_prompt: str,
                                       object_detection_results: Dict[str, Any],
                                       video_info: Dict[str, Any],
                                       ollama_client=None) -> Dict[str, Any]:
        """
        Analyze the best reframing strategy based on prompt and object detection.
        
        Args:
            user_prompt: User's prompt
            object_detection_results: Results from object detection
            video_info: Video metadata
            ollama_client: Optional Ollama client for AI analysis
            
        Returns:
            Reframing strategy analysis
        """
        # Check cache
        cache_key = f"{user_prompt}_{hash(str(object_detection_results))}"
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        strategy = {
            "reframing_type": "standard",
            "focus_strategy": "center",
            "tracking_objects": [],
            "temporal_behavior": "stable",
            "zoom_behavior": "fixed",
            "crop_priorities": {},
            "ai_decisions": {}
        }
        
        # Basic analysis from object detection
        tracking_results = object_detection_results.get("object_tracks", [])
        prompt_analysis = object_detection_results.get("prompt_analysis", {})
        
        # Determine primary focus objects
        focus_objects = self._identify_focus_objects(tracking_results, prompt_analysis)
        strategy["tracking_objects"] = focus_objects
        
        # Determine framing strategy based on prompt
        strategy.update(self._analyze_prompt_for_framing(user_prompt, prompt_analysis))
        
        # AI-enhanced strategy analysis
        if ollama_client:
            try:
                ai_strategy = await self._get_ai_reframing_strategy(
                    user_prompt, 
                    object_detection_results, 
                    video_info,
                    ollama_client
                )
                strategy["ai_decisions"] = ai_strategy
                strategy = self._merge_ai_strategy(strategy, ai_strategy)
            except Exception as e:
                self.logger.warning(f"AI strategy analysis failed: {e}")
        
        # Cache the result
        self.strategy_cache[cache_key] = strategy
        
        return strategy
    
    def _identify_focus_objects(self, 
                              tracking_results: List[Any],
                              prompt_analysis: Dict[str, Any]) -> List[str]:
        """Identify which objects should be the primary focus for reframing."""
        focus_objects = []
        
        # Sort tracking results by relevance
        sorted_tracks = sorted(
            tracking_results,
            key=lambda x: (x.prompt_relevance, x.temporal_consistency, x.avg_confidence),
            reverse=True
        )
        
        # Select top relevant objects
        for track in sorted_tracks[:3]:  # Top 3 objects
            if track.prompt_relevance > 0.5:
                focus_objects.append(track.class_name)
        
        # Add priority objects from prompt analysis
        priority_objects = prompt_analysis.get("priority_objects", [])
        for obj in priority_objects:
            if obj not in focus_objects:
                focus_objects.append(obj)
        
        return focus_objects
    
    def _analyze_prompt_for_framing(self, 
                                  user_prompt: str,
                                  prompt_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prompt to determine framing strategy."""
        prompt_lower = user_prompt.lower()
        strategy_updates = {}
        
        # Action-based framing
        if any(word in prompt_lower for word in ["goal", "shot", "action", "movement"]):
            strategy_updates.update({
                "reframing_type": "action_focused",
                "focus_strategy": "object_tracking",
                "temporal_behavior": "dynamic",
                "zoom_behavior": "adaptive"
            })
        
        # Person-focused framing
        elif any(word in prompt_lower for word in ["player", "person", "character", "face"]):
            strategy_updates.update({
                "reframing_type": "person_focused",
                "focus_strategy": "person_tracking",
                "temporal_behavior": "smooth_follow",
                "zoom_behavior": "medium_close"
            })
        
        # Object-focused framing
        elif any(word in prompt_lower for word in ["ball", "object", "equipment"]):
            strategy_updates.update({
                "reframing_type": "object_focused",
                "focus_strategy": "object_tracking",
                "temporal_behavior": "precise_follow",
                "zoom_behavior": "close_tracking"
            })
        
        # Wide shot requirements
        elif any(word in prompt_lower for word in ["scene", "environment", "overview", "context"]):
            strategy_updates.update({
                "reframing_type": "contextual",
                "focus_strategy": "scene_overview",
                "temporal_behavior": "stable",
                "zoom_behavior": "wide_shot"
            })
        
        # Determine crop priorities
        strategy_updates["crop_priorities"] = self._determine_crop_priorities(prompt_lower)
        
        return strategy_updates
    
    def _determine_crop_priorities(self, prompt_lower: str) -> Dict[str, float]:
        """Determine priorities for different elements in cropping."""
        priorities = {
            "primary_object": 1.0,
            "secondary_objects": 0.6,
            "persons": 0.8,
            "background": 0.2,
            "center_bias": 0.3
        }
        
        # Adjust based on prompt keywords
        if "goal" in prompt_lower or "target" in prompt_lower:
            priorities["primary_object"] = 1.0
            priorities["background"] = 0.1
        
        if "player" in prompt_lower or "person" in prompt_lower:
            priorities["persons"] = 1.0
            priorities["primary_object"] = 0.7
        
        if "scene" in prompt_lower or "context" in prompt_lower:
            priorities["background"] = 0.6
            priorities["center_bias"] = 0.5
        
        return priorities
    
    async def _get_ai_reframing_strategy(self, 
                                       user_prompt: str,
                                       object_detection_results: Dict[str, Any],
                                       video_info: Dict[str, Any],
                                       ollama_client) -> Dict[str, Any]:
        """Get AI-powered reframing strategy using LLM."""
        
        # Prepare context for AI (objects are now dictionaries)
        detected_objects = [obj.get('class_name', '') for obj in object_detection_results.get("detected_objects", [])]
        tracking_summary = object_detection_results.get("summary", {})
        
        prompt_template = f"""
        Analyze this video processing request for optimal reframing strategy:
        
        User Prompt: "{user_prompt}"
        Video Duration: {video_info.get('duration', 'unknown')} seconds
        Detected Objects: {list(set(detected_objects))}
        Object Tracking Summary: {tracking_summary}
        
        Determine the best reframing approach:
        1. What should be the primary focus of cropping?
        2. How should the camera movement behave (static, smooth follow, dynamic)?
        3. What zoom level would work best?
        4. Should we prioritize object tracking or scene context?
        5. What temporal behavior is most appropriate?
        
        Consider the user's intent and provide specific recommendations.
        
        Respond in JSON format:
        {{
            "primary_focus": "object_name or area",
            "camera_behavior": "static/smooth_follow/dynamic_track",
            "zoom_strategy": "wide/medium/close/adaptive",
            "tracking_priority": 0.0-1.0,
            "temporal_smoothing": 0.0-1.0,
            "crop_aggressiveness": 0.0-1.0,
            "reasoning": "explanation of decisions"
        }}
        """
        
        response = await ollama_client.generate_response(prompt_template)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                ai_strategy = json.loads(json_match.group())
                return ai_strategy
        except (json.JSONDecodeError, AttributeError) as e:
            self.logger.warning(f"Failed to parse AI reframing strategy: {e}")
        
        return {}
    
    def _merge_ai_strategy(self, 
                         base_strategy: Dict[str, Any],
                         ai_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Merge AI recommendations with base strategy."""
        merged = base_strategy.copy()
        
        # Map AI recommendations to strategy parameters
        if "camera_behavior" in ai_strategy:
            behavior_mapping = {
                "static": "stable",
                "smooth_follow": "smooth_follow",
                "dynamic_track": "dynamic"
            }
            ai_behavior = ai_strategy["camera_behavior"]
            if ai_behavior in behavior_mapping:
                merged["temporal_behavior"] = behavior_mapping[ai_behavior]
        
        if "zoom_strategy" in ai_strategy:
            zoom_mapping = {
                "wide": "wide_shot",
                "medium": "medium_close",
                "close": "close_tracking",
                "adaptive": "adaptive"
            }
            ai_zoom = ai_strategy["zoom_strategy"]
            if ai_zoom in zoom_mapping:
                merged["zoom_behavior"] = zoom_mapping[ai_zoom]
        
        # Update priorities based on AI recommendations
        if "tracking_priority" in ai_strategy:
            tracking_priority = float(ai_strategy["tracking_priority"])
            if tracking_priority > 0.7:
                merged["focus_strategy"] = "object_tracking"
            elif tracking_priority < 0.3:
                merged["focus_strategy"] = "scene_overview"
        
        return merged
    
    async def generate_reframing_plan(self, 
                                    reframing_strategy: Dict[str, Any],
                                    object_detection_results: Dict[str, Any],
                                    video_info: Dict[str, Any]) -> List[FramingDecision]:
        """
        Generate a detailed reframing plan with frame-by-frame decisions.
        
        Args:
            reframing_strategy: Strategy analysis results
            object_detection_results: Object detection results
            video_info: Video metadata
            
        Returns:
            List of framing decisions
        """
        framing_decisions = []
        
        # Get video dimensions
        frame_width = video_info.get('width', 1920)
        frame_height = video_info.get('height', 1080)
        frame_size = (frame_width, frame_height)
        
        # Get object tracking results
        tracking_results = object_detection_results.get("object_tracks", [])
        
        # Get focus objects from strategy
        focus_objects = reframing_strategy.get("tracking_objects", [])
        
        # Generate base crop regions from object detection
        base_crop_regions = self._calculate_base_crop_regions(
            tracking_results, 
            frame_size,
            focus_objects
        )
        
        # Apply strategy-specific modifications
        refined_regions = self._apply_strategy_refinements(
            base_crop_regions,
            reframing_strategy,
            frame_size
        )
        
        # Apply temporal smoothing
        if reframing_strategy.get("temporal_behavior") in ["smooth_follow", "stable"]:
            smoothed_regions = self._apply_temporal_smoothing(refined_regions)
        else:
            smoothed_regions = refined_regions
        
        # Convert to framing decisions
        for region in smoothed_regions:
            decision = FramingDecision(
                timestamp=region["timestamp"],
                crop_region=region["crop_box"],
                focus_objects=focus_objects,
                confidence=region.get("confidence", 0.8),
                reasoning=self._generate_reasoning(region, reframing_strategy),
                zoom_level=self._calculate_zoom_level(region, frame_size),
                pan_speed=self._calculate_pan_speed(region, refined_regions)
            )
            framing_decisions.append(decision)
        
        # Fill gaps in timeline if needed
        framing_decisions = self._fill_timeline_gaps(
            framing_decisions, 
            video_info.get('duration', 600),
            frame_size
        )
        
        return framing_decisions
    
    def _calculate_base_crop_regions(self, 
                                   tracking_results: List[Any],
                                   frame_size: Tuple[int, int],
                                   focus_objects: List[str]) -> List[Dict[str, Any]]:
        """Calculate base crop regions from object tracking."""
        frame_width, frame_height = frame_size
        crop_regions = []
        
        # Calculate target crop dimensions
        target_ratio = self.params.target_aspect_ratio[0] / self.params.target_aspect_ratio[1]
        
        if frame_width / frame_height > target_ratio:
            crop_height = frame_height
            crop_width = int(crop_height * target_ratio)
        else:
            crop_width = frame_width
            crop_height = int(crop_width / target_ratio)
        
        # Group objects by timestamp
        timestamp_objects = {}
        for track in tracking_results:
            if track.class_name in focus_objects or track.prompt_relevance > 0.6:
                for obj in track.appearances:
                    timestamp = obj.frame_timestamp
                    if timestamp not in timestamp_objects:
                        timestamp_objects[timestamp] = []
                    timestamp_objects[timestamp].append(obj)
        
        # Calculate crop region for each timestamp
        for timestamp, objects in timestamp_objects.items():
            if not objects:
                continue
            
            # Calculate weighted center
            center = self._calculate_weighted_center(objects)
            
            # Apply padding and constraints
            crop_x1 = max(0, center[0] - crop_width // 2)
            crop_y1 = max(0, center[1] - crop_height // 2)
            crop_x2 = min(frame_width, crop_x1 + crop_width)
            crop_y2 = min(frame_height, crop_y1 + crop_height)
            
            # Adjust if out of bounds
            if crop_x2 > frame_width:
                crop_x1 = frame_width - crop_width
                crop_x2 = frame_width
            if crop_y2 > frame_height:
                crop_y1 = frame_height - crop_height
                crop_y2 = frame_height
            
            crop_region = {
                "timestamp": timestamp,
                "crop_box": (crop_x1, crop_y1, crop_x2, crop_y2),
                "center": center,
                "objects_count": len(objects),
                "total_relevance": sum(obj.relevance_score for obj in objects),
                "confidence": min(1.0, sum(obj.confidence for obj in objects) / len(objects))
            }
            
            crop_regions.append(crop_region)
        
        return sorted(crop_regions, key=lambda x: x["timestamp"])
    
    def _calculate_weighted_center(self, objects: List[Any]) -> Tuple[int, int]:
        """Calculate weighted center of objects."""
        if not objects:
            return (0, 0)
        
        weighted_x = 0
        weighted_y = 0
        total_weight = 0
        
        for obj in objects:
            # Weight by relevance and confidence
            weight = obj.relevance_score * obj.confidence * obj.area
            center_x, center_y = obj.center
            
            weighted_x += center_x * weight
            weighted_y += center_y * weight
            total_weight += weight
        
        if total_weight > 0:
            return (int(weighted_x / total_weight), int(weighted_y / total_weight))
        else:
            # Fallback to simple average
            centers = [obj.center for obj in objects]
            avg_x = sum(c[0] for c in centers) // len(centers)
            avg_y = sum(c[1] for c in centers) // len(centers)
            return (avg_x, avg_y)
    
    def _apply_strategy_refinements(self, 
                                  crop_regions: List[Dict[str, Any]],
                                  strategy: Dict[str, Any],
                                  frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Apply strategy-specific refinements to crop regions."""
        refined = crop_regions.copy()
        
        zoom_behavior = strategy.get("zoom_behavior", "fixed")
        focus_strategy = strategy.get("focus_strategy", "center")
        
        # Apply zoom behavior
        if zoom_behavior == "close_tracking":
            refined = self._apply_close_tracking(refined, frame_size)
        elif zoom_behavior == "wide_shot":
            refined = self._apply_wide_shot(refined, frame_size)
        elif zoom_behavior == "adaptive":
            refined = self._apply_adaptive_zoom(refined, frame_size)
        
        # Apply focus strategy
        if focus_strategy == "object_tracking":
            refined = self._enhance_object_focus(refined)
        elif focus_strategy == "scene_overview":
            refined = self._enhance_scene_context(refined, frame_size)
        
        return refined
    
    def _apply_close_tracking(self, 
                            crop_regions: List[Dict[str, Any]],
                            frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Apply close tracking zoom behavior."""
        frame_width, frame_height = frame_size
        
        for region in crop_regions:
            x1, y1, x2, y2 = region["crop_box"]
            current_width = x2 - x1
            current_height = y2 - y1
            
            # Reduce crop size for closer tracking (zoom in)
            zoom_factor = 0.8
            new_width = int(current_width * zoom_factor)
            new_height = int(current_height * zoom_factor)
            
            center_x, center_y = region["center"]
            
            new_x1 = max(0, center_x - new_width // 2)
            new_y1 = max(0, center_y - new_height // 2)
            new_x2 = min(frame_width, new_x1 + new_width)
            new_y2 = min(frame_height, new_y1 + new_height)
            
            region["crop_box"] = (new_x1, new_y1, new_x2, new_y2)
        
        return crop_regions
    
    def _apply_wide_shot(self, 
                       crop_regions: List[Dict[str, Any]],
                       frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Apply wide shot zoom behavior."""
        frame_width, frame_height = frame_size
        
        for region in crop_regions:
            # Use larger crop area for context
            zoom_factor = 1.2
            current_width = region["crop_box"][2] - region["crop_box"][0]
            current_height = region["crop_box"][3] - region["crop_box"][1]
            
            new_width = min(frame_width, int(current_width * zoom_factor))
            new_height = min(frame_height, int(current_height * zoom_factor))
            
            center_x, center_y = region["center"]
            
            new_x1 = max(0, center_x - new_width // 2)
            new_y1 = max(0, center_y - new_height // 2)
            new_x2 = min(frame_width, new_x1 + new_width)
            new_y2 = min(frame_height, new_y1 + new_height)
            
            region["crop_box"] = (new_x1, new_y1, new_x2, new_y2)
        
        return crop_regions
    
    def _apply_adaptive_zoom(self, 
                           crop_regions: List[Dict[str, Any]],
                           frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Apply adaptive zoom based on object density and relevance."""
        for region in crop_regions:
            # Zoom in more when objects are highly relevant
            relevance = region.get("total_relevance", 0.5)
            object_count = region.get("objects_count", 1)
            
            # More objects = wider shot, higher relevance = closer shot
            zoom_factor = 0.9 + (relevance * 0.3) - (object_count * 0.1)
            zoom_factor = max(0.7, min(1.3, zoom_factor))
            
            # Apply zoom
            x1, y1, x2, y2 = region["crop_box"]
            current_width = x2 - x1
            current_height = y2 - y1
            
            new_width = int(current_width * zoom_factor)
            new_height = int(current_height * zoom_factor)
            
            center_x, center_y = region["center"]
            frame_width, frame_height = frame_size
            
            new_x1 = max(0, center_x - new_width // 2)
            new_y1 = max(0, center_y - new_height // 2)
            new_x2 = min(frame_width, new_x1 + new_width)
            new_y2 = min(frame_height, new_y1 + new_height)
            
            region["crop_box"] = (new_x1, new_y1, new_x2, new_y2)
        
        return crop_regions
    
    def _apply_temporal_smoothing(self, 
                                crop_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply temporal smoothing to reduce jittery camera movement."""
        if len(crop_regions) < 3:
            return crop_regions
        
        timestamps = [r["timestamp"] for r in crop_regions]
        
        # Extract crop coordinates
        x1_coords = [r["crop_box"][0] for r in crop_regions]
        y1_coords = [r["crop_box"][1] for r in crop_regions]
        x2_coords = [r["crop_box"][2] for r in crop_regions]
        y2_coords = [r["crop_box"][3] for r in crop_regions]
        
        # Apply Gaussian smoothing
        sigma = self.params.smoothing_window / 6.0  # Convert window to sigma
        
        smooth_x1 = gaussian_filter1d(x1_coords, sigma=sigma)
        smooth_y1 = gaussian_filter1d(y1_coords, sigma=sigma)
        smooth_x2 = gaussian_filter1d(x2_coords, sigma=sigma)
        smooth_y2 = gaussian_filter1d(y2_coords, sigma=sigma)
        
        # Update crop regions with smoothed coordinates
        for i, region in enumerate(crop_regions):
            region["crop_box"] = (
                int(smooth_x1[i]),
                int(smooth_y1[i]),
                int(smooth_x2[i]),
                int(smooth_y2[i])
            )
        
        return crop_regions
    
    def _enhance_object_focus(self, crop_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance focus on detected objects."""
        # Increase confidence for regions with high object relevance
        for region in crop_regions:
            if region.get("total_relevance", 0) > 0.7:
                region["confidence"] = min(1.0, region.get("confidence", 0.8) * 1.2)
        
        return crop_regions
    
    def _enhance_scene_context(self, 
                             crop_regions: List[Dict[str, Any]],
                             frame_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Enhance scene context by expanding crop regions."""
        frame_width, frame_height = frame_size
        
        for region in crop_regions:
            # Expand crop region for better context
            x1, y1, x2, y2 = region["crop_box"]
            expand_factor = 1.15
            
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            new_width = min(frame_width, int(width * expand_factor))
            new_height = min(frame_height, int(height * expand_factor))
            
            new_x1 = max(0, center_x - new_width // 2)
            new_y1 = max(0, center_y - new_height // 2)
            new_x2 = min(frame_width, new_x1 + new_width)
            new_y2 = min(frame_height, new_y1 + new_height)
            
            region["crop_box"] = (new_x1, new_y1, new_x2, new_y2)
        
        return crop_regions
    
    def _generate_reasoning(self, 
                          region: Dict[str, Any],
                          strategy: Dict[str, Any]) -> str:
        """Generate reasoning for framing decision."""
        focus_strategy = strategy.get("focus_strategy", "center")
        objects_count = region.get("objects_count", 0)
        relevance = region.get("total_relevance", 0)
        
        if focus_strategy == "object_tracking" and objects_count > 0:
            return f"Tracking {objects_count} relevant objects (relevance: {relevance:.2f})"
        elif focus_strategy == "scene_overview":
            return "Maintaining scene context for overview"
        elif relevance > 0.8:
            return "High relevance objects detected, focusing tightly"
        else:
            return "Standard framing based on detected content"
    
    def _calculate_zoom_level(self, 
                            region: Dict[str, Any],
                            frame_size: Tuple[int, int]) -> float:
        """Calculate zoom level for the region."""
        frame_width, frame_height = frame_size
        crop_width = region["crop_box"][2] - region["crop_box"][0]
        crop_height = region["crop_box"][3] - region["crop_box"][1]
        
        zoom_x = frame_width / crop_width
        zoom_y = frame_height / crop_height
        
        return min(zoom_x, zoom_y)
    
    def _calculate_pan_speed(self, 
                           current_region: Dict[str, Any],
                           all_regions: List[Dict[str, Any]]) -> float:
        """Calculate pan speed for smooth transitions."""
        current_time = current_region["timestamp"]
        
        # Find previous region
        prev_region = None
        for region in reversed(all_regions):
            if region["timestamp"] < current_time:
                prev_region = region
                break
        
        if not prev_region:
            return 0.0
        
        # Calculate movement distance
        curr_center = current_region["center"]
        prev_center = prev_region["center"]
        
        distance = np.sqrt(
            (curr_center[0] - prev_center[0])**2 + 
            (curr_center[1] - prev_center[1])**2
        )
        
        time_diff = current_time - prev_region["timestamp"]
        
        if time_diff > 0:
            return distance / time_diff
        else:
            return 0.0
    
    def _fill_timeline_gaps(self, 
                          framing_decisions: List[FramingDecision],
                          video_duration: float,
                          frame_size: Tuple[int, int]) -> List[FramingDecision]:
        """Fill gaps in the timeline with interpolated decisions."""
        if not framing_decisions:
            # Create default decision for entire video
            frame_width, frame_height = frame_size
            default_crop = (0, 0, frame_width, frame_height)
            
            return [FramingDecision(
                timestamp=0.0,
                crop_region=default_crop,
                focus_objects=[],
                confidence=0.5,
                reasoning="Default framing - no objects detected",
                zoom_level=1.0,
                pan_speed=0.0
            )]
        
        # Sort by timestamp
        framing_decisions.sort(key=lambda x: x.timestamp)
        
        # Fill gaps between decisions
        filled_decisions = []
        prev_decision = None
        
        for decision in framing_decisions:
            if prev_decision and decision.timestamp - prev_decision.timestamp > 2.0:
                # Insert interpolated decision
                mid_time = (prev_decision.timestamp + decision.timestamp) / 2
                interpolated = self._interpolate_decisions(prev_decision, decision, mid_time)
                filled_decisions.append(interpolated)
            
            filled_decisions.append(decision)
            prev_decision = decision
        
        return filled_decisions
    
    def _interpolate_decisions(self, 
                             decision1: FramingDecision,
                             decision2: FramingDecision,
                             target_time: float) -> FramingDecision:
        """Interpolate between two framing decisions."""
        # Linear interpolation factor
        t = (target_time - decision1.timestamp) / (decision2.timestamp - decision1.timestamp)
        t = max(0.0, min(1.0, t))
        
        # Interpolate crop region
        x1 = int(decision1.crop_region[0] + t * (decision2.crop_region[0] - decision1.crop_region[0]))
        y1 = int(decision1.crop_region[1] + t * (decision2.crop_region[1] - decision1.crop_region[1]))
        x2 = int(decision1.crop_region[2] + t * (decision2.crop_region[2] - decision1.crop_region[2]))
        y2 = int(decision1.crop_region[3] + t * (decision2.crop_region[3] - decision1.crop_region[3]))
        
        # Interpolate other values
        confidence = decision1.confidence + t * (decision2.confidence - decision1.confidence)
        zoom_level = decision1.zoom_level + t * (decision2.zoom_level - decision1.zoom_level)
        pan_speed = decision1.pan_speed + t * (decision2.pan_speed - decision1.pan_speed)
        
        return FramingDecision(
            timestamp=target_time,
            crop_region=(x1, y1, x2, y2),
            focus_objects=decision1.focus_objects,  # Use first decision's objects
            confidence=confidence,
            reasoning="Interpolated framing decision",
            zoom_level=zoom_level,
            pan_speed=pan_speed
        )
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            self.strategy_cache.clear()
            self.logger.info("AIReframingProcessor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
