# src/smart_zoom/object_aware_zoom.py
"""Object-aware smart zoom integration that uses YOLO object detection results"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass


@dataclass
class ObjectAwareZoomDecision:
    """Enhanced zoom decision that considers object detection results."""
    start_time: float
    end_time: float
    zoom_level: float
    crop_position: Tuple[int, int, int, int]  # x, y, width, height
    target_objects: List[Dict]
    confidence: float
    strategy: str
    reasoning: str
    object_focus_weight: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ObjectAwareZoomDecision to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'zoom_level': self.zoom_level,
            'crop_position': list(self.crop_position),
            'target_objects': self.target_objects,
            'confidence': self.confidence,
            'strategy': self.strategy,
            'reasoning': self.reasoning,
            'object_focus_weight': self.object_focus_weight
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectAwareZoomDecision':
        """Create ObjectAwareZoomDecision from dictionary."""
        return cls(
            start_time=data['start_time'],
            end_time=data['end_time'],
            zoom_level=data['zoom_level'],
            crop_position=tuple(data['crop_position']),
            target_objects=data['target_objects'],
            confidence=data['confidence'],
            strategy=data['strategy'],
            reasoning=data['reasoning'],
            object_focus_weight=data.get('object_focus_weight', 0.0)
        )


class ObjectAwareZoomProcessor:
    """Enhanced smart zoom processor that integrates object detection results."""
    
    def __init__(self, 
                 base_zoom_processor,
                 object_detector=None,
                 ai_reframer=None):
        """
        Initialize object-aware zoom processor.
        
        Args:
            base_zoom_processor: Base SmartZoomProcessor instance
            object_detector: IntelligentObjectDetector instance
            ai_reframer: AIReframingProcessor instance
        """
        self.logger = logging.getLogger(__name__)
        self.base_zoom_processor = base_zoom_processor
        self.object_detector = object_detector
        self.ai_reframer = ai_reframer
        
        # Configuration
        self.object_tracking_enabled = True
        self.min_object_confidence = 0.5
        self.min_relevance_score = 0.3
        
        self.logger.info("ObjectAwareZoomProcessor initialized")
    
    async def process_video_with_objects(self,
                                       video_path: str,
                                       output_path: str,
                                       user_prompt: str = None,
                                       audio_analysis: Dict = None,
                                       target_aspect_ratio: List[int] = None,
                                       is_vertical: bool = True,
                                       canvas_type: str = "shorts") -> Dict[str, Any]:
        """
        Process video with object-aware smart zoom.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            user_prompt: User prompt for object detection
            audio_analysis: Audio analysis results
            target_aspect_ratio: Target aspect ratio [width, height]
            is_vertical: Whether output is vertical
            canvas_type: Canvas type (shorts, horizontal, etc.)
            
        Returns:
            Processing results with object detection and zoom information
        """
        try:
            self.logger.info(f"Processing video with object-aware zoom: {video_path}")
            
            # Step 1: Perform object detection if prompt provided
            object_detection_results = None
            reframing_strategy = None
            
            if user_prompt and self.object_detector:
                self.logger.info(f"Running object detection for prompt: '{user_prompt}'")
                
                # Analyze prompt for objects
                prompt_analysis = await self.object_detector.analyze_prompt_for_objects(user_prompt)
                
                # Perform object detection
                object_detection_results = await self.object_detector.detect_objects_in_video(
                    video_path=video_path,
                    prompt_analysis=prompt_analysis,
                    sample_rate=2.0,  # Higher sample rate for better tracking
                    save_debug_frames=True
                )
                
                # Generate AI reframing strategy if available
                if self.ai_reframer and object_detection_results:
                    video_info = self._get_video_info(video_path)
                    reframing_strategy = await self.ai_reframer.analyze_reframing_strategy(
                        user_prompt=user_prompt,
                        object_detection_results=object_detection_results,
                        video_info=video_info
                    )
                    
                    self.logger.info(f"AI reframing strategy: {reframing_strategy.get('reframing_type', 'standard')}")
            
            # Step 2: Enhanced zoom processing with object awareness
            if object_detection_results and reframing_strategy:
                # Use object-aware processing
                zoom_results = await self._process_with_object_awareness(
                    video_path=video_path,
                    output_path=output_path,
                    object_detection_results=object_detection_results,
                    reframing_strategy=reframing_strategy,
                    audio_analysis=audio_analysis,
                    target_aspect_ratio=target_aspect_ratio,
                    is_vertical=is_vertical
                )
            else:
                # Fallback to standard smart zoom
                self.logger.info("Falling back to standard smart zoom processing")
                zoom_results = self.base_zoom_processor.process_video(
                    input_path=video_path,
                    output_path=output_path,
                    audio_analysis=audio_analysis,
                    target_aspect_ratio=target_aspect_ratio,
                    is_vertical=is_vertical,
                    canvas_type=canvas_type
                )
            
            # Combine results
            results = {
                'status': 'success',
                'processing_time': zoom_results.get('processing_time', 0),
                'object_detection_enabled': object_detection_results is not None,
                'ai_reframing_enabled': reframing_strategy is not None,
                'zoom_decisions': zoom_results.get('zoom_decisions', []),
                'object_tracking_stats': self._calculate_object_stats(object_detection_results),
                'reframing_strategy': reframing_strategy
            }
            
            self.logger.info(f"Object-aware zoom processing completed in {results['processing_time']:.2f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Object-aware zoom processing failed: {e}")
            # Fallback to standard processing
            return self.base_zoom_processor.process_video(
                input_path=video_path,
                output_path=output_path,
                audio_analysis=audio_analysis,
                target_aspect_ratio=target_aspect_ratio,
                is_vertical=is_vertical,
                canvas_type=canvas_type
            )
    
    async def _process_with_object_awareness(self,
                                           video_path: str,
                                           output_path: str,
                                           object_detection_results: Dict,
                                           reframing_strategy: Dict,
                                           audio_analysis: Dict = None,
                                           target_aspect_ratio: List[int] = None,
                                           is_vertical: bool = True) -> Dict[str, Any]:
        """Process video with object detection guidance."""
        import cv2
        import time
        
        start_time = time.time()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate target dimensions
        if target_aspect_ratio:
            target_width, target_height = target_aspect_ratio
            if is_vertical:
                output_width = int(height * target_width / target_height)
                output_height = height
            else:
                output_width = width
                output_height = int(width * target_height / target_width)
        else:
            output_width, output_height = width, height
        
        # Setup temporary video writer for frames only
        temp_video_path = output_path.replace('.mp4', '_temp_frames.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (output_width, output_height))
        
        # Get object tracking data
        objects_by_frame = object_detection_results.get('objects_by_frame', {})
        object_tracks = object_detection_results.get('object_tracks', [])
        
        # Generate zoom decisions based on objects
        zoom_decisions = self._generate_object_aware_zoom_decisions(
            objects_by_frame, 
            object_tracks, 
            reframing_strategy,
            fps,
            total_frames
        )
        
        self.logger.info(f"Generated {len(zoom_decisions)} object-aware zoom decisions")
        
        # Process frames
        frame_count = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Find applicable zoom decision
            current_zoom = self._get_zoom_decision_for_timestamp(zoom_decisions, timestamp)
            
            if current_zoom:
                # Apply object-aware cropping
                cropped_frame = self._apply_object_aware_crop(
                    frame, 
                    current_zoom,
                    output_width,
                    output_height
                )
            else:
                # Default center crop
                cropped_frame = self._default_crop(frame, output_width, output_height)
            
            out.write(cropped_frame)
            processed_frames += 1
            frame_count += 1
            
            # Log progress
            if processed_frames % 300 == 0:
                progress = (frame_count / total_frames) * 100
                self.logger.info(f"Object-aware processing: {progress:.1f}% ({processed_frames}/{total_frames})")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Now merge the processed frames with original audio using FFmpeg
        self.logger.info("Merging processed video with original audio using FFmpeg")
        self._merge_video_with_audio(temp_video_path, video_path, output_path)
        
        # Clean up temporary file
        import os
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        processing_time = time.time() - start_time
        
        return {
            'processing_time': processing_time,
            'zoom_decisions': zoom_decisions,
            'frames_processed': processed_frames,
            'object_aware_crops': len([zd for zd in zoom_decisions if zd.object_focus_weight > 0])
        }
    
    def _generate_object_aware_zoom_decisions(self,
                                            objects_by_frame: Dict,
                                            object_tracks: List,
                                            reframing_strategy: Dict,
                                            fps: float,
                                            total_frames: int) -> List[ObjectAwareZoomDecision]:
        """Generate zoom decisions based on object tracking results."""
        decisions = []
        
        # Get high-relevance object tracks
        relevant_tracks = [
            track for track in object_tracks 
            if track.prompt_relevance > self.min_relevance_score
        ]
        
        if not relevant_tracks:
            self.logger.info("No relevant object tracks found, using default zoom")
            return []
        
        # Sort tracks by relevance and temporal consistency
        relevant_tracks.sort(
            key=lambda t: (t.prompt_relevance, t.temporal_consistency, t.avg_confidence),
            reverse=True
        )
        
        # Group consecutive frames with similar object configurations
        decision_segments = self._group_frames_by_object_presence(
            objects_by_frame, 
            relevant_tracks, 
            fps
        )
        
        # Generate zoom decisions for each segment
        for segment in decision_segments:
            start_time = segment['start_time']
            end_time = segment['end_time']
            dominant_objects = segment['objects']
            
            # Calculate zoom parameters based on objects
            zoom_level, crop_position = self._calculate_object_based_zoom(
                dominant_objects,
                reframing_strategy
            )
            
            decision = ObjectAwareZoomDecision(
                start_time=start_time,
                end_time=end_time,
                zoom_level=zoom_level,
                crop_position=crop_position,
                target_objects=dominant_objects,
                confidence=self._calculate_decision_confidence(dominant_objects),
                strategy=reframing_strategy.get('reframing_type', 'object_focused'),
                reasoning=f"Focusing on {len(dominant_objects)} relevant objects",
                object_focus_weight=self._calculate_object_focus_weight(dominant_objects)
            )
            
            decisions.append(decision)
        
        self.logger.info(f"Generated {len(decisions)} object-aware zoom decisions")
        return decisions
    
    def _get_video_info(self, video_path: str) -> Dict:
        """Get basic video information."""
        cap = cv2.VideoCapture(video_path)
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info
    
    def _calculate_object_stats(self, object_detection_results: Dict) -> Dict:
        """Calculate object detection statistics."""
        if not object_detection_results:
            return {}
        
        return {
            'total_detections': object_detection_results.get('total_detections', 0),
            'relevant_detections': object_detection_results.get('relevant_detections', 0),
            'detection_rate': object_detection_results.get('detection_rate', 0),
            'relevance_rate': object_detection_results.get('relevance_rate', 0)
        }
    
    def _group_frames_by_object_presence(self, 
                                       objects_by_frame: Dict,
                                       relevant_tracks: List,
                                       fps: float) -> List[Dict]:
        """Group consecutive frames with similar object configurations."""
        segments = []
        current_segment = None
        
        frame_numbers = sorted([int(f) for f in objects_by_frame.keys()])
        
        for frame_num in frame_numbers:
            timestamp = frame_num / fps
            frame_objects = objects_by_frame.get(str(frame_num), [])
            
            # Filter relevant objects in this frame
            relevant_frame_objects = [
                obj for obj in frame_objects
                if hasattr(obj, 'relevance_score') and obj.relevance_score > self.min_relevance_score
            ]
            
            if relevant_frame_objects:
                if current_segment is None:
                    # Start new segment
                    current_segment = {
                        'start_time': timestamp,
                        'end_time': timestamp,
                        'objects': relevant_frame_objects,
                        'frame_count': 1
                    }
                else:
                    # Extend current segment
                    current_segment['end_time'] = timestamp
                    current_segment['frame_count'] += 1
                    # Update objects (could do more sophisticated merging here)
                    current_segment['objects'] = relevant_frame_objects
            else:
                # End current segment if exists
                if current_segment is not None:
                    segments.append(current_segment)
                    current_segment = None
        
        # Add final segment if exists
        if current_segment is not None:
            segments.append(current_segment)
        
        return segments
    
    def _calculate_object_based_zoom(self, 
                                   objects: List,
                                   reframing_strategy: Dict) -> Tuple[float, Tuple[int, int, int, int]]:
        """Calculate zoom level and crop position based on objects."""
        if not objects:
            return 1.0, (0, 0, 1920, 1080)  # Default values
        
        # Calculate bounding box that encompasses all relevant objects
        min_x = min(obj.bbox[0] for obj in objects)
        min_y = min(obj.bbox[1] for obj in objects)
        max_x = max(obj.bbox[2] for obj in objects)
        max_y = max(obj.bbox[3] for obj in objects)
        
        # Add padding based on reframing strategy
        padding_factor = reframing_strategy.get('crop_padding', 0.2)
        width = max_x - min_x
        height = max_y - min_y
        
        padded_width = width * (1 + padding_factor)
        padded_height = height * (1 + padding_factor)
        
        # Center the crop on the object cluster
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        crop_x = max(0, center_x - padded_width / 2)
        crop_y = max(0, center_y - padded_height / 2)
        
        # Calculate zoom level (inverse of crop size relative to frame)
        zoom_level = min(1920 / padded_width, 1080 / padded_height)
        zoom_level = max(1.0, min(zoom_level, 3.0))  # Clamp between 1x and 3x
        
        return zoom_level, (int(crop_x), int(crop_y), int(padded_width), int(padded_height))
    
    def _calculate_decision_confidence(self, objects: List) -> float:
        """Calculate confidence score for zoom decision."""
        if not objects:
            return 0.0
        
        # Average of object confidences and relevance scores
        confidences = [obj.confidence for obj in objects if hasattr(obj, 'confidence')]
        relevances = [obj.relevance_score for obj in objects if hasattr(obj, 'relevance_score')]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_relevance = sum(relevances) / len(relevances) if relevances else 0.0
        
        return (avg_confidence + avg_relevance) / 2.0
    
    def _calculate_object_focus_weight(self, objects: List) -> float:
        """Calculate how much this decision is influenced by objects."""
        if not objects:
            return 0.0
        
        # Higher weight for more relevant and confident objects
        weights = []
        for obj in objects:
            if hasattr(obj, 'relevance_score') and hasattr(obj, 'confidence'):
                weight = obj.relevance_score * obj.confidence
                weights.append(weight)
        
        return sum(weights) / len(weights) if weights else 0.0
    
    def _get_zoom_decision_for_timestamp(self, 
                                       decisions: List[ObjectAwareZoomDecision],
                                       timestamp: float) -> Optional[ObjectAwareZoomDecision]:
        """Get the applicable zoom decision for a given timestamp."""
        for decision in decisions:
            if decision.start_time <= timestamp <= decision.end_time:
                return decision
        return None
    
    def _apply_object_aware_crop(self,
                               frame: np.ndarray,
                               zoom_decision: ObjectAwareZoomDecision,
                               output_width: int,
                               output_height: int) -> np.ndarray:
        """Apply object-aware cropping to frame."""
        height, width = frame.shape[:2]
        
        crop_x, crop_y, crop_width, crop_height = zoom_decision.crop_position
        
        # Ensure crop is within frame bounds
        crop_x = max(0, min(crop_x, width - crop_width))
        crop_y = max(0, min(crop_y, height - crop_height))
        crop_width = min(crop_width, width - crop_x)
        crop_height = min(crop_height, height - crop_y)
        
        # Extract crop
        cropped = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]
        
        # Resize to output dimensions
        resized = cv2.resize(cropped, (output_width, output_height))
        
        return resized
    
    def _default_crop(self, 
                     frame: np.ndarray,
                     output_width: int,
                     output_height: int) -> np.ndarray:
        """Apply default center crop."""
        height, width = frame.shape[:2]
        
        # Center crop
        start_x = (width - output_width) // 2
        start_y = (height - output_height) // 2
        
        if start_x < 0 or start_y < 0:
            # Need to resize first
            resized = cv2.resize(frame, (output_width, output_height))
            return resized
        else:
            cropped = frame[start_y:start_y+output_height, start_x:start_x+output_width]
            return cropped

    def _merge_video_with_audio(self, video_only_path: str, original_video_path: str, output_path: str):
        """Merge processed video frames with original audio using FFmpeg."""
        import subprocess
        
        try:
            # Build FFmpeg command to merge video and audio
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', video_only_path,  # Video input (no audio)
                '-i', original_video_path,  # Audio source
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',   # Re-encode audio for compatibility
                '-map', '0:v:0',  # Map video from first input
                '-map', '1:a:0',  # Map audio from second input  
                '-shortest',      # End when shortest stream ends
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output
                output_path
            ]
            
            self.logger.debug(f"Merging audio with FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully merged video with audio: {output_path}")
            else:
                self.logger.error(f"FFmpeg audio merge failed: {result.stderr}")
                # Fallback: just copy the video-only file
                import shutil
                shutil.copy2(video_only_path, output_path)
                self.logger.warning(f"Fallback: copied video without audio to {output_path}")
                
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg audio merge timed out")
        except Exception as e:
            self.logger.error(f"Error merging audio: {e}")
            # Fallback: copy video-only file
            import shutil
            shutil.copy2(video_only_path, output_path)
