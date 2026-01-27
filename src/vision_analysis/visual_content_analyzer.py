# src/vision_analysis/visual_content_analyzer.py
"""Visual content analysis using Ollama vision models"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import time
from ..ai_integration.ollama_client import OllamaClient


class VisualContentAnalyzer:
    """
    Analyzes visual content using Ollama vision models to enhance video segment selection.
    """
    
    def __init__(self, ollama_client: OllamaClient = None):
        """
        Initialize visual content analyzer.
        
        Args:
            ollama_client: Existing Ollama client instance (optional)
        """
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        
        # Vision analysis prompts
        self.vision_prompts = {
            'scene_description': """
            Analyze this video frame and provide a brief description focusing on:
            1. What type of scene is this? (talking head, demonstration, action, text/slides, etc.)
            2. Are there people visible? How many? What are they doing?
            3. Is this an engaging visual scene for social media?
            4. Rate the visual interest level from 1-10
            
            Keep response concise (1-2 sentences). Respond in JSON format:
            {{
                "scene_type": "talking_head|demonstration|action|slides|other",
                "people_count": number,
                "activity": "brief description",
                "visual_interest": number,
                "engaging": true/false,
                "description": "brief scene description"
            }}
            """,
            
            'content_type_detection': """
            Looking at this video frame, determine the content type:
            - Is this educational content, entertainment, demonstration, or presentation?
            - Are there visual elements that enhance the content (graphics, text, demonstrations)?
            - Would this frame work well in a vertical (9:16) format for social media?
            
            Respond in JSON format:
            {{
                "content_type": "educational|entertainment|demonstration|presentation|other",
                "has_graphics": true/false,
                "has_text": true/false,
                "vertical_suitable": true/false,
                "visual_elements": ["element1", "element2"]
            }}
            """,
            
            'engagement_assessment': """
            Assess this frame for social media engagement potential:
            1. Is this visually interesting or compelling?
            2. Does it show emotion, action, or important visual information?
            3. Would this grab attention in a social media feed?
            4. Rate engagement potential 1-10
            
            Brief response in JSON:
            {{
                "engagement_score": number,
                "attention_grabbing": true/false,
                "has_emotion": true/false,
                "has_action": true/false,
                "social_media_suitable": true/false
            }}
            """,
            
            'scene_transition': """
            Analyze this frame for scene transition markers:
            - Is this the start/end of a scene or topic?
            - Are there visual cues indicating a transition?
            - Is this a good cut point for video editing?
            
            JSON response:
            {{
                "transition_point": true/false,
                "scene_start": true/false,
                "scene_end": true/false,
                "cut_quality": "good|fair|poor"
            }}
            """
        }
    
    async def analyze_frames(self, frames: List[Dict], analysis_type: str = 'scene_description') -> List[Dict]:
        """
        Analyze multiple frames with vision models.
        
        Args:
            frames: List of frame dictionaries with base64 data
            analysis_type: Type of analysis to perform
            
        Returns:
            List of analysis results
        """
        if not frames:
            return []
        
        if not self.ollama_client:
            self.logger.warning("No Ollama client available for vision analysis")
            return []
        
        try:
            # Get the appropriate prompt
            prompt = self.vision_prompts.get(analysis_type, self.vision_prompts['scene_description'])
            
            # Process frames in batches to avoid overwhelming the API
            batch_size = 3  # Process 3 frames at once
            results = []
            
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                
                # Process batch in parallel
                batch_tasks = []
                for frame in batch:
                    if 'base64' in frame:
                        task = self._analyze_single_frame(frame, prompt, analysis_type)
                        batch_tasks.append(task)
                
                if batch_tasks:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for j, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            self.logger.warning(f"Frame analysis failed: {result}")
                            # Add fallback result
                            results.append({
                                'timestamp': batch[j].get('timestamp', 0),
                                'analysis_type': analysis_type,
                                'error': str(result),
                                'scene_type': 'unknown',
                                'visual_interest': 5  # Default medium interest
                            })
                        else:
                            # Add timestamp and metadata
                            if isinstance(result, dict):
                                result['timestamp'] = batch[j].get('timestamp', 0)
                                result['analysis_type'] = analysis_type
                            results.append(result)
                    
                    # Brief pause between batches to avoid rate limiting
                    if i + batch_size < len(frames):
                        await asyncio.sleep(0.5)
            
            self.logger.info(f"Analyzed {len(frames)} frames with {analysis_type}, got {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch frame analysis: {e}")
            return []
    
    async def _analyze_single_frame(self, frame: Dict, prompt: str, analysis_type: str) -> Dict:
        """Analyze a single frame with the vision model."""
        try:
            # Use vision-capable model
            vision_model = self.ollama_client.get_best_model("vision")
            
            # Make request with image
            response_text = await self.ollama_client._make_request(
                prompt=prompt,
                model=vision_model,
                cache_key=f"vision_{analysis_type}_{hash(frame.get('base64', '')[:100])}",  # Cache key from partial base64
                images=[frame['base64']]
            )
            
            # Parse response
            result = self.ollama_client._parse_json_response(response_text)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing single frame: {e}")
            return {
                'error': str(e),
                'scene_type': 'unknown',
                'visual_interest': 5
            }
    
    async def analyze_segment_visuals(self, segment_frames: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Analyze visual content for multiple segments.
        
        Args:
            segment_frames: Dictionary mapping segment IDs to frame lists
            
        Returns:
            Dictionary mapping segment IDs to visual analysis results
        """
        segment_analyses = {}
        
        try:
            # Process each segment
            for segment_id, frames in segment_frames.items():
                if not frames:
                    continue
                
                self.logger.debug(f"Analyzing visuals for {segment_id} with {len(frames)} frames")
                
                # Analyze frames for scene description
                scene_analyses = await self.analyze_frames(frames, 'scene_description')
                
                # Analyze frames for engagement
                engagement_analyses = await self.analyze_frames(frames, 'engagement_assessment')
                
                # Combine and summarize results
                segment_analysis = self._summarize_segment_analysis(
                    segment_id, frames, scene_analyses, engagement_analyses
                )
                
                segment_analyses[segment_id] = segment_analysis
                
                # Brief pause between segments
                await asyncio.sleep(0.3)
            
            self.logger.info(f"Completed visual analysis for {len(segment_analyses)} segments")
            return segment_analyses
            
        except Exception as e:
            self.logger.error(f"Error in segment visual analysis: {e}")
            return segment_analyses
    
    def _summarize_segment_analysis(self, 
                                  segment_id: str,
                                  frames: List[Dict], 
                                  scene_analyses: List[Dict], 
                                  engagement_analyses: List[Dict]) -> Dict:
        """Summarize analysis results for a segment."""
        try:
            # Calculate average scores
            visual_interest_scores = [
                a.get('visual_interest', 5) for a in scene_analyses 
                if isinstance(a.get('visual_interest'), (int, float))
            ]
            
            engagement_scores = [
                a.get('engagement_score', 5) for a in engagement_analyses 
                if isinstance(a.get('engagement_score'), (int, float))
            ]
            
            avg_visual_interest = sum(visual_interest_scores) / len(visual_interest_scores) if visual_interest_scores else 5.0
            avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 5.0
            
            # Determine dominant scene type
            scene_types = [a.get('scene_type', 'unknown') for a in scene_analyses if a.get('scene_type')]
            dominant_scene_type = max(set(scene_types), key=scene_types.count) if scene_types else 'unknown'
            
            # Check for visual elements
            has_people = any(a.get('people_count', 0) > 0 for a in scene_analyses)
            has_action = any(a.get('has_action', False) for a in engagement_analyses)
            has_emotion = any(a.get('has_emotion', False) for a in engagement_analyses)
            
            # Overall recommendation
            visual_quality = 'high' if avg_visual_interest >= 7 else 'medium' if avg_visual_interest >= 5 else 'low'
            engagement_quality = 'high' if avg_engagement >= 7 else 'medium' if avg_engagement >= 5 else 'low'
            
            return {
                'segment_id': segment_id,
                'frame_count': len(frames),
                'visual_interest_avg': round(avg_visual_interest, 2),
                'engagement_score_avg': round(avg_engagement, 2),
                'dominant_scene_type': dominant_scene_type,
                'has_people': has_people,
                'has_action': has_action,
                'has_emotion': has_emotion,
                'visual_quality': visual_quality,
                'engagement_quality': engagement_quality,
                'recommendation': self._generate_visual_recommendation(
                    avg_visual_interest, avg_engagement, dominant_scene_type, has_people, has_action, has_emotion
                ),
                'scene_analyses': scene_analyses,
                'engagement_analyses': engagement_analyses
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing segment analysis for {segment_id}: {e}")
            return {
                'segment_id': segment_id,
                'frame_count': len(frames),
                'error': str(e),
                'visual_quality': 'unknown',
                'engagement_quality': 'unknown'
            }
    
    def _generate_visual_recommendation(self, 
                                      visual_interest: float, 
                                      engagement: float, 
                                      scene_type: str,
                                      has_people: bool, 
                                      has_action: bool, 
                                      has_emotion: bool) -> str:
        """Generate a recommendation based on visual analysis."""
        
        # High-quality visuals
        if visual_interest >= 7 and engagement >= 7:
            if has_emotion or has_action:
                return "excellent_for_shorts"
            else:
                return "good_for_shorts"
        
        # Medium-quality visuals
        elif visual_interest >= 5 and engagement >= 5:
            if scene_type in ['talking_head', 'demonstration'] and has_people:
                return "suitable_for_shorts"
            else:
                return "consider_for_shorts"
        
        # Low-quality visuals
        else:
            if scene_type == 'slides' or not has_people:
                return "poor_for_shorts"
            else:
                return "marginal_for_shorts"
    
    async def quick_scene_assessment(self, frames: List[Dict]) -> Dict:
        """
        Quick assessment of scene content for fast processing.
        
        Args:
            frames: List of frame dictionaries
            
        Returns:
            Quick assessment results
        """
        if not frames or not self.ollama_client:
            return {
                'scene_type': 'unknown',
                'visual_interest': 5,
                'suitable_for_shorts': False
            }
        
        try:
            # Use only the best quality frame for quick assessment
            best_frame = max(frames, key=lambda f: f.get('quality_score', 0))
            
            # Simple prompt for quick analysis
            quick_prompt = """
            Quickly assess this video frame:
            1. Scene type (talking_head, demonstration, action, slides)
            2. Visual interest rating 1-10
            3. Suitable for social media shorts? (yes/no)
            
            Brief JSON response:
            {{"scene_type": "type", "visual_interest": number, "suitable": true/false}}
            """
            
            result = await self._analyze_single_frame(best_frame, quick_prompt, 'quick_assessment')
            
            return {
                'scene_type': result.get('scene_type', 'unknown'),
                'visual_interest': result.get('visual_interest', 5),
                'suitable_for_shorts': result.get('suitable', False),
                'timestamp': best_frame.get('timestamp', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in quick scene assessment: {e}")
            return {
                'scene_type': 'unknown',
                'visual_interest': 5,
                'suitable_for_shorts': False,
                'error': str(e)
            }
    
    async def analyze_segment_visuals_efficient(self, segment_frames: Dict) -> Dict:
        """
        Efficiently analyze visual content for video segments using smart sampling.
        
        Args:
            segment_frames: Dictionary mapping segment IDs to frame lists
            
        Returns:
            Enhanced visual analysis results optimized for segment selection
        """
        try:
            if not self.ollama_client:
                self.logger.warning("No Ollama client available for vision analysis")
                return {'status': 'no_client', 'segments': []}
            
            # Handle both dictionary and list input formats
            if isinstance(segment_frames, dict):
                # Convert dict format to list format for processing
                segment_list = []
                for segment_id, frames in segment_frames.items():
                    # Extract timing from first frame if available
                    if frames:
                        start_time = min(frame.get('timestamp', 0) for frame in frames)
                        end_time = max(frame.get('timestamp', 0) for frame in frames)
                        segment_list.append({
                            'segment_id': segment_id,
                            'start_time': start_time,
                            'end_time': end_time,
                            'frames': frames
                        })
                segment_frames_list = segment_list
            else:
                # Already in list format
                segment_frames_list = segment_frames
            
            self.logger.info(f"Starting efficient visual analysis for {len(segment_frames_list)} segments")
            
            results = []
            
            for segment_info in segment_frames_list:
                segment_result = await self._analyze_segment_efficient(segment_info)
                if segment_result:
                    results.append(segment_result)
            
            return {
                'status': 'success',
                'segments': results,
                'analysis_method': 'efficient_visual_analysis',
                'total_segments': len(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error in efficient segment visual analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    async def _analyze_segment_efficient(self, segment_info: Dict) -> Optional[Dict]:
        """
        Analyze a single segment efficiently with smart frame selection.
        
        Args:
            segment_info: Dictionary containing segment timing and frame data
            
        Returns:
            Segment analysis results
        """
        try:
            start_time = segment_info.get('start_time', 0)
            end_time = segment_info.get('end_time', 0)
            frames = segment_info.get('frames', [])
            
            if not frames:
                return None
            
            # Smart frame selection: analyze up to 2 representative frames per segment
            selected_frames = self._select_representative_frames(frames)
            
            # Analyze selected frames with batch processing
            if hasattr(self.ollama_client, 'batch_analyze_frames'):
                frame_analyses = await self.ollama_client.batch_analyze_frames(
                    selected_frames, 'scene_analysis'
                )
            else:
                # Fallback to individual analysis
                frame_analyses = []
                for frame in selected_frames:
                    if 'base64' in frame:
                        analysis = await self.ollama_client.analyze_visual_content(
                            frame['base64'], 'scene_analysis'
                        )
                        if analysis and not analysis.get('error'):
                            frame_analyses.append(analysis)
            
            if not frame_analyses:
                return None
            
            # Aggregate frame analyses into segment result
            segment_analysis = self._aggregate_frame_analyses(
                frame_analyses, start_time, end_time
            )
            
            return segment_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing segment: {e}")
            return None
    
    def _select_representative_frames(self, frames: List[Dict], segment_duration: float = 30.0) -> List[Dict]:
        """
        Select the most representative frames from a segment for analysis.
        
        ENHANCED: Now samples more frames for better visual understanding.
        
        Args:
            frames: List of frame dictionaries
            segment_duration: Duration of the segment in seconds
            
        Returns:
            Selected representative frames (3-5 frames for better coverage)
        """
        if not frames:
            return []
        
        # If we have few frames, analyze all
        if len(frames) <= 3:
            return frames
        
        # Calculate how many frames to sample based on duration
        # Minimum 3, maximum 5 frames per segment
        num_frames = max(3, min(5, int(segment_duration // 10)))
        
        selected = []
        
        # First frame (beginning of segment - important for hook)
        selected.append(frames[0])
        
        # Evenly distributed frames across the segment
        if len(frames) >= num_frames:
            step = len(frames) // (num_frames - 1)
            for i in range(1, num_frames - 1):
                idx = min(i * step, len(frames) - 1)
                if frames[idx] not in selected:
                    selected.append(frames[idx])
            
            # Last frame (end of segment)
            if frames[-1] not in selected:
                selected.append(frames[-1])
        else:
            # If fewer frames than desired, take all
            selected = frames[:num_frames]
        
        self.logger.debug(f"Selected {len(selected)} representative frames from {len(frames)} total")
        return selected[:5]  # Maximum 5 frames per segment
    
    def _aggregate_frame_analyses(self, frame_analyses: List[Dict], start_time: float, end_time: float) -> Dict:
        """
        Aggregate multiple frame analyses into a single segment analysis.
        
        ENHANCED: Better scoring with variance detection and action/emotion tracking.
        
        Args:
            frame_analyses: List of individual frame analysis results
            start_time: Segment start time
            end_time: Segment end time
            
        Returns:
            Aggregated segment analysis with comprehensive visual metrics
        """
        if not frame_analyses:
            return {
                'start_time': start_time,
                'end_time': end_time,
                'visual_score': 0.0,  # Changed from 0.5 to 0.0 for no data
                'scene_type': 'unknown',
                'visual_interest': 0,
                'has_vision_data': False
            }
        
        # Aggregate scores
        visual_interests = [analysis.get('visual_interest', 5) for analysis in frame_analyses]
        engagement_scores = [analysis.get('engagement_score', 5) for analysis in frame_analyses]
        
        avg_visual_interest = sum(visual_interests) / len(visual_interests)
        max_visual_interest = max(visual_interests)
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else avg_visual_interest
        
        # Calculate variance in visual interest (indicates dynamic content)
        visual_variance = 0.0
        if len(visual_interests) > 1:
            mean = avg_visual_interest
            visual_variance = sum((x - mean) ** 2 for x in visual_interests) / len(visual_interests)
        
        # Determine dominant scene type
        scene_types = [analysis.get('scene_type', 'unknown') for analysis in frame_analyses]
        dominant_scene_type = max(set(scene_types), key=scene_types.count)
        
        # Check for people presence
        people_counts = [analysis.get('people_count', 0) for analysis in frame_analyses]
        max_people = max(people_counts) if people_counts else 0
        avg_people = sum(people_counts) / len(people_counts) if people_counts else 0
        
        # Check for action/emotion indicators
        has_action = any(analysis.get('has_action', False) for analysis in frame_analyses)
        has_emotion = any(analysis.get('has_emotion', False) for analysis in frame_analyses)
        
        # Calculate overall visual score (0-1 scale) - ENHANCED
        visual_score = avg_visual_interest / 10.0  # Convert from 1-10 to 0-1
        
        # Bonus for peak visual interest (captures highlights)
        if max_visual_interest > avg_visual_interest + 1:
            visual_score += 0.05
        
        # Apply bonuses for engaging content
        if max_people > 0:
            visual_score += 0.08  # Increased bonus for people presence
            if max_people > 1:
                visual_score += 0.04  # Extra for multiple people
        
        if dominant_scene_type in ['demonstration', 'action']:
            visual_score += 0.10  # Increased bonus for dynamic content
        elif dominant_scene_type in ['presentation', 'interview', 'talking_head']:
            visual_score += 0.05
        
        if has_action:
            visual_score += 0.05
        
        if has_emotion:
            visual_score += 0.03
        
        # Bonus for visual dynamism (high variance = changing/interesting content)
        if visual_variance > 2.0:
            visual_score += 0.05
        
        visual_score = min(1.0, visual_score)  # Cap at 1.0
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'visual_score': visual_score,
            'visual_interest': avg_visual_interest,
            'max_visual_interest': max_visual_interest,
            'engagement_score': avg_engagement / 10.0,  # Normalize to 0-1
            'scene_type': dominant_scene_type,
            'people_count': max_people,
            'avg_people_count': avg_people,
            'people_visible': max_people > 0,
            'has_action': has_action,
            'has_emotion': has_emotion,
            'visual_variance': visual_variance,
            'frames_analyzed': len(frame_analyses),
            'audiovisual_alignment': 0.7,  # Default alignment score
            'analysis_quality': 'comprehensive' if len(frame_analyses) >= 3 else ('efficient' if len(frame_analyses) >= 2 else 'basic'),
            'has_vision_data': True
        }
