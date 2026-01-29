# src/content_analysis/content_analyzer.py
"""Content analysis for identifying optimal short video segments"""

import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .prompt_based_analyzer import PromptBasedAnalyzer
from .comprehensive_segment_generator import ComprehensiveSegmentGenerator
from ..object_detection.intelligent_object_detector import IntelligentObjectDetector
from ..object_detection.ai_reframing_processor import AIReframingProcessor, ReframingParameters
from ..utils.speech_boundary_adjuster import SpeechBoundaryAdjuster

from ..face_insights.celebrity_index import (
    load_celebrity_index,
    actor_coverage_for_segment,
    compute_celebrity_score,
)


@dataclass
class VideoSegment:
    """Represents a video segment with metadata."""
    start_time: float
    end_time: float
    quality_score: float
    engagement_score: float
    content_type: str
    description: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert VideoSegment to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'quality_score': self.quality_score,
            'engagement_score': self.engagement_score,
            'content_type': self.content_type,
            'description': self.description,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoSegment':
        """Create VideoSegment from dictionary."""
        return cls(
            start_time=data['start_time'],
            end_time=data['end_time'],
            quality_score=data['quality_score'],
            engagement_score=data['engagement_score'],
            content_type=data['content_type'],
            description=data['description'],
            confidence=data['confidence']
        )


class ContentAnalyzer:
    """
    Analyzes video content to identify optimal segments for short videos.
    Enhanced with object detection and AI-powered reframing capabilities.
    """
    
    def __init__(self, ollama_client=None, enable_object_detection=True, enable_ai_reframing=True):
        """Initialize content analyzer."""
        self.logger = logging.getLogger(__name__)
        self.ollama_client = ollama_client
        
        # Initialize specialized analyzers
        self.prompt_analyzer = PromptBasedAnalyzer(ollama_client)
        self.segment_generator = ComprehensiveSegmentGenerator()
        self.boundary_adjuster = SpeechBoundaryAdjuster(self.logger)
        
        # Initialize object detection and reframing if enabled
        self.enable_object_detection = enable_object_detection
        self.enable_ai_reframing = enable_ai_reframing
        
        if enable_object_detection:
            self.object_detector = IntelligentObjectDetector()
            # Import and initialize object-aware zoom processor with base processor
            from ..smart_zoom.object_aware_zoom import ObjectAwareZoomProcessor
            from ..smart_zoom.smart_zoom_processor import SmartZoomProcessor
            base_zoom_processor = SmartZoomProcessor()
            self.object_aware_zoom_processor = ObjectAwareZoomProcessor(
                base_zoom_processor=base_zoom_processor,
                object_detector=self.object_detector
            )
            self.logger.info("Object detection and object-aware zoom enabled")
        else:
            self.object_detector = None
            self.object_aware_zoom_processor = None
            
        if enable_ai_reframing:
            self.ai_reframer = AIReframingProcessor()
            self.logger.info("AI reframing enabled")
        else:
            self.ai_reframer = None


    def _enhance_segments_with_celebrity_scores(self,
                                                segments: List[Dict],
                                                celebrity_index_path: str = None,
                                                celebrity_index_data: Dict = None,
                                                min_coverage_threshold: float = 0.01) -> List[Dict]:
        """Enhance segments with celebrity coverage and score metadata."""
        self.logger.info(f"Enhancing {len(segments)} segments with celebrity scores...")
        try:
            if celebrity_index_data:
                appearances_per_actor, actor_conf = celebrity_index_data
            elif celebrity_index_path:
                if not hasattr(self, '_celebrity_index_cache'):
                    self._celebrity_index_cache = {}
                if celebrity_index_path not in self._celebrity_index_cache:
                    self._celebrity_index_cache[celebrity_index_path] = load_celebrity_index(celebrity_index_path)
                appearances_per_actor, actor_conf = self._celebrity_index_cache[celebrity_index_path]
            else:
                return segments

            for seg in segments:
                per_actor = actor_coverage_for_segment(
                    seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf
                )
                if per_actor:
                    score = compute_celebrity_score(per_actor)
                    seg['celebrity_score'] = float(score)
                    self.logger.info(f"Segment {seg['start_time']}-{seg['end_time']}s celebrity score: {seg['celebrity_score']:.4f}")
                    seg['celebrity_actors'] = sorted(per_actor.items(), key=lambda x: x[1]['coverage'], reverse=True)
                    seg['has_celebrity'] = seg['celebrity_score'] >= min_coverage_threshold
                    seg['top_celebrity'] = seg['celebrity_actors'][0][0] if seg['celebrity_actors'] else None
                else:
                    seg['celebrity_score'] = 0.0
                    seg['celebrity_actors'] = []
                    seg['has_celebrity'] = False
                    seg['top_celebrity'] = None

            self.logger.info(f"Enhanced {len(segments)} segments with celebrity scores (path={celebrity_index_path})")
            return segments

        except Exception as e:
            self.logger.warning(f"Failed to enhance segments with celebrity scores: {e}")
            return segments


    def _calculate_combined_score(self, segment: Dict) -> float:
        """Calculate combined score from prompt match, object detection and celebrity presence."""
        self.logger.info(f"Calculating combined score for segment {segment.get('start_time', 0)}-{segment.get('end_time', 0)}s")
        prompt_score = segment.get('prompt_match_score', 0.0)
        object_relevance = segment.get('object_relevance_score', 0.0)
        object_confidence = segment.get('object_confidence_score', 0.0)
        object_prompt_match = segment.get('object_prompt_match_score', 0.0)
        celebrity_score = segment.get('celebrity_score', 0.0)
        object_score_from_segment = segment.get('object_score', 0.0)

        # Weighted combination — add celebrity weight (0.25) so actor-presence matters
        base_score = prompt_score * 0.25
        object_score = (object_relevance * 0.3 + object_confidence * 0.2 + object_prompt_match * 0.4) * 0.4
        celeb_score = celebrity_score * 0.25

        # High priority for segments with reference-matched objects
        object_reference_bonus = 0.0
        if segment.get('has_object') and object_score_from_segment > 0:
            object_reference_bonus = 0.8  # High boost for reference-matched object segments

        # Bonus for segments with multiple relevant objects/actors
        objects_detected = segment.get('objects_detected', 0)
        unique_classes = segment.get('unique_object_classes', 0)
        object_bonus = min(0.1, (objects_detected * 0.03) + (unique_classes * 0.05))

        combined = base_score + object_score + celeb_score + object_reference_bonus + object_bonus
        return min(1.0, combined)
    #     """
    #     Analyzes video content to identify optimal segments for short videos.
    #     Enhanced with object detection and AI-powered reframing capabilities.
    #     """
    
    # def __init__(self, ollama_client=None, enable_object_detection=True, enable_ai_reframing=True):
    #     """Initialize content analyzer."""
    #     self.logger = logging.getLogger(__name__)
    #     self.ollama_client = ollama_client
        
    #     # Initialize specialized analyzers
    #     self.prompt_analyzer = PromptBasedAnalyzer(ollama_client)
    #     self.segment_generator = ComprehensiveSegmentGenerator()
    #     self.boundary_adjuster = SpeechBoundaryAdjuster(self.logger)
        
    #     # Initialize object detection and reframing if enabled
    #     self.enable_object_detection = enable_object_detection
    #     self.enable_ai_reframing = enable_ai_reframing
        
    #     if enable_object_detection:
    #         self.object_detector = IntelligentObjectDetector()
    #         # Import and initialize object-aware zoom processor with base processor
    #         from ..smart_zoom.object_aware_zoom import ObjectAwareZoomProcessor
    #         from ..smart_zoom.smart_zoom_processor import SmartZoomProcessor
    #         base_zoom_processor = SmartZoomProcessor()
    #         self.object_aware_zoom_processor = ObjectAwareZoomProcessor(
    #             base_zoom_processor=base_zoom_processor,
    #             object_detector=self.object_detector
    #         )
    #         self.logger.info("Object detection and object-aware zoom enabled")
    #     else:
    #         self.object_detector = None
    #         self.object_aware_zoom_processor = None
            
    #     if enable_ai_reframing:
    #         self.ai_reframer = AIReframingProcessor()
    #         self.logger.info("AI reframing enabled")
    #     else:
    #         self.ai_reframer = None
    
    async def analyze_with_user_prompt(self,
                                     user_prompt: str,
                                     video_path: str,
                                     video_info: Dict,
                                     audio_analysis: Dict,
                                     scene_analysis: Dict,
                                     vision_analysis: Optional[Dict] = None,
                                     target_duration: Tuple[int, int] = (15, 60),
                                     max_shorts: int = 10,
                                     ai_reframe: bool = False,
                                     llm_provider: str = "ollama",
                                     celebrity_index_path: str = None) -> Dict:
        """
        Analyze video content based on user prompt for theme-specific short creation.
        Enhanced with object detection, AI reframing, and contextual understanding.
        
        Args:
            user_prompt: User's prompt describing desired content (e.g., "goals in basketball", "comedy shorts")
            video_path: Path to the video file
            video_info: Video metadata
            audio_analysis: Audio analysis results
            scene_analysis: Scene break analysis
            vision_analysis: Vision analysis results (optional)
            target_duration: Target duration range (min, max) in seconds
            max_shorts: Maximum number of shorts to generate
            ai_reframe: Whether to enable AI-powered reframing
            llm_provider: LLM provider to use ('openai' or 'ollama')
            
        Returns:
            Prompt-based analysis results with selected segments and reframing data
        """
        try:
            self.logger.info(f"Analyzing video with user prompt: '{user_prompt}' (AI reframe: {ai_reframe})")
            
            # PHASE 1 ENHANCEMENT: Content Overview Analysis
            self.logger.info("Performing content overview analysis for contextual understanding...")
            content_overview = await self._analyze_content_overview(
                audio_analysis, vision_analysis, video_info
            )
            
            # PHASE 1 ENHANCEMENT: Enhanced Intent Analysis
            self.logger.info("Performing enhanced user intent analysis...")
            intent_analysis = await self._analyze_user_intent_comprehensive(
                user_prompt, content_overview, video_info
            )
            
            # Step 1: Object Detection Analysis (if enabled)
            object_detection_results = {}
            if self.enable_object_detection and self.object_detector:
                self.logger.info("Performing object detection analysis...")
                
                # Analyze prompt for object detection strategy (with fallback for empty prompts)
                effective_prompt = user_prompt if user_prompt and user_prompt.strip() else "detect all objects in the scene"
                prompt_analysis = await self.object_detector.analyze_prompt_for_objects(
                    effective_prompt, self.ollama_client
                )
                
                # Perform object detection on video
                object_detection_results = await self.object_detector.detect_objects_in_video(
                    video_path, prompt_analysis, sample_rate=1.0
                )
                
                self.logger.info(f"Object detection completed: {object_detection_results.get('total_detections', 0)} objects detected")
            
            # Step 2: Generate ALL possible candidate segments (or skip if actor-only)
            # OPTIMIZATION: Detect actor-only requests early to skip expensive candidate generation
            actor_only = False
            if celebrity_index_path and user_prompt:
                try:
                    appearances_per_actor, _ = load_celebrity_index(celebrity_index_path)
                    prompt_lower = user_prompt.lower()
                    for actor in appearances_per_actor.keys():
                        if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
                            actor_only = True
                            break
                    if actor_only:
                        self.logger.info(f"🎯 DETECTED ACTOR-ONLY REQUEST - will skip comprehensive candidate generation and extract directly from precomputed timestamps")
                except Exception as e:
                    self.logger.warning(f"Could not inspect celebrity index for actor-only detection: {e}")

            # CRITICAL OPTIMIZATION: For actor-only requests, skip expensive candidate segment generation
            # The prompt analyzer will extract segments directly from precomputed actor timestamps
            if actor_only:
                self.logger.info("⚡ ACTOR-ONLY MODE: Skipping comprehensive candidate generation for efficiency")
                all_candidates = []  # Empty list - prompt_analyzer will generate segments from precomputed timestamps
            else:
                self.logger.info("Generating comprehensive candidate segments...")
                all_candidates = self.segment_generator.generate_all_possible_segments(
                    video_info=video_info,
                    audio_analysis=audio_analysis,
                    scene_analysis=scene_analysis,
                    target_duration=target_duration,
                    max_total_segments=300,  # Increased for comprehensive analysis
                    celebrity_index_path=celebrity_index_path,
                    actor_only=actor_only
                )
            
            # DETAILED DEBUG: Log candidate generation results
            self.logger.info(f"🔍 CANDIDATE SEGMENT GENERATION DEBUG:")
            self.logger.info(f"📊 Video duration: {video_info.get('duration', 'unknown')}s")
            self.logger.info(f"📊 Target duration: {target_duration}")
            self.logger.info(f"📊 Audio segments: {len(audio_analysis.get('transcription', {}).get('segments', []))}")
            self.logger.info(f"📊 Scene breaks: {len(scene_analysis.get('scene_breaks', []))}")
            self.logger.info(f"📊 Generated candidates: {len(all_candidates)}")
            
            if len(all_candidates) == 0:
                self.logger.warning("⚠️ NO CANDIDATE SEGMENTS GENERATED!")
                self.logger.warning("🔍 Debugging segment generation failure:")
                self.logger.warning(f"  - Video info: {video_info}")
                self.logger.warning(f"  - Audio transcription segments: {len(audio_analysis.get('transcription', {}).get('segments', []))}")
                self.logger.warning(f"  - Scene analysis: {scene_analysis.get('combined_breaks', [])}")
                
                # Log first few audio segments if available
                audio_segments = audio_analysis.get('transcription', {}).get('segments', [])
                if audio_segments:
                    self.logger.warning(f"  - First 3 audio segments:")
                    for i, seg in enumerate(audio_segments[:3]):
                        self.logger.warning(f"    {i}: {seg.get('start', 'no_start')}-{seg.get('end', 'no_end')}s: '{seg.get('text', 'no_text')[:50]}...'")
                else:
                    self.logger.warning(f"  - No audio segments found in transcription")
            else:
                self.logger.info(f"📊 Candidate segments generated successfully:")
                for i, candidate in enumerate(all_candidates[:5]):  # Show first 5
                    self.logger.info(f"  Candidate {i}: {candidate.get('start_time', 'no_start'):.1f}s-{candidate.get('end_time', 'no_end'):.1f}s, text: '{candidate.get('segment_text', 'no_text')[:50]}...'")
                if len(all_candidates) > 5:
                    self.logger.info(f"  ... and {len(all_candidates) - 5} more candidates")
            
            self.logger.info(f"Generated {len(all_candidates)} comprehensive candidate segments")
            
            # Step 3: Analyze segments with user prompt (ENHANCED with context and object detection)
            self.logger.info("Performing context-aware prompt-based segment analysis...")
            # Pass celebrity_index_path to prompt analyzer via video_info
            video_info_with_celebrity = dict(video_info) if video_info else {}
            if celebrity_index_path:
                video_info_with_celebrity['celebrity_index_path'] = celebrity_index_path
            prompt_results = await self.prompt_analyzer.analyze_with_prompt(
                user_prompt=user_prompt,
                audio_analysis=audio_analysis,
                vision_analysis=vision_analysis,
                scene_analysis=scene_analysis,
                video_info=video_info_with_celebrity,
                candidate_segments=all_candidates,
                object_detection_results=object_detection_results,
                content_overview=content_overview,  # ENHANCED: Pass content overview
                intent_analysis=intent_analysis,    # ENHANCED: Pass intent analysis
                target_duration=target_duration,
                llm_provider=llm_provider
            )
            # Debug raw prompt_results
            self.logger.debug(f"DEBUG prompt_results: {prompt_results}")
            if isinstance(prompt_results, dict):
                status = prompt_results.get('status')
                error = prompt_results.get('error')
                self.logger.debug(f"Prompt-based analysis status: {status}, error: {error}")
            else:
                self.logger.error(f"prompt_results is not dict: {type(prompt_results)}")
            
            # Step 4: AI Reframing Analysis (if enabled)
            reframing_data = {}
            if ai_reframe and self.enable_ai_reframing and self.ai_reframer:
                self.logger.info("Performing AI reframing analysis...")
                
                # Analyze reframing strategy
                reframing_strategy = await self.ai_reframer.analyze_reframing_strategy(
                    user_prompt, object_detection_results, video_info, self.ollama_client
                )
                
                # Generate reframing plan
                if prompt_results['status'] == 'success':
                    reframing_plan = await self.ai_reframer.generate_reframing_plan(
                        reframing_strategy, object_detection_results, video_info
                    )
                    
                    reframing_data = {
                        "strategy": reframing_strategy,
                        "framing_decisions": reframing_plan,
                        "enabled": True
                    }
                    
                    self.logger.info(f"AI reframing plan generated with {len(reframing_plan)} framing decisions")
            
            # Defensive validation: ensure prompt_results is a proper dictionary
            if not isinstance(prompt_results, dict):
                self.logger.error(f"CRITICAL: prompt_results is not a dictionary, got {type(prompt_results)}: {prompt_results}")
                # Force it to be a proper failed result dictionary
                prompt_results = {
                    'status': 'error', 
                    'error': f'Invalid prompt results format: got {type(prompt_results)} instead of dict', 
                    'segments': [],
                    'prompt_analysis': {}
                }
            
            # Ensure all required keys exist with proper types
            if 'status' not in prompt_results or not isinstance(prompt_results['status'], str):
                prompt_results['status'] = 'error'
            if 'segments' not in prompt_results or not isinstance(prompt_results['segments'], list):
                prompt_results['segments'] = []
            if 'prompt_analysis' not in prompt_results or not isinstance(prompt_results['prompt_analysis'], dict):
                prompt_results['prompt_analysis'] = {}
                
            # Log the prompt_results structure for debugging
            self.logger.debug(f"Prompt results keys: {list(prompt_results.keys())}")
            self.logger.debug(f"Prompt results status: {prompt_results.get('status', 'unknown')}")
            
            # Step 5: Enhanced Phase 1 + Phase 2 Analysis Pipeline
            if prompt_results.get('status') == 'success':
                # Handle both regular segments and climax_segments keys
                prompt_segments = prompt_results.get('segments', prompt_results.get('climax_segments', []))
                
                # Ensure prompt_segments is a list, not something else
                if not isinstance(prompt_segments, list):
                    self.logger.warning(f"prompt_segments is not a list, got {type(prompt_segments)}: {prompt_segments}")
                    prompt_segments = []
                
                # Include object reference segments if they exist (ensure they are always considered)
                object_reference_segments = [seg for seg in all_candidates if seg.get('has_object') and seg.get('object_score', 0) > 0]
                for seg in object_reference_segments:
                    if seg not in prompt_segments:
                        prompt_segments.append(seg)
                        self.logger.info(f"Added object reference segment: {seg.get('start_time', 0)}-{seg.get('end_time', 0)}s")
                
                self.logger.info(f"Found {len(prompt_segments)} prompt-matched segments")
                
                # Log segment details with actor verification info
                if prompt_segments and len(prompt_segments) > 0:
                    actor_info = ""
                    if prompt_segments[0].get('actor_focus'):
                        actor_info = f" (Actor: {prompt_segments[0].get('actor_focus')})"
                    
                    self.logger.info(f"📊 Prompt-matched segment coverage{actor_info}:")
                    for i, seg in enumerate(prompt_segments[:10]):  # Log first 10
                        appearance_ts = seg.get('appearance_timestamp_sec', 'N/A')
                        start = seg.get('start_time', 'N/A')
                        end = seg.get('end_time', 'N/A')
                        source = seg.get('source', 'unknown')
                        self.logger.info(
                            f"  [{i+1}] {start:.1f}s-{end:.1f}s (appearance at {appearance_ts}s, source: {source})"
                        )
                    if len(prompt_segments) > 10:
                        self.logger.info(f"  ... and {len(prompt_segments) - 10} more segments")
                
                # Phase 2: Multi-pass analysis pipeline (if available)
                if self.ollama_client and content_overview and intent_analysis and len(prompt_segments) > 0:
                    self.logger.info("Initiating Phase 2 multi-pass analysis pipeline...")
                    
                    phase2_result = await self._phase2_multipass_analysis(
                        initial_segments=prompt_segments,
                        user_prompt=user_prompt,
                        comprehensive_segments=all_candidates,
                        audio_analysis=audio_analysis,
                        vision_analysis=vision_analysis,
                        scene_analysis=scene_analysis,
                        content_overview=content_overview,
                        intent_analysis=intent_analysis,
                        target_duration=target_duration,
                        video_info=video_info,
                        max_shorts=max_shorts
                    )
                    
                    if phase2_result['status'] == 'success':
                        # Apply final object detection and reframing
                        final_segments = self._select_final_prompt_segments(
                            phase2_result['segments'],
                            audio_analysis=audio_analysis,
                            user_prompt=user_prompt,
                            object_detection_results=object_detection_results,
                            reframing_data=reframing_data,
                            max_shorts=max_shorts,
                            target_duration=target_duration,
                            celebrity_index_path=celebrity_index_path,
                            actor_only=actor_only
                        )
                        
                        # Safely extract prompt_analysis with defensive check
                        self.logger.info("No of prompt matched segments : " + str(len(prompt_segments)))
                        safe_prompt_analysis = {}
                        if isinstance(prompt_results, dict) and 'prompt_analysis' in prompt_results:
                            safe_prompt_analysis = prompt_results['prompt_analysis']
                            if not isinstance(safe_prompt_analysis, dict):
                                safe_prompt_analysis = {}
                        
                        return {
                            'status': 'success',
                            'user_prompt': user_prompt,
                            'analysis_method': 'phase2_multipass_enhanced',
                            'total_candidates_analyzed': len(all_candidates),
                            'prompt_matched_segments': len(prompt_segments),
                            'phase2_enhanced_segments': len(phase2_result['segments']),
                            'final_selected_segments': len(final_segments),
                            'segments': final_segments,
                            'content_overview': content_overview,
                            'intent_analysis': intent_analysis,
                            'multipass_details': phase2_result.get('multipass_details', {}),
                            'prompt_analysis': safe_prompt_analysis,
                            'object_detection_results': object_detection_results,
                            'reframing_data': reframing_data,
                            'comprehensive_coverage': True,
                            'ai_reframe_enabled': ai_reframe,
                            'context_confidence': phase2_result.get('context_confidence', 0.7),
                            'enhancement_level': 'phase2_multipass'
                        }
                    else:
                        self.logger.warning("Phase 2 analysis failed, falling back to Phase 1 results")
                
                # Fallback to Phase 1 analysis
                try:
                    final_segments = self._select_final_prompt_segments(
                        prompt_segments,
                        audio_analysis=audio_analysis,
                        user_prompt=user_prompt,
                        object_detection_results=object_detection_results,
                        reframing_data=reframing_data,
                        max_shorts=max_shorts,
                        target_duration=target_duration,
                        celebrity_index_path=celebrity_index_path,
                        actor_only=actor_only
                    )
                except Exception as e:
                    self.logger.error(f"Error in _select_final_prompt_segments: {e}")
                    raise e
                
                # Defensive check for intent_analysis before using it
                if not isinstance(intent_analysis, dict):
                    self.logger.warning(f"intent_analysis is not a dict, got {type(intent_analysis)}: {intent_analysis}")
                    intent_analysis = {}
                
                # Defensive check for content_overview before using it
                if not isinstance(content_overview, dict):
                    self.logger.warning(f"content_overview is not a dict, got {type(content_overview)}: {content_overview}")
                    content_overview = {}
                
                # Defensive check for object_detection_results before using it
                if not isinstance(object_detection_results, dict):
                    self.logger.warning(f"object_detection_results is not a dict, got {type(object_detection_results)}: {object_detection_results}")
                    object_detection_results = {}
                
                # Defensive check for reframing_data before using it
                if not isinstance(reframing_data, dict):
                    self.logger.warning(f"reframing_data is not a dict, got {type(reframing_data)}: {reframing_data}")
                    reframing_data = {}
                
                # Safely extract prompt_analysis with defensive check
                safe_prompt_analysis = {}
                if isinstance(prompt_results, dict) and 'prompt_analysis' in prompt_results:
                    safe_prompt_analysis = prompt_results['prompt_analysis']
                    if not isinstance(safe_prompt_analysis, dict):
                        safe_prompt_analysis = {}
                
                return {
                    'status': 'success',
                    'user_prompt': user_prompt,
                    'analysis_method': 'enhanced_contextual_with_objects',
                    'total_candidates_analyzed': len(all_candidates),
                    'prompt_matched_segments': len(prompt_segments),
                    'final_selected_segments': len(final_segments),
                    'segments': final_segments,
                    'content_overview': content_overview,  # ENHANCED: Include content overview
                    'intent_analysis': intent_analysis,    # ENHANCED: Include intent analysis
                    'prompt_analysis': safe_prompt_analysis,
                    'object_detection_results': object_detection_results,
                    'reframing_data': reframing_data,
                    'comprehensive_coverage': True,
                    'ai_reframe_enabled': ai_reframe,
                    'context_confidence': intent_analysis.get('confidence_assessment', {}).get('overall_confidence', 0.5),
                    'enhancement_level': 'phase1_contextual'
                }
            else:
                # Prompt analysis failed, log the error and fall back
                error_msg = prompt_results.get('error', 'Unknown error in prompt analysis')
                self.logger.warning(f"Prompt analysis failed with error: {error_msg}")
                self.logger.warning("Falling back to standard analysis")
                return await self._fallback_to_standard_analysis(
                    all_candidates, video_info, audio_analysis, scene_analysis, 
                    vision_analysis, target_duration, max_shorts
                )
                
        except Exception as e:
            # Handle the specific theme_templates corruption error
            if "'list' object has no attribute 'items'" in str(e):
                self.logger.error(f"CRITICAL: theme_templates corruption detected in ContentAnalyzer: {e}")
                self.logger.warning("Attempting recovery by reinitializing PromptBasedAnalyzer theme_templates...")
                
                # Force recovery in the PromptBasedAnalyzer
                try:
                    if hasattr(self.prompt_analyzer, '_reinitialize_theme_templates'):
                        self.prompt_analyzer._reinitialize_theme_templates()
                        self.logger.info("theme_templates recovery successful, retrying analysis...")
                        
                        # Retry the analysis once after recovery
                        prompt_results = await self.prompt_analyzer.analyze_with_prompt(
                            user_prompt=user_prompt,
                            audio_analysis=audio_analysis,
                            vision_analysis=vision_analysis,
                            scene_analysis=scene_analysis,
                            video_info=video_info,
                            candidate_segments=all_candidates,
                            object_detection_results=object_detection_results,
                            content_overview=content_overview,
                            intent_analysis=intent_analysis,
                            llm_provider=llm_provider
                        )
                        
                        if prompt_results['status'] == 'success':
                            self.logger.info("Analysis retry successful after theme_templates recovery")
                            # Continue with the normal flow
                            prompt_segments = prompt_results.get('segments', [])
                            final_segments = self._select_final_prompt_segments(
                                prompt_segments,
                                audio_analysis=audio_analysis,
                                user_prompt=user_prompt,
                                object_detection_results=object_detection_results,
                                reframing_data={},  # Skip reframing on retry
                                max_shorts=max_shorts,
                                target_duration=target_duration,
                                celebrity_index_path=celebrity_index_path
                            )
                            
                            return {
                                'status': 'success',
                                'user_prompt': user_prompt,
                                'analysis_method': 'recovered_after_corruption',
                                'total_candidates_analyzed': len(all_candidates),
                                'prompt_matched_segments': len(prompt_segments),
                                'final_selected_segments': len(final_segments),
                                'segments': final_segments,
                                'recovery_applied': True,
                                'enhancement_level': 'recovered_basic'
                            }
                        else:
                            self.logger.warning("Analysis retry failed after recovery, falling back to standard analysis")
                    
                except Exception as retry_error:
                    self.logger.error(f"Recovery retry failed: {retry_error}")
                
            self.logger.error(f"Error in prompt-based analysis2: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to standard analysis - CRITICAL FIX: preserve all_candidates
            return await self._fallback_to_standard_analysis(
                all_candidates, video_info, audio_analysis, scene_analysis, 
                vision_analysis, target_duration, max_shorts
            )
    
    def _select_final_prompt_segments(self,
                                    prompt_segments: List[Dict],
                                    audio_analysis: Dict = None,
                                    user_prompt: str = "",
                                    object_detection_results: Dict = None,
                                    reframing_data: Dict = None,
                                    max_shorts: int = 10,
                                    target_duration: Tuple[int, int] = (15, 60),
                                    celebrity_index_path: str = None,
                                    actor_only: bool = False) -> List[Dict]:
        """
        Select final segments from prompt-matched segments with quality and diversity.
        Enhanced with object detection and AI reframing data.
        
        Args:
            prompt_segments: Segments matched by prompt analysis
            audio_analysis: Audio analysis with transcription for boundary adjustment
            user_prompt: User's search prompt for context
            object_detection_results: Object detection results
            reframing_data: AI reframing data
            max_shorts: Maximum number of shorts
            target_duration: Target duration range
            celebrity_index_path: Optional path to celebrity result JSON for per-segment scoring
            
        Returns:
            Final selected segments with enhanced metadata
        """
        if not prompt_segments:
            return []

        # Enhance segments with celebrity scores (if available) before object scoring
        if celebrity_index_path:
            try:
                prompt_segments = self._enhance_segments_with_celebrity_scores(prompt_segments, celebrity_index_path=celebrity_index_path)
                self.logger.info("No of Enhanced prompt segments with celebrity data before final selection: " + str(len(prompt_segments)))
            except Exception as e:
                self.logger.warning(f"Failed to enhance prompt segments with celebrity data: {e}")

        # If user explicitly requested actor(s), enforce strict overlap with celebrity timestamps
        actor_matches = []
        if celebrity_index_path and user_prompt:
            try:
                prompt_lower = user_prompt.lower()
                appearances_per_actor, actor_conf = load_celebrity_index(celebrity_index_path)
                for actor in appearances_per_actor.keys():
                    if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
                        actor_matches.append(actor)

                if actor_matches:
                    self.logger.info(f"Actor(s) requested in prompt: {actor_matches}. Enforcing overlap with celebrity timestamps.")
                    self.logger.info(f"No of prompt_segments for requested actors before filtering:{len(prompt_segments)}")  
                    try:
                        from ..face_insights.celebrity_index import actor_coverage_for_segment
                    except Exception:
                        from src.face_insights.celebrity_index import actor_coverage_for_segment

                    # self.logger.info(f"No of prompt_segments for requested actors before filtering:{len(prompt_segments)}")    

                    filtered = []
                    for seg in prompt_segments:
                        per_actor_cov = actor_coverage_for_segment(seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf)
                        # Accept segment if any requested actor has coverage > 0
                        if any(a for a in per_actor_cov.keys() if a.lower() in [am.lower() for am in actor_matches]):
                            # Boost prompt score to prefer these
                            seg['prompt_match_score'] = max(seg.get('prompt_match_score', 0.0), 0.9)
                            filtered.append(seg)

                    if filtered:
                        prompt_segments = filtered
                        self.logger.info(f"Strictly filtered prompt_segments to {len(prompt_segments)} segments overlapping requested actors")
                    else:
                        self.logger.warning("Strict actor overlap filtering found no segments; falling back to non-strict actor-aware selection")
            except Exception as e:
                self.logger.warning(f"Could not enforce strict actor overlap filtering: {e}")

        # If the user explicitly requested an actor, filter prompt_segments to those with celebrity appearances
        actor_matches = []
        if celebrity_index_path and user_prompt:
            try:
                appearances_per_actor, _ = load_celebrity_index(celebrity_index_path)
                prompt_lower = user_prompt.lower()
                for actor in appearances_per_actor.keys():
                    if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
                        actor_matches.append(actor)
                if actor_matches:
                    self.logger.info(f"Actor-only prompt detected for: {actor_matches}. Filtering to celebrity-overlapping segments.")
                    filtered = []
                    for seg in prompt_segments:
                        ca = seg.get('celebrity_actors', [])
                        actors_in_seg = []
                        if isinstance(ca, (list, tuple)):
                            for a in ca:
                                if isinstance(a, (list, tuple)) and len(a) > 0:
                                    actors_in_seg.append(str(a[0]).lower())
                                elif isinstance(a, dict):
                                    actors_in_seg.append(str(a.get('name','')).lower())
                                else:
                                    actors_in_seg.append(str(a).lower())
                        elif isinstance(ca, dict):
                            actors_in_seg.append(str(ca.get('name','')).lower())

                        if seg.get('has_celebrity') and any(a.lower() in actors_in_seg for a in actor_matches):
                            # Boost prompt score so actor segments are favored in selection
                            seg['prompt_match_score'] = max(seg.get('prompt_match_score', 0.0), 0.9)
                            filtered.append(seg)

                    if filtered:
                        prompt_segments = filtered
                        self.logger.info(f"Filtered prompt segments to {len(prompt_segments)} actor-overlapping segments")

                        # If the caller explicitly requested actor-only clips, select up to max_shorts
                        # distinct appearance-based segments (do not aggressively dedupe overlapping timestamps)
                        if actor_only:
                            try:
                                # Build a list of unique appearance timestamps for requested actors
                                appearance_timestamps = []
                                for a in actor_matches:
                                    for t in appearances_per_actor.get(a, []):
                                        if t not in appearance_timestamps:
                                            appearance_timestamps.append(t)
                                appearance_timestamps = sorted(appearance_timestamps)

                                selected = []

                                # Warn if not enough unique appearances to satisfy requested count
                                if len(appearance_timestamps) < max_shorts:
                                    self.logger.warning(f"Not enough unique actor appearances ({len(appearance_timestamps)}) to satisfy max_shorts={max_shorts}; filling remaining with best actor-overlapping segments")

                                # For each appearance timestamp, try to pick the best segment that covers that timestamp
                                for t in appearance_timestamps:
                                    if len(selected) >= max_shorts:
                                        break

                                    # Candidate segments that include the timestamp
                                    candidates = [s for s in filtered if s['start_time'] <= float(t) <= s['end_time']]

                                    # If no candidate covers the timestamp exactly, pick nearest segments by center distance within a small window
                                    if not candidates:
                                        window = max(5.0, (target_duration[1] if target_duration else 30) / 2.0)
                                        def center(seg):
                                            return (seg['start_time'] + seg['end_time']) / 2.0
                                        candidates = [s for s in filtered if abs(center(s) - float(t)) <= window]

                                    if candidates:
                                        # Select best candidate by celebrity_score then prompt_match_score
                                        best = sorted(candidates, key=lambda x: (x.get('celebrity_score', 0.0), x.get('prompt_match_score', 0.0)), reverse=True)[0]
                                        if best not in selected:
                                            selected.append(best)

                                # If not enough unique appearances, fill remaining slots with other top actor-overlapping segments
                                if len(selected) < max_shorts:
                                    remaining = [s for s in sorted(filtered, key=lambda x: (x.get('celebrity_score', 0.0), x.get('prompt_match_score', 0.0)), reverse=True) if s not in selected]
                                    for s in remaining:
                                        if len(selected) >= max_shorts:
                                            break
                                        selected.append(s)

                                # Limit to requested count
                                selected = selected[:max_shorts]

                                # Adjust boundaries to natural speech breaks if transcription is available
                                if selected and audio_analysis and 'transcription' in audio_analysis:
                                    selected = self.boundary_adjuster.batch_adjust_segments(
                                        segments=selected,
                                        transcription=audio_analysis['transcription'],
                                        user_prompt=user_prompt
                                    )

                                self.logger.info(f"Actor-only mode: selected {len(selected)} actor-overlapping segments (up to max_shorts={max_shorts})")
                                return selected

                            except Exception as e:
                                self.logger.warning(f"Error while computing actor-only selection: {e}")
                                # Fallback to previous simple unique behavior if grouping failed
                                unique = []
                                seen = set()
                                for s in filtered:
                                    key = (round(s.get('start_time', 0), 2), round(s.get('end_time', 0), 2))
                                    if key not in seen:
                                        seen.add(key)
                                        unique.append(s)
                                unique.sort(key=lambda x: x.get('start_time', 0))

                                if unique and audio_analysis and 'transcription' in audio_analysis:
                                    unique = self.boundary_adjuster.batch_adjust_segments(
                                        segments=unique,
                                        transcription=audio_analysis['transcription'],
                                        user_prompt=user_prompt
                                    )

                                self.logger.info(f"Actor-only fallback: returning {len(unique)} actor-overlapping segments")
                                return unique

                    else:
                        self.logger.warning("No prompt-matched segments overlapped requested actor(s); continuing with unfiltered set")
            except Exception as e:
                self.logger.warning(f"Could not load celebrity index for actor filtering: {e}")

        # Enhance segments with object detection scores
        if object_detection_results:
            prompt_segments = self._enhance_segments_with_object_scores(
                prompt_segments, object_detection_results
            )
        
        # Ensure segments have a usable prompt_match_score (derive from contextual/heuristic scores when absent)
        prompt_segments = self._ensure_prompt_scores(prompt_segments)
        
        # CRITICAL: Check for object reference segments FIRST
        # These are segments marked by the segment generator as having reference_match.score
        object_reference_segments = [seg for seg in prompt_segments if seg.get('is_object_reference_segment', False)]
        
        if not object_reference_segments:
            # Fallback: check for segments with has_object and object_score
            object_reference_segments = [seg for seg in prompt_segments if seg.get('has_object') and seg.get('object_score', 0) > 0]
        
        if object_reference_segments:
            self.logger.info(f"🎯 Found {len(object_reference_segments)} object reference segments - will use EXCLUSIVELY")
        
        # CRITICAL: Check for object reference segments FIRST
        # These are segments marked by the segment generator as having reference_match.score
        object_reference_segments = [seg for seg in prompt_segments if seg.get('is_object_reference_segment', False)]
        
        if not object_reference_segments:
            # Fallback: check for segments with has_object and object_score
            object_reference_segments = [seg for seg in prompt_segments if seg.get('has_object') and seg.get('object_score', 0) > 0]
        
        if object_reference_segments:
            self.logger.info(f"🎯 Found {len(object_reference_segments)} object reference segments - will use EXCLUSIVELY")
        
        # Sort by combined score (prompt match + object relevance)
        sorted_segments = sorted(prompt_segments, 
                               key=lambda x: self._calculate_combined_score(x), 
                               reverse=True)
        
        # Apply quality filtering - SPECIAL HANDLING FOR OBJECT REFERENCE SEGMENTS
        min_quality_threshold = 0.05
        
        # CRITICAL: If we have object reference segments, use ONLY those - bypass all other filtering
        if object_reference_segments:
            self.logger.info(f"🎯 OBJECT REFERENCE MODE: Using ONLY {len(object_reference_segments)} object reference segments")
            # Sort object reference segments by start_time for consistency
            quality_filtered = sorted(object_reference_segments, key=lambda x: x.get('start_time', 0))
        elif actor_matches:
            # When user explicitly requested actor(s), prefer segments with celebrity coverage
            quality_filtered = [seg for seg in sorted_segments if seg.get('has_celebrity')]
            if not quality_filtered:
                # Fallback to low threshold if no celeb-overlapping segments are available
                quality_filtered = [seg for seg in sorted_segments if seg.get('prompt_match_score', 0) >= 0.01]
        else:
            quality_filtered = [
                seg for seg in sorted_segments 
                if seg.get('prompt_match_score', 0) >= min_quality_threshold
            ]
        
        # Log segment scores for debugging
        if sorted_segments:
            top_scores = [seg.get('prompt_match_score', 0) for seg in sorted_segments[:5]]
            self.logger.info(f"Top 5 segment scores: {top_scores}")
            self.logger.info(f"Quality filtered: {len(quality_filtered)}/{len(sorted_segments)} segments (threshold: {min_quality_threshold})")
        
        # If not enough high-quality segments, lower threshold
        if len(quality_filtered) < max_shorts // 2:
            min_quality_threshold = 0.01
            quality_filtered = [
                seg for seg in sorted_segments 
                if seg.get('prompt_match_score', 0) >= min_quality_threshold
            ]
            self.logger.info(f"Lowered threshold to {min_quality_threshold}, now have {len(quality_filtered)} segments")
        
        # Apply diversity filtering (avoid temporal clustering)
        # EXCEPTION: Skip diversity filtering for object reference segments - use them all
        if object_reference_segments and quality_filtered == object_reference_segments:
            # For object reference segments, use all of them up to max_shorts
            final_segments = quality_filtered[:max_shorts]
            self.logger.info(f"🎯 Object reference mode: using all {len(final_segments)} reference segments without diversity filtering")
        else:
            # Standard diversity filtering for other segment types
            final_segments = []
            used_time_ranges = []
            min_gap = 15.0  # Minimum 15 seconds between segments
            
            for segment in quality_filtered:
                if len(final_segments) >= max_shorts:
                    break
                
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Check for temporal conflicts
                conflict = False
                for used_start, used_end in used_time_ranges:
                    # Check if segments are too close
                    gap_start = min(abs(start_time - used_end), abs(end_time - used_start))
                    gap_end = min(abs(end_time - used_start), abs(start_time - used_end))
                    
                    if gap_start < min_gap or gap_end < min_gap:
                        # Check for actual overlap
                        if not (end_time <= used_start or start_time >= used_end):
                            conflict = True
                            break
                
                if not conflict:
                    final_segments.append(segment)
                    used_time_ranges.append((start_time, end_time))
        
        # If we don't have enough segments, add more with relaxed gap requirements
        # EXCEPTION: Skip this for object reference segments - they're already all included
        if len(final_segments) < max_shorts // 2 and not (object_reference_segments and quality_filtered == object_reference_segments):
            relaxed_gap = min_gap / 2
            
            for segment in quality_filtered:
                if segment in final_segments:
                    continue
                if len(final_segments) >= max_shorts:
                    break
                
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Check with relaxed gap
                conflict = False
                for used_start, used_end in used_time_ranges:
                    if not (end_time <= used_start or start_time >= used_end):
                        conflict = True
                        break
                
                if not conflict:
                    final_segments.append(segment)
                    used_time_ranges.append((start_time, end_time))
        
        # Sort final segments by start time
        final_segments.sort(key=lambda x: x['start_time'])
        
        # ENHANCEMENT: Adjust segment boundaries to natural speech breaks
        if final_segments and audio_analysis and 'transcription' in audio_analysis:
            self.logger.info(f"🎬 Adjusting {len(final_segments)} segment boundaries to natural speech breaks...")
            transcription = audio_analysis['transcription']
            adjusted_segments = self.boundary_adjuster.batch_adjust_segments(
                segments=final_segments,
                transcription=transcription,
                user_prompt=user_prompt
            )
            final_segments = adjusted_segments
            self.logger.info(f"✅ Boundary adjustment complete")
        
        # Enhanced logging for debugging
        self.logger.info(f"Final segment selection: {len(final_segments)} segments selected from {len(prompt_segments)} prompt matches")

        # If actor matches were requested, ensure final segments overlap requested actor timestamps
        if actor_matches:
            try:
                appearances_per_actor, actor_conf = load_celebrity_index(celebrity_index_path)
                try:
                    from ..face_insights.celebrity_index import actor_coverage_for_segment
                except Exception:
                    from src.face_insights.celebrity_index import actor_coverage_for_segment

                final_actor_filtered = []
                for seg in final_segments:
                    per_actor_cov = actor_coverage_for_segment(seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf)
                    if any(a for a in per_actor_cov.keys() if a.lower() in [am.lower() for am in actor_matches]):
                        final_actor_filtered.append(seg)

                if final_actor_filtered:
                    final_segments = final_actor_filtered
                    self.logger.info(f"Filtered final_segments to {len(final_segments)} actor-overlapping segments")
                else:
                    self.logger.warning("Final actor overlap check found no segments; keeping current final selection")
            except Exception as e:
                self.logger.warning(f"Could not perform final actor overlap filtering: {e}")

        # CRITICAL FIX: Add intelligent fallback when no segments are selected
        if len(final_segments) == 0 and len(sorted_segments) > 0:
            self.logger.warning("No segments passed quality filter. Implementing intelligent fallback...")
            
            # Fallback Strategy 1: Select highest-scoring segments regardless of threshold
            top_segments = sorted_segments[:max_shorts]
            
            # Fallback Strategy 2: Ensure we select the most engaging content
            # Score segments by engagement rather than just prompt matching
            for segment in top_segments:
                # Boost engagement score for fallback selection
                engagement_score = self._calculate_engagement_score(segment)
                segment['fallback_score'] = engagement_score
                segment['prompt_match_score'] = max(segment.get('prompt_match_score', 0), engagement_score)
            
            # Re-sort by enhanced scores
            top_segments.sort(key=lambda x: x.get('fallback_score', 0), reverse=True)
            
            # Apply temporal diversity for fallback segments
            fallback_segments = []
            used_ranges = []
            min_gap = 10.0  # Reduced gap for fallback
            
            for segment in top_segments:
                if len(fallback_segments) >= max_shorts:
                    break
                    
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Check for conflicts with reduced constraints
                conflict = False
                for used_start, used_end in used_ranges:
                    if not (end_time <= used_start + min_gap or start_time >= used_end - min_gap):
                        conflict = True
                        break
                
                if not conflict:
                    fallback_segments.append(segment)
                    used_ranges.append((start_time, end_time))
            
            final_segments = fallback_segments
            self.logger.info(f"Fallback selection applied: {len(final_segments)} engaging segments selected")
        
        if final_segments:
            avg_score = sum(seg.get('prompt_match_score', 0) for seg in final_segments) / len(final_segments)
            self.logger.info(f"Selected segments average score: {avg_score:.3f}")
            self.logger.info(f"Selected segments time ranges: {[(seg['start_time'], seg['end_time']) for seg in final_segments]}")
        
        return final_segments
    
    async def _fallback_to_standard_analysis(self,
                                          all_candidates: List[Dict],
                                          video_info: Dict,
                                          audio_analysis: Dict,
                                          scene_analysis: Dict,
                                          vision_analysis: Optional[Dict],
                                          target_duration: Tuple[int, int],
                                          max_shorts: int) -> Dict:
        """Fallback to standard segment analysis if prompt analysis fails."""
        try:
            # Use existing method or generate segments if candidates not available
            if not all_candidates:
                all_candidates = self.segment_generator.generate_all_possible_segments(
                    video_info=video_info,
                    audio_analysis=audio_analysis,
                    scene_analysis=scene_analysis,
                    target_duration=target_duration,
                    max_total_segments=200
                )
            
            # Score segments using standard approach
            scored_segments = []
            for segment in all_candidates:
                scored_segment = self._score_segment(segment, audio_analysis, None)
                scored_segments.append(scored_segment)
            
            # Apply vision enhancement if available
            if vision_analysis:
                scored_segments = self._enhance_segments_with_vision(scored_segments, vision_analysis)
            
            # Select final segments
            final_segments = self._select_diverse_content(scored_segments, max_shorts)
            
            return {
                'status': 'success',
                'analysis_method': 'standard_comprehensive_fallback',
                'total_candidates_analyzed': len(all_candidates),
                'final_selected_segments': len(final_segments),
                'segments': final_segments,
                'comprehensive_coverage': True,
                'fallback_reason': 'prompt_analysis_unavailable'
            }
            
        except Exception as e:
            self.logger.error(f"Fallback analysis also failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    def get_supported_themes(self) -> Dict[str, str]:
        """
        Get list of supported themes for prompt-based analysis.
        
        Returns:
            Dictionary of theme names and descriptions
        """
        try:
            return self.prompt_analyzer.get_supported_themes()
        except AttributeError as e:
            # Handle the specific 'list' object has no attribute 'items' error
            if "'list' object has no attribute 'items'" in str(e):
                self.logger.error(f"CRITICAL: theme_templates corruption detected in ContentAnalyzer: {e}")
                self.logger.warning("Attempting to recover by accessing PromptBasedAnalyzer recovery methods")
                
                # Force recovery in the PromptBasedAnalyzer
                if hasattr(self.prompt_analyzer, '_reinitialize_theme_templates'):
                    self.prompt_analyzer._reinitialize_theme_templates()
                    return self.prompt_analyzer.get_supported_themes()
                else:
                    # Fallback themes if recovery fails
                    return {
                        'general': 'Generally engaging and interesting content',
                        'action': 'High-energy, dynamic, action-packed moments',
                        'educational': 'Informative, instructional, educational content',
                        'emotional': 'Emotionally engaging, touching, inspiring moments',
                        'comedy': 'Humorous, entertaining, light-hearted moments'
                    }
            else:
                # Re-raise other AttributeErrors
                raise
        except Exception as e:
            # Catch any other exceptions and provide fallback
            self.logger.error(f"Error in get_supported_themes: {e}")
            return {
                'general': 'Generally engaging and interesting content',
                'action': 'High-energy, dynamic, action-packed moments', 
                'educational': 'Informative, instructional, educational content',
                'emotional': 'Emotionally engaging, touching, inspiring moments',
                'comedy': 'Humorous, entertaining, light-hearted moments'
            }
    
    def identify_short_segments(self, 
                              video_info: Dict,
                              audio_analysis: Dict,
                              ollama_analysis: Optional[Dict],
                              scene_analysis: Dict,
                              target_duration: Tuple[int, int] = (60, 120),
                              max_shorts: int = 10) -> List[Dict]:
        """
        Identify optimal segments for short videos with content diversity.
        
        Args:
            video_info: Video metadata
            audio_analysis: Audio analysis results
            ollama_analysis: AI analysis results (optional)
            scene_analysis: Scene break analysis
            target_duration: Target duration range (min, max) in seconds
            max_shorts: Maximum number of shorts to generate
            
        Returns:
            List of optimal segments with diverse content
        """
        try:
            min_duration, max_duration = target_duration
            self.logger.info(f"Identifying segments with target duration range: {min_duration}s - {max_duration}s")
            
            # Detect pre-roll content for filtering
            pre_roll_end = self._detect_pre_roll_content(audio_analysis, video_info['duration'])
            self.logger.info(f"Detected pre-roll content ending at {pre_roll_end:.2f}s")
            
            # Get all potential break points, filtering out pre-roll
            break_points = self._get_all_break_points(scene_analysis, audio_analysis, pre_roll_end)
            
            # ADD VISUAL SEGMENT DETECTION for silent/action content
            self.logger.info("Detecting visually interesting segments (including silent content)")
            visual_segments = self._identify_visual_segments(
                video_info, audio_analysis, scene_analysis, target_duration, max_shorts, pre_roll_end, []
            )
            
            # Generate candidate segments, avoiding pre-roll content
            candidates = self._generate_candidate_segments(
                break_points, 
                video_info['duration'], 
                target_duration,
                pre_roll_end
            )
            
            # MERGE VISUAL SEGMENTS with audio-based candidates
            if visual_segments:
                self.logger.info(f"Adding {len(visual_segments)} visual segments to {len(candidates)} audio-based candidates")
                candidates.extend(visual_segments)
                # Remove duplicates based on similar timing
                candidates = self._deduplicate_segments(candidates, time_threshold=5.0)
            
            # Score each segment using HYBRID approach (audio + visual)
            scored_segments = []
            for segment in candidates:
                # Use hybrid scoring if visual segments are available
                if visual_segments and any(vs.get('is_visual_priority') for vs in visual_segments):
                    score = self._score_hybrid_segments([segment], audio_analysis, scene_analysis)[0]
                else:
                    score = self._score_segment(segment, audio_analysis, ollama_analysis)
                scored_segments.append(score)
            
            # Sort by quality score
            scored_segments.sort(key=lambda x: x['quality_score'], reverse=True)
            
            # Remove overlapping segments
            non_overlapping = self._remove_overlapping_segments(scored_segments)
            
            # Apply content diversity selection
            diverse_segments = self._select_diverse_content(non_overlapping, max_shorts)
            
            # Final sort by start time
            diverse_segments.sort(key=lambda x: x['start_time'])
            
            return diverse_segments
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {e}")
            return []
    
    def _detect_pre_roll_content(self, audio_analysis: Dict, video_duration: float) -> float:
        """
        Detect pre-roll content (intro music, silence, generic greetings) at the beginning of video.
        
        Args:
            audio_analysis: Audio analysis results
            video_duration: Total video duration in seconds
            
        Returns:
            Timestamp where pre-roll content ends (0 if none detected)
        """
        # Default pre-roll threshold (10% of video or 30 seconds, whichever is less)
        pre_roll_threshold = min(video_duration * 0.1, 30.0)
        
        # Check for explicitly detected pre-roll in audio analysis
        likely_preroll = audio_analysis.get('zoom_analysis', {}).get('likely_preroll', [])
        if likely_preroll:
            # Find the end time of the last pre-roll segment
            last_preroll_end = max(segment['end'] for segment in likely_preroll)
            return last_preroll_end
        
        # Check for music segments at beginning
        music_segments = audio_analysis.get('zoom_analysis', {}).get('music_segments', [])
        for segment in music_segments:
            if segment['start'] < pre_roll_threshold and segment['end'] < pre_roll_threshold * 1.2:
                # Found music in the beginning, likely intro music
                return segment['end']
        
        # Check for silence periods at beginning
        silence_segments = audio_analysis.get('zoom_analysis', {}).get('silence_segments', [])
        for segment in silence_segments:
            if segment['start'] < pre_roll_threshold and segment['end'] < pre_roll_threshold * 1.2:
                # Found silence in the beginning, could be pre-roll
                return segment['end']
        
        # Check for standard introductory phrases in first segments
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        intro_phrases = [
            'welcome to', 'hello everyone', 'hi everyone', 'hey guys', 
            'welcome back', 'in this video', 'today we\'re going to'
        ]
        
        for segment in segments:
            if segment['start'] < pre_roll_threshold:
                text_lower = segment['text'].lower()
                if any(phrase in text_lower for phrase in intro_phrases):
                    # Found intro greeting, consider it pre-roll
                    return segment['end']
        
        # If no clear pre-roll detected, return 0
        return 0.0
    
    def _get_all_break_points(self, scene_analysis: Dict, audio_analysis: Dict, pre_roll_end: float = 0.0) -> List[float]:
        """
        Get all potential break points from various sources, filtering out pre-roll.
        
        Args:
            scene_analysis: Scene break analysis
            audio_analysis: Audio analysis results
            pre_roll_end: Timestamp where pre-roll content ends
            
        Returns:
            List of filtered break points
        """
        break_points = set()
        
        # Add scene breaks, filtering out pre-roll (VISUAL PRIORITY)
        for break_point in scene_analysis.get('combined_breaks', []):
            timestamp = break_point['timestamp']
            if timestamp > pre_roll_end:
                break_points.add(timestamp)
        
        # Add PURE VISUAL breaks (important for silent scenes)
        for break_point in scene_analysis.get('scene_breaks', []):
            timestamp = break_point['timestamp']
            if timestamp > pre_roll_end:
                break_points.add(timestamp)
                
        # Add audio breaks, filtering out pre-roll
        for break_point in audio_analysis.get('zoom_analysis', {}).get('recommended_cuts', []):
            timestamp = break_point['timestamp']
            if timestamp > pre_roll_end:
                break_points.add(timestamp)
        
        # Add silence periods, filtering out pre-roll (these could be visually interesting)
        transcription = audio_analysis.get('transcription', {})
        for silence in transcription.get('silence_periods', []):
            start, end = silence
            if end > pre_roll_end:
                # Only add if the silence period is after pre-roll
                break_points.add(max(start, pre_roll_end))  # Ensure break point is after pre-roll
                break_points.add(end)
                
        # ADD MOTION-BASED BREAKS for action sequences
        if 'motion_analysis' in scene_analysis:
            for motion_break in scene_analysis['motion_analysis'].get('high_motion_segments', []):
                if motion_break['start'] > pre_roll_end:
                    break_points.add(motion_break['start'])
                    break_points.add(motion_break['end'])
                break_points.add(end)
        
        return sorted(list(break_points))
    
    def _deduplicate_segments(self, segments: List[Dict], time_threshold: float = 5.0) -> List[Dict]:
        """
        Remove duplicate segments that are too similar in timing.
        
        Args:
            segments: List of segment dictionaries
            time_threshold: Maximum time difference (seconds) to consider segments as duplicates
            
        Returns:
            Deduplicated list of segments
        """
        if not segments:
            return segments
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])
        deduplicated = [sorted_segments[0]]
        
        for segment in sorted_segments[1:]:
            # Check if this segment is too similar to the last added segment
            last_segment = deduplicated[-1]
            
            start_diff = abs(segment['start_time'] - last_segment['start_time'])
            end_diff = abs(segment['end_time'] - last_segment['end_time'])
            
            # If both start and end times are within threshold, consider it a duplicate
            if start_diff <= time_threshold and end_diff <= time_threshold:
                # Keep the one with higher quality score
                if segment.get('quality_score', 0) > last_segment.get('quality_score', 0):
                    deduplicated[-1] = segment
                # Skip this segment (it's a duplicate)
            else:
                deduplicated.append(segment)
        
        self.logger.info(f"Deduplicated {len(sorted_segments)} segments to {len(deduplicated)} unique segments")
        return deduplicated
    
    def _generate_candidate_segments(self, 
                                   break_points: List[float],
                                   video_duration: float,
                                   target_duration: Tuple[int, int],
                                   pre_roll_end: float = 0.0) -> List[Dict]:
        """
        Generate candidate segments from break points, avoiding pre-roll content.
        
        Args:
            break_points: List of potential break points
            video_duration: Total video duration in seconds
            target_duration: Target duration range (min, max) in seconds
            pre_roll_end: Timestamp where pre-roll content ends
            
        Returns:
            List of candidate segments
        """
        min_duration, max_duration = target_duration
        candidates = []
        
        # Add video start (after pre-roll) and end as break points
        all_points = [max(0.0, pre_roll_end)] + break_points + [video_duration]
        all_points = sorted(set(all_points))
        
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                start_time = all_points[i]
                end_time = all_points[j]
                duration = end_time - start_time
                
                if min_duration <= duration <= max_duration:
                    candidates.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
        
        return candidates
    
    def _score_segment(self, 
                      segment: Dict,
                      audio_analysis: Dict,
                      ollama_analysis: Optional[Dict]) -> Dict:
        """Score a segment based on various factors."""
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # Initialize scores
        quality_score = 0.5
        engagement_score = 0.5
        content_type = 'general'
        confidence = 0.5
        
        # Analyze audio content in segment
        transcription = audio_analysis.get('transcription', {})
        segment_text = self._extract_segment_text(transcription, start_time, end_time)
        
        if segment_text:
            # Score based on text content
            quality_score += self._score_text_quality(segment_text)
            engagement_score += self._score_text_engagement(segment_text)
            content_type = self._classify_content_type(segment_text)
        
        # Use Ollama analysis if available
        if ollama_analysis:
            ollama_score = self._extract_ollama_score(ollama_analysis, start_time, end_time)
            if ollama_score:
                quality_score = (quality_score + ollama_score['quality']) / 2
                engagement_score = (engagement_score + ollama_score['engagement']) / 2
                confidence = ollama_score['confidence']
        
        # Apply duration penalty/bonus
        duration_score = self._score_duration(segment['duration'])
        quality_score *= duration_score
        
        # Ensure scores are in valid range
        quality_score = max(0.0, min(1.0, quality_score))
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'duration': segment['duration'],
            'quality_score': quality_score,
            'engagement_score': engagement_score,
            'content_type': content_type,
            'confidence': confidence,
            'segment_text': segment_text
        }
    
    def _extract_segment_text(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """Extract text from transcription for a specific time segment."""
        segments = transcription.get('segments', [])
        text_parts = []
        
        for segment in segments:
            if segment['start'] >= start_time and segment['end'] <= end_time:
                text_parts.append(segment['text'])
            elif segment['start'] < end_time and segment['end'] > start_time:
                # Partial overlap
                text_parts.append(segment['text'])
        
        return ' '.join(text_parts).strip()
    
    def _score_text_quality(self, text: str) -> float:
        """Score text quality based on completeness and coherence."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length factor
        word_count = len(text.split())
        if 10 <= word_count <= 100:  # Optimal range
            score += 0.3
        elif word_count >= 5:
            score += 0.1
        
        # Sentence completeness
        sentences = text.split('.')
        complete_sentences = [s.strip() for s in sentences if s.strip()]
        if len(complete_sentences) >= 1:
            score += 0.2
        
        # Avoid fragments
        if not text.endswith(('...', '-', ',')):
            score += 0.1
        
        # Check for questions or exclamations (engaging)
        if '?' in text or '!' in text:
            score += 0.1
        
        return score
    
    def _score_text_engagement(self, text: str) -> float:
        """Score text engagement potential with enhanced yoga/fitness awareness."""
        if not text:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        # General engagement keywords
        engagement_keywords = [
            'amazing', 'incredible', 'wow', 'look', 'see', 'watch',
            'important', 'key', 'secret', 'tip', 'hack', 'trick',
            'you', 'your', 'we', 'us', 'let\'s', 'remember'
        ]
        
        # Yoga/fitness specific engaging content
        yoga_action_words = [
            'breathe', 'inhale', 'exhale', 'stretch', 'hold', 'feel',
            'move', 'shift', 'come into', 'transition', 'flow',
            'balance', 'center', 'ground', 'lift', 'open', 'close'
        ]
        
        # Instructional engagement words
        instruction_words = [
            'now', 'next', 'here', 'let\'s', 'we\'re going to',
            'take a', 'bring your', 'find your', 'notice'
        ]
        
        # Count different types of engaging content
        keyword_count = sum(1 for word in engagement_keywords if word in text_lower)
        yoga_count = sum(1 for word in yoga_action_words if word in text_lower)
        instruction_count = sum(1 for word in instruction_words if word in text_lower)
        
        score += min(0.2, keyword_count * 0.05)
        score += min(0.3, yoga_count * 0.08)  # Higher weight for yoga actions
        score += min(0.2, instruction_count * 0.06)
        
        # Questions increase engagement
        question_count = text.count('?')
        score += min(0.15, question_count * 0.08)
        
        # Exclamations show energy
        exclamation_count = text.count('!')
        score += min(0.1, exclamation_count * 0.05)
        
        # Direct address (very important for yoga instruction)
        if any(word in text_lower for word in ['you', 'your', 'we', 'us', 'let\'s']):
            score += 0.15
        
        # Breathing cues (high value for yoga content)
        if any(word in text_lower for word in ['breathe', 'inhale', 'exhale', 'breath']):
            score += 0.1
        
        # Movement descriptions (valuable for demonstration)
        movement_words = ['move', 'shift', 'come', 'go', 'bring', 'take', 'lift', 'lower']
        if any(word in text_lower for word in movement_words):
            score += 0.1
        
        return min(1.0, score)
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type with yoga/fitness awareness."""
        if not text:
            return 'general'
        
        text_lower = text.lower()
        
        # Yoga/fitness specific classifications
        if any(word in text_lower for word in ['pose', 'asana', 'position', 'stretch', 'yoga']):
            if any(word in text_lower for word in ['breathe', 'inhale', 'exhale', 'hold']):
                return 'yoga_breathing'
            elif any(word in text_lower for word in ['move', 'flow', 'transition']):
                return 'yoga_movement'
            else:
                return 'yoga_instruction'
        
        # Movement/action content
        if any(word in text_lower for word in ['move', 'shift', 'come into', 'step', 'walk']):
            return 'movement'
        
        # Breathing/mindfulness content
        if any(word in text_lower for word in ['breathe', 'breath', 'inhale', 'exhale', 'mindful']):
            return 'breathing'
        
        # Educational content
        if any(word in text_lower for word in ['learn', 'teach', 'explain', 'how to', 'tutorial']):
            return 'educational'
        
        # Interactive/engagement content
        if '?' in text or any(word in text_lower for word in ['feel', 'notice', 'you', 'your']):
            return 'interactive'
        
        # Motivational content
        if any(word in text_lower for word in ['motivate', 'inspire', 'success', 'achieve', 'believe']):
            return 'motivational'
        
        # Introduction/welcome content
        if any(word in text_lower for word in ['welcome', 'hello', 'hi', 'today', 'begin']):
            return 'introduction'
        
        # Closing/conclusion content
        if any(word in text_lower for word in ['thank', 'thanks', 'end', 'finish', 'complete', 'namaste']):
            return 'conclusion'
        
        return 'instruction'
    
    def _extract_ollama_score(self, ollama_analysis: Dict, start_time: float, end_time: float) -> Optional[Dict]:
        """Extract Ollama analysis score for a segment."""
        if not ollama_analysis:
            return None
        
        # Try to find matching segment in Ollama analysis
        engagement_analysis = ollama_analysis.get('content_engagement', {})
        if 'engagement_analysis' in engagement_analysis:
            for segment in engagement_analysis['engagement_analysis']:
                if (segment['start_time'] <= start_time <= segment['end_time'] or
                    segment['start_time'] <= end_time <= segment['end_time']):
                    return {
                        'quality': segment.get('engagement_score', 0.5),
                        'engagement': segment.get('engagement_score', 0.5),
                        'confidence': 0.8
                    }
        
        return None
    
    def _score_duration(self, duration: float) -> float:
        """Score segment based on duration (60-120 seconds is optimal for engagement)."""
        if 60 <= duration <= 90:
            return 1.0  # Perfect length for engagement
        elif 90 < duration <= 120:
            return 0.95  # Very good length
        elif 45 <= duration < 60:
            return 0.8  # A bit short but acceptable
        elif 120 < duration <= 150:
            return 0.7  # A bit long but acceptable 
        else:
            return 0.7  # Too long
    
    def _remove_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
        """Remove overlapping segments, keeping the highest quality ones."""
        if not segments:
            return []
        
        # Sort by quality score (descending)
        segments.sort(key=lambda x: x['quality_score'], reverse=True)
        
        non_overlapping = []
        
        for segment in segments:
            overlap_found = False
            
            for existing in non_overlapping:
                if self._segments_overlap(segment, existing):
                    overlap_found = True
                    break
            
            if not overlap_found:
                non_overlapping.append(segment)
        
        return non_overlapping
    
    def _select_diverse_content(self, segments: List[Dict], max_shorts: int) -> List[Dict]:
        """
        Select diverse content to avoid repetitive shorts.
        Ensures variety in content types and timing distribution.
        """
        if not segments:
            return []
        
        if len(segments) <= max_shorts:
            return segments
        
        # Categorize segments by content type and characteristics
        categorized = self._categorize_segments_for_diversity(segments)
        
        # Select segments ensuring diversity
        selected = []
        content_type_counts = {}
        
        # First pass: select top segment from each content type
        for content_type, type_segments in categorized.items():
            if len(selected) < max_shorts and type_segments:
                best_segment = max(type_segments, key=lambda x: x['quality_score'])
                selected.append(best_segment)
                content_type_counts[content_type] = 1
                # Remove selected segment from all categories
                for cat_list in categorized.values():
                    if best_segment in cat_list:
                        cat_list.remove(best_segment)
        
        # Second pass: fill remaining slots with temporal diversity
        remaining_slots = max_shorts - len(selected)
        if remaining_slots > 0:
            # Get all remaining segments
            all_remaining = []
            for type_segments in categorized.values():
                all_remaining.extend(type_segments)
            
            # Sort by quality but ensure temporal spacing
            all_remaining.sort(key=lambda x: x['quality_score'], reverse=True)
            
            for segment in all_remaining:
                if len(selected) >= max_shorts:
                    break
                
                # Check temporal diversity (avoid clustering)
                if self._has_good_temporal_spacing(segment, selected):
                    selected.append(segment)
        
        self.logger.info(f"Selected {len(selected)} diverse segments from {len(segments)} candidates")
        return selected
    
    def _categorize_segments_for_diversity(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize segments for content diversity."""
        categories = {
            'celebrity': [],        # Segments with celebrity appearances
            'instruction': [],      # Teaching/talking segments
            'demonstration': [],    # Active practice/movement
            'transition': [],       # Pose changes/movements
            'introduction': [],     # Opening segments
            'conclusion': [],       # Closing segments
            'high_energy': [],      # Dynamic/energetic content
            'calm': [],            # Peaceful/relaxing content
            'interactive': []       # Questions/engagement
        }
        
        total_duration = max(seg['end_time'] for seg in segments) if segments else 1
        
        for segment in segments:
            text = segment.get('segment_text', '').lower()
            start_ratio = segment['start_time'] / total_duration
            
            # Classify based on text content and timing
            if start_ratio < 0.1:
                categories['introduction'].append(segment)
            elif start_ratio > 0.9:
                categories['conclusion'].append(segment)
            
            # Text-based classification
            if any(word in text for word in ['how', 'let\'s', 'we\'re going to', 'now', 'next']):
                categories['instruction'].append(segment)
            
            if any(word in text for word in ['breathe', 'inhale', 'exhale', 'hold', 'stretch']):
                categories['demonstration'].append(segment)
            
            if any(word in text for word in ['move', 'shift', 'transition', 'come into', 'step']):
                categories['transition'].append(segment)
            
            if '?' in text or any(word in text for word in ['you', 'feel', 'notice']):
                categories['interactive'].append(segment)
            
            # Celebrity classification
            if segment.get('has_celebrity') or segment.get('celebrity_score', 0) > 0:
                categories['celebrity'].append(segment)
            
            # Energy level classification based on keywords
            high_energy_words = ['jump', 'quick', 'fast', 'energy', 'power', 'strong']
            calm_words = ['relax', 'gentle', 'soft', 'peaceful', 'calm', 'rest', 'slow']
            
            if any(word in text for word in high_energy_words):
                categories['high_energy'].append(segment)
            elif any(word in text for word in calm_words):
                categories['calm'].append(segment)
            else:
                # Default categorization based on duration and position
                if segment['duration'] < 25:
                    categories['demonstration'].append(segment)
                else:
                    categories['instruction'].append(segment)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _has_good_temporal_spacing(self, candidate: Dict, selected: List[Dict], min_gap: float = 15.0) -> bool:
        """Check if candidate segment has good temporal spacing from selected segments."""
        if not selected:
            return True
        
        candidate_start = candidate['start_time']
        
        for existing in selected:
            existing_start = existing['start_time']
            
            # Ensure minimum gap between segments
            if abs(candidate_start - existing_start) < min_gap:
                return False
        
        return True
    
    def _segments_overlap(self, seg1: Dict, seg2: Dict) -> bool:
        """Check if two segments overlap."""
        return not (seg1['end_time'] <= seg2['start_time'] or seg2['end_time'] <= seg1['start_time'])
    
    def select_final_segments(self,
                            video_info: Dict,
                            audio_analysis: Dict,
                            ollama_analysis: Optional[Dict],
                            scene_analysis: Dict,
                            vision_analysis: Optional[Dict],
                            target_duration: Tuple[int, int] = (60, 120),
                            max_shorts: int = 10,
                            quality_threshold: float = 0.7,
                            candidate_segments: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Select final segments based on all analysis results including vision analysis.
        Guarantees a non-empty selection by falling back to engagement-first ranking
        when thresholds filter out all candidates.
        """
        try:
            # Use provided candidates or generate new ones
            if candidate_segments:
                self.logger.info(f"Using {len(candidate_segments)} pre-analyzed candidate segments")
                initial_segments = candidate_segments
            else:
                # Start with initial segment identification
                initial_segments = self.identify_short_segments(
                    video_info=video_info,
                    audio_analysis=audio_analysis,
                    ollama_analysis=ollama_analysis,
                    scene_analysis=scene_analysis,
                    target_duration=target_duration,
                    max_shorts=max_shorts * 3  # Get more candidates for better selection
                )
            
            # Apply vision enhancement if available
            if vision_analysis and initial_segments:
                enhanced_segments = self._enhance_segments_with_vision(
                    initial_segments, vision_analysis
                )
            else:
                enhanced_segments = initial_segments
            
            # Apply intelligent quality threshold with content-aware preference
            def meets_intelligent_threshold(seg):
                base_quality = seg.get('quality_score', 0)
                method = seg.get('generation_method', '')
                
                # Lower threshold for content-aware segments (they're more intelligent)
                if method == 'content_aware':
                    adjusted_threshold = quality_threshold * 0.7  # 30% lower threshold
                    if seg.get('high_value_content'):
                        adjusted_threshold = quality_threshold * 0.6  # Even lower for high-value
                elif method == 'quality_driven':
                    adjusted_threshold = quality_threshold * 0.8  # 20% lower threshold
                else:
                    adjusted_threshold = quality_threshold  # Standard threshold for others
                
                return base_quality >= adjusted_threshold
            
            quality_filtered = [seg for seg in enhanced_segments if meets_intelligent_threshold(seg)]
            
            # If not enough segments pass quality threshold, lower it slightly
            if len(quality_filtered) < max_shorts and enhanced_segments:
                lower_threshold = quality_threshold * 0.8
                quality_filtered = [
                    seg for seg in enhanced_segments 
                    if seg.get('quality_score', 0) >= lower_threshold
                ]
                self.logger.info(f"Lowered quality threshold to {lower_threshold:.2f} to get more segments")
            
            # Select diverse content up to max_shorts
            final_segments = self._select_diverse_content(quality_filtered, max_shorts)
            
            # Engagement-first forced fallback: if still empty or too few, pick top-N by engagement + vision
            if (not final_segments) and enhanced_segments:
                self.logger.info("No segments met the thresholds; applying engagement-first fallback selection")
                # Rank by engagement-first composite
                def engagement_rank(seg: Dict) -> float:
                    # Core metrics
                    eng = seg.get('engagement_score', 0.5)
                    qual = seg.get('quality_score', 0.5)
                    text_qual = seg.get('text_quality_score', 0.5)
                    
                    # Vision analysis
                    vision = seg.get('vision_score', seg.get('visual_interest', 0.5))
                    if isinstance(vision, (int, float)):
                        v = max(0.0, min(1.0, vision/10.0)) if vision > 1 else max(0.0, min(1.0, vision))
                    else:
                        v = 0.5
                    
                    # Content intelligence bonuses
                    method_bonus = 0.0
                    if seg.get('generation_method') == 'content_aware':
                        method_bonus += 0.20  # Strong preference for content-aware
                        if seg.get('high_value_content'):
                            method_bonus += 0.10
                        if seg.get('trigger_keywords'):
                            method_bonus += 0.05
                    elif seg.get('generation_method') == 'quality_driven':
                        method_bonus += 0.10  # Moderate preference for quality-driven
                    
                    # Audio transcription quality
                    audio_bonus = 0.0
                    if seg.get('has_complete_sentences', False):
                        audio_bonus += 0.05
                    if seg.get('word_count', 0) > 100:
                        audio_bonus += 0.03
                    
                    # People detection bonus
                    people_bonus = 0.05 if (seg.get('people_visible') or seg.get('people_count', 0) > 0) else 0.0
                    
                    # Celebrity detection bonus
                    celebrity_bonus = 0.0
                    if seg.get('has_celebrity') or seg.get('celebrity_score', 0) > 0:
                        celebrity_bonus += 0.15  # Strong preference for celebrity segments
                        celebrity_score = seg.get('celebrity_score', 0)
                        if celebrity_score > 0.8:
                            celebrity_bonus += 0.10  # Extra bonus for high-confidence celebrities
                    
                    # Object detection bonus
                    object_bonus = 0.0
                    if seg.get('has_object') or seg.get('object_score', 0) > 0:
                        object_bonus += 0.15  # Strong preference for object segments
                        object_score = seg.get('object_score', 0)
                        if object_score > 0.8:
                            object_bonus += 0.10  # Extra bonus for high-confidence objects
                    
                    # Comprehensive weighted score
                    return (
                        eng * 0.30 +           # Engagement
                        qual * 0.25 +          # Base quality
                        text_qual * 0.15 +     # Text quality
                        v * 0.10 +             # Visual interest
                        method_bonus +         # Content intelligence
                        audio_bonus +          # Audio quality
                        people_bonus +         # People detection
                        celebrity_bonus +      # Celebrity detection
                        object_bonus           # Object detection
                    )
                ranked = sorted(enhanced_segments, key=engagement_rank, reverse=True)
                # Light temporal spacing
                picked: List[Dict] = []
                used_times: List[Tuple[float, float]] = []
                min_gap = 15.0
                for seg in ranked:
                    if len(picked) >= max_shorts:
                        break
                    s, e = seg.get('start_time', 0.0), seg.get('end_time', 0.0)
                    if all(e <= us or s >= ue or min(abs(s-ue), abs(e-us)) >= min_gap for us, ue in used_times):
                        picked.append(seg)
                        used_times.append((s, e))
                final_segments = picked
                self.logger.info(f"Engagement-first fallback selected {len(final_segments)} segments")
            
            self.logger.info(f"Selected {len(final_segments)} final segments from {len(initial_segments)} candidates")
            return final_segments
            
        except Exception as e:
            self.logger.error(f"Error selecting final segments: {e}")
            # Return simplified segments as fallback
            return initial_segments[:max_shorts] if initial_segments else []
    
    def _enhance_segments_with_vision(self, 
                                    segments: List[Dict], 
                                    vision_analysis: Dict) -> List[Dict]:
        """
        Enhance segment scoring with vision analysis data.
        
        Args:
            segments: Initial segments from audio/content analysis
            vision_analysis: Vision analysis results
            
        Returns:
            Enhanced segments with vision-based scoring
        """
        try:
            enhanced_segments = []
            vision_segments = vision_analysis.get('segments', [])
            
            # Defensive check: ensure vision_segments is a list
            if not isinstance(vision_segments, list):
                self.logger.warning(f"vision_segments is not a list, got {type(vision_segments)}: {vision_segments}")
                return segments
            
            # Create a lookup for vision data by time
            vision_lookup = {}
            for v_seg in vision_segments:
                # Defensive check: ensure v_seg is a dictionary
                if not isinstance(v_seg, dict):
                    self.logger.warning(f"Vision segment is not a dict, got {type(v_seg)}: {v_seg}")
                    continue
                    
                start_time = v_seg.get('start_time', 0)
                vision_lookup[start_time] = v_seg
            
            for segment in segments:
                enhanced_segment = segment.copy()
                
                # Find matching or closest vision segment
                best_vision_match = None
                min_time_diff = float('inf')
                
                for v_start, v_data in vision_lookup.items():
                    time_diff = abs(v_start - segment['start_time'])
                    if time_diff < min_time_diff and time_diff < 10.0:  # Within 10 seconds
                        min_time_diff = time_diff
                        best_vision_match = v_data
                
                if best_vision_match:
                    # Enhance scoring with vision data
                    vision_score = best_vision_match.get('visual_score', 0.5)
                    visual_interest = best_vision_match.get('visual_interest', 0.5)
                    people_count = best_vision_match.get('people_count', 0)
                    scene_type = best_vision_match.get('scene_type', 'unknown')
                    
                    # Combine audio and vision scores
                    original_quality = enhanced_segment.get('quality_score', 0.5)
                    original_engagement = enhanced_segment.get('engagement_score', 0.5)
                    
                    # Weighted combination (60% audio, 40% vision)
                    combined_quality = (original_quality * 0.6) + (vision_score * 0.4)
                    combined_engagement = (original_engagement * 0.6) + (visual_interest * 0.4)
                    
                    # Bonus for segments with people visible
                    if people_count > 0:
                        combined_quality += 0.1
                        combined_engagement += 0.1
                    
                    # Bonus for interesting visual content
                    if scene_type in ['action', 'demonstration', 'presentation']:
                        combined_engagement += 0.05
                    
                    # Update segment with enhanced scores
                    enhanced_segment['quality_score'] = min(1.0, combined_quality)
                    enhanced_segment['engagement_score'] = min(1.0, combined_engagement)
                    enhanced_segment['has_vision_data'] = True
                    enhanced_segment['vision_score'] = vision_score
                    enhanced_segment['people_visible'] = people_count > 0
                    enhanced_segment['scene_type'] = scene_type
                else:
                    # No vision data available, keep original scores
                    enhanced_segment['has_vision_data'] = False
                
                enhanced_segments.append(enhanced_segment)
            
            self.logger.info(f"Enhanced {len(enhanced_segments)} segments with vision analysis")
            return enhanced_segments
            
        except Exception as e:
            self.logger.error(f"Error enhancing segments with vision: {e}")
            return segments  # Return original segments on error

    def get_segment_summary(self, segments: List[Dict]) -> Dict:
        """Generate summary of selected segments."""
        if not segments:
            return {
                'total_segments': 0,
                'total_duration': 0.0,
                'average_quality': 0.0,
                'content_types': {}
            }
        
        total_duration = sum(seg['duration'] for seg in segments)
        average_quality = sum(seg['quality_score'] for seg in segments) / len(segments)
        
        # Count content types
        content_types = {}
        for segment in segments:
            content_type = segment['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            'total_segments': len(segments),
            'total_duration': total_duration,
            'average_quality': average_quality,
            'average_engagement': sum(seg['engagement_score'] for seg in segments) / len(segments),
            'content_types': content_types,
            'duration_range': (min(seg['duration'] for seg in segments), max(seg['duration'] for seg in segments))
        }

    def _identify_audio_segments(self, 
                               video_info: Dict,
                               audio_analysis: Dict,
                               ollama_analysis: Optional[Dict],
                               scene_analysis: Optional[Dict],
                               target_duration: Tuple[int, int],
                               max_shorts: int,
                               pre_roll_end: float) -> List[Dict]:
        """Identify segments using traditional audio-based approach."""
        # Get all potential break points
        break_points = self._get_all_break_points(scene_analysis or {}, audio_analysis, pre_roll_end)
        
        # Generate candidate segments
        candidate_segments = self._generate_candidate_segments(
            break_points, 
            video_info.get('duration', 0),
            target_duration,
            pre_roll_end
        )
        
        # Score segments based on content quality
        scored_segments = self._score_segments(
            candidate_segments, 
            audio_analysis, 
            ollama_analysis,
            scene_analysis
        )
        
        return scored_segments

    def _identify_visual_segments(self, 
                                video_info: Dict,
                                audio_analysis: Dict,
                                scene_analysis: Optional[Dict],
                                target_duration: Tuple[int, int],
                                max_shorts: int,
                                pre_roll_end: float,
                                existing_audio_segments: List[Dict]) -> List[Dict]:
        """Identify segments based on visual content (silent scenes, action, etc.)."""
        try:
            from .visual_segment_detector import VisualSegmentDetector
            
            # Initialize visual detector
            visual_detector = VisualSegmentDetector()
            
            # For now, focus on silent periods that might be visually interesting
            visual_segments = []
            
            # Find silent periods from transcription
            transcription = audio_analysis.get('transcription', {})
            silence_periods = transcription.get('silence_periods', [])
            
            for silence_start, silence_end in silence_periods:
                duration = silence_end - silence_start
                
                # Only consider silent periods that are:
                # 1. After pre-roll
                # 2. Long enough to be interesting (>10 seconds)
                # 3. Within target duration range
                if (silence_start > pre_roll_end and 
                    duration >= 10.0 and 
                    duration <= target_duration[1]):
                    
                    # Check if this overlaps significantly with existing audio segments
                    overlap_ratio = self._calculate_max_overlap_with_existing(
                        silence_start, silence_end, existing_audio_segments
                    )
                    
                    # Only add if minimal overlap with audio segments
                    if overlap_ratio < 0.3:
                        visual_segments.append({
                            'start_time': silence_start,
                            'end_time': min(silence_end, silence_start + target_duration[1]),
                            'duration': min(duration, target_duration[1]),
                            'quality_score': 0.6,  # Moderate quality for silent scenes
                            'engagement_score': 0.5,  # Could be visually engaging
                            'content_type': 'visual_silent',
                            'source': 'visual_detection',
                            'visual_priority': True,
                            'segment_text': '[SILENT VISUAL CONTENT]'
                        })
            
            # Look for motion-based segments from scene analysis
            if scene_analysis and 'motion_analysis' in scene_analysis:
                motion_segments = scene_analysis['motion_analysis'].get('high_motion_segments', [])
                
                for motion_seg in motion_segments:
                    start_time = motion_seg['start']
                    end_time = motion_seg['end']
                    duration = end_time - start_time
                    
                    if (start_time > pre_roll_end and 
                        target_duration[0] <= duration <= target_duration[1]):
                        
                        overlap_ratio = self._calculate_max_overlap_with_existing(
                            start_time, end_time, existing_audio_segments
                        )
                        
                        if overlap_ratio < 0.5:  # Allow more overlap for motion segments
                            visual_segments.append({
                                'start_time': start_time,
                                'end_time': end_time,
                                'duration': duration,
                                'quality_score': 0.8,  # High quality for motion
                                'engagement_score': 0.9,  # Motion is typically engaging
                                'content_type': 'visual_action',
                                'source': 'visual_detection',
                                'visual_priority': True,
                                'segment_text': '[HIGH MOTION VISUAL CONTENT]'
                            })
            
            self.logger.info(f"Identified {len(visual_segments)} visual-based segments")
            return visual_segments
            
        except Exception as e:
            self.logger.warning(f"Visual segment detection failed: {e}")
            return []

    def _enhance_segments_with_object_scores(self, 
                                           segments: List[Dict], 
                                           object_detection_results: Dict) -> List[Dict]:
        """Enhance segments with object detection relevance scores."""
        try:
            if not object_detection_results or object_detection_results.get('status') != 'success':
                return segments
            
            detected_objects = object_detection_results.get('detected_objects', [])
            tracking_results = object_detection_results.get('object_tracks', [])
            
            # Defensive check: ensure detected_objects is a list
            if not isinstance(detected_objects, list):
                self.logger.warning(f"detected_objects is not a list, got {type(detected_objects)}: {detected_objects}")
                return segments
            
            # Create object timeline (objects are now dictionaries)
            object_timeline = {}
            for obj in detected_objects:
                # Defensive check: ensure obj is a dictionary
                if not isinstance(obj, dict):
                    self.logger.warning(f"Object in detected_objects is not a dict, got {type(obj)}: {obj}")
                    continue
                    
                timestamp = obj.get('frame_timestamp')
                if timestamp not in object_timeline:
                    object_timeline[timestamp] = []
                object_timeline[timestamp].append(obj)
            
            # Enhance each segment
            for segment in segments:
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Find objects in this time range
                segment_objects = []
                for timestamp, objects in object_timeline.items():
                    if start_time <= timestamp <= end_time:
                        segment_objects.extend(objects)
                
                if segment_objects:
                    # Calculate object relevance score (objects are now dictionaries)
                    object_relevance = sum(obj.get('relevance_score', 0) for obj in segment_objects) / len(segment_objects)
                    object_confidence = sum(obj.get('confidence', 0) for obj in segment_objects) / len(segment_objects)
                    prompt_match = sum(obj.get('prompt_match_score', 0) for obj in segment_objects) / len(segment_objects)
                    
                    # Add object-based scores
                    segment['object_relevance_score'] = object_relevance
                    segment['object_confidence_score'] = object_confidence
                    segment['object_prompt_match_score'] = prompt_match
                    segment['objects_detected'] = len(segment_objects)
                    segment['unique_object_classes'] = len(set(obj.get('class_name', '') for obj in segment_objects))
                else:
                    # No objects detected in this segment
                    segment['object_relevance_score'] = 0.0
                    segment['object_confidence_score'] = 0.0
                    segment['object_prompt_match_score'] = 0.0
                    segment['objects_detected'] = 0
                    segment['unique_object_classes'] = 0
            
            return segments
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance segments with object scores: {e}")
            return segments
    
    def _calculate_combined_score(self, segment: Dict) -> float:
        """Calculate combined score from prompt match and object detection."""
        prompt_score = segment.get('prompt_match_score', 0.0)
        object_relevance = segment.get('object_relevance_score', 0.0)
        object_confidence = segment.get('object_confidence_score', 0.0)
        object_prompt_match = segment.get('object_prompt_match_score', 0.0)
        
        # Weighted combination
        base_score = prompt_score * 0.4
        object_score = (object_relevance * 0.3 + object_confidence * 0.2 + object_prompt_match * 0.4) * 0.6
        
        # Bonus for segments with multiple relevant objects
        objects_detected = segment.get('objects_detected', 0)
        unique_classes = segment.get('unique_object_classes', 0)
        
        object_bonus = min(0.2, (objects_detected * 0.05) + (unique_classes * 0.1))
        
        return min(1.0, base_score + object_score + object_bonus)
    
    def _ensure_prompt_scores(self, segments: List[Dict]) -> List[Dict]:
        """Ensure each segment has a non-zero prompt_match_score by deriving it from available scores when missing.
        This makes the final selection robust to segments produced by contextual/heuristic analyzers.
        """
        derived = 0
        for seg in segments:
            current = seg.get('prompt_match_score')
            if current is None or current == 0:
                candidates = [
                    # Climax-specific scores
                    seg.get('final_climax_score'),
                    seg.get('climax_score'),
                    # Contextual / heuristic scores
                    seg.get('contextual_overall_score'),
                    seg.get('composite_validation_score'),
                    seg.get('contextual_relevance_score'),
                    seg.get('heuristic_score'),
                    seg.get('quality_score'),
                    seg.get('ai_refined_score')
                ]
                # Filter out None and negatives
                numeric = [c for c in candidates if isinstance(c, (int, float)) and c is not None]
                if numeric:
                    new_score = max(0.0, min(1.0, max(numeric)))
                else:
                    # Minimal non-zero score if we have any text content
                    new_score = 0.05 if seg.get('segment_text') else 0.0
                seg['prompt_match_score'] = new_score
                if new_score > 0:
                    derived += 1
        if derived:
            self.logger.info(f"Derived prompt_match_score for {derived} segments from contextual/heuristic metrics")
        return segments
    
    async def cleanup(self):
        """Clean up resources from all analyzers."""
        try:
            if self.object_detector:
                await self.object_detector.cleanup()
            
            if self.ai_reframer:
                await self.ai_reframer.cleanup()
            
            self.logger.info("ContentAnalyzer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during ContentAnalyzer cleanup: {e}")
    
    # ================================
    # PHASE 2: MULTI-PASS ANALYSIS PIPELINE
    # ================================
    
    async def _phase2_multipass_analysis(
        self,
        initial_segments,
        user_prompt,
        comprehensive_segments,
        audio_analysis,
        vision_analysis,
        scene_analysis,
        content_overview,
        intent_analysis,
        video_info,
        target_duration,
        max_shorts
    ):
        """
        Phase 2: Multi-pass analysis pipeline for iterative segment refinement.
        
        This implements a sophisticated multi-pass system that:
        1. Cross-validates initial results
        2. Performs iterative refinement
        3. Quality scoring and ranking
        4. Final optimization
        """
        try:
            self.logger.info("Starting Phase 2 multi-pass analysis pipeline")
            
            # Pass 1: Cross-validation and quality assessment
            cross_validation_result = await self._pass1_cross_validation(
                initial_segments, user_prompt, comprehensive_segments,
                audio_analysis, vision_analysis, scene_analysis,
                content_overview, intent_analysis
            )
            
            # Pass 2: Iterative refinement
            refinement_result = await self._pass2_iterative_refinement(
                cross_validation_result['segments'], user_prompt,
                comprehensive_segments, audio_analysis, vision_analysis,
                scene_analysis, content_overview, intent_analysis, video_info
            )
            
            # Pass 3: Quality scoring and final optimization
            final_result = await self._pass3_quality_optimization(
                refinement_result['segments'], user_prompt,
                content_overview, intent_analysis, video_info,
                target_duration, max_shorts
            )
            
            # Compile comprehensive results
            multipass_confidence = (
                cross_validation_result.get('confidence', 0.7) * 0.3 +
                refinement_result.get('confidence', 0.7) * 0.4 +
                final_result.get('confidence', 0.7) * 0.3
            )
            
            return {
                'status': 'success',
                'segments': final_result['segments'],
                'analysis_method': 'phase2_multipass_enhanced',
                'content_overview': content_overview,
                'intent_analysis': intent_analysis,
                'context_confidence': multipass_confidence,
                'multipass_details': {
                    'pass1_cross_validation': cross_validation_result.get('metrics', {}),
                    'pass2_iterative_refinement': refinement_result.get('metrics', {}),
                    'pass3_quality_optimization': final_result.get('metrics', {}),
                    'total_analysis_passes': 3,
                    'enhancement_level': 'phase2_multipass'
                },
                'generation_details': {
                    'initial_segments': len(initial_segments),
                    'total_candidates': len(comprehensive_segments),
                    'final_selected_count': len(final_result['segments']),
                    'multipass_confidence': multipass_confidence,
                    'enhancement_level': 'phase2_multipass'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Phase 2 multi-pass analysis failed: {e}")
            return {
                'status': 'error',
                'error': f"Multi-pass analysis failed: {e}",
                'fallback_required': True
            }
    
    async def _pass1_cross_validation(
        self, initial_segments, user_prompt, comprehensive_segments,
        audio_analysis, vision_analysis, scene_analysis,
        content_overview, intent_analysis
    ):
        """Pass 1: Cross-validation between different analysis methods."""
        try:
            self.logger.info("Phase 2 Pass 1: Cross-validation analysis")
            
            # Method 1: LLM-based validation
            llm_validation = await self._validate_segments_with_llm(
                initial_segments, user_prompt, content_overview, intent_analysis
            )
            
            # Method 2: Heuristic-based validation
            heuristic_validation = self._validate_segments_heuristically(
                initial_segments, audio_analysis, scene_analysis, intent_analysis
            )
            
            # Method 3: Content coherence validation
            coherence_validation = self._validate_content_coherence(
                initial_segments, audio_analysis, content_overview
            )
            
            # Cross-validate and score segments
            validated_segments = []
            for segment in initial_segments:
                segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
                validation_scores = {
                    'llm_score': llm_validation.get(segment_id, {}).get('score', 0.5),
                    'heuristic_score': heuristic_validation.get(segment_id, {}).get('score', 0.5),
                    'coherence_score': coherence_validation.get(segment_id, {}).get('score', 0.5)
                }
                
                # Calculate composite validation score
                composite_score = (
                    validation_scores['llm_score'] * 0.5 +
                    validation_scores['heuristic_score'] * 0.3 +
                    validation_scores['coherence_score'] * 0.2
                )
                
                # Add validation metadata
                enhanced_segment = segment.copy()
                enhanced_segment.update({
                    'segment_id': segment_id,
                    'validation_scores': validation_scores,
                    'composite_validation_score': composite_score,
                    'cross_validation_passed': composite_score >= 0.6,
                    'validation_confidence': min(validation_scores.values())
                })
                
                validated_segments.append(enhanced_segment)
            
            # Filter segments that passed cross-validation
            passed_segments = [seg for seg in validated_segments if seg.get('cross_validation_passed', False)]
            
            self.logger.info(f"Pass 1: {len(passed_segments)}/{len(initial_segments)} segments passed cross-validation")
            
            return {
                'segments': passed_segments,
                'confidence': sum(seg.get('composite_validation_score', 0.5) for seg in passed_segments) / max(len(passed_segments), 1),
                'metrics': {
                    'initial_count': len(initial_segments),
                    'validated_count': len(passed_segments),
                    'validation_rate': len(passed_segments) / max(len(initial_segments), 1),
                    'average_validation_score': sum(seg.get('composite_validation_score', 0.5) for seg in validated_segments) / max(len(validated_segments), 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pass 1 cross-validation failed: {e}")
            return {
                'segments': initial_segments,
                'confidence': 0.5,
                'metrics': {'error': str(e)}
            }
    
    async def _pass2_iterative_refinement(
        self, validated_segments, user_prompt, comprehensive_segments,
        audio_analysis, vision_analysis, scene_analysis,
        content_overview, intent_analysis, video_info
    ):
        """Pass 2: Iterative refinement of segment selection and timing."""
        try:
            self.logger.info("Phase 2 Pass 2: Iterative refinement")
            
            refined_segments = validated_segments.copy()
            iterations = 0
            max_iterations = 3
            improvement_threshold = 0.05
            
            previous_score = sum(seg.get('composite_validation_score', 0.5) for seg in refined_segments) / max(len(refined_segments), 1)
            
            while iterations < max_iterations:
                iterations += 1
                self.logger.info(f"Refinement iteration {iterations}/{max_iterations}")
                
                # Refinement 1: Timing optimization
                timing_refined = await self._refine_segment_timing(
                    refined_segments, audio_analysis, scene_analysis
                )
                
                # Refinement 2: Content overlap detection and resolution
                overlap_resolved = self._resolve_segment_overlaps(
                    timing_refined, video_info
                )
                
                # Refinement 3: Gap analysis and filling
                gap_filled = await self._analyze_and_fill_gaps(
                    overlap_resolved, comprehensive_segments, user_prompt,
                    content_overview, intent_analysis
                )
                
                # Calculate improvement score
                current_score = sum(seg.get('composite_validation_score', 0.5) for seg in gap_filled) / max(len(gap_filled), 1)
                improvement = current_score - previous_score
                
                if improvement < improvement_threshold:
                    self.logger.info(f"Refinement converged after {iterations} iterations (improvement: {improvement:.3f})")
                    break
                
                refined_segments = gap_filled
                previous_score = current_score
                self.logger.info(f"Iteration {iterations} improvement: {improvement:.3f}")
            
            self.logger.info(f"Pass 2: Iterative refinement completed in {iterations} iterations")
            
            return {
                'segments': refined_segments,
                'confidence': current_score,
                'metrics': {
                    'iterations': iterations,
                    'final_improvement': improvement,
                    'refinement_convergence': improvement < improvement_threshold,
                    'segment_count': len(refined_segments)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pass 2 iterative refinement failed: {e}")
            return {
                'segments': validated_segments,
                'confidence': 0.6,
                'metrics': {'error': str(e)}
            }
    
    async def _pass3_quality_optimization(
        self, refined_segments, user_prompt, content_overview,
        intent_analysis, video_info, target_duration, max_shorts
    ):
        """Pass 3: Quality scoring and final optimization."""
        try:
            self.logger.info("Phase 2 Pass 3: Quality optimization")
            
            # Enhanced quality scoring
            quality_scored_segments = []
            for segment in refined_segments:
                quality_score = await self._calculate_enhanced_quality_score(
                    segment, user_prompt, content_overview, intent_analysis
                )
                
                enhanced_segment = segment.copy()
                enhanced_segment.update({
                    'quality_score': quality_score,
                    'phase2_enhanced': True,
                    'final_optimization_applied': True
                })
                
                quality_scored_segments.append(enhanced_segment)
            
            # Sort by quality score
            quality_scored_segments.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Final selection optimization
            optimized_segments = self._optimize_final_selection(
                quality_scored_segments, target_duration, max_shorts, intent_analysis
            )
            
            # Calculate final confidence
            final_confidence = sum(seg.get('quality_score', 0.5) for seg in optimized_segments) / max(len(optimized_segments), 1)
            
            self.logger.info(f"Pass 3: Quality optimization completed - {len(optimized_segments)} final segments")
            
            return {
                'segments': optimized_segments,
                'confidence': final_confidence,
                'metrics': {
                    'quality_scored_count': len(quality_scored_segments),
                    'final_selected_count': len(optimized_segments),
                    'average_quality_score': final_confidence,
                    'optimization_applied': True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pass 3 quality optimization failed: {e}")
            return {
                'segments': refined_segments,
                'confidence': 0.7,
                'metrics': {'error': str(e)}
            }

    async def _analyze_content_overview(self,
                                      audio_analysis: Dict,
                                      vision_analysis: Optional[Dict],
                                      video_info: Dict) -> Dict:
        """
        PHASE 1 ENHANCEMENT: Comprehensive content overview analysis.
        
        This provides the LLM with full context about the video content,
        narrative structure, and overall characteristics before segment selection.
        """
        try:
            # Prepare comprehensive content summary
            transcription = audio_analysis.get('transcription', {})
            segments = transcription.get('segments', [])
            
            # Extract key content indicators
            total_duration = video_info.get('duration', 0)
            speaker_count = len(set(seg.get('speaker', 'unknown') for seg in segments))
            
            # Create content analysis prompt
            overview_prompt = f"""
            You are analyzing a video to understand its overall content and structure for short-form content creation.
            
            VIDEO METADATA:
            - Duration: {total_duration} seconds
            - Estimated speakers: {speaker_count}
            - Has visual analysis: {vision_analysis is not None}
            
            TRANSCRIPTION SAMPLE (first 10 segments):
            {json.dumps(segments[:10], indent=2)}
            
            
            VISUAL CONTENT SUMMARY (if available):
            {json.dumps(vision_analysis.get('summary', {}) if vision_analysis else {}, indent=2)}
            
            ANALYSIS TASK:
            Provide a comprehensive overview to help with intelligent scene selection:
            
            1. What type of content is this? (movie, tutorial, interview, presentation, sports, etc.)
            2. What is the overall narrative structure and flow?
            3. What are the main themes and topics discussed?
            4. Where are the most engaging/important moments likely to be?
            5. What would work best as short-form social media content?
            
            Respond in JSON format:
            {{
                "content_type": "movie|tutorial|interview|presentation|sports|documentary|other",
                "genre": "specific genre if applicable",
                "narrative_structure": {{
                    "has_clear_beginning": boolean,
                    "has_development": boolean,
                    "has_climax_or_peak": boolean,
                    "has_resolution": boolean,
                    "content_flow": "linear|episodic|instructional|conversational"
                }},
                "main_themes": ["theme1", "theme2", "theme3"],
                "content_characteristics": {{
                    "is_educational": boolean,
                    "is_entertaining": boolean,
                    "is_narrative_driven": boolean,
                    "has_demonstrations": boolean,
                    "interaction_level": "monologue|dialogue|interactive"
                }},
                "engagement_patterns": {{
                    "peak_likely_at": "beginning|middle|end|distributed",
                    "content_density": "high|medium|low",
                    "emotional_variation": "high|medium|low"
                }},
                "short_form_potential": {{
                    "best_segment_types": ["type1", "type2"],
                    "ideal_segment_length": "15-30|30-60|60-90 seconds",
                    "key_selection_criteria": ["criteria1", "criteria2"]
                }}
            }}
            """
            
            if self.ollama_client:
                response = await self.ollama_client._make_request(
                    prompt=overview_prompt,
                    model=self.ollama_client.get_best_model("analysis"),
                    cache_key=f"content_overview_{hash(str(audio_analysis.get('transcription', {})) + str(video_info))}"
                )
                
                overview_data = self.ollama_client._parse_json_response(response)
                self.logger.info(f"Content overview analysis completed: {overview_data.get('content_type', 'unknown')} type")
                return overview_data
            else:
                return self._fallback_content_overview(audio_analysis, video_info)
                
        except Exception as e:
            self.logger.error(f"Content overview analysis failed: {e}")
            return self._fallback_content_overview(audio_analysis, video_info)
    
    async def _analyze_user_intent_comprehensive(self,
                                               user_prompt: str,
                                               content_overview: Dict,
                                               video_info: Dict) -> Dict:
        """
        PHASE 1 ENHANCEMENT: Enhanced user intent analysis with full content context.
        
        This goes beyond keyword matching to understand what the user
        really wants based on the specific video content.
        """
        try:
            intent_prompt = f"""
            You are an expert video editor who understands user intent for creating short-form content.
            
            USER REQUEST: "{user_prompt}"
            
            VIDEO CONTENT OVERVIEW:
            {json.dumps(content_overview, indent=2)}
            
            VIDEO METADATA:
            - Duration: {video_info.get('duration', 0)} seconds
            - Type: {content_overview.get('content_type', 'unknown')}
            
            ANALYSIS TASK:
            Based on the user's request and the actual video content, determine:
            
            1. What the user's true intent is (beyond literal keywords)
            2. What type of content from this specific video would satisfy their request
            3. How to adapt their request to this video's content and structure
            4. What quality criteria should be used for segment selection
            
            Consider the video's actual content type and structure when interpreting the request.
            
            Provide detailed intent analysis:
            {{
                "intent_interpretation": {{
                    "literal_request": "what they said",
                    "contextual_intent": "what they actually want given this video type",
                    "content_alignment": "how well this video can satisfy their request",
                    "adaptation_strategy": "how to best fulfill their intent with available content"
                }},
                "selection_criteria": {{
                    "primary_factors": ["most important selection criteria"],
                    "secondary_factors": ["additional considerations"],
                    "content_position_preference": "beginning|middle|end|peak|any",
                    "quality_thresholds": {{
                        "minimum_engagement": 0.4,
                        "minimum_completeness": 0.3,
                        "minimum_relevance": 0.5
                    }}
                }},
                "content_requirements": {{
                    "emotional_tone": "required emotional characteristics",
                    "must_include": ["required elements"],
                    "must_avoid": ["elements to avoid"],
                    "duration_preference": "ideal length in seconds",
                    "standalone_viability": "high|medium|low requirement"
                }},
                "confidence_assessment": {{
                    "intent_clarity": 0.8,
                    "content_availability": 0.7,
                    "match_likelihood": 0.8,
                    "overall_confidence": 0.8
                }}
            }}
            """
            
            if self.ollama_client:
                response = await self.ollama_client._make_request(
                    prompt=intent_prompt,
                    model=self.ollama_client.get_best_model("analysis"),
                    cache_key=f"intent_analysis_{hash(user_prompt + str(content_overview))}"
                )
                
                intent_data = self.ollama_client._parse_json_response(response)
                self.logger.info(f"Intent analysis completed with {intent_data.get('confidence_assessment', {}).get('overall_confidence', 0):.2f} confidence")
                return intent_data
            else:
                return self._fallback_intent_analysis(user_prompt)
                
        except Exception as e:
            self.logger.error(f"Intent analysis failed: {e}")
            return self._fallback_intent_analysis(user_prompt)
    
    def _fallback_content_overview(self, audio_analysis: Dict, video_info: Dict) -> Dict:
        """Fallback content overview when LLM analysis fails."""
        # Simple heuristic-based content analysis
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        # Basic content type detection
        all_text = ' '.join(seg.get('text', '') for seg in segments).lower()
        
        content_type = 'other'
        if any(word in all_text for word in ['tutorial', 'how to', 'learn', 'explain']):
            content_type = 'tutorial'
        elif any(word in all_text for word in ['interview', 'conversation', 'discuss']):
            content_type = 'interview'
        elif len(segments) > 0 and len(set(seg.get('speaker', 'unknown') for seg in segments)) > 2:
            content_type = 'presentation'
        
        return {
            'content_type': content_type,
            'narrative_structure': {
                'has_clear_beginning': True,
                'has_development': True,
                'has_climax_or_peak': False,
                'has_resolution': True,
                'content_flow': 'linear'
            },
            'main_themes': ['general'],
            'content_characteristics': {
                'is_educational': content_type == 'tutorial',
                'is_entertaining': content_type not in ['tutorial', 'presentation'],
                'is_narrative_driven': False,
                'has_demonstrations': content_type == 'tutorial',
                'interaction_level': 'monologue'
            },
            'engagement_patterns': {
                'peak_likely_at': 'middle',
                'content_density': 'medium',
                'emotional_variation': 'medium'
            },
            'short_form_potential': {
                'best_segment_types': ['highlights', 'key_points'],
                'ideal_segment_length': '30-60 seconds',
                'key_selection_criteria': ['completeness', 'engagement']
            }
        }
    
    def _fallback_intent_analysis(self, user_prompt: str) -> Dict:
        """Fallback intent analysis when LLM analysis fails."""
        # Simple keyword-based intent detection
        prompt_lower = user_prompt.lower()
        
        intent_type = 'general'
        if any(word in prompt_lower for word in ['climax', 'peak', 'best', 'highlight']):
            intent_type = 'highlights'
        elif any(word in prompt_lower for word in ['comedy', 'funny', 'humor']):
            intent_type = 'comedy'
        elif any(word in prompt_lower for word in ['emotional', 'touching', 'moving']):
            intent_type = 'emotional'
        elif any(word in prompt_lower for word in ['educational', 'learn', 'tutorial']):
            intent_type = 'educational'
        
        return {
            'intent_interpretation': {
                'literal_request': user_prompt,
                'contextual_intent': f'User wants {intent_type} content',
                'content_alignment': 'medium',
                'adaptation_strategy': 'Find segments matching keywords'
            },
            'selection_criteria': {
                'primary_factors': ['keyword_match', 'quality'],
                'secondary_factors': ['duration', 'completeness'],
                'content_position_preference': 'any',
                'quality_thresholds': {
                    'minimum_engagement': 0.3,
                    'minimum_completeness': 0.3,
                    'minimum_relevance': 0.3
                }
            },
            'content_requirements': {
                'emotional_tone': intent_type,
                'must_include': [],
                'must_avoid': ['silence', 'intro'],
                'duration_preference': '30-60 seconds',
                'standalone_viability': 'medium'
            },
            'confidence_assessment': {
                'intent_clarity': 0.5,
                'content_availability': 0.5,
                'match_likelihood': 0.5,
                'overall_confidence': 0.3
            }
        }
    
    # ================================
    # PHASE 2: HELPER METHODS FOR MULTI-PASS ANALYSIS
    # ================================
    
    async def _validate_segments_with_llm(self, segments, user_prompt, content_overview, intent_analysis):
        """LLM-based validation of segment quality and relevance."""
        try:
            if not self.ollama_client:
                return {}
            
            validation_results = {}
            
            for segment in segments:
                segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
                segment_text = segment.get('segment_text', '')
                
                # Create validation prompt
                validation_prompt = f"""
                Analyze this video segment for quality and relevance:
                
                User Request: "{user_prompt}"
                Content Type: {content_overview.get('content_type', 'unknown')}
                Main Themes: {', '.join(content_overview.get('main_themes', []))}
                User Intent: {intent_analysis.get('intent_interpretation', {}).get('contextual_intent', '')}
                
                Segment Text: "{segment_text}"
                Duration: {segment.get('duration', 0):.1f}s
                
                Rate this segment (0-1) on:
                1. Relevance to user request
                2. Content quality
                3. Standalone viability
                4. Engagement potential
                
                Respond with: score:X.X reasoning:"explanation"
                """
                
                try:
                    response = await self.ollama_client.generate_response(
                        validation_prompt,
                        "mistral-small3.2:latest"
                    )
                    
                    # Parse LLM response
                    score = 0.5  # default
                    if 'score:' in response:
                        score_str = response.split('score:')[1].split()[0]
                        try:
                            score = float(score_str)
                        except:
                            pass
                    
                    validation_results[segment_id] = {
                        'score': max(0, min(1, score)),
                        'reasoning': response,
                        'method': 'llm_validation'
                    }
                    
                except Exception as e:
                    validation_results[segment_id] = {
                        'score': 0.5,
                        'error': str(e),
                        'method': 'llm_validation'
                    }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}")
            return {}
    
    def _validate_segments_heuristically(self, segments, audio_analysis, scene_analysis, intent_analysis):
        """Heuristic-based validation using audio and scene analysis."""
        validation_results = {}
        
        try:
            for segment in segments:
                segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
                
                # Heuristic scoring factors
                factors = {
                    'duration_score': self._score_duration_appropriateness(segment),
                    'audio_quality_score': self._score_audio_quality(segment, audio_analysis),
                    'scene_transition_score': self._score_scene_transitions(segment, scene_analysis),
                    'content_density_score': self._score_content_density(segment)
                }
                
                # Weighted heuristic score
                heuristic_score = (
                    factors['duration_score'] * 0.25 +
                    factors['audio_quality_score'] * 0.3 +
                    factors['scene_transition_score'] * 0.25 +
                    factors['content_density_score'] * 0.2
                )
                
                validation_results[segment_id] = {
                    'score': heuristic_score,
                    'factors': factors,
                    'method': 'heuristic_validation'
                }
        
        except Exception as e:
            self.logger.error(f"Heuristic validation failed: {e}")
        
        return validation_results
    
    def _validate_content_coherence(self, segments, audio_analysis, content_overview):
        """Validate content coherence and narrative flow."""
        validation_results = {}
        
        try:
            content_type = content_overview.get('content_type', 'unknown')
            narrative_structure = content_overview.get('narrative_structure', {})
            
            for segment in segments:
                segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
                segment_text = segment.get('segment_text', '')
                
                # Coherence scoring
                coherence_factors = {
                    'text_coherence': self._score_text_coherence(segment_text),
                    'narrative_fit': self._score_narrative_fit(segment, content_type, narrative_structure),
                    'standalone_quality': self._score_standalone_quality(segment_text, content_type)
                }
                
                coherence_score = sum(coherence_factors.values()) / len(coherence_factors)
                
                validation_results[segment_id] = {
                    'score': coherence_score,
                    'factors': coherence_factors,
                    'method': 'coherence_validation'
                }
        
        except Exception as e:
            self.logger.error(f"Coherence validation failed: {e}")
        
        return validation_results
    
    async def _refine_segment_timing(self, segments, audio_analysis, scene_analysis):
        """Refine segment timing based on audio and scene analysis."""
        refined_segments = []
        
        try:
            for segment in segments:
                # Extract relevant audio segments
                start_time = segment.get('start_time', 0)
                end_time = segment.get('end_time', 0)
                
                # Find optimal boundaries
                optimal_start = self._find_optimal_start_boundary(start_time, audio_analysis, scene_analysis)
                optimal_end = self._find_optimal_end_boundary(end_time, audio_analysis, scene_analysis)
                
                # Create refined segment
                refined_segment = segment.copy()
                refined_segment.update({
                    'start_time': optimal_start,
                    'end_time': optimal_end,
                    'duration': optimal_end - optimal_start,
                    'timing_refined': True,
                    'original_start': start_time,
                    'original_end': end_time
                })
                
                refined_segments.append(refined_segment)
        
        except Exception as e:
            self.logger.error(f"Timing refinement failed: {e}")
            return segments
        
        return refined_segments
    
    def _resolve_segment_overlaps(self, segments, video_info):
        """Resolve overlapping segments through intelligent merging or splitting."""
        resolved_segments = []
        
        try:
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
            
            for i, segment in enumerate(sorted_segments):
                if i == 0:
                    resolved_segments.append(segment)
                    continue
                
                prev_segment = resolved_segments[-1]
                current_start = segment.get('start_time', 0)
                prev_end = prev_segment.get('end_time', 0)
                
                # Check for overlap
                if current_start < prev_end:
                    # Resolve overlap
                    if self._should_merge_segments(prev_segment, segment):
                        # Merge segments
                        merged_segment = self._merge_segments(prev_segment, segment)
                        resolved_segments[-1] = merged_segment
                    else:
                        # Adjust boundaries
                        split_point = (prev_end + current_start) / 2
                        prev_segment['end_time'] = split_point
                        prev_segment['duration'] = split_point - prev_segment.get('start_time', 0)
                        
                        adjusted_segment = segment.copy()
                        adjusted_segment['start_time'] = split_point
                        adjusted_segment['duration'] = adjusted_segment.get('end_time', 0) - split_point
                        
                        resolved_segments.append(adjusted_segment)
                else:
                    resolved_segments.append(segment)
        
        except Exception as e:
            self.logger.error(f"Overlap resolution failed: {e}")
            return segments
        
        return resolved_segments
    
    async def _analyze_and_fill_gaps(self, segments, comprehensive_segments, user_prompt, content_overview, intent_analysis):
        """Analyze gaps between segments and potentially fill with relevant content."""
        try:
            # Sort segments by start time
            sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
            
            # Identify gaps
            gaps = []
            for i in range(len(sorted_segments) - 1):
                gap_start = sorted_segments[i].get('end_time', 0)
                gap_end = sorted_segments[i + 1].get('start_time', 0)
                gap_duration = gap_end - gap_start
                
                if gap_duration > 5.0:  # Only consider significant gaps
                    gaps.append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'position': i + 1
                    })
            
            # Analyze each gap for potential content
            filled_segments = sorted_segments.copy()
            
            for gap in gaps:
                gap_candidates = [
                    seg for seg in comprehensive_segments
                    if (seg.get('start_time', 0) >= gap['start'] and
                        seg.get('end_time', 0) <= gap['end'])
                ]
                
                if gap_candidates:
                    # Evaluate gap candidates
                    best_candidate = await self._evaluate_gap_candidates(
                        gap_candidates, user_prompt, content_overview, intent_analysis
                    )
                    
                    if best_candidate and best_candidate.get('relevance_score', 0) > 0.6:
                        filled_segments.insert(gap['position'], best_candidate)
            
            return filled_segments
        
        except Exception as e:
            self.logger.error(f"Gap analysis failed: {e}")
            return segments
    
    async def _calculate_enhanced_quality_score(self, segment, user_prompt, content_overview, intent_analysis):
        """Calculate enhanced quality score for Phase 2 optimization."""
        try:
            # Base scores from existing validation
            base_score = segment.get('composite_validation_score', 0.5)
            
            # Enhanced scoring factors
            enhanced_factors = {
                'contextual_relevance': self._score_contextual_relevance(segment, user_prompt, intent_analysis),
                'content_type_alignment': self._score_content_type_alignment(segment, content_overview),
                'engagement_potential': self._score_engagement_potential(segment),
                'technical_quality': self._score_technical_quality(segment),
                'narrative_value': self._score_narrative_value(segment, content_overview)
            }
            
            # Weighted enhanced score
            enhanced_score = (
                base_score * 0.3 +
                enhanced_factors['contextual_relevance'] * 0.25 +
                enhanced_factors['content_type_alignment'] * 0.15 +
                enhanced_factors['engagement_potential'] * 0.15 +
                enhanced_factors['technical_quality'] * 0.1 +
                enhanced_factors['narrative_value'] * 0.05
            )
            
            return max(0, min(1, enhanced_score))
        
        except Exception as e:
            self.logger.error(f"Enhanced quality scoring failed: {e}")
            return segment.get('composite_validation_score', 0.5)
    
    def _optimize_final_selection(self, quality_scored_segments, target_duration, max_shorts, intent_analysis):
        """Optimize final segment selection based on quality scores and constraints."""
        try:
            # Sort by quality score
            sorted_segments = sorted(quality_scored_segments, key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Apply selection constraints
            min_duration, max_duration = target_duration
            selected_segments = []
            
            for segment in sorted_segments:
                if len(selected_segments) >= max_shorts:
                    break
                
                duration = segment.get('duration', 0)
                if min_duration <= duration <= max_duration:
                    # Check for diversity
                    if self._ensures_selection_diversity(segment, selected_segments, intent_analysis):
                        selected_segments.append(segment)
            
            return selected_segments
        
        except Exception as e:
            self.logger.error(f"Final selection optimization failed: {e}")
            return quality_scored_segments[:max_shorts]
    
    # Helper scoring methods
    def _score_duration_appropriateness(self, segment):
        """Score segment duration appropriateness for short-form content."""
        duration = segment.get('duration', 0)
        if 15 <= duration <= 60:
            return 1.0
        elif 10 <= duration < 15 or 60 < duration <= 90:
            return 0.7
        else:
            return 0.3
    
    def _score_audio_quality(self, segment, audio_analysis):
        """Score audio quality based on transcription confidence and clarity."""
        # Implementation would analyze audio transcription quality
        return 0.7  # Placeholder
    
    def _score_scene_transitions(self, segment, scene_analysis):
        """Score scene transition quality within segment."""
        # Implementation would analyze scene breaks and transitions
        return 0.6  # Placeholder
    
    def _score_content_density(self, segment):
        """Score content density and information richness."""
        text_length = len(segment.get('segment_text', ''))
        duration = segment.get('duration', 1)
        words_per_second = text_length / max(duration, 1) / 5  # Approximate words
        
        if 2 <= words_per_second <= 4:
            return 1.0
        elif 1 <= words_per_second < 2 or 4 < words_per_second <= 6:
            return 0.7
        else:
            return 0.4
    
    def _score_text_coherence(self, text):
        """Score text coherence and readability."""
        if not text:
            return 0.0
        
        # Simple coherence metrics
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if 8 <= avg_sentence_length <= 20:
            return 0.8
        else:
            return 0.5
    
    def _score_narrative_fit(self, segment, content_type, narrative_structure):
        """Score how well segment fits the overall narrative."""
        # Implementation would analyze narrative position and structure
        return 0.6  # Placeholder
    
    def _score_standalone_quality(self, text, content_type):
        """Score standalone quality of segment text."""
        if not text:
            return 0.0
        
        # Check for complete thoughts, context
        has_complete_thought = '.' in text or '!' in text or '?' in text
        has_context_clues = any(word in text.lower() for word in ['this', 'that', 'here', 'now', 'today'])
        
        score = 0.5
        if has_complete_thought:
            score += 0.3
        if has_context_clues:
            score += 0.2
        
        return min(1.0, score)
    
    def _find_optimal_start_boundary(self, start_time, audio_analysis, scene_analysis):
        """Find optimal start boundary based on audio and scene analysis."""
        # Implementation would find natural start points
        return start_time  # Placeholder
    
    def _find_optimal_end_boundary(self, end_time, audio_analysis, scene_analysis):
        """Find optimal end boundary based on audio and scene analysis."""
        # Implementation would find natural end points
        return end_time  # Placeholder
    
    def _should_merge_segments(self, segment1, segment2):
        """Determine if two overlapping segments should be merged."""
        # Check content similarity, timing, and thematic coherence
        return False  # Placeholder - conservative approach
    
    def _merge_segments(self, segment1, segment2):
        """Merge two segments into one."""
        return {
            'start_time': min(segment1.get('start_time', 0), segment2.get('start_time', 0)),
            'end_time': max(segment1.get('end_time', 0), segment2.get('end_time', 0)),
            'segment_text': f"{segment1.get('segment_text', '')} {segment2.get('segment_text', '')}",
            'merged_from': [segment1, segment2],
            'generation_method': 'phase2_merge'
        }
    
    async def _evaluate_gap_candidates(self, candidates, user_prompt, content_overview, intent_analysis):
        """Evaluate candidates for filling gaps between segments."""
        if not candidates:
            return None
        
        # Score each candidate
        scored_candidates = []
        for candidate in candidates:
            relevance_score = self._score_contextual_relevance(candidate, user_prompt, intent_analysis)
            candidate['relevance_score'] = relevance_score
            scored_candidates.append(candidate)
        
        # Return best candidate
        return max(scored_candidates, key=lambda x: x.get('relevance_score', 0))
    
    def _score_contextual_relevance(self, segment, user_prompt, intent_analysis):
        """Score contextual relevance to user prompt and intent."""
        segment_text = segment.get('segment_text', '').lower()
        user_prompt_lower = user_prompt.lower()
        
        # Simple keyword matching (can be enhanced with semantic analysis)
        keywords = user_prompt_lower.split()
        matches = sum(1 for keyword in keywords if keyword in segment_text)
        keyword_score = matches / max(len(keywords), 1)
        
        # Intent alignment score
        intent_keywords = intent_analysis.get('selection_criteria', {}).get('primary_factors', [])
        intent_matches = sum(1 for factor in intent_keywords if factor.lower() in segment_text)
        intent_score = intent_matches / max(len(intent_keywords), 1) if intent_keywords else 0.5
        
        return (keyword_score * 0.6 + intent_score * 0.4)
    
    def _score_content_type_alignment(self, segment, content_overview):
        """Score alignment with identified content type."""
        content_type = content_overview.get('content_type', 'unknown')
        segment_text = segment.get('segment_text', '').lower()
        
        # Content type specific scoring
        if content_type == 'tutorial':
            tutorial_keywords = ['learn', 'how', 'step', 'method', 'technique', 'way', 'process']
            matches = sum(1 for keyword in tutorial_keywords if keyword in segment_text)
            return min(1.0, matches / 3)
        elif content_type == 'entertainment':
            entertainment_keywords = ['fun', 'funny', 'amazing', 'incredible', 'wow', 'great']
            matches = sum(1 for keyword in entertainment_keywords if keyword in segment_text)
            return min(1.0, matches / 2)
        else:
            return 0.5
    
    def _score_engagement_potential(self, segment):
        """Score potential for audience engagement."""
        text = segment.get('segment_text', '').lower()
        
        # Engagement indicators
        engagement_words = ['amazing', 'incredible', 'important', 'key', 'best', 'must', 'wow', 'great']
        questions = text.count('?')
        exclamations = text.count('!')
        
        engagement_score = 0.5
        engagement_score += min(0.3, len([w for w in engagement_words if w in text]) * 0.1)
        engagement_score += min(0.1, questions * 0.05)
        engagement_score += min(0.1, exclamations * 0.05)
        
        return min(1.0, engagement_score)
    
    def _score_technical_quality(self, segment):
        """Score technical quality of the segment."""
        # Placeholder for technical quality metrics
        return 0.7
    
    def _score_narrative_value(self, segment, content_overview):
        """Score narrative value within content structure."""
        # Placeholder for narrative analysis
        return 0.6
    
    def _ensures_selection_diversity(self, segment, selected_segments, intent_analysis):
        """Ensure selection diversity to avoid repetitive content."""
        if not selected_segments:
            return True
        
        # Check for content similarity (simplified)
        segment_text = segment.get('segment_text', '').lower()
        for selected in selected_segments:
            selected_text = selected.get('segment_text', '').lower()
            # Simple overlap check
            words_segment = set(segment_text.split())
            words_selected = set(selected_text.split())
            overlap = len(words_segment & words_selected) / max(len(words_segment | words_selected), 1)
            
            if overlap > 0.7:  # Too similar
                return False
        
        return True












# # src/content_analysis/content_analyzer.py
# """Content analysis for identifying optimal short video segments"""

# import numpy as np
# import logging
# import json
# from typing import Dict, List, Tuple, Optional, Any
# from dataclasses import dataclass

# from .prompt_based_analyzer import PromptBasedAnalyzer
# from .comprehensive_segment_generator import ComprehensiveSegmentGenerator
# from ..object_detection.intelligent_object_detector import IntelligentObjectDetector
# from ..object_detection.ai_reframing_processor import AIReframingProcessor, ReframingParameters
# from ..utils.speech_boundary_adjuster import SpeechBoundaryAdjuster

# from ..face_insights.celebrity_index import (
#     load_celebrity_index,
#     actor_coverage_for_segment,
#     compute_celebrity_score,
# )


# @dataclass
# class VideoSegment:
#     """Represents a video segment with metadata."""
#     start_time: float
#     end_time: float
#     quality_score: float
#     engagement_score: float
#     content_type: str
#     description: str
#     confidence: float
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert VideoSegment to dictionary for JSON serialization."""
#         return {
#             'start_time': self.start_time,
#             'end_time': self.end_time,
#             'quality_score': self.quality_score,
#             'engagement_score': self.engagement_score,
#             'content_type': self.content_type,
#             'description': self.description,
#             'confidence': self.confidence
#         }
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> 'VideoSegment':
#         """Create VideoSegment from dictionary."""
#         return cls(
#             start_time=data['start_time'],
#             end_time=data['end_time'],
#             quality_score=data['quality_score'],
#             engagement_score=data['engagement_score'],
#             content_type=data['content_type'],
#             description=data['description'],
#             confidence=data['confidence']
#         )


# class ContentAnalyzer:
#     """
#     Analyzes video content to identify optimal segments for short videos.
#     Enhanced with object detection and AI-powered reframing capabilities.
#     """
    
#     def __init__(self, ollama_client=None, enable_object_detection=True, enable_ai_reframing=True):
#         """Initialize content analyzer."""
#         self.logger = logging.getLogger(__name__)
#         self.ollama_client = ollama_client
        
#         # Initialize specialized analyzers
#         self.prompt_analyzer = PromptBasedAnalyzer(ollama_client)
#         self.segment_generator = ComprehensiveSegmentGenerator()
#         self.boundary_adjuster = SpeechBoundaryAdjuster(self.logger)
        
#         # Initialize object detection and reframing if enabled
#         self.enable_object_detection = enable_object_detection
#         self.enable_ai_reframing = enable_ai_reframing
        
#         if enable_object_detection:
#             self.object_detector = IntelligentObjectDetector()
#             # Import and initialize object-aware zoom processor with base processor
#             from ..smart_zoom.object_aware_zoom import ObjectAwareZoomProcessor
#             from ..smart_zoom.smart_zoom_processor import SmartZoomProcessor
#             base_zoom_processor = SmartZoomProcessor()
#             self.object_aware_zoom_processor = ObjectAwareZoomProcessor(
#                 base_zoom_processor=base_zoom_processor,
#                 object_detector=self.object_detector
#             )
#             self.logger.info("Object detection and object-aware zoom enabled")
#         else:
#             self.object_detector = None
#             self.object_aware_zoom_processor = None
            
#         if enable_ai_reframing:
#             self.ai_reframer = AIReframingProcessor()
#             self.logger.info("AI reframing enabled")
#         else:
#             self.ai_reframer = None


#     def _enhance_segments_with_celebrity_scores(self,
#                                                 segments: List[Dict],
#                                                 celebrity_index_path: str = None,
#                                                 celebrity_index_data: Dict = None,
#                                                 min_coverage_threshold: float = 0.01) -> List[Dict]:
#         """Enhance segments with celebrity coverage and score metadata."""
#         self.logger.info(f"Enhancing {len(segments)} segments with celebrity scores...")
#         try:
#             if celebrity_index_data:
#                 appearances_per_actor, actor_conf = celebrity_index_data
#             elif celebrity_index_path:
#                 if not hasattr(self, '_celebrity_index_cache'):
#                     self._celebrity_index_cache = {}
#                 if celebrity_index_path not in self._celebrity_index_cache:
#                     self._celebrity_index_cache[celebrity_index_path] = load_celebrity_index(celebrity_index_path)
#                 appearances_per_actor, actor_conf = self._celebrity_index_cache[celebrity_index_path]
#             else:
#                 return segments

#             for seg in segments:
#                 per_actor = actor_coverage_for_segment(
#                     seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf
#                 )
#                 if per_actor:
#                     score = compute_celebrity_score(per_actor)
#                     seg['celebrity_score'] = float(score)
#                     self.logger.info(f"Segment {seg['start_time']}-{seg['end_time']}s celebrity score: {seg['celebrity_score']:.4f}")
#                     seg['celebrity_actors'] = sorted(per_actor.items(), key=lambda x: x[1]['coverage'], reverse=True)
#                     seg['has_celebrity'] = seg['celebrity_score'] >= min_coverage_threshold
#                     seg['top_celebrity'] = seg['celebrity_actors'][0][0] if seg['celebrity_actors'] else None
#                 else:
#                     seg['celebrity_score'] = 0.0
#                     seg['celebrity_actors'] = []
#                     seg['has_celebrity'] = False
#                     seg['top_celebrity'] = None

#             self.logger.info(f"Enhanced {len(segments)} segments with celebrity scores (path={celebrity_index_path})")
#             return segments

#         except Exception as e:
#             self.logger.warning(f"Failed to enhance segments with celebrity scores: {e}")
#             return segments


#     def _calculate_combined_score(self, segment: Dict) -> float:
#         """Calculate combined score from prompt match, object detection and celebrity presence."""
#         self.logger.info(f"Calculating combined score for segment {segment.get('start_time', 0)}-{segment.get('end_time', 0)}s")
#         prompt_score = segment.get('prompt_match_score', 0.0)
#         object_relevance = segment.get('object_relevance_score', 0.0)
#         object_confidence = segment.get('object_confidence_score', 0.0)
#         object_prompt_match = segment.get('object_prompt_match_score', 0.0)
#         celebrity_score = segment.get('celebrity_score', 0.0)

#         # Weighted combination — add celebrity weight (0.25) so actor-presence matters
#         base_score = prompt_score * 0.25
#         object_score = (object_relevance * 0.3 + object_confidence * 0.2 + object_prompt_match * 0.4) * 0.4
#         celeb_score = celebrity_score * 90.25

#         # Bonus for segments with multiple relevant objects/actors
#         objects_detected = segment.get('objects_detected', 0)
#         unique_classes = segment.get('unique_object_classes', 0)
#         object_bonus = min(0.1, (objects_detected * 0.03) + (unique_classes * 0.05))

#         combined = base_score + object_score + celeb_score + object_bonus
#         return min(1.0, combined)
#     #     """
#     #     Analyzes video content to identify optimal segments for short videos.
#     #     Enhanced with object detection and AI-powered reframing capabilities.
#     #     """
    
#     # def __init__(self, ollama_client=None, enable_object_detection=True, enable_ai_reframing=True):
#     #     """Initialize content analyzer."""
#     #     self.logger = logging.getLogger(__name__)
#     #     self.ollama_client = ollama_client
        
#     #     # Initialize specialized analyzers
#     #     self.prompt_analyzer = PromptBasedAnalyzer(ollama_client)
#     #     self.segment_generator = ComprehensiveSegmentGenerator()
#     #     self.boundary_adjuster = SpeechBoundaryAdjuster(self.logger)
        
#     #     # Initialize object detection and reframing if enabled
#     #     self.enable_object_detection = enable_object_detection
#     #     self.enable_ai_reframing = enable_ai_reframing
        
#     #     if enable_object_detection:
#     #         self.object_detector = IntelligentObjectDetector()
#     #         # Import and initialize object-aware zoom processor with base processor
#     #         from ..smart_zoom.object_aware_zoom import ObjectAwareZoomProcessor
#     #         from ..smart_zoom.smart_zoom_processor import SmartZoomProcessor
#     #         base_zoom_processor = SmartZoomProcessor()
#     #         self.object_aware_zoom_processor = ObjectAwareZoomProcessor(
#     #             base_zoom_processor=base_zoom_processor,
#     #             object_detector=self.object_detector
#     #         )
#     #         self.logger.info("Object detection and object-aware zoom enabled")
#     #     else:
#     #         self.object_detector = None
#     #         self.object_aware_zoom_processor = None
            
#     #     if enable_ai_reframing:
#     #         self.ai_reframer = AIReframingProcessor()
#     #         self.logger.info("AI reframing enabled")
#     #     else:
#     #         self.ai_reframer = None
    
#     async def analyze_with_user_prompt(self,
#                                      user_prompt: str,
#                                      video_path: str,
#                                      video_info: Dict,
#                                      audio_analysis: Dict,
#                                      scene_analysis: Dict,
#                                      vision_analysis: Optional[Dict] = None,
#                                      target_duration: Tuple[int, int] = (15, 60),
#                                      max_shorts: int = 10,
#                                      ai_reframe: bool = False,
#                                      llm_provider: str = "ollama",
#                                      celebrity_index_path: str = None) -> Dict:
#         """
#         Analyze video content based on user prompt for theme-specific short creation.
#         Enhanced with object detection, AI reframing, and contextual understanding.
        
#         Args:
#             user_prompt: User's prompt describing desired content (e.g., "goals in basketball", "comedy shorts")
#             video_path: Path to the video file
#             video_info: Video metadata
#             audio_analysis: Audio analysis results
#             scene_analysis: Scene break analysis
#             vision_analysis: Vision analysis results (optional)
#             target_duration: Target duration range (min, max) in seconds
#             max_shorts: Maximum number of shorts to generate
#             ai_reframe: Whether to enable AI-powered reframing
#             llm_provider: LLM provider to use ('openai' or 'ollama')
            
#         Returns:
#             Prompt-based analysis results with selected segments and reframing data
#         """
#         try:
#             self.logger.info(f"Analyzing video with user prompt: '{user_prompt}' (AI reframe: {ai_reframe})")
            
#             # PHASE 1 ENHANCEMENT: Content Overview Analysis
#             self.logger.info("Performing content overview analysis for contextual understanding...")
#             content_overview = await self._analyze_content_overview(
#                 audio_analysis, vision_analysis, video_info
#             )
            
#             # PHASE 1 ENHANCEMENT: Enhanced Intent Analysis
#             self.logger.info("Performing enhanced user intent analysis...")
#             intent_analysis = await self._analyze_user_intent_comprehensive(
#                 user_prompt, content_overview, video_info
#             )
            
#             # Step 1: Object Detection Analysis (if enabled)
#             object_detection_results = {}
#             if self.enable_object_detection and self.object_detector:
#                 self.logger.info("Performing object detection analysis...")
                
#                 # Analyze prompt for object detection strategy (with fallback for empty prompts)
#                 effective_prompt = user_prompt if user_prompt and user_prompt.strip() else "detect all objects in the scene"
#                 prompt_analysis = await self.object_detector.analyze_prompt_for_objects(
#                     effective_prompt, self.ollama_client
#                 )
                
#                 # Perform object detection on video
#                 object_detection_results = await self.object_detector.detect_objects_in_video(
#                     video_path, prompt_analysis, sample_rate=1.0
#                 )
                
#                 self.logger.info(f"Object detection completed: {object_detection_results.get('total_detections', 0)} objects detected")
            
#             # Step 2: Generate ALL possible candidate segments
#             self.logger.info("Generating comprehensive candidate segments...")

#             # Decide whether prompt explicitly requests actor-only clips
#             actor_only = False
#             if celebrity_index_path and user_prompt:
#                 try:
#                     appearances_per_actor, _ = load_celebrity_index(celebrity_index_path)
#                     prompt_lower = user_prompt.lower()
#                     for actor in appearances_per_actor.keys():
#                         if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
#                             actor_only = True
#                             break
#                     if actor_only:
#                         self.logger.info("Detected actor-only request from prompt; generating actor-only candidates")
#                 except Exception as e:
#                     self.logger.warning(f"Could not inspect celebrity index for actor-only detection: {e}")

#             all_candidates = self.segment_generator.generate_all_possible_segments(
#                 video_info=video_info,
#                 audio_analysis=audio_analysis,
#                 scene_analysis=scene_analysis,
#                 target_duration=target_duration,
#                 max_total_segments=300,  # Increased for comprehensive analysis
#                 celebrity_index_path=celebrity_index_path,
#                 actor_only=actor_only
#             )
            
#             # DETAILED DEBUG: Log candidate generation results
#             self.logger.info(f"🔍 CANDIDATE SEGMENT GENERATION DEBUG:")
#             self.logger.info(f"📊 Video duration: {video_info.get('duration', 'unknown')}s")
#             self.logger.info(f"📊 Target duration: {target_duration}")
#             self.logger.info(f"📊 Audio segments: {len(audio_analysis.get('transcription', {}).get('segments', []))}")
#             self.logger.info(f"📊 Scene breaks: {len(scene_analysis.get('scene_breaks', []))}")
#             self.logger.info(f"📊 Generated candidates: {len(all_candidates)}")
            
#             if len(all_candidates) == 0:
#                 self.logger.warning("⚠️ NO CANDIDATE SEGMENTS GENERATED!")
#                 self.logger.warning("🔍 Debugging segment generation failure:")
#                 self.logger.warning(f"  - Video info: {video_info}")
#                 self.logger.warning(f"  - Audio transcription segments: {len(audio_analysis.get('transcription', {}).get('segments', []))}")
#                 self.logger.warning(f"  - Scene analysis: {scene_analysis.get('combined_breaks', [])}")
                
#                 # Log first few audio segments if available
#                 audio_segments = audio_analysis.get('transcription', {}).get('segments', [])
#                 if audio_segments:
#                     self.logger.warning(f"  - First 3 audio segments:")
#                     for i, seg in enumerate(audio_segments[:3]):
#                         self.logger.warning(f"    {i}: {seg.get('start', 'no_start')}-{seg.get('end', 'no_end')}s: '{seg.get('text', 'no_text')[:50]}...'")
#                 else:
#                     self.logger.warning(f"  - No audio segments found in transcription")
#             else:
#                 self.logger.info(f"📊 Candidate segments generated successfully:")
#                 for i, candidate in enumerate(all_candidates[:5]):  # Show first 5
#                     self.logger.info(f"  Candidate {i}: {candidate.get('start_time', 'no_start'):.1f}s-{candidate.get('end_time', 'no_end'):.1f}s, text: '{candidate.get('segment_text', 'no_text')[:50]}...'")
#                 if len(all_candidates) > 5:
#                     self.logger.info(f"  ... and {len(all_candidates) - 5} more candidates")
            
#             self.logger.info(f"Generated {len(all_candidates)} comprehensive candidate segments")
            
#             # Step 3: Analyze segments with user prompt (ENHANCED with context and object detection)
#             self.logger.info("Performing context-aware prompt-based segment analysis...")
#             # Pass celebrity_index_path to prompt analyzer via video_info
#             video_info_with_celebrity = dict(video_info) if video_info else {}
#             if celebrity_index_path:
#                 video_info_with_celebrity['celebrity_index_path'] = celebrity_index_path
#             prompt_results = await self.prompt_analyzer.analyze_with_prompt(
#                 user_prompt=user_prompt,
#                 audio_analysis=audio_analysis,
#                 vision_analysis=vision_analysis,
#                 scene_analysis=scene_analysis,
#                 video_info=video_info_with_celebrity,
#                 candidate_segments=all_candidates,
#                 object_detection_results=object_detection_results,
#                 content_overview=content_overview,  # ENHANCED: Pass content overview
#                 intent_analysis=intent_analysis,    # ENHANCED: Pass intent analysis
#                 llm_provider=llm_provider
#             )
#             # Debug raw prompt_results
#             self.logger.debug(f"DEBUG prompt_results: {prompt_results}")
#             if isinstance(prompt_results, dict):
#                 status = prompt_results.get('status')
#                 error = prompt_results.get('error')
#                 self.logger.debug(f"Prompt-based analysis status: {status}, error: {error}")
#             else:
#                 self.logger.error(f"prompt_results is not dict: {type(prompt_results)}")
            
#             # Step 4: AI Reframing Analysis (if enabled)
#             reframing_data = {}
#             if ai_reframe and self.enable_ai_reframing and self.ai_reframer:
#                 self.logger.info("Performing AI reframing analysis...")
                
#                 # Analyze reframing strategy
#                 reframing_strategy = await self.ai_reframer.analyze_reframing_strategy(
#                     user_prompt, object_detection_results, video_info, self.ollama_client
#                 )
                
#                 # Generate reframing plan
#                 if prompt_results['status'] == 'success':
#                     reframing_plan = await self.ai_reframer.generate_reframing_plan(
#                         reframing_strategy, object_detection_results, video_info
#                     )
                    
#                     reframing_data = {
#                         "strategy": reframing_strategy,
#                         "framing_decisions": reframing_plan,
#                         "enabled": True
#                     }
                    
#                     self.logger.info(f"AI reframing plan generated with {len(reframing_plan)} framing decisions")
            
#             # Defensive validation: ensure prompt_results is a proper dictionary
#             if not isinstance(prompt_results, dict):
#                 self.logger.error(f"CRITICAL: prompt_results is not a dictionary, got {type(prompt_results)}: {prompt_results}")
#                 # Force it to be a proper failed result dictionary
#                 prompt_results = {
#                     'status': 'error', 
#                     'error': f'Invalid prompt results format: got {type(prompt_results)} instead of dict', 
#                     'segments': [],
#                     'prompt_analysis': {}
#                 }
            
#             # Ensure all required keys exist with proper types
#             if 'status' not in prompt_results or not isinstance(prompt_results['status'], str):
#                 prompt_results['status'] = 'error'
#             if 'segments' not in prompt_results or not isinstance(prompt_results['segments'], list):
#                 prompt_results['segments'] = []
#             if 'prompt_analysis' not in prompt_results or not isinstance(prompt_results['prompt_analysis'], dict):
#                 prompt_results['prompt_analysis'] = {}
                
#             # Log the prompt_results structure for debugging
#             self.logger.debug(f"Prompt results keys: {list(prompt_results.keys())}")
#             self.logger.debug(f"Prompt results status: {prompt_results.get('status', 'unknown')}")
            
#             # Step 5: Enhanced Phase 1 + Phase 2 Analysis Pipeline
#             if prompt_results.get('status') == 'success':
#                 # Handle both regular segments and climax_segments keys
#                 prompt_segments = prompt_results.get('segments', prompt_results.get('climax_segments', []))
                
#                 # Ensure prompt_segments is a list, not something else
#                 if not isinstance(prompt_segments, list):
#                     self.logger.warning(f"prompt_segments is not a list, got {type(prompt_segments)}: {prompt_segments}")
#                     prompt_segments = []
                
#                 self.logger.info(f"Found {len(prompt_segments)} prompt-matched segments")
                
#                 # Phase 2: Multi-pass analysis pipeline (if available)
#                 if self.ollama_client and content_overview and intent_analysis and len(prompt_segments) > 0:
#                     self.logger.info("Initiating Phase 2 multi-pass analysis pipeline...")
                    
#                     phase2_result = await self._phase2_multipass_analysis(
#                         initial_segments=prompt_segments,
#                         user_prompt=user_prompt,
#                         comprehensive_segments=all_candidates,
#                         audio_analysis=audio_analysis,
#                         vision_analysis=vision_analysis,
#                         scene_analysis=scene_analysis,
#                         content_overview=content_overview,
#                         intent_analysis=intent_analysis,
#                         video_info=video_info,
#                         target_duration=target_duration,
#                         max_shorts=max_shorts
#                     )
                    
#                     if phase2_result['status'] == 'success':
#                         # Apply final object detection and reframing
#                         final_segments = self._select_final_prompt_segments(
#                             phase2_result['segments'],
#                             audio_analysis=audio_analysis,
#                             user_prompt=user_prompt,
#                             object_detection_results=object_detection_results,
#                             reframing_data=reframing_data,
#                             max_shorts=max_shorts,
#                             target_duration=target_duration,
#                             celebrity_index_path=celebrity_index_path,
#                             actor_only=actor_only
#                         )
                        
#                         # Safely extract prompt_analysis with defensive check
#                         self.logger.info("No of prompt matched segments : " + str(len(prompt_segments)))
#                         safe_prompt_analysis = {}
#                         if isinstance(prompt_results, dict) and 'prompt_analysis' in prompt_results:
#                             safe_prompt_analysis = prompt_results['prompt_analysis']
#                             if not isinstance(safe_prompt_analysis, dict):
#                                 safe_prompt_analysis = {}
                        
#                         return {
#                             'status': 'success',
#                             'user_prompt': user_prompt,
#                             'analysis_method': 'phase2_multipass_enhanced',
#                             'total_candidates_analyzed': len(all_candidates),
#                             'prompt_matched_segments': len(prompt_segments),
#                             'phase2_enhanced_segments': len(phase2_result['segments']),
#                             'final_selected_segments': len(final_segments),
#                             'segments': final_segments,
#                             'content_overview': content_overview,
#                             'intent_analysis': intent_analysis,
#                             'multipass_details': phase2_result.get('multipass_details', {}),
#                             'prompt_analysis': safe_prompt_analysis,
#                             'object_detection_results': object_detection_results,
#                             'reframing_data': reframing_data,
#                             'comprehensive_coverage': True,
#                             'ai_reframe_enabled': ai_reframe,
#                             'context_confidence': phase2_result.get('context_confidence', 0.7),
#                             'enhancement_level': 'phase2_multipass'
#                         }
#                     else:
#                         self.logger.warning("Phase 2 analysis failed, falling back to Phase 1 results")
                
#                 # Fallback to Phase 1 analysis
#                 try:
#                     final_segments = self._select_final_prompt_segments(
#                         prompt_segments,
#                         audio_analysis=audio_analysis,
#                         user_prompt=user_prompt,
#                         object_detection_results=object_detection_results,
#                         reframing_data=reframing_data,
#                         max_shorts=max_shorts,
#                         target_duration=target_duration,
#                         celebrity_index_path=celebrity_index_path,
#                         actor_only=actor_only
#                     )
#                 except Exception as e:
#                     self.logger.error(f"Error in _select_final_prompt_segments: {e}")
#                     raise e
                
#                 # Defensive check for intent_analysis before using it
#                 if not isinstance(intent_analysis, dict):
#                     self.logger.warning(f"intent_analysis is not a dict, got {type(intent_analysis)}: {intent_analysis}")
#                     intent_analysis = {}
                
#                 # Defensive check for content_overview before using it
#                 if not isinstance(content_overview, dict):
#                     self.logger.warning(f"content_overview is not a dict, got {type(content_overview)}: {content_overview}")
#                     content_overview = {}
                
#                 # Defensive check for object_detection_results before using it
#                 if not isinstance(object_detection_results, dict):
#                     self.logger.warning(f"object_detection_results is not a dict, got {type(object_detection_results)}: {object_detection_results}")
#                     object_detection_results = {}
                
#                 # Defensive check for reframing_data before using it
#                 if not isinstance(reframing_data, dict):
#                     self.logger.warning(f"reframing_data is not a dict, got {type(reframing_data)}: {reframing_data}")
#                     reframing_data = {}
                
#                 # Safely extract prompt_analysis with defensive check
#                 safe_prompt_analysis = {}
#                 if isinstance(prompt_results, dict) and 'prompt_analysis' in prompt_results:
#                     safe_prompt_analysis = prompt_results['prompt_analysis']
#                     if not isinstance(safe_prompt_analysis, dict):
#                         safe_prompt_analysis = {}
                
#                 return {
#                     'status': 'success',
#                     'user_prompt': user_prompt,
#                     'analysis_method': 'enhanced_contextual_with_objects',
#                     'total_candidates_analyzed': len(all_candidates),
#                     'prompt_matched_segments': len(prompt_segments),
#                     'final_selected_segments': len(final_segments),
#                     'segments': final_segments,
#                     'content_overview': content_overview,  # ENHANCED: Include content overview
#                     'intent_analysis': intent_analysis,    # ENHANCED: Include intent analysis
#                     'prompt_analysis': safe_prompt_analysis,
#                     'object_detection_results': object_detection_results,
#                     'reframing_data': reframing_data,
#                     'comprehensive_coverage': True,
#                     'ai_reframe_enabled': ai_reframe,
#                     'context_confidence': intent_analysis.get('confidence_assessment', {}).get('overall_confidence', 0.5),
#                     'enhancement_level': 'phase1_contextual'
#                 }
#             else:
#                 # Prompt analysis failed, log the error and fall back
#                 error_msg = prompt_results.get('error', 'Unknown error in prompt analysis')
#                 self.logger.warning(f"Prompt analysis failed with error: {error_msg}")
#                 self.logger.warning("Falling back to standard analysis")
#                 return await self._fallback_to_standard_analysis(
#                     all_candidates, video_info, audio_analysis, scene_analysis, 
#                     vision_analysis, target_duration, max_shorts
#                 )
                
#         except Exception as e:
#             # Handle the specific theme_templates corruption error
#             if "'list' object has no attribute 'items'" in str(e):
#                 self.logger.error(f"CRITICAL: theme_templates corruption detected in ContentAnalyzer: {e}")
#                 self.logger.warning("Attempting recovery by reinitializing PromptBasedAnalyzer theme_templates...")
                
#                 # Force recovery in the PromptBasedAnalyzer
#                 try:
#                     if hasattr(self.prompt_analyzer, '_reinitialize_theme_templates'):
#                         self.prompt_analyzer._reinitialize_theme_templates()
#                         self.logger.info("theme_templates recovery successful, retrying analysis...")
                        
#                         # Retry the analysis once after recovery
#                         prompt_results = await self.prompt_analyzer.analyze_with_prompt(
#                             user_prompt=user_prompt,
#                             audio_analysis=audio_analysis,
#                             vision_analysis=vision_analysis,
#                             scene_analysis=scene_analysis,
#                             video_info=video_info,
#                             candidate_segments=all_candidates,
#                             object_detection_results=object_detection_results,
#                             content_overview=content_overview,
#                             intent_analysis=intent_analysis,
#                             llm_provider=llm_provider
#                         )
                        
#                         if prompt_results['status'] == 'success':
#                             self.logger.info("Analysis retry successful after theme_templates recovery")
#                             # Continue with the normal flow
#                             prompt_segments = prompt_results.get('segments', [])
#                             final_segments = self._select_final_prompt_segments(
#                                 prompt_segments,
#                                 audio_analysis=audio_analysis,
#                                 user_prompt=user_prompt,
#                                 object_detection_results=object_detection_results,
#                                 reframing_data={},  # Skip reframing on retry
#                                 max_shorts=max_shorts,
#                                 target_duration=target_duration,
#                                 celebrity_index_path=celebrity_index_path
#                             )
                            
#                             return {
#                                 'status': 'success',
#                                 'user_prompt': user_prompt,
#                                 'analysis_method': 'recovered_after_corruption',
#                                 'total_candidates_analyzed': len(all_candidates),
#                                 'prompt_matched_segments': len(prompt_segments),
#                                 'final_selected_segments': len(final_segments),
#                                 'segments': final_segments,
#                                 'recovery_applied': True,
#                                 'enhancement_level': 'recovered_basic'
#                             }
#                         else:
#                             self.logger.warning("Analysis retry failed after recovery, falling back to standard analysis")
                    
#                 except Exception as retry_error:
#                     self.logger.error(f"Recovery retry failed: {retry_error}")
                
#             self.logger.error(f"Error in prompt-based analysis2: {e}")
#             self.logger.error(f"Exception type: {type(e).__name__}")
#             import traceback
#             self.logger.error(f"Traceback: {traceback.format_exc()}")
#             # Fallback to standard analysis - CRITICAL FIX: preserve all_candidates
#             return await self._fallback_to_standard_analysis(
#                 all_candidates, video_info, audio_analysis, scene_analysis, 
#                 vision_analysis, target_duration, max_shorts
#             )
    
#     def _select_final_prompt_segments(self,
#                                     prompt_segments: List[Dict],
#                                     audio_analysis: Dict = None,
#                                     user_prompt: str = "",
#                                     object_detection_results: Dict = None,
#                                     reframing_data: Dict = None,
#                                     max_shorts: int = 10,
#                                     target_duration: Tuple[int, int] = (15, 60),
#                                     celebrity_index_path: str = None,
#                                     actor_only: bool = False) -> List[Dict]:
#         """
#         Select final segments from prompt-matched segments with quality and diversity.
#         Enhanced with object detection and AI reframing data.
        
#         Args:
#             prompt_segments: Segments matched by prompt analysis
#             audio_analysis: Audio analysis with transcription for boundary adjustment
#             user_prompt: User's search prompt for context
#             object_detection_results: Object detection results
#             reframing_data: AI reframing data
#             max_shorts: Maximum number of shorts
#             target_duration: Target duration range
#             celebrity_index_path: Optional path to celebrity result JSON for per-segment scoring
            
#         Returns:
#             Final selected segments with enhanced metadata
#         """
#         if not prompt_segments:
#             return []

#         # Enhance segments with celebrity scores (if available) before object scoring
#         if celebrity_index_path:
#             try:
#                 prompt_segments = self._enhance_segments_with_celebrity_scores(prompt_segments, celebrity_index_path=celebrity_index_path)
#                 self.logger.info("No of Enhanced prompt segments with celebrity data before final selection: " + str(len(prompt_segments)))
#             except Exception as e:
#                 self.logger.warning(f"Failed to enhance prompt segments with celebrity data: {e}")

#         # If user explicitly requested actor(s), enforce strict overlap with celebrity timestamps
#         actor_matches = []
#         if celebrity_index_path and user_prompt:
#             try:
#                 prompt_lower = user_prompt.lower()
#                 appearances_per_actor, actor_conf = load_celebrity_index(celebrity_index_path)
#                 for actor in appearances_per_actor.keys():
#                     if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
#                         actor_matches.append(actor)

#                 if actor_matches:
#                     self.logger.info(f"Actor(s) requested in prompt: {actor_matches}. Enforcing overlap with celebrity timestamps.")
#                     self.logger.info(f"No of prompt_segments for requested actors before filtering:{len(prompt_segments)}")  
#                     try:
#                         from ..face_insights.celebrity_index import actor_coverage_for_segment
#                     except Exception:
#                         from src.face_insights.celebrity_index import actor_coverage_for_segment

#                     # self.logger.info(f"No of prompt_segments for requested actors before filtering:{len(prompt_segments)}")    

#                     filtered = []
#                     for seg in prompt_segments:
#                         per_actor_cov = actor_coverage_for_segment(seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf)
#                         # Accept segment if any requested actor has coverage > 0
#                         if any(a for a in per_actor_cov.keys() if a.lower() in [am.lower() for am in actor_matches]):
#                             # Boost prompt score to prefer these
#                             seg['prompt_match_score'] = max(seg.get('prompt_match_score', 0.0), 0.9)
#                             filtered.append(seg)

#                     if filtered:
#                         prompt_segments = filtered
#                         self.logger.info(f"Strictly filtered prompt_segments to {len(prompt_segments)} segments overlapping requested actors")
#                     else:
#                         self.logger.warning("Strict actor overlap filtering found no segments; falling back to non-strict actor-aware selection")
#             except Exception as e:
#                 self.logger.warning(f"Could not enforce strict actor overlap filtering: {e}")

#         # If the user explicitly requested an actor, filter prompt_segments to those with celebrity appearances
#         actor_matches = []
#         if celebrity_index_path and user_prompt:
#             try:
#                 appearances_per_actor, _ = load_celebrity_index(celebrity_index_path)
#                 prompt_lower = user_prompt.lower()
#                 for actor in appearances_per_actor.keys():
#                     if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
#                         actor_matches.append(actor)
#                 if actor_matches:
#                     self.logger.info(f"Actor-only prompt detected for: {actor_matches}. Filtering to celebrity-overlapping segments.")
#                     filtered = []
#                     for seg in prompt_segments:
#                         ca = seg.get('celebrity_actors', [])
#                         actors_in_seg = []
#                         if isinstance(ca, (list, tuple)):
#                             for a in ca:
#                                 if isinstance(a, (list, tuple)) and len(a) > 0:
#                                     actors_in_seg.append(str(a[0]).lower())
#                                 elif isinstance(a, dict):
#                                     actors_in_seg.append(str(a.get('name','')).lower())
#                                 else:
#                                     actors_in_seg.append(str(a).lower())
#                         elif isinstance(ca, dict):
#                             actors_in_seg.append(str(ca.get('name','')).lower())

#                         if seg.get('has_celebrity') and any(a.lower() in actors_in_seg for a in actor_matches):
#                             # Boost prompt score so actor segments are favored in selection
#                             seg['prompt_match_score'] = max(seg.get('prompt_match_score', 0.0), 0.9)
#                             filtered.append(seg)

#                     if filtered:
#                         prompt_segments = filtered
#                         self.logger.info(f"Filtered prompt segments to {len(prompt_segments)} actor-overlapping segments")

#                         # If the caller explicitly requested actor-only clips, select up to max_shorts
#                         # distinct appearance-based segments (do not aggressively dedupe overlapping timestamps)
#                         if actor_only:
#                             try:
#                                 groups = {}

#                                 # Helper: find an appearance timestamp that falls within segment or nearest within margin
#                                 def _find_closest_appearance(seg, actors, appearances_map, margin=15.0):
#                                     # Prefer exact inclusion inside seg
#                                     for a in actors:
#                                         for t in appearances_map.get(a, []):
#                                             if seg['start_time'] <= float(t) <= seg['end_time']:
#                                                 return int(t)
#                                     # Otherwise pick nearest timestamp within margin to center
#                                     center = (seg['start_time'] + seg['end_time']) / 2.0
#                                     best = None
#                                     best_dist = margin + 10
#                                     for a in actors:
#                                         for t in appearances_map.get(a, []):
#                                             dist = abs(float(t) - center)
#                                             if dist <= margin and dist < best_dist:
#                                                 best_dist = dist
#                                                 best = int(t)
#                                     return best

#                                 # Build groups keyed by appearance timestamp (if found) or segment center rounded
#                                 for seg in filtered:
#                                     key_ts = None
#                                     # Try to find a requested actor presence
#                                     if 'celebrity_actors' in seg and seg.get('celebrity_actors'):
#                                         # Normalize actor names for matching
#                                         seg_actor_names = []
#                                         for a in seg.get('celebrity_actors', []):
#                                             if isinstance(a, (list, tuple)) and len(a) > 0:
#                                                 seg_actor_names.append(str(a[0]))
#                                             elif isinstance(a, dict):
#                                                 seg_actor_names.append(str(a.get('name', '')))
#                                             else:
#                                                 seg_actor_names.append(str(a))
#                                         seg_actor_names = [s for s in seg_actor_names if s]
#                                         key_ts = _find_closest_appearance(seg, [am for am in actor_matches], appearances_per_actor)

#                                     if key_ts is None:
#                                         # fallback to appearance(s) that overlap final segment region
#                                         key_ts = _find_closest_appearance(seg, actor_matches, appearances_per_actor)

#                                     if key_ts is None:
#                                         # ultimate fallback: use rounded center timestamp
#                                         key_ts = int(round((seg['start_time'] + seg['end_time']) / 2.0))

#                                     groups.setdefault(key_ts, []).append(seg)

#                                 # Choose best segment per appearance timestamp (by celebrity_score then prompt_match_score)
#                                 selected = []
#                                 for ts in sorted(groups.keys()):
#                                     group = groups[ts]
#                                     best = sorted(group, key=lambda x: (x.get('celebrity_score', 0.0), x.get('prompt_match_score', 0.0)), reverse=True)[0]
#                                     selected.append(best)

#                                 # If not enough unique appearances, fill remaining slots with other top actor-overlapping segments
#                                 if len(selected) < max_shorts:
#                                     remaining = [s for s in sorted(filtered, key=lambda x: (x.get('celebrity_score', 0.0), x.get('prompt_match_score', 0.0)), reverse=True) if s not in selected]
#                                     for s in remaining:
#                                         if len(selected) >= max_shorts:
#                                             break
#                                         selected.append(s)

#                                 # At this point we have one candidate per appearance timestamp (best per appearance)
#                                 # Now enforce temporal diversity and uniqueness while filling up to `max_shorts`.

#                                 def _overlap_seconds(a, b):
#                                     s1, e1 = a['start_time'], a['end_time']
#                                     s2, e2 = b['start_time'], b['end_time']
#                                     inter = max(0.0, min(e1, e2) - max(s1, s2))
#                                     return inter

#                                 def _overlap_ratio(a, b):
#                                     inter = _overlap_seconds(a, b)
#                                     if inter <= 0:
#                                         return 0.0
#                                     # Normalize by min duration so small segments count as high overlap
#                                     return inter / min(a.get('duration', max(1.0, a['end_time'] - a['start_time'])), b.get('duration', max(1.0, b['end_time'] - b['start_time'])))

#                                 # Candidate pool ordered by celebrity_score and prompt_match_score
#                                 candidates = sorted(selected + [s for s in filtered if s not in selected],
#                                                     key=lambda x: (x.get('celebrity_score', 0.0), x.get('prompt_match_score', 0.0)),
#                                                     reverse=True)

#                                 # Greedy selection with progressive gap relaxation
#                                 chosen = []
#                                 # Compute a coverage-aware initial min_gap (based on candidate span and requested count)
#                                 try:
#                                     cand_starts = [s['start_time'] for s in candidates]
#                                     cand_ends = [s['end_time'] for s in candidates]
#                                     cand_span = max(cand_ends) - min(cand_starts) if cand_starts and cand_ends else 0.0
#                                     min_gap = max(15.0, (cand_span / max_shorts) if max_shorts > 0 else 15.0)
#                                 except Exception:
#                                     min_gap = 15.0

#                                 def _is_conflicting(seg, chosen_list, gap):
#                                     for c in chosen_list:
#                                         # Consider both gap distance and overlap
#                                         if _overlap_seconds(seg, c) > 0:
#                                             if _overlap_ratio(seg, c) > 0.5:
#                                                 return True
#                                             # Allow small overlaps but enforce a minimum separation
#                                         center_gap = abs(((seg['start_time'] + seg['end_time']) / 2.0) - ((c['start_time'] + c['end_time']) / 2.0))
#                                         if center_gap < gap:
#                                             return True
#                                     return False

#                                 # Iteratively try to pick up to max_shorts using decreasing gap thresholds
#                                 gap_values = [min_gap, max(min_gap/2, 5.0), 3.0, 1.0, 0.0]
#                                 for gap in gap_values:
#                                     for cand in candidates:
#                                         if len(chosen) >= max_shorts:
#                                             break
#                                         if cand in chosen:
#                                             continue
#                                         if not _is_conflicting(cand, chosen, gap):
#                                             chosen.append(cand)
#                                     if len(chosen) >= max_shorts:
#                                         break

#                                 # Final fallback: if still not enough, fill with top candidates regardless of gap (but unique)
#                                 if len(chosen) < max_shorts:
#                                     for cand in candidates:
#                                         if len(chosen) >= max_shorts:
#                                             break
#                                         if cand not in chosen:
#                                             chosen.append(cand)

#                                 # Trim to requested count
#                                 chosen = chosen[:max_shorts]

#                                 # Adjust boundaries to natural speech breaks if transcription is available
#                                 if chosen and audio_analysis and 'transcription' in audio_analysis:
#                                     chosen = self.boundary_adjuster.batch_adjust_segments(
#                                         segments=chosen,
#                                         transcription=audio_analysis['transcription'],
#                                         user_prompt=user_prompt
#                                     )

#                                 self.logger.info(f"Actor-only mode: selected {len(chosen)} actor-overlapping segments (up to max_shorts={max_shorts})")
#                                 return chosen

#                             except Exception as e:
#                                 self.logger.warning(f"Error while computing actor-only selection: {e}")
#                                 # Fallback to previous simple unique behavior if grouping failed
#                                 unique = []
#                                 seen = set()
#                                 for s in filtered:
#                                     key = (round(s.get('start_time', 0), 2), round(s.get('end_time', 0), 2))
#                                     if key not in seen:
#                                         seen.add(key)
#                                         unique.append(s)
#                                 unique.sort(key=lambda x: x.get('start_time', 0))

#                                 if unique and audio_analysis and 'transcription' in audio_analysis:
#                                     unique = self.boundary_adjuster.batch_adjust_segments(
#                                         segments=unique,
#                                         transcription=audio_analysis['transcription'],
#                                         user_prompt=user_prompt
#                                     )

#                                 self.logger.info(f"Actor-only fallback: returning {len(unique)} actor-overlapping segments")
#                                 return unique

#                     else:
#                         self.logger.warning("No prompt-matched segments overlapped requested actor(s); continuing with unfiltered set")
#             except Exception as e:
#                 self.logger.warning(f"Could not load celebrity index for actor filtering: {e}")

#         # Enhance segments with object detection scores
#         if object_detection_results:
#             prompt_segments = self._enhance_segments_with_object_scores(
#                 prompt_segments, object_detection_results
#             )
        
#         # Ensure segments have a usable prompt_match_score (derive from contextual/heuristic scores when absent)
#         prompt_segments = self._ensure_prompt_scores(prompt_segments)
        
#         # Sort by combined score (prompt match + object relevance)
#         sorted_segments = sorted(prompt_segments, 
#                                key=lambda x: self._calculate_combined_score(x), 
#                                reverse=True)
        
#         # Apply quality filtering
#         min_quality_threshold = 0.05
#         if actor_matches:
#             # When user explicitly requested actor(s), prefer segments with celebrity coverage
#             quality_filtered = [seg for seg in sorted_segments if seg.get('has_celebrity')]
#             if not quality_filtered:
#                 # Fallback to low threshold if no celeb-overlapping segments are available
#                 quality_filtered = [seg for seg in sorted_segments if seg.get('prompt_match_score', 0) >= 0.01]
#         else:
#             quality_filtered = [
#                 seg for seg in sorted_segments 
#                 if seg.get('prompt_match_score', 0) >= min_quality_threshold
#             ]
        
#         # Log segment scores for debugging
#         if sorted_segments:
#             top_scores = [seg.get('prompt_match_score', 0) for seg in sorted_segments[:5]]
#             self.logger.info(f"Top 5 segment scores: {top_scores}")
#             self.logger.info(f"Quality filtered: {len(quality_filtered)}/{len(sorted_segments)} segments (threshold: {min_quality_threshold})")
        
#         # If not enough high-quality segments, lower threshold
#         if len(quality_filtered) < max_shorts // 2:
#             min_quality_threshold = 0.01
#             quality_filtered = [
#                 seg for seg in sorted_segments 
#                 if seg.get('prompt_match_score', 0) >= min_quality_threshold
#             ]
#             self.logger.info(f"Lowered threshold to {min_quality_threshold}, now have {len(quality_filtered)} segments")
        
#         # Apply diversity filtering (avoid temporal clustering) using coverage-aware center-gap enforcement
#         final_segments = []
#         used_centers = []

#         # Compute coverage span from available prompt_segments to determine reasonable spacing
#         try:
#             starts = [s['start_time'] for s in prompt_segments]
#             ends = [s['end_time'] for s in prompt_segments]
#             coverage_span = max(ends) - min(starts) if starts and ends else 0.0
#         except Exception:
#             coverage_span = 0.0

#         # Desired minimum gap: at least 15s, or coverage_span / max_shorts if that yields larger spacing
#         desired_min_gap = max(15.0, (coverage_span / max_shorts) if max_shorts > 0 else 15.0)

#         # Greedy selection by score while enforcing center gap and content diversity
#         for segment in quality_filtered:
#             if len(final_segments) >= max_shorts:
#                 break

#             start_time = segment['start_time']
#             end_time = segment['end_time']
#             center = (start_time + end_time) / 2.0

#             # Check temporal distance to already selected centers
#             too_close = any(abs(center - c) < desired_min_gap for c in used_centers)

#             # Also check content similarity against already selected segments to avoid near-duplicates
#             similar_content = False
#             for sel in final_segments:
#                 try:
#                     if self._calculate_content_similarity(segment, sel) > 0.75:
#                         similar_content = True
#                         break
#                 except Exception:
#                     # Be conservative if similarity calculation fails
#                     continue

#             if not too_close and not similar_content:
#                 final_segments.append(segment)
#                 used_centers.append(center)

#         # If we still don't have enough, relax constraints gradually to fill slots
#         if len(final_segments) < max_shorts:
#             # First relax the minimum gap to 2/3 of desired gap and allow more similar content
#             relaxed_gap = desired_min_gap * 0.66
#             for segment in quality_filtered:
#                 if len(final_segments) >= max_shorts:
#                     break
#                 if segment in final_segments:
#                     continue
#                 start_time = segment['start_time']
#                 end_time = segment['end_time']
#                 center = (start_time + end_time) / 2.0
#                 too_close = any(abs(center - c) < relaxed_gap for c in used_centers)
#                 if not too_close:
#                     final_segments.append(segment)
#                     used_centers.append(center)

#         if len(final_segments) < max_shorts:
#             # Final relaxation: allow any remaining top segments until we reach max_shorts
#             for segment in quality_filtered:
#                 if len(final_segments) >= max_shorts:
#                     break
#                 if segment not in final_segments:
#                     final_segments.append(segment)

#         # Ensure final segments are sorted by start time for downstream processing
#         final_segments.sort(key=lambda x: x['start_time'])
        
#         # ENHANCEMENT: Adjust segment boundaries to natural speech breaks
#         if final_segments and audio_analysis and 'transcription' in audio_analysis:
#             self.logger.info(f"🎬 Adjusting {len(final_segments)} segment boundaries to natural speech breaks...")
#             transcription = audio_analysis['transcription']
#             adjusted_segments = self.boundary_adjuster.batch_adjust_segments(
#                 segments=final_segments,
#                 transcription=transcription,
#                 user_prompt=user_prompt
#             )
#             final_segments = adjusted_segments
#             self.logger.info(f"✅ Boundary adjustment complete")
        
#         # Enhanced logging for debugging
#         self.logger.info(f"Final segment selection: {len(final_segments)} segments selected from {len(prompt_segments)} prompt matches")

#         # If actor matches were requested, ensure final segments overlap requested actor timestamps
#         if actor_matches:
#             try:
#                 appearances_per_actor, actor_conf = load_celebrity_index(celebrity_index_path)
#                 try:
#                     from ..face_insights.celebrity_index import actor_coverage_for_segment
#                 except Exception:
#                     from src.face_insights.celebrity_index import actor_coverage_for_segment

#                 final_actor_filtered = []
#                 for seg in final_segments:
#                     per_actor_cov = actor_coverage_for_segment(seg['start_time'], seg['end_time'], appearances_per_actor, actor_conf)
#                     if any(a for a in per_actor_cov.keys() if a.lower() in [am.lower() for am in actor_matches]):
#                         final_actor_filtered.append(seg)

#                 if final_actor_filtered:
#                     final_segments = final_actor_filtered
#                     self.logger.info(f"Filtered final_segments to {len(final_segments)} actor-overlapping segments")
#                 else:
#                     self.logger.warning("Final actor overlap check found no segments; keeping current final selection")
#             except Exception as e:
#                 self.logger.warning(f"Could not perform final actor overlap filtering: {e}")

#         # CRITICAL FIX: Add intelligent fallback when no segments are selected
#         if len(final_segments) == 0 and len(sorted_segments) > 0:
#             self.logger.warning("No segments passed quality filter. Implementing intelligent fallback...")
            
#             # Fallback Strategy 1: Select highest-scoring segments regardless of threshold
#             top_segments = sorted_segments[:max_shorts]
            
#             # Fallback Strategy 2: Ensure we select the most engaging content
#             # Score segments by engagement rather than just prompt matching
#             for segment in top_segments:
#                 # Boost engagement score for fallback selection
#                 engagement_score = self._calculate_engagement_score(segment)
#                 segment['fallback_score'] = engagement_score
#                 segment['prompt_match_score'] = max(segment.get('prompt_match_score', 0), engagement_score)
            
#             # Re-sort by enhanced scores
#             top_segments.sort(key=lambda x: x.get('fallback_score', 0), reverse=True)
            
#             # Apply temporal diversity for fallback segments
#             fallback_segments = []
#             used_ranges = []
#             min_gap = 10.0  # Reduced gap for fallback
            
#             for segment in top_segments:
#                 if len(fallback_segments) >= max_shorts:
#                     break
                    
#                 start_time = segment['start_time']
#                 end_time = segment['end_time']
                
#                 # Check for conflicts with reduced constraints
#                 conflict = False
#                 for used_start, used_end in used_ranges:
#                     if not (end_time <= used_start + min_gap or start_time >= used_end - min_gap):
#                         conflict = True
#                         break
                
#                 if not conflict:
#                     fallback_segments.append(segment)
#                     used_ranges.append((start_time, end_time))
            
#             final_segments = fallback_segments
#             self.logger.info(f"Fallback selection applied: {len(final_segments)} engaging segments selected")
        
#         if final_segments:
#             avg_score = sum(seg.get('prompt_match_score', 0) for seg in final_segments) / len(final_segments)
#             self.logger.info(f"Selected segments average score: {avg_score:.3f}")
#             self.logger.info(f"Selected segments time ranges: {[(seg['start_time'], seg['end_time']) for seg in final_segments]}")
        
#         return final_segments
    
#     async def _fallback_to_standard_analysis(self,
#                                           all_candidates: List[Dict],
#                                           video_info: Dict,
#                                           audio_analysis: Dict,
#                                           scene_analysis: Dict,
#                                           vision_analysis: Optional[Dict],
#                                           target_duration: Tuple[int, int],
#                                           max_shorts: int) -> Dict:
#         """Fallback to standard segment analysis if prompt analysis fails."""
#         try:
#             # Use existing method or generate segments if candidates not available
#             if not all_candidates:
#                 all_candidates = self.segment_generator.generate_all_possible_segments(
#                     video_info=video_info,
#                     audio_analysis=audio_analysis,
#                     scene_analysis=scene_analysis,
#                     target_duration=target_duration,
#                     max_total_segments=200
#                 )
            
#             # Score segments using standard approach
#             scored_segments = []
#             for segment in all_candidates:
#                 scored_segment = self._score_segment(segment, audio_analysis, None)
#                 scored_segments.append(scored_segment)
            
#             # Apply vision enhancement if available
#             if vision_analysis:
#                 scored_segments = self._enhance_segments_with_vision(scored_segments, vision_analysis)
            
#             # Select final segments
#             final_segments = self._select_diverse_content(scored_segments, max_shorts)
            
#             return {
#                 'status': 'success',
#                 'analysis_method': 'standard_comprehensive_fallback',
#                 'total_candidates_analyzed': len(all_candidates),
#                 'final_selected_segments': len(final_segments),
#                 'segments': final_segments,
#                 'comprehensive_coverage': True,
#                 'fallback_reason': 'prompt_analysis_unavailable'
#             }
            
#         except Exception as e:
#             self.logger.error(f"Fallback analysis also failed: {e}")
#             return {
#                 'status': 'error',
#                 'error': str(e),
#                 'segments': []
#             }
    
#     def get_supported_themes(self) -> Dict[str, str]:
#         """
#         Get list of supported themes for prompt-based analysis.
        
#         Returns:
#             Dictionary of theme names and descriptions
#         """
#         try:
#             return self.prompt_analyzer.get_supported_themes()
#         except AttributeError as e:
#             # Handle the specific 'list' object has no attribute 'items' error
#             if "'list' object has no attribute 'items'" in str(e):
#                 self.logger.error(f"CRITICAL: theme_templates corruption detected in ContentAnalyzer: {e}")
#                 self.logger.warning("Attempting to recover by accessing PromptBasedAnalyzer recovery methods")
                
#                 # Force recovery in the PromptBasedAnalyzer
#                 if hasattr(self.prompt_analyzer, '_reinitialize_theme_templates'):
#                     self.prompt_analyzer._reinitialize_theme_templates()
#                     return self.prompt_analyzer.get_supported_themes()
#                 else:
#                     # Fallback themes if recovery fails
#                     return {
#                         'general': 'Generally engaging and interesting content',
#                         'action': 'High-energy, dynamic, action-packed moments',
#                         'educational': 'Informative, instructional, educational content',
#                         'emotional': 'Emotionally engaging, touching, inspiring moments',
#                         'comedy': 'Humorous, entertaining, light-hearted moments'
#                     }
#             else:
#                 # Re-raise other AttributeErrors
#                 raise
#         except Exception as e:
#             # Catch any other exceptions and provide fallback
#             self.logger.error(f"Error in get_supported_themes: {e}")
#             return {
#                 'general': 'Generally engaging and interesting content',
#                 'action': 'High-energy, dynamic, action-packed moments', 
#                 'educational': 'Informative, instructional, educational content',
#                 'emotional': 'Emotionally engaging, touching, inspiring moments',
#                 'comedy': 'Humorous, entertaining, light-hearted moments'
#             }
    
#     def identify_short_segments(self, 
#                               video_info: Dict,
#                               audio_analysis: Dict,
#                               ollama_analysis: Optional[Dict],
#                               scene_analysis: Dict,
#                               target_duration: Tuple[int, int] = (60, 120),
#                               max_shorts: int = 10) -> List[Dict]:
#         """
#         Identify optimal segments for short videos with content diversity.
        
#         Args:
#             video_info: Video metadata
#             audio_analysis: Audio analysis results
#             ollama_analysis: AI analysis results (optional)
#             scene_analysis: Scene break analysis
#             target_duration: Target duration range (min, max) in seconds
#             max_shorts: Maximum number of shorts to generate
            
#         Returns:
#             List of optimal segments with diverse content
#         """
#         try:
#             min_duration, max_duration = target_duration
#             self.logger.info(f"Identifying segments with target duration range: {min_duration}s - {max_duration}s")
            
#             # Detect pre-roll content for filtering
#             pre_roll_end = self._detect_pre_roll_content(audio_analysis, video_info['duration'])
#             self.logger.info(f"Detected pre-roll content ending at {pre_roll_end:.2f}s")
            
#             # Get all potential break points, filtering out pre-roll
#             break_points = self._get_all_break_points(scene_analysis, audio_analysis, pre_roll_end)
            
#             # ADD VISUAL SEGMENT DETECTION for silent/action content
#             self.logger.info("Detecting visually interesting segments (including silent content)")
#             visual_segments = self._identify_visual_segments(
#                 video_info, audio_analysis, scene_analysis, target_duration, max_shorts, pre_roll_end, []
#             )
            
#             # Generate candidate segments, avoiding pre-roll content
#             candidates = self._generate_candidate_segments(
#                 break_points, 
#                 video_info['duration'], 
#                 target_duration,
#                 pre_roll_end
#             )
            
#             # MERGE VISUAL SEGMENTS with audio-based candidates
#             if visual_segments:
#                 self.logger.info(f"Adding {len(visual_segments)} visual segments to {len(candidates)} audio-based candidates")
#                 candidates.extend(visual_segments)
#                 # Remove duplicates based on similar timing
#                 candidates = self._deduplicate_segments(candidates, time_threshold=5.0)
            
#             # Score each segment using HYBRID approach (audio + visual)
#             scored_segments = []
#             for segment in candidates:
#                 # Use hybrid scoring if visual segments are available
#                 if visual_segments and any(vs.get('is_visual_priority') for vs in visual_segments):
#                     score = self._score_hybrid_segments([segment], audio_analysis, scene_analysis)[0]
#                 else:
#                     score = self._score_segment(segment, audio_analysis, ollama_analysis)
#                 scored_segments.append(score)
            
#             # Sort by quality score
#             scored_segments.sort(key=lambda x: x['quality_score'], reverse=True)
            
#             # Remove overlapping segments
#             non_overlapping = self._remove_overlapping_segments(scored_segments)
            
#             # Apply content diversity selection
#             diverse_segments = self._select_diverse_content(non_overlapping, max_shorts)
            
#             # Final sort by start time
#             diverse_segments.sort(key=lambda x: x['start_time'])
            
#             return diverse_segments
            
#         except Exception as e:
#             self.logger.error(f"Content analysis failed: {e}")
#             return []
    
#     def _detect_pre_roll_content(self, audio_analysis: Dict, video_duration: float) -> float:
#         """
#         Detect pre-roll content (intro music, silence, generic greetings) at the beginning of video.
        
#         Args:
#             audio_analysis: Audio analysis results
#             video_duration: Total video duration in seconds
            
#         Returns:
#             Timestamp where pre-roll content ends (0 if none detected)
#         """
#         # Default pre-roll threshold (10% of video or 30 seconds, whichever is less)
#         pre_roll_threshold = min(video_duration * 0.1, 30.0)
        
#         # Check for explicitly detected pre-roll in audio analysis
#         likely_preroll = audio_analysis.get('zoom_analysis', {}).get('likely_preroll', [])
#         if likely_preroll:
#             # Find the end time of the last pre-roll segment
#             last_preroll_end = max(segment['end'] for segment in likely_preroll)
#             return last_preroll_end
        
#         # Check for music segments at beginning
#         music_segments = audio_analysis.get('zoom_analysis', {}).get('music_segments', [])
#         for segment in music_segments:
#             if segment['start'] < pre_roll_threshold and segment['end'] < pre_roll_threshold * 1.2:
#                 # Found music in the beginning, likely intro music
#                 return segment['end']
        
#         # Check for silence periods at beginning
#         silence_segments = audio_analysis.get('zoom_analysis', {}).get('silence_segments', [])
#         for segment in silence_segments:
#             if segment['start'] < pre_roll_threshold and segment['end'] < pre_roll_threshold * 1.2:
#                 # Found silence in the beginning, could be pre-roll
#                 return segment['end']
        
#         # Check for standard introductory phrases in first segments
#         transcription = audio_analysis.get('transcription', {})
#         segments = transcription.get('segments', [])
        
#         intro_phrases = [
#             'welcome to', 'hello everyone', 'hi everyone', 'hey guys', 
#             'welcome back', 'in this video', 'today we\'re going to'
#         ]
        
#         for segment in segments:
#             if segment['start'] < pre_roll_threshold:
#                 text_lower = segment['text'].lower()
#                 if any(phrase in text_lower for phrase in intro_phrases):
#                     # Found intro greeting, consider it pre-roll
#                     return segment['end']
        
#         # If no clear pre-roll detected, return 0
#         return 0.0
    
#     def _get_all_break_points(self, scene_analysis: Dict, audio_analysis: Dict, pre_roll_end: float = 0.0) -> List[float]:
#         """
#         Get all potential break points from various sources, filtering out pre-roll.
        
#         Args:
#             scene_analysis: Scene break analysis
#             audio_analysis: Audio analysis results
#             pre_roll_end: Timestamp where pre-roll content ends
            
#         Returns:
#             List of filtered break points
#         """
#         break_points = set()
        
#         # Add scene breaks, filtering out pre-roll (VISUAL PRIORITY)
#         for break_point in scene_analysis.get('combined_breaks', []):
#             timestamp = break_point['timestamp']
#             if timestamp > pre_roll_end:
#                 break_points.add(timestamp)
        
#         # Add PURE VISUAL breaks (important for silent scenes)
#         for break_point in scene_analysis.get('scene_breaks', []):
#             timestamp = break_point['timestamp']
#             if timestamp > pre_roll_end:
#                 break_points.add(timestamp)
                
#         # Add audio breaks, filtering out pre-roll
#         for break_point in audio_analysis.get('zoom_analysis', {}).get('recommended_cuts', []):
#             timestamp = break_point['timestamp']
#             if timestamp > pre_roll_end:
#                 break_points.add(timestamp)
        
#         # Add silence periods, filtering out pre-roll (these could be visually interesting)
#         transcription = audio_analysis.get('transcription', {})
#         for silence in transcription.get('silence_periods', []):
#             start, end = silence
#             if end > pre_roll_end:
#                 # Only add if the silence period is after pre-roll
#                 break_points.add(max(start, pre_roll_end))  # Ensure break point is after pre-roll
#                 break_points.add(end)
                
#         # ADD MOTION-BASED BREAKS for action sequences
#         if 'motion_analysis' in scene_analysis:
#             for motion_break in scene_analysis['motion_analysis'].get('high_motion_segments', []):
#                 if motion_break['start'] > pre_roll_end:
#                     break_points.add(motion_break['start'])
#                     break_points.add(motion_break['end'])
#                 break_points.add(end)
        
#         return sorted(list(break_points))
    
#     def _deduplicate_segments(self, segments: List[Dict], time_threshold: float = 5.0) -> List[Dict]:
#         """
#         Remove duplicate segments that are too similar in timing.
        
#         Args:
#             segments: List of segment dictionaries
#             time_threshold: Maximum time difference (seconds) to consider segments as duplicates
            
#         Returns:
#             Deduplicated list of segments
#         """
#         if not segments:
#             return segments
            
#         # Sort segments by start time
#         sorted_segments = sorted(segments, key=lambda x: x['start_time'])
#         deduplicated = [sorted_segments[0]]
        
#         for segment in sorted_segments[1:]:
#             # Check if this segment is too similar to the last added segment
#             last_segment = deduplicated[-1]
            
#             start_diff = abs(segment['start_time'] - last_segment['start_time'])
#             end_diff = abs(segment['end_time'] - last_segment['end_time'])
            
#             # If both start and end times are within threshold, consider it a duplicate
#             if start_diff <= time_threshold and end_diff <= time_threshold:
#                 # Keep the one with higher quality score
#                 if segment.get('quality_score', 0) > last_segment.get('quality_score', 0):
#                     deduplicated[-1] = segment
#                 # Skip this segment (it's a duplicate)
#             else:
#                 deduplicated.append(segment)
        
#         self.logger.info(f"Deduplicated {len(sorted_segments)} segments to {len(deduplicated)} unique segments")
#         return deduplicated
    
#     def _generate_candidate_segments(self, 
#                                    break_points: List[float],
#                                    video_duration: float,
#                                    target_duration: Tuple[int, int],
#                                    pre_roll_end: float = 0.0) -> List[Dict]:
#         """
#         Generate candidate segments from break points, avoiding pre-roll content.
        
#         Args:
#             break_points: List of potential break points
#             video_duration: Total video duration in seconds
#             target_duration: Target duration range (min, max) in seconds
#             pre_roll_end: Timestamp where pre-roll content ends
            
#         Returns:
#             List of candidate segments
#         """
#         min_duration, max_duration = target_duration
#         candidates = []
        
#         # Add video start (after pre-roll) and end as break points
#         all_points = [max(0.0, pre_roll_end)] + break_points + [video_duration]
#         all_points = sorted(set(all_points))
        
#         for i in range(len(all_points)):
#             for j in range(i + 1, len(all_points)):
#                 start_time = all_points[i]
#                 end_time = all_points[j]
#                 duration = end_time - start_time
                
#                 if min_duration <= duration <= max_duration:
#                     candidates.append({
#                         'start_time': start_time,
#                         'end_time': end_time,
#                         'duration': duration
#                     })
        
#         return candidates
    
#     def _score_segment(self, 
#                       segment: Dict,
#                       audio_analysis: Dict,
#                       ollama_analysis: Optional[Dict]) -> Dict:
#         """Score a segment based on various factors."""
#         start_time = segment['start_time']
#         end_time = segment['end_time']
        
#         # Initialize scores
#         quality_score = 0.5
#         engagement_score = 0.5
#         content_type = 'general'
#         confidence = 0.5
        
#         # Analyze audio content in segment
#         transcription = audio_analysis.get('transcription', {})
#         segment_text = self._extract_segment_text(transcription, start_time, end_time)
        
#         if segment_text:
#             # Score based on text content
#             quality_score += self._score_text_quality(segment_text)
#             engagement_score += self._score_text_engagement(segment_text)
#             content_type = self._classify_content_type(segment_text)
        
#         # Use Ollama analysis if available
#         if ollama_analysis:
#             ollama_score = self._extract_ollama_score(ollama_analysis, start_time, end_time)
#             if ollama_score:
#                 quality_score = (quality_score + ollama_score['quality']) / 2
#                 engagement_score = (engagement_score + ollama_score['engagement']) / 2
#                 confidence = ollama_score['confidence']
        
#         # Apply duration penalty/bonus
#         duration_score = self._score_duration(segment['duration'])
#         quality_score *= duration_score
        
#         # Ensure scores are in valid range
#         quality_score = max(0.0, min(1.0, quality_score))
#         engagement_score = max(0.0, min(1.0, engagement_score))
        
#         return {
#             'start_time': start_time,
#             'end_time': end_time,
#             'duration': segment['duration'],
#             'quality_score': quality_score,
#             'engagement_score': engagement_score,
#             'content_type': content_type,
#             'confidence': confidence,
#             'segment_text': segment_text
#         }
    
#     def _extract_segment_text(self, transcription: Dict, start_time: float, end_time: float) -> str:
#         """Extract text from transcription for a specific time segment."""
#         segments = transcription.get('segments', [])
#         text_parts = []
        
#         for segment in segments:
#             if segment['start'] >= start_time and segment['end'] <= end_time:
#                 text_parts.append(segment['text'])
#             elif segment['start'] < end_time and segment['end'] > start_time:
#                 # Partial overlap
#                 text_parts.append(segment['text'])
        
#         return ' '.join(text_parts).strip()
    
#     def _score_text_quality(self, text: str) -> float:
#         """Score text quality based on completeness and coherence."""
#         if not text:
#             return 0.0
        
#         score = 0.0
        
#         # Length factor
#         word_count = len(text.split())
#         if 10 <= word_count <= 100:  # Optimal range
#             score += 0.3
#         elif word_count >= 5:
#             score += 0.1
        
#         # Sentence completeness
#         sentences = text.split('.')
#         complete_sentences = [s.strip() for s in sentences if s.strip()]
#         if len(complete_sentences) >= 1:
#             score += 0.2
        
#         # Avoid fragments
#         if not text.endswith(('...', '-', ',')):
#             score += 0.1
        
#         # Check for questions or exclamations (engaging)
#         if '?' in text or '!' in text:
#             score += 0.1
        
#         return score
    
#     def _score_text_engagement(self, text: str) -> float:
#         """Score text engagement potential with enhanced yoga/fitness awareness."""
#         if not text:
#             return 0.0
        
#         score = 0.0
#         text_lower = text.lower()
        
#         # General engagement keywords
#         engagement_keywords = [
#             'amazing', 'incredible', 'wow', 'look', 'see', 'watch',
#             'important', 'key', 'secret', 'tip', 'hack', 'trick',
#             'you', 'your', 'we', 'us', 'let\'s', 'remember'
#         ]
        
#         # Yoga/fitness specific engaging content
#         yoga_action_words = [
#             'breathe', 'inhale', 'exhale', 'stretch', 'hold', 'feel',
#             'move', 'shift', 'come into', 'transition', 'flow',
#             'balance', 'center', 'ground', 'lift', 'open', 'close'
#         ]
        
#         # Instructional engagement words
#         instruction_words = [
#             'now', 'next', 'here', 'let\'s', 'we\'re going to',
#             'take a', 'bring your', 'find your', 'notice'
#         ]
        
#         # Count different types of engaging content
#         keyword_count = sum(1 for word in engagement_keywords if word in text_lower)
#         yoga_count = sum(1 for word in yoga_action_words if word in text_lower)
#         instruction_count = sum(1 for word in instruction_words if word in text_lower)
        
#         score += min(0.2, keyword_count * 0.05)
#         score += min(0.3, yoga_count * 0.08)  # Higher weight for yoga actions
#         score += min(0.2, instruction_count * 0.06)
        
#         # Questions increase engagement
#         question_count = text.count('?')
#         score += min(0.15, question_count * 0.08)
        
#         # Exclamations show energy
#         exclamation_count = text.count('!')
#         score += min(0.1, exclamation_count * 0.05)
        
#         # Direct address (very important for yoga instruction)
#         if any(word in text_lower for word in ['you', 'your', 'we', 'us', 'let\'s']):
#             score += 0.15
        
#         # Breathing cues (high value for yoga content)
#         if any(word in text_lower for word in ['breathe', 'inhale', 'exhale', 'breath']):
#             score += 0.1
        
#         # Movement descriptions (valuable for demonstration)
#         movement_words = ['move', 'shift', 'come', 'go', 'bring', 'take', 'lift', 'lower']
#         if any(word in text_lower for word in movement_words):
#             score += 0.1
        
#         return min(1.0, score)
    
#     def _classify_content_type(self, text: str) -> str:
#         """Classify content type with yoga/fitness awareness."""
#         if not text:
#             return 'general'
        
#         text_lower = text.lower()
        
#         # Yoga/fitness specific classifications
#         if any(word in text_lower for word in ['pose', 'asana', 'position', 'stretch', 'yoga']):
#             if any(word in text_lower for word in ['breathe', 'inhale', 'exhale', 'hold']):
#                 return 'yoga_breathing'
#             elif any(word in text_lower for word in ['move', 'flow', 'transition']):
#                 return 'yoga_movement'
#             else:
#                 return 'yoga_instruction'
        
#         # Movement/action content
#         if any(word in text_lower for word in ['move', 'shift', 'come into', 'step', 'walk']):
#             return 'movement'
        
#         # Breathing/mindfulness content
#         if any(word in text_lower for word in ['breathe', 'breath', 'inhale', 'exhale', 'mindful']):
#             return 'breathing'
        
#         # Educational content
#         if any(word in text_lower for word in ['learn', 'teach', 'explain', 'how to', 'tutorial']):
#             return 'educational'
        
#         # Interactive/engagement content
#         if '?' in text or any(word in text_lower for word in ['feel', 'notice', 'you', 'your']):
#             return 'interactive'
        
#         # Motivational content
#         if any(word in text_lower for word in ['motivate', 'inspire', 'success', 'achieve', 'believe']):
#             return 'motivational'
        
#         # Introduction/welcome content
#         if any(word in text_lower for word in ['welcome', 'hello', 'hi', 'today', 'begin']):
#             return 'introduction'
        
#         # Closing/conclusion content
#         if any(word in text_lower for word in ['thank', 'thanks', 'end', 'finish', 'complete', 'namaste']):
#             return 'conclusion'
        
#         return 'instruction'
    
#     def _extract_ollama_score(self, ollama_analysis: Dict, start_time: float, end_time: float) -> Optional[Dict]:
#         """Extract Ollama analysis score for a segment."""
#         if not ollama_analysis:
#             return None
        
#         # Try to find matching segment in Ollama analysis
#         engagement_analysis = ollama_analysis.get('content_engagement', {})
#         if 'engagement_analysis' in engagement_analysis:
#             for segment in engagement_analysis['engagement_analysis']:
#                 if (segment['start_time'] <= start_time <= segment['end_time'] or
#                     segment['start_time'] <= end_time <= segment['end_time']):
#                     return {
#                         'quality': segment.get('engagement_score', 0.5),
#                         'engagement': segment.get('engagement_score', 0.5),
#                         'confidence': 0.8
#                     }
        
#         return None
    
#     def _score_duration(self, duration: float) -> float:
#         """Score segment based on duration (60-120 seconds is optimal for engagement)."""
#         if 60 <= duration <= 90:
#             return 1.0  # Perfect length for engagement
#         elif 90 < duration <= 120:
#             return 0.95  # Very good length
#         elif 45 <= duration < 60:
#             return 0.8  # A bit short but acceptable
#         elif 120 < duration <= 150:
#             return 0.7  # A bit long but acceptable 
#         else:
#             return 0.7  # Too long
    
#     def _remove_overlapping_segments(self, segments: List[Dict]) -> List[Dict]:
#         """Remove overlapping segments, keeping the highest quality ones."""
#         if not segments:
#             return []
        
#         # Sort by quality score (descending)
#         segments.sort(key=lambda x: x['quality_score'], reverse=True)
        
#         non_overlapping = []
        
#         for segment in segments:
#             overlap_found = False
            
#             for existing in non_overlapping:
#                 if self._segments_overlap(segment, existing):
#                     overlap_found = True
#                     break
            
#             if not overlap_found:
#                 non_overlapping.append(segment)
        
#         return non_overlapping
    
#     def _select_diverse_content(self, segments: List[Dict], max_shorts: int) -> List[Dict]:
#         """
#         Select diverse content to avoid repetitive shorts.
#         Ensures variety in content types and timing distribution.
#         """
#         if not segments:
#             return []
        
#         if len(segments) <= max_shorts:
#             return segments
        
#         # Categorize segments by content type and characteristics
#         categorized = self._categorize_segments_for_diversity(segments)
        
#         # Select segments ensuring diversity
#         selected = []
#         content_type_counts = {}
        
#         # First pass: select top segment from each content type
#         for content_type, type_segments in categorized.items():
#             if len(selected) < max_shorts and type_segments:
#                 best_segment = max(type_segments, key=lambda x: x['quality_score'])
#                 selected.append(best_segment)
#                 content_type_counts[content_type] = 1
#                 # Remove selected segment from all categories
#                 for cat_list in categorized.values():
#                     if best_segment in cat_list:
#                         cat_list.remove(best_segment)
        
#         # Second pass: fill remaining slots with temporal diversity
#         remaining_slots = max_shorts - len(selected)
#         if remaining_slots > 0:
#             # Get all remaining segments
#             all_remaining = []
#             for type_segments in categorized.values():
#                 all_remaining.extend(type_segments)
            
#             # Sort by quality but ensure temporal spacing
#             all_remaining.sort(key=lambda x: x['quality_score'], reverse=True)
            
#             for segment in all_remaining:
#                 if len(selected) >= max_shorts:
#                     break
                
#                 # Check temporal diversity (avoid clustering)
#                 if self._has_good_temporal_spacing(segment, selected):
#                     selected.append(segment)
        
#         self.logger.info(f"Selected {len(selected)} diverse segments from {len(segments)} candidates")
#         return selected
    
#     def _categorize_segments_for_diversity(self, segments: List[Dict]) -> Dict[str, List[Dict]]:
#         """Categorize segments for content diversity."""
#         categories = {
#             'instruction': [],      # Teaching/talking segments
#             'demonstration': [],    # Active practice/movement
#             'transition': [],       # Pose changes/movements
#             'introduction': [],     # Opening segments
#             'conclusion': [],       # Closing segments
#             'high_energy': [],      # Dynamic/energetic content
#             'calm': [],            # Peaceful/relaxing content
#             'interactive': []       # Questions/engagement
#         }
        
#         total_duration = max(seg['end_time'] for seg in segments) if segments else 1
        
#         for segment in segments:
#             text = segment.get('segment_text', '').lower()
#             start_ratio = segment['start_time'] / total_duration
            
#             # Classify based on text content and timing
#             if start_ratio < 0.1:
#                 categories['introduction'].append(segment)
#             elif start_ratio > 0.9:
#                 categories['conclusion'].append(segment)
            
#             # Text-based classification
#             if any(word in text for word in ['how', 'let\'s', 'we\'re going to', 'now', 'next']):
#                 categories['instruction'].append(segment)
            
#             if any(word in text for word in ['breathe', 'inhale', 'exhale', 'hold', 'stretch']):
#                 categories['demonstration'].append(segment)
            
#             if any(word in text for word in ['move', 'shift', 'transition', 'come into', 'step']):
#                 categories['transition'].append(segment)
            
#             if '?' in text or any(word in text for word in ['you', 'feel', 'notice']):
#                 categories['interactive'].append(segment)
            
#             # Energy level classification based on keywords
#             high_energy_words = ['jump', 'quick', 'fast', 'energy', 'power', 'strong']
#             calm_words = ['relax', 'gentle', 'soft', 'peaceful', 'calm', 'rest', 'slow']
            
#             if any(word in text for word in high_energy_words):
#                 categories['high_energy'].append(segment)
#             elif any(word in text for word in calm_words):
#                 categories['calm'].append(segment)
#             else:
#                 # Default categorization based on duration and position
#                 if segment['duration'] < 25:
#                     categories['demonstration'].append(segment)
#                 else:
#                     categories['instruction'].append(segment)
        
#         # Remove empty categories
#         return {k: v for k, v in categories.items() if v}
    
#     def _has_good_temporal_spacing(self, candidate: Dict, selected: List[Dict], min_gap: float = 90.0) -> bool:
#         """Check if candidate segment has good temporal spacing from selected segments."""
#         if not selected:
#             return True
        
#         candidate_start = candidate['start_time']
        
#         for existing in selected:
#             existing_start = existing['start_time']
            
#             # Ensure minimum gap between segments
#             if abs(candidate_start - existing_start) < min_gap:
#                 return False
        
#         return True
    
#     def _segments_overlap(self, seg1: Dict, seg2: Dict) -> bool:
#         """Check if two segments overlap."""
#         return not (seg1['end_time'] <= seg2['start_time'] or seg2['end_time'] <= seg1['start_time'])
    
#     def select_final_segments(self,
#                             video_info: Dict,
#                             audio_analysis: Dict,
#                             ollama_analysis: Optional[Dict],
#                             scene_analysis: Dict,
#                             vision_analysis: Optional[Dict],
#                             target_duration: Tuple[int, int] = (60, 120),
#                             max_shorts: int = 10,
#                             quality_threshold: float = 0.7,
#                             candidate_segments: Optional[List[Dict]] = None) -> List[Dict]:
#         """
#         Select final segments based on all analysis results including vision analysis.
#         Guarantees a non-empty selection by falling back to engagement-first ranking
#         when thresholds filter out all candidates.
#         """
#         try:
#             # Use provided candidates or generate new ones
#             if candidate_segments:
#                 self.logger.info(f"Using {len(candidate_segments)} pre-analyzed candidate segments")
#                 initial_segments = candidate_segments
#             else:
#                 # Start with initial segment identification
#                 initial_segments = self.identify_short_segments(
#                     video_info=video_info,
#                     audio_analysis=audio_analysis,
#                     ollama_analysis=ollama_analysis,
#                     scene_analysis=scene_analysis,
#                     target_duration=target_duration,
#                     max_shorts=max_shorts * 3  # Get more candidates for better selection
#                 )
            
#             # Apply vision enhancement if available
#             if vision_analysis and initial_segments:
#                 enhanced_segments = self._enhance_segments_with_vision(
#                     initial_segments, vision_analysis
#                 )
#             else:
#                 enhanced_segments = initial_segments
            
#             # Apply intelligent quality threshold with content-aware preference
#             def meets_intelligent_threshold(seg):
#                 base_quality = seg.get('quality_score', 0)
#                 method = seg.get('generation_method', '')
                
#                 # Lower threshold for content-aware segments (they're more intelligent)
#                 if method == 'content_aware':
#                     adjusted_threshold = quality_threshold * 0.7  # 30% lower threshold
#                     if seg.get('high_value_content'):
#                         adjusted_threshold = quality_threshold * 0.6  # Even lower for high-value
#                 elif method == 'quality_driven':
#                     adjusted_threshold = quality_threshold * 0.8  # 20% lower threshold
#                 else:
#                     adjusted_threshold = quality_threshold  # Standard threshold for others
                
#                 return base_quality >= adjusted_threshold
            
#             quality_filtered = [seg for seg in enhanced_segments if meets_intelligent_threshold(seg)]
            
#             # If not enough segments pass quality threshold, lower it slightly
#             if len(quality_filtered) < max_shorts and enhanced_segments:
#                 lower_threshold = quality_threshold * 0.8
#                 quality_filtered = [
#                     seg for seg in enhanced_segments 
#                     if seg.get('quality_score', 0) >= lower_threshold
#                 ]
#                 self.logger.info(f"Lowered quality threshold to {lower_threshold:.2f} to get more segments")
            
#             # Select diverse content up to max_shorts
#             final_segments = self._select_diverse_content(quality_filtered, max_shorts)
            
#             # Engagement-first forced fallback: if still empty or too few, pick top-N by engagement + vision
#             if (not final_segments) and enhanced_segments:
#                 self.logger.info("No segments met the thresholds; applying engagement-first fallback selection")
#                 # Rank by engagement-first composite
#                 def engagement_rank(seg: Dict) -> float:
#                     # Core metrics
#                     eng = seg.get('engagement_score', 0.5)
#                     qual = seg.get('quality_score', 0.5)
#                     text_qual = seg.get('text_quality_score', 0.5)
                    
#                     # Vision analysis
#                     vision = seg.get('vision_score', seg.get('visual_interest', 0.5))
#                     if isinstance(vision, (int, float)):
#                         v = max(0.0, min(1.0, vision/10.0)) if vision > 1 else max(0.0, min(1.0, vision))
#                     else:
#                         v = 0.5
                    
#                     # Content intelligence bonuses
#                     method_bonus = 0.0
#                     if seg.get('generation_method') == 'content_aware':
#                         method_bonus += 0.20  # Strong preference for content-aware
#                         if seg.get('high_value_content'):
#                             method_bonus += 0.10
#                         if seg.get('trigger_keywords'):
#                             method_bonus += 0.05
#                     elif seg.get('generation_method') == 'quality_driven':
#                         method_bonus += 0.10  # Moderate preference for quality-driven
                    
#                     # Audio transcription quality
#                     audio_bonus = 0.0
#                     if seg.get('has_complete_sentences', False):
#                         audio_bonus += 0.05
#                     if seg.get('word_count', 0) > 100:
#                         audio_bonus += 0.03
                    
#                     # People detection bonus
#                     people_bonus = 0.05 if (seg.get('people_visible') or seg.get('people_count', 0) > 0) else 0.0
                    
#                     # Comprehensive weighted score
#                     return (
#                         eng * 0.30 +           # Engagement
#                         qual * 0.25 +          # Base quality
#                         text_qual * 0.15 +     # Text quality
#                         v * 0.10 +             # Visual interest
#                         method_bonus +         # Content intelligence
#                         audio_bonus +          # Audio quality
#                         people_bonus           # People detection
#                     )
#                 ranked = sorted(enhanced_segments, key=engagement_rank, reverse=True)
#                 # Light temporal spacing
#                 picked: List[Dict] = []
#                 used_times: List[Tuple[float, float]] = []
#                 min_gap = 10.0
#                 for seg in ranked:
#                     if len(picked) >= max_shorts:
#                         break
#                     s, e = seg.get('start_time', 0.0), seg.get('end_time', 0.0)
#                     if all(e <= us or s >= ue or min(abs(s-ue), abs(e-us)) >= min_gap for us, ue in used_times):
#                         picked.append(seg)
#                         used_times.append((s, e))
#                 final_segments = picked
#                 self.logger.info(f"Engagement-first fallback selected {len(final_segments)} segments")
            
#             self.logger.info(f"Selected {len(final_segments)} final segments from {len(initial_segments)} candidates")
#             return final_segments
            
#         except Exception as e:
#             self.logger.error(f"Error selecting final segments: {e}")
#             # Return simplified segments as fallback
#             return initial_segments[:max_shorts] if initial_segments else []
    
#     def _enhance_segments_with_vision(self, 
#                                     segments: List[Dict], 
#                                     vision_analysis: Dict) -> List[Dict]:
#         """
#         Enhance segment scoring with vision analysis data.
        
#         Args:
#             segments: Initial segments from audio/content analysis
#             vision_analysis: Vision analysis results
            
#         Returns:
#             Enhanced segments with vision-based scoring
#         """
#         try:
#             enhanced_segments = []
#             vision_segments = vision_analysis.get('segments', [])
            
#             # Defensive check: ensure vision_segments is a list
#             if not isinstance(vision_segments, list):
#                 self.logger.warning(f"vision_segments is not a list, got {type(vision_segments)}: {vision_segments}")
#                 return segments
            
#             # Create a lookup for vision data by time
#             vision_lookup = {}
#             for v_seg in vision_segments:
#                 # Defensive check: ensure v_seg is a dictionary
#                 if not isinstance(v_seg, dict):
#                     self.logger.warning(f"Vision segment is not a dict, got {type(v_seg)}: {v_seg}")
#                     continue
                    
#                 start_time = v_seg.get('start_time', 0)
#                 vision_lookup[start_time] = v_seg
            
#             for segment in segments:
#                 enhanced_segment = segment.copy()
                
#                 # Find matching or closest vision segment
#                 best_vision_match = None
#                 min_time_diff = float('inf')
                
#                 for v_start, v_data in vision_lookup.items():
#                     time_diff = abs(v_start - segment['start_time'])
#                     if time_diff < min_time_diff and time_diff < 10.0:  # Within 10 seconds
#                         min_time_diff = time_diff
#                         best_vision_match = v_data
                
#                 if best_vision_match:
#                     # Enhance scoring with vision data
#                     vision_score = best_vision_match.get('visual_score', 0.5)
#                     visual_interest = best_vision_match.get('visual_interest', 0.5)
#                     people_count = best_vision_match.get('people_count', 0)
#                     scene_type = best_vision_match.get('scene_type', 'unknown')
                    
#                     # Combine audio and vision scores
#                     original_quality = enhanced_segment.get('quality_score', 0.5)
#                     original_engagement = enhanced_segment.get('engagement_score', 0.5)
                    
#                     # Weighted combination (60% audio, 40% vision)
#                     combined_quality = (original_quality * 0.6) + (vision_score * 0.4)
#                     combined_engagement = (original_engagement * 0.6) + (visual_interest * 0.4)
                    
#                     # Bonus for segments with people visible
#                     if people_count > 0:
#                         combined_quality += 0.1
#                         combined_engagement += 0.1
                    
#                     # Bonus for interesting visual content
#                     if scene_type in ['action', 'demonstration', 'presentation']:
#                         combined_engagement += 0.05
                    
#                     # Update segment with enhanced scores
#                     enhanced_segment['quality_score'] = min(1.0, combined_quality)
#                     enhanced_segment['engagement_score'] = min(1.0, combined_engagement)
#                     enhanced_segment['has_vision_data'] = True
#                     enhanced_segment['vision_score'] = vision_score
#                     enhanced_segment['people_visible'] = people_count > 0
#                     enhanced_segment['scene_type'] = scene_type
#                 else:
#                     # No vision data available, keep original scores
#                     enhanced_segment['has_vision_data'] = False
                
#                 enhanced_segments.append(enhanced_segment)
            
#             self.logger.info(f"Enhanced {len(enhanced_segments)} segments with vision analysis")
#             return enhanced_segments
            
#         except Exception as e:
#             self.logger.error(f"Error enhancing segments with vision: {e}")
#             return segments  # Return original segments on error

#     def get_segment_summary(self, segments: List[Dict]) -> Dict:
#         """Generate summary of selected segments."""
#         if not segments:
#             return {
#                 'total_segments': 0,
#                 'total_duration': 0.0,
#                 'average_quality': 0.0,
#                 'content_types': {}
#             }
        
#         total_duration = sum(seg['duration'] for seg in segments)
#         average_quality = sum(seg['quality_score'] for seg in segments) / len(segments)
        
#         # Count content types
#         content_types = {}
#         for segment in segments:
#             content_type = segment['content_type']
#             content_types[content_type] = content_types.get(content_type, 0) + 1
        
#         return {
#             'total_segments': len(segments),
#             'total_duration': total_duration,
#             'average_quality': average_quality,
#             'average_engagement': sum(seg['engagement_score'] for seg in segments) / len(segments),
#             'content_types': content_types,
#             'duration_range': (min(seg['duration'] for seg in segments), max(seg['duration'] for seg in segments))
#         }

#     def _identify_audio_segments(self, 
#                                video_info: Dict,
#                                audio_analysis: Dict,
#                                ollama_analysis: Optional[Dict],
#                                scene_analysis: Optional[Dict],
#                                target_duration: Tuple[int, int],
#                                max_shorts: int,
#                                pre_roll_end: float) -> List[Dict]:
#         """Identify segments using traditional audio-based approach."""
#         # Get all potential break points
#         break_points = self._get_all_break_points(scene_analysis or {}, audio_analysis, pre_roll_end)
        
#         # Generate candidate segments
#         candidate_segments = self._generate_candidate_segments(
#             break_points, 
#             video_info.get('duration', 0),
#             target_duration,
#             pre_roll_end
#         )
        
#         # Score segments based on content quality
#         scored_segments = self._score_segments(
#             candidate_segments, 
#             audio_analysis, 
#             ollama_analysis,
#             scene_analysis
#         )
        
#         return scored_segments

#     def _identify_visual_segments(self, 
#                                 video_info: Dict,
#                                 audio_analysis: Dict,
#                                 scene_analysis: Optional[Dict],
#                                 target_duration: Tuple[int, int],
#                                 max_shorts: int,
#                                 pre_roll_end: float,
#                                 existing_audio_segments: List[Dict]) -> List[Dict]:
#         """Identify segments based on visual content (silent scenes, action, etc.)."""
#         try:
#             from .visual_segment_detector import VisualSegmentDetector
            
#             # Initialize visual detector
#             visual_detector = VisualSegmentDetector()
            
#             # For now, focus on silent periods that might be visually interesting
#             visual_segments = []
            
#             # Find silent periods from transcription
#             transcription = audio_analysis.get('transcription', {})
#             silence_periods = transcription.get('silence_periods', [])
            
#             for silence_start, silence_end in silence_periods:
#                 duration = silence_end - silence_start
                
#                 # Only consider silent periods that are:
#                 # 1. After pre-roll
#                 # 2. Long enough to be interesting (>10 seconds)
#                 # 3. Within target duration range
#                 if (silence_start > pre_roll_end and 
#                     duration >= 10.0 and 
#                     duration <= target_duration[1]):
                    
#                     # Check if this overlaps significantly with existing audio segments
#                     overlap_ratio = self._calculate_max_overlap_with_existing(
#                         silence_start, silence_end, existing_audio_segments
#                     )
                    
#                     # Only add if minimal overlap with audio segments
#                     if overlap_ratio < 0.3:
#                         visual_segments.append({
#                             'start_time': silence_start,
#                             'end_time': min(silence_end, silence_start + target_duration[1]),
#                             'duration': min(duration, target_duration[1]),
#                             'quality_score': 0.6,  # Moderate quality for silent scenes
#                             'engagement_score': 0.5,  # Could be visually engaging
#                             'content_type': 'visual_silent',
#                             'source': 'visual_detection',
#                             'visual_priority': True,
#                             'segment_text': '[SILENT VISUAL CONTENT]'
#                         })
            
#             # Look for motion-based segments from scene analysis
#             if scene_analysis and 'motion_analysis' in scene_analysis:
#                 motion_segments = scene_analysis['motion_analysis'].get('high_motion_segments', [])
                
#                 for motion_seg in motion_segments:
#                     start_time = motion_seg['start']
#                     end_time = motion_seg['end']
#                     duration = end_time - start_time
                    
#                     if (start_time > pre_roll_end and 
#                         target_duration[0] <= duration <= target_duration[1]):
                        
#                         overlap_ratio = self._calculate_max_overlap_with_existing(
#                             start_time, end_time, existing_audio_segments
#                         )
                        
#                         if overlap_ratio < 0.5:  # Allow more overlap for motion segments
#                             visual_segments.append({
#                                 'start_time': start_time,
#                                 'end_time': end_time,
#                                 'duration': duration,
#                                 'quality_score': 0.8,  # High quality for motion
#                                 'engagement_score': 0.9,  # Motion is typically engaging
#                                 'content_type': 'visual_action',
#                                 'source': 'visual_detection',
#                                 'visual_priority': True,
#                                 'segment_text': '[HIGH MOTION VISUAL CONTENT]'
#                             })
            
#             self.logger.info(f"Identified {len(visual_segments)} visual-based segments")
#             return visual_segments
            
#         except Exception as e:
#             self.logger.warning(f"Visual segment detection failed: {e}")
#             return []

#     def _enhance_segments_with_object_scores(self, 
#                                            segments: List[Dict], 
#                                            object_detection_results: Dict) -> List[Dict]:
#         """Enhance segments with object detection relevance scores."""
#         try:
#             if not object_detection_results or object_detection_results.get('status') != 'success':
#                 return segments
            
#             detected_objects = object_detection_results.get('detected_objects', [])
#             tracking_results = object_detection_results.get('object_tracks', [])
            
#             # Defensive check: ensure detected_objects is a list
#             if not isinstance(detected_objects, list):
#                 self.logger.warning(f"detected_objects is not a list, got {type(detected_objects)}: {detected_objects}")
#                 return segments
            
#             # Create object timeline (objects are now dictionaries)
#             object_timeline = {}
#             for obj in detected_objects:
#                 # Defensive check: ensure obj is a dictionary
#                 if not isinstance(obj, dict):
#                     self.logger.warning(f"Object in detected_objects is not a dict, got {type(obj)}: {obj}")
#                     continue
                    
#                 timestamp = obj.get('frame_timestamp')
#                 if timestamp not in object_timeline:
#                     object_timeline[timestamp] = []
#                 object_timeline[timestamp].append(obj)
            
#             # Enhance each segment
#             for segment in segments:
#                 start_time = segment['start_time']
#                 end_time = segment['end_time']
                
#                 # Find objects in this time range
#                 segment_objects = []
#                 for timestamp, objects in object_timeline.items():
#                     if start_time <= timestamp <= end_time:
#                         segment_objects.extend(objects)
                
#                 if segment_objects:
#                     # Calculate object relevance score (objects are now dictionaries)
#                     object_relevance = sum(obj.get('relevance_score', 0) for obj in segment_objects) / len(segment_objects)
#                     object_confidence = sum(obj.get('confidence', 0) for obj in segment_objects) / len(segment_objects)
#                     prompt_match = sum(obj.get('prompt_match_score', 0) for obj in segment_objects) / len(segment_objects)
                    
#                     # Add object-based scores
#                     segment['object_relevance_score'] = object_relevance
#                     segment['object_confidence_score'] = object_confidence
#                     segment['object_prompt_match_score'] = prompt_match
#                     segment['objects_detected'] = len(segment_objects)
#                     segment['unique_object_classes'] = len(set(obj.get('class_name', '') for obj in segment_objects))
#                 else:
#                     # No objects detected in this segment
#                     segment['object_relevance_score'] = 0.0
#                     segment['object_confidence_score'] = 0.0
#                     segment['object_prompt_match_score'] = 0.0
#                     segment['objects_detected'] = 0
#                     segment['unique_object_classes'] = 0
            
#             return segments
            
#         except Exception as e:
#             self.logger.warning(f"Failed to enhance segments with object scores: {e}")
#             return segments
    
#     def _calculate_combined_score(self, segment: Dict) -> float:
#         """Calculate combined score from prompt match and object detection."""
#         prompt_score = segment.get('prompt_match_score', 0.0)
#         object_relevance = segment.get('object_relevance_score', 0.0)
#         object_confidence = segment.get('object_confidence_score', 0.0)
#         object_prompt_match = segment.get('object_prompt_match_score', 0.0)
        
#         # Weighted combination
#         base_score = prompt_score * 0.4
#         object_score = (object_relevance * 0.3 + object_confidence * 0.2 + object_prompt_match * 0.4) * 0.6
        
#         # Bonus for segments with multiple relevant objects
#         objects_detected = segment.get('objects_detected', 0)
#         unique_classes = segment.get('unique_object_classes', 0)
        
#         object_bonus = min(0.2, (objects_detected * 0.05) + (unique_classes * 0.1))
        
#         return min(1.0, base_score + object_score + object_bonus)
    
#     def _ensure_prompt_scores(self, segments: List[Dict]) -> List[Dict]:
#         """Ensure each segment has a non-zero prompt_match_score by deriving it from available scores when missing.
#         This makes the final selection robust to segments produced by contextual/heuristic analyzers.
#         """
#         derived = 0
#         for seg in segments:
#             current = seg.get('prompt_match_score')
#             if current is None or current == 0:
#                 candidates = [
#                     # Climax-specific scores
#                     seg.get('final_climax_score'),
#                     seg.get('climax_score'),
#                     # Contextual / heuristic scores
#                     seg.get('contextual_overall_score'),
#                     seg.get('composite_validation_score'),
#                     seg.get('contextual_relevance_score'),
#                     seg.get('heuristic_score'),
#                     seg.get('quality_score'),
#                     seg.get('ai_refined_score')
#                 ]
#                 # Filter out None and negatives
#                 numeric = [c for c in candidates if isinstance(c, (int, float)) and c is not None]
#                 if numeric:
#                     new_score = max(0.0, min(1.0, max(numeric)))
#                 else:
#                     # Minimal non-zero score if we have any text content
#                     new_score = 0.05 if seg.get('segment_text') else 0.0
#                 seg['prompt_match_score'] = new_score
#                 if new_score > 0:
#                     derived += 1
#         if derived:
#             self.logger.info(f"Derived prompt_match_score for {derived} segments from contextual/heuristic metrics")
#         return segments
    
#     async def cleanup(self):
#         """Clean up resources from all analyzers."""
#         try:
#             if self.object_detector:
#                 await self.object_detector.cleanup()
            
#             if self.ai_reframer:
#                 await self.ai_reframer.cleanup()
            
#             self.logger.info("ContentAnalyzer cleanup completed")
            
#         except Exception as e:
#             self.logger.error(f"Error during ContentAnalyzer cleanup: {e}")
    
#     # ================================
#     # PHASE 2: MULTI-PASS ANALYSIS PIPELINE
#     # ================================
    
#     async def _phase2_multipass_analysis(
#         self,
#         initial_segments,
#         user_prompt,
#         comprehensive_segments,
#         audio_analysis,
#         vision_analysis,
#         scene_analysis,
#         content_overview,
#         intent_analysis,
#         video_info,
#         target_duration,
#         max_shorts
#     ):
#         """
#         Phase 2: Multi-pass analysis pipeline for iterative segment refinement.
        
#         This implements a sophisticated multi-pass system that:
#         1. Cross-validates initial results
#         2. Performs iterative refinement
#         3. Quality scoring and ranking
#         4. Final optimization
#         """
#         try:
#             self.logger.info("Starting Phase 2 multi-pass analysis pipeline")
            
#             # Pass 1: Cross-validation and quality assessment
#             cross_validation_result = await self._pass1_cross_validation(
#                 initial_segments, user_prompt, comprehensive_segments,
#                 audio_analysis, vision_analysis, scene_analysis,
#                 content_overview, intent_analysis
#             )
            
#             # Pass 2: Iterative refinement
#             refinement_result = await self._pass2_iterative_refinement(
#                 cross_validation_result['segments'], user_prompt,
#                 comprehensive_segments, audio_analysis, vision_analysis,
#                 scene_analysis, content_overview, intent_analysis, video_info
#             )
            
#             # Pass 3: Quality scoring and final optimization
#             final_result = await self._pass3_quality_optimization(
#                 refinement_result['segments'], user_prompt,
#                 content_overview, intent_analysis, video_info,
#                 target_duration, max_shorts
#             )
            
#             # Compile comprehensive results
#             multipass_confidence = (
#                 cross_validation_result.get('confidence', 0.7) * 0.3 +
#                 refinement_result.get('confidence', 0.7) * 0.4 +
#                 final_result.get('confidence', 0.7) * 0.3
#             )
            
#             return {
#                 'status': 'success',
#                 'segments': final_result['segments'],
#                 'analysis_method': 'phase2_multipass_enhanced',
#                 'content_overview': content_overview,
#                 'intent_analysis': intent_analysis,
#                 'context_confidence': multipass_confidence,
#                 'multipass_details': {
#                     'pass1_cross_validation': cross_validation_result.get('metrics', {}),
#                     'pass2_iterative_refinement': refinement_result.get('metrics', {}),
#                     'pass3_quality_optimization': final_result.get('metrics', {}),
#                     'total_analysis_passes': 3,
#                     'enhancement_level': 'phase2_multipass'
#                 },
#                 'generation_details': {
#                     'initial_segments': len(initial_segments),
#                     'total_candidates': len(comprehensive_segments),
#                     'final_selected_count': len(final_result['segments']),
#                     'multipass_confidence': multipass_confidence,
#                     'enhancement_level': 'phase2_multipass'
#                 }
#             }
            
#         except Exception as e:
#             self.logger.error(f"Phase 2 multi-pass analysis failed: {e}")
#             return {
#                 'status': 'error',
#                 'error': f"Multi-pass analysis failed: {e}",
#                 'fallback_required': True
#             }
    
#     async def _pass1_cross_validation(
#         self, initial_segments, user_prompt, comprehensive_segments,
#         audio_analysis, vision_analysis, scene_analysis,
#         content_overview, intent_analysis
#     ):
#         """Pass 1: Cross-validation between different analysis methods."""
#         try:
#             self.logger.info("Phase 2 Pass 1: Cross-validation analysis")
            
#             # Method 1: LLM-based validation
#             llm_validation = await self._validate_segments_with_llm(
#                 initial_segments, user_prompt, content_overview, intent_analysis
#             )
            
#             # Method 2: Heuristic-based validation
#             heuristic_validation = self._validate_segments_heuristically(
#                 initial_segments, audio_analysis, scene_analysis, intent_analysis
#             )
            
#             # Method 3: Content coherence validation
#             coherence_validation = self._validate_content_coherence(
#                 initial_segments, audio_analysis, content_overview
#             )
            
#             # Cross-validate and score segments
#             validated_segments = []
#             for segment in initial_segments:
#                 segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
#                 validation_scores = {
#                     'llm_score': llm_validation.get(segment_id, {}).get('score', 0.5),
#                     'heuristic_score': heuristic_validation.get(segment_id, {}).get('score', 0.5),
#                     'coherence_score': coherence_validation.get(segment_id, {}).get('score', 0.5)
#                 }
                
#                 # Calculate composite validation score
#                 composite_score = (
#                     validation_scores['llm_score'] * 0.5 +
#                     validation_scores['heuristic_score'] * 0.3 +
#                     validation_scores['coherence_score'] * 0.2
#                 )
                
#                 # Add validation metadata
#                 enhanced_segment = segment.copy()
#                 enhanced_segment.update({
#                     'segment_id': segment_id,
#                     'validation_scores': validation_scores,
#                     'composite_validation_score': composite_score,
#                     'cross_validation_passed': composite_score >= 0.6,
#                     'validation_confidence': min(validation_scores.values())
#                 })
                
#                 validated_segments.append(enhanced_segment)
            
#             # Filter segments that passed cross-validation
#             passed_segments = [seg for seg in validated_segments if seg.get('cross_validation_passed', False)]
            
#             self.logger.info(f"Pass 1: {len(passed_segments)}/{len(initial_segments)} segments passed cross-validation")
            
#             return {
#                 'segments': passed_segments,
#                 'confidence': sum(seg.get('composite_validation_score', 0.5) for seg in passed_segments) / max(len(passed_segments), 1),
#                 'metrics': {
#                     'initial_count': len(initial_segments),
#                     'validated_count': len(passed_segments),
#                     'validation_rate': len(passed_segments) / max(len(initial_segments), 1),
#                     'average_validation_score': sum(seg.get('composite_validation_score', 0.5) for seg in validated_segments) / max(len(validated_segments), 1)
#                 }
#             }
            
#         except Exception as e:
#             self.logger.error(f"Pass 1 cross-validation failed: {e}")
#             return {
#                 'segments': initial_segments,
#                 'confidence': 0.5,
#                 'metrics': {'error': str(e)}
#             }
    
#     async def _pass2_iterative_refinement(
#         self, validated_segments, user_prompt, comprehensive_segments,
#         audio_analysis, vision_analysis, scene_analysis,
#         content_overview, intent_analysis, video_info
#     ):
#         """Pass 2: Iterative refinement of segment selection and timing."""
#         try:
#             self.logger.info("Phase 2 Pass 2: Iterative refinement")
            
#             refined_segments = validated_segments.copy()
#             iterations = 0
#             max_iterations = 3
#             improvement_threshold = 0.05
            
#             previous_score = sum(seg.get('composite_validation_score', 0.5) for seg in refined_segments) / max(len(refined_segments), 1)
            
#             while iterations < max_iterations:
#                 iterations += 1
#                 self.logger.info(f"Refinement iteration {iterations}/{max_iterations}")
                
#                 # Refinement 1: Timing optimization
#                 timing_refined = await self._refine_segment_timing(
#                     refined_segments, audio_analysis, scene_analysis
#                 )
                
#                 # Refinement 2: Content overlap detection and resolution
#                 overlap_resolved = self._resolve_segment_overlaps(
#                     timing_refined, video_info
#                 )
                
#                 # Refinement 3: Gap analysis and filling
#                 gap_filled = await self._analyze_and_fill_gaps(
#                     overlap_resolved, comprehensive_segments, user_prompt,
#                     content_overview, intent_analysis
#                 )
                
#                 # Calculate improvement score
#                 current_score = sum(seg.get('composite_validation_score', 0.5) for seg in gap_filled) / max(len(gap_filled), 1)
#                 improvement = current_score - previous_score
                
#                 if improvement < improvement_threshold:
#                     self.logger.info(f"Refinement converged after {iterations} iterations (improvement: {improvement:.3f})")
#                     break
                
#                 refined_segments = gap_filled
#                 previous_score = current_score
#                 self.logger.info(f"Iteration {iterations} improvement: {improvement:.3f}")
            
#             self.logger.info(f"Pass 2: Iterative refinement completed in {iterations} iterations")
            
#             return {
#                 'segments': refined_segments,
#                 'confidence': current_score,
#                 'metrics': {
#                     'iterations': iterations,
#                     'final_improvement': improvement,
#                     'refinement_convergence': improvement < improvement_threshold,
#                     'segment_count': len(refined_segments)
#                 }
#             }
            
#         except Exception as e:
#             self.logger.error(f"Pass 2 iterative refinement failed: {e}")
#             return {
#                 'segments': validated_segments,
#                 'confidence': 0.6,
#                 'metrics': {'error': str(e)}
#             }
    
#     async def _pass3_quality_optimization(
#         self, refined_segments, user_prompt, content_overview,
#         intent_analysis, video_info, target_duration, max_shorts
#     ):
#         """Pass 3: Quality scoring and final optimization."""
#         try:
#             self.logger.info("Phase 2 Pass 3: Quality optimization")
            
#             # Enhanced quality scoring
#             quality_scored_segments = []
#             for segment in refined_segments:
#                 quality_score = await self._calculate_enhanced_quality_score(
#                     segment, user_prompt, content_overview, intent_analysis
#                 )
                
#                 enhanced_segment = segment.copy()
#                 enhanced_segment.update({
#                     'quality_score': quality_score,
#                     'phase2_enhanced': True,
#                     'final_optimization_applied': True
#                 })
                
#                 quality_scored_segments.append(enhanced_segment)
            
#             # Sort by quality score
#             quality_scored_segments.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
#             # Final selection optimization
#             optimized_segments = self._optimize_final_selection(
#                 quality_scored_segments, target_duration, max_shorts, intent_analysis
#             )
            
#             # Calculate final confidence
#             final_confidence = sum(seg.get('quality_score', 0.5) for seg in optimized_segments) / max(len(optimized_segments), 1)
            
#             self.logger.info(f"Pass 3: Quality optimization completed - {len(optimized_segments)} final segments")
            
#             return {
#                 'segments': optimized_segments,
#                 'confidence': final_confidence,
#                 'metrics': {
#                     'quality_scored_count': len(quality_scored_segments),
#                     'final_selected_count': len(optimized_segments),
#                     'average_quality_score': final_confidence,
#                     'optimization_applied': True
#                 }
#             }
            
#         except Exception as e:
#             self.logger.error(f"Pass 3 quality optimization failed: {e}")
#             return {
#                 'segments': refined_segments,
#                 'confidence': 0.7,
#                 'metrics': {'error': str(e)}
#             }

#     async def _analyze_content_overview(self,
#                                       audio_analysis: Dict,
#                                       vision_analysis: Optional[Dict],
#                                       video_info: Dict) -> Dict:
#         """
#         PHASE 1 ENHANCEMENT: Comprehensive content overview analysis.
        
#         This provides the LLM with full context about the video content,
#         narrative structure, and overall characteristics before segment selection.
#         """
#         try:
#             # Prepare comprehensive content summary
#             transcription = audio_analysis.get('transcription', {})
#             segments = transcription.get('segments', [])
            
#             # Extract key content indicators
#             total_duration = video_info.get('duration', 0)
#             speaker_count = len(set(seg.get('speaker', 'unknown') for seg in segments))
            
#             # Create content analysis prompt
#             overview_prompt = f"""
#             You are analyzing a video to understand its overall content and structure for short-form content creation.
            
#             VIDEO METADATA:
#             - Duration: {total_duration} seconds
#             - Estimated speakers: {speaker_count}
#             - Has visual analysis: {vision_analysis is not None}
            
#             TRANSCRIPTION SAMPLE (first 10 segments):
#             {json.dumps(segments[:10], indent=2)}
            
            
#             VISUAL CONTENT SUMMARY (if available):
#             {json.dumps(vision_analysis.get('summary', {}) if vision_analysis else {}, indent=2)}
            
#             ANALYSIS TASK:
#             Provide a comprehensive overview to help with intelligent scene selection:
            
#             1. What type of content is this? (movie, tutorial, interview, presentation, sports, etc.)
#             2. What is the overall narrative structure and flow?
#             3. What are the main themes and topics discussed?
#             4. Where are the most engaging/important moments likely to be?
#             5. What would work best as short-form social media content?
            
#             Respond in JSON format:
#             {{
#                 "content_type": "movie|tutorial|interview|presentation|sports|documentary|other",
#                 "genre": "specific genre if applicable",
#                 "narrative_structure": {{
#                     "has_clear_beginning": boolean,
#                     "has_development": boolean,
#                     "has_climax_or_peak": boolean,
#                     "has_resolution": boolean,
#                     "content_flow": "linear|episodic|instructional|conversational"
#                 }},
#                 "main_themes": ["theme1", "theme2", "theme3"],
#                 "content_characteristics": {{
#                     "is_educational": boolean,
#                     "is_entertaining": boolean,
#                     "is_narrative_driven": boolean,
#                     "has_demonstrations": boolean,
#                     "interaction_level": "monologue|dialogue|interactive"
#                 }},
#                 "engagement_patterns": {{
#                     "peak_likely_at": "beginning|middle|end|distributed",
#                     "content_density": "high|medium|low",
#                     "emotional_variation": "high|medium|low"
#                 }},
#                 "short_form_potential": {{
#                     "best_segment_types": ["type1", "type2"],
#                     "ideal_segment_length": "15-30|30-60|60-90 seconds",
#                     "key_selection_criteria": ["criteria1", "criteria2"]
#                 }}
#             }}
#             """
            
#             if self.ollama_client:
#                 response = await self.ollama_client._make_request(
#                     prompt=overview_prompt,
#                     model=self.ollama_client.get_best_model("analysis"),
#                     cache_key=f"content_overview_{hash(str(audio_analysis.get('transcription', {})) + str(video_info))}"
#                 )
                
#                 overview_data = self.ollama_client._parse_json_response(response)
#                 self.logger.info(f"Content overview analysis completed: {overview_data.get('content_type', 'unknown')} type")
#                 return overview_data
#             else:
#                 return self._fallback_content_overview(audio_analysis, video_info)
                
#         except Exception as e:
#             self.logger.error(f"Content overview analysis failed: {e}")
#             return self._fallback_content_overview(audio_analysis, video_info)
    
#     async def _analyze_user_intent_comprehensive(self,
#                                                user_prompt: str,
#                                                content_overview: Dict,
#                                                video_info: Dict) -> Dict:
#         """
#         PHASE 1 ENHANCEMENT: Enhanced user intent analysis with full content context.
        
#         This goes beyond keyword matching to understand what the user
#         really wants based on the specific video content.
#         """
#         try:
#             intent_prompt = f"""
#             You are an expert video editor who understands user intent for creating short-form content.
            
#             USER REQUEST: "{user_prompt}"
            
#             VIDEO CONTENT OVERVIEW:
#             {json.dumps(content_overview, indent=2)}
            
#             VIDEO METADATA:
#             - Duration: {video_info.get('duration', 0)} seconds
#             - Type: {content_overview.get('content_type', 'unknown')}
            
#             ANALYSIS TASK:
#             Based on the user's request and the actual video content, determine:
            
#             1. What the user's true intent is (beyond literal keywords)
#             2. What type of content from this specific video would satisfy their request
#             3. How to adapt their request to this video's content and structure
#             4. What quality criteria should be used for segment selection
            
#             Consider the video's actual content type and structure when interpreting the request.
            
#             Provide detailed intent analysis:
#             {{
#                 "intent_interpretation": {{
#                     "literal_request": "what they said",
#                     "contextual_intent": "what they actually want given this video type",
#                     "content_alignment": "how well this video can satisfy their request",
#                     "adaptation_strategy": "how to best fulfill their intent with available content"
#                 }},
#                 "selection_criteria": {{
#                     "primary_factors": ["most important selection criteria"],
#                     "secondary_factors": ["additional considerations"],
#                     "content_position_preference": "beginning|middle|end|peak|any",
#                     "quality_thresholds": {{
#                         "minimum_engagement": 0.4,
#                         "minimum_completeness": 0.3,
#                         "minimum_relevance": 0.5
#                     }}
#                 }},
#                 "content_requirements": {{
#                     "emotional_tone": "required emotional characteristics",
#                     "must_include": ["required elements"],
#                     "must_avoid": ["elements to avoid"],
#                     "duration_preference": "ideal length in seconds",
#                     "standalone_viability": "high|medium|low requirement"
#                 }},
#                 "confidence_assessment": {{
#                     "intent_clarity": 0.8,
#                     "content_availability": 0.7,
#                     "match_likelihood": 0.8,
#                     "overall_confidence": 0.8
#                 }}
#             }}
#             """
            
#             if self.ollama_client:
#                 response = await self.ollama_client._make_request(
#                     prompt=intent_prompt,
#                     model=self.ollama_client.get_best_model("analysis"),
#                     cache_key=f"intent_analysis_{hash(user_prompt + str(content_overview))}"
#                 )
                
#                 intent_data = self.ollama_client._parse_json_response(response)
#                 self.logger.info(f"Intent analysis completed with {intent_data.get('confidence_assessment', {}).get('overall_confidence', 0):.2f} confidence")
#                 return intent_data
#             else:
#                 return self._fallback_intent_analysis(user_prompt)
                
#         except Exception as e:
#             self.logger.error(f"Intent analysis failed: {e}")
#             return self._fallback_intent_analysis(user_prompt)
    
#     def _fallback_content_overview(self, audio_analysis: Dict, video_info: Dict) -> Dict:
#         """Fallback content overview when LLM analysis fails."""
#         # Simple heuristic-based content analysis
#         transcription = audio_analysis.get('transcription', {})
#         segments = transcription.get('segments', [])
        
#         # Basic content type detection
#         all_text = ' '.join(seg.get('text', '') for seg in segments).lower()
        
#         content_type = 'other'
#         if any(word in all_text for word in ['tutorial', 'how to', 'learn', 'explain']):
#             content_type = 'tutorial'
#         elif any(word in all_text for word in ['interview', 'conversation', 'discuss']):
#             content_type = 'interview'
#         elif len(segments) > 0 and len(set(seg.get('speaker', 'unknown') for seg in segments)) > 2:
#             content_type = 'presentation'
        
#         return {
#             'content_type': content_type,
#             'narrative_structure': {
#                 'has_clear_beginning': True,
#                 'has_development': True,
#                 'has_climax_or_peak': False,
#                 'has_resolution': True,
#                 'content_flow': 'linear'
#             },
#             'main_themes': ['general'],
#             'content_characteristics': {
#                 'is_educational': content_type == 'tutorial',
#                 'is_entertaining': content_type not in ['tutorial', 'presentation'],
#                 'is_narrative_driven': False,
#                 'has_demonstrations': content_type == 'tutorial',
#                 'interaction_level': 'monologue'
#             },
#             'engagement_patterns': {
#                 'peak_likely_at': 'middle',
#                 'content_density': 'medium',
#                 'emotional_variation': 'medium'
#             },
#             'short_form_potential': {
#                 'best_segment_types': ['highlights', 'key_points'],
#                 'ideal_segment_length': '30-60 seconds',
#                 'key_selection_criteria': ['completeness', 'engagement']
#             }
#         }
    
#     def _fallback_intent_analysis(self, user_prompt: str) -> Dict:
#         """Fallback intent analysis when LLM analysis fails."""
#         # Simple keyword-based intent detection
#         prompt_lower = user_prompt.lower()
        
#         intent_type = 'general'
#         if any(word in prompt_lower for word in ['climax', 'peak', 'best', 'highlight']):
#             intent_type = 'highlights'
#         elif any(word in prompt_lower for word in ['comedy', 'funny', 'humor']):
#             intent_type = 'comedy'
#         elif any(word in prompt_lower for word in ['emotional', 'touching', 'moving']):
#             intent_type = 'emotional'
#         elif any(word in prompt_lower for word in ['educational', 'learn', 'tutorial']):
#             intent_type = 'educational'
        
#         return {
#             'intent_interpretation': {
#                 'literal_request': user_prompt,
#                 'contextual_intent': f'User wants {intent_type} content',
#                 'content_alignment': 'medium',
#                 'adaptation_strategy': 'Find segments matching keywords'
#             },
#             'selection_criteria': {
#                 'primary_factors': ['keyword_match', 'quality'],
#                 'secondary_factors': ['duration', 'completeness'],
#                 'content_position_preference': 'any',
#                 'quality_thresholds': {
#                     'minimum_engagement': 0.3,
#                     'minimum_completeness': 0.3,
#                     'minimum_relevance': 0.3
#                 }
#             },
#             'content_requirements': {
#                 'emotional_tone': intent_type,
#                 'must_include': [],
#                 'must_avoid': ['silence', 'intro'],
#                 'duration_preference': '30-60 seconds',
#                 'standalone_viability': 'medium'
#             },
#             'confidence_assessment': {
#                 'intent_clarity': 0.5,
#                 'content_availability': 0.5,
#                 'match_likelihood': 0.5,
#                 'overall_confidence': 0.3
#             }
#         }
    
#     # ================================
#     # PHASE 2: HELPER METHODS FOR MULTI-PASS ANALYSIS
#     # ================================
    
#     async def _validate_segments_with_llm(self, segments, user_prompt, content_overview, intent_analysis):
#         """LLM-based validation of segment quality and relevance."""
#         try:
#             if not self.ollama_client:
#                 return {}
            
#             validation_results = {}
            
#             for segment in segments:
#                 segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
#                 segment_text = segment.get('segment_text', '')
                
#                 # Create validation prompt
#                 validation_prompt = f"""
#                 Analyze this video segment for quality and relevance:
                
#                 User Request: "{user_prompt}"
#                 Content Type: {content_overview.get('content_type', 'unknown')}
#                 Main Themes: {', '.join(content_overview.get('main_themes', []))}
#                 User Intent: {intent_analysis.get('intent_interpretation', {}).get('contextual_intent', '')}
                
#                 Segment Text: "{segment_text}"
#                 Duration: {segment.get('duration', 0):.1f}s
                
#                 Rate this segment (0-1) on:
#                 1. Relevance to user request
#                 2. Content quality
#                 3. Standalone viability
#                 4. Engagement potential
                
#                 Respond with: score:X.X reasoning:"explanation"
#                 """
                
#                 try:
#                     response = await self.ollama_client.generate_response(
#                         validation_prompt,
#                         "mistral-small3.2:latest"
#                     )
                    
#                     # Parse LLM response
#                     score = 0.5  # default
#                     if 'score:' in response:
#                         score_str = response.split('score:')[1].split()[0]
#                         try:
#                             score = float(score_str)
#                         except:
#                             pass
                    
#                     validation_results[segment_id] = {
#                         'score': max(0, min(1, score)),
#                         'reasoning': response,
#                         'method': 'llm_validation'
#                     }
                    
#                 except Exception as e:
#                     validation_results[segment_id] = {
#                         'score': 0.5,
#                         'error': str(e),
#                         'method': 'llm_validation'
#                     }
            
#             return validation_results
            
#         except Exception as e:
#             self.logger.error(f"LLM validation failed: {e}")
#             return {}
    
#     def _validate_segments_heuristically(self, segments, audio_analysis, scene_analysis, intent_analysis):
#         """Heuristic-based validation using audio and scene analysis."""
#         validation_results = {}
        
#         try:
#             for segment in segments:
#                 segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
                
#                 # Heuristic scoring factors
#                 factors = {
#                     'duration_score': self._score_duration_appropriateness(segment),
#                     'audio_quality_score': self._score_audio_quality(segment, audio_analysis),
#                     'scene_transition_score': self._score_scene_transitions(segment, scene_analysis),
#                     'content_density_score': self._score_content_density(segment)
#                 }
                
#                 # Weighted heuristic score
#                 heuristic_score = (
#                     factors['duration_score'] * 0.25 +
#                     factors['audio_quality_score'] * 0.3 +
#                     factors['scene_transition_score'] * 0.25 +
#                     factors['content_density_score'] * 0.2
#                 )
                
#                 validation_results[segment_id] = {
#                     'score': heuristic_score,
#                     'factors': factors,
#                     'method': 'heuristic_validation'
#                 }
        
#         except Exception as e:
#             self.logger.error(f"Heuristic validation failed: {e}")
        
#         return validation_results
    
#     def _validate_content_coherence(self, segments, audio_analysis, content_overview):
#         """Validate content coherence and narrative flow."""
#         validation_results = {}
        
#         try:
#             content_type = content_overview.get('content_type', 'unknown')
#             narrative_structure = content_overview.get('narrative_structure', {})
            
#             for segment in segments:
#                 segment_id = f"{segment.get('start_time', 0)}-{segment.get('end_time', 0)}"
#                 segment_text = segment.get('segment_text', '')
                
#                 # Coherence scoring
#                 coherence_factors = {
#                     'text_coherence': self._score_text_coherence(segment_text),
#                     'narrative_fit': self._score_narrative_fit(segment, content_type, narrative_structure),
#                     'standalone_quality': self._score_standalone_quality(segment_text, content_type)
#                 }
                
#                 coherence_score = sum(coherence_factors.values()) / len(coherence_factors)
                
#                 validation_results[segment_id] = {
#                     'score': coherence_score,
#                     'factors': coherence_factors,
#                     'method': 'coherence_validation'
#                 }
        
#         except Exception as e:
#             self.logger.error(f"Coherence validation failed: {e}")
        
#         return validation_results
    
#     async def _refine_segment_timing(self, segments, audio_analysis, scene_analysis):
#         """Refine segment timing based on audio and scene analysis."""
#         refined_segments = []
        
#         try:
#             for segment in segments:
#                 # Extract relevant audio segments
#                 start_time = segment.get('start_time', 0)
#                 end_time = segment.get('end_time', 0)
                
#                 # Find optimal boundaries
#                 optimal_start = self._find_optimal_start_boundary(start_time, audio_analysis, scene_analysis)
#                 optimal_end = self._find_optimal_end_boundary(end_time, audio_analysis, scene_analysis)
                
#                 # Create refined segment
#                 refined_segment = segment.copy()
#                 refined_segment.update({
#                     'start_time': optimal_start,
#                     'end_time': optimal_end,
#                     'duration': optimal_end - optimal_start,
#                     'timing_refined': True,
#                     'original_start': start_time,
#                     'original_end': end_time
#                 })
                
#                 refined_segments.append(refined_segment)
        
#         except Exception as e:
#             self.logger.error(f"Timing refinement failed: {e}")
#             return segments
        
#         return refined_segments
    
#     def _resolve_segment_overlaps(self, segments, video_info):
#         """Resolve overlapping segments through intelligent merging or splitting."""
#         resolved_segments = []
        
#         try:
#             # Sort segments by start time
#             sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
            
#             for i, segment in enumerate(sorted_segments):
#                 if i == 0:
#                     resolved_segments.append(segment)
#                     continue
                
#                 prev_segment = resolved_segments[-1]
#                 current_start = segment.get('start_time', 0)
#                 prev_end = prev_segment.get('end_time', 0)
                
#                 # Check for overlap
#                 if current_start < prev_end:
#                     # Resolve overlap
#                     if self._should_merge_segments(prev_segment, segment):
#                         # Merge segments
#                         merged_segment = self._merge_segments(prev_segment, segment)
#                         resolved_segments[-1] = merged_segment
#                     else:
#                         # Adjust boundaries
#                         split_point = (prev_end + current_start) / 2
#                         prev_segment['end_time'] = split_point
#                         prev_segment['duration'] = split_point - prev_segment.get('start_time', 0)
                        
#                         adjusted_segment = segment.copy()
#                         adjusted_segment['start_time'] = split_point
#                         adjusted_segment['duration'] = adjusted_segment.get('end_time', 0) - split_point
                        
#                         resolved_segments.append(adjusted_segment)
#                 else:
#                     resolved_segments.append(segment)
        
#         except Exception as e:
#             self.logger.error(f"Overlap resolution failed: {e}")
#             return segments
        
#         return resolved_segments
    
#     async def _analyze_and_fill_gaps(self, segments, comprehensive_segments, user_prompt, content_overview, intent_analysis):
#         """Analyze gaps between segments and potentially fill with relevant content."""
#         try:
#             # Sort segments by start time
#             sorted_segments = sorted(segments, key=lambda x: x.get('start_time', 0))
            
#             # Identify gaps
#             gaps = []
#             for i in range(len(sorted_segments) - 1):
#                 gap_start = sorted_segments[i].get('end_time', 0)
#                 gap_end = sorted_segments[i + 1].get('start_time', 0)
#                 gap_duration = gap_end - gap_start
                
#                 if gap_duration > 5.0:  # Only consider significant gaps
#                     gaps.append({
#                         'start': gap_start,
#                         'end': gap_end,
#                         'duration': gap_duration,
#                         'position': i + 1
#                     })
            
#             # Analyze each gap for potential content
#             filled_segments = sorted_segments.copy()
            
#             for gap in gaps:
#                 gap_candidates = [
#                     seg for seg in comprehensive_segments
#                     if (seg.get('start_time', 0) >= gap['start'] and
#                         seg.get('end_time', 0) <= gap['end'])
#                 ]
                
#                 if gap_candidates:
#                     # Evaluate gap candidates
#                     best_candidate = await self._evaluate_gap_candidates(
#                         gap_candidates, user_prompt, content_overview, intent_analysis
#                     )
                    
#                     if best_candidate and best_candidate.get('relevance_score', 0) > 0.6:
#                         filled_segments.insert(gap['position'], best_candidate)
            
#             return filled_segments
        
#         except Exception as e:
#             self.logger.error(f"Gap analysis failed: {e}")
#             return segments
    
#     async def _calculate_enhanced_quality_score(self, segment, user_prompt, content_overview, intent_analysis):
#         """Calculate enhanced quality score for Phase 2 optimization."""
#         try:
#             # Base scores from existing validation
#             base_score = segment.get('composite_validation_score', 0.5)
            
#             # Enhanced scoring factors
#             enhanced_factors = {
#                 'contextual_relevance': self._score_contextual_relevance(segment, user_prompt, intent_analysis),
#                 'content_type_alignment': self._score_content_type_alignment(segment, content_overview),
#                 'engagement_potential': self._score_engagement_potential(segment),
#                 'technical_quality': self._score_technical_quality(segment),
#                 'narrative_value': self._score_narrative_value(segment, content_overview)
#             }
            
#             # Weighted enhanced score
#             enhanced_score = (
#                 base_score * 0.3 +
#                 enhanced_factors['contextual_relevance'] * 0.25 +
#                 enhanced_factors['content_type_alignment'] * 0.15 +
#                 enhanced_factors['engagement_potential'] * 0.15 +
#                 enhanced_factors['technical_quality'] * 0.1 +
#                 enhanced_factors['narrative_value'] * 0.05
#             )
            
#             return max(0, min(1, enhanced_score))
        
#         except Exception as e:
#             self.logger.error(f"Enhanced quality scoring failed: {e}")
#             return segment.get('composite_validation_score', 0.5)
    
#     def _optimize_final_selection(self, quality_scored_segments, target_duration, max_shorts, intent_analysis):
#         """Optimize final segment selection based on quality scores and constraints."""
#         try:
#             # Sort by quality score
#             sorted_segments = sorted(quality_scored_segments, key=lambda x: x.get('quality_score', 0), reverse=True)
            
#             # Apply selection constraints
#             min_duration, max_duration = target_duration
#             selected_segments = []
            
#             for segment in sorted_segments:
#                 if len(selected_segments) >= max_shorts:
#                     break
                
#                 duration = segment.get('duration', 0)
#                 if min_duration <= duration <= max_duration:
#                     # Check for diversity
#                     if self._ensures_selection_diversity(segment, selected_segments, intent_analysis):
#                         selected_segments.append(segment)
            
#             return selected_segments
        
#         except Exception as e:
#             self.logger.error(f"Final selection optimization failed: {e}")
#             return quality_scored_segments[:max_shorts]
    
#     # Helper scoring methods
#     def _score_duration_appropriateness(self, segment):
#         """Score segment duration appropriateness for short-form content."""
#         duration = segment.get('duration', 0)
#         if 15 <= duration <= 60:
#             return 1.0
#         elif 10 <= duration < 15 or 60 < duration <= 90:
#             return 0.7
#         else:
#             return 0.3
    
#     def _score_audio_quality(self, segment, audio_analysis):
#         """Score audio quality based on transcription confidence and clarity."""
#         # Implementation would analyze audio transcription quality
#         return 0.7  # Placeholder
    
#     def _score_scene_transitions(self, segment, scene_analysis):
#         """Score scene transition quality within segment."""
#         # Implementation would analyze scene breaks and transitions
#         return 0.6  # Placeholder
    
#     def _score_content_density(self, segment):
#         """Score content density and information richness."""
#         text_length = len(segment.get('segment_text', ''))
#         duration = segment.get('duration', 1)
#         words_per_second = text_length / max(duration, 1) / 5  # Approximate words
        
#         if 2 <= words_per_second <= 4:
#             return 1.0
#         elif 1 <= words_per_second < 2 or 4 < words_per_second <= 6:
#             return 0.7
#         else:
#             return 0.4
    
#     def _score_text_coherence(self, text):
#         """Score text coherence and readability."""
#         if not text:
#             return 0.0
        
#         # Simple coherence metrics
#         sentences = text.split('.')
#         avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
#         if 8 <= avg_sentence_length <= 20:
#             return 0.8
#         else:
#             return 0.5
    
#     def _score_narrative_fit(self, segment, content_type, narrative_structure):
#         """Score how well segment fits the overall narrative."""
#         # Implementation would analyze narrative position and structure
#         return 0.6  # Placeholder
    
#     def _score_standalone_quality(self, text, content_type):
#         """Score standalone quality of segment text."""
#         if not text:
#             return 0.0
        
#         # Check for complete thoughts, context
#         has_complete_thought = '.' in text or '!' in text or '?' in text
#         has_context_clues = any(word in text.lower() for word in ['this', 'that', 'here', 'now', 'today'])
        
#         score = 0.5
#         if has_complete_thought:
#             score += 0.3
#         if has_context_clues:
#             score += 0.2
        
#         return min(1.0, score)
    
#     def _find_optimal_start_boundary(self, start_time, audio_analysis, scene_analysis):
#         """Find optimal start boundary based on audio and scene analysis."""
#         # Implementation would find natural start points
#         return start_time  # Placeholder
    
#     def _find_optimal_end_boundary(self, end_time, audio_analysis, scene_analysis):
#         """Find optimal end boundary based on audio and scene analysis."""
#         # Implementation would find natural end points
#         return end_time  # Placeholder
    
#     def _should_merge_segments(self, segment1, segment2):
#         """Determine if two overlapping segments should be merged."""
#         # Check content similarity, timing, and thematic coherence
#         return False  # Placeholder - conservative approach
    
#     def _merge_segments(self, segment1, segment2):
#         """Merge two segments into one."""
#         return {
#             'start_time': min(segment1.get('start_time', 0), segment2.get('start_time', 0)),
#             'end_time': max(segment1.get('end_time', 0), segment2.get('end_time', 0)),
#             'segment_text': f"{segment1.get('segment_text', '')} {segment2.get('segment_text', '')}",
#             'merged_from': [segment1, segment2],
#             'generation_method': 'phase2_merge'
#         }
    
#     async def _evaluate_gap_candidates(self, candidates, user_prompt, content_overview, intent_analysis):
#         """Evaluate candidates for filling gaps between segments."""
#         if not candidates:
#             return None
        
#         # Score each candidate
#         scored_candidates = []
#         for candidate in candidates:
#             relevance_score = self._score_contextual_relevance(candidate, user_prompt, intent_analysis)
#             candidate['relevance_score'] = relevance_score
#             scored_candidates.append(candidate)
        
#         # Return best candidate
#         return max(scored_candidates, key=lambda x: x.get('relevance_score', 0))
    
#     def _score_contextual_relevance(self, segment, user_prompt, intent_analysis):
#         """Score contextual relevance to user prompt and intent."""
#         segment_text = segment.get('segment_text', '').lower()
#         user_prompt_lower = user_prompt.lower()
        
#         # Simple keyword matching (can be enhanced with semantic analysis)
#         keywords = user_prompt_lower.split()
#         matches = sum(1 for keyword in keywords if keyword in segment_text)
#         keyword_score = matches / max(len(keywords), 1)
        
#         # Intent alignment score
#         intent_keywords = intent_analysis.get('selection_criteria', {}).get('primary_factors', [])
#         intent_matches = sum(1 for factor in intent_keywords if factor.lower() in segment_text)
#         intent_score = intent_matches / max(len(intent_keywords), 1) if intent_keywords else 0.5
        
#         return (keyword_score * 0.6 + intent_score * 0.4)
    
#     def _score_content_type_alignment(self, segment, content_overview):
#         """Score alignment with identified content type."""
#         content_type = content_overview.get('content_type', 'unknown')
#         segment_text = segment.get('segment_text', '').lower()
        
#         # Content type specific scoring
#         if content_type == 'tutorial':
#             tutorial_keywords = ['learn', 'how', 'step', 'method', 'technique', 'way', 'process']
#             matches = sum(1 for keyword in tutorial_keywords if keyword in segment_text)
#             return min(1.0, matches / 3)
#         elif content_type == 'entertainment':
#             entertainment_keywords = ['fun', 'funny', 'amazing', 'incredible', 'wow', 'great']
#             matches = sum(1 for keyword in entertainment_keywords if keyword in segment_text)
#             return min(1.0, matches / 2)
#         else:
#             return 0.5
    
#     def _score_engagement_potential(self, segment):
#         """Score potential for audience engagement."""
#         text = segment.get('segment_text', '').lower()
        
#         # Engagement indicators
#         engagement_words = ['amazing', 'incredible', 'important', 'key', 'best', 'must', 'wow', 'great']
#         questions = text.count('?')
#         exclamations = text.count('!')
        
#         engagement_score = 0.5
#         engagement_score += min(0.3, len([w for w in engagement_words if w in text]) * 0.1)
#         engagement_score += min(0.1, questions * 0.05)
#         engagement_score += min(0.1, exclamations * 0.05)
        
#         return min(1.0, engagement_score)
    
#     def _score_technical_quality(self, segment):
#         """Score technical quality of the segment."""
#         # Placeholder for technical quality metrics
#         return 0.7
    
#     def _score_narrative_value(self, segment, content_overview):
#         """Score narrative value within content structure."""
#         # Placeholder for narrative analysis
#         return 0.6
    
#     def _ensures_selection_diversity(self, segment, selected_segments, intent_analysis):
#         """Ensure selection diversity to avoid repetitive content."""
#         if not selected_segments:
#             return True
        
#         # Check for content similarity (simplified)
#         segment_text = segment.get('segment_text', '').lower()
#         for selected in selected_segments:
#             selected_text = selected.get('segment_text', '').lower()
#             # Simple overlap check
#             words_segment = set(segment_text.split())
#             words_selected = set(selected_text.split())
#             overlap = len(words_segment & words_selected) / max(len(words_segment | words_selected), 1)
            
#             if overlap > 0.7:  # Too similar
#                 return False
        
#         return True
