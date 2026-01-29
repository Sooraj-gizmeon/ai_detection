# src/content_analysis/prompt_based_analyzer.py
"""Prompt-based content analyzer for theme-specific short video creation"""

import logging
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..ai_integration.llm_provider import create_llm_provider, ManagedLLMProvider


class PromptBasedAnalyzer:
    """
    Analyzes video content based on user prompts to find theme-specific segments.
    Supports prompts like "climax scenes", "comedy shorts", "emotional parts", etc.
    """
    
    def __init__(self, ollama_client=None):
        """
        Initialize prompt-based analyzer.
        
        Args:
            ollama_client: Ollama client for AI analysis (used as fallback or when not using OpenAI)
        """
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize configurable LLM provider
        self.llm_provider = None
        
        # Note: LLM provider selection is now done per-request based on llm_provider parameter
        self.logger.info("PromptBasedAnalyzer initialized - LLM provider will be determined per request")
        
        # Initialize predefined prompt templates for common themes
        # CRITICAL FIX: This initialization was moved from _get_llm_provider method where it was
        # unreachable code after return statements, causing AttributeError: 'list' has no attribute 'items'
        self.theme_templates = {
            'climax': {
                'keywords': ['climax', 'peak', 'highest point', 'crucial moment', 'turning point'],
                'description': 'intense, high-energy, dramatic moments with emotional peaks',
                'emotional_indicators': ['excitement', 'tension', 'drama', 'intensity'],
                'audio_patterns': ['increased volume', 'rapid speech', 'emphasis', 'passion'],
                'visual_patterns': ['action', 'close-ups', 'dramatic lighting', 'movement']
            },
            'comedy': {
                'keywords': ['funny', 'humor', 'joke', 'laugh', 'amusing', 'hilarious'],
                'description': 'humorous, entertaining, light-hearted moments',
                'emotional_indicators': ['joy', 'amusement', 'laughter', 'playfulness'],
                'audio_patterns': ['laughter', 'funny voices', 'comedic timing', 'punchlines'],
                'visual_patterns': ['expressions', 'gestures', 'reactions', 'visual gags']
            },
            'emotional': {
                'keywords': ['emotional', 'touching', 'heartfelt', 'moving', 'inspiring'],
                'description': 'emotionally engaging, touching, inspiring moments',
                'emotional_indicators': ['sadness', 'joy', 'inspiration', 'empathy', 'connection'],
                'audio_patterns': ['soft speech', 'emotional tone', 'pauses', 'sincerity'],
                'visual_patterns': ['close-ups', 'expressions', 'intimate moments', 'reactions']
            },
            'educational': {
                'keywords': ['learn', 'explain', 'tutorial', 'how to', 'educational'],
                'description': 'informative, instructional, educational content',
                'emotional_indicators': ['clarity', 'understanding', 'curiosity', 'enlightenment'],
                'audio_patterns': ['clear speech', 'explanatory tone', 'structured delivery'],
                'visual_patterns': ['diagrams', 'demonstrations', 'step-by-step visuals']
            },
            'action': {
                'keywords': ['action', 'fast', 'dynamic', 'movement', 'energy', 'exciting'],
                'description': 'high-energy, dynamic, action-packed moments',
                'emotional_indicators': ['excitement', 'adrenaline', 'intensity', 'energy'],
                'audio_patterns': ['fast speech', 'dynamic audio', 'sound effects', 'music'],
                'visual_patterns': ['rapid cuts', 'movement', 'dynamic angles', 'effects']
            },
            'motivational': {
                'keywords': ['motivational', 'inspiring', 'uplifting', 'positive', 'encourage'],
                'description': 'inspiring, uplifting, motivational content',
                'emotional_indicators': ['inspiration', 'hope', 'determination', 'positivity'],
                'audio_patterns': ['passionate delivery', 'crescendo', 'emphasis', 'conviction'],
                'visual_patterns': ['uplifting imagery', 'positive expressions', 'success visuals']
            },
            'general': {
                'keywords': ['engagement', 'interesting', 'captivating', 'attention'],
                'description': 'generally engaging and interesting content',
                'emotional_indicators': ['interest', 'curiosity', 'engagement', 'attention'],
                'audio_patterns': ['varied tone', 'clear delivery', 'engaging pace'],
                'visual_patterns': ['varied shots', 'good composition', 'visual interest']
            }
        }
        
        # Initialize intelligent climax detector
        from .intelligent_climax_detector import IntelligentClimaxDetector
        self.climax_detector = IntelligentClimaxDetector(ollama_client)
        
        # Variation manager (will be set externally)
        self.variation_manager = None
    
    def _should_use_openai(self) -> bool:
        """Check if OpenAI should be used based on environment configuration (legacy method)"""
        model_name = os.getenv('MODEL_NAME', '').lower()
        return model_name in ['gpt4', 'gpt-4', 'openai'] or 'gpt' in model_name or 'openai' in model_name
    
    def _should_use_openai_for_provider(self, llm_provider: str) -> bool:
        """Check if OpenAI should be used based on the llm_provider parameter"""
        if llm_provider.lower() == "openai":
            return True
        elif llm_provider.lower() == "ollama":
            return False
        else:
            # Fallback to environment variable if provider is not explicitly specified
            return self._should_use_openai()

    def _detect_actor_from_prompt(self, user_prompt: str, appearances_per_actor: Dict[str, List[int]]) -> List[str]:
        """Simple actor name matching from prompt. Returns list of matched actor names."""
        prompt_lower = user_prompt.lower()
        matches = []
        for actor in appearances_per_actor.keys():
            # match by whole name or last name token to be permissive
            if actor.lower() in prompt_lower or any(tok in prompt_lower for tok in actor.lower().split()):
                matches.append(actor)
        self.logger.info(f"Detected actors from prompt: {matches}")      
        return matches
    
    def _detect_object_from_prompt(self, user_prompt: str, appearances_per_object: Dict[str, List[Dict]], object_labels: Dict[str, str] = None) -> List[str]:
        """Simple object detection from prompt. Returns list of matched object_ids."""
        prompt_lower = user_prompt.lower()
        matches = []
        
        if object_labels is None:
            object_labels = {}
        
        for object_id, segments in appearances_per_object.items():
            label = object_labels.get(object_id, "").lower()
            # Check if the object label appears in the prompt
            if label and label in prompt_lower:
                matches.append(object_id)
        
        # If no specific matches, but prompt mentions objects generally, return all objects with high scores
        if not matches and any(word in prompt_lower for word in ['object', 'objects', 'item', 'items', 'thing', 'things']):
            # Return objects sorted by their average score
            scored_objects = []
            for object_id, segments in appearances_per_object.items():
                avg_score = sum(seg['score'] for seg in segments) / len(segments) if segments else 0
                scored_objects.append((object_id, avg_score))
            scored_objects.sort(key=lambda x: x[1], reverse=True)
            matches = [obj_id for obj_id, score in scored_objects[:5]]  # Top 5 objects
            
        self.logger.info(f"Detected objects from prompt: {matches}")
        return matches
    
    async def _get_llm_provider(self, llm_provider: str = "ollama"):
        """Get configured LLM provider for textual analysis"""
        should_use_openai = self._should_use_openai_for_provider(llm_provider)
        
        if should_use_openai:
            self.logger.info(f"Using OpenAI for textual analysis (llm_provider={llm_provider})")
            return create_llm_provider("openai")
        else:
            self.logger.info(f"Using Ollama for textual analysis (llm_provider={llm_provider})")
            return create_llm_provider("ollama", self.ollama_client)
    
    async def analyze_segments(self,
                             user_prompt: str,
                             candidate_segments: List[Dict],
                             audio_analysis: Dict,
                             vision_analysis: Optional[Dict],
                             scene_analysis: Dict,
                             content_overview: Optional[Dict] = None,
                             intent_analysis: Optional[Dict] = None,
                             llm_provider: str = "ollama") -> Dict:
        """
        Phase 2-compatible segment analysis method.
        
        This method is specifically designed for Phase 2 multi-pass analysis
        and provides enhanced contextual understanding.
        
        Args:
            user_prompt: User's prompt describing desired content type
            candidate_segments: List of candidate segments to analyze
            audio_analysis: Audio transcription and analysis
            vision_analysis: Visual content analysis (optional)
            scene_analysis: Scene break analysis
            content_overview: Content overview from Phase 1 (optional)
            intent_analysis: Intent analysis from Phase 1 (optional)
            
        Returns:
            Dict containing analysis results with enhanced Phase 2 metadata
        """
        try:
            self.logger.info(f"Analyzing {len(candidate_segments)} segments with Phase 2 compatibility")
            
            # Use enhanced contextual analysis if available
            if self.ollama_client and content_overview and intent_analysis:
                return await self._contextual_analysis_approach(
                    user_prompt, candidate_segments, audio_analysis,
                    vision_analysis, scene_analysis, content_overview, intent_analysis,
                    llm_provider=llm_provider
                )
            else:
                # Fallback to enhanced heuristic analysis
                return self._enhanced_heuristic_analysis(
                    user_prompt, candidate_segments, audio_analysis,
                    vision_analysis, scene_analysis
                )
                
        except Exception as e:
            self.logger.error(f"Segment analysis failed: {e}")
            return {
                'status': 'error',
                'error': f"Segment analysis failed: {e}",
                'segments': []
            }

    async def analyze_with_prompt(self, 
                                user_prompt: str,
                                audio_analysis: Dict,
                                vision_analysis: Optional[Dict],
                                scene_analysis: Dict,
                                video_info: Dict,
                                candidate_segments: List[Dict],
                                object_detection_results: Optional[Dict] = None,
                                content_overview: Optional[Dict] = None,
                                intent_analysis: Optional[Dict] = None,
                                llm_provider: str = "ollama") -> Dict:
        """
        Analyze video content based on user prompt to find matching segments.
        Enhanced with object detection results and contextual understanding.
        
        Args:
            user_prompt: User's prompt describing desired content type
            audio_analysis: Audio transcription and analysis
            vision_analysis: Visual content analysis (optional)
            scene_analysis: Scene detection results
            video_info: Video metadata
            candidate_segments: All potential segments to analyze
            object_detection_results: Object detection results (optional)
            content_overview: Content overview analysis (optional)
            intent_analysis: User intent analysis (optional)
            llm_provider: LLM provider to use ('openai' or 'ollama')
            
        Returns:
            Analysis results with prompt-matched segments
        """
        try:
            # Defensive type checking for all input parameters
            if not isinstance(audio_analysis, dict):
                self.logger.error(f"audio_analysis is not a dict: {type(audio_analysis)}")
                audio_analysis = {}
            if not isinstance(scene_analysis, dict):
                self.logger.error(f"scene_analysis is not a dict: {type(scene_analysis)}")
                scene_analysis = {}
            if not isinstance(video_info, dict):
                self.logger.error(f"video_info is not a dict: {type(video_info)}")
                video_info = {}
            if not isinstance(candidate_segments, list):
                self.logger.error(f"candidate_segments is not a list: {type(candidate_segments)}")
                candidate_segments = []
            if object_detection_results is not None and not isinstance(object_detection_results, dict):
                self.logger.error(f"object_detection_results is not a dict: {type(object_detection_results)}")
                object_detection_results = None
            if content_overview is not None and not isinstance(content_overview, dict):
                self.logger.error(f"content_overview is not a dict: {type(content_overview)}")
                content_overview = None
            if intent_analysis is not None and not isinstance(intent_analysis, dict):
                self.logger.error(f"intent_analysis is not a dict: {type(intent_analysis)}")
                intent_analysis = None
                
            self.logger.info(f"Analyzing content with user prompt: '{user_prompt}'")

            # Celebrity index support: check for celebrity_index_path in video_info or intent_analysis
            celebrity_index_path = None
            if video_info and isinstance(video_info, dict):
                celebrity_index_path = video_info.get('celebrity_index_path')
            if not celebrity_index_path and intent_analysis and isinstance(intent_analysis, dict):
                celebrity_index_path = intent_analysis.get('celebrity_index_path')

            appearances_per_actor, actor_conf = {}, {}
            appearances_per_object, object_conf, object_labels = {}, {}, {}
            if celebrity_index_path:
                try:
                    from ..face_insights.celebrity_index import load_celebrity_index, load_object_index, actor_coverage_for_segment, compute_celebrity_score, object_coverage_for_segment, compute_object_score
                    appearances_per_actor, actor_conf = load_celebrity_index(celebrity_index_path)
                    appearances_per_object, object_conf, object_labels = load_object_index(celebrity_index_path)
                    self.logger.info(f"Loaded celebrity index with {len(appearances_per_actor)} actors and {len(appearances_per_object)} objects from {celebrity_index_path}")
                except Exception as e:
                    self.logger.warning(f"Could not load celebrity/object index: {e}")

            # If user explicitly asked for actor clips, do actor-first flow
            actor_matches = []
            if appearances_per_actor:
                actor_matches = self._detect_actor_from_prompt(user_prompt, appearances_per_actor)

            if actor_matches:
                # STRICT IMPLEMENTATION: Use ONLY precomputed actor timestamps
                # Do NOT recompute confidence scores, do NOT use candidate segments
                # Do NOT fall back to random segment generation
                
                self.logger.info(
                    f"🎯 STRICT ACTOR MODE: Extracting segments ONLY from precomputed "
                    f"'{actor_matches}' timestamps (user_prompt: '{user_prompt}')"
                )
                
                try:
                    from .actor_segment_extractor import ActorSegmentExtractor
                    extractor = ActorSegmentExtractor()
                    
                    # Get video duration from video_info
                    video_duration = video_info.get('duration', 3600) if video_info else 3600
                    
                    # Extract segments exclusively from precomputed actor appearances
                    selected = extractor.extract_multiple_actors_segments(
                        actor_matches,
                        appearances_per_actor,
                        actor_conf,
                        min_duration=60,  # Default minimum
                        max_duration=120,  # Default maximum
                        video_start=0.0,
                        video_end=video_duration
                    )
                    
                    if selected:
                        # Log actor timestamps for verification
                        for actor in actor_matches:
                            actor_timestamps = appearances_per_actor.get(actor, [])
                            if actor_timestamps:
                                timestamps_str = ", ".join([f"{int(t)}s" for t in sorted(set(actor_timestamps))])
                                self.logger.info(
                                    f"✓ Actor '{actor}' appears at: [{timestamps_str}]"
                                )
                        
                        self.logger.info(
                            f"✅ Generated {len(selected)} segments from precomputed actor timestamps "
                            f"(NO fallback to candidate segments, NO recomputed scores)"
                        )
                        
                        return {
                            'status': 'success',
                            'analysis_method': 'actor_only_strict',
                            'user_prompt': user_prompt,
                            'matched_actors': actor_matches,
                            'segments': selected,
                            'celebrity_index_path': celebrity_index_path,
                            'actor_appearances': {actor: sorted(set(appearances_per_actor.get(actor, []))) for actor in actor_matches},
                            'generation_note': 'Segments extracted exclusively from precomputed actor detection timestamps. Confidence scores from precomputed results. NO recomputation.'
                        }
                    else:
                        self.logger.warning(
                            f"No segments generated from precomputed actor timestamps for: {actor_matches}"
                        )
                        
                except Exception as e:
                    self.logger.error(f"Error in actor-only segment extraction: {e}", exc_info=True)
                    # Fall through to other analysis methods below

            # Check for object-based requests
            object_matches = []
            if appearances_per_object:
                object_matches = self._detect_object_from_prompt(user_prompt, appearances_per_object, object_labels)

            if object_matches:
                # Interpret "all clips with X" intent (simple heuristics)
                want_all = any(word in user_prompt.lower() for word in ['all', 'every', 'every clip', 'every clip with', 'all clips', 'clips with'])

                selected = []
                for seg in candidate_segments:
                    per_object = object_coverage_for_segment(seg['start_time'], seg['end_time'], appearances_per_object, object_conf)
                    score = compute_object_score(per_object)
                    seg_copy = seg.copy()
                    seg_copy['object_score'] = score
                    seg_copy['object_details'] = per_object
                    # Ensure object-matched segments get a baseline prompt_match_score so they survive quality filtering
                    if want_all:
                        # keep even low coverage, give small baseline
                        seg_copy['prompt_match_score'] = max(seg_copy.get('prompt_match_score', 0.0), score, 0.05)
                        if score > 0.01:
                            selected.append(seg_copy)
                    else:
                        # Stronger baseline for explicit object requests
                        seg_copy['prompt_match_score'] = max(seg_copy.get('prompt_match_score', 0.0), score, 0.25)
                        if score > 0:
                            selected.append(seg_copy)

                # If we found some matching segments, return them prioritized by object_score
                if selected:
                    selected.sort(key=lambda x: x.get('object_score', 0), reverse=True)
                    return {
                        'status': 'success',
                        'analysis_method': 'object_direct_match',
                        'user_prompt': user_prompt,
                        'matched_objects': object_matches,
                        'segments': selected,
                        'celebrity_index_path': celebrity_index_path
                    }

            # SMART APPROACH: Let LLM understand ANY user prompt intelligently
            self.logger.info(f"Using intelligent LLM-based analysis for prompt: '{user_prompt}'")
            
            # PHASE 1 ENHANCEMENT: Use contextual understanding if available
            if content_overview and intent_analysis:
                self.logger.info("Using enhanced contextual analysis approach")
                return await self._contextual_analysis_approach(
                    user_prompt=user_prompt,
                    candidate_segments=candidate_segments,
                    audio_analysis=audio_analysis,
                    vision_analysis=vision_analysis,
                    scene_analysis=scene_analysis,
                    content_overview=content_overview,
                    intent_analysis=intent_analysis,
                    object_detection_results=object_detection_results,
                    llm_provider=llm_provider
                )
            
            # Use direct LLM analysis without rigid classification
            return await self._intelligent_segment_analysis(
                user_prompt=user_prompt,
                candidate_segments=candidate_segments,
                audio_analysis=audio_analysis,
                vision_analysis=vision_analysis,
                scene_analysis=scene_analysis,
                object_detection_results=object_detection_results,
                llm_provider=llm_provider
            )
            
            # Defensive check: ensure prompt_analysis is a dictionary
            if not isinstance(prompt_analysis, dict):
                self.logger.error(f"prompt_analysis is not a dict, got {type(prompt_analysis)}: {prompt_analysis}")
                prompt_analysis = {
                    'primary_theme': 'general',
                    'theme_config': {},
                    'duration_preferences': {},
                    'quantity_preferences': {},
                    'content_requirements': {}
                }
            
            # Enhance prompt analysis with object detection if available
            if object_detection_results:
                prompt_analysis = self._enhance_prompt_analysis_with_objects(
                    prompt_analysis, object_detection_results
                )
            
            # Score all candidate segments based on prompt matching
            scored_segments = await self._score_segments_for_prompt(
                candidate_segments, 
                prompt_analysis,
                audio_analysis,
                vision_analysis,
                scene_analysis
            )
            
            # Apply score variation before AI enhancement for more diverse selection
            if self.variation_manager:
                scored_segments = self.variation_manager.apply_score_variation(scored_segments)
                self.logger.info(f"Applied score variation to {len(scored_segments)} segments")
            
            # Use AI to enhance segment selection if available
            if self.ollama_client:
                enhanced_segments = await self._ai_enhance_prompt_matching(
                    scored_segments,
                    prompt_analysis,
                    audio_analysis,
                    vision_analysis
                )
            else:
                enhanced_segments = scored_segments
            
            # Select the best segments based on prompt matching
            final_segments = self._select_best_prompt_matches(
                enhanced_segments,
                prompt_analysis,
                max_segments=10,
                variation_manager=getattr(self, 'variation_manager', None)
            )
            
            return {
                'status': 'success',
                'user_prompt': user_prompt,
                'prompt_analysis': prompt_analysis,
                'total_candidates_analyzed': len(candidate_segments),
                'prompt_matched_segments': len(final_segments),
                'segments': final_segments,
                'analysis_method': 'prompt_based_with_ai' if self.ollama_client else 'prompt_based_heuristic'
            }
            
        except Exception as e:
            self.logger.error(f"Error in prompt-based analysis3: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    def _is_climax_request(self, user_prompt: str) -> bool:
        """
        Detect if the user prompt is requesting climax/dramatic content.
        Uses semantic analysis instead of hardcoded keywords.
        """
        prompt_lower = user_prompt.lower()
        
        # First check for explicit non-climax content types that should take priority
        comedy_keywords = ['comedy', 'funny', 'humor', 'humorous', 'joke', 'laugh', 'amusing', 'hilarious']
        emotional_keywords = ['emotional', 'touching', 'moving', 'heartfelt', 'sentimental']
        educational_keywords = ['educational', 'learn', 'tutorial', 'teach', 'explain', 'instructional']
        
        # If it's explicitly comedy, emotional, or educational content, don't treat as climax
        if any(keyword in prompt_lower for keyword in comedy_keywords):
            self.logger.info(f"Prompt contains comedy keywords - skipping climax detection")
            return False
            
        if any(keyword in prompt_lower for keyword in emotional_keywords) and 'peak' not in prompt_lower:
            self.logger.info(f"Prompt contains emotional keywords (non-peak) - skipping climax detection")
            return False
            
        if any(keyword in prompt_lower for keyword in educational_keywords):
            self.logger.info(f"Prompt contains educational keywords - skipping climax detection")
            return False
        
        # Climax-related terms (more specific to avoid false positives)
        climax_indicators = [
            # Direct climax terms
            'climax', 'peak', 'apex', 'zenith', 'pinnacle', 'summit', 'crescendo',
            'high point', 'turning point', 'crucial moment', 'pivotal moment',
            'decisive moment', 'critical point', 'breaking point', 'tipping point',
            
            # Dramatic intensity terms (must be combined with dramatic context)
            'dramatic scenes', 'intense scenes', 'powerful scenes', 'gripping scenes',
            'thrilling scenes', 'electrifying scenes', 'breathtaking scenes',
            'explosive scenes', 'spectacular scenes', 'mind-blowing scenes',
            
            # Best scenes/highlights (only when clearly dramatic context)
            'best dramatic scenes', 'most important scenes', 'essential dramatic scenes',
            'defining dramatic moments', 'iconic dramatic scenes',
            
            # Emotional peaks (specifically intense emotional moments)
            'emotional peaks', 'most emotional scenes', 'emotional climax',
            
            # Action intensity
            'action-packed scenes', 'most exciting action', 'intense action scenes',
            'thrilling action sequences',
            
            # Story structure
            'finale', 'ending climax', 'dramatic conclusion', 'dramatic resolution', 
            'showdown', 'confrontation', 'face-off', 'final battle', 'ultimate dramatic moment'
        ]
        
        # Check for semantic matches (more restrictive scoring)
        climax_score = 0
        for indicator in climax_indicators:
            if indicator in prompt_lower:
                climax_score += 1.0
            # Remove partial matching on single words to avoid false positives
            # Only count partial matches for multi-word phrases
            elif len(indicator.split()) > 1 and any(word in prompt_lower for word in indicator.split() if len(word) > 4):
                climax_score += 0.3  # Reduced partial match score
        
        # Additional context clues (more specific)
        context_clues = [
            'from the movie climax', 'from the film climax', 'dramatic part of',
            'most dramatic', 'most intense', 'most gripping'
        ]
        
        context_score = sum(1 for clue in context_clues if clue in prompt_lower) * 0.5
        
        total_score = climax_score + context_score
        
        # Higher threshold for climax detection to reduce false positives
        is_climax = total_score >= 1.2
        
        self.logger.info(f"Climax detection score: {total_score:.2f} (threshold: 1.2) -> {'YES' if is_climax else 'NO'}")
        
        return is_climax
    
    def _is_comedy_request(self, user_prompt: str) -> bool:
        """
        Detect if the user prompt is specifically requesting comedy/humorous content.
        """
        prompt_lower = user_prompt.lower()
        
        # Comedy-specific keywords
        comedy_keywords = [
            'comedy', 'funny', 'humor', 'humorous', 'joke', 'jokes', 'laugh', 'laughter',
            'amusing', 'hilarious', 'witty', 'entertaining', 'comedic', 'comic',
            'light-hearted', 'light hearted', 'playful', 'silly', 'goofy'
        ]
        
        # Comedy phrases
        comedy_phrases = [
            'comedy moments', 'funny moments', 'funny scenes', 'comedy scenes',
            'humorous parts', 'amusing parts', 'entertaining moments',
            'laugh out loud', 'make me laugh', 'something funny'
        ]
        
        # Check for exact comedy matches
        comedy_score = 0
        for keyword in comedy_keywords:
            if keyword in prompt_lower:
                comedy_score += 1.0
                
        for phrase in comedy_phrases:
            if phrase in prompt_lower:
                comedy_score += 1.5  # Higher weight for specific comedy phrases
        
        is_comedy = comedy_score >= 1.0
        
        self.logger.info(f"Comedy detection score: {comedy_score:.2f} (threshold: 1.0) -> {'YES' if is_comedy else 'NO'}")
        
        return is_comedy
    
    def _analyze_user_prompt(self, user_prompt: str) -> Dict:
        """
        Analyze and parse the user prompt to understand intent.
        
        Args:
            user_prompt: User's prompt string
            
        Returns:
            Parsed prompt analysis
        """
        prompt_lower = user_prompt.lower()
        
        # Identify primary theme(s)
        detected_themes = []
        theme_confidence = {}
        
        # DEFENSIVE CODING: Check if theme_templates is corrupted
        if not isinstance(self.theme_templates, dict):
            self.logger.error(f"CRITICAL: theme_templates corrupted in _analyze_user_prompt! Type: {type(self.theme_templates)}")
            self._reinitialize_theme_templates()
        
        for theme, config in self.theme_templates.items():
            confidence = 0.0
            
            # Special handling for comedy themes when comedy is detected
            if theme == 'comedy' and self._is_comedy_request(user_prompt):
                confidence += 0.8  # Strong boost for explicit comedy requests
                self.logger.info(f"Comedy theme boosted due to explicit comedy request")
            
            # Check for exact theme match
            if theme in prompt_lower:
                confidence += 0.5
            
            # Check for keyword matches
            keyword_matches = sum(1 for keyword in config['keywords'] if keyword in prompt_lower)
            confidence += (keyword_matches / len(config['keywords'])) * 0.3
            
            # Check for emotional indicator matches
            emotion_matches = sum(1 for emotion in config['emotional_indicators'] if emotion in prompt_lower)
            confidence += (emotion_matches / len(config['emotional_indicators'])) * 0.2
            
            if confidence > 0.1:  # Threshold for detection
                detected_themes.append(theme)
                theme_confidence[theme] = confidence
        
        # Parse duration preferences from prompt
        duration_prefs = self._extract_duration_preferences(prompt_lower)
        
        # Parse quantity preferences
        quantity_prefs = self._extract_quantity_preferences(prompt_lower)
        
        # Identify specific content requirements
        content_requirements = self._extract_content_requirements(prompt_lower)
        
        return {
            'original_prompt': user_prompt,
            'detected_themes': detected_themes,
            'theme_confidence': theme_confidence,
            'primary_theme': max(theme_confidence.keys(), key=theme_confidence.get) if theme_confidence else 'general',
            'duration_preferences': duration_prefs,
            'quantity_preferences': quantity_prefs,
            'content_requirements': content_requirements,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _extract_duration_preferences(self, prompt: str) -> Dict:
        """Extract duration preferences from prompt."""
        prefs = {
            'min_duration': 15,  # Default
            'max_duration': 60,  # Default
            'preferred_duration': None
        }
        
        # Look for duration keywords
        if any(word in prompt for word in ['short', 'brief', 'quick']):
            prefs['max_duration'] = 30
        elif any(word in prompt for word in ['long', 'extended', 'detailed']):
            prefs['min_duration'] = 45
            prefs['max_duration'] = 120
        
        return prefs
    
    def _extract_quantity_preferences(self, prompt: str) -> Dict:
        """Extract quantity preferences from prompt."""
        prefs = {
            'min_count': 3,
            'max_count': 10,
            'preferred_count': 5
        }
        
        # Look for quantity keywords
        if any(word in prompt for word in ['few', 'couple', 'some']):
            prefs['max_count'] = 5
        elif any(word in prompt for word in ['many', 'several', 'multiple']):
            prefs['min_count'] = 7
            prefs['max_count'] = 15
        
        return prefs
    
    def _extract_content_requirements(self, prompt: str) -> Dict:
        """Extract specific content requirements from prompt."""
        requirements = {
            'must_have_speech': True,
            'must_have_visuals': False,
            'must_have_people': False,
            'must_have_action': False,
            'avoid_silence': True,
            'prefer_complete_thoughts': True
        }
        
        # Visual requirements
        if any(word in prompt for word in ['visual', 'see', 'show', 'watch', 'look']):
            requirements['must_have_visuals'] = True
        
        # People requirements
        if any(word in prompt for word in ['people', 'person', 'speaker', 'presenter']):
            requirements['must_have_people'] = True
        
        # Action requirements
        if any(word in prompt for word in ['action', 'movement', 'demo', 'demonstrate']):
            requirements['must_have_action'] = True
        
        # Speech requirements
        if any(word in prompt for word in ['silent', 'quiet', 'no speech', 'visual only']):
            requirements['must_have_speech'] = False
            requirements['avoid_silence'] = False
        
        return requirements
    
    async def _score_segments_for_prompt(self,
                                       segments: List[Dict],
                                       prompt_analysis: Dict,
                                       audio_analysis: Dict,
                                       vision_analysis: Optional[Dict],
                                       scene_analysis: Dict) -> List[Dict]:
        """
        Score segments based on how well they match the user prompt.
        
        Args:
            segments: List of candidate segments
            prompt_analysis: Parsed prompt analysis
            audio_analysis: Audio transcription and analysis
            vision_analysis: Visual content analysis
            scene_analysis: Scene detection results
            
        Returns:
            Segments with prompt matching scores
        """
        scored_segments = []
        primary_theme = prompt_analysis['primary_theme']
        
        # DEFENSIVE CODING: Check if theme_templates is corrupted
        if not isinstance(self.theme_templates, dict):
            self.logger.error(f"CRITICAL: theme_templates corrupted in _score_segments_for_prompt! Type: {type(self.theme_templates)}")
            self._reinitialize_theme_templates()
        
        theme_config = self.theme_templates.get(primary_theme, {})
        content_requirements = prompt_analysis['content_requirements']
        
        # Detect refresh rate specs & generalized focus tokens in user prompt for boosting
        user_prompt_text = prompt_analysis.get('original_prompt', '')
        spec_targets = self._detect_refresh_specs(user_prompt_text)
        focus_tokens = self._extract_focus_tokens(user_prompt_text)
        if focus_tokens:
            self.logger.debug(f"Focus tokens extracted from prompt: {sorted(list(focus_tokens))}")

        for segment in segments:
            # Base segment info
            scored_segment = segment.copy()
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Extract segment text from transcription
            segment_text = self._extract_segment_text(audio_analysis, start_time, end_time)
            scored_segment['segment_text'] = segment_text
            
            # Calculate prompt matching score
            prompt_score = 0.0
            
            # 1. Theme-based text analysis (40% weight)
            if segment_text and theme_config:
                text_score = self._score_text_for_theme(segment_text, theme_config)
                prompt_score += text_score * 0.4
            
            # 2. Visual content matching (30% weight)
            if vision_analysis:
                visual_score = self._score_visuals_for_theme(
                    start_time, end_time, vision_analysis, theme_config
                )
                prompt_score += visual_score * 0.3
            
            # 3. Content requirements compliance (20% weight)
            requirement_score = self._score_content_requirements(
                segment, segment_text, vision_analysis, content_requirements
            )
            prompt_score += requirement_score * 0.2
            
            # 4. Duration and timing preferences (10% weight)
            duration_score = self._score_duration_preferences(
                segment, prompt_analysis['duration_preferences']
            )
            prompt_score += duration_score * 0.1
            
            # Store all scoring components
            scored_segment.update({
                'prompt_match_score': prompt_score,
                'text_theme_score': text_score if segment_text and theme_config else 0.0,
                'visual_theme_score': visual_score if vision_analysis else 0.0,
                'requirement_compliance_score': requirement_score,
                'duration_preference_score': duration_score,
                'primary_theme': primary_theme,
                'theme_confidence': prompt_analysis['theme_confidence'].get(primary_theme, 0.0)
            })
            
            if segment_text:
                # SPEC BOOST: If segment text contains requested refresh rate spec, modestly boost
                if spec_targets:
                    seg_specs = self._detect_refresh_specs(segment_text)
                    if seg_specs & spec_targets:
                        scored_segment['prompt_match_score'] = min(1.0, scored_segment['prompt_match_score'] + 0.15)
                        scored_segment['spec_match_boost'] = True
                        scored_segment['spec_matches'] = list(seg_specs & spec_targets)

                # GENERIC FOCUS TOKEN BOOST (numbers+units, component/spec keywords, brand/model tokens)
                if focus_tokens:
                    matched_tokens = self._match_focus_tokens(segment_text, focus_tokens)
                    if matched_tokens:
                        # Boost scaling: base 0.05 + 0.02 * (matched_count-1) capped + extra for numeric-unit presence
                        numeric_unit_present = any(self._is_numeric_unit_token(tok) for tok in matched_tokens)
                        boost = 0.05 + 0.02 * (len(matched_tokens) - 1)
                        if numeric_unit_present:
                            boost += 0.05
                        # If spec present as part of focus tokens and matched, add slight extra synergy
                        if spec_targets and (matched_tokens & spec_targets):
                            boost += 0.03
                        boost = min(0.25, boost)
                        scored_segment['prompt_match_score'] = min(1.0, scored_segment['prompt_match_score'] + boost)
                        scored_segment['focus_token_boost'] = boost
                        scored_segment['focus_token_matches'] = list(matched_tokens)

            scored_segments.append(scored_segment)
        
        # Sort by prompt matching score
        scored_segments.sort(key=lambda x: x['prompt_match_score'], reverse=True)
        
        self.logger.info(f"Scored {len(scored_segments)} segments for prompt '{prompt_analysis['original_prompt']}'")
        self.logger.info(f"Top 5 scores: {[s['prompt_match_score'] for s in scored_segments[:5]]}")
        
        return scored_segments
    
    def _extract_segment_text(self, audio_analysis: Dict, start_time: float, end_time: float) -> str:
        """Extract transcription text for a specific segment."""
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        text_parts = []
        for seg in segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check for overlap with our segment
            if seg_start < end_time and seg_end > start_time:
                text = seg.get('text', '').strip()
                if text:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _score_text_for_theme(self, text: str, theme_config: Dict) -> float:
        """Score text content based on theme matching."""
        if not text or not theme_config:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        
        # Keyword matching (50% of text score)
        keyword_matches = sum(1 for keyword in theme_config['keywords'] if keyword in text_lower)
        keyword_score = (keyword_matches / len(theme_config['keywords'])) if theme_config['keywords'] else 0
        score += keyword_score * 0.5
        
        # Emotional indicator matching (30% of text score)
        emotion_matches = sum(1 for emotion in theme_config['emotional_indicators'] if emotion in text_lower)
        emotion_score = (emotion_matches / len(theme_config['emotional_indicators'])) if theme_config['emotional_indicators'] else 0
        score += emotion_score * 0.3
        
        # Audio pattern matching (20% of text score)
        audio_patterns = theme_config.get('audio_patterns', [])
        pattern_matches = sum(1 for pattern in audio_patterns if any(word in text_lower for word in pattern.split()))
        pattern_score = (pattern_matches / len(audio_patterns)) if audio_patterns else 0
        score += pattern_score * 0.2
        
        return min(1.0, score)
    
    def _score_visuals_for_theme(self, 
                               start_time: float, 
                               end_time: float, 
                               vision_analysis: Dict, 
                               theme_config: Dict) -> float:
        """Score visual content based on theme matching."""
        if not vision_analysis or not theme_config:
            return 0.0
        
        # Find matching visual segments
        vision_segments = vision_analysis.get('segments', [])
        matching_visuals = []
        
        for v_seg in vision_segments:
            v_start = v_seg.get('start_time', 0)
            v_end = v_seg.get('end_time', v_start + 1)
            
            # Check for overlap
            if v_start < end_time and v_end > start_time:
                matching_visuals.append(v_seg)
        
        if not matching_visuals:
            return 0.0
        
        # Score based on visual patterns
        visual_patterns = theme_config.get('visual_patterns', [])
        total_score = 0.0
        
        for visual in matching_visuals:
            scene_type = visual.get('scene_type', '').lower()
            has_people = visual.get('has_people', False)
            has_action = visual.get('has_action', False)
            visual_interest = visual.get('visual_interest', 5) / 10.0  # Normalize to 0-1
            
            visual_score = visual_interest * 0.4  # Base visual interest
            
            # Pattern matching
            for pattern in visual_patterns:
                if pattern.lower() in scene_type:
                    visual_score += 0.2
                elif pattern == 'people' and has_people:
                    visual_score += 0.15
                elif pattern == 'action' and has_action:
                    visual_score += 0.15
            
            total_score += visual_score
        
        # Average across matching visuals
        return min(1.0, total_score / len(matching_visuals))
    
    def _score_content_requirements(self, 
                                  segment: Dict, 
                                  segment_text: str, 
                                  vision_analysis: Optional[Dict],
                                  requirements: Dict) -> float:
        """Score segment based on content requirements compliance."""
        score = 0.0
        requirement_count = len(requirements)
        
        # Must have speech
        if requirements.get('must_have_speech', True):
            if segment_text and len(segment_text.strip()) > 10:
                score += 1.0 / requirement_count
        elif not requirements.get('must_have_speech', True):
            if not segment_text or len(segment_text.strip()) <= 10:
                score += 1.0 / requirement_count
        
        # Must have visuals (if vision analysis available)
        if requirements.get('must_have_visuals', False) and vision_analysis:
            # Check if segment has good visual content
            visual_interest = segment.get('visual_interest', 5)
            if visual_interest >= 6:
                score += 1.0 / requirement_count
        
        # Must have people
        if requirements.get('must_have_people', False):
            has_people = segment.get('has_people', True)  # Default to True
            if has_people:
                score += 1.0 / requirement_count
        
        # Must have action
        if requirements.get('must_have_action', False):
            has_action = segment.get('has_action', False)
            if has_action:
                score += 1.0 / requirement_count
        
        # Avoid silence
        if requirements.get('avoid_silence', True):
            if segment_text and len(segment_text.strip()) > 20:
                score += 1.0 / requirement_count
        
        # Prefer complete thoughts
        if requirements.get('prefer_complete_thoughts', True):
            if segment_text and ('.' in segment_text or '!' in segment_text or '?' in segment_text):
                score += 1.0 / requirement_count
        
        return score
    
    def _score_duration_preferences(self, segment: Dict, duration_prefs: Dict) -> float:
        """Score segment based on duration preferences."""
        duration = segment.get('duration', segment.get('end_time', 0) - segment.get('start_time', 0))
        
        min_dur = duration_prefs.get('min_duration', 15)
        max_dur = duration_prefs.get('max_duration', 60)
        preferred = duration_prefs.get('preferred_duration')
        
        if preferred:
            # Score based on distance from preferred duration
            distance = abs(duration - preferred)
            max_distance = max(preferred - min_dur, max_dur - preferred)
            score = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
        else:
            # Score based on being within range
            if min_dur <= duration <= max_dur:
                # Perfect score if within range
                score = 1.0
            elif duration < min_dur:
                # Penalty for being too short
                score = duration / min_dur
            else:
                # Penalty for being too long
                score = max_dur / duration
        
        return max(0.0, min(1.0, score))
    
    async def _ai_enhance_prompt_matching(self,
                                        scored_segments: List[Dict],
                                        prompt_analysis: Dict,
                                        audio_analysis: Dict,
                                        vision_analysis: Optional[Dict]) -> List[Dict]:
        """
        Use AI to enhance prompt matching by analyzing content with the user prompt.
        
        Args:
            scored_segments: Segments with initial scores
            prompt_analysis: Parsed prompt analysis
            audio_analysis: Audio analysis
            vision_analysis: Visual analysis
            
        Returns:
            Enhanced segments with AI-refined scores
        """
        try:
            # Take top candidates for AI analysis (to manage API costs)
            top_candidates = scored_segments[:20]  # Analyze top 20 candidates
            
            # Create AI prompt for enhanced analysis
            ai_prompt = self._create_ai_enhancement_prompt(
                prompt_analysis['original_prompt'],
                prompt_analysis['primary_theme'],
                top_candidates,
                audio_analysis
            )
            
            # Get AI analysis
            ai_response = await self.ollama_client._make_request(
                prompt=ai_prompt,
                model=self.ollama_client.get_best_model("analysis"),
                cache_key=f"prompt_enhance_{hash(prompt_analysis['original_prompt'] + str(len(top_candidates)))}"
            )
            
            ai_analysis = self.ollama_client._parse_json_response(ai_response)
            
            # Defensive check: ensure ai_analysis is a dictionary
            if not isinstance(ai_analysis, dict):
                self.logger.warning(f"AI analysis parsing returned {type(ai_analysis)} instead of dict: {ai_analysis}")
                ai_analysis = {'segment_refinements': []}
            
            # Apply AI enhancements to segments
            enhanced_segments = self._apply_ai_enhancements(scored_segments, ai_analysis)
            
            self.logger.info(f"AI enhanced {len(top_candidates)} segments for prompt matching")
            
            return enhanced_segments
            
        except Exception as e:
            self.logger.warning(f"AI enhancement failed, using heuristic scores: {e}")
            return scored_segments
    
    def _create_ai_enhancement_prompt(self,
                                    user_prompt: str,
                                    primary_theme: str,
                                    candidates: List[Dict],
                                    audio_analysis: Dict) -> str:
        """Create AI prompt for enhanced segment analysis."""
        
        # Prepare candidate summaries
        candidate_summaries = []
        for i, segment in enumerate(candidates):
            summary = {
                'index': i,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment.get('duration', segment['end_time'] - segment['start_time']),
                'text': segment.get('segment_text', ''),
                'initial_score': segment.get('prompt_match_score', 0.0)
            }
            candidate_summaries.append(summary)
        
        prompt = f"""
        You are an expert video content analyst. A user wants to create short videos with this specific request:
        
        USER REQUEST: "{user_prompt}"
        
        PRIMARY THEME DETECTED: {primary_theme}
        
        I have {len(candidates)} candidate video segments. For each segment, analyze how well it matches the user's specific request.
        
        Consider:
        1. Does the content directly relate to what the user asked for?
        2. Would this segment work well as a standalone short video for social media?
        3. Does it capture the essence of "{primary_theme}" content?
        4. Is the content engaging and complete enough for the user's intent?
        
        CANDIDATE SEGMENTS:
        {json.dumps(candidate_summaries, indent=2)}
        
        For each segment, provide:
        - A refined match score (0.0 to 1.0) based on how well it fulfills the user's specific request
        - A brief explanation of why it does or doesn't match the user's intent
        - Whether it should be prioritized for the final selection
        
        Respond in JSON format:
        {{
            "analysis_summary": "brief overview of how well the candidates match the user request",
            "theme_interpretation": "your understanding of what the user is looking for",
            "segment_refinements": [
                {{
                    "index": segment_index,
                    "refined_score": score_0_to_1,
                    "match_explanation": "explanation of match quality",
                    "priority": "high|medium|low",
                    "recommendation": "include|consider|exclude"
                }}
            ]
        }}
        """
        
        return prompt
    
    def _apply_ai_enhancements(self, scored_segments: List[Dict], ai_analysis: Dict) -> List[Dict]:
        """Apply AI refinements to segment scores."""
        enhanced_segments = scored_segments.copy()
        
        # Defensive check: ensure ai_analysis is a dictionary
        if not isinstance(ai_analysis, dict):
            self.logger.warning(f"ai_analysis is not a dict, got {type(ai_analysis)}: {ai_analysis}")
            return enhanced_segments
        
        refinements = ai_analysis.get('segment_refinements', [])
        
        for refinement in refinements:
            try:
                index = refinement.get('index')
                if index is not None and 0 <= index < len(enhanced_segments):
                    segment = enhanced_segments[index]
                    
                    # Combine original score with AI refinement
                    original_score = segment.get('prompt_match_score', 0.0)
                    ai_score = refinement.get('refined_score', original_score)
                    
                    # Weighted combination (70% AI, 30% heuristic)
                    combined_score = (ai_score * 0.7) + (original_score * 0.3)
                    
                    # Update segment with AI insights
                    segment.update({
                        'prompt_match_score': combined_score,
                        'ai_refined_score': ai_score,
                        'ai_explanation': refinement.get('match_explanation', ''),
                        'ai_priority': refinement.get('priority', 'medium'),
                        'ai_recommendation': refinement.get('recommendation', 'consider')
                    })
                    
            except Exception as e:
                self.logger.warning(f"Error applying AI refinement for index {index}: {e}")
        
        # Re-sort by combined scores
        enhanced_segments.sort(key=lambda x: x['prompt_match_score'], reverse=True)
        
        return enhanced_segments
    
    def _select_best_prompt_matches(self,
                                  enhanced_segments: List[Dict],
                                  prompt_analysis: Dict,
                                  max_segments: int = 10,
                                  variation_manager=None) -> List[Dict]:
        """
        Select the best segments that match the user prompt with optional variation.
        
        Args:
            enhanced_segments: Segments with all scoring applied
            prompt_analysis: Parsed prompt analysis
            max_segments: Maximum number of segments to return
            variation_manager: Optional VariationManager for segment diversity
            
        Returns:
            Final selected segments
        """
        # Apply quantity preferences
        quantity_prefs = prompt_analysis['quantity_preferences']
        min_count = quantity_prefs.get('min_count', 3)
        max_count = min(quantity_prefs.get('max_count', max_segments), max_segments)
        
        # Apply variation to quality threshold if variation manager is provided
        base_threshold = 0.3  # Base minimum prompt match score
        if variation_manager:
            min_score_threshold = variation_manager.apply_quality_threshold_variation(base_threshold)
        else:
            min_score_threshold = base_threshold
            
        qualified_segments = [
            seg for seg in enhanced_segments 
            if seg.get('prompt_match_score', 0) >= min_score_threshold
        ]
        
        # Ensure we have enough segments
        if len(qualified_segments) < min_count:
            # Lower threshold if needed
            min_score_threshold = 0.1
            qualified_segments = [
                seg for seg in enhanced_segments 
                if seg.get('prompt_match_score', 0) >= min_score_threshold
            ]
        
        # Apply variation to segment selection if variation manager is provided
        if variation_manager:
            # Use variation manager for intelligent segment selection
            final_segments = variation_manager.apply_segment_variation(qualified_segments, max_count)
            self.logger.info(f"Applied segment variation: selected {len(final_segments)} segments with diversity")
        else:
            # Use traditional deterministic selection
            final_segments = []
            used_time_ranges = []
            
            for segment in qualified_segments:
                if len(final_segments) >= max_count:
                    break
                
                # Check for temporal overlap with already selected segments
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                overlap = False
                for used_start, used_end in used_time_ranges:
                    if not (end_time <= used_start or start_time >= used_end):
                        overlap = True
                        break
                
                if not overlap:
                    final_segments.append(segment)
                    used_time_ranges.append((start_time, end_time))
            
            # Ensure minimum count if possible
            if len(final_segments) < min_count and len(qualified_segments) >= min_count:
                # Add more segments even with some overlap
                remaining = [seg for seg in qualified_segments if seg not in final_segments]
                remaining.sort(key=lambda x: x['prompt_match_score'], reverse=True)
                
                for segment in remaining:
                    if len(final_segments) >= min_count:
                        break
                    final_segments.append(segment)
            
            # Sort final selection by score
            final_segments.sort(key=lambda x: x['prompt_match_score'], reverse=True)
        
        self.logger.info(f"Selected {len(final_segments)} final segments for prompt '{prompt_analysis['original_prompt']}'")
        
        return final_segments

    async def _intelligent_segment_analysis(self,
                                           user_prompt: str,
                                           candidate_segments: List[Dict],
                                           audio_analysis: Dict,
                                           vision_analysis: Optional[Dict],
                                           scene_analysis: Dict,
                                           object_detection_results: Optional[Dict] = None,
                                           llm_provider: str = "ollama") -> Dict:
        """
        Intelligent segment analysis that lets the LLM understand ANY user prompt
        without rigid keyword classification. The LLM decides what segments match best.
        """
        try:
            self.logger.info(f"🧠 Starting intelligent analysis for prompt: '{user_prompt}'")
            
            # Use configurable LLM provider
            async with ManagedLLMProvider(
                provider_type=llm_provider,
                ollama_client=self.ollama_client
            ) as llm:
                
                # Process segments in batches to manage prompt size
                batch_size = 10
                evaluated_segments = []
                
                for i in range(0, len(candidate_segments), batch_size):
                    batch = candidate_segments[i:i + batch_size]
                    
                    # Create intelligent analysis prompt
                    analysis_prompt = self._create_intelligent_analysis_prompt(
                        user_prompt, batch, audio_analysis
                    )
                    
                    try:
                        response = await llm.generate_response(
                            prompt=analysis_prompt,
                            model=llm.get_best_model("analysis"),
                            cache_key=f"intelligent_analysis_{hash(user_prompt + str(i))}"
                        )
                        
                        evaluation_data = llm._parse_json_response(response)
                        
                        # Process batch results
                        if isinstance(evaluation_data, dict) and 'segments' in evaluation_data:
                            for seg_eval in evaluation_data['segments']:
                                if 'index' in seg_eval and 0 <= seg_eval['index'] < len(batch):
                                    segment = batch[seg_eval['index']].copy()
                                    
                                    # Add intelligent scoring
                                    segment.update({
                                        'intelligent_relevance_score': seg_eval.get('relevance_score', 0.0),
                                        'intelligent_quality_score': seg_eval.get('quality_score', 0.0),
                                        'intelligent_engagement_score': seg_eval.get('engagement_score', 0.0),
                                        'intelligent_overall_score': seg_eval.get('overall_score', 0.0),
                                        'intelligent_reasoning': seg_eval.get('reasoning', ''),
                                        'intelligent_recommended': seg_eval.get('recommended', False),
                                        'analysis_method': 'intelligent_llm'
                                    })
                                    
                                    evaluated_segments.append(segment)
                    
                    except Exception as e:
                        self.logger.warning(f"Batch {i//batch_size + 1} analysis failed: {e}")
                        # Add segments with default scores as fallback
                        for segment in batch:
                            segment_copy = segment.copy()
                            segment_copy.update({
                                'intelligent_overall_score': 0.3,
                                'intelligent_recommended': False,
                                'analysis_method': 'fallback_heuristic'
                            })
                            evaluated_segments.append(segment_copy)
                
                # Select best segments based on intelligent analysis
                final_segments = self._select_intelligent_segments(
                    evaluated_segments, user_prompt, max_segments=10
                )
                
                self.logger.info(f"🎯 Intelligent analysis completed: {len(final_segments)} segments selected")
                
                return {
                    'status': 'success',
                    'user_prompt': user_prompt,
                    'analysis_method': 'intelligent_llm',
                    'total_candidates_analyzed': len(candidate_segments),
                    'intelligent_matched_segments': len(final_segments),
                    'segments': final_segments
                }
                
        except Exception as e:
            self.logger.error(f"Intelligent analysis failed: {e}")
            # Fallback to basic analysis
            return await self._fallback_to_basic_analysis(
                user_prompt, candidate_segments, audio_analysis
            )
    
    def _create_intelligent_analysis_prompt(self,
                                          user_prompt: str,
                                          batch_segments: List[Dict],
                                          audio_analysis: Dict) -> str:
        """
        Create an intelligent analysis prompt that lets the LLM understand
        any user request and select matching segments.
        """
        # Prepare segment data for analysis
        segments_data = []
        for i, segment in enumerate(batch_segments):
            # Extract text for this segment
            start_time = segment['start_time']
            end_time = segment['end_time']
            duration = segment.get('duration', end_time - start_time)
            
            # Get transcription text
            segment_text = self._extract_segment_text(audio_analysis, start_time, end_time)
            
            segments_data.append({
                'index': i,
                'start_time': round(start_time, 1),
                'end_time': round(end_time, 1),
                'duration': round(duration, 1),
                'text': segment_text[:500] if segment_text else '',  # Limit text length
                'word_count': len(segment_text.split()) if segment_text else 0
            })
        
        prompt = f"""You are an expert video content analyst. A user wants to create short videos based on this request:

USER REQUEST: "{user_prompt}"

Your task is to analyze video segments and determine which ones best match what the user is looking for. Be intelligent about understanding the user's intent - they might want:
- Funny/comedy moments
- Dramatic/emotional scenes  
- Action sequences
- Educational content
- Interesting conversations
- Specific topics or themes
- Or anything else!

Don't just match keywords - understand the MEANING and INTENT behind their request.

SEGMENTS TO ANALYZE:
{json.dumps(segments_data, indent=2)}

For each segment, analyze:
1. How well does the content match what the user is asking for?
2. Would this make a good standalone short video?
3. Is the content engaging and complete?
4. Does it fulfill the user's intent?

Respond in JSON format:
{{
    "user_intent_interpretation": "your understanding of what the user wants",
    "segments": [
        {{
            "index": segment_index,
            "relevance_score": 0.0-1.0,
            "quality_score": 0.0-1.0,
            "engagement_score": 0.0-1.0,
            "overall_score": 0.0-1.0,
            "reasoning": "why this segment does/doesn't match the user's request",
            "recommended": true/false
        }}
    ]
}}"""
        
        return prompt
    
    def _select_intelligent_segments(self,
                                   evaluated_segments: List[Dict],
                                   user_prompt: str,
                                   max_segments: int = 10) -> List[Dict]:
        """
        Select the best segments based on intelligent LLM analysis.
        """
        # Sort by overall score
        sorted_segments = sorted(
            evaluated_segments,
            key=lambda x: x.get('intelligent_overall_score', 0),
            reverse=True
        )
        
        # First, try to use recommended segments
        recommended_segments = [
            seg for seg in sorted_segments
            if seg.get('intelligent_recommended', False)
        ]
        
        if recommended_segments:
            self.logger.info(f"✅ Using {len(recommended_segments)} recommended segments")
            candidate_pool = recommended_segments
        else:
            # Fallback: use segments with good scores
            good_segments = [
                seg for seg in sorted_segments
                if seg.get('intelligent_overall_score', 0) >= 0.5
            ]
            
            if good_segments:
                self.logger.info(f"✅ Using {len(good_segments)} high-scoring segments (≥0.5)")
                candidate_pool = good_segments
            else:
                # Last resort: take top scoring segments
                candidate_pool = sorted_segments[:max_segments * 2]
                self.logger.warning(f"⚠️ Using top {len(candidate_pool)} segments by score")
        
        # Apply diversity filtering
        final_segments = []
        used_time_ranges = []
        min_gap = 30.0  # Minimum 30 seconds between segments
        
        for segment in candidate_pool:
            if len(final_segments) >= max_segments:
                break
            
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Check for temporal conflicts
            conflict = False
            for used_start, used_end in used_time_ranges:
                if not (end_time <= used_start or start_time >= used_end):
                    conflict = True
                    break
            
            if not conflict:
                final_segments.append(segment)
                used_time_ranges.append((start_time, end_time))
        
        # Ensure we have at least one segment
        if not final_segments and sorted_segments:
            final_segments = [sorted_segments[0]]
            self.logger.warning("⚠️ Emergency fallback: selected single best segment")
        
        # Sort by start time
        final_segments.sort(key=lambda x: x['start_time'])
        
        self.logger.info(f"📊 Final selection: {len(final_segments)} segments")
        if final_segments:
            scores = [seg.get('intelligent_overall_score', 0) for seg in final_segments]
            self.logger.info(f"📊 Score range: {min(scores):.2f} - {max(scores):.2f}")
        
        return final_segments
    
    async def _fallback_to_basic_analysis(self,
                                        user_prompt: str,
                                        candidate_segments: List[Dict],
                                        audio_analysis: Dict) -> Dict:
        """
        Basic fallback analysis when intelligent analysis fails.
        """
        self.logger.warning("Using basic fallback analysis")
        
        # Simple keyword matching as last resort
        prompt_keywords = user_prompt.lower().split()
        scored_segments = []
        
        for segment in candidate_segments:
            segment_text = self._extract_segment_text(
                audio_analysis, segment['start_time'], segment['end_time']
            ).lower()
            
            # Basic scoring based on keyword presence
            score = 0.0
            for keyword in prompt_keywords:
                if keyword in segment_text:
                    score += 0.2
            
            # Basic quality indicators
            if segment_text:
                word_count = len(segment_text.split())
                if 10 <= word_count <= 100:
                    score += 0.3
                if any(punct in segment_text for punct in ['.', '!', '?']):
                    score += 0.2
            
            segment_copy = segment.copy()
            segment_copy.update({
                'intelligent_overall_score': min(1.0, score),
                'intelligent_recommended': score > 0.4,
                'analysis_method': 'basic_fallback'
            })
            scored_segments.append(segment_copy)
        
        # Take top segments
        scored_segments.sort(key=lambda x: x['intelligent_overall_score'], reverse=True)
        final_segments = scored_segments[:5]  # Take top 5
        
        return {
            'status': 'success',
            'user_prompt': user_prompt,
            'analysis_method': 'basic_fallback',
            'total_candidates_analyzed': len(candidate_segments),
            'intelligent_matched_segments': len(final_segments),
            'segments': final_segments
        }
    
    def _get_dynamic_threshold(self, user_prompt: str, evaluated_segments: List[Dict]) -> float:
        """
        Get dynamic threshold based on user prompt and available segments.
        Different prompts might need different quality thresholds.
        """
        prompt_lower = user_prompt.lower()
        
        # Get base scores
        if evaluated_segments:
            scores = [seg.get('contextual_overall_score', 0) for seg in evaluated_segments]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
        else:
            avg_score = 0.5
            max_score = 0.5
        
        # Default threshold
        base_threshold = 0.6
        
        # Lower threshold for specific content types that might be harder to find
        comedy_keywords = ['funny', 'comedy', 'humor', 'joke', 'laugh', 'amusing']
        emotional_keywords = ['emotional', 'touching', 'heartfelt', 'moving']
        educational_keywords = ['educational', 'learn', 'tutorial', 'explain']
        
        if any(kw in prompt_lower for kw in comedy_keywords):
            base_threshold = 0.4  # Comedy might be subjective
        elif any(kw in prompt_lower for kw in emotional_keywords):
            base_threshold = 0.45  # Emotional content varies
        elif any(kw in prompt_lower for kw in educational_keywords):
            base_threshold = 0.5  # Educational content should be clearer
        
        # Adaptive threshold based on available content quality
        if max_score < 0.7:
            # If no segments score very high, lower the threshold
            adaptive_threshold = max(0.3, avg_score * 0.8)
        else:
            # If we have high-quality segments, maintain standards
            adaptive_threshold = base_threshold
        
        final_threshold = min(base_threshold, adaptive_threshold)
        
        self.logger.info(f"🎯 Dynamic threshold: {final_threshold:.2f} (base: {base_threshold:.2f}, adaptive: {adaptive_threshold:.2f})")
        
        return final_threshold
    
    def _derive_target_keywords(self, user_prompt: str) -> List[str]:
        """
        Derive target keywords from user prompt for content filtering.
        Enhanced to extract semantic keywords and related terms.
        """
        prompt_lower = user_prompt.lower()
        keywords = []
        
        # Extract meaningful words (skip common stop words)
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has',
            'that', 'this', 'from', 'as', 'it', 'its', 'their', 'there', 'about'
        }
        words = [word.strip('.,!?()[]{}"\'-') for word in prompt_lower.split()]
        base_keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        keywords.extend(base_keywords)
        
        # Add semantic expansions for common themes
        keyword_expansions = {
            'christmas': ['holiday', 'festive', 'santa', 'xmas', 'celebration', 'december', 'gift', 'winter'],
            'comedy': ['funny', 'laugh', 'joke', 'humor', 'amusing', 'hilarious', 'witty'],
            'action': ['fast', 'movement', 'dynamic', 'intense', 'exciting', 'chase', 'fight'],
            'emotional': ['touching', 'moving', 'heartfelt', 'tears', 'sad', 'happy', 'love'],
            'educational': ['learn', 'explain', 'tutorial', 'teach', 'lesson', 'understand'],
            'dramatic': ['tension', 'conflict', 'climax', 'intense', 'serious', 'powerful'],
            'romantic': ['love', 'relationship', 'romance', 'couple', 'kiss', 'date'],
            'scary': ['horror', 'fear', 'creepy', 'spooky', 'frightening', 'terror'],
            'celebration': ['party', 'festive', 'happy', 'joy', 'event', 'special'],
            'travel': ['journey', 'trip', 'destination', 'explore', 'adventure', 'visit'],
            'food': ['cooking', 'recipe', 'meal', 'dish', 'eat', 'taste', 'delicious'],
            'sports': ['game', 'play', 'team', 'competition', 'match', 'athlete'],
            'music': ['song', 'sing', 'performance', 'concert', 'band', 'melody'],
            'nature': ['outdoor', 'wildlife', 'landscape', 'scenery', 'environment', 'natural'],
            'technology': ['tech', 'digital', 'computer', 'device', 'innovation', 'gadget']
        }
        
        # Add expansions for found keywords
        for keyword in base_keywords:
            if keyword in keyword_expansions:
                keywords.extend(keyword_expansions[keyword])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        self.logger.info(f"🔑 Derived {len(unique_keywords)} keywords from prompt: {unique_keywords[:10]}...")
        return unique_keywords[:15]  # Limit to top 15 keywords
    
    def _inject_keywords_into_prompt(self, base_prompt: str, user_prompt: str) -> str:
        """
        Inject derived keywords into LLM prompt for better understanding.
        """
        keywords = self._derive_target_keywords(user_prompt)
        if keywords:
            keyword_hint = f"\nKEY TERMS TO FOCUS ON: {', '.join(keywords)}\n"
            return base_prompt + keyword_hint
        return base_prompt
        """
        Enhance prompt analysis with object detection insights.
        
        Args:
            prompt_analysis: Original prompt analysis
            object_detection_results: Object detection results
            
        Returns:
            Enhanced prompt analysis
        """
        try:
            enhanced = prompt_analysis.copy()
            
            # Add object-based context
            if object_detection_results.get('status') == 'success':
                object_summary = object_detection_results.get('summary', {})
                prompt_obj_analysis = object_detection_results.get('prompt_analysis', {})
                
                # Merge object insights 
                enhanced['object_context'] = {
                    'total_objects_detected': object_summary.get('total_objects_detected', 0),
                    'relevant_objects_count': object_summary.get('relevant_objects_count', 0),
                    'detection_quality': object_summary.get('prompt_match_quality', 0.0),
                    'sport_type': prompt_obj_analysis.get('sport_type'),
                    'target_objects': prompt_obj_analysis.get('target_objects', []),
                    'priority_objects': prompt_obj_analysis.get('priority_objects', [])
                }
                
                # Enhance keywords with detected objects
                detected_classes = list(set(
                    obj.get('class_name', '') for obj in object_detection_results.get('detected_objects', [])
                ))
                enhanced['keywords'].extend(detected_classes)
                enhanced['keywords'] = list(set(enhanced['keywords']))  # Remove duplicates
                
                # Boost score if objects match prompt intent
                if object_summary.get('prompt_match_quality', 0) > 0.7:
                    enhanced['confidence_boost'] = 0.3
                
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Failed to enhance prompt analysis with objects: {e}")
            return prompt_analysis

    def set_variation_manager(self, variation_manager):
        """Set the variation manager for output diversity."""
        self.variation_manager = variation_manager
        
    def get_supported_themes(self) -> Dict[str, str]:
        """
        Get list of supported themes and their descriptions.
        Enhanced with object detection capabilities.
        
        Returns:
            Dictionary of theme names and descriptions
        """
        # DEFENSIVE CODING: Check if theme_templates is corrupted
        if not isinstance(self.theme_templates, dict):
            self.logger.error(f"CRITICAL: theme_templates corrupted! Type: {type(self.theme_templates)}, Value: {self.theme_templates}")
            # Attempt to recover by reinitializing theme_templates
            self._reinitialize_theme_templates()
        
        try:
            themes = {
                theme: config['description'] 
                for theme, config in self.theme_templates.items()
            }
        except AttributeError as e:
            self.logger.error(f"CRITICAL: theme_templates.items() failed: {e}")
            # Last resort recovery
            self._reinitialize_theme_templates()
            themes = {
                theme: config['description'] 
                for theme, config in self.theme_templates.items()
            }
        
        # Add object detection specific themes
        themes.update({
            "sports_goals": "Sports goal moments with object detection (basketball hoops, soccer goals, etc.)",
            "sports_highlights": "Key sports moments with player and ball tracking",
            "object_focused": "Content focused on specific objects mentioned in prompt",
            "person_focused": "Content focused on tracking people and characters",
            "action_objects": "Dynamic scenes with moving objects and equipment"
        })
        
        return themes
    
    def _reinitialize_theme_templates(self):
        """
        Emergency recovery method to reinitialize theme_templates if corrupted.
        This should only be called when theme_templates becomes non-dict.
        """
        self.logger.warning("RECOVERY: Reinitializing corrupted theme_templates")
        
        self.theme_templates = {
            'climax': {
                'keywords': ['climax', 'peak', 'highest point', 'crucial moment', 'turning point'],
                'description': 'intense, high-energy, dramatic moments with emotional peaks',
                'emotional_indicators': ['excitement', 'tension', 'drama', 'intensity'],
                'audio_patterns': ['increased volume', 'rapid speech', 'emphasis', 'passion'],
                'visual_patterns': ['action', 'close-ups', 'dramatic lighting', 'movement']
            },
            'comedy': {
                'keywords': ['funny', 'humor', 'joke', 'laugh', 'amusing', 'hilarious'],
                'description': 'humorous, entertaining, light-hearted moments',
                'emotional_indicators': ['joy', 'amusement', 'laughter', 'playfulness'],
                'audio_patterns': ['laughter', 'funny voices', 'comedic timing', 'punchlines'],
                'visual_patterns': ['expressions', 'gestures', 'reactions', 'visual gags']
            },
            'emotional': {
                'keywords': ['emotional', 'touching', 'heartfelt', 'moving', 'inspiring'],
                'description': 'emotionally engaging, touching, inspiring moments',
                'emotional_indicators': ['sadness', 'joy', 'inspiration', 'empathy', 'connection'],
                'audio_patterns': ['soft speech', 'emotional tone', 'pauses', 'sincerity'],
                'visual_patterns': ['close-ups', 'expressions', 'intimate moments', 'reactions']
            },
            'educational': {
                'keywords': ['learn', 'explain', 'tutorial', 'how to', 'educational'],
                'description': 'informative, instructional, educational content',
                'emotional_indicators': ['clarity', 'understanding', 'curiosity', 'enlightenment'],
                'audio_patterns': ['clear speech', 'explanatory tone', 'structured delivery'],
                'visual_patterns': ['diagrams', 'demonstrations', 'step-by-step visuals']
            },
            'action': {
                'keywords': ['action', 'fast', 'dynamic', 'movement', 'energy', 'exciting'],
                'description': 'high-energy, dynamic, action-packed moments',
                'emotional_indicators': ['excitement', 'adrenaline', 'intensity', 'energy'],
                'audio_patterns': ['fast speech', 'dynamic audio', 'sound effects', 'music'],
                'visual_patterns': ['rapid cuts', 'movement', 'dynamic angles', 'effects']
            },
            'motivational': {
                'keywords': ['motivational', 'inspiring', 'uplifting', 'positive', 'encourage'],
                'description': 'inspiring, uplifting, motivational content',
                'emotional_indicators': ['inspiration', 'hope', 'determination', 'positivity'],
                'audio_patterns': ['passionate delivery', 'crescendo', 'emphasis', 'conviction'],
                'visual_patterns': ['uplifting imagery', 'positive expressions', 'success visuals']
            },
            'general': {
                'keywords': ['engagement', 'interesting', 'captivating', 'attention'],
                'description': 'generally engaging and interesting content',
                'emotional_indicators': ['interest', 'curiosity', 'engagement', 'attention'],
                'audio_patterns': ['varied tone', 'clear delivery', 'engaging pace'],
                'visual_patterns': ['varied shots', 'good composition', 'visual interest']
            }
        }
        
        self.logger.warning("RECOVERY: theme_templates successfully reinitialized")
    
    async def _contextual_analysis_approach(self,
                                          user_prompt: str,
                                          candidate_segments: List[Dict],
                                          audio_analysis: Dict,
                                          vision_analysis: Optional[Dict],
                                          scene_analysis: Dict,
                                          content_overview: Dict,
                                          intent_analysis: Dict,
                                          object_detection_results: Optional[Dict] = None,
                                          llm_provider: str = "ollama") -> Dict:
        """
        PHASE 1 ENHANCEMENT: Contextual analysis approach using content overview and intent understanding.
        
        This method leverages the enhanced contextual understanding to provide
        more accurate and relevant segment selection.
        """
        try:
            self.logger.info("Performing contextual analysis with LLM-enhanced understanding")
            
            # Extract key information from context
            content_type = content_overview.get('content_type', 'unknown')
            user_intent = intent_analysis.get('intent_interpretation', {})
            selection_criteria = intent_analysis.get('selection_criteria', {})
            
            # Use LLM for context-aware segment evaluation
            should_use_openai = self._should_use_openai_for_provider(llm_provider)
            
            evaluated_segments = []
            evaluation_method = "unknown"
            
            if should_use_openai or self.ollama_client:
                try:
                    evaluated_segments = await self._llm_evaluate_segments_with_context(
                        candidate_segments=candidate_segments,
                        user_prompt=user_prompt,
                        content_overview=content_overview,
                        intent_analysis=intent_analysis,
                        audio_analysis=audio_analysis,
                        vision_analysis=vision_analysis,
                        llm_provider=llm_provider
                    )
                    evaluation_method = "llm_contextual"
                    self.logger.info(f"✅ LLM evaluation completed: {len(evaluated_segments)} segments")
                except Exception as e:
                    self.logger.error(f"LLM evaluation failed: {e}")
                    evaluated_segments = []
            
            # INTELLIGENT FALLBACK: If LLM evaluation failed or returned no results
            if not evaluated_segments:
                self.logger.warning("⚠️ LLM evaluation failed - applying enhanced heuristic fallback")
                evaluated_segments = self._enhanced_heuristic_evaluation(
                    candidate_segments, user_prompt, content_overview, intent_analysis, audio_analysis
                )
                evaluation_method = "heuristic_fallback"
            
            # Apply selection criteria from intent analysis
            quality_thresholds = selection_criteria.get('quality_thresholds', {})
            min_relevance = quality_thresholds.get('minimum_relevance', 0.3)
            min_engagement = quality_thresholds.get('minimum_engagement', 0.3)
            
            # DEFENSIVE: Verify theme_templates integrity after LLM operations
            if not isinstance(self.theme_templates, dict):
                self.logger.error(f"CRITICAL: theme_templates corrupted after LLM evaluation! Type: {type(self.theme_templates)}")
                self._reinitialize_theme_templates()
            
            # Filter segments based on contextual quality thresholds
            qualified_segments = [
                seg for seg in evaluated_segments
                if (seg.get('contextual_relevance_score', 0) >= min_relevance and
                    seg.get('contextual_engagement_score', 0) >= min_engagement)
            ]
            
            # CRITICAL FALLBACK: If no segments meet quality thresholds, lower them
            if not qualified_segments and evaluated_segments:
                self.logger.warning(f"⚠️ No segments met quality thresholds (relevance≥{min_relevance}, engagement≥{min_engagement})")
                self.logger.info("🔄 Lowering quality thresholds to ensure segment selection")
                
                # Progressive threshold reduction
                fallback_relevance = max(0.1, min_relevance - 0.2)
                fallback_engagement = max(0.1, min_engagement - 0.2)
                
                qualified_segments = [
                    seg for seg in evaluated_segments
                    if (seg.get('contextual_relevance_score', 0) >= fallback_relevance and
                        seg.get('contextual_engagement_score', 0) >= fallback_engagement)
                ]
                
                # Last resort: take top 50% of segments by score
                if not qualified_segments:
                    sorted_by_score = sorted(evaluated_segments, 
                                           key=lambda x: x.get('contextual_overall_score', 0), 
                                           reverse=True)
                    qualified_segments = sorted_by_score[:len(sorted_by_score)//2] or sorted_by_score[:3]
                    self.logger.info(f"🔄 Emergency fallback: Selected top {len(qualified_segments)} segments by score")
            
            # Apply final selection with enhanced diversity
            final_segments = self._select_best_contextual_matches(
                qualified_segments,
                intent_analysis,
                max_segments=10
            )
            
            # FINAL SAFETY CHECK: Ensure we have at least one segment
            if not final_segments and candidate_segments:
                # Emergency: select the best single segment
                best_segment = max(candidate_segments, 
                                 key=lambda x: x.get('contextual_overall_score', 
                                                   x.get('prompt_match_score', 0)))
                final_segments = [best_segment]
                self.logger.warning("🚨 Emergency fallback: Selected single best segment to prevent empty result")
                evaluation_method += "_emergency"
            
            return {
                'status': 'success',
                'user_prompt': user_prompt,
                'analysis_method': 'enhanced_contextual',
                'content_type': content_type,
                'user_intent_interpretation': user_intent.get('contextual_intent', ''),
                'intent_confidence': intent_analysis.get('confidence_assessment', {}).get('overall_confidence', 0.5),
                'total_candidates_analyzed': len(candidate_segments),
                'qualified_segments': len(qualified_segments),
                'prompt_matched_segments': len(final_segments),
                'segments': final_segments,
                'content_overview': content_overview,
                'intent_analysis': intent_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Contextual analysis failed: {e}")
            # Fallback to original approach
            return await self._fallback_to_original_analysis(
                user_prompt, candidate_segments, audio_analysis, vision_analysis, scene_analysis
            )
    
    async def _llm_evaluate_segments_with_context(self,
                                                candidate_segments: List[Dict],
                                                user_prompt: str,
                                                content_overview: Dict,
                                                intent_analysis: Dict,
                                                audio_analysis: Dict,
                                                vision_analysis: Optional[Dict],
                                                llm_provider: str = "ollama") -> List[Dict]:
        """
        Use LLM to evaluate segments with full contextual understanding.
        """
        try:
            # Use configurable LLM provider
            async with ManagedLLMProvider(
                provider_type=llm_provider,
                ollama_client=self.ollama_client
            ) as llm:
                
                # Narrow candidate pool using derived keywords to reduce prompt size & improve precision
                derived_kws = self._derive_target_keywords(user_prompt)
                filtered_segments = candidate_segments
                if derived_kws:
                    lowered_kws = [k.lower() for k in derived_kws]
                    tmp = []
                    for seg in candidate_segments:
                        text = (seg.get('segment_text') or '').lower()
                        if any(k in text for k in lowered_kws):
                            tmp.append(seg)
                    # Use filtered set if it’s non-trivial; otherwise keep originals
                    if len(tmp) >= max(5, int(0.2 * len(candidate_segments))):
                        self.logger.info(f"🔎 Keyword filter applied: {len(tmp)}/{len(candidate_segments)} segments match {derived_kws[:6]}...")
                        filtered_segments = tmp
                    else:
                        self.logger.info("🔎 Keyword filter found too few matches; keeping full candidate set")
                
                # Tighter, adaptive batching to keep prompts small
                base_batch_size = 8
                min_batch_size = 3
                evaluated_segments = []
                
                # Helper to build compact segment payloads
                def build_segment_data(batch: List[Dict], total_duration: float, text_char_limit: int = 280) -> List[Dict]:
                    data = []
                    for j, segment in enumerate(batch):
                        text = segment.get('segment_text', '') or ''
                        # Trim per-segment text to limit tokens
                        trimmed = (text[:text_char_limit] + '…') if len(text) > text_char_limit else text
                        data.append({
                            'index': j,
                            'start_time': round(segment['start_time'], 2),
                            'end_time': round(segment['end_time'], 2),
                            'duration': round(segment.get('duration', segment['end_time'] - segment['start_time']), 2),
                            'text_excerpt': trimmed,
                            'pos': round(segment['start_time'] / total_duration, 4) if total_duration > 0 else 0
                        })
                    return data
                
                total_duration = float(content_overview.get('total_duration', 0) or 0)
                batch_size = base_batch_size
                
                i = 0
                while i < len(filtered_segments):
                    batch = filtered_segments[i:i + batch_size]
                    segment_data = build_segment_data(batch, total_duration)
                    # Compact JSON (no pretty-print) to reduce tokens
                    segment_json = json.dumps(segment_data, separators=(',', ':'))
                    
                    base_instruction = (
                        "🎯 CRITICAL: Your PRIMARY task is to find segments that match the user's request. Prioritize user intent above all else.\n\n"
                        f"USER REQUEST: \"{user_prompt}\"\n"
                        f"USER'S TRUE INTENT: {intent_analysis.get('intent_interpretation', {}).get('contextual_intent', '')}\n\n"
                        "⚠️ STRICT MATCHING REQUIREMENTS:\n"
                        "1. Segments MUST directly relate to the user's request\n"
                        "2. If the user asks for 'Christmas', ONLY select segments mentioning Christmas/holidays/festive content\n"
                        "3. If the user asks for 'comedy', ONLY select funny/humorous segments\n"
                        "4. Generic or unrelated content should score LOW even if technically good quality\n"
                        "5. Keyword matching is MANDATORY - segments without relevant keywords should be rejected\n\n"
                        f"VIDEO CONTEXT (secondary consideration):\n"
                        f"- Content Type: {content_overview.get('content_type', 'unknown')}\n"
                        f"- Characteristics: {json.dumps(content_overview.get('content_characteristics', {}), separators=(',', ':'))}\n"
                        f"- Selection Criteria: {json.dumps(intent_analysis.get('selection_criteria', {}), separators=(',', ':'))}\n\n"
                    )
                    base_instruction = self._inject_keywords_into_prompt(base_instruction, user_prompt)
                    evaluation_prompt = (
                        base_instruction +
                        f"SEGMENTS:{segment_json}\n\n"
                        "SCORING RULES:\n"
                        "- contextual_relevance_score: How well does the content match the user's specific request? (0.0-1.0)\n"
                        "- intent_alignment_score: Does this fulfill the user's intent? (0.0-1.0)\n"
                        "- overall_score: Average of relevance and intent (must be ≥0.7 for recommendation)\n"
                        "- recommended: true ONLY if overall_score ≥0.7 AND segment directly matches user request\n\n"
                        "Return strict JSON with key 'evaluations' only, and for each item include: segment_index, contextual_relevance_score, content_quality_score, standalone_viability_score, engagement_potential_score, intent_alignment_score, overall_score, reasoning, recommended."
                    )
                    
                    prompt_length = len(evaluation_prompt)
                    self.logger.info(f"📏 Evaluation prompt length: {prompt_length} characters (batch={len(batch)})")
                    
                    # If prompt is too long, shrink batch or text window
                    if prompt_length > 8000 and batch_size > min_batch_size:
                        batch_size = max(min_batch_size, batch_size - 2)
                        self.logger.warning(f"⚠️ Prompt too long ({prompt_length}). Reducing batch size to {batch_size} and retrying this window.")
                        continue
                    if prompt_length > 10000:
                        # As last resort trim text further
                        segment_data = build_segment_data(batch, total_duration, text_char_limit=180)
                        segment_json = json.dumps(segment_data, separators=(',', ':'))
                        base_instruction = self._inject_keywords_into_prompt((
                            "🎯 CRITICAL: Find segments matching the user's request. Prioritize user intent.\n\n"
                            f"USER REQUEST: \"{user_prompt}\"\n\n"
                            "RULES: Segments MUST directly relate to user request. Score low if irrelevant.\n"
                        ), user_prompt)
                        evaluation_prompt = (
                            base_instruction +
                            f"SEGMENTS:{segment_json}\n\n"
                            "Return strict JSON: 'evaluations' with segment_index, contextual_relevance_score, intent_alignment_score, overall_score, reasoning, recommended (true only if overall≥0.7 AND matches request)."
                        )
                        self.logger.warning(f"⚠️ Aggressively trimmed segment text excerpts due to oversized prompt.")
                    
                    try:
                        response = await llm.generate_response(
                            prompt=evaluation_prompt,
                            model=llm.get_best_model("analysis"),
                            cache_key=f"contextual_segment_eval_{hash(segment_json + user_prompt)}"
                        )
                    except Exception as e:
                        # If rate-limited or error, try smaller batch next
                        if 'rate limit' in str(e).lower() or '429' in str(e):
                            if batch_size > min_batch_size:
                                batch_size = max(min_batch_size, batch_size - 2)
                                self.logger.warning(f"⚠️ Rate limited. Reducing batch size to {batch_size} and retrying.")
                                continue
                        # On other errors, log and skip this batch
                        self.logger.error(f"LLM request failed for batch starting at {i}: {e}")
                        i += batch_size
                        continue
                    
                    evaluation_data = llm._parse_json_response(response)
                    
                    # Defensive: ensure dict
                    if not isinstance(evaluation_data, dict):
                        self.logger.warning(f"Parsed evaluation was not a dict, got {type(evaluation_data)}. Skipping batch.")
                        i += batch_size
                        continue
                    
                    evaluations = evaluation_data.get('evaluations', [])
                    if not isinstance(evaluations, list):
                        self.logger.warning("No 'evaluations' list found in response. Skipping batch.")
                        i += batch_size
                        continue
                    
                    batch_evaluated_count = 0
                    for evaluation in evaluations:
                        segment_index = evaluation.get('segment_index')
                        if isinstance(segment_index, int) and 0 <= segment_index < len(batch):
                            enhanced_segment = batch[segment_index].copy()
                            enhanced_segment.update({
                                'contextual_relevance_score': evaluation.get('contextual_relevance_score', 0.5),
                                'contextual_quality_score': evaluation.get('content_quality_score', 0.5),
                                'contextual_standalone_score': evaluation.get('standalone_viability_score', 0.5),
                                'contextual_engagement_score': evaluation.get('engagement_potential_score', 0.5),
                                'contextual_intent_alignment': evaluation.get('intent_alignment_score', 0.5),
                                'contextual_overall_score': evaluation.get('overall_score', 0.5),
                                'contextual_reasoning': evaluation.get('reasoning', ''),
                                'contextual_recommended': evaluation.get('recommended', False)
                            })
                            evaluated_segments.append(enhanced_segment)
                            batch_evaluated_count += 1
                        else:
                            self.logger.warning(f"⚠️ Invalid segment_index {segment_index} for batch size {len(batch)}")
                    
                    self.logger.info(f"✅ Batch completed: {batch_evaluated_count}/{len(batch)} segments evaluated")
                    i += batch_size
                
                self.logger.info(f"LLM evaluated {len(evaluated_segments)} segments with contextual understanding using {llm_provider}")
                
                # FINAL DEBUG SUMMARY
                self.logger.info(f"🎯 FINAL LLM EVALUATION SUMMARY:")
                self.logger.info(f"📊 Input segments: {len(candidate_segments)}")
                self.logger.info(f"📊 Evaluated segments: {len(evaluated_segments)}")
                self.logger.info(f"📊 Success rate: {len(evaluated_segments)/len(candidate_segments)*100:.1f}%" if candidate_segments else "N/A")
                
                if evaluated_segments:
                    scores = [seg.get('contextual_overall_score', 0) for seg in evaluated_segments]
                    recommended_count = sum(1 for seg in evaluated_segments if seg.get('contextual_recommended', False))
                    
                    # Also check segments that meet dynamic threshold criteria
                    dynamic_threshold = self._get_dynamic_threshold(user_prompt, evaluated_segments)
                    threshold_qualified_count = sum(1 for seg in evaluated_segments if seg.get('contextual_overall_score', 0) >= dynamic_threshold)
                    
                    self.logger.info(f"📊 Score range: {min(scores):.2f} - {max(scores):.2f}")
                    self.logger.info(f"📊 Average score: {sum(scores)/len(scores):.2f}")
                    self.logger.info(f"📊 Recommended segments: {recommended_count}/{len(evaluated_segments)}")
                    self.logger.info(f"📊 Dynamic threshold qualified: {threshold_qualified_count}/{len(evaluated_segments)} (threshold={dynamic_threshold:.2f})")
                    
                    # ENHANCED FIX: Dynamic threshold based on content type and keyword matches
                    if recommended_count == 0 and scores:
                        # Check for specific keyword matches to lower threshold
                        keyword_match_threshold = self._get_dynamic_threshold(user_prompt, evaluated_segments)
                        
                        # First try with dynamic threshold (could be lower for specific keywords)
                        high_scoring_segments = [seg for seg in evaluated_segments if seg.get('contextual_overall_score', 0) >= keyword_match_threshold]
                        if high_scoring_segments:
                            # Force recommendation for segments meeting dynamic threshold
                            for seg in high_scoring_segments:
                                seg['contextual_recommended'] = True
                            self.logger.warning(f"🔄 KEYWORD FALLBACK: Force-recommended {len(high_scoring_segments)} segments (score ≥ {keyword_match_threshold:.2f})")
                        else:
                            # Fallback: try standard 0.6 threshold
                            high_scoring_segments = [seg for seg in evaluated_segments if seg.get('contextual_overall_score', 0) >= 0.6]
                            if high_scoring_segments:
                                for seg in high_scoring_segments:
                                    seg['contextual_recommended'] = True
                                self.logger.warning(f"🔄 FALLBACK: Force-recommended {len(high_scoring_segments)} high-scoring segments (score ≥ 0.6)")
                            else:
                                # Final fallback: recommend top 3 segments regardless of score
                                sorted_segments = sorted(evaluated_segments, key=lambda x: x.get('contextual_overall_score', 0), reverse=True)
                                top_segments = sorted_segments[:3]
                                for seg in top_segments:
                                    seg['contextual_recommended'] = True
                                self.logger.warning(f"🔄 EMERGENCY FALLBACK: Force-recommended top {len(top_segments)} segments to prevent empty results")
                
                return evaluated_segments
            
        except Exception as e:
            self.logger.error(f"LLM contextual evaluation failed: {e}")
            return candidate_segments  # Return original segments as fallback
    
    def _enhanced_heuristic_evaluation(self,
                                     candidate_segments: List[Dict],
                                     user_prompt: str,
                                     content_overview: Dict,
                                     intent_analysis: Dict,
                                     audio_analysis: Dict) -> List[Dict]:
        """
        Enhanced heuristic evaluation when LLM is not available.
        """
        enhanced_segments = []
        
        for segment in candidate_segments:
            # Apply contextual scoring based on content overview and intent
            contextual_score = 0.5  # Base score
            
            # Content type alignment
            content_type = content_overview.get('content_type', 'unknown')
            if content_type == 'tutorial' and 'educational' in user_prompt.lower():
                contextual_score += 0.2
            elif content_type in ['movie', 'documentary'] and any(word in user_prompt.lower() for word in ['climax', 'dramatic', 'best']):
                contextual_score += 0.2
            
            # Position-based scoring
            position_preference = intent_analysis.get('selection_criteria', {}).get('content_position_preference', 'any')
            total_duration = content_overview.get('total_duration', 1)
            position_ratio = segment['start_time'] / total_duration if total_duration > 0 else 0.5
            
            if position_preference == 'peak' and 0.3 <= position_ratio <= 0.8:
                contextual_score += 0.1
            elif position_preference == 'end' and position_ratio > 0.7:
                contextual_score += 0.1
            elif position_preference == 'beginning' and position_ratio < 0.3:
                contextual_score += 0.1
            
            # Text quality assessment
            text_content = segment.get('segment_text', '')
            if text_content:
                # Basic quality indicators
                word_count = len(text_content.split())
                has_complete_sentences = any(punct in text_content for punct in ['.', '!', '?'])
                
                if 10 <= word_count <= 100 and has_complete_sentences:
                    contextual_score += 0.15
            
            enhanced_segment = segment.copy()
            enhanced_segment.update({
                'contextual_relevance_score': min(1.0, contextual_score),
                'contextual_engagement_score': min(1.0, contextual_score * 0.9),
                'contextual_overall_score': min(1.0, contextual_score),
                'contextual_reasoning': 'Heuristic-based contextual evaluation',
                'contextual_recommended': contextual_score > 0.6
            })
            
            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _select_best_contextual_matches(self,
                                      qualified_segments: List[Dict],
                                      intent_analysis: Dict,
                                      max_segments: int = 10) -> List[Dict]:
        """
        Select the best contextual matches with intelligent fallback logic.
        
        CRITICAL FIX: Always ensure we select the most engaging segments,
        even if LLM doesn't explicitly "recommend" any segments.
        """
        # Sort by contextual overall score first
        sorted_segments = sorted(
            qualified_segments,
            key=lambda x: x.get('contextual_overall_score', 0),
            reverse=True
        )
        
        # INTELLIGENT FALLBACK LOGIC: Check if we have recommended segments
        recommended_segments = [
            seg for seg in sorted_segments 
            if seg.get('contextual_recommended', False)
        ]
        
        # If no segments are explicitly recommended, use intelligent fallback
        if not recommended_segments:
            self.logger.warning("⚠️ No segments explicitly recommended by LLM - applying intelligent fallback")
            
            # Fallback Strategy 1: Use highest scoring segments
            high_scoring_segments = [
                seg for seg in sorted_segments 
                if seg.get('contextual_overall_score', 0) >= 0.6
            ]
            
            if high_scoring_segments:
                self.logger.info(f"✅ Fallback 1: Found {len(high_scoring_segments)} high-scoring segments (≥0.6)")
                candidate_pool = high_scoring_segments
            else:
                # Fallback Strategy 2: Use top scoring segments regardless of threshold
                top_segment_count = min(max_segments * 2, len(sorted_segments))
                candidate_pool = sorted_segments[:top_segment_count]
                self.logger.info(f"✅ Fallback 2: Using top {len(candidate_pool)} scoring segments")
        else:
            self.logger.info(f"✅ Using {len(recommended_segments)} explicitly recommended segments")
            # NEW: If we have fewer recommended segments than the target, supplement with next best scoring segments
            if len(recommended_segments) < max_segments:
                supplemental_needed = max_segments - len(recommended_segments)
                supplemental = []
                for seg in sorted_segments:
                    if seg in recommended_segments:
                        continue
                    supplemental.append(seg)
                    if len(supplemental) >= supplemental_needed * 2:  # over-fill for diversity filter
                        break
                if supplemental:
                    self.logger.info(
                        f"🔄 Supplementing {len(recommended_segments)} recommended segment(s) with {len(supplemental)} high-scoring non-recommended segment(s) to aim for {max_segments} total"
                    )
                candidate_pool = recommended_segments + supplemental
            else:
                candidate_pool = recommended_segments
        
        # Apply diversity filtering to selected candidates
        final_segments = []
        used_time_ranges = []
        min_gap = 30.0  # Minimum 30 seconds between segments
        
        for segment in candidate_pool:
            if len(final_segments) >= max_segments:
                break
            
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Check for temporal conflicts
            conflict = False
            for used_start, used_end in used_time_ranges:
                gap_start = min(abs(start_time - used_end), abs(end_time - used_start))
                if gap_start < min_gap:
                    if not (end_time <= used_start or start_time >= used_end):
                        conflict = True
                        break
            
            if not conflict:
                final_segments.append(segment)
                used_time_ranges.append((start_time, end_time))
        
        # ADDITIONAL FALLBACK: If still no segments, force select the best one
        if not final_segments and sorted_segments:
            self.logger.warning("⚠️ No segments passed diversity filter - force selecting best segment")
            final_segments = [sorted_segments[0]]

        # SECONDARY SUPPLEMENTATION: If after diversity we still have fewer than desired and we started with some recommendations,
        # attempt to fill remaining slots with additional non-conflicting segments from the global sorted list.
        if 0 < len(final_segments) < max_segments:
            needed = max_segments - len(final_segments)
            added = 0
            for seg in sorted_segments:
                if seg in final_segments:
                    continue
                # Temporal diversity check (reuse earlier logic)
                start_time = seg['start_time']
                end_time = seg['end_time']
                conflict = False
                for used_start, used_end in used_time_ranges:
                    gap_start = min(abs(start_time - used_end), abs(end_time - used_start))
                    if gap_start < min_gap and not (end_time <= used_start or start_time >= used_end):
                        conflict = True
                        break
                if conflict:
                    continue
                final_segments.append(seg)
                used_time_ranges.append((start_time, end_time))
                added += 1
                if added >= needed:
                    break
            if added > 0:
                self.logger.info(f"🔧 Diversity supplementation added {added} additional segment(s) to reach {len(final_segments)}/{max_segments}")
        
        # Sort final segments by start time
        final_segments.sort(key=lambda x: x['start_time'])
        
        # Log selection strategy used
        strategy_used = "recommended" if recommended_segments else "intelligent_fallback"
        self.logger.info(f"📊 Selection strategy: {strategy_used}, Final segments: {len(final_segments)}")
        
        return final_segments

    # ========= Refresh Rate Spec Utilities =========
    def _detect_refresh_specs(self, text: str) -> set:
        """Detect refresh rate specs like '120 hz', '144Hz', returning normalized tokens (e.g., '120hz')."""
        if not text:
            return set()
        import re
        lower = text.lower()
        specs = set()
        pattern = r'\b(60|75|90|100|120|144|150|165|170|180|200|240|300|360)\s*-?\s*(hz|hertz)\b'
        for m in re.finditer(pattern, lower):
            specs.add(f"{m.group(1)}hz")
        generic_indicators = [
            'high refresh', 'higher refresh', 'fast refresh', 'smooth display',
            '120 fps', 'high fps', 'buttery smooth'
        ]
        if any(g in lower for g in generic_indicators):
            specs.add('high_refresh')
        return specs

    def _boost_contextual_segments_for_specs(self, evaluated_segments: List[Dict], user_prompt: str) -> int:
        """After contextual/LLM evaluation, ensure refresh spec segments are recommended."""
        spec_targets = self._detect_refresh_specs(user_prompt)
        if not spec_targets:
            return 0
        boosted = 0
        for seg in evaluated_segments:
            if seg.get('contextual_recommended'):
                continue
            seg_text = seg.get('segment_text', '').lower()
            seg_specs = self._detect_refresh_specs(seg_text)
            if seg_specs & spec_targets and seg.get('contextual_overall_score', 0) >= 0.3:
                seg['contextual_recommended'] = True
                seg['contextual_reasoning'] = (seg.get('contextual_reasoning', '') + ' | spec_refresh_rate_match_boost').strip()
                seg['spec_matches'] = list(seg_specs & spec_targets)
                boosted += 1
        if boosted:
            self.logger.info(f"🔧 Spec Boost: Marked {boosted} segment(s) due to refresh spec match ({', '.join(spec_targets)})")
        else:
            self.logger.info(f"🔧 Spec Boost: No segments matched refresh specs ({', '.join(spec_targets)})")
        return boosted

    # ========= General Focus Token Utilities =========
    def _extract_focus_tokens(self, prompt: str) -> set:
        """Extract focus tokens from user prompt (numbers+units, specs, component keywords, brand/model names)."""
        if not prompt:
            return set()
        import re
        lower = prompt.lower()
        tokens = set()
        # Reuse refresh specs
        tokens |= self._detect_refresh_specs(lower)
        # Number+unit patterns (mAh, gb, g, w, nm, fps, nit(s), inch, mm, ghz, k, kwh)
        num_unit_pattern = r'\b(\d+[\d\.,]*)(\s?)(mah|gb|g|w|nm|fps|nit|nits|inch|in|mm|ghz|khz|mhz|k|kwh|%|mp)\b'
        for m in re.finditer(num_unit_pattern, lower):
            val = m.group(1).replace(',', '')
            unit = m.group(3)
            tokens.add(f"{val}{unit}")
        # Resolution patterns 4k / 8k / 2k / 1080p / 1440p / 2160p
        res_pattern = r'\b(4k|8k|2k|1080p|1440p|2160p|720p)\b'
        for m in re.finditer(res_pattern, lower):
            tokens.add(m.group(1))
        # Component keywords
        component_keywords = [
            'battery','display','screen','panel','camera','sensor','lens','chip','chipset','processor','cpu','gpu',
            'ram','storage','charge','charging','fast charge','wireless charging','brightness','nits','refresh','frame rate',
            'speaker','audio','mic','microphone','stabilization','ois','eis','dynamic range','hdr','amoled','oled','lcd','mini led'
        ]
        for kw in component_keywords:
            if kw in lower:
                tokens.add(kw)
        # Brand / model patterns (simplistic extraction of capitalized alphanumerics with digits from original prompt, keep case-insensitive)
        model_pattern = r'\b([a-zA-Z]+\d+[a-zA-Z\d]*)\b'
        for m in re.finditer(model_pattern, prompt):
            tokens.add(m.group(1).lower())
        # Filter trivial tokens
        stop = {'the','and','with','that','this','what','when','your','you','for','from','video','clip','short'}
        tokens = {t for t in tokens if len(t) >= 3 and t not in stop}
        return tokens

    def _match_focus_tokens(self, text: str, focus_tokens: set) -> set:
        if not text or not focus_tokens:
            return set()
        lower = text.lower()
        matched = set()
        for tok in focus_tokens:
            if tok in lower:
                matched.add(tok)
        return matched

    def _is_numeric_unit_token(self, token: str) -> bool:
        import re
        return bool(re.match(r'^\d+[\d\.,]*(mah|gb|g|w|nm|fps|nit|nits|inch|in|mm|ghz|khz|mhz|k|kwh|%|mp)$', token))

    def _boost_contextual_segments_for_focus_tokens(self, evaluated_segments: List[Dict], user_prompt: str) -> int:
        focus_tokens = self._extract_focus_tokens(user_prompt)
        if not focus_tokens:
            return 0
        boosted = 0
        for seg in evaluated_segments:
            if seg.get('contextual_recommended'):
                continue
            seg_text = seg.get('segment_text', '').lower()
            matched = self._match_focus_tokens(seg_text, focus_tokens)
            if matched and seg.get('contextual_overall_score', 0) >= 0.3:
                seg['contextual_recommended'] = True
                seg['contextual_reasoning'] = (seg.get('contextual_reasoning', '') + ' | focus_token_match_boost').strip()
                seg['focus_token_matches'] = list(matched)
                boosted += 1
        if boosted:
            self.logger.info(f"🔧 Focus Token Boost: Marked {boosted} segment(s) due to focus token matches")
        else:
            self.logger.info("🔧 Focus Token Boost: No segments matched focus tokens")
        return boosted
    
    async def _fallback_to_original_analysis(self,
                                           user_prompt: str,
                                           candidate_segments: List[Dict],
                                           audio_analysis: Dict,
                                           vision_analysis: Optional[Dict],
                                           scene_analysis: Dict) -> Dict:
        """
        Fallback to original analysis approach when contextual analysis fails.
        """
        self.logger.warning("Falling back to original prompt analysis approach")
        
        # Use original prompt analysis workflow
        prompt_analysis = self._analyze_user_prompt(user_prompt)
        
        scored_segments = await self._score_segments_for_prompt(
            candidate_segments,
            prompt_analysis,
            audio_analysis,
            vision_analysis,
            scene_analysis
        )
        
        final_segments = self._select_best_prompt_matches(
            scored_segments,
            prompt_analysis,
            max_segments=10
        )
        
        return {
            'status': 'success',
            'user_prompt': user_prompt,
            'analysis_method': 'fallback_original',
            'prompt_analysis': prompt_analysis,
            'total_candidates_analyzed': len(candidate_segments),
            'prompt_matched_segments': len(final_segments),
            'segments': final_segments
        }
    
    def _enhanced_heuristic_analysis(self,
                                   user_prompt: str,
                                   candidate_segments: List[Dict],
                                   audio_analysis: Dict,
                                   vision_analysis: Optional[Dict],
                                   scene_analysis: Dict) -> Dict:
        """
        Enhanced heuristic analysis for Phase 2 compatibility without LLM.
        
        This provides improved heuristic analysis when LLM is not available
        but still maintains Phase 2 compatibility and enhanced scoring.
        """
        try:
            self.logger.info("Using enhanced heuristic analysis for Phase 2 compatibility")
            
            # Analyze user prompt for intent keywords
            prompt_keywords = self._extract_intent_keywords(user_prompt)
            content_type = self._infer_content_type_heuristic(user_prompt, audio_analysis)
            
            # Score segments with enhanced heuristics
            scored_segments = []
            for segment in candidate_segments:
                # Enhanced scoring factors
                heuristic_score = self._calculate_enhanced_heuristic_score(
                    segment, prompt_keywords, content_type, audio_analysis, scene_analysis
                )
                
                # Add Phase 2 compatible metadata
                enhanced_segment = segment.copy()
                enhanced_segment.update({
                    'heuristic_score': heuristic_score,
                    'prompt_keywords_matched': self._count_keyword_matches(segment, prompt_keywords),
                    'content_type_alignment': self._score_content_type_alignment_heuristic(segment, content_type),
                    'phase2_compatible': True,
                    'analysis_method': 'enhanced_heuristic'
                })
                
                scored_segments.append(enhanced_segment)
            
            # Filter and sort segments
            qualifying_segments = [seg for seg in scored_segments if seg['heuristic_score'] >= 0.5]
            qualifying_segments.sort(key=lambda x: x['heuristic_score'], reverse=True)
            
            self.logger.info(f"Enhanced heuristic analysis: {len(qualifying_segments)}/{len(candidate_segments)} segments qualified")
            
            return {
                'status': 'success',
                'segments': qualifying_segments,
                'analysis_method': 'enhanced_heuristic_phase2',
                'phase2_compatible': True,
                'heuristic_analysis': {
                    'prompt_keywords': prompt_keywords,
                    'inferred_content_type': content_type,
                    'total_analyzed': len(candidate_segments),
                    'qualified_segments': len(qualifying_segments),
                    'average_score': sum(seg['heuristic_score'] for seg in scored_segments) / max(len(scored_segments), 1)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced heuristic analysis failed: {e}")
            return {
                'status': 'error',
                'error': f"Enhanced heuristic analysis failed: {e}",
                'segments': candidate_segments
            }
    
    def _extract_intent_keywords(self, user_prompt: str) -> List[str]:
        """Extract intent keywords from user prompt for heuristic analysis."""
        prompt_lower = user_prompt.lower()
        
        # Intent keyword mapping
        intent_keywords = {
            'climax': ['climax', 'peak', 'intense', 'dramatic', 'tension', 'exciting', 'thrilling'],
            'comedy': ['funny', 'humor', 'laugh', 'joke', 'amusing', 'entertaining', 'hilarious'],
            'educational': ['learn', 'teach', 'explain', 'instruction', 'tutorial', 'guide', 'how'],
            'emotional': ['emotion', 'feel', 'touching', 'moving', 'heartfelt', 'inspiring'],
            'action': ['action', 'fast', 'quick', 'dynamic', 'movement', 'energy'],
            'highlights': ['best', 'important', 'key', 'main', 'crucial', 'significant'],
            'summary': ['summary', 'overview', 'recap', 'conclusion', 'points']
        }
        
        # Extract relevant keywords
        extracted_keywords = []
        for category, keywords in intent_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                extracted_keywords.extend(keywords)

        
        # Add prompt words directly
        prompt_words = [word.strip('.,?!') for word in prompt_lower.split() if len(word) > 3]
        extracted_keywords.extend(prompt_words)
        
        return list(set(extracted_keywords))
    
    def _infer_content_type_heuristic(self, user_prompt: str, audio_analysis: Dict) -> str:
        """Infer content type using heuristic analysis."""
        prompt_lower = user_prompt.lower()
        
        # Check prompt for content type indicators
        if any(word in prompt_lower for word in ['tutorial', 'learn', 'how', 'teach', 'explain']):
            return 'tutorial'
        elif any(word in prompt_lower for word in ['funny', 'comedy', 'humor', 'laugh']):
            return 'comedy'
        elif any(word in prompt_lower for word in ['climax', 'dramatic', 'intense', 'action']):
            return 'dramatic'
        elif any(word in prompt_lower for word in ['summary', 'overview', 'highlights', 'best']):
            return 'highlights'
        
        # Analyze transcription for content type clues
        transcription_text = ""
        if 'transcription' in audio_analysis and 'segments' in audio_analysis['transcription']:
            transcription_text = " ".join([
                seg.get('text', '') for seg in audio_analysis['transcription']['segments']
            ]).lower()
        
        if 'learn' in transcription_text or 'tutorial' in transcription_text:
            return 'tutorial'
        elif 'funny' in transcription_text or 'laugh' in transcription_text:
            return 'comedy'
        else:
            return 'general'
    
    def _calculate_enhanced_heuristic_score(self,
                                          segment: Dict,
                                          prompt_keywords: List[str],
                                          content_type: str,
                                          audio_analysis: Dict,
                                          scene_analysis: Dict) -> float:
        """Calculate enhanced heuristic score for Phase 2 compatibility."""
        segment_text = segment.get('segment_text', '').lower()
        
        # Scoring factors
        scores = {
            'keyword_match': self._score_keyword_matching(segment_text, prompt_keywords),
            'content_type_alignment': self._score_content_type_alignment_heuristic(segment, content_type),
            'duration_appropriateness': self._score_duration_appropriateness_heuristic(segment),
            'content_density': self._score_content_density_heuristic(segment),
            'position_bonus': self._score_position_bonus_heuristic(segment, audio_analysis),
            'engagement_score': self._score_content_engagement(segment, audio_analysis, scene_analysis)
        }
        
        # Weighted combination
        weighted_score = (
            scores['keyword_match'] * 0.3 +
            scores['content_type_alignment'] * 0.2 +
            scores['duration_appropriateness'] * 0.15 +
            scores['content_density'] * 0.15 +
            scores['position_bonus'] * 0.1 +
            scores['engagement_score'] * 0.2
        )
        
        return max(0, min(1, weighted_score))
    
    def _score_keyword_matching(self, segment_text: str, keywords: List[str]) -> float:
        """Score keyword matching in segment text."""
        if not keywords:
            return 0.5
        
        matches = sum(1 for keyword in keywords if keyword in segment_text)
        return min(1.0, matches / len(keywords) * 2)  # Scale up to reward multiple matches
    
    def _score_content_type_alignment_heuristic(self, segment: Dict, content_type: str) -> float:
        """Score content type alignment using heuristics."""
        segment_text = segment.get('segment_text', '').lower()
        
        type_indicators = {
            'tutorial': ['step', 'learn', 'how', 'method', 'process', 'technique'],
            'comedy': ['funny', 'laugh', 'joke', 'humor', 'amusing'],
            'dramatic': ['intense', 'dramatic', 'peak', 'climax', 'tension'],
            'highlights': ['important', 'key', 'best', 'main', 'crucial'],
            'general': []
        }
        
        indicators = type_indicators.get(content_type, [])
        if not indicators:
            return 0.6  # Neutral score for general content
        
        matches = sum(1 for indicator in indicators if indicator in segment_text)
        return min(1.0, 0.4 + (matches / len(indicators)) * 0.6)
    
    def _score_duration_appropriateness_heuristic(self, segment: Dict) -> float:
        """Score duration appropriateness for short-form content."""
        duration = segment.get('duration', 0)
        
        if 15 <= duration <= 45:
            return 1.0
        elif 10 <= duration < 15 or 45 < duration <= 60:
            return 0.8
        elif 5 <= duration < 10 or 60 < duration <= 90:
            return 0.5
        else:
            return 0.2
    
    def _score_content_density_heuristic(self, segment: Dict) -> float:
        """Score content density and information richness."""
        text = segment.get('segment_text', '')
        duration = segment.get('duration', 1)
        
        if not text:
            return 0.3
        
        words_per_second = len(text.split()) / max(duration, 1)
        
        if 1.5 <= words_per_second <= 3.5:
            return 1.0
        elif 1 <= words_per_second < 1.5 or 3.5 < words_per_second <= 5:
            return 0.7
        else:
            return 0.4
    
    def _score_position_bonus_heuristic(self, segment: Dict, audio_analysis: Dict) -> float:
        """Score based on position in video (beginnings and endings often important)."""
        start_time = segment.get('start_time', 0)
        
        # Get total duration from audio analysis
        total_duration = 0
        if 'transcription' in audio_analysis and 'segments' in audio_analysis['transcription']:
            segments = audio_analysis['transcription']['segments']
            if segments:
                total_duration = max(seg.get('end', 0) for seg in segments)
        
        if total_duration == 0:
            return 0.5
        
        # Calculate position ratio
        position_ratio = start_time / total_duration
        
        # Bonus for beginning (first 10%) and ending (last 15%)
        if position_ratio <= 0.1:
            return 0.8  # Beginning bonus
        elif position_ratio >= 0.85:
            return 0.9  # Ending bonus
        elif 0.4 <= position_ratio <= 0.6:
            return 0.7  # Middle highlight bonus
        else:
            return 0.5  # Neutral
    
    def _score_content_engagement(self, segment: Dict, audio_analysis: Dict, scene_analysis: Dict) -> float:
        """Score content engagement potential - NEW ENHANCED METHOD"""
        engagement_score = 0.5  # Base score
        
        # Audio engagement indicators
        segment_text = segment.get('segment_text', '').lower()
        
        # Emotional content indicators
        emotional_words = ['amazing', 'incredible', 'wow', 'fantastic', 'perfect', 'best', 'great', 
                          'excellent', 'awesome', 'brilliant', 'outstanding', 'remarkable']
        action_words = ['action', 'moving', 'fast', 'quick', 'dynamic', 'intense', 'energy']
        engagement_words = ['look', 'watch', 'see', 'check', 'notice', 'observe', 'focus']
        
        emotional_count = sum(1 for word in emotional_words if word in segment_text)
        action_count = sum(1 for word in action_words if word in segment_text)
        engagement_count = sum(1 for word in engagement_words if word in segment_text)
        
        # Boost for emotional content
        if emotional_count > 0:
            engagement_score += min(0.3, emotional_count * 0.1)
        
        # Boost for action content
        if action_count > 0:
            engagement_score += min(0.2, action_count * 0.1)
        
        # Boost for engaging language
        if engagement_count > 0:
            engagement_score += min(0.15, engagement_count * 0.05)
        
        # Speech pattern analysis
        words = segment_text.split()
        if len(words) > 5:
            # Varied sentence length indicates natural speech
            avg_word_length = sum(len(word) for word in words) / len(words)
            if 4 <= avg_word_length <= 7:  # Optimal word length range
                engagement_score += 0.1
            
            # Question indicators (engaging content often has questions)
            if '?' in segment_text or any(word in segment_text for word in ['what', 'how', 'why', 'when', 'where']):
                engagement_score += 0.1
        
        # Scene change indicators (visual engagement)
        if scene_analysis and 'transitions' in scene_analysis:
            segment_start = segment['start_time']
            segment_end = segment['end_time']
            
            # Check for scene transitions within segment
            transitions_in_segment = [
                t for t in scene_analysis['transitions'] 
                if segment_start <= t.get('timestamp', 0) <= segment_end
            ]
            
            # Multiple scene changes indicate dynamic content
            if len(transitions_in_segment) >= 2:
                engagement_score += 0.2
            elif len(transitions_in_segment) == 1:
                engagement_score += 0.1
        
        return min(1.0, engagement_score)
    
    def _score_scene_quality(self, segment: Dict, scene_analysis: Dict) -> float:
        """Score visual scene quality - NEW ENHANCED METHOD"""
        if not scene_analysis:
            return 0.5
        
        quality_score = 0.5
        segment_start = segment['start_time']
        segment_end = segment['end_time']
        
        # Check for scene quality indicators
        if 'quality_metrics' in scene_analysis:
            quality_metrics = scene_analysis['quality_metrics']
            
            # Look for quality indicators in time range
            if isinstance(quality_metrics, list):
                relevant_metrics = [
                    m for m in quality_metrics 
                    if segment_start <= m.get('timestamp', 0) <= segment_end
                ]
                
                if relevant_metrics:
                    avg_quality = sum(m.get('quality_score', 0.5) for m in relevant_metrics) / len(relevant_metrics)
                    quality_score = avg_quality
        
        # Scene variety boost
        if 'scene_types' in scene_analysis:
            scene_types = scene_analysis.get('scene_types', [])
            relevant_scenes = [
                s for s in scene_types 
                if segment_start <= s.get('start_time', 0) <= segment_end or 
                   segment_start <= s.get('end_time', 0) <= segment_end
            ]
            
            # Multiple scene types indicate rich content
            unique_scene_types = len(set(s.get('type', '') for s in relevant_scenes))
            if unique_scene_types >= 2:
                quality_score += 0.2
            elif unique_scene_types == 1:
                quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _derive_target_keywords(self, user_prompt: str) -> List[str]:
        """Derive topic-specific keywords/synonyms from the user prompt.
        Special handling for 'Pro motion' (Apple ProMotion 120 Hz) cases.
        """
        s = (user_prompt or '').lower().strip()
        kws: List[str] = []
        # Handle common misspellings/variations and map to ProMotion semantics
        if ('pro motion' in s) or ('pro-motion' in s) or ('promotion' in s and 'display' in s) or ('pro motion display' in s):
            kws.extend([
                'promotion', 'pro motion', 'pro-motion',
                'promtion', 'pro-motion display', 'promotion display',
                '120 hz', '120hz', 'high refresh', 'refresh rate', 'variable refresh', 'adaptive refresh',
                'ltpo', 'smooth scrolling', 'screen tearing', 'motion blur', 'frame rate', 'fps',
                'display', 'screen', 'iphone', 'ipad pro', 'apple', 'pro models'
            ])
        # Generic display/refresh hints
        if any(x in s for x in ['refresh', 'hz', 'display', 'screen', 'scroll']):
            kws.extend(['refresh rate', '120 hz', '90 hz', '60 hz', 'adaptive refresh', 'ltpo', 'smooth scrolling'])
        # Deduplicate
        return sorted(set(kws))

    def _inject_keywords_into_prompt(self, base_prompt: str, user_prompt: str) -> str:
        """Append derived keywords/synonyms to the instruction to guide the LLM."""
        kws = self._derive_target_keywords(user_prompt)
        if not kws:
            return base_prompt
        hint = f"\n\nTarget keywords/synonyms (treat as equivalent): {json.dumps(kws)}\n"
        return base_prompt + hint
    
    def _get_dynamic_threshold(self, user_prompt, evaluated_segments):
        """
        Calculate dynamic recommendation threshold based on user prompt and content matches
        
        Args:
            user_prompt: User's search query
            evaluated_segments: List of evaluated segments with scores
            
        Returns:
            float: Threshold value (0.3-0.6) based on keyword specificity
        """
        try:
            if not user_prompt or not evaluated_segments:
                return 0.6  # Default threshold
                
            # Derive target keywords from user prompt
            target_keywords = self._derive_target_keywords(user_prompt)
            
            # Check for explicit keyword matches in segment text
            keyword_match_segments = []
            for seg in evaluated_segments:
                segment_text = seg.get('text', '').lower()
                
                # Count direct keyword matches
                matches = 0
                for keyword in target_keywords:
                    if keyword.lower() in segment_text:
                        matches += 1
                        
                if matches > 0:
                    keyword_match_segments.append({
                        'segment': seg,
                        'matches': matches,
                        'score': seg.get('contextual_overall_score', 0)
                    })
            
            # Dynamic threshold logic
            if keyword_match_segments:
                # We have explicit keyword matches - be more lenient
                max_matches = max(item['matches'] for item in keyword_match_segments)
                avg_match_score = sum(item['score'] for item in keyword_match_segments) / len(keyword_match_segments)
                
                if max_matches >= 3:
                    # Multiple keyword matches - very specific content
                    threshold = 0.3
                elif max_matches >= 2:
                    # Some keyword matches - moderately specific
                    threshold = 0.4
                elif max_matches >= 1 and avg_match_score >= 0.25:
                    # Single keyword match with decent score
                    threshold = 0.45
                else:
                    # Weak keyword matches
                    threshold = 0.55
                    
                self.logger.info(f"🎯 DYNAMIC THRESHOLD: {threshold:.2f} (keyword_segments={len(keyword_match_segments)}, max_matches={max_matches}, avg_score={avg_match_score:.3f})")
                return threshold
            else:
                # No explicit keyword matches - use standard threshold
                self.logger.info(f"🎯 STANDARD THRESHOLD: 0.6 (no keyword matches found)")
                return 0.6
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating dynamic threshold: {e}")
            return 0.6  # Safe fallback
