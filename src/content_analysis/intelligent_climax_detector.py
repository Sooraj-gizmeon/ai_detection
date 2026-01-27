# src/content_analysis/intelligent_climax_detector.py
"""Intelligent climax and dramatic peak detection using semantic analysis"""

import logging
import json
import re
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np


class IntelligentClimaxDetector:
    """
    Advanced climax detection using semantic understanding, emotional analysis,
    and narrative structure recognition instead of hardcoded keywords.
    """
    
    def __init__(self, ollama_client=None):
        """Initialize intelligent climax detector."""
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        
        # Semantic groups for climax-related concepts (expandable)
        self.climax_semantic_groups = {
            'intensity_peaks': {
                'synonyms': [
                    'climax', 'peak', 'apex', 'zenith', 'pinnacle', 'summit', 'crescendo',
                    'high point', 'turning point', 'crucial moment', 'pivotal moment',
                    'decisive moment', 'critical point', 'breaking point', 'tipping point'
                ],
                'weight': 1.0
            },
            'dramatic_moments': {
                'synonyms': [
                    'dramatic', 'intense', 'powerful', 'gripping', 'compelling',
                    'thrilling', 'electrifying', 'breathtaking', 'stunning',
                    'explosive', 'spectacular', 'mind-blowing', 'jaw-dropping'
                ],
                'weight': 0.8
            },
            'emotional_peaks': {
                'synonyms': [
                    'emotional', 'touching', 'moving', 'heartbreaking', 'inspiring',
                    'overwhelming', 'profound', 'deep', 'powerful', 'stirring',
                    'passionate', 'intense feelings', 'emotional roller coaster'
                ],
                'weight': 0.7
            },
            'narrative_structure': {
                'synonyms': [
                    'best scenes', 'key moments', 'highlights', 'memorable parts',
                    'standout scenes', 'most important', 'essential scenes',
                    'defining moments', 'unforgettable', 'iconic scenes'
                ],
                'weight': 0.9
            },
            'action_intensity': {
                'synonyms': [
                    'action-packed', 'fast-paced', 'adrenaline', 'excitement',
                    'high-energy', 'dynamic', 'explosive action', 'intense action',
                    'thrilling sequences', 'edge-of-seat', 'heart-pounding'
                ],
                'weight': 0.8
            },
            'conflict_resolution': {
                'synonyms': [
                    'resolution', 'conclusion', 'finale', 'showdown', 'confrontation',
                    'face-off', 'final battle', 'ultimate moment', 'decisive action',
                    'moment of truth', 'final act', 'endgame'
                ],
                'weight': 0.9
            }
        }
        
        # Emotional intensity indicators for audio analysis
        self.intensity_indicators = {
            'volume_patterns': ['loud', 'shouting', 'yelling', 'screaming', 'whisper then loud'],
            'speech_patterns': ['rapid speech', 'slow and dramatic', 'emphasis', 'pausing'],
            'emotional_words': ['amazing', 'incredible', 'unbelievable', 'shocking', 'stunning'],
            'exclamation_patterns': ['!', '?!', 'wow', 'oh my god', 'no way', 'incredible']
        }
        
        # Visual intensity markers (for vision analysis integration)
        self.visual_intensity_markers = {
            'movement': ['fast motion', 'sudden movement', 'camera shake', 'quick cuts'],
            'lighting': ['dramatic lighting', 'shadows', 'bright flash', 'spotlight'],
            'composition': ['close-up', 'extreme close-up', 'wide shot for impact'],
            'color': ['high contrast', 'vivid colors', 'dramatic colors']
        }
    
    async def detect_climax_segments(self,
                                   user_prompt: str,
                                   candidate_segments: List[Dict],
                                   audio_analysis: Dict,
                                   vision_analysis: Optional[Dict] = None,
                                   scene_analysis: Optional[Dict] = None,
                                   llm_provider: str = "ollama") -> Dict:
        """
        Intelligently detect climax segments using semantic understanding.
        
        Args:
            user_prompt: User's request (e.g., "get the best scenes from the climax of the movie")
            candidate_segments: All possible video segments
            audio_analysis: Audio transcription and analysis
            vision_analysis: Visual content analysis
            scene_analysis: Scene detection results
            llm_provider: LLM provider to use ('openai' or 'ollama')
            
        Returns:
            Ranked climax segments with confidence scores
        """
        try:
            self.logger.info(f"Intelligent climax detection for: '{user_prompt}'")
            
            # 1. Semantic analysis of user prompt
            self.logger.debug("Step 1: Starting semantic analysis...")
            semantic_analysis = await self._analyze_prompt_semantically(user_prompt)
            self.logger.debug(f"Semantic analysis completed: {semantic_analysis.get('overall_climax_confidence', 0):.2f} confidence")
            
            # 2. Emotional intensity mapping across video
            self.logger.debug("Step 2: Starting emotional intensity mapping...")
            intensity_timeline = self._map_emotional_intensity(audio_analysis, scene_analysis)
            self.logger.debug(f"Intensity mapping completed: {len(intensity_timeline.get('timeline', []))} timeline points")
            
            # 3. Narrative structure analysis
            self.logger.debug("Step 3: Starting narrative structure analysis...")
            narrative_analysis = await self._analyze_narrative_structure(
                audio_analysis, intensity_timeline, llm_provider
            )
            self.logger.debug("Narrative analysis completed")
            
            # 4. Score segments using multiple intelligence layers
            self.logger.debug("Step 4: Starting segment scoring...")
            scored_segments = await self._score_segments_intelligently(
                candidate_segments,
                semantic_analysis,
                intensity_timeline,
                narrative_analysis,
                audio_analysis,
                vision_analysis
            )
            self.logger.debug(f"Segment scoring completed: {len(scored_segments)} segments scored")
            
            # 5. Apply climax-specific ranking
            self.logger.debug("Step 5: Starting climax ranking...")
            final_segments = self._rank_by_climax_likelihood(
                scored_segments,
                semantic_analysis,
                max_segments=10
            )
            self.logger.debug(f"Climax ranking completed: {len(final_segments)} final segments")
            
            return {
                'status': 'success',
                'detection_method': 'intelligent_semantic',
                'user_prompt': user_prompt,
                'semantic_analysis': semantic_analysis,
                'intensity_peaks_detected': len(intensity_timeline.get('peaks', [])),
                'narrative_structure': narrative_analysis,
                'climax_segments': final_segments,
                'confidence_distribution': self._calculate_confidence_distribution(final_segments)
            }
            
        except Exception as e:
            self.logger.error(f"Intelligent climax detection failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'climax_segments': []
            }
    
    async def _analyze_prompt_semantically(self, user_prompt: str) -> Dict:
        """
        Analyze user prompt using semantic understanding instead of exact keyword matching.
        """
        prompt_lower = user_prompt.lower()
        
        # Calculate semantic similarity for each group
        semantic_scores = {}
        detected_concepts = []
        
        for group_name, group_data in self.climax_semantic_groups.items():
            synonyms = group_data['synonyms']
            weight = group_data['weight']
            
            # Count matches (flexible partial matching)
            matches = 0
            matched_terms = []
            
            for synonym in synonyms:
                # Exact match
                if synonym in prompt_lower:
                    matches += 1.0
                    matched_terms.append(synonym)
                # Partial word match (e.g., "dramatic" matches "drama")
                elif any(word in prompt_lower for word in synonym.split()):
                    matches += 0.5
                    matched_terms.append(f"~{synonym}")
            
            if matches > 0:
                score = (matches / len(synonyms)) * weight
                semantic_scores[group_name] = score
                detected_concepts.extend(matched_terms)
        
        # Use AI to enhance semantic understanding if available
        ai_semantic_enhancement = {}
        if self.ollama_client:
            ai_semantic_enhancement = await self._ai_enhance_semantic_analysis(
                user_prompt, semantic_scores
            )
        
        return {
            'semantic_scores': semantic_scores,
            'detected_concepts': detected_concepts,
            'dominant_concept_group': max(semantic_scores.keys(), key=semantic_scores.get) if semantic_scores else 'general',
            'overall_climax_confidence': sum(semantic_scores.values()) / len(semantic_scores) if semantic_scores else 0.0,
            'ai_enhancement': ai_semantic_enhancement,
            'interpretation': self._interpret_semantic_analysis(semantic_scores)
        }
    
    def _map_emotional_intensity(self, audio_analysis: Dict, scene_analysis: Optional[Dict]) -> Dict:
        """
        Map emotional intensity across the video timeline.
        """
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        intensity_timeline = []
        peaks = []
        
        for segment in segments:
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').lower()
            
            # Calculate multiple intensity factors
            volume_intensity = self._calculate_volume_intensity(segment, text)
            emotional_intensity = self._calculate_emotional_intensity(text)
            speech_pattern_intensity = self._calculate_speech_pattern_intensity(segment, text)
            
            # Combine intensities
            combined_intensity = (
                volume_intensity * 0.4 +
                emotional_intensity * 0.4 +
                speech_pattern_intensity * 0.2
            )
            
            timeline_point = {
                'start_time': start_time,
                'end_time': end_time,
                'intensity': combined_intensity,
                'volume_component': volume_intensity,
                'emotional_component': emotional_intensity,
                'speech_component': speech_pattern_intensity,
                'text_sample': text[:100] + '...' if len(text) > 100 else text
            }
            
            intensity_timeline.append(timeline_point)
            
            # Identify peaks (intensity > 0.7)
            if combined_intensity > 0.7:
                peaks.append(timeline_point)
        
        # Smooth the timeline and identify trends
        smoothed_timeline = self._smooth_intensity_timeline(intensity_timeline)
        
        return {
            'timeline': smoothed_timeline,
            'peaks': peaks,
            'average_intensity': np.mean([p['intensity'] for p in intensity_timeline]) if intensity_timeline else 0,
            'max_intensity': max(p['intensity'] for p in intensity_timeline) if intensity_timeline else 0,
            'intensity_variance': np.var([p['intensity'] for p in intensity_timeline]) if intensity_timeline else 0
        }
    
    async def _analyze_narrative_structure(self, audio_analysis: Dict, intensity_timeline: Dict, llm_provider: str = "ollama") -> Dict:
        """
        Analyze narrative structure to identify likely climax positions.
        
        Args:
            audio_analysis: Audio analysis data
            intensity_timeline: Emotional intensity timeline
            llm_provider: LLM provider to use ('openai' or 'ollama')
        """
        if not self.ollama_client and llm_provider.lower() == "ollama":
            return self._heuristic_narrative_analysis(intensity_timeline)
        
        # Use configurable LLM provider
        try:
            from ..ai_integration.llm_provider import create_llm_provider
            
            # Determine which provider to use
            if llm_provider.lower() == "openai":
                llm = create_llm_provider("openai")
            else:
                if not self.ollama_client:
                    return self._heuristic_narrative_analysis(intensity_timeline)
                llm = create_llm_provider("ollama", self.ollama_client)
        except ImportError:
            # Fallback if LLM provider system not available
            if not self.ollama_client:
                return self._heuristic_narrative_analysis(intensity_timeline)
            llm = self.ollama_client
        
        # Get full transcript
        transcription = audio_analysis.get('transcription', {})
        full_text = ' '.join([seg.get('text', '') for seg in transcription.get('segments', [])])
        
        # AI-powered narrative analysis
        narrative_prompt = f"""
        Analyze this video transcript for narrative structure and identify the most likely climax segments:
        
        Transcript: {full_text[:4000]}...  # Truncate for context window
        
        Intensity Timeline Summary:
        - Average intensity: {intensity_timeline.get('average_intensity', 0):.2f}
        - Peak moments: {len(intensity_timeline.get('peaks', []))}
        - Max intensity: {intensity_timeline.get('max_intensity', 0):.2f}
        
        Analyze:
        1. Story structure and progression
        2. Emotional arc development
        3. Most likely climax timing (as percentage of total duration)
        4. Key dramatic moments
        5. Resolution patterns
        
        Respond in JSON:
        {{
            "story_structure": "three_act|hero_journey|episodic|other",
            "estimated_climax_position": 0.75,
            "climax_confidence": 0.85,
            "dramatic_arc": ["setup", "rising_action", "climax", "resolution"],
            "key_moments": [
                {{"timestamp_percent": 0.65, "description": "conflict escalation", "intensity": 0.8}},
                {{"timestamp_percent": 0.75, "description": "climax moment", "intensity": 0.95}}
            ],
            "narrative_type": "action|drama|comedy|documentary|educational"
        }}
        """
        
        try:
            if hasattr(llm, 'generate_response'):
                response = await llm.generate_response(narrative_prompt)
            elif hasattr(llm, '_make_request'):
                response = await llm._make_request(narrative_prompt)
            else:
                # Fallback for direct ollama_client usage
                response = await llm.generate_response(narrative_prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"AI narrative analysis failed: {e}")
        
        return self._heuristic_narrative_analysis(intensity_timeline)
    
    def _heuristic_narrative_analysis(self, intensity_timeline: Dict) -> Dict:
        """Fallback heuristic narrative analysis with credits exclusion."""
        timeline = intensity_timeline.get('timeline', [])
        if not timeline:
            return {'estimated_climax_position': 0.75, 'climax_confidence': 0.5}
        
        try:
            # Get total duration
            total_duration = timeline[-1]['end_time'] if timeline else 100
            
            # Exclude the last 10% of the video (likely credits/end sequences)
            credits_threshold = total_duration * 0.9
            story_timeline = [p for p in timeline if p['end_time'] <= credits_threshold]
            
            if not story_timeline:
                # If everything is in the last 10%, use the last 20% instead
                credits_threshold = total_duration * 0.8
                story_timeline = [p for p in timeline if p['end_time'] <= credits_threshold]
            
            if not story_timeline:
                # Fallback to full timeline if still empty
                story_timeline = timeline
            
            # Find the highest intensity period in the story content
            max_intensity_point = max(story_timeline, key=lambda x: x['intensity'])
            climax_position = max_intensity_point['start_time'] / total_duration
            
            # Prefer climax positions between 60-85% of the video (typical story structure)
            ideal_range_start = 0.6
            ideal_range_end = 0.85
            
            if ideal_range_start <= climax_position <= ideal_range_end:
                position_bonus = 0.2  # Boost confidence if in ideal range
            else:
                position_bonus = 0.0
                
        except (ValueError, KeyError, IndexError):
            # Fallback if timeline is malformed
            return {'estimated_climax_position': 0.75, 'climax_confidence': 0.5}
        
        # Confidence based on intensity variance (higher variance = clearer structure)
        variance = intensity_timeline.get('intensity_variance', 0)
        base_confidence = min(variance * 2, 1.0)  # Scale variance to confidence
        final_confidence = min(base_confidence + position_bonus, 1.0)
        
        return {
            'story_structure': 'three_act' if final_confidence > 0.6 else 'unknown',
            'estimated_climax_position': climax_position,
            'climax_confidence': final_confidence,
            'narrative_type': 'dramatic' if max_intensity_point['intensity'] > 0.7 else 'unknown',
            'credits_excluded': True,
            'story_duration_analyzed': credits_threshold / total_duration,
            'key_moments': [
                {
                    'timestamp_percent': climax_position,
                    'description': 'highest intensity story moment (credits excluded)',
                    'intensity': max_intensity_point['intensity'],
                    'in_ideal_range': ideal_range_start <= climax_position <= ideal_range_end
                }
            ]
        }
    
    async def _score_segments_intelligently(self,
                                          segments: List[Dict],
                                          semantic_analysis: Dict,
                                          intensity_timeline: Dict,
                                          narrative_analysis: Dict,
                                          audio_analysis: Dict,
                                          vision_analysis: Optional[Dict]) -> List[Dict]:
        """
        Score segments using multiple intelligence layers.
        """
        scored_segments = []
        estimated_climax_pos = narrative_analysis.get('estimated_climax_position', 0.75)
        
        for segment in segments:
            start_time = segment['start_time']
            end_time = segment['end_time']
            duration = end_time - start_time
            
            # Get total video duration for positioning analysis
            try:
                total_duration = max(p['end_time'] for p in intensity_timeline.get('timeline', [{'end_time': 100}]))
            except (ValueError, KeyError):
                total_duration = 100  # Fallback duration
            
            segment_position = (start_time + end_time) / 2 / total_duration
            
            scored_segment = segment.copy()
            
            # 1. Semantic relevance score (30%)
            semantic_score = sum(semantic_analysis['semantic_scores'].values()) * 0.3
            
            # 2. Intensity alignment score (25%)
            intensity_score = self._calculate_segment_intensity_score(
                start_time, end_time, intensity_timeline
            ) * 0.25
            
            # 3. Narrative position score (20%)
            position_distance = abs(segment_position - estimated_climax_pos)
            position_score = max(0, (1 - position_distance * 2)) * 0.2  # Closer to estimated climax = higher score
            
            # 4. Content quality score (15%)
            content_score = segment.get('quality_score', 0.5) * 0.15
            
            # 5. Duration appropriateness (10%)
            duration_score = self._score_duration_for_climax(duration) * 0.1
            
            # 6. Enhanced credits detection and penalty
            credits_penalty = 0.0
            is_credits = self._is_credits_content(segment, audio_analysis, vision_analysis)
            
            if is_credits:
                credits_penalty = -0.9  # Very heavy penalty for detected credits
                self.logger.debug(f"Credits detected and penalized for segment at {segment_position:.1%}")
            elif segment_position > 0.9:  # Last 10% of video (fallback position check)
                credits_penalty = -0.7  # Heavy penalty for likely credits
                self.logger.debug(f"Late-video penalty applied to segment at {segment_position:.1%}")
            elif segment_position > 0.87:  # Last 13% gets smaller penalty
                credits_penalty = -0.4
                self.logger.debug(f"End-sequence penalty applied to segment at {segment_position:.1%}")
            
            # 7. Optimal climax range bonus - bias toward 60-80% range
            optimal_climax_range_bonus = 0.0
            if 0.6 <= segment_position <= 0.8:
                optimal_climax_range_bonus = 0.15  # Boost for ideal climax timing
            elif 0.5 <= segment_position <= 0.85:
                optimal_climax_range_bonus = 0.1   # Smaller boost for acceptable range
            
            # Combine all scores
            total_climax_score = (
                semantic_score + intensity_score + 
                position_score + content_score + duration_score + 
                credits_penalty + optimal_climax_range_bonus
            )
            
            scored_segment.update({
                'climax_score': max(0, total_climax_score),  # Don't allow negative scores
                'semantic_relevance': semantic_score / 0.3,  # Normalize back
                'intensity_alignment': intensity_score / 0.25,
                'narrative_position_score': position_score / 0.2,
                'content_quality': content_score / 0.15,
                'duration_appropriateness': duration_score / 0.1,
                'credits_penalty': credits_penalty,
                'optimal_range_bonus': optimal_climax_range_bonus,
                'segment_position_in_video': segment_position,
                'distance_from_estimated_climax': position_distance,
                'is_likely_credits': is_credits or segment_position > 0.9,
                'credits_detected': is_credits
            })
            
            scored_segments.append(scored_segment)
        
        return sorted(scored_segments, key=lambda x: x['climax_score'], reverse=True)
    
    def _calculate_segment_intensity_score(self, start_time: float, end_time: float, intensity_timeline: Dict) -> float:
        """Calculate average intensity for a segment."""
        timeline = intensity_timeline.get('timeline', [])
        relevant_points = [
            point for point in timeline
            if point['start_time'] < end_time and point['end_time'] > start_time
        ]
        
        if not relevant_points:
            return 0.0
        
        return sum(point['intensity'] for point in relevant_points) / len(relevant_points)
    
    def _score_duration_for_climax(self, duration: float) -> float:
        """Score duration appropriateness for climax scenes."""
        # Climax scenes are typically 30-90 seconds for shorts
        if 30 <= duration <= 90:
            return 1.0
        elif 20 <= duration <= 120:
            return 0.8
        elif 15 <= duration <= 150:
            return 0.6
        else:
            return 0.3
    
    def _rank_by_climax_likelihood(self, segments: List[Dict], semantic_analysis: Dict, max_segments: int = 10) -> List[Dict]:
        """Final ranking by climax likelihood."""
        # Apply additional climax-specific filters
        climax_segments = []
        
        for segment in segments[:max_segments * 2]:  # Consider more candidates
            climax_score = segment['climax_score']
            
            # Boost score if segment has high semantic relevance
            semantic_boost = semantic_analysis['overall_climax_confidence'] * 0.1
            adjusted_score = climax_score + semantic_boost
            
            # Apply minimum thresholds
            if (adjusted_score > 0.4 and  # Minimum quality
                segment.get('intensity_alignment', 0) > 0.3):  # Minimum intensity
                
                segment['final_climax_score'] = adjusted_score
                climax_segments.append(segment)
        
        # Return top segments
        final_segments = sorted(climax_segments, key=lambda x: x['final_climax_score'], reverse=True)
        return final_segments[:max_segments]
    
    # Helper methods for intensity calculation
    def _is_credits_content(self, segment: Dict, audio_analysis: Dict, vision_analysis: Optional[Dict] = None) -> bool:
        """
        Detect if segment contains credits/end sequence content using multiple methods.
        Enhanced to handle visual credits detection since many credits have no audio.
        """
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        
        # Method 1: Position-based detection (most reliable)
        # Get total video duration for position calculation
        total_duration = 100  # Default fallback
        try:
            # Try to get duration from audio analysis
            if audio_analysis.get('segments'):
                last_segment = max(audio_analysis['segments'], key=lambda x: x.get('end', 0))
                total_duration = last_segment.get('end', 100)
        except:
            pass
        
        segment_position = (start_time + end_time) / 2 / total_duration
        
        # Strong position indicator: last 8% of video is very likely credits
        if segment_position > 0.92:
            self.logger.debug(f"Position-based credits detection: segment at {segment_position:.1%} is likely credits")
            return True
        
        # Method 2: Audio transcription analysis (for credits with spoken text)
        credits_keywords = [
            'credits', 'directed by', 'produced by', 'executive producer',
            'written by', 'screenplay by', 'music by', 'cinematography',
            'editing by', 'cast', 'starring', 'featuring', 'special thanks',
            'acknowledgments', 'copyright', '©', 'all rights reserved',
            'soundtrack', 'original music', 'theme song', 'end credits',
            'closing credits', 'rolling credits', 'courtesy of', 'licensed',
            'warner bros', 'universal pictures', 'paramount', 'disney',
            'columbia pictures', 'twentieth century', 'mgm', 'studio'
        ]
        
        # Get transcription segments that overlap with this video segment
        transcription_segments = audio_analysis.get('segments', [])
        overlapping_transcripts = [
            seg for seg in transcription_segments
            if seg.get('start', 0) < end_time and seg.get('end', 0) > start_time
        ]
        
        # Check if transcription contains credits keywords
        combined_text = ' '.join([seg.get('text', '').lower() for seg in overlapping_transcripts])
        credits_matches = sum(1 for keyword in credits_keywords if keyword in combined_text)
        
        if credits_matches >= 2:
            self.logger.debug(f"Audio-based credits detection: found {credits_matches} credits keywords")
            return True
        
        # Method 3: Visual analysis (if available)
        if vision_analysis and vision_analysis.get('segments'):
            visual_credits_score = self._analyze_visual_credits_indicators(
                segment, vision_analysis
            )
            if visual_credits_score > 0.7:
                self.logger.debug(f"Visual-based credits detection: score {visual_credits_score:.2f}")
                return True
        
        # Method 4: Audio pattern analysis (silence or continuous music without speech)
        if self._is_credits_audio_pattern(segment, audio_analysis):
            # Only apply this if in the last 15% of video
            if segment_position > 0.85:
                self.logger.debug("Audio pattern suggests credits in late video position")
                return True
        
        # Method 5: Text pattern analysis (lots of capitalized names/titles)
        if combined_text and len(combined_text.split()) > 5:
            words = combined_text.split()
            capitalized_words = sum(1 for word in words if word and word[0].isupper())
            if capitalized_words / len(words) > 0.7:  # More than 70% capitalized
                self.logger.debug("Text pattern suggests credits (high capitalization)")
                return True
        
        return False
    
    def _analyze_visual_credits_indicators(self, segment: Dict, vision_analysis: Dict) -> float:
        """
        Analyze visual patterns that indicate credits using vision analysis data.
        Returns a score from 0.0 to 1.0 indicating likelihood of credits.
        """
        credits_score = 0.0
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        
        # Get vision analysis segments that overlap with this video segment
        vision_segments = vision_analysis.get('segments', [])
        overlapping_vision = [
            seg for seg in vision_segments
            if seg.get('start_time', 0) < end_time and seg.get('end_time', 0) > start_time
        ]
        
        for vision_seg in overlapping_vision:
            description = vision_seg.get('description', '').lower()
            key_elements = vision_seg.get('key_elements', [])
            scene_type = vision_seg.get('scene_type', '')
            
            # Visual credits indicators
            credits_visual_indicators = [
                'text overlay', 'scrolling text', 'rolling credits', 'names scrolling',
                'black background', 'white text', 'credits roll', 'end credits',
                'text on screen', 'logo', 'studio logo', 'production company',
                'copyright notice', 'legal text', 'acknowledgments',
                'staff names', 'cast list', 'crew list', 'title card'
            ]
            
            # Check description for credits indicators
            visual_matches = sum(1 for indicator in credits_visual_indicators 
                               if indicator in description)
            credits_score += visual_matches * 0.15
            
            # Check key elements for text/credits indicators
            if key_elements:
                text_elements = [elem for elem in key_elements 
                               if any(text_term in str(elem).lower() 
                                     for text_term in ['text', 'title', 'name', 'logo', 'credit'])]
                credits_score += len(text_elements) * 0.1
            
            # Scene type indicators
            if scene_type in ['text_overlay', 'credits', 'title_sequence', 'logo']:
                credits_score += 0.3
        
        return min(credits_score, 1.0)
    
    def _is_credits_audio_pattern(self, segment: Dict, audio_analysis: Dict) -> bool:
        """
        Detect credits based on audio patterns:
        - Long periods of music without speech
        - Silence
        - Very minimal speech
        """
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', 0)
        duration = end_time - start_time
        
        # Get overlapping transcription segments
        transcription_segments = audio_analysis.get('segments', [])
        overlapping_transcripts = [
            seg for seg in transcription_segments
            if seg.get('start', 0) < end_time and seg.get('end', 0) > start_time
        ]
        
        if not overlapping_transcripts:
            # No speech detected - could be credits with music only
            return True
        
        # Calculate speech density
        total_speech_duration = sum(
            min(seg.get('end', 0), end_time) - max(seg.get('start', 0), start_time)
            for seg in overlapping_transcripts
        )
        
        speech_density = total_speech_duration / duration if duration > 0 else 0
        
        # Low speech density suggests credits/music
        if speech_density < 0.3:  # Less than 30% speech
            return True
        
        # Check for very short, sparse speech (typical of credits acknowledgments)
        if len(overlapping_transcripts) > 0:
            avg_speech_length = total_speech_duration / len(overlapping_transcripts)
            if avg_speech_length < 2.0:  # Very short speech segments
                return True
        
        return False
    
    def _calculate_volume_intensity(self, segment: Dict, text: str) -> float:
        """Calculate volume-based intensity."""
        # Use text cues for volume estimation
        volume_indicators = ['!', '?', 'loud', 'shout', 'yell', 'scream', 'whisper']
        matches = sum(1 for indicator in volume_indicators if indicator in text)
        return min(matches * 0.2, 1.0)
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity from text."""
        emotional_words = [
            'amazing', 'incredible', 'shocking', 'stunning', 'unbelievable',
            'dramatic', 'intense', 'powerful', 'overwhelming', 'breathtaking'
        ]
        matches = sum(1 for word in emotional_words if word in text)
        return min(matches * 0.15, 1.0)
    
    def _calculate_speech_pattern_intensity(self, segment: Dict, text: str) -> float:
        """Calculate speech pattern intensity."""
        words = len(text.split())
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        if duration <= 0:
            return 0.0
        
        words_per_second = words / duration
        
        # Fast speech (>3 wps) or very slow speech (<1 wps) indicates intensity
        if words_per_second > 3 or words_per_second < 1:
            return 0.8
        elif words_per_second > 2.5 or words_per_second < 1.5:
            return 0.6
        else:
            return 0.4
    
    def _smooth_intensity_timeline(self, timeline: List[Dict]) -> List[Dict]:
        """Apply smoothing to intensity timeline."""
        if len(timeline) < 3:
            return timeline
        
        smoothed = []
        for i, point in enumerate(timeline):
            if i == 0 or i == len(timeline) - 1:
                smoothed.append(point)
            else:
                # Simple moving average
                prev_intensity = timeline[i-1]['intensity']
                curr_intensity = point['intensity']
                next_intensity = timeline[i+1]['intensity']
                
                smoothed_point = point.copy()
                smoothed_point['intensity'] = (prev_intensity + curr_intensity + next_intensity) / 3
                smoothed.append(smoothed_point)
        
        return smoothed
    
    def _interpret_semantic_analysis(self, semantic_scores: Dict) -> str:
        """Interpret semantic analysis results."""
        if not semantic_scores:
            return "Generic content analysis - no specific climax indicators detected"
        
        top_concepts = sorted(semantic_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if top_concepts[0][1] > 0.7:
            return f"Strong {top_concepts[0][0].replace('_', ' ')} indicators detected"
        elif top_concepts[0][1] > 0.4:
            return f"Moderate {top_concepts[0][0].replace('_', ' ')} patterns found"
        else:
            return "Weak climax indicators - using general content analysis"
    
    def _calculate_confidence_distribution(self, segments: List[Dict]) -> Dict:
        """Calculate confidence distribution for results."""
        if not segments:
            return {}
        
        scores = [s.get('final_climax_score', s.get('climax_score', 0)) for s in segments]
        
        return {
            'high_confidence': len([s for s in scores if s > 0.7]),
            'medium_confidence': len([s for s in scores if 0.4 <= s <= 0.7]),
            'low_confidence': len([s for s in scores if s < 0.4]),
            'average_confidence': np.mean(scores) if scores else 0,
            'confidence_range': f"{min(scores):.2f} - {max(scores):.2f}" if scores else "0.00 - 0.00"
        }
    
    async def _ai_enhance_semantic_analysis(self, user_prompt: str, semantic_scores: Dict) -> Dict:
        """Use AI to enhance semantic understanding."""
        if not self.ollama_client:
            return {}
        
        enhancement_prompt = f"""
        Analyze this user request for video content: "{user_prompt}"
        
        Current semantic analysis detected these concept groups:
        {json.dumps(semantic_scores, indent=2)}
        
        Questions:
        1. What is the user REALLY asking for?
        2. Are there any concepts we missed?
        3. What should be the priority focus?
        4. What synonyms or related terms should we consider?
        
        Respond in JSON:
        {{
            "user_intent": "clear description of what they want",
            "missed_concepts": ["concept1", "concept2"],
            "priority_focus": "main thing to look for",
            "additional_synonyms": ["term1", "term2"],
            "confidence_in_interpretation": 0.85
        }}
        """
        
        try:
            response = await self.ollama_client.generate_response(enhancement_prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            self.logger.warning(f"AI semantic enhancement failed: {e}")
        
        return {}
