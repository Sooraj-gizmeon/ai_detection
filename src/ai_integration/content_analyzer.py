# src/ai_integration/content_analyzer.py
"""Content analyzer using AI integration for intelligent video analysis"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from .ollama_client import OllamaClient


@dataclass
class ContentAnalysisResult:
    """Result of content analysis."""
    content_type: str
    engagement_score: float
    viral_potential: float
    key_moments: List[Dict]
    recommended_cuts: List[Dict]
    emotional_analysis: Dict
    topic_analysis: Dict
    audience_insights: Dict
    optimization_suggestions: List[str]
    confidence_score: float


class ContentAnalyzer:
    """
    AI-powered content analyzer for video understanding and optimization.
    """
    
    def __init__(self, ollama_client: OllamaClient = None, enable_object_detection: bool = True, enable_ai_reframing: bool = True):
        """
        Initialize content analyzer.
         
        Args:
            ollama_client: Ollama client instance (optional)
            enable_object_detection: Enable object detection features (backward compatibility)
            enable_ai_reframing: Enable AI reframing features (backward compatibility)
        """ 
        self.logger = logging.getLogger(__name__)
        self.ollama_client = ollama_client or OllamaClient()
        
        # Store object detection and AI reframing flags for potential future use
        self.enable_object_detection = enable_object_detection
        self.enable_ai_reframing = enable_ai_reframing
        
        # Add segment_generator for backward compatibility
        self.segment_generator = None  # Placeholder for compatibility
        
        if enable_object_detection:
            self.logger.info("Object detection support noted (ai_integration ContentAnalyzer)")
        if enable_ai_reframing:
            self.logger.info("AI reframing support noted (ai_integration ContentAnalyzer)")
        
        # Analysis templates
        self.analysis_prompts = {
            'content_classification': """
            Analyze this video transcript and classify the content:
            
            Transcript: {transcript}
            
            Please provide a JSON response with:
            {{
                "primary_category": "educational|entertainment|lifestyle|business|tech|other",
                "secondary_categories": ["list", "of", "categories"],
                "content_style": "formal|casual|conversational|instructional|storytelling",
                "target_audience": "general|young_adults|professionals|students|creators",
                "confidence": 0.0-1.0
            }}
            """,
            
            'engagement_analysis': """
            Analyze this transcript for engagement potential:
            
            Transcript: {transcript}
            
            Rate engagement factors in JSON format:
            {{
                "hook_strength": 0.0-1.0,
                "emotional_impact": 0.0-1.0,
                "clarity_score": 0.0-1.0,
                "actionability": 0.0-1.0,
                "relatability": 0.0-1.0,
                "curiosity_factor": 0.0-1.0,
                "overall_engagement": 0.0-1.0,
                "key_engagement_moments": [
                    {{"timestamp": 0.0, "reason": "description", "impact": 0.0-1.0}}
                ]
            }}
            """,
            
            'viral_potential_analysis': """
            Assess the viral potential of this content:
            
            Transcript: {transcript}
            
            Provide JSON analysis:
            {{
                "viral_score": 0.0-1.0,
                "viral_elements": ["list", "of", "viral", "elements"],
                "missing_viral_elements": ["what", "could", "improve", "virality"],
                "platform_suitability": {{
                    "tiktok": 0.0-1.0,
                    "youtube_shorts": 0.0-1.0,
                    "instagram_reels": 0.0-1.0
                }},
                "viral_moments": [
                    {{"timestamp": 0.0, "element": "description", "potential": 0.0-1.0}}
                ]
            }}
            """,
            
            'key_moments_detection': """
            Identify key moments and optimal cutting points:
            
            Transcript: {transcript}
            
            Return JSON with:
            {{
                "key_moments": [
                    {{
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "importance": 0.0-1.0,
                        "reason": "why this moment is important",
                        "content_type": "hook|climax|explanation|conclusion|transition"
                    }}
                ],
                "optimal_cuts": [
                    {{
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "duration": 0.0,
                        "cut_score": 0.0-1.0,
                        "cut_reason": "why this would make a good short"
                    }}
                ],
                "natural_breaks": [0.0, 0.0, 0.0]
            }}
            """,
            
            'emotional_analysis': """
            Analyze the emotional content and tone:
            
            Transcript: {transcript}
            
            Provide emotional analysis in JSON:
            {{
                "overall_tone": "positive|negative|neutral|mixed",
                "emotional_intensity": 0.0-1.0,
                "emotional_journey": [
                    {{"timestamp": 0.0, "emotion": "emotion_name", "intensity": 0.0-1.0}}
                ],
                "dominant_emotions": ["emotion1", "emotion2", "emotion3"],
                "emotional_peaks": [
                    {{"timestamp": 0.0, "emotion": "emotion", "intensity": 0.0-1.0, "reason": "description"}}
                ],
                "audience_emotional_response": "predicted response"
            }}
            """
        }
    
    async def analyze_content(self, 
                             transcription: Dict,
                             audio_analysis: Dict = None,
                             visual_analysis: Dict = None) -> ContentAnalysisResult:
        """
        Perform comprehensive content analysis.
        
        Args:
            transcription: Whisper transcription results
            audio_analysis: Audio analysis results (optional)
            visual_analysis: Visual analysis results (optional)
            
        Returns:
            Comprehensive content analysis results
        """
        try:
            # Extract full transcript text
            full_transcript = self._extract_full_transcript(transcription)
            
            if not full_transcript.strip():
                self.logger.warning("Empty transcript - returning minimal analysis")
                return self._create_minimal_analysis()
            
            # Run multiple analysis types concurrently
            analysis_tasks = [
                self._classify_content(full_transcript),
                self._analyze_engagement(full_transcript),
                self._analyze_viral_potential(full_transcript),
                self._detect_key_moments(full_transcript),
                self._analyze_emotions(full_transcript)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            content_classification = results[0] if not isinstance(results[0], Exception) else {}
            engagement_analysis = results[1] if not isinstance(results[1], Exception) else {}
            viral_analysis = results[2] if not isinstance(results[2], Exception) else {}
            moments_analysis = results[3] if not isinstance(results[3], Exception) else {}
            emotional_analysis = results[4] if not isinstance(results[4], Exception) else {}
            
            # Create comprehensive result
            return self._compile_analysis_result(
                content_classification,
                engagement_analysis,
                viral_analysis,
                moments_analysis,
                emotional_analysis,
                full_transcript
            )
            
        except Exception as e:
            self.logger.error(f"Error in content analysis: {e}")
            return self._create_minimal_analysis()
    
    async def _classify_content(self, transcript: str) -> Dict:
        """Classify content type and style."""
        try:
            async with self.ollama_client as client:
                response = await client._make_request(
                    prompt=self.analysis_prompts['viral_potential_analysis'].format(transcript=transcript),
                    model=client.get_best_model("analysis"),
                    cache_key=f"viral_analysis_{hash(transcript)}"
                )
                return client._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Viral analysis error: {e}")
            return {}
    
    async def _detect_key_moments(self, transcript: str) -> Dict:
        """Detect key moments and cutting points."""
        try:
            async with self.ollama_client as client:
                response = await client._make_request(
                    prompt=self.analysis_prompts['key_moments_detection'].format(transcript=transcript),
                    model=client.get_best_model("analysis"),
                    cache_key=f"key_moments_{hash(transcript)}"
                )
                return client._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Key moments detection error: {e}")
            return {}
    
    async def _analyze_emotions(self, transcript: str) -> Dict:
        """Analyze emotional content."""
        try:
            async with self.ollama_client as client:
                response = await client._make_request(
                    prompt=self.analysis_prompts['emotional_analysis'].format(transcript=transcript),
                    model=client.get_best_model("analysis"),
                    cache_key=f"emotional_analysis_{hash(transcript)}"
                )
                return client._parse_json_response(response)
        except Exception as e:
            self.logger.error(f"Emotional analysis error: {e}")
            return {}
    
    def _extract_full_transcript(self, transcription: Dict) -> str:
        """Extract full transcript text."""
        if not transcription:
            return ""
        
        if isinstance(transcription, str):
            return transcription
        
        if 'text' in transcription:
            return transcription['text']
        
        if 'segments' in transcription:
            return ' '.join(
                segment.get('text', '') 
                for segment in transcription['segments']
                if segment.get('text', '').strip()
            )
        
        return ""
    
    def _compile_analysis_result(self,
                               content_classification: Dict,
                               engagement_analysis: Dict,
                               viral_analysis: Dict,
                               moments_analysis: Dict,
                               emotional_analysis: Dict,
                               transcript: str) -> ContentAnalysisResult:
        """Compile all analysis results into a single result."""
        
        # Extract content type
        content_type = content_classification.get('primary_category', 'general')
        
        # Calculate engagement score
        engagement_score = engagement_analysis.get('overall_engagement', 0.5)
        
        # Calculate viral potential
        viral_potential = viral_analysis.get('viral_score', 0.3)
        
        # Extract key moments
        key_moments = moments_analysis.get('key_moments', [])
        
        # Extract recommended cuts
        recommended_cuts = moments_analysis.get('optimal_cuts', [])
        
        # Extract emotional analysis
        emotions = {
            'overall_tone': emotional_analysis.get('overall_tone', 'neutral'),
            'emotional_intensity': emotional_analysis.get('emotional_intensity', 0.5),
            'dominant_emotions': emotional_analysis.get('dominant_emotions', []),
            'emotional_peaks': emotional_analysis.get('emotional_peaks', [])
        }
        
        # Extract topic analysis
        topic_analysis = {
            'primary_category': content_type,
            'secondary_categories': content_classification.get('secondary_categories', []),
            'content_style': content_classification.get('content_style', 'conversational'),
            'platform_suitability': viral_analysis.get('platform_suitability', {})
        }
        
        # Extract audience insights
        audience_insights = {
            'target_audience': content_classification.get('target_audience', 'general'),
            'predicted_response': emotional_analysis.get('audience_emotional_response', 'mixed'),
            'engagement_factors': engagement_analysis.get('key_engagement_moments', [])
        }
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            content_classification,
            engagement_analysis,
            viral_analysis,
            emotional_analysis
        )
        
        # Calculate overall confidence
        confidence_scores = [
            content_classification.get('confidence', 0.5),
            min(1.0, engagement_score + 0.2),  # Boost if analysis completed
            min(1.0, viral_potential + 0.2),   # Boost if analysis completed
        ]
        confidence_score = sum(confidence_scores) / len(confidence_scores)
        
        return ContentAnalysisResult(
            content_type=content_type,
            engagement_score=engagement_score,
            viral_potential=viral_potential,
            key_moments=key_moments,
            recommended_cuts=recommended_cuts,
            emotional_analysis=emotions,
            topic_analysis=topic_analysis,
            audience_insights=audience_insights,
            optimization_suggestions=optimization_suggestions,
            confidence_score=confidence_score
        )
    
    def _generate_optimization_suggestions(self,
                                         content_classification: Dict,
                                         engagement_analysis: Dict,
                                         viral_analysis: Dict,
                                         emotional_analysis: Dict) -> List[str]:
        """Generate actionable optimization suggestions."""
        suggestions = []
        
        # Engagement improvements
        engagement_score = engagement_analysis.get('overall_engagement', 0.5)
        if engagement_score < 0.6:
            suggestions.append("Add more engaging hooks at the beginning")
            suggestions.append("Include more questions to increase interactivity")
        
        if engagement_analysis.get('hook_strength', 0.5) < 0.6:
            suggestions.append("Strengthen the opening hook to capture attention immediately")
        
        if engagement_analysis.get('curiosity_factor', 0.5) < 0.6:
            suggestions.append("Add elements of curiosity or suspense to maintain viewer interest")
        
        # Viral improvements
        viral_score = viral_analysis.get('viral_score', 0.3)
        if viral_score < 0.5:
            missing_elements = viral_analysis.get('missing_viral_elements', [])
            for element in missing_elements[:3]:  # Top 3 suggestions
                suggestions.append(f"Consider adding: {element}")
        
        # Emotional improvements
        emotional_intensity = emotional_analysis.get('emotional_intensity', 0.5)
        if emotional_intensity < 0.6:
            suggestions.append("Increase emotional impact with more expressive language")
        
        # Content-specific suggestions
        content_type = content_classification.get('primary_category', 'general')
        if content_type == 'educational':
            suggestions.append("Break down complex concepts into digestible chunks")
            suggestions.append("Add clear takeaways or action items")
        elif content_type == 'entertainment':
            suggestions.append("Enhance comedic timing and punchlines")
            suggestions.append("Add visual or audio elements for better entertainment value")
        elif content_type == 'lifestyle':
            suggestions.append("Make content more relatable to daily experiences")
            suggestions.append("Include personal anecdotes or examples")
        
        # Platform-specific suggestions
        # Debugging: ensure platform_suitability is a dict
        raw_platform = viral_analysis.get('platform_suitability', {})
        if not isinstance(raw_platform, dict):
            self.logger.warning(
                f"Expected 'platform_suitability' to be dict but got {type(raw_platform).__name__}: {raw_platform}"
            )
            platform_scores = {}
        else:
            platform_scores = raw_platform
        low_scoring_platforms = []
        try:
            low_scoring_platforms = [
                platform for platform, score in platform_scores.items()
                if isinstance(score, (int, float)) and score < 0.5
            ]
        except Exception as e:
            self.logger.warning(f"Error iterating platform_suitability items: {e}")
        
        for platform in low_scoring_platforms:
            if platform == 'tiktok':
                suggestions.append("Optimize for TikTok: Add trending sounds or challenges")
            elif platform == 'youtube_shorts':
                suggestions.append("Optimize for YouTube Shorts: Include clear value proposition")
            elif platform == 'instagram_reels':
                suggestions.append("Optimize for Instagram: Focus on visual appeal and aesthetics")
        
        # Remove duplicates and limit suggestions
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:8]
    
    def _create_minimal_analysis(self) -> ContentAnalysisResult:
        """Create minimal analysis result for fallback cases."""
        return ContentAnalysisResult(
            content_type='general',
            engagement_score=0.5,
            viral_potential=0.3,
            key_moments=[],
            recommended_cuts=[],
            emotional_analysis={
                'overall_tone': 'neutral',
                'emotional_intensity': 0.5,
                'dominant_emotions': [],
                'emotional_peaks': []
            },
            topic_analysis={
                'primary_category': 'general',
                'secondary_categories': [],
                'content_style': 'conversational',
                'platform_suitability': {}
            },
            audience_insights={
                'target_audience': 'general',
                'predicted_response': 'mixed',
                'engagement_factors': []
            },
            optimization_suggestions=[
                "Add clear hooks at the beginning",
                "Include engaging questions",
                "Optimize for target platform"
            ],
            confidence_score=0.3
        )
    
    async def analyze_segment(self, 
                             segment_transcript: str,
                             segment_start: float,
                             segment_end: float) -> Dict:
        """
        Analyze a specific segment of content.
        
        Args:
            segment_transcript: Transcript for the segment
            segment_start: Start time of segment
            segment_end: End time of segment
            
        Returns:
            Segment-specific analysis
        """
        try:
            if not segment_transcript.strip():
                return {'error': 'Empty segment transcript'}
            
            # Focused analysis for segment
            segment_prompt = f"""
            Analyze this video segment (duration: {segment_end - segment_start:.1f}s):
            
            Transcript: {segment_transcript}
            
            Provide JSON analysis:
            {{
                "segment_quality": 0.0-1.0,
                "standalone_potential": 0.0-1.0,
                "hook_effectiveness": 0.0-1.0,
                "conclusion_strength": 0.0-1.0,
                "content_completeness": 0.0-1.0,
                "engagement_factors": ["list", "of", "factors"],
                "improvement_suggestions": ["specific", "suggestions"],
                "optimal_duration": "recommended duration range",
                "best_platform": "tiktok|youtube_shorts|instagram_reels"
            }}
            """
            
            async with self.ollama_client as client:
                response = await client._make_request(
                    prompt=segment_prompt,
                    model=client.get_best_model("analysis"),
                    cache_key=f"segment_analysis_{hash(segment_transcript)}_{segment_start}_{segment_end}"
                )
                
                analysis = client._parse_json_response(response)
                
                # Add timing information
                analysis['segment_timing'] = {
                    'start_time': segment_start,
                    'end_time': segment_end,
                    'duration': segment_end - segment_start
                }
                
                return analysis
                
        except Exception as e:
            self.logger.error(f"Segment analysis error: {e}")
            return {
                'error': str(e),
                'segment_quality': 0.5,
                'standalone_potential': 0.5
            }
    
    async def test_connection(self) -> Dict:
        """Test the AI connection."""
        try:
            async with self.ollama_client as client:
                result = await client.test_connection()
                return result
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'message': 'Failed to connect to AI service'
            }
