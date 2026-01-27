# src/vision_analysis/vision_processor.py
"""Main vision processing coordinator that integrates frame analysis with Ollama vision models"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .frame_analyzer import FrameAnalyzer
from .visual_content_analyzer import VisualContentAnalyzer
from ..ai_integration.ollama_client import OllamaClient


class VisionProcessor:
    """
    Main coordinator for vision-based analysis that combines frame extraction
    with Ollama vision model analysis for enhanced video understanding.
    """
    
    def __init__(self, 
                 ollama_client: OllamaClient = None,
                 frame_target_size: Tuple[int, int] = (512, 288),
                 sample_rate: float = 0.5):
        """
        Initialize vision processor.
        
        Args:
            ollama_client: Ollama client for vision analysis
            frame_target_size: Target size for frame processing (width, height)
            sample_rate: Frame sampling rate for general analysis
        """
        self.ollama_client = ollama_client
        self.frame_analyzer = FrameAnalyzer(
            target_width=frame_target_size[0],
            target_height=frame_target_size[1],
            sample_rate=sample_rate
        )
        self.visual_analyzer = VisualContentAnalyzer(ollama_client)
        
        self.logger = logging.getLogger(__name__)
        
        # Processing configuration
        self.config = {
            'max_frames_per_segment': 3,
            'quick_mode': True,  # Use fast analysis by default
            'enable_caching': True,
            'vision_analysis_timeout': 30.0
        }
    
    async def analyze_video_segments(self, 
                                   video_path: str, 
                                   segments: List[Dict],
                                   audio_transcription: Dict = None) -> Dict:
        """
        Analyze video segments using both frame extraction and vision models.
        
        Args:
            video_path: Path to video file
            segments: List of segment dictionaries with start_time and end_time
            audio_transcription: Audio transcription for context (optional)
            
        Returns:
            Combined visual analysis results
        """
        try:
            start_time = time.time()
            
            self.logger.info(f"VisionProcessor: beginning analyze_video_segments for {len(segments)} segments from {video_path}")
            
            # Extract frames for segments
            segment_frames = self.frame_analyzer.extract_frames_for_segments(video_path, segments)
            # Log frame extraction details
            frame_counts = {seg_id: len(frames) for seg_id, frames in segment_frames.items()}
            self.logger.info(f"VisionProcessor: extracted frames per segment: {frame_counts}")
            
            if not segment_frames:
                self.logger.warning("No frames extracted from video segments")
                return {
                    'status': 'no_frames',
                    'segments': [],
                    'error': 'No frames could be extracted from video segments'
                }
            
            # Analyze visual content if Ollama client is available
            visual_analysis = {}
            if self.ollama_client:
                self.logger.info("VisionProcessor: Ollama client detected, invoking visual analyzer")
                try:
                    # Set timeout for vision analysis
                    visual_analysis = await asyncio.wait_for(
                        self.visual_analyzer.analyze_segment_visuals_efficient(segment_frames),
                        timeout=self.config['vision_analysis_timeout']
                    )
                    # Log analysis summary with better detail
                    if isinstance(visual_analysis, dict):
                        if 'status' in visual_analysis and visual_analysis['status'] == 'success':
                            segment_count = len(visual_analysis.get('segments', []))
                            self.logger.info(f"VisionProcessor: visual analysis completed successfully for {segment_count} segments")
                        else:
                            self.logger.warning(f"VisionProcessor: visual analysis completed with status: {visual_analysis.get('status', 'unknown')}")
                            if 'error' in visual_analysis:
                                self.logger.warning(f"VisionProcessor: visual analysis error: {visual_analysis['error']}")
                        
                        seg_keys = list(visual_analysis.keys())
                        self.logger.info(f"VisionProcessor: visual analysis result keys: {seg_keys}")
                    else:
                        self.logger.warning(f"VisionProcessor: visual analysis returned unexpected type: {type(visual_analysis)}")
                        visual_analysis = {}
                except asyncio.TimeoutError:
                    self.logger.warning(f"Vision analysis timed out after {self.config['vision_analysis_timeout']}s")
                    visual_analysis = {}
                except Exception as e:
                    self.logger.error(f"Vision analysis failed: {e}")
                    visual_analysis = {}
            else:
                self.logger.warning("No Ollama client available, skipping vision model analysis")
            
            # Combine results
            combined_results = self._combine_analysis_results(
                segments, segment_frames, visual_analysis, audio_transcription
            )
            
            processing_time = time.time() - start_time
            
            return {
                'status': 'success',
                'processing_time': processing_time,
                'segments_analyzed': len(segments),
                'frames_extracted': sum(len(frames) for frames in segment_frames.values()),
                'vision_analysis_available': bool(visual_analysis),
                'segments': combined_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in vision analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    def _combine_analysis_results(self, 
                                segments: List[Dict],
                                segment_frames: Dict[str, List[Dict]],
                                visual_analysis: Dict,
                                audio_transcription: Dict = None) -> List[Dict]:
        """Combine frame data with visual analysis results."""
        
        combined_results = []
        
        # FIXED: Handle the actual structure returned by analyze_segment_visuals_efficient
        # It returns {'status': 'success', 'segments': [...]} not a dict keyed by segment_id
        visual_segments_list = []
        if isinstance(visual_analysis, dict):
            if 'segments' in visual_analysis:
                # New format: {'status': 'success', 'segments': [...]}
                visual_segments_list = visual_analysis.get('segments', [])
            else:
                # Old format: {'segment_0': {...}, 'segment_1': {...}}
                # Convert to list for matching
                visual_segments_list = []
        
        # Build a lookup by time range for matching
        visual_lookup = {}
        for v_seg in visual_segments_list:
            if isinstance(v_seg, dict):
                start = v_seg.get('start_time', 0)
                end = v_seg.get('end_time', 0)
                visual_lookup[(start, end)] = v_seg
        
        self.logger.debug(f"Visual analysis has {len(visual_segments_list)} segments, built lookup with {len(visual_lookup)} entries")
        
        for i, segment in enumerate(segments):
            segment_id = f"segment_{i}"
            seg_start = segment.get('start_time', 0)
            seg_end = segment.get('end_time', 0)
            
            # Base segment info
            result = {
                'segment_index': i,
                'start_time': seg_start,
                'end_time': seg_end,
                'duration': seg_end - seg_start
            }
            
            # Add frame information
            if segment_id in segment_frames:
                frames = segment_frames[segment_id]
                result.update({
                    'frame_count': len(frames),
                    'frame_quality_avg': sum(f.get('quality_score', 0) for f in frames) / len(frames) if frames else 0,
                    'frames': frames
                })
            else:
                result.update({
                    'frame_count': 0,
                    'frame_quality_avg': 0,
                    'frames': []
                })
            
            # FIXED: Match visual analysis by time range instead of segment_id
            matched_analysis = None
            
            # Try exact match first
            if (seg_start, seg_end) in visual_lookup:
                matched_analysis = visual_lookup[(seg_start, seg_end)]
            else:
                # Try fuzzy match (within 2 seconds tolerance)
                for (v_start, v_end), v_data in visual_lookup.items():
                    if abs(v_start - seg_start) < 2.0 and abs(v_end - seg_end) < 2.0:
                        matched_analysis = v_data
                        break
            
            # Add visual analysis if matched
            if matched_analysis:
                result.update({
                    'visual_interest': matched_analysis.get('visual_interest', matched_analysis.get('visual_interest_avg', 5)),
                    'engagement_score': matched_analysis.get('engagement_score', matched_analysis.get('engagement_score_avg', 5)),
                    'scene_type': matched_analysis.get('scene_type', matched_analysis.get('dominant_scene_type', 'unknown')),
                    'has_people': matched_analysis.get('has_people', matched_analysis.get('people_visible', False)),
                    'has_action': matched_analysis.get('has_action', False),
                    'has_emotion': matched_analysis.get('has_emotion', False),
                    'visual_quality': matched_analysis.get('visual_quality', 'unknown'),
                    'engagement_quality': matched_analysis.get('engagement_quality', 'unknown'),
                    'visual_recommendation': matched_analysis.get('recommendation', matched_analysis.get('visual_recommendation', 'unknown')),
                    'visual_score': matched_analysis.get('visual_score', 0.5),
                    'people_count': matched_analysis.get('people_count', 0),
                    'vision_analysis': matched_analysis,
                    'has_vision_data': True
                })
            else:
                # Default values when no vision analysis is available
                result.update({
                    'visual_interest': 5,
                    'engagement_score': 5,
                    'scene_type': 'unknown',
                    'has_people': True,  # Assume presence for safety
                    'has_action': False,
                    'has_emotion': False,
                    'visual_quality': 'unknown',
                    'engagement_quality': 'unknown',
                    'visual_recommendation': 'unknown',
                    'visual_score': 0.5,
                    'people_count': 0,
                    'has_vision_data': False
                })
            
            # Add audio context if available
            if audio_transcription:
                result['has_audio_context'] = True
            else:
                result['has_audio_context'] = False
            
            combined_results.append(result)
        
        return combined_results
    
    async def quick_video_assessment(self, video_path: str, max_frames: int = 5) -> Dict:
        """
        Quick assessment of video content using limited frame sampling.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to analyze
            
        Returns:
            Quick assessment results
        """
        try:
            self.logger.info(f"Starting quick video assessment for {video_path}")
            
            # Extract a few representative frames
            frames = self.frame_analyzer.extract_key_frames(video_path)
            
            # Limit frames for quick analysis
            if len(frames) > max_frames:
                # Select frames with best quality scores
                frames = sorted(frames, key=lambda f: f.get('quality_score', 0), reverse=True)[:max_frames]
            
            if not frames:
                return {
                    'status': 'no_frames',
                    'assessment': 'unknown',
                    'confidence': 0
                }
            
            # Quick visual assessment if Ollama is available
            if self.ollama_client:
                assessment = await self.visual_analyzer.quick_scene_assessment(frames)
            else:
                # Fallback assessment based on frame quality
                avg_quality = sum(f.get('quality_score', 0) for f in frames) / len(frames)
                assessment = {
                    'scene_type': 'unknown',
                    'visual_interest': min(10, max(1, avg_quality * 10)),
                    'suitable_for_shorts': avg_quality > 0.6
                }
            
            return {
                'status': 'success',
                'frames_analyzed': len(frames),
                'scene_type': assessment.get('scene_type', 'unknown'),
                'visual_interest': assessment.get('visual_interest', 5),
                'suitable_for_shorts': assessment.get('suitable_for_shorts', False),
                'assessment': 'good' if assessment.get('visual_interest', 5) >= 7 else 'fair' if assessment.get('visual_interest', 5) >= 5 else 'poor',
                'confidence': 0.8 if self.ollama_client else 0.3  # Lower confidence without vision analysis
            }
            
        except Exception as e:
            self.logger.error(f"Error in quick video assessment: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'assessment': 'unknown',
                'confidence': 0
            }
    
    async def analyze_with_audio_context(self, 
                                       video_path: str, 
                                       segments: List[Dict],
                                       transcription: Dict) -> Dict:
        """
        Analyze video segments with audio transcription context for better decision making.
        
        Args:
            video_path: Path to video file
            segments: List of segment dictionaries
            transcription: Audio transcription with timestamps
            
        Returns:
            Enhanced analysis combining visual and audio insights
        """
        try:
            # Get standard vision analysis
            vision_results = await self.analyze_video_segments(video_path, segments, transcription)
            
            if vision_results['status'] != 'success':
                return vision_results
            
            # Enhance segments with audio-visual correlation
            enhanced_segments = []
            
            for segment_result in vision_results['segments']:
                enhanced_segment = segment_result.copy()
                
                # Extract audio text for this segment
                start_time = segment_result['start_time']
                end_time = segment_result['end_time']
                
                segment_text = self._extract_segment_text(transcription, start_time, end_time)
                enhanced_segment['transcription_text'] = segment_text
                
                # Calculate audio-visual alignment score
                alignment_score = self._calculate_audiovisual_alignment(
                    segment_result, segment_text
                )
                enhanced_segment['audiovisual_alignment'] = alignment_score
                
                # Generate combined recommendation
                combined_recommendation = self._generate_combined_recommendation(
                    segment_result, segment_text, alignment_score
                )
                enhanced_segment['combined_recommendation'] = combined_recommendation
                
                enhanced_segments.append(enhanced_segment)
            
            # Update results
            vision_results['segments'] = enhanced_segments
            vision_results['has_audio_context'] = True
            
            return vision_results
            
        except Exception as e:
            self.logger.error(f"Error in audio-visual analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    def _extract_segment_text(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """Extract transcription text for a specific time segment."""
        if not transcription or 'segments' not in transcription:
            return ""
        
        segment_text = []
        for segment in transcription['segments']:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with our time range
            if seg_start < end_time and seg_end > start_time:
                text = segment.get('text', '').strip()
                if text:
                    segment_text.append(text)
        
        return ' '.join(segment_text)
    
    def _calculate_audiovisual_alignment(self, segment_result: Dict, segment_text: str) -> float:
        """Calculate how well visual and audio content align."""
        try:
            # Base score
            alignment_score = 0.5
            
            scene_type = segment_result.get('scene_type', 'unknown')
            has_people = segment_result.get('has_people', False)
            has_action = segment_result.get('has_action', False)
            
            if not segment_text:
                return 0.3  # Low score if no audio
            
            text_lower = segment_text.lower()
            
            # Check for action words matching visual action
            action_words = ['move', 'show', 'demonstrate', 'look', 'see', 'watch']
            has_action_words = any(word in text_lower for word in action_words)
            
            if has_action and has_action_words:
                alignment_score += 0.2
            
            # Check for personal/emotional content matching people presence
            personal_words = ['i', 'you', 'we', 'feel', 'think', 'believe']
            has_personal_content = any(word in text_lower for word in personal_words)
            
            if has_people and has_personal_content:
                alignment_score += 0.2
            
            # Check for instructional content matching demonstration scene
            instruction_words = ['how', 'step', 'first', 'next', 'then', 'now']
            has_instruction = any(word in text_lower for word in instruction_words)
            
            if scene_type == 'demonstration' and has_instruction:
                alignment_score += 0.2
            
            # Penalize mismatches
            if scene_type == 'slides' and has_personal_content:
                alignment_score -= 0.1
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating audiovisual alignment: {e}")
            return 0.5
    
    def _generate_combined_recommendation(self, 
                                        segment_result: Dict, 
                                        segment_text: str, 
                                        alignment_score: float) -> str:
        """Generate recommendation based on combined visual and audio analysis."""
        
        visual_rec = segment_result.get('visual_recommendation', 'unknown')
        visual_interest = segment_result.get('visual_interest', 5)
        engagement_score = segment_result.get('engagement_score', 5)
        
        # Text quality indicators
        has_meaningful_text = len(segment_text.split()) > 5
        text_engagement = self._assess_text_engagement(segment_text)
        
        # Combined scoring
        visual_score = (visual_interest + engagement_score) / 2
        audio_score = text_engagement * 5  # Convert to 0-10 scale
        combined_score = (visual_score * 0.6 + audio_score * 0.4) * alignment_score
        
        # Generate recommendation
        if combined_score >= 8 and has_meaningful_text:
            return "highly_recommended"
        elif combined_score >= 6 and has_meaningful_text:
            return "recommended"
        elif combined_score >= 4 or (visual_score >= 6 and not has_meaningful_text):
            return "consider"
        else:
            return "not_recommended"
    
    def _assess_text_engagement(self, text: str) -> float:
        """Assess engagement potential of text content."""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        
        # Engagement indicators
        engagement_words = ['amazing', 'incredible', 'important', 'key', 'secret', 'wow']
        question_count = text.count('?')
        exclamation_count = text.count('!')
        direct_address = any(word in text_lower for word in ['you', 'your', 'we', 'let\'s'])
        
        # Calculate engagement score
        score = 0.0
        score += min(0.3, len([w for w in engagement_words if w in text_lower]) * 0.1)
        score += min(0.2, question_count * 0.1)
        score += min(0.1, exclamation_count * 0.05)
        score += 0.2 if direct_address else 0.0
        
        return min(1.0, score)
    
    def set_config(self, **kwargs):
        """Update processor configuration."""
        self.config.update(kwargs)
        self.logger.info(f"Updated vision processor config: {kwargs}")


# Add import at the top
import time
