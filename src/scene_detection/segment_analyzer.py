# src/scene_detection/segment_analyzer.py
"""Segment analysis for identifying optimal video segments"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class VideoSegment:
    """Represents a video segment with analysis data."""
    start_time: float
    end_time: float
    duration: float
    quality_score: float
    engagement_score: float
    content_type: str
    has_speech: bool
    has_faces: bool
    motion_level: str
    transcription: str = ""
    subjects: List[Dict] = None
    
    def __post_init__(self):
        if self.subjects is None:
            self.subjects = []


class SegmentAnalyzer:
    """
    Analyze video segments for quality, engagement, and suitability for shorts.
    """
    
    def __init__(self, 
                 min_segment_duration: float = 60.0,
                 max_segment_duration: float = 120.0,
                 quality_threshold: float = 0.6):
        """
        Initialize segment analyzer.
        
        Args:
            min_segment_duration: Minimum segment duration in seconds
            max_segment_duration: Maximum segment duration in seconds
            quality_threshold: Minimum quality score for segments
        """
        self.min_duration = min_segment_duration
        self.max_duration = max_segment_duration
        self.quality_threshold = quality_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Analysis weights for scoring
        self.scoring_weights = {
            'speech_quality': 0.25,
            'visual_quality': 0.25,
            'engagement_factors': 0.30,
            'technical_quality': 0.20
        }
        
        # Content type classifiers
        self.content_keywords = {
            'educational': ['learn', 'tutorial', 'how to', 'explain', 'understand'],
            'entertainment': ['funny', 'hilarious', 'amazing', 'incredible', 'wow'],
            'motivational': ['success', 'achieve', 'inspire', 'motivation', 'goals'],
            'informational': ['news', 'update', 'information', 'facts', 'data'],
            'conversational': ['discussion', 'talk', 'conversation', 'interview']
        }
    
    def analyze_segments(self, 
                        segments: List[Dict],
                        audio_analysis: Dict,
                        visual_analysis: Dict,
                        transcription: Dict) -> List[VideoSegment]:
        """
        Analyze segments and return scored video segments.
        
        Args:
            segments: List of segment dictionaries with start/end times
            audio_analysis: Audio analysis results
            visual_analysis: Visual analysis results
            transcription: Whisper transcription results
            
        Returns:
            List of analyzed video segments
        """
        analyzed_segments = []
        
        for segment_data in segments:
            try:
                segment = self._analyze_single_segment(
                    segment_data, audio_analysis, visual_analysis, transcription
                )
                
                if segment and segment.quality_score >= self.quality_threshold:
                    analyzed_segments.append(segment)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing segment {segment_data}: {e}")
                continue
        
        # Sort by engagement score
        analyzed_segments.sort(key=lambda x: x.engagement_score, reverse=True)
        
        self.logger.info(f"Analyzed {len(analyzed_segments)} quality segments from {len(segments)} total")
        
        return analyzed_segments
    
    def _analyze_single_segment(self,
                               segment_data: Dict,
                               audio_analysis: Dict,
                               visual_analysis: Dict,
                               transcription: Dict) -> Optional[VideoSegment]:
        """Analyze a single segment."""
        start_time = segment_data.get('start_time', 0.0)
        end_time = segment_data.get('end_time', 0.0)
        duration = end_time - start_time
        
        # Skip segments that are too short or too long
        if duration < self.min_duration or duration > self.max_duration:
            return None
        
        # Extract transcription for this segment
        segment_transcription = self._extract_segment_transcription(
            transcription, start_time, end_time
        )
        
        # Analyze speech content
        speech_analysis = self._analyze_speech_content(segment_transcription)
        
        # Analyze visual content
        visual_segment_analysis = self._analyze_visual_content(
            visual_analysis, start_time, end_time
        )
        
        # Calculate scores
        quality_score = self._calculate_quality_score(
            speech_analysis, visual_segment_analysis, duration
        )
        
        engagement_score = self._calculate_engagement_score(
            speech_analysis, visual_segment_analysis, segment_transcription
        )
        
        # Determine content type
        content_type = self._classify_content_type(segment_transcription)
        
        return VideoSegment(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            quality_score=quality_score,
            engagement_score=engagement_score,
            content_type=content_type,
            has_speech=len(segment_transcription.strip()) > 0,
            has_faces=visual_segment_analysis.get('has_faces', False),
            motion_level=visual_segment_analysis.get('motion_level', 'medium'),
            transcription=segment_transcription,
            subjects=visual_segment_analysis.get('subjects', [])
        )
    
    def _extract_segment_transcription(self,
                                     transcription: Dict,
                                     start_time: float,
                                     end_time: float) -> str:
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
    
    def _analyze_speech_content(self, transcription: str) -> Dict:
        """Analyze speech content for quality indicators."""
        analysis = {
            'word_count': 0,
            'sentence_count': 0,
            'avg_words_per_sentence': 0,
            'engagement_keywords': 0,
            'question_count': 0,
            'emotional_words': 0,
            'technical_terms': 0,
            'filler_words': 0,
            'speech_clarity_score': 0.0
        }
        
        if not transcription:
            return analysis
        
        words = transcription.lower().split()
        sentences = transcription.split('.')
        
        analysis['word_count'] = len(words)
        analysis['sentence_count'] = len([s for s in sentences if s.strip()])
        
        if analysis['sentence_count'] > 0:
            analysis['avg_words_per_sentence'] = analysis['word_count'] / analysis['sentence_count']
        
        # Count engagement indicators
        engagement_words = ['amazing', 'incredible', 'important', 'key', 'essential', 'crucial']
        analysis['engagement_keywords'] = sum(1 for word in words if word in engagement_words)
        
        # Count questions
        analysis['question_count'] = transcription.count('?')
        
        # Count emotional words
        emotional_words = ['love', 'hate', 'excited', 'amazing', 'terrible', 'wonderful']
        analysis['emotional_words'] = sum(1 for word in words if word in emotional_words)
        
        # Count filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        analysis['filler_words'] = sum(1 for word in words if word in filler_words)
        
        # Calculate speech clarity (fewer fillers = higher clarity)
        if analysis['word_count'] > 0:
            analysis['speech_clarity_score'] = 1.0 - (analysis['filler_words'] / analysis['word_count'])
        
        return analysis
    
    def _analyze_visual_content(self,
                               visual_analysis: Dict,
                               start_time: float,
                               end_time: float) -> Dict:
        """Analyze visual content for the segment."""
        analysis = {
            'has_faces': False,
            'face_count': 0,
            'subjects': [],
            'motion_level': 'low',
            'scene_changes': 0,
            'visual_quality_score': 0.0,
            'composition_score': 0.0
        }
        
        # Extract visual data for this time segment
        if 'face_detections' in visual_analysis:
            faces_in_segment = [
                face for face in visual_analysis['face_detections']
                if start_time <= face.get('timestamp', 0) <= end_time
            ]
            
            analysis['has_faces'] = len(faces_in_segment) > 0
            analysis['face_count'] = len(set(face.get('face_id', 0) for face in faces_in_segment))
        
        # Analyze motion
        if 'motion_analysis' in visual_analysis:
            motion_data = visual_analysis['motion_analysis']
            segment_motion = [
                motion for motion in motion_data
                if start_time <= motion.get('timestamp', 0) <= end_time
            ]
            
            if segment_motion:
                avg_motion = sum(motion.get('motion_score', 0) for motion in segment_motion) / len(segment_motion)
                if avg_motion > 0.7:
                    analysis['motion_level'] = 'high'
                elif avg_motion > 0.3:
                    analysis['motion_level'] = 'medium'
                else:
                    analysis['motion_level'] = 'low'
        
        # Analyze scene changes
        if 'scene_changes' in visual_analysis:
            scene_changes_in_segment = [
                change for change in visual_analysis['scene_changes']
                if start_time <= change.get('timestamp', 0) <= end_time
            ]
            analysis['scene_changes'] = len(scene_changes_in_segment)
        
        # Calculate visual quality score
        analysis['visual_quality_score'] = self._calculate_visual_quality(analysis)
        
        return analysis
    
    def _calculate_visual_quality(self, visual_analysis: Dict) -> float:
        """Calculate visual quality score."""
        score = 0.0
        
        # Face presence bonus
        if visual_analysis['has_faces']:
            score += 0.3
        
        # Motion level scoring
        motion_scores = {'low': 0.2, 'medium': 0.3, 'high': 0.25}
        score += motion_scores.get(visual_analysis['motion_level'], 0.2)
        
        # Scene stability (fewer changes = better for shorts)
        if visual_analysis['scene_changes'] <= 2:
            score += 0.2
        elif visual_analysis['scene_changes'] <= 5:
            score += 0.1
        
        # Multiple subjects can be good for engagement
        if visual_analysis['face_count'] >= 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_quality_score(self,
                               speech_analysis: Dict,
                               visual_analysis: Dict,
                               duration: float) -> float:
        """Calculate overall quality score for the segment."""
        scores = {}
        
        # Speech quality component
        speech_score = 0.0
        if speech_analysis['word_count'] > 0:
            # Good speech clarity
            speech_score += speech_analysis['speech_clarity_score'] * 0.4
            
            # Appropriate length (words per minute)
            wpm = speech_analysis['word_count'] / (duration / 60)
            if 120 <= wpm <= 180:  # Ideal speaking rate
                speech_score += 0.3
            elif 100 <= wpm <= 200:
                speech_score += 0.2
            else:
                speech_score += 0.1
            
            # Engagement indicators
            if speech_analysis['engagement_keywords'] > 0:
                speech_score += 0.2
            
            if speech_analysis['question_count'] > 0:
                speech_score += 0.1
        
        scores['speech_quality'] = speech_score
        
        # Visual quality component
        scores['visual_quality'] = visual_analysis['visual_quality_score']
        
        # Engagement factors
        engagement_score = 0.0
        if speech_analysis['emotional_words'] > 0:
            engagement_score += 0.3
        
        if visual_analysis['has_faces']:
            engagement_score += 0.3
        
        if visual_analysis['motion_level'] in ['medium', 'high']:
            engagement_score += 0.2
        
        if speech_analysis['question_count'] > 0:
            engagement_score += 0.2
        
        scores['engagement_factors'] = engagement_score
        
        # Technical quality
        technical_score = 0.5  # Base score
        
        # Duration appropriateness
        if 20 <= duration <= 45:  # Ideal duration for shorts
            technical_score += 0.3
        elif 15 <= duration <= 60:
            technical_score += 0.2
        
        # Speech presence
        if speech_analysis['word_count'] > 10:
            technical_score += 0.2
        
        scores['technical_quality'] = technical_score
        
        # Calculate weighted final score
        final_score = sum(
            scores[component] * self.scoring_weights[component]
            for component in self.scoring_weights
        )
        
        return min(1.0, final_score)
    
    def _calculate_engagement_score(self,
                                  speech_analysis: Dict,
                                  visual_analysis: Dict,
                                  transcription: str) -> float:
        """Calculate engagement potential score."""
        engagement_score = 0.0
        
        # High-energy words and phrases
        high_energy_words = ['amazing', 'incredible', 'unbelievable', 'wow', 'shocking']
        energy_count = sum(1 for word in high_energy_words if word in transcription.lower())
        engagement_score += min(0.3, energy_count * 0.1)
        
        # Questions engage viewers
        engagement_score += min(0.2, speech_analysis['question_count'] * 0.1)
        
        # Emotional content
        engagement_score += min(0.2, speech_analysis['emotional_words'] * 0.05)
        
        # Visual engagement
        if visual_analysis['has_faces']:
            engagement_score += 0.15
        
        if visual_analysis['motion_level'] == 'high':
            engagement_score += 0.1
        elif visual_analysis['motion_level'] == 'medium':
            engagement_score += 0.05
        
        # Multiple people can be more engaging
        if visual_analysis['face_count'] >= 2:
            engagement_score += 0.1
        
        # Key phrases that typically perform well
        viral_phrases = ['you won\'t believe', 'this is crazy', 'wait for it', 'watch this']
        for phrase in viral_phrases:
            if phrase in transcription.lower():
                engagement_score += 0.15
                break
        
        return min(1.0, engagement_score)
    
    def _classify_content_type(self, transcription: str) -> str:
        """Classify content type based on transcription."""
        if not transcription:
            return 'unknown'
        
        text_lower = transcription.lower()
        
        # Count keywords for each category
        category_scores = {}
        for category, keywords in self.content_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return 'general'
    
    def filter_segments_by_criteria(self, 
                                  segments: List[VideoSegment],
                                  max_segments: int = 10,
                                  prefer_content_types: List[str] = None) -> List[VideoSegment]:
        """
        Filter and prioritize segments based on criteria.
        
        Args:
            segments: List of analyzed segments
            max_segments: Maximum number of segments to return
            prefer_content_types: Preferred content types to prioritize
            
        Returns:
            Filtered list of segments
        """
        filtered_segments = segments.copy()
        
        # Prioritize preferred content types
        if prefer_content_types:
            def priority_score(segment):
                base_score = segment.engagement_score
                if segment.content_type in prefer_content_types:
                    base_score += 0.2
                return base_score
            
            filtered_segments.sort(key=priority_score, reverse=True)
        
        # Remove overlapping segments (keep highest scoring)
        non_overlapping = []
        for segment in filtered_segments:
            overlaps = False
            for existing in non_overlapping:
                if self._segments_overlap(segment, existing):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(segment)
                
                if len(non_overlapping) >= max_segments:
                    break
        
        self.logger.info(f"Filtered to {len(non_overlapping)} non-overlapping segments")
        
        return non_overlapping
    
    def _segments_overlap(self, seg1: VideoSegment, seg2: VideoSegment) -> bool:
        """Check if two segments overlap significantly."""
        # Calculate overlap
        overlap_start = max(seg1.start_time, seg2.start_time)
        overlap_end = min(seg1.end_time, seg2.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)
        
        # Consider significant if overlap is more than 30% of either segment
        threshold = 0.3
        overlap_ratio_1 = overlap_duration / seg1.duration
        overlap_ratio_2 = overlap_duration / seg2.duration
        
        return overlap_ratio_1 > threshold or overlap_ratio_2 > threshold
    
    def generate_segment_report(self, segments: List[VideoSegment]) -> Dict:
        """Generate analysis report for segments."""
        if not segments:
            return {'error': 'No segments to analyze'}
        
        # Calculate statistics
        total_duration = sum(seg.duration for seg in segments)
        avg_quality = sum(seg.quality_score for seg in segments) / len(segments)
        avg_engagement = sum(seg.engagement_score for seg in segments) / len(segments)
        
        # Content type distribution
        content_types = {}
        for segment in segments:
            content_types[segment.content_type] = content_types.get(segment.content_type, 0) + 1
        
        # Quality distribution
        quality_ranges = {'high': 0, 'medium': 0, 'low': 0}
        for segment in segments:
            if segment.quality_score >= 0.8:
                quality_ranges['high'] += 1
            elif segment.quality_score >= 0.6:
                quality_ranges['medium'] += 1
            else:
                quality_ranges['low'] += 1
        
        return {
            'summary': {
                'total_segments': len(segments),
                'total_duration': total_duration,
                'average_duration': total_duration / len(segments),
                'average_quality_score': avg_quality,
                'average_engagement_score': avg_engagement
            },
            'content_distribution': content_types,
            'quality_distribution': quality_ranges,
            'top_segments': [
                {
                    'start_time': seg.start_time,
                    'end_time': seg.end_time,
                    'duration': seg.duration,
                    'quality_score': seg.quality_score,
                    'engagement_score': seg.engagement_score,
                    'content_type': seg.content_type,
                    'preview_text': seg.transcription[:100] + '...' if len(seg.transcription) > 100 else seg.transcription
                }
                for seg in segments[:5]  # Top 5 segments
            ]
        }
