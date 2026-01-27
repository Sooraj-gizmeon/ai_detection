# src/content_analysis/segment_optimizer.py
"""Segment optimization for creating the best possible short videos"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path
import json
import math


@dataclass
class OptimizedSegment:
    """Represents an optimized video segment."""
    start_time: float
    end_time: float
    duration: float
    optimization_score: float
    content_highlights: List[str]
    recommended_title: str
    tags: List[str]
    target_audience: str
    viral_potential: float
    technical_notes: Dict
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'optimization_score': self.optimization_score,
            'content_highlights': self.content_highlights,
            'recommended_title': self.recommended_title,
            'tags': self.tags,
            'target_audience': self.target_audience,
            'viral_potential': self.viral_potential,
            'technical_notes': self.technical_notes
        }


class SegmentOptimizer:
    """
    Optimize video segments for maximum engagement and viral potential.
    """
    
    def __init__(self):
        """Initialize segment optimizer."""
        self.logger = logging.getLogger(__name__)
        
        # Optimization parameters
        self.viral_keywords = [
            'amazing', 'incredible', 'unbelievable', 'shocking', 'mind-blowing',
            'secret', 'trick', 'hack', 'tip', 'mistake', 'fail', 'win',
            'before', 'after', 'transformation', 'reveal', 'exposed'
        ]
        
        self.engagement_hooks = [
            'you won\'t believe', 'this will shock you', 'wait for it',
            'watch this', 'this is crazy', 'you need to see this',
            'here\'s why', 'the truth about', 'what happens next'
        ]
        
        self.target_audiences = {
            'general': {'min_age': 13, 'max_age': 65, 'interests': ['entertainment']},
            'young_adults': {'min_age': 18, 'max_age': 35, 'interests': ['lifestyle', 'tech', 'trends']},
            'professionals': {'min_age': 25, 'max_age': 55, 'interests': ['business', 'education', 'productivity']},
            'creators': {'min_age': 16, 'max_age': 40, 'interests': ['tutorials', 'tips', 'behind-scenes']},
            'students': {'min_age': 16, 'max_age': 25, 'interests': ['education', 'study tips', 'career']}
        }
        
        # Content categories for better optimization
        self.content_categories = {
            'educational': {
                'optimal_duration': (30, 60),
                'key_elements': ['clear explanation', 'visual aids', 'step-by-step'],
                'title_patterns': ['How to {}', 'Learn {} in {} minutes', '{} explained']
            },
            'entertainment': {
                'optimal_duration': (15, 30),
                'key_elements': ['humor', 'surprise', 'relatability'],
                'title_patterns': ['This is hilarious', 'You have to see this', 'When {}']
            },
            'motivational': {
                'optimal_duration': (20, 45),
                'key_elements': ['inspiration', 'success story', 'actionable advice'],
                'title_patterns': ['How {} changed my life', 'The secret to {}', 'Why you should {}']
            },
            'lifestyle': {
                'optimal_duration': (25, 50),
                'key_elements': ['personal experience', 'tips', 'relatable content'],
                'title_patterns': ['My {} routine', 'Why I {}', 'Things I wish I knew about {}']
            }
        }
    
    def optimize_segments(self, 
                         segments: List[Dict],
                         transcription: Dict,
                         audio_analysis: Dict,
                         visual_analysis: Dict,
                         ai_analysis: Dict = None) -> List[OptimizedSegment]:
        """
        Optimize segments for maximum engagement.
        
        Args:
            segments: List of analyzed segments
            transcription: Whisper transcription results
            audio_analysis: Audio analysis results
            visual_analysis: Visual analysis results
            ai_analysis: AI analysis results from Ollama
            
        Returns:
            List of optimized segments
        """
        optimized_segments = []
        
        for segment in segments:
            try:
                optimized = self._optimize_single_segment(
                    segment, transcription, audio_analysis, visual_analysis, ai_analysis
                )
                
                if optimized:
                    optimized_segments.append(optimized)
                    
            except Exception as e:
                self.logger.error(f"Error optimizing segment: {e}")
                continue
        
        # Sort by optimization score
        optimized_segments.sort(key=lambda x: x.optimization_score, reverse=True)
        
        self.logger.info(f"Optimized {len(optimized_segments)} segments")
        
        return optimized_segments
    
    def _optimize_single_segment(self,
                                segment: Dict,
                                transcription: Dict,
                                audio_analysis: Dict,
                                visual_analysis: Dict,
                                ai_analysis: Dict = None) -> Optional[OptimizedSegment]:
        """Optimize a single segment."""
        start_time = segment.get('start_time', 0.0)
        end_time = segment.get('end_time', 0.0)
        duration = end_time - start_time
        
        # Extract segment transcription
        segment_text = self._extract_segment_text(transcription, start_time, end_time)
        
        # Analyze content
        content_analysis = self._analyze_content_for_optimization(segment_text)
        
        # Calculate viral potential
        viral_potential = self._calculate_viral_potential(
            segment_text, content_analysis, visual_analysis
        )
        
        # Generate content highlights
        highlights = self._extract_content_highlights(segment_text, content_analysis)
        
        # Generate recommended title
        title = self._generate_optimized_title(segment_text, content_analysis)
        
        # Generate tags
        tags = self._generate_tags(segment_text, content_analysis)
        
        # Determine target audience
        target_audience = self._determine_target_audience(segment_text, content_analysis)
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            content_analysis, viral_potential, duration, segment.get('quality_score', 0.5)
        )
        
        # Generate technical notes
        technical_notes = self._generate_technical_notes(
            segment, content_analysis, visual_analysis
        )
        
        return OptimizedSegment(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            optimization_score=optimization_score,
            content_highlights=highlights,
            recommended_title=title,
            tags=tags,
            target_audience=target_audience,
            viral_potential=viral_potential,
            technical_notes=technical_notes
        )
    
    def _extract_segment_text(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """Extract transcription text for a segment."""
        if not transcription or 'segments' not in transcription:
            return ""
        
        segment_text = []
        
        for segment in transcription['segments']:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            if seg_start < end_time and seg_end > start_time:
                text = segment.get('text', '').strip()
                if text:
                    segment_text.append(text)
        
        return ' '.join(segment_text)
    
    def _analyze_content_for_optimization(self, text: str) -> Dict:
        """Analyze content for optimization opportunities."""
        analysis = {
            'word_count': 0,
            'viral_keywords_count': 0,
            'engagement_hooks_count': 0,
            'questions_count': 0,
            'action_words_count': 0,
            'emotional_intensity': 0.0,
            'content_category': 'general',
            'readability_score': 0.0,
            'urgency_indicators': 0,
            'social_proof_indicators': 0
        }
        
        if not text:
            return analysis
        
        text_lower = text.lower()
        words = text_lower.split()
        
        analysis['word_count'] = len(words)
        analysis['questions_count'] = text.count('?')
        
        # Count viral keywords
        analysis['viral_keywords_count'] = sum(
            1 for keyword in self.viral_keywords if keyword in text_lower
        )
        
        # Count engagement hooks
        analysis['engagement_hooks_count'] = sum(
            1 for hook in self.engagement_hooks if hook in text_lower
        )
        
        # Count action words
        action_words = ['discover', 'learn', 'find', 'get', 'achieve', 'master', 'unlock']
        analysis['action_words_count'] = sum(
            1 for word in action_words if word in text_lower
        )
        
        # Analyze emotional intensity
        emotional_words = {
            'positive': ['amazing', 'incredible', 'awesome', 'fantastic', 'brilliant'],
            'negative': ['terrible', 'awful', 'horrible', 'disaster', 'worst'],
            'surprise': ['shocking', 'unbelievable', 'unexpected', 'surprising'],
            'urgency': ['now', 'immediately', 'urgent', 'quick', 'fast']
        }
        
        emotional_score = 0
        for category, word_list in emotional_words.items():
            count = sum(1 for word in word_list if word in text_lower)
            emotional_score += count * 0.2
        
        analysis['emotional_intensity'] = min(1.0, emotional_score)
        
        # Determine content category
        analysis['content_category'] = self._classify_content_category(text_lower)
        
        # Count urgency indicators
        urgency_words = ['now', 'today', 'immediately', 'urgent', 'limited time', 'hurry']
        analysis['urgency_indicators'] = sum(
            1 for word in urgency_words if word in text_lower
        )
        
        # Count social proof indicators
        social_proof_words = ['everyone', 'millions', 'thousands', 'proven', 'tested', 'verified']
        analysis['social_proof_indicators'] = sum(
            1 for word in social_proof_words if word in text_lower
        )
        
        return analysis
    
    def _classify_content_category(self, text: str) -> str:
        """Classify content into categories."""
        category_keywords = {
            'educational': ['learn', 'tutorial', 'how to', 'explain', 'understand', 'guide'],
            'entertainment': ['funny', 'hilarious', 'comedy', 'joke', 'laugh'],
            'motivational': ['success', 'achieve', 'inspire', 'motivation', 'goals', 'dream'],
            'lifestyle': ['daily', 'routine', 'life', 'personal', 'experience', 'tips'],
            'tech': ['technology', 'app', 'software', 'digital', 'online', 'internet'],
            'business': ['money', 'business', 'entrepreneur', 'profit', 'sales', 'marketing']
        }
        
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            if category_scores[best_category] > 0:
                return best_category
        
        return 'general'
    
    def _calculate_viral_potential(self, 
                                 text: str, 
                                 content_analysis: Dict, 
                                 visual_analysis: Dict) -> float:
        """Calculate viral potential score."""
        viral_score = 0.0
        
        # Content factors
        viral_score += min(0.3, content_analysis['viral_keywords_count'] * 0.1)
        viral_score += min(0.2, content_analysis['engagement_hooks_count'] * 0.1)
        viral_score += min(0.15, content_analysis['emotional_intensity'] * 0.15)
        viral_score += min(0.1, content_analysis['questions_count'] * 0.05)
        viral_score += min(0.1, content_analysis['urgency_indicators'] * 0.05)
        viral_score += min(0.1, content_analysis['social_proof_indicators'] * 0.05)
        
        # Visual factors
        visual_factors = visual_analysis.get('visual_engagement', {})
        if visual_factors.get('has_faces', False):
            viral_score += 0.05
        
        if visual_factors.get('motion_level', 'low') == 'high':
            viral_score += 0.05
        
        return min(1.0, viral_score)
    
    def _extract_content_highlights(self, text: str, content_analysis: Dict) -> List[str]:
        """Extract key content highlights."""
        if not text:
            return []
        
        highlights = []
        sentences = text.split('.')
        
        # Find sentences with viral keywords
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length
                for keyword in self.viral_keywords:
                    if keyword in sentence.lower():
                        highlights.append(sentence)
                        break
        
        # Find sentences with questions
        question_sentences = [s.strip() for s in sentences if '?' in s and len(s.strip()) > 10]
        highlights.extend(question_sentences[:2])  # Max 2 questions
        
        # Find sentences with emotional words
        emotional_sentences = []
        emotional_words = ['amazing', 'incredible', 'shocking', 'unbelievable']
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                for word in emotional_words:
                    if word in sentence.lower():
                        emotional_sentences.append(sentence)
                        break
        
        highlights.extend(emotional_sentences[:2])
        
        # Remove duplicates and limit to top 5
        unique_highlights = list(dict.fromkeys(highlights))  # Preserve order
        
        return unique_highlights[:5]
    
    def _generate_optimized_title(self, text: str, content_analysis: Dict) -> str:
        """Generate an optimized title for the segment."""
        category = content_analysis['content_category']
        
        # Get title patterns for this category
        patterns = self.content_categories.get(category, {}).get('title_patterns', [])
        
        # Extract key phrases from text
        key_phrases = self._extract_key_phrases(text)
        
        # Generate title based on content
        if content_analysis['viral_keywords_count'] > 0:
            # Use viral approach
            viral_starters = [
                "This Will Blow Your Mind:",
                "You Won't Believe This:",
                "This Is Incredible:",
                "Watch This Amazing"
            ]
            if key_phrases:
                return f"{viral_starters[0]} {key_phrases[0]}"
        
        if content_analysis['questions_count'] > 0:
            # Question-based title
            questions = [s.strip() for s in text.split('.') if '?' in s]
            if questions:
                return questions[0][:50] + ("..." if len(questions[0]) > 50 else "")
        
        # Category-specific titles
        if patterns and key_phrases:
            pattern = patterns[0]
            if '{}' in pattern:
                return pattern.format(key_phrases[0])
            else:
                return f"{pattern} {key_phrases[0]}"
        
        # Fallback to key phrase
        if key_phrases:
            return key_phrases[0][:50] + ("..." if len(key_phrases[0]) > 50 else "")
        
        # Ultimate fallback
        return "Amazing Short Video"
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        if not text:
            return []
        
        # Split into sentences and find important ones
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        
        # Score sentences based on importance indicators
        scored_sentences = []
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Boost score for viral keywords
            for keyword in self.viral_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Boost score for action words
            action_words = ['learn', 'discover', 'find', 'get', 'achieve']
            for word in action_words:
                if word in sentence_lower:
                    score += 1
            
            scored_sentences.append((sentence, score))
        
        # Sort by score and return top phrases
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        return [sentence for sentence, score in scored_sentences[:3] if score > 0]
    
    def _generate_tags(self, text: str, content_analysis: Dict) -> List[str]:
        """Generate relevant tags for the segment."""
        tags = []
        
        # Category-based tags
        category = content_analysis['content_category']
        category_tags = {
            'educational': ['learn', 'tutorial', 'education', 'howto', 'tips'],
            'entertainment': ['funny', 'comedy', 'entertainment', 'viral', 'trending'],
            'motivational': ['motivation', 'inspiration', 'success', 'mindset', 'goals'],
            'lifestyle': ['lifestyle', 'daily', 'personal', 'life', 'routine'],
            'tech': ['tech', 'technology', 'digital', 'innovation', 'gadgets'],
            'business': ['business', 'entrepreneur', 'money', 'success', 'marketing']
        }
        
        tags.extend(category_tags.get(category, ['general', 'content']))
        
        # Add viral tags if applicable
        if content_analysis['viral_keywords_count'] > 0:
            tags.extend(['viral', 'trending', 'amazing', 'mustwatch'])
        
        # Add engagement tags
        if content_analysis['questions_count'] > 0:
            tags.append('interactive')
        
        if content_analysis['emotional_intensity'] > 0.5:
            tags.append('emotional')
        
        # Universal short-form tags
        tags.extend(['shorts', 'short', 'quickwatch', 'vertical'])
        
        # Remove duplicates and limit
        unique_tags = list(dict.fromkeys(tags))
        
        return unique_tags[:10]
    
    def _determine_target_audience(self, text: str, content_analysis: Dict) -> str:
        """Determine the target audience for the content."""
        category = content_analysis['content_category']
        
        # Category-based audience mapping
        audience_mapping = {
            'educational': 'students',
            'business': 'professionals',
            'lifestyle': 'young_adults',
            'tech': 'young_adults',
            'entertainment': 'general'
        }
        
        # Check for age-specific indicators
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['college', 'university', 'student', 'study']):
            return 'students'
        
        if any(word in text_lower for word in ['business', 'professional', 'career', 'work']):
            return 'professionals'
        
        if any(word in text_lower for word in ['creator', 'content', 'youtube', 'tiktok']):
            return 'creators'
        
        # Default to category mapping
        return audience_mapping.get(category, 'general')
    
    def _calculate_optimization_score(self, 
                                    content_analysis: Dict, 
                                    viral_potential: float, 
                                    duration: float, 
                                    base_quality: float) -> float:
        """Calculate overall optimization score."""
        score_components = []
        
        # Base quality (40% weight)
        score_components.append(base_quality * 0.4)
        
        # Viral potential (30% weight)
        score_components.append(viral_potential * 0.3)
        
        # Content optimization (20% weight)
        content_score = 0.0
        content_score += min(0.3, content_analysis['viral_keywords_count'] * 0.1)
        content_score += min(0.3, content_analysis['engagement_hooks_count'] * 0.15)
        content_score += min(0.2, content_analysis['emotional_intensity'])
        content_score += min(0.2, content_analysis['action_words_count'] * 0.1)
        
        score_components.append(content_score * 0.2)
        
        # Duration optimization (10% weight)
        duration_score = 0.0
        if 15 <= duration <= 30:  # Optimal for shorts
            duration_score = 1.0
        elif 30 <= duration <= 45:
            duration_score = 0.8
        elif 45 <= duration <= 60:
            duration_score = 0.6
        else:
            duration_score = 0.3
        
        score_components.append(duration_score * 0.1)
        
        return sum(score_components)
    
    def _generate_technical_notes(self, 
                                segment: Dict, 
                                content_analysis: Dict, 
                                visual_analysis: Dict) -> Dict:
        """Generate technical optimization notes."""
        notes = {
            'recommended_edits': [],
            'audio_recommendations': [],
            'visual_recommendations': [],
            'pacing_notes': [],
            'engagement_tips': []
        }
        
        # Editing recommendations
        if content_analysis['emotional_intensity'] > 0.7:
            notes['recommended_edits'].append('Add dynamic cuts during emotional peaks')
        
        if content_analysis['questions_count'] > 0:
            notes['recommended_edits'].append('Pause briefly after questions for impact')
        
        # Audio recommendations
        notes['audio_recommendations'].append('Ensure clear speech throughout')
        
        if content_analysis['urgency_indicators'] > 0:
            notes['audio_recommendations'].append('Consider faster pacing for urgency')
        
        # Visual recommendations
        visual_data = visual_analysis.get('visual_engagement', {})
        
        if visual_data.get('has_faces', False):
            notes['visual_recommendations'].append('Ensure faces are clearly visible')
        
        if visual_data.get('motion_level', 'low') == 'low':
            notes['visual_recommendations'].append('Consider adding motion or zoom effects')
        
        # Pacing notes
        duration = segment.get('duration', 0)
        if duration > 45:
            notes['pacing_notes'].append('Consider faster pacing or cutting content')
        elif duration < 20:
            notes['pacing_notes'].append('Could expand with additional context')
        
        # Engagement tips
        if content_analysis['viral_keywords_count'] == 0:
            notes['engagement_tips'].append('Consider adding hook words or phrases')
        
        if content_analysis['questions_count'] == 0:
            notes['engagement_tips'].append('Add rhetorical questions to engage viewers')
        
        return notes
    
    def rank_segments_for_platform(self, 
                                  segments: List[OptimizedSegment], 
                                  platform: str = 'general') -> List[OptimizedSegment]:
        """
        Rank segments optimized for specific platforms.
        
        Args:
            segments: List of optimized segments
            platform: Target platform ('tiktok', 'youtube_shorts', 'instagram', 'general')
            
        Returns:
            Reranked segments
        """
        platform_preferences = {
            'tiktok': {
                'preferred_duration': (15, 30),
                'content_preference': ['entertainment', 'lifestyle', 'educational'],
                'viral_weight': 0.4,
                'engagement_weight': 0.6
            },
            'youtube_shorts': {
                'preferred_duration': (20, 60),
                'content_preference': ['educational', 'entertainment', 'tech'],
                'viral_weight': 0.3,
                'engagement_weight': 0.7
            },
            'instagram': {
                'preferred_duration': (15, 45),
                'content_preference': ['lifestyle', 'entertainment', 'motivational'],
                'viral_weight': 0.35,
                'engagement_weight': 0.65
            }
        }
        
        preferences = platform_preferences.get(platform, platform_preferences['youtube_shorts'])
        
        # Calculate platform-specific scores
        for segment in segments:
            platform_score = segment.optimization_score
            
            # Duration preference adjustment
            pref_min, pref_max = preferences['preferred_duration']
            if pref_min <= segment.duration <= pref_max:
                platform_score += 0.1
            
            # Content preference adjustment
            if segment.target_audience in preferences['content_preference']:
                platform_score += 0.05
            
            # Adjust viral vs engagement weight
            viral_weight = preferences['viral_weight']
            engagement_weight = preferences['engagement_weight']
            
            weighted_score = (
                segment.viral_potential * viral_weight +
                segment.optimization_score * engagement_weight
            )
            
            # Store platform score in technical notes
            segment.technical_notes['platform_score'] = weighted_score
        
        # Sort by platform score
        segments.sort(key=lambda x: x.technical_notes.get('platform_score', x.optimization_score), reverse=True)
        
        return segments
