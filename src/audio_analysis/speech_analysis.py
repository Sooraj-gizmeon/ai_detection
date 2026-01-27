# src/audio_analysis/speech_analysis.py
"""Speech analysis and processing utilities"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json


class SpeechAnalyzer:
    """
    Analyzes speech patterns and characteristics for video processing.
    """
    
    def __init__(self):
        """Initialize SpeechAnalyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Speech pattern keywords for different zoom strategies
        self.zoom_keywords = {
            'close_up': [
                'important', 'key', 'focus', 'listen', 'crucial', 'essential',
                'remember', 'note', 'pay attention', 'highlight', 'emphasis',
                'secret', 'confidential', 'personal', 'intimate'
            ],
            'wide_shot': [
                'everyone', 'all', 'group', 'together', 'audience', 'crowd',
                'team', 'community', 'collective', 'whole', 'entire',
                'overview', 'general', 'broad', 'context'
            ],
            'medium_shot': [
                'explain', 'describe', 'discuss', 'talk about', 'conversation',
                'dialogue', 'presentation', 'interview', 'normal', 'regular'
            ],
            'dynamic': [
                'move', 'show', 'demonstrate', 'look', 'point', 'gesture',
                'action', 'activity', 'performance', 'dance', 'exercise'
            ]
        }
        
        # Emotional intensity keywords
        self.emotional_keywords = {
            'high_intensity': [
                'amazing', 'incredible', 'wow', 'fantastic', 'awesome',
                'terrible', 'horrible', 'disaster', 'crisis', 'emergency',
                'excited', 'thrilled', 'shocked', 'surprised', 'angry',
                'furious', 'love', 'hate', 'breakthrough', 'revolutionary'
            ],
            'medium_intensity': [
                'good', 'bad', 'nice', 'interesting', 'cool', 'strange',
                'weird', 'happy', 'sad', 'concerned', 'worried', 'pleased',
                'disappointed', 'satisfied', 'frustrated'
            ],
            'low_intensity': [
                'okay', 'fine', 'normal', 'regular', 'standard', 'typical',
                'usual', 'common', 'average', 'moderate', 'calm', 'peaceful'
            ]
        }
    
    def analyze_speech_patterns(self, transcription: Dict) -> Dict:
        """
        Analyze speech patterns from Whisper transcription.
        
        Args:
            transcription: Whisper transcription result
            
        Returns:
            Speech pattern analysis
        """
        analysis = {
            'segments': [],
            'speaking_rate': [],
            'pause_patterns': [],
            'emotional_intensity': [],
            'zoom_recommendations': [],
            'speaker_changes': [],
            'confidence_scores': []
        }
        
        segments = transcription.get('segments', [])
        
        for i, segment in enumerate(segments):
            segment_analysis = self._analyze_segment(segment)
            
            # Add segment metadata
            segment_analysis.update({
                'segment_id': i,
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 0),
                'duration': segment.get('end', 0) - segment.get('start', 0)
            })
            
            analysis['segments'].append(segment_analysis)
            analysis['speaking_rate'].append(segment_analysis['speaking_rate'])
            analysis['emotional_intensity'].append(segment_analysis['emotional_intensity'])
            analysis['zoom_recommendations'].append(segment_analysis['zoom_strategy'])
            analysis['confidence_scores'].append(segment.get('avg_logprob', 0))
        
        # Analyze pause patterns
        analysis['pause_patterns'] = self._analyze_pause_patterns(segments)
        
        # Detect speaker changes (basic heuristic)
        analysis['speaker_changes'] = self._detect_speaker_changes(segments)
        
        # Overall statistics
        analysis['statistics'] = self._calculate_speech_statistics(analysis)
        
        return analysis
    
    def _analyze_segment(self, segment: Dict) -> Dict:
        """
        Analyze individual speech segment.
        
        Args:
            segment: Individual transcription segment
            
        Returns:
            Segment analysis
        """
        text = segment.get('text', '').lower()
        words = text.split()
        duration = segment.get('end', 0) - segment.get('start', 0)
        
        analysis = {
            'text': segment.get('text', ''),
            'word_count': len(words),
            'speaking_rate': len(words) / (duration + 0.001),  # words per second
            'emotional_intensity': self._calculate_emotional_intensity(text),
            'zoom_strategy': self._determine_zoom_strategy(text),
            'content_type': self._classify_content_type(text),
            'emphasis_level': self._calculate_emphasis_level(text),
            'action_indicators': self._detect_action_indicators(text)
        }
        
        return analysis
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """
        Calculate emotional intensity of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emotional intensity score (0.0 to 1.0)
        """
        high_count = sum(1 for keyword in self.emotional_keywords['high_intensity'] 
                        if keyword in text)
        medium_count = sum(1 for keyword in self.emotional_keywords['medium_intensity'] 
                          if keyword in text)
        low_count = sum(1 for keyword in self.emotional_keywords['low_intensity'] 
                       if keyword in text)
        
        # Weight the counts
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        intensity_score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.1) / total_words
        
        # Check for punctuation that indicates intensity
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
        
        punctuation_boost = (exclamation_count * 0.2 + question_count * 0.1 + caps_words * 0.1) / total_words
        
        return min(intensity_score + punctuation_boost, 1.0)
    
    def _determine_zoom_strategy(self, text: str) -> str:
        """
        Determine optimal zoom strategy based on text content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Recommended zoom strategy
        """
        scores = {}
        
        for strategy, keywords in self.zoom_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[strategy] = score
        
        # Return strategy with highest score, default to medium_shot
        if max(scores.values()) == 0:
            return 'medium_shot'
        
        return max(scores, key=scores.get)
    
    def _classify_content_type(self, text: str) -> str:
        """
        Classify content type based on text patterns.
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type classification
        """
        # Question patterns
        if '?' in text or any(word in text for word in ['what', 'why', 'how', 'when', 'where', 'who']):
            return 'question'
        
        # Instruction patterns
        if any(word in text for word in ['let', 'make', 'do', 'try', 'follow', 'step']):
            return 'instruction'
        
        # Explanation patterns
        if any(word in text for word in ['because', 'since', 'therefore', 'so', 'explain', 'reason']):
            return 'explanation'
        
        # Story patterns
        if any(word in text for word in ['once', 'then', 'after', 'before', 'story', 'happened']):
            return 'narrative'
        
        # Opinion patterns
        if any(word in text for word in ['think', 'believe', 'feel', 'opinion', 'personally']):
            return 'opinion'
        
        return 'general'
    
    def _calculate_emphasis_level(self, text: str) -> float:
        """
        Calculate emphasis level based on text characteristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Emphasis level (0.0 to 1.0)
        """
        emphasis_indicators = [
            'very', 'really', 'extremely', 'absolutely', 'definitely',
            'certainly', 'surely', 'indeed', 'particularly', 'especially'
        ]
        
        words = text.split()
        if not words:
            return 0.0
        
        emphasis_count = sum(1 for word in words if word in emphasis_indicators)
        caps_ratio = sum(1 for word in words if word.isupper()) / len(words)
        punctuation_intensity = (text.count('!') + text.count('?')) / len(text)
        
        emphasis_score = (emphasis_count / len(words)) + caps_ratio + punctuation_intensity
        
        return min(emphasis_score, 1.0)
    
    def _detect_action_indicators(self, text: str) -> List[str]:
        """
        Detect action indicators in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected action indicators
        """
        action_verbs = [
            'move', 'go', 'come', 'run', 'walk', 'jump', 'dance', 'perform',
            'show', 'point', 'gesture', 'demonstrate', 'present', 'display',
            'look', 'see', 'watch', 'observe', 'examine', 'check',
            'turn', 'rotate', 'spin', 'flip', 'slide', 'push', 'pull'
        ]
        
        detected_actions = []
        words = text.split()
        
        for word in words:
            if word in action_verbs:
                detected_actions.append(word)
        
        return detected_actions
    
    def _analyze_pause_patterns(self, segments: List[Dict]) -> List[Dict]:
        """
        Analyze pause patterns between segments.
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Pause pattern analysis
        """
        pauses = []
        
        for i in range(len(segments) - 1):
            current_end = segments[i].get('end', 0)
            next_start = segments[i + 1].get('start', 0)
            
            pause_duration = next_start - current_end
            
            if pause_duration > 0.1:  # Only consider pauses > 100ms
                pause_type = self._classify_pause(pause_duration)
                
                pauses.append({
                    'start_time': current_end,
                    'end_time': next_start,
                    'duration': pause_duration,
                    'type': pause_type,
                    'context': {
                        'before_text': segments[i].get('text', ''),
                        'after_text': segments[i + 1].get('text', '')
                    }
                })
        
        return pauses
    
    def _classify_pause(self, duration: float) -> str:
        """
        Classify pause based on duration.
        
        Args:
            duration: Pause duration in seconds
            
        Returns:
            Pause classification
        """
        if duration < 0.5:
            return 'short'
        elif duration < 1.5:
            return 'medium'
        elif duration < 3.0:
            return 'long'
        else:
            return 'extended'
    
    def _detect_speaker_changes(self, segments: List[Dict]) -> List[Dict]:
        """
        Detect potential speaker changes (basic heuristic).
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Speaker change indicators
        """
        speaker_changes = []
        
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            # Check for indicators of speaker change
            pause_duration = next_segment.get('start', 0) - current_segment.get('end', 0)
            
            # Long pause might indicate speaker change
            if pause_duration > 2.0:
                speaker_changes.append({
                    'time': current_segment.get('end', 0),
                    'confidence': min(pause_duration / 5.0, 1.0),
                    'indicator': 'long_pause'
                })
            
            # Change in speaking characteristics
            current_rate = self._calculate_speaking_rate(current_segment)
            next_rate = self._calculate_speaking_rate(next_segment)
            
            rate_change = abs(current_rate - next_rate) / (current_rate + 0.001)
            
            if rate_change > 0.5:  # Significant change in speaking rate
                speaker_changes.append({
                    'time': next_segment.get('start', 0),
                    'confidence': min(rate_change, 1.0),
                    'indicator': 'rate_change'
                })
        
        return speaker_changes
    
    def _calculate_speaking_rate(self, segment: Dict) -> float:
        """
        Calculate speaking rate for a segment.
        
        Args:
            segment: Transcription segment
            
        Returns:
            Speaking rate (words per second)
        """
        text = segment.get('text', '')
        duration = segment.get('end', 0) - segment.get('start', 0)
        word_count = len(text.split())
        
        return word_count / (duration + 0.001)
    
    def _calculate_speech_statistics(self, analysis: Dict) -> Dict:
        """
        Calculate overall speech statistics.
        
        Args:
            analysis: Complete speech analysis
            
        Returns:
            Summary statistics
        """
        segments = analysis['segments']
        
        if not segments:
            return {}
        
        speaking_rates = [s['speaking_rate'] for s in segments]
        emotional_intensities = [s['emotional_intensity'] for s in segments]
        
        stats = {
            'total_segments': len(segments),
            'average_speaking_rate': np.mean(speaking_rates),
            'speaking_rate_variance': np.var(speaking_rates),
            'average_emotional_intensity': np.mean(emotional_intensities),
            'peak_emotional_intensity': np.max(emotional_intensities),
            'total_pauses': len(analysis['pause_patterns']),
            'average_pause_duration': np.mean([p['duration'] for p in analysis['pause_patterns']]) if analysis['pause_patterns'] else 0,
            'speaker_change_count': len(analysis['speaker_changes']),
            'zoom_strategy_distribution': self._calculate_zoom_distribution(segments)
        }
        
        return stats
    
    def _calculate_zoom_distribution(self, segments: List[Dict]) -> Dict:
        """
        Calculate distribution of zoom strategies.
        
        Args:
            segments: List of analyzed segments
            
        Returns:
            Zoom strategy distribution
        """
        strategies = [s['zoom_strategy'] for s in segments]
        distribution = {}
        
        for strategy in ['close_up', 'medium_shot', 'wide_shot', 'dynamic']:
            count = strategies.count(strategy)
            distribution[strategy] = {
                'count': count,
                'percentage': (count / len(strategies)) * 100 if strategies else 0
            }
        
        return distribution
    
    def export_analysis(self, analysis: Dict, output_path: str):
        """
        Export speech analysis to JSON file.
        
        Args:
            analysis: Speech analysis results
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            self.logger.info(f"Speech analysis exported to: {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to export analysis: {e}")
            raise
