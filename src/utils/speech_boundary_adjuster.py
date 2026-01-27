"""
Speech Boundary Adjuster - Ensures video segments cut at natural speech boundaries.

This module adjusts segment start/end times to align with:
- Complete sentences
- Natural pauses in speech
- Word boundaries
- Logical thought completion

ENHANCED: Now includes:
- Clean start detection (avoid starting mid-conversation)
- Clean end detection (avoid cutting mid-dialogue)
- Flexible duration adjustment for better flow

Prevents jarring mid-word or mid-sentence cuts.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional


class SpeechBoundaryAdjuster:
    """
    Adjusts segment boundaries to natural speech breaks for better viewer experience.
    
    ENHANCED with:
    - Conversation context awareness (don't start with "and", "but", "so")
    - Dialogue flow protection (extend to complete thought)
    - Flexible duration (allow ±5s variance for better boundaries)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Speech Boundary Adjuster.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Sentence ending punctuation
        self.sentence_endings = ['.', '!', '?', '...']
        
        # Natural pause indicators
        self.pause_indicators = [',', ';', ':', '--', '—']
        
        # Words that indicate CONTINUATION (bad to start with)
        self.continuation_words = [
            'and', 'but', 'so', 'because', 'however', 'also', 'then',
            'therefore', 'moreover', 'furthermore', 'although', 'though',
            'yet', 'still', 'besides', 'meanwhile', 'otherwise', 'hence'
        ]
        
        # Words that indicate STRONG OPENING (good to start with)
        self.strong_openers = [
            'what', 'why', 'how', 'when', 'where', 'who', 'which',
            'the', 'this', 'there', 'here', 'now', 'today', 'imagine',
            'let', 'think', 'consider', 'remember', 'look', 'see',
            'first', 'one', 'i', 'we', 'you', 'if', 'every', 'all'
        ]
        
        # Incomplete dialogue patterns (bad to end with)
        self.incomplete_patterns = [
            r'\b(and|but|or|so|because|if|when|while|as|that|which)\s*$',
            r'\bthe\s*$',
            r'\bto\s*$',
            r'\ba\s*$',
            r'\bin\s*$',
            r'\bof\s*$',
            r',\s*$',  # Trailing comma
        ]
    
    def adjust_segment_boundaries_flexible(self,
                                          start_time: float,
                                          end_time: float,
                                          transcription: Dict,
                                          target_duration: Tuple[int, int],
                                          max_adjustment: float = 5.0) -> Tuple[float, float, Dict]:
        """
        ENHANCED: Adjust segment boundaries with FLEXIBLE DURATION for better flow.
        
        This method allows the segment duration to vary within the target range
        if it results in better natural boundaries (complete thoughts, clean starts/ends).
        
        Args:
            start_time: Original start time
            end_time: Original end time  
            transcription: Full transcription with word-level timestamps
            target_duration: Tuple of (min_duration, max_duration) in seconds
            max_adjustment: Maximum seconds to adjust each boundary (default 5.0)
            
        Returns:
            Tuple of (adjusted_start, adjusted_end, adjustment_info)
        """
        min_duration, max_duration = target_duration
        original_duration = end_time - start_time
        
        try:
            # Step 1: Find the BEST start point (clean conversation start)
            best_start = self._find_clean_start(
                start_time, transcription, max_adjustment
            )
            
            # Step 2: Find the BEST end point (complete thought)
            best_end = self._find_clean_end(
                end_time, transcription, max_adjustment
            )
            
            # Step 3: Validate duration and adjust if needed
            new_duration = best_end - best_start
            
            adjustment_info = {
                'original_start': start_time,
                'original_end': end_time,
                'original_duration': original_duration,
                'adjusted_start': best_start,
                'adjusted_end': best_end,
                'adjusted_duration': new_duration,
                'start_change': best_start - start_time,
                'end_change': best_end - end_time,
                'flow_quality': 'unknown'
            }
            
            # Check if new duration is within acceptable range
            if new_duration < min_duration:
                # Too short - try extending end first, then moving start earlier
                self.logger.debug(f"Duration {new_duration:.1f}s too short, extending...")
                best_end = self._extend_to_duration(best_start, min_duration, transcription, max_duration)
                new_duration = best_end - best_start
                adjustment_info['adjusted_end'] = best_end
                adjustment_info['adjusted_duration'] = new_duration
                
            elif new_duration > max_duration:
                # Too long - try shortening from end while keeping clean boundary
                self.logger.debug(f"Duration {new_duration:.1f}s too long, trimming...")
                best_end = self._trim_to_duration(best_start, max_duration, transcription)
                new_duration = best_end - best_start
                adjustment_info['adjusted_end'] = best_end
                adjustment_info['adjusted_duration'] = new_duration
            
            # Step 4: Assess flow quality
            segment_text = self._extract_segment_text(transcription, best_start, best_end)
            flow_quality = self._assess_flow_quality(segment_text)
            adjustment_info['flow_quality'] = flow_quality
            adjustment_info['segment_text_preview'] = segment_text[:150] + '...' if len(segment_text) > 150 else segment_text
            
            self.logger.info(
                f"🎬 Flexible boundary: [{start_time:.1f}s-{end_time:.1f}s] → "
                f"[{best_start:.1f}s-{best_end:.1f}s] "
                f"(duration: {original_duration:.1f}s → {new_duration:.1f}s, flow: {flow_quality})"
            )
            
            return best_start, best_end, adjustment_info
            
        except Exception as e:
            self.logger.warning(f"Flexible boundary adjustment failed: {e}")
            return start_time, end_time, {'error': str(e), 'flow_quality': 'failed'}
    
    def _find_clean_start(self, 
                         start_time: float, 
                         transcription: Dict, 
                         max_adjustment: float) -> float:
        """
        Find a clean start point that doesn't begin mid-conversation.
        
        Avoids starting with: "And", "But", "So", etc.
        Prefers starting with: Strong openers, questions, new topics
        """
        segments = transcription.get('segments', [])
        candidates = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            text = segment.get('text', '').strip()
            
            # Only consider starts within our adjustment window
            if abs(seg_start - start_time) > max_adjustment:
                continue
                
            # Skip empty segments
            if not text:
                continue
            
            # Calculate start quality score
            start_quality = self._score_start_quality(text, segment, segments)
            
            candidates.append({
                'time': seg_start,
                'text': text,
                'quality': start_quality,
                'distance': abs(seg_start - start_time)
            })
        
        if not candidates:
            return start_time
        
        # Sort by quality (descending), then by distance (ascending)
        candidates.sort(key=lambda x: (-x['quality'], x['distance']))
        
        best = candidates[0]
        if best['quality'] < 0.3:
            # Even best candidate is poor - log warning
            self.logger.warning(
                f"No ideal start found near {start_time:.1f}s. "
                f"Best option starts with: '{best['text'][:30]}...' (quality: {best['quality']:.2f})"
            )
        
        return best['time']
    
    def _score_start_quality(self, text: str, segment: Dict, all_segments: List[Dict]) -> float:
        """
        Score how good a starting point this is (0.0 to 1.0).
        """
        score = 0.5  # Base score
        text_lower = text.lower().strip()
        first_word = text_lower.split()[0] if text_lower.split() else ''
        
        # PENALTY: Starts with continuation word (-0.4)
        if first_word in self.continuation_words:
            score -= 0.4
            
        # BONUS: Starts with strong opener (+0.3)
        if first_word in self.strong_openers:
            score += 0.3
        
        # BONUS: Previous segment ended with sentence ending (+0.3)
        if self._is_sentence_start(segment, all_segments):
            score += 0.3
        
        # BONUS: Starts with question word (+0.2)
        if first_word in ['what', 'why', 'how', 'when', 'where', 'who']:
            score += 0.2
        
        # BONUS: Has significant pause before (+0.2)
        if self._has_pause_before(segment, all_segments):
            score += 0.2
        
        # PENALTY: Starts with lowercase (likely mid-sentence) (-0.2)
        if text and text[0].islower():
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _find_clean_end(self,
                       end_time: float,
                       transcription: Dict,
                       max_adjustment: float) -> float:
        """
        Find a clean end point that completes the thought/dialogue.
        
        Avoids ending with: trailing conjunctions, incomplete phrases
        Prefers ending with: Complete sentences, natural pauses
        """
        segments = transcription.get('segments', [])
        candidates = []
        
        for segment in segments:
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Only consider ends within our adjustment window
            if abs(seg_end - end_time) > max_adjustment:
                continue
            
            if not text:
                continue
            
            # Calculate end quality score
            end_quality = self._score_end_quality(text)
            
            candidates.append({
                'time': seg_end,
                'text': text,
                'quality': end_quality,
                'distance': abs(seg_end - end_time)
            })
        
        if not candidates:
            return end_time
        
        # Sort by quality (descending), then by distance (ascending)
        candidates.sort(key=lambda x: (-x['quality'], x['distance']))
        
        best = candidates[0]
        if best['quality'] < 0.3:
            self.logger.warning(
                f"No ideal end found near {end_time:.1f}s. "
                f"Best option ends with: '...{best['text'][-30:]}' (quality: {best['quality']:.2f})"
            )
        
        return best['time']
    
    def _score_end_quality(self, text: str) -> float:
        """
        Score how good an ending point this is (0.0 to 1.0).
        """
        score = 0.5  # Base score
        text = text.strip()
        
        if not text:
            return 0.0
        
        # BONUS: Ends with sentence-ending punctuation (+0.4)
        if any(text.endswith(p) for p in self.sentence_endings):
            score += 0.4
        
        # BONUS: Ends with natural pause (+0.2)
        elif any(text.endswith(p) for p in self.pause_indicators):
            score += 0.2
        
        # PENALTY: Matches incomplete patterns (-0.4)
        for pattern in self.incomplete_patterns:
            if re.search(pattern, text.lower()):
                score -= 0.4
                break
        
        # PENALTY: Very short segment (likely incomplete) (-0.2)
        if len(text.split()) < 3:
            score -= 0.2
        
        # BONUS: Contains closing words (+0.1)
        closing_words = ['thank', 'thanks', 'finally', 'conclusion', 'summary', 'remember']
        if any(word in text.lower() for word in closing_words):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _extend_to_duration(self,
                           start_time: float,
                           min_duration: float,
                           transcription: Dict,
                           max_duration: float) -> float:
        """
        Extend segment end to meet minimum duration while finding clean boundary.
        """
        segments = transcription.get('segments', [])
        target_end = start_time + min_duration
        
        # Find segments near target end
        candidates = []
        for segment in segments:
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Only consider ends that would give us valid duration
            new_duration = seg_end - start_time
            if new_duration < min_duration or new_duration > max_duration:
                continue
            
            end_quality = self._score_end_quality(text)
            candidates.append({
                'time': seg_end,
                'quality': end_quality,
                'duration': new_duration
            })
        
        if not candidates:
            return start_time + min_duration
        
        # Prefer quality, then shorter duration
        candidates.sort(key=lambda x: (-x['quality'], x['duration']))
        return candidates[0]['time']
    
    def _trim_to_duration(self,
                         start_time: float,
                         max_duration: float,
                         transcription: Dict) -> float:
        """
        Trim segment end to meet maximum duration while finding clean boundary.
        """
        segments = transcription.get('segments', [])
        target_end = start_time + max_duration
        
        candidates = []
        for segment in segments:
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            new_duration = seg_end - start_time
            # Only consider ends that would give us valid duration
            if new_duration > max_duration:
                continue
            if new_duration < max_duration * 0.8:  # Don't trim too much
                continue
            
            end_quality = self._score_end_quality(text)
            candidates.append({
                'time': seg_end,
                'quality': end_quality,
                'duration': new_duration
            })
        
        if not candidates:
            return start_time + max_duration
        
        # Prefer quality, then longer duration
        candidates.sort(key=lambda x: (-x['quality'], -x['duration']))
        return candidates[0]['time']
    
    def _assess_flow_quality(self, text: str) -> str:
        """
        Assess overall flow quality of the segment text.
        
        Returns: 'excellent', 'good', 'acceptable', 'poor'
        """
        if not text:
            return 'poor'
        
        text = text.strip()
        score = 0.0
        
        # Check start quality
        first_word = text.lower().split()[0] if text.split() else ''
        if first_word not in self.continuation_words and text[0].isupper():
            score += 0.25
        
        # Check end quality
        if any(text.endswith(p) for p in self.sentence_endings):
            score += 0.25
        
        # Check for complete sentences
        sentence_count = len([s for s in re.split(r'[.!?]', text) if s.strip()])
        if sentence_count >= 2:
            score += 0.25
        elif sentence_count >= 1:
            score += 0.15
        
        # Check word count
        word_count = len(text.split())
        if 20 <= word_count <= 150:
            score += 0.25
        elif 10 <= word_count <= 200:
            score += 0.15
        
        if score >= 0.8:
            return 'excellent'
        elif score >= 0.6:
            return 'good'
        elif score >= 0.4:
            return 'acceptable'
        else:
            return 'poor'
    
    def _extract_segment_text(self, transcription: Dict, start_time: float, end_time: float) -> str:
        """Extract text from transcription for a specific time range."""
        segments = transcription.get('segments', [])
        text_parts = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with our time range
            if seg_start < end_time and seg_end > start_time:
                text = segment.get('text', '').strip()
                if text:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
        
    def adjust_segment_boundaries(self,
                                 start_time: float,
                                 end_time: float,
                                 transcription: Dict,
                                 max_adjustment: float = 3.0,
                                 prefer_extension: bool = True) -> Tuple[float, float]:
        """
        Adjust segment boundaries to natural speech breaks.
        
        Args:
            start_time: Original start time
            end_time: Original end time
            transcription: Full transcription with word-level timestamps
            max_adjustment: Maximum seconds to adjust boundaries (default 3.0)
            prefer_extension: Prefer extending segments over shortening (default True)
            
        Returns:
            Tuple of (adjusted_start_time, adjusted_end_time)
        """
        try:
            adjusted_start = self._adjust_start_boundary(
                start_time, transcription, max_adjustment, prefer_extension
            )
            adjusted_end = self._adjust_end_boundary(
                end_time, transcription, max_adjustment, prefer_extension
            )
            
            # Log adjustments if significant
            if abs(adjusted_start - start_time) > 0.5 or abs(adjusted_end - end_time) > 0.5:
                self.logger.info(
                    f"📍 Boundary adjusted: [{start_time:.2f}s, {end_time:.2f}s] → "
                    f"[{adjusted_start:.2f}s, {adjusted_end:.2f}s]"
                )
            
            return adjusted_start, adjusted_end
            
        except Exception as e:
            self.logger.warning(f"Boundary adjustment failed: {e}, using original boundaries")
            return start_time, end_time
    
    def _adjust_start_boundary(self,
                              start_time: float,
                              transcription: Dict,
                              max_adjustment: float,
                              prefer_extension: bool) -> float:
        """
        Adjust start boundary to begin at a natural break.
        
        Strategy:
        1. Find nearest sentence start before or after
        2. If no sentence boundary, find natural pause
        3. If no pause, find word boundary
        """
        segments = transcription.get('segments', [])
        
        # Find best start position within adjustment window
        candidates = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Check if segment is within adjustment window
            if abs(seg_start - start_time) <= max_adjustment:
                
                # Priority 1: Sentence start (previous sentence ended)
                if self._is_sentence_start(segment, segments):
                    candidates.append({
                        'time': seg_start,
                        'priority': 1,
                        'type': 'sentence_start',
                        'distance': abs(seg_start - start_time)
                    })
                
                # Priority 2: After a natural pause
                elif self._has_pause_before(segment, segments):
                    candidates.append({
                        'time': seg_start,
                        'priority': 2,
                        'type': 'after_pause',
                        'distance': abs(seg_start - start_time)
                    })
                
                # Priority 3: Word boundary
                else:
                    candidates.append({
                        'time': seg_start,
                        'priority': 3,
                        'type': 'word_boundary',
                        'distance': abs(seg_start - start_time)
                    })
        
        if not candidates:
            return start_time
        
        # Sort by priority first, then by distance
        candidates.sort(key=lambda x: (x['priority'], x['distance']))
        
        # If prefer_extension, choose earlier time when priorities are equal
        if prefer_extension and len(candidates) > 1:
            best_priority = candidates[0]['priority']
            same_priority = [c for c in candidates if c['priority'] == best_priority]
            if same_priority:
                # Choose the one that extends the segment (earlier start)
                return min(c['time'] for c in same_priority)
        
        return candidates[0]['time']
    
    def _adjust_end_boundary(self,
                            end_time: float,
                            transcription: Dict,
                            max_adjustment: float,
                            prefer_extension: bool) -> float:
        """
        Adjust end boundary to end at a natural break.
        
        Strategy:
        1. Find nearest complete sentence ending
        2. If no sentence boundary, find natural pause
        3. If no pause, find word boundary
        """
        segments = transcription.get('segments', [])
        
        # Find best end position within adjustment window
        candidates = []
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Check if segment is within adjustment window
            if abs(seg_end - end_time) <= max_adjustment:
                
                # Priority 1: Complete sentence ending
                if self._is_sentence_ending(text):
                    candidates.append({
                        'time': seg_end,
                        'priority': 1,
                        'type': 'sentence_end',
                        'distance': abs(seg_end - end_time),
                        'text': text
                    })
                
                # Priority 2: Natural pause
                elif self._has_natural_pause(text):
                    candidates.append({
                        'time': seg_end,
                        'priority': 2,
                        'type': 'natural_pause',
                        'distance': abs(seg_end - end_time),
                        'text': text
                    })
                
                # Priority 3: Word boundary
                else:
                    candidates.append({
                        'time': seg_end,
                        'priority': 3,
                        'type': 'word_boundary',
                        'distance': abs(seg_end - end_time),
                        'text': text
                    })
        
        if not candidates:
            return end_time
        
        # Sort by priority first, then by distance
        candidates.sort(key=lambda x: (x['priority'], x['distance']))
        
        # If prefer_extension, choose later time when priorities are equal
        if prefer_extension and len(candidates) > 1:
            best_priority = candidates[0]['priority']
            same_priority = [c for c in candidates if c['priority'] == best_priority]
            if same_priority:
                # Choose the one that extends the segment (later end)
                best_time = max(c['time'] for c in same_priority)
                self.logger.debug(
                    f"Extended end to complete thought: '{[c['text'] for c in same_priority if c['time'] == best_time][0][:50]}...'"
                )
                return best_time
        
        return candidates[0]['time']
    
    def _is_sentence_start(self, segment: Dict, all_segments: List[Dict]) -> bool:
        """Check if segment starts a new sentence."""
        text = segment.get('text', '').strip()
        
        # Empty or very short text is not a sentence start
        if not text or len(text) < 3:
            return False
        
        # Check if previous segment ended with sentence ending punctuation
        try:
            seg_index = all_segments.index(segment)
        except ValueError:
            return False
            
        if seg_index > 0:
            prev_segment = all_segments[seg_index - 1]
            prev_text = prev_segment.get('text', '').strip()
            
            if any(prev_text.endswith(ending) for ending in self.sentence_endings):
                # Check if current segment starts with capital letter (common in transcriptions)
                if text[0].isupper() or text.startswith('"') or text.startswith("'"):
                    return True
        
        return False
    
    def _has_pause_before(self, segment: Dict, all_segments: List[Dict]) -> bool:
        """Check if there's a natural pause before this segment."""
        try:
            seg_index = all_segments.index(segment)
        except ValueError:
            return False
            
        if seg_index > 0:
            prev_segment = all_segments[seg_index - 1]
            prev_end = prev_segment.get('end', 0)
            curr_start = segment.get('start', 0)
            
            # Significant gap indicates pause (0.5 seconds or more)
            if curr_start - prev_end >= 0.5:
                return True
            
            # Previous segment ended with pause indicator
            prev_text = prev_segment.get('text', '').strip()
            if any(prev_text.endswith(p) for p in self.pause_indicators):
                return True
        
        return False
    
    def _is_sentence_ending(self, text: str) -> bool:
        """Check if text ends with sentence-ending punctuation."""
        if not text:
            return False
        text = text.strip()
        return any(text.endswith(ending) for ending in self.sentence_endings)
    
    def _has_natural_pause(self, text: str) -> bool:
        """Check if text ends with natural pause punctuation."""
        if not text:
            return False
        text = text.strip()
        return any(text.endswith(p) for p in self.pause_indicators)
    
    def batch_adjust_segments(self,
                             segments: List[Dict],
                             transcription: Dict,
                             user_prompt: str = "",
                             target_duration: Tuple[int, int] = (30, 60)) -> List[Dict]:
        """
        Batch adjust boundaries for multiple segments.
        
        Args:
            segments: List of segment dictionaries with start_time and end_time
            transcription: Full transcription with word-level timestamps
            user_prompt: User's search prompt for context
            target_duration: Target duration range (min, max) in seconds
            
        Returns:
            List of segments with adjusted boundaries
        """
        adjusted_segments = []
        
        for segment in segments:
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            
            # Use flexible adjustment for better flow
            adjusted_start, adjusted_end, adjustment_info = self.adjust_segment_boundaries_flexible(
                start_time=start_time,
                end_time=end_time,
                transcription=transcription,
                target_duration=target_duration,
                max_adjustment=5.0
            )
            
            # Create adjusted segment
            adjusted_segment = segment.copy()
            adjusted_segment['start_time'] = adjusted_start
            adjusted_segment['end_time'] = adjusted_end
            adjusted_segment['duration'] = adjusted_end - adjusted_start
            adjusted_segment['boundary_adjusted'] = True
            adjusted_segment['flow_quality'] = adjustment_info.get('flow_quality', 'unknown')
            adjusted_segment['adjustment_info'] = adjustment_info
            
            adjusted_segments.append(adjusted_segment)
        
        return adjusted_segments
