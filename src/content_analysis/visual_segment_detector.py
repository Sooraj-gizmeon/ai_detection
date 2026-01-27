"""
Visual-first segment detection for silent or visually-rich content.
Identifies segments based on visual activity rather than audio content.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass


@dataclass
class VisualSegment:
    start_time: float
    end_time: float
    duration: float
    visual_activity_score: float
    scene_type: str
    motion_level: str
    has_people: bool
    color_diversity: float
    visual_complexity: float
    silence_ratio: float  # 0-1, how much of this segment is silent


class VisualSegmentDetector:
    """
    Detects segments based on visual content rather than audio.
    Particularly useful for silent demonstrations, action sequences, and visual storytelling.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for visual activity detection
        self.motion_threshold = 30.0  # Pixel difference threshold
        self.color_diversity_threshold = 0.5
        self.complexity_threshold = 0.6
        
        # Segment parameters
        self.min_segment_duration = 30.0  # Minimum 30 seconds
        self.max_segment_duration = 180.0  # Maximum 3 minutes
        self.overlap_threshold = 0.3  # Allow some overlap with audio segments
        
    def detect_visual_segments(self, 
                             video_path: str, 
                             audio_analysis: Dict,
                             existing_segments: List[Dict] = None) -> List[VisualSegment]:
        """
        Detect visually interesting segments that might be missed by audio-only analysis.
        
        Args:
            video_path: Path to video file
            audio_analysis: Audio analysis results
            existing_segments: Already detected audio-based segments
            
        Returns:
            List of visual segments that complement audio-based segments
        """
        try:
            self.logger.info("Starting visual-first segment detection")
            
            # Analyze video for visual activity
            visual_activity = self._analyze_visual_activity(video_path)
            
            # Detect silent periods that might be visually interesting
            silent_periods = self._find_silent_periods(audio_analysis)
            
            # Find high-motion segments
            motion_segments = self._detect_motion_segments(visual_activity)
            
            # Find visual complexity changes
            complexity_segments = self._detect_complexity_segments(visual_activity)
            
            # Combine and filter segments
            visual_segments = self._combine_visual_segments(
                motion_segments, complexity_segments, silent_periods
            )
            
            # Remove overlaps with existing audio segments
            if existing_segments:
                visual_segments = self._filter_overlapping_segments(
                    visual_segments, existing_segments
                )
            
            self.logger.info(f"Detected {len(visual_segments)} visual-first segments")
            return visual_segments
            
        except Exception as e:
            self.logger.error(f"Error in visual segment detection: {e}")
            return []
    
    def _analyze_visual_activity(self, video_path: str) -> Dict:
        """Analyze visual activity throughout the video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Sample every 2 seconds for analysis
        sample_interval = int(fps * 2)
        
        activity_data = []
        prev_frame = None
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            timestamp = frame_idx / fps
            
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate motion if we have a previous frame
            motion_score = 0.0
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
            
            # Calculate visual complexity (edge density)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Calculate color diversity
            color_hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_diversity = cv2.compareHist(color_hist, np.ones_like(color_hist), cv2.HISTCMP_CHISQR)
            
            activity_data.append({
                'timestamp': timestamp,
                'motion_score': motion_score,
                'complexity': complexity,
                'color_diversity': color_diversity,
                'frame_index': frame_idx
            })
            
            prev_frame = gray
        
        cap.release()
        
        return {
            'duration': duration,
            'fps': fps,
            'activity_timeline': activity_data
        }
    
    def _find_silent_periods(self, audio_analysis: Dict) -> List[Dict]:
        """Find periods with little to no audio that might be visually interesting."""
        silent_periods = []
        
        # Get transcription segments
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        if not segments:
            return silent_periods
        
        # Find gaps between speech segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end']
            next_start = segments[i + 1]['start']
            
            gap_duration = next_start - current_end
            
            # If gap is significant (>5 seconds), it might be visually interesting
            if gap_duration >= 5.0:
                silent_periods.append({
                    'start': current_end,
                    'end': next_start,
                    'duration': gap_duration,
                    'type': 'speech_gap'
                })
        
        # Also check for explicit silence segments
        for silence in transcription.get('silence_periods', []):
            start, end = silence
            duration = end - start
            
            if duration >= 5.0:  # At least 5 seconds of silence
                silent_periods.append({
                    'start': start,
                    'end': end,
                    'duration': duration,
                    'type': 'detected_silence'
                })
        
        return silent_periods
    
    def _detect_motion_segments(self, visual_activity: Dict) -> List[Dict]:
        """Detect segments with high visual motion/activity."""
        activity_timeline = visual_activity['activity_timeline']
        motion_segments = []
        
        # Find periods of sustained high motion
        in_motion_segment = False
        segment_start = 0.0
        
        for data_point in activity_timeline:
            timestamp = data_point['timestamp']
            motion_score = data_point['motion_score']
            
            if motion_score > self.motion_threshold:
                if not in_motion_segment:
                    segment_start = timestamp
                    in_motion_segment = True
            else:
                if in_motion_segment:
                    duration = timestamp - segment_start
                    if duration >= 10.0:  # At least 10 seconds of motion
                        motion_segments.append({
                            'start': segment_start,
                            'end': timestamp,
                            'duration': duration,
                            'type': 'high_motion',
                            'avg_motion': motion_score
                        })
                    in_motion_segment = False
        
        return motion_segments
    
    def _detect_complexity_segments(self, visual_activity: Dict) -> List[Dict]:
        """Detect segments with high visual complexity (detailed scenes)."""
        activity_timeline = visual_activity['activity_timeline']
        complexity_segments = []
        
        # Calculate average complexity for baseline
        avg_complexity = np.mean([d['complexity'] for d in activity_timeline])
        complexity_threshold = avg_complexity * 1.5  # 50% above average
        
        in_complex_segment = False
        segment_start = 0.0
        
        for data_point in activity_timeline:
            timestamp = data_point['timestamp']
            complexity = data_point['complexity']
            
            if complexity > complexity_threshold:
                if not in_complex_segment:
                    segment_start = timestamp
                    in_complex_segment = True
            else:
                if in_complex_segment:
                    duration = timestamp - segment_start
                    if duration >= 15.0:  # At least 15 seconds of complexity
                        complexity_segments.append({
                            'start': segment_start,
                            'end': timestamp,
                            'duration': duration,
                            'type': 'high_complexity',
                            'avg_complexity': complexity
                        })
                    in_complex_segment = False
        
        return complexity_segments
    
    def _combine_visual_segments(self, 
                               motion_segments: List[Dict],
                               complexity_segments: List[Dict],
                               silent_periods: List[Dict]) -> List[VisualSegment]:
        """Combine different types of visual segments into final candidates."""
        all_segments = []
        
        # Process motion segments
        for seg in motion_segments:
            all_segments.append(VisualSegment(
                start_time=seg['start'],
                end_time=seg['end'],
                duration=seg['duration'],
                visual_activity_score=0.8,  # High for motion
                scene_type='action',
                motion_level='high',
                has_people=True,  # Assume motion involves people
                color_diversity=0.6,
                visual_complexity=0.5,
                silence_ratio=0.0  # Motion segments unlikely to be silent
            ))
        
        # Process complexity segments
        for seg in complexity_segments:
            all_segments.append(VisualSegment(
                start_time=seg['start'],
                end_time=seg['end'],
                duration=seg['duration'],
                visual_activity_score=0.7,
                scene_type='detailed',
                motion_level='medium',
                has_people=False,  # Complex doesn't necessarily mean people
                color_diversity=0.8,  # Complex scenes often colorful
                visual_complexity=0.9,
                silence_ratio=0.3
            ))
        
        # Process silent periods (potential visual storytelling)
        for seg in silent_periods:
            if seg['duration'] >= self.min_segment_duration:
                all_segments.append(VisualSegment(
                    start_time=seg['start'],
                    end_time=seg['end'],
                    duration=seg['duration'],
                    visual_activity_score=0.6,  # Could be interesting
                    scene_type='silent_visual',
                    motion_level='unknown',
                    has_people=False,
                    color_diversity=0.5,
                    visual_complexity=0.6,
                    silence_ratio=1.0  # By definition
                ))
        
        # Sort by start time and merge overlapping segments
        all_segments.sort(key=lambda x: x.start_time)
        merged_segments = self._merge_overlapping_segments(all_segments)
        
        return merged_segments
    
    def _merge_overlapping_segments(self, segments: List[VisualSegment]) -> List[VisualSegment]:
        """Merge overlapping visual segments."""
        if not segments:
            return []
        
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current.start_time <= last.end_time + 5.0:  # 5 second tolerance
                # Merge segments
                merged[-1] = VisualSegment(
                    start_time=last.start_time,
                    end_time=max(last.end_time, current.end_time),
                    duration=max(last.end_time, current.end_time) - last.start_time,
                    visual_activity_score=max(last.visual_activity_score, current.visual_activity_score),
                    scene_type=f"{last.scene_type}+{current.scene_type}",
                    motion_level=current.motion_level if current.motion_level != 'unknown' else last.motion_level,
                    has_people=last.has_people or current.has_people,
                    color_diversity=max(last.color_diversity, current.color_diversity),
                    visual_complexity=max(last.visual_complexity, current.visual_complexity),
                    silence_ratio=(last.silence_ratio + current.silence_ratio) / 2
                )
            else:
                merged.append(current)
        
        return merged
    
    def _filter_overlapping_segments(self, 
                                   visual_segments: List[VisualSegment],
                                   existing_segments: List[Dict]) -> List[VisualSegment]:
        """Filter out visual segments that significantly overlap with existing audio segments."""
        filtered = []
        
        for visual_seg in visual_segments:
            has_significant_overlap = False
            
            for audio_seg in existing_segments:
                overlap = self._calculate_overlap(
                    visual_seg.start_time, visual_seg.end_time,
                    audio_seg['start_time'], audio_seg['end_time']
                )
                
                if overlap > self.overlap_threshold:
                    has_significant_overlap = True
                    break
            
            # Keep visual segments that don't significantly overlap with audio segments
            if not has_significant_overlap:
                filtered.append(visual_seg)
        
        return filtered
    
    def _calculate_overlap(self, start1: float, end1: float, start2: float, end2: float) -> float:
        """Calculate overlap ratio between two time segments."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        segment1_duration = end1 - start1
        
        return overlap_duration / segment1_duration if segment1_duration > 0 else 0.0


def integrate_visual_segments_with_content_analyzer():
    """
    Example of how to integrate visual segment detection with the main content analyzer.
    """
    # This would be called from ContentAnalyzer.identify_short_segments()
    
    # 1. Run normal audio-based segment detection
    # audio_segments = self.identify_short_segments(...)
    
    # 2. Run visual segment detection
    # visual_detector = VisualSegmentDetector()
    # visual_segments = visual_detector.detect_visual_segments(video_path, audio_analysis, audio_segments)
    
    # 3. Combine and prioritize
    # combined_segments = audio_segments + [convert_visual_to_dict(vs) for vs in visual_segments]
    
    # 4. Score and select best segments
    # final_segments = self._select_diverse_content(combined_segments, max_shorts)
    
    pass
