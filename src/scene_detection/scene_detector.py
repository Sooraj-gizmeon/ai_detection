# src/scene_detection/scene_detector.py
"""Scene detection using computer vision and audio analysis"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class SceneDetector:
    """
    Detects scene changes and natural breakpoints in video content.
    """
    
    def __init__(self, threshold: float = 0.3, min_scene_length: float = 5.0):
        """
        Initialize scene detector.
        
        Args:
            threshold: Sensitivity threshold for scene detection
            min_scene_length: Minimum scene length in seconds
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.logger = logging.getLogger(__name__)
    
    def detect_scenes(self, video_path: str) -> List[Dict]:
        """
        Detect scene changes in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of scene break points with timestamps
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            scene_breaks = []
            prev_hist = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate histogram for current frame
                hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                
                if prev_hist is not None:
                    # Calculate histogram difference
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # If difference exceeds threshold, mark as scene break
                    if diff < (1 - self.threshold):
                        timestamp = frame_count / fps
                        
                        # Check minimum scene length
                        if not scene_breaks or (timestamp - scene_breaks[-1]['timestamp']) >= self.min_scene_length:
                            scene_breaks.append({
                                'timestamp': timestamp,
                                'frame_number': frame_count,
                                'confidence': 1 - diff,
                                'type': 'visual_change'
                            })
                
                prev_hist = hist
                frame_count += 1
                
                # Sample every 10th frame for efficiency
                for _ in range(9):
                    ret, _ = cap.read()
                    if not ret:
                        break
                    frame_count += 1
            
            cap.release()
            
            self.logger.info(f"Detected {len(scene_breaks)} scene breaks")
            return scene_breaks
            
        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            return []
    
    def merge_breaks(self, visual_breaks: List[Dict], audio_breaks: List[Dict]) -> List[Dict]:
        """
        Merge visual and audio scene breaks.
        
        Args:
            visual_breaks: Visual scene breaks
            audio_breaks: Audio scene breaks
            
        Returns:
            Merged list of scene breaks
        """
        all_breaks = []
        
        # Add visual breaks
        for break_point in visual_breaks:
            all_breaks.append({
                'timestamp': break_point['timestamp'],
                'type': 'visual',
                'confidence': break_point['confidence'],
                'source': 'computer_vision'
            })
        
        # Add audio breaks
        for break_point in audio_breaks:
            all_breaks.append({
                'timestamp': break_point['timestamp'],
                'type': 'audio',
                'confidence': break_point.get('confidence', 0.8),
                'source': 'audio_analysis'
            })
        
        # Sort by timestamp
        all_breaks.sort(key=lambda x: x['timestamp'])
        
        # Merge nearby breaks (within 2 seconds)
        merged_breaks = []
        for break_point in all_breaks:
            if not merged_breaks or (break_point['timestamp'] - merged_breaks[-1]['timestamp']) >= 2.0:
                merged_breaks.append(break_point)
            else:
                # Update existing break with higher confidence
                if break_point['confidence'] > merged_breaks[-1]['confidence']:
                    merged_breaks[-1].update(break_point)
        
        return merged_breaks
        
    def merge_breaks_with_filtering(self, visual_breaks: List[Dict], audio_breaks: List[Dict], preroll_end: float = 0.0, audio_analysis: Optional[Dict] = None) -> List[Dict]:
        """
        Merge visual and audio scene breaks with preroll filtering.
        
        Args:
            visual_breaks: Visual scene breaks
            audio_breaks: Audio breaks
            preroll_end: End timestamp of pre-roll content (in seconds)
            audio_analysis: Full audio analysis data for additional filtering
            
        Returns:
            Filtered and merged list of scene breaks
        """
        # First get merged breaks using standard method
        merged_breaks = self.merge_breaks(visual_breaks, audio_breaks)
        
        # If no preroll filtering needed, return standard merged breaks
        if preroll_end <= 0 and not audio_analysis:
            return merged_breaks
        
        # Check audio analysis for music segments if available
        music_segments = []
        silence_segments = []
        likely_preroll = []
        
        if audio_analysis and 'zoom_analysis' in audio_analysis:
            zoom_analysis = audio_analysis['zoom_analysis']
            
            # Get music segments
            if 'music_segments' in zoom_analysis:
                music_segments = zoom_analysis['music_segments']
                
            # Get silence segments
            if 'silence_segments' in zoom_analysis:
                silence_segments = zoom_analysis['silence_segments']
                
            # Get explicitly detected preroll
            if 'likely_preroll' in zoom_analysis:
                likely_preroll = zoom_analysis['likely_preroll']
                # Update preroll_end if needed
                if likely_preroll:
                    detected_preroll_end = max(segment['end'] for segment in likely_preroll)
                    preroll_end = max(preroll_end, detected_preroll_end)
        
        # Check transcription for additional silence periods
        if audio_analysis and 'transcription' in audio_analysis:
            transcription = audio_analysis['transcription']
            if 'silence_periods' in transcription:
                for start, end in transcription['silence_periods']:
                    # Only consider silence in the first 10% of the video
                    video_duration = transcription.get('stats', {}).get('total_duration', 0)
                    if video_duration > 0 and end < video_duration * 0.1:
                        silence_segments.append({
                            'start': start,
                            'end': end,
                            'type': 'silence'
                        })
        
        # Combine all preroll-like segments
        preroll_segments = likely_preroll + [
            seg for seg in music_segments + silence_segments 
            if seg['start'] < preroll_end
        ]
        
        # Filter out breaks that fall within preroll segments
        filtered_breaks = []
        for break_point in merged_breaks:
            timestamp = break_point['timestamp']
            
            # Skip if timestamp is before preroll_end
            if timestamp <= preroll_end:
                self.logger.info(f"Filtered out break at {timestamp:.2f}s (before preroll_end at {preroll_end:.2f}s)")
                continue
                
            # Check if this break point is within any preroll segment
            in_preroll_segment = False
            for segment in preroll_segments:
                if segment['start'] <= timestamp <= segment['end']:
                    in_preroll_segment = True
                    break
                    
            if not in_preroll_segment:
                filtered_breaks.append(break_point)
            else:
                self.logger.info(f"Filtered out break at {timestamp:.2f}s (in preroll segment)")
                
        self.logger.info(f"Filtered {len(merged_breaks) - len(filtered_breaks)} breaks in preroll segments")
        
        # Final check: if no breaks remain after filtering, add at least one break after preroll
        if not filtered_breaks and merged_breaks:
            # Find the first break after preroll_end
            after_preroll = [b for b in merged_breaks if b['timestamp'] > preroll_end]
            if after_preroll:
                filtered_breaks.append(after_preroll[0])
                self.logger.info(f"Added back first break after preroll at {after_preroll[0]['timestamp']:.2f}s")
        
        return filtered_breaks
