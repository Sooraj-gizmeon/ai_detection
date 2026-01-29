# src/content_analysis/comprehensive_segment_generator.py
"""Comprehensive segment generator that analyzes ALL potential segments for maximum quality"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations


class ComprehensiveSegmentGenerator:
    """
    Generates and analyzes ALL possible video segments within target duration ranges
    to ensure maximum coverage and quality in segment selection.
    """
    
    def __init__(self):
        """Initialize comprehensive segment generator."""
        self.logger = logging.getLogger(__name__)
        
        # Configuration for comprehensive analysis
        self.config = {
            'sliding_window_step': 5.0,    # Generate segments every 5 seconds
            'overlap_tolerance': 0.5,       # INCREASED: Allow 50% overlap to preserve diverse content
            'quality_sampling_rate': 2.0,   # Sample every 2 seconds for quality assessment
            'max_segments_per_duration': 50, # Maximum segments per duration range
            'enable_micro_segments': True,   # Include very short segments
            'enable_extended_segments': True, # Include longer segments
            'content_similarity_threshold': 0.9  # NEW: Only dedupe if content is >60% similar
        }
    
    def generate_all_possible_segments(self,
                                     video_info: Dict,
                                     audio_analysis: Dict,
                                     scene_analysis: Dict,
                                     target_duration: Tuple[int, int] = (15, 60),
                                     max_total_segments: int = 200,
                                     celebrity_index_path: Optional[str] = None,
                                     actor_only: bool = True) -> List[Dict]:
        """
        Generate ALL possible segments within target duration using multiple strategies.
        If `celebrity_index_path` is provided, generate high-priority actor-based segments
        from the celebrity appearances so prompt-based actor requests can be satisfied.
        
        Args:
            video_info: Video metadata
            audio_analysis: Audio analysis results
            scene_analysis: Scene detection results
            target_duration: Target duration range (min, max) in seconds OR single target duration
            max_total_segments: Maximum total segments to generate
            celebrity_index_path: Optional path to celebrity result JSON to prioritize actor segments
            
        Returns:
            Comprehensive list of all possible segments
        """
        try:
            # Handle both tuple (min, max) and single integer target_duration
            if isinstance(target_duration, (tuple, list)) and len(target_duration) == 2:
                min_duration, max_duration = target_duration
            elif isinstance(target_duration, (int, float)):
                # Single duration - create reasonable range around it
                target = int(target_duration)
                min_duration = max(15, target - 15)  # At least 15s minimum
                max_duration = target + 15  # Allow up to 15s longer
                self.logger.info(f"🔧 Converted single target duration {target}s to range {min_duration}-{max_duration}s")
            else:
                # Fallback to default range
                min_duration, max_duration = 15, 60
                self.logger.warning(f"⚠️ Invalid target_duration format: {target_duration}, using default range {min_duration}-{max_duration}s")
            video_duration = video_info.get('duration', 600)
            
            self.logger.info(f"Generating ALL possible segments: duration {min_duration}-{max_duration}s from {video_duration:.1f}s video")
            
            # Detect and filter out pre-roll content
            pre_roll_end = self._detect_comprehensive_preroll(audio_analysis, video_duration)
            effective_start = pre_roll_end
            effective_end = video_duration - 10.0  # Leave 10s at end
            
            self.logger.info(f"Analysis range: {effective_start:.1f}s to {effective_end:.1f}s (excluding pre-roll)")
            
            all_segments = []
            
            # Strategy 0: Celebrity/actor-based segments (HIGHEST PRIORITY)
            if celebrity_index_path:
                try:
                    # Load celebrity JSON directly to avoid heavy package side-effects
                    from pathlib import Path
                    import json
                    raw = json.loads(Path(celebrity_index_path).read_text())
                    appearances_per_actor = {}
                    actor_conf = {}
                    for celeb in raw.get('celebrities', []):
                        name = celeb.get('name')
                        if not name:
                            continue
                        conf = float(celeb.get('confidence', 1.0))
                        times = sorted(int(a['timestamp_sec']) for a in celeb.get('appearances', []) if 'timestamp_sec' in a)
                        if times:
                            appearances_per_actor[name] = times
                            actor_conf[name] = conf

                    # Load object data - only segments with reference_match.score
                    appearances_per_object = {}
                    object_conf = {}
                    object_labels = {}
                    max_object_end = 0.0  # Track the latest object timestamp
                    for obj in raw.get('objects', []):
                        object_id = obj.get('object_id')
                        label = obj.get('label')
                        if not object_id or not label:
                            continue
                        
                        segments = []
                        for seg in obj.get('segments', []):
                            start_sec = seg.get('start_sec')
                            end_sec = seg.get('end_sec')
                            reference_match = seg.get('reference_match')
                            if start_sec is not None and end_sec is not None and reference_match and isinstance(reference_match, dict):
                                score = float(reference_match.get('score', 1.0))
                                segments.append({
                                    'start_sec': float(start_sec),
                                    'end_sec': float(end_sec),
                                    'score': score
                                })
                                # Track the latest object time to extend analysis range
                                max_object_end = max(max_object_end, float(end_sec))
                        
                        if segments:
                            appearances_per_object[object_id] = segments
                            # Use the highest score among segments for object confidence
                            max_score = max(seg['score'] for seg in segments)
                            object_conf[object_id] = max_score
                            object_labels[object_id] = label
                            self.logger.info(f"🎯 Loaded object {object_id} ({label}): {len(segments)} segments with max_score={max_score:.3f}")
                    
                    # CRITICAL FIX: If we have object reference segments, extend effective_end to include them
                    if max_object_end > 0:
                        # Extend effective_end to be at least 20s after the latest object (to have context)
                        # IMPORTANT: Do NOT cap to video_duration - trust the object timestamps from result JSON
                        adjusted_end = max_object_end + 50.0
                        # Only cap to video_duration if adjusted_end would exceed it significantly
                        # This allows objects near the end of the video to be included
                        if adjusted_end > video_duration:
                            adjusted_end = video_duration - 2.0  # Leave at least 2s margin
                        if adjusted_end > effective_end:
                            self.logger.info(f"🎯 EXTENDING ANALYSIS RANGE for object segments: {effective_end:.1f}s → {adjusted_end:.1f}s (last object at {max_object_end:.1f}s, video_duration={video_duration:.1f}s)")
                            effective_end = adjusted_end
                        else:
                            self.logger.info(f"🎯 Object end ({max_object_end:.1f}s) already within range {effective_start:.1f}s-{effective_end:.1f}s")

                    actor_segments = self._generate_actor_based_segments(
                        appearances_per_actor,
                        actor_conf,
                        effective_start,
                        effective_end,
                        min_duration,
                        max_duration
                    )

                    if actor_segments:
                        for s in actor_segments:
                            s.setdefault('method_priority', 0)
                            # Ensure celebrity metadata persists so these segments are prioritized
                            s.setdefault('has_celebrity', True)
                            s.setdefault('celebrity_score', 1.0)
                            s.setdefault('celebrity_actors', [s.get('actor_focus')])
                            s.setdefault('prompt_match_score', max(0.6, s.get('prompt_match_score', 0.6)))

                        # If caller requested actor-only segments, return early with just actor segments
                        if actor_only:
                            self.logger.info(f"Actor-only request: returning {len(actor_segments)} actor-based segments")
                            unique = self._deduplicate_comprehensive_segments(actor_segments, max_total_segments)
                            return self._enhance_segments_with_metadata(unique, audio_analysis, scene_analysis)

                        all_segments.extend(actor_segments)
                        self.logger.info(f"Generated {len(actor_segments)} actor-based segments from celebrity index")

                    # Generate object-based segments
                    object_segments = self._generate_object_based_segments(
                        appearances_per_object,
                        object_conf,
                        effective_start,
                        effective_end,
                        min_duration,
                        max_duration
                    )

                    if object_segments:
                        for s in object_segments:
                            s.setdefault('method_priority', 0)
                            # Ensure object metadata persists so these segments are prioritized
                            s.setdefault('has_object', True)
                            s.setdefault('object_score', s.get('object_score', 1.0))
                            s.setdefault('object_focus', s.get('object_focus'))
                            # HIGH SCORE FOR OBJECT REFERENCE SEGMENTS - ensure they pass all filters
                            s.setdefault('prompt_match_score', 0.95)
                            # CRITICAL FLAG: Mark as object reference segment for downstream processing
                            s['is_object_reference_segment'] = True

                        all_segments.extend(object_segments)
                        self.logger.info(f"Generated {len(object_segments)} object-based segments from celebrity index")
                        
                        # CRITICAL: If we have object reference segments, return them immediately without further analysis
                        if object_segments:
                            self.logger.info(f"🎯 OBJECT-REFERENCE MODE: Returning {len(object_segments)} object segments directly")
                            unique = self._deduplicate_comprehensive_segments(object_segments, max_total_segments)
                            return self._enhance_segments_with_metadata(unique, audio_analysis, scene_analysis)
                        
                except Exception as e:
                    self.logger.warning(f"Could not generate actor/object-based segments: {e}")
            
            # SEGMENT GENERATION PRIORITY ORDER (1=highest, 10=lowest):
            # 1. Content-aware segments (coherent topics/themes)
            # 2. Quality-driven segments (high-quality speech areas) & Scene-based segments (natural boundaries)  
            # 3. Audio-cue segments (speech patterns and pauses)
            # 4. Content-aware partial segments
            # 10. Sliding window segments (linear cuts - only as last resort)
            
            # Strategy 1: Content-aware segments (PRIORITY 1)
            content_segments = self._generate_content_aware_segments(
                audio_analysis, effective_start, effective_end, min_duration, max_duration
            )
            all_segments.extend(content_segments) 
            
            # Strategy 2: Quality-driven segments (PRIORITY 2)
            quality_segments = self._generate_quality_driven_segments(
                audio_analysis, effective_start, effective_end, min_duration, max_duration
            )
            all_segments.extend(quality_segments)
            
            # Strategy 3: Scene-based segments (PRIORITY 2) 
            scene_segments = self._generate_scene_based_segments(
                scene_analysis, effective_start, effective_end, min_duration, max_duration
            )
            all_segments.extend(scene_segments)
            
            # Strategy 4: Audio-cue based segments (PRIORITY 3)
            audio_segments = self._generate_audio_cue_segments(
                audio_analysis, effective_start, effective_end, min_duration, max_duration
            )
            all_segments.extend(audio_segments)
            
            # Strategy 5: Sliding window approach (PRIORITY 10 - FALLBACK ONLY)
            sliding_segments = self._generate_sliding_window_segments(
                effective_start, effective_end, min_duration, max_duration
            )
            all_segments.extend(sliding_segments)
            
            # Remove duplicates and overlaps
            unique_segments = self._deduplicate_comprehensive_segments(
                all_segments, max_total_segments
            )
            
            # Add metadata to each segment
            enhanced_segments = self._enhance_segments_with_metadata(
                unique_segments, audio_analysis, scene_analysis
            )
            
            self.logger.info(f"Generated {len(enhanced_segments)} unique segments from {len(all_segments)} total candidates")
            
            return enhanced_segments
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive segment generation: {e}")
            return []
    
    def _detect_comprehensive_preroll(self, audio_analysis: Dict, video_duration: float) -> float:
        """
        Comprehensive pre-roll detection using multiple signals.
        
        Args:
            audio_analysis: Audio analysis results
            video_duration: Total video duration
            
        Returns:
            End time of pre-roll content
        """
        # Start with explicit pre-roll detection from audio analysis
        pre_roll_end = 0.0
        
        # Check for explicit pre-roll segments
        likely_preroll = audio_analysis.get('zoom_analysis', {}).get('likely_preroll', [])
        if likely_preroll:
            pre_roll_end = max(segment['end'] for segment in likely_preroll)
        
        # Check transcription for intro patterns
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        # Look for intro/greeting patterns in first 60 seconds
        intro_threshold = min(60.0, video_duration * 0.15)  # First 15% or 60s
        
        for segment in segments:
            if segment['start'] > intro_threshold:
                break
                
            text = segment.get('text', '').lower()
            
            # Common intro patterns
            intro_patterns = [
                'welcome to', 'hello everyone', 'hi everyone', 'hey guys',
                'welcome back', 'in this video', "today we're going to",
                'before we start', 'let me introduce', 'this is', 'my name is'
            ]
            
            if any(pattern in text for pattern in intro_patterns):
                pre_roll_end = max(pre_roll_end, segment['end'])
        
        # Check for music-only segments
        music_segments = audio_analysis.get('zoom_analysis', {}).get('music_segments', [])
        for music_seg in music_segments:
            if music_seg['start'] < intro_threshold:
                pre_roll_end = max(pre_roll_end, music_seg['end'])
        
        # Check for silence at beginning
        silence_periods = transcription.get('silence_periods', [])
        for silence_start, silence_end in silence_periods:
            if silence_start < intro_threshold and silence_end - silence_start > 5.0:
                pre_roll_end = max(pre_roll_end, silence_end)
        
        # Ensure minimum content is preserved
        max_preroll = video_duration * 0.25  # Never exclude more than 25%
        pre_roll_end = min(pre_roll_end, max_preroll)
        
        return pre_roll_end
    
    def _generate_sliding_window_segments(self,
                                        start_time: float,
                                        end_time: float,
                                        min_duration: int,
                                        max_duration: int) -> List[Dict]:
        """Generate segments using sliding window approach."""
        segments = []
        step_size = self.config['sliding_window_step']
        
        # Generate segments of various durations
        durations = []
        
        # Fine-grained duration sampling
        for duration in range(min_duration, max_duration + 1, 5):  # Every 5 seconds
            durations.append(duration)
        
        # Add some intermediate durations for better coverage
        for duration in [min_duration + 2, min_duration + 7, 
                        (min_duration + max_duration) // 2,
                        max_duration - 7, max_duration - 2]:
            if min_duration <= duration <= max_duration:
                durations.append(duration)
        
        durations = sorted(set(durations))
        
        for duration in durations:
            current_start = start_time
            
            while current_start + duration <= end_time:
                segments.append({
                    'start_time': current_start,
                    'end_time': current_start + duration,
                    'duration': duration,
                    'generation_method': 'sliding_window',
                    'method_priority': 10  # Lowest priority - only as last resort fallback
                })
                
                current_start += step_size
        
        return segments
    
    def _generate_scene_based_segments(self,
                                     scene_analysis: Dict,
                                     start_time: float,
                                     end_time: float,
                                     min_duration: int,
                                     max_duration: int) -> List[Dict]:
        """Generate segments based on scene breaks and natural boundaries."""
        segments = []
        
        # Get scene breaks - handle both dict and float formats
        scene_breaks = scene_analysis.get('combined_breaks', [])
        break_points = []
        
        for bp in scene_breaks:
            if isinstance(bp, dict) and 'timestamp' in bp:
                timestamp = bp['timestamp']
            elif isinstance(bp, (int, float)):
                timestamp = float(bp)
            else:
                continue  # Skip invalid format
                
            if start_time <= timestamp <= end_time:
                break_points.append(timestamp)
        
        # Add start and end points
        all_points = [start_time] + break_points + [end_time]
        all_points = sorted(set(all_points))
        
        # Generate segments between scene breaks
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                seg_start = all_points[i]
                seg_end = all_points[j]
                duration = seg_end - seg_start
                
                if min_duration <= duration <= max_duration:
                    segments.append({
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'duration': duration,
                        'generation_method': 'scene_based',
                        'method_priority': 2,  # High priority - natural scene boundaries
                        'scene_break_count': j - i - 1
                    })
        
        return segments
    
    def _generate_audio_cue_segments(self,
                                   audio_analysis: Dict,
                                   start_time: float,
                                   end_time: float,
                                   min_duration: int,
                                   max_duration: int) -> List[Dict]:
        """Generate segments based on audio cues and speech patterns."""
        segments = []
        
        transcription = audio_analysis.get('transcription', {})
        transcript_segments = transcription.get('segments', [])
        
        # Find natural speech boundaries
        speech_breaks = []
        
        for i, segment in enumerate(transcript_segments):
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            if not (start_time <= seg_start <= end_time):
                continue
            
            # Add sentence endings as potential breaks
            text = segment.get('text', '')
            if any(punct in text for punct in ['.', '!', '?']):
                speech_breaks.append(seg_end)
            
            # Add pause points (gaps between segments)
            if i < len(transcript_segments) - 1:
                next_segment = transcript_segments[i + 1]
                gap = next_segment.get('start', 0) - seg_end
                if gap > 1.0:  # Significant pause
                    speech_breaks.append(seg_end)
        
        # Generate segments using speech boundaries
        all_points = [start_time] + speech_breaks + [end_time]
        all_points = sorted(set(all_points))
        
        for i in range(len(all_points)):
            for j in range(i + 1, len(all_points)):
                seg_start = all_points[i]
                seg_end = all_points[j]
                duration = seg_end - seg_start
                
                if min_duration <= duration <= max_duration:
                    segments.append({
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'duration': duration,
                        'generation_method': 'audio_cue',
                        'method_priority': 3,  # Good priority - speech patterns
                        'speech_break_count': j - i - 1
                    })
        
        return segments
    
    def _generate_content_aware_segments(self,
                                       audio_analysis: Dict,
                                       start_time: float,
                                       end_time: float,
                                       min_duration: int,
                                       max_duration: int) -> List[Dict]:
        """Generate segments based on content themes and topics."""
        segments = []
        
        transcription = audio_analysis.get('transcription', {})
        transcript_segments = transcription.get('segments', [])
        
        # Group segments by content similarity
        content_groups = self._group_segments_by_content(transcript_segments, start_time, end_time)
        
        # Also generate segments around high-value content markers
        high_value_segments = self._find_high_value_content(transcript_segments, start_time, end_time, min_duration, max_duration)
        segments.extend(high_value_segments)
        
        for group in content_groups:
            if len(group) < 1:  # Changed from 2 to 1 - single segments can be valuable too
                continue
                
            group_start = min(seg['start'] for seg in group)
            group_end = max(seg['end'] for seg in group)
            duration = group_end - group_start
            
            # Create segments within this content group
            if duration >= min_duration:
                # Full group segment
                if duration <= max_duration:
                    segments.append({
                        'start_time': group_start,
                        'end_time': group_end,
                        'duration': duration,
                        'generation_method': 'content_aware',
                        'method_priority': 1,  # Highest priority - content coherence
                        'content_coherence': True
                    })
                
                # Partial group segments
                for sub_duration in range(min_duration, min(max_duration + 1, int(duration))):
                    for offset in range(0, int(duration - sub_duration) + 1, 10):
                        seg_start = group_start + offset
                        seg_end = seg_start + sub_duration
                        
                        segments.append({
                            'start_time': seg_start,
                            'end_time': seg_end,
                            'duration': sub_duration,
                            'generation_method': 'content_aware_partial',
                            'method_priority': 3,  # Good priority - partial content coherence
                            'content_coherence': True
                        })
        
        return segments

    def _generate_actor_based_segments(self,
                                       appearances_per_actor: Dict[str, List[int]],
                                       actor_conf: Dict[str, float],
                                       start_time: float,
                                       end_time: float,
                                       min_duration: int,
                                       max_duration: int) -> List[Dict]:
        """Generate segments centered on celebrity appearance timestamps.
        Attempts to compute a reasonable celebrity_score based on actor_conf and timestamp coverage.
        """
        segments = []
        try:
            try:
                from ..face_insights.celebrity_index import actor_coverage_for_segment, compute_celebrity_score
            except Exception:
                from src.face_insights.celebrity_index import actor_coverage_for_segment, compute_celebrity_score
        except Exception:
            actor_coverage_for_segment = None
            compute_celebrity_score = None

        for actor, timestamps in appearances_per_actor.items():
            for t in timestamps:
                t = float(t)
                if t < start_time or t > end_time:
                    continue
                # Primary short segment centered on timestamp
                duration = max(min_duration, min(max_duration, min_duration))
                start = max(start_time, t - duration / 2.0)
                end = min(end_time, start + duration)

                seg = {
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'generation_method': 'celebrity_appearance',
                    'method_priority': 0,
                    'actor_focus': actor,
                    'appearance_timestamp_sec': int(t)
                }

                # Compute celebrity score for this specific segment if helpers available
                if actor_coverage_for_segment and compute_celebrity_score:
                    per_actor = actor_coverage_for_segment(start, end, appearances_per_actor, actor_conf)
                    if per_actor:
                        seg['celebrity_actors'] = sorted(per_actor.items(), key=lambda x: x[1]['coverage'], reverse=True)
                        seg['celebrity_score'] = float(compute_celebrity_score(per_actor))
                        seg['has_celebrity'] = seg['celebrity_score'] > 0
                        seg['top_celebrity'] = seg['celebrity_actors'][0][0] if seg['celebrity_actors'] else None

                # Baseline prompt match to ensure survive downstream filtering
                seg['prompt_match_score'] = seg.get('prompt_match_score', 0.6)
                segments.append(seg)

                # Secondary longer option
                long_duration = min(max_duration, duration * 2)
                if long_duration > duration:
                    start2 = max(start_time, t - long_duration / 2.0)
                    end2 = min(end_time, start2 + long_duration)
                    seg2 = {
                        'start_time': start2,
                        'end_time': end2,
                        'duration': end2 - start2,
                        'generation_method': 'celebrity_appearance_long',
                        'method_priority': 0,
                        'actor_focus': actor,
                        'appearance_timestamp_sec': int(t)
                    }
                    if actor_coverage_for_segment and compute_celebrity_score:
                        per_actor2 = actor_coverage_for_segment(start2, end2, appearances_per_actor, actor_conf)
                        if per_actor2:
                            seg2['celebrity_actors'] = sorted(per_actor2.items(), key=lambda x: x[1]['coverage'], reverse=True)
                            seg2['celebrity_score'] = float(compute_celebrity_score(per_actor2))
                            seg2['has_celebrity'] = seg2['celebrity_score'] > 0
                            seg2['top_celebrity'] = seg2['celebrity_actors'][0][0] if seg2.get('celebrity_actors') else None
                    seg2['prompt_match_score'] = seg2.get('prompt_match_score', 0.5)
                    segments.append(seg2)
        return segments

    def _generate_object_based_segments(self,
                                       appearances_per_object: Dict[str, List[Dict]],
                                       object_conf: Dict[str, float],
                                       start_time: float,
                                       end_time: float,
                                       min_duration: int,
                                       max_duration: int) -> List[Dict]:
        """Generate segments centered on object appearance segments.
        Uses object scores from reference_match to prioritize segments.
        """
        self.logger.info(f"🎯 GENERATING OBJECT SEGMENTS: {len(appearances_per_object)} objects to process")
        segments = []
        try:
            try:
                from ..face_insights.celebrity_index import object_coverage_for_segment, compute_object_score
            except Exception:
                from src.face_insights.celebrity_index import object_coverage_for_segment, compute_object_score
        except Exception:
            object_coverage_for_segment = None
            compute_object_score = None

        for object_id, object_segments in appearances_per_object.items():
            self.logger.info(f"🎯 Processing object {object_id}: {len(object_segments)} segments")
            for obj_seg in object_segments:
                obj_start = obj_seg['start_sec']
                obj_end = obj_seg['end_sec']
                score = obj_seg['score']
                
                # Skip if object segment doesn't overlap with analysis range
                if obj_end < start_time or obj_start > end_time:
                    continue
                
                # Primary short segment centered on object appearance
                duration = max(min_duration, min(max_duration, min_duration))
                center_time = (obj_start + obj_end) / 2.0
                start = max(start_time, center_time - duration / 2.0)
                end = min(end_time, start + duration)

                seg = {
                    'start_time': start,
                    'end_time': end,
                    'duration': end - start,
                    'generation_method': 'object_appearance',
                    'method_priority': 0,
                    'object_focus': object_id,
                    'object_appearance_start': obj_start,
                    'object_appearance_end': obj_end,
                    'object_score': score
                }

                # Compute object score for this specific segment if helpers available
                if object_coverage_for_segment and compute_object_score:
                    per_object = object_coverage_for_segment(start, end, appearances_per_object, object_conf)
                    if per_object:
                        seg['object_details'] = sorted(per_object.items(), key=lambda x: x[1]['score'], reverse=True)
                        seg['object_score'] = float(compute_object_score(per_object))
                        seg['has_object'] = seg['object_score'] > 0
                        seg['top_object'] = seg['object_details'][0][0] if seg['object_details'] else None

                # Baseline prompt match to ensure survive downstream filtering
                seg['prompt_match_score'] = seg.get('prompt_match_score', 0.9)
                segments.append(seg)
                self.logger.info(f"🎯 Added object segment: {object_id} @ {start:.1f}-{end:.1f}s (object @ {obj_start}-{obj_end}s)")


                # Secondary longer option if object segment is long enough
                obj_duration = obj_end - obj_start
                if obj_duration > duration:
                    long_duration = min(max_duration, obj_duration)
                    if long_duration > duration:
                        start2 = max(start_time, obj_start)
                        end2 = min(end_time, obj_start + long_duration)
                        seg2 = {
                            'start_time': start2,
                            'end_time': end2,
                            'duration': end2 - start2,
                            'generation_method': 'object_appearance_long',
                            'method_priority': 0,
                            'object_focus': object_id,
                            'object_appearance_start': obj_start,
                            'object_appearance_end': obj_end,
                            'object_score': score
                        }
                        if object_coverage_for_segment and compute_object_score:
                            per_object2 = object_coverage_for_segment(start2, end2, appearances_per_object, object_conf)
                            if per_object2:
                                seg2['object_details'] = sorted(per_object2.items(), key=lambda x: x[1]['score'], reverse=True)
                                seg2['object_score'] = float(compute_object_score(per_object2))
                                seg2['has_object'] = seg2['object_score'] > 0
                                seg2['top_object'] = seg2['object_details'][0][0] if seg2.get('object_details') else None
                        seg2['prompt_match_score'] = seg2.get('prompt_match_score', 0.8)
                        segments.append(seg2)
        self.logger.info(f"🎯 GENERATED {len(segments)} total object segments")
        return segments

    def _generate_quality_driven_segments(self,
                                        audio_analysis: Dict,
                                        start_time: float,
                                        end_time: float,
                                        min_duration: int,
                                        max_duration: int) -> List[Dict]:
        """Generate segments focusing on high-quality content areas."""
        segments = []
        
        transcription = audio_analysis.get('transcription', {})
        transcript_segments = transcription.get('segments', [])
        
        # Identify high-quality speech segments
        quality_segments = []
        
        for segment in transcript_segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            if not (start_time <= seg_start <= end_time):
                continue
            
            text = segment.get('text', '')
            
            # Quality indicators
            quality_score = 0.0
            
            # Clear speech (word count)
            word_count = len(text.split())
            if word_count >= 5:
                quality_score += 0.3
            
            # Complete sentences
            if any(punct in text for punct in ['.', '!', '?']):
                quality_score += 0.2
            
            # Engaging content
            engagement_words = ['important', 'key', 'amazing', 'incredible', 'secret', 'tip']
            if any(word in text.lower() for word in engagement_words):
                quality_score += 0.3
            
            # Direct address
            if any(word in text.lower() for word in ['you', 'your', 'we', "let's"]):
                quality_score += 0.2
            
            if quality_score >= 0.4:  # Threshold for quality
                quality_segments.append({
                    'start': seg_start,
                    'end': seg_end,
                    'quality_score': quality_score,
                    'text': text
                })
        
        # Generate segments around high-quality areas
        for qual_seg in quality_segments:
            center_time = (qual_seg['start'] + qual_seg['end']) / 2
            
            # Generate segments centered on quality content
            for duration in range(min_duration, max_duration + 1, 10):
                seg_start = max(start_time, center_time - duration / 2)
                seg_end = min(end_time, seg_start + duration)
                
                if seg_end - seg_start >= min_duration:
                    segments.append({
                        'start_time': seg_start,
                        'end_time': seg_end,
                        'duration': seg_end - seg_start,
                        'generation_method': 'quality_driven',
                        'method_priority': 2,  # High priority - quality content
                        'quality_score': qual_seg['quality_score'],
                        'contains_quality_content': True
                    })
        
        return segments
    
    def _group_segments_by_content(self, 
                                 transcript_segments: List[Dict],
                                 start_time: float,
                                 end_time: float) -> List[List[Dict]]:
        """Group transcript segments by content similarity."""
        # Simple content grouping based on keywords and timing
        groups = []
        current_group = []
        
        for segment in transcript_segments:
            seg_start = segment.get('start', 0)
            
            if not (start_time <= seg_start <= end_time):
                continue
            
            text = segment.get('text', '').lower()
            
            # If current group is empty or content is similar, add to group
            if not current_group:
                current_group.append(segment)
            else:
                # Check content similarity with last segment in group
                last_text = current_group[-1].get('text', '').lower()
                
                # Simple similarity check (common words)
                common_words = set(text.split()) & set(last_text.split())
                similarity = len(common_words) / max(len(text.split()), len(last_text.split()))
                
                # Check temporal proximity
                time_gap = seg_start - current_group[-1].get('end', 0)
                
                if similarity > 0.2 or time_gap < 5.0:  # Similar content or close in time
                    current_group.append(segment)
                else:
                    # Start new group
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [segment]
        
        # Add final group
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def _deduplicate_comprehensive_segments(self, 
                                          all_segments: List[Dict],
                                          max_segments: int) -> List[Dict]:
        """
        Remove duplicate and heavily overlapping segments while preserving diversity.
        
        Args:
            all_segments: All generated segments
            max_segments: Maximum number of segments to keep
            
        Returns:
            Deduplicated segments
        """
        if not all_segments:
            return []
        
        # Sort by comprehensive intelligence metrics with ENHANCED visual weighting
        def comprehensive_score(segment):
            # Priority 1: Method priority (content_aware=1, sliding_window=10)
            method_priority = segment.get('method_priority', 10)
            
            # Priority 2: Quality and content intelligence
            quality_score = segment.get('quality_score', 0.5)
            text_quality = segment.get('text_quality_score', 0.5)
            engagement = segment.get('engagement_score', 0.5)
            
            # Priority 3: Vision and audio intelligence - ENHANCED
            vision_score = segment.get('vision_score', 0.0)
            visual_interest = segment.get('visual_interest', 0.0)
            
            # Normalize vision scores (handle both 0-1 and 1-10 scales)
            if isinstance(vision_score, (int, float)) and vision_score > 1:
                vision_score = vision_score / 10.0
            if isinstance(visual_interest, (int, float)) and visual_interest > 1:
                visual_interest = visual_interest / 10.0
            
            # Combined visual score with fallback
            vision_normalized = max(vision_score, visual_interest, 0.0)
            if vision_normalized == 0:
                vision_normalized = 0.5  # Default only if no vision data
            
            # Scene type bonus for dynamic/engaging content
            scene_bonus = 0.0
            scene_type = segment.get('scene_type', 'unknown')
            if scene_type in ['action', 'demonstration']:
                scene_bonus = 0.10
            elif scene_type in ['presentation', 'interview', 'talking_head']:
                scene_bonus = 0.05
            
            # Content-aware specific bonuses
            content_bonus = 0.0
            if segment.get('generation_method') == 'content_aware':
                content_bonus += 0.12  # Strong bonus for content-aware
                if segment.get('high_value_content'):
                    content_bonus += 0.08  # Additional bonus for high-value content
                if segment.get('trigger_keywords'):
                    content_bonus += 0.04  # Bonus for keyword triggers
            
            # Audio transcription quality bonus
            if segment.get('has_complete_sentences', False):
                content_bonus += 0.04
            if segment.get('word_count', 0) > 100:  # Substantial content
                content_bonus += 0.02
            
            # People/visual interest bonus - INCREASED
            people_bonus = 0.0
            if segment.get('people_visible') or segment.get('people_count', 0) > 0:
                people_bonus = 0.08
                # Extra bonus for multiple people (social/conversation content)
                if segment.get('people_count', 0) > 1:
                    people_bonus += 0.04
            
            # Motion/action bonus
            motion_bonus = 0.05 if segment.get('has_action', False) else 0.0
            
            # Calculate composite intelligence score - REBALANCED for visual
            intelligence_score = (
                quality_score * 0.18 +      # Reduced from 0.25
                text_quality * 0.15 +       # Reduced from 0.20
                engagement * 0.15 +         # Reduced from 0.20
                vision_normalized * 0.25 +  # INCREASED from 0.15
                scene_bonus +               # NEW
                content_bonus +
                people_bonus +              # INCREASED
                motion_bonus                # NEW
            )
            
            # Return tuple for sorting: (method_priority, -intelligence_score, -duration)
            # Lower method_priority is better, higher intelligence_score is better
            return (method_priority, -intelligence_score, -segment.get('duration', 0))
        
        all_segments.sort(key=comprehensive_score)
        
        unique_segments = []
        overlap_threshold = self.config['overlap_tolerance']
        content_sim_threshold = self.config.get('content_similarity_threshold', 0.6)
        
        for segment in all_segments:
            if len(unique_segments) >= max_segments:
                break
            
            # Check overlap with existing segments - ENHANCED with content similarity
            should_skip = False
            for existing in unique_segments:
                overlap_ratio = self._calculate_overlap_ratio(segment, existing)
                
                # Only skip if BOTH temporal overlap AND content similarity are high
                if overlap_ratio > overlap_threshold:
                    content_similarity = self._calculate_content_similarity(segment, existing)
                    if content_similarity > content_sim_threshold:
                        should_skip = True
                        break
                    else:
                        # Overlapping but DIFFERENT content - log and potentially keep
                        self.logger.debug(
                            f"Keeping overlapping segment - different content "
                            f"(overlap: {overlap_ratio:.2f}, content_sim: {content_similarity:.2f})"
                        )
            
            if not should_skip:
                unique_segments.append(segment)
        
        # If we still have space, add more segments with slightly higher overlap tolerance
        if len(unique_segments) < max_segments * 0.8:  # If we have less than 80% capacity
            higher_threshold = overlap_threshold + 0.2
            
            for segment in all_segments:
                if segment in unique_segments:
                    continue
                    
                if len(unique_segments) >= max_segments:
                    break
                
                should_skip = False
                for existing in unique_segments:
                    overlap_ratio = self._calculate_overlap_ratio(segment, existing)
                    if overlap_ratio > higher_threshold:
                        content_similarity = self._calculate_content_similarity(segment, existing)
                        if content_similarity > content_sim_threshold:
                            should_skip = True
                            break
                
                if not should_skip:
                    unique_segments.append(segment)
        
        self.logger.info(f"Content-aware dedup: {len(all_segments)} → {len(unique_segments)} segments")
        return unique_segments
    
    def _calculate_content_similarity(self, seg1: Dict, seg2: Dict) -> float:
        """
        Calculate content similarity between two segments.
        Returns 0.0 (completely different) to 1.0 (identical content).
        """
        similarity_score = 0.0
        weights_used = 0.0
        
        # 1. Keyword overlap (weight: 0.3)
        keywords1 = set(seg1.get('trigger_keywords', []) or [])
        keywords2 = set(seg2.get('trigger_keywords', []) or [])
        if keywords1 or keywords2:
            keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
            similarity_score += keyword_overlap * 0.3
            weights_used += 0.3
        
        # 2. Text content similarity (weight: 0.5)
        text1 = seg1.get('segment_text', seg1.get('text', '')).lower()
        text2 = seg2.get('segment_text', seg2.get('text', '')).lower()
        if text1 and text2:
            words1 = set(text1.split())
            words2 = set(text2.split())
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'it', 'of', 'that', 'this'}
            words1 = words1 - stop_words
            words2 = words2 - stop_words
            if words1 or words2:
                text_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
                similarity_score += text_overlap * 0.5
                weights_used += 0.5
        
        # 3. Generation method match (weight: 0.2)
        method1 = seg1.get('generation_method', 'unknown')
        method2 = seg2.get('generation_method', 'unknown')
        if method1 != 'unknown' or method2 != 'unknown':
            method_match = 1.0 if method1 == method2 else 0.0
            similarity_score += method_match * 0.2
            weights_used += 0.2
        
        if weights_used > 0:
            return similarity_score / weights_used
        return 0.5  # Default if no content info
    
    def _calculate_overlap_ratio(self, seg1: Dict, seg2: Dict) -> float:
        """Calculate overlap ratio between two segments."""
        start1, end1 = seg1['start_time'], seg1['end_time']
        start2, end2 = seg2['start_time'], seg2['end_time']
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        min_duration = min(end1 - start1, end2 - start2)
        
        return overlap_duration / min_duration if min_duration > 0 else 0.0
    
    def _enhance_segments_with_metadata(self,
                                      segments: List[Dict],
                                      audio_analysis: Dict,
                                      scene_analysis: Dict) -> List[Dict]:
        """Add comprehensive metadata to each segment."""
        enhanced_segments = []
        
        transcription = audio_analysis.get('transcription', {})
        transcript_segments = transcription.get('segments', [])
        
        for segment in segments:
            enhanced_segment = segment.copy()
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            # Extract text content
            segment_text = self._extract_segment_text(transcript_segments, start_time, end_time)
            enhanced_segment['segment_text'] = segment_text
            
            # Calculate text-based metrics
            if segment_text:
                word_count = len(segment_text.split())
                sentence_count = segment_text.count('.') + segment_text.count('!') + segment_text.count('?')
                
                enhanced_segment.update({
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'text_density': word_count / segment['duration'] if segment['duration'] > 0 else 0,
                    'has_complete_sentences': sentence_count > 0,
                    'text_quality_score': self._calculate_text_quality(segment_text)
                })
            else:
                enhanced_segment.update({
                    'word_count': 0,
                    'sentence_count': 0,
                    'text_density': 0,
                    'has_complete_sentences': False,
                    'text_quality_score': 0.0
                })
            
            # Add scene break information
            scene_breaks_in_segment = self._count_scene_breaks_in_segment(
                scene_analysis, start_time, end_time
            )
            enhanced_segment['scene_breaks_count'] = scene_breaks_in_segment
            
            # Add preliminary quality assessment
            enhanced_segment['preliminary_quality'] = self._assess_preliminary_quality(enhanced_segment)

            # Preserve and boost celebrity metadata if present so actor segments rank higher
            if enhanced_segment.get('actor_focus') or enhanced_segment.get('has_celebrity'):
                enhanced_segment.setdefault('has_celebrity', True)
                enhanced_segment.setdefault('celebrity_score', enhanced_segment.get('celebrity_score', 0.9))
                enhanced_segment.setdefault('celebrity_actors', enhanced_segment.get('celebrity_actors', [enhanced_segment.get('actor_focus')]))
                # Boost preliminary quality and ensure prompt_match baseline so these survive filtering
                enhanced_segment['preliminary_quality'] = min(1.0, enhanced_segment.get('preliminary_quality', 0) + 0.2)
                enhanced_segment['prompt_match_score'] = max(enhanced_segment.get('prompt_match_score', 0.0), 0.6)

            enhanced_segments.append(enhanced_segment)
        
        return enhanced_segments
    
    def _extract_segment_text(self, transcript_segments: List[Dict], start_time: float, end_time: float) -> str:
        """Extract text from transcript segments within time range."""
        text_parts = []
        
        for seg in transcript_segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            
            # Check for overlap
            if seg_start < end_time and seg_end > start_time:
                text = seg.get('text', '').strip()
                if text:
                    text_parts.append(text)
        
        return ' '.join(text_parts)
    
    def _find_high_value_content(self, transcript_segments: List[Dict], start_time: float, end_time: float, min_duration: int, max_duration: int) -> List[Dict]:
        """Find high-value content segments based on keywords and patterns."""
        segments = []
        
        # High-value keywords that indicate interesting content
        high_value_keywords = [
            'amazing', 'incredible', 'breakthrough', 'important', 'key', 'secret',
            'shocking', 'surprising', 'reveal', 'discover', 'explain', 'show',
            'first', 'best', 'worst', 'never', 'always', 'must', 'should'
        ]
        
        for i, segment in enumerate(transcript_segments):
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').lower()
            
            if not (start_time <= seg_start <= end_time):
                continue
            
            # Check for high-value keywords
            has_high_value = any(keyword in text for keyword in high_value_keywords)
            
            if has_high_value:
                # Create segments around this high-value content
                for duration in range(min_duration, max_duration + 1, 15):
                    # Center the segment around the high-value content
                    center_time = (seg_start + seg_end) / 2
                    segment_start = max(start_time, center_time - duration / 2)
                    segment_end = min(end_time, segment_start + duration)
                    
                    if segment_end - segment_start >= min_duration:
                        segments.append({
                            'start_time': segment_start,
                            'end_time': segment_end,
                            'duration': segment_end - segment_start,
                            'generation_method': 'content_aware',
                            'method_priority': 1,  # Highest priority
                            'high_value_content': True,
                            'trigger_keywords': [kw for kw in high_value_keywords if kw in text]
                        })
        
        return segments
    
    def _calculate_text_quality(self, text: str) -> float:
        """Calculate text quality score."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Length factor
        word_count = len(text.split())
        if 5 <= word_count <= 50:
            score += 0.3
        elif word_count > 0:
            score += 0.1
        
        # Completeness (sentences)
        if any(punct in text for punct in ['.', '!', '?']):
            score += 0.3
        
        # Engagement indicators
        engagement_words = ['important', 'key', 'amazing', 'incredible', 'you', 'we']
        engagement_count = sum(1 for word in engagement_words if word in text.lower())
        score += min(0.2, engagement_count * 0.1)
        
        # Clarity (not too fragmented)
        if not text.endswith(('...', '-', ',')):
            score += 0.2
        
        return min(1.0, score)
    
    def _count_scene_breaks_in_segment(self, scene_analysis: Dict, start_time: float, end_time: float) -> int:
        """Count scene breaks within segment."""
        scene_breaks = scene_analysis.get('combined_breaks', [])
        
        count = 0
        for break_point in scene_breaks:
            if isinstance(break_point, dict) and 'timestamp' in break_point:
                timestamp = break_point['timestamp']
            elif isinstance(break_point, (int, float)):
                timestamp = float(break_point)
            else:
                continue  # Skip invalid format
                
            if start_time < timestamp < end_time:
                count += 1
        
        return count
    
    def _assess_preliminary_quality(self, segment: Dict) -> float:
        """Assess preliminary quality of segment."""
        score = 0.0
        
        # Text quality weight: 40%
        text_quality = segment.get('text_quality_score', 0)
        score += text_quality * 0.4
        
        # Duration appropriateness weight: 20%
        duration = segment.get('duration', 0)
        if 20 <= duration <= 90:
            score += 0.2
        elif 15 <= duration <= 120:
            score += 0.1
        
        # Method priority weight: 20%
        method_priority = segment.get('method_priority', 10)
        priority_score = max(0, (10 - method_priority) / 10)
        score += priority_score * 0.2
        
        # Content coherence weight: 20%
        if segment.get('content_coherence', False):
            score += 0.15
        if segment.get('contains_quality_content', False):
            score += 0.05
        
        return min(1.0, score)
















# # src/content_analysis/comprehensive_segment_generator.py
# """Comprehensive segment generator that analyzes ALL potential segments for maximum quality"""

# import logging
# import numpy as np
# from typing import Dict, List, Tuple, Optional
# from itertools import combinations


# class ComprehensiveSegmentGenerator:
#     """
#     Generates and analyzes ALL possible video segments within target duration ranges
#     to ensure maximum coverage and quality in segment selection.
#     """
    
#     def __init__(self):
#         """Initialize comprehensive segment generator."""
#         self.logger = logging.getLogger(__name__)
        
#         # Configuration for comprehensive analysis
#         self.config = {
#             'sliding_window_step': 5.0,    # Generate segments every 5 seconds
#             'overlap_tolerance': 0.5,       # INCREASED: Allow 50% overlap to preserve diverse content
#             'quality_sampling_rate': 2.0,   # Sample every 2 seconds for quality assessment
#             'max_segments_per_duration': 50, # Maximum segments per duration range
#             'enable_micro_segments': True,   # Include very short segments
#             'enable_extended_segments': True, # Include longer segments
#             'content_similarity_threshold': 0.9,  # NEW: Only dedupe if content is >60% similar
#             'min_output_gap': 20.0  # Minimum seconds between final output segments to reduce duplicates
#         }
    
#     def generate_all_possible_segments(self,
#                                      video_info: Dict,
#                                      audio_analysis: Dict,
#                                      scene_analysis: Dict,
#                                      target_duration: Tuple[int, int] = (15, 60),
#                                      max_total_segments: int = 200,
#                                      celebrity_index_path: Optional[str] = None,
#                                      actor_only: bool = True) -> List[Dict]:
#         """
#         Generate ALL possible segments within target duration using multiple strategies.
#         If `celebrity_index_path` is provided, generate high-priority actor-based segments
#         from the celebrity appearances so prompt-based actor requests can be satisfied.
        
#         Args:
#             video_info: Video metadata
#             audio_analysis: Audio analysis results
#             scene_analysis: Scene detection results
#             target_duration: Target duration range (min, max) in seconds OR single target duration
#             max_total_segments: Maximum total segments to generate
#             celebrity_index_path: Optional path to celebrity result JSON to prioritize actor segments
            
#         Returns:
#             Comprehensive list of all possible segments
#         """
#         try:
#             # Handle both tuple (min, max) and single integer target_duration
#             if isinstance(target_duration, (tuple, list)) and len(target_duration) == 2:
#                 min_duration, max_duration = target_duration
#             elif isinstance(target_duration, (int, float)):
#                 # Single duration - create reasonable range around it
#                 target = int(target_duration)
#                 min_duration = max(15, target - 15)  # At least 15s minimum
#                 max_duration = target + 15  # Allow up to 15s longer
#                 self.logger.info(f"🔧 Converted single target duration {target}s to range {min_duration}-{max_duration}s")
#             else:
#                 # Fallback to default range
#                 min_duration, max_duration = 15, 60
#                 self.logger.warning(f"⚠️ Invalid target_duration format: {target_duration}, using default range {min_duration}-{max_duration}s")
#             video_duration = video_info.get('duration', 600)
            
#             self.logger.info(f"Generating ALL possible segments: duration {min_duration}-{max_duration}s from {video_duration:.1f}s video")
            
#             # Detect and filter out pre-roll content
#             pre_roll_end = self._detect_comprehensive_preroll(audio_analysis, video_duration)
#             effective_start = pre_roll_end
#             effective_end = video_duration - 10.0  # Leave 10s at end
            
#             self.logger.info(f"Analysis range: {effective_start:.1f}s to {effective_end:.1f}s (excluding pre-roll)")
            
#             all_segments = []
            
#             # Strategy 0: Celebrity/actor-based segments (HIGHEST PRIORITY)
#             if celebrity_index_path:
#                 try:
#                     # Load celebrity JSON directly to avoid heavy package side-effects
#                     from pathlib import Path
#                     import json
#                     raw = json.loads(Path(celebrity_index_path).read_text())
#                     appearances_per_actor = {}
#                     actor_conf = {}
#                     for celeb in raw.get('celebrities', []):
#                         name = celeb.get('name')
#                         if not name:
#                             continue
#                         conf = float(celeb.get('confidence', 1.0))
#                         times = sorted(int(a['timestamp_sec']) for a in celeb.get('appearances', []) if 'timestamp_sec' in a)
#                         if times:
#                             appearances_per_actor[name] = times
#                             actor_conf[name] = conf

#                     actor_segments = self._generate_actor_based_segments(
#                         appearances_per_actor,
#                         actor_conf,
#                         effective_start,
#                         effective_end,
#                         min_duration,
#                         max_duration
#                     )

#                     if actor_segments:
#                         for s in actor_segments:
#                             s.setdefault('method_priority', 0)
#                             # Ensure celebrity metadata persists so these segments are prioritized
#                             s.setdefault('has_celebrity', True)
#                             s.setdefault('celebrity_score', 1.0)
#                             s.setdefault('celebrity_actors', [s.get('actor_focus')])
#                             s.setdefault('prompt_match_score', max(0.6, s.get('prompt_match_score', 0.6)))

#                         # If caller requested actor-only segments, return early with just actor segments
#                         if actor_only:
#                             self.logger.info(f"Actor-only request: returning {len(actor_segments)} actor-based segments")
#                             unique = self._deduplicate_comprehensive_segments(actor_segments, max_total_segments)
#                             #unique = actor_segments[:max_total_segments]
#                             return self._enhance_segments_with_metadata(unique, audio_analysis, scene_analysis)

#                         all_segments.extend(actor_segments)
#                         self.logger.info(f"Generated {len(actor_segments)} actor-based segments from celebrity index")
#                 except Exception as e:
#                     self.logger.warning(f"Could not generate actor-based segments: {e}")
            
#             # SEGMENT GENERATION PRIORITY ORDER (1=highest, 10=lowest):
#             # 1. Content-aware segments (coherent topics/themes)
#             # 2. Quality-driven segments (high-quality speech areas) & Scene-based segments (natural boundaries)  
#             # 3. Audio-cue segments (speech patterns and pauses)
#             # 4. Content-aware partial segments
#             # 10. Sliding window segments (linear cuts - only as last resort)
            
#             # Strategy 1: Content-aware segments (PRIORITY 1)
#             content_segments = self._generate_content_aware_segments(
#                 audio_analysis, effective_start, effective_end, min_duration, max_duration
#             )
#             all_segments.extend(content_segments) 
            
#             # Strategy 2: Quality-driven segments (PRIORITY 2)
#             quality_segments = self._generate_quality_driven_segments(
#                 audio_analysis, effective_start, effective_end, min_duration, max_duration
#             )
#             all_segments.extend(quality_segments)
            
#             # Strategy 3: Scene-based segments (PRIORITY 2) 
#             scene_segments = self._generate_scene_based_segments(
#                 scene_analysis, effective_start, effective_end, min_duration, max_duration
#             )
#             all_segments.extend(scene_segments)
            
#             # Strategy 4: Audio-cue based segments (PRIORITY 3)
#             audio_segments = self._generate_audio_cue_segments(
#                 audio_analysis, effective_start, effective_end, min_duration, max_duration
#             )
#             all_segments.extend(audio_segments)
            
#             # Strategy 5: Sliding window approach (PRIORITY 10 - FALLBACK ONLY)
#             sliding_segments = self._generate_sliding_window_segments(
#                 effective_start, effective_end, min_duration, max_duration
#             )
#             all_segments.extend(sliding_segments)
            
#             # Remove duplicates and overlaps
#             unique_segments = self._deduplicate_comprehensive_segments(
#                 all_segments, max_total_segments
#             )
            
#             # Add metadata to each segment
#             enhanced_segments = self._enhance_segments_with_metadata(
#                 unique_segments, audio_analysis, scene_analysis
#             )
            
#             self.logger.info(f"Generated {len(enhanced_segments)} unique segments from {len(all_segments)} total candidates")
            
#             return enhanced_segments
            
#         except Exception as e:
#             self.logger.error(f"Error in comprehensive segment generation: {e}")
#             return []
    
#     def _detect_comprehensive_preroll(self, audio_analysis: Dict, video_duration: float) -> float:
#         """
#         Comprehensive pre-roll detection using multiple signals.
        
#         Args:
#             audio_analysis: Audio analysis results
#             video_duration: Total video duration
            
#         Returns:
#             End time of pre-roll content
#         """
#         # Start with explicit pre-roll detection from audio analysis
#         pre_roll_end = 0.0
        
#         # Check for explicit pre-roll segments
#         likely_preroll = audio_analysis.get('zoom_analysis', {}).get('likely_preroll', [])
#         if likely_preroll:
#             pre_roll_end = max(segment['end'] for segment in likely_preroll)
        
#         # Check transcription for intro patterns
#         transcription = audio_analysis.get('transcription', {})
#         segments = transcription.get('segments', [])
        
#         # Look for intro/greeting patterns in first 60 seconds
#         intro_threshold = min(60.0, video_duration * 0.15)  # First 15% or 60s
        
#         for segment in segments:
#             if segment['start'] > intro_threshold:
#                 break
                
#             text = segment.get('text', '').lower()
            
#             # Common intro patterns
#             intro_patterns = [
#                 'welcome to', 'hello everyone', 'hi everyone', 'hey guys',
#                 'welcome back', 'in this video', "today we're going to",
#                 'before we start', 'let me introduce', 'this is', 'my name is'
#             ]
            
#             if any(pattern in text for pattern in intro_patterns):
#                 pre_roll_end = max(pre_roll_end, segment['end'])
        
#         # Check for music-only segments
#         music_segments = audio_analysis.get('zoom_analysis', {}).get('music_segments', [])
#         for music_seg in music_segments:
#             if music_seg['start'] < intro_threshold:
#                 pre_roll_end = max(pre_roll_end, music_seg['end'])
        
#         # Check for silence at beginning
#         silence_periods = transcription.get('silence_periods', [])
#         for silence_start, silence_end in silence_periods:
#             if silence_start < intro_threshold and silence_end - silence_start > 5.0:
#                 pre_roll_end = max(pre_roll_end, silence_end)
        
#         # Ensure minimum content is preserved
#         max_preroll = video_duration * 0.25  # Never exclude more than 25%
#         pre_roll_end = min(pre_roll_end, max_preroll)
        
#         return pre_roll_end
    
#     def _generate_sliding_window_segments(self,
#                                         start_time: float,
#                                         end_time: float,
#                                         min_duration: int,
#                                         max_duration: int) -> List[Dict]:
#         """Generate segments using sliding window approach."""
#         segments = []
#         step_size = self.config['sliding_window_step']
        
#         # Generate segments of various durations
#         durations = []
        
#         # Fine-grained duration sampling
#         for duration in range(min_duration, max_duration + 1, 5):  # Every 5 seconds
#             durations.append(duration)
        
#         # Add some intermediate durations for better coverage
#         for duration in [min_duration + 2, min_duration + 7, 
#                         (min_duration + max_duration) // 2,
#                         max_duration - 7, max_duration - 2]:
#             if min_duration <= duration <= max_duration:
#                 durations.append(duration)
        
#         durations = sorted(set(durations))
        
#         for duration in durations:
#             current_start = start_time
            
#             while current_start + duration <= end_time:
#                 segments.append({
#                     'start_time': current_start,
#                     'end_time': current_start + duration,
#                     'duration': duration,
#                     'generation_method': 'sliding_window',
#                     'method_priority': 10  # Lowest priority - only as last resort fallback
#                 })
                
#                 current_start += step_size
        
#         return segments
    
#     def _generate_scene_based_segments(self,
#                                      scene_analysis: Dict,
#                                      start_time: float,
#                                      end_time: float,
#                                      min_duration: int,
#                                      max_duration: int) -> List[Dict]:
#         """Generate segments based on scene breaks and natural boundaries."""
#         segments = []
        
#         # Get scene breaks - handle both dict and float formats
#         scene_breaks = scene_analysis.get('combined_breaks', [])
#         break_points = []
        
#         for bp in scene_breaks:
#             if isinstance(bp, dict) and 'timestamp' in bp:
#                 timestamp = bp['timestamp']
#             elif isinstance(bp, (int, float)):
#                 timestamp = float(bp)
#             else:
#                 continue  # Skip invalid format
                
#             if start_time <= timestamp <= end_time:
#                 break_points.append(timestamp)
        
#         # Add start and end points
#         all_points = [start_time] + break_points + [end_time]
#         all_points = sorted(set(all_points))
        
#         # Generate segments between scene breaks
#         for i in range(len(all_points)):
#             for j in range(i + 1, len(all_points)):
#                 seg_start = all_points[i]
#                 seg_end = all_points[j]
#                 duration = seg_end - seg_start
                
#                 if min_duration <= duration <= max_duration:
#                     segments.append({
#                         'start_time': seg_start,
#                         'end_time': seg_end,
#                         'duration': duration,
#                         'generation_method': 'scene_based',
#                         'method_priority': 2,  # High priority - natural scene boundaries
#                         'scene_break_count': j - i - 1
#                     })
        
#         return segments
    
#     def _generate_audio_cue_segments(self,
#                                    audio_analysis: Dict,
#                                    start_time: float,
#                                    end_time: float,
#                                    min_duration: int,
#                                    max_duration: int) -> List[Dict]:
#         """Generate segments based on audio cues and speech patterns."""
#         segments = []
        
#         transcription = audio_analysis.get('transcription', {})
#         transcript_segments = transcription.get('segments', [])
        
#         # Find natural speech boundaries
#         speech_breaks = []
        
#         for i, segment in enumerate(transcript_segments):
#             seg_start = segment.get('start', 0)
#             seg_end = segment.get('end', 0)
            
#             if not (start_time <= seg_start <= end_time):
#                 continue
            
#             # Add sentence endings as potential breaks
#             text = segment.get('text', '')
#             if any(punct in text for punct in ['.', '!', '?']):
#                 speech_breaks.append(seg_end)
            
#             # Add pause points (gaps between segments)
#             if i < len(transcript_segments) - 1:
#                 next_segment = transcript_segments[i + 1]
#                 gap = next_segment.get('start', 0) - seg_end
#                 if gap > 1.0:  # Significant pause
#                     speech_breaks.append(seg_end)
        
#         # Generate segments using speech boundaries
#         all_points = [start_time] + speech_breaks + [end_time]
#         all_points = sorted(set(all_points))
        
#         for i in range(len(all_points)):
#             for j in range(i + 1, len(all_points)):
#                 seg_start = all_points[i]
#                 seg_end = all_points[j]
#                 duration = seg_end - seg_start
                
#                 if min_duration <= duration <= max_duration:
#                     segments.append({
#                         'start_time': seg_start,
#                         'end_time': seg_end,
#                         'duration': duration,
#                         'generation_method': 'audio_cue',
#                         'method_priority': 3,  # Good priority - speech patterns
#                         'speech_break_count': j - i - 1
#                     })
        
#         return segments
    
#     def _generate_content_aware_segments(self,
#                                        audio_analysis: Dict,
#                                        start_time: float,
#                                        end_time: float,
#                                        min_duration: int,
#                                        max_duration: int) -> List[Dict]:
#         """Generate segments based on content themes and topics."""
#         segments = []
        
#         transcription = audio_analysis.get('transcription', {})
#         transcript_segments = transcription.get('segments', [])
        
#         # Group segments by content similarity
#         content_groups = self._group_segments_by_content(transcript_segments, start_time, end_time)
        
#         # Also generate segments around high-value content markers
#         high_value_segments = self._find_high_value_content(transcript_segments, start_time, end_time, min_duration, max_duration)
#         segments.extend(high_value_segments)
        
#         for group in content_groups:
#             if len(group) < 1:  # Changed from 2 to 1 - single segments can be valuable too
#                 continue
                
#             group_start = min(seg['start'] for seg in group)
#             group_end = max(seg['end'] for seg in group)
#             duration = group_end - group_start
            
#             # Create segments within this content group
#             if duration >= min_duration:
#                 # Full group segment
#                 if duration <= max_duration:
#                     segments.append({
#                         'start_time': group_start,
#                         'end_time': group_end,
#                         'duration': duration,
#                         'generation_method': 'content_aware',
#                         'method_priority': 1,  # Highest priority - content coherence
#                         'content_coherence': True
#                     })
                
#                 # Partial group segments
#                 for sub_duration in range(min_duration, min(max_duration + 1, int(duration))):
#                     for offset in range(0, int(duration - sub_duration) + 1, 10):
#                         seg_start = group_start + offset
#                         seg_end = seg_start + sub_duration
                        
#                         segments.append({
#                             'start_time': seg_start,
#                             'end_time': seg_end,
#                             'duration': sub_duration,
#                             'generation_method': 'content_aware_partial',
#                             'method_priority': 3,  # Good priority - partial content coherence
#                             'content_coherence': True
#                         })
        
#         return segments

#     def _generate_actor_based_segments(self,
#                                        appearances_per_actor: Dict[str, List[int]],
#                                        actor_conf: Dict[str, float],
#                                        start_time: float,
#                                        end_time: float,
#                                        min_duration: int,
#                                        max_duration: int) -> List[Dict]:
#         """Generate segments centered on celebrity appearance timestamps.
#         Attempts to compute a reasonable celebrity_score based on actor_conf and timestamp coverage.
#         """
#         segments = []
#         try:
#             try:
#                 from ..face_insights.celebrity_index import actor_coverage_for_segment, compute_celebrity_score
#             except Exception:
#                 from src.face_insights.celebrity_index import actor_coverage_for_segment, compute_celebrity_score
#         except Exception:
#             actor_coverage_for_segment = None
#             compute_celebrity_score = None

#         for actor, timestamps in appearances_per_actor.items():
#             for t in timestamps:
#                 t = float(t)
#                 if t < start_time or t > end_time:
#                     continue
#                 # Primary short segment centered on timestamp
#                 duration = max(min_duration, min(max_duration, min_duration))
#                 start = max(start_time, t - duration / 2.0)
#                 end = min(end_time, start + duration)

#                 seg = {
#                     'start_time': start,
#                     'end_time': end,
#                     'duration': end - start,
#                     'generation_method': 'celebrity_appearance',
#                     'method_priority': 0,
#                     'actor_focus': actor,
#                     'appearance_timestamp_sec': int(t)
#                 }

#                 # Compute celebrity score for this specific segment if helpers available
#                 if actor_coverage_for_segment and compute_celebrity_score:
#                     per_actor = actor_coverage_for_segment(start, end, appearances_per_actor, actor_conf)
#                     if per_actor:
#                         seg['celebrity_actors'] = sorted(per_actor.items(), key=lambda x: x[1]['coverage'], reverse=True)
#                         seg['celebrity_score'] = float(compute_celebrity_score(per_actor))
#                         seg['has_celebrity'] = seg['celebrity_score'] > 0
#                         seg['top_celebrity'] = seg['celebrity_actors'][0][0] if seg['celebrity_actors'] else None

#                 # Baseline prompt match to ensure survive downstream filtering
#                 seg['prompt_match_score'] = seg.get('prompt_match_score', 0.6)
#                 segments.append(seg)

#                 # Secondary longer option
#                 long_duration = min(max_duration, duration * 2)
#                 if long_duration > duration:
#                     start2 = max(start_time, t - long_duration / 2.0)
#                     end2 = min(end_time, start2 + long_duration)
#                     seg2 = {
#                         'start_time': start2,
#                         'end_time': end2,
#                         'duration': end2 - start2,
#                         'generation_method': 'celebrity_appearance_long',
#                         'method_priority': 0,
#                         'actor_focus': actor,
#                         'appearance_timestamp_sec': int(t)
#                     }
#                     if actor_coverage_for_segment and compute_celebrity_score:
#                         per_actor2 = actor_coverage_for_segment(start2, end2, appearances_per_actor, actor_conf)
#                         if per_actor2:
#                             seg2['celebrity_actors'] = sorted(per_actor2.items(), key=lambda x: x[1]['coverage'], reverse=True)
#                             seg2['celebrity_score'] = float(compute_celebrity_score(per_actor2))
#                             seg2['has_celebrity'] = seg2['celebrity_score'] > 0
#                             seg2['top_celebrity'] = seg2['celebrity_actors'][0][0] if seg2.get('celebrity_actors') else None
#                     seg2['prompt_match_score'] = seg2.get('prompt_match_score', 0.5)
#                     segments.append(seg2)
#         return segments

#     def _generate_quality_driven_segments(self,
#                                         audio_analysis: Dict,
#                                         start_time: float,
#                                         end_time: float,
#                                         min_duration: int,
#                                         max_duration: int) -> List[Dict]:
#         """Generate segments focusing on high-quality content areas."""
#         segments = []
        
#         transcription = audio_analysis.get('transcription', {})
#         transcript_segments = transcription.get('segments', [])
        
#         # Identify high-quality speech segments
#         quality_segments = []
        
#         for segment in transcript_segments:
#             seg_start = segment.get('start', 0)
#             seg_end = segment.get('end', 0)
            
#             if not (start_time <= seg_start <= end_time):
#                 continue
            
#             text = segment.get('text', '')
            
#             # Quality indicators
#             quality_score = 0.0
            
#             # Clear speech (word count)
#             word_count = len(text.split())
#             if word_count >= 5:
#                 quality_score += 0.3
            
#             # Complete sentences
#             if any(punct in text for punct in ['.', '!', '?']):
#                 quality_score += 0.2
            
#             # Engaging content
#             engagement_words = ['important', 'key', 'amazing', 'incredible', 'secret', 'tip']
#             if any(word in text.lower() for word in engagement_words):
#                 quality_score += 0.3
            
#             # Direct address
#             if any(word in text.lower() for word in ['you', 'your', 'we', "let's"]):
#                 quality_score += 0.2
            
#             if quality_score >= 0.4:  # Threshold for quality
#                 quality_segments.append({
#                     'start': seg_start,
#                     'end': seg_end,
#                     'quality_score': quality_score,
#                     'text': text
#                 })
        
#         # Generate segments around high-quality areas
#         for qual_seg in quality_segments:
#             center_time = (qual_seg['start'] + qual_seg['end']) / 2
            
#             # Generate segments centered on quality content
#             for duration in range(min_duration, max_duration + 1, 10):
#                 seg_start = max(start_time, center_time - duration / 2)
#                 seg_end = min(end_time, seg_start + duration)
                
#                 if seg_end - seg_start >= min_duration:
#                     segments.append({
#                         'start_time': seg_start,
#                         'end_time': seg_end,
#                         'duration': seg_end - seg_start,
#                         'generation_method': 'quality_driven',
#                         'method_priority': 2,  # High priority - quality content
#                         'quality_score': qual_seg['quality_score'],
#                         'contains_quality_content': True
#                     })
        
#         return segments
    
#     def _group_segments_by_content(self, 
#                                  transcript_segments: List[Dict],
#                                  start_time: float,
#                                  end_time: float) -> List[List[Dict]]:
#         """Group transcript segments by content similarity."""
#         # Simple content grouping based on keywords and timing
#         groups = []
#         current_group = []
        
#         for segment in transcript_segments:
#             seg_start = segment.get('start', 0)
            
#             if not (start_time <= seg_start <= end_time):
#                 continue
            
#             text = segment.get('text', '').lower()
            
#             # If current group is empty or content is similar, add to group
#             if not current_group:
#                 current_group.append(segment)
#             else:
#                 # Check content similarity with last segment in group
#                 last_text = current_group[-1].get('text', '').lower()
                
#                 # Simple similarity check (common words)
#                 common_words = set(text.split()) & set(last_text.split())
#                 similarity = len(common_words) / max(len(text.split()), len(last_text.split()))
                
#                 # Check temporal proximity
#                 time_gap = seg_start - current_group[-1].get('end', 0)
                
#                 if similarity > 0.2 or time_gap < 5.0:  # Similar content or close in time
#                     current_group.append(segment)
#                 else:
#                     # Start new group
#                     if len(current_group) >= 2:
#                         groups.append(current_group)
#                     current_group = [segment]
        
#         # Add final group
#         if len(current_group) >= 2:
#             groups.append(current_group)
        
#         return groups
    
#     def _deduplicate_comprehensive_segments(self, 
#                                           all_segments: List[Dict],
#                                           max_segments: int) -> List[Dict]:
#         """
#         Remove duplicate and heavily overlapping segments while preserving diversity.
        
#         Args:
#             all_segments: All generated segments
#             max_segments: Maximum number of segments to keep
            
#         Returns:
#             Deduplicated segments
#         """
#         if not all_segments:
#             return []
        
#         # Sort by comprehensive intelligence metrics with ENHANCED visual weighting
#         def comprehensive_score(segment):
#             # Priority 1: Method priority (content_aware=1, sliding_window=10)
#             method_priority = segment.get('method_priority', 10)
            
#             # Priority 2: Quality and content intelligence
#             quality_score = segment.get('quality_score', 0.5)
#             text_quality = segment.get('text_quality_score', 0.5)
#             engagement = segment.get('engagement_score', 0.5)
            
#             # Priority 3: Vision and audio intelligence - ENHANCED
#             vision_score = segment.get('vision_score', 0.0)
#             visual_interest = segment.get('visual_interest', 0.0)
            
#             # Normalize vision scores (handle both 0-1 and 1-10 scales)
#             if isinstance(vision_score, (int, float)) and vision_score > 1:
#                 vision_score = vision_score / 10.0
#             if isinstance(visual_interest, (int, float)) and visual_interest > 1:
#                 visual_interest = visual_interest / 10.0
            
#             # Combined visual score with fallback
#             vision_normalized = max(vision_score, visual_interest, 0.0)
#             if vision_normalized == 0:
#                 vision_normalized = 0.5  # Default only if no vision data
            
#             # Scene type bonus for dynamic/engaging content
#             scene_bonus = 0.0
#             scene_type = segment.get('scene_type', 'unknown')
#             if scene_type in ['action', 'demonstration']:
#                 scene_bonus = 0.10
#             elif scene_type in ['presentation', 'interview', 'talking_head']:
#                 scene_bonus = 0.05
            
#             # Content-aware specific bonuses
#             content_bonus = 0.0
#             if segment.get('generation_method') == 'content_aware':
#                 content_bonus += 0.12  # Strong bonus for content-aware
#                 if segment.get('high_value_content'):
#                     content_bonus += 0.08  # Additional bonus for high-value content
#                 if segment.get('trigger_keywords'):
#                     content_bonus += 0.04  # Bonus for keyword triggers
            
#             # Audio transcription quality bonus
#             if segment.get('has_complete_sentences', False):
#                 content_bonus += 0.04
#             if segment.get('word_count', 0) > 100:  # Substantial content
#                 content_bonus += 0.02
            
#             # People/visual interest bonus - INCREASED
#             people_bonus = 0.0
#             if segment.get('people_visible') or segment.get('people_count', 0) > 0:
#                 people_bonus = 0.08
#                 # Extra bonus for multiple people (social/conversation content)
#                 if segment.get('people_count', 0) > 1:
#                     people_bonus += 0.04
            
#             # Motion/action bonus
#             motion_bonus = 0.05 if segment.get('has_action', False) else 0.0
            
#             # Calculate composite intelligence score - REBALANCED for visual
#             intelligence_score = (
#                 quality_score * 0.18 +      # Reduced from 0.25
#                 text_quality * 0.15 +       # Reduced from 0.20
#                 engagement * 0.15 +         # Reduced from 0.20
#                 vision_normalized * 0.25 +  # INCREASED from 0.15
#                 scene_bonus +               # NEW
#                 content_bonus +
#                 people_bonus +              # INCREASED
#                 motion_bonus                # NEW
#             )
            
#             # Return tuple for sorting: (method_priority, -intelligence_score, -duration)
#             # Lower method_priority is better, higher intelligence_score is better
#             return (method_priority, -intelligence_score, -segment.get('duration', 0))
        
#         all_segments.sort(key=comprehensive_score)
        
#         unique_segments = []
#         overlap_threshold = self.config['overlap_tolerance']
#         content_sim_threshold = self.config.get('content_similarity_threshold', 0.4)
        
#         for segment in all_segments:
#             if len(unique_segments) >= max_segments:
#                 break
            
#             # Check overlap with existing segments - ENHANCED with content similarity
#             should_skip = False
#             for existing in unique_segments:
#                 overlap_ratio = self._calculate_overlap_ratio(segment, existing)
                
#                 # Only skip if BOTH temporal overlap AND content similarity are high
#                 if overlap_ratio > overlap_threshold:
#                     content_similarity = self._calculate_content_similarity(segment, existing)
#                     if content_similarity > content_sim_threshold:
#                         should_skip = True
#                         break
#                     else:
#                         # Overlapping but DIFFERENT content - log and potentially keep
#                         self.logger.debug(
#                             f"Keeping overlapping segment - different content "
#                             f"(overlap: {overlap_ratio:.2f}, content_sim: {content_similarity:.2f})"
#                         )
            
#             if not should_skip:
#                 unique_segments.append(segment)
        
#         # If we still have space, add more segments with slightly higher overlap tolerance
#         if len(unique_segments) < max_segments * 0.8:  # If we have less than 80% capacity
#             higher_threshold = overlap_threshold + 0.2
            
#             for segment in all_segments:
#                 if segment in unique_segments:
#                     continue
                    
#                 if len(unique_segments) >= max_segments:
#                     break
                
#                 should_skip = False
#                 for existing in unique_segments:
#                     overlap_ratio = self._calculate_overlap_ratio(segment, existing)
#                     if overlap_ratio > higher_threshold:
#                         content_similarity = self._calculate_content_similarity(segment, existing)
#                         if content_similarity > content_sim_threshold:
#                             should_skip = True
#                             break
                
#                 if not should_skip:
#                     unique_segments.append(segment)
        
#         self.logger.info(f"Content-aware dedup: {len(all_segments)} → {len(unique_segments)} segments")

#         # Enforce a minimum temporal gap between final output segments to avoid heavy duplication
#         min_gap = float(self.config.get('min_output_gap', 20.0))

#         def _center(seg):
#             return (seg['start_time'] + seg['end_time']) / 2.0

#         final_segments = []
#         # Iterate in priority order (unique_segments is already sorted by comprehensive score)
#         for seg in unique_segments:
#             if len(final_segments) >= max_segments:
#                 break
#             conflict = False
#             for s in final_segments:
#                 center_gap = abs(_center(seg) - _center(s))
#                 if center_gap < min_gap or self._calculate_overlap_ratio(seg, s) > 0.6:
#                     conflict = True
#                     break
#             if not conflict:
#                 final_segments.append(seg)

#         # If we still don't have enough segments, relax gap thresholds progressively
#         if len(final_segments) < max_segments:
#             gap_vals = [min_gap / 2.0, min_gap / 4.0, 0.0]
#             for gap in gap_vals:
#                 if len(final_segments) >= max_segments:
#                     break
#                 for seg in unique_segments:
#                     if seg in final_segments:
#                         continue
#                     # check against current final list with relaxed gap
#                     conflict = False
#                     for s in final_segments:
#                         center_gap = abs(_center(seg) - _center(s))
#                         if center_gap < gap and self._calculate_overlap_ratio(seg, s) > 0.6:
#                             conflict = True
#                             break
#                     if not conflict:
#                         final_segments.append(seg)
#                     if len(final_segments) >= max_segments:
#                         break

#         final_segments = final_segments[:max_segments]
#         self.logger.info(f"Post-gap enforcement: {len(final_segments)} segments returned (min_gap={min_gap}s)")
#         return final_segments
    
#     def _calculate_content_similarity(self, seg1: Dict, seg2: Dict) -> float:
#         """
#         Calculate content similarity between two segments.
#         Returns 0.0 (completely different) to 1.0 (identical content).
#         """
#         similarity_score = 0.0
#         weights_used = 0.0
        
#         # 1. Keyword overlap (weight: 0.3)
#         keywords1 = set(seg1.get('trigger_keywords', []) or [])
#         keywords2 = set(seg2.get('trigger_keywords', []) or [])
#         if keywords1 or keywords2:
#             keyword_overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
#             similarity_score += keyword_overlap * 0.3
#             weights_used += 0.3
        
#         # 2. Text content similarity (weight: 0.5)
#         text1 = seg1.get('segment_text', seg1.get('text', '')).lower()
#         text2 = seg2.get('segment_text', seg2.get('text', '')).lower()
#         if text1 and text2:
#             words1 = set(text1.split())
#             words2 = set(text2.split())
#             stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'it', 'of', 'that', 'this'}
#             words1 = words1 - stop_words
#             words2 = words2 - stop_words
#             if words1 or words2:
#                 text_overlap = len(words1 & words2) / max(len(words1 | words2), 1)
#                 similarity_score += text_overlap * 0.5
#                 weights_used += 0.5
        
#         # 3. Generation method match (weight: 0.2)
#         method1 = seg1.get('generation_method', 'unknown')
#         method2 = seg2.get('generation_method', 'unknown')
#         if method1 != 'unknown' or method2 != 'unknown':
#             method_match = 1.0 if method1 == method2 else 0.0
#             similarity_score += method_match * 0.2
#             weights_used += 0.2
        
#         if weights_used > 0:
#             return similarity_score / weights_used
#         return 0.5  # Default if no content info
    
#     def _calculate_overlap_ratio(self, seg1: Dict, seg2: Dict) -> float:
#         """Calculate overlap ratio between two segments."""
#         start1, end1 = seg1['start_time'], seg1['end_time']
#         start2, end2 = seg2['start_time'], seg2['end_time']
        
#         # Calculate overlap
#         overlap_start = max(start1, start2)
#         overlap_end = min(end1, end2)
        
#         if overlap_end <= overlap_start:
#             return 0.0
        
#         overlap_duration = overlap_end - overlap_start
#         min_duration = min(end1 - start1, end2 - start2)
        
#         return overlap_duration / min_duration if min_duration > 0 else 0.0
    
#     def _enhance_segments_with_metadata(self,
#                                       segments: List[Dict],
#                                       audio_analysis: Dict,
#                                       scene_analysis: Dict) -> List[Dict]:
#         """Add comprehensive metadata to each segment."""
#         enhanced_segments = []
        
#         transcription = audio_analysis.get('transcription', {})
#         transcript_segments = transcription.get('segments', [])
        
#         for segment in segments:
#             enhanced_segment = segment.copy()
#             start_time = segment['start_time']
#             end_time = segment['end_time']
            
#             # Extract text content
#             segment_text = self._extract_segment_text(transcript_segments, start_time, end_time)
#             enhanced_segment['segment_text'] = segment_text
            
#             # Calculate text-based metrics
#             if segment_text:
#                 word_count = len(segment_text.split())
#                 sentence_count = segment_text.count('.') + segment_text.count('!') + segment_text.count('?')
                
#                 enhanced_segment.update({
#                     'word_count': word_count,
#                     'sentence_count': sentence_count,
#                     'text_density': word_count / segment['duration'] if segment['duration'] > 0 else 0,
#                     'has_complete_sentences': sentence_count > 0,
#                     'text_quality_score': self._calculate_text_quality(segment_text)
#                 })
#             else:
#                 enhanced_segment.update({
#                     'word_count': 0,
#                     'sentence_count': 0,
#                     'text_density': 0,
#                     'has_complete_sentences': False,
#                     'text_quality_score': 0.0
#                 })
            
#             # Add scene break information
#             scene_breaks_in_segment = self._count_scene_breaks_in_segment(
#                 scene_analysis, start_time, end_time
#             )
#             enhanced_segment['scene_breaks_count'] = scene_breaks_in_segment
            
#             # Add preliminary quality assessment
#             enhanced_segment['preliminary_quality'] = self._assess_preliminary_quality(enhanced_segment)

#             # Preserve and boost celebrity metadata if present so actor segments rank higher
#             if enhanced_segment.get('actor_focus') or enhanced_segment.get('has_celebrity'):
#                 enhanced_segment.setdefault('has_celebrity', True)
#                 enhanced_segment.setdefault('celebrity_score', enhanced_segment.get('celebrity_score', 0.9))
#                 enhanced_segment.setdefault('celebrity_actors', enhanced_segment.get('celebrity_actors', [enhanced_segment.get('actor_focus')]))
#                 # Boost preliminary quality and ensure prompt_match baseline so these survive filtering
#                 enhanced_segment['preliminary_quality'] = min(1.0, enhanced_segment.get('preliminary_quality', 0) + 0.2)
#                 enhanced_segment['prompt_match_score'] = max(enhanced_segment.get('prompt_match_score', 0.0), 0.6)

#             enhanced_segments.append(enhanced_segment)
        
#         return enhanced_segments
    
#     def _extract_segment_text(self, transcript_segments: List[Dict], start_time: float, end_time: float) -> str:
#         """Extract text from transcript segments within time range."""
#         text_parts = []
        
#         for seg in transcript_segments:
#             seg_start = seg.get('start', 0)
#             seg_end = seg.get('end', 0)
            
#             # Check for overlap
#             if seg_start < end_time and seg_end > start_time:
#                 text = seg.get('text', '').strip()
#                 if text:
#                     text_parts.append(text)
        
#         return ' '.join(text_parts)
    
#     def _find_high_value_content(self, transcript_segments: List[Dict], start_time: float, end_time: float, min_duration: int, max_duration: int) -> List[Dict]:
#         """Find high-value content segments based on keywords and patterns."""
#         segments = []
        
#         # High-value keywords that indicate interesting content
#         high_value_keywords = [
#             'amazing', 'incredible', 'breakthrough', 'important', 'key', 'secret',
#             'shocking', 'surprising', 'reveal', 'discover', 'explain', 'show',
#             'first', 'best', 'worst', 'never', 'always', 'must', 'should'
#         ]
        
#         for i, segment in enumerate(transcript_segments):
#             seg_start = segment.get('start', 0)
#             seg_end = segment.get('end', 0)
#             text = segment.get('text', '').lower()
            
#             if not (start_time <= seg_start <= end_time):
#                 continue
            
#             # Check for high-value keywords
#             has_high_value = any(keyword in text for keyword in high_value_keywords)
            
#             if has_high_value:
#                 # Create segments around this high-value content
#                 for duration in range(min_duration, max_duration + 1, 15):
#                     # Center the segment around the high-value content
#                     center_time = (seg_start + seg_end) / 2
#                     segment_start = max(start_time, center_time - duration / 2)
#                     segment_end = min(end_time, segment_start + duration)
                    
#                     if segment_end - segment_start >= min_duration:
#                         segments.append({
#                             'start_time': segment_start,
#                             'end_time': segment_end,
#                             'duration': segment_end - segment_start,
#                             'generation_method': 'content_aware',
#                             'method_priority': 1,  # Highest priority
#                             'high_value_content': True,
#                             'trigger_keywords': [kw for kw in high_value_keywords if kw in text]
#                         })
        
#         return segments
    
#     def _calculate_text_quality(self, text: str) -> float:
#         """Calculate text quality score."""
#         if not text:
#             return 0.0
        
#         score = 0.0
        
#         # Length factor
#         word_count = len(text.split())
#         if 5 <= word_count <= 50:
#             score += 0.3
#         elif word_count > 0:
#             score += 0.1
        
#         # Completeness (sentences)
#         if any(punct in text for punct in ['.', '!', '?']):
#             score += 0.3
        
#         # Engagement indicators
#         engagement_words = ['important', 'key', 'amazing', 'incredible', 'you', 'we']
#         engagement_count = sum(1 for word in engagement_words if word in text.lower())
#         score += min(0.2, engagement_count * 0.1)
        
#         # Clarity (not too fragmented)
#         if not text.endswith(('...', '-', ',')):
#             score += 0.2
        
#         return min(1.0, score)
    
#     def _count_scene_breaks_in_segment(self, scene_analysis: Dict, start_time: float, end_time: float) -> int:
#         """Count scene breaks within segment."""
#         scene_breaks = scene_analysis.get('combined_breaks', [])
        
#         count = 0
#         for break_point in scene_breaks:
#             if isinstance(break_point, dict) and 'timestamp' in break_point:
#                 timestamp = break_point['timestamp']
#             elif isinstance(break_point, (int, float)):
#                 timestamp = float(break_point)
#             else:
#                 continue  # Skip invalid format
                
#             if start_time < timestamp < end_time:
#                 count += 1
        
#         return count
    
#     def _assess_preliminary_quality(self, segment: Dict) -> float:
#         """Assess preliminary quality of segment."""
#         score = 0.0
        
#         # Text quality weight: 40%
#         text_quality = segment.get('text_quality_score', 0)
#         score += text_quality * 0.4
        
#         # Duration appropriateness weight: 20%
#         duration = segment.get('duration', 0)
#         if 20 <= duration <= 90:
#             score += 0.2
#         elif 15 <= duration <= 120:
#             score += 0.1
        
#         # Method priority weight: 20%
#         method_priority = segment.get('method_priority', 10)
#         priority_score = max(0, (10 - method_priority) / 10)
#         score += priority_score * 0.2
        
#         # Content coherence weight: 20%
#         if segment.get('content_coherence', False):
#             score += 0.15
#         if segment.get('contains_quality_content', False):
#             score += 0.05
        
#         return min(1.0, score)
