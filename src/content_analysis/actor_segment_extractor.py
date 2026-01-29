"""
Actor-specific segment extractor for strict actor-based clip generation.

This module ensures that when users request "generate clips with [actor name]",
the system ONLY generates segments from the precomputed timestamps where that
actor is present, without any fallback or random segment generation.
"""

import logging
from typing import Dict, List, Optional, Tuple


class ActorSegmentExtractor:
    """
    Extracts segments exclusively from precomputed actor appearance timestamps.
    Prevents overlapping/duplicate segments by enforcing non-overlapping time windows.
    """
    
    def __init__(self):
        """Initialize the actor segment extractor."""
        self.logger = logging.getLogger(__name__)
        self.generated_time_ranges = []  # Track all generated segment time ranges to prevent overlaps
    
    def _segments_overlap(self, seg1_start: float, seg1_end: float, 
                         seg2_start: float, seg2_end: float) -> bool:
        """
        Check if two time segments overlap.
        
        Args:
            seg1_start: Start time of first segment
            seg1_end: End time of first segment
            seg2_start: Start time of second segment
            seg2_end: End time of second segment
            
        Returns:
            True if segments overlap, False otherwise
        """
        # Segments overlap if one starts before the other ends
        return seg1_start < seg2_end and seg2_start < seg1_end
    
    def _has_overlap_with_existing(self, start_time: float, end_time: float) -> bool:
        """
        Check if a segment overlaps with any previously generated segment.
        
        Args:
            start_time: Start time of segment to check
            end_time: End time of segment to check
            
        Returns:
            True if segment overlaps with any existing segment, False otherwise
        """
        for existing_start, existing_end in self.generated_time_ranges:
            if self._segments_overlap(start_time, end_time, existing_start, existing_end):
                return True
        return False
    
    def _record_segment(self, start_time: float, end_time: float) -> None:
        """
        Record a generated segment's time range to prevent future overlaps.
        
        Args:
            start_time: Start time of segment
            end_time: End time of segment
        """
        self.generated_time_ranges.append((start_time, end_time))
        self.logger.debug(f"Recorded segment time range: {start_time:.2f}s-{end_time:.2f}s")
    
    def reset_time_ranges(self) -> None:
        """
        Reset the tracked time ranges. Call this when starting a new actor's segment generation.
        """
        self.generated_time_ranges = []
        self.logger.debug("Reset time range tracking for new actor")

    
    def extract_actor_only_segments(self,
                                   actor_name: str,
                                   appearances_per_actor: Dict[str, List[int]],
                                   actor_conf: Dict[str, float],
                                   min_duration: int,
                                   max_duration: int,
                                   video_start: float = 0.0,
                                   video_end: float = float('inf')) -> List[Dict]:
        """
        Extract segments ONLY from timestamps where the specified actor appears.
        
        This is the strict implementation - NO fallback segment generation,
        NO random segments outside actor detection ranges.
        ENFORCES: No overlapping segments - once a segment is generated,
        its entire time range is excluded from further candidate generation.
        
        Args:
            actor_name: Name of the actor to extract segments for (must match exactly or in appearances_per_actor)
            appearances_per_actor: Dict mapping actor names to lists of timestamp seconds
            actor_conf: Dict mapping actor names to confidence scores
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            video_start: Video start time (default 0.0)
            video_end: Video end time (default infinity)
            
        Returns:
            List of segment dicts with ONLY precomputed actor timestamps as basis (non-overlapping)
        """
        segments = []
        
        # Reset tracking for this actor's segment generation
        self.reset_time_ranges()
        
        # Find matching actor(s) in appearances_per_actor
        # Note: Same actor can appear multiple times with different face_ids
        matching_actors = []
        for actor_key in appearances_per_actor.keys():
            if actor_key.lower() == actor_name.lower():
                matching_actors.append(actor_key)
        
        if not matching_actors:
            self.logger.warning(
                f"Actor '{actor_name}' not found in precomputed results. "
                f"Available actors: {list(appearances_per_actor.keys())}"
            )
            return []
        
        # Consolidate all timestamps from all matching actors (same person, different face detections)
        all_timestamps = []
        max_confidence = 0.0
        
        for actor_key in matching_actors:
            timestamps = appearances_per_actor.get(actor_key, [])
            confidence = actor_conf.get(actor_key, 0.0)
            max_confidence = max(max_confidence, confidence)
            all_timestamps.extend(timestamps)
        
        # Remove duplicates and sort
        all_timestamps = sorted(set(float(t) for t in all_timestamps))
        
        self.logger.info(
            f"Extracting segments for actor '{actor_name}' "
            f"with {len(all_timestamps)} total precomputed appearances across {len(matching_actors)} detections "
            f"(max confidence: {max_confidence:.2f})"
        )
        
        # Log all appearance timestamps for verification
        timestamps_str = ", ".join([f"{int(t)}s" for t in all_timestamps])
        self.logger.info(
            f"📍 Actor '{actor_name}' appearance timestamps: [{timestamps_str}]"
        )
        
        if not all_timestamps:
            self.logger.warning(f"Actor '{actor_name}' has no appearance timestamps")
            return []
        
        # Filter timestamps to video bounds
        timestamps = [t for t in all_timestamps if video_start <= t <= video_end]
        
        # Generate segments centered on each timestamp, enforcing non-overlapping constraint
        for timestamp in timestamps:
            # Check if we can still generate a segment around this timestamp
            # Primary segment: centered on timestamp
            segment_start = max(video_start, timestamp - min_duration / 2.0)
            segment_end = min(video_end, segment_start + min_duration)
            
            # Ensure minimum duration
            if segment_end - segment_start < min_duration:
                segment_end = min(video_end, segment_start + min_duration)
            
            # Ensure max duration constraint
            if segment_end - segment_start > max_duration:
                segment_end = segment_start + max_duration
            
            # CHECK FOR OVERLAP: Skip this segment if it overlaps with any previously generated segment
            if self._has_overlap_with_existing(segment_start, segment_end):
                self.logger.debug(
                    f"⏭️  Skipping segment {segment_start:.2f}s-{segment_end:.2f}s for timestamp {int(timestamp)}s "
                    f"(overlaps with previously generated segment)"
                )
                continue
            
            segment = {
                'start_time': segment_start,
                'end_time': segment_end,
                'duration': segment_end - segment_start,
                'generation_method': 'actor_only_strict',
                'method_priority': 100,  # Highest priority - precomputed
                'actor_focus': actor_name,  # Use original actor name requested
                'appearance_timestamp_sec': int(timestamp),
                'actor_confidence': float(max_confidence),
                'source': 'precomputed_detection',
                'prompt_match_score': 1.0,  # Maximum confidence - from precomputed
                'celebrity_score': float(max_confidence),
                'has_celebrity': True,
                'top_celebrity': actor_name
            }
            
            segments.append(segment)
            # Record this segment's time range to prevent future overlaps
            self._record_segment(segment_start, segment_end)
            self.logger.debug(
                f"✅ Generated primary segment {segment_start:.2f}s-{segment_end:.2f}s "
                f"for appearance at {int(timestamp)}s"
            )
            
            # Optional: Secondary longer segment for additional coverage
            # ONLY if it doesn't overlap with existing segments
            long_duration = min(max_duration, min_duration * 2)
            if long_duration > min_duration:
                long_start = max(video_start, timestamp - long_duration / 2.0)
                long_end = min(video_end, long_start + long_duration)
                
                # CHECK FOR OVERLAP: Skip extended segment if it overlaps
                if not self._has_overlap_with_existing(long_start, long_end):
                    long_segment = {
                        'start_time': long_start,
                        'end_time': long_end,
                        'duration': long_end - long_start,
                        'generation_method': 'actor_only_strict_extended',
                        'method_priority': 99,  # High priority - precomputed
                        'actor_focus': actor_name,  # Use original actor name requested
                        'appearance_timestamp_sec': int(timestamp),
                        'actor_confidence': float(max_confidence),
                        'source': 'precomputed_detection',
                        'prompt_match_score': 0.95,  # Slightly lower for extended
                        'celebrity_score': float(max_confidence),
                        'has_celebrity': True,
                        'top_celebrity': actor_name
                    }
                    
                    segments.append(long_segment)
                    # Record this segment's time range to prevent future overlaps
                    self._record_segment(long_start, long_end)
                    self.logger.debug(
                        f"✅ Generated extended segment {long_start:.2f}s-{long_end:.2f}s "
                        f"for appearance at {int(timestamp)}s"
                    )
                else:
                    self.logger.debug(
                        f"⏭️  Skipping extended segment {long_start:.2f}s-{long_end:.2f}s for timestamp {int(timestamp)}s "
                        f"(overlaps with previously generated segment)"
                    )
        
        self.logger.info(
            f"Generated {len(segments)} non-overlapping segments exclusively from "
            f"precomputed '{actor_name}' timestamps ({len(all_timestamps)} unique appearance points, "
            f"deduped from {len(timestamps)} video-range appearances)"
        )
        
        # Log segment coverage with actor appearances
        if segments:
            for i, segment in enumerate(segments[:5]):  # Log first 5 for brevity
                appearance_ts = segment.get('appearance_timestamp_sec', 'N/A')
                start = segment['start_time']
                end = segment['end_time']
                self.logger.info(
                    f"  Segment {i+1}: {start:.2f}s-{end:.2f}s (covers actor appearance at {appearance_ts}s)"
                )
            if len(segments) > 5:
                self.logger.info(f"  ... and {len(segments) - 5} more segments")
        
        return segments
    
    def extract_multiple_actors_segments(self,
                                        actor_names: List[str],
                                        appearances_per_actor: Dict[str, List[int]],
                                        actor_conf: Dict[str, float],
                                        min_duration: int,
                                        max_duration: int,
                                        video_start: float = 0.0,
                                        video_end: float = float('inf')) -> List[Dict]:
        """
        Extract segments for multiple actors (union of appearances).
        Each actor's segments are generated independently with non-overlapping constraint.
        
        Args:
            actor_names: List of actor names to extract segments for
            appearances_per_actor: Dict mapping actor names to lists of timestamp seconds
            actor_conf: Dict mapping actor names to confidence scores
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            video_start: Video start time (default 0.0)
            video_end: Video end time (default infinity)
            
        Returns:
            List of segments from all specified actors (non-overlapping within each actor)
        """
        all_segments = []
        
        for actor_name in actor_names:
            # Reset time ranges for each actor to track overlaps independently
            self.reset_time_ranges()
            
            actor_segments = self.extract_actor_only_segments(
                actor_name, appearances_per_actor, actor_conf,
                min_duration, max_duration, video_start, video_end
            )
            all_segments.extend(actor_segments)
        
        self.logger.info(
            f"Generated {len(all_segments)} total non-overlapping segments for {len(actor_names)} actors"
        )
        
        return all_segments
    
    def validate_actor_request(self,
                              actor_name: str,
                              appearances_per_actor: Dict[str, List[int]]) -> Tuple[bool, str]:
        """
        Validate that an actor is present in the precomputed results.
        Accounts for same actor being detected multiple times with different face_ids.
        
        Args:
            actor_name: Name of actor to validate
            appearances_per_actor: Dict mapping actor names to lists of timestamps
            
        Returns:
            Tuple of (is_valid, message)
        """
        matching_actors = []
        total_appearances = 0
        
        for actor_key in appearances_per_actor.keys():
            if actor_key.lower() == actor_name.lower():
                matching_actors.append(actor_key)
                timestamps = appearances_per_actor.get(actor_key, [])
                total_appearances += len(timestamps)
        
        if not matching_actors:
            return False, f"Actor '{actor_name}' not found. Available: {list(appearances_per_actor.keys())}"
        
        return True, f"Actor '{actor_name}' found in {len(matching_actors)} detection(s) with {total_appearances} total precomputed appearances"
