#!/usr/bin/env python3
"""
Variation Manager - Adds controlled randomization to video processing
to prevent identical outputs while maintaining quality standards.
"""

import random
import time
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

class VariationManager:
    """
    Manages controlled randomization across video processing pipeline
    to ensure different outputs for multiple runs of the same video.
    """
    
    def __init__(self, enable_variation: bool = True, variation_seed: Optional[int] = None):
        """
        Initialize variation manager.
        
        Args:
            enable_variation: Whether to enable variation (default: True)
            variation_seed: Optional seed for reproducible variation (for testing)
        """
        self.enable_variation = enable_variation
        self.logger = logging.getLogger(__name__)
        
        # Set random seed - use provided seed, or time-based for true randomization
        if variation_seed is not None:
            self.seed = variation_seed
            random.seed(variation_seed)
            self.logger.info(f"🎲 Variation enabled with fixed seed: {variation_seed}")
        else:
            # Use time-based seed with microsecond precision and additional randomness for true variation
            import os
            microsecond_time = int(time.time() * 1000000) % 1000000000
            process_random = os.getpid() * random.randint(1, 10000)
            self.seed = (microsecond_time + process_random) % 1000000000
            random.seed(self.seed)
            self.logger.info(f"🎲 Variation enabled with dynamic seed: {self.seed}")
        
        # Variation settings
        self.segment_diversity_factor = 0.3  # How much to randomize segment selection
        self.title_variation_strength = 0.4  # How much to vary title generation
        self.cache_bust_probability = 0.8   # Probability of cache busting
        
        self.logger.info(f"🎯 VariationManager initialized (enabled={enable_variation})")
    
    def get_varied_cache_key(self, base_key: str, content_hash: str = "") -> str:
        """
        Generate a varied cache key to prevent identical cached responses.
        
        Args:
            base_key: Original cache key
            content_hash: Hash of content for variation
            
        Returns:
            Modified cache key with variation component
        """
        if not self.enable_variation:
            return base_key
        
        # Add variation component to cache key
        if random.random() < self.cache_bust_probability:
            # Include timestamp and seed for variation
            timestamp = int(time.time())
            variation_id = f"{self.seed}_{timestamp}"
            varied_key = f"{base_key}_var_{hashlib.md5(variation_id.encode()).hexdigest()[:8]}"
            self.logger.debug(f"🔄 Cache key varied: {base_key} -> {varied_key}")
            return varied_key
        else:
            return base_key
    
    def apply_segment_variation(self, 
                              scored_segments: List[Dict], 
                              target_count: int = 5) -> List[Dict]:
        """
        Apply controlled randomization to segment selection while maintaining quality.
        
        Args:
            scored_segments: List of segments with scores (sorted by score)
            target_count: Target number of segments to return
            
        Returns:
            List of segments with applied variation
        """
        if not self.enable_variation or len(scored_segments) <= target_count:
            return scored_segments[:target_count]
        
        # More aggressive variation: expand quality tiers
        high_quality = []   # Top 50% by score (increased from 30%)
        medium_quality = [] # Bottom 50% by score  
        
        total_segments = len(scored_segments)
        high_cutoff = int(total_segments * 0.5)  # Increased from 0.3
        
        for i, segment in enumerate(scored_segments):
            if i < high_cutoff:
                high_quality.append(segment)
            else:
                medium_quality.append(segment)
        
        # Shuffle segments before selection to increase randomness
        import random as rnd
        rnd.shuffle(high_quality)
        rnd.shuffle(medium_quality)
        
        # More varied selection strategy
        selected_segments = []
        
        # Even more aggressive reduction of high-quality dominance (30-60% instead of 40-70%)
        high_percentage = random.uniform(0.3, 0.6)
        high_count = max(1, int(target_count * high_percentage))
        high_count = min(high_count, len(high_quality))
        
        # Add position-based variation - sometimes prefer segments from different parts of video
        position_bias = random.choice(['early', 'middle', 'late', 'random'])
        
        if position_bias != 'random' and len(high_quality) > high_count:
            # Apply position preference to high quality segments
            if position_bias == 'early':
                # Bias towards earlier segments in the video
                high_quality_sorted = sorted(high_quality, key=lambda x: x.get('start_time', 0))
                weights = [2.0 / (i + 1) for i in range(len(high_quality_sorted))]
            elif position_bias == 'late':
                # Bias towards later segments in the video
                high_quality_sorted = sorted(high_quality, key=lambda x: x.get('start_time', 0), reverse=True)
                weights = [2.0 / (i + 1) for i in range(len(high_quality_sorted))]
            else:  # middle
                # Bias towards middle segments
                high_quality_sorted = sorted(high_quality, key=lambda x: x.get('start_time', 0))
                mid_point = len(high_quality_sorted) // 2
                weights = [1.0 / (abs(i - mid_point) + 1) for i in range(len(high_quality_sorted))]
            
            selected_high = random.choices(high_quality_sorted, weights=weights, k=high_count)
        else:
            # Random selection from high quality
            if len(high_quality) > high_count:
                selected_high = random.sample(high_quality, high_count)
            else:
                selected_high = high_quality
        
        selected_segments.extend(selected_high)
        remaining_slots = target_count - len(selected_segments)
        
        # Fill remaining with medium quality - use more aggressive selection
        if remaining_slots > 0 and medium_quality:
            medium_count = min(remaining_slots, len(medium_quality))
            
            # Sometimes completely randomize medium selection
            if random.random() < 0.3:  # 30% chance for full randomization
                selected_medium = random.sample(medium_quality, medium_count)
            else:
                # Weighted selection
                medium_weights = [1.0 / (i + 1) for i in range(len(medium_quality))]
                selected_medium = random.choices(
                    medium_quality, 
                    weights=medium_weights, 
                    k=medium_count
                )
            
            selected_segments.extend(selected_medium)
        
        # Remove duplicates while preserving order
        seen_start_times = set()
        final_segments = []
        for segment in selected_segments:
            start_time = segment.get('start_time', 0)
            if start_time not in seen_start_times:
                seen_start_times.add(start_time)
                final_segments.append(segment)
        
        # Sort by start time for logical ordering
        final_segments.sort(key=lambda x: x.get('start_time', 0))
        
        self.logger.info(f"🎲 Segment variation applied: {len(scored_segments)} -> {len(final_segments)} "
                        f"(position_bias={position_bias}, high:{len(selected_high)}, "
                        f"medium:{len(selected_segments) - len(selected_high)})")
        
        return final_segments
    
    def apply_score_variation(self, segments: List[Dict]) -> List[Dict]:
        """
        Apply variation to segment scores to create more diverse selection.
        
        Args:
            segments: List of segments with scores
            
        Returns:
            Segments with varied scores
        """
        if not self.enable_variation:
            return segments
        
        varied_segments = []
        for segment in segments:
            varied_segment = segment.copy()
            original_score = segment.get('prompt_match_score', 0)
            
            # More aggressive random variation (±30% instead of ±15%)
            variation_factor = random.uniform(0.7, 1.3)
            varied_score = min(1.0, max(0.0, original_score * variation_factor))
            
            # More frequent boosting of lower-scored segments (40% chance instead of 20%)
            if original_score < 0.8 and random.random() < 0.4:
                boost = random.uniform(0.15, 0.4)  # Larger boost
                varied_score = min(1.0, varied_score + boost)
                varied_segment['score_boosted'] = True
            
            # Occasionally penalize high-scoring segments to prevent dominance
            if original_score > 0.85 and random.random() < 0.3:  # 30% chance
                penalty = random.uniform(0.1, 0.25)
                varied_score = max(0.0, varied_score - penalty)
                varied_segment['score_penalized'] = True
            
            varied_segment['prompt_match_score'] = varied_score
            varied_segment['original_score'] = original_score
            varied_segments.append(varied_segment)
        
        self.logger.debug(f"🎯 Applied aggressive score variation to {len(segments)} segments")
        return varied_segments
    
    def get_title_variation_prompt_modifier(self, base_prompt: str, content_type: str = "social_media") -> str:
        """
        Add variation to title generation prompts to produce different outputs.
        
        Args:
            base_prompt: Original prompt for title generation
            content_type: Type of content (social_media, educational, entertainment)
            
        Returns:
            Modified prompt with variation instructions
        """
        if not self.enable_variation:
            return base_prompt
        
        # Variation instructions based on content type
        variation_instructions = {
            'social_media': [
                "Create a catchy, trending-style title that would go viral.",
                "Focus on an intriguing hook that makes people want to click.",
                "Use a different emotional angle (excitement, curiosity, shock).",
                "Try a question-based or how-to format for engagement.",
                "Emphasize the most surprising or interesting aspect."
            ],
            'educational': [
                "Focus on the key learning outcome or skill taught.",
                "Use a problem-solution angle for the title.",
                "Emphasize the practical benefit or tip shared.",
                "Create a tutorial-style title that promises knowledge.",
                "Highlight the expertise or authority in the content."
            ],
            'entertainment': [
                "Emphasize the fun, humorous, or entertaining aspect.",
                "Use an exciting or dramatic angle for the title.",
                "Focus on the emotional reaction it will create.",
                "Try a before/after or transformation angle.",
                "Highlight the most amusing or surprising moment."
            ]
        }
        
        # Select random variation instruction
        instructions = variation_instructions.get(content_type, variation_instructions['social_media'])
        selected_instruction = random.choice(instructions)
        
        # Add variation modifier to prompt
        variation_modifier = f"\n\nVARIATION FOCUS: {selected_instruction}\n"
        variation_modifier += f"Make this title unique and different from typical titles for similar content.\n"
        
        # Occasionally add creativity boosters
        if random.random() < 0.3:
            creativity_boosters = [
                "Be creative and think outside the box.",
                "Use an unexpected or fresh perspective.",
                "Try a different tone or style than usual.",
                "Consider what would make this stand out in a feed."
            ]
            variation_modifier += f"CREATIVITY BOOST: {random.choice(creativity_boosters)}\n"
        
        modified_prompt = base_prompt + variation_modifier
        
        self.logger.debug(f"🎨 Title prompt variation added: {selected_instruction}")
        return modified_prompt
    
    def add_description_variation(self, base_description_prompt: str) -> str:
        """
        Add variation to description generation prompts.
        
        Args:
            base_description_prompt: Original description prompt
            
        Returns:
            Modified prompt with variation instructions
        """
        if not self.enable_variation:
            return base_description_prompt
        
        # Description variation approaches
        variation_approaches = [
            "Focus on creating urgency or FOMO (fear of missing out).",
            "Use a storytelling approach to describe the content.",
            "Emphasize the value or benefit viewers will get.",
            "Create curiosity with a teaser approach.",
            "Use social proof or trending elements.",
            "Focus on the emotional impact or reaction.",
            "Try a behind-the-scenes or insider perspective."
        ]
        
        selected_approach = random.choice(variation_approaches)
        
        # Add hashtag variation
        hashtag_styles = [
            "Use trending, popular hashtags",
            "Focus on niche, specific hashtags", 
            "Mix popular and niche hashtags",
            "Use descriptive, content-specific hashtags"
        ]
        
        hashtag_instruction = random.choice(hashtag_styles)
        
        variation_modifier = f"\n\nDESCRIPTION VARIATION: {selected_approach}\n"
        variation_modifier += f"HASHTAG STRATEGY: {hashtag_instruction}\n"
        variation_modifier += "Make this description unique and engaging in a different way than typical descriptions.\n"
        
        return base_description_prompt + variation_modifier
    
    def should_use_alternative_analysis_path(self) -> bool:
        """
        Determine if an alternative analysis path should be used for variation.
        
        Returns:
            bool: True if alternative path should be used
        """
        if not self.enable_variation:
            return False
        
        # 30% chance to use alternative analysis path
        return random.random() < 0.3
    
    def get_segment_position_variation(self, total_duration: float) -> Tuple[str, float, float]:
        """
        Get varied segment position preferences for different video regions.
        
        Args:
            total_duration: Total video duration in seconds
            
        Returns:
            Tuple of (position_name, start_ratio, end_ratio)
        """
        if not self.enable_variation:
            return ("any", 0.0, 1.0)
        
        # Define different position preferences with variation
        position_options = [
            ("opening", 0.0, 0.3),      # First 30%
            ("early", 0.1, 0.4),       # 10-40%
            ("middle", 0.3, 0.7),      # 30-70%
            ("late", 0.6, 0.9),        # 60-90%
            ("ending", 0.7, 1.0),      # Last 30%
            ("peak", 0.4, 0.8),        # Peak content area
            ("any", 0.0, 1.0)          # No preference
        ]
        
        # Weight preferences (favor middle and peak regions slightly)
        weights = [0.1, 0.15, 0.2, 0.15, 0.1, 0.2, 0.1]
        
        selected_position = random.choices(position_options, weights=weights, k=1)[0]
        
        self.logger.debug(f"🎯 Position variation: {selected_position[0]} ({selected_position[1]:.1%}-{selected_position[2]:.1%})")
        
        return selected_position
    
    def apply_quality_threshold_variation(self, base_threshold: float = 0.6) -> float:
        """
        Apply variation to quality thresholds to allow different quality segments.
        
        Args:
            base_threshold: Base quality threshold
            
        Returns:
            Varied threshold value
        """
        if not self.enable_variation:
            return base_threshold
        
        # Vary threshold by ±20% to allow different quality ranges
        variation_range = 0.2
        min_threshold = max(0.3, base_threshold - variation_range)
        max_threshold = min(0.9, base_threshold + variation_range)
        
        varied_threshold = random.uniform(min_threshold, max_threshold)
        
        self.logger.debug(f"📊 Quality threshold variation: {base_threshold:.2f} -> {varied_threshold:.2f}")
        
        return varied_threshold
    
    def get_variation_summary(self) -> Dict[str, Any]:
        """
        Get summary of variation settings and current state.
        
        Returns:
            Dictionary with variation information
        """
        return {
            'enabled': self.enable_variation,
            'seed': self.seed,
            'timestamp': datetime.now().isoformat(),
            'settings': {
                'segment_diversity_factor': self.segment_diversity_factor,
                'title_variation_strength': self.title_variation_strength,
                'cache_bust_probability': self.cache_bust_probability
            }
        }
    
    def apply_aggressive_variation(self, segments: List[Dict], target_count: int = 5) -> List[Dict]:
        """
        Apply very aggressive variation by purposely avoiding highest-scoring segments.
        
        Args:
            segments: List of segments with scores (sorted by score)
            target_count: Target number of segments to return
            
        Returns:
            List of segments with aggressive variation applied
        """
        if not self.enable_variation or len(segments) <= target_count:
            return segments[:target_count]
        
        # Shuffle all segments first
        import random as rnd
        shuffled_segments = segments.copy()
        rnd.shuffle(shuffled_segments)
        
        # Categorize segments more aggressively
        top_tier = []       # Top 20% (usually avoid these)
        high_tier = []      # Next 30% (prefer these)
        medium_tier = []    # Next 30% (sometimes use these)
        lower_tier = []     # Bottom 20% (occasionally use)
        
        total = len(shuffled_segments)
        top_cutoff = int(total * 0.2)
        high_cutoff = int(total * 0.5)
        medium_cutoff = int(total * 0.8)
        
        for i, segment in enumerate(shuffled_segments):
            if i < top_cutoff:
                top_tier.append(segment)
            elif i < high_cutoff:
                high_tier.append(segment)
            elif i < medium_cutoff:
                medium_tier.append(segment)
            else:
                lower_tier.append(segment)
        
        selected_segments = []
        
        # Selection strategy with forced diversity
        strategy = random.choice(['avoid_top', 'middle_focus', 'mixed_random'])
        
        if strategy == 'avoid_top':
            # Completely avoid top tier, focus on high and medium
            candidates = high_tier + medium_tier
            if len(candidates) >= target_count:
                selected_segments = random.sample(candidates, target_count)
            else:
                selected_segments = candidates + random.sample(lower_tier, 
                                                               min(target_count - len(candidates), len(lower_tier)))
        
        elif strategy == 'middle_focus':
            # Focus on medium tier with some high tier
            medium_count = min(target_count // 2, len(medium_tier))
            high_count = min(target_count - medium_count, len(high_tier))
            remaining = target_count - medium_count - high_count
            
            selected_segments.extend(random.sample(medium_tier, medium_count) if medium_tier else [])
            selected_segments.extend(random.sample(high_tier, high_count) if high_tier else [])
            
            if remaining > 0:
                remaining_candidates = lower_tier + top_tier
                if remaining_candidates:
                    selected_segments.extend(random.sample(remaining_candidates, 
                                                          min(remaining, len(remaining_candidates))))
        
        else:  # mixed_random
            # Completely random selection from all tiers
            all_candidates = top_tier + high_tier + medium_tier + lower_tier
            selected_segments = random.sample(all_candidates, min(target_count, len(all_candidates)))
        
        # Remove duplicates and sort by start time
        seen_times = set()
        final_segments = []
        for segment in selected_segments:
            start_time = segment.get('start_time', 0)
            if start_time not in seen_times:
                seen_times.add(start_time)
                final_segments.append(segment)
        
        final_segments.sort(key=lambda x: x.get('start_time', 0))
        
        self.logger.info(f"🎲 Aggressive variation applied: {len(segments)} -> {len(final_segments)} "
                        f"(strategy={strategy}, avoided_top={len(top_tier)})")
        
        return final_segments
    
    @staticmethod
    def create_session_variation_manager(enable_variation: bool = True) -> 'VariationManager':
        """
        Create a variation manager for a processing session.
        
        Args:
            enable_variation: Whether to enable variation
            
        Returns:
            VariationManager instance
        """
        return VariationManager(
            enable_variation=enable_variation,
            variation_seed=None  # Use time-based seed for true variation
        )