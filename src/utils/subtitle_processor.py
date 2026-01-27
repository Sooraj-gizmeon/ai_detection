# src/utils/subtitle_processor.py
"""Subtitle processing utilities for adding transcription overlays to videos"""

import os
import logging
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import re

from .subtitle_config_loader import SubtitleConfigLoader
from .kinetic_subtitle_processor import KineticSubtitleProcessor


class SubtitleProcessor:
    """
    Handles adding subtitle overlays to videos using transcription data.
    """
    
    def __init__(self):
        """Initialize subtitle processor."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize subtitle config loader
        self.config_loader = SubtitleConfigLoader()
        
        # Initialize kinetic subtitle processor
        self.kinetic_processor = KineticSubtitleProcessor()
        
        # Define legacy preset styles for backward compatibility
        # These will be deprecated in favor of JSON configuration
        self.preset_styles = {
            'classic': {
                'font_color': 'white',
                'background_color': 'black@0.7',
                'border_color': 'black',
                'shadow_color': 'black@0.9'
            },
            'modern': {
                'font_color': 'white',
                'background_color': 'black@0.3',  # Very subtle background
                'border_color': 'black',
                'shadow_color': 'black@0.8'
            },
            'high_contrast': {
                'font_color': 'yellow',
                'background_color': 'black@0.8',
                'border_color': 'black',
                'shadow_color': 'black@1.0'
            },
            'elegant': {
                'font_color': 'white',
                'background_color': '#333333@0.6',  # Dark gray background
                'border_color': '#222222',
                'shadow_color': 'black@0.8'
            },
            'vibrant': {
                'font_color': 'white',
                'background_color': '#FF6B35@0.7',  # Orange background
                'border_color': '#CC5529',
                'shadow_color': 'black@0.9'
            },
            'minimal': {
                'font_color': 'white',
                'background_color': 'transparent',  # No background
                'border_color': 'black',
                'border_width': 4,  # Thicker border for readability without background
                'shadow_color': 'black@1.0'
            },
            'blue_theme': {
                'font_color': 'white',
                'background_color': '#1E3A8A@0.7',  # Blue background
                'border_color': '#1E40AF',
                'shadow_color': 'black@0.8'
            },
            'green_theme': {
                'font_color': 'white',
                'background_color': '#047857@0.7',  # Green background
                'border_color': '#065F46',
                'shadow_color': 'black@0.8'
            }
        }
    
    def convert_api_style_to_ffmpeg_style(self, api_style: Dict) -> Dict:
        """
        Convert API style object to FFmpeg-compatible style format.
        
        Args:
            api_style: Style object from API with font_style, font_weight, default_color, etc.
            
        Returns:
            FFmpeg-compatible style dictionary
        """
        # Start with default style
        style = self._get_default_style()
        
        if not api_style:
            return style
        
        # Map API style properties to FFmpeg style
        if 'default_color' in api_style:
            style['font_color'] = api_style['default_color']
        
        if 'pronunciation_color' in api_style:
            style['highlight_color'] = api_style['pronunciation_color']
        
        if 'font_family' in api_style:
            style['font_family'] = api_style['font_family']
        
        if 'font_weight' in api_style:
            # Map font weight to ASS bold setting (-1 = bold, 0 = normal)
            weight = api_style['font_weight']
            # Convert integer weight to bold/normal (600+ is considered bold)
            if isinstance(weight, int) and weight >= 600:
                style['ass_bold'] = -1  # ASS uses -1 for bold
                style['font_weight'] = 'bold'  # Keep for FFmpeg compatibility
            elif isinstance(weight, str) and weight in ['bold', '600', '700', '800', '900']:
                # Handle legacy string values for backward compatibility
                style['ass_bold'] = -1  # ASS uses -1 for bold
                style['font_weight'] = 'bold'  # Keep for FFmpeg compatibility
            else:
                style['ass_bold'] = 0   # ASS uses 0 for normal
                style['font_weight'] = 'normal'
        else:
            style['ass_bold'] = 0  # Default to normal
        
        if 'font_style' in api_style:
            # Map font style to ASS italic setting (1 = italic, 0 = normal)
            font_style = api_style['font_style']
            if font_style == 'italic':
                style['ass_italic'] = 1  # ASS uses 1 for italic
                style['font_style'] = 'italic'
            else:
                style['ass_italic'] = 0  # ASS uses 0 for normal
                style['font_style'] = 'normal'
        else:
            style['ass_italic'] = 0  # Default to normal
        
        if 'text_transform' in api_style:
            style['text_transform'] = api_style['text_transform'] if api_style['text_transform'] else None
        
        if 'letter_spacing' in api_style:
            # Parse letter spacing (e.g., "3px" -> 3)
            spacing = api_style['letter_spacing']
            if spacing and isinstance(spacing, str):
                # Extract numeric value from string like "3px"
                import re
                match = re.search(r'(\d+)', spacing)
                if match:
                    spacing_pixels = int(match.group(1))
                    style['letter_spacing'] = spacing_pixels  # For FFmpeg
                    style['ass_spacing'] = spacing_pixels      # For ASS format
            else:
                style['ass_spacing'] = 0  # Default spacing
        else:
            style['ass_spacing'] = 0  # Default spacing
        
        # Set default ASS values if not already set
        style.setdefault('ass_underline', 0)  # No underline by default
        style.setdefault('ass_strikeout', 0)  # No strikeout by default
        style.setdefault('ass_scale_x', 100)  # Normal width scaling
        style.setdefault('ass_scale_y', 100)  # Normal height scaling
        style.setdefault('ass_angle', 0)      # No rotation
        
        self.logger.info(f"Converted API style to FFmpeg format with ASS properties: {style}")
        return style

    def get_style_by_id(self, style_id: str, api_style: Optional[Dict] = None) -> Dict:
        """
        Get subtitle style configuration by JSON config ID or API style.
        
        Args:
            style_id: Style ID from subtitle_config.json (e.g., 'cinematic', 'elegant', 'tech')
            api_style: Custom API style object that overrides config-based styling
            
        Returns:
            Complete style configuration dictionary
        """
        # If API style is provided, use it instead of config-based styling
        if api_style:
            self.logger.info("Using API-provided style configuration")
            return self.convert_api_style_to_ffmpeg_style(api_style)
        
        # Try to get style from JSON configuration first
        json_style = self.config_loader.get_style_by_id(style_id)
        
        if json_style:
            # Convert JSON style to FFmpeg-compatible format
            style = self.config_loader.convert_json_style_to_ffmpeg_style(json_style)
            self.logger.info(f"Using JSON subtitle style: {style_id} ({json_style.get('name', 'Unknown')})")
            return style
        
        # Fallback to legacy preset styles for backward compatibility
        if style_id in self.preset_styles:
            style = self._get_default_style()
            style.update(self.preset_styles[style_id])
            self.logger.info(f"Using legacy subtitle style preset: {style_id}")
            return style
        
        # If neither found, use default and log warning
        self.logger.warning(f"Style ID '{style_id}' not found in JSON config or legacy presets")
        available_json_ids = self.config_loader.get_available_style_ids()
        available_legacy_ids = list(self.preset_styles.keys())
        self.logger.info(f"Available JSON style IDs: {available_json_ids}")
        self.logger.info(f"Available legacy style IDs: {available_legacy_ids}")
        
        return self._get_default_style()

    def get_style_by_preset(self, preset_name: str = 'classic') -> Dict:
        """
        Get subtitle style configuration by preset name.
        
        Args:
            preset_name: Name of the preset style ('classic', 'modern', 'high_contrast', etc.)
            
        Returns:
            Complete style configuration dictionary
        """
        # Start with default style
        style = self._get_default_style()
        
        # Apply preset if it exists
        if preset_name in self.preset_styles:
            style.update(self.preset_styles[preset_name])
            self.logger.info(f"Applied '{preset_name}' subtitle style preset")
        else:
            self.logger.warning(f"Preset '{preset_name}' not found, using default style")
            available_presets = list(self.preset_styles.keys())
            self.logger.info(f"Available presets: {', '.join(available_presets)}")
        
        return style


    
    def srttodict(srt_path):
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            srt_text = f.read()

        entries = re.split(r'\n\s*\n', srt_text.strip())
        segments = []
        for entry in entries:
            lines = entry.strip().splitlines()
            if len(lines) < 2:
                continue
            # Find timestamp line
            timestamp_line = None
            for ln in lines:
                if '-->' in ln:
                    timestamp_line = ln.strip()
                    break
            if not timestamp_line:
                continue
            try:
                start_str, end_str = [s.strip() for s in timestamp_line.split('-->')]
                # convert to seconds
                def srt_time_to_seconds(tstr):
                    h, m, rest = tstr.split(':')
                    s, ms = rest.split(',')
                    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                start_secs = srt_time_to_seconds(start_str)
                end_secs = srt_time_to_seconds(end_str)
                text_lines = []
                saw_timestamp = False
                for ln in lines:
                    if saw_timestamp:
                        text_lines.append(ln)
                    if '-->' in ln:
                        saw_timestamp = True
                text = "\n".join(text_lines).strip()
                segments.append({'start': start_secs, 'end': end_secs, 'text': text})
            except Exception:
                continue
        return {'segments': segments, 'has_word_timing': False}   

    
    def add_subtitle_overlay(self, 
                           input_video_path: str,
                           output_video_path: str,
                           transcription_data: Dict,
                           segment_start_time: float,
                           segment_end_time: float,
                           style_config: Optional[Dict] = None,
                           style_preset: str = 'classic',
                           style_id: Optional[str] = None,
                           api_style: Optional[Dict] = None,
                           caption_x: Optional[int] = None,
                           caption_y: Optional[int] = None,
                           canvas_type: str = "shorts",
                           external_srt_path: Optional[str] = None,
                           timing_offset: float = 1.700) -> bool:
        """
        Add subtitle overlay to a video segment using transcription data.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path for output video with subtitles
            transcription_data: Whisper transcription data with segments
            segment_start_time: Start time of the video segment in original video
            segment_end_time: End time of the video segment in original video
            style_config: Custom styling configuration for subtitles (overrides preset and style_id)
            style_preset: Legacy preset style name (deprecated, use style_id instead)
            style_id: Style ID from subtitle_config.json (e.g., 'cinematic', 'elegant', 'tech')
            api_style: API-provided style object (overrides all other style options)
            caption_x: X coordinate for caption position (0 = left edge, increases rightward)
            caption_y: Y coordinate for caption position (0 = bottom edge, increases upward)
            canvas_type: Canvas type ("shorts" or "clips") for coordinate system scaling
            timing_offset: Seconds to add to all subtitle timings (default 1.700s to fix early display)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get style configuration - priority: api_style > custom config > style_id > legacy preset
            if api_style:
                style = self.convert_api_style_to_ffmpeg_style(api_style)
                self.logger.info("Using API-provided style configuration")
            elif style_config:
                style = self._get_default_style()
                style.update(style_config)
                self.logger.info("Using custom style configuration")
            elif style_id:
                style = self.get_style_by_id(style_id)
                self.logger.info(f"Using JSON style ID: '{style_id}'")
            else:
                style = self.get_style_by_preset(style_preset)
                self.logger.info(f"Using legacy preset: '{style_preset}'")
            
            # TEMPORARY FIX: Disable coordinate positioning system - force bottom center
            # if caption_x is not None and caption_y is not None:
            #     style = self._apply_coordinate_positioning(style, caption_x, caption_y, canvas_type, input_video_path)
            #     self.logger.info(f"Applied coordinate positioning: ({caption_x}, {caption_y}) for {canvas_type}")
            
            # TEMPORARY FIX: Force bottom-center positioning for ALL subtitles
            if caption_x is not None and caption_y is not None:
                self.logger.info(f"TEMPORARY: Ignoring coordinate positioning ({caption_x}, {caption_y}) and using bottom-center")
            else:
                self.logger.info("TEMPORARY: No coordinates provided, ensuring bottom-center positioning")
            
            # Set bottom-center alignment for all subtitles
            style['alignment'] = 2  # Center alignment
            style['margin_v'] = 50  # 50px from bottom
            style['margin_l'] = 0   # No left margin
            style['margin_r'] = 0   # No right margin

            # If an external SRT path was provided, parse it and convert to transcription segments
            if external_srt_path:
                self.logger.info(f"Using external SRT file for subtitles: {external_srt_path}")
                transcription_data = srttodict(external_srt_path)



            # Filter transcription segments that overlap with this video segment
            relevant_segments = self._filter_transcription_segments(
                transcription_data,
                segment_start_time,
                segment_end_time,
                timing_offset
            )
            
            if not relevant_segments:
                self.logger.info("No transcription segments found for this video segment, copying without subtitles")
                # If no transcription for this segment, just copy the input to output
                return self._copy_video(input_video_path, output_video_path)
            
            # Create SRT subtitle file
            srt_path = self._create_srt_file(relevant_segments, segment_start_time)
            
            if not srt_path:
                self.logger.warning("Failed to create SRT file, copying without subtitles")
                return self._copy_video(input_video_path, output_video_path)
            
            try:
                # Apply subtitles using FFmpeg
                success = self._apply_subtitles_with_ffmpeg(
                    input_video_path,
                    output_video_path,
                    #'/app/font/subtitles2.srt',
                    srt_path,
                    style
                )
                
                return success
                
            finally:
                # Cleanup temporary SRT file
                if os.path.exists(srt_path):
                    os.unlink(srt_path)
                    
        except Exception as e:
            self.logger.error(f"Error adding subtitle overlay: {e}", exc_info=True)
            # Fallback: copy input to output without subtitles
            return self._copy_video(input_video_path, output_video_path)

    
    def add_kinetic_subtitle_overlay(self, 
                                    input_video_path: str,
                                    output_video_path: str,
                                    transcription_data: Dict,
                                    segment_start_time: float,
                                    segment_end_time: float,
                                    style_config: Optional[Dict] = None,
                                    style_preset: str = 'classic',
                                    style_id: Optional[str] = None,
                                    api_style: Optional[Dict] = None,
                                    caption_x: Optional[int] = None,
                                    caption_y: Optional[int] = None,
                                    canvas_type: str = "shorts",
                                    kinetic_mode: str = "karaoke") -> bool:
        """
        Add kinetic subtitle overlay with word-level timing animations.
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path for output video with kinetic subtitles
            transcription_data: Whisper transcription data with word-level timing
            segment_start_time: Start time of the video segment in original video
            segment_end_time: End time of the video segment in original video
            style_config: Custom styling configuration for subtitles
            style_preset: Legacy preset style name
            style_id: Style ID from subtitle_config.json
            api_style: API-provided style object (overrides all other style options)
            caption_x: X coordinate for caption position
            caption_y: Y coordinate for caption position  
            canvas_type: Canvas type ("shorts" or "clips")
            kinetic_mode: Type of kinetic effect ("karaoke", "typewriter", "highlight")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if word-level timing is available
            has_word_timing = self._has_word_level_timing(transcription_data)
            
            if not has_word_timing:
                self.logger.warning("No word-level timing available, falling back to standard captions")
                return self.add_subtitle_overlay(
                    input_video_path, output_video_path, transcription_data,
                    segment_start_time, segment_end_time, style_config, 
                    style_preset, style_id, api_style, caption_x, caption_y, canvas_type
                )
            
            # Get style configuration - priority: api_style > custom config > style_id > legacy preset
            if api_style:
                style = self.convert_api_style_to_ffmpeg_style(api_style)
                self.logger.info("Using API-provided style configuration for kinetic captions")
            elif style_config:
                style = self._get_default_style()
                style.update(style_config)
                self.logger.info("Using custom style configuration for kinetic captions")
            elif style_id:
                style = self.get_style_by_id(style_id)
                self.logger.info(f"Using JSON style ID for kinetic captions: '{style_id}'")
            else:
                style = self.get_style_by_preset(style_preset)
                self.logger.info(f"Using legacy preset for kinetic captions: '{style_preset}'")
            
            # TEMPORARY FIX: Disable coordinate positioning system - force bottom center
            # if caption_x is not None and caption_y is not None:
            #     style = self._apply_coordinate_positioning(style, caption_x, caption_y, canvas_type, input_video_path)
            #     self.logger.info(f"Applied coordinate positioning for kinetic captions: ({caption_x}, {caption_y}) for {canvas_type}")
            
            # TEMPORARY FIX: Force bottom-center positioning for ALL kinetic subtitles
            if caption_x is not None and caption_y is not None:
                self.logger.info(f"TEMPORARY: Ignoring kinetic coordinate positioning ({caption_x}, {caption_y}) and using bottom-center")
            else:
                self.logger.info("TEMPORARY: No coordinates provided for kinetic, ensuring bottom-center positioning")
            
            # Set bottom-center alignment for all kinetic subtitles
            style['alignment'] = 2  # Center alignment
            style['margin_v'] = 50  # 50px from bottom
            style['margin_l'] = 0   # No left margin
            style['margin_r'] = 0   # No right margin
            
            # Add highlight color for kinetic effects
            if 'highlight_color' not in style:
                style['highlight_color'] = 'yellow'  # Default highlight color
            
            # Apply kinetic subtitles
            self.logger.info(f"Applying kinetic subtitles with mode: {kinetic_mode}")
            success = self.kinetic_processor.apply_kinetic_subtitles(
                input_video_path,
                output_video_path,
                transcription_data,
                segment_start_time,
                segment_end_time,
                style,
                kinetic_mode
            )
            
            if success:
                self.logger.info(f"Successfully applied kinetic subtitles with {kinetic_mode} effect")
            else:
                self.logger.error("Failed to apply kinetic subtitles")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error adding kinetic subtitle overlay: {e}", exc_info=True)
            # Fallback: use standard subtitles
            self.logger.info("Falling back to standard subtitles due to kinetic subtitle error")
            return self.add_subtitle_overlay(
                input_video_path, output_video_path, transcription_data,
                segment_start_time, segment_end_time, style_config,
                style_preset, style_id, api_style, caption_x, caption_y, canvas_type
            )
    
    def _has_word_level_timing(self, transcription_data: Dict) -> bool:
        """Check if transcription has word-level timing data."""
        # First check the has_word_timing flag if available (set by WhisperAnalyzer)
        if 'has_word_timing' in transcription_data:
            has_timing = transcription_data['has_word_timing']
            self.logger.info(f"Transcription has_word_timing flag: {has_timing}")
            if has_timing:
                return True
        
        # Fallback: check segments for word-level data with proper timing
        segments = transcription_data.get('segments', [])
        if not segments:
            self.logger.info("No segments found in transcription data")
            return False

    def add_subtitle_overlay_from_srt(self,
                                      input_video_path: str,
                                      output_video_path: str,
                                      external_srt_path: str,
                                      segment_start_time: float,
                                      segment_end_time: float,
                                      style_config: Optional[Dict] = None,
                                      style_preset: str = 'classic',
                                      style_id: Optional[str] = None,
                                      api_style: Optional[Dict] = None,
                                      caption_x: Optional[int] = None,
                                      caption_y: Optional[int] = None,
                                      canvas_type: str = "shorts") -> bool:
        """
        Apply an external full-video SRT to a clipped segment by extracting the relevant
        subtitle lines, time-shifting them to start at 0 for the clip, and burning them
        into the clip using FFmpeg.

        Returns True on success, False otherwise.
        """
        try:
            # Validate external SRT
            if not external_srt_path or not os.path.exists(external_srt_path):
                self.logger.error(f"External SRT not found: {external_srt_path}")
                return False

            # Read SRT content
            with open(external_srt_path, 'r', encoding='utf-8', errors='ignore') as f:
                srt_text = f.read()

            # Simple SRT parser - split into blocks
            entries = re.split(r'\n\s*\n', srt_text.strip())

            filtered_entries = []
            seq = 1
            srt_start_time = None
            srt_end_time = None
           
            for entry in entries:
                lines = entry.strip().splitlines()
                if len(lines) < 2:
                    continue
                # First non-empty line can be index or timestamp
                # Find the timestamp line
                timestamp_line = None
                for ln in lines:
                    if '-->' in ln:
                        timestamp_line = ln.strip()
                        break
                if not timestamp_line:
                    continue

                try:
                    start_str, end_str = [s.strip() for s in timestamp_line.split('-->')]
                    def srt_time_to_seconds(tstr):
                        h, m, rest = tstr.split(':')
                        s, ms = rest.split(',')
                        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

                    start_secs = srt_time_to_seconds(start_str)
                    end_secs = srt_time_to_seconds(end_str)
                except Exception:
                    continue

                   # Track SRT time range
                if srt_start_time is None or start_secs < srt_start_time:
                    srt_start_time = start_secs
                if srt_end_time is None or end_secs > srt_end_time:
                    srt_end_time = end_secs

                # Check overlap
                if end_secs <= segment_start_time or start_secs >= segment_end_time:
                    self.logger.debug(f"SRT entry {start_str}-->{end_str} ({start_secs:.2f}-->{end_secs:.2f}s) does not overlap with segment ({segment_start_time:.2f}-->{segment_end_time:.2f}s)")
                    continue

                self.logger.debug(f"Found overlapping SRT entry: {start_str}-->{end_str} ({start_secs:.2f}-->{end_secs:.2f}s) overlaps with segment ({segment_start_time:.2f}-->{segment_end_time:.2f}s)")

                # Clip to segment bounds and shift times relative to segment_start_time
                new_start = max(0.0, start_secs - segment_start_time)
                new_end = max(0.0, end_secs - segment_start_time)

                # Build new SRT block
                # Compose text lines (lines after timestamp)
                text_lines = []
                saw_timestamp = False
                for ln in lines:
                    if saw_timestamp:
                        text_lines.append(ln)
                    if '-->' in ln:
                        saw_timestamp = True

                # Format times back to SRT string
                def seconds_to_srt(ts):
                    hours = int(ts // 3600)
                    minutes = int((ts % 3600) // 60)
                    secs = int(ts % 60)
                    milliseconds = int((ts % 1) * 1000)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

                srt_block = f"{seq}\n{seconds_to_srt(new_start)} --> {seconds_to_srt(new_end)}\n" + "\n".join(text_lines)
                filtered_entries.append(srt_block)
                seq += 1

            if not filtered_entries:
                   # Debug: Show SRT coverage
                    if srt_start_time is not None and srt_end_time is not None:
                       self.logger.warning(f"SRT file covers {srt_start_time:.2f}-->{srt_end_time:.2f}s, but clip is {segment_start_time:.2f}-->{segment_end_time:.2f}s - NO OVERLAP!")
                    else:
                       self.logger.warning(f"SRT file is empty or unparseable")
                    self.logger.info("No subtitle lines overlap with the clip time range; skipping subtitle overlay")

                    # If no overlap detected, try a workaround: assume SRT is universal (meant for all clips)
                    # Reset and parse SRT again, but treat all times as 0-based (for the clip, not original video)
                    self.logger.info("Attempting workaround: treating SRT as universal subtitles (0-based timing)")
            
                    filtered_entries = []
                    seq = 1
                    for entry in entries:
                        lines = entry.strip().splitlines()
                        if len(lines) < 2:
                            continue
                        timestamp_line = None
                        for ln in lines:
                            if '-->' in ln:
                                timestamp_line = ln.strip()
                                break
                        if not timestamp_line:
                            continue
                
                        try:
                            start_str, end_str = [s.strip() for s in timestamp_line.split('-->')]
                            def srt_time_to_seconds(tstr):
                                h, m, rest = tstr.split(':')
                                s, ms = rest.split(',')
                                return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
                    
                            start_secs = srt_time_to_seconds(start_str)
                            end_secs = srt_time_to_seconds(end_str)
                    
                            # For universal subtitles, just check if they fit within the clip duration
                            clip_duration = segment_end_time - segment_start_time
                            if end_secs <= clip_duration:
                                # Use SRT times as-is (0-based for the clip)
                                new_start = start_secs
                                new_end = end_secs
                            else:
                                # SRT entry extends beyond clip, trim it
                                new_start = start_secs
                                new_end = min(end_secs, clip_duration)
                                if new_end <= new_start:
                                    continue  # Skip if trimming makes it invalid
                    
                            # Build SRT block
                            text_lines = []
                            saw_timestamp = False
                            for ln in lines:
                                if saw_timestamp:
                                    text_lines.append(ln)
                                if '-->' in ln:
                                    saw_timestamp = True
                    
                            def seconds_to_srt(ts):
                                hours = int(ts // 3600)
                                minutes = int((ts % 3600) // 60)
                                secs = int(ts % 60)
                                milliseconds = int((ts % 1) * 1000)
                                return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
                    
                            srt_block = f"{seq}\n{seconds_to_srt(new_start)} --> {seconds_to_srt(new_end)}\n" + "\n".join(text_lines)
                            filtered_entries.append(srt_block)
                            seq += 1
                        except Exception as e:
                            self.logger.debug(f"Error parsing SRT entry in workaround: {e}")
                            continue
            
                    if not filtered_entries:
                        self.logger.error(f"Workaround also failed - no subtitles to apply. SRT file may not be compatible with clips.")
                        # Return False to signal that no subtitles were applied
                        return False
                    else:
                        self.logger.info(f"Workaround succeeded: using {len(filtered_entries)} universal subtitle entries")
            # Write temporary SRT file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.srt')
            tmp_path = tmp.name
            try:
                with open(tmp_path, 'w', encoding='utf-8') as out_f:
                    out_f.write('\n\n'.join(filtered_entries))

                # Determine style (reuse same logic as add_subtitle_overlay)
                if api_style:
                    style = self.convert_api_style_to_ffmpeg_style(api_style)
                    self.logger.info("Using API-provided style configuration for external SRT overlay")
                elif style_config:
                    style = self._get_default_style()
                    style.update(style_config)
                    self.logger.info("Using custom style configuration for external SRT overlay")
                elif style_id:
                    style = self.get_style_by_id(style_id)
                    self.logger.info(f"Using JSON style ID: '{style_id}' for external SRT overlay")
                else:
                    style = self.get_style_by_preset(style_preset)

                # Apply subtitles using FFmpeg
                success = self._apply_subtitles_with_ffmpeg(
                    input_video_path,
                    output_video_path,
                    tmp_path,
                    style
                )

                return success
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Error applying external SRT overlay: {e}", exc_info=True)
            return False
        
        # Check if any segment has word-level data with proper timing
        words_with_timing = 0
        total_words = 0
        
        for segment in segments:
            if 'words' in segment and segment['words']:
                for word_data in segment['words']:
                    total_words += 1
                    if (isinstance(word_data, dict) and 
                        'start' in word_data and 'end' in word_data and
                        word_data['start'] != word_data['end']):
                        words_with_timing += 1
        
        has_word_timing = words_with_timing > 0
        if total_words > 0:
            coverage = (words_with_timing / total_words) * 100
            self.logger.info(f"Word timing check: {words_with_timing}/{total_words} words ({coverage:.1f}%) have timing")
        else:
            self.logger.info("No words found in transcription segments")
        
        return has_word_timing
    
    def _get_default_style(self) -> Dict:
        """
        Get default subtitle styling configuration.
        
        Color Options:
        - Named colors: 'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta'
        - Hex colors: '#FFFFFF', '#000000', '#FF0000', etc.
        - Colors with transparency: 'white@0.9', 'black@0.7', '#000000@0.5'
        
        Popular Color Combinations:
        1. Classic: white text, black@0.7 background
        2. Modern: white text, black@0.3 background (very subtle)
        3. High contrast: yellow text, black@0.8 background
        4. Elegant: white text, #333333@0.6 background
        5. Vibrant: white text, #FF6B35@0.7 background (orange)
        """
        return {
            'font_family': 'Arial',
            'font_size': 14,  # Reduced to 50% of original size (was 28)
            'font_color': 'white',  # Main text color
            'background_color': 'black@0.7',  # Semi-transparent background (higher opacity for better readability)
            'border_width': 3,  # Increased border for better contrast
            'border_color': 'black',  # Border/outline color
            'alignment': 2,  # Bottom center (1=left, 2=center, 3=right)
            'margin_v': 60,  # Vertical margin from bottom (pixels)
            'margin_l': 0,  # Left margin (pixels)
            'margin_r': 0,  # Right margin (pixels)
            'shadow_offset': '3,3',  # Shadow offset (x,y)
            'shadow_color': 'black@0.9'  # Shadow color with transparency
        }
    
    def _filter_transcription_segments(self, 
                                     transcription_data: Dict,
                                     segment_start: float,
                                     segment_end: float,
                                     timing_offset: float = 1.700) -> List[Dict]:
        """
        Filter transcription segments that overlap with the video segment.
        
        Args:
            transcription_data: Whisper transcription data
            segment_start: Start time of video segment
            segment_end: End time of video segment
            
        Returns:
            List of relevant transcription segments with adjusted timestamps
        """
        relevant_segments = []
        
        # Get segments from transcription data
        segments = transcription_data.get('segments', [])
        
        # Debug logging
        self.logger.info(f"Filtering transcription segments: video_segment={segment_start:.2f}-{segment_end:.2f}s")
        self.logger.info(f"Available transcription segments: {len(segments)}")
        if not segments:
            self.logger.warning("No segments found in transcription_data")
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Skip empty text segments
            if not text:
                continue
            
            # Check if segment overlaps with our video segment
            if seg_end <= segment_start or seg_start >= segment_end:
                self.logger.debug(f"No overlap: seg({seg_start:.2f}-{seg_end:.2f}) vs video({segment_start:.2f}-{segment_end:.2f})")
                continue  # No overlap
            
            self.logger.info(f"Found overlapping segment: {seg_start:.2f}-{seg_end:.2f}s")
            
            # Calculate overlap
            overlap_start = max(seg_start, segment_start)
            overlap_end = min(seg_end, segment_end)
            
            # Check if too much of this segment was spoken before the video segment
            # This prevents subtitles from before the clip appearing at the start
            segment_duration = seg_end - seg_start
            overlap_duration = overlap_end - overlap_start
            overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
            
            # Only include segments where at least 50% is within the video segment
            if overlap_percentage < 0.5:
                self.logger.debug(f"Excluding segment {seg_start:.2f}-{seg_end:.2f}s: only {overlap_percentage:.1%} overlaps with video segment")
                continue
            
            # Adjust timestamps relative to the video segment start
            adjusted_start = overlap_start - segment_start
            adjusted_end = overlap_end - segment_start
            
            # Apply configurable timing offset to fix subtitle sync
            adjusted_start += timing_offset
            adjusted_end += timing_offset
            
            # Ensure positive timestamps and minimum duration
            adjusted_start = max(0, adjusted_start)
            adjusted_end = max(adjusted_start + 0.1, adjusted_end)  # Minimum 0.1s duration
            
            # Apply tiered threshold based on overlap percentage
            # High overlap = more permissive timing, Low overlap = stricter timing
            if overlap_percentage >= 0.8:
                min_start_threshold = 0.0  # Allow immediate start for high overlap
            elif overlap_percentage >= 0.6:
                min_start_threshold = 0.1  # Small delay for good overlap
            else:  # 0.5 <= overlap_percentage < 0.6
                min_start_threshold = 0.3  # Moderate delay for marginal overlap
            
            if adjusted_start < min_start_threshold:
                self.logger.debug(f"Excluding segment {seg_start:.2f}-{seg_end:.2f}s: would start at {adjusted_start:.2f}s < {min_start_threshold:.1f}s threshold (overlap: {overlap_percentage:.1%})")
                continue
            
            relevant_segments.append({
                'start': adjusted_start,
                'end': adjusted_end,
                'text': text,
                'original_start': seg_start,
                'original_end': seg_end
            })
        
        # Sort by start time
        relevant_segments.sort(key=lambda x: x['start'])
        
        self.logger.info(f"Found {len(relevant_segments)} relevant transcription segments for video segment")
        
        return relevant_segments
    
    def _create_srt_file(self, segments: List[Dict], segment_start_time: float) -> Optional[str]:
        """
        Create a temporary SRT subtitle file from transcription segments.
        
        Args:
            segments: List of transcription segments with adjusted timestamps
            segment_start_time: Start time of the video segment (for logging)
            
        Returns:
            Path to temporary SRT file, or None if failed
        """
        try:
            # Create temporary SRT file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
                srt_path = f.name
                
                for i, segment in enumerate(segments, 1):
                    start_time = segment['start']
                    end_time = segment['end']
                    text = segment['text'].strip()
                    
                    # Clean up text (remove excessive whitespace, fix encoding issues)
                    text = self._clean_subtitle_text(text)
                    
                    # Skip if text is empty after cleaning
                    if not text:
                        continue
                    
                    # Convert seconds to SRT time format (HH:MM:SS,mmm)
                    start_srt = self._seconds_to_srt_time(start_time)
                    end_srt = self._seconds_to_srt_time(end_time)
                    
                    # Write SRT entry
                    f.write(f"{i}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{text}\n\n")
                    
                    self.logger.debug(f"Added subtitle: {start_srt} --> {end_srt}: {text[:50]}...")
            
            self.logger.info(f"Created SRT file with {len(segments)} subtitle entries: {srt_path}")
            return srt_path
            
        except Exception as e:
            self.logger.error(f"Failed to create SRT file: {e}")
            return None
    
    def _clean_subtitle_text(self, text: str) -> str:
        """
        Clean subtitle text for better display.
        
        Args:
            text: Raw text from transcription
            
        Returns:
            Cleaned text suitable for subtitles
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove or replace problematic characters
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # Limit line length (split long lines)
        max_chars_per_line = 40
        if len(text) > max_chars_per_line:
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Limit to 2 lines maximum
            if len(lines) > 2:
                lines = lines[:2]
                lines[1] = lines[1] + '...'
            
            text = '\n'.join(lines)
        
        return text
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm).
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Time string in SRT format
        """
        hours = int(seconds // 3600)  # Fixed: 3600 seconds in an hour, not 3450
        minutes = int((seconds % 3600) // 60)  # Fixed: use 3600 for proper calculation
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _apply_subtitles_with_ffmpeg(self,
                                   input_path: str,
                                   output_path: str,
                                   srt_path: str,
                                   style: Dict) -> bool:
        """
        Apply subtitles to video using FFmpeg.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            srt_path: SRT subtitle file path
            style: Subtitle styling configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Escape the SRT file path for FFmpeg
            srt_escaped = srt_path.replace('\\', '\\\\').replace(':', '\\:')
            
            # Build FFmpeg subtitle filter with styling
            # Use simplified color format for better compatibility
            font_color = self._color_to_ffmpeg(style['font_color'])
            back_color = self._color_to_ffmpeg(style['background_color'])
            outline_color = self._color_to_ffmpeg(style['border_color'])
            shadow_color = self._color_to_ffmpeg(style['shadow_color'])
            
            subtitle_filter = (
                f"subtitles='{srt_escaped}'"
                f":force_style='Fontname=\"Noto Sans\","
                f"Fontsize={style['font_size']},"
                f"PrimaryColour={font_color},"
                f"BackColour={back_color}," 
                f"OutlineColour={outline_color},"
                f"Outline={style['border_width']},"
                f"BorderStyle=1,"
                f"Alignment={style['alignment']},"
                f"MarginV={style['margin_v']},"
                f"MarginL={style.get('margin_l', 0)},"
                f"MarginR={style.get('margin_r', 0)},"
                f"Shadow=2,"
                f"ShadowColour={shadow_color}'"
            )
            
            self.logger.info(f"Using font color: {font_color} for style: {style['font_color']}")
            self.logger.info(f"Subtitle filter: {subtitle_filter}")
            
            # Build FFmpeg command with explicit stream mapping to avoid subtitle stream issues
            # First check if input has audio stream
            import subprocess as sp
            
            # Probe for audio stream
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0', 
                '-show_entries', 'stream=index', '-of', 'csv=p=0', input_path
            ]
            
            try:
                probe_result = sp.run(probe_cmd, capture_output=True, text=True, timeout=10)
                has_audio = probe_result.returncode == 0 and probe_result.stdout.strip()
            except:
                has_audio = False
            
            # Get original video frame rate to preserve exact timing
            frame_rate_probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0', 
                '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', 
                input_path
            ]
            
            frame_rate = "30"  # Default fallback
            try:
                frame_rate_result = sp.run(frame_rate_probe_cmd, capture_output=True, text=True, timeout=10)
                if frame_rate_result.returncode == 0 and frame_rate_result.stdout.strip():
                    frame_rate_str = frame_rate_result.stdout.strip()
                    # Handle fractional frame rates like "30000/1001" for 29.97 fps
                    if '/' in frame_rate_str:
                        num, den = frame_rate_str.split('/')
                        frame_rate = f"{int(num)}/{int(den)}"
                    else:
                        frame_rate = frame_rate_str
                    self.logger.info(f"Detected original frame rate: {frame_rate}")
            except Exception as e:
                self.logger.warning(f"Could not detect frame rate, using default: {e}")

            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-vf', subtitle_filter,
                '-r', str(frame_rate),    # CRITICAL: Preserve exact original frame rate
                '-map', '0:v:0',  # Map only the first video stream
            ]
            
            # Only map audio if it exists
            if has_audio:
                ffmpeg_cmd.extend(['-map', '0:a:0', '-c:a', 'copy'])  # Copy audio without re-encoding
            
            ffmpeg_cmd.extend([
                '-c:v', 'libx264',  # Re-encode video to apply subtitles
                '-preset', 'medium',
                '-crf', '23',
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-vsync', '1',              # Ensure proper video frame sync
                '-async', '1',              # Fix audio sync to video (when audio exists)
                '-copytb', '1',             # Copy input timebase to output
                '-fflags', '+genpts',       # Generate presentation timestamps
                '-y',  # Overwrite output file
                output_path
            ])
            
            self.logger.debug(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully added subtitles to video: {output_path}")
                return True
            else:
                self.logger.error(f"FFmpeg failed with return code {result.returncode}")
                self.logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg timed out while adding subtitles")
            return False
        except Exception as e:
            self.logger.error(f"Error running FFmpeg for subtitles: {e}")
            return False
    
    def _color_to_ffmpeg(self, color: str) -> str:
        """
        Convert color specification to FFmpeg ASS format.
        
        Args:
            color: Color specification (e.g., 'white', 'black@0.5', '#FFFFFF')
            
        Returns:
            FFmpeg-compatible color string
        """
        # Handle transparency
        if '@' in color:
            base_color, alpha = color.split('@')
            alpha_val = int(float(alpha) * 255)
        else:
            base_color = color
            alpha_val = 255  # Fully opaque
        
        # Convert named colors to hex
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': 'FF0000',
            'green': '00FF00',
            'blue': '0000FF',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'magenta': 'FF00FF'
        }
        
        if base_color.lower() in color_map:
            hex_color = color_map[base_color.lower()]
        elif base_color.startswith('#'):
            hex_color = base_color[1:]
        else:
            hex_color = 'FFFFFF'  # Default to white
        
        # For transparent backgrounds, use special handling
        if base_color.lower() == 'transparent':
            return "&H00000000"  # Fully transparent
        
        # FFmpeg uses ABGR format (Alpha, Blue, Green, Red)
        if len(hex_color) == 6:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Convert to ABGR format: &HAABBGGRR
            # For text color, we want full opacity for better visibility
            if base_color.lower() == 'white' and alpha_val == 255:
                return "&HFFFFFF"  # Simplified format for pure white
            
            return f"&H{alpha_val:02X}{b:02X}{g:02X}{r:02X}"
        
        return "&HFFFFFF"  # Default solid white
    
    def _copy_video(self, input_path: str, output_path: str) -> bool:
        """
        Copy video without modification (fallback when no subtitles needed).
        
        Args:
            input_path: Input video path
            output_path: Output video path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if input has audio stream
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0', 
                '-show_entries', 'stream=index', '-of', 'csv=p=0', input_path
            ]
            
            try:
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                has_audio = probe_result.returncode == 0 and probe_result.stdout.strip()
            except:
                has_audio = False
            
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_path,
                '-map', '0:v:0',  # Map only the first video stream
            ]
            
            # Only map audio if it exists
            if has_audio:
                ffmpeg_cmd.extend(['-map', '0:a:0', '-c:a', 'copy'])  # Copy audio without re-encoding
            
            ffmpeg_cmd.extend([
                '-c:v', 'copy',  # Copy video without re-encoding
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-y',  # Overwrite output file
                output_path
            ])
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully copied video without subtitles: {output_path}")
                return True
            else:
                self.logger.error(f"FFmpeg copy failed with return code {result.returncode}")
                self.logger.error(f"FFmpeg stderr: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error copying video: {e}")
            return False
    
    def _apply_coordinate_positioning(self, style: Dict, caption_x: int, caption_y: int, 
                                    canvas_type: str, video_path: str) -> Dict:
        """
        Apply coordinate-based positioning to subtitle style.
        
        Args:
            style: Base style configuration
            caption_x: X coordinate (0 = left edge, increases rightward)
            caption_y: Y coordinate (0 = bottom edge, increases upward)
            canvas_type: Canvas type ("shorts" or "clips") for coordinate system scaling
            video_path: Path to video file for dimension detection
            
        Returns:
            Updated style configuration with positioning
        """
        try:
            # Get video dimensions
            video_info = self._get_video_info(video_path)
            if not video_info:
                self.logger.error("Failed to retrieve video info for positioning")
                self.logger.warning("Could not get video dimensions, using defaults")
                video_width = 1080 if canvas_type == "shorts" else 1920
                video_height = 1920 if canvas_type == "shorts" else 1080
            else:
                self.logger.info(f"Video dimensions: {video_info.get('width')}x{video_info.get('height')}, duration: {video_info.get('duration'):.2f}s")
                video_width = video_info.get('width', 1080)
                video_height = video_info.get('height', 1920)
            
            # Scale coordinates based on canvas type
            if canvas_type == "shorts":
                # For shorts (9:16), scale from (0,0)-(105,450) to video dimensions
                max_x, max_y = 105, 450
            else:  # clips (16:9)
                # For clips (16:9), scale from (0,0)-(315,240) to video dimensions
                max_x, max_y = 315, 240
            
            # Clamp coordinates to prevent out-of-bounds positioning
            clamped_x = max(0, min(caption_x, max_x))
            clamped_y = max(0, min(caption_y, max_y))
            
            if clamped_x != caption_x or clamped_y != caption_y:
                self.logger.warning(f"Caption coordinates clamped from ({caption_x}, {caption_y}) to ({clamped_x}, {clamped_y}) for {canvas_type} canvas (max: {max_x}, {max_y})")
            
            # Scale coordinates to video dimensions
            scaled_x = int((clamped_x / max_x) * video_width)
            scaled_y = int((clamped_y / max_y) * video_height)
            
            # Additional safety: ensure scaled coordinates don't exceed video dimensions
            scaled_x = max(0, min(scaled_x, video_width - 50))  # Leave 50px margin for text
            scaled_y = max(50, min(scaled_y, video_height - 50))  # Leave 50px margin for text
            
            # Convert Y coordinate from bottom-up to top-down for FFmpeg positioning
            # FFmpeg uses top-left origin, we use bottom-left
            ffmpeg_y = video_height - scaled_y
            
            # Update style with calculated position
            # FFmpeg subtitle positioning uses MarginV (vertical margin from bottom) and MarginL/MarginR
            style = style.copy()  # Don't modify original
            
            # Set horizontal alignment based on position
            if scaled_x < video_width / 3:
                # Left third
                style['alignment'] = 1  # Left alignment
                style['margin_l'] = scaled_x
                style['margin_r'] = 0
            elif scaled_x > 2 * video_width / 3:
                # Right third  
                style['alignment'] = 3  # Right alignment
                style['margin_l'] = 0
                style['margin_r'] = video_width - scaled_x
            else:
                # Center third
                style['alignment'] = 2  # Center alignment
                style['margin_l'] = 0
                style['margin_r'] = 0
            
            # Set vertical margin from bottom
            style['margin_v'] = scaled_y
            
            self.logger.info(f"Caption positioning transformation:")
            self.logger.info(f"  Input: ({caption_x}, {caption_y}) -> Clamped: ({clamped_x}, {clamped_y})")
            self.logger.info(f"  Video dims: {video_width}x{video_height}, Canvas: {canvas_type}, Limits: ({max_x}, {max_y})")
            self.logger.info(f"  Scaled: ({scaled_x}, {scaled_y}) -> FFmpeg: alignment={style['alignment']}, margin_v={style['margin_v']}")
            
            return style
            
        except Exception as e:
            self.logger.error(f"Error applying coordinate positioning: {e}")
            return style  # Return original style on error

    def _get_video_info(self, video_path: str) -> Optional[Dict]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'duration': float(probe_data.get('format', {}).get('duration', 0))
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            return None
