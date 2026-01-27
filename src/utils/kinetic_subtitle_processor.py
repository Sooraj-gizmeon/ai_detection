# src/utils/kinetic_subtitle_processor.py
"""Kinetic subtitle processor for word-level animated captions"""

import os
import logging
import tempfile
import subprocess
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class KineticSubtitleProcessor:
    """
    Handles creating kinetic captions with word-level timing animations.
    """
    
    def __init__(self):
        """Initialize kinetic subtitle processor."""
        self.logger = logging.getLogger(__name__)
        
    def _validate_video_file(self, video_path: str) -> bool:
        """Validate that a video file is readable and has streams."""
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Video file does not exist: {video_path}")
                return False
                
            # Use ffprobe to validate video file
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"FFprobe failed for {video_path}: {result.stderr}")
                return False
            
            import json
            probe_data = json.loads(result.stdout)
            video_streams = [s for s in probe_data.get('streams', []) if s.get('codec_type') == 'video']
            
            if not video_streams:
                self.logger.error(f"No video streams found in {video_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(video_path)
            if file_size < 1000:  # Less than 1KB indicates corruption
                self.logger.error(f"Video file too small ({file_size} bytes): {video_path}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Video validation failed for {video_path}: {e}")
            return False
    
    def create_kinetic_ass(self, 
                          transcription_data: Dict,
                          segment_start_time: float,
                          segment_end_time: float,
                          style_config: Dict,
                          kinetic_mode: str = "karaoke",
                          timing_offset: float = 1.700) -> str:
        """
        Create ASS subtitle file with kinetic word-level timing.
        
        Args:
            transcription_data: Whisper transcription with word-level timing
            segment_start_time: Start time of video segment
            segment_end_time: End time of video segment  
            style_config: Subtitle styling configuration
            kinetic_mode: Type of kinetic effect ("karaoke", "typewriter", "highlight")
            timing_offset: Seconds to add to all subtitle timings (default 1.700s to fix early display)
            
        Returns:
            ASS file content as string
        """
        # Check if word-level timing is available
        if not self._has_word_level_timing(transcription_data):
            self.logger.warning("No word-level timing available, falling back to standard captions")
            return None
        
        # Filter relevant segments
        relevant_segments = self._filter_transcription_segments(
            transcription_data, segment_start_time, segment_end_time
        )
        
        if not relevant_segments:
            self.logger.warning("No relevant segments found for kinetic captions")
            return None
        
        # Generate ASS content
        ass_header = self._create_ass_header(style_config)
        ass_events = self._create_kinetic_events(relevant_segments, segment_start_time, kinetic_mode, timing_offset)
        
        return f"{ass_header}\n{ass_events}"
    
    def _has_word_level_timing(self, transcription_data: Dict) -> bool:
        """Check if transcription has word-level timing data."""
        # First check the has_word_timing flag if available (set by WhisperAnalyzer)
        if 'has_word_timing' in transcription_data:
            has_timing = transcription_data['has_word_timing']
            self.logger.info(f"Transcription has_word_timing flag: {has_timing}")
            if has_timing:
                return True
        
        # Fallback: check segments for word-level data
        segments = transcription_data.get('segments', [])
        if not segments:
            self.logger.info("No segments found in transcription data")
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
    
    def _filter_transcription_segments(self, 
                                     transcription_data: Dict,
                                     segment_start: float,
                                     segment_end: float) -> List[Dict]:
        """Filter transcription segments that overlap with video segment."""
        relevant_segments = []
        
        segments = transcription_data.get('segments', [])
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # Check if segment overlaps with our video segment
            if seg_end <= segment_start or seg_start >= segment_end:
                continue  # No overlap
            
            # Only include segments with word-level data
            if 'words' not in segment or not segment['words']:
                continue
            
            # Check if too much of this segment was spoken before the video segment
            # This prevents subtitles from before the clip appearing at the start
            overlap_start = max(seg_start, segment_start)
            overlap_end = min(seg_end, segment_end)
            segment_duration = seg_end - seg_start
            overlap_duration = overlap_end - overlap_start
            overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
            
            # Only include segments where at least 50% is within the video segment
            if overlap_percentage < 0.5:
                continue
            
            # Adjust word timestamps relative to video segment start
            adjusted_words = []
            for word_data in segment['words']:
                word_start = word_data.get('start', 0)
                word_end = word_data.get('end', 0)
                
                # Skip words outside our segment
                if word_end <= segment_start or word_start >= segment_end:
                    continue
                
                # Adjust timestamps with higher precision
                adjusted_word = word_data.copy()
                # Use higher precision timestamp adjustment (no rounding to 0.1)
                adjusted_start = word_start - segment_start
                adjusted_end = word_end - segment_start
                
                # Apply tiered threshold based on segment overlap percentage
                if overlap_percentage >= 0.8:
                    min_start_threshold = 0.0
                elif overlap_percentage >= 0.6:
                    min_start_threshold = 0.1
                else:
                    min_start_threshold = 0.3
                
                # Additional check: exclude words that would start too early in the clip
                if adjusted_start < min_start_threshold:
                    continue
                
                adjusted_word['start'] = max(0.0, adjusted_start)
                adjusted_word['end'] = max(adjusted_word['start'] + 0.01, adjusted_end)
                adjusted_words.append(adjusted_word)
            
            if adjusted_words:
                adjusted_segment = segment.copy()
                adjusted_segment['words'] = adjusted_words
                # Use higher precision for segment timing as well
                adjusted_segment_start = max(0.0, seg_start - segment_start)
                adjusted_segment_end = max(adjusted_segment_start + 0.01, seg_end - segment_start)
                
                # Apply tiered threshold for segment-level timing
                if overlap_percentage >= 0.8:
                    min_segment_threshold = 0.0
                elif overlap_percentage >= 0.6:
                    min_segment_threshold = 0.1
                else:
                    min_segment_threshold = 0.3
                
                # Additional check for segment-level timing
                if adjusted_segment_start < min_segment_threshold:
                    # Adjust the segment start to the threshold time
                    adjusted_segment_start = min_segment_threshold
                
                adjusted_segment['start'] = adjusted_segment_start
                adjusted_segment['end'] = adjusted_segment_end
                relevant_segments.append(adjusted_segment)
        
        self.logger.info(f"Found {len(relevant_segments)} segments with word-level timing")
        return relevant_segments
    
    def _create_ass_header(self, style_config: Dict) -> str:
        """Create ASS file header with style definitions."""
        
        # Extract style properties
        font_name = style_config.get('font_family', 'Arial')
        font_size = style_config.get('font_size', 24)
        primary_color = self._color_to_ass(style_config.get('font_color', 'white'))
        
        # Use pronunciation_color from API style for highlight color, fallback to highlight_color, then yellow
        highlight_color = (style_config.get('highlight_color') or 
                          style_config.get('pronunciation_color', 'yellow'))
        secondary_color = self._color_to_ass(highlight_color)
        
        outline_color = self._color_to_ass(style_config.get('border_color', 'black'))
        back_color = self._color_to_ass(style_config.get('background_color', 'black@0.7'))
        outline_width = style_config.get('border_width', 2)
        alignment = style_config.get('alignment', 2)  # Bottom center
        margin_v = style_config.get('margin_v', 60)
        
        # Get ASS-specific formatting values from API style conversion
        bold = style_config.get('ass_bold', 0)          # -1 = bold, 0 = normal
        italic = style_config.get('ass_italic', 0)      # 1 = italic, 0 = normal
        underline = style_config.get('ass_underline', 0) # 1 = underline, 0 = normal
        strikeout = style_config.get('ass_strikeout', 0) # 1 = strikeout, 0 = normal
        scale_x = style_config.get('ass_scale_x', 100)   # Width scaling percentage
        scale_y = style_config.get('ass_scale_y', 100)   # Height scaling percentage
        spacing = style_config.get('ass_spacing', 0)     # Letter spacing in pixels
        angle = style_config.get('ass_angle', 0)         # Rotation angle
        
        # Store highlight color and text transform for use in kinetic events
        self.highlight_color = highlight_color
        self.text_transform = style_config.get('text_transform')
        
        header = f"""[Script Info]
Title: Kinetic Captions
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},{secondary_color},{outline_color},{back_color},{bold},{italic},{underline},{strikeout},{scale_x},{scale_y},{spacing},{angle},1,{outline_width},1,{alignment},0,0,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""
        
        return header
    
    def _create_kinetic_events(self, segments: List[Dict], offset: float, kinetic_mode: str, timing_offset: float = 1.700) -> str:
        """Create ASS events with kinetic timing."""
        
        events = []
        
        for segment in segments:
            words = segment.get('words', [])
            if not words:
                continue
            
            # Group words into caption lines (3-5 words per line)
            word_chunks = self._chunk_words(words, max_words=4)
            
            for chunk in word_chunks:
                if kinetic_mode == "karaoke":
                    event = self._create_karaoke_event(chunk, timing_offset)
                elif kinetic_mode == "typewriter":
                    event = self._create_typewriter_event(chunk, timing_offset)
                elif kinetic_mode == "highlight":
                    event = self._create_highlight_event(chunk, timing_offset)
                else:
                    event = self._create_karaoke_event(chunk, timing_offset)  # Default
                
                if event:
                    events.append(event)
        
        return "\n".join(events)
    
    def _chunk_words(self, words: List[Dict], max_words: int = 4) -> List[List[Dict]]:
        """Chunk words into caption groups for better readability."""
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            
            # Break on natural pauses or max words
            if (len(current_chunk) >= max_words or 
                word.get('word', '').endswith(('.', '!', '?', ','))):
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add remaining words
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _apply_text_transform(self, text: str) -> str:
        """Apply text transformation based on API style."""
        if hasattr(self, 'text_transform') and self.text_transform:
            if self.text_transform.lower() == 'uppercase':
                return text.upper()
            elif self.text_transform.lower() == 'lowercase':
                return text.lower()
            elif self.text_transform.lower() == 'capitalize':
                return text.capitalize()
        return text
    
    def _create_karaoke_event(self, words: List[Dict], timing_offset: float = 1.700) -> str:
        """Create karaoke-style ASS event where words change color as spoken."""
        
        if not words:
            return ""
        
        start_time = words[0]['start'] + timing_offset
        end_time = words[-1]['end'] + timing_offset
        
        # Create karaoke text with \k timing tags
        karaoke_text = ""
        for word_data in words:
            # Try both 'text' and 'word' keys (whisper-timestamped uses 'text')
            word = word_data.get('text', word_data.get('word', '')).strip()
            if not word:
                continue
            
            # Apply text transformation
            word = self._apply_text_transform(word)
                
            # Duration remains the same (it's relative timing between words)
            duration = word_data['end'] - word_data['start']
            duration_cs = int(duration * 100)  # Convert to centiseconds
            
            # ASS karaoke format: \k<duration>word
            karaoke_text += f"{{\\k{duration_cs}}}{word} "
        
        # Format times for ASS
        start_ass = self._seconds_to_ass_time(start_time)
        end_ass = self._seconds_to_ass_time(end_time)
        
        return f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{karaoke_text.strip()}"
    
    def _create_typewriter_event(self, words: List[Dict], timing_offset: float = 1.700) -> str:
        """Create typewriter-style effect where characters appear progressively."""
        
        if not words:
            return ""
        
        start_time = words[0]['start'] + timing_offset
        end_time = words[-1]['end'] + timing_offset
        
        # For typewriter effect, we'll create multiple events
        # This is a simplified version - each word appears at its timing
        events = []
        cumulative_text = ""
        
        for i, word_data in enumerate(words):
            # Try both 'text' and 'word' keys (whisper-timestamped uses 'text')
            word = word_data.get('text', word_data.get('word', '')).strip()
            if not word:
                continue
            
            # Apply text transformation
            word = self._apply_text_transform(word)
            cumulative_text += word + " "
            word_start = word_data['start'] + timing_offset  # Apply same timing offset
            word_end = word_data.get('end', word_start + 0.5) + timing_offset
            
            start_ass = self._seconds_to_ass_time(word_start)
            end_ass = self._seconds_to_ass_time(end_time)  # Show until end of chunk
            
            # Create event showing cumulative text
            event = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{cumulative_text.strip()}"
            events.append(event)
        
        return "\n".join(events)
    
    def _create_highlight_event(self, words: List[Dict], timing_offset: float = 1.700) -> str:
        """Create highlight effect where current word is emphasized."""
        
        if not words:
            return ""
        
        events = []
        # Use 'text' key (whisper-timestamped) with fallback to 'word'
        full_text = " ".join([w.get('text', w.get('word', '')).strip() for w in words if w.get('text', w.get('word', '')).strip()])
        
        # Get the pronunciation color (for highlighting) in ASS format
        highlight_color_ass = self._color_to_ass(getattr(self, 'highlight_color', 'yellow'))
        default_color_ass = self._color_to_ass('white')  # Default text color
        
        for i, word_data in enumerate(words):
            # Try both 'text' and 'word' keys (whisper-timestamped uses 'text')
            word = word_data.get('text', word_data.get('word', '')).strip()
            if not word:
                continue
            
            word_start = word_data['start'] + timing_offset
            word_end = word_data['end'] + timing_offset
            
            # Create text with current word highlighted
            highlighted_text = ""
            for j, w in enumerate(words):
                # Try both 'text' and 'word' keys (whisper-timestamped uses 'text')
                w_text = w.get('text', w.get('word', '')).strip()
                if not w_text:
                    continue
                
                # Apply text transformation
                w_text = self._apply_text_transform(w_text)
                    
                if j == i:
                    # Highlight current word with pronunciation_color from API
                    highlighted_text += f"{{\\c{highlight_color_ass}}}{w_text}{{\\c{default_color_ass}}} "
                else:
                    highlighted_text += f"{w_text} "
            
            start_ass = self._seconds_to_ass_time(word_start)
            end_ass = self._seconds_to_ass_time(word_end)
            
            event = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{highlighted_text.strip()}"
            events.append(event)
        
        return "\n".join(events)
    
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.CC)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        
        # ASS format: H:MM:SS.CC where CC is centiseconds (hundredths)
        return f"{hours}:{minutes:02d}:{secs:06.2f}"
    
    def _color_to_ass(self, color: str) -> str:
        """Convert color specification to ASS format."""
        # Handle transparency
        if '@' in color:
            base_color, alpha = color.split('@')
            alpha_val = int(float(alpha) * 255)
        else:
            base_color = color
            alpha_val = 0  # Fully opaque in ASS (0 = opaque, 255 = transparent)
        
        # Convert named colors to hex
        color_map = {
            'white': 'FFFFFF',
            'black': '000000',
            'red': '0000FF',  # ASS uses BGR format
            'green': '00FF00',
            'blue': 'FF0000',
            'yellow': '00FFFF',
            'cyan': 'FFFF00',
            'magenta': 'FF00FF'
        }
        
        if base_color.lower() in color_map:
            hex_color = color_map[base_color.lower()]
        elif base_color.startswith('#'):
            # Convert RGB to BGR
            hex_color = base_color[1:]
            if len(hex_color) == 6:
                r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
                hex_color = f"{b}{g}{r}"  # BGR format
        else:
            hex_color = 'FFFFFF'  # Default to white
        
        # ASS color format: &H<alpha><color>&
        return f"&H{alpha_val:02X}{hex_color}&"
    
    def apply_kinetic_subtitles(self,
                               input_video_path: str,
                               output_video_path: str,
                               transcription_data: Dict,
                               segment_start_time: float,
                               segment_end_time: float,
                               style_config: Dict,
                               kinetic_mode: str = "karaoke",
                               timing_offset: float = 1.700) -> bool:
        """
        Apply kinetic subtitles to video.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path for output video  
            transcription_data: Transcription with word-level timing
            segment_start_time: Start time of segment
            segment_end_time: End time of segment
            style_config: Subtitle styling
            kinetic_mode: Type of kinetic effect
            timing_offset: Seconds to add to all subtitle timings (default 1.700s to fix early display)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input video file first
            if not self._validate_video_file(input_video_path):
                self.logger.error(f"Input video file is invalid or corrupted: {input_video_path}")
                return False
                
            # Generate ASS content
            ass_content = self.create_kinetic_ass(
                transcription_data, segment_start_time, segment_end_time, 
                style_config, kinetic_mode, timing_offset
            )
            
            if not ass_content:
                self.logger.error("Failed to generate kinetic ASS content")
                return False
            
            # Create temporary ASS file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as f:
                f.write(ass_content)
                ass_path = f.name
            
            try:
                # Get original video frame rate to preserve exact timing
                probe_cmd = [
                    'ffprobe', '-v', 'quiet', '-select_streams', 'v:0', 
                    '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', 
                    input_video_path
                ]
                
                frame_rate = "30"  # Default fallback
                try:
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                    if probe_result.returncode == 0 and probe_result.stdout.strip():
                        frame_rate_str = probe_result.stdout.strip()
                        # Handle fractional frame rates like "30000/1001" for 29.97 fps
                        if '/' in frame_rate_str:
                            num, den = frame_rate_str.split('/')
                            frame_rate = f"{int(num)}/{int(den)}"
                        else:
                            frame_rate = frame_rate_str
                        self.logger.info(f"Detected original frame rate: {frame_rate}")
                except Exception as e:
                    self.logger.warning(f"Could not detect frame rate, using default: {e}")
                
                # Apply ASS subtitles with FFmpeg - ENHANCED: Frame rate preservation + A/V sync
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', input_video_path,
                    '-vf', f"ass='{ass_path}'",
                    '-c:v', 'libx264',          # Encode video to ensure proper sync
                    '-c:a', 'aac',              # Encode audio to ensure proper sync
                    '-r', str(frame_rate),      # CRITICAL: Preserve exact original frame rate
                    '-map', '0:v:0?',           # Optional map - first video stream
                    '-map', '0:a:0?',           # Optional map - first audio stream
                    '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                    '-vsync', '1',              # Ensure proper video frame sync
                    '-async', '1',              # Fix audio sync to video
                    '-copytb', '1',             # Copy input timebase to output
                    '-fflags', '+genpts',       # Generate presentation timestamps
                    '-crf', '23',               # Good quality setting
                    '-preset', 'medium',        # Balance speed vs quality
                    '-shortest',                # End with shortest stream
                    '-y', output_video_path
                ]
                
                self.logger.info(f"Applying kinetic subtitles with mode: {kinetic_mode}")
                self.logger.info(f"🔧 FFmpeg command: {' '.join(ffmpeg_cmd)}")
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    # Validate the output file
                    if self._validate_video_file(output_video_path):
                        self.logger.info(f"Successfully applied kinetic subtitles: {output_video_path}")
                        return True
                    else:
                        self.logger.error(f"Output video file is corrupted: {output_video_path}")
                        return False
                else:
                    self.logger.error(f"FFmpeg failed: {result.stderr}")
                    # Check if output was created anyway (sometimes FFmpeg reports errors but succeeds)
                    if os.path.exists(output_video_path) and self._validate_video_file(output_video_path):
                        self.logger.warning("FFmpeg reported error but output seems valid, continuing...")
                        return True
                    return False
                    
            finally:
                # Cleanup temporary ASS file
                if os.path.exists(ass_path):
                    os.unlink(ass_path)
                    
        except Exception as e:
            self.logger.error(f"Error applying kinetic subtitles: {e}")
            return False
