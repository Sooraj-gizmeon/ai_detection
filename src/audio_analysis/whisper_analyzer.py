# src/audio_analysis/whisper_analyzer.py
"""OpenAI Whisper integration for audio analysis with GPU acceleration and precise word-level timestamps"""

import os
import torch
import whisper_timestamped as whisper
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
import hashlib
from ..utils.cache_manager import CacheManager
from ..utils.gpu_utils import get_device_info
from ..utils.database_api_client import DatabaseAPIClient, extract_transcription_text


class WhisperAnalyzer:
    """
    OpenAI Whisper integration for speech recognition and audio analysis
    with GPU acceleration and intelligent caching.
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "cuda",
                 cache_dir: str = "audio_cache",
                 compute_type: str = "float16"):
        """
        Initialize Whisper analyzer with GPU acceleration.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu')
            cache_dir: Directory for caching transcriptions
            compute_type: Compute precision ('float16' or 'float32')
        """
        self.model_size = model_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.compute_type = compute_type
        
        # Initialize cache manager
        self.cache_manager = CacheManager(cache_dir)
        
        # Initialize database API client
        self.db_api_client = DatabaseAPIClient()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self.model = None
        self._load_model()
        
        # GPU information
        self.gpu_info = get_device_info()
        self.logger.info(f"Whisper initialized on {self.device} with {self.model_size} model")
    
    def get_transcription_from_db(self, video_id: str = None, uploaded_video_id: str = None) -> Optional[Dict]:
        """Retrieve transcription from database via API by video_id or uploaded_video_id"""
        if not video_id and not uploaded_video_id:
            return None
        
        # Determine which ID to use and convert if necessary
        effective_id = None
        id_type = None
        
        if video_id:
            # Try to convert video_id to integer
            try:
                effective_id = int(video_id)
                id_type = "video_id"
            except (ValueError, TypeError):
                self.logger.error(f"Invalid video_id format: {video_id}. Must be convertible to integer.")
                return None
        else:
            # Use uploaded_video_id as string
            effective_id = uploaded_video_id
            id_type = "uploaded_video_id"
            
        self.logger.info(f"Checking for existing transcription for {id_type}: {effective_id}")
        
        try:
            # Call API with appropriate parameter
            if id_type == "video_id":
                result = self.db_api_client.get_transcription(video_id=effective_id)
            else:
                result = self.db_api_client.get_transcription(uploaded_video_id=effective_id)
            
            if result.get('success', True) and result.get('data'):
                # Extract transcription from API response
                data = result['data']
                if isinstance(data, list) and len(data) > 0:
                    transcription_text = data[0].get('transcription', '')
                    if transcription_text:
                        # Found in DB: return parsed JSON if possible, otherwise build structured result
                        self.logger.info(f"Found existing transcription for {id_type}: {effective_id}")
                        try:
                            return json.loads(transcription_text)
                        except json.JSONDecodeError:
                            # If not JSON, create a simple structure with better segments
                            # Split text into sentences for better analysis
                            sentences = self._split_into_sentences(transcription_text)
                            segments = []
                            
                            # Create segments from sentences with estimated timestamps
                            cumulative_time = 0
                            for i, sentence in enumerate(sentences):
                                word_count = len(sentence.split())
                                # Estimate duration: ~2 words per second
                                estimated_duration = max(1, word_count * 0.5)
                                
                                segments.append({
                                    "text": sentence,
                                    "start": cumulative_time,
                                    "end": cumulative_time + estimated_duration
                                })
                                cumulative_time += estimated_duration
                            
                            return {
                                "text": transcription_text,
                                "segments": segments
                            }
                            
            self.logger.info(f"No existing transcription found for {id_type}: {effective_id}")
            return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving transcription via API: {e}")
            return None

    def save_transcription_to_db(self, video_id: str = None, uploaded_video_id: str = None, transcription: Dict = None) -> bool:
        """Save complete transcription with timestamps to database via API"""
        if not video_id and not uploaded_video_id:
            return False
        
        if not transcription:
            return False
        
        # Determine which ID to use and convert if necessary
        effective_id = None
        id_type = None
        
        if video_id:
            # Try to convert video_id to integer
            try:
                effective_id = int(video_id)
                id_type = "video_id"
            except (ValueError, TypeError):
                self.logger.error(f"Invalid video_id format: {video_id}. Must be convertible to integer.")
                return False
        else:
            # Use uploaded_video_id as string
            effective_id = uploaded_video_id
            id_type = "uploaded_video_id"
            
        self.logger.info(f"Saving complete transcription with timestamps for {id_type}: {effective_id}")
        
        try:
            # Save the COMPLETE transcription as JSON (preserving timestamps)
            transcription_json = json.dumps(transcription)
            
            # Call API to save transcription with appropriate parameter
            if id_type == "video_id":
                result = self.db_api_client.add_transcription(video_id=effective_id, transcription=transcription_json)
            else:
                result = self.db_api_client.add_transcription(uploaded_video_id=effective_id, transcription=transcription_json)
            
            if result.get('success', True):  # Assume success if no explicit success field
                segment_count = len(transcription.get('segments', []))
                self.logger.info(f"Successfully saved transcription with {segment_count} segments for {id_type}: {effective_id}")
                return True
            else:
                self.logger.error(f"Failed to save transcription via API: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving transcription via API: {e}")
            return False
    
    def _load_model(self):
        """Load Whisper model with error handling using whisper-timestamped."""
        try:
            self.model = whisper.load_model(
                self.model_size,
                device=self.device,
                backend="openai-whisper",
                download_root=str(Path("models") / "whisper")
            )
            self.logger.info(f"Whisper {self.model_size} model loaded successfully with whisper-timestamped backend")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def extract_audio(self, video_path: str, audio_path: str = None) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to input video file
            audio_path: Path for output audio file (optional)
            
        Returns:
            Path to extracted audio file
        """
        import ffmpeg
        
        if audio_path is None:
            video_stem = Path(video_path).stem
            audio_path = str(self.cache_dir / f"{video_stem}_audio.wav")
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(audio_path, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            self.logger.info(f"Audio extracted to {audio_path}")
            return audio_path
        except Exception as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise
    
    def transcribe(self, audio_path: str, language: str = None, video_id: str = None, uploaded_video_id: str = None) -> Dict:
        """
        Transcribe audio file with Whisper and database caching.
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            video_id: Video ID for database caching (optional)
            uploaded_video_id: Uploaded video ID for database caching (optional)
            
        Returns:
            Transcription result dictionary
        """
        # Check database cache first if video_id or uploaded_video_id is provided
        if video_id or uploaded_video_id:
            cached_result = self.get_transcription_from_db(video_id=video_id, uploaded_video_id=uploaded_video_id)
            if cached_result:
                id_label = f"video_id: {video_id}" if video_id else f"uploaded_video_id: {uploaded_video_id}"
                self.logger.info(f"Using cached transcription from database for {id_label}")
                return cached_result
        
        # If not in database, check file cache
        cache_key = self._generate_cache_key(audio_path, language)
        cached_result = self.cache_manager.get_cached_result(cache_key)
        if cached_result:
            self.logger.info("Using cached transcription from file cache")
            # Save to database if video_id or uploaded_video_id is provided
            if video_id or uploaded_video_id:
                self.save_transcription_to_db(video_id=video_id, uploaded_video_id=uploaded_video_id, transcription=cached_result)
            return cached_result
        
        # No cached transcription found, generate a new one
        id_label = f"video_id: {video_id}" if video_id else (f"uploaded_video_id: {uploaded_video_id}" if uploaded_video_id else "audio")
        self.logger.info(f"Generating new transcription for {id_label}")
        
        try:
            # Use whisper-timestamped for precise word-level timestamps
            self.logger.info("Using whisper-timestamped for precise word-level timing")
            result = whisper.transcribe(
                self.model,
                audio_path,
                language='en',
                verbose=False,
                # Options for better accuracy and word-level timing
                compute_word_confidence=True,
                include_punctuation_in_confidence=False,
                refine_whisper_precision=0.5,
                min_word_duration=0.02,
                # Use VAD for better transcription quality 
                vad=True,
                # Remove disfluencies detection for cleaner output
                detect_disfluencies=False,
                # Trust whisper timestamps but refine them
                trust_whisper_timestamps=True,
                # Use efficient default options
                temperature=0.0,
                best_of=1,
                beam_size=1,
                condition_on_previous_text=False
            )
            
            self.logger.info(f"Whisper-timestamped result type: {type(result)}")
            
            # Check if result is the expected dictionary format
            if not isinstance(result, dict):
                self.logger.error(f"Unexpected result type: {type(result)}, value: {result}")
                # Create a minimal result structure
                result = {
                    'text': str(result) if result else "",
                    'segments': [],
                    'language': language or 'en'
                }
            
            # Ensure required keys exist
            if 'segments' not in result:
                result['segments'] = []
            if 'text' not in result:
                result['text'] = ""
            if 'language' not in result:
                result['language'] = language or 'en'
            
            self.logger.info(f"Processing whisper-timestamped result with {len(result.get('segments', []))} segments")
            
            # Validate word-level timestamps from whisper-timestamped
            word_timing_available = False
            total_words = 0
            total_words_with_timing = 0
            
            for segment in result.get('segments', []):
                if 'words' in segment and segment['words']:
                    segment_words = len(segment['words'])
                    total_words += segment_words
                    
                    # Check if words have proper timing information
                    words_with_timing = sum(1 for word in segment['words'] 
                                          if isinstance(word, dict) and 'start' in word and 'end' in word)
                    total_words_with_timing += words_with_timing
                    
                    if words_with_timing > 0:
                        word_timing_available = True
                        self.logger.debug(f"Segment has {words_with_timing}/{segment_words} words with precise timestamps")
                else:
                    self.logger.debug("Segment missing word-level timestamps")
            
            if word_timing_available and total_words_with_timing > 0:
                coverage = (total_words_with_timing / total_words * 100) if total_words > 0 else 0
                self.logger.info(f"✅ Word-level timing available: {total_words_with_timing}/{total_words} words ({coverage:.1f}%) with precise timestamps")
                result['has_word_timing'] = True
                result['word_timing_coverage'] = coverage
            else:
                self.logger.warning("❌ No word-level timing data available from whisper-timestamped")
                result['has_word_timing'] = False
                result['word_timing_coverage'] = 0
                
                # This shouldn't happen with whisper-timestamped, but add fallback just in case
                self._add_synthetic_word_timestamps(result)
            
            # Enhance result with additional analysis
            enhanced_result = self._enhance_transcription(result)
            
            # Cache the result in file cache
            self.cache_manager.cache_result(cache_key, enhanced_result)
            
            # Save to database if video_id or uploaded_video_id is provided
            if video_id or uploaded_video_id:
                self.save_transcription_to_db(video_id=video_id, uploaded_video_id=uploaded_video_id, transcription=enhanced_result)
            
            self.logger.info(f"Transcription completed: {len(enhanced_result['segments'])} segments")
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _enhance_transcription(self, result: Dict) -> Dict:
        """
        Enhance transcription with additional analysis.
        
        Args:
            result: Raw Whisper transcription result
            
        Returns:
            Enhanced transcription with additional metadata
        """
        self.logger.info(f"_enhance_transcription input type: {type(result)}")
        self.logger.info(f"_enhance_transcription input keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        
        enhanced = result.copy()
        
        # Add segment analysis
        enhanced['segments'] = [
            self._analyze_segment(segment) 
            for segment in result['segments']
        ]
        
        # Add overall statistics
        enhanced['stats'] = self._calculate_stats(result)
        
        # Add speaker change detection
        enhanced['speaker_changes'] = self._detect_speaker_changes(result)
        
        # Add silence detection
        enhanced['silence_periods'] = self._detect_silence_periods(result)
        
        return enhanced
    
    def _analyze_segment(self, segment: Dict) -> Dict:
        """
        Analyze individual segment for additional metadata.
        
        Args:
            segment: Whisper segment dictionary
            
        Returns:
            Enhanced segment with additional analysis
        """
        # Validate segment format
        if not isinstance(segment, dict):
            self.logger.warning(f"Invalid segment type: {type(segment)}, value: {segment}")
            # Create a minimal segment structure
            return {
                'start': 0,
                'end': 0,
                'text': str(segment) if segment else "",
                'duration': 0,
                'speaking_rate': 0,
                'avg_confidence': 0,
                'emphasis_keywords': []
            }
        
        enhanced_segment = segment.copy()
        
        # Safely get required fields with defaults
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '')
        
        # Calculate speaking rate (words per minute)
        duration = end - start
        word_count = len(text.split())
        speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
        
        # Analyze confidence and stability
        tokens = segment.get('tokens', [])
        if tokens and isinstance(tokens, list):
            confidences = [token.get('probability', 0) for token in tokens if isinstance(token, dict)]
            avg_confidence = np.mean(confidences) if confidences else 0
        else:
            avg_confidence = 0
        
        # Detect emphasis keywords
        emphasis_keywords = self._detect_emphasis_keywords(text)
        
        # Add metadata
        enhanced_segment.update({
            'duration': duration,
            'word_count': word_count,
            'speaking_rate': speaking_rate,
            'avg_confidence': avg_confidence,
            'emphasis_keywords': emphasis_keywords,
            'energy_level': self._calculate_energy_level(segment),
            'suitable_for_zoom': self._is_suitable_for_zoom(segment)
        })
        
        return enhanced_segment
    
    def _detect_emphasis_keywords(self, text: str) -> List[str]:
        """Detect keywords that might indicate emphasis or important content."""
        emphasis_words = [
            "important", "key", "crucial", "significant", "essential",
            "remember", "notice", "look", "see", "watch", "focus",
            "amazing", "incredible", "fantastic", "wow", "great"
        ]
        
        words = text.lower().split()
        return [word for word in words if word in emphasis_words]
    
    def _calculate_energy_level(self, segment: Dict) -> str:
        """Calculate energy level of segment based on various factors."""
        # This is a simplified version - in practice, you'd analyze audio features
        text = segment['text'].lower()
        
        # Handle segments with or without timestamps
        if 'start' in segment and 'end' in segment:
            duration = segment['end'] - segment['start']
        else:
            # Estimate duration based on word count
            word_count = len(text.split())
            duration = word_count * 0.5  # Rough estimate: 2 words per second
        
        word_count = len(text.split())
        
        # Simple heuristic based on speaking rate and punctuation
        speaking_rate = (word_count / duration) * 60 if duration > 0 else 0
        exclamation_count = text.count('!') + text.count('?')
        
        if speaking_rate > 180 or exclamation_count > 0:
            return "high"
        elif speaking_rate > 120:
            return "medium"
        else:
            return "low"
    
    def _is_suitable_for_zoom(self, segment: Dict) -> bool:
        """Determine if segment is suitable for close-up zoom."""
        # Check for personal pronouns, emphasis, or direct address
        text = segment['text'].lower()
        personal_indicators = ['i', 'you', 'we', 'let me', 'look at', 'see this']
        
        return any(indicator in text for indicator in personal_indicators)
    
    def _calculate_stats(self, result: Dict) -> Dict:
        """Calculate overall transcription statistics."""
        segments = result['segments']
        total_duration = max(seg['end'] for seg in segments) if segments else 0
        total_words = sum(len(seg['text'].split()) for seg in segments)
        
        return {
            'total_duration': total_duration,
            'total_words': total_words,
            'total_segments': len(segments),
            'avg_words_per_segment': total_words / len(segments) if segments else 0,
            'overall_speaking_rate': (total_words / total_duration) * 60 if total_duration > 0 else 0
        }
    
    def _detect_speaker_changes(self, result: Dict) -> List[float]:
        """Detect potential speaker changes based on audio characteristics."""
        # Simplified speaker change detection
        # In practice, you'd use more sophisticated audio analysis
        changes = []
        segments = result['segments']
        
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
        
            # Check for significant pause or change in speaking characteristics
            # Handle segments with or without timestamps
            if 'start' in curr_segment and 'end' in prev_segment:
                pause_duration = curr_segment['start'] - prev_segment['end']
                
                if pause_duration > 2.0:  # 2 second pause might indicate speaker change
                    changes.append(curr_segment['start'])
            else:
                # For segments without timestamps, skip speaker change detection
                continue
        
        return changes
    
    def _detect_silence_periods(self, result: Dict) -> List[Tuple[float, float]]:
        """Detect silence periods between segments."""
        silence_periods = []
        segments = result['segments']
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']
            
            if curr_start > prev_end + 0.5:  # 0.5 second minimum silence
                silence_periods.append((prev_end, curr_start))
        
        return silence_periods
    
    def _generate_cache_key(self, audio_path: str, language: str = None) -> str:
        """Generate unique cache key for transcription."""
        # Include file modification time and parameters in cache key
        file_stat = os.stat(audio_path)
        key_data = f"{audio_path}:{file_stat.st_mtime}:{self.model_size}:{language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def analyze_for_zoom_decisions(self, transcription: Dict) -> Dict:
        """
        Analyze transcription to generate zoom recommendations.
        
        Args:
            transcription: Enhanced transcription result
            
        Returns:
            Zoom decision recommendations
        """
        # Validate transcription format
        if not isinstance(transcription, dict):
            self.logger.error(f"Invalid transcription format: {type(transcription)}")
            return {
                'zoom_decisions': [],
                'recommended_cuts': [],
                'engagement_scores': {}
            }
        
        if 'segments' not in transcription or not isinstance(transcription['segments'], list):
            self.logger.warning("No segments found in transcription")
            return {
                'zoom_decisions': [],
                'recommended_cuts': [],
                'engagement_scores': {}
            }
        
        zoom_decisions = []
        speech_segments = []
        silence_segments = []
        music_segments = []
        
        # Analyze segments for speech, silence, and music
        for segment in transcription['segments']:
            # Handle segments that may not have timestamps (from database cache)
            if 'start' in segment and 'end' in segment:
                segment_duration = segment['end'] - segment['start']
                words_per_second = len(segment['text'].split()) / segment_duration if segment_duration > 0 else 0
            else:
                # For segments without timestamps, estimate based on word count
                word_count = len(segment['text'].split())
                segment_duration = word_count * 0.5  # Rough estimate: 2 words per second
                words_per_second = word_count / segment_duration if segment_duration > 0 else 0
            
            # Detect if segment is likely music-only or silence
            has_music = self._detect_music_pattern(segment['text'])
            has_silence = words_per_second < 0.5  # Very low word rate
            
            if has_music:
                music_segments.append(segment)
            elif has_silence:
                silence_segments.append(segment)
            else:
                speech_segments.append(segment)
                
                # Determine zoom strategy based on content
                zoom_strategy = self._determine_zoom_strategy(segment)
                
                # Handle segments with or without timestamps
                if 'start' in segment and 'end' in segment:
                    start_time = segment['start']
                    end_time = segment['end']
                else:
                    # For segments without timestamps, use estimated values
                    start_time = 0  # Will be overridden by actual video processing
                    end_time = segment_duration
                
                zoom_decisions.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'zoom_level': zoom_strategy['zoom_level'],
                    'strategy': zoom_strategy['strategy'],
                    'confidence': zoom_strategy['confidence'],
                    'reasoning': zoom_strategy['reasoning'],
                    'content_type': 'speech'
                })
        
        # Identify segments that look like pre-roll
        likely_preroll = []
        total_duration = transcription.get('stats', {}).get('total_duration', 0)
        
        if total_duration > 0:
            # Check first 10% of video
            first_tenth_end = total_duration * 0.1
            
            # Add music/silence segments in first 10% to likely_preroll
            for segment in music_segments + silence_segments:
                # Handle segments with or without timestamps
                if 'start' in segment and 'end' in segment:
                    if segment['start'] < first_tenth_end:
                        likely_preroll.append({
                            'start': segment['start'],
                            'end': segment['end'],
                            'type': 'music' if segment in music_segments else 'silence',
                            'should_skip': True
                        })
                else:
                    # For segments without timestamps, assume they might be pre-roll if they're music/silence
                    likely_preroll.append({
                        'start': 0,
                        'end': 10,  # Rough estimate
                        'type': 'music' if segment in music_segments else 'silence',
                        'should_skip': True
                    })
        
        # Generate recommended cuts based on speech segments
        recommended_cuts = self._recommend_cuts(transcription)
        
        return {
            'zoom_decisions': zoom_decisions,
            'recommended_cuts': recommended_cuts,
            'likely_preroll': likely_preroll,
            'speech_segments': speech_segments,
            'silence_segments': silence_segments,
            'music_segments': music_segments,
            'engagement_scores': self._calculate_engagement_scores(transcription)
        }
        
    def _detect_music_pattern(self, text: str) -> bool:
        """Detect if text appears to be music transcription."""
        # Common patterns in music transcription
        music_indicators = [
            '[Music]', '♪', '♫', '[Instrumental]', '[Intro music]',
            'Music playing', 'instrumental', 'theme music', 'background music'
        ]
        
        # Check if text contains music indicators
        for indicator in music_indicators:
            if indicator.lower() in text.lower():
                return True
                
        # Check for repeated phrases that might indicate lyrics
        words = text.lower().split()
        if len(words) > 8:  # Only check if enough words
            # Simple heuristic for repetition - could be improved
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # If any word appears more than 3 times in a short segment, might be lyrics
            repeats = [word for word, count in word_counts.items() if count > 3 and len(word) > 2]
            if repeats:
                return True
                
        return False
    
    def _determine_zoom_strategy(self, segment: Dict) -> Dict:
        """Determine optimal zoom strategy for a segment."""
        text = segment['text'].lower()
        energy = segment.get('energy_level', 'medium')
        emphasis_keywords = segment.get('emphasis_keywords', [])
        
        # Default strategy
        strategy = {
            'zoom_level': 1.5,
            'strategy': 'medium_shot',
            'confidence': 0.5,
            'reasoning': 'Default medium shot'
        }
        
        # Close-up for emphasis or personal content
        if emphasis_keywords or segment.get('suitable_for_zoom', False):
            strategy = {
                'zoom_level': 2.0,
                'strategy': 'close_up',
                'confidence': 0.8,
                'reasoning': 'Contains emphasis keywords or personal content'
            }
        
        # Wide shot for group references
        elif any(word in text for word in ['everyone', 'all', 'group', 'together']):
            strategy = {
                'zoom_level': 1.0,
                'strategy': 'wide_shot',
                'confidence': 0.7,
                'reasoning': 'References group or multiple people'
            }
        
        # Dynamic for high energy content
        elif energy == 'high':
            strategy = {
                'zoom_level': 1.8,
                'strategy': 'dynamic',
                'confidence': 0.6,
                'reasoning': 'High energy content'
            }
        
        return strategy
    
    def _recommend_cuts(self, transcription: Dict) -> List[Dict]:
        """Recommend optimal cut points for short videos."""
        if not isinstance(transcription, dict) or 'segments' not in transcription:
            return []
            
        cuts = []
        segments = transcription['segments']
        silence_periods = transcription.get('silence_periods', [])
        
        # Find natural break points
        for silence_start, silence_end in silence_periods:
            if silence_end - silence_start > 1.0:  # 1 second minimum silence
                cuts.append({
                    'timestamp': silence_start,
                    'type': 'natural_break',
                    'confidence': 0.8
                })
        
        # Find topic transitions (simplified)
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
            # Simple heuristic: if very different content, might be topic change
            if self._segments_different_topic(prev_segment, curr_segment):
                # Handle segments with or without timestamps
                timestamp = curr_segment.get('start', i * 10)  # Default to estimated time
                cuts.append({
                    'timestamp': timestamp,
                    'type': 'topic_change',
                    'confidence': 0.6
                })
        
        return cuts
    
    def _segments_different_topic(self, seg1: Dict, seg2: Dict) -> bool:
        """Simple heuristic to detect topic changes between segments."""
        # This is a simplified version - in practice, you'd use more sophisticated NLP
        text1_words = set(seg1['text'].lower().split())
        text2_words = set(seg2['text'].lower().split())
        
        # Calculate word overlap
        overlap = len(text1_words.intersection(text2_words))
        union = len(text1_words.union(text2_words))
        
        similarity = overlap / union if union > 0 else 0
        return similarity < 0.3  # Less than 30% word overlap
    
    def _calculate_engagement_scores(self, transcription: Dict) -> List[Dict]:
        """Calculate engagement scores for each segment."""
        if not isinstance(transcription, dict) or 'segments' not in transcription:
            return []
            
        scores = []
        
        for segment in transcription['segments']:
            score = self._calculate_segment_engagement(segment)
            scores.append({
                'start_time': segment.get('start', 0),
                'end_time': segment.get('end', 10),  # Default 10 seconds if no end time
                'engagement_score': score,
                'viral_potential': self._assess_viral_potential(segment, score)
            })
        
        return scores
    
    def _calculate_segment_engagement(self, segment: Dict) -> float:
        """Calculate engagement score for a single segment."""
        score = 0.5  # Base score
        
        # Factors that increase engagement
        if segment.get('emphasis_keywords'):
            score += 0.2
        
        if segment.get('energy_level') == 'high':
            score += 0.2
        elif segment.get('energy_level') == 'medium':
            score += 0.1
        
        if segment.get('suitable_for_zoom'):
            score += 0.1
        
        # Speaking rate factor
        speaking_rate = segment.get('speaking_rate', 0)
        if 120 <= speaking_rate <= 200:  # Optimal speaking rate
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _assess_viral_potential(self, segment: Dict, engagement_score: float) -> str:
        """Assess viral potential of a segment."""
        if engagement_score > 0.8:
            return "high"
        elif engagement_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            import shutil
            temp_files = list(self.cache_dir.glob("*_audio.*"))
            for temp_file in temp_files:
                temp_file.unlink()
            self.logger.info(f"Cleaned up {len(temp_files)} temporary audio files")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp files: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_size': self.model_size,
            'device': self.device,
            'compute_type': self.compute_type,
            'gpu_info': self.gpu_info,
            'cache_dir': str(self.cache_dir)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better segment creation."""
        import re
        
        # Simple sentence splitting using common punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If no sentences found, split by length
        if not sentences:
            # Split long text into chunks of ~50 words
            words = text.split()
            chunk_size = 50
            sentences = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        return sentences
    
    def _add_synthetic_word_timestamps(self, result: Dict):
        """
        Add synthetic word timestamps when real word timestamps are not available.
        This creates estimated word timing based on segment timing and word count.
        """
        if not isinstance(result, dict) or 'segments' not in result:
            return
            
        self.logger.info("Creating synthetic word timestamps for segments without word-level timing")
        
        for segment in result.get('segments', []):
            if 'words' not in segment or not segment['words']:
                # Create synthetic word timestamps for this segment
                text = segment.get('text', '').strip()
                if not text:
                    continue
                    
                words = text.split()
                if not words:
                    continue
                
                start_time = segment.get('start', 0)
                end_time = segment.get('end', start_time + len(words) * 0.5)  # Default: 0.5 seconds per word
                duration = end_time - start_time
                
                # Create synthetic word entries
                synthetic_words = []
                for i, word in enumerate(words):
                    # Distribute words evenly across the segment duration
                    word_start = start_time + (i * duration / len(words))
                    word_end = start_time + ((i + 1) * duration / len(words))
                    
                    synthetic_words.append({
                        'word': word,
                        'start': word_start,
                        'end': word_end,
                        'probability': 0.8,  # Default confidence for synthetic timestamps
                        'synthetic': True  # Mark as synthetic for downstream processing
                    })
                
                segment['words'] = synthetic_words
                self.logger.debug(f"Added {len(synthetic_words)} synthetic word timestamps to segment")
        
        # Update the result to indicate we now have word timing (synthetic)
        result['has_word_timing'] = True
        result['synthetic_word_timing'] = True
        self.logger.info("✅ Synthetic word-level timing created - kinetic captions will work with estimated timing")
