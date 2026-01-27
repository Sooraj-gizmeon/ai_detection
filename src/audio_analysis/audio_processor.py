# src/audio_analysis/audio_processor.py
"""Audio processing utilities for video analysis"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import tempfile
import os


class AudioProcessor:
    """
    Handles audio extraction and preprocessing for video analysis.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 temp_dir: str = "temp"):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            temp_dir: Directory for temporary audio files
        """
        self.sample_rate = sample_rate
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file
        """
        import ffmpeg
        
        video_path = Path(video_path)
        audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        try:
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(str(video_path))
                .output(str(audio_path), 
                       acodec='pcm_s16le', 
                       ar=self.sample_rate,
                       ac=1)  # Mono audio
                .overwrite_output()
                .run(quiet=True)
            )
            
            self.logger.info(f"Audio extracted to: {audio_path}")
            return str(audio_path)
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio from {video_path}: {e}")
            raise
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio_data, sr
        except Exception as e:
            self.logger.error(f"Failed to load audio from {audio_path}: {e}")
            raise
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data for analysis.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Remove silence at beginning and end
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        return audio_data
    
    def detect_speech_segments(self, audio_data: np.ndarray, 
                             sample_rate: int) -> List[Dict]:
        """
        Detect speech segments in audio.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            List of speech segments with timestamps
        """
        # Use voice activity detection
        from scipy.signal import butter, filtfilt
        
        # Apply band-pass filter for speech frequencies (300-3400 Hz)
        nyquist = sample_rate // 2
        low_freq = 300 / nyquist
        high_freq = 3400 / nyquist
        
        b, a = butter(4, [low_freq, high_freq], btype='band')
        filtered_audio = filtfilt(b, a, audio_data)
        
        # Calculate energy
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        energy = []
        for i in range(0, len(filtered_audio) - frame_length, hop_length):
            frame = filtered_audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Threshold for speech detection
        threshold = np.mean(energy) * 0.3
        
        # Find speech segments
        speech_frames = energy > threshold
        segments = []
        
        in_speech = False
        start_frame = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech segment
                start_frame = i
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                end_frame = i
                start_time = start_frame * hop_length / sample_rate
                end_time = end_frame * hop_length / sample_rate
                
                if end_time - start_time > 0.5:  # Minimum 0.5 seconds
                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time
                    })
                
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            end_time = len(speech_frames) * hop_length / sample_rate
            start_time = start_frame * hop_length / sample_rate
            if end_time - start_time > 0.5:
                segments.append({
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time
                })
        
        return segments
    
    def extract_features(self, audio_data: np.ndarray, 
                        sample_rate: int) -> Dict:
        """
        Extract audio features for analysis.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Dictionary of audio features
        """
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_data, sr=sample_rate)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = tempo
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    def analyze_audio_quality(self, audio_data: np.ndarray, 
                            sample_rate: int) -> Dict:
        """
        Analyze audio quality metrics.
        
        Args:
            audio_data: Audio data
            sample_rate: Sample rate
            
        Returns:
            Audio quality metrics
        """
        quality_metrics = {}
        
        # Signal-to-noise ratio estimation
        # Use spectral subtraction approach
        stft = librosa.stft(audio_data)
        magnitude = np.abs(stft)
        
        # Estimate noise from quiet portions
        energy_per_frame = np.mean(magnitude, axis=0)
        noise_threshold = np.percentile(energy_per_frame, 10)
        noise_frames = energy_per_frame < noise_threshold
        
        if np.any(noise_frames):
            noise_level = np.mean(energy_per_frame[noise_frames])
            signal_level = np.mean(energy_per_frame[~noise_frames])
            snr = 20 * np.log10(signal_level / (noise_level + 1e-10))
            quality_metrics['snr_db'] = snr
        else:
            quality_metrics['snr_db'] = float('inf')
        
        # Dynamic range
        rms = librosa.feature.rms(y=audio_data)[0]
        dynamic_range = 20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))
        quality_metrics['dynamic_range_db'] = dynamic_range
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        quality_metrics['clipping_percentage'] = (clipped_samples / len(audio_data)) * 100
        
        # Frequency range analysis
        freqs = librosa.fft_frequencies(sr=sample_rate)
        stft_mean = np.mean(np.abs(stft), axis=1)
        
        # Find frequency range with significant energy
        energy_threshold = np.max(stft_mean) * 0.1
        significant_freqs = freqs[stft_mean > energy_threshold]
        
        if len(significant_freqs) > 0:
            quality_metrics['frequency_range_hz'] = {
                'min': float(np.min(significant_freqs)),
                'max': float(np.max(significant_freqs))
            }
        
        return quality_metrics
    
    def save_audio_segment(self, audio_data: np.ndarray, 
                          sample_rate: int,
                          start_time: float,
                          end_time: float,
                          output_path: str) -> str:
        """
        Save a segment of audio to file.
        
        Args:
            audio_data: Full audio data
            sample_rate: Sample rate
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Output file path
            
        Returns:
            Path to saved audio segment
        """
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        segment = audio_data[start_sample:end_sample]
        
        sf.write(output_path, segment, sample_rate)
        return output_path
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        try:
            for audio_file in self.temp_dir.glob("*_audio.wav"):
                audio_file.unlink()
            self.logger.info("Temporary audio files cleaned up")
        except Exception as e:
            self.logger.warning(f"Failed to clean up temp files: {e}")
