# src/smart_zoom/smart_zoom_processor.py
"""Smart Zoom Processor - Main coordinator for AI-powered dynamic framing"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from ..subject_detection import SubjectDetector
from ..utils.gpu_utils import get_device_info
from ..utils.video_utils import VideoReader, VideoWriter
from .zoom_calculator import ZoomCalculator
from .framing_optimizer import FramingOptimizer
from config.smart_zoom_settings import SMART_ZOOM_CONFIG, QUALITY_STANDARDS


@dataclass
class ZoomDecision:
    """Represents a zoom decision for a specific time range."""
    start_time: float
    end_time: float
    zoom_level: float
    crop_position: Tuple[int, int, int, int]  # x, y, width, height
    subjects: List[Dict]
    confidence: float
    strategy: str
    reasoning: str


class SmartZoomProcessor:
    """
    AI-powered smart zoom processor that dynamically frames subjects
    for optimal vertical video content.
    """
    
    def __init__(self, 
                 config: Dict = None,
                 device: str = "cuda",
                 cache_dir: str = "tracking_cache"):
        """
        Initialize Smart Zoom Processor.
        
        Args:
            config: Configuration dictionary (uses default if None)
            device: Device to use for processing ('cuda' or 'cpu')
            cache_dir: Directory for caching tracking data
        """
        self.config = config or SMART_ZOOM_CONFIG
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.subject_detector = SubjectDetector(device=self.device)
        self.zoom_calculator = ZoomCalculator()  # Uses default (9, 16) aspect ratio
        self.framing_optimizer = FramingOptimizer()  # Uses default (9, 16) aspect ratio
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.current_zoom_level = 1.0
        self.zoom_history = []
        self.tracking_data = {}
        
        # GPU information
        self.gpu_info = get_device_info()
        self.logger.info(f"SmartZoomProcessor initialized on {self.device}")
    
    def process_video(self, 
                     video_path: str,
                     output_path: str,
                     audio_analysis: Dict = None,
                     target_aspect_ratio: Tuple[int, int] = (9, 16),
                     is_vertical: bool = True,
                     canvas_type: str = "shorts") -> Dict:
        """
        Process video with smart zoom and dynamic framing.
        
        Args:
            video_path: Path to input video
            output_path: Path to output video
            audio_analysis: Optional audio analysis data from Whisper
            target_aspect_ratio: Target content aspect ratio (width, height)
            is_vertical: Whether the output is vertical (True) or horizontal (False)
            canvas_type: Type of canvas ('shorts' or 'clips')
            
        Returns:
            Processing results and statistics
        """
        try:
            # Determine canvas aspect ratio based on canvas_type
            canvas_aspect_ratio = (9, 16) if canvas_type == "shorts" else (16, 9)
            
            self.logger.info(f"Processing video with canvas_type={canvas_type}, " +
                           f"canvas_aspect_ratio={canvas_aspect_ratio}, " + 
                           f"content_aspect_ratio={target_aspect_ratio}, " +
                           f"is_vertical={is_vertical}")
            
            # Initialize video reader and writer
            reader = VideoReader(video_path)
            writer = VideoWriter(output_path, 
                               fps=reader.fps,
                               target_aspect_ratio=canvas_aspect_ratio)
            
            # Set input video path for audio merging
            writer.set_input_video_path(video_path)
            
            # Get video information
            video_info = reader.get_info()
            
            # Generate zoom decisions for content framing
            zoom_decisions = self._generate_zoom_decisions(
                video_path, audio_analysis, video_info, target_aspect_ratio
            )
            
            # 🎯 OPTIMIZATION: Check if zoom is actually needed
            # If all zoom levels are 1.0, skip smart zoom processing and just do aspect ratio conversion
            all_zoom_levels = [decision.zoom_level for decision in zoom_decisions]
            max_zoom = max(all_zoom_levels) if all_zoom_levels else 1.0
            min_zoom = min(all_zoom_levels) if all_zoom_levels else 1.0
            
            if max_zoom <= 1.01 and min_zoom >= 0.99:  # Allow small tolerance for floating point
                self.logger.info("🎯 OPTIMIZATION: No zoom needed (all zoom levels ≈ 1.0), skipping smart zoom processing")
                self.logger.info("   Using fast aspect ratio conversion instead")
                
                # Fast path: Just convert aspect ratio without zoom processing
                results = self._fast_aspect_ratio_conversion(
                    reader, writer, target_aspect_ratio, is_vertical, canvas_type
                )
            else:
                self.logger.info(f"Smart zoom active: zoom range [{min_zoom:.2f} - {max_zoom:.2f}]")
                
                # Process frames with smart zoom
                results = self._process_frames(
                    reader, writer, zoom_decisions, target_aspect_ratio, is_vertical, canvas_type
                )
            
            # Cleanup
            reader.close()
            writer.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Smart zoom processing failed: {e}")
            raise
    
    def _generate_zoom_decisions(self,
                               video_path: str,
                               audio_analysis: Dict,
                               video_info: Dict,
                               target_aspect_ratio: Tuple[int, int] = (9, 16)) -> List[ZoomDecision]:
        """
        Generate zoom decisions for the entire video.
        
        Args:
            video_path: Path to video file
            audio_analysis: Audio analysis data
            video_info: Video metadata
            target_aspect_ratio: Target aspect ratio (width, height)
            
        Returns:
            List of zoom decisions
        """
        decisions = []
        
        # Sample frames for analysis
        sample_frames = self._sample_frames_for_analysis(video_path, video_info)
        
        # Analyze each sample frame
        for frame_data in sample_frames:
            timestamp = frame_data['timestamp']
            frame = frame_data['frame']
            
            # Detect subjects in frame
            subjects = self.subject_detector.detect_subjects(frame)
            
            # Calculate optimal zoom
            zoom_result = self.zoom_calculator.calculate_smart_zoom(
                frame, subjects, audio_analysis, timestamp, target_aspect_ratio
            )
            
            # Create zoom decision
            decision = ZoomDecision(
                start_time=timestamp,
                end_time=timestamp + 1.0,  # Will be adjusted later
                zoom_level=zoom_result['zoom_level'],
                crop_position=zoom_result['crop_position'],
                subjects=subjects,
                confidence=zoom_result['confidence'],
                strategy=zoom_result['strategy'],
                reasoning=zoom_result['reasoning']
            )
            
            decisions.append(decision)
        
        # Smooth and optimize decisions
        decisions = self._smooth_zoom_decisions(decisions)
        
        # Merge with audio analysis
        if audio_analysis:
            decisions = self._merge_with_audio_analysis(decisions, audio_analysis)
        
        return decisions
    
    def _sample_frames_for_analysis(self, video_path: str, video_info: Dict) -> List[Dict]:
        """
        GPU-accelerated frame sampling from video for zoom analysis.
        Uses FFmpeg with hardware decode for faster frame extraction.
        
        Args:
            video_path: Path to video file
            video_info: Video metadata
            
        Returns:
            List of sampled frames with timestamps
        """
        import subprocess
        import tempfile
        import os
        
        fps = video_info['fps']
        duration = video_info['duration']
        
        # Sample every 2 seconds for analysis
        sample_interval = 2.0
        timestamps = np.arange(0, duration, sample_interval)
        
        frames = []
        
        try:
            # Try GPU-accelerated frame extraction
            self.logger.info(f"🚀 GPU-accelerated frame sampling: {len(timestamps)} frames")
            
            # Create temporary directory for extracted frames
            with tempfile.TemporaryDirectory() as temp_dir:
                # Build select filter for specific timestamps
                select_expr = '+'.join([f'eq(n\\,{int(ts * fps)})' for ts in timestamps[:50]])  # Limit to prevent command too long
                
                # Extract frames using GPU decode
                cmd = [
                    'ffmpeg',
                    '-hwaccel', 'cuda',
                    '-i', video_path,
                    '-vf', f"select='{select_expr}'",
                    '-vsync', '0',
                    '-frame_pts', '1',
                    os.path.join(temp_dir, 'frame_%d.jpg')
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode != 0:
                    # Fallback to CPU decode
                    self.logger.warning(f"⚠️ GPU frame extraction failed, using CPU")
                    raise Exception("GPU extraction failed")
                
                # Load extracted frames
                for idx, timestamp in enumerate(timestamps):
                    frame_path = os.path.join(temp_dir, f'frame_{idx + 1}.jpg')
                    if os.path.exists(frame_path):
                        frame = cv2.imread(frame_path)
                        if frame is not None:
                            frames.append({
                                'timestamp': timestamp,
                                'frame': frame,
                                'frame_number': int(timestamp * fps)
                            })
                
                self.logger.info(f"✅ GPU extracted {len(frames)} frames")
                
        except Exception as e:
            self.logger.warning(f"GPU frame sampling failed: {e}, falling back to OpenCV")
            # Fallback to original OpenCV method
            frames = self._sample_frames_opencv(video_path, timestamps, fps)
        
        return frames
    
    def _sample_frames_opencv(self, video_path: str, timestamps: np.ndarray, fps: float) -> List[Dict]:
        """
        CPU fallback for frame sampling using OpenCV.
        
        Args:
            video_path: Path to video file
            timestamps: Array of timestamps to sample
            fps: Video FPS
            
        Returns:
            List of sampled frames
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        for timestamp in timestamps:
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                frames.append({
                    'timestamp': timestamp,
                    'frame': frame,
                    'frame_number': frame_number
                })
        
        cap.release()
        self.logger.info(f"⚠️ CPU extracted {len(frames)} frames")
        return frames
    
    def _process_frames(self, 
                       reader: VideoReader,
                       writer: VideoWriter,
                       zoom_decisions: List[ZoomDecision],
                       target_aspect_ratio: Tuple[int, int],
                       is_vertical: bool = True,
                       canvas_type: str = "shorts") -> Dict:
        """
        GPU-accelerated frame processing with smart zoom applied.
        Uses batch processing and GPU tensors for maximum performance.
        
        Args:
            reader: Video reader
            writer: Video writer
            zoom_decisions: List of zoom decisions
            target_aspect_ratio: Target aspect ratio
            is_vertical: Whether the output is vertical (True) or horizontal (False)
            canvas_type: Type of canvas ('shorts' or 'clips')
            
        Returns:
            Processing results
        """
        import time
        
        frame_count = 0
        processed_frames = 0
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'zoom_changes': 0,
            'average_zoom_level': 0.0,
            'subject_detection_rate': 0.0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        # Check if we can use GPU acceleration for frame processing
        use_gpu_tensors = torch.cuda.is_available() and self.config.get('gpu_frame_processing', False)
        
        if use_gpu_tensors:
            self.logger.info("🚀 Using GPU tensor processing for frames")
        
        # Batch processing parameters
        batch_size = 32 if use_gpu_tensors else 1
        frame_batch = []
        decision_batch = []
        
        while True:
            frame = reader.read_frame()
            if frame is None:
                # Process remaining frames in batch
                if frame_batch:
                    processed = self._process_frame_batch_gpu(
                        frame_batch, decision_batch, target_aspect_ratio, 
                        canvas_type
                    ) if use_gpu_tensors else self._process_frame_batch_cpu(
                        frame_batch, decision_batch, target_aspect_ratio
                    )
                    
                    for proc_frame in processed:
                        writer.write_frame(proc_frame)
                        processed_frames += 1
                break
            
            frame_count += 1
            timestamp = frame_count / reader.fps
            
            # Get current zoom decision
            current_decision = self._get_decision_for_timestamp(
                zoom_decisions, timestamp
            )
            
            if current_decision:
                frame_batch.append(frame)
                decision_batch.append(current_decision)
                
                # Update statistics
                stats['average_zoom_level'] += current_decision.zoom_level
                
                # Check for zoom changes
                if (hasattr(self, 'previous_zoom_level') and 
                    abs(current_decision.zoom_level - self.previous_zoom_level) > 0.1):
                    stats['zoom_changes'] += 1
                
                self.previous_zoom_level = current_decision.zoom_level
                
                # Process batch when full
                if len(frame_batch) >= batch_size:
                    processed = self._process_frame_batch_gpu(
                        frame_batch, decision_batch, target_aspect_ratio,
                        canvas_type
                    ) if use_gpu_tensors else self._process_frame_batch_cpu(
                        frame_batch, decision_batch, target_aspect_ratio
                    )
                    
                    for proc_frame in processed:
                        writer.write_frame(proc_frame)
                        processed_frames += 1
                    
                    frame_batch = []
                    decision_batch = []
            
            # Progress logging
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_rate = frame_count / elapsed if elapsed > 0 else 0
                self.logger.info(f"Processed {frame_count} frames ({fps_rate:.1f} fps)")
        
        # Calculate final statistics
        stats['total_frames'] = frame_count
        stats['processed_frames'] = processed_frames
        stats['processing_time'] = time.time() - start_time
        
        if processed_frames > 0:
            stats['average_zoom_level'] /= processed_frames
            stats['subject_detection_rate'] = processed_frames / frame_count
        
        avg_fps = frame_count / stats['processing_time'] if stats['processing_time'] > 0 else 0
        self.logger.info(f"✅ Smart zoom processing completed: {stats} (avg {avg_fps:.1f} fps)")
        return stats
    
    def _process_frame_batch_gpu(self,
                                frames: List[np.ndarray],
                                decisions: List[ZoomDecision],
                                target_aspect_ratio: Tuple[int, int],
                                canvas_type: str) -> List[np.ndarray]:
        """
        Process a batch of frames using GPU tensors.
        
        Args:
            frames: List of frames to process
            decisions: Corresponding zoom decisions
            target_aspect_ratio: Target aspect ratio
            canvas_type: Canvas type
            
        Returns:
            List of processed frames
        """
        try:
            # Convert frames to GPU tensor batch
            frame_tensors = []
            for frame in frames:
                # Convert BGR to RGB and normalize
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame_rgb).float().cuda()
                tensor = tensor.permute(2, 0, 1)  # HWC to CHW
                frame_tensors.append(tensor)
            
            # Stack into batch
            batch = torch.stack(frame_tensors)
            
            # Process batch with GPU operations
            processed_batch = []
            for idx, (frame_tensor, decision) in enumerate(zip(frame_tensors, decisions)):
                # Apply zoom using GPU tensor operations
                x, y, w, h = decision.crop_position
                
                # Crop on GPU
                cropped = frame_tensor[:, y:y+h, x:x+w]
                
                # Resize on GPU
                canvas_h = 1920 if canvas_type == "shorts" else 1080
                canvas_w = 1080 if canvas_type == "shorts" else 1920
                
                resized = torch.nn.functional.interpolate(
                    cropped.unsqueeze(0),
                    size=(canvas_h, canvas_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                
                # Convert back to numpy
                frame_np = resized.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                processed_batch.append(frame_bgr)
            
            return processed_batch
            
        except Exception as e:
            self.logger.warning(f"GPU batch processing failed: {e}, falling back to CPU")
            return self._process_frame_batch_cpu(frames, decisions, target_aspect_ratio)
    
    def _process_frame_batch_cpu(self,
                                frames: List[np.ndarray],
                                decisions: List[ZoomDecision],
                                target_aspect_ratio: Tuple[int, int]) -> List[np.ndarray]:
        """
        CPU fallback for batch frame processing.
        
        Args:
            frames: List of frames
            decisions: Corresponding decisions
            target_aspect_ratio: Target aspect ratio
            
        Returns:
            List of processed frames
        """
        processed = []
        for frame, decision in zip(frames, decisions):
            proc_frame = self._apply_smart_zoom(frame, decision, target_aspect_ratio)
            processed.append(proc_frame)
        return processed
    
    def _get_decision_for_timestamp(self, 
                                  decisions: List[ZoomDecision],
                                  timestamp: float) -> Optional[ZoomDecision]:
        """
        Get the appropriate zoom decision for a given timestamp.
        
        Args:
            decisions: List of zoom decisions
            timestamp: Current timestamp
            
        Returns:
            Appropriate zoom decision or None
        """
        for decision in decisions:
            if decision.start_time <= timestamp <= decision.end_time:
                return decision
        
        # Return closest decision if no exact match
        if decisions:
            closest_decision = min(decisions, 
                                 key=lambda d: abs(d.start_time - timestamp))
            return closest_decision
        
        return None
    
    def _fast_aspect_ratio_conversion(self,
                                     reader: VideoReader,
                                     writer: VideoWriter,
                                     target_aspect_ratio: Tuple[int, int],
                                     is_vertical: bool = True,
                                     canvas_type: str = "shorts") -> Dict:
        """
        GPU-accelerated fast path for aspect ratio conversion when no zoom is needed.
        Uses FFmpeg with hardware acceleration for maximum speed.
        
        Args:
            reader: Video reader
            writer: Video writer
            target_aspect_ratio: Target aspect ratio (width, height)
            is_vertical: Whether the output is vertical
            canvas_type: Type of canvas ('shorts' or 'clips')
            
        Returns:
            Processing results
        """
        import time
        import subprocess
        import os
        
        start_time = time.time()
        
        # Determine canvas dimensions
        canvas_aspect_ratio = (9, 16) if canvas_type == "shorts" else (16, 9)
        if canvas_aspect_ratio[0] == 9 and canvas_aspect_ratio[1] == 16:
            output_width, output_height = 1080, 1920
        else:
            base_height = 1080
            output_height = base_height
            output_width = int(base_height * (canvas_aspect_ratio[0] / canvas_aspect_ratio[1]))
        
        target_aspect = target_aspect_ratio[0] / target_aspect_ratio[1]
        
        # Get input video path and output path from reader/writer
        input_path = reader.video_path
        output_path = writer.output_path if hasattr(writer, 'output_path') else input_path.replace('.mp4', '_processed.mp4')
        
        # Build FFmpeg filter for center crop + scale
        crop_filter = f"crop=ih*{target_aspect}:ih:(iw-ih*{target_aspect})/2:0"
        scale_filter = f"scale={output_width}:{output_height}:flags=lanczos"
        
        self.logger.info(f"🚀 GPU-accelerated fast conversion: {input_path} → {output_path}")
        
        try:
            # Try GPU-accelerated encoding first (NVIDIA)
            gpu_cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',
                '-i', input_path,
                '-vf', f"{crop_filter},{scale_filter}",
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-cq', '23',
                '-c:a', 'aac', '-b:a', '128k',
                '-movflags', '+faststart',
                output_path
            ]
            
            self.logger.info(f"🎯 Attempting GPU encoding (h264_nvenc)...")
            result = subprocess.run(gpu_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                # Fallback to CPU encoding if GPU fails
                self.logger.warning(f"⚠️ GPU encoding failed: {result.stderr[:200]}")
                self.logger.info(f"🔄 Falling back to CPU encoding...")
                
                cpu_cmd = [
                    'ffmpeg', '-y',
                    '-i', input_path,
                    '-vf', f"{crop_filter},{scale_filter}",
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',
                    '-crf', '23',
                    '-c:a', 'aac', '-b:a', '128k',
                    '-movflags', '+faststart',
                    output_path
                ]
                subprocess.run(cpu_cmd, check=True, capture_output=True, text=True, timeout=600)
            
            processing_time = time.time() - start_time
            
            # Get frame count from output file
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_frames',
                '-show_entries', 'stream=nb_read_frames',
                '-of', 'csv=p=0',
                output_path
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            frame_count = int(probe_result.stdout.strip()) if probe_result.returncode == 0 else 0
            
            stats = {
                'total_frames': frame_count,
                'processed_frames': frame_count,
                'zoom_changes': 0,
                'average_zoom_level': 1.0,
                'subject_detection_rate': 1.0,
                'processing_time': processing_time,
                'optimization': 'gpu_accelerated_ffmpeg'
            }
            
            self.logger.info(f"✅ GPU-accelerated conversion completed in {processing_time:.2f}s: {frame_count} frames")
            return stats
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ FFmpeg process timed out after 10 minutes")
            raise
        except Exception as e:
            self.logger.error(f"❌ GPU-accelerated conversion failed: {e}")
            # Fall back to original frame-by-frame processing
            self.logger.info("🔄 Falling back to frame-by-frame processing...")
            return self._fast_aspect_ratio_conversion_fallback(
                reader, writer, target_aspect_ratio, is_vertical, canvas_type, 
                output_width, output_height, target_aspect
            )
    
    def _fast_aspect_ratio_conversion_fallback(self,
                                              reader: VideoReader,
                                              writer: VideoWriter,
                                              target_aspect_ratio: Tuple[int, int],
                                              is_vertical: bool,
                                              canvas_type: str,
                                              output_width: int,
                                              output_height: int,
                                              target_aspect: float) -> Dict:
        """
        CPU fallback for aspect ratio conversion.
        
        Args:
            reader: Video reader
            writer: Video writer
            target_aspect_ratio: Target aspect ratio
            is_vertical: Whether output is vertical
            canvas_type: Canvas type
            output_width: Output width
            output_height: Output height
            target_aspect: Target aspect ratio as float
            
        Returns:
            Processing statistics
        """
        import time
        
        frame_count = 0
        processed_frames = 0
        start_time = time.time()
        
        self.logger.info("⚠️ Using CPU frame-by-frame processing (slower)")
        
        while True:
            frame = reader.read_frame()
            if frame is None:
                break
            
            frame_count += 1
            frame_h, frame_w = frame.shape[:2]
            
            # Center crop to target aspect ratio
            frame_aspect = frame_w / frame_h
            
            if frame_aspect > target_aspect:
                new_width = int(frame_h * target_aspect)
                start_x = (frame_w - new_width) // 2
                cropped = frame[:, start_x:start_x + new_width]
            else:
                new_height = int(frame_w / target_aspect)
                start_y = (frame_h - new_height) // 2
                cropped = frame[start_y:start_y + new_height, :]
            
            # Resize to output dimensions
            resized = cv2.resize(cropped, (output_width, output_height), 
                               interpolation=cv2.INTER_LANCZOS4)
            
            writer.write_frame(resized)
            processed_frames += 1
            
            if frame_count % 100 == 0:
                self.logger.info(f"CPU processing: {frame_count} frames")
        
        processing_time = time.time() - start_time
        
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'zoom_changes': 0,
            'average_zoom_level': 1.0,
            'subject_detection_rate': 1.0,
            'processing_time': processing_time,
            'optimization': 'cpu_fallback'
        }
        
        self.logger.info(f"CPU conversion completed in {processing_time:.2f}s: {stats}")
        return stats
    
    def _apply_smart_zoom(self, 
                         frame: np.ndarray,
                         decision: ZoomDecision,
                         target_aspect_ratio: Tuple[int, int]) -> np.ndarray:
        """
        Apply smart zoom to a frame based on zoom decision.
        
        Args:
            frame: Input frame
            decision: Zoom decision
            target_aspect_ratio: Target content aspect ratio
            
        Returns:
            Processed frame with smart zoom applied
        """
        # Get crop position and size
        x, y, w, h = decision.crop_position
        
        # Ensure crop is within frame boundaries
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - w))
        y = max(0, min(y, frame_h - h))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)
        
        # Crop the frame based on subject detection
        cropped_frame = frame[y:y+h, x:x+w]
        
        # Calculate content dimensions maintaining content aspect ratio
        content_w, content_h = target_aspect_ratio
        content_aspect = content_w / content_h
        
        # Get cropped frame dimensions
        crop_h, crop_w = cropped_frame.shape[:2]
        crop_aspect = crop_w / crop_h
        
        # Determine how to crop to exact content aspect ratio
        if crop_aspect > content_aspect:
            # Cropped frame is wider than target content - need to crop width
            new_width = int(crop_h * content_aspect)
            start_x = (crop_w - new_width) // 2
            final_crop = cropped_frame[:, start_x:start_x + new_width]
        else:
            # Cropped frame is taller than target content - need to crop height
            new_height = int(crop_w / content_aspect)
            start_y = (crop_h - new_height) // 2
            final_crop = cropped_frame[start_y:start_y + new_height, :]
        
        # Now determine the canvas dimensions based on whether this is a shorts or other format
        # For shorts, always use 9:16 as the final output dimensions
        # For other formats, use the content aspect ratio
        canvas_aspect_ratio = (9, 16)  # Default for shorts (vertical)
        
        # Calculate output resolution based on canvas type
        if canvas_aspect_ratio[0] == 9 and canvas_aspect_ratio[1] == 16:
            output_width, output_height = 1080, 1920  # Standard shorts resolution
        else:
            # For other aspect ratios, scale to reasonable size
            base_height = 1080
            output_height = base_height
            output_width = int(base_height * (canvas_aspect_ratio[0] / canvas_aspect_ratio[1]))
        
        # Resize content to fit within the canvas while maintaining the content aspect ratio
        content_h, content_w = final_crop.shape[:2]
        content_aspect_ratio = content_w / content_h
        
        # Calculate how content fits in the canvas
        if output_width / output_height > content_aspect_ratio:
            # Canvas is wider than content - fit by height
            content_height = output_height
            content_width = int(content_height * content_aspect_ratio)
            content_x = (output_width - content_width) // 2
            content_y = 0
        else:
            # Canvas is taller than content - fit by width
            content_width = output_width
            content_height = int(content_width / content_aspect_ratio)
            content_x = 0
            content_y = (output_height - content_height) // 2
        
        # Resize the content
        resized_content = cv2.resize(final_crop, (content_width, content_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Create the canvas (black background)
        canvas = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Place the content on the canvas
        canvas[content_y:content_y+content_height, content_x:content_x+content_width] = resized_content
        
        # Apply smoothing and stabilization
        stabilized_frame = self._apply_stabilization(canvas)
        
        return stabilized_frame
    
    def _apply_stabilization(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply stabilization and smoothing to the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Stabilized frame
        """
        # Simple stabilization - in practice, you'd use more sophisticated methods
        # This is a placeholder for actual stabilization algorithms
        
        # Apply slight blur to reduce jitter
        kernel_size = 3
        stabilized = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Blend with original frame
        alpha = 0.9
        stabilized = cv2.addWeighted(frame, alpha, stabilized, 1 - alpha, 0)
        
        return stabilized
    
    def _smooth_zoom_decisions(self, decisions: List[ZoomDecision]) -> List[ZoomDecision]:
        """
        Smooth zoom decisions to avoid jarring transitions.
        
        Args:
            decisions: List of raw zoom decisions
            
        Returns:
            Smoothed zoom decisions
        """
        if len(decisions) < 2:
            return decisions
        
        smoothed = []
        window_size = 3
        
        for i in range(len(decisions)):
            # Get window of decisions around current decision
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(decisions), i + window_size // 2 + 1)
            window = decisions[start_idx:end_idx]
            
            # Calculate smoothed zoom level
            zoom_levels = [d.zoom_level for d in window]
            smoothed_zoom = np.mean(zoom_levels)
            
            # Apply smoothing constraints
            max_change = self.config['zoom_behavior']['max_zoom_per_second']
            if i > 0:
                prev_zoom = smoothed[-1].zoom_level
                time_diff = decisions[i].start_time - smoothed[-1].start_time
                max_allowed_change = max_change * time_diff
                
                if abs(smoothed_zoom - prev_zoom) > max_allowed_change:
                    if smoothed_zoom > prev_zoom:
                        smoothed_zoom = prev_zoom + max_allowed_change
                    else:
                        smoothed_zoom = prev_zoom - max_allowed_change
            
            # Create smoothed decision
            smoothed_decision = ZoomDecision(
                start_time=decisions[i].start_time,
                end_time=decisions[i].end_time,
                zoom_level=smoothed_zoom,
                crop_position=decisions[i].crop_position,
                subjects=decisions[i].subjects,
                confidence=decisions[i].confidence,
                strategy=decisions[i].strategy,
                reasoning=f"Smoothed: {decisions[i].reasoning}"
            )
            
            smoothed.append(smoothed_decision)
        
        return smoothed
    
    def _merge_with_audio_analysis(self, 
                                 decisions: List[ZoomDecision],
                                 audio_analysis: Dict) -> List[ZoomDecision]:
        """
        Merge zoom decisions with audio analysis data.
        
        Args:
            decisions: Visual zoom decisions
            audio_analysis: Audio analysis data
            
        Returns:
            Merged zoom decisions
        """
        # Get audio-based zoom recommendations
        # Handle cases where audio_analysis might contain error data
        if not audio_analysis or not isinstance(audio_analysis, dict):
            self.logger.warning("Invalid audio analysis data, using visual decisions only")
            return decisions
            
        audio_zoom_data = audio_analysis.get('zoom_decisions', [])
        
        # Ensure audio_zoom_data is a list
        if not isinstance(audio_zoom_data, list):
            self.logger.warning("Audio zoom data is not a list, using visual decisions only")
            return decisions
        
        merged_decisions = []
        
        for decision in decisions:
            # Find corresponding audio recommendation
            audio_recommendation = self._find_audio_recommendation(
                decision, audio_zoom_data
            )
            
            if audio_recommendation:
                # Merge visual and audio recommendations
                merged_decision = self._merge_recommendations(
                    decision, audio_recommendation
                )
                merged_decisions.append(merged_decision)
            else:
                merged_decisions.append(decision)
        
        return merged_decisions
    
    def _find_audio_recommendation(self, 
                                 decision: ZoomDecision,
                                 audio_zoom_data: List[Dict]) -> Optional[Dict]:
        """
        Find corresponding audio recommendation for a visual decision.
        
        Args:
            decision: Visual zoom decision
            audio_zoom_data: Audio-based zoom recommendations
            
        Returns:
            Matching audio recommendation or None
        """
        for audio_rec in audio_zoom_data:
            # Ensure audio_rec is a dictionary and has required keys
            if not isinstance(audio_rec, dict):
                continue
                
            # Check for required keys with safe access
            try:
                start_time = audio_rec.get('start_time')
                end_time = audio_rec.get('end_time')
                
                if start_time is None or end_time is None:
                    continue
                
                # Convert to float to ensure proper comparison
                start_time = float(start_time)
                end_time = float(end_time)
                
                # Check for time overlap
                if (start_time <= decision.start_time <= end_time or
                    decision.start_time <= start_time <= decision.end_time):
                    return audio_rec
                    
            except (TypeError, ValueError, KeyError) as e:
                self.logger.warning(f"Invalid audio recommendation data: {e}")
                continue
        
        return None
    
    def _merge_recommendations(self, 
                             visual_decision: ZoomDecision,
                             audio_recommendation: Dict) -> ZoomDecision:
        """
        Merge visual and audio zoom recommendations.
        
        Args:
            visual_decision: Visual-based zoom decision
            audio_recommendation: Audio-based zoom recommendation
            
        Returns:
            Merged zoom decision
        """
        # Weight visual and audio recommendations
        visual_weight = 0.7
        audio_weight = 0.3
        
        # Safely extract audio recommendation values with defaults
        try:
            audio_zoom_level = float(audio_recommendation.get('zoom_level', visual_decision.zoom_level))
            audio_confidence = float(audio_recommendation.get('confidence', 0.5))
            audio_strategy = str(audio_recommendation.get('strategy', 'default'))
            audio_reasoning = str(audio_recommendation.get('reasoning', 'No audio reasoning available'))
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid audio recommendation values, using visual decision: {e}")
            return visual_decision
        
        # Merge zoom levels
        merged_zoom = (visual_decision.zoom_level * visual_weight + 
                      audio_zoom_level * audio_weight)
        
        # Merge confidence
        merged_confidence = (visual_decision.confidence * visual_weight +
                           audio_confidence * audio_weight)
        
        # Choose strategy based on higher confidence
        if visual_decision.confidence > audio_confidence:
            strategy = visual_decision.strategy
            reasoning = f"Visual: {visual_decision.reasoning}, Audio: {audio_reasoning}"
        else:
            strategy = audio_strategy
            reasoning = f"Audio: {audio_reasoning}, Visual: {visual_decision.reasoning}"
        
        return ZoomDecision(
            start_time=visual_decision.start_time,
            end_time=visual_decision.end_time,
            zoom_level=merged_zoom,
            crop_position=visual_decision.crop_position,
            subjects=visual_decision.subjects,
            confidence=merged_confidence,
            strategy=strategy,
            reasoning=reasoning
        )
    
    def validate_zoom_quality(self, decisions: List[ZoomDecision]) -> Dict:
        """
        Validate zoom decisions against quality standards.
        
        Args:
            decisions: List of zoom decisions
            
        Returns:
            Quality validation results
        """
        standards = QUALITY_STANDARDS
        
        validation_results = {
            'overall_quality': 0.0,
            'stability_score': 0.0,
            'composition_score': 0.0,
            'subject_visibility_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check stability
        zoom_changes = []
        for i in range(1, len(decisions)):
            zoom_change = abs(decisions[i].zoom_level - decisions[i-1].zoom_level)
            time_diff = decisions[i].start_time - decisions[i-1].start_time
            zoom_changes.append(zoom_change / time_diff if time_diff > 0 else 0)
        
        max_change_rate = max(zoom_changes) if zoom_changes else 0
        stability_score = 1.0 - min(max_change_rate / 0.5, 1.0)  # Penalty for fast changes
        
        # Check subject visibility
        visibility_scores = []
        for decision in decisions:
            if decision.subjects:
                # Calculate subject size in frame
                subject_areas = [self._calculate_subject_area(sub) for sub in decision.subjects]
                avg_area = np.mean(subject_areas)
                visibility_score = min(avg_area / 0.3, 1.0)  # Target 30% of frame
                visibility_scores.append(visibility_score)
        
        subject_visibility_score = np.mean(visibility_scores) if visibility_scores else 0.0
        
        # Overall quality score
        overall_quality = (stability_score + subject_visibility_score) / 2
        
        validation_results.update({
            'overall_quality': overall_quality,
            'stability_score': stability_score,
            'subject_visibility_score': subject_visibility_score,
        })
        
        # Generate recommendations
        if stability_score < 0.8:
            validation_results['issues'].append("Zoom transitions too fast")
            validation_results['recommendations'].append("Apply more smoothing to zoom transitions")
        
        if subject_visibility_score < 0.6:
            validation_results['issues'].append("Subjects not well-framed")
            validation_results['recommendations'].append("Adjust zoom levels to better frame subjects")
        
        return validation_results
    
    def _calculate_subject_area(self, subject: Dict) -> float:
        """Calculate the area of a subject as a fraction of the frame."""
        bbox = subject.get('bbox', [0, 0, 0, 0])
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics and performance metrics."""
        return {
            'device': self.device,
            'gpu_info': self.gpu_info,
            'config': self.config,
            'zoom_history': self.zoom_history,
            'cache_dir': str(self.cache_dir)
        }
