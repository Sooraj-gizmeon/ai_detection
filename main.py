# main.py
"""Main entry point for the Video-to-Shorts Pipeline"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import JSON utilities for serialization fixes
from src.utils.json_utils import clean_for_serialization
from src.utils.video_analysis_logger import VideoAnalysisLogger
from src.utils.database_api_client import DatabaseAPIClient, create_clip_data

from src.audio_analysis import WhisperAnalyzer
from src.smart_zoom import SmartZoomProcessor
from src.smart_zoom.object_aware_zoom import ObjectAwareZoomProcessor
from src.subject_detection import SubjectDetector
from src.scene_detection import SceneDetector
from src.content_analysis import ContentAnalyzer
from src.video_processing import VideoProcessor
from src.ai_integration import OllamaClient
from src.vision_analysis import VisionProcessor
from src.utils import setup_logging, get_device_info, ensure_directory, SubtitleProcessor
from src.utils.title_generator import TitleGenerator
from src.utils.variation_manager import VariationManager
from config.smart_zoom_settings import SMART_ZOOM_CONFIG


class VideoToShortsProcessor:
    """
    Main coordinator for the Video-to-Shorts pipeline.
    Orchestrates all components to convert horizontal videos into vertical shorts.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the processor with configuration.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Setup logging
        setup_logging(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize directories
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        self.cache_dir = Path("cache")
        
        # Get absolute paths for debugging
        abs_input_dir = self.input_dir.absolute()
        abs_output_dir = self.output_dir.absolute()
        abs_temp_dir = self.temp_dir.absolute()
        abs_cache_dir = self.cache_dir.absolute()
        
        self.logger.info(f"Absolute directory paths:")
        self.logger.info(f"  Input dir: {abs_input_dir}")
        self.logger.info(f"  Output dir: {abs_output_dir}")
        self.logger.info(f"  Temp dir: {abs_temp_dir}")
        self.logger.info(f"  Cache dir: {abs_cache_dir}")
        
        for directory in [self.input_dir, self.output_dir, self.temp_dir, self.cache_dir]:
            ensure_directory(directory)
            
        # Double check directory permissions
        self.logger.info("Checking directory permissions:")
        self.logger.info(f"  Output dir writable: {os.access(str(self.output_dir), os.W_OK)}")
        self.logger.info(f"  Output dir exists after ensure: {self.output_dir.exists()}")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.whisper_analyzer = None
        self.smart_zoom_processor = None
        self.object_aware_zoom_processor = None
        self.subject_detector = None
        self.scene_detector = None
        self.content_analyzer = None
        self.video_processor = None
        self.ollama_client = None
        self.vision_processor = None
        self.subtitle_processor = None
        self.title_generator = None
        self.database_api_client = None
        self.variation_manager = None
        
        # Processing statistics
        self.stats = {
            'videos_processed': 0,
            'shorts_generated': 0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        self.logger.info("VideoToShortsProcessor initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file or use defaults."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    if config_data:
                        self.logger.info(f"Successfully loaded config from {config_path}")
                        return config_data
                    else:
                        self.logger.warning(f"Config file {config_path} is empty, using defaults")
            except json.JSONDecodeError:
                self.logger.error(f"Failed to parse config file {config_path} (invalid JSON)")
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        else:
            if config_path:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
            else:
                self.logger.info("No config file specified, using defaults")
        
        # Return default configuration
        return {
            'whisper_model': 'base',
            'target_short_duration': (60, 120),
            'target_aspect_ratio': (9, 16),
            'max_shorts_per_video': 10,
            'quality_threshold': 0.7,
            'smart_zoom_enabled': True,
            'ollama_analysis_enabled': True,
            'vision_analysis_enabled': True,
            'cleanup_temp_files': True,
            'linear_cut': False,
            'linear_duration': None,
            'append_part_number': True
        }
    
    async def initialize_components(self):
        """Initialize all processing components."""
        try:
            # Get device info
            device_info = get_device_info()
            device = "cuda" if device_info['cuda_available'] else "cpu"
            
            self.logger.info(f"Initializing components on device: {device}")
            
            # Initialize Whisper analyzer
            self.whisper_analyzer = WhisperAnalyzer(
                model_size=self.config['whisper_model'],
                device=device,
                cache_dir=str(self.cache_dir / "whisper")
            )
            
            # Initialize Smart Zoom processor
            self.smart_zoom_processor = SmartZoomProcessor(
                config=SMART_ZOOM_CONFIG,
                device=device,
                cache_dir=str(self.cache_dir / "smart_zoom")
            )
            
            # Initialize Object-Aware Zoom processor
            self.object_aware_zoom_processor = None  # Will be initialized after content analyzer
            
            # Initialize Subject detector
            self.subject_detector = SubjectDetector(
                device=device,
                model_dir=str(Path("models"))
            )
            
            # Initialize Scene detector
            self.scene_detector = SceneDetector()
            
            # Initialize Content analyzer with backward compatibility
            object_detection_enabled = self.config.get('enable_object_detection', True)
            ai_reframing_enabled = self.config.get('enable_ai_reframing', True)
            
            try:
                # Try new ContentAnalyzer constructor with object detection support
                if self.ollama_client:
                    self.content_analyzer = ContentAnalyzer(
                        ollama_client=self.ollama_client,
                        enable_object_detection=object_detection_enabled,
                        enable_ai_reframing=ai_reframing_enabled
                    )
                else:
                    self.content_analyzer = ContentAnalyzer(
                        enable_object_detection=object_detection_enabled,
                        enable_ai_reframing=ai_reframing_enabled
                    )
                self.logger.info("ContentAnalyzer initialized with object detection support")
                
                # Initialize Object-Aware Zoom processor if object detection is enabled
                if object_detection_enabled and hasattr(self.content_analyzer, 'object_detector') and self.content_analyzer.object_detector:
                    try:
                        ai_reframer = getattr(self.content_analyzer, 'ai_reframer', None)
                        self.object_aware_zoom_processor = ObjectAwareZoomProcessor(
                            base_zoom_processor=self.smart_zoom_processor,
                            object_detector=self.content_analyzer.object_detector,
                            ai_reframer=ai_reframer
                        )
                        self.logger.info("Object-aware zoom processor initialized with object detection integration")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize object-aware zoom processor: {e}")
                        self.object_aware_zoom_processor = None
                else:
                    self.logger.info("Object-aware zoom processor not initialized (object detection disabled or not available)")
                    
            except (TypeError, ImportError) as e:
                # Fallback to old ContentAnalyzer constructor for backward compatibility
                self.logger.warning(f"New ContentAnalyzer constructor failed ({e}), falling back to legacy version")
                if self.ollama_client:
                    self.content_analyzer = ContentAnalyzer(ollama_client=self.ollama_client)
                else:
                    self.content_analyzer = ContentAnalyzer()
                self.logger.info("ContentAnalyzer initialized in legacy mode (object detection disabled)")
                self.object_aware_zoom_processor = None
            
            # Initialize Video processor
            self.video_processor = VideoProcessor(
                temp_dir=str(self.temp_dir),
                output_dir=str(self.output_dir)
            )
            
            # Initialize Ollama client
            if self.config['ollama_analysis_enabled']:
                self.ollama_client = OllamaClient(
                    cache_dir=str(self.cache_dir / "ollama")
                )
                
                # Test connection
                connection_test = await self.ollama_client.test_connection()
                if connection_test['status'] == 'error':
                    self.logger.warning("Ollama connection failed, continuing without AI analysis")
                    self.ollama_client = None
            
            # Initialize Vision processor if enabled in config (frame extraction runs even without Ollama client)
            if self.config.get('vision_analysis_enabled', True):
                self.vision_processor = VisionProcessor(
                    ollama_client=self.ollama_client,
                    frame_target_size=(512, 288),  # Lower resolution for efficiency
                    sample_rate=0.5  # Sample 0.5 frames per second
                )
                self.logger.info("Vision processor initialized")
                if not self.ollama_client:
                    self.logger.warning("Ollama client unavailable: vision model steps will be skipped but frames will still be processed")
            else:
                self.logger.info("Vision analysis disabled in config")
            
            # Initialize Subtitle processor
            self.subtitle_processor = SubtitleProcessor()
            self.logger.info("Subtitle processor initialized")
            
            # Initialize Title generator
            self.title_generator = TitleGenerator(ollama_client=self.ollama_client)
            self.logger.info("Title generator initialized")
            
            # Initialize Database API Client for API-based database operations
            self.database_api_client = DatabaseAPIClient()
            self.logger.info("Database API client initialized")
            
            # Initialize Variation Manager for output diversity
            enable_variation = self.config.get('enable_output_variation', True)
            self.variation_manager = VariationManager.create_session_variation_manager(enable_variation)
            if enable_variation:
                self.logger.info(f"Variation manager initialized - seed: {self.variation_manager.seed}")
            else:
                self.logger.info("Output variation disabled")
            
            # Set variation manager on content analyzer if it has a prompt-based analyzer
            if hasattr(self.content_analyzer, 'prompt_analyzer') and self.content_analyzer.prompt_analyzer:
                self.content_analyzer.prompt_analyzer.set_variation_manager(self.variation_manager)
                self.logger.info("Variation manager set on content analyzer")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    async def _analyze_with_user_prompt(self,
                                      user_prompt: str,
                                      video_path: str,
                                      audio_analysis: Dict,
                                      scene_analysis: Dict,
                                      analysis_logger: VideoAnalysisLogger,
                                      llm_provider: str = "ollama") -> Dict:
        """
        Analyze video content based on user prompt.
        
        Args:
            user_prompt: User's prompt for content selection
            video_path: Path to video file
            audio_analysis: Audio analysis results
            scene_analysis: Scene analysis results
            analysis_logger: Logger for analysis details
            llm_provider: LLM provider to use ('openai' or 'ollama')
            
        Returns:
            Prompt-based analysis results
        """
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Perform vision analysis if available
            vision_analysis = None
            if self.vision_processor and self.config.get('vision_analysis_enabled', True):
                self.logger.info("Performing vision analysis for prompt-based selection...")
                
                # Generate a representative set of segments for vision analysis
                temp_segments = await self._generate_temp_segments_for_vision(
                    video_path, audio_analysis, scene_analysis
                )
                
                if temp_segments:
                    vision_analysis = await self._analyze_visual_content(
                        video_path, temp_segments, audio_analysis
                    )
            
            # Use content analyzer with prompt support and enhanced features
            ai_reframe_enabled = self.config.get('ai_reframe', False)
            
            # PHASE 1 ENHANCEMENT: Enhanced context-aware analysis
            # Try to find existing celebrity results for object detection
            celebrity_index_path = self.config.get('celebrity_index_path')
            if not celebrity_index_path:
                # Look for the most recent celebrity result file
                import os
                from pathlib import Path
                result_dir = Path("output/celebrity_results")
                if result_dir.exists():
                    result_files = list(result_dir.glob("*.json"))
                    if result_files:
                        # Use the most recent file
                        celebrity_index_path = str(sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
                        self.logger.info(f"Found existing celebrity result file: {celebrity_index_path}")
            
            self.logger.info(f"Passing celebrity_index_path to ContentAnalyzer: {celebrity_index_path}")
            analysis_results = await self.content_analyzer.analyze_with_user_prompt(
                user_prompt=user_prompt,
                video_path=video_path,
                video_info=video_info,
                audio_analysis=audio_analysis,
                scene_analysis=scene_analysis,
                vision_analysis=vision_analysis,
                target_duration=self.config['target_short_duration'],
                max_shorts=self.config['max_shorts_per_video'],
                ai_reframe=ai_reframe_enabled,
                celebrity_index_path=self.config.get('celebrity_index_path'),
                llm_provider=llm_provider
            )
            
            # Defensive check: ensure analysis_results is a dictionary
            if not isinstance(analysis_results, dict):
                self.logger.error(f"Analysis results is not a dictionary, got {type(analysis_results)}: {analysis_results}")
                return {
                    'status': 'error',
                    'error': f'Invalid analysis results type: {type(analysis_results)}',
                    'segments': []
                }
            
            # Log object detection results if available
            if 'object_detection_results' in analysis_results:
                object_results = analysis_results['object_detection_results']
                if object_results:
                    analysis_logger.log_object_detection(object_results)
                    self.logger.info(f"Logged object detection results: {object_results.get('total_detections', 0)} detections")
                else:
                    analysis_logger.log_object_detection_disabled("empty_results")
            else:
                analysis_logger.log_object_detection_disabled("not_in_results")
            return analysis_results
            
        except Exception as e:
            # DEFENSIVE FIX: Check for the specific theme_templates corruption error
            if "'list' object has no attribute 'items'" in str(e):
                self.logger.error(f"DETECTED theme_templates corruption error: {e}")
                self.logger.warning("Attempting to recover from theme_templates corruption...")
                
                # Try to recover by calling get_supported_themes which has defensive fixes
                try:
                    self.content_analyzer.get_supported_themes()
                    self.logger.info("Recovery attempt completed, retrying analysis...")
                    
                    # Retry the analysis once after recovery
                    self.logger.info(f"Passing celebrity_index_path to ContentAnalyzer (retry): {self.config.get('celebrity_index_path')}")
                    analysis_results = await self.content_analyzer.analyze_with_user_prompt(
                        user_prompt=user_prompt,
                        video_path=video_path,
                        video_info=video_info,
                        audio_analysis=audio_analysis,
                        scene_analysis=scene_analysis,
                        vision_analysis=vision_analysis,
                        target_duration=self.config['target_short_duration'],
                        max_shorts=self.config['max_shorts_per_video'],
                        ai_reframe=ai_reframe_enabled,
                        celebrity_index_path=self.config.get('celebrity_index_path'),
                        llm_provider=llm_provider
                    )
                    
                    self.logger.info("Analysis retry after recovery was successful!")
                    return analysis_results
                    
                except Exception as retry_error:
                    self.logger.error(f"Recovery attempt failed: {retry_error}")
                    # Fall through to normal error handling
            
            self.logger.error(f"Error in prompt-based analysis1: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    async def _analyze_comprehensive_content(self,
                                           video_path: str,
                                           audio_analysis: Dict,
                                           scene_analysis: Dict,
                                           analysis_logger: VideoAnalysisLogger) -> Dict:
        """
        Perform comprehensive content analysis without user prompt.
        
        Args:
            video_path: Path to video file
            audio_analysis: Audio analysis results
            scene_analysis: Scene analysis results
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Check for celebrity_index_path (which may contain object detection results)
            celebrity_index_path = self.config.get('celebrity_index_path')
            
            # Generate comprehensive candidate segments
            all_candidate_segments = self.content_analyzer.segment_generator.generate_all_possible_segments(
                video_info=video_info,
                audio_analysis=audio_analysis,
                scene_analysis=scene_analysis,
                target_duration=self.config['target_short_duration'],
                max_total_segments=300,
                celebrity_index_path=celebrity_index_path
            )
            
            self.logger.info(f"Generated {len(all_candidate_segments)} comprehensive candidate segments")
            analysis_logger.log_segment_generation(all_candidate_segments, "comprehensive_generation")
            
            # CRITICAL FIX: Check if we have object-based segments with reference_match.score
            # If so, prioritize them for object-detection-based short generation
            object_reference_segments = [seg for seg in all_candidate_segments if seg.get('has_object') and seg.get('object_score', 0) > 0]
            if object_reference_segments:
                self.logger.info(f"Found {len(object_reference_segments)} object reference segments - prioritizing for object-based generation")
                
                # For object-based generation, we don't need further analysis
                # Just return the object reference segments as final segments
                final_segments = object_reference_segments[:self.config['max_shorts_per_video']]
                
                return {
                    'status': 'success',
                    'analysis_method': 'object_based_detection',
                    'total_candidates_analyzed': len(all_candidate_segments),
                    'object_reference_segments': len(object_reference_segments),
                    'final_selected_segments': len(final_segments),
                    'segments': final_segments,
                    'comprehensive_coverage': True,
                    'object_based_generation': True
                }
            
            # Perform object detection if enabled (even without specific prompts)
            object_detection_results = {}
            if (hasattr(self.content_analyzer, 'enable_object_detection') and 
                self.content_analyzer.enable_object_detection and 
                hasattr(self.content_analyzer, 'object_detector') and 
                self.content_analyzer.object_detector):
                self.logger.info("Performing general object detection analysis...")
                try:
                    # Use general object detection prompt
                    general_prompt_analysis = await self.content_analyzer.object_detector.analyze_prompt_for_objects(
                        "detect all objects in the scene", self.ollama_client
                    )
                    
                    object_detection_results = await self.content_analyzer.object_detector.detect_objects_in_video(
                        video_path, general_prompt_analysis, sample_rate=1.0
                    )
                    
                    if object_detection_results:
                        analysis_logger.log_object_detection(object_detection_results, general_prompt_analysis)
                        self.logger.info(f"Object detection completed: {object_detection_results.get('total_detections', 0)} objects detected")
                except Exception as e:
                    self.logger.warning(f"Object detection failed: {e}")
                    object_detection_results = {}
                    analysis_logger.log_object_detection_disabled("detection_failed")
            else:
                analysis_logger.log_object_detection_disabled("not_enabled_or_configured")
            
            # Perform vision analysis if enabled
            vision_analysis = None
            if self.vision_processor and self.config.get('vision_analysis_enabled', True):
                self.logger.info("Performing comprehensive vision analysis...")
                try:
                    # Add timeout protection to prevent hanging
                    vision_analysis = await asyncio.wait_for(
                        self._analyze_visual_content(video_path, all_candidate_segments, audio_analysis),
                        timeout=120  # 2 minute timeout
                    )
                    self.logger.info("Vision analysis completed successfully")
                    analysis_logger.log_vision_analysis(vision_analysis)
                except asyncio.TimeoutError:
                    self.logger.warning("Vision analysis timed out after 2 minutes, proceeding without vision data")
                    vision_analysis = None
                except Exception as e:
                    self.logger.warning(f"Vision analysis failed: {e}, proceeding without vision data")
                    vision_analysis = None
            
            # Enhanced content analysis with Ollama
            ollama_analysis = None
            if self.ollama_client:
                self.logger.info("Analyzing content with Ollama...")
                try:
                    # Add timeout protection for Ollama analysis
                    llm_start_time = datetime.now()
                    if vision_analysis:
                        # Use dual-modal analysis
                        self.logger.info("Performing dual-modal analysis with audio and vision data")
                        ollama_analysis = await asyncio.wait_for(
                            self._analyze_content_with_dual_modal(audio_analysis, vision_analysis),
                            timeout=180  # 3 minute timeout
                        )
                    else:
                        # Use audio-only analysis
                        ollama_analysis = await asyncio.wait_for(
                            self._analyze_content_with_ollama(audio_analysis),
                            timeout=180  # 3 minute timeout
                        )
                    llm_duration = (datetime.now() - llm_start_time).total_seconds()
                    self.logger.info("Dual-modal analysis completed successfully")
                    
                    # Log LLM interaction to analysis logger
                    analysis_logger.log_llm_interaction(
                        interaction_type="comprehensive_content_analysis",
                        prompt="[Combined audio transcription and vision analysis for content understanding]",
                        response=json.dumps(ollama_analysis) if ollama_analysis else "No response",
                        model=self.ollama_client.get_best_model("analysis") if hasattr(self.ollama_client, 'get_best_model') else "unknown",
                        duration_seconds=llm_duration,
                        metadata={"vision_enabled": vision_analysis is not None}
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Ollama analysis timed out after 3 minutes, proceeding without LLM analysis")
                    ollama_analysis = None
                except Exception as e:
                    self.logger.warning(f"Ollama analysis failed: {e}, proceeding without LLM analysis")
                    ollama_analysis = None
            
            # Select final segments using enhanced selection
            final_segments = await self._select_final_segments(
                video_path,
                audio_analysis,
                ollama_analysis,
                scene_analysis,
                vision_analysis,
                all_candidate_segments
            )
            
            # Log segment selection
            analysis_logger.log_segment_selection(
                final_segments,
                "comprehensive_enhanced_selection",
                self.config.get('quality_threshold', 0.7),
                len(all_candidate_segments)
            )
            
            # Emergency fallback: ensure we always have some segments
            if not final_segments and all_candidate_segments:
                self.logger.warning("No segments selected by standard analysis, applying emergency fallback")
                # Sort by text quality and duration, take top segments
                sorted_candidates = sorted(
                    all_candidate_segments,
                    key=lambda x: (
                        x.get('text_quality_score', 0) * 0.4 +
                        x.get('preliminary_quality', 0) * 0.6
                    ),
                    reverse=True
                )
                final_segments = sorted_candidates[:self.config['max_shorts_per_video']]
                self.logger.info(f"Emergency fallback selected {len(final_segments)} segments")
            
            return {
                'status': 'success',
                'analysis_method': 'comprehensive_standard_with_fallback',
                'total_candidates_analyzed': len(all_candidate_segments),
                'final_selected_segments': len(final_segments),
                'segments': final_segments,
                'comprehensive_coverage': True,
                'ollama_analysis': ollama_analysis,
                'vision_analysis': vision_analysis,
                'used_emergency_fallback': not final_segments and all_candidate_segments
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'segments': []
            }
    
    async def _generate_temp_segments_for_vision(self,
                                               video_path: str,
                                               audio_analysis: Dict,
                                               scene_analysis: Dict) -> List[Dict]:
        """
        Generate a representative set of segments for vision analysis.
        Used when we need vision data for prompt-based analysis.
        
        Args:
            video_path: Path to video file
            audio_analysis: Audio analysis results
            scene_analysis: Scene analysis results
            
        Returns:
            Representative segments for vision analysis
        """
        try:
            video_info = self.video_processor.get_video_info(video_path)
            
            # Generate a smaller set of representative segments
            temp_segments = self.content_analyzer.segment_generator.generate_all_possible_segments(
                video_info=video_info,
                audio_analysis=audio_analysis,
                scene_analysis=scene_analysis,
                target_duration=self.config['target_short_duration'],
                max_total_segments=50  # Smaller set for vision analysis
            )
            
            # Select diverse segments for vision analysis
            if len(temp_segments) > 20:
                # Sample segments across the video timeline
                sorted_segments = sorted(temp_segments, key=lambda x: x['start_time'])
                step = len(sorted_segments) // 20
                temp_segments = [sorted_segments[i] for i in range(0, len(sorted_segments), step)][:20]
            
            return temp_segments
            
        except Exception as e:
            self.logger.error(f"Error generating temp segments for vision: {e}")
            return []
    
    async def process_video(self, video_path: str, video_id: str = None, uploaded_video_id: str = None, user_prompt: str = None, llm_provider: str = "ollama", template: str = None, intro_outro_config: Dict = None) -> Dict:
        """
        Process a single video file to generate shorts.
        
        Args:
            video_path: Path to input video file
            video_id: Video ID for database caching (optional, use this OR uploaded_video_id)
            uploaded_video_id: Uploaded video ID for database caching (optional, use this OR video_id)
            user_prompt: User prompt for theme-specific content (e.g., "climax scenes", "comedy shorts")
            llm_provider: LLM provider to use for analysis
            template: Template to apply to final clips (e.g., "podcast")
            intro_outro_config: Configuration for intro/outro processing with keys:
                - intro_url: Wasabi path to intro video
                - outro_url: Wasabi path to outro video  
                - storage_bucket: Storage bucket for downloading files
            
        Returns:
            Processing results dictionary
        """
        # Get effective video ID for logging (video_id takes precedence)
        effective_id = video_id if video_id else uploaded_video_id
        id_label = f"video_id: {video_id}" if video_id else (f"uploaded_video_id: {uploaded_video_id}" if uploaded_video_id else "no ID")
        start_time = datetime.now()
        video_name = Path(video_path).stem
        
        # Initialize video analysis logger
        analysis_logger = VideoAnalysisLogger(video_name)
        
        # Check if linear cut mode is enabled
        linear_cut_mode = self.config.get('linear_cut', False)
        
        # Check if object-based generation should be used
        object_based_generation = False
        celebrity_index_path = self.config.get('celebrity_index_path')
        if celebrity_index_path and Path(celebrity_index_path).exists():
            try:
                import json
                with open(celebrity_index_path, 'r') as f:
                    data = json.load(f)
                
                objects = data.get('objects', [])
                object_reference_segments = []
                for obj in objects:
                    for seg in obj.get('segments', []):
                        if seg.get('reference_match') and isinstance(seg.get('reference_match'), dict):
                            object_reference_segments.append(seg)
                
                if object_reference_segments:
                    object_based_generation = True
            except Exception as e:
                self.logger.warning(f"Could not check for object reference segments in logging: {e}")
        
        if linear_cut_mode:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎬 LINEAR CUT MODE - Processing video: {video_name} ({id_label})")
            self.logger.info(f"{'='*80}\n")
        elif user_prompt:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎬 Processing video with USER PROMPT: {video_name} ({id_label})")
            self.logger.info(f"   💬 User Prompt: '{user_prompt}'")
            self.logger.info(f"{'='*80}\n")
        elif object_based_generation:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎬 OBJECT-BASED GENERATION - Processing video: {video_name} ({id_label})")
            self.logger.info(f"   📦 Using object detection results from: {Path(celebrity_index_path).name}")
            self.logger.info(f"{'='*80}\n")
        else:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"🎬 Processing video: {video_name} ({id_label})")
            self.logger.info(f"{'='*80}\n")
        
        try:
            # Get video info for logger
            video_info = self.video_processor.get_video_info(video_path)
            analysis_logger.log_video_metadata(video_info, video_path)
            analysis_logger.log_user_request(user_prompt, self.config)
            
            # LINEAR CUT MODE: Skip AI analysis and generate simple linear segments
            if linear_cut_mode:
                self.logger.info("LINEAR CUT MODE: Skipping AI analysis, generating linear segments...")
                
                # Step 1: Extract audio for subtitle generation only (if enabled)
                audio_analysis = None
                if self.config.get('subtitle_overlay', False):
                    self.logger.info("Step 1: Analyzing audio with Whisper for subtitles...")
                    audio_analysis = await self._analyze_audio(video_path, video_id, uploaded_video_id)
                    analysis_logger.log_audio_analysis(audio_analysis)
                else:
                    self.logger.info("Step 1: Skipping audio analysis (subtitles disabled)")
                    # Create minimal audio analysis for compatibility
                    audio_analysis = {
                        'transcription': {'text': '', 'segments': [], 'language': 'en'},
                        'zoom_analysis': {'recommended_cuts': []},
                        'audio_path': None
                    }
                
                # Step 2: Generate linear segments
                self.logger.info("Step 2: Generating linear segments...")
                no_of_videos = self.config.get('no_of_videos', 5)
                linear_duration = self.config.get('linear_duration', 60)
                final_segments = await self._generate_linear_segments(
                    video_path,
                    no_of_videos,
                    linear_duration
                )
                
                if not final_segments:
                    raise ValueError("Failed to generate linear segments")
                
                self.logger.info(f"Generated {len(final_segments)} linear segments")
                
                # Create content_analysis_results and scene_analysis for compatibility
                content_analysis_results = {
                    'status': 'success',
                    'mode': 'linear_cut',
                    'segments': final_segments
                }
                
                # Create minimal scene_analysis for compatibility (not used in linear mode)
                scene_analysis = {
                    'scene_breaks': [],
                    'audio_breaks': [],
                    'combined_breaks': [],
                    'preroll_end': 0.0
                }
                
            # NORMAL AI-POWERED MODE: Full analysis pipeline
            else:
                # Step 1: Extract and analyze audio
                self.logger.info("Step 1: Analyzing audio with Whisper...")
                audio_analysis = await self._analyze_audio(video_path, video_id, uploaded_video_id)
                analysis_logger.log_audio_analysis(audio_analysis)
                
                # Step 2: Detect scenes and segments
                self.logger.info("Step 2: Detecting scenes and segments...")
                scene_analysis = await self._detect_scenes(video_path, audio_analysis)
                
                # Step 3: Vision analysis on comprehensive candidates (if enabled)
                vision_analysis = None
                vision_enabled = self.config.get('vision_analysis_enabled', True)
                self.logger.info(f"Vision analysis enabled flag: {vision_enabled}")
                self.logger.info(f"Ollama client available: {self.ollama_client is not None}, vision_processor initialized: {self.vision_processor is not None}")
                
                # Step 4: Enhanced content analysis with user prompt support
                # Check if we have object detection results that should trigger object-based generation
                celebrity_index_path = self.config.get('celebrity_index_path')
                object_based_generation = False
                
                if celebrity_index_path and Path(celebrity_index_path).exists():
                    try:
                        import json
                        with open(celebrity_index_path, 'r') as f:
                            data = json.load(f)
                        
                        # Check if there are objects with reference_match.score
                        objects = data.get('objects', [])
                        object_reference_segments = []
                        for obj in objects:
                            for seg in obj.get('segments', []):
                                if seg.get('reference_match') and isinstance(seg.get('reference_match'), dict):
                                    object_reference_segments.append(seg)
                        
                        if object_reference_segments:
                            object_based_generation = True
                            self.logger.info(f"Detected {len(object_reference_segments)} object reference segments - using object-based generation")
                    except Exception as e:
                        self.logger.warning(f"Could not check for object reference segments: {e}")
                
                if user_prompt:
                    self.logger.info(f"Step 4: Performing prompt-based analysis for: '{user_prompt}'")
                    content_analysis_results = await self._analyze_with_user_prompt(
                        user_prompt, video_path, audio_analysis, scene_analysis, analysis_logger, llm_provider
                    )
                elif object_based_generation:
                    self.logger.info("Step 4: Performing object-based analysis (object detection results available)")
                    content_analysis_results = await self._analyze_comprehensive_content(
                        video_path, audio_analysis, scene_analysis, analysis_logger
                    )
                else:
                    self.logger.info("Step 4: Performing comprehensive analysis (no user prompt or object detection)")
                    content_analysis_results = await self._analyze_comprehensive_content(
                        video_path, audio_analysis, scene_analysis, analysis_logger
                    )
                
                # Extract final segments from analysis results
                if content_analysis_results['status'] == 'success':
                    final_segments = content_analysis_results['segments']
                    self.logger.info(f"Content analysis successful: {len(final_segments)} segments selected")
                    # If analysis returned zero segments, generate fallback segments to guarantee output
                    if not final_segments:
                        self.logger.warning("Analysis returned 0 segments; generating fallback segments to ensure output")
                        final_segments = await self._generate_fallback_segments(video_path, audio_analysis)
                else:
                    self.logger.warning("Content analysis failed, falling back to basic segmentation")
                    final_segments = await self._generate_fallback_segments(video_path, audio_analysis)
            
            # Step 5: Apply smart zoom and process videos
            self.logger.info("Step 5: Applying smart zoom and processing...")
            processed_shorts = await self._process_shorts_with_smart_zoom(
                video_path, 
                final_segments, 
                audio_analysis,
                user_prompt=user_prompt,
                template=template,
                intro_outro_config=intro_outro_config
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Verify output files exist
            output_files = [short['output_path'] for short in processed_shorts]
            self.logger.info(f"Checking {len(output_files)} output files:")
            for file_path in output_files:
                path = Path(file_path)
                file_exists = path.exists()
                file_size = path.stat().st_size if file_exists else 0
                self.logger.info(f"  Output file {path}: exists={file_exists}, size={file_size} bytes")
            
            # Check output directory contents
            try:
                output_dir_contents = list(self.output_dir.glob('*'))
                self.logger.info(f"Output directory ({self.output_dir}) contains {len(output_dir_contents)} files:")
                for item in output_dir_contents[:10]:  # Show first 10 items
                    file_size = item.stat().st_size if item.is_file() else 0
                    self.logger.info(f"  {item.name} - {'directory' if item.is_dir() else 'file'} - {file_size} bytes")
                if len(output_dir_contents) > 10:
                    self.logger.info(f"  ... and {len(output_dir_contents) - 10} more items")
            except Exception as e:
                self.logger.error(f"Error listing output directory: {e}")
            
            # Update statistics
            self.stats['videos_processed'] += 1
            self.stats['shorts_generated'] += len(processed_shorts)
            self.stats['total_processing_time'] += processing_time
            
            # Log final outputs to analysis logger
            analysis_logger.log_final_outputs(output_files, processing_time)
            
            # Save comprehensive analysis log
            log_file_path = analysis_logger.save_log()
            if log_file_path:
                self.logger.info(f"Video analysis log saved: {log_file_path}")
            
            results = {
                'video_name': video_name,
                'input_path': video_path,
                'user_prompt': user_prompt,
                'shorts_generated': len(processed_shorts),
                'output_files': [short['output_path'] for short in processed_shorts],
                'processing_time': processing_time,
                'audio_analysis': audio_analysis,
                'content_analysis_results': content_analysis_results,
                'scene_analysis': scene_analysis,
                'shorts_details': processed_shorts,
                'analysis_log': log_file_path
            }
            
            self.logger.info(f"Successfully processed {video_name}: {len(processed_shorts)} shorts generated")
            
            # Clean results for JSON serialization to prevent int64 errors
            cleaned_results = clean_for_serialization(results)
            return cleaned_results
            
        except Exception as e:
            # DEFENSIVE FIX: Handle theme_templates corruption at the highest level
            if "'list' object has no attribute 'items'" in str(e):
                self.logger.error(f"DETECTED theme_templates corruption in process_video: {e}")
                self.logger.warning("Attempting global recovery from theme_templates corruption...")
                
                # Force recovery by accessing get_supported_themes
                try:
                    if hasattr(self, 'content_analyzer') and self.content_analyzer:
                        self.content_analyzer.get_supported_themes()
                        self.logger.info("Global recovery attempt completed")
                except Exception as recovery_error:
                    self.logger.error(f"Global recovery failed: {recovery_error}")
            
            self.logger.error(f"Error processing video {video_name}: {e}")
            # Still try to save the analysis log even on error
            try:
                if 'analysis_logger' in locals():
                    analysis_logger.log_final_outputs([], (datetime.now() - start_time).total_seconds())
                    analysis_logger.save_log()
            except:
                pass
            self.stats['errors'].append({
                'video': video_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    async def _analyze_audio(self, video_path: str, video_id: str = None, uploaded_video_id: str = None) -> Dict:
        """Analyze audio using Whisper."""
        # Extract audio
        audio_path = self.whisper_analyzer.extract_audio(video_path)
        
        # Transcribe audio with video_id or uploaded_video_id for caching
        transcription = self.whisper_analyzer.transcribe(audio_path, video_id=video_id, uploaded_video_id=uploaded_video_id)
        
        # Analyze for zoom decisions
        zoom_analysis = self.whisper_analyzer.analyze_for_zoom_decisions(transcription)
        
        return {
            'transcription': transcription,
            'zoom_analysis': zoom_analysis,
            'audio_path': audio_path
        }
    
    async def _analyze_content_with_ollama(self, audio_analysis: Dict) -> Dict:
        """Analyze content using Ollama AI."""
        transcription = audio_analysis['transcription']
        
        # Generate comprehensive analysis
        analysis = await self.ollama_client.generate_comprehensive_analysis(transcription)
        
        return analysis
    
    async def _analyze_content_with_dual_modal(self, audio_analysis: Dict, vision_analysis: Dict) -> Dict:
        """Analyze content using both audio transcription and vision analysis."""
        try:
            self.logger.info("Performing dual-modal analysis with audio and vision data")
            
            # Get transcription
            transcription = audio_analysis['transcription']
            
            # Use Ollama client's dual-modal analysis method
            if hasattr(self.ollama_client, 'analyze_with_vision_context'):
                analysis = await self.ollama_client.analyze_with_vision_context(
                    transcription, vision_analysis
                )
                
                # Enhance the analysis with segment-specific insights
                enhanced_analysis = await self._enhance_dual_modal_analysis(
                    analysis, audio_analysis, vision_analysis
                )
                self.logger.info("Dual-modal analysis completed successfully")
            
                return enhanced_analysis
            else:
                self.logger.warning("Ollama client does not support dual-modal analysis, falling back to audio-only")
                self.logger.info("Using audio-only analysis as fallback")
                # Fallback to audio-only analysis if dual-modal not available
                self.logger.warning("Dual-modal analysis not available, falling back to audio-only")
                return await self._analyze_content_with_ollama(audio_analysis)
                
        except Exception as e:
            self.logger.error(f"Error in dual-modal analysis: {e}")
            # Fallback to audio-only analysis
            return await self._analyze_content_with_ollama(audio_analysis)
    
    async def _enhance_dual_modal_analysis(self, 
                                         base_analysis: Dict, 
                                         audio_analysis: Dict, 
                                         vision_analysis: Dict) -> Dict:
        """Enhance the dual-modal analysis with additional segment insights."""
        try:
            enhanced = base_analysis.copy()
            
            # Add vision-enhanced segment scoring
            if 'segments' in enhanced and 'segments' in vision_analysis:
                vision_segments = {
                    seg['start_time']: seg for seg in vision_analysis.get('segments', [])
                }
                
                for segment in enhanced.get('segments', []):
                    start_time = segment.get('start_time', 0)
                    
                    # Find matching vision segment
                    vision_seg = None
                    for v_start, v_seg in vision_segments.items():
                        if abs(v_start - start_time) < 5.0:  # Within 5 seconds
                            vision_seg = v_seg
                            break
                    
                    if vision_seg:
                        # Enhance segment with vision data
                        segment['vision_score'] = vision_seg.get('visual_score', 0.5)
                        segment['visual_interest'] = vision_seg.get('visual_interest', 0.5)
                        segment['scene_type'] = vision_seg.get('scene_type', 'unknown')
                        segment['people_visible'] = vision_seg.get('people_count', 0) > 0
                        
                        # Adjust combined score based on audio-visual alignment
                        audio_score = segment.get('audio_score', 0.5)
                        vision_score = segment.get('vision_score', 0.5)
                        alignment_bonus = vision_seg.get('audiovisual_alignment', 0.5)
                        
                        # Calculate enhanced combined score
                        combined_score = (audio_score * 0.6 + vision_score * 0.4) * (1 + alignment_bonus * 0.2)
                        segment['enhanced_combined_score'] = min(1.0, combined_score)
            
            # Add overall dual-modal metrics
            enhanced['analysis_type'] = 'dual_modal'
            enhanced['has_vision_context'] = True
            enhanced['vision_segments_analyzed'] = len(vision_analysis.get('segments', []))
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing dual-modal analysis: {e}")
            return base_analysis
    
    async def _detect_scenes(self, video_path: str, audio_analysis: Dict) -> Dict:
        """Detect scenes and segments in the video."""
        # Use scene detector to find natural breaks
        scene_breaks = self.scene_detector.detect_scenes(video_path)
        
        # Combine with audio analysis
        transcription = audio_analysis['transcription']
        audio_breaks = audio_analysis['zoom_analysis']['recommended_cuts']
        
        # Detect pre-roll content from audio analysis
        preroll_end = 0.0
        if 'zoom_analysis' in audio_analysis and 'likely_preroll' in audio_analysis['zoom_analysis']:
            likely_preroll = audio_analysis['zoom_analysis']['likely_preroll']
            if likely_preroll:
                preroll_end = max(segment['end'] for segment in likely_preroll)
                self.logger.info(f"Detected pre-roll content ending at {preroll_end:.2f}s")
        
        # Merge scene and audio breaks with pre-roll filtering
        combined_breaks = self.scene_detector.merge_breaks_with_filtering(
            scene_breaks, 
            audio_breaks,
            preroll_end=preroll_end,
            audio_analysis=audio_analysis
        )
        
        return {
            'scene_breaks': scene_breaks,
            'audio_breaks': audio_breaks,
            'combined_breaks': combined_breaks,
            'preroll_end': preroll_end
        }
    
    async def _generate_all_candidate_segments(self, 
                                         video_path: str, 
                                         audio_analysis: Dict, 
                                         scene_analysis: Dict) -> List[Dict]:
        """Generate ALL potential candidate segments for AI analysis (no pre-filtering)."""
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Generate ALL segments using content analyzer (with expanded target)
            all_segments = self.content_analyzer.identify_short_segments(
                video_info=video_info,
                audio_analysis=audio_analysis,
                ollama_analysis=None,  # Will be added later in the process
                scene_analysis=scene_analysis,
                target_duration=self.config['target_short_duration'],
                max_shorts=self.config['max_shorts_per_video'] * 4  # Get 4x more candidates for AI analysis
            )
            
            self.logger.info(f"Generated {len(all_segments)} candidate segments for AI analysis")
            return all_segments
            
        except Exception as e:
            self.logger.error(f"Error generating candidate segments: {e}")
            return []

    async def _generate_initial_segments(self, 
                                       video_path: str, 
                                       audio_analysis: Dict, 
                                       scene_analysis: Dict) -> List[Dict]:
        """Generate initial segments based on scene and audio analysis (DEPRECATED - use _generate_all_candidate_segments)."""
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Generate segments using content analyzer
            segments = self.content_analyzer.identify_short_segments(
                video_info=video_info,
                audio_analysis=audio_analysis,
                ollama_analysis=None,  # Will be added later
                scene_analysis=scene_analysis,
                target_duration=self.config['target_short_duration'],
                max_shorts=self.config['max_shorts_per_video']
            )
            
            self.logger.info(f"Generated {len(segments)} initial segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error generating initial segments: {e}")
            return []
    
    async def _analyze_visual_content(self, 
                                    video_path: str, 
                                    segments: List[Dict], 
                                    audio_analysis: Dict) -> Dict:
        """Analyze visual content of video segments."""
        try:
            if not self.vision_processor:
                self.logger.warning("Vision processor not available")
                return {}
            
            # Analyze visual content for each segment
            # Log the Vision API request payload for verification
            self.logger.info(f"Vision API request: video_path={video_path}, segments={segments}, audio_analysis_keys={list(audio_analysis.keys())}")
            vision_results = await self.vision_processor.analyze_video_segments(
                video_path, segments, audio_analysis
            )
            
            self.logger.info(f"Completed vision analysis for {len(segments)} segments")
            # Log the results for debugging
            self.logger.debug(f"Vision analysis results: {vision_results}")

            return vision_results
            
        except Exception as e:
            self.logger.error(f"Error in visual content analysis: {e}")
            return {}
    
    async def _select_final_segments(self, 
                                   video_path: str,
                                   audio_analysis: Dict,
                                   ollama_analysis: Optional[Dict],
                                   scene_analysis: Dict,
                                   vision_analysis: Optional[Dict],
                                   all_candidate_segments: List[Dict] = None) -> List[Dict]:
        """Select final segments based on all analysis results from ALL candidates."""
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            
            # Use provided candidates or generate them if not provided (fallback)
            if all_candidate_segments:
                self.logger.info(f"Selecting from {len(all_candidate_segments)} AI-analyzed candidate segments")
                candidate_segments = all_candidate_segments
            else:
                self.logger.warning("No candidate segments provided, generating them now (less optimal)")
                candidate_segments = await self._generate_all_candidate_segments(video_path, audio_analysis, scene_analysis)
            
            # Use content analyzer to select final segments with AI-enhanced criteria
            final_segments = self.content_analyzer.select_final_segments(
                video_info=video_info,
                audio_analysis=audio_analysis,
                ollama_analysis=ollama_analysis,
                scene_analysis=scene_analysis,
                vision_analysis=vision_analysis,
                target_duration=self.config['target_short_duration'],
                max_shorts=self.config['max_shorts_per_video'],
                quality_threshold=self.config['quality_threshold'],
                candidate_segments=candidate_segments  # Pass all AI-analyzed candidates
            )
            
            self.logger.info(f"Selected {len(final_segments)} final segments from {len(candidate_segments)} AI-analyzed candidates")
            return final_segments
            
        except Exception as e:
            self.logger.error(f"Error selecting final segments: {e}")
            # Fallback to simple segment generation
            return await self._generate_fallback_segments(video_path, audio_analysis)
    
    async def _generate_fallback_segments(self, 
                                        video_path: str, 
                                        audio_analysis: Dict) -> List[Dict]:
        """Generate fallback segments when advanced analysis fails.
        Always returns at least one segment, adapting to short videos.
        """
        try:
            video_info = self.video_processor.get_video_info(video_path)
            duration = max(0, float(video_info.get('duration', 0)) or 0)

            segments: List[Dict] = []

            if duration <= 0:
                # Unknown duration: produce a single 30s segment starting at 0
                segments.append({
                    'start_time': 0.0,
                    'end_time': 30.0,
                    'quality_score': 0.7,
                    'engagement_score': 0.6,
                    'reason': 'fallback_segment_unknown_duration'
                })
                self.logger.info(f"Generated 1 fallback segment (unknown duration)")
                return segments

            # Choose a target clip length based on duration
            # Try to keep around 45-60s but adjust to available length
            target_len = 60.0
            margin = min(10.0, duration * 0.05)  # small head/tail margin relative to video

            if duration <= 45.0:
                # Very short videos: use the whole content
                start_time = 0.0
                end_time = max(5.0, duration - 0.1)  # ensure > 0
                segments.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'quality_score': 0.7,
                    'engagement_score': 0.6,
                    'reason': 'fallback_segment_short_video'
                })
                self.logger.info(f"Generated 1 fallback segment for short video ({duration:.2f}s)")
                return segments

            # For medium videos, create 1–N overlapping segments
            start = max(0.0, margin)
            end_limit = max(0.0, duration - margin)

            # Step allows overlap (50%) to increase chances of good content
            step = target_len * 0.5
            while start + 15.0 < end_limit:  # ensure at least 15s segments
                end_time = min(end_limit, start + target_len)
                if end_time - start >= 15.0:
                    segments.append({
                        'start_time': float(start),
                        'end_time': float(end_time),
                        'quality_score': 0.7,
                        'engagement_score': 0.6,
                        'reason': 'fallback_segment'
                    })
                start += step
                # Limit to configured max shorts
                if len(segments) >= self.config.get('max_shorts_per_video', 10):
                    break

            if not segments:
                # As a last resort, create a single centered segment
                usable = max(15.0, duration - 2 * margin)
                clip_len = min(target_len, usable)
                center = duration / 2.0
                start_time = max(0.0, center - clip_len / 2.0)
                end_time = min(duration, start_time + clip_len)
                segments.append({
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'quality_score': 0.7,
                    'engagement_score': 0.6,
                    'reason': 'fallback_segment_centered'
                })

            self.logger.info(f"Generated {len(segments)} fallback segment(s)")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error generating fallback segments: {e}")
            # Absolute minimum: return a single 30s segment
            return [{
                'start_time': 0.0,
                'end_time': 30.0,
                'quality_score': 0.7,
                'engagement_score': 0.6,
                'reason': 'fallback_segment_error'
            }]
    
    async def _generate_linear_segments(self,
                                       video_path: str,
                                       no_of_videos: int,
                                       linear_duration: int) -> List[Dict]:
        """Generate linear segments by dividing video into equal parts.
        
        Args:
            video_path: Path to video file
            no_of_videos: Number of segments to create
            linear_duration: Duration of each segment in seconds
            
        Returns:
            List of segment dictionaries with start, end, and part number
        """
        try:
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            total_duration = video_info.get('duration', 0)
            
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"LINEAR CUT MODE - Generating {no_of_videos} segments of {linear_duration}s each")
            self.logger.info(f"Total video duration: {total_duration:.2f}s")
            self.logger.info(f"{'='*80}\n")
            
            segments = []
            
            for i in range(no_of_videos):
                start_time = i * linear_duration
                end_time = min((i + 1) * linear_duration, total_duration)
                
                # Stop if we've reached the end of the video
                if start_time >= total_duration:
                    self.logger.warning(f"Segment {i+1} starts at {start_time}s but video ends at {total_duration}s - skipping")
                    break
                
                segment = {
                    'start_time': float(start_time),
                    'end_time': float(end_time),
                    'duration': float(end_time - start_time),
                    'part_number': i + 1,
                    'total_parts': no_of_videos,
                    # keep compatibility fields used by downstream code
                    'quality_score': 1.0,  # Max quality for linear cuts
                    'reason': f'Linear segment {i+1} of {no_of_videos}'
                }
                
                segments.append(segment)
                self.logger.info(f"  Segment {i+1}: {start_time:.2f}s - {end_time:.2f}s ({segment['duration']:.2f}s)")
            
            self.logger.info(f"\nGenerated {len(segments)} linear segments")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error generating linear segments: {e}")
            return []
    
    async def _generate_shorts(self, 
                             video_path: str,
                             audio_analysis: Dict,
                             ollama_analysis: Optional[Dict],
                             scene_analysis: Dict) -> List[Dict]:
        """Generate short video segments."""
        # Get video info
        video_info = self.video_processor.get_video_info(video_path)
        
        # Determine optimal segments
        segments = self.content_analyzer.identify_short_segments(
            video_info=video_info,
            audio_analysis=audio_analysis,
            ollama_analysis=ollama_analysis,
            scene_analysis=scene_analysis,
            target_duration=self.config['target_short_duration'],
            max_shorts=self.config['max_shorts_per_video']
        )
        
        # Filter segments by quality
        quality_threshold = self.config['quality_threshold']
        filtered_segments = [
            seg for seg in segments 
            if seg.get('quality_score', 0) >= quality_threshold
        ]
        
        return filtered_segments
    
    async def _process_shorts_with_smart_zoom(self, 
                                            video_path: str,
                                            shorts: List[Dict],
                                            audio_analysis: Dict,
                                            user_prompt: str = None,
                                            template: str = None,
                                            intro_outro_config: Dict = None) -> List[Dict]:
        """Process shorts with smart zoom applied.
        Guarantees an output by falling back to standard zoom or basic conversion if needed.
        Template processing (e.g., podcast) is applied AFTER clipping for better performance.
        """
        processed_shorts = []
        
        # Store intro/outro configuration for use in processing loop
        self._intro_outro_config = intro_outro_config or {}
        
        # Get original video info for duration logging
        video_info = self.video_processor.get_video_info(video_path)
        original_duration = video_info.get('duration', 0)
        self.logger.info(f"📹 ORIGINAL VIDEO INFO: Duration={original_duration:.2f}s, Path={video_path}")
        
        # Check if final_video_ids are provided in the config
        final_video_ids = self.config.get('final_video_ids', [])
        self.logger.info(f"Using final_video_ids from config: {final_video_ids}, type: {type(final_video_ids)}")
        
        for i, short in enumerate(shorts):
            # Determine canvas type and aspect ratio
            canvas_type = self.config.get('canvas_type', 'shorts')
            
            # Get input video name for the filename
            input_video_name = Path(video_path).stem
            
            # Use final_video_ids if available, otherwise use default naming
            if final_video_ids and i < len(final_video_ids) and final_video_ids[i]:
                vid_id = str(final_video_ids[i]) if not isinstance(final_video_ids[i], str) else final_video_ids[i]
                short_name = f"{input_video_name}_{vid_id}_{canvas_type}"
                self.logger.info(f"Using final_video_id {vid_id} for output: {short_name}")
            else:
                short_name = f"{input_video_name}_short_{i+1:02d}_{canvas_type}"
            
            output_path = self.output_dir / f"{short_name}.mp4"
            
            # Log detailed segment information for verification
            start_time = short['start_time']
            end_time = short['end_time']
            segment_duration = end_time - start_time
            start_percentage = (start_time / original_duration * 100) if original_duration > 0 else 0
            end_percentage = (end_time / original_duration * 100) if original_duration > 0 else 0
            
            self.logger.info(f"🎬 SEGMENT #{i+1} EXTRACTION:")
            self.logger.info(f"   📍 Source Position: {start_time:.2f}s - {end_time:.2f}s ({start_percentage:.1f}% - {end_percentage:.1f}% of original)")
            self.logger.info(f"   ⏱️  Segment Duration: {segment_duration:.2f}s")
            self.logger.info(f"   📂 Original Video: {original_duration:.2f}s → Extracting {segment_duration:.2f}s ({segment_duration/original_duration*100:.1f}%)")
            
            # Extract segment from original video
            segment_path = self.temp_dir / f"{short_name}_segment.mp4"
            
            self.video_processor.extract_segment(
                video_path,
                str(segment_path),
                start_time,
                end_time
            )
            
            # Verify extracted segment duration
            try:
                extracted_info = self.video_processor.get_video_info(str(segment_path))
                extracted_duration = extracted_info.get('duration', 0)
                duration_difference = abs(extracted_duration - segment_duration)
                
                self.logger.info(f"✅ EXTRACTION VERIFIED:")
                self.logger.info(f"   📊 Expected Duration: {segment_duration:.2f}s")
                self.logger.info(f"   📊 Actual Duration: {extracted_duration:.2f}s")
                self.logger.info(f"   📊 Difference: {duration_difference:.2f}s ({duration_difference/segment_duration*100:.1f}% error)")
                
                if duration_difference > 1.0:  # More than 1 second difference
                    self.logger.warning(f"⚠️ Duration mismatch detected! Expected {segment_duration:.2f}s, got {extracted_duration:.2f}s")
                    
            except Exception as e:
                self.logger.warning(f"Could not verify extracted segment duration: {e}")
            
            # Apply intro/outro processing to extracted segment BEFORE smart zoom processing
            # This ensures intro/outro works on the properly extracted segment
            if hasattr(self, '_intro_outro_config') and self._intro_outro_config:
                intro_url = self._intro_outro_config.get('intro_url')
                outro_url = self._intro_outro_config.get('outro_url')
                storage_bucket = self._intro_outro_config.get('storage_bucket')
                
                if intro_url or outro_url:
                    self.logger.info(f"🎬 Applying intro/outro to extracted segment #{i+1}")
                    
                    # Verify segment file exists and has content before processing
                    if not segment_path.exists():
                        self.logger.error(f"❌ Segment file does not exist: {segment_path}")
                    elif segment_path.stat().st_size == 0:
                        self.logger.error(f"❌ Segment file is empty: {segment_path}")
                    else:
                        original_size = segment_path.stat().st_size
                        self.logger.info(f"📊 Original segment size: {original_size} bytes")
                        
                        try:
                            from src.utils.intro_outro_handler import IntroOutroHandler
                            
                            handler = IntroOutroHandler(storage_bucket=storage_bucket)
                            
                            # Process the extracted segment with intro/outro
                            final_segment_path = handler.process_clip_with_intro_outro(
                                clip_path=str(segment_path),
                                intro_url=intro_url,
                                outro_url=outro_url,
                                storage_bucket=storage_bucket
                            )
                            
                            # Verify the processed file and its size
                            if segment_path.exists():
                                new_size = segment_path.stat().st_size
                                self.logger.info(f"📊 After intro/outro size: {new_size} bytes")
                                
                                if new_size > original_size:
                                    self.logger.info(f"✅ Successfully applied intro/outro to segment #{i+1} (size increased from {original_size} to {new_size} bytes)")
                                else:
                                    self.logger.warning(f"⚠️ Intro/outro processing may have failed - size did not increase: {original_size} → {new_size} bytes")
                            else:
                                self.logger.error(f"❌ Segment file disappeared after intro/outro processing: {segment_path}")
                            
                            handler.cleanup()
                            
                        except Exception as e:
                            self.logger.error(f"❌ Error applying intro/outro to segment #{i+1}: {e}")
                            # Continue with original segment if intro/outro fails
            
            self.logger.info(f"🎯 Processing segment #{i+1} with smart zoom...")
            
            canvas_type = self.config.get('canvas_type', 'shorts')
            self.logger.info(f"Using canvas_type from config: {canvas_type}")
            canvas_aspect_ratio = (9, 16) if canvas_type == 'shorts' else (16, 9)
            content_aspect_ratio = self.config.get('aspect_ratio', canvas_aspect_ratio)
            self.logger.info(f"Using content_aspect_ratio from config: {content_aspect_ratio}")
            is_vertical = canvas_aspect_ratio[0] < canvas_aspect_ratio[1]
            
            # Check if smart zoom should be disabled for linear cut mode
            linear_cut = self.config.get('linear_cut', False)
            disable_smart_zoom_linear_cut = self.config.get('disable_smart_zoom_linear_cut', True)
            skip_smart_zoom = linear_cut and disable_smart_zoom_linear_cut
            
            # Try object-aware zoom first if available
            file_verified = False
            smart_zoom_results = {'processing_time': 0.0}
            
            if skip_smart_zoom:
                # ⚡ LINEAR CUT FAST PATH: Skip smart zoom for maximum speed
                self.logger.info(f"⚡ LINEAR CUT OPTIMIZATION: Skipping smart zoom for maximum speed")
                self.logger.info(f"   Using fast aspect ratio conversion (7-10x faster)")
                
                try:
                    # Simple center crop + resize without any AI analysis
                    import cv2
                    import numpy as np
                    
                    # Open video
                    cap = cv2.VideoCapture(str(segment_path))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Set up output dimensions
                    if canvas_aspect_ratio[0] == 9 and canvas_aspect_ratio[1] == 16:
                        output_width, output_height = 1080, 1920
                    else:
                        base_height = 1080
                        output_height = base_height
                        output_width = int(base_height * (canvas_aspect_ratio[0] / canvas_aspect_ratio[1]))
                    
                    # Set up video writer
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    temp_output = str(output_path).replace('.mp4', '_temp.mp4')
                    out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, output_height))
                    
                    target_aspect = content_aspect_ratio[0] / content_aspect_ratio[1]
                    frames_processed = 0
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_h, frame_w = frame.shape[:2]
                        frame_aspect = frame_w / frame_h
                        
                        # Center crop to target aspect ratio
                        if frame_aspect > target_aspect:
                            new_width = int(frame_h * target_aspect)
                            start_x = (frame_w - new_width) // 2
                            cropped = frame[:, start_x:start_x + new_width]
                        else:
                            new_height = int(frame_w / target_aspect)
                            start_y = (frame_h - new_height) // 2
                            cropped = frame[start_y:start_y + new_height, :]
                        
                        # Resize to output dimensions
                        resized = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                        out.write(resized)
                        frames_processed += 1
                    
                    cap.release()
                    out.release()
                    
                    # Re-encode with audio using ffmpeg
                    import subprocess
                    final_output = str(output_path)
                    cmd = [
                        'ffmpeg', '-y', '-i', temp_output, '-i', str(segment_path),
                        '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                        '-c:a', 'aac', '-b:a', '128k',
                        '-map', '0:v:0', '-map', '1:a:0?',
                        '-movflags', '+faststart',
                        final_output
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Clean up temp file
                    import os
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    
                    file_verified = output_path.exists() and output_path.stat().st_size > 0
                    smart_zoom_results['processing_time'] = 0.5  # Minimal time
                    smart_zoom_results['optimization'] = 'fast_aspect_ratio_conversion'
                    
                    self.logger.info(f"⚡ Fast conversion completed: {frames_processed} frames")
                    
                except Exception as e:
                    self.logger.error(f"❌ Fast conversion failed: {e}, falling back to smart zoom")
                    skip_smart_zoom = False
            
            if not skip_smart_zoom and self.config['smart_zoom_enabled']:
                if (user_prompt and 
                    self.object_aware_zoom_processor and 
                    self.config.get('enable_object_detection', True)):
                    try:
                        self.logger.info(f"Using object-aware zoom processing for prompt: '{user_prompt}'")
                        smart_zoom_results = await self.object_aware_zoom_processor.process_video_with_objects(
                            video_path=str(segment_path),
                            output_path=str(output_path),
                            user_prompt=user_prompt,
                            audio_analysis=audio_analysis,
                            target_aspect_ratio=content_aspect_ratio,
                            is_vertical=is_vertical,
                            canvas_type=canvas_type
                        )
                        file_verified = self._ensure_output_file(output_path)
                    except Exception as e:
                        self.logger.warning(f"Object-aware zoom failed: {e}")
                        file_verified = False
                
                # If object-aware failed or not used, try standard smart zoom
                if not file_verified:
                    self.logger.info("Falling back to standard smart zoom processing")
                    try:
                        smart_zoom_results = self.smart_zoom_processor.process_video(
                            str(segment_path),
                            str(output_path),
                            audio_analysis=audio_analysis,
                            target_aspect_ratio=content_aspect_ratio,
                            is_vertical=is_vertical,
                            canvas_type=canvas_type
                        )
                        file_verified = self._ensure_output_file(output_path)
                    except Exception as e:
                        self.logger.warning(f"Standard smart zoom failed: {e}")
                        file_verified = False
            
            # Absolute last resort: basic conversion without smart zoom
            if not file_verified:
                self.logger.info("Falling back to basic conversion (no smart zoom)")
                try:
                    if is_vertical:
                        self.video_processor.convert_to_vertical(
                            str(segment_path),
                            str(output_path),
                            target_aspect_ratio=canvas_aspect_ratio
                        )
                    else:
                        self.video_processor.convert_to_horizontal(
                            str(segment_path),
                            str(output_path),
                            target_aspect_ratio=canvas_aspect_ratio
                        )
                    file_verified = self._ensure_output_file(output_path)
                except Exception as e:
                    self.logger.error(f"Basic conversion failed: {e}")
                    file_verified = False
            
            # Apply podcast template if specified (after clipping for better performance)
            template_applied = False
            if template == 'podcast' and file_verified:
                self.logger.info(f"Applying podcast template to clipped video: {short_name}")
                
                # Import the robust processing function
                from src.queue_system.tasks import apply_robust_podcast_template_processing
                
                # Create temporary path for podcast processing
                podcast_output_path = self.temp_dir / f"{short_name}_podcast.mp4"
                
                try:
                    # Apply podcast template to the already clipped video
                    result_path = apply_robust_podcast_template_processing(
                        str(output_path),
                        str(podcast_output_path),
                        template
                    )
                    
                    # If processing was successful, replace the original output
                    if result_path == str(podcast_output_path) and podcast_output_path.exists():
                        # Replace the original output with the podcast-processed version
                        import shutil
                        if output_path.exists():
                            output_path.unlink()  # Remove original
                        try:
                            podcast_output_path.rename(output_path)  # Try rename first (faster)
                        except OSError:
                            # If rename fails due to cross-device link, use copy+delete
                            shutil.copy2(podcast_output_path, output_path)
                            podcast_output_path.unlink()
                        
                        template_applied = True
                        self.logger.info(f"Successfully applied podcast template to {short_name}")
                    else:
                        self.logger.warning(f"Podcast template processing failed for {short_name}, using original clip")
                        # Cleanup failed file
                        if podcast_output_path.exists():
                            podcast_output_path.unlink()
                            
                except Exception as e:
                    self.logger.error(f"Error applying podcast template to {short_name}: {e}")
                    # Cleanup failed file
                    if podcast_output_path.exists():
                        podcast_output_path.unlink()
            elif template == 'podcast':
                self.logger.warning(f"Podcast template requested but video processing failed for {short_name}")
            
            # Apply subtitle overlay or kinetic captions if enabled
            subtitle_applied = False
            if (self.config.get('subtitle_overlay', False) or self.config.get('kinetic_captions', True)) and self.subtitle_processor:
                # Get subtitle style ID from config, default to 'cinematic'
                subtitle_style_id = self.config.get('subtitle_overlay_style_id', 'cinematic')
                # Get API-provided style object
                api_style = self.config.get('api_style')
                # Get caption coordinates from config
                caption_x = self.config.get('caption_x', 30)
                caption_y = self.config.get('caption_y', 40)
                canvas_type = self.config.get('canvas_type', 'shorts')
                
                # Get kinetic caption settings
                kinetic_enabled = self.config.get('kinetic_captions', True)  # Enable by default
                kinetic_mode = self.config.get('kinetic_mode', 'karaoke')  # Default to karaoke effect
                
                if api_style:
                    self.logger.info(f"Applying subtitle overlay to {short_name} with API style at position ({caption_x}, {caption_y})")
                else:
                    self.logger.info(f"Applying subtitle overlay to {short_name} with style ID '{subtitle_style_id}' at position ({caption_x}, {caption_y})")
                
                # Create temporary file for video with subtitles
                subtitle_output_path = self.temp_dir / f"{short_name}_with_subtitles.mp4"
                
                # Try kinetic captions first if enabled, fallback to standard if subtitle overlay is enabled
                if kinetic_enabled:
                    self.logger.info(f"Attempting kinetic captions with {kinetic_mode} effect")
                    subtitle_success = self.subtitle_processor.add_kinetic_subtitle_overlay(
                        input_video_path=str(output_path),
                        output_video_path=str(subtitle_output_path),
                        transcription_data=audio_analysis['transcription'],
                        segment_start_time=short['start_time'],
                        segment_end_time=short['end_time'],
                        style_id=subtitle_style_id,
                        api_style=api_style,
                        caption_x=caption_x,
                        caption_y=caption_y,
                        canvas_type=canvas_type,
                        kinetic_mode=kinetic_mode
                    )
                elif self.config.get('subtitle_overlay', False):
                    # Apply standard subtitles only if subtitle overlay is explicitly enabled
                    self.logger.info(f"Applying standard subtitles with style ID '{subtitle_style_id}'")
                    subtitle_success = self.subtitle_processor.add_subtitle_overlay(
                        input_video_path=str(output_path),
                        output_video_path=str(subtitle_output_path),
                        transcription_data=audio_analysis['transcription'],
                        segment_start_time=short['start_time'],
                        segment_end_time=short['end_time'],
                        style_id=subtitle_style_id,
                        api_style=api_style,
                        caption_x=caption_x,
                        caption_y=caption_y,
                        canvas_type=canvas_type
                    )
                else:
                    # Neither kinetic captions nor standard subtitles are enabled
                    self.logger.info("No subtitle processing enabled")
                    subtitle_success = False
                
                if subtitle_success and subtitle_output_path.exists():
                    # Replace the original output with the subtitled version
                    import shutil
                    if output_path.exists():
                        output_path.unlink()  # Remove original
                    try:
                        subtitle_output_path.rename(output_path)  # Try rename first (faster)
                    except OSError:
                        # If rename fails due to cross-device link, use copy+delete
                        shutil.copy2(subtitle_output_path, output_path)
                        subtitle_output_path.unlink()
                    subtitle_applied = True
                    self.logger.info(f"Successfully applied subtitle overlay to {short_name}")
                else:
                    # Cleanup failed subtitle file
                    if subtitle_output_path.exists():
                        subtitle_output_path.unlink()
                    self.logger.warning(f"Failed to apply subtitle overlay to {short_name}, continuing without subtitles")
            elif self.config.get('subtitle_overlay', False):
                self.logger.warning("Subtitle overlay requested but subtitle processor not initialized")
            
            # Verify output file exists
            self.logger.info(f"Checking if output file exists at: {output_path}")
            file_exists = output_path.exists()
            file_size = output_path.stat().st_size if file_exists else 0
            self.logger.info(f"Output file exists: {file_exists}, size: {file_size} bytes")
            
            # Get absolute path for clarity
            abs_output_path = output_path.absolute()
            self.logger.info(f"Absolute output path: {abs_output_path}")
            
            processed_short = {
                'name': short_name,
                'canvas_type_applied': canvas_type,
                'content_aspect_ratio_applied': content_aspect_ratio,
                'output_path': str(output_path),
                'absolute_output_path': str(abs_output_path),
                'output_file_exists': file_exists,
                'output_file_size': file_size,
                'start_time': short['start_time'],
                'end_time': short['end_time'],
                'duration': short['end_time'] - short['start_time'],
                'quality_score': short.get('quality_score', 0.0),
                'engagement_score': short.get('engagement_score', 0.0),
                'smart_zoom_applied': self.config['smart_zoom_enabled'],
                'template_applied': template_applied,
                'template_requested': template,
                'subtitle_overlay_applied': subtitle_applied,
                'subtitle_overlay_requested': self.config.get('subtitle_overlay', False),
                'processing_results': smart_zoom_results
            }
            
            # Generate title and description for this short video
            if self.title_generator:
                try:
                    self.logger.info(f"Generating title and description for {short_name}")
                    
                    # Extract transcription for this segment
                    segment_transcription = self.title_generator.extract_segment_transcription(
                        audio_analysis['transcription'],
                        short['start_time'],
                        short['end_time']
                    )
                    
                    # Generate title and description with variation
                    title_data = await self.title_generator.generate_title_and_description(
                        filename=input_video_name,
                        transcription=segment_transcription,
                        bucket_path=str(abs_output_path),
                        duration=short['end_time'] - short['start_time'],
                        variation_manager=self.variation_manager
                    )
                    
                    # Append part number for linear cut mode if enabled
                    linear_cut_mode = self.config.get('linear_cut', False)
                    append_part_number = self.config.get('append_part_number', True)
                    
                    if linear_cut_mode and append_part_number and 'part_number' in short and 'total_parts' in short:
                        part_num = short['part_number']
                        total_parts = short['total_parts']
                        
                        # Append part number to title
                        original_title = title_data.get('title', 'Video Clip')
                        title_data['title'] = f"{original_title} - Part {part_num} of {total_parts}"
                        
                        self.logger.info(f"Appended part number to title: {title_data['title']}")
                    
                    # Save clip data for later API update (instead of direct database save)
                    if final_video_ids and i < len(final_video_ids) and final_video_ids[i]:
                        clip_id = final_video_ids[i]  # Keep as integer
                        
                        # Store clip data in the processed short for later API update by tasks.py
                        title_data['clip_id'] = clip_id
                        title_data['api_update_ready'] = True
                        title_data['db_saved'] = False  # Will be updated via API later
                        
                        self.logger.info(f"Prepared clip data for API update - clip_id: {clip_id}")
                    else:
                        title_data['clip_id'] = None
                        title_data['api_update_ready'] = False
                        title_data['db_saved'] = False
                    
                    # Add title data to processed short
                    processed_short.update({
                        'title': title_data.get('title', ''),
                        'description': title_data.get('description', ''),
                        'content_type': title_data.get('content_type', 'social_media'),
                        'tags': title_data.get('tags', []),
                        'title_generated': True,
                        'api_update_ready': title_data.get('api_update_ready', False),
                        'db_saved': title_data.get('db_saved', False),
                        'clip_id': title_data.get('clip_id')
                    })
                    
                    self.logger.info(f"Generated title: {title_data.get('title', 'N/A')}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate title for {short_name}: {e}")
                    processed_short.update({
                        'title': '',
                        'description': '',
                        'content_type': 'social_media',
                        'tags': [],
                        'title_generated': False,
                        'db_saved': False,
                        'clip_id': final_video_ids[i] if final_video_ids and i < len(final_video_ids) else None,
                        'title_error': str(e)
                    })
            else:
                # Title generator not available
                processed_short.update({
                    'title': '',
                    'description': '',
                    'content_type': 'social_media',
                    'tags': [],
                    'title_generated': False,
                    'db_saved': False,
                    'clip_id': final_video_ids[i] if final_video_ids and i < len(final_video_ids) else None,
                    'title_error': 'Title generator not initialized'
                })
            
            self.logger.info(f"Generated short video: {short_name}.mp4 with canvas_type={canvas_type}")
            
            processed_shorts.append(processed_short)
            
            # Cleanup temporary segment file
            if segment_path.exists():
                segment_path.unlink()
        
        # Log final summary of all generated clips
        self.logger.info(f"📋 FINAL CLIP GENERATION SUMMARY:")
        self.logger.info(f"   🎬 Original Video: {original_duration:.2f}s")
        self.logger.info(f"   ✂️  Generated Clips: {len(processed_shorts)}")
        
        total_extracted_duration = 0
        for i, short in enumerate(processed_shorts):
            duration = short['duration']
            total_extracted_duration += duration
            percentage = (duration / original_duration * 100) if original_duration > 0 else 0
            self.logger.info(f"   📹 Clip #{i+1}: {short['start_time']:.2f}s - {short['end_time']:.2f}s → {duration:.2f}s ({percentage:.1f}% of original)")
        
        coverage_percentage = (total_extracted_duration / original_duration * 100) if original_duration > 0 else 0
        self.logger.info(f"   📊 Total Extracted: {total_extracted_duration:.2f}s ({coverage_percentage:.1f}% of original video)")
        
        return processed_shorts
    
    def _ensure_output_file(self, file_path: Path) -> bool:
        """
        Ensure output file exists and is accessible. Handles Docker volume paths if needed.
        
        Args:
            file_path: Path to output file
            
        Returns:
            True if file exists or was successfully fixed, False otherwise
        """
        if file_path.exists() and file_path.stat().st_size > 0:
            self.logger.info(f"Output file verified: {file_path}")
            return True
            
        self.logger.warning(f"Output file not found or empty: {file_path}")
        
        # Check if we're in a Docker container
        in_docker = os.path.exists('/.dockerenv')
        self.logger.info(f"Running in Docker container: {in_docker}")
        
        if in_docker:
            # Try to locate the file in common Docker volume paths
            docker_paths = [
                # Common Docker volume mappings
                Path("/app/output") / file_path.name, 
                Path("/output") / file_path.name,
                Path("/data/output") / file_path.name,
                # Original path but with different volume mountpoints
                Path("/app") / file_path
            ]
            
            for docker_path in docker_paths:
                if docker_path.exists() and docker_path.stat().st_size > 0:
                    self.logger.info(f"Found output file at Docker path: {docker_path}")
                    try:
                        # Try to copy to the expected location
                        import shutil
                        file_path.parent.mkdir(exist_ok=True)
                        shutil.copy2(str(docker_path), str(file_path))
                        self.logger.info(f"Copied file from {docker_path} to {file_path}")
                        return True
                    except Exception as e:
                        self.logger.error(f"Failed to copy file: {e}")
        
        # Try to force creation of a minimal file as a test
        try:
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
            # Try to create a small test file to check permissions
            test_path = file_path.parent / f"test_{file_path.name}.txt"
            with open(test_path, 'w') as f:
                f.write("Permission test")
            
            if test_path.exists():
                self.logger.info(f"Successfully created test file: {test_path}")
                test_path.unlink() # Clean up
            else:
                self.logger.error(f"Failed to create test file: {test_path}")
        except Exception as e:
            self.logger.error(f"Permission test failed: {e}")
            
        return False
    
    async def process_directory(self, input_dir: str = None, user_prompt: str = None) -> Dict:
        """
        Process all videos in the input directory.
        
        Args:
            input_dir: Directory containing input videos (default: ./input)
            user_prompt: User prompt for theme-specific content selection
            
        Returns:
            Overall processing results
        """
        input_path = Path(input_dir) if input_dir else self.input_dir
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(list(input_path.glob(f"*{ext}")))
        
        if not video_files:
            self.logger.warning(f"No video files found in {input_path}")
            return {'videos_processed': 0, 'results': []}
        
        if user_prompt:
            self.logger.info(f"Found {len(video_files)} video files to process with prompt: '{user_prompt}'")
        else:
            self.logger.info(f"Found {len(video_files)} video files to process")
        
        # Process each video
        results = []
        for video_file in video_files:
            try:
                result = await self.process_video(str(video_file), user_prompt=user_prompt)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {video_file}: {e}")
                results.append({
                    'video_name': video_file.stem,
                    'error': str(e),
                    'success': False
                })
        
        # Generate summary
        summary = self._generate_processing_summary(results)
        
        return {
            'videos_processed': len(results),
            'user_prompt': user_prompt,
            'results': results,
            'summary': summary,
            'stats': self.stats
        }
    
    def _generate_processing_summary(self, results: List[Dict]) -> Dict:
        """Generate processing summary."""
        successful_results = [r for r in results if 'error' not in r]
        
        total_shorts = sum(r.get('shorts_generated', 0) for r in successful_results)
        total_time = sum(r.get('processing_time', 0) for r in successful_results)
        
        return {
            'total_videos': len(results),
            'successful_videos': len(successful_results),
            'failed_videos': len(results) - len(successful_results),
            'total_shorts_generated': total_shorts,
            'total_processing_time': total_time,
            'average_processing_time': total_time / len(successful_results) if successful_results else 0,
            'average_shorts_per_video': total_shorts / len(successful_results) if successful_results else 0
        }
    
    async def cleanup(self):
        """Cleanup resources and temporary files."""
        self.logger.info("Cleaning up resources...")
        
        try:
            # Cleanup individual components
            if self.whisper_analyzer:
                self.whisper_analyzer.cleanup_temp_files()
            
            if self.subject_detector:
                self.subject_detector.cleanup()
            
            if self.ollama_client:
                await self.ollama_client.cleanup()
            
            # Cleanup temporary files if configured
            if self.config['cleanup_temp_files']:
                import shutil
                if self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    self.temp_dir.mkdir()
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


async def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Video-to-Shorts Pipeline")
    parser.add_argument('--input', '-i', type=str, help='Input video file or directory')
    parser.add_argument('--output', '-o', type=str, help='Output directory')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--model', '-m', type=str, default='base', help='Whisper model size')
    parser.add_argument('--max-shorts', type=int, default=10, help='Maximum shorts per video')
    parser.add_argument('--no-smart-zoom', action='store_true', help='Disable smart zoom')
    parser.add_argument('--no-ollama', action='store_true', help='Disable Ollama analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--user-prompt', '-p', type=str, help='User prompt for theme-specific content (e.g., "goals in basketball", "comedy shorts", "emotional parts")')
    parser.add_argument('--ai-reframe', action='store_true', help='Enable AI-powered reframing based on object detection')
    parser.add_argument('--no-object-detection', action='store_true', help='Disable YOLO object detection')
    parser.add_argument('--final-video-ids', type=str, help='Comma-separated list of final video IDs (integers) for output files')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize processor
    processor = VideoToShortsProcessor(config_path=args.config)
    
    # Update configuration with command line arguments
    if args.model:
        processor.config['whisper_model'] = args.model
    if args.max_shorts:
        processor.config['max_shorts_per_video'] = args.max_shorts
    if args.no_smart_zoom:
        processor.config['smart_zoom_enabled'] = False
    if args.no_ollama:
        processor.config['ollama_analysis_enabled'] = False
    if args.ai_reframe:
        processor.config['ai_reframe'] = True
        processor.config['enable_ai_reframing'] = True
    if args.no_object_detection:
        processor.config['enable_object_detection'] = False
    if args.output:
        processor.output_dir = Path(args.output)
        ensure_directory(processor.output_dir)
    
    # Process final_video_ids if provided
    if args.final_video_ids:
        try:
            # Convert comma-separated string to list of integers
            final_video_ids = [int(vid.strip()) for vid in args.final_video_ids.split(',')]
            processor.config['final_video_ids'] = final_video_ids
            logging.info(f"Using final_video_ids: {final_video_ids}")
        except ValueError as e:
            logging.error(f"Error converting final_video_ids to integers: {e}")
            logging.warning("Will use default sequential numbering instead")
            processor.config['final_video_ids'] = []
    
    try:
        # Initialize components
        await processor.initialize_components()
        
        # Process videos
        if args.input:
            input_path = Path(args.input)
            if input_path.is_file():
                # Process single file
                result = await processor.process_video(str(input_path), user_prompt=args.user_prompt)
                if args.user_prompt:
                    print(f"Successfully processed {result['video_name']} with prompt '{args.user_prompt}': {result['shorts_generated']} shorts generated")
                else:
                    print(f"Successfully processed {result['video_name']}: {result['shorts_generated']} shorts generated")
            else:
                # Process directory
                results = await processor.process_directory(str(input_path), user_prompt=args.user_prompt)
                if args.user_prompt:
                    print(f"Processed {results['summary']['successful_videos']} videos successfully with prompt '{args.user_prompt}'")
                else:
                    print(f"Processed {results['summary']['successful_videos']} videos successfully")
                print(f"Generated {results['summary']['total_shorts_generated']} shorts")
        else:
            # Process default input directory
            results = await processor.process_directory(user_prompt=args.user_prompt)
            if args.user_prompt:
                print(f"Processed {results['summary']['successful_videos']} videos successfully with prompt '{args.user_prompt}'")
            else:
                print(f"Processed {results['summary']['successful_videos']} videos successfully")
            print(f"Generated {results['summary']['total_shorts_generated']} shorts")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    finally:
        await processor.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
