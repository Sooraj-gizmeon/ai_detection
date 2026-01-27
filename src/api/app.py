"""
FastAPI application for video processing API endpoints.
Independent from the Celery worker state.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Body, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo, ConfigDict, ValidationError
import uvicorn
import redis
import json
from celery.result import AsyncResult
from dotenv import load_dotenv

# Import Celery app and tasks - only for task submission, not for worker state
from src.queue_system.celery_app import celery_app

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Video Processing API",
    description="""
    API for processing videos into shorts with AI-powered content selection or linear cutting.
    
    ### Linear Cut Mode
    
    The **linear_cut** parameter enables simple linear segmentation without AI analysis:
    
    - **linear_cut=True**: Divide video into equal parts without AI content analysis
      - **linear_duration**: Duration of each segment in seconds (required when linear_cut=True)
      - **final_video_ids**: List of integer IDs for output videos (required when linear_cut=True)
      - **no_of_videos**: Number of segments to create (must match length of final_video_ids)
      - **append_part_number**: Add "Part X of Y" to titles (default: True)
    
    When linear_cut is enabled:
    - Kept: Aspect ratio conversion, subtitles, overlays, intro/outro, template processing, AI title/description
    - Skipped: AI scene detection, content analysis, smart segment selection
    
    **Example Request**:
    ```json
    {
      "video_id": "video_abc",
      "linear_cut": true,
      "linear_duration": 60,
      "no_of_videos": 5,
      "final_video_ids": [1001, 1002, 1003, 1004, 1005],
      "pubid": "pub123",
      "bucket_path": "path/to/video.mp4"
    }
    ```
    
    ### User Prompt-Based Content Selection (AI Mode)
    
    The **user_prompt** parameter allows you to specify the type of content you want to extract:
    
    - **"climax scenes"** - Extract intense, dramatic, high-energy moments
    - **"comedy shorts"** - Find humorous, entertaining, light-hearted moments  
    - **"emotional parts"** - Identify touching, inspiring, emotionally engaging content
    - **"educational content"** - Extract informative, instructional segments
    - **"motivational moments"** - Find inspiring, empowering content
    - **"action sequences"** - Capture high-energy, dynamic movement
    - **"dramatic scenes"** - Extract serious, intense, impactful moments
    
    Examples:
    - `"create shorts on the climax scenes"`
    - `"Create comedy shorts"`
    - `"create a shorts so that it conveys the emotional parts of the movie"`
    - `"extract the most educational segments"`
    - `"find motivational moments for inspiration"`

    ### Intro and Outro Videos
    
    Add **intro_url** and **outro_url** to automatically prepend and append videos to all generated clips:
    
    - **intro_url**: Wasabi storage path to intro video (e.g., "video/intro/100628-video-720.mp4")
    - **outro_url**: Wasabi storage path to outro video (e.g., "video/outro/100627-video-720.mp4")
    
    When provided, each generated clip will be: **intro + generated_clip + outro**
    
    - Intro/outro videos are automatically downloaded from Wasabi storage
    - Both intro and outro are optional - you can use one, both, or neither
    - Videos are concatenated seamlessly with proper encoding
    
    ### Canvas Type vs. Content Aspect Ratio
    
    - **canvas_type**: Determines the output container dimensions
      - `shorts`: 9:16 vertical video container
      - `clips`: 16:9 horizontal video container
    
    - **aspect_ratio**: Determines the aspect ratio of the content within the canvas
      - Example: `1:1` for square content within a vertical shorts container
      - Example: `2.35:1` for cinematic content within a shorts container
      - If not specified, defaults to match the canvas_type aspect ratio
    
    ### Template-Based Processing
    
    - **template**: Specialized video processing templates
      - `podcast`: Speaker-based cropping for 2-speaker videos
        - Detects 2 speakers in video frames
        - Crops left speaker to top half of vertical video
        - Crops right speaker to bottom half of vertical video
        - Perfect for podcast interviews, talk shows, panel discussions
      - `null`: Default processing (no template)
    
    ### Video Duration Settings
    
    - **min_duration**: Minimum duration for each output video (in seconds)
      - Default: 15 seconds
      - Range: 5-300 seconds
      
    - **max_duration**: Maximum duration for each output video (in seconds)
      - Default: 60 seconds
      - Range: 10-600 seconds
      
    These duration settings control the length of the generated short videos.
    
    ### AI-Powered Analysis
    
    The system uses:
    - **Audio transcription** for understanding spoken content
    - **Vision analysis** for visual content understanding  
    - **Dual-modal AI analysis** combining both audio and visual insights
    - **Comprehensive segment generation** analyzing ALL possible segments
    - **Quality optimization** ensuring maximum output quality
    """,
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware to log all requests for debugging 422 errors
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests for debugging.
    """
    # Log request details
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"   Headers: {dict(request.headers)}")
    
    # Log request body for POST requests
    if request.method == "POST":
        try:
            body = await request.body()
            # log the request body
            logger.info(f"   Body: {body.decode() if body else '<empty>'}")
            if body:
                # Try to parse as JSON for better logging
                try:
                    body_json = json.loads(body.decode())
                    logger.info(f"   Body (JSON): {json.dumps(body_json, indent=2)}")
                except:
                    logger.info(f"   Body (raw): {body.decode()}")
            else:
                logger.info("   Body: <empty>")
                
            # Re-create request with the body (since we consumed it)
            async def receive():
                return {"type": "http.request", "body": body}
            request._receive = receive
        except Exception as e:
            logger.error(f"   Error reading request body: {e}")
    
    # Process the request
    try:
        response = await call_next(request)
        logger.info(f"   Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"   Request processing error: {e}")
        logger.error(f"   Error type: {type(e)}")
        raise

# Add exception handler for 422 Unprocessable Entity errors
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    """
    Handle 422 Unprocessable Entity errors with detailed logging.
    """
    # Get the request body for logging
    try:
        body = await request.body()
        body_str = body.decode() if body else "No body"
    except Exception as e:
        body_str = f"Error reading body: {str(e)}"
    
    # Log detailed error information
    logger.error("422 Unprocessable Entity Error Details:")
    logger.error(f"   URL: {request.url}")
    logger.error(f"   Method: {request.method}")
    logger.error(f"   Headers: {dict(request.headers)}")
    logger.error(f"   Body: {body_str}")
    logger.error(f"   Exception: {exc}")
    logger.error(f"   Exception type: {type(exc)}")
    
    # Try to get more detailed validation error information
    if hasattr(exc, 'errors'):
        logger.error(f"   Validation errors: {exc.errors()}")
    if hasattr(exc, 'body'):
        logger.error(f"   Exception body: {exc.body}")
    if hasattr(exc, 'detail'):
        logger.error(f"   Exception detail: {exc.detail}")
    
    # Return a detailed error response
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation Error - Check logs for detailed information",
            "error_type": str(type(exc)),
            "url": str(request.url),
            "method": request.method,
            "validation_errors": getattr(exc, 'errors', lambda: [])() if hasattr(exc, 'errors') else []
        }
    )

# Add validation error handler specifically for Pydantic validation errors
@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handle Pydantic validation errors with detailed logging.
    """
    # Get the request body for logging
    try:
        body = await request.body()
        body_str = body.decode() if body else "No body"
    except Exception as e:
        body_str = f"Error reading body: {str(e)}"
    
    # Log detailed error information
    logger.error("Pydantic Validation Error Details:")
    logger.error(f"   URL: {request.url}")
    logger.error(f"   Method: {request.method}")
    logger.error(f"   Body: {body_str}")
    logger.error(f"   Validation errors: {exc.errors()}")
    logger.error(f"   Error count: {exc.error_count()}")
    
    # Return a detailed error response
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Pydantic Validation Error",
            "errors": exc.errors(),
            "error_count": exc.error_count(),
            "url": str(request.url),
            "method": request.method
        }
    )

# Add general exception handler for any unhandled request validation errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle any unhandled exceptions with detailed logging.
    """
    # Only handle if it's a validation-related error that might cause 422
    if any(keyword in str(exc).lower() for keyword in ['validation', 'invalid', 'required', 'missing']):
        # Get the request body for logging
        try:
            body = await request.body()
            body_str = body.decode() if body else "No body"
        except Exception as e:
            body_str = f"Error reading body: {str(e)}"
        
        # Log detailed error information
        logger.error("General Validation-Related Error:")
        logger.error(f"   URL: {request.url}")
        logger.error(f"   Method: {request.method}")
        logger.error(f"   Body: {body_str}")
        logger.error(f"   Exception: {exc}")
        logger.error(f"   Exception type: {type(exc)}")
        logger.error(f"   Exception args: {exc.args}")
        
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Validation-related error: {str(exc)}",
                "error_type": str(type(exc)),
                "url": str(request.url),
                "method": request.method
            }
        )
    
    # For non-validation errors, re-raise them to be handled by FastAPI's default handler
    raise exc

# Redis client for task status tracking
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=int(os.getenv("REDIS_DB", "0")),
    password=os.getenv("REDIS_PASSWORD", ""),
    decode_responses=True  # Return strings instead of bytes
)

# Define models
class VideoProcessingRequest(BaseModel):
    """Model for video processing request."""
    
    video_id: str = Field(default=None, description="Unique identifier for the video job (use this OR uploaded_video_id)")
    uploaded_video_id: str = Field(default=None, description="Alternative unique identifier for uploaded videos (use this OR video_id)")
    final_video_ids: List[int] = Field(default_factory=list, description="List of integer IDs for the output videos (required when linear_cut=True, optional otherwise)")
    canvas_type: str = Field(default="shorts", description="Canvas type: 'shorts' (9:16) or 'clips' (16:9)")
    no_of_videos: int = Field(default=5, ge=1, le=20, description="Number of output videos to generate")
    linear_cut: bool = Field(default=False, description="Enable linear cut mode - divide video into equal parts without AI analysis")
    linear_duration: int = Field(default=None, description="Duration of each segment in linear cut mode (required when linear_cut=True)")
    disable_smart_zoom_linear_cut: bool = Field(default=True, description="Disable smart zoom for linear cut mode for faster processing (7-10x speedup)")
    min_duration: int = Field(default=15, ge=5, le=300, description="Minimum duration of each output video in seconds (5-300 seconds, not used in linear_cut mode)")
    max_duration: int = Field(default=60, ge=10, le=600, description="Maximum duration of each output video in seconds (10-600 seconds, not used in linear_cut mode)")
    append_part_number: bool = Field(default=True, description="Append 'Part X of Y' to AI-generated titles in linear_cut mode")
    aspect_ratio: str = Field(default=None, description="Content aspect ratio within the canvas ('9:16', '16:9', '1:1', etc.)")
    subtitle_overlay: bool = Field(default=False, description="Whether to add subtitle overlay with transcription text")
    subtitle_overlay_style_id: str = Field(default="cinematic", description="Subtitle style ID from config/subtitle_config.json (e.g., 'cinematic', 'elegant', 'tech')")
    caption_x: int = Field(default=30, ge=0, description="Caption X position in pixels (0 = left edge, increases rightward)")
    caption_y: int = Field(default=40, ge=0, description="Caption Y position in pixels (0 = bottom edge, increases upward)")
    kinetic_captions: bool = Field(default=True, description="Enable kinetic captions with word-level timing animations")
    kinetic_mode: str = Field(default="karaoke", description="Kinetic caption effect mode: 'karaoke', 'typewriter', 'highlight'")
    subtitle_overlay_style: Optional[Dict] = Field(default=None, description="Custom subtitle style object with font_family, font_style, font_weight, default_color, letter_spacing, text_transform, pronunciation_color")
    user_prompt: str = Field(default=None, description="User prompt for theme-specific content selection (e.g., 'goals in basketball', 'comedy shorts', 'emotional parts')")
    ai_reframe: bool = Field(default=False, description="Enable AI-powered reframing based on object detection and prompt analysis")
    enable_object_detection: bool = Field(default=True, description="Enable YOLO object detection for enhanced content analysis")
    llm_provider: str = Field(default="ollama", description="LLM provider for textual analysis: 'openai' or 'ollama'")
    template: str = Field(default=None, description="Video template type: 'podcast' for speaker-based cropping, None for default processing")
    intro_url: str = Field(default=None, description="Optional URL to intro video in Wasabi storage to prepend to generated clips")
    outro_url: str = Field(default=None, description="Optional URL to outro video in Wasabi storage to append to generated clips")
    brand_logo: str = Field(default=None, description="Optional URL to brand logo image to overlay on generated clips")
    overlay_x: int = Field(default=10, ge=0, description="Logo overlay X position in pixels (0 = left edge, increases rightward)")
    overlay_y: int = Field(default=20, ge=0, description="Logo overlay Y position in pixels (0 = bottom edge, increases upward)")
    pubid: str = Field(..., description="Publisher ID")
    channelid: str = Field(default="000", description="Channel ID")
    bucket_path: str = Field(..., description="Path to the input video in the storage bucket")
    storage_bucket: str = Field(default="videos", description="Name of the storage bucket")
    storage_type: str = Field(default="wasabi", description="Storage type: 'wasabi' or 'aws' (default: wasabi)")
    external_srt_url: Optional[str] = Field(default=None, description="Optional URL to external .srt subtitle file to overlay on final clips")
    celebrity_detection: bool = Field(default=False, description="Enable celebrity detection in video")

    model_config = ConfigDict(
        # Log validation errors
        validate_assignment=True,
        # Allow extra fields (for forward compatibility)
        extra="allow"
    )
    
    def __init__(self, **data):
        """Initialize with detailed logging for debugging 422 errors."""
        logger.info(f"Creating VideoProcessingRequest with data keys: {list(data.keys())}")
        logger.info(f"Raw data: {json.dumps(data, indent=2, default=str)}")
        
        # Check for required fields before validation
        # Note: Either video_id OR uploaded_video_id is required (not both)
        required_fields = ['pubid', 'bucket_path']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            logger.error(f"Available fields: {list(data.keys())}")
        
        # Check final_video_ids specifically since it's mentioned in the log
        if 'final_video_ids' in data:
            logger.info(f"final_video_ids value: {data['final_video_ids']} (type: {type(data['final_video_ids'])})")
            if isinstance(data['final_video_ids'], list):
                for i, vid_id in enumerate(data['final_video_ids']):
                    logger.info(f"   [{i}]: {vid_id} (type: {type(vid_id)})")
            else:
                logger.error(f"final_video_ids is not a list: {type(data['final_video_ids'])}")
        
        super().__init__(**data)
    
    @model_validator(mode='after')
    def validate_video_identifiers(self):
        """Ensure at least one video identifier is provided."""
        if not self.video_id and not self.uploaded_video_id:
            raise ValueError("Either video_id or uploaded_video_id must be provided")
        return self
    
    @model_validator(mode='after')
    def validate_linear_cut_parameters(self):
        """Ensure linear_duration and final_video_ids are provided when linear_cut is enabled."""
        if self.linear_cut:
            if not self.linear_duration:
                raise ValueError("linear_duration is required when linear_cut=True")
            if self.linear_duration < 5 or self.linear_duration > 600:
                raise ValueError("linear_duration must be between 5 and 600 seconds")
            if not self.final_video_ids:
                raise ValueError("final_video_ids is required when linear_cut=True")
            if len(self.final_video_ids) != self.no_of_videos:
                raise ValueError(f"Length of final_video_ids ({len(self.final_video_ids)}) must match no_of_videos ({self.no_of_videos})")
        
        return self
    
    def get_effective_video_id(self) -> str:
        """Get the effective video ID (video_id takes precedence over uploaded_video_id)."""
        return self.video_id if self.video_id else self.uploaded_video_id
    
    @field_validator('canvas_type')
    @classmethod
    def validate_canvas_type(cls, v):
        """Validate canvas type."""
        logger.info(f"Validating canvas_type: '{v}'")
        if v not in ['shorts', 'clips']:
            logger.error(f"Invalid canvas_type: '{v}' - must be 'shorts' or 'clips'")
            raise ValueError("canvas_type must be 'shorts' or 'clips'")
        logger.info(f"Canvas type validation passed: '{v}'")
        return v
    
    @field_validator('final_video_ids')
    @classmethod
    def validate_final_video_ids(cls, v):
        """Validate final_video_ids are all valid integers."""
        logger.info(f"Validating final_video_ids: {v}")
        if not all(isinstance(vid_id, int) for vid_id in v):
            logger.error(f"Invalid final_video_ids: {v} - all must be integers")
            raise ValueError("All final_video_ids must be integers")
        logger.info(f"Final video IDs validation passed: {v}")
        return v
    
    @field_validator('max_duration')
    @classmethod
    def validate_duration(cls, v, info: ValidationInfo):
        """Validate max_duration is greater than min_duration."""
        values = info.data if info.data else {}
        logger.info(f"Validating max_duration: {v}, min_duration: {values.get('min_duration', 'N/A')}")
        if 'min_duration' in values and v <= values['min_duration']:
            logger.error(f"Invalid duration: max_duration ({v}) must be greater than min_duration ({values['min_duration']})")
            raise ValueError("max_duration must be greater than min_duration")
        logger.info(f"Duration validation passed: max={v}, min={values.get('min_duration', 'N/A')}")
        return v
    
    @field_validator('aspect_ratio')
    @classmethod
    def validate_aspect_ratio(cls, v):
        """Validate aspect ratio."""
        logger.info(f"Validating aspect_ratio: '{v}'")
        # Allow None (will default based on canvas_type)
        if v is None:
            logger.info("Aspect ratio validation passed: None (will use default)")
            return v
            
        try:
            width, height = v.split(':')
            int(width), int(height)  # Check if they are integers
            logger.info(f"Aspect ratio validation passed: '{v}' -> {width}:{height}")
        except (ValueError, TypeError):
            logger.error(f"Invalid aspect_ratio: '{v}' - must be in format 'width:height' (e.g., '9:16')")
            raise ValueError("aspect_ratio must be in format 'width:height' (e.g., '9:16')")
        return v
    
    @field_validator('intro_url', 'outro_url')
    @classmethod
    def validate_media_urls(cls, v):
        """Validate intro/outro URLs."""
        if v is None:
            return v
        
        logger.info(f"Validating media URL: '{v}'")
        
        # Check if URL is a valid Wasabi path (basic validation)
        if not isinstance(v, str) or not v.strip():
            logger.error(f"Invalid media URL: must be a non-empty string")
            raise ValueError("Media URL must be a non-empty string")
        
        # Basic path validation - should contain common video extensions
        valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            logger.warning(f"Media URL '{v}' doesn't have a common video extension")
        
        logger.info(f"Media URL validation passed: '{v}'")
        return v
    
    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        """Validate template type."""
        logger.info(f"Validating template: '{v}'")
        # Allow None for default processing
        if v is None:
            logger.info("Template validation passed: None (default processing)")
            return v
            
        valid_templates = ['podcast']
        if v not in valid_templates:
            logger.error(f"Invalid template: '{v}' - must be one of {valid_templates}")
            raise ValueError(f"template must be one of {valid_templates}")
        
        logger.info(f"Template validation passed: '{v}'")
        return v
    
    @field_validator('llm_provider')
    @classmethod
    def validate_llm_provider(cls, v):
        """Validate LLM provider."""
        logger.info(f"Validating llm_provider: '{v}'")
        if v not in ['openai', 'ollama']:
            logger.error(f"Invalid llm_provider: '{v}' - must be 'openai' or 'ollama'")
            raise ValueError("llm_provider must be 'openai' or 'ollama'")
        logger.info(f"LLM provider validation passed: '{v}'")
        return v
    
    @field_validator('overlay_x', 'overlay_y', 'caption_x', 'caption_y')
    @classmethod
    def validate_overlay_coordinates(cls, v, info: ValidationInfo):
        """Validate overlay and caption coordinates."""
        field_name = info.field_name
        logger.info(f"Validating {field_name}: {v}")
        if v < 0:
            raise ValueError(f"{field_name} must be >= 0 (got {v})")
        if v > 2000:  # Reasonable upper limit for coordinates
            raise ValueError(f"{field_name} must be <= 2000 pixels (got {v})")
        logger.info(f"{field_name} validation passed: {v}")
        return v
    
    @field_validator('subtitle_overlay_style')
    @classmethod
    def validate_style(cls, v):
        """Validate subtitle_overlay_style object for subtitle customization."""
        if v is None:
            return v
        
        logger.info(f"Validating subtitle_overlay_style: {v}")
        
        if not isinstance(v, dict):
            logger.error(f"Invalid subtitle_overlay_style: must be a dictionary")
            raise ValueError("subtitle_overlay_style must be a dictionary")
        
        # Define expected style properties
        valid_properties = {
            'font_family': str,  # 'Arial', 'Impact', etc.
            'font_style': str,  # 'italic' or None
            'font_weight': int,  # 100, 300, 400, 600, 700, 800, 900
            'default_color': str,  # '#ccc', 'white', etc.
            'letter_spacing': str,  # '3px'
            'text_transform': str,  # 'uppercase' or None
            'pronunciation_color': str  # '#4361EE' for highlight color
        }
        
        # Validate each provided property
        for prop, value in v.items():
            if prop not in valid_properties:
                logger.warning(f"Unknown subtitle style property: {prop}")
                continue
                
            expected_type = valid_properties[prop]
            if not isinstance(value, expected_type) and value is not None:
                logger.error(f"Invalid subtitle style property {prop}: expected {expected_type.__name__}, got {type(value).__name__}")
                raise ValueError(f"subtitle_overlay_style property '{prop}' must be {expected_type.__name__} or null")
        
        # Validate color formats for color properties
        color_properties = ['default_color', 'pronunciation_color']
        for prop in color_properties:
            if prop in v and v[prop] is not None:
                color_value = v[prop]
                if not isinstance(color_value, str):
                    continue
                    
                # Basic color validation (hex format or named colors)
                if color_value.startswith('#'):
                    if len(color_value) not in [4, 7]:  # #RGB or #RRGGBB
                        logger.error(f"Invalid hex color format for {prop}: {color_value}")
                        raise ValueError(f"Invalid hex color format for {prop}: {color_value}")
                elif color_value not in ['white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta']:
                    # Allow any color string, but warn about non-standard ones
                    logger.warning(f"Non-standard color for {prop}: {color_value}")
        
        logger.info(f"Subtitle overlay style validation passed: {v}")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def auto_enable_kinetic_captions(cls, values):
        """Automatically enable kinetic captions when subtitle_overlay is True."""
        # Handle both dict and model instances
        if isinstance(values, dict):
            data = values
        else:
            data = values.__dict__.copy()

        subtitle_overlay = data.get('subtitle_overlay', False)
        kinetic_captions = data.get('kinetic_captions', True)  # Default is True
        external_srt = data.get('external_srt_url')
        logger.info(f"Auto-validating subtitle and caption settings: subtitle_overlay={subtitle_overlay}, kinetic_captions={kinetic_captions}, external_srt_url={external_srt}")

        # REQUIREMENT: If subtitle_overlay is False, kinetic_captions must also be False
        if not subtitle_overlay:
            if kinetic_captions:
                data['kinetic_captions'] = False
                logger.info(f"Auto-disabled kinetic_captions because subtitle_overlay=False")
            else:
                logger.info(f"No captions will be rendered (both subtitle_overlay and kinetic_captions are False)")
        else:
            # subtitle_overlay is True
            # If an external SRT is provided, prefer burned subtitles and disable kinetic captions
            if external_srt:
                if kinetic_captions:
                    data['kinetic_captions'] = False
                    logger.info(f"Auto-disabled kinetic_captions because external_srt_url was provided")
                # Ensure subtitle overlay is enabled when external SRT is provided
                if not subtitle_overlay:
                    data['subtitle_overlay'] = True
                    logger.info(f"Auto-enabled subtitle_overlay because external_srt_url was provided")
            else:
                if not kinetic_captions:
                    # Auto-enable kinetic captions when subtitle overlay is requested
                    data['kinetic_captions'] = True
                    logger.info(f"Auto-enabled kinetic_captions because subtitle_overlay=True")
                else:
                    logger.info(f"Both subtitle_overlay and kinetic_captions are enabled")

        return data if isinstance(values, dict) else values
        
    def get_canvas_aspect_ratio_tuple(self) -> tuple:
        """
        Get canvas aspect ratio based on canvas_type.
        
        Returns:
            Tuple of (width, height) for the canvas aspect ratio
        """
        # Canvas aspect ratio is always determined by canvas_type
        return (9, 16) if self.canvas_type == 'shorts' else (16, 9)
        
    def get_content_aspect_ratio_tuple(self) -> tuple:
        """
        Convert aspect_ratio string to tuple, or default based on canvas_type.
        
        Returns:
            Tuple of (width, height) for content aspect ratio
        """
        # Try aspect_ratio first
        if self.aspect_ratio:
            try:
                width, height = self.aspect_ratio.split(':')
                return (int(width), int(height))
            except (ValueError, TypeError):
                pass
                
        # Default to same as canvas aspect ratio
        return self.get_canvas_aspect_ratio_tuple()

class VideoProcessingResponse(BaseModel):
    """Model for video processing response."""
    
    task_id: str = Field(..., description="Celery task ID")
    video_id: str = Field(..., description="Video job ID")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")

class TaskStatusResponse(BaseModel):
    """Model for task status response."""
    
    task_id: str = Field(..., description="Celery task ID")
    video_id: str = Field(None, description="Video job ID")
    status: str = Field(..., description="Task status")
    result: Optional[Dict] = Field(None, description="Task result if available")
    error: Optional[str] = Field(None, description="Error message if failed")

# Function to store task metadata in Redis
def store_task_metadata(task_id: str, video_id: str, request_data: Dict[str, Any]):
    """Store task metadata in Redis for independent tracking."""
    try:
        # Create metadata - include both video_id and uploaded_video_id if available
        metadata = {
            "task_id": task_id,
            "video_id": video_id,  # This is the effective ID (video_id or uploaded_video_id)
            "status": "PENDING",
            "request_data": request_data,
            "created_at": datetime.now().isoformat()
        }
        
        # Store in Redis with appropriate keys
        redis_client.set(f"task:{task_id}", json.dumps(metadata))
        
        # Store task ID lookup for video_id if present in request_data
        if request_data.get('video_id'):
            redis_client.set(f"video:{request_data['video_id']}", task_id)
            logger.info(f"Stored video_id lookup: video:{request_data['video_id']} -> {task_id}")
        
        # Store task ID lookup for uploaded_video_id if present in request_data
        if request_data.get('uploaded_video_id'):
            redis_client.set(f"uploaded_video:{request_data['uploaded_video_id']}", task_id)
            logger.info(f"Stored uploaded_video_id lookup: uploaded_video:{request_data['uploaded_video_id']} -> {task_id}")
        
        redis_client.sadd("active_tasks", task_id)
        
        logger.info(f"Stored metadata for task {task_id} (effective video ID: {video_id})")
    except Exception as e:
        logger.error(f"Failed to store task metadata: {e}")

def log_api_request_body(request_data: Dict[str, Any], endpoint: str):
    """Log API request body to a JSON file in the logs directory."""
    try:
        # Create logs directory if it doesn't exist
        # Use relative path that works both in Docker and local development
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "request_body": request_data,
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to JSON file
        log_filename = f"api_request_{endpoint.replace('/', '_')}_{timestamp}.json"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        with open(log_filepath, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)
        
        logger.info(f"API request body logged to: {log_filepath}")
        
    except Exception as e:
        logger.error(f"Failed to log API request body: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Attempted logs directory: {logs_dir if 'logs_dir' in locals() else 'undefined'}")
        logger.error(f"Failed to log API request body: {e}")

# API routes
@app.post("/api/queue/video", response_model=VideoProcessingResponse)
async def queue_video_processing(request: VideoProcessingRequest, background_tasks: BackgroundTasks, raw_request: Request = None):
    """
    Queue a video for processing.
    
    Either video_id or uploaded_video_id must be provided (video_id takes precedence).
    """
    # Get effective video ID for logging and task identification
    effective_video_id = request.get_effective_video_id()
    """
    Queue a video for processing with AI-powered content selection.
    
    This endpoint accepts video processing parameters and adds the job to the Celery queue.
    
    **New Feature: User Prompt-Based Content Selection**
    
    Use the `user_prompt` parameter to specify what type of content you want to extract:
    
    - "climax scenes" - Extract dramatic, intense, high-energy moments
    - "comedy shorts" - Find humorous, entertaining segments
    - "emotional parts" - Identify touching, inspiring content
    - "educational content" - Extract informative segments
    - "motivational moments" - Find inspiring content
    - "action sequences" - Capture dynamic movement
    
    **Technical Details:**
    
    The canvas_type determines the output container dimensions:
    - shorts: 9:16 vertical video container
    - clips: 16:9 horizontal video container
    
    The aspect_ratio determines how the content is framed within the canvas.
    If not specified, aspect_ratio defaults to match the canvas_type.
    
    The subtitle_overlay flag enables automatic transcription overlay on the generated videos.
    When enabled, the video transcription will be displayed as subtitles on the output videos.
    
    **AI Analysis Pipeline:**
    
    1. **Comprehensive Segment Generation** - Analyzes ALL possible segments instead of just a subset
    2. **Audio Transcription Analysis** - Uses Whisper for speech understanding
    3. **Vision Analysis** - Analyzes visual content for better selection
    4. **Dual-Modal AI Analysis** - Combines audio and visual insights
    5. **Prompt-Based Filtering** - Matches segments to user's specific request
    6. **Quality Optimization** - Ensures maximum output quality
    """
    # Log the complete request body to logs directory
    request_dict = request.dict()
    log_api_request_body(request_dict, "api/queue/video")
    
    # Log incoming request details for debugging 422 errors
    logger.info("Incoming video processing request:")
    logger.info(f"   URL: {raw_request.url if raw_request else 'N/A'}")
    logger.info(f"   Method: {raw_request.method if raw_request else 'N/A'}")
    logger.info(f"   Video ID: {request.video_id}")
    logger.info(f"   Uploaded Video ID: {request.uploaded_video_id}")
    logger.info(f"   Effective ID: {effective_video_id}")
    logger.info(f"   Canvas Type: {request.canvas_type}")
    logger.info(f"   Final Video IDs: {request.final_video_ids}")
    logger.info(f"   Duration: {request.min_duration}-{request.max_duration}s")
    logger.info(f"   Linear Cut: {request.linear_cut}")
    if request.linear_cut:
        logger.info(f"   Linear Duration: {request.linear_duration}s")
        logger.info(f"   Disable Smart Zoom: {request.disable_smart_zoom_linear_cut}")
        logger.info(f"   Append Part Numbers: {request.append_part_number}")
    logger.info(f"   Aspect Ratio: {request.aspect_ratio}")
    logger.info(f"   User Prompt: '{request.user_prompt}'")
    logger.info(f"   LLM Provider: {request.llm_provider}")
    logger.info(f"   Template: {request.template}")
    logger.info(f"   AI Reframe: {request.ai_reframe}")
    logger.info(f"   Object Detection: {request.enable_object_detection}")
    logger.info(f"   Subtitle Overlay: {request.subtitle_overlay}")
    logger.info(f"   Caption Position: ({request.caption_x}, {request.caption_y})")
    logger.info(f"   Intro URL: {request.intro_url}")
    logger.info(f"   Outro URL: {request.outro_url}")
    logger.info(f"   Brand Logo: {request.brand_logo}")
    logger.info(f"   Logo Position: ({request.overlay_x}, {request.overlay_y})")
    logger.info(f"   Channel ID: {request.channelid}")
    logger.info(f"   Publisher ID: {request.pubid}")
    logger.info(f"   Bucket Path: {request.bucket_path}")
    logger.info(f"   Storage Type: {request.storage_type}")
    logger.info(f"   Storage Bucket: {request.storage_bucket}")
    
    try:
        # Convert Pydantic model to dictionary
        task_data = request.dict()
        
        # Calculate canvas aspect ratio based on canvas_type
        canvas_aspect_ratio = request.get_canvas_aspect_ratio_tuple()
        task_data['canvas_aspect_ratio_tuple'] = canvas_aspect_ratio
        
        # Calculate content aspect ratio based on aspect_ratio or default
        content_aspect_ratio = request.get_content_aspect_ratio_tuple()
        task_data['content_aspect_ratio_tuple'] = content_aspect_ratio
        
        # Add explicit strings for canvas_type and aspect_ratio to avoid any ambiguity
        task_data['canvas_type_str'] = request.canvas_type
        task_data['aspect_ratio_str'] = request.aspect_ratio
        
        logger.info(f"Processing video with canvas_type={request.canvas_type}, " +
                   f"canvas_aspect_ratio={canvas_aspect_ratio}, " +
                   f"content_aspect_ratio={content_aspect_ratio}, " +
                   f"aspect_ratio_str={request.aspect_ratio}, " +
                   f"user_prompt='{request.user_prompt}'")
        
        # Submit task to Celery queue
        task = celery_app.send_task(
            "process_video_task",
            kwargs={"task_data": task_data},
            queue="video_processing",
            routing_key="video_processing"
        )
        
        # Store metadata in Redis (in background to avoid blocking)
        background_tasks.add_task(
            store_task_metadata,
            task.id,
            effective_video_id,
            task_data
        )
        
        prompt_message = f" with user prompt: '{request.user_prompt}'" if request.user_prompt else ""
        
        return {
            "task_id": task.id,
            "video_id": effective_video_id,
            "responded_time": datetime.now().isoformat(),
            "status": "queued",
            "message": f"Video processing task has been queued e successfully with canvas_type={request.canvas_type} ({canvas_aspect_ratio}), " +
                      f"content aspect_ratio={content_aspect_ratio}{prompt_message}"
        }
    
    except ValidationError as ve:
        # Log Pydantic validation errors specifically
        logger.error("Pydantic Validation Error in queue_video_processing:")
        logger.error(f"   Validation errors: {ve.errors()}")
        logger.error(f"   Error count: {ve.error_count()}")
        logger.error(f"   Video ID attempted: {getattr(request, 'video_id', 'N/A')}")
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    
    except Exception as e:
        logger.error(f"Failed to queue task: {e}", exc_info=True)
        logger.error(f"   Video ID: {getattr(request, 'video_id', 'N/A')}")
        logger.error(f"   Exception type: {type(e)}")
        logger.error(f"   Exception details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to queue task: {str(e)}")

@app.get("/api/queue/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a task.
    
    This endpoint returns the status of a task by its ID using Redis for state tracking.
    """
    try:
        # First check Redis for task metadata
        task_data_str = redis_client.get(f"task:{task_id}")
        
        if not task_data_str:
            # Task not found in Redis, try Celery directly as fallback
            task_result = AsyncResult(task_id, app=celery_app)
            
            return {
                "task_id": task_id,
                "status": task_result.status,
                "result": task_result.result if task_result.ready() and task_result.successful() else None,
                "error": str(task_result.result) if task_result.ready() and not task_result.successful() else None
            }
        
        # Parse task data from Redis
        task_data = json.loads(task_data_str)
        video_id = task_data.get("video_id")
        
        # Check if we have a result
        result_data_str = redis_client.get(f"result:{task_id}")
        result_data = json.loads(result_data_str) if result_data_str else None
        
        # Check Celery for the current status (we don't depend on the worker for results)
        try:
            task_result = AsyncResult(task_id, app=celery_app)
            status = task_result.status
        except:
            # If Celery is unavailable, use the status from Redis
            status = task_data.get("status", "UNKNOWN")
        
        return {
            "task_id": task_id,
            "video_id": video_id,
            "status": status,
            "result": result_data.get("result") if result_data else None,
            "error": result_data.get("error") if result_data else None
        }
    
    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@app.get("/api/queue/video/{video_id}", response_model=TaskStatusResponse)
async def get_status_by_video_id(video_id: str):
    """
    Get the status of a task by video ID or uploaded_video_id.
    
    This endpoint returns the status of a task using the video ID or uploaded_video_id.
    """
    try:
        # Try to get task ID from video_id first
        task_id = redis_client.get(f"video:{video_id}")
        
        # If not found, try uploaded_video_id pattern
        if not task_id:
            task_id = redis_client.get(f"uploaded_video:{video_id}")
        
        if not task_id:
            raise HTTPException(status_code=404, detail=f"No task found for video ID: {video_id}")
        
        # Reuse the task status endpoint
        return await get_task_status(task_id)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task by video ID: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get task by video ID: {str(e)}")

@app.get("/api/queue/list")
async def list_tasks():
    """
    List all active tasks.
    
    This endpoint returns a list of all active tasks using Redis for state tracking.
    """
    try:
        # Get all active tasks from Redis
        task_ids = redis_client.smembers("active_tasks")
        
        if not task_ids:
            return {"tasks": []}
        
        tasks = []
        for task_id in task_ids:
            task_data_str = redis_client.get(f"task:{task_id}")
            if task_data_str:
                task_data = json.loads(task_data_str)
                
                # Check current status from Celery if available
                try:
                    task_result = AsyncResult(task_id, app=celery_app)
                    status = task_result.status
                except:
                    # If Celery is unavailable, use the status from Redis
                    status = task_data.get("status", "UNKNOWN")
                
                tasks.append({
                    "task_id": task_id,
                    "video_id": task_data.get("video_id"),
                    "status": status,
                    "created_at": task_data.get("created_at")
                })
        
        return {"tasks": tasks}
    
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")



@app.get("/api/supported-themes")
async def get_supported_themes():
    """
    Get list of supported themes for user prompt-based content selection.
    Enhanced with object detection capabilities.
    
    Returns a list of supported themes and their descriptions for use in the user_prompt parameter.
    """
    try:
        # Import here to avoid circular imports and only when needed
        from src.content_analysis.prompt_based_analyzer import PromptBasedAnalyzer
        
        analyzer = PromptBasedAnalyzer()
        themes = analyzer.get_supported_themes()
        
        return {
            "supported_themes": themes,
            "examples": {
                "climax": "create shorts on the climax scenes",
                "comedy": "Create comedy shorts", 
                "emotional": "create a shorts so that it conveys the emotional parts of the movie",
                "educational": "extract the most educational segments",
                "motivational": "find motivational moments for inspiration",
                "action": "capture action sequences",
                "dramatic": "extract dramatic scenes",
                "sports_goals": "get the goals in this basketball match",
                "sports_highlights": "extract the best plays from this football game",
                "object_focused": "focus on the ball movements in tennis",
                "person_focused": "track the main character throughout the scene"
            },
            "object_detection_examples": {
                "basketball": "get the goals in this basketball match",
                "football": "capture touchdown moments",
                "soccer": "find all goal scoring moments",
                "tennis": "extract ace serves and winning shots",
                "baseball": "capture home run moments",
                "general_sports": "track the ball and key players"
            },
            "ai_reframing_note": "Set 'ai_reframe': true to enable intelligent cropping based on detected objects",
            "usage_note": "Use these themes in the user_prompt field to get theme-specific content selection with object detection"
        }
    
    except Exception as e:
        logger.error(f"Failed to get supported themes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get supported themes: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    # Check Redis connection
    redis_status = "ok"
    try:
        redis_client.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        redis_status = f"error: {str(e)}"
    
    # Check Celery connection (broker only)
    celery_status = "ok"
    try:
        celery_app.control.ping(timeout=1.0)
    except Exception as e:
        logger.error(f"Celery broker health check failed: {e}")
        celery_status = f"error: {str(e)}"
    
    return {
        "status": "ok",
        "service": "video-processing-api",
        "components": {
            "redis": redis_status,
            "celery_broker": celery_status
        }
    }

def start_api():
    """Start the FastAPI server."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_api()
