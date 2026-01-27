# config/smart_zoom_settings.py
"""Smart Zoom Configuration Settings"""

SMART_ZOOM_CONFIG = {
    "detection": {
        "face_confidence": 0.7,
        "person_confidence": 0.8,
        "object_confidence": 0.6,
        "tracking_persistence": 10,  # frames
    },
    
    "framing": {
        "vertical_aspect_ratio": (4, 5),
        "subject_padding_ratio": 0.15,
        "min_subject_size": 0.3,  # of frame height
        "max_subject_size": 0.8,  # of frame height
    },
    
    "zoom_behavior": {
        "zoom_speed": 0.0,  # No zoom per frame
        "max_zoom_per_second": 0.0,  # No zooming allowed
        "stability_threshold": 0.0,  # No zoom change
        "smoothing_factor": 1.0,  # No smoothing needed
        "fixed_zoom_level": 1.0,  # Always use 1.0 (no zoom)
    },
    
    "content_based_zoom": {
        "close_up_keywords": ["important", "key", "focus", "listen"],
        "wide_shot_keywords": ["everyone", "all", "group", "together"],
        "action_keywords": ["move", "show", "demonstrate", "look"],
    },
    
    # Vision enhancement settings
    "vision_analysis": {
        "enabled": True,
        "max_frames_per_segment": 2,
        "frame_target_size": (512, 288),
        "vision_analysis_timeout": 30.0,
        "quality_threshold": 0.6,
        "efficient_mode": True,
        "cache_results": True,
        "batch_processing": True,
        "fallback_to_audio_only": True
    }
}

# Model configurations
MODEL_CONFIG = {
    "yolo": {
        "model_path": "yolov8n.pt",
        "device": "cuda",
        "conf_threshold": 0.5,
        "iou_threshold": 0.45,
        "enabled": True,  # Set to False to disable YOLO object detection
    },
    "mediapipe": {
        "face_detection": {
            "model_selection": 0,
            "min_detection_confidence": 0.5,
        },
        "pose_detection": {
            "model_complexity": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
        },
    },
    "whisper": {
        "model_size": "base",
        "device": "cuda",
        "compute_type": "float16",
    },
}

# Quality standards
QUALITY_STANDARDS = {
    "subject_visibility": {
        "min_size_ratio": 0.3,
        "max_size_ratio": 0.8,
        "padding_ratio": 0.15,
    },
    "stability": {
        "max_zoom_change_per_frame": 0.02,
        "smoothing_window": 5,
    },
    "composition": {
        "rule_of_thirds": True,
        "eye_level_ratio": 0.5,  # Eyes at 50% from top
        "headroom_ratio": 0.1,   # 10% headroom
    },
}

# Ollama integration
OLLAMA_CONFIG = {
    "base_url": "http://localhost:11434",
    "timeout": 30,  # Default timeout in seconds
    "max_retries": 3,  # Increased retries with smart backoff
    "retry_delay": 1,  # Initial delay between retries
    "fast_mode": True,  # Use shorter, simpler prompts for faster processing
    "skip_on_error": True,  # Continue with basic processing if AI analysis fails
    
    # Enhanced request configuration
    "request_settings": {
        "temperature": 0.1,  # Low temperature for deterministic responses
        "num_predict": 768,  # Default token limit for responses
        "num_ctx": 4096,     # Default context window size
        "keep_alive": "20m",  # Keep model loaded for 20 minutes
    },
    
    # Connection optimization
    "connection_settings": {
        "connection_limit": 20,       # Max concurrent connections
        "connection_limit_per_host": 10,  # Max connections per host
        "keepalive_timeout": 60.0,    # Keep connections alive for 60s
    },
    
    # Streaming configuration
    "streaming": {
        "enabled": False,           # Enable streaming for large requests
        "large_request_threshold": 5000,  # Characters threshold for streaming
        "very_large_threshold": 15000,    # Threshold for optimized processing
        "max_streaming_timeout": 180.0,   # Maximum timeout for streaming (3 min)
    },
    
    # Performance optimization
    "performance": {
        "batch_processing": True,  # Enable batch processing
        "batch_size": 5,           # Process segments in batches of 5
        "preload_models": True,    # Preload models at startup
        "adaptive_timeouts": True, # Scale timeouts with request complexity
    },
    
    # Model configuration for different tasks
    "models": {
        "text": "mistral-small3.2:latest",     # Default text model
        "analysis": "mistral-small3.2:latest", # Content analysis model
        "vision": "llava:latest",             # Vision-capable model (install with: ollama pull llava:latest)
        "fallback": "mistral-small3.2:latest" # Fallback if specific models unavailable
    },
    
    # Model selection
    "models": {
        "vision": "mistral-small3.2:latest",  # Default vision model
        "analysis": "mistral-small3.2:latest", # Default analysis model
        "text": "mistral-small3.2:latest",     # Default text model
    },
    
    # Error handling
    "error_handling": {
        "fallback_parsing": True,   # Use advanced fallback for JSON parsing
        "default_responses": True,  # Provide default responses on failure
        "collect_diagnostics": True, # Collect detailed error diagnostics
    }
}

# Processing optimization
PROCESSING_CONFIG = {
    "batch_size": 32,
    "max_workers": 8,
    "gpu_memory_fraction": 0.8,
    "temp_cleanup": True,
    "cache_results": True,
}
