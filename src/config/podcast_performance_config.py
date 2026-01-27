"""
Performance configuration for optimized podcast template processing.

Adjust these settings based on your hardware capabilities and quality requirements.
"""

# Performance optimization settings for podcast template processing
PODCAST_PERFORMANCE_CONFIG = {
    # Detection Optimization
    "detection_interval": 5,  # Process every Nth frame for detection
                             # Higher = faster but less accurate tracking
                             # Recommended: 3-10 (3=best quality, 10=fastest)
    
    # Parallel Processing
    "batch_size": 8,         # Number of frames to process in parallel
                             # Higher = more memory usage but faster processing
                             # Recommended: 4-16 based on available RAM
    
    "max_workers": 4,        # Number of detection threads
                             # Should match CPU cores available
                             # Recommended: 2-8 based on CPU
    
    # Quality Settings
    "speaker_detection_threshold": 0.7,  # Confidence threshold for speaker detection
                                        # Lower = more detections but less accurate
                                        # Higher = fewer detections but more accurate
    
    # Memory Optimization
    "detection_frame_max_width": 1280,  # Downscale frames for detection if larger
                                       # Smaller = faster detection but less accurate
                                       # Recommended: 1280-1920
    
    # Interpolation Settings
    "interpolation_quality": "linear",   # OpenCV interpolation method
                                        # Options: "linear", "cubic", "area"
                                        # linear = fastest, cubic = best quality
}

# Preset configurations for different use cases
PERFORMANCE_PRESETS = {
    "maximum_quality": {
        "detection_interval": 2,
        "batch_size": 4,
        "max_workers": 2,
        "speaker_detection_threshold": 0.8,
        "detection_frame_max_width": 1920,
        "interpolation_quality": "cubic"
    },
    
    "balanced": {
        "detection_interval": 5,
        "batch_size": 8,
        "max_workers": 4,
        "speaker_detection_threshold": 0.7,
        "detection_frame_max_width": 1280,
        "interpolation_quality": "linear"
    },
    
    "maximum_speed": {
        "detection_interval": 10,
        "batch_size": 16,
        "max_workers": 8,
        "speaker_detection_threshold": 0.6,
        "detection_frame_max_width": 960,
        "interpolation_quality": "linear"
    },
    
    "ultra_fast": {
        "detection_interval": 15,
        "batch_size": 20,
        "max_workers": 8,
        "speaker_detection_threshold": 0.5,
        "detection_frame_max_width": 720,
        "interpolation_quality": "area"
    }
}

# Hardware-specific recommendations
HARDWARE_RECOMMENDATIONS = {
    "cpu_cores_2_4": {
        "max_workers": 2,
        "batch_size": 4,
        "detection_interval": 8
    },
    
    "cpu_cores_4_8": {
        "max_workers": 4,
        "batch_size": 8,
        "detection_interval": 5
    },
    
    "cpu_cores_8_plus": {
        "max_workers": 8,
        "batch_size": 16,
        "detection_interval": 5
    },
    
    "gpu_available": {
        "detection_frame_max_width": 1920,
        "speaker_detection_threshold": 0.8,
        "batch_size": 16
    },
    
    "limited_memory": {
        "batch_size": 4,
        "detection_frame_max_width": 960,
        "max_workers": 2
    }
}


def get_optimal_config(preset_name: str = "balanced") -> dict:
    """
    Get optimal configuration based on preset.
    
    Args:
        preset_name: Name of the preset to use
        
    Returns:
        Configuration dictionary
    """
    if preset_name in PERFORMANCE_PRESETS:
        return PERFORMANCE_PRESETS[preset_name].copy()
    else:
        return PODCAST_PERFORMANCE_CONFIG.copy()


def get_hardware_optimized_config(hardware_type: str = "cpu_cores_4_8") -> dict:
    """
    Get hardware-optimized configuration.
    
    Args:
        hardware_type: Type of hardware configuration
        
    Returns:
        Configuration dictionary optimized for hardware
    """
    base_config = PODCAST_PERFORMANCE_CONFIG.copy()
    
    if hardware_type in HARDWARE_RECOMMENDATIONS:
        base_config.update(HARDWARE_RECOMMENDATIONS[hardware_type])
    
    return base_config


def estimate_processing_time(video_duration_seconds: float, 
                           fps: float, 
                           detection_interval: int = 5) -> dict:
    """
    Estimate processing time based on video properties and settings.
    
    Args:
        video_duration_seconds: Duration of video in seconds
        fps: Frames per second of video
        detection_interval: Detection interval setting
        
    Returns:
        Dictionary with time estimates
    """
    total_frames = video_duration_seconds * fps
    frames_to_detect = total_frames / detection_interval
    
    # Rough estimates based on performance testing
    # These will vary based on hardware
    detection_time_per_frame = 0.1  # seconds per frame for detection
    processing_time_per_frame = 0.02  # seconds per frame for processing
    
    estimated_detection_time = frames_to_detect * detection_time_per_frame
    estimated_processing_time = total_frames * processing_time_per_frame
    estimated_total_time = estimated_detection_time + estimated_processing_time
    
    # Original implementation estimate (for comparison)
    original_estimated_time = total_frames * 0.15  # Much slower
    
    speedup_factor = original_estimated_time / estimated_total_time
    
    return {
        "estimated_total_time_minutes": estimated_total_time / 60,
        "estimated_detection_time_minutes": estimated_detection_time / 60,
        "estimated_processing_time_minutes": estimated_processing_time / 60,
        "original_estimated_time_minutes": original_estimated_time / 60,
        "estimated_speedup_factor": speedup_factor,
        "total_frames": int(total_frames),
        "frames_to_detect": int(frames_to_detect)
    }
