"""
Ultra-high-performance configuration specifically optimized for NVIDIA A6000 Ada GPU.

This configuration maximizes GPU utilization and processing speed while maintaining accuracy.
"""
import torch

# A6000 Ada specific optimizations
A6000_ADA_CONFIG = {
    # GPU Optimization
    "device": "cuda",
    "gpu_memory_fraction": 0.8,  # Use 80% of GPU memory
    "enable_mixed_precision": True,
    "enable_tensorrt": True,
    "enable_cuda_streams": True,
    "enable_gpu_pipeline": True,
    
    # Detection Optimization (Aggressive for A6000)
    "detection_interval": 8,  # Skip more frames for speed
    "batch_size": 24,         # Large batch for A6000's memory
    "max_workers": 12,        # Utilize multiple CPU cores
    
    # Frame Processing Optimization
    "detection_frame_max_width": 1920,  # Keep high res for accuracy
    "processing_frame_parallel": True,
    "gpu_frame_preprocessing": True,
    
    # MediaPipe Optimization
    "mediapipe_model_complexity": 0,  # Fastest model
    "mediapipe_max_num_faces": 4,
    "mediapipe_min_detection_confidence": 0.3,  # Lower for speed
    "mediapipe_min_tracking_confidence": 0.3,
    
    # Memory Optimization
    "enable_memory_pool": True,
    "prealloc_frame_buffers": True,
    "frame_buffer_count": 32,
    
    # Threading Optimization
    "detection_thread_priority": "high",
    "processing_thread_priority": "normal",
    "io_thread_priority": "low",
    
    # Quality vs Speed Balance (Optimized for A6000)
    "speaker_detection_threshold": 0.6,  # Slightly lower for speed
    "interpolation_quality": "linear",   # Fastest
    "enable_smart_caching": True,
    "cache_detection_results": True,
    
    # Video Processing Optimization
    "video_decode_threads": 4,
    "video_encode_threads": 8,
    "enable_hardware_decode": True,
    "enable_hardware_encode": True,
    
    # Advanced GPU Features
    "enable_tensor_cores": True,
    "enable_cuda_graphs": True,
    "enable_async_processing": True,
}

# Ultra-fast preset for A6000 Ada
A6000_ULTRA_FAST_CONFIG = {
    **A6000_ADA_CONFIG,
    "detection_interval": 12,  # Even more aggressive
    "batch_size": 32,
    "max_workers": 16,
    "detection_frame_max_width": 1280,
    "speaker_detection_threshold": 0.5,
    "mediapipe_min_detection_confidence": 0.2,
    "frame_buffer_count": 48,
}

# Balanced high-performance preset
A6000_BALANCED_CONFIG = {
    **A6000_ADA_CONFIG,
    "detection_interval": 6,
    "batch_size": 16,
    "max_workers": 8,
    "speaker_detection_threshold": 0.7,
    "mediapipe_min_detection_confidence": 0.4,
}

# Maximum quality preset (still fast on A6000)
A6000_QUALITY_CONFIG = {
    **A6000_ADA_CONFIG,
    "detection_interval": 4,
    "batch_size": 12,
    "max_workers": 6,
    "detection_frame_max_width": 1920,
    "speaker_detection_threshold": 0.8,
    "mediapipe_min_detection_confidence": 0.5,
    "interpolation_quality": "cubic",
    "mediapipe_model_complexity": 1,
}

# Quality accelerated preset (high accuracy with better performance)
A6000_QUALITY_ACCELERATED_CONFIG = {
    **A6000_ADA_CONFIG,
    "detection_interval": 4,  # Slightly more aggressive than quality
    "batch_size": 20,         # Larger batch for better GPU utilization
    "max_workers": 10,        # More workers than quality preset
    "detection_frame_max_width": 1920,  # Keep high resolution
    "speaker_detection_threshold": 0.75,  # Slightly lower than quality for speed
    "mediapipe_min_detection_confidence": 0.45,  # Balanced confidence
    "interpolation_quality": "cubic",  # Keep cubic for quality
    "mediapipe_model_complexity": 1,   # Keep model complexity for accuracy
    "frame_buffer_count": 40,          # More buffers for smoother processing
    "enable_smart_caching": True,      # Enhanced caching
    "cache_detection_results": True,   # Cache results for efficiency
}


def get_a6000_config(preset: str = "ultra_fast") -> dict:
    """
    Get A6000 Ada optimized configuration.
    
    Args:
        preset: Configuration preset ('ultra_fast', 'balanced', 'quality', 'quality_accelerated', 'default')
        
    Returns:
        Optimized configuration dictionary
    """
    configs = {
        "ultra_fast": A6000_ULTRA_FAST_CONFIG,
        "balanced": A6000_BALANCED_CONFIG,
        "quality": A6000_QUALITY_CONFIG,
        "quality_accelerated": A6000_QUALITY_ACCELERATED_CONFIG,
        "default": A6000_ADA_CONFIG
    }
    
    return configs.get(preset, A6000_ADA_CONFIG).copy()


def check_a6000_availability() -> dict:
    """
    Check if A6000 or similar high-end GPU is available.
    
    Returns:
        Dictionary with GPU information
    """
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA not available"}
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    
    # Check for high-end GPUs
    high_end_gpus = ["A6000", "A100", "V100", "RTX 4090", "RTX 4080", "RTX 3090"]
    is_high_end = any(gpu in gpu_name for gpu in high_end_gpus)
    
    return {
        "available": True,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "is_high_end": is_high_end,
        "recommended_preset": "ultra_fast" if is_high_end and gpu_memory >= 24 else "balanced"
    }


def estimate_a6000_performance(video_duration_seconds: float, fps: float, preset: str = "ultra_fast") -> dict:
    """
    Estimate processing performance on A6000 Ada.
    
    Args:
        video_duration_seconds: Video duration in seconds
        fps: Frames per second
        preset: Configuration preset
        
    Returns:
        Performance estimates
    """
    config = get_a6000_config(preset)
    
    total_frames = video_duration_seconds * fps
    frames_to_detect = total_frames / config["detection_interval"]
    
    # A6000 performance estimates (much faster than general estimates)
    detection_time_per_frame = 0.02  # Very fast with GPU acceleration
    processing_time_per_frame = 0.005  # GPU-accelerated processing
    
    estimated_detection_time = frames_to_detect * detection_time_per_frame
    estimated_processing_time = total_frames * processing_time_per_frame
    estimated_total_time = estimated_detection_time + estimated_processing_time
    
    # Baseline comparison (CPU-only processing)
    baseline_time = total_frames * 0.15
    speedup_factor = baseline_time / estimated_total_time
    
    # Expected processing FPS
    processing_fps = total_frames / estimated_total_time
    
    return {
        "estimated_total_time_minutes": estimated_total_time / 60,
        "estimated_detection_time_minutes": estimated_detection_time / 60,
        "estimated_processing_time_minutes": estimated_processing_time / 60,
        "baseline_time_minutes": baseline_time / 60,
        "speedup_factor": speedup_factor,
        "processing_fps": processing_fps,
        "total_frames": int(total_frames),
        "frames_to_detect": int(frames_to_detect),
        "preset_used": preset
    }
