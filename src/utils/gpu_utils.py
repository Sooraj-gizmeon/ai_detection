# src/utils/gpu_utils.py
"""GPU utilities for optimization and device management"""

import torch
import logging
from typing import Dict, Optional, List
import subprocess
import json
import psutil


def get_device_info() -> Dict:
    """
    Get information about available GPU devices.
    
    Returns:
        Dictionary with GPU device information
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'current_device': None,
        'devices': [],
        'cuda_version': None,
        'cudnn_version': None,
        'total_memory': 0,
        'free_memory': 0
    }
    
    try:
        if torch.cuda.is_available():
            device_info['device_count'] = torch.cuda.device_count()
            device_info['current_device'] = torch.cuda.current_device()
            device_info['cuda_version'] = torch.version.cuda
            device_info['cudnn_version'] = torch.backends.cudnn.version()
            
            # Get information for each device
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                device_data = {
                    'index': i,
                    'name': device_props.name,
                    'total_memory': device_props.total_memory,
                    'major': device_props.major,
                    'minor': device_props.minor,
                    'multi_processor_count': device_props.multi_processor_count,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_cached': torch.cuda.memory_reserved(i),
                    'memory_free': device_props.total_memory - torch.cuda.memory_allocated(i)
                }
                device_info['devices'].append(device_data)
            
            # Current device memory info
            if device_info['devices']:
                current_device = device_info['devices'][device_info['current_device']]
                device_info['total_memory'] = current_device['total_memory']
                device_info['free_memory'] = current_device['memory_free']
    
    except Exception as e:
        logging.error(f"Error getting GPU info: {e}")
    
    return device_info


def optimize_gpu_memory(device: Optional[str] = None) -> bool:
    """
    Optimize GPU memory usage by clearing cache and setting memory fraction.
    
    Args:
        device: Specific device to optimize (None for current device)
        
    Returns:
        True if optimization was successful
    """
    try:
        if not torch.cuda.is_available():
            return False
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Set memory fraction to prevent out-of-memory errors
        if device is not None:
            torch.cuda.set_device(device)
        
        # Get available memory
        total_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        
        # Calculate optimal memory fraction (leave 20% buffer)
        memory_fraction = min(0.8, (total_memory - allocated_memory) / total_memory)
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        logging.info(f"GPU memory optimized: {memory_fraction:.2%} allocation")
        return True
        
    except Exception as e:
        logging.error(f"Error optimizing GPU memory: {e}")
        return False


def get_optimal_batch_size(model_size: str, available_memory: int) -> int:
    """
    Calculate optimal batch size based on model size and available memory.
    
    Args:
        model_size: Size of the model ('small', 'medium', 'large')
        available_memory: Available GPU memory in bytes
        
    Returns:
        Optimal batch size
    """
    # Memory requirements per batch item (rough estimates)
    memory_per_batch = {
        'small': 512 * 1024 * 1024,    # 512MB
        'medium': 1024 * 1024 * 1024,  # 1GB
        'large': 2048 * 1024 * 1024,   # 2GB
    }
    
    base_memory = memory_per_batch.get(model_size, 1024 * 1024 * 1024)
    
    # Calculate batch size with safety margin
    safety_margin = 0.7  # Use 70% of available memory
    usable_memory = available_memory * safety_margin
    
    batch_size = max(1, int(usable_memory // base_memory))
    
    # Cap batch size for practical reasons
    return min(batch_size, 32)


def monitor_gpu_usage() -> Dict:
    """
    Monitor current GPU usage.
    
    Returns:
        Dictionary with GPU usage statistics
    """
    usage_stats = {
        'gpu_utilization': 0.0,
        'memory_utilization': 0.0,
        'temperature': 0.0,
        'power_usage': 0.0,
        'processes': []
    }
    
    try:
        # Try to get nvidia-smi output
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                values = line.split(', ')
                if len(values) >= 4:
                    usage_stats['gpu_utilization'] = float(values[0])
                    memory_used = float(values[1])
                    memory_total = float(values[2])
                    usage_stats['memory_utilization'] = (memory_used / memory_total) * 100
                    usage_stats['temperature'] = float(values[3])
                    if len(values) >= 5 and values[4] != '[N/A]':
                        usage_stats['power_usage'] = float(values[4])
        
        # Get GPU processes
        result = subprocess.run([
            'nvidia-smi', '--query-compute-apps=pid,name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    values = line.split(', ')
                    if len(values) >= 3:
                        usage_stats['processes'].append({
                            'pid': int(values[0]),
                            'name': values[1],
                            'memory_used': int(values[2])
                        })
    
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available or failed
        pass
    
    return usage_stats


def get_recommended_device() -> str:
    """
    Get recommended device for processing.
    
    Returns:
        Recommended device string ('cuda' or 'cpu')
    """
    if not torch.cuda.is_available():
        return 'cpu'
    
    # Check if GPU has enough memory
    device_info = get_device_info()
    
    if device_info['devices']:
        # Find device with most free memory
        best_device = max(device_info['devices'], key=lambda d: d['memory_free'])
        
        # Require at least 2GB free memory for GPU processing
        if best_device['memory_free'] > 2 * 1024 * 1024 * 1024:
            return 'cuda'
    
    return 'cpu'


def setup_gpu_for_inference(memory_fraction: float = 0.8) -> bool:
    """
    Setup GPU for inference with optimal settings.
    
    Args:
        memory_fraction: Fraction of GPU memory to use
        
    Returns:
        True if setup was successful
    """
    try:
        if not torch.cuda.is_available():
            return False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # Set optimal settings for inference
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        logging.info(f"GPU setup completed with {memory_fraction:.1%} memory allocation")
        return True
        
    except Exception as e:
        logging.error(f"Error setting up GPU: {e}")
        return False


def cleanup_gpu_memory():
    """Clean up GPU memory and reset cache."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logging.info("GPU memory cleaned up")
    except Exception as e:
        logging.error(f"Error cleaning up GPU memory: {e}")


def check_gpu_compatibility() -> Dict:
    """
    Check GPU compatibility for the pipeline.
    
    Returns:
        Dictionary with compatibility information
    """
    compatibility = {
        'cuda_available': torch.cuda.is_available(),
        'compatible': False,
        'warnings': [],
        'recommendations': []
    }
    
    if not torch.cuda.is_available():
        compatibility['warnings'].append("CUDA not available")
        compatibility['recommendations'].append("Install CUDA-enabled PyTorch")
        return compatibility
    
    device_info = get_device_info()
    
    # Check compute capability
    for device in device_info['devices']:
        compute_capability = f"{device['major']}.{device['minor']}"
        
        if device['major'] < 6:
            compatibility['warnings'].append(f"GPU {device['name']} has old compute capability {compute_capability}")
            compatibility['recommendations'].append("Consider upgrading GPU for better performance")
        
        # Check memory
        memory_gb = device['total_memory'] / (1024**3)
        if memory_gb < 4:
            compatibility['warnings'].append(f"GPU {device['name']} has limited memory: {memory_gb:.1f}GB")
            compatibility['recommendations'].append("Consider using CPU or upgrading GPU")
        elif memory_gb >= 8:
            compatibility['compatible'] = True
    
    return compatibility


def get_system_info() -> Dict:
    """
    Get comprehensive system information.
    
    Returns:
        Dictionary with system information
    """
    system_info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'memory_percent': psutil.virtual_memory().percent,
        'disk_usage': psutil.disk_usage('/').percent,
        'gpu_info': get_device_info(),
        'gpu_compatibility': check_gpu_compatibility()
    }
    
    return system_info
