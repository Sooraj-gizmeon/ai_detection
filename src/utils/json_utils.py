"""
JSON serialization utilities for handling NumPy types and other non-serializable objects.

This module provides utilities to convert NumPy types and other objects to JSON-serializable
formats, preventing serialization errors in Celery tasks and Redis storage.
"""

import json
import numpy as np
from typing import Any, Dict, List, Union


def convert_numpy_types(obj):
    """
    Convert NumPy types and dataclasses to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types or dataclasses
        
    Returns:
        Object with NumPy types and dataclasses converted to Python native types
    """
    # Handle dataclasses with to_dict() method first
    if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
        return convert_numpy_types(obj.to_dict())
    # Handle dataclasses without to_dict() method
    elif hasattr(obj, '__dataclass_fields__'):
        return {field.name: convert_numpy_types(getattr(obj, field.name)) 
                for field in obj.__dataclass_fields__.values()}
    # Handle NumPy types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def safe_json_serialize(obj):
    """
    Safely serialize an object to JSON, handling NumPy types and other problematic types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    try:
        # First try direct serialization
        return json.dumps(obj, default=str)
    except TypeError:
        # If that fails, convert NumPy types and try again
        converted_obj = convert_numpy_types(obj)
        return json.dumps(converted_obj, default=str)


def make_json_serializable(data: Any) -> Any:
    """
    Make any data structure JSON serializable by converting problematic types.
    
    This function recursively traverses data structures and converts:
    - NumPy integers to Python int
    - NumPy floats to Python float
    - NumPy arrays to Python lists
    - NumPy booleans to Python bool
    - Dataclasses to dictionaries (using to_dict() if available)
    
    Args:
        data: Any data structure that may contain NumPy types or dataclasses
        
    Returns:
        Data structure with all types converted to JSON-serializable equivalents
    """
    return convert_numpy_types(data)


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that can handle NumPy types and dataclasses.
    """
    def default(self, obj):
        # Handle dataclasses with to_dict() method first
        if hasattr(obj, 'to_dict') and callable(getattr(obj, 'to_dict')):
            return obj.to_dict()
        # Handle dataclasses without to_dict() method
        elif hasattr(obj, '__dataclass_fields__'):
            return {field.name: getattr(obj, field.name) 
                    for field in obj.__dataclass_fields__.values()}
        # Handle NumPy types
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def clean_for_serialization(result: Dict) -> Dict:
    """
    Clean a result dictionary for safe JSON serialization.
    
    This is specifically designed for video processing results that may contain
    NumPy types from YOLO object detection, audio analysis, etc.
    
    Args:
        result: Result dictionary from video processing
        
    Returns:
        Cleaned result dictionary safe for JSON serialization
    """
    try:
        # Convert the entire result structure
        cleaned_result = make_json_serializable(result)
        
        # Validate that it's actually serializable
        json.dumps(cleaned_result)
        
        return cleaned_result
    except Exception as e:
        # If cleaning still fails, create a safe fallback
        return {
            'status': 'error',
            'error': f'Serialization cleaning failed: {str(e)}',
            'original_status': result.get('status', 'unknown'),
            'fallback_applied': True
        }
