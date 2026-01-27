# src/utils/cache_manager.py
"""Cache management for storing processing results and avoiding recomputation"""

import json
import hashlib
import pickle
import os
from pathlib import Path
from typing import Any, Optional, Dict
import logging
import time
from datetime import datetime, timedelta


class CacheManager:
    """
    Manages caching of processing results to avoid recomputation.
    Supports both JSON and pickle serialization.
    """
    
    def __init__(self, cache_dir: str = "cache", max_age_days: int = 30):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for storing cache files
            max_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        
        # Create subdirectories for different cache types
        self.json_cache_dir = self.cache_dir / "json"
        self.pickle_cache_dir = self.cache_dir / "pickle"
        self.temp_cache_dir = self.cache_dir / "temp"
        
        for dir_path in [self.json_cache_dir, self.pickle_cache_dir, self.temp_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'errors': 0
        }
        
        self.logger.info(f"CacheManager initialized with directory: {self.cache_dir}")
    
    def get_cached_result(self, cache_key: str, cache_type: str = "json") -> Optional[Any]:
        """
        Get cached result by key.
        
        Args:
            cache_key: Unique identifier for the cached item
            cache_type: Type of cache ('json' or 'pickle')
            
        Returns:
            Cached result or None if not found or expired
        """
        try:
            cache_file = self._get_cache_file_path(cache_key, cache_type)
            
            if not cache_file.exists():
                self.stats['misses'] += 1
                return None
            
            # Check if cache is expired
            if self._is_cache_expired(cache_file):
                self.logger.debug(f"Cache expired for key: {cache_key}")
                cache_file.unlink()  # Remove expired cache
                self.stats['misses'] += 1
                return None
            
            # Load cached data
            if cache_type == "json":
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:  # pickle
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
            
            self.stats['hits'] += 1
            self.logger.debug(f"Cache hit for key: {cache_key}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading cache for key {cache_key}: {e}")
            self.stats['errors'] += 1
            return None
    
    def cache_result(self, cache_key: str, data: Any, cache_type: str = "json") -> bool:
        """
        Cache a result.
        
        Args:
            cache_key: Unique identifier for the cached item
            data: Data to cache
            cache_type: Type of cache ('json' or 'pickle')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_file = self._get_cache_file_path(cache_key, cache_type)
            
            # Create metadata
            metadata = {
                'cached_at': datetime.now().isoformat(),
                'cache_key': cache_key,
                'data_type': type(data).__name__
            }
            
            if cache_type == "json":
                # Try to serialize as JSON
                cache_data = {
                    'metadata': metadata,
                    'data': data
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, indent=2, ensure_ascii=False)
            else:  # pickle
                cache_data = {
                    'metadata': metadata,
                    'data': data
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            self.stats['writes'] += 1
            self.logger.debug(f"Cached result for key: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching result for key {cache_key}: {e}")
            self.stats['errors'] += 1
            return False
    
    def _get_cache_file_path(self, cache_key: str, cache_type: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            cache_type: Type of cache
            
        Returns:
            Path to cache file
        """
        # Create safe filename from cache key
        safe_key = hashlib.md5(cache_key.encode()).hexdigest()
        
        if cache_type == "json":
            return self.json_cache_dir / f"{safe_key}.json"
        else:
            return self.pickle_cache_dir / f"{safe_key}.pkl"
    
    def _is_cache_expired(self, cache_file: Path) -> bool:
        """
        Check if cache file is expired.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            True if expired, False otherwise
        """
        if not cache_file.exists():
            return True
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age > timedelta(days=self.max_age_days)
    
    def invalidate_cache(self, cache_key: str, cache_type: str = "json") -> bool:
        """
        Invalidate (delete) a cached result.
        
        Args:
            cache_key: Cache key to invalidate
            cache_type: Type of cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cache_file = self._get_cache_file_path(cache_key, cache_type)
            
            if cache_file.exists():
                cache_file.unlink()
                self.logger.debug(f"Invalidated cache for key: {cache_key}")
                return True
            else:
                self.logger.debug(f"Cache key not found for invalidation: {cache_key}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error invalidating cache for key {cache_key}: {e}")
            return False
    
    def clear_cache(self, cache_type: str = None) -> Dict[str, int]:
        """
        Clear cache files.
        
        Args:
            cache_type: Specific cache type to clear ('json' or 'pickle'), or None for all
            
        Returns:
            Dictionary with count of cleared files
        """
        cleared = {'json': 0, 'pickle': 0, 'temp': 0}
        
        try:
            if cache_type is None or cache_type == "json":
                cleared['json'] = self._clear_directory(self.json_cache_dir)
            
            if cache_type is None or cache_type == "pickle":
                cleared['pickle'] = self._clear_directory(self.pickle_cache_dir)
            
            if cache_type is None:
                cleared['temp'] = self._clear_directory(self.temp_cache_dir)
            
            self.logger.info(f"Cache cleared: {cleared}")
            return cleared
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return cleared
    
    def _clear_directory(self, directory: Path) -> int:
        """
        Clear all files in a directory.
        
        Args:
            directory: Directory to clear
            
        Returns:
            Number of files cleared
        """
        count = 0
        if directory.exists():
            for file_path in directory.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
        return count
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        Remove expired cache files.
        
        Returns:
            Dictionary with count of removed files
        """
        removed = {'json': 0, 'pickle': 0}
        
        try:
            # Check JSON cache
            for cache_file in self.json_cache_dir.iterdir():
                if cache_file.is_file() and self._is_cache_expired(cache_file):
                    cache_file.unlink()
                    removed['json'] += 1
            
            # Check pickle cache
            for cache_file in self.pickle_cache_dir.iterdir():
                if cache_file.is_file() and self._is_cache_expired(cache_file):
                    cache_file.unlink()
                    removed['pickle'] += 1
            
            self.logger.info(f"Expired cache cleaned: {removed}")
            return removed
            
        except Exception as e:
            self.logger.error(f"Error cleaning expired cache: {e}")
            return removed
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count cache files
        json_count = len(list(self.json_cache_dir.glob("*.json")))
        pickle_count = len(list(self.pickle_cache_dir.glob("*.pkl")))
        
        # Calculate cache directory size
        json_size = sum(f.stat().st_size for f in self.json_cache_dir.glob("*.json"))
        pickle_size = sum(f.stat().st_size for f in self.pickle_cache_dir.glob("*.pkl"))
        
        # Calculate hit rate
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'cache_directory': str(self.cache_dir),
            'file_counts': {
                'json': json_count,
                'pickle': pickle_count,
                'total': json_count + pickle_count
            },
            'cache_sizes': {
                'json_bytes': json_size,
                'pickle_bytes': pickle_size,
                'total_bytes': json_size + pickle_size,
                'json_mb': json_size / (1024 * 1024),
                'pickle_mb': pickle_size / (1024 * 1024),
                'total_mb': (json_size + pickle_size) / (1024 * 1024)
            },
            'access_stats': {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'writes': self.stats['writes'],
                'errors': self.stats['errors'],
                'hit_rate': hit_rate
            },
            'max_age_days': self.max_age_days
        }
    
    def create_cache_key(self, *args, **kwargs) -> str:
        """
        Create a cache key from arguments.
        
        Args:
            *args: Arguments to include in key
            **kwargs: Keyword arguments to include in key
            
        Returns:
            Cache key string
        """
        # Create a string representation of all arguments
        key_parts = []
        
        # Add positional arguments
        for arg in args:
            if hasattr(arg, '__dict__'):
                # For objects, use their string representation
                key_parts.append(str(arg))
            else:
                key_parts.append(str(arg))
        
        # Add keyword arguments
        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}={value}")
        
        # Join all parts and create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_temp_file_path(self, prefix: str = "temp", suffix: str = ".tmp") -> Path:
        """
        Get a temporary file path in the temp cache directory.
        
        Args:
            prefix: Prefix for temporary file name
            suffix: Suffix for temporary file name
            
        Returns:
            Path to temporary file
        """
        timestamp = str(int(time.time() * 1000))
        filename = f"{prefix}_{timestamp}{suffix}"
        return self.temp_cache_dir / filename
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files cleaned up
        """
        count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        try:
            for temp_file in self.temp_cache_dir.iterdir():
                if temp_file.is_file():
                    file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        temp_file.unlink()
                        count += 1
            
            self.logger.info(f"Cleaned up {count} temporary files")
            return count
            
        except Exception as e:
            self.logger.error(f"Error cleaning temporary files: {e}")
            return count
