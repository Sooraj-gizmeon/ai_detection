# src/utils/file_utils.py
"""File management utilities"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
import logging
import hashlib
import json


def ensure_directory(directory: Path) -> bool:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"Failed to create directory {directory}: {e}")
        return False


def clean_temp_files(temp_dir: str, max_age_hours: int = 24) -> int:
    """
    Clean temporary files older than specified age.
    
    Args:
        temp_dir: Temporary directory path
        max_age_hours: Maximum age in hours
        
    Returns:
        Number of files cleaned
    """
    import time
    from datetime import datetime, timedelta
    
    cleaned_count = 0
    cutoff_time = time.time() - (max_age_hours * 3600)
    
    try:
        temp_path = Path(temp_dir)
        if not temp_path.exists():
            return 0
        
        for file_path in temp_path.rglob("*"):
            if file_path.is_file():
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_count += 1
        
        logging.info(f"Cleaned {cleaned_count} temporary files")
        return cleaned_count
        
    except Exception as e:
        logging.error(f"Error cleaning temporary files: {e}")
        return 0


def get_file_hash(file_path: str) -> str:
    """
    Get MD5 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logging.error(f"Failed to calculate hash for {file_path}: {e}")
        return ""


def save_json(data: dict, file_path: str) -> bool:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Data to save
        file_path: Output file path
        
    Returns:
        True if successful
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[dict]:
    """
    Load JSON file as dictionary.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON from {file_path}: {e}")
        return None


def get_directory_size(directory: str) -> int:
    """
    Get total size of directory in bytes.
    
    Args:
        directory: Directory path
        
    Returns:
        Size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logging.error(f"Failed to calculate directory size: {e}")
    
    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def find_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """
    Find all files with specified extensions in directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.mp4', '.avi'])
        
    Returns:
        List of file paths
    """
    files = []
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return files
        
        for ext in extensions:
            files.extend([str(f) for f in dir_path.rglob(f"*{ext}")])
        
        return sorted(files)
        
    except Exception as e:
        logging.error(f"Failed to find files: {e}")
        return []


def create_backup(source_file: str, backup_dir: str = "backups") -> Optional[str]:
    """
    Create backup of a file.
    
    Args:
        source_file: Source file path
        backup_dir: Backup directory
        
    Returns:
        Backup file path or None if failed
    """
    try:
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        source_path = Path(source_file)
        if not source_path.exists():
            return None
        
        # Create backup filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_filepath = backup_path / backup_filename
        
        # Copy file
        shutil.copy2(source_file, backup_filepath)
        
        logging.info(f"Created backup: {backup_filepath}")
        return str(backup_filepath)
        
    except Exception as e:
        logging.error(f"Failed to create backup: {e}")
        return None


def cleanup_old_backups(backup_dir: str, max_backups: int = 10) -> int:
    """
    Clean up old backup files, keeping only the most recent ones.
    
    Args:
        backup_dir: Backup directory
        max_backups: Maximum number of backups to keep
        
    Returns:
        Number of backups cleaned up
    """
    try:
        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return 0
        
        # Get all backup files sorted by modification time
        backup_files = sorted(
            backup_path.glob("*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups
        cleaned_count = 0
        for backup_file in backup_files[max_backups:]:
            if backup_file.is_file():
                backup_file.unlink()
                cleaned_count += 1
        
        logging.info(f"Cleaned up {cleaned_count} old backups")
        return cleaned_count
        
    except Exception as e:
        logging.error(f"Failed to cleanup old backups: {e}")
        return 0
