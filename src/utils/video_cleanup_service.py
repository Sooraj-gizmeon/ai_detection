"""
Intelligent Video Cleanup Service
Manages automatic deletion of old videos from input and output directories
with configurable policies and safety mechanisms.
"""

import os
import logging
import shutil
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum


class CleanupPolicy(Enum):
    """Video cleanup policies"""
    TIME_BASED = "time_based"  # Remove files older than specified time
    SIZE_BASED = "size_based"  # Remove oldest files when size limit exceeded
    COUNT_BASED = "count_based"  # Keep only N most recent files
    DISK_SPACE = "disk_space"  # Cleanup when disk space is low
    SUCCESS_BASED = "success_based"  # Remove input only after successful processing


@dataclass
class CleanupStats:
    """Cleanup operation statistics"""
    files_scanned: int = 0
    files_deleted: int = 0
    space_freed_mb: float = 0.0
    errors: List[str] = None
    duration_seconds: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CleanupConfig:
    """Configuration for video cleanup operations"""
    
    # Time-based cleanup
    input_retention_days: int = 2  # Keep input files for 2 days
    output_retention_days: int = 2  # Keep output files for 2 days
    temp_retention_hours: int = 24  # Keep temp files for 24 hours
    
    # Size-based cleanup
    max_input_size_gb: float = 150.0  # Max 150GB in input directory
    max_output_size_gb: float = 50.0  # Max 50GB in output directory
    max_temp_size_gb: float = 5.0  # Max 5GB in temp directory
    
    # Count-based cleanup
    max_input_files: int = 100  # Keep max 100 input files
    max_output_files: int = 500  # Keep max 500 output files
    
    # Disk space monitoring
    min_free_space_gb: float = 5.0  # Trigger cleanup when <5GB free
    cleanup_target_free_gb: float = 10.0  # Cleanup until 10GB free
    
    # Safety settings
    never_delete_extensions: List[str] = None  # Extensions to never delete
    never_delete_patterns: List[str] = None  # Filename patterns to never delete
    dry_run: bool = False  # If True, only log what would be deleted
    
    # Processing tracking
    track_processing_success: bool = True  # Track successful processing
    success_tracking_file: str = "cleanup_success_log.json"
    
    # Scheduling
    auto_cleanup_enabled: bool = True
    cleanup_schedule_hours: int = 6  # Run cleanup every 6 hours
    
    def __post_init__(self):
        if self.never_delete_extensions is None:
            self.never_delete_extensions = ['.log', '.json', '.srt', '.ass']
        if self.never_delete_patterns is None:
            self.never_delete_patterns = ['example', 'sample', 'demo']


class VideoCleanupService:
    """
    Intelligent video cleanup service with multiple policies and safety mechanisms.
    """
    
    def __init__(self, 
                 input_dir: str = "input",
                 output_dir: str = "output", 
                 temp_dir: str = "temp",
                 config: Optional[CleanupConfig] = None,
                 log_dir: str = "logs"):
        """
        Initialize the cleanup service.
        
        Args:
            input_dir: Path to input directory
            output_dir: Path to output directory 
            temp_dir: Path to temp directory
            config: Cleanup configuration
            log_dir: Directory for cleanup logs
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.log_dir = Path(log_dir)
        
        self.config = config or CleanupConfig()
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.VideoCleanupService")
        
        # Success tracking
        self.success_log_path = self.log_dir / self.config.success_tracking_file
        self._ensure_directories()
        
        # Video extensions to track
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        
        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.input_dir, self.output_dir, self.temp_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    async def start_auto_cleanup(self):
        """Start automatic cleanup scheduler."""
        if not self.config.auto_cleanup_enabled:
            self.logger.info("Auto cleanup is disabled")
            return
        
        if self._cleanup_task and not self._cleanup_task.done():
            self.logger.warning("Auto cleanup is already running")
            return
        
        self.logger.info(f"Starting auto cleanup scheduler (every {self.config.cleanup_schedule_hours} hours)")
        self._cleanup_task = asyncio.create_task(self._cleanup_scheduler())
    
    async def stop_auto_cleanup(self):
        """Stop automatic cleanup scheduler."""
        self.logger.info("Stopping auto cleanup scheduler...")
        self._shutdown_event.set()
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                self.logger.info("Auto cleanup scheduler stopped")
    
    async def _cleanup_scheduler(self):
        """Background scheduler for automatic cleanup."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Run comprehensive cleanup
                    await self.run_comprehensive_cleanup()
                    
                    # Wait for next schedule or shutdown
                    schedule_seconds = self.config.cleanup_schedule_hours * 3600
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), 
                        timeout=schedule_seconds
                    )
                    
                except asyncio.TimeoutError:
                    # Timeout is expected - continue with next cleanup cycle
                    continue
                except Exception as e:
                    self.logger.error(f"Error in cleanup scheduler: {e}")
                    # Wait a bit before retrying
                    await asyncio.sleep(300)  # 5 minutes
                    
        except asyncio.CancelledError:
            self.logger.info("Cleanup scheduler cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in cleanup scheduler: {e}")
    
    async def run_comprehensive_cleanup(self) -> CleanupStats:
        """
        Run comprehensive cleanup using all policies.
        
        Returns:
            Combined cleanup statistics
        """
        start_time = datetime.now()
        self.logger.info("Starting comprehensive video cleanup")
        
        total_stats = CleanupStats()
        
        try:
            # 1. Check disk space first
            if await self._should_cleanup_for_disk_space():
                self.logger.info("Low disk space detected, running emergency cleanup")
                disk_stats = await self.cleanup_by_disk_space()
                self._merge_stats(total_stats, disk_stats)
            
            # 2. Time-based cleanup
            time_stats = await self.cleanup_by_time()
            self._merge_stats(total_stats, time_stats)
            
            # 3. Size-based cleanup for each directory
            size_stats = await self.cleanup_by_size()
            self._merge_stats(total_stats, size_stats)
            
            # 4. Count-based cleanup
            count_stats = await self.cleanup_by_count()
            self._merge_stats(total_stats, count_stats)
            
            # 5. Clean temp files aggressively
            temp_stats = await self.cleanup_temp_files()
            self._merge_stats(total_stats, temp_stats)
            
            # Calculate duration
            total_stats.duration_seconds = (datetime.now() - start_time).total_seconds()
            
            # Log summary
            self.logger.info(f"Comprehensive cleanup completed:")
            self.logger.info(f"  Files scanned: {total_stats.files_scanned}")
            self.logger.info(f"  Files deleted: {total_stats.files_deleted}")
            self.logger.info(f"  Space freed: {total_stats.space_freed_mb:.1f} MB")
            self.logger.info(f"  Duration: {total_stats.duration_seconds:.1f} seconds")
            
            if total_stats.errors:
                self.logger.warning(f"  Errors: {len(total_stats.errors)}")
                for error in total_stats.errors[:5]:  # Show first 5 errors
                    self.logger.warning(f"    {error}")
            
            # Save cleanup log
            await self._save_cleanup_log(total_stats)
            
            return total_stats
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive cleanup: {e}")
            total_stats.errors.append(f"Comprehensive cleanup error: {str(e)}")
            return total_stats
    
    async def cleanup_by_time(self) -> CleanupStats:
        """
        Remove files older than configured retention periods.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running time-based cleanup")
        
        try:
            # Input directory cleanup
            input_cutoff = datetime.now() - timedelta(days=self.config.input_retention_days)
            input_stats = await self._cleanup_directory_by_time(
                self.input_dir, input_cutoff, "input"
            )
            self._merge_stats(stats, input_stats)
            
            # Output directory cleanup
            output_cutoff = datetime.now() - timedelta(days=self.config.output_retention_days)
            output_stats = await self._cleanup_directory_by_time(
                self.output_dir, output_cutoff, "output"
            )
            self._merge_stats(stats, output_stats)
            
            # Temp directory cleanup (more aggressive)
            temp_cutoff = datetime.now() - timedelta(hours=self.config.temp_retention_hours)
            temp_stats = await self._cleanup_directory_by_time(
                self.temp_dir, temp_cutoff, "temp"
            )
            self._merge_stats(stats, temp_stats)
            
        except Exception as e:
            error_msg = f"Time-based cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    async def cleanup_by_size(self) -> CleanupStats:
        """
        Remove oldest files when directory size exceeds limits.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running size-based cleanup")
        
        try:
            # Input directory
            input_stats = await self._cleanup_directory_by_size(
                self.input_dir, self.config.max_input_size_gb, "input"
            )
            self._merge_stats(stats, input_stats)
            
            # Output directory
            output_stats = await self._cleanup_directory_by_size(
                self.output_dir, self.config.max_output_size_gb, "output"
            )
            self._merge_stats(stats, output_stats)
            
            # Temp directory
            temp_stats = await self._cleanup_directory_by_size(
                self.temp_dir, self.config.max_temp_size_gb, "temp"
            )
            self._merge_stats(stats, temp_stats)
            
        except Exception as e:
            error_msg = f"Size-based cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    async def cleanup_by_count(self) -> CleanupStats:
        """
        Keep only N most recent files in each directory.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running count-based cleanup")
        
        try:
            # Input directory
            input_stats = await self._cleanup_directory_by_count(
                self.input_dir, self.config.max_input_files, "input"
            )
            self._merge_stats(stats, input_stats)
            
            # Output directory
            output_stats = await self._cleanup_directory_by_count(
                self.output_dir, self.config.max_output_files, "output"
            )
            self._merge_stats(stats, output_stats)
            
        except Exception as e:
            error_msg = f"Count-based cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    async def cleanup_by_disk_space(self) -> CleanupStats:
        """
        Emergency cleanup when disk space is low.
        Removes oldest files until target free space is reached.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running disk space cleanup")
        
        try:
            target_bytes = self.config.cleanup_target_free_gb * (1024 ** 3)
            
            # Get all video files sorted by modification time (oldest first)
            all_files = []
            
            # Collect from temp first (most aggressive)
            all_files.extend(await self._get_files_by_age(self.temp_dir, reverse=False))
            # Then output (can be regenerated)  
            all_files.extend(await self._get_files_by_age(self.output_dir, reverse=False))
            # Finally input (more precious, but still cleanable)
            all_files.extend(await self._get_files_by_age(self.input_dir, reverse=False))
            
            # Delete files until we have enough space
            for file_path, file_info in all_files:
                current_free = await self._get_free_space()
                if current_free >= target_bytes:
                    break
                
                if await self._should_delete_file(file_path):
                    size_mb = file_info['size'] / (1024 * 1024)
                    
                    if self.config.dry_run:
                        self.logger.info(f"[DRY RUN] Would delete: {file_path} ({size_mb:.1f} MB)")
                    else:
                        try:
                            file_path.unlink()
                            stats.files_deleted += 1
                            stats.space_freed_mb += size_mb
                            self.logger.info(f"Deleted for disk space: {file_path} ({size_mb:.1f} MB)")
                        except Exception as e:
                            error_msg = f"Failed to delete {file_path}: {str(e)}"
                            stats.errors.append(error_msg)
                
                stats.files_scanned += 1
            
        except Exception as e:
            error_msg = f"Disk space cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    async def cleanup_temp_files(self) -> CleanupStats:
        """
        Aggressively clean temporary files.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running temp files cleanup")
        
        try:
            # Clean any file older than 1 hour in temp
            one_hour_ago = datetime.now() - timedelta(hours=1)
            temp_stats = await self._cleanup_directory_by_time(
                self.temp_dir, one_hour_ago, "temp", include_non_video=True
            )
            self._merge_stats(stats, temp_stats)
            
        except Exception as e:
            error_msg = f"Temp files cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    async def mark_processing_success(self, input_file: str, output_files: List[str]):
        """
        Mark successful processing of an input file.
        
        Args:
            input_file: Path to input file that was processed
            output_files: List of output files that were created
        """
        try:
            success_log = await self._load_success_log()
            
            entry = {
                'input_file': str(input_file),
                'output_files': [str(f) for f in output_files],
                'timestamp': datetime.now().isoformat(),
                'safe_to_delete': True
            }
            
            success_log[str(input_file)] = entry
            await self._save_success_log(success_log)
            
            self.logger.info(f"Marked successful processing: {input_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to mark processing success: {e}")
    
    async def cleanup_successfully_processed(self) -> CleanupStats:
        """
        Remove input files that have been successfully processed.
        
        Returns:
            Cleanup statistics
        """
        stats = CleanupStats()
        self.logger.info("Running success-based cleanup")
        
        try:
            success_log = await self._load_success_log()
            
            for input_file_str, entry in success_log.items():
                input_path = Path(input_file_str)
                stats.files_scanned += 1
                
                # Check if input file still exists
                if not input_path.exists():
                    continue
                
                # Check if all output files exist
                output_files_exist = all(
                    Path(output_file).exists() 
                    for output_file in entry['output_files']
                )
                
                if output_files_exist and entry.get('safe_to_delete', False):
                    if await self._should_delete_file(input_path):
                        size_mb = input_path.stat().st_size / (1024 * 1024)
                        
                        if self.config.dry_run:
                            self.logger.info(f"[DRY RUN] Would delete successfully processed: {input_path}")
                        else:
                            try:
                                input_path.unlink()
                                stats.files_deleted += 1
                                stats.space_freed_mb += size_mb
                                self.logger.info(f"Deleted successfully processed: {input_path}")
                            except Exception as e:
                                error_msg = f"Failed to delete {input_path}: {str(e)}"
                                stats.errors.append(error_msg)
                
        except Exception as e:
            error_msg = f"Success-based cleanup error: {str(e)}"
            self.logger.error(error_msg)
            stats.errors.append(error_msg)
        
        return stats
    
    # Helper methods
    
    async def _should_cleanup_for_disk_space(self) -> bool:
        """Check if cleanup is needed due to low disk space."""
        try:
            free_space = await self._get_free_space()
            min_space_bytes = self.config.min_free_space_gb * (1024 ** 3)
            return free_space < min_space_bytes
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return False
    
    async def _get_free_space(self) -> int:
        """Get free disk space in bytes."""
        import shutil
        total, used, free = shutil.disk_usage(self.input_dir.parent)
        return free
    
    async def _cleanup_directory_by_time(self, 
                                       directory: Path, 
                                       cutoff_time: datetime, 
                                       dir_name: str,
                                       include_non_video: bool = False) -> CleanupStats:
        """Clean directory by removing files older than cutoff time."""
        stats = CleanupStats()
        
        try:
            if not directory.exists():
                return stats
            
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                
                # Check file extension
                if not include_non_video and file_path.suffix.lower() not in self.video_extensions:
                    continue
                
                stats.files_scanned += 1
                
                # Check modification time
                mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if mod_time < cutoff_time and await self._should_delete_file(file_path):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    
                    if self.config.dry_run:
                        self.logger.info(f"[DRY RUN] Would delete old {dir_name} file: {file_path}")
                    else:
                        try:
                            file_path.unlink()
                            stats.files_deleted += 1
                            stats.space_freed_mb += size_mb
                            self.logger.info(f"Deleted old {dir_name} file: {file_path}")
                        except Exception as e:
                            error_msg = f"Failed to delete {file_path}: {str(e)}"
                            stats.errors.append(error_msg)
                            
        except Exception as e:
            error_msg = f"Directory cleanup error for {directory}: {str(e)}"
            stats.errors.append(error_msg)
        
        return stats
    
    async def _cleanup_directory_by_size(self, 
                                       directory: Path, 
                                       max_size_gb: float, 
                                       dir_name: str) -> CleanupStats:
        """Clean directory by removing oldest files when size limit exceeded."""
        stats = CleanupStats()
        
        try:
            current_size = await self._get_directory_size(directory)
            max_size_bytes = max_size_gb * (1024 ** 3)
            
            if current_size <= max_size_bytes:
                return stats
            
            self.logger.info(f"{dir_name} directory size ({current_size / (1024**3):.1f} GB) exceeds limit ({max_size_gb} GB)")
            
            # Get files sorted by modification time (oldest first)
            files = await self._get_files_by_age(directory, reverse=False)
            
            bytes_to_delete = current_size - max_size_bytes
            deleted_bytes = 0
            
            for file_path, file_info in files:
                if deleted_bytes >= bytes_to_delete:
                    break
                
                stats.files_scanned += 1
                
                if await self._should_delete_file(file_path):
                    size_mb = file_info['size'] / (1024 * 1024)
                    
                    if self.config.dry_run:
                        self.logger.info(f"[DRY RUN] Would delete for size limit: {file_path}")
                    else:
                        try:
                            file_path.unlink()
                            stats.files_deleted += 1
                            stats.space_freed_mb += size_mb
                            deleted_bytes += file_info['size']
                            self.logger.info(f"Deleted for size limit: {file_path}")
                        except Exception as e:
                            error_msg = f"Failed to delete {file_path}: {str(e)}"
                            stats.errors.append(error_msg)
                            
        except Exception as e:
            error_msg = f"Size-based cleanup error for {directory}: {str(e)}"
            stats.errors.append(error_msg)
        
        return stats
    
    async def _cleanup_directory_by_count(self, 
                                        directory: Path, 
                                        max_files: int, 
                                        dir_name: str) -> CleanupStats:
        """Clean directory by keeping only N most recent files."""
        stats = CleanupStats()
        
        try:
            # Get files sorted by modification time (newest first)
            files = await self._get_files_by_age(directory, reverse=True)
            
            if len(files) <= max_files:
                return stats
            
            self.logger.info(f"{dir_name} directory has {len(files)} files, keeping {max_files} most recent")
            
            # Delete excess files (oldest ones)
            files_to_delete = files[max_files:]
            
            for file_path, file_info in files_to_delete:
                stats.files_scanned += 1
                
                if await self._should_delete_file(file_path):
                    size_mb = file_info['size'] / (1024 * 1024)
                    
                    if self.config.dry_run:
                        self.logger.info(f"[DRY RUN] Would delete for count limit: {file_path}")
                    else:
                        try:
                            file_path.unlink()
                            stats.files_deleted += 1
                            stats.space_freed_mb += size_mb
                            self.logger.info(f"Deleted for count limit: {file_path}")
                        except Exception as e:
                            error_msg = f"Failed to delete {file_path}: {str(e)}"
                            stats.errors.append(error_msg)
                            
        except Exception as e:
            error_msg = f"Count-based cleanup error for {directory}: {str(e)}"
            stats.errors.append(error_msg)
        
        return stats
    
    async def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes."""
        total_size = 0
        
        try:
            if not directory.exists():
                return 0
            
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    
        except Exception as e:
            self.logger.error(f"Error calculating directory size for {directory}: {e}")
        
        return total_size
    
    async def _get_files_by_age(self, directory: Path, reverse: bool = True) -> List[Tuple[Path, Dict]]:
        """
        Get video files sorted by modification time.
        
        Args:
            directory: Directory to scan
            reverse: If True, newest first; if False, oldest first
            
        Returns:
            List of (file_path, file_info) tuples sorted by age
        """
        files = []
        
        try:
            if not directory.exists():
                return files
            
            for file_path in directory.iterdir():
                if not file_path.is_file():
                    continue
                
                # Only consider video files
                if file_path.suffix.lower() not in self.video_extensions:
                    continue
                
                try:
                    stat = file_path.stat()
                    file_info = {
                        'size': stat.st_size,
                        'mtime': stat.st_mtime,
                        'atime': stat.st_atime
                    }
                    files.append((file_path, file_info))
                except Exception as e:
                    self.logger.warning(f"Error getting stats for {file_path}: {e}")
            
            # Sort by modification time
            files.sort(key=lambda x: x[1]['mtime'], reverse=reverse)
            
        except Exception as e:
            self.logger.error(f"Error scanning directory {directory}: {e}")
        
        return files
    
    async def _should_delete_file(self, file_path: Path) -> bool:
        """
        Check if a file should be deleted based on safety rules.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file can be safely deleted
        """
        try:
            # Check file extension whitelist
            if file_path.suffix.lower() in self.config.never_delete_extensions:
                return False
            
            # Check filename patterns
            file_name = file_path.name.lower()
            for pattern in self.config.never_delete_patterns:
                if pattern.lower() in file_name:
                    return False
            
            # Additional safety checks
            # Don't delete files that are currently being accessed (< 5 minutes old)
            current_time = datetime.now().timestamp()
            file_mtime = file_path.stat().st_mtime
            
            if current_time - file_mtime < 300:  # 5 minutes
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking if file should be deleted {file_path}: {e}")
            return False
    
    async def _load_success_log(self) -> Dict:
        """Load success tracking log."""
        try:
            if self.success_log_path.exists():
                with open(self.success_log_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading success log: {e}")
        
        return {}
    
    async def _save_success_log(self, success_log: Dict):
        """Save success tracking log."""
        try:
            with open(self.success_log_path, 'w') as f:
                json.dump(success_log, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving success log: {e}")
    
    async def _save_cleanup_log(self, stats: CleanupStats):
        """Save cleanup operation log."""
        try:
            cleanup_log_path = self.log_dir / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(cleanup_log_path, 'w') as f:
                json.dump(asdict(stats), f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving cleanup log: {e}")
    
    def _merge_stats(self, target: CleanupStats, source: CleanupStats):
        """Merge cleanup statistics."""
        target.files_scanned += source.files_scanned
        target.files_deleted += source.files_deleted
        target.space_freed_mb += source.space_freed_mb
        target.errors.extend(source.errors)
    
    # Public utility methods
    
    async def get_directory_info(self) -> Dict:
        """Get information about all monitored directories."""
        info = {}
        
        for name, directory in [
            ("input", self.input_dir),
            ("output", self.output_dir), 
            ("temp", self.temp_dir)
        ]:
            try:
                files = await self._get_files_by_age(directory)
                size_bytes = await self._get_directory_size(directory)
                
                info[name] = {
                    'path': str(directory),
                    'exists': directory.exists(),
                    'file_count': len(files),
                    'size_mb': size_bytes / (1024 * 1024),
                    'size_gb': size_bytes / (1024 ** 3),
                    'oldest_file': files[-1][0].name if files else None,
                    'newest_file': files[0][0].name if files else None
                }
                
            except Exception as e:
                info[name] = {
                    'path': str(directory),
                    'error': str(e)
                }
        
        # Add system disk space info
        try:
            total, used, free = shutil.disk_usage(self.input_dir.parent)
            info['disk_space'] = {
                'total_gb': total / (1024 ** 3),
                'used_gb': used / (1024 ** 3),
                'free_gb': free / (1024 ** 3),
                'usage_percent': (used / total) * 100
            }
        except Exception as e:
            info['disk_space'] = {'error': str(e)}
        
        return info
    
    async def simulate_cleanup(self) -> Dict:
        """
        Simulate cleanup without actually deleting files.
        
        Returns:
            Dictionary with simulation results
        """
        original_dry_run = self.config.dry_run
        self.config.dry_run = True
        
        try:
            stats = await self.run_comprehensive_cleanup()
            
            return {
                'would_delete_files': stats.files_deleted,
                'would_free_mb': stats.space_freed_mb,
                'files_scanned': stats.files_scanned,
                'errors': stats.errors
            }
            
        finally:
            self.config.dry_run = original_dry_run