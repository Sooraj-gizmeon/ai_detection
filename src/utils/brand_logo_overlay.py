"""
Brand logo overlay utility for adding logos to videos using FFmpeg.
"""

import logging
import subprocess
import requests
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class BrandLogoOverlay:
    """Handles brand logo overlay operations using FFmpeg."""
    
    # Logo position mappings
    POSITION_MAP = {
        'tl': 'top-left',     # Top-left
        'tr': 'top-right',    # Top-right  
        'bl': 'bottom-left',  # Bottom-left
        'br': 'bottom-right', # Bottom-right
        'tc': 'top-center',   # Top-center
        'bc': 'bottom-center' # Bottom-center
    }
    
    def __init__(self, logo_size: Tuple[int, int] = (120, 120), margin: int = 20):
        """
        Initialize brand logo overlay.
        
        Args:
            logo_size: Target logo size (width, height) in pixels
            margin: Margin from video edges in pixels
        """
        self.logo_size = logo_size
        self.margin = margin
        self.logger = logging.getLogger(__name__)
    
    def download_logo(self, logo_url: str) -> Optional[str]:
        """
        Download logo from URL to temporary file.
        
        Args:
            logo_url: URL of the logo image
            
        Returns:
            Path to downloaded logo file or None if failed
        """
        try:
            self.logger.info(f"Downloading brand logo from: {logo_url}")
            
            # Create temporary file for logo
            temp_dir = tempfile.gettempdir()
            logo_filename = f"brand_logo_{hash(logo_url) % 100000}.png"
            logo_path = os.path.join(temp_dir, logo_filename)
            
            # Download logo
            response = requests.get(logo_url, timeout=30, stream=True)
            response.raise_for_status()
            
            with open(logo_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.logger.info(f"Logo downloaded successfully: {logo_path}")
            return logo_path
            
        except Exception as e:
            self.logger.error(f"Failed to download logo from {logo_url}: {e}")
            return None
    
    def get_ffmpeg_overlay_filter(self, logo_path: str, overlay_x: int = None, overlay_y: int = None, 
                                 logo_position: str = None, video_width: int = 1080, video_height: int = 1920, 
                                 canvas_type: str = "shorts") -> str:
        """
        Generate FFmpeg overlay filter string for logo positioning.
        
        Args:
            logo_path: Path to logo image file
            overlay_x: X coordinate (0 = left edge, increases rightward)
            overlay_y: Y coordinate (0 = bottom edge, increases upward)
            logo_position: Legacy position code ('tl', 'tr', 'bl', 'br', 'tc', 'bc') - used if coordinates not provided
            video_width: Video width in pixels
            video_height: Video height in pixels
            canvas_type: Canvas type ("shorts" or "clips") for coordinate system scaling
            
        Returns:
            FFmpeg overlay filter string
        """
        # Calculate logo dimensions
        logo_w, logo_h = self.logo_size
        
        # Use coordinate system if provided, otherwise fall back to legacy positioning
        if overlay_x is not None and overlay_y is not None:
            # Convert from our coordinate system to FFmpeg coordinates
            # Our system: (0,0) = bottom-left, Y increases upward
            # FFmpeg system: (0,0) = top-left, Y increases downward
            
            # Scale coordinates based on canvas type and video resolution
            if canvas_type == "shorts":
                # For shorts (9:16 - 1080x1920), scale from (0,0)-(225,580) to video dimensions
                max_x, max_y = 225, 435
            else:  # clips (16:9 - 1920x1080)
                # For clips (16:9 - 1920x1080), scale from (0,0)-(580,225) to video dimensions
                max_x, max_y = 435, 225
            
            # Scale coordinates to video dimensions, accounting for logo size
            # The coordinate system maps to usable area minus logo dimensions
            usable_width = video_width - logo_w
            usable_height = video_height - logo_h
            
            scaled_x = int((overlay_x / max_x) * usable_width)
            scaled_y = int((overlay_y / max_y) * usable_height)
            
            # Ensure coordinates stay within bounds
            scaled_x = max(0, min(scaled_x, usable_width))
            scaled_y = max(0, min(scaled_y, usable_height))
            
            # Convert Y coordinate from bottom-up to top-down
            ffmpeg_x = scaled_x
            ffmpeg_y = video_height - scaled_y - logo_h
            
            self.logger.info(f"Coordinate positioning: ({overlay_x}, {overlay_y}) -> scaled: ({scaled_x}, {scaled_y}) -> FFmpeg: ({ffmpeg_x}, {ffmpeg_y})")
            
        else:
            # Legacy positioning system
            margin = self.margin
            
            # Position calculations for legacy system
            if logo_position == 'tl':  # Top-left
                ffmpeg_x, ffmpeg_y = margin, margin
            elif logo_position == 'tr':  # Top-right
                ffmpeg_x, ffmpeg_y = f"W-w-{margin}", margin
            elif logo_position == 'bl':  # Bottom-left
                ffmpeg_x, ffmpeg_y = margin, f"H-h-{margin}"
            elif logo_position == 'br':  # Bottom-right
                ffmpeg_x, ffmpeg_y = f"W-w-{margin}", f"H-h-{margin}"
            elif logo_position == 'tc':  # Top-center
                ffmpeg_x, ffmpeg_y = f"(W-w)/2", margin
            elif logo_position == 'bc':  # Bottom-center
                ffmpeg_x, ffmpeg_y = f"(W-w)/2", f"H-h-{margin}"
            else:
                # Default to top-right if unknown position
                self.logger.warning(f"Unknown logo position '{logo_position}', defaulting to top-right")
                ffmpeg_x, ffmpeg_y = f"W-w-{margin}", margin
        
        # Create scale and overlay filter
        filter_complex = f"[1:v]scale={logo_w}:{logo_h}[logo];[0:v][logo]overlay={ffmpeg_x}:{ffmpeg_y}"
        
        return filter_complex
    
    def add_logo_to_video(self, input_video_path: str, output_video_path: str, 
                         logo_url: str, overlay_x: int = None, overlay_y: int = None, 
                         logo_position: str = None, canvas_type: str = "shorts") -> bool:
        """
        Add brand logo overlay to video using FFmpeg.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video with logo
            logo_url: URL of the logo image
            overlay_x: X coordinate (0 = left edge, increases rightward)
            overlay_y: Y coordinate (0 = bottom edge, increases upward)
            logo_position: Legacy position code ('tl', 'tr', 'bl', 'br', 'tc', 'bc') - used if coordinates not provided
            canvas_type: Canvas type ("shorts" or "clips") for coordinate system scaling
            
        Returns:
            True if successful, False otherwise
        """
        logo_path = None
        
        try:
            # Download logo
            logo_path = self.download_logo(logo_url)
            if not logo_path:
                return False
            
            # Get video dimensions
            video_info = self._get_video_info(input_video_path)
            if not video_info:
                return False
            
            video_width = video_info.get('width', 1080)
            video_height = video_info.get('height', 1920)
            
            # Generate overlay filter with new coordinate system
            overlay_filter = self.get_ffmpeg_overlay_filter(
                logo_path, overlay_x, overlay_y, logo_position, 
                video_width, video_height, canvas_type
            )
            
            # Build FFmpeg command
            ffmpeg_cmd = [
                'ffmpeg',
                '-i', input_video_path,      # Video input
                '-i', logo_path,             # Logo input
                '-filter_complex', overlay_filter,  # Apply logo overlay
                '-c:a', 'copy',              # Copy audio stream
                '-preset', 'fast',           # Fast encoding
                '-crf', '23',                # Quality setting
                '-y',                        # Overwrite output
                output_video_path
            ]
            
            self.logger.info(f"Adding brand logo with FFmpeg: {' '.join(ffmpeg_cmd[:8])}...")
            
            # Execute FFmpeg command
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=225  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully added brand logo to video: {output_video_path}")
                return True
            else:
                self.logger.error(f"FFmpeg logo overlay failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg logo overlay timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error adding logo overlay: {e}")
            return False
        finally:
            # Clean up downloaded logo
            if logo_path and os.path.exists(logo_path):
                try:
                    os.remove(logo_path)
                    self.logger.debug(f"Cleaned up logo file: {logo_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up logo file {logo_path}: {e}")
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get video information using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information or None if failed
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in probe_data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'duration': float(probe_data.get('format', {}).get('duration', 0))
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            return None


def add_brand_logo_if_needed(input_video_path: str, output_video_path: str, 
                           brand_logo: Optional[str] = None, 
                           overlay_x: Optional[int] = None,
                           overlay_y: Optional[int] = None,
                           logo_position: Optional[str] = None,
                           canvas_type: str = "shorts") -> str:
    """
    Add brand logo to video if logo parameters are provided.
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to output video
        brand_logo: Brand logo URL (optional)
        overlay_x: X coordinate (0 = left edge, increases rightward)
        overlay_y: Y coordinate (0 = bottom edge, increases upward)
        logo_position: Legacy position code (optional, used if coordinates not provided)
        canvas_type: Canvas type ("shorts" or "clips") for coordinate system scaling
        
    Returns:
        Path to final video (with or without logo)
    """
    if not brand_logo:
        # No logo URL provided, return original path
        logger.debug("No brand logo URL provided, skipping logo overlay")
        return input_video_path
    
    # Check if we have either coordinates or legacy position
    has_coordinates = overlay_x is not None and overlay_y is not None
    has_legacy_position = logo_position is not None
    
    if not has_coordinates and not has_legacy_position:
        logger.warning("No logo positioning parameters provided, skipping logo overlay")
        return input_video_path
    
    # Validate legacy logo position if using legacy system
    if has_legacy_position and not has_coordinates:
        valid_positions = ['tl', 'tr', 'bl', 'br', 'tc', 'bc']
        if logo_position not in valid_positions:
            logger.warning(f"Invalid logo position '{logo_position}', skipping logo overlay")
            return input_video_path
    
    try:
        # Create logo overlay instance
        logo_overlay = BrandLogoOverlay()
        
        # Add logo to video
        success = logo_overlay.add_logo_to_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            logo_url=brand_logo,
            overlay_x=overlay_x,
            overlay_y=overlay_y,
            logo_position=logo_position,
            canvas_type=canvas_type
        )
        
        if success:
            logger.info(f"Brand logo added successfully to {output_video_path}")
            return output_video_path
        else:
            logger.warning("Brand logo overlay failed, using original video")
            return input_video_path
            
    except Exception as e:
        logger.error(f"Error during brand logo overlay: {e}")
        return input_video_path
