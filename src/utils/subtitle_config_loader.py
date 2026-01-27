# src/utils/subtitle_config_loader.py
"""Subtitle configuration loader for loading styles from JSON config"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional


class SubtitleConfigLoader:
    """Loads and manages subtitle configurations from JSON file."""
    
    def __init__(self, config_path: str = None):
        """Initialize subtitle config loader.
        
        Args:
            config_path: Path to subtitle_config.json, defaults to config/subtitle_config.json
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Default to config/subtitle_config.json relative to project root
            self.config_path = Path(__file__).parent.parent.parent / "config" / "subtitle_config.json"
        else:
            self.config_path = Path(config_path)
            
        self._config_cache = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load subtitle configuration from JSON file."""
        try:
            if not self.config_path.exists():
                self.logger.error(f"Subtitle config file not found: {self.config_path}")
                self._config_cache = []
                return
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_cache = json.load(f)
                
            self.logger.info(f"Loaded {len(self._config_cache)} subtitle styles from {self.config_path}")
            
            # Log available style IDs for debugging
            available_ids = [style.get('id') for style in self._config_cache if style.get('id')]
            self.logger.debug(f"Available subtitle style IDs: {available_ids}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in subtitle config file: {e}")
            self._config_cache = []
        except Exception as e:
            self.logger.error(f"Error loading subtitle config: {e}")
            self._config_cache = []
    
    def get_style_by_id(self, style_id: str) -> Optional[Dict]:
        """Get subtitle style configuration by ID.
        
        Args:
            style_id: The style ID to look up (e.g., 'cinematic', 'elegant', 'tech')
            
        Returns:
            Style configuration dictionary or None if not found
        """
        if not self._config_cache:
            self.logger.warning("No subtitle styles loaded, using default")
            return None
            
        for style in self._config_cache:
            if style.get('id') == style_id:
                self.logger.info(f"Found subtitle style: {style_id} ({style.get('name', 'Unknown')})")
                return style
                
        self.logger.warning(f"Subtitle style ID '{style_id}' not found")
        available_ids = [s.get('id') for s in self._config_cache if s.get('id')]
        self.logger.info(f"Available style IDs: {available_ids}")
        return None
    
    def get_all_styles(self) -> list:
        """Get all available subtitle styles.
        
        Returns:
            List of all style configurations
        """
        return self._config_cache or []
    
    def get_available_style_ids(self) -> list:
        """Get list of all available style IDs.
        
        Returns:
            List of style ID strings
        """
        if not self._config_cache:
            return []
        return [style.get('id') for style in self._config_cache if style.get('id')]
    
    def convert_json_style_to_ffmpeg_style(self, json_style: Dict) -> Dict:
        """Convert JSON style configuration to FFmpeg-compatible format.
        
        The JSON config uses CSS-like properties, but FFmpeg needs specific formatting.
        
        Args:
            json_style: Style configuration from JSON
            
        Returns:
            FFmpeg-compatible style dictionary
        """
        if not json_style or 'style' not in json_style:
            return self._get_default_ffmpeg_style()
            
        css_style = json_style['style']
        
        # Map CSS properties to FFmpeg style properties
        ffmpeg_style = {
            # Font properties
            'font_family': self._map_font_family(css_style.get('font_family', 'Arial')),
            'font_size': 24,  # Default size, can be adjusted
            'font_color': css_style.get('default_color', '#ccc'),
            
            # Typography
            'font_weight': css_style.get('font_weight', 'normal'),
            'font_style': css_style.get('font_style', 'normal'),
            'text_transform': css_style.get('text_transform'),
            'letter_spacing': css_style.get('letter_spacing', '0px'),
            
            # Colors
            'background_color': 'black@0.7',  # Default semi-transparent background
            'border_color': 'black',
            'shadow_color': 'black@0.9',
            
            # Layout properties (FFmpeg defaults)
            'border_width': 2,
            'alignment': 2,  # Center
            'margin_v': 60,
            'shadow_offset': '2,2',
        }
        
        # Handle pronunciation color (special highlighting)
        if 'pronunciation_color' in css_style:
            # Store for potential future use in highlighting specific words
            ffmpeg_style['pronunciation_color'] = css_style['pronunciation_color']
        
        self.logger.debug(f"Converted style {json_style.get('id', 'unknown')} to FFmpeg format")
        return ffmpeg_style
    
    def _map_font_family(self, css_font_family: str) -> str:
        """Map CSS font family to FFmpeg-compatible font name.
        
        Args:
            css_font_family: CSS font family string
            
        Returns:
            FFmpeg-compatible font name
        """
        # Extract the primary font name from CSS font stack
        # Example: "Impact, sans-serif" -> "Impact"
        if ',' in css_font_family:
            primary_font = css_font_family.split(',')[0].strip()
        else:
            primary_font = css_font_family.strip()
            
        # Remove quotes if present
        primary_font = primary_font.strip('\'"')
        
        # Map some common CSS fonts to FFmpeg equivalents
        font_mapping = {
            'Black Ops One': 'Arial',  # Fallback for special fonts
            'Press Start 2P': 'Courier New',
            'Gloria Hallelujah': 'Arial',
            'UnifrakturMaguntia': 'Times New Roman',
            'Cinzel': 'Times New Roman',
            'Lobster': 'Arial',
            'Satisfy': 'Arial',
        }
        
        return font_mapping.get(primary_font, primary_font)
    
    def _get_default_ffmpeg_style(self) -> Dict:
        """Get default FFmpeg style configuration.
        
        Returns:
            Default style dictionary
        """
        return {
            'font_family': 'Arial',
            'font_size': 24,
            'font_color': 'white',
            'background_color': 'black@0.7',
            'border_width': 2,
            'border_color': 'black',
            'alignment': 2,
            'margin_v': 60,
            'shadow_offset': '2,2',
            'shadow_color': 'black@0.9'
        }
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self.logger.info("Reloading subtitle configuration")
        self._config_cache = None
        self._load_config()


# Global instance for easy access
_subtitle_config_loader = None

def get_subtitle_config_loader() -> SubtitleConfigLoader:
    """Get global subtitle configuration loader instance.
    
    Returns:
        SubtitleConfigLoader instance
    """
    global _subtitle_config_loader
    if _subtitle_config_loader is None:
        _subtitle_config_loader = SubtitleConfigLoader()
    return _subtitle_config_loader