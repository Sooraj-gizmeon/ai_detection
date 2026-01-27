# config/subtitle_styles.py
"""
Subtitle styling configuration for the video processing pipeline.
This file allows easy customization of subtitle appearance.
"""

# Current active style preset
# Options: 'classic', 'modern', 'high_contrast', 'elegant', 'vibrant', 'minimal', 'blue_theme', 'green_theme'
ACTIVE_SUBTITLE_STYLE = 'modern'

# Custom style overrides (optional)
# These will override the preset style settings
CUSTOM_SUBTITLE_STYLE = {
    # Text appearance
    # 'font_family': 'Arial',  # Font family: Arial, Helvetica, Times, etc.
    # 'font_size': 28,         # Font size in pixels
    # 'font_color': 'white',   # Text color: 'white', 'yellow', '#FFFFFF', etc.
    
    # Background
    # 'background_color': 'black@0.7',  # Background color with transparency
    
    # Border/Outline
    # 'border_width': 3,       # Border thickness in pixels
    # 'border_color': 'black', # Border color
    
    # Shadow
    # 'shadow_offset': '3,3',  # Shadow offset (x,y) in pixels
    # 'shadow_color': 'black@0.9',  # Shadow color with transparency
    
    # Position
    # 'alignment': 2,          # Text alignment: 1=left, 2=center, 3=right
    # 'margin_v': 60,          # Vertical margin from bottom in pixels
}

# Style presets with descriptions
STYLE_PRESETS = {
    'classic': {
        'description': 'White text with semi-transparent black background - most readable',
        'font_color': 'white',
        'background_color': 'black@0.7',
        'border_color': 'black',
        'use_case': 'General purpose, works well on all video types'
    },
    
    'modern': {
        'description': 'White text with subtle background - clean and minimal',
        'font_color': 'white',
        'background_color': 'black@0.3',
        'border_color': 'black',
        'use_case': 'Modern videos with dark content'
    },
    
    'high_contrast': {
        'description': 'Yellow text with strong black background - maximum readability',
        'font_color': 'yellow',
        'background_color': 'black@0.8',
        'border_color': 'black',
        'use_case': 'Videos with busy backgrounds or accessibility needs'
    },
    
    'elegant': {
        'description': 'White text with dark gray background - sophisticated look',
        'font_color': 'white',
        'background_color': '#333333@0.6',
        'border_color': '#222222',
        'use_case': 'Professional or educational content'
    },
    
    'vibrant': {
        'description': 'White text with orange background - eye-catching',
        'font_color': 'white',
        'background_color': '#FF6B35@0.7',
        'border_color': '#CC5529',
        'use_case': 'Entertainment or lifestyle content'
    },
    
    'minimal': {
        'description': 'White text with thick border, no background - clean appearance',
        'font_color': 'white',
        'background_color': 'transparent',
        'border_color': 'black',
        'border_width': 4,
        'use_case': 'Videos with consistent dark backgrounds'
    },
    
    'blue_theme': {
        'description': 'White text with blue background - corporate/tech feel',
        'font_color': 'white',
        'background_color': '#1E3A8A@0.7',
        'border_color': '#1E40AF',
        'use_case': 'Technology or business content'
    },
    
    'green_theme': {
        'description': 'White text with green background - natural/health theme',
        'font_color': 'white',
        'background_color': '#047857@0.7',
        'border_color': '#065F46',
        'use_case': 'Health, nature, or environment content'
    }
}

# Color reference for easy customization
COLOR_REFERENCE = {
    'named_colors': {
        'white': 'Pure white text',
        'black': 'Pure black',
        'red': 'Bright red',
        'green': 'Bright green',
        'blue': 'Bright blue',
        'yellow': 'Bright yellow (high visibility)',
        'cyan': 'Bright cyan',
        'magenta': 'Bright magenta'
    },
    
    'hex_colors': {
        '#FFFFFF': 'Pure white',
        '#000000': 'Pure black',
        '#FF0000': 'Pure red',
        '#00FF00': 'Pure green',
        '#0000FF': 'Pure blue',
        '#FFFF00': 'Pure yellow',
        '#FF6B35': 'Orange',
        '#1E3A8A': 'Dark blue',
        '#047857': 'Dark green',
        '#333333': 'Dark gray',
        '#666666': 'Medium gray',
        '#999999': 'Light gray'
    },
    
    'transparency_examples': {
        'color@1.0': 'Fully opaque (no transparency)',
        'color@0.8': '80% opacity (slight transparency)',
        'color@0.7': '70% opacity (good for backgrounds)',
        'color@0.5': '50% opacity (medium transparency)',
        'color@0.3': '30% opacity (very transparent)',
        'color@0.0': 'Fully transparent (invisible)',
        'transparent': 'Completely transparent (no background)'
    }
}

# Quick style examples for common use cases
QUICK_EXAMPLES = {
    'youtube_style': {
        'font_color': 'white',
        'background_color': 'black@0.8',
        'font_size': 30,
        'border_width': 2
    },
    
    'tiktok_style': {
        'font_color': 'white',
        'background_color': 'transparent',
        'font_size': 32,
        'border_width': 4,
        'border_color': 'black'
    },
    
    'instagram_style': {
        'font_color': 'white',
        'background_color': 'black@0.6',
        'font_size': 26,
        'border_width': 2
    }
}
