# src/utils/title_generator.py
"""Title and description generator for short videos using LLM"""

import logging
import re
import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import psycopg2
from psycopg2.extras import RealDictCursor

class TitleGenerator:
    """
    Generates titles and descriptions for short videos using LLM analysis
    of transcription content and video metadata.
    """
    
    def __init__(self, ollama_client):
        """
        Initialize title generator with Ollama client.
        
        Args:
            ollama_client: Configured Ollama client for LLM requests
        """
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        
        # Database connection parameters
        self.db_config = {
            'host': os.getenv("DB_HOST", "18.234.149.248"),
            'port': os.getenv("DB_PORT", "5432"),
            'dbname': os.getenv("DB_NAME", "postgres"),
            'user': os.getenv("DB_USER", "gizmott2011"),
            'password': os.getenv("DB_PASSWORD", "Giz2017$#321JHj$$")
        }
        
        # Title generation prompts for different content types
        self.title_prompts = {
            'social_media': """
            You are a social media content expert. Create an engaging title for a short video.
            
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Create a catchy, engaging title (max 60 characters)
            - Use trending keywords and hooks
            - Include emojis where appropriate
            - Make it clickable and shareable
            - Focus on the most interesting part of the content
            
            Return only the title, nothing else.
            """,
            
            'educational': """
            You are an educational content creator. Create a clear, informative title for this video segment.
            
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Create a clear, educational title (max 80 characters)
            - Focus on the key learning point or insight
            - Use action words and clear language
            - Make it searchable and informative
            
            Return only the title, nothing else.
            """,
            
            'entertainment': """
            You are an entertainment content creator. Create a fun, engaging title for this video clip.
            
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Create an entertaining, fun title (max 70 characters)
            - Use humor, excitement, or intrigue
            - Include relevant emojis
            - Make it shareable and memorable
            - Capture the entertainment value
            
            Return only the title, nothing else.
            """
        }
        
        self.description_prompts = {
            'social_media': """
            Create a compelling description for this short video for social media platforms.
            
            Title: {title}
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Write 2-3 sentences (max 150 characters)
            - Include relevant hashtags (3-5)
            - Make it engaging and shareable
            - Call-to-action for engagement
            - Match the tone of the title
            
            Format:
            Description text here. #hashtag1 #hashtag2 #hashtag3
            
            Return only the description, nothing else.
            """,
            
            'educational': """
            Create an informative description for this educational video segment.
            
            Title: {title}
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Write 2-3 sentences (max 200 characters)
            - Highlight key learning points
            - Include educational hashtags
            - Encourage learning and sharing
            
            Format:
            Description text here. #hashtag1 #hashtag2 #hashtag3
            
            Return only the description, nothing else.
            """,
            
            'entertainment': """
            Create a fun description for this entertainment video clip.
            
            Title: {title}
            Original Video: {original_filename}
            Video Content: {transcription}
            Duration: {duration}s
            
            Requirements:
            - Write 2-3 sentences (max 180 characters)
            - Match the entertainment tone
            - Include fun, relevant hashtags
            - Encourage engagement and sharing
            
            Format:
            Description text here. #hashtag1 #hashtag2 #hashtag3
            
            Return only the description, nothing else.
            """
        }
    
    def get_db_connection(self):
        """
        Get database connection.
        
        DEPRECATED: Direct database access is deprecated. Use API-based methods instead.
        This method remains for backward compatibility but will be removed in a future version.
        """
        self.logger.warning("DEPRECATED: Direct database access is deprecated. Use API-based methods instead.")
        
        try:
            # Database credentials should be set as environment variables
            conn = psycopg2.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                database=os.getenv('DB_NAME', 'video_db'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'password'),
                port=os.getenv('DB_PORT', '5432')
            )
            return conn
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return None
    
    def save_to_database(self, clip_id: int, title: str, description: str, 
                        tags: List[str], pubid: str = None, channelid: str = None) -> bool:
        """
        Save title and description data via API instead of direct database access.
        
        Args:
            clip_id: Video clip ID (integer)
            title: Generated AI title
            description: Generated AI description
            tags: List of tags/keywords
            pubid: Publisher ID for API authentication (required)
            channelid: Channel ID for API authentication (optional)
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure clip_id is an integer
            if isinstance(clip_id, str):
                clip_id = int(clip_id)
            
            # Check if we have required authentication parameters
            if not pubid:
                self.logger.error("Cannot save to database via API: pubid is required")
                return False
                
            # Import here to avoid circular imports
            from src.utils.database_api_client import DatabaseAPIClient, create_clip_data
            
            # Create API client
            api_client = DatabaseAPIClient()
            
            # Create clip data for API update
            clip_data = create_clip_data(
                key="",  # Key will be set by the calling context
                thumbnail="",  # Thumbnail will be set by the calling context
                ai_title=title,
                ai_description=description,
                tags=tags,
                clip_id=clip_id,
                video_id=0,  # Video ID will be set by the calling context
                duration=0   # Duration will be set by the calling context
            )
            
            # Update clip via API
            result = api_client.update_single_clip(clip_data, pubid, channelid)
            
            if result.get('success', True):  # Assume success if no explicit success field
                self.logger.info(f"Successfully updated clip {clip_id} via API")
                return True
            else:
                self.logger.error(f"Failed to update clip {clip_id} via API: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to save title data via API: {e}")
            return False
    
    def detect_content_type(self, transcription: str, original_filename: str) -> str:
        """
        Detect the content type based on transcription and filename.
        
        Args:
            transcription: Video transcription text
            original_filename: Original video filename
            
        Returns:
            Content type: 'educational', 'entertainment', or 'social_media'
        """
        # Convert to lowercase for analysis
        text_lower = transcription.lower()
        filename_lower = original_filename.lower()
        
        # Educational keywords
        educational_keywords = [
            'learn', 'tutorial', 'how to', 'explain', 'lesson', 'course',
            'teach', 'education', 'study', 'guide', 'training', 'tip',
            'technique', 'method', 'process', 'step', 'instruction'
        ]
        
        # Entertainment keywords
        entertainment_keywords = [
            'funny', 'laugh', 'comedy', 'joke', 'entertainment', 'fun',
            'amazing', 'incredible', 'wow', 'cool', 'awesome', 'epic',
            'hilarious', 'crazy', 'wild', 'unbelievable', 'shocking'
        ]
        
        # Check filename for hints
        if any(keyword in filename_lower for keyword in ['tutorial', 'lesson', 'course', 'guide', 'how']):
            return 'educational'
        
        if any(keyword in filename_lower for keyword in ['funny', 'comedy', 'entertainment', 'fun']):
            return 'entertainment'
        
        # Check transcription content
        educational_score = sum(1 for keyword in educational_keywords if keyword in text_lower)
        entertainment_score = sum(1 for keyword in entertainment_keywords if keyword in text_lower)
        
        if educational_score > entertainment_score and educational_score > 2:
            return 'educational'
        elif entertainment_score > 2:
            return 'entertainment'
        else:
            return 'social_media'  # Default for general content
    
    def clean_filename_for_context(self, filename: str) -> str:
        """
        Clean filename to make it more readable for LLM context.
        
        Args:
            filename: Original filename
            
        Returns:
            Cleaned filename for better LLM understanding
        """
        # Remove file extension
        name = Path(filename).stem
        
        # Replace common separators with spaces
        name = re.sub(r'[_\-\.]', ' ', name)
        
        # Remove video quality indicators
        name = re.sub(r'\b(720p|1080p|4k|hd|sd|mp4|avi|mov)\b', '', name, flags=re.IGNORECASE)
        
        # Remove excess whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Capitalize words for better readability
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name
    
    def extract_segment_transcription(self, full_transcription: Dict, start_time: float, end_time: float) -> str:
        """
        Extract transcription text for a specific video segment.
        
        Args:
            full_transcription: Full video transcription data
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            Transcription text for the segment
        """
        segment_text = []
        
        segments = full_transcription.get('segments', [])
        
        for segment in segments:
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            text = segment.get('text', '').strip()
            
            # Check if segment overlaps with our time range
            if seg_end > start_time and seg_start < end_time:
                segment_text.append(text)
        
        # Join all segment texts
        full_text = ' '.join(segment_text).strip()
        
        # Clean up the text
        full_text = re.sub(r'\s+', ' ', full_text)  # Remove extra whitespace
        full_text = full_text.replace('\n', ' ')    # Replace newlines with spaces
        
        # Limit length for LLM prompt (keep most relevant part)
        if len(full_text) > 500:
            # Take first and last part if too long
            words = full_text.split()
            if len(words) > 80:
                first_part = ' '.join(words[:40])
                last_part = ' '.join(words[-40:])
                full_text = f"{first_part} ... {last_part}"
        
        return full_text
    
    async def generate_title_and_description(self, 
                                     filename: str,
                                     transcription: str,
                                     bucket_path: str,
                                     duration: float,
                                     variation_manager=None) -> Dict[str, str]:
        """
        Generate title and description for a video segment with optional variation.
        
        Args:
            filename: Original video filename
            transcription: Video transcription text
            bucket_path: Path to the video file in bucket
            duration: Video duration in seconds
            variation_manager: Optional VariationManager for output diversity
            
        Returns:
            Dictionary with 'title', 'description', 'content_type', and 'tags'
        """
        try:
            if not transcription.strip():
                # Fallback for segments without transcription
                transcription = "Video content without clear audio transcription"
            
            # Clean filename for better context
            clean_filename = self.clean_filename_for_context(filename)
            
            # Detect content type
            content_type = self.detect_content_type(transcription, filename)
            
            # Calculate duration
            duration_int = int(duration)
            
            # Create unique cache key including transcription hash for segment differentiation
            import hashlib
            transcription_hash = hashlib.md5(transcription.encode()).hexdigest()[:8]
            base_cache_key = f"title_{clean_filename}_{duration_int}_{transcription_hash}"
            
            # Apply variation to cache key if variation manager is provided
            if variation_manager:
                cache_key = variation_manager.get_varied_cache_key(base_cache_key, transcription_hash)
            else:
                cache_key = base_cache_key
            
            self.logger.info(f"Generating title for {content_type} content: {clean_filename} ({duration_int}s) - Cache key: {cache_key}")
            
            # Generate title with optional variation
            title_prompt = self.title_prompts[content_type].format(
                original_filename=clean_filename,
                transcription=transcription,
                duration=duration_int
            )
            
            # Apply title variation if variation manager is provided
            if variation_manager:
                title_prompt = variation_manager.get_title_variation_prompt_modifier(title_prompt, content_type)
            
            # Request plain text response (not JSON) for title generation
            title_response = await self.ollama_client._make_request(
                prompt=title_prompt,
                model="mistral-small3.2:latest",
                cache_key=cache_key,
                response_format=None  # Don't force JSON format for simple text
            )
            
            self.logger.debug(f"Raw title response: {title_response}")
            
            # Since we're now requesting plain text (not JSON), handle response appropriately
            if isinstance(title_response, dict):
                # Check if this is a cached response format
                if 'data' in title_response and isinstance(title_response['data'], str):
                    title = title_response['data']
                else:
                    # If response is still a dict (shouldn't happen with plain text format), extract text
                    title = str(title_response.get('response', title_response.get('content', str(title_response))))
            else:
                # Plain text response - use directly
                title = str(title_response) if title_response else ""
            
            self.logger.debug(f"Extracted title before cleaning: {title}")
            # Clean up the title (remove quotes, extra whitespace, etc.)
            title = self._clean_generated_text(title)
            self.logger.debug(f"Title after cleaning: {title}")
            
            # Enhanced fallback detection - check for invalid/placeholder titles
            invalid_title_keywords = [
                'unique_title_generator', 'internet_search', 'web_search', 
                'search', 'tool', 'function', 'error', 'null', 'undefined',
                'none', 'n/a', 'untitled'
            ]
            
            is_invalid_title = (
                not title or 
                title.strip() == "" or 
                len(title.strip()) < 5 or  # Too short to be a real title
                any(keyword in title.lower() for keyword in invalid_title_keywords) or
                title.startswith('{') or  # Looks like JSON
                title.startswith('[')     # Looks like array
            )
            
            if is_invalid_title:
                self.logger.warning(f"Invalid title generated: '{title}', using fallback")
                # Fallback title generation
                title = self._generate_fallback_title(clean_filename, transcription, duration_int)
            
            self.logger.info(f"Generated title: {title}")
            
            # Generate description with optional variation
            description_prompt = self.description_prompts[content_type].format(
                title=title,
                original_filename=clean_filename,
                transcription=transcription,
                duration=duration_int
            )
            
            # Apply description variation if variation manager is provided
            if variation_manager:
                description_prompt = variation_manager.add_description_variation(description_prompt)
            
            # Create varied cache key for description
            desc_base_cache_key = f"desc_{clean_filename}_{duration_int}_{transcription_hash}"
            if variation_manager:
                desc_cache_key = variation_manager.get_varied_cache_key(desc_base_cache_key, transcription_hash)
            else:
                desc_cache_key = desc_base_cache_key
            
            # Request plain text response (not JSON) for description generation
            description_response = await self.ollama_client._make_request(
                prompt=description_prompt,
                model="mistral-small3.2:latest",
                cache_key=desc_cache_key,
                response_format=None  # Don't force JSON format for simple text
            )
            
            # Handle both string and dict responses (including cached responses)
            if isinstance(description_response, dict):
                # Check if this is a cached response format
                if 'data' in description_response and isinstance(description_response['data'], str):
                    description = description_response['data']
                else:
                    # If response is a dict, extract text content
                    description = str(description_response.get('response', description_response.get('content', str(description_response))))
            else:
                description = str(description_response) if description_response else ""
            
            description = self._clean_generated_text(description)
            
            if not description:
                # Fallback description generation
                description = self._generate_fallback_description(title, content_type)
            
            # Extract tags from description
            tags = self._extract_tags_from_description(description)
            
            self.logger.info(f"Generated description: {description}")
            
            return {
                'title': title,
                'description': description,
                'content_type': content_type,
                'tags': tags,
                'duration': duration_int
            }
            
        except Exception as e:
            self.logger.error(f"Error generating title and description: {e}")
            
            # Return fallback title and description
            clean_filename = self.clean_filename_for_context(filename)
            duration_int = int(duration)
            
            return {
                'title': f"Clip from {clean_filename}",
                'description': f"Short video clip ({duration_int}s) from {clean_filename} #shorts #video",
                'content_type': 'social_media',
                'tags': ['shorts', 'video'],
                'duration': duration_int
            }
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean and validate generated text from LLM.
        Now optimized for plain text responses since we no longer force JSON format.
        """
        if not text:
            return ""
        
        # Since we're now getting plain text responses, JSON parsing should only be needed
        # for legacy cached responses. Try JSON parsing only if it looks like JSON.
        if text.strip().startswith('{') or text.strip().startswith('['):
            try:
                import json
                parsed = json.loads(text)
                
                # Extract text from common JSON response formats
                if isinstance(parsed, dict):
                    # Try different possible keys
                    for key in ['title', 'description', 'response', 'content', 'text', 'message']:
                        if key in parsed and isinstance(parsed[key], str):
                            text = parsed[key]
                            self.logger.warning(f"Found title/description in JSON format (legacy cache?): key='{key}'")
                            break
                    else:
                        # If no expected key found, this might be an error response
                        if 'tool' in parsed or 'function' in parsed or 'error' in parsed:
                            self.logger.error(f"Received unexpected JSON structure: {parsed}")
                            return ""  # Return empty to trigger fallback
                        
                        # Try to stringify the first string value
                        for value in parsed.values():
                            if isinstance(value, str) and len(value.strip()) > 0:
                                text = value
                                break
                elif isinstance(parsed, str):
                    text = parsed
            except (json.JSONDecodeError, TypeError) as e:
                # Not valid JSON despite looking like it - this is suspicious
                self.logger.warning(f"Text looks like JSON but failed to parse: {e}")
                # Continue with original text
                pass
        
        # Remove quotes if the LLM wrapped the response
        text = text.strip('"\'')
        
        # Remove common LLM prefixes
        text = re.sub(r'^(Title:|Description:)\s*', '', text, flags=re.IGNORECASE)
        
        # Remove markdown formatting if present
        text = text.strip('*_`')
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_tags_from_description(self, description: str) -> List[str]:
        """Extract hashtags from description text."""
        if not description:
            return []
        
        # Find all hashtags in the description
        hashtags = re.findall(r'#(\w+)', description)
        
        # Convert to lowercase and remove duplicates
        tags = list(set(tag.lower() for tag in hashtags))
        
        return tags
    
    def _generate_fallback_title(self, filename: str, transcription: str, duration: int) -> str:
        """Generate fallback title when LLM fails."""
        if transcription and len(transcription) > 20:
            # Extract first few meaningful words
            words = transcription.split()[:5]
            title = ' '.join(words)
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        else:
            return f"{filename} - {duration}s Clip"
    
    def _generate_fallback_description(self, title: str, content_type: str) -> str:
        """Generate fallback description when LLM fails."""
        hashtags = {
            'educational': '#learn #education #tutorial #tips',
            'entertainment': '#funny #entertainment #fun #viral',
            'social_media': '#shorts #video #viral #content'
        }
        
        return f"Check out this video! {hashtags.get(content_type, '#shorts #video')}"
    
    def batch_generate_titles_and_descriptions(self, 
                                             shorts_details: List[Dict],
                                             original_filename: str,
                                             full_transcription: Dict) -> List[Dict]:
        """
        Generate titles and descriptions for multiple short videos.
        
        Args:
            shorts_details: List of short video details with timing info
            original_filename: Original video filename
            full_transcription: Full video transcription data
            
        Returns:
            Updated shorts_details with title and description
        """
        updated_shorts = []
        
        for i, short in enumerate(shorts_details):
            try:
                self.logger.info(f"Generating title and description for short {i+1}/{len(shorts_details)}")
                
                # Generate title and description
                title_data = self.generate_title_and_description(
                    original_filename=original_filename,
                    full_transcription=full_transcription,
                    segment_start=short.get('start_time', 0),
                    segment_end=short.get('end_time', 60)
                )
                
                # Add title data to short details
                short.update(title_data)
                updated_shorts.append(short)
                
                self.logger.info(f"Short {i+1}: '{title_data['title']}' ({title_data['content_type']})")
                
            except Exception as e:
                self.logger.error(f"Error processing short {i+1}: {e}")
                # Add fallback data
                short.update({
                    'title': f"Video Clip {i+1}",
                    'description': f"Short video clip #shorts #video",
                    'content_type': 'social_media',
                    'duration': int(short.get('end_time', 60) - short.get('start_time', 0))
                })
                updated_shorts.append(short)
        
        return updated_shorts
