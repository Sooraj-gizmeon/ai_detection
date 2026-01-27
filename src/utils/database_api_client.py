"""
API client for external database operations.
Replaces direct PostgreSQL operations with API calls.
"""

import logging
import requests
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseAPIClient:
    """Client for interacting with external database APIs."""
    
    def __init__(self, base_url: str = "https://api.gizmott.com/dashboard/v1"):
        """
        Initialize the API client. 
        
        Args:
            base_url: Base URL for the API endpoints
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def add_transcription(self, video_id: int = None, uploaded_video_id: str = None, transcription: str = None) -> Dict[str, Any]:
        """
        Add transcription for a video.
        
        Args:
            video_id: The video ID (integer, optional)
            uploaded_video_id: The uploaded video ID (string, optional)
            transcription: The transcription text
            
        Returns:
            API response
        """
        # Determine which ID to use (video_id takes precedence)
        effective_id = video_id if video_id else uploaded_video_id
        if not effective_id:
            return {"success": False, "error": "Either video_id or uploaded_video_id must be provided"}
        
        url = f"{self.base_url}/ai/transcription/add"
        payload = {
            "transcription": transcription
        }
        
        # Add the appropriate ID field to payload
        if video_id:
            payload["video_id"] = video_id
        else:
            payload["uploaded_video_id"] = uploaded_video_id
        
        try:
            logger.info(f"Adding transcription for video {video_id}")
            
            # Log request details
            logger.info(f"🔍 Add Transcription Request:")
            logger.info(f"   URL: {url}")
            logger.info(f"   Payload: {json.dumps(payload, indent=2)}")
            
            response = self.session.post(url, json=payload, timeout=30)
            
            # Log response details
            logger.info(f"📡 Add Transcription Response:")
            logger.info(f"   Status Code: {response.status_code}")
            try:
                response_text = response.text
                logger.info(f"   Response Body: {response_text}")
            except Exception as resp_error:
                logger.warning(f"   Could not read response body: {resp_error}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Successfully added transcription for video {video_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to add transcription for video {video_id}: {e}")
            
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Error Status Code: {e.response.status_code}")
                try:
                    error_body = e.response.text
                    logger.error(f"   Error Response Body: {error_body}")
                except Exception as err_parse:
                    logger.error(f"   Could not parse error response: {err_parse}")
            
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error adding transcription: {e}")
            return {"success": False, "error": str(e)}
    
    def get_transcription(self, video_id: int = None, uploaded_video_id: str = None) -> Dict[str, Any]:
        """
        Get transcription for a video.
        
        Args:
            video_id: The video ID (integer, optional)
            uploaded_video_id: The uploaded video ID (string, optional)
            
        Returns:
            API response with transcription data
        """
        # Determine which ID to use (video_id takes precedence)
        effective_id = video_id if video_id else uploaded_video_id
        if not effective_id:
            return {"success": False, "error": "Either video_id or uploaded_video_id must be provided"}
        
        # Construct URL based on which ID type is provided
        if video_id:
            url = f"{self.base_url}/ai/transcription/fetch?video_id={video_id}"
        else:
            url = f"{self.base_url}/ai/transcription/fetch?uploaded_video_id={uploaded_video_id}"
        
        try:
            logger.info(f"Getting transcription for video {video_id}")
            
            # Log request details
            logger.info(f"🔍 Get Transcription Request:")
            logger.info(f"   URL: {url}")
            
            response = self.session.get(url, timeout=30)
            
            # Log response details
            logger.info(f"📡 Get Transcription Response:")
            logger.info(f"   Status Code: {response.status_code}")
            try:
                response_text = response.text
                logger.info(f"   Response Body: {response_text}")
            except Exception as resp_error:
                logger.warning(f"   Could not read response body: {resp_error}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Successfully retrieved transcription for video {video_id}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to get transcription for video {video_id}: {e}")
            
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Error Status Code: {e.response.status_code}")
                try:
                    error_body = e.response.text
                    logger.error(f"   Error Response Body: {error_body}")
                except Exception as err_parse:
                    logger.error(f"   Could not parse error response: {err_parse}")
            
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error getting transcription: {e}")
            return {"success": False, "error": str(e)}                   
    
    def update_clips(self, clips: List[Dict[str, Any]], pubid: str, channelid: str = None, status: str = None) -> Dict[str, Any]:
        """
        Update clips information.
        
        Args:
            clips: List of clip data dictionaries
            pubid: Publisher ID for authentication
            channelid: Channel ID for authentication (optional)
            status: Processing status (e.g., "completed") for the entire batch (optional)
            
        Returns:
            API response
        """
        url = f"{self.base_url}/ai/clips/update"
        
        # Set pubid and channelid in headers
        headers = {"pubid": pubid}
        if channelid:
            headers["channelid"] = channelid
        
        payload = {"clips": clips}
        
        # Add status to payload root if provided
        if status:
            payload["status"] = status
        
        try:
            if channelid:
                logger.info(f"Updating {len(clips)} clips with pubid {pubid} and channelid {channelid}")
            else:
                logger.info(f"Updating {len(clips)} clips with pubid {pubid}")
            
            # Log the complete request details for debugging
            logger.info(f"🔍 API Request Details:")
            logger.info(f"   URL: {url}")
            logger.info(f"   Headers: {json.dumps(headers, indent=2, ensure_ascii=False)}")
            logger.info(f"   Request Body: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            
            # Serialize payload with proper Unicode handling
            json_payload = json.dumps(payload, ensure_ascii=False)
            
            response = self.session.post(
                url, 
                data=json_payload, 
                headers={**headers, 'Content-Type': 'application/json'}, 
                timeout=60
            )
            
            # Log response details
            logger.info(f"📡 API Response:")
            logger.info(f"   Status Code: {response.status_code}")
            logger.info(f"   Response Headers: {dict(response.headers)}")
            
            # Log response body if available
            try:
                response_text = response.text
                logger.info(f"   Response Body: {response_text}")
            except Exception as resp_error:
                logger.warning(f"   Could not read response body: {resp_error}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Successfully updated {len(clips)} clips")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to update clips: {e}")
            
            # Log additional error details if response is available
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Error Status Code: {e.response.status_code}")
                logger.error(f"   Error Response Headers: {dict(e.response.headers)}")
                try:
                    error_body = e.response.text
                    logger.error(f"   Error Response Body: {error_body}")
                except Exception as err_parse:
                    logger.error(f"   Could not parse error response: {err_parse}")
            
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error updating clips: {e}")
            return {"success": False, "error": str(e)}
    
    def update_single_clip(self, clip_data: Dict[str, Any], pubid: str, channelid: str = None, status: str = None) -> Dict[str, Any]:
        """
        Update a single clip (convenience method).
        
        Args:
            clip_data: Clip data dictionary
            pubid: Publisher ID for authentication
            channelid: Channel ID for authentication (optional)
            status: Processing status (e.g., "completed") - optional
            
        Returns:
            API response
        """
        return self.update_clips([clip_data], pubid, channelid, status)
    
    def close(self):
        """Close the HTTP session."""
        if self.session:
            self.session.close()


def create_clip_data(key: str, thumbnail: str, ai_title: str, ai_description: str, 
                    tags: List[str], clip_id: int, duration: float,
                    video_id: int = None, uploaded_video_id: str = None,
                    brand_logo: str = None, overlay_x: int = None, overlay_y: int = None,
                    canvas_type: str = None) -> Dict[str, Any]:
    """
    Create a clip data dictionary for API submission.
    
    Args:
        key: S3 object key for the clip
        thumbnail: Thumbnail URL
        ai_title: AI-generated title
        ai_description: AI-generated description  
        tags: List of tags
        clip_id: Clip ID (from final_video_ids)
        duration: Duration in seconds
        video_id: Video ID (integer, optional)
        uploaded_video_id: Uploaded video ID (string, optional)
        brand_logo: Brand logo URL (optional)
        overlay_x: Logo overlay X position (optional)
        overlay_y: Logo overlay Y position (optional)
        canvas_type: Canvas type (e.g., "shorts", "landscape") (optional)
        
    Returns:
        Formatted clip data dictionary
    """
    clip_data = {
        "key": key,
        "thumbnail": thumbnail,
        "ai_title": ai_title,
        "ai_description": ai_description,
        "tags": tags,
        "clip_id": clip_id,
        "duration": duration
    }
    
    # Add video identifier (video_id takes precedence)
    if video_id:
        clip_data["video_id"] = video_id
    elif uploaded_video_id:
        clip_data["uploaded_video_id"] = uploaded_video_id
    
    # Add canvas_type if provided
    if canvas_type:
        clip_data["canvas_type"] = canvas_type
    
    # Add brand logo parameters if provided
    # if brand_logo:
    #     clip_data["brand_logo"] = brand_logo
    # if logo_position:
    #     clip_data["logo_position"] = logo_position
    
    return clip_data


def extract_transcription_text(audio_analysis: Dict[str, Any]) -> str:
    """
    Extract full transcription text from audio analysis results.
    
    Args:
        audio_analysis: Audio analysis results containing transcription
        
    Returns:
        Combined transcription text
    """
    if not audio_analysis or 'transcription' not in audio_analysis:
        return ""
    
    transcription = audio_analysis['transcription']
    if 'segments' not in transcription:
        return ""
    
    segments = transcription['segments']
    if not segments:
        return ""
    
    # Combine all segment texts
    transcription_text = " ".join([
        seg.get('text', '').strip() for seg in segments 
        if seg.get('text', '').strip()
    ])
    
    return transcription_text.strip()


def generate_ai_title_and_description(segment_data: Dict[str, Any], 
                                     prompt_analysis: Dict[str, Any] = None) -> tuple[str, str]:
    """
    Generate AI title and description for a clip based on segment data.
    
    Args:
        segment_data: Segment information including transcript, timing, etc.
        prompt_analysis: Optional prompt analysis for context
        
    Returns:
        Tuple of (ai_title, ai_description)
    """
    # Extract key information
    transcript = segment_data.get('transcript', '').strip()
    start_time = segment_data.get('start_time', 0)
    end_time = segment_data.get('end_time', 0)
    duration = end_time - start_time
    
    # Generate title based on content
    if transcript:
        # Use first meaningful words for title
        words = transcript.split()[:8]  # First 8 words
        ai_title = " ".join(words)
        if len(transcript.split()) > 8:
            ai_title += "..."
    else:
        ai_title = f"Video Clip ({int(start_time)}s-{int(end_time)}s)"
    
    # Generate description
    if transcript:
        ai_description = f"Clip from {int(start_time)}s to {int(end_time)}s: {transcript[:200]}"
        if len(transcript) > 200:
            ai_description += "..."
    else:
        ai_description = f"Video clip segment ({duration:.1f}s duration)"
    
    # Add context from prompt analysis if available
    if prompt_analysis:
        intent = prompt_analysis.get('intent', '')
        if intent:
            ai_description += f" | Intent: {intent}"
    
    return ai_title[:100], ai_description[:500]  # Limit lengths


def generate_tags_from_analysis(segment_data: Dict[str, Any], 
                               prompt_analysis: Dict[str, Any] = None,
                               object_context: Dict[str, Any] = None) -> List[str]:
    """
    Generate tags for a clip based on analysis results.
    
    Args:
        segment_data: Segment information
        prompt_analysis: Prompt analysis results
        object_context: Object detection context
        
    Returns:
        List of relevant tags
    """
    tags = []
    
    # Add tags from prompt analysis
    if prompt_analysis:
        keywords = prompt_analysis.get('keywords', [])
        tags.extend([kw.lower() for kw in keywords[:3]])  # First 3 keywords
        
        intent = prompt_analysis.get('intent', '')
        if intent:
            tags.append(intent.lower())
    
    # Add tags from object context
    if object_context:
        detected_objects = object_context.get('detected_objects', [])
        object_classes = list(set([obj.lower() for obj in detected_objects[:3]]))  # First 3 unique objects
        tags.extend(object_classes)
    
    # Add content type tags
    transcript = segment_data.get('transcript', '').lower()
    if 'funny' in transcript or 'laugh' in transcript:
        tags.append('funny')
    if 'learn' in transcript or 'tutorial' in transcript:
        tags.append('educational')
    if 'exciting' in transcript or 'amazing' in transcript:
        tags.append('exciting')
    
    # Default tags
    tags.extend(['ai', 'shorts'])
    
    # Remove duplicates and limit to 8 tags
    unique_tags = list(dict.fromkeys(tags))[:8]
    return [tag for tag in unique_tags if tag and len(tag) > 1]
