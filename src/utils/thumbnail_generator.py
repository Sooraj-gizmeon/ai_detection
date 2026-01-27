"""
Utility functions for generating and uploading thumbnails.
"""

import os
import cv2
import logging
import tempfile
import requests
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_thumbnail(video_path: str, output_path: str = None, frame_position: float = 0.25) -> str:
    """
    Generate a thumbnail from a video file.
    
    Args:
        video_path: Path to the input video file
        output_path: Path to save the thumbnail. If None, a temporary file will be created
        frame_position: Position in the video to extract the frame (0.0 to 1.0, as a fraction of total duration)
        
    Returns:
        Path to the generated thumbnail
    """
    try:
        # Create output path if not provided
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            video_name = os.path.basename(video_path).split('.')[0]
            output_path = os.path.join(temp_dir, f"{video_name}_thumbnail.jpg")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not video.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return None
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            logger.error(f"Invalid frame count for video: {video_path}")
            return None
        
        # Calculate frame position
        target_frame = int(total_frames * frame_position)
        
        # Set video position to target frame
        video.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read the frame
        success, frame = video.read()
        
        if not success:
            logger.error(f"Failed to read frame at position {target_frame} from {video_path}")
            return None
        
        # Resize to reasonable thumbnail size (maintaining aspect ratio)
        height, width = frame.shape[:2]
        max_dimension = 800
        
        if height > width:
            new_height = min(height, max_dimension)
            new_width = int((new_height / height) * width)
        else:
            new_width = min(width, max_dimension)
            new_height = int((new_width / width) * height)
        
        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Save the frame as a JPEG
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        
        # Release the video capture object
        video.release()
        
        logger.info(f"Generated thumbnail at {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error generating thumbnail: {e}")
        return None

def upload_thumbnail(thumbnail_path: str, pubid: str, channelid: str = None) -> dict:
    """
    Upload a thumbnail to the API.
    
    Args:
        thumbnail_path: Path to the thumbnail file
        pubid: Publisher ID for the API request
        channelid: Channel ID for the API request (optional)
        
    Returns:
        Dictionary with upload results, including the thumbnail URL if successful
    """
    try:
        # Check if thumbnail exists
        if not os.path.exists(thumbnail_path):
            logger.error(f"Thumbnail file not found: {thumbnail_path}")
            return {"success": False, "error": "Thumbnail file not found"}
        
        # API endpoint
        url = "https://api.gizmott.com/dashboard/v1/upload/thumbnail"
        
        # Headers
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Origin": "https://revamp.gizmott.com",
            "Referer": "https://revamp.gizmott.com/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
            "pubid": pubid,
            "sec-ch-ua": "\"Not)A;Brand\";v=\"8\", \"Chromium\";v=\"138\", \"Google Chrome\";v=\"138\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\""
        }
        
        # Add channelid to headers if provided
        if channelid:
            headers["channelid"] = channelid
        
        # Files to upload
        files = {
            'thumbnailWebsite': (os.path.basename(thumbnail_path), open(thumbnail_path, 'rb'), 'image/jpeg'),
            'is_show_video': (None, '0')
        }
        
        # Make the POST request
        logger.info(f"Uploading thumbnail {thumbnail_path} to {url}")
        logger.info(f"Request headers: {headers}")
        logger.info(f"Request files: thumbnailWebsite={os.path.basename(thumbnail_path)}, is_show_video=0")
        
        response = requests.post(url, headers=headers, files=files)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Thumbnail upload successful: {result}")
            
            # Extract the thumbnail URL from the response
            if result.get('success') and result.get('data') and len(result['data']) > 0:
                thumbnail_url = result['data'][0].get('path')
                return {
                    "success": True,
                    "thumbnail_url": thumbnail_url,
                    "response": result
                }
            else:
                logger.warning(f"Thumbnail upload succeeded but response format unexpected: {result}")
                return {
                    "success": True,
                    "response": result,
                    "error": "Unexpected response format"
                }
        else:
            logger.error(f"Thumbnail upload failed with status code {response.status_code}: {response.text}")
            return {
                "success": False,
                "status_code": response.status_code,
                "error": response.text
            }
    
    except Exception as e:
        logger.error(f"Error uploading thumbnail: {e}")
        return {
            "success": False,
            "error": str(e)
        }
