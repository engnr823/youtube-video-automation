# file: video_clients/heygen_client.py
import os
import requests
import time
import logging
import json

# Configure logging
logger = logging.getLogger(__name__)

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")

class HeyGenError(Exception):
    """Custom exception for HeyGen API errors."""
    pass

def _request(method, endpoint, json_data=None, timeout=60):
    """Helper to make HTTP requests to HeyGen."""
    if not HEYGEN_API_KEY:
        raise HeyGenError("HEYGEN_API_KEY not found in environment variables.")

    url = f"https://api.heygen.com{endpoint}"
    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.request(method, url, headers=headers, json=json_data, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_msg = f"HeyGen API Error: {e.response.text}"
        if response.status_code == 400 or response.status_code == 404:
             # Identify Avatar Not Found specifically
             if "avatar_id" in response.text or "not found" in response.text:
                 raise HeyGenError(f"AVATAR_NOT_FOUND: {response.text}")
        raise HeyGenError(error_msg)
    except Exception as e:
        raise HeyGenError(str(e))

def get_all_avatars():
    """Fetches list of available avatars."""
    try:
        data = _request("GET", "/v2/avatars")
        return data.get("data", {}).get("avatars", [])
    except Exception as e:
        logger.error(f"Failed to fetch avatars: {e}")
        return []

def check_video_status(video_id):
    """Checks status of video generation."""
    data = _request("GET", f"/v1/video_status.get?video_id={video_id}")
    return data.get("data", {})

def generate_heygen_video(avatar_id, audio_url, aspect_ratio="16:9", background_image_url=None, use_green_screen=False):
    """
    Generates a video. 
    SMART LOGIC: Detects if ID is a 'Talking Photo' (UUID) or 'Avatar' (String).
    """
    logger.info(f"Submitting Job: Avatar={avatar_id}")
    
    # --- SMART DETECTION LOGIC ---
    # 3D Avatars usually have underscores (e.g., 'josh_lite3_20230714')
    # Talking Photos are usually UUIDs (e.g., '4343bfb447bf4028a48b598ae297f5dc')
    is_talking_photo = False
    if len(avatar_id) == 32 and "_" not in avatar_id:
        is_talking_photo = True
        logger.info(f"💡 Detected UUID format. Switching to TALKING PHOTO mode for {avatar_id}")

    # Prepare Aspect Ratio / Dimension
    # Talking Photos often require specific dimensions, but we'll try to map ratio
    dimension = {"width": 1080, "height": 1920} if aspect_ratio == "9:16" else {"width": 1920, "height": 1080}

    # Payload Construction
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar" if not is_talking_photo else "talking_photo",
                    "avatar_id": avatar_id if not is_talking_photo else None,
                    "talking_photo_id": avatar_id if is_talking_photo else None,
                    "avatar_style": "normal" if not is_talking_photo else None,
                    "scale": 1.0
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "background": {
                    "type": "color",
                    "value": "#00FF00" # Green screen for compositing
                }
            }
        ],
        "dimension": dimension
    }

    # Clean up None values
    if is_talking_photo:
        del payload["video_inputs"][0]["character"]["avatar_id"]
        del payload["video_inputs"][0]["character"]["avatar_style"]
    else:
        del payload["video_inputs"][0]["character"]["talking_photo_id"]

    try:
        # Submit
        response = _request("POST", "/v2/video/generate", payload)
        video_id = response.get("data", {}).get("video_id")
        
        if not video_id:
            raise HeyGenError("No video_id returned")
            
        logger.info(f"HeyGen Job Submitted. Video ID: {video_id}")
        
        # Poll
        for _ in range(60): # Wait up to 5-10 mins (they can be slow)
            status_data = check_video_status(video_id)
            status = status_data.get("status")
            
            if status == "completed":
                url = status_data.get("video_url")
                logger.info(f"✅ Video Generated: {url}")
                return url
            elif status == "failed":
                err = status_data.get("error")
                raise HeyGenError(f"Generation Failed: {err}")
            
            logger.info(f"HeyGen Polling for {video_id}: {status}...")
            time.sleep(10)
            
        raise HeyGenError("Timeout waiting for HeyGen")

    except Exception as e:
        logger.error(f"HeyGen Error: {e}")
        return None
