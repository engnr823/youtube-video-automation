import os
import time
import requests
import logging
from typing import Optional, Dict, Any

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_BASE_URL = "https://api.heygen.com"

POLL_INTERVAL = 5        # seconds
MAX_WAIT_TIME = 600      # seconds (10 minutes)

logging.basicConfig(level=logging.INFO)

# Global cache to store IDs so we don't ask the API every single time
CACHED_AVATARS = {
    "male": None,
    "female": None
}

class HeyGenError(Exception):
    pass

# -------------------------------------------------
# LOW LEVEL REQUEST HANDLER
# -------------------------------------------------

def _request(method: str, endpoint: str, payload: Optional[Dict] = None) -> Dict[str, Any]:
    if not HEYGEN_API_KEY:
        raise HeyGenError("HEYGEN_API_KEY is missing")

    headers = {
        "X-Api-Key": HEYGEN_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.request(
            method=method,
            url=f"{HEYGEN_BASE_URL}{endpoint}",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code >= 400:
            try:
                err_data = response.json()
                msg = err_data.get("error", {}).get("message") or err_data.get("message")
                logging.error(f"HeyGen API Error: {msg}")
            except:
                logging.error(f"HeyGen Raw Error: {response.text}")
        
        response.raise_for_status()
        return response.json()

    except Exception as e:
        raise HeyGenError(f"HeyGen Error: {str(e)}")

# -------------------------------------------------
# DYNAMIC ID FETCHER (THE BYPASS)
# -------------------------------------------------

def _fetch_dynamic_avatar_id(gender_target: str) -> str:
    """
    Asks HeyGen for a list of avatars available to THIS account
    and returns the first one that matches the gender.
    """
    global CACHED_AVATARS
    
    # Return cached if available
    if CACHED_AVATARS.get(gender_target):
        return CACHED_AVATARS[gender_target]

    logging.info(f"🔍 Auto-discovering a valid {gender_target} avatar from HeyGen API...")
    
    try:
        data = _request("GET", "/v2/avatars")
        avatars = data.get("data", {}).get("avatars", [])
        
        for avatar in avatars:
            # Check if gender matches (or take any if gender is None)
            av_gender = avatar.get("gender", "male").lower()
            av_id = avatar.get("avatar_id")
            
            if av_id and (gender_target == "any" or av_gender == gender_target):
                logging.info(f"✅ Found Valid Avatar: {avatar.get('avatar_name')} (ID: {av_id})")
                CACHED_AVATARS[gender_target] = av_id
                return av_id
                
        # If no specific gender found, just return the very first avatar ID
        if avatars:
            fallback = avatars[0].get("avatar_id")
            logging.warning(f"⚠️ No {gender_target} found. Using fallback: {fallback}")
            return fallback
            
    except Exception as e:
        logging.error(f"Failed to auto-discover avatars: {e}")
        
    # Ultimate hardcoded fallback if API list fails
    return "Avatar_Expressive_20240520_01"

# -------------------------------------------------
# JOB POLLING
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    logging.info(f"HeyGen Polling for Video ID: {video_id}")
    start = time.time()

    while time.time() - start < MAX_WAIT_TIME:
        data = _request("GET", f"/v2/video/status?video_id={video_id}")
        inner = data.get("data", {})
        status = inner.get("status")

        logging.info(f"Video Status: {status}")

        if status == "completed":
            return inner.get("video_url")
        if status == "failed":
            raise HeyGenError(f"Job failed: {inner.get('error')}")

        time.sleep(POLL_INTERVAL)

    raise HeyGenError("Timeout waiting for video")

# -------------------------------------------------
# MAIN VIDEO GENERATION
# -------------------------------------------------

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    aspect_ratio: str = "9:16",
    background_color: str = "#000000"
) -> str:
    
    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    # Detect if we need to use the auto-discovered ID
    # If the incoming ID is the broken one, fetch a new one dynamically
    real_id = avatar_id
    if "4343" in str(avatar_id) or "Avatar_Expressive" in str(avatar_id):
        # We assume male context if the broken ID was passed
        real_id = _fetch_dynamic_avatar_id("male")

    character = {
        "type": "avatar",
        "avatar_id": real_id,
        "avatar_style": "normal"
    }

    payload = {
        "video_inputs": [{
            "character": character,
            "voice": {"type": "audio", "audio_url": audio_url},
            "background": {"type": "color", "value": background_color}
        }],
        "dimension": dimension
    }

    logging.info(f"Submitting HeyGen Job using Avatar ID: {real_id}")
    response = _request("POST", "/v2/video/generate", payload)
    
    video_id = response.get("data", {}).get("video_id")
    if not video_id:
        # Fallback for V2 sometimes returning 'video_id' at root or 'job_id'
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id: 
        raise HeyGenError("No video_id returned")

    return _wait_for_job(video_id)

# -------------------------------------------------
# AVATAR HELPERS
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    """
    Uses the API to find a valid avatar ID automatically.
    """
    return _fetch_dynamic_avatar_id(avatar_type)
