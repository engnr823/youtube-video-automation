import os
import time
import requests
import logging
from typing import Optional, Dict, Any, List

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_BASE_URL = "https://api.heygen.com"
SAFE_AVATAR_ID = "josh_lite3_20230714" 

# Standard Green Screen color for potential chroma keying
GREEN_SCREEN_COLOR = "#00B140"

# Increased polling settings for stability
POLL_INTERVAL = 10       
MAX_WAIT_TIME = 600      # 10 minutes

logging.basicConfig(level=logging.INFO)
logging.info("*** NEW HEYGEN CLIENT CODE LOADED (v7-BACKGROUNDS) ***")

class HeyGenError(Exception):
    pass

# -------------------------------------------------
# LOW LEVEL REQUEST HANDLER
# -------------------------------------------------

def _request(method: str, endpoint: str, payload: Optional[Dict] = None, timeout: int = 120) -> Dict[str, Any]:
    """
    Sends HTTP requests with robust error handling for API V2.
    """
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
            timeout=timeout
        )
        
        # Handle 404 gracefully during status checks (job initializing)
        if response.status_code == 404 and "status" in endpoint:
            logging.warning(f"Status 404 for {endpoint}. Retrying...")
            return None

        if response.status_code >= 400:
            try:
                err_data = response.json()
                msg = err_data.get("error", {}).get("message") or err_data.get("message")
                
                # Check for specific avatar errors to trigger fallback
                if "not found" in str(msg).lower():
                    raise HeyGenError(f"AVATAR_NOT_FOUND: {msg}")
                
                logging.error(f"HeyGen API Error: {msg}")
            except Exception as e:
                if isinstance(e, HeyGenError): raise e
                logging.error(f"HeyGen Raw Error: {response.text}")
        
        response.raise_for_status()
        return response.json()

    except HeyGenError:
        raise
    except Exception as e:
        # Swallow 404 errors during polling
        if "404" in str(e) and "status" in endpoint:
            return None
        raise HeyGenError(f"HeyGen Error: {str(e)}")

# -------------------------------------------------
# AVATAR MANAGEMENT
# -------------------------------------------------

def get_all_avatars() -> List[Dict]:
    """
    Fetches available avatars. Retries on timeout.
    """
    retries = 2
    for attempt in range(retries + 1):
        try:
            logging.info(f"Fetching avatars (Attempt {attempt+1})...")
            data = _request("GET", "/v2/avatars", timeout=120)
            if data:
                avatars = data.get("data", {}).get("avatars", [])
                logging.info(f"✅ Found {len(avatars)} avatars.")
                return avatars
        except Exception:
            time.sleep(2)
    return []

def get_safe_fallback_id():
    return SAFE_AVATAR_ID

# -------------------------------------------------
# JOB POLLING
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    logging.info(f"HeyGen Polling for Video ID: {video_id}")
    time.sleep(15) # Allow job to register
    start = time.time()
    
    while time.time() - start < MAX_WAIT_TIME:
        # Use V1 status endpoint for reliability
        data = _request("GET", f"/v1/video_status.get?video_id={video_id}")
        
        if data is None:
            time.sleep(POLL_INTERVAL)
            continue

        inner = data.get("data", {})
        status = inner.get("status")
        logging.info(f"Video Status: {status}")

        if status == "completed":
            url = inner.get("video_url")
            if url: return url
        
        if status == "failed":
            raise HeyGenError(f"Job failed: {inner.get('error')}")

        time.sleep(POLL_INTERVAL)

    raise HeyGenError("Timeout waiting for video")

# -------------------------------------------------
# MAIN GENERATION FUNCTION
# -------------------------------------------------

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    aspect_ratio: str = "9:16",
    background_image_url: str = None,
    background_color: str = "#000000"
) -> str:
    """
    Generates a video scene. Supports background images.
    """
    
    # 1. Dimensions
    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    # 2. Configure Background
    if background_image_url:
        background = {"type": "image", "url": background_image_url}
    else:
        background = {"type": "color", "value": background_color}

    # 3. Build Payload
    payload = {
        "video_inputs": [{
            "character": {
                "type": "avatar",
                "avatar_id": avatar_id,
                "avatar_style": "normal"
            },
            "voice": {
                "type": "audio",
                "audio_url": audio_url
            },
            "background": background
        }],
        "dimension": dimension
    }

    # 4. Submit with Fallback
    try:
        logging.info(f"Submitting Job: Avatar={avatar_id}, BG={'Image' if background_image_url else 'Color'}")
        response = _request("POST", "/v2/video/generate", payload)
    
    except HeyGenError as e:
        # Retry with Safe Avatar if original was not found
        if "AVATAR_NOT_FOUND" in str(e) and avatar_id != SAFE_AVATAR_ID:
            logging.warning(f"🚨 Avatar {avatar_id} not found. Retrying with {SAFE_AVATAR_ID}...")
            payload["video_inputs"][0]["character"]["avatar_id"] = SAFE_AVATAR_ID
            response = _request("POST", "/v2/video/generate", payload)
        else:
            raise e

    # 5. Get Video ID
    video_id = response.get("data", {}).get("video_id")
    if not video_id:
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id: 
        raise HeyGenError("No video_id returned from API")

    return _wait_for_job(video_id)
