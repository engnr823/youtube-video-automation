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

# Increased polling interval to be safe
POLL_INTERVAL = 10       
MAX_WAIT_TIME = 600      # 10 minutes

logging.basicConfig(level=logging.INFO)

# LOGGING PROOF THAT NEW CODE IS ACTIVE
logging.info("*** NEW HEYGEN CLIENT CODE LOADED (v4-FIXED) ***")

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
        
        # Handle 404 gracefully during status checks
        if response.status_code == 404 and "status" in endpoint:
            logging.warning(f"Status 404 for {endpoint}. Job might be initializing. Retrying...")
            return None

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
        # Swallow 404 errors during status checks so we can retry
        if "404" in str(e) and "status" in endpoint:
            return None
        raise HeyGenError(f"HeyGen Error: {str(e)}")

# -------------------------------------------------
# JOB POLLING (With V1 Endpoint Fix)
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    logging.info(f"HeyGen Polling for Video ID: {video_id}")
    
    # Wait 15s initially to allow HeyGen to register the ID
    time.sleep(15)
    
    start = time.time()
    
    while time.time() - start < MAX_WAIT_TIME:
        # --- CRITICAL FIX: USE V1 ENDPOINT FOR STATUS CHECK ---
        # The V2 endpoint often returns 404 for specific avatar jobs. 
        # V1 is more reliable for status polling.
        data = _request("GET", f"/v1/video_status.get?video_id={video_id}")
        
        if data is None:
            logging.info(f"Video {video_id} not found yet... waiting.")
            time.sleep(POLL_INTERVAL)
            continue

        # V1 response structure: { "data": { "status": "...", "video_url": "..." } }
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

    # --- CRITICAL FIX: REMOVED FORCED "TALKING PHOTO" MODE ---
    # We now always use the standard avatar mode, which supports full video generation.
    character = {
        "type": "avatar",
        "avatar_id": avatar_id,
        "avatar_style": "normal"
    }
    logging.info(f"Using Standard Avatar mode for ID: {avatar_id}")

    payload = {
        "video_inputs": [{
            "character": character,
            "voice": {"type": "audio", "audio_url": audio_url},
            "background": {"type": "color", "value": background_color}
        }],
        "dimension": dimension
    }

    logging.info(f"Submitting HeyGen Job...")
    response = _request("POST", "/v2/video/generate", payload)
    
    video_id = response.get("data", {}).get("video_id")
    if not video_id:
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id: 
        logging.error(f"Full Response: {response}")
        raise HeyGenError("No video_id returned")

    return _wait_for_job(video_id)

# -------------------------------------------------
# AVATAR HELPERS
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    AVATARS = {
        "male": "Avatar_Expressive_20240520_02", # Tyler (Public)
        "female": "Avatar_Expressive_20240520_02" # Rachel (Public) - Change if you have a specific female ID
    }
    return AVATARS.get(avatar_type.lower(), "Avatar_Expressive_20240520_02")
