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

    # FORCE PUBLIC AVATAR MODE
    # We ignore the passed 'avatar_id' if it matches your broken one
    # and swap it for a known working public avatar.
    
    real_avatar_id = avatar_id
    if "4343" in avatar_id:
        logging.warning(f"Swapping broken ID {avatar_id} for public 'Edward'")
        real_avatar_id = "Avatar_Expressive_20240520_01"

    character = {
        "type": "avatar",
        "avatar_id": real_avatar_id,
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

    logging.info(f"Submitting HeyGen Job for Avatar: {real_avatar_id}")
    response = _request("POST", "/v2/video/generate", payload)
    
    video_id = response.get("data", {}).get("video_id")
    if not video_id: raise HeyGenError("No video_id returned")

    return _wait_for_job(video_id)

# -------------------------------------------------
# AVATAR HELPERS
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    AVATARS = {
        # Edward (Public Male)
        "male": "Avatar_Expressive_20240520_01",
        # Tyler (Public Female)
        "female": "Avatar_Expressive_20240520_02"
    }
    # Default to Edward
    return AVATARS.get(avatar_type.lower(), "Avatar_Expressive_20240520_01")
