# file: heygen_client.py

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

# Define the Green Screen Color for Compositing compatibility
GREEN_SCREEN_COLOR = "#00B140"

# Increased polling interval to be safe
POLL_INTERVAL = 10      
MAX_WAIT_TIME = 600      # 10 minutes

logging.basicConfig(level=logging.INFO)
logging.info("*** NEW HEYGEN CLIENT CODE LOADED (v6-ROBUST) ***")

class HeyGenError(Exception):
    pass

# -------------------------------------------------
# LOW LEVEL REQUEST HANDLER
# -------------------------------------------------

def _request(method: str, endpoint: str, payload: Optional[Dict] = None, timeout: int = 120) -> Dict[str, Any]:
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
            timeout=timeout  # Increased default timeout
        )
        
        # Handle 404 gracefully during status checks
        if response.status_code == 404 and "status" in endpoint:
            logging.warning(f"Status 404 for {endpoint}. Job might be initializing. Retrying...")
            return None

        if response.status_code >= 400:
            try:
                err_data = response.json()
                msg = err_data.get("error", {}).get("message") or err_data.get("message")
                # Specific check for Avatar Not Found to allow fallback logic
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
        # Swallow 404 errors during status checks so we can retry
        if "404" in str(e) and "status" in endpoint:
            return None
        raise HeyGenError(f"HeyGen Error: {str(e)}")

# -------------------------------------------------
# ROBUST AVATAR FETCHER
# -------------------------------------------------
def get_all_avatars() -> List[Dict]:
    """
    Fetches the list of available avatars with retries.
    Returns a list of avatar dictionaries or a fallback list if API fails.
    """
    retries = 2
    for attempt in range(retries + 1):
        try:
            logging.info(f"Fetching available avatars from HeyGen (Attempt {attempt+1})...")
            data = _request("GET", "/v2/avatars", timeout=120)
            
            if data:
                avatars = data.get("data", {}).get("avatars", [])
                logging.info(f"✅ Found {len(avatars)} avatars in account.")
                return avatars
                
        except Exception as e:
            logging.warning(f"Avatar fetch attempt {attempt+1} failed: {e}")
            time.sleep(2)

    logging.warning("⚠️ Failed to fetch avatars. Returning empty list (Logic will use fallback).")
    return []

# -------------------------------------------------
# JOB POLLING (V1 Endpoint Fix)
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    logging.info(f"HeyGen Polling for Video ID: {video_id}")
    time.sleep(15)
    start = time.time()
    
    while time.time() - start < MAX_WAIT_TIME:
        # V1 is more reliable for status polling
        data = _request("GET", f"/v1/video_status.get?video_id={video_id}")
        
        if data is None:
            logging.info(f"Video {video_id} not found yet... waiting.")
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
# MAIN VIDEO GENERATION
# -------------------------------------------------

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    aspect_ratio: str = "9:16",
    background_color: str = "#000000" # This parameter is used for Green Screen request
) -> str:
    
    # 1. Configuration
    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    # 2. Define Character
    character = {
        "type": "avatar",
        "avatar_id": avatar_id,
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

    # 3. Submit Job (With Auto-Fallback for Invalid IDs)
    try:
        logging.info(f"Submitting HeyGen Job for Avatar: {avatar_id}...")
        response = _request("POST", "/v2/video/generate", payload)
    
    except HeyGenError as e:
        # SAFETY NET: If the specific avatar is not found, retry with the SAFE ID
        if "AVATAR_NOT_FOUND" in str(e) and avatar_id != SAFE_AVATAR_ID:
            logging.warning(f"🚨 Avatar {avatar_id} failed. Retrying with SAFE AVATAR: {SAFE_AVATAR_ID}")
            
            # Update payload to use Safe Avatar
            payload["video_inputs"][0]["character"]["avatar_id"] = SAFE_AVATAR_ID
            response = _request("POST", "/v2/video/generate", payload)
        else:
            raise e

    # 4. Extract Video ID
    video_id = response.get("data", {}).get("video_id")
    if not video_id:
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id: 
        raise HeyGenError("No video_id returned from API")

    # 5. Poll for completion
    return _wait_for_job(video_id)

# -------------------------------------------------
# HELPER FOR WORKER
# -------------------------------------------------
def get_safe_fallback_id():
    return SAFE_AVATAR_ID
