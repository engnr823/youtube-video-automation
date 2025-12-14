# file: video_clients/heygen_client.py
import os
import time
import requests
import logging
from typing import Optional, Dict, Any, List
import random

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_BASE_URL = "https://api.heygen.com"

# --- CRITICAL FIX: UPDATED BACKUP AVATAR LIST ---
# If your account has no avatars, we rotate through these public stock IDs.
BACKUP_STOCK_AVATARS = [
    "daf0aeb75914449980d4407b1e427d14", # Pierce (Generic Male)
    "0f82e18b45614ee28864700057e494a8", # Wayne (Generic Male)
    "a83859550e6347149a4253109a933de3"  # Fin (Generic Male)
]
# Fallback if list fails
SAFE_AVATAR_ID = "daf0aeb75914449980d4407b1e427d14" 

# Standard Green Screen color for potential chroma keying
GREEN_SCREEN_COLOR = "#00B140"

POLL_INTERVAL = 10       
MAX_WAIT_TIME = 600      

logging.basicConfig(level=logging.INFO)
logging.info("*** NEW HEYGEN CLIENT CODE LOADED (v8-ROBUST-STOCK) ***")

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
            timeout=timeout
        )
        
        # Swallow 404s during polling (job initializing)
        if response.status_code == 404 and "status" in endpoint:
            logging.warning(f"Status 404 for {endpoint}. Retrying...")
            return None

        if response.status_code >= 400:
            try:
                err_data = response.json()
                msg = err_data.get("error", {}).get("message") or err_data.get("message")
                
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
        if "404" in str(e) and "status" in endpoint:
            return None
        raise HeyGenError(f"HeyGen Error: {str(e)}")

# -------------------------------------------------
# AVATAR MANAGEMENT
# -------------------------------------------------

def get_all_avatars() -> List[Dict]:
    """
    Fetches available avatars. If empty, returns generic STOCK avatars
    so the pipeline doesn't crash.
    """
    retries = 2
    for attempt in range(retries + 1):
        try:
            logging.info(f"Fetching avatars (Attempt {attempt+1})...")
            data = _request("GET", "/v2/avatars", timeout=120)
            if data:
                avatars = data.get("data", {}).get("avatars", [])
                
                # --- CRITICAL FIX: Handle Empty Account ---
                if not avatars:
                    logging.warning("⚠️ No Personal Avatars found! Injecting Stock Backups.")
                    return [
                        {"avatar_id": pid, "name": f"Stock_Backup_{i}"} 
                        for i, pid in enumerate(BACKUP_STOCK_AVATARS)
                    ]
                
                logging.info(f"✅ Found {len(avatars)} avatars.")
                return avatars
        except Exception as e:
            logging.warning(f"Avatar fetch failed: {e}")
            time.sleep(2)
            
    # Final safety net
    return [{"avatar_id": SAFE_AVATAR_ID, "name": "System_Fallback"}]

def get_safe_fallback_id():
    return SAFE_AVATAR_ID

# -------------------------------------------------
# JOB POLLING
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    logging.info(f"HeyGen Polling for Video ID: {video_id}")
    time.sleep(15) 
    start = time.time()
    
    while time.time() - start < MAX_WAIT_TIME:
        data = _request("GET", f"/v1/video_status.get?video_id={video_id}")
        
        if data is None:
            time.sleep(POLL_INTERVAL)
            continue

        inner = data.get("data", {})
        status = inner.get("status")
        
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
    
    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    if background_image_url:
        background = {"type": "image", "url": background_image_url}
    else:
        background = {"type": "color", "value": background_color}

    # Standardize Payload
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

    try:
        logging.info(f"Submitting Job: Avatar={avatar_id}")
        response = _request("POST", "/v2/video/generate", payload)
    
    except HeyGenError as e:
        # --- CRITICAL FIX: ROTATING FALLBACK ---
        # If the requested avatar is dead, try the backups
        if "AVATAR_NOT_FOUND" in str(e):
            logging.warning(f"🚨 Avatar {avatar_id} failed. Trying backup stock avatar...")
            
            # Pick a random backup that isn't the one we just tried
            backup_id = random.choice(BACKUP_STOCK_AVATARS)
            payload["video_inputs"][0]["character"]["avatar_id"] = backup_id
            
            try:
                logging.info(f"♻️ Retrying with Backup: {backup_id}")
                response = _request("POST", "/v2/video/generate", payload)
            except Exception as final_e:
                logging.error("All backups failed.")
                raise final_e
        else:
            raise e

    video_id = response.get("data", {}).get("video_id")
    if not video_id:
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id: 
        raise HeyGenError("No video_id returned from API")

    return _wait_for_job(video_id)
