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


# -------------------------------------------------
# EXCEPTIONS
# -------------------------------------------------

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
            timeout=40
        )
        # Handle 400 errors specifically to show the message
        if response.status_code >= 400:
             try:
                 err_data = response.json()
                 err_msg = err_data.get("error", {}).get("message") or err_data.get("message")
                 logging.error(f"HeyGen API Error Detail: {err_msg}")
             except:
                 pass
        
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        try:
            data = e.response.json()
            message = data.get("message") or data.get("detail") or str(data)
        except Exception:
            message = e.response.text[:200]

        raise HeyGenError(f"HeyGen API ERROR [{e.response.status_code}]: {message}")

    except Exception as e:
        raise HeyGenError(f"HeyGen Network Error: {str(e)}")


# -------------------------------------------------
# JOB POLLING
# -------------------------------------------------

def _wait_for_job(job_id: str) -> str:
    logging.info(f"HeyGen Job started: {job_id}")
    start = time.time()

    while time.time() - start < MAX_WAIT_TIME:
        time.sleep(POLL_INTERVAL)

        data = _request("GET", f"/v1/jobs/{job_id}")
        status = data.get("status")

        logging.info(f"HeyGen Job {job_id} status: {status}")

        if status == "completed":
            result = data.get("result", {})
            video_url = result.get("video_url")
            if not video_url:
                raise HeyGenError("Job completed but video_url missing")
            return video_url

        if status == "failed":
            error = data.get("error_message", "Unknown failure")
            raise HeyGenError(f"HeyGen job failed: {error}")

    raise HeyGenError("HeyGen job timeout")


# -------------------------------------------------
# MAIN VIDEO GENERATION
# -------------------------------------------------

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    aspect_ratio: str = "9:16",
    background_color: str = "#000000"
) -> str:
    """
    Generate HeyGen video using external audio.
    """

    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    # CRITICAL FIX: Structure payload correctly for HeyGen V2 API
    # The API expects 'voice' (not 'audio') and 'character' inside video_inputs
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
                    "scale": 1.0,
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "audio",
                    "audio_url": audio_url
                },
                "background": {
                    "type": "color",
                    "value": background_color
                }
            }
        ],
        "dimension": dimension
    }

    logging.info("Submitting HeyGen video job...")
    response = _request("POST", "/v2/video/generate", payload)

    # Some endpoints return 'job_id', others 'data' -> 'job_id'
    data = response.get("data", response)
    job_id = data.get("job_id")
    
    if not job_id:
        # Fallback check
        job_id = response.get("job_id")

    if not job_id:
        logging.error(f"HeyGen Response: {response}")
        raise HeyGenError("HeyGen did not return job_id")

    return _wait_for_job(job_id)


# -------------------------------------------------
# AVATAR HELPERS (SAFE & REPLACEABLE)
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    """
    Replace these IDs with avatars from YOUR HeyGen dashboard.
    """
    # Updated IDs to likely valid stock ones (Example IDs)
    AVATARS = {
        "male": "37f4d924115147908b88eb342e47c17d", 
        "female": "48f4d924115147908b88eb342e47c17d" 
    }
    
    # Fallback to the ID seen in your logs (which seems valid as a string)
    fallback = "4343bfb447bf4028a48b598ae297f5dc" 

    avatar_id = AVATARS.get(avatar_type.lower(), fallback)
    return avatar_id
