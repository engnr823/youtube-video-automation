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
        raise HeyGenError("HEYGEN_API_KEY is missing from environment variables")

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
        
        # Log detail before raising for status
        if response.status_code >= 400:
            try:
                err_data = response.json()
                # Extract deeper error message if available
                detail = err_data.get("error", {}).get("message") or err_data.get("message")
                logging.error(f"HeyGen API Error Detail: {detail}")
            except:
                logging.error(f"HeyGen Raw Error: {response.text}")
        
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        try:
            data = e.response.json()
            message = data.get("error", {}).get("message") or data.get("message") or str(data)
        except:
            message = e.response.text[:200]
        raise HeyGenError(f"HeyGen API ERROR [{e.response.status_code}]: {message}")

    except Exception as e:
        raise HeyGenError(f"HeyGen Network/Request Error: {str(e)}")


# -------------------------------------------------
# JOB POLLING (FIXED FOR V2)
# -------------------------------------------------

def _wait_for_job(video_id: str) -> str:
    """
    Polls the status of a video generated via V2 API.
    Note: V2 uses /v2/video/status?video_id=...
    """
    logging.info(f"HeyGen Polling started for Video ID: {video_id}")
    start = time.time()

    while time.time() - start < MAX_WAIT_TIME:
        # IMPORTANT: For V2 generation, we check status via video_id
        data = _request("GET", f"/v2/video/status?video_id={video_id}")
        
        # V2 response structure is usually {"data": {"status": "...", "video_url": "..."}}
        inner_data = data.get("data", {})
        status = inner_data.get("status")

        logging.info(f"HeyGen Video {video_id} status: {status}")

        if status == "completed":
            video_url = inner_data.get("video_url")
            if not video_url:
                raise HeyGenError("Job completed but video_url is missing from response")
            return video_url

        if status == "failed":
            error_info = inner_data.get("error") or "Unknown failure"
            raise HeyGenError(f"HeyGen job failed: {error_info}")

        time.sleep(POLL_INTERVAL)

    raise HeyGenError("HeyGen job timed out after 10 minutes")


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
    Generate HeyGen video using V2 API.
    """

    if aspect_ratio == "9:16":
        dimension = {"width": 1080, "height": 1920}
    else:
        dimension = {"width": 1280, "height": 720}

    # Structure payload for HeyGen V2 API
    payload = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": avatar_id,
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

    logging.info(f"Submitting HeyGen V2 video job (Avatar: {avatar_id})...")
    response = _request("POST", "/v2/video/generate", payload)

    # V2 returns {"data": {"video_id": "..."}}
    video_id = response.get("data", {}).get("video_id")
    
    if not video_id:
        # Fallback check for different response formats
        video_id = response.get("video_id") or response.get("job_id")

    if not video_id:
        logging.error(f"Invalid HeyGen Response: {response}")
        raise HeyGenError("HeyGen did not return a video_id/job_id")

    return _wait_for_job(video_id)


# -------------------------------------------------
# AVATAR HELPERS (UPDATED WITH YOUR IDS)
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    """
    Uses the specific IDs provided:
    Male: Your personal image avatar
    Female: HeyGen Public avatar
    """
    AVATARS = {
        "male": "4343bfb447bf4028a48b598ae297f5dc",   # Your ID
        "female": "26f5fc9be1fc47eab0ef65df30d47a4e" # Public Female ID
    }
    
    # Default to your male ID if type is not recognized
    fallback = "4343bfb447bf4028a48b598ae297f5dc" 

    avatar_id = AVATARS.get(avatar_type.lower(), fallback)
    return avatar_id
