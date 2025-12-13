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

    payload = {
        "avatar_id": avatar_id,
        "dimension": dimension,
        "video_inputs": [
            {
                "audio": {
                    "type": "audio_url",
                    "url": audio_url
                },
                "background": {
                    "type": "color",
                    "value": background_color
                }
            }
        ]
    }

    logging.info("Submitting HeyGen video job...")
    response = _request("POST", "/v2/video/generate", payload)

    job_id = response.get("job_id")
    if not job_id:
        raise HeyGenError("HeyGen did not return job_id")

    return _wait_for_job(job_id)


# -------------------------------------------------
# AVATAR HELPERS (SAFE & REPLACEABLE)
# -------------------------------------------------

def get_stock_avatar(avatar_type: str = "male") -> str:
    """
    Replace these IDs with avatars from YOUR HeyGen dashboard.
    """
    AVATARS = {
        "male": "4343bfb447bf4028a48b598ae297f5dc",
        "female": "26f5fc9be1fc47eab0ef65df30d47a4e"
    }

    avatar_id = AVATARS.get(avatar_type.lower())
    if not avatar_id:
        raise HeyGenError("Invalid avatar type")

    return avatar_id
