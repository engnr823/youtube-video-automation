# file: video_clients/heygen_client.py

import os
import time
import requests
import logging
import uuid
from typing import Optional, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)

# Assuming HeyGen API key is stored in the environment
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
# CRITICAL FIX: Base URL must be only the domain for correct path concatenation
HEYGEN_API_URL = "https://api.heygen.com" 

class HeyGenError(Exception):
    """Custom exception for HeyGen API errors."""
    pass

def retry_if_job_not_ready(exception):
    """Retry condition: only retry if the job is processing or pending."""
    return isinstance(exception, HeyGenError) and ("processing" in str(exception) or "pending" in str(exception))

def _make_request(method: str, endpoint: str, **kwargs) -> Any:
    """Handles API requests and common error checking."""
    if not HEYGEN_API_KEY:
        raise HeyGenError("HEYGEN_API_KEY is not set.")
    
    # HeyGen V2 requires the API key in the header
    headers = {"X-Api-Key": HEYGEN_API_KEY, "Content-Type": "application/json"}
    
    try:
        response = requests.request(
            method,
            f"{HEYGEN_API_URL}{endpoint}",
            headers=headers,
            timeout=30,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code
        
        # Robust Error Detail Extraction
        error_detail = f"API Error {status_code}"
        try:
            error_data = e.response.json()
            error_detail = error_data.get('detail', error_data.get('message', error_detail))
        except requests.exceptions.JSONDecodeError:
            error_detail = f"Non-JSON response for status {status_code}: {e.response.text[:150]}..."
        except Exception:
            pass
        
        raise HeyGenError(f"API Request Failed ({status_code}): {error_detail}") from e
    except Exception as e:
        raise HeyGenError(f"Network Error: {e}") from e


def _poll_job_status(job_id: str, max_wait: int = 400) -> str:
    """Polls the API until the video job is complete (success or failure)."""
    # NOTE: Assuming status endpoint uses the V1 path /jobs/{id}/status
    endpoint = f"/v1/jobs/{job_id}" 
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        time.sleep(5) # Poll every 5 seconds
        try:
            status_data = _make_request("GET", endpoint)
            status = status_data.get("status", "pending")
            
            if status == "completed":
                logging.info(f"HeyGen Job {job_id} COMPLETED.")
                return status_data.get("video_url")
            
            if status == "failed":
                error_msg = status_data.get("error_message", "Job failed for unknown reason.")
                raise HeyGenError(f"Job {job_id} failed: {error_msg}")
            
            logging.info(f"HeyGen Job {job_id} status: {status}. Polling...")
            
        except HeyGenError as e:
            if "Job failed" in str(e):
                 raise
            logging.warning(f"Polling error for Job {job_id}: {e}")
            
    raise HeyGenError(f"Job {job_id} timed out after {max_wait} seconds.")


# ----------------- HEYGEN UTILITY FUNCTIONS (Conceptual Implementation) -----------------
# These functions are stubs for the full paid/custom avatar workflow.

def _upload_image_to_heygen(image_url: str) -> Optional[str]:
    """CONCEPTUAL: Returns image_key."""
    logging.warning("HEYGEN: Custom Avatar creation simulated. Requires real Upload Asset API.")
    return "MOCK_IMAGE_KEY_" + str(uuid.uuid4())[:8]

def _create_photo_avatar(name: str, image_key: str) -> Optional[str]:
    """CONCEPTUAL: Creates a Photo Avatar Group. Returns avatar_id."""
    logging.warning("HEYGEN: Custom Avatar creation simulated. Requires real Create Photo Avatar Group API.")
    return "AVATAR_GROUP_" + str(uuid.uuid4())[:8] 

# ----------------- MAIN VIDEO GENERATION FUNCTION (Final Payload Fix) -----------------

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    scene_prompt: str,
    scene_duration: float,
    aspect: str,
    ref_image_url: Optional[str] = None
) -> str:
    """
    Submits a video generation job with V2 API structure.
    """
    # V2 payload structure enforcement
    request_data = {
        "avatar_id": avatar_id,
        "video_inputs": [
            {
                "audio_type": "tts", # Required for external audio sources like ElevenLabs
                "audio_url": audio_url,
                "text": " ", # Placeholder text field to satisfy schema requirements
                "script_id": "MOCK_SCRIPT_ID", # Mock ID to satisfy schema
                "avatar_id": avatar_id
            }
        ],
        "ratio": aspect,
        "test": True, # Use test mode to check API connection and credits
    }

    # 2. Submit Job
    logging.info(f"HeyGen: Submitting job for Avatar ID {avatar_id} with aspect {aspect}...")
    
    # CRITICAL FIX: Using the /v2/video/generate endpoint.
    submission_data = _make_request("POST", "/v2/video/generate", json=request_data)
    
    job_id = submission_data.get("job_id")
    
    if not job_id:
        video_url = submission_data.get("video_url")
        if video_url:
             logging.info("HeyGen Job COMPLETED synchronously.")
             return video_url
        
        raise HeyGenError("Job submission failed to return a Job ID or Video URL.")

    # 3. Poll Status
    video_url = _poll_job_status(job_id)
    
    if not video_url:
        raise HeyGenError("Job completed but returned no video URL.")
        
    return video_url


# ----------------- MAIN AVATAR CREATION LOGIC (Final ID Configuration) -----------------

def create_or_get_avatar(char_name: str, ref_image: Optional[str] = None) -> Optional[str]:
    """
    Handles multi-mode character assignment: Custom Avatar (for Mentor) or Stock Avatar (for Apprentice).
    """
    
    # 1. --- FINAL AVATAR IDs ---
    # User's Custom ID for the male character slot (Mentor/Ali):
    STOCK_ID_MALE = "Max-inTshirt-20220820" 
    
    # User-provided Public Stock Female Avatar ID (Zara/Apprentice):
    STOCK_ID_FEMALE = "26f5fc9be1fc47eab0ef65df30d47a4e" 
    # ---------------------------

    # 2. Custom Avatar Mode (User Uploaded Image)
    # The logic ensures that if an image is uploaded AND the character is male, we use the user's custom ID.
    if ref_image and ref_image.startswith("http"):
        name_key = char_name.upper()
        
        if name_key in ('ALI', 'KABIR', 'MENTOR') or 'MALE' in name_key:
            logging.warning(f"Custom Avatar Mode active. Using user's pre-created ID: {STOCK_ID_MALE}")
            return STOCK_ID_MALE

    # 3. Stock Avatar Mode (Default Fallback)
    
    name_key = char_name.upper()

    if name_key in ('ALI', 'KABIR', 'MENTOR') or 'MALE' in name_key:
         return STOCK_ID_MALE
    elif name_key in ('ZARA', 'APPRENTICE') or 'FEMALE' in name_key:
         return STOCK_ID_FEMALE
    
    # Final default fallback
    return STOCK_ID_MALE
