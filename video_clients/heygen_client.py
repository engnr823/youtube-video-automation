# file: video_clients/heygen_client.py

import os
import time
import requests
import logging
from typing import Optional, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)

# Assuming HeyGen API key is stored in the environment
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_API_URL = "https://api.heygen.com/v1" # Example URL

class HeyGenError(Exception):
    """Custom exception for HeyGen API errors."""
    pass

def retry_if_job_not_ready(exception):
    """Retry condition: only retry if the job is processing or pending."""
    return isinstance(exception, HeyGenError) and ("processing" in str(exception) or "pending" in str(exception))

# file: video_clients/heygen_client.py (CRITICAL FIX APPLIED TO _make_request)

# ... (imports and HEYGEN_API_URL definitions above)

def _make_request(method: str, endpoint: str, **kwargs) -> Any:
    """Handles API requests and common error checking."""
    if not HEYGEN_API_KEY:
        raise HeyGenError("HEYGEN_API_KEY is not set.")
    
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
        # Catch specific API errors and raise a clean exception
        status_code = e.response.status_code
        
        # --- CRITICAL FIX START: Robust Error Detail Extraction ---
        error_detail = f"API Error {status_code}"
        try:
            # Attempt to parse the API's JSON error message
            error_data = e.response.json()
            # Use 'detail' (common) or 'message' (less common)
            error_detail = error_data.get('detail', error_data.get('message', error_detail))
        except requests.exceptions.JSONDecodeError:
            # If the response body is empty/not JSON (which is what crashed it)
            error_detail = f"Non-JSON response for status {status_code}: {e.response.text[:150]}..."
        except Exception:
            pass
        # --- CRITICAL FIX END ---
        
        # Raise a custom error with better information
        raise HeyGenError(f"API Request Failed ({status_code}): {error_detail}") from e
    except Exception as e:
        raise HeyGenError(f"Network Error: {e}") from e

# ... (rest of the file contents below)


def _poll_job_status(job_id: str, max_wait: int = 400) -> str:
    """Polls the API until the video job is complete (success or failure)."""
    endpoint = f"/jobs/{job_id}/status"
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
            # If the error is network related, we allow the outer function's retry to handle it.
            if "Job failed" in str(e):
                 raise # Re-raise failure immediately
            # Otherwise, continue polling if it's not a definitive failure status
            logging.warning(f"Polling error for Job {job_id}: {e}")
            
    raise HeyGenError(f"Job {job_id} timed out after {max_wait} seconds.")


# file: video_clients/heygen_client.py

# ... (function definitions above)

def generate_heygen_video(
    avatar_id: str,
    audio_url: str,
    scene_prompt: str,
    scene_duration: float,
    aspect: str,
    ref_image_url: Optional[str] = None
) -> str:
    """
    Submits a video generation job and polls the status until complete.
    This replaces the SadTalker/Wan-Video multi-step process.
    """
    from .heygen_models import HeyGenVideoRequest
    
    # 1. Prepare and Validate Request Body
    request_data = HeyGenVideoRequest(
        avatar_id=avatar_id,
        audio_url=audio_url,
        scene_prompt=scene_prompt,
        ratio=aspect,
        scene_duration=scene_duration,
        ref_image_url=ref_image_url
    )
    
    # 2. Submit Job
    logging.info(f"HeyGen: Submitting job for Avatar ID {avatar_id}...")
    
    # CRITICAL FIX: Changing the incorrect API endpoint (/jobs/video) 
    # to the standard V1 asynchronous video generation endpoint (/video.generate)
    submission_data = _make_request("POST", "/video.generate", json=request_data.dict())
    
    job_id = submission_data.get("job_id")
    
    if not job_id:
        # NOTE: Some API versions return the video_url directly on success.
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



# Example of avatar creation (for your casting step)
def create_or_get_avatar(char_name: str, ref_image: Optional[str] = None) -> Optional[str]:
    """
    Simulated function to return a persistent HeyGen Avatar ID based on character name.
    
    In a real setup:
    1. Check a local database (CHAR_DB) for an existing 'heygen_avatar_id' for 'char_name'.
    2. If not found, call HeyGen's Avatar Creation API using 'ref_image' (if provided).
    3. Cache and return the new HeyGen ID.
    
    For now, we use a simple lookup for simulation to enable multi-character logic in the worker.
    """
    logging.warning(f"HeyGen: Simulating avatar creation/lookup for {char_name}.")

    # 1. Simple, simulated persistent ID assignment based on name
    name_key = char_name.upper().replace(" ", "_").strip()

    # Use a dictionary to map common names to unique, persistent (mock) HeyGen IDs
    MOCK_AVATAR_MAP = {
        "ALI": "AVTR_MALE_CONSISTENT_001",
        "ZARA": "AVTR_FEMALE_CONSISTENT_002",
        "KABIR": "AVTR_MALE_CONSISTENT_003",
        "NARRATOR": "AVTR_NARRATOR_GENERIC_000",
        # Add more names as needed based on common storyboards
    }
    
    # 2. Check if the name is in the map, otherwise create a new mock ID
    avatar_id = MOCK_AVATAR_MAP.get(name_key)
    
    if not avatar_id:
        # Create a simpler, unique ID for consistency if not a predefined name
        avatar_id = f"AVTR_DYNAMIC_{name_key}"
        logging.info(f"Assigned dynamic mock avatar ID: {avatar_id}")

    return avatar_id
