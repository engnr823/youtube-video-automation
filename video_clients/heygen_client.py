# file: video_clients/heygen_client.py

import os
import time
import requests
import logging
from typing import Optional, Any
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Assuming HeyGen API key is stored in the environment
HEYGEN_API_KEY = os.getenv("HEYGEN_API_KEY")
HEYGEN_API_URL = "https://api.heygen.com/v1" # Example URL

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
        error_detail = e.response.json().get('detail', 'Unknown API Error')
        raise HeyGenError(f"API Error {status_code}: {error_detail}") from e
    except Exception as e:
        raise HeyGenError(f"Network Error: {e}") from e


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
    submission_data = _make_request("POST", "/jobs/video", json=request_data.dict())
    job_id = submission_data.get("job_id")
    
    if not job_id:
        raise HeyGenError("Job submission failed to return a Job ID.")

    # 3. Poll Status
    video_url = _poll_job_status(job_id)
    
    if not video_url:
        raise HeyGenError("Job completed but returned no video URL.")
        
    return video_url


# Example of avatar creation (for your casting step)
def create_or_get_avatar(char_name: str, ref_image: str) -> Optional[str]:
    """
    Conceptual function to check if avatar exists or create it, returning the Avatar ID.
    This replaces the Flux image generation in the casting step.
    """
    # NOTE: HeyGen requires a specific image format for avatar creation.
    logging.warning(f"HeyGen: Simulating avatar creation for {char_name}. Replace with real API call.")
    # In a real setup, this would call /avatars/create and return the new ID
    return f"AVTR_{char_name.upper()}_REAL"
