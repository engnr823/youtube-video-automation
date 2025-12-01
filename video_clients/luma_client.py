import os
import logging
import time
import requests

# --- Luma AI API Configuration ---
LUMA_API_KEY = os.getenv("LUMA_API_KEY")
API_HOST = "https://api.luma.ai"
GENERATION_ENDPOINT = f"{API_HOST}/v1/dream"
STATUS_ENDPOINT = f"{API_HOST}/v1/tasks/"

if not LUMA_API_KEY:
    logging.warning("🔴 WARNING: LUMA_API_KEY is not set. Video generation will fail.")

def generate_video_scene_and_upload(prompt: str, duration: int, aspect: str = "16:9") -> str:
    """
    Generates a video scene using Luma AI's asynchronous API.
    
    This function starts a generation job, polls for its completion, and returns the
    final video URL hosted by Luma. No separate upload to Cloudinary is needed.

    Args:
        prompt (str): The visual prompt for the video scene.
        duration (int): The desired duration (note: Luma's API may have its own limits).
        aspect (str): The aspect ratio (e.g., "16:9" or "9:16").

    Returns:
        str: The URL of the generated MP4 video file.
    """
    if not LUMA_API_KEY:
        raise ConnectionError("Luma client is not initialized. Please set the LUMA_API_KEY.")

    headers = {"Authorization": f"Bearer {LUMA_API_KEY}"}
    
    # --- 1. Initiate Generation ---
    payload = {
        "user_prompt": prompt,
        "aspect_ratio": aspect,  # [FIX] Dynamic Aspect Ratio used here
        # Note: Luma's API might have specific ways to handle duration or it might be fixed.
    }
    
    logging.info(f"Initiating Luma video generation for prompt: '{prompt[:70]}...' | Aspect: {aspect}")
    try:
        response = requests.post(GENERATION_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        job_id = response.json().get("id")
        if not job_id:
            raise ValueError("Luma API did not return a job ID.")
        logging.info(f"Luma job started successfully with ID: {job_id}")
    except requests.RequestException as e:
        logging.error(f"🔴 Failed to initiate Luma generation: {e}")
        raise

    # --- 2. Poll for Completion ---
    # Poll for up to 10 minutes (60 tries * 10 seconds)
    for i in range(60):
        try:
            time.sleep(10)
            logging.info(f"Polling Luma job {job_id} (Attempt {i+1}/60)...")
            status_response = requests.get(f"{STATUS_ENDPOINT}{job_id}", headers=headers, timeout=30)
            status_response.raise_for_status()
            
            status_data = status_response.json()
            state = status_data.get("state")
            
            if state == "succeeded":
                video_url = status_data.get("video", {}).get("url")
                if not video_url:
                    raise ValueError("Luma job succeeded but no video URL was found.")
                logging.info(f"✅ Luma job {job_id} succeeded. Video URL: {video_url}")
                return video_url
            
            elif state == "failed":
                error_message = status_data.get("error", {}).get("message", "Unknown error")
                raise RuntimeError(f"Luma job {job_id} failed: {error_message}")
                
            # If state is 'pending' or 'processing', the loop continues.
            
        except requests.RequestException as e:
            logging.warning(f"Polling for Luma job {job_id} failed on attempt {i+1}: {e}. Retrying...")

    raise TimeoutError(f"Luma job {job_id} timed out after 10 minutes.")
