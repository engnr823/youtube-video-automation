import os
import logging
import replicate

# --- Replicate Client Initialization ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    logging.warning("🔴 WARNING: REPLICATE_API_TOKEN is not set. Video generation will fail.")

# The specific identifier for the Zeroscope v2 XL model on Replicate
# This model is a great, cost-effective choice for text-to-video.
MODEL_ID = "anotherjesse/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351"

def generate_video_scene_with_replicate(prompt: str, duration: int) -> str:
    """
    Generates a video scene using a model on the Replicate platform.

    Args:
        prompt (str): The visual prompt for the video scene.
        duration (int): The desired duration of the clip (used to calculate frames).

    Returns:
        str: The URL of the generated MP4 video file.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate client is not initialized. Please set the REPLICATE_API_TOKEN.")

    logging.info(f"Initiating Replicate video generation for prompt: '{prompt[:70]}...'")

    # This model generates video at 24 frames per second.
    # We calculate the number of frames needed based on the desired duration.
    num_frames = duration * 24

    try:
        # Define the input payload for the model
        input_payload = {
            "prompt": prompt,
            "num_frames": num_frames,
            "width": 1024,  # Standard 16:9 aspect ratio width
            "height": 576, # Standard 16:9 aspect ratio height
        }

        # Run the model on Replicate and wait for the output
        output_url = replicate.run(
            MODEL_ID,
            input=input_payload
        )

        if not output_url:
            raise ValueError("Replicate job succeeded but did not return a video URL.")

        logging.info(f"✅ Replicate job succeeded. Video URL: {output_url}")
        return output_url

    except Exception as e:
        logging.error(f"🔴 An error occurred during Replicate video generation: {e}")
        # Reraise the exception to let Celery know the task failed
        raise
