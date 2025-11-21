import os
import logging
import replicate

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
# Using Wan 2.1 Image-to-Video (This is much cheaper than Luma/Runway but great quality)
MODEL_ID = "wan-video/wan-2.1-1.3b"

def generate_video_scene_with_replicate(prompt: str, image_url: str = None) -> str:
    """
    Generates a video from a source image (Image-to-Video).
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logging.info(f"🎬 Animating Scene: '{prompt}' from Image...")

    try:
        input_payload = {
            "prompt": prompt,  # e.g., "Camera pans right, character waves"
            "negative_prompt": "distortion, morphing, static, low quality",
            "aspect_ratio": "16:9",
            "quality": "high"
        }
        
        # IF we provided an image (which we should for consistency), add it
        if image_url:
            input_payload["image"] = image_url
            # Wan 2.1 specific parameter for image influence
            # Some models call it 'start_image' or 'image'. Check Replicate docs for specific model version.
        
        output = replicate.run(MODEL_ID, input=input_payload)

        # Handle output (Wan usually returns a list)
        video_url = output[0] if isinstance(output, list) else output
        return video_url

    except Exception as e:
        logging.error(f"🔴 Replicate Animation Error: {e}")
        raise
