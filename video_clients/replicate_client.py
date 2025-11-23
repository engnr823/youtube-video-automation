import os
import logging
import replicate

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
# Using Wan 2.1 (Good choice for cost/performance)
MODEL_ID = "wan-video/wan-2.1-1.3b"

def generate_video_scene_with_replicate(prompt: str, image_url: str = None, aspect: str = "16:9") -> str:
    """
    Generates a video from a source image.
    [CRITICAL UPDATE] Now accepts 'aspect' to prevent Worker crashes.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logging.info(f"🎬 Animating Scene: '{prompt[:30]}...' | Aspect: {aspect}")

    try:
        input_payload = {
            "prompt": prompt, 
            "negative_prompt": "distortion, morphing, static, low quality, watermark, text",
            "aspect_ratio": aspect, # Now uses the dynamic variable (16:9 or 9:16)
            "quality": "high"
        }
        
        # Wan 2.1 specific: ensure image is passed if available
        if image_url:
            input_payload["image"] = image_url
        
        output = replicate.run(MODEL_ID, input=input_payload)

        # Handle output (Replicate usually returns a list for video models)
        video_url = output[0] if isinstance(output, list) else str(output)
        return video_url

    except Exception as e:
        logging.error(f"🔴 Replicate Animation Error: {e}")
        raise
