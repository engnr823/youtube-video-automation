import os
import logging
import replicate

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- MODEL CONFIGURATION ---
# Model 1: Cinematic B-Roll (Wan 2.1 - High Quality, No Lip Sync)
SCENE_MODEL_ID = "wan-video/wan-2.1-1.3b"

# Model 2: Lip-Sync Engine (SadTalker - Makes static images talk)
# Use this if you pass specific audio per scene.
LIP_SYNC_MODEL_ID = "cjwbw/sadtalker:3aa3dac9353cc4d6bd62a8f95957bd844003b401ca4e4a9b33baa574c549d376"

def generate_video_scene_with_replicate(prompt: str, image_url: str = None, aspect: str = "16:9") -> str:
    """
    Generates a cinematic B-Roll video using Wan 2.1.
    Handles Aspect Ratio correctly.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logging.info(f"🎬 Animating Cinematic Scene: '{prompt[:30]}...' | Aspect: {aspect}")

    try:
        input_payload = {
            "prompt": prompt, 
            "negative_prompt": "distortion, morphing, static, low quality, watermark, text, bad anatomy",
            "aspect_ratio": aspect,
            "quality": "high"
        }
        
        # Wan 2.1 specific: ensure image is passed if available
        if image_url:
            input_payload["image"] = image_url
        
        output = replicate.run(SCENE_MODEL_ID, input=input_payload)

        # Handle output (Replicate usually returns a list for video models)
        video_url = output[0] if isinstance(output, list) else str(output)
        return video_url

    except Exception as e:
        logging.error(f"🔴 Replicate Cinematic Animation Error: {e}")
        # If video generation fails, return the image so the pipeline doesn't break
        return image_url 

def generate_lip_sync_with_replicate(image_url: str, audio_url: str) -> str:
    """
    Generates a talking head video synced to audio using SadTalker.
    Requires: A face image and an audio file URL.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logging.info(f"🗣️ Generating Lip-Sync for image...")

    try:
        output = replicate.run(
            LIP_SYNC_MODEL_ID,
            input={
                "source_image": image_url,
                "driven_audio": audio_url,
                "enhancer": "gfpgan", # Improves face quality
                "preprocess": "full",
                "still": True # Keeps head stable, focuses on mouth movement
            }
        )
        # SadTalker returns a single URL string
        return str(output)

    except Exception as e:
        logging.error(f"🔴 Lip-Sync Error: {e}")
        raise
