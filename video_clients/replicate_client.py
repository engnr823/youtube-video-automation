import os
import logging
import replicate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- MODEL CONFIGURATION ---
# Model 1: Cinematic B-Roll (Wan 2.1 - High Quality)
SCENE_MODEL_ID = "wan-video/wan-2.1-1.3b"

# Model 2: Lip-Sync Engine (SadTalker - Verified Public ID)
# [CRITICAL FIX] Using the stable 'lucataco' version to avoid 422 Errors
LIP_SYNC_MODEL_ID = "lucataco/sadtalker:85c698db7c0a66d5011435d0191bd32305a9c7499252a9041270252565697697"

def generate_video_scene_with_replicate(prompt: str, image_url: str = None, aspect: str = "16:9") -> str:
    """
    Generates a cinematic B-Roll video using Wan 2.1.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logger.info(f"🎬 Animating Cinematic Scene: '{prompt[:30]}...' | Aspect: {aspect}")

    try:
        input_payload = {
            "prompt": prompt, 
            "negative_prompt": "distortion, morphing, static, low quality, watermark, text, bad anatomy, cartoon",
            "aspect_ratio": aspect
        }
        
        # Wan 2.1 optionally accepts an image for image-to-video
        if image_url:
            input_payload["image"] = image_url
        
        output = replicate.run(SCENE_MODEL_ID, input=input_payload)

        # Handle list vs string output safely
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return str(output)

    except Exception as e:
        logger.error(f"🔴 Replicate Cinematic Animation Error: {e}")
        # Return None or raise depending on preference. Raising allows retry logic in worker.
        raise

def generate_lip_sync_with_replicate(image_url: str, audio_url: str) -> str:
    """
    Generates a talking head video using SadTalker with Natural Motion settings.
    """
    if not REPLICATE_API_TOKEN:
        raise ConnectionError("Replicate token missing.")

    logger.info(f"🗣️ Generating Lip-Sync for image...")

    try:
        output = replicate.run(
            LIP_SYNC_MODEL_ID,
            input={
                "source_image": image_url,
                "driven_audio": audio_url,
                "still": True,             # Keep head relatively stable to prevent background warp
                "enhancer": "gfpgan",      # Sharpen face (HD Fix)
                "preprocess": "full",      # Use full frame
                "expression_scale": 1.1,   # [FIX] Add 10% emotion movement
                "ref_eyeblink": None,      # [FIX] Auto-generate blinking
                "ref_pose": None           # [FIX] Auto-generate head motion
            }
        )
        
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return str(output)

    except Exception as e:
        logger.error(f"🔴 Lip-Sync Error: {e}")
        raise
