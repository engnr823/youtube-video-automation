import os
import logging
import replicate

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# -------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------

# Model 1: Wan 2.1 (Video Generation)
SCENE_MODEL_ID = "wan-video/wan-2.1-1.3b"

# Model 2: SadTalker (Stable Version from lucataco)
# Confirmed working and DOES NOT give 422 error.
LIP_SYNC_MODEL_ID = (
    "lucataco/sadtalker:85c698db7c0a66d5011435d0191bd32305a9c7499252a9041270252565697697"
)

# -------------------------------------------------
# WAN 2.1 VIDEO GENERATOR
# -------------------------------------------------
def generate_video_scene_with_replicate(prompt: str, image_url: str = None, aspect: str = "16:9") -> str:
    """
    Generates a cinematic video scene using Wan 2.1.
    Supports both text-to-video and image-to-video.
    """

    if not REPLICATE_API_TOKEN:
        raise ConnectionError("REPLICATE_API_TOKEN is missing in environment variables.")

    logger.info(f"🎬 Generating WAN 2.1 Video | Prompt: {prompt[:40]}... | Aspect: {aspect}")

    try:
        payload = {
            "prompt": prompt,
            "negative_prompt": (
                "distortion, deformed face, bad anatomy, low quality, flickering, watermark, text, cartoon"
            ),
            "aspect_ratio": aspect,
        }

        # If user uploads an image for image-to-video
        if image_url:
            payload["image"] = image_url

        output = replicate.run(SCENE_MODEL_ID, input=payload)

        # Replicate may return: list[video_url] or just a string
        if isinstance(output, list) and len(output):
            return output[0]
        return str(output)

    except Exception as e:
        logger.error(f"🔴 WAN Video Generation Error: {e}")
        raise


# -------------------------------------------------
# SADTALKER LIP SYNC GENERATOR
# -------------------------------------------------
def generate_lip_sync_with_replicate(image_url: str, audio_url: str) -> str:
    """
    Generates a speaking avatar using SadTalker.
    Fully compatible with lucataco repo version.
    """

    if not REPLICATE_API_TOKEN:
        raise ConnectionError("REPLICATE_API_TOKEN is missing in environment variables.")

    logger.info("🗣️ Running SadTalker Lip Sync...")

    try:
        payload = {
            "source_image": image_url,
            "driven_audio": audio_url,

            # ---------- Motion & Stability Settings ----------
            "still": True,                 # Prevents background shaking
            "enhancer": "gfpgan",          # Face HD enhancer
            "preprocess": "full",          # Full frame detection
            "expression_scale": 1.1,       # Adds realistic emotional expression (FIX)

            # ---------- Auto Blink & Auto Pose ----------
            "ref_eyeblink": None,          # Auto-generate blinking
            "ref_pose": None,              # Auto head motion
        }

        output = replicate.run(LIP_SYNC_MODEL_ID, input=payload)

        if isinstance(output, list) and len(output):
            return output[0]
        return str(output)

    except Exception as e:
        logger.error(f"🔴 SadTalker Lip Sync Error: {e}")
        raise

