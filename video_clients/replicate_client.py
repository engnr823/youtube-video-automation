import os
import logging
import replicate
import time

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# --- CLIENT INITIALIZATION ---
if REPLICATE_API_TOKEN:
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
else:
    client = None
    logger.warning("⚠️ REPLICATE_API_TOKEN is missing.")

# --- HELPER: SAFE RUNNER (Fixes 404 Errors) ---
def run_replicate_safe(model_ref: str, input_data: dict) -> str:
    """
    Runs a Replicate model safely.
    - If it's an official model (no version hash needed), runs directly.
    - If it's a community model, fetches the latest version first.
    """
    if not client:
        raise ConnectionError("Replicate Client not initialized.")

    # 1. Official Models (Do NOT fetch version, run directly)
    official_models = ["black-forest-labs/flux-schnell", "wan-video/wan-2.1-1.3b"]
    
    try:
        if model_ref in official_models:
            logger.info(f"🚀 Running Official Model: {model_ref}")
            output = client.run(model_ref, input=input_data)
        else:
            # 2. Community Models (Fetch Version ID)
            if ":" not in model_ref:
                model = client.models.get(model_ref)
                version = model.versions.list()[0].id
                model_ref = f"{model_ref}:{version}"
            
            logger.info(f"🚀 Running Community Model: {model_ref}")
            output = client.run(model_ref, input=input_data)

        # Normalize Output
        if isinstance(output, list) and output:
            return str(output[0])
        elif isinstance(output, dict):
            return output.get("url") or output.get("output") or str(output)
        return str(output)

    except Exception as e:
        logger.error(f"🔴 Replicate Error ({model_ref}): {e}")
        raise

# --- 1. IMAGE GENERATION (FLUX) ---
def generate_image_flux(prompt: str, aspect_ratio: str = "16:9") -> str:
    return run_replicate_safe(
        "black-forest-labs/flux-schnell",
        {"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "jpg"}
    )

# --- 2. VIDEO GENERATION (WAN 2.1) ---
def generate_video_wan(prompt: str, image_url: str = None, aspect_ratio: str = "16:9") -> str:
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "negative_prompt": "distortion, low quality, text, watermark"
    }
    if image_url:
        payload["image"] = image_url  # Image-to-Video
    
    return run_replicate_safe("wan-video/wan-2.1-1.3b", payload)

# --- 3. LIP SYNC (SADTALKER) ---
def generate_lip_sync(face_url: str, audio_url: str) -> str:
    return run_replicate_safe(
        "lucataco/sadtalker",
        {
            "source_image": face_url,
            "driven_audio": audio_url,
            "still": True,
            "enhancer": "gfpgan",
            "expression_scale": 1.1
        }
    )
