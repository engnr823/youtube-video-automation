# video_clients/elevenlabs_client.py

import os
import logging
import uuid
import tempfile
import cloudinary
import cloudinary.uploader
from pathlib import Path
from elevenlabs.client import ElevenLabs

# --- Client Initialization ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logging.warning("🔴 WARNING: ELEVENLABS_API_KEY is not set. Voice generation will fail.")
    client = None
else:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def generate_voiceover_and_upload(script: str, voice_id: str) -> str:
    """
    Generates audio using the ElevenLabs API, saves it to a cross-platform temp file,
    uploads it to Cloudinary, and cleans up.
    """
    if not client:
        raise ConnectionError("ElevenLabs client is not initialized.")

    if not script:
        logging.warning("Script is empty. Returning empty string.")
        return ""

    logging.info(f"🎙️ Generating voiceover for: {voice_id}...")

    # [IMPROVEMENT] Use cross-platform temp directory instead of hardcoded "/tmp/"
    temp_dir = tempfile.gettempdir()
    temp_filename = os.path.join(temp_dir, f"voiceover_{uuid.uuid4()}.mp3")

    try:
        # 1. Generate audio stream from ElevenLabs
        # [CONFIRMATION] 'convert' returns a generator of bytes. This is the correct V1+ usage.
        audio_generator = client.text_to_speech.convert(
            text=script,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2", # Valid Flagship Model (Not Deprecated)
            output_format="mp3_44100_128"      # [ADDED] Explicitly request high-quality MP3
        )

        # 2. Write stream to file
        with open(temp_filename, 'wb') as f:
            for chunk in audio_generator:
                if chunk:
                    f.write(chunk)

        # 3. Upload to Cloudinary
        # [CONFIRMATION] resource_type="video" is CORRECT. Cloudinary treats audio as video.
        logging.info(f"☁️ Uploading to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            temp_filename,
            resource_type="video", 
            folder="voiceovers" # [ADDED] Keep your bucket organized
        )

        secure_url = upload_result.get('secure_url')
        if not secure_url:
            raise RuntimeError("Cloudinary upload failed: No secure_url returned")

        logging.info(f"✅ Voiceover Ready: {secure_url}")
        return secure_url

    except Exception as e:
        logging.error(f"🔴 Voiceover Generation Failed: {e}")
        raise  # Re-raise to trigger Celery retry logic

    finally:
        # 4. Safe Cleanup
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                logging.debug(f"🧹 Cleaned up: {temp_filename}")
        except Exception as cleanup_error:
            logging.warning(f"⚠️ Failed to delete temp file: {cleanup_error}")
