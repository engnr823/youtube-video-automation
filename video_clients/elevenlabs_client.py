import os
import logging
import uuid
import cloudinary
import cloudinary.uploader
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
    Generates audio using the ElevenLabs API, saves it to a temporary file,
    uploads it to Cloudinary, and then cleans up the local file.

    Args:
        script (str): The text script to be converted to speech.
        voice_id (str): The ID of the ElevenLabs voice to use (e.g., "Rachel").

    Returns:
        str: The secure URL of the uploaded audio file on Cloudinary.
    """
    if not client:
        raise ConnectionError("ElevenLabs client is not initialized. Please set the ELEVENLABS_API_KEY.")

    if not script:
        logging.warning("Script is empty. Returning empty string.")
        return ""

    logging.info(f"Generating voiceover with voice: {voice_id}...")

    try:
        # 1. Generate audio bytes from ElevenLabs
        # Note: Using the modern 'text_to_speech.convert' method
        # generator returns an iterator, so we must consume it to write bytes
        audio_generator = client.text_to_speech.convert(
            text=script,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )

        # 2. Save audio bytes to a temporary file
        temp_filename = f"/tmp/voiceover_{uuid.uuid4()}.mp3"
        
        # Consume the generator to write the file
        with open(temp_filename, 'wb') as f:
            for chunk in audio_generator:
                f.write(chunk)

        # 3. Upload the temporary file to Cloudinary
        logging.info(f"Uploading voiceover '{temp_filename}' to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            temp_filename,
            resource_type="video"  # Cloudinary treats audio as a 'video' resource type
        )

        secure_url = upload_result.get('secure_url')
        if not secure_url:
            raise Exception("Cloudinary upload failed, no secure_url returned.")

        logging.info(f"✅ Voiceover uploaded successfully: {secure_url}")
        return secure_url

    except Exception as e:
        logging.error(f"🔴 An error occurred during voiceover generation or upload: {e}")
        raise
    finally:
        # 4. Clean up the temporary file
        if 'temp_filename' in locals() and os.path.exists(temp_filename):
            os.remove(temp_filename)
            logging.info(f"Cleaned up temporary file: {temp_filename}")
