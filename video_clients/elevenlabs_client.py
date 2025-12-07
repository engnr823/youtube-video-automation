import os
import uuid
import logging
import tempfile
from typing import Optional, List, Dict
from elevenlabs import ElevenLabs, VoiceSettings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Client Initialization ---
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.warning("🔴 WARNING: ELEVENLABS_API_KEY is not set. Voice generation will fail.")
    client = None
else:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def generate_audio_for_scene(text: str, voice_id: str) -> Optional[str]:
    """
    Generates audio for a single scene, saves it locally, and returns the path.
    Includes 'Nasal Voice Fix' settings.
    """
    if not client:
        logger.error("ElevenLabs client is not initialized.")
        return None

    # Fallback if voice_id is missing or empty
    if not voice_id:
        voice_id = "pqHfZKP75CvOlQylNhV4" # Default to 'Bill' (Deep Male)

    try:
        # Generate a unique filename in /tmp
        safe_id = str(uuid.uuid4())
        output_path = os.path.join(tempfile.gettempdir(), f"scene_audio_{safe_id}.mp3")

        logger.info(f"🎙️ Generating audio for voice: {voice_id}...")

        # [CRITICAL FIX] Settings for Natural Urdu/Hindi (Removes Robotic/Nasal tone)
        # stability=0.4: Allows more emotion and natural accent fluctuation
        # similarity_boost=0.5: Keeps the voice recognizable but not rigid
        audio_stream = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings=VoiceSettings(
                stability=0.4, 
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
        )

        # Write stream to file
        with open(output_path, "wb") as f:
            for chunk in audio_stream:
                if chunk:
                    f.write(chunk)
        
        logger.info(f"✅ Audio generated successfully: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"🔴 ElevenLabs Generation Failed: {e}")
        # Optional: Add retry logic here with a fallback voice if needed
        return None

# --- Legacy Wrappers (Kept for compatibility if other files call them) ---

def generate_voiceover_and_upload(script: str, voice_id: str) -> str:
    """
    Wrapper that generates audio and returns a path. 
    (The worker now handles the upload to Cloudinary, so we just return the path).
    """
    return generate_audio_for_scene(script, voice_id) or ""

def generate_multi_voice_audio(segments: List[Dict[str, str]]) -> str:
    """
    Placeholder. The new Celery Worker handles stitching via FFmpeg directly.
    """
    logger.warning("⚠️ generate_multi_voice_audio is deprecated. Logic moved to Celery Worker.")
    return ""
