import os
import uuid
import logging
import tempfile
from openai import OpenAI

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def generate_openai_speech(text: str, voice_category: str = "neutral") -> str:
    """
    Generates TTS using OpenAI's 'tts-1' model.
    Returns the path to the saved MP3 file.
    """
    if not client:
        logger.error("🔴 OpenAI API Key missing. Cannot generate speech.")
        raise ValueError("OpenAI API Key missing.")

    try:
        # Map generic voice categories to OpenAI presets
        voice_map = {
            "male": "onyx",
            "female": "nova",
            "intense": "onyx",
            "happy": "alloy",
            "neutral": "alloy"
        }
        
        # Simple logic to pick a voice
        selected_voice = "alloy"
        for key, val in voice_map.items():
            if key in voice_category.lower():
                selected_voice = val
                break

        safe_id = str(uuid.uuid4())
        output_path = os.path.join(tempfile.gettempdir(), f"openai_tts_{safe_id}.mp3")

        logger.info(f"🎙️ OpenAI TTS: Generating '{text[:20]}...' with voice '{selected_voice}'")

        response = client.audio.speech.create(
            model="tts-1",
            voice=selected_voice,
            input=text
        )
        
        response.stream_to_file(output_path)
        logger.info(f"✅ OpenAI Audio saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"🔴 OpenAI TTS Failed: {e}")
        raise
