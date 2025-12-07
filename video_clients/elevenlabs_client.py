import os
import uuid
import logging
import tempfile
from typing import Optional, List, Dict
from elevenlabs import ElevenLabs, VoiceSettings
from time import sleep

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# ElevenLabs Client Initialization
# -----------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not ELEVENLABS_API_KEY:
    logger.warning("🔴 ELEVENLABS_API_KEY is not set. Voice generation will fail.")
    client = None
else:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# -----------------------------
# Audio Generation Function
# -----------------------------
def generate_audio_for_scene(text: str, voice_id: Optional[str] = None, retries: int = 2) -> Optional[Dict[str, str]]:
    """
    Generates audio for a single scene and returns a dict containing:
        - path: local audio file path
        - voice_id: the voice used
        - text: original text

    Retries are handled for transient API errors.
    """
    if not client:
        logger.error("ElevenLabs client not initialized. Cannot generate audio.")
        return None

    # Default voice if none provided
    if not voice_id:
        voice_id = "pqHfZKP75CvOlQylNhV4"  # Default 'Bill' Deep Male

    attempt = 0
    while attempt <= retries:
        try:
            safe_id = str(uuid.uuid4())
            output_path = os.path.join(tempfile.gettempdir(), f"scene_audio_{safe_id}.mp3")
            logger.info(f"🎙️ Generating audio for voice_id={voice_id}, attempt {attempt + 1}...")

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

            # Write to file
            with open(output_path, "wb") as f:
                for chunk in audio_stream:
                    if chunk:
                        f.write(chunk)

            logger.info(f"✅ Audio generated: {output_path}")
            return {"path": output_path, "voice_id": voice_id, "text": text}

        except Exception as e:
            logger.error(f"🔴 Audio generation failed: {e}")
            attempt += 1
            if attempt <= retries:
                sleep(1)  # brief wait before retry
            else:
                return None

# -----------------------------
# Multi-Scene / Multi-Character Support
# -----------------------------
def generate_audio_for_scenes(scenes: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Accepts a list of scenes:
        [
            {"text": "Scene 1 dialogue", "voice_id": "xyz"},
            {"text": "Scene 2 narration", "voice_id": "abc"},
            ...
        ]
    Returns a list of dicts with audio paths.
    """
    results = []
    for scene in scenes:
        text = scene.get("text")
        voice_id = scene.get("voice_id")
        audio = generate_audio_for_scene(text, voice_id)
        if audio:
            results.append(audio)
        else:
            logger.warning(f"⚠️ Failed to generate audio for scene: {text}")
    return results

# -----------------------------
# Legacy Wrapper (Optional)
# -----------------------------
def generate_voiceover_and_upload(script: str, voice_id: Optional[str] = None) -> str:
    """
    Single script wrapper for legacy compatibility.
    Returns path or empty string.
    """
    audio = generate_audio_for_scene(script, voice_id)
    return audio["path"] if audio else ""

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Example: 4 scenes with different characters
    test_scenes = [
        {"text": "Hello! I am the narrator.", "voice_id": "pqHfZKP75CvOlQylNhV4"},
        {"text": "Ali enters the room and looks around.", "voice_id": "xyz123VoiceID"},
        {"text": "The sun sets over the horizon.", "voice_id": "pqHfZKP75CvOlQylNhV4"},
        {"text": "End scene with dramatic music.", "voice_id": "abc456VoiceID"}
    ]
    audio_files = generate_audio_for_scenes(test_scenes)
    print("Generated audio files:", audio_files)

