import os
import logging
import uuid
import tempfile
import subprocess
import cloudinary
import cloudinary.uploader
from pathlib import Path
from typing import List, Dict
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
    Single-voice generation (Standard Mode).
    """
    if not client:
        raise ConnectionError("ElevenLabs client is not initialized.")

    if not script:
        logging.warning("Script is empty. Returning empty string.")
        return ""

    logging.info(f"🎙️ Generating single voiceover for: {voice_id}...")
    
    temp_dir = tempfile.gettempdir()
    temp_filename = os.path.join(temp_dir, f"voiceover_{uuid.uuid4()}.mp3")

    try:
        # Generate audio stream
        audio_generator = client.text_to_speech.convert(
            text=script,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        # Write stream to file
        with open(temp_filename, 'wb') as f:
            for chunk in audio_generator:
                f.write(chunk)

        logging.info(f"☁️ Uploading to Cloudinary...")
        upload_result = cloudinary.uploader.upload(
            temp_filename,
            resource_type="video", 
            folder="voiceovers"
        )
        return upload_result.get('secure_url')

    except Exception as e:
        logging.error(f"🔴 Voiceover Generation Failed: {e}")
        raise
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def generate_multi_voice_audio(segments: List[Dict[str, str]]) -> str:
    """
    Generates multiple audio clips and stitches them into one file.
    Compatible with the SaaS 'refine_script_with_roles' worker function.
    """
    if not client: raise ConnectionError("ElevenLabs client is not initialized.")
    if not segments: return ""

    logging.info(f"🎙️ Generating Multi-Voice Audio ({len(segments)} segments)...")
    temp_dir = tempfile.gettempdir()
    audio_files = []
    unique_run_id = str(uuid.uuid4())

    # Valid Fallback Voice (Adam) - Use this if a custom ID fails
    FALLBACK_VOICE_ID = "pNInz6obpgDQGcFmaJgB" 

    try:
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            voice_id = seg.get("voice_id", "").strip()
            
            if not text: continue
            
            # Use fallback if ID is missing
            if not voice_id: 
                voice_id = FALLBACK_VOICE_ID

            segment_filename = os.path.join(temp_dir, f"seg_{unique_run_id}_{i}.mp3")
            
            try:
                # Try generation with requested voice
                audio_gen = client.text_to_speech.convert(
                    text=text, voice_id=voice_id, model_id="eleven_multilingual_v2", output_format="mp3_44100_128"
                )
                with open(segment_filename, 'wb') as f:
                    for chunk in audio_gen: f.write(chunk)
                audio_files.append(segment_filename)
                
            except Exception as e:
                logging.warning(f"⚠️ Failed segment {i} with voice {voice_id}: {e}")
                # RETRY with Fallback Voice
                try:
                    logging.info(f"🔄 Retrying segment {i} with Fallback Voice (Adam)...")
                    audio_gen = client.text_to_speech.convert(
                        text=text, voice_id=FALLBACK_VOICE_ID, model_id="eleven_multilingual_v2", output_format="mp3_44100_128"
                    )
                    with open(segment_filename, 'wb') as f:
                        for chunk in audio_gen: f.write(chunk)
                    audio_files.append(segment_filename)
                except Exception as final_e:
                    logging.error(f"❌ Segment {i} failed completely: {final_e}")

        if not audio_files: 
            raise RuntimeError("No audio segments generated. Check API Key and Voice IDs.")

        # --- STITCHING LOGIC (Requires FFmpeg) ---
        concat_list_path = os.path.join(temp_dir, f"concat_list_{unique_run_id}.txt")
        final_output_path = os.path.join(temp_dir, f"final_dialogue_{unique_run_id}.mp3")
        
        # Create FFmpeg concat list
        with open(concat_list_path, "w") as f:
            for path in audio_files:
                # Escape single quotes for FFmpeg list file
                safe_path = path.replace("'", "'\\''") 
                f.write(f"file '{safe_path}'\n")

        # Run FFmpeg to merge MP3s
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path, "-c", "copy", final_output_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Upload final merged audio
        upload_result = cloudinary.uploader.upload(final_output_path, resource_type="video", folder="voiceovers")
        return upload_result.get('secure_url')

    except Exception as e:
        logging.error(f"Multi-voice process failed: {e}")
        raise
    finally:
        # Cleanup temp files
        for f in audio_files: 
            if os.path.exists(f): os.remove(f)
        if 'concat_list_path' in locals() and os.path.exists(concat_list_path): os.remove(concat_list_path)
        if 'final_output_path' in locals() and os.path.exists(final_output_path): os.remove(final_output_path)

# --- CORRECTED PLACEMENT: Indentation fixed here ---
def generate_audio_for_scene(text: str, voice_id: str) -> str:
    """Generates audio for a single scene and returns the local file path."""
    if not client: raise ConnectionError("ElevenLabs client not initialized")
    
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f"scene_audio_{uuid.uuid4()}.mp3")
    
    try:
        audio_gen = client.text_to_speech.convert(
            text=text, voice_id=voice_id, model_id="eleven_multilingual_v2"
        )
        with open(filename, 'wb') as f:
            for chunk in audio_gen: f.write(chunk)
        return filename
    except Exception as e:
        logging.error(f"Audio gen failed: {e}")
        return None
