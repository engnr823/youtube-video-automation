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
        audio_generator = client.text_to_speech.convert(
            text=script,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

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
    Multi-voice generation (Dialogue Mode).
    Args:
        segments: List of dicts like [{'text': 'Hello', 'voice_id': '...'}, ...]
    Returns:
        Cloudinary URL of the stitched conversation.
    """
    if not client:
        raise ConnectionError("ElevenLabs client is not initialized.")
    
    if not segments:
        logging.warning("No dialogue segments provided.")
        return ""

    logging.info(f"🎙️ Generating Multi-Voice Audio ({len(segments)} segments)...")
    
    temp_dir = tempfile.gettempdir()
    audio_files = []
    unique_run_id = str(uuid.uuid4())

    try:
        # 1. Generate individual audio clips for each segment
        for i, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            voice_id = seg.get("voice_id")
            
            if not text or not voice_id:
                continue
            
            # Create a temp file for this specific sentence
            segment_filename = os.path.join(temp_dir, f"seg_{unique_run_id}_{i}.mp3")
            
            try:
                audio_gen = client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                
                with open(segment_filename, 'wb') as f:
                    for chunk in audio_gen:
                        f.write(chunk)
                
                audio_files.append(segment_filename)
            except Exception as e:
                logging.error(f"Failed to generate segment {i} ({voice_id}): {e}")
                # We continue loop to try saving the rest, or break depending on strictness.
                # Here we continue, but audio might have gaps.
                continue

        if not audio_files:
            raise RuntimeError("No audio segments were successfully generated.")

        # 2. Stitch them together using FFmpeg
        # We create a text file listing all mp3s to concatenate
        concat_list_path = os.path.join(temp_dir, f"concat_list_{unique_run_id}.txt")
        final_output_path = os.path.join(temp_dir, f"final_dialogue_{unique_run_id}.mp3")
        
        with open(concat_list_path, "w") as f:
            for path in audio_files:
                # FFmpeg concat requires single quotes around paths and escaping
                safe_path = path.replace("'", "'\\''") 
                f.write(f"file '{safe_path}'\n")

        logging.info(f"🧵 Stitching {len(audio_files)} clips...")
        
        # Run FFmpeg concat demuxer
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy", # Direct stream copy (very fast, no re-encoding)
            final_output_path
        ]
        
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 3. Upload Stitched Result
        logging.info(f"☁️ Uploading stitched dialogue...")
        upload_result = cloudinary.uploader.upload(
            final_output_path,
            resource_type="video", 
            folder="voiceovers"
        )
        return upload_result.get('secure_url')

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg stitching failed: {e.stderr.decode()}")
        raise
    except Exception as e:
        logging.error(f"Multi-voice generation failed: {e}")
        raise
    finally:
        # 4. Aggressive Cleanup
        # Delete individual clip files
        for f in audio_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        # Delete list file
        if 'concat_list_path' in locals() and os.path.exists(concat_list_path):
            try: os.remove(concat_list_path)
            except: pass
        # Delete final output
        if 'final_output_path' in locals() and os.path.exists(final_output_path):
            try: os.remove(final_output_path)
            except: pass
