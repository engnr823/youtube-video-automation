import os
import logging
import subprocess
import uuid
import shutil
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_media_duration(file_path):
    """Returns the duration of a media file in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def stitch_video_audio_pairs(scene_pairs: List[Tuple[str, str]], output_path: str) -> bool:
    """
    Stitches (video, audio) pairs. 
    - Ensures Video matches Audio length (loops/cuts video).
    - Uses UUIDs to prevent file conflicts in Flask.
    """
    
    # Generate a unique ID for this specific request to prevent collision
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"render_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []

    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for Request ID: {request_id}")

        # 1. Process chunks (Sync Video length to Audio length)
        for i, (video, audio) in enumerate(scene_pairs):
            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            
            # Get duration to force video match
            audio_dur = get_media_duration(audio)
            if audio_dur == 0:
                logging.warning(f"Audio duration is 0 for {audio}, skipping chunk.")
                continue

            # Command: Loop video (-stream_loop -1) + Cut at audio length (-t)
            # Added: -pix_fmt yuv420p for better compatibility with players
            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(audio_dur), 
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                "-c:a", "aac", "-shortest",
                chunk_name
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        # 2. Write the Concat List
        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                # Use absolute path and escape single quotes for safety
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        # 3. Concatenate Chunks
        logging.info("Concatenating chunks...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", input_list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)

        logging.info(f"✅ Video successfully saved to {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        return False
    except Exception as e:
        logging.error(f"General stitching error: {e}")
        return False
    finally:
        # cleanup the whole temp directory for this request
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp dir: {temp_dir}")
