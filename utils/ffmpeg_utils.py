import os
import logging
import subprocess
import uuid
from typing import List

def stitch_video_with_audio(scene_paths: List[str], voiceover_path: str, output_path: str) -> bool:
    """
    Stitches multiple video clips together, adds a voiceover, and saves the result.

    This function uses a robust ffmpeg command that re-encodes video streams
    to ensure compatibility between clips from different sources.

    Args:
        scene_paths (List[str]): A list of local file paths for the video scenes.
        voiceover_path (str): The local file path for the voiceover audio.
        output_path (str): The local file path to save the final assembled video.

    Returns:
        bool: True if the stitching was successful, False otherwise.
    """
    logging.info("Invoking FFMPEG utility to stitch video...")

    if not scene_paths:
        logging.error("🔴 FFMPEG Error: No scene paths provided.")
        return False
    if not voiceover_path or not os.path.exists(voiceover_path):
        logging.error(f"🔴 FFMPEG Error: Voiceover path '{voiceover_path}' does not exist.")
        return False

    # Create a temporary file list for ffmpeg's concat demuxer
    concat_file_path = f"/tmp/concat_{uuid.uuid4()}.txt"

    try:
        with open(concat_file_path, "w") as f:
            for path in scene_paths:
                if os.path.exists(path):
                    # Sanitize file path for ffmpeg
                    f.write(f"file '{os.path.abspath(path)}'\n")
                else:
                    logging.warning(f"⚠️ Warning: Scene path not found and skipped: {path}")
        
        # Build the robust ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",                   # Overwrite output file if it exists
            "-f", "concat",         # Use the concat demuxer
            "-safe", "0",           # Disable safety checks for file paths
            "-i", concat_file_path, # Input file list
            "-i", voiceover_path,   # Second input: the voiceover audio
            "-c:v", "libx264",      # Re-encode video to the standard H.264 codec
            "-pix_fmt", "yuv420p",  # Pixel format for maximum compatibility
            "-preset", "fast",      # A good balance between encoding speed and quality
            "-c:a", "aac",          # Encode audio to the standard AAC codec
            "-shortest",            # Finish encoding when the shortest input stream ends (the video)
            output_path             # The final output file path
        ]
        
        logging.info(f"Executing FFMPEG command: {' '.join(cmd)}")
        
        # Execute the command
        subprocess.run(
            cmd,
            check=True,             # Raise an exception if ffmpeg returns a non-zero exit code
            capture_output=True,    # Capture stdout and stderr
            text=True
        )
        
        logging.info(f"✅ FFMPEG stitching complete. Output saved to: {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        # Log the detailed error message from ffmpeg if it fails
        logging.error(f"🔴 FFMPEG Error during stitching: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"🔴 An unexpected error occurred in the FFMPEG utility: {e}")
        return False
    finally:
        # Clean up the temporary concat file list
        if os.path.exists(concat_file_path):
            os.remove(concat_file_path)
            logging.info(f"Cleaned up temporary concat file: {concat_file_path}")
