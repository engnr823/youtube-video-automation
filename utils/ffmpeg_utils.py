# file: utils/ffmpeg_utils.py

import os
import logging
import subprocess
import uuid
import shutil
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_media_duration(file_path: str) -> float:
    """Returns the duration of a media file in seconds using ffprobe."""
    try:
        if not os.path.exists(file_path): return 0.0
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        val = result.stdout.strip()
        return float(val) if val else 0.0
    except Exception as e:
        logging.error(f"Error getting duration for {file_path}: {e}")
        return 0.0

def process_and_stitch_scenes(scene_pairs: List[Tuple[str, str]], temp_visual_path: str) -> bool:
    """
    Stitches pre-generated video clips.
    CRITICAL FIX: Forces Audio to Stereo (2 Channels) @ 44.1kHz to prevent silence.
    """
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"stitch_request_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []

    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for Stitching.")

        for i, (video, audio) in enumerate(scene_pairs):
            if not os.path.exists(video): continue

            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            
            # Determine duration
            audio_dur = get_media_duration(audio) if audio and os.path.exists(audio) else 0.0
            video_dur = get_media_duration(video)
            # Use audio duration if valid, otherwise video duration
            final_dur = audio_dur if audio_dur > 0.5 else video_dur
            if final_dur <= 0: final_dur = 5.0 # Fallback

            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video
            ]
            
            # Add Audio Input
            if audio and os.path.exists(audio):
                cmd.extend(["-i", audio, "-map", "0:v", "-map", "1:a"])
            else:
                # Generate Silent Audio (Stereo/44.1kHz)
                cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-map", "0:v", "-map", "1:a"])

            cmd.extend([
                "-t", str(final_dur),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
                # CRITICAL AUDIO FIX: Force AAC Stereo 44.1kHz
                "-c:a", "aac", "-ac", "2", "-ar", "44100",
                "-shortest",
                chunk_name
            ])
            
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths: return False

        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        logging.info("Concatenating chunks...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path, "-c", "copy", temp_visual_path
        ], check=True, capture_output=True)

        return True

    except Exception as e:
        logging.error(f"Stitching failed: {e}")
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def composite_green_screen_scene(
    background_image_path: str,
    green_screen_video_path: str,
    output_path: str
) -> bool:
    """
    Composites a Green Screen video over a Background.
    FIX: Uses 'aac' audio codec instead of 'copy' to ensure compatibility.
    """
    try:
        if not os.path.exists(background_image_path) or not os.path.exists(green_screen_video_path):
            logging.error("Missing input files for composite.")
            return False

        duration = get_media_duration(green_screen_video_path)
        if duration <= 0: duration = 5.0

        # Scale Logic: Fits the 'Talking Photo' or 'Avatar' nicely
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", background_image_path,  # Input 0: BG
            "-i", green_screen_video_path,             # Input 1: Actor
            "-filter_complex",
            # Background: Slight zoom effect for 'Cinema' feel
            f"[0:v]scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*25)+100}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[bg];"
            # Actor: Scale to width 1080 (full width fit) or 400 (corner) depending on preference.
            # Current setting: 1080 width (Full presence) for Cinematic feel
            "[1:v]scale=1080:-1[actor];"
            "[bg][actor]overlay=(W-w)/2:H-h:shortest=1",
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", # Re-encode audio to safe format
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)

    except Exception as e:
        logging.error(f"Composite failed: {e}")
        return False

def mix_music_and_finalize(temp_visual_path: str, music_path: str, final_output_path: str) -> bool:
    """
    Mixes dialogue and background music.
    CRITICAL FIX: Replaced 'loudnorm' with simple Volume filter to prevent silence on short clips.
    """
    if not os.path.exists(music_path):
        shutil.move(temp_visual_path, final_output_path)
        return True

    logging.info("Mixing background music...")

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_visual_path,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex",
            # FIX: Simple volume mix. Voice=1.0 (100%), Music=0.15 (15%)
            "[0:a]volume=1.0[v];[1:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first[outa]",
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "copy", # Copy video stream (fast)
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", final_output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True

    except Exception as e:
        logging.error(f"Music mix failed: {e}")
        # Fallback: Return video without music if mixing crashes
        shutil.move(temp_visual_path, final_output_path)
        return False
