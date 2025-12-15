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
    Stitches pre-generated video clips (from HeyGen) and their dialogue audio into a single base video.
    Output is a video file containing only synchronized video and dialogue audio (temp_visual_path).
    """
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"stitch_request_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []

    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for Stitching.")

        # 1. Process chunks (Sync Video length to Audio length)
        for i, (video, audio) in enumerate(scene_pairs):
            if not os.path.exists(video) or not os.path.exists(audio): continue

            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            audio_dur = get_media_duration(audio)
            if audio_dur == 0: continue

            # Command: Use fast preset and CRF 23 for better quality
            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(audio_dur),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac", "-shortest",
                chunk_name
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths: return False

        # 2. Write the Concat List
        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        # 3. Concatenate Chunks into the temporary visual base file
        logging.info("Concatenating chunks into visual base file...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path, "-c", "copy", temp_visual_path
        ], check=True, capture_output=True)

        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg scene stitching failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logging.error(f"General stitching error: {e}")
        return False
    finally:
        # cleanup the whole temp directory for this request
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logging.info(f"Cleaned up temp dir: {temp_dir}")


def composite_green_screen_scene(
    background_image_path: str,
    green_screen_video_path: str,
    output_path: str
) -> bool:
    """
    Composites a Green Screen video (HeyGen) over a Static Background (Flux/DALL-E).
    Adds a subtle 'Zoom In' effect to the background to make it feel cinematic.
    """
    try:
        if not os.path.exists(background_image_path) or not os.path.exists(green_screen_video_path):
            logging.error("Missing input files for composite.")
            return False

        # Get duration of the video to match the background length
        duration = get_media_duration(green_screen_video_path)
        if duration <= 0: duration = 5.0

        # FFmpeg Filter Explanation:
        # 1. [0:v] zoompan: Zooms into the image slowly over 25fps * duration.
        # 2. [1:v] colorkey: Removes the #00FF00 green color.
        # 3. overlay: Puts the actor on top of the zoomed background.
        
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", background_image_path,  # Input 0: Background
            "-i", green_screen_video_path,             # Input 1: Actor
            "-filter_complex",
            f"[0:v]scale=8000:-1,zoompan=z='min(zoom+0.0015,1.5)':d={int(duration*25)+100}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920[bg];"
            "[1:v]colorkey=0x00FF00:0.1:0.1[actor];"
            "[bg][actor]overlay=(W-w)/2:(H-h):shortest=1",
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "copy",
            output_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)

    except subprocess.CalledProcessError as e:
        logging.error(f"Composite failed: {e.stderr.decode() if e.stderr else e}")
        return False
    except Exception as e:
        logging.error(f"General composite error: {e}")
        return False


def mix_music_and_finalize(temp_visual_path: str, music_path: str, final_output_path: str) -> bool:
    """
    Applies Loudness Normalization and mixes the dialogue track with the background music track.
    This function is CRITICAL for the professional sound of The Apex Archive channel.
    """
    if not os.path.exists(music_path):
        # If music path is bad, just copy the base visual file
        shutil.move(temp_visual_path, final_output_path)
        logging.warning("Music mix skipped. Copied base video only.")
        return True

    logging.info("Applying Loudness Normalization and mixing audio...")

    try:
        # CRITICAL FIX: Loudnorm and Amix filter complex for balanced audio
        cmd = [
            "ffmpeg", "-y", 
            "-i", temp_visual_path, 
            "-stream_loop", "-1", "-i", music_path, 
            
            # Filter Complex: Loudnorm dialogue, Loudnorm music, then Amix them.
            "-filter_complex", 
            # [0:a] is dialogue. Apply loudnorm (EBU R 128 standard) and reduce volume slightly.
            "[0:a]loudnorm=I=-14:LRA=7:tp=-2,volume=0.9[dia];" 
            # [1:a] is music. Apply loudnorm to music to a lower background level.
            "[1:a]loudnorm=I=-22:LRA=7:tp=-2[bg];" 
            # Amix: Mix dialogue [dia] and background music [bg]
            "[dia][bg]amix=inputs=2:duration=first:weights=1 1[outa]", 
            
            # Mapping and Encoding:
            "-map", "0:v", "-map", "[outa]",
            "-c:v", "libx264", 
            "-vf", "scale='min(720,iw)':-2", 
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", final_output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg Loudness Mix failed: {e.stderr.decode()}. Copying base video.")
        shutil.move(temp_visual_path, final_output_path)
        return False
    except Exception as e:
        logging.error(f"General mix error: {e}")
        shutil.move(temp_visual_path, final_output_path)
        return False
