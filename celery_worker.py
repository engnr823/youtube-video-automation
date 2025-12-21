# file: celery_worker.py
import os
import sys
import logging
import json
import uuid
import shutil
import subprocess
import requests
import math
import traceback  # <--- FIXED: Added missing import
import re         # <--- Added for Google Drive ID extraction
from pathlib import Path
from datetime import timedelta

# --- SETUP PATHS ---
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR)

import cloudinary
import cloudinary.uploader
import replicate
from openai import OpenAI
from celery_init import celery

# --- CONFIG ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (EDITOR): %(message)s")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

# -------------------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# -------------------------------------------------------------------------

def transform_drive_url(url):
    """Converts a Google Drive 'View' link to a 'Direct Download' link."""
    if "drive.google.com" in url and "/file/d/" in url:
        try:
            file_id = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url).group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        except AttributeError:
            return url
    return url

def download_file(url, dest_path):
    """Downloads video from any public link (Cloudinary, Drive, S3)."""
    # Fix Google Drive Links automatically
    download_url = transform_drive_url(url)
    
    logging.info(f"⬇️ Downloading: {download_url}")
    try:
        with requests.get(download_url, stream=True, timeout=120) as r:
            # Check for 403/401 errors specifically
            if r.status_code == 403 or r.status_code == 401:
                raise RuntimeError("⛔ Access Denied. Please make sure the Google Drive link is set to 'Anyone with the link'.")
            
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")

def get_video_info(file_path):
    """Returns duration, width, height using FFprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration", 
        "-of", "json", file_path
    ]
    result = subprocess.check_output(cmd).decode('utf-8')
    info = json.loads(result)['streams'][0]
    return float(info.get('duration', 0)), int(info['width']), int(info['height'])

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# -------------------------------------------------------------------------
# ✂️ FEATURE 1: SMART CROP (16:9 -> 9:16)
# -------------------------------------------------------------------------
def crop_to_vertical(input_path, output_path):
    """
    Centers the video and crops it to 9:16.
    This effectively REMOVES side watermarks (like Veo) automatically.
    """
    logging.info("📐 Auto-Cropping to Vertical (9:16)...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=-1:1920,crop=1080:1920:((iw-1080)/2):0,setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)

# -------------------------------------------------------------------------
# ✂️ FEATURE 2: SILENCE REMOVAL (JUMP CUTS)
# -------------------------------------------------------------------------
def remove_silence(input_path, output_path, db_threshold=-30, min_silence_duration=0.6):
    logging.info("✂️ Processing Jump Cuts...")
    
    cmd = ["ffmpeg", "-i", input_path, "-af", f"silencedetect=noise={db_threshold}dB:d={min_silence_duration}", "-f", "null", "-"]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    
    silence_starts = []
    silence_ends = []
    for line in result.stderr.splitlines():
        if "silence_start" in line:
            silence_starts.append(float(line.split("silence_start: ")[1]))
        if "silence_end" in line:
            if "silence_end" in line: silence_ends.append(float(line.split("silence_end: ")[1].split(" ")[0]))

    if not silence_starts:
        logging.info("No silence found. Skipping cut.")
        shutil.copy(input_path, output_path)
        return

    duration, _, _ = get_video_info(input_path)
    filter_complex = ""
    concat_idx = 0
    current_time = 0.0
    periods = list(zip(silence_starts, silence_ends))
    
    for start, end in periods:
        if start > current_time:
            filter_complex += f"[0:v]trim=start={current_time}:end={start},setpts=PTS-STARTPTS[v{concat_idx}];"
            filter_complex += f"[0:a]atrim=start={current_time}:end={start},asetpts=PTS-STARTPTS[a{concat_idx}];"
            concat_idx += 1
        current_time = end
        
    if current_time < duration:
        filter_complex += f"[0:v]trim=start={current_time}:end={duration},setpts=PTS-STARTPTS[v{concat_idx}];"
        filter_complex += f"[0:a]atrim=start={current_time}:end={duration},asetpts=PTS-STARTPTS[a{concat_idx}];"
        concat_idx += 1

    filter_complex += "".join([f"[v{i}][a{i}]" for i in range(concat_idx)])
    filter_complex += f"concat=n={concat_idx}:v=1:a=1[outv][outa]"

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", output_path
    ], check=True)

# -------------------------------------------------------------------------
# 📝 FEATURE 3: TRANSCRIPTION & SUBTITLES
# -------------------------------------------------------------------------
def generate_subtitles(audio_path):
    logging.info("🎙️ Transcribing with Whisper...")
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="verbose_json"
        )
    
    srt_content = ""
    full_text = ""
    for i, segment in enumerate(transcript.segments):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_content += f"{i+1}\n{start} --> {end}\n{text}\n\n"
        full_text += text + " "
        
    return srt_content, full_text

# -------------------------------------------------------------------------
# 🖼️ FEATURE 4: FINAL POLISH (Blur + Subs)
# -------------------------------------------------------------------------
def apply_final_polish(input_path, srt_path, output_path, blur_watermarks=True):
    logging.info("✨ Applying Final Polish (Blur + Subs)...")
    
    style = "Alignment=2,MarginV=50,Fontname=Arial,FontSize=24,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,Bold=1"
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    
    filters = []
    if blur_watermarks:
        filters.append("boxblur=luma_radius=20:luma_power=1:enable='between(y,0,150)+between(y,h-200,h)'")
        
    if os.path.exists(srt_path):
        filters.append(f"subtitles='{safe_srt}':force_style='{style}'")
        
    filter_str = ",".join(filters) if filters else "null"
    
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-vf", filter_str,
        "-c:a", "copy", output_path
    ], check=True)

# -------------------------------------------------------------------------
# 🧠 FEATURE 5: METADATA & THUMBNAIL
# -------------------------------------------------------------------------
def generate_packaging(transcript_text, duration):
    logging.info("📦 Generating Viral Packaging...")
    
    prompt = f"""
    Analyze this video transcript: "{transcript_text[:1000]}..."
    Generate:
    1. A Viral YouTube Shorts Title (max 60 chars, clickbait style).
    2. A SEO Description with keywords.
    3. 5 hashtags.
    4. A Flux Image Prompt for a thumbnail that represents the emotion of the video.
    Output JSON.
    """
    res = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role":"user", "content":prompt}], 
        response_format={"type": "json_object"}
    )
    meta = json.loads(res.choices[0].message.content)
    
    logging.info(f"🎨 Generating Thumbnail: {meta.get('thumbnail_prompt')[:30]}...")
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": "Cinematic YouTube Thumbnail, " + meta.get('thumbnail_prompt', 'Viral scene'), "aspect_ratio": "9:16"}
        )
        thumb_url = str(output[0])
    except:
        thumb_url = None
        
    return meta, thumb_url

# -------------------------------------------------------------------------
# 🏭 MAIN TASK
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Paths
    raw_path = os.path.join(temp_dir, "raw.mp4")
    cropped_path = os.path.join(temp_dir, "cropped.mp4") # Vertical conversion
    cut_path = os.path.join(temp_dir, "cut.mp4")         # Silence removed
    final_path = os.path.join(temp_dir, "final.mp4")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    srt_path = os.path.join(temp_dir, "subs.srt")
    
    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})
        
        # 1. Ingest
        update("Downloading Video...")
        # Note: download_file now handles Google Drive conversion internally
        download_file(form_data['video_url'], raw_path)
        
        # Check Source Aspect Ratio
        dur, w, h = get_video_info(raw_path)
        is_landscape = w > h
        
        current_video = raw_path
        
        # 2. Smart Crop (Landscape -> Portrait)
        if is_landscape:
            update("Converting to Vertical (9:16)...")
            crop_to_vertical(raw_path, cropped_path)
            current_video = cropped_path
        
        # 3. Silence Removal
        if form_data.get('remove_silence') == 'true':
            update("Removing Silence...")
            remove_silence(current_video, cut_path)
            current_video = cut_path
            
        # 4. Transcription
        if form_data.get('add_subtitles') == 'true':
            update("Generating Subtitles...")
            subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
            srt_content, transcript_text = generate_subtitles(audio_path)
            with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
        else:
            transcript_text = "Video content analysis."
            
        # 5. Final Assembly (Blur + Burn Subs)
        update("Polishing Video...")
        apply_final_polish(
            current_video, 
            srt_path if form_data.get('add_subtitles') == 'true' else None,
            final_path,
            blur_watermarks=(form_data.get('blur_watermarks') == 'true')
        )
        
        # 6. Packaging
        update("Creating Thumbnail & SEO...")
        meta, thumb_url = generate_packaging(transcript_text, dur)
        
        # 7. Upload
        update("Uploading Final Video...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_content if os.path.exists(srt_path) else None
        }

    except Exception as e:
        error_msg = f"Workflow failed: {str(e)}"
        # FIXED: traceback is now imported, so this won't crash
        logging.error(f"Task Exception: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": error_msg})
        return {"status": "error", "message": error_msg}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
