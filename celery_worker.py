import os
import sys
import logging
import json
import uuid
import shutil
import subprocess
import requests
import traceback
import re
from pathlib import Path
from datetime import timedelta
from string import Template

# --- SETUP PATHS ---
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR)

import cloudinary
import cloudinary.uploader
import replicate
from openai import OpenAI
from celery_init import celery

# --- CONFIG ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SAAS-ENGINE): %(message)s")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

# ===================================================================
# --- 🧠 ADVANCED YOUTUBE SEO PROMPT V6.0
# ===================================================================

SEO_METADATA_PROMPT = """
# --- SYSTEM PROMPT V6.0 — YOUTUBE SEO MASTER
Analyze the TRANSCRIPT and generate Viral YouTube Metadata.
TRANSCRIPT: ${transcript}
FORMAT: ${video_type}

# --- OUTPUT SCHEMA (STRICT JSON ONLY)
{
  "title": "Viral Title",
  "description": "Engaging description with keywords.",
  "tags": ["tag1", "tag2"],
  "thumbnail_prompt": "Cinematic Flux prompt based on script",
  "primary_keyword": "topic"
}
"""

# -------------------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# -------------------------------------------------------------------------

def transform_drive_url(url):
    patterns = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    file_id = None
    for p in patterns:
        match = re.search(p, url)
        if match:
            file_id = match.group(1)
            break
    return f"https://drive.google.com/uc?export=download&id={file_id}" if file_id else url

def download_file(url, dest_path):
    download_url = transform_drive_url(url) if "drive.google.com" in url else url
    headers = {'User-Agent': 'Mozilla/5.0'}
    with requests.get(download_url, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    return dest_path

def get_video_info(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration", "-of", "json", file_path]
        result = subprocess.check_output(cmd).decode('utf-8')
        info = json.loads(result)['streams'][0]
        return float(info.get('duration', 0)), int(info['width']), int(info['height'])
    except: return 0.0, 1080, 1920

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def ensure_font(temp_dir):
    font_path = os.path.join(temp_dir, "Arial.ttf")
    if not os.path.exists(font_path):
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        r = requests.get(url, timeout=10)
        with open(font_path, 'wb') as f: f.write(r.content)
    return font_path

# -------------------------------------------------------------------------
# ✂️ THE "ABSOLUTE BOTTOM" RENDER ENGINE
# -------------------------------------------------------------------------

def apply_absolute_bottom_polish(input_path, srt_path, font_path, output_path, channel_name, blur_watermarks=True, is_vertical=True):
    """
    V13 FINAL RENDER:
    - Subtitles: Absolute Bottom (MarginV=20).
    - Subtitle Size: 22px (Clean & Small).
    - Subtitle Wrapping: Max 2 lines.
    - Branding: Replaced by Subtitles (or moved to top if needed).
    - Blur: Bottom 7% (Covers old watermark).
    """
    logging.info(f"✨ Rendering Absolute Bottom Subtitles...")
    
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(font_path).replace("\\", "/").replace(":", "\\:")
    safe_font_path = font_path.replace("\\", "/").replace(":", "\\:")
    
    # Subtitle Style: 
    # MarginV=20 (Lowest possible area)
    # Alignment=2 (Centered)
    # Outline=2 (High visibility)
    if is_vertical:
        sub_style = "FontName=Arial,Alignment=2,MarginV=20,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"
        # We move Branding to the TOP so it doesn't fight with subtitles
        brand_y = "40" 
        brand_size = 24
    else:
        sub_style = "FontName=Arial,Alignment=2,MarginV=10,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Bold=1"
        brand_y = "20"
        brand_size = 18

    cmd = ["ffmpeg", "-y", "-i", input_path]
    filter_chain = []
    last_label = "0:v"

    # 1. Blur 7% at bottom (To hide old watermarks)
    if blur_watermarks and is_vertical:
        filter_chain.append(f"[{last_label}]crop=iw:ih*0.07:0:ih*0.93,boxblur=luma_radius=20[bot_blur]")
        filter_chain.append(f"[{last_label}][bot_blur]overlay=0:H-h[v_blurred]")
        last_label = "v_blurred"

    # 2. Add Branding Watermark at the TOP (Safe area)
    safe_brand = channel_name.replace(":", "\\:").replace("'", "")
    filter_chain.append(f"[{last_label}]drawtext=fontfile='{safe_font_path}':text='{safe_brand}':fontcolor=white@0.5:fontsize={brand_size}:x=(w-text_w)/2:y={brand_y}:shadowcolor=black:shadowx=1:shadowy=1[v_branded]")
    last_label = "v_branded"

    # 3. Add Subtitles at the ABSOLUTE BOTTOM
    if os.path.exists(srt_path):
        filter_chain.append(f"[{last_label}]subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'[v_final]")
        last_label = "v_final"

    if filter_chain:
        full_filter = ";".join(filter_chain)
        cmd.extend(["-filter_complex", full_filter, "-map", f"[{last_label}]", "-map", "0:a?", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-c:a", "copy"])
    else:
        cmd.extend(["-c", "copy"])
        
    cmd.append(output_path)
    subprocess.run(cmd, check=True)

# -------------------------------------------------------------------------
# 🏭 MAIN TASK
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    raw_path, final_path = os.path.join(temp_dir, "raw.mp4"), os.path.join(temp_dir, "final.mp4")
    audio_path, srt_path = os.path.join(temp_dir, "audio.mp3"), os.path.join(temp_dir, "subs.srt")
    
    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})
        
        font_path = ensure_font(temp_dir)
        update("Downloading media...")
        download_file(form_data['video_url'], raw_path)
        
        dur, w, h = get_video_info(raw_path)
        target_format = form_data.get('output_format', '9:16')
        user_brand = form_data.get('channel_name', '@ViralShorts')

        # Formatting
        current_video = raw_path
        if (target_format == '9:16') and w > h:
            update("Vertical Resizing...")
            cropped = os.path.join(temp_dir, "cropped.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-vf", "scale=-1:1920,crop=1080:1920,setsar=1", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", cropped], check=True)
            current_video = cropped
        
        update("AI Transcription...")
        subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
        transcript = openai_client.audio.translations.create(model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json")
        
        srt_content, full_text = "", ""
        for i, seg in enumerate(transcript.segments):
            srt_content += f"{i+1}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
            full_text += seg.text + " "
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)

        update("YouTube SEO Engine...")
        seo_prompt = Template(SEO_METADATA_PROMPT).safe_substitute(video_type=target_format, transcript=full_text[:2500])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": seo_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        
        update("AI Thumbnail Artist...")
        flux_out = replicate.run("black-forest-labs/flux-schnell", input={"prompt": meta['thumbnail_prompt'], "aspect_ratio": target_format})
        thumb_url = str(flux_out[0])

        update("Final Polish...")
        apply_absolute_bottom_polish(current_video, srt_path, font_path, final_path, user_brand, True, (target_format == '9:16'))
        
        update("Uploading...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {"status": "success", "video_url": cloud_res.get("secure_url"), "thumbnail_url": thumb_url, "metadata": meta, "transcript_srt": srt_content}

    except Exception as e:
        logging.error(f"Task Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
