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
# --- 🧠 ENHANCED YOUTUBE SEO ARCHITECT (FOR RELEVANCY)
# ===================================================================

SEO_METADATA_PROMPT = """
# --- SYSTEM PROMPT V5.5 — YOUTUBE SEO ARCHITECT
Your task is to analyze the following TRANSCRIPT and generate highly relevant YouTube SEO metadata.
YOU MUST BE FACTUALLY ACCURATE TO THE TRANSCRIPT.

TRANSCRIPT: ${transcript}
FORMAT: ${video_type}

# --- OUTPUT REQUIREMENTS (JSON ONLY)
{
  "title": "A punchy viral title based on script",
  "description": "Comprehensive description including the story in the script.",
  "tags": ["relevant_tag_1", "relevant_tag_2"],
  "thumbnail_prompt": "A detailed cinematic image description of the MAIN PERSON or ACTION in the script for Flux AI.",
  "primary_keyword": "main topic"
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
# ✂️ THE V12 RENDER ENGINE
# -------------------------------------------------------------------------

def apply_v12_polish(input_path, srt_path, font_path, output_path, channel_name, blur_watermarks=True, is_vertical=True):
    """
    V12 FINAL RENDER:
    - Blur: 7% Bottom
    - Subtitle size: 20 (Exactly as requested)
    - Subtitle Position: MarginV=45 (Perfect bottom placement)
    - Branding: MarginV=10 (Very bottom edge)
    """
    logging.info(f"✨ V12 Render Initiation (Branding: {channel_name})...")
    
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(font_path).replace("\\", "/").replace(":", "\\:")
    safe_font_path = font_path.replace("\\", "/").replace(":", "\\:")
    safe_brand = channel_name.replace(":", "\\:").replace("'", "")

    # Layout Strategy:
    # We move subtitles to MarginV=45 (about 45 pixels from bottom)
    # We move Branding to MarginV=10 (10 pixels from bottom)
    if is_vertical:
        sub_style = "FontName=Arial,Alignment=2,MarginV=45,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"
        brand_size = 22
        brand_y = "h-th-10" 
    else:
        sub_style = "FontName=Arial,Alignment=2,MarginV=25,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Bold=1"
        brand_size = 18
        brand_y = "h-th-5"

    cmd = ["ffmpeg", "-y", "-i", input_path]
    filter_chain = []
    last_label = "0:v"

    # 1. Precise 7% Blur
    if blur_watermarks and is_vertical:
        filter_chain.append(f"[{last_label}]crop=iw:ih*0.07:0:ih*0.93,boxblur=luma_radius=20[bot_blur]")
        filter_chain.append(f"[{last_label}][bot_blur]overlay=0:H-h[v_blurred]")
        last_label = "v_blurred"

    # 2. Add Branding Watermark
    filter_chain.append(f"[{last_label}]drawtext=fontfile='{safe_font_path}':text='{safe_brand}':fontcolor=white:fontsize={brand_size}:x=(w-text_w)/2:y={brand_y}:shadowcolor=black:shadowx=1:shadowy=1[v_branded]")
    last_label = "v_branded"

    # 3. Add Subtitles
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
    
    raw_path = os.path.join(temp_dir, "raw.mp4")
    final_path = os.path.join(temp_dir, "final.mp4")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    srt_path = os.path.join(temp_dir, "subs.srt")
    
    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})
        
        font_path = ensure_font(temp_dir)
        update("Downloading video...")
        download_file(form_data['video_url'], raw_path)
        
        dur, w, h = get_video_info(raw_path)
        target_format = form_data.get('output_format', '9:16')
        user_brand = form_data.get('channel_name', '@ViralShorts')

        update("AI Transcription...")
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-q:a", "0", "-map", "a", audio_path], check=True)
        transcript = openai_client.audio.translations.create(model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json")
        
        srt_content, full_text = "", ""
        for i, seg in enumerate(transcript.segments):
            srt_content += f"{i+1}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
            full_text += seg.text + " "
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)

        # GENERATE RELEVANT SEO & THUMBNAIL
        update("Ranking SEO Metadata...")
        seo_prompt = Template(SEO_METADATA_PROMPT).safe_substitute(video_type=target_format, transcript=full_text[:3000])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": seo_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        
        update("Generating AI Thumbnail...")
        flux_out = replicate.run("black-forest-labs/flux-schnell", input={"prompt": meta['thumbnail_prompt'], "aspect_ratio": target_format})
        thumb_url = str(flux_out[0])

        # FINAL RENDER
        update("Final Polish...")
        apply_v12_polish(raw_path, srt_path, font_path, final_path, user_brand, True, (target_format == '9:16'))
        
        update("Uploading...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_content
        }

    except Exception as e:
        logging.error(f"Task Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
