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
# --- 🧠 ADVANCED YOUTUBE SEO PROMPT V5.0
# ===================================================================

SEO_METADATA_PROMPT = """
# --- SYSTEM PROMPT V5.0 — SENIOR YOUTUBE SEO GROWTH ARCHITECT
Task: Generate High-Performance Metadata for a Viral Short/Video.
Strategy: Use Hook-Rehook-Action framework. Target high-volume, low-competition semantic keywords.

# --- INPUT CONTEXT
TRANSCRIPT: ${transcript}
FORMAT: ${video_type}

# --- OUTPUT SCHEMA (STRICT JSON ONLY)
{
  "title": "SHOCKING: [Primary Keyword] + [Emotional Trigger] #Shorts",
  "description": "👉 [HOOK: 1st sentence]. \\n\\n[DEEP DIVE: 150 words using LSI keywords]. \\n\\n🚀 [CALL TO ACTION]. \\n\\n#Tags #SEO",
  "tags": ["primary keyword", "long tail keyword", "viral topic", "trending"],
  "thumbnail_prompt": "Cinematic, photorealistic Flux prompt, high contrast, emotional face, 8k",
  "primary_keyword": "keyword"
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
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest_path

def get_video_info(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration", "-of", "json", file_path]
        result = subprocess.check_output(cmd).decode('utf-8')
        info = json.loads(result)['streams'][0]
        return float(info.get('duration', 0)), int(info['width']), int(info['height'])
    except:
        return 0.0, 1080, 1920

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
# ✂️ THE GOLDEN RENDER ENGINE
# -------------------------------------------------------------------------

def apply_bulletproof_polish(input_path, srt_path, font_path, output_path, channel_name, blur_watermarks=True, is_vertical=True):
    """
    ULTIMATE RENDERER:
    - 7% Blur at the very bottom.
    - Branding handle at the very bottom edge.
    - Subtitles REDUCED 40% (Size 24-26) and moved to the BOTTOM 10% zone.
    """
    logging.info(f"🚀 SaaS Rendering: {channel_name}")
    
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(font_path).replace("\\", "/").replace(":", "\\:")
    safe_font_path = font_path.replace("\\", "/").replace(":", "\\:")
    safe_brand = channel_name.replace(":", "\\:").replace("'", "")

    # COORDINATE SYSTEM FIX:
    # On 1920px height:
    # Bottom 7% (Blur) = 0 to 134px from bottom.
    # Subtitles must be ABOVE branding.
    # MarginV is pixels from BOTTOM.
    if is_vertical:
        # FontSize=26 (Reduced 40% from 45/50)
        # MarginV=160 (Moves text down but keeps it clear of the branding at the very bottom)
        sub_style = "FontName=Arial,Alignment=2,MarginV=160,FontSize=26,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"
        brand_size = 32
        brand_y = "h-th-40" 
    else:
        sub_style = "FontName=Arial,Alignment=2,MarginV=40,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Bold=1"
        brand_size = 20
        brand_y = "h-th-20"

    cmd = ["ffmpeg", "-y", "-i", input_path]
    filter_chain = []
    last_label = "0:v"

    # 1. 7% Blur (Precise)
    if blur_watermarks and is_vertical:
        filter_chain.append(f"[{last_label}]crop=iw:ih*0.07:0:ih*0.93,boxblur=luma_radius=20[bot_blur]")
        filter_chain.append(f"[{last_label}][bot_blur]overlay=0:H-h[v_blurred]")
        last_label = "v_blurred"

    # 2. Branding (Dynamic Channel Name)
    filter_chain.append(f"[{last_label}]drawtext=fontfile='{safe_font_path}':text='{safe_brand}':fontcolor=white@0.8:fontsize={brand_size}:x=(w-text_w)/2:y={brand_y}:shadowcolor=black:shadowx=1:shadowy=1[v_branded]")
    last_label = "v_branded"

    # 3. Burn Subtitles (V8 logic with fontsdir)
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
        update("Downloading media...")
        download_file(form_data['video_url'], raw_path)
        
        dur, w, h = get_video_info(raw_path)
        target_format = form_data.get('output_format', '9:16')
        is_vertical = (target_format == '9:16')
        user_brand = form_data.get('channel_name', '@ViralShorts')

        # 1. Vertical Auto-Format
        current_video = raw_path
        if is_vertical and w > h:
            update("Smart Cropping...")
            cropped = os.path.join(temp_dir, "cropped.mp4")
            subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-vf", "scale=-1:1920,crop=1080:1920,setsar=1", "-c:v", "libx264", "-crf", "18", "-c:a", "copy", cropped], check=True)
            current_video = cropped
        
        # 2. Transcription
        update("AI Transcription...")
        subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
        transcript = openai_client.audio.translations.create(model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json")
        
        srt_content = ""
        full_text = ""
        for i, seg in enumerate(transcript.segments):
            srt_content += f"{i+1}\n{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}\n{seg.text.strip()}\n\n"
            full_text += seg.text + " "
        with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)

        # 3. ADVANCED SEO & THUMBNAIL (GENERATE FIRST TO INCLUDE IN RETURN)
        update("Advanced SEO Engine...")
        seo_prompt = Template(SEO_METADATA_PROMPT).safe_substitute(video_type=target_format, transcript=full_text[:2500])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": seo_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        
        # Thumbnail via Flux
        flux_out = replicate.run("black-forest-labs/flux-schnell", input={"prompt": meta['thumbnail_prompt'], "aspect_ratio": target_format})
        thumb_url = str(flux_out[0])

        # 4. FINAL RENDER
        update("Applying Polish...")
        apply_bulletproof_polish(current_video, srt_path, font_path, final_path, user_brand, True, is_vertical)
        
        # 5. UPLOAD
        update("Finalizing...")
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
