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

# ===================================================================
# --- 🧠 SYSTEM PROMPTS V4.0
# ===================================================================

SEO_METADATA_PROMPT = """
# --- SYSTEM PROMPT V4.0 — SENIOR YOUTUBE SEO STRATEGIST
You are a Senior YouTube Growth Strategist.
Goal: Maximize CTR and SEO.

# --- INPUT CONTEXT
VIDEO TYPE: ${video_type}
SCRIPT CONTEXT: ${transcript}

# --- OPTIMIZATION RULES
1. TITLE: [Primary Keyword] + [Power Word]. If "reel", add #Shorts. Max 60 chars.
2. DESCRIPTION: Hook (2 sentences) + Deep Dive (100 words) + Tags.
3. TAGS: Broad + Exact + Long-Tail.

# --- OUTPUT SCHEMA (JSON)
{
  "title": "Optimized Title #Shorts",
  "description": "Full description...",
  "tags": ["tag1", "tag2"],
  "primary_keyword": "keyword"
}
"""

THUMBNAIL_ARTIST_PROMPT = """
# --- SYSTEM PROMPT V4.0 — THUMBNAIL ARTIST
Write ONE detailed text-to-image prompt for Flux AI based on: ${context}
Focus on: High Emotion (Fear/Joy), Cinematic Lighting (Rembrandt), 8k texture.
Negative: No text, no blurry, no cartoon.
Output ONLY the prompt text.
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
    if file_id:
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def download_file(url, dest_path):
    if "drive.google.com" in url:
        download_url = transform_drive_url(url)
    else:
        download_url = url
    
    logging.info(f"⬇️ Downloading: {download_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(download_url, stream=True, timeout=300, headers=headers) as r:
            if r.status_code in [401, 403]:
                raise RuntimeError("⛔ Access Denied. Public links only.")
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")

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
        logging.info("📥 Downloading Portable Font (Arial)...")
        url = "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf"
        try:
            r = requests.get(url, timeout=10)
            with open(font_path, 'wb') as f:
                f.write(r.content)
        except:
            logging.warning("Font download failed.")
    return font_path

# -------------------------------------------------------------------------
# ✂️ VIDEO PROCESSING
# -------------------------------------------------------------------------

def crop_to_vertical_force(input_path, output_path):
    logging.info("📐 Force Cropping to Vertical (9:16)...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=-1:1920,crop=1080:1920:((iw-1080)/2):0,setsar=1",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)

def remove_silence_optimized(input_path, output_path):
    shutil.copy(input_path, output_path)

def generate_subtitles_english(audio_path):
    logging.info("🎙️ Transcribing & Translating to English...")
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.translations.create(
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

def apply_final_polish_v6(input_path, srt_path, font_path, output_path, channel_name, blur_watermarks=True, is_vertical=True):
    logging.info(f"✨ Applying Final Polish (Branding: {channel_name})...")
    
    # 1. Prepare Paths
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(font_path).replace("\\", "/").replace(":", "\\:")
    safe_font_path = font_path.replace("\\", "/").replace(":", "\\:")
    
    # 2. Escape Branding text
    safe_channel = channel_name.replace(":", "\\:").replace("'", "")

    # 3. Subtitle Style (Lower Position for 7% blur area)
    # MarginV=80 puts it right above the very bottom
    # FontSize=65 is standard large for 1080p
    if is_vertical:
        style = "FontName=Arial,Alignment=2,MarginV=80,FontSize=65,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Bold=1"
        brand_size = 35
        brand_y = "h-th-15" # 15px from the absolute bottom
    else:
        style = "FontName=Arial,Alignment=2,MarginV=40,FontSize=40,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"
        brand_size = 20
        brand_y = "h-th-10"

    cmd = ["ffmpeg", "-y", "-i", input_path]
    filter_chain = []
    last_label = "0:v"

    # 4. Blur Watermarks (Only 7% at bottom as requested)
    if blur_watermarks and is_vertical:
        filter_chain.append(f"[{last_label}]crop=iw:ih*0.07:0:ih*0.93,boxblur=luma_radius=20[bot_blur]")
        filter_chain.append(f"[{last_label}][bot_blur]overlay=0:H-h[v_blurred]")
        last_label = "v_blurred"

    # 5. Add Branding Layer (Channel Name)
    filter_chain.append(f"[{last_label}]drawtext=fontfile='{safe_font_path}':text='{safe_channel}':fontcolor=white@0.8:fontsize={brand_size}:x=(w-text_w)/2:y={brand_y}:shadowcolor=black:shadowx=1:shadowy=1[v_branded]")
    last_label = "v_branded"

    # 6. Burn Subtitles
    if os.path.exists(srt_path):
        filter_chain.append(f"[{last_label}]subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{style}'[v_final]")
        last_label = "v_final"

    if filter_chain:
        full_filter = ";".join(filter_chain)
        cmd.extend(["-filter_complex", full_filter, "-map", f"[{last_label}]", "-map", "0:a?", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "copy"])
    else:
        cmd.extend(["-c", "copy"])
        
    cmd.append(output_path)
    subprocess.run(cmd, check=True)

def generate_packaging_v4(transcript_text, duration, output_format="9:16"):
    logging.info("📦 Generating V4.0 Packaging...")
    is_short = (output_format == "9:16")
    video_type_str = "reel" if is_short else "youtube"
    try:
        formatted_prompt = Template(SEO_METADATA_PROMPT).safe_substitute(video_type=video_type_str, transcript=transcript_text[:2000])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": formatted_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        context_data = meta.get('description', transcript_text[:500])
        thumb_gen_prompt = Template(THUMBNAIL_ARTIST_PROMPT).safe_substitute(context=context_data)
        output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": thumb_gen_prompt + ", photorealistic, 8k", "aspect_ratio": "9:16" if is_short else "16:9"})
        return meta, str(output[0])
    except Exception:
        return {"title": "Viral Edit #Shorts", "description": "", "tags": []}, None

# -------------------------------------------------------------------------
# 🏭 MAIN TASK
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    raw_path = os.path.join(temp_dir, "raw.mp4")
    cropped_path = os.path.join(temp_dir, "cropped.mp4")
    processed_path = os.path.join(temp_dir, "processed.mp4")
    final_path = os.path.join(temp_dir, "final.mp4")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    srt_path = os.path.join(temp_dir, "subs.srt")
    
    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})
        
        font_path = ensure_font(temp_dir)
        update("Downloading Media...")
        download_file(form_data['video_url'], raw_path)
        dur, w, h = get_video_info(raw_path)
        
        target_format = form_data.get('output_format', '9:16')
        is_landscape = w > h
        is_vertical_output = (target_format == '9:16')
        
        # CAPTURE DYNAMIC CHANNEL NAME FROM FORM
        user_channel = form_data.get('channel_name', '@ViralShorts')
        
        current_video = raw_path

        # FORCE CROP
        if target_format == '9:16' and is_landscape:
            update("Formatting to Vertical...")
            crop_to_vertical_force(raw_path, cropped_path)
            current_video = cropped_path
            is_landscape = False 

        remove_silence_optimized(current_video, processed_path)
        current_video = processed_path
        
        transcript_text = "Video Content"
        srt_exists = False
        if form_data.get('add_subtitles') == 'true':
            update("AI Captioning...")
            subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
            srt_content, transcript_text = generate_subtitles_english(audio_path)
            if len(srt_content) > 10:
                with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
                srt_exists = True
            
        # APPLY POLISH (V6)
        update("Applying Polish...")
        apply_final_polish_v6(
            current_video, 
            srt_path if srt_exists else None,
            font_path,
            final_path,
            channel_name=user_channel,
            blur_watermarks=(form_data.get('blur_watermarks') == 'true'),
            is_vertical=is_vertical_output
        )
        
        meta, thumb_url = generate_packaging_v4(transcript_text, dur, target_format)
        update("Finalizing...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_content if srt_exists else None
        }

    except Exception as e:
        logging.error(f"Task Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
