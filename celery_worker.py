import os
import sys
import json
import uuid
import shutil
import logging
import subprocess
import traceback
import requests
import re
from pathlib import Path
from datetime import timedelta
from string import Template

import cloudinary
import cloudinary.uploader
import replicate
from openai import OpenAI
from celery_init import celery

# ------------------------------------------------------------------
# CONFIG & AI AGENTS
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (SAAS-ENGINE): %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

SEO_PROMPT_TEMPLATE = """
# --- SYSTEM PROMPT V6.0 — YOUTUBE SEO MASTER
Analyze the TRANSCRIPT and generate Viral YouTube Metadata.
YOU MUST BE FACTUALLY ACCURATE TO THE TRANSCRIPT.

TRANSCRIPT: ${transcript}
FORMAT: ${video_type}

# --- OUTPUT REQUIREMENTS (VALID JSON ONLY)
{
  "title": "A high-CTR viral title based on script",
  "description": "Comprehensive description with keywords and story summary.",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "thumbnail_prompt": "A detailed cinematic image description for Flux AI based on the video's climax.",
  "primary_keyword": "main topic"
}
"""

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def transform_drive_url(url):
    patterns = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
    for p in patterns:
        m = re.search(p, url)
        if m: return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url

def download_file(url, path):
    url = transform_drive_url(url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    return path

def get_video_info(path):
    try:
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration", "-of", "json", path])
        s = json.loads(out)["streams"][0]
        return float(s["duration"]), int(s["width"]), int(s["height"])
    except: return 0.0, 1080, 1920

def format_ts(sec):
    td = timedelta(seconds=sec)
    ms = int(td.microseconds / 1000)
    s = int(td.total_seconds())
    return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},{ms:03}"

def ensure_font(tmp):
    font = os.path.join(tmp, "Arial.ttf")
    if not os.path.exists(font):
        r = requests.get("https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf", timeout=10)
        with open(font, "wb") as f: f.write(r.content)
    return font

# ------------------------------------------------------------------
# THE ULTIMATE RENDERER
# ------------------------------------------------------------------
def render_video_final(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width
    
    # MarginV=25 is the absolute bottom safe zone
    # MarginV=65 places subs directly above branding
    if is_vertical:
        play_res = "PlayResX=1080,PlayResY=1920"
        font_size = 14  # Professional small size
        sub_margin = 65 
        brand_y = "h-55" # Precise Golden branding position
    else:
        play_res = "PlayResX=1920,PlayResY=1080"
        font_size = 20
        sub_margin = 35
        brand_y = "h-30"

    # Subtitle Style: Reduced 20%, Absolute Bottom center
    sub_style = (
        f"FontName=Arial,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,"
        f"Alignment=2,MarginV={sub_margin},WrapStyle=2,{play_res}"
    )

    safe_srt = srt_file.replace("\\", "/").replace(":", "\\:")
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(safe_font)
    brand = channel_name.replace(":", "").replace("'", "")

    # DRAWTEXT FILTERS: 
    # 1. Shadow Layer for Glow
    # 2. Gold Text Layer
    # 3. Subtitles Layer
    filters = [
        f"drawtext=fontfile='{safe_font}':text='{brand}':fontcolor='#FFD700':fontsize={font_size+6}:"
        f"x=(w-text_w)/2:y={brand_y}:shadowcolor='#FFD700@0.4':shadowx=0:shadowy=0:box=1:boxcolor=black@0.2:boxborderw=5",
        
        f"subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'"
    ]

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "19", "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)

# ------------------------------------------------------------------
# MAIN CELERY TASK
# ------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data):
    task_id = str(uuid.uuid4())
    tmp = f"/tmp/job_{task_id}"
    os.makedirs(tmp, exist_ok=True)

    raw, srt, final = os.path.join(tmp, "raw.mp4"), os.path.join(tmp, "subs.srt"), os.path.join(tmp, "final.mp4")
    audio = os.path.join(tmp, "audio.mp3")

    try:
        def update(m): self.update_state(state="PROGRESS", meta={"message": m})

        update("Downloading & Analyzing")
        download_file(form_data["video_url"], raw)
        duration, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        update("AI Transcription")
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-map", "a", "-q:a", "0", audio], check=True)
        transcript = openai_client.audio.translations.create(model="whisper-1", file=open(audio, "rb"), response_format="verbose_json")

        srt_text, full_text = "", ""
        for i, seg in enumerate(transcript.segments):
            text = seg.text.strip()
            # Professional 2-line wrapping
            if len(text) > 42:
                words = text.split()
                mid = len(words) // 2
                text = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
            srt_text += f"{i+1}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{text}\n\n"
            full_text += seg.text + " "
        with open(srt, "w", encoding="utf-8") as f: f.write(srt_text)

        update("SEO & Thumbnail Architect")
        seo_prompt = Template(SEO_PROMPT_TEMPLATE).safe_substitute(video_type="9:16", transcript=full_text[:3000])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": seo_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        
        # Flux Schnell Thumbnail
        flux_out = replicate.run("black-forest-labs/flux-schnell", input={"prompt": meta['thumbnail_prompt'], "aspect_ratio": "9:16"})
        thumb_url = str(flux_out[0])

        update("Glowing Final Render")
        render_video_final(raw, srt, font, final, channel, w, h)

        update("Final Upload")
        cloud = cloudinary.uploader.upload(final, resource_type="video", folder="viral_edits")

        return {
            "status": "success",
            "video_url": cloud["secure_url"],
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_text
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
