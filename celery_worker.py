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
import math
from datetime import timedelta

import cloudinary
import cloudinary.uploader
from openai import OpenAI
from celery_init import celery

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (SAAS-ENGINE): %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Use the new client format
openai_client = OpenAI(api_key=OPENAI_API_KEY)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# -------------------------------------------------
# HELPERS & EXPERT SEO
# -------------------------------------------------

def generate_expert_seo(transcript_text):
    """
    Generates Viral YouTube SEO Metadata based on the transcript.
    """
    try:
        system_prompt = (
            "You are a YouTube SEO Expert and Viral Content Strategist. "
            "Generate metadata for this video transcript. "
            "Return ONLY a raw JSON object (no markdown) with keys: 'title', 'description', 'tags'."
        )
        
        user_prompt = (
            f"TRANSCRIPT: {transcript_text[:4000]}...\n\n"
            "REQUIREMENTS:\n"
            "1. Title: High CTR, Clickbait style, under 60 chars.\n"
            "2. Description: 3 sentences using keywords, plus hashtags.\n"
            "3. Tags: Comma separated high volume keywords."
        )

        response = openai_client.chat.completions.create(
            model="gpt-4o", # Or gpt-3.5-turbo
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"SEO Generation failed: {e}")
        return {
            "title": "Viral Video",
            "description": "Watch this amazing video!",
            "tags": "viral, shorts, video"
        }

def transform_drive_url(url):
    patterns = [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url

def download_file(url, path):
    url = transform_drive_url(url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
    return path

def get_video_info(path):
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", path
        ])
        s = json.loads(out)["streams"][0]
        return float(s.get("duration", 0)), int(s["width"]), int(s["height"])
    except Exception:
        return 0.0, 1920, 1080

def format_ts(sec):
    td = timedelta(seconds=sec)
    s = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},{ms:03}"

def ensure_font(tmp_dir):
    fonts_dir = os.path.join(tmp_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    # Using Arial Bold for better visibility
    font_path = os.path.join(fonts_dir, "Arial.ttf")
    if not os.path.exists(font_path):
        # Fallback to a reliable font source
        r = requests.get(
            "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf",
            timeout=10
        )
        with open(font_path, "wb") as f:
            f.write(r.content)
    return font_path

# -------------------------------------------------
# THUMBNAIL GENERATOR (CANVA STYLE)
# -------------------------------------------------
def generate_thumbnail(video_path, font_path, output_path, title, duration, width, height):
    """
    Extracts a frame from the middle and burns the Title on it.
    """
    try:
        timestamp = duration / 2
        safe_font = font_path.replace("\\", "/").replace(":", "\\:")
        
        # Calculate font size based on resolution
        font_size = int(width / 10) 
        
        # Filter to scale image if needed and draw text
        # Draw a semi-transparent black box behind text for readability
        vf = (
            f"drawtext=fontfile='{safe_font}':"
            f"text='{title.upper()}':"
            f"fontcolor=white:"
            f"fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:" # Center
            f"box=1:boxcolor=black@0.6:boxborderw=10"
        )

        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vf", vf,
            "-frames:v", "1",
            output_path
        ], check=True)
        return True
    except Exception as e:
        logging.error(f"Thumbnail generation error: {e}")
        return False

# -------------------------------------------------
# RENDER ENGINE
# -------------------------------------------------
def render_video(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width

    # --- 1. CONFIGURATION ---
    if is_vertical:
        # Shorts / Reels
        play_res = "PlayResX=1080,PlayResY=1920"
        # Subtitles (Bottom)
        sub_font_size = 22 # Larger for mobile
        sub_margin_v = 150 # Lifted slightly from very bottom
        
        # Branding (Top Right)
        brand_font_size = 40
        brand_x = "w-text_w-40" # 40px padding from right
        brand_y = "120"         # 120px padding from top (avoids UI)
    else:
        # Long Video (YouTube Landscape)
        play_res = "PlayResX=1920,PlayResY=1080"
        # Subtitles
        sub_font_size = 28
        sub_margin_v = 80
        
        # Branding
        brand_font_size = 50
        brand_x = "w-text_w-50"
        brand_y = "50"

    # --- 2. SUBTITLE STYLING (High Visibility) ---
    # Outline=3 (Thick black border), PrimaryColour is White
    sub_style = (
        f"FontName=Arial,"
        f"FontSize={sub_font_size},"
        f"PrimaryColour=&H00FFFFFF," 
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline=3," 
        f"Shadow=1,"
        f"Alignment=2," # 2 = Bottom Center
        f"MarginV={sub_margin_v},"
        f"WrapStyle=2,"
        f"{play_res}"
    )

    safe_srt = srt_file.replace("\\", "/").replace(":", "\\:")
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    font_dir = safe_font.rsplit("/", 1)[0]
    
    # Remove special chars from brand
    brand = channel_name.replace(":", "").replace("'", "").replace("\"", "")

    # --- 3. FILTER GRAPH ---
    # 1. Branding: Golden Color (#FFD700), Top Right, Pulsing Alpha
    # alpha='0.7+0.3*sin(2*PI*t/3)' -> Pulses opacity between 0.4 and 1.0 every 3 seconds
    branding_filter = (
        f"drawtext=fontfile='{safe_font}':"
        f"text='{brand}':"
        f"fontcolor=#FFD700:" # GOLD
        f"fontsize={brand_font_size}:"
        f"x={brand_x}:"
        f"y={brand_y}:"
        f"shadowcolor=black@0.6:"
        f"shadowx=3:shadowy=3:"
        f"alpha='0.7+0.3*sin(2*PI*t/3)'" # ANIMATION
    )

    # 2. Subtitles: Using strict styling
    subtitle_filter = (
        f"subtitles='{safe_srt}':"
        f"fontsdir='{font_dir}':"
        f"force_style='{sub_style}'"
    )

    # Combine filters
    filters = f"{branding_filter},{subtitle_filter}"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", filters,
        "-map", "0:v",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "superfast", # Faster rendering
        "-crf", "23",
        "-c:a", "aac", # Re-encode audio to ensure compatibility
        output_video
    ]

    subprocess.run(cmd, check=True)

# -------------------------------------------------
# CELERY TASK
# -------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data):
    task_id = str(uuid.uuid4())
    tmp = f"/tmp/job_{task_id}"
    os.makedirs(tmp, exist_ok=True)

    raw = os.path.join(tmp, "raw.mp4")
    audio = os.path.join(tmp, "audio.mp3")
    srt = os.path.join(tmp, "subs.srt")
    final = os.path.join(tmp, "final.mp4")
    thumb = os.path.join(tmp, "thumbnail.jpg")

    try:
        def update(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})

        update("Downloading video")
        download_file(form_data["video_url"], raw)

        duration, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        update("Extracting audio")
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw, "-map", "a", "-q:a", "0", audio],
            check=True
        )

        update("AI Transcribing & SEO")
        # 1. Transcribe
        transcript = openai_client.audio.translations.create(
            model="whisper-1",
            file=open(audio, "rb"),
            response_format="verbose_json"
        )
        
        # 2. Generate Expert SEO Data
        full_text = transcript.text
        seo_data = generate_expert_seo(full_text)

        # 3. Create SRT
        srt_text = ""
        for i, seg in enumerate(transcript.segments):
            srt_text += (
                f"{i+1}\n"
                f"{format_ts(seg.start)} --> {format_ts(seg.end)}\n"
                f"{seg.text.strip()}\n\n"
            )

        with open(srt, "w", encoding="utf-8") as f:
            f.write(srt_text)

        update("Rendering Viral Video")
        render_video(raw, srt, font, final, channel, w, h)

        update("Generating Expert Thumbnail")
        has_thumb = generate_thumbnail(final, font, thumb, seo_data['title'], duration, w, h)

        update("Uploading to Cloud")
        # Upload Video
        vid_upload = cloudinary.uploader.upload(
            final,
            resource_type="video",
            folder="viral_edits"
        )
        
        # Upload Thumbnail if generated
        thumb_url = None
        if has_thumb:
            thumb_upload = cloudinary.uploader.upload(
                thumb,
                resource_type="image",
                folder="viral_thumbnails"
            )
            thumb_url = thumb_upload["secure_url"]

        return {
            "status": "success",
            "video_url": vid_upload["secure_url"],
            "thumbnail_url": thumb_url,
            "transcript_srt": srt_text,
            "seo_metadata": seo_data
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
