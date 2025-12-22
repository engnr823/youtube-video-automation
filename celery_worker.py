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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# -------------------------------------------------
# HELPER: PROMPT LOADER
# -------------------------------------------------
def load_prompt_from_file(filepath, default_text):
    """
    Attempts to read the custom prompt file. 
    If file is missing, returns the default expert prompt.
    """
    try:
        # Assuming the script runs from root, adjust path if needed
        full_path = os.path.join(os.getcwd(), filepath)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                logging.info(f"Loaded custom prompt from: {filepath}")
                return f.read()
    except Exception as e:
        logging.warning(f"Could not load prompt {filepath}: {e}")
    
    return default_text

# -------------------------------------------------
# HELPER: FFMPEG SANITIZER (CRITICAL FIX)
# -------------------------------------------------
def sanitize_text_for_ffmpeg(text):
    """
    Escapes special characters that crash FFmpeg (Colons, quotes, backslashes).
    """
    if not text: return ""
    # 1. Replace backslash with double backslash
    # 2. Escape colons (CRITICAL FIX for 'Exit 234')
    # 3. Escape single quotes
    clean = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\''")
    return clean

# -------------------------------------------------
# EXPERT AI ENGINE
# -------------------------------------------------
def generate_expert_metadata(transcript_text):
    """Generates SEO using your custom repo prompt."""
    
    default_prompt = (
        "You are a YouTube Algorithm Expert. "
        "Generate a JSON with 'title' (CLICKBAIT, under 60 chars), "
        "'description' (SEO optimized), and 'tags'."
    )
    
    # LOAD FROM YOUR SPECIFIC FILE
    system_instruction = load_prompt_from_file(
        "prompts/prompt_youtube_metadata_generator.txt", 
        default_prompt
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Transcript: {transcript_text[:4000]}"}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        # Fallback if AI fails
        return {"title": "Viral Video", "description": "", "tags": ""}

def generate_thumbnail_text(transcript_text, meta_title):
    """Generates short punchy text for the thumbnail image using your image prompt."""
    
    default_img_prompt = (
        "Extract a 3-5 word shocking phrase from this transcript "
        "suitable for a YouTube Thumbnail overlay."
    )
    
    # LOAD FROM YOUR SPECIFIC FILE
    system_instruction = load_prompt_from_file(
        "prompts/prompt_image_synthesizer.txt", 
        default_img_prompt
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Title: {meta_title}\nTranscript: {transcript_text[:1000]}"}
            ]
        )
        # Clean up quotes for the visual overlay
        return response.choices[0].message.content.strip().replace('"', '').upper()
    except:
        return meta_title.upper()

# -------------------------------------------------
# FILE HANDLING
# -------------------------------------------------
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
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration", "-of", "json", path
        ])
        s = json.loads(out)["streams"][0]
        return float(s.get("duration", 0)), int(s["width"]), int(s["height"])
    except:
        return 0.0, 1920, 1080

def format_ts(sec):
    td = timedelta(seconds=sec)
    s = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},{ms:03}"

def ensure_font(tmp_dir):
    fonts_dir = os.path.join(tmp_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    font_path = os.path.join(fonts_dir, "Arial.ttf")
    if not os.path.exists(font_path):
        r = requests.get("https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf", timeout=10)
        with open(font_path, "wb") as f:
            f.write(r.content)
    return font_path

# -------------------------------------------------
# THUMBNAIL GENERATOR (FIXED)
# -------------------------------------------------
def generate_thumbnail(video_path, font_path, output_path, overlay_text, duration, width):
    """
    Creates a Canva-style thumbnail.
    FIXED: Uses sanitize_text_for_ffmpeg to prevent crashing on colons/quotes.
    """
    try:
        timestamp = duration / 2
        safe_font = sanitize_text_for_ffmpeg(font_path)
        
        # SANITIZE THE TITLE TEXT TO PREVENT CRASH
        safe_text = sanitize_text_for_ffmpeg(overlay_text)
        
        # Dynamic font size (bigger for visibility)
        font_size = int(width / 8) 
        
        # Filters:
        # 1. Grab frame
        # 2. Draw Text with Background Box (easier to read)
        vf = (
            f"drawtext=fontfile='{safe_font}':"
            f"text='{safe_text}':"
            f"fontcolor=white:"
            f"fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"box=1:boxcolor=black@0.7:boxborderw=20" # Thick background box
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vf", vf,
            "-frames:v", "1",
            "-q:v", "2", # High quality jpg
            output_path
        ]
        
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Thumbnail Gen Failed: {e}")
        return False

# -------------------------------------------------
# RENDER ENGINE (IMPROVED BRANDING & SUBS)
# -------------------------------------------------
def render_video(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width

    # --- SETTINGS ---
    if is_vertical:
        # Shorts settings
        play_res = "PlayResX=1080,PlayResY=1920"
        sub_font_size = 26      # INCREASED SIZE
        sub_margin_v = 150      # Position from bottom
        brand_size = 45
        # Tight top-right corner
        brand_x = "w-text_w-20" 
        brand_y = "40"
    else:
        # Landscape settings
        play_res = "PlayResX=1920,PlayResY=1080"
        sub_font_size = 30      # INCREASED SIZE
        sub_margin_v = 80
        brand_size = 55
        brand_x = "w-text_w-30"
        brand_y = "30"

    # --- SUBTITLE STYLE (High Visibility) ---
    # Outline=3 (Thick), BackColour (Semi-transparent background for subs)
    sub_style = (
        f"FontName=Arial,"
        f"FontSize={sub_font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BackColour=&H80000000," # Semi-transparent black background
        f"BorderStyle=3,"         # 3 = Opaque Box (Best for readability)
        f"Outline=3,"
        f"Shadow=0,"
        f"Alignment=2,"
        f"MarginV={sub_margin_v},"
        f"WrapStyle=2,"
        f"{play_res}"
    )

    # Sanitize paths
    safe_srt = sanitize_text_for_ffmpeg(srt_file)
    safe_font = sanitize_text_for_ffmpeg(font_path)
    font_dir = os.path.dirname(safe_font)
    
    # Sanitize Brand Name
    brand = sanitize_text_for_ffmpeg(channel_name)

    # --- FILTER GRAPH ---
    # 1. Branding: Top Right, Pulsing Gold
    branding_filter = (
        f"drawtext=fontfile='{safe_font}':"
        f"text='{brand}':"
        f"fontcolor=#FFD700:" # Gold
        f"fontsize={brand_size}:"
        f"x={brand_x}:" # Pushed to Right
        f"y={brand_y}:" # Pushed to Top
        f"alpha='0.6+0.4*sin(2*PI*t/2)'" # Animation
    )

    # 2. Subtitles
    subtitle_filter = (
        f"subtitles='{safe_srt}':"
        f"fontsdir='{font_dir}':"
        f"force_style='{sub_style}'"
    )

    filters = f"{branding_filter},{subtitle_filter}"

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", filters,
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "23",
        "-c:a", "aac",
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

        # 1. Download
        update("Downloading video")
        download_file(form_data["video_url"], raw)
        duration, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        # 2. Audio Extraction
        update("Extracting audio")
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-map", "a", "-q:a", "0", audio], check=True)

        # 3. Transcription & SEO
        update("AI Processing (SEO & Scripts)")
        transcript_obj = openai_client.audio.translations.create(
            model="whisper-1", file=open(audio, "rb"), response_format="verbose_json"
        )
        full_text = transcript_obj.text
        
        # GENERATE METADATA (Using your prompt file)
        seo_data = generate_expert_metadata(full_text)
        
        # GENERATE THUMBNAIL TEXT (Using your prompt file)
        thumb_text = generate_thumbnail_text(full_text, seo_data.get('title', 'Viral Video'))

        # Create SRT
        srt_text = ""
        for i, seg in enumerate(transcript_obj.segments):
            srt_text += f"{i+1}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{seg.text.strip()}\n\n"
        with open(srt, "w", encoding="utf-8") as f: f.write(srt_text)

        # 4. Rendering
        update("Rendering Final Video")
        render_video(raw, srt, font, final, channel, w, h)

        # 5. Thumbnail Generation
        update("Generating Thumbnail")
        has_thumb = generate_thumbnail(final, font, thumb, thumb_text, duration, w)

        # 6. Upload
        update("Uploading")
        vid_res = cloudinary.uploader.upload(final, resource_type="video", folder="viral_edits")
        
        thumb_url = None
        if has_thumb:
            thumb_res = cloudinary.uploader.upload(thumb, resource_type="image", folder="viral_thumbnails")
            thumb_url = thumb_res["secure_url"]

        return {
            "status": "success",
            "video_url": vid_res["secure_url"],
            "thumbnail_url": thumb_url,
            "metadata": seo_data
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
