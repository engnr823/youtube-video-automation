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
# CONFIGURATION
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
# HELPERS
# -------------------------------------------------
def load_prompt_from_file(filepath, default_text):
    try:
        full_path = os.path.join(os.getcwd(), filepath)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return default_text

def sanitize_text_for_ffmpeg(text):
    if not text: return ""
    return text.replace("'", "").replace('"', '').replace(":", "\\:").replace(",", "\\,")

def transform_drive_url(url):
    patterns = [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]
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
        with open(font_path, "wb") as f: f.write(r.content)
    return font_path

# -------------------------------------------------
# AI GENERATORS
# -------------------------------------------------
def generate_expert_metadata(transcript_text):
    default_prompt = "Generate JSON with 'title', 'description', 'tags'."
    system_instruction = load_prompt_from_file("prompts/prompt_youtube_metadata_generator.txt", default_prompt)
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
    except:
        return {"title": "Viral Video", "description": "", "tags": ""}

def generate_thumbnail_hook(transcript_text, title):
    system_prompt = (
        "You are a Viral Content Expert. "
        "Create a 3 to 5 word 'Visual Hook' text for a YouTube Thumbnail. "
        "Return ONLY the text. No quotes."
    )
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\nTranscript: {transcript_text[:1000]}"}
            ]
        )
        return response.choices[0].message.content.strip().upper()
    except:
        return title.upper()[:20]

# -------------------------------------------------
# RENDER ENGINE
# -------------------------------------------------
def render_video(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width

    if is_vertical:
        play_res = "PlayResX=1080,PlayResY=1920"
        # MASSIVE INCREASE for visibility on mobile
        sub_font_size = 80 
        sub_margin_v = 350 # Higher up
        
        # Fixed Branding
        brand_size = 30
        brand_x = "w-text_w-20"
        brand_y = "50"
    else:
        play_res = "PlayResX=1920,PlayResY=1080"
        sub_font_size = 60
        sub_margin_v = 100
        
        brand_size = 30
        brand_x = "w-text_w-30"
        brand_y = "30"

    # 
    sub_style = (
        f"FontName=Arial,"
        f"FontSize={sub_font_size},"
        f"PrimaryColour=&H00FFFFFF,"   # White
        f"OutlineColour=&H00000000,"   # Black Outline
        f"BackColour=&H60000000,"      # Black Box (60% Alpha)
        f"BorderStyle=3,"              # Opaque Box Mode
        f"Outline=1,"
        f"Shadow=0,"
        f"Alignment=2,"
        f"MarginV={sub_margin_v},"
        f"WrapStyle=2,"
        f"{play_res}"
    )

    safe_srt = sanitize_text_for_ffmpeg(srt_file)
    safe_font = sanitize_text_for_ffmpeg(font_path)
    font_dir = os.path.dirname(safe_font)
    safe_brand = sanitize_text_for_ffmpeg(channel_name)

    branding_filter = (
        f"drawtext=fontfile='{safe_font}':"
        f"text='{safe_brand}':"
        f"fontcolor=#FFD700:" 
        f"fontsize={brand_size}:"
        f"x={brand_x}:y={brand_y}:"
        f"alpha='0.8+0.2*sin(2*PI*t/2)'"
    )

    subtitle_filter = (
        f"subtitles='{safe_srt}':"
        f"fontsdir='{font_dir}':"
        f"force_style='{sub_style}'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", f"{branding_filter},{subtitle_filter}",
        "-c:v", "libx264",
        "-preset", "superfast",
        "-crf", "23",
        "-c:a", "aac",
        output_video
    ]
    subprocess.run(cmd, check=True)

def generate_thumbnail(video_path, font_path, output_path, text_overlay, duration, width):
    try:
        timestamp = duration / 2
        safe_font = sanitize_text_for_ffmpeg(font_path)
        safe_text = sanitize_text_for_ffmpeg(text_overlay)
        
        # Make font massive (1/6th of width)
        font_size = int(width / 6)
        
        # Box with wider padding (boxborderw=40) for "Design" look
        vf = (
            f"drawtext=fontfile='{safe_font}':"
            f"text='{safe_text}':"
            f"fontcolor=white:"
            f"fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"box=1:boxcolor=black@0.7:boxborderw=40"
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(timestamp),
            "-i", video_path,
            "-vf", vf,
            "-frames:v", "1",
            "-update", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except:
        return False

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
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})

        update("Downloading")
        download_file(form_data["video_url"], raw)
        duration, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        update("AI Analysis")
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-map", "a", "-q:a", "0", audio], check=True)
        
        transcript = openai_client.audio.translations.create(
            model="whisper-1", file=open(audio, "rb"), response_format="verbose_json"
        )
        full_text = transcript.text

        seo_data = generate_expert_metadata(full_text)
        thumb_text = generate_thumbnail_hook(full_text, seo_data.get("title", ""))

        srt_content = ""
        for i, seg in enumerate(transcript.segments):
            srt_content += f"{i+1}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{seg.text.strip()}\n\n"
        with open(srt, "w", encoding="utf-8") as f: f.write(srt_content)

        update("Rendering")
        render_video(raw, srt, font, final, channel, w, h)

        update("Thumbnail")
        has_thumb = generate_thumbnail(final, font, thumb, thumb_text, duration, w)

        update("Uploading")
        vid_res = cloudinary.uploader.upload(final, resource_type="video", folder="viral_edits")
        
        thumb_url = None
        if has_thumb:
            thumb_res = cloudinary.uploader.upload(thumb, resource_type="image", folder="viral_thumbnails")
            thumb_url = thumb_res["secure_url"]

        # EXPLICIT RETURN OF TRANSCRIPT TEXT
        return {
            "status": "success",
            "video_url": vid_res["secure_url"],
            "thumbnail_url": thumb_url,
            "metadata": seo_data,
            "transcript_text": full_text 
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
