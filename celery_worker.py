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
openai_client = OpenAI(api_key=OPENAI_API_KEY)

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def load_prompt(filepath, default):
    try:
        path = os.path.join(os.getcwd(), filepath)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except:
        pass
    return default

def sanitize(text):
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
# AI ENGINE
# -------------------------------------------------
def generate_metadata(transcript):
    sys_prompt = load_prompt("prompts/prompt_youtube_metadata_generator.txt", "Generate viral JSON title/desc/tags.")
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Transcript: {transcript[:4000]}"}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"title": "Viral Video", "description": "", "tags": ""}

def generate_thumbnail_image(transcript, title, tmp_dir):
    """Generates an AI Image using DALL-E 3 instead of a screenshot."""
    img_prompt_sys = load_prompt("prompts/prompt_image_synthesizer.txt", "Describe a high quality youtube thumbnail background image without text.")
    
    # 1. Get Visual Description
    try:
        desc_res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": img_prompt_sys},
                {"role": "user", "content": f"Title: {title}\nTranscript: {transcript[:1000]}"}
            ]
        )
        visual_prompt = desc_res.choices[0].message.content.strip()[:1000] # Limit char count
    except:
        visual_prompt = f"High quality cinematic photo for video: {title}"

    # 2. Generate Image
    try:
        img_res = openai_client.images.generate(
            model="dall-e-3",
            prompt=visual_prompt,
            size="1024x1792", # Vertical aspect ratio approx
            quality="standard",
            n=1
        )
        image_url = img_res.data[0].url
        
        # Download
        local_path = os.path.join(tmp_dir, "ai_thumb_bg.png")
        with requests.get(image_url) as r:
            with open(local_path, "wb") as f:
                f.write(r.content)
        return local_path
    except Exception as e:
        logging.error(f"DALL-E Failed: {e}")
        return None

def generate_hook_text(transcript, title):
    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Generate a 3-5 word shocking visual hook text for a thumbnail. Output ONLY text."},
                {"role": "user", "content": f"Title: {title}\nTranscript: {transcript[:500]}"}
            ]
        )
        return res.choices[0].message.content.strip().upper().replace('"', '')
    except:
        return "WATCH THIS"

# -------------------------------------------------
# SRT FORMATTER (HORMOZI SPLIT)
# -------------------------------------------------
def create_hormozi_srt(segments, output_path):
    """
    Splits long segments into shorter chunks (max 4 words) to ensure
    subtitles stay at the bottom and don't take up too much width.
    """
    srt_content = ""
    index = 1
    
    for seg in segments:
        words = seg.text.strip().split()
        start = seg.start
        end = seg.end
        duration = end - start
        
        if not words: continue

        # Determine how many chunks
        chunk_size = 4 # Max words per line
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        chunk_duration = duration / len(chunks)
        
        current_start = start
        for chunk in chunks:
            current_end = current_start + chunk_duration
            text_line = " ".join(chunk)
            
            srt_content += f"{index}\n"
            srt_content += f"{format_ts(current_start)} --> {format_ts(current_end)}\n"
            srt_content += f"{text_line}\n\n"
            
            current_start = current_end
            index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

# -------------------------------------------------
# RENDER ENGINE
# -------------------------------------------------
def render_video(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width

    if is_vertical:
        # Shorts
        play_res = "PlayResX=1080,PlayResY=1920"
        sub_font_size = 85  # Readable size
        sub_margin_v = 150  # Lowered (was 350, too high). 150 is bottom-safe area.
        brand_size = 30     # Fixed as requested
        brand_x = "w-text_w-30"
        brand_y = "50"
    else:
        # Landscape
        play_res = "PlayResX=1920,PlayResY=1080"
        sub_font_size = 60
        sub_margin_v = 80
        brand_size = 30
        brand_x = "w-text_w-30"
        brand_y = "30"

    # STYLE: Opaque Box (3) for readability
    sub_style = (
        f"FontName=Arial,FontSize={sub_font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BackColour=&H60000000,BorderStyle=3,"
        f"Outline=1,Shadow=0,Alignment=2,MarginV={sub_margin_v},"
        f"WrapStyle=2,{play_res}"
    )

    safe_srt = sanitize(srt_file)
    safe_font = sanitize(font_path)
    safe_brand = sanitize(channel_name)
    font_dir = os.path.dirname(safe_font)

    # BRANDING: Top Right
    brand_filter = (
        f"drawtext=fontfile='{safe_font}':text='{safe_brand}':"
        f"fontcolor=#FFD700:fontsize={brand_size}:"
        f"x={brand_x}:y={brand_y}:"
        f"shadowcolor=black@0.8:shadowx=2:shadowy=2:"
        f"alpha='0.8+0.2*sin(2*PI*t/2)'"
    )

    # SUBTITLES
    sub_filter = f"subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"{brand_filter},{sub_filter}",
        "-c:v", "libx264", "-preset", "superfast", "-crf", "23", "-c:a", "aac",
        output_video
    ]
    subprocess.run(cmd, check=True)

def create_final_thumbnail(bg_image_path, font_path, output_path, text, width):
    """Overlays text on the generated AI image."""
    try:
        safe_font = sanitize(font_path)
        safe_text = sanitize(text)
        font_size = int(width / 6) # Big Text
        
        vf = (
            f"drawtext=fontfile='{safe_font}':text='{safe_text}':"
            f"fontcolor=white:fontsize={font_size}:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:"
            f"box=1:boxcolor=black@0.6:boxborderw=30"
        )
        
        # Scale AI image to video width/height first if needed, but here we assume close match or crop
        # We'll just run filter on the image
        cmd = [
            "ffmpeg", "-y", "-i", bg_image_path,
            "-vf", vf,
            "-frames:v", "1", "-update", "1", # Fix for single image output
            output_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        logging.error(f"Thumb overlay error: {e}")
        return False

# -------------------------------------------------
# TASK
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
    thumb_path = os.path.join(tmp, "thumbnail.jpg")

    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})

        update("Downloading")
        download_file(form_data["video_url"], raw)
        dur, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        update("Processing Audio")
        subprocess.run(["ffmpeg", "-y", "-i", raw, "-map", "a", "-q:a", "0", audio], check=True)
        
        transcript_obj = openai_client.audio.translations.create(
            model="whisper-1", file=open(audio, "rb"), response_format="verbose_json"
        )
        full_text = transcript_obj.text

        update("Generating Assets")
        seo = generate_metadata(full_text)
        hook_text = generate_hook_text(full_text, seo.get("title", ""))
        
        # Generate SRT with splitting for "Hormozi" look
        create_hormozi_srt(transcript_obj.segments, srt)

        update("Rendering Video")
        render_video(raw, srt, font, final, channel, w, h)

        update("Creating AI Thumbnail")
        # 1. Generate AI Background
        ai_bg = generate_thumbnail_image(full_text, seo.get("title",""), tmp)
        has_thumb = False
        if ai_bg:
            # 2. Overlay Text
            has_thumb = create_final_thumbnail(ai_bg, font, thumb_path, hook_text, w)
        else:
            # Fallback to screenshot if AI fails
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(dur/2), "-i", final,
                    "-vf", f"drawtext=fontfile='{sanitize(font)}':text='{sanitize(hook_text)}':fontcolor=white:fontsize={int(w/7)}:x=(w-text_w)/2:y=(h-text_h)/2:box=1:boxcolor=black@0.6:boxborderw=20",
                    "-frames:v", "1", "-update", "1", thumb_path
                ], check=True)
                has_thumb = True
            except: pass

        update("Uploading")
        vid_res = cloudinary.uploader.upload(final, resource_type="video", folder="viral_edits")
        
        thumb_url = None
        if has_thumb:
            thumb_res = cloudinary.uploader.upload(thumb_path, resource_type="image", folder="viral_thumbnails")
            thumb_url = thumb_res["secure_url"]

        return {
            "status": "success",
            "video_url": vid_res["secure_url"],
            "thumbnail_url": thumb_url,
            "metadata": seo,
            "transcript_text": full_text
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
``` Your video is ready!
