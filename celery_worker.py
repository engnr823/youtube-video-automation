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
import replicate
from openai import OpenAI
from celery_init import celery

# [NEW] Google API Imports for YouTube Upload
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (SAAS-ENGINE): %(message)s"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Ensure REPLICATE_API_TOKEN is in your environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# -------------------------------------------------
# UTILITIES
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
    """Escapes special characters for FFmpeg."""
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

def generate_thumbnail_image(transcript, title, tmp_dir, width, height):
    """Generates an AI Image using Replicate (Flux-Schnell)."""
    img_prompt_sys = load_prompt("prompts/prompt_image_synthesizer.txt", "Describe a high quality youtube thumbnail background image.")
    
    # 1. Get Visual Prompt
    try:
        desc_res = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": img_prompt_sys},
                {"role": "user", "content": f"Title: {title}\nTranscript: {transcript[:800]}"}
            ]
        )
        visual_prompt = desc_res.choices[0].message.content.strip()[:900]
    except:
        visual_prompt = f"Cinematic hyper-realistic photo representing {title}, high emotional impact, 8k resolution."

    aspect_ratio = "9:16" if height > width else "16:9"

    # 2. Generate Image (Replicate)
    try:
        logging.info(f"🎨 Generating Flux Image ({aspect_ratio}): {visual_prompt[:50]}...")
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": visual_prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": "png",
                "disable_safety_checker": True
            }
        )
        image_url = output[0]
        
        local_path = os.path.join(tmp_dir, "ai_thumb_bg.png")
        with requests.get(str(image_url)) as r:
            with open(local_path, "wb") as f:
                f.write(r.content)
        return local_path
    except Exception as e:
        logging.error(f"⚠️ Flux Generation Failed: {e}")
        return None

# -------------------------------------------------
# SRT FORMATTER (5 WORDS, NO SPLIT)
# -------------------------------------------------
def create_5word_srt(segments, output_path):
    """
    Formats subtitles to appear in chunks of exactly 5 words (or less at end).
    Does NOT split inside the chunk, keeping it as one line.
    """
    srt_content = ""
    index = 1
    
    for seg in segments:
        words = seg.text.strip().split()
        start = seg.start
        end = seg.end
        duration = end - start
        
        if not words: continue

        # EXACTLY 5 WORDS per chunk
        chunk_size = 5
        chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
        chunk_duration = duration / len(chunks)
        
        current_start = start
        for chunk in chunks:
            current_end = current_start + chunk_duration
            
            # Join words with space (One line)
            text_block = " ".join(chunk)
            
            srt_content += f"{index}\n"
            srt_content += f"{format_ts(current_start)} --> {format_ts(current_end)}\n"
            srt_content += f"{text_block}\n\n"
            
            current_start = current_end
            index += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

def generate_formatted_transcript(segments):
    """Generates a readable transcript with timestamps."""
    formatted_text = ""
    for seg in segments:
        start_ts = format_ts(seg.start).split(',')[0] # Remove milliseconds
        end_ts = format_ts(seg.end).split(',')[0]
        formatted_text += f"[{start_ts} - {end_ts}] {seg.text.strip()}\n"
    return formatted_text

# -------------------------------------------------
# RENDER ENGINE
# -------------------------------------------------
def render_video(input_video, srt_file, font_path, output_video, channel_name, width, height):
    is_vertical = height > width

    if is_vertical:
        play_res = "PlayResX=1080,PlayResY=1920"
        sub_font_size = 70  # Readable size
        sub_margin_v = 140  # Bottom Safe
        brand_size = 30
        brand_x = "w-text_w-30"
        brand_y = "50"
    else:
        play_res = "PlayResX=1920,PlayResY=1080"
        sub_font_size = 55
        sub_margin_v = 80
        brand_size = 30
        brand_x = "w-text_w-30"
        brand_y = "30"

    # STYLE: Opaque Grey Box with Black Outline
    sub_style = (
        f"FontName=Arial,FontSize={sub_font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"  # Black Outline
        f"BackColour=&H00808080,"      # Opaque Grey Box
        f"BorderStyle=3,"             # Box Mode
        f"Outline=2,"                 # Thicker Outline
        f"Shadow=0,Alignment=2,MarginV={sub_margin_v},"
        f"WrapStyle=2,{play_res}"
    )

    safe_srt = sanitize(srt_file)
    safe_font = sanitize(font_path)
    safe_brand = sanitize(channel_name)
    font_dir = os.path.dirname(safe_font)

    brand_filter = (
        f"drawtext=fontfile='{safe_font}':text='{safe_brand}':"
        f"fontcolor=#FFD700:fontsize={brand_size}:"
        f"x={brand_x}:y={brand_y}:"
        f"shadowcolor=black@0.8:shadowx=2:shadowy=2:"
        f"alpha='0.8+0.2*sin(2*PI*t/2)'"
    )

    sub_filter = f"subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"{brand_filter},{sub_filter}",
        "-c:v", "libx264", "-preset", "superfast", "-crf", "23", "-c:a", "aac",
        output_video
    ]
    subprocess.run(cmd, check=True)

# -------------------------------------------------
# [UPDATED] YOUTUBE UPLOAD FUNCTION
# -------------------------------------------------
def upload_video_to_youtube(creds_dict, video_path, metadata, transcript_text=None):
    try:
        logging.info("🚀 Initiating YouTube Upload...")
        credentials = Credentials(**creds_dict)
        youtube = build('youtube', 'v3', credentials=credentials)

        title = metadata.get("title", "AI Generated Video")[:100] # YouTube limit
        description = metadata.get("description", "Uploaded via AI Viral Editor")
        
        # [NEW] Append Transcript to Description if available
        if transcript_text:
            description += "\n\n" + ("="*20) + "\nTRANSCRIPT:\n" + transcript_text[:4000] # Safe limit

        tags = metadata.get("tags", [])
        if isinstance(tags, str): tags = [tags]

        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags[:15],
                'categoryId': '22' # People & Blogs
            },
            'status': {
                'privacyStatus': 'private', # Start private for safety
                'selfDeclaredMadeForKids': False
            }
        }

        media = MediaFileUpload(video_path, chunksize=-1, resumable=True)
        
        request = youtube.videos().insert(
            part=','.join(body.keys()),
            body=body,
            media_body=media
        )
        
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logging.info(f"Upload progress: {int(status.progress() * 100)}%")

        video_id = response.get("id")
        logging.info(f"✅ YouTube Upload Success! Video ID: {video_id}")
        return f"https://youtu.be/{video_id}"

    except Exception as e:
        logging.error(f"❌ YouTube Upload Failed: {e}")
        return None

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
        
        # GENERATE FORMATTED TRANSCRIPT
        full_transcript_text = generate_formatted_transcript(transcript_obj.segments)

        update("Generating Assets")
        seo = generate_metadata(full_transcript_text)
        
        # Create SRT (5 Words max per line)
        create_5word_srt(transcript_obj.segments, srt)

        update("Rendering Video")
        render_video(raw, srt, font, final, channel, w, h)

        update("Creating AI Thumbnail")
        # Generate using Flux
        ai_bg = generate_thumbnail_image(full_transcript_text, seo.get("title",""), tmp, w, h)
        has_thumb = False
        
        if ai_bg:
            # NO TEXT OVERLAY - Use pure AI image
            # Convert PNG to JPG for thumbnail
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", ai_bg, "-q:v", "2", thumb_path
                ], check=True)
                has_thumb = True
            except Exception as e:
                logging.error(f"Thumb conversion failed: {e}")
        
        # Fallback to screenshot if AI failed
        if not has_thumb:
            logging.info("Fallback: Using Video Frame for Thumbnail")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-ss", str(dur/2), "-i", final,
                    "-frames:v", "1", "-update", "1", thumb_path
                ], check=True)
                has_thumb = True
            except: pass

        update("Uploading to Cloud")
        vid_res = cloudinary.uploader.upload(final, resource_type="video", folder="viral_edits")
        
        thumb_url = None
        if has_thumb:
            thumb_res = cloudinary.uploader.upload(thumb_path, resource_type="image", folder="viral_thumbnails")
            thumb_url = thumb_res["secure_url"]

        # [UPDATED] Handle YouTube Upload with Transcript
        youtube_link = None
        if form_data.get("youtube_creds"):
            update("Uploading to YouTube")
            youtube_link = upload_video_to_youtube(
                form_data["youtube_creds"], 
                final, 
                seo,
                full_transcript_text # <--- Passed here
            )

        return {
            "status": "success",
            "video_url": vid_res["secure_url"],
            "thumbnail_url": thumb_url,
            "youtube_url": youtube_link, 
            "metadata": seo,
            "transcript_srt": full_transcript_text 
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
