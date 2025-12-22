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
# CONFIG
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

# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------
def transform_drive_url(url):
    patterns = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)']
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
        return float(s["duration"]), int(s["width"]), int(s["height"])
    except:
        return 0.0, 1920, 1080


def format_ts(sec):
    td = timedelta(seconds=sec)
    s = int(td.total_seconds())
    return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},000"


def ensure_font(tmp):
    font = os.path.join(tmp, "Arial.ttf")
    if not os.path.exists(font):
        r = requests.get(
            "https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf",
            timeout=10
        )
        with open(font, "wb") as f:
            f.write(r.content)
    return font

# ------------------------------------------------------------------
# ABSOLUTE BOTTOM SUBTITLE RENDER (UNIVERSAL)
# ------------------------------------------------------------------
def render_video(
    input_video,
    srt_file,
    font_path,
    output_video,
    channel_name,
    width,
    height
):
    is_vertical = height > width

    if is_vertical:
        play_res = "PlayResX=1080,PlayResY=1920"
        font_size = 18
        margin_v = 6
        brand_y = 40
    else:
        play_res = "PlayResX=1920,PlayResY=1080"
        font_size = 20
        margin_v = 8
        brand_y = 20

    sub_style = (
        f"FontName=Arial,"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline=1,"
        f"Shadow=0,"
        f"Bold=0,"
        f"Alignment=2,"
        f"MarginV={margin_v},"
        f"WrapStyle=2,"
        f"{play_res}"
    )

    safe_srt = srt_file.replace("\\", "/").replace(":", "\\:")
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(safe_font)
    brand = channel_name.replace(":", "").replace("'", "")

    filters = [
        f"drawtext=fontfile='{safe_font}':"
        f"text='{brand}':"
        f"fontcolor=white@0.5:"
        f"fontsize=18:"
        f"x=(w-text_w)/2:"
        f"y={brand_y}:"
        f"shadowcolor=black:"
        f"shadowx=1:"
        f"shadowy=1",
        f"subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'"
    ]

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-vf", ",".join(filters),
        "-map", "0:v",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        output_video
    ]

    subprocess.run(cmd, check=True)

# ------------------------------------------------------------------
# CELERY TASK
# ------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data):
    task_id = str(uuid.uuid4())
    tmp = f"/tmp/job_{task_id}"
    os.makedirs(tmp, exist_ok=True)

    raw = os.path.join(tmp, "raw.mp4")
    audio = os.path.join(tmp, "audio.mp3")
    srt = os.path.join(tmp, "subs.srt")
    final = os.path.join(tmp, "final.mp4")

    try:
        def update(m):
            self.update_state(state="PROGRESS", meta={"message": m})

        update("Downloading video")
        download_file(form_data["video_url"], raw)

        duration, w, h = get_video_info(raw)
        font = ensure_font(tmp)
        channel = form_data.get("channel_name", "@ViralShorts")

        update("Extracting audio")
        subprocess.run([
            "ffmpeg", "-y",
            "-i", raw,
            "-map", "a",
            "-q:a", "0",
            audio
        ], check=True)

        update("Transcribing")
        transcript = openai_client.audio.translations.create(
            model="whisper-1",
            file=open(audio, "rb"),
            response_format="verbose_json"
        )

        srt_text = ""
        for i, seg in enumerate(transcript.segments):
            srt_text += (
                f"{i+1}\n"
                f"{format_ts(seg.start)} --> {format_ts(seg.end)}\n"
                f"{seg.text.strip()}\n\n"
            )

        with open(srt, "w", encoding="utf-8") as f:
            f.write(srt_text)

        update("Rendering final video")
        render_video(raw, srt, font, final, channel, w, h)

        update("Uploading")
        cloud = cloudinary.uploader.upload(
            final,
            resource_type="video",
            folder="viral_edits"
        )

        return {
            "status": "success",
            "video_url": cloud["secure_url"],
            "transcript_srt": srt_text
        }

    except Exception as e:
        logging.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
