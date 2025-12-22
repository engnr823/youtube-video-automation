import os
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

# ------------------------------------------------------------------
# CONFIG & AI AGENTS
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SAAS-ENGINE]: %(message)s")

# Load these from your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SEO_PROMPT_TEMPLATE = """
{
  "title": "Viral Clicky Title",
  "description": "SEO Description with hashtags",
  "tags": ["tag1", "tag2"],
  "thumbnail_prompt": "Cinematic, high-detail, 8k, realistic scene of: ${climax}",
  "primary_keyword": "topic"
}
Analyze this transcript: ${transcript}
Return ONLY valid JSON.
"""

# ------------------------------------------------------------------
# DYNAMIC RENDERING ENGINE (The Core Fix)
# ------------------------------------------------------------------
def render_video_final(input_video, srt_file, font_path, output_video, channel_name, width, height):
    """
    Calculates subtitle size and position based on aspect ratio.
    Removes background blurness by focusing on direct overlay.
    """
    is_vertical = height > width
    
    # Logic: Base scaling on height to keep text readable
    # 9:16 (Vertical) usually 1920px high | 16:9 (Full) usually 1080px high
    if is_vertical:
        # SHORT/REEL SETTINGS
        play_res_y = 1920
        play_res_x = 1080
        font_size = 12   # Reduced size for cleaner look
        sub_margin = 120 # Higher up to clear UI buttons
        title_y = "h-80" # Brand title at absolute bottom
    else:
        # CINEMATIC/FULL SETTINGS
        play_res_y = 1080
        play_res_x = 1920
        font_size = 18   # Scaled for horizontal
        sub_margin = 70 
        title_y = "h-50"

    sub_style = (
        f"FontName=Arial,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,BorderStyle=1,Outline=0.5,Shadow=0,"
        f"Alignment=2,MarginV={sub_margin},WrapStyle=2,"
        f"PlayResX={play_res_x},PlayResY={play_res_y}"
    )

    safe_srt = srt_file.replace("\\", "/").replace(":", "\\:")
    safe_font = font_path.replace("\\", "/").replace(":", "\\:")
    font_dir = os.path.dirname(safe_font)
    brand = channel_name.upper()

    # FILTER CHAIN:
    # 1. Draw Title at very bottom
    # 2. Draw Subtitles above it
    filters = [
        f"drawtext=fontfile='{safe_font}':text='{brand}':fontcolor='#FFD700':"
        f"fontsize={font_size + 4}:x=(w-text_w)/2:y={title_y}:"
        f"shadowcolor='black@0.6':shadowx=2:shadowy=2",
        
        f"subtitles='{safe_srt}':fontsdir='{font_dir}':force_style='{sub_style}'"
    ]

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "fast", "-crf", "21", "-c:a", "copy",
        output_video
    ]
    subprocess.run(cmd, check=True)

# ------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------
def get_video_info(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
           "-show_entries", "stream=width,height,duration", "-of", "json", path]
    out = subprocess.check_output(cmd)
    s = json.loads(out)["streams"][0]
    return float(s["duration"]), int(s["width"]), int(s["height"])

def format_ts(sec):
    td = timedelta(seconds=sec)
    total_sec = int(td.total_seconds())
    ms = int(td.microseconds / 1000)
    return f"{total_sec//3600:02}:{(total_sec%3600)//60:02}:{total_sec%60:02},{ms:03}"

def ensure_font(tmp):
    font_path = os.path.join(tmp, "Arial.ttf")
    if not os.path.exists(font_path):
        r = requests.get("https://github.com/matomo-org/travis-scripts/raw/master/fonts/Arial.ttf")
        with open(font_path, "wb") as f: f.write(r.content)
    return font_path

# ------------------------------------------------------------------
# MAIN PROCESSOR
# ------------------------------------------------------------------
def process_video_locally(video_url, channel_name="@MyChannel"):
    task_id = str(uuid.uuid4())
    tmp = f"./job_{task_id}"
    os.makedirs(tmp, exist_ok=True)

    raw_path = os.path.join(tmp, "raw.mp4")
    audio_path = os.path.join(tmp, "audio.mp3")
    srt_path = os.path.join(tmp, "subs.srt")
    final_path = os.path.join(tmp, "final_output.mp4")

    try:
        # 1. Download
        print("--- Downloading ---")
        with requests.get(video_url, stream=True) as r:
            with open(raw_path, "wb") as f: shutil.copyfileobj(r.raw, f)

        duration, w, h = get_video_info(raw_path)
        font = ensure_font(tmp)

        # 2. Transcribe
        print("--- AI Transcribing ---")
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-q:a", "0", "-map", "a", audio_path], check=True)
        with open(audio_path, "rb") as audio_file:
            transcript = openai_client.audio.translations.create(model="whisper-1", file=audio_file, response_format="verbose_json")

        # 3. Create SRT
        full_text = ""
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(transcript.segments):
                f.write(f"{i+1}\n{format_ts(seg.start)} --> {format_ts(seg.end)}\n{seg.text.strip()}\n\n")
                full_text += seg.text + " "

        # 4. SEO & Thumbnail
        print("--- Generating SEO & Thumbnail ---")
        seo_prompt = Template(SEO_PROMPT_TEMPLATE).safe_substitute(transcript=full_text[:2000], climax=full_text[:100])
        res = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role":"user", "content": seo_prompt}], response_format={"type": "json_object"})
        meta = json.loads(res.choices[0].message.content)
        
        flux_out = replicate.run("black-forest-labs/flux-schnell", input={"prompt": meta['thumbnail_prompt'], "aspect_ratio": "9:16" if h>w else "16:9"})
        
        # 5. Render
        print("--- Final Rendering ---")
        render_video_final(raw_path, srt_path, font, final_path, channel_name, w, h)

        print(f"DONE! File saved at: {final_path}")
        print(f"Thumbnail: {flux_out[0]}")
        print(f"SEO Metadata: {meta}")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
    finally:
        # shutil.rmtree(tmp) # Uncomment to clean up after testing
        pass

if __name__ == "__main__":
    # Test with a sample URL
    TEST_VIDEO = "https://www.w3schools.com/html/mov_bbb.mp4" 
    process_video_locally(TEST_VIDEO, "@ViralMaster")
