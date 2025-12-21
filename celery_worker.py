# file: celery_worker.py
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

# -------------------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# -------------------------------------------------------------------------

def transform_drive_url(url):
    """Converts a Google Drive 'View' link to a 'Direct Download' link."""
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
    """Downloads video from any public link."""
    if "drive.google.com" in url:
        download_url = transform_drive_url(url)
    else:
        download_url = url
    
    logging.info(f"⬇️ Downloading: {download_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(download_url, stream=True, timeout=300, headers=headers) as r:
            if r.status_code in [401, 403]:
                raise RuntimeError("⛔ Access Denied. Make sure link is Public.")
            if "text/html" in r.headers.get('Content-Type', ''):
                 raise RuntimeError("⛔ Link returned HTML page, not video file.")
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")

def get_video_info(file_path):
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration", 
            "-of", "json", file_path
        ]
        result = subprocess.check_output(cmd).decode('utf-8')
        info = json.loads(result)['streams'][0]
        return float(info.get('duration', 0)), int(info['width']), int(info['height'])
    except Exception:
        return 0.0, 1080, 1920

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# -------------------------------------------------------------------------
# ✂️ VIDEO PROCESSING FUNCTIONS
# -------------------------------------------------------------------------

def crop_to_vertical(input_path, output_path):
    logging.info("📐 Auto-Cropping to Vertical (9:16)...")
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "scale=-1:1920,crop=1080:1920:((iw-1080)/2):0,setsar=1",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)

def remove_silence(input_path, output_path, db_threshold=-30, min_silence_duration=0.6):
    logging.info("✂️ Processing Jump Cuts...")
    try:
        cmd = ["ffmpeg", "-i", input_path, "-af", f"silencedetect=noise={db_threshold}dB:d={min_silence_duration}", "-f", "null", "-"]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        
        silence_starts = []
        silence_ends = []
        for line in result.stderr.splitlines():
            if "silence_start" in line:
                silence_starts.append(float(line.split("silence_start: ")[1]))
            if "silence_end" in line:
                if "silence_end" in line: silence_ends.append(float(line.split("silence_end: ")[1].split(" ")[0]))

        if not silence_starts:
            logging.info("No silence found. Skipping cut.")
            shutil.copy(input_path, output_path)
            return

        dur, _, _ = get_video_info(input_path)
        filter_complex = ""
        concat_idx = 0
        current_time = 0.0
        periods = list(zip(silence_starts, silence_ends))
        
        for start, end in periods:
            if start > current_time:
                filter_complex += f"[0:v]trim=start={current_time}:end={start},setpts=PTS-STARTPTS[v{concat_idx}];"
                filter_complex += f"[0:a]atrim=start={current_time}:end={start},asetpts=PTS-STARTPTS[a{concat_idx}];"
                concat_idx += 1
            current_time = end
            
        if current_time < dur:
            filter_complex += f"[0:v]trim=start={current_time}:end={dur},setpts=PTS-STARTPTS[v{concat_idx}];"
            filter_complex += f"[0:a]atrim=start={current_time}:end={dur},asetpts=PTS-STARTPTS[a{concat_idx}];"
            concat_idx += 1

        filter_complex += "".join([f"[v{i}][a{i}]" for i in range(concat_idx)])
        filter_complex += f"concat=n={concat_idx}:v=1:a=1[outv][outa]"

        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", output_path
        ], check=True)
    except Exception:
        logging.warning("Silence removal failed, using original.")
        shutil.copy(input_path, output_path)

def generate_subtitles(audio_path):
    logging.info("🎙️ Transcribing with Whisper (Source Language)...")
    # Using 'transcriptions' preserves original language (e.g. Urdu/Hindi)
    # Using 'translations' would force English.
    # Choose 'transcriptions' for original language accuracy.
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
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

def apply_final_polish(input_path, srt_path, output_path, blur_watermarks=True, is_vertical=True):
    logging.info(f"✨ Applying Final Polish (Vertical: {is_vertical})...")
    
    # --- PRO SUBTITLE STYLE ---
    # Fontname: Removed (Let fontconfig pick Noto Sans for Urdu/Hindi support)
    # PrimaryColour=&H00FFFFFF (White Text)
    # OutlineColour=&H00000000 (Black Outline)
    # BorderStyle=1 (Outline Only - NO BOX)
    # MarginV=25 (Very Bottom, sits inside the blur area)
    
    if is_vertical:
        style = "Alignment=2,MarginV=25,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"
    else:
        style = "Alignment=2,MarginV=25,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1,Shadow=0,Bold=1"

    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", input_path]
    
    filter_chain = ""
    current_stream = "0:v"
    
    # --- BLUR LOGIC (Bottom 10% Only) ---
    if blur_watermarks and is_vertical:
        # Crop bottom 10% (0.10) -> Blur -> Overlay back
        # y = ih*0.90 (Starts at 90% down)
        filter_chain += f"[{current_stream}]crop=iw:ih*0.10:0:ih*0.90,boxblur=luma_radius=20[bot_blur];" \
                        f"[{current_stream}][bot_blur]overlay=0:H-h[v_blurred]"
        current_stream = "v_blurred" 

    # --- SUBTITLE LOGIC ---
    if os.path.exists(srt_path):
        if filter_chain:
            filter_chain += f";[{current_stream}]subtitles='{safe_srt}':force_style='{style}'[v_final]"
        else:
            filter_chain = f"[{current_stream}]subtitles='{safe_srt}':force_style='{style}'[v_final]"
        current_stream = "v_final"

    # --- EXECUTE ---
    if filter_chain:
        cmd.extend(["-filter_complex", filter_chain, "-map", f"[{current_stream}]", "-map", "0:a?", "-c:v", "libx264", "-c:a", "copy"])
    else:
        cmd.extend(["-c", "copy"])
        
    cmd.append(output_path)
    subprocess.run(cmd, check=True)

def generate_packaging(transcript_text, duration, output_format="9:16"):
    logging.info("📦 Generating Advanced Viral Packaging (SEO V4.0)...")
    
    # Context
    is_short = (output_format == "9:16")
    video_type_str = "reel" if is_short else "youtube"
    
    # --- SENIOR SEO STRATEGIST PROMPT ---
    seo_system_prompt = f"""
    You are a Senior YouTube Growth Strategist.
    Goal: Maximize CTR and SEO using semantic keywords and psychological hooks.
    
    INPUT CONTEXT:
    - VIDEO TYPE: {video_type_str}
    - TRANSCRIPT: {transcript_text[:1500]}...
    
    OUTPUT REQUIREMENTS:
    1. TITLE: [Primary Keyword] + [Power Word]. If reel, append #Shorts. Max 60 chars.
    2. DESCRIPTION: "SEO Sandwich". Hook (2 sentences) + Deep Dive (100 words) + Tags.
    3. TAGS: Mix of Broad, Exact, and Long-Tail tags.
    4. THUMBNAIL_PROMPT: A highly detailed, cinematic image prompt for Flux AI. Focus on emotion, lighting (Rembrandt/Volumetric), and texture (8k, raw photo).
    
    Return ONLY valid JSON: {{ "title": "...", "description": "...", "tags": [...], "thumbnail_prompt": "..." }}
    """

    try:
        res = openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role":"user", "content": seo_system_prompt}], 
            response_format={"type": "json_object"}
        )
        meta = json.loads(res.choices[0].message.content)
        
        # --- THUMBNAIL GENERATION (Using the Expert Prompt) ---
        thumb_prompt = meta.get('thumbnail_prompt', 'Cinematic viral video thumbnail')
        logging.info(f"🎨 Generating Thumbnail: {thumb_prompt[:40]}...")
        
        flux_aspect = "9:16" if is_short else "16:9"
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": thumb_prompt + ", highly detailed, 8k, cinematic lighting, photorealistic", 
                "aspect_ratio": flux_aspect
            }
        )
        thumb_url = str(output[0])
        return meta, thumb_url

    except Exception as e:
        logging.warning(f"Packaging failed: {e}")
        return {
            "title": "Viral Video #Shorts",
            "description": "Watch this amazing video.",
            "tags": ["viral", "trending"]
        }, None

# -------------------------------------------------------------------------
# 🏭 MAIN TASK
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Paths
    raw_path = os.path.join(temp_dir, "raw.mp4")
    processed_path = os.path.join(temp_dir, "processed.mp4")
    final_path = os.path.join(temp_dir, "final.mp4")
    audio_path = os.path.join(temp_dir, "audio.mp3")
    srt_path = os.path.join(temp_dir, "subs.srt")
    
    try:
        def update(msg): self.update_state(state="PROGRESS", meta={"message": msg})
        
        # 1. Ingest
        update("Downloading Video...")
        download_file(form_data['video_url'], raw_path)
        
        dur, w, h = get_video_info(raw_path)
        
        # Determine Format
        target_format = form_data.get('output_format', '9:16')
        is_landscape = w > h
        
        current_video = raw_path
        
        # 2. Smart Format Logic
        if target_format == '9:16' and is_landscape:
            update("Converting to Vertical (9:16)...")
            crop_to_vertical(raw_path, processed_path)
            current_video = processed_path
            is_vertical_output = True
        else:
            is_vertical_output = (target_format == '9:16')
            if is_landscape and not is_vertical_output:
                current_video = raw_path # Keep original
            elif not is_landscape and is_vertical_output:
                current_video = raw_path # Already vertical
            else:
                 current_video = raw_path
            
        # 3. Silence Removal
        if form_data.get('remove_silence') == 'true':
            update("Removing Silence...")
            remove_silence(current_video, processed_path.replace(".mp4", "_cut.mp4"))
            current_video = processed_path.replace(".mp4", "_cut.mp4")
            
        # 4. Transcription
        transcript_text = "Video Content"
        if form_data.get('add_subtitles') == 'true':
            update("Transcribing...")
            subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
            srt_content, transcript_text = generate_subtitles(audio_path)
            with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
            
        # 5. Final Polish (Blur + Subs)
        update("Applying Polish...")
        apply_final_polish(
            current_video, 
            srt_path if form_data.get('add_subtitles') == 'true' else None,
            final_path,
            blur_watermarks=(form_data.get('blur_watermarks') == 'true'),
            is_vertical=is_vertical_output
        )
        
        # 6. Packaging (SEO + Thumbnail)
        update("Creating Thumbnail & SEO...")
        meta, thumb_url = generate_packaging(transcript_text, dur, target_format)
        
        # 7. Upload
        update("Uploading Final Video...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_content if os.path.exists(srt_path) else None
        }

    except Exception as e:
        logging.error(f"Task Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
