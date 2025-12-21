# file: celery_worker.py
import os
import sys
import logging
import json
import uuid
import shutil
import subprocess
import requests
import traceback  # <--- CRITICAL FIX: Prevents NameError crash
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
    # Matches /file/d/ID/view or /file/d/ID
    match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', url)
    if match:
        file_id = match.group(1)
        # Construct direct download URL
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def download_file(url, dest_path):
    """Downloads video from any public link."""
    # 1. Fix Google Drive Links automatically
    if "drive.google.com" in url:
        download_url = transform_drive_url(url)
        logging.info(f"🔄 Converted Drive Link to: {download_url}")
    else:
        download_url = url
    
    logging.info(f"⬇️ Downloading: {download_url}")
    
    try:
        # User-Agent header helps avoid some 403 Forbidden errors on generic hosts
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(download_url, stream=True, timeout=300, headers=headers) as r:
            
            # Specific check for Google Drive Permission/Virus page
            if "drive.google.com" in download_url and r.status_code != 200:
                 raise RuntimeError("⛔ Google Drive Access Denied. Make sure the link is set to 'Anyone with the link'.")
            
            if "text/html" in r.headers.get('Content-Type', ''):
                 raise RuntimeError("⛔ The link returned a Web Page instead of a Video File. Check permissions.")

            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")

def get_video_info(file_path):
    """Returns duration, width, height using FFprobe."""
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
        # Fallback
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
    # scale height to 1920, then crop width to 1080 from center
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
        # Detect silence
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

        # Build filter to skip silences
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
        # Fallback if silence logic fails: Just use original
        logging.warning("Silence removal failed, using original.")
        shutil.copy(input_path, output_path)

def generate_subtitles(audio_path):
    logging.info("🎙️ Transcribing with Whisper...")
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

def apply_final_polish(input_path, srt_path, output_path, blur_watermarks=True):
    logging.info("✨ Applying Final Polish...")
    
    # Style: Yellow Text, Black Outline, Bottom Center
    style = "Alignment=2,MarginV=50,Fontname=Arial,FontSize=24,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,Bold=1"
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    
    filters = []
    
    if blur_watermarks:
        # Blurs top 150px and bottom 150px
        filters.append("boxblur=luma_radius=20:luma_power=1:enable='between(y,0,150)+between(y,h-200,h)'")
        
    if os.path.exists(srt_path):
        filters.append(f"subtitles='{safe_srt}':force_style='{style}'")
        
    filter_str = ",".join(filters) if filters else "null"
    
    cmd = ["ffmpeg", "-y", "-i", input_path]
    if filter_str != "null":
        cmd.extend(["-vf", filter_str])
    cmd.extend(["-c:a", "copy", output_path])
    
    subprocess.run(cmd, check=True)

def generate_packaging(transcript_text, duration):
    logging.info("📦 Generating Viral Packaging...")
    try:
        prompt = f"Analyze transcript: '{transcript_text[:800]}...'. Output JSON with keys: title (viral short), description (seo), tags (5 hashtags), thumbnail_prompt (cinematic)."
        res = openai_client.chat.completions.create(
            model="gpt-4o", messages=[{"role":"user", "content":prompt}], response_format={"type": "json_object"}
        )
        meta = json.loads(res.choices[0].message.content)
        
        # Generate Thumbnail
        logging.info(f"🎨 Generating Thumbnail: {meta.get('thumbnail_prompt')[:20]}...")
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": meta.get('thumbnail_prompt', 'Viral video thumbnail'), "aspect_ratio": "9:16"}
        )
        thumb_url = str(output[0])
        return meta, thumb_url
    except Exception as e:
        logging.warning(f"Packaging failed: {e}")
        return {}, None

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
        is_landscape = w > h
        current_video = raw_path
        
        # 2. Crop
        if is_landscape:
            update("Auto-Cropping to Vertical...")
            crop_to_vertical(raw_path, processed_path)
            current_video = processed_path
        
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
            
        # 5. Final Polish
        update("Applying Polish...")
        apply_final_polish(
            current_video, 
            srt_path if form_data.get('add_subtitles') == 'true' else "/dev/null",
            final_path,
            blur_watermarks=(form_data.get('blur_watermarks') == 'true')
        )
        
        # 6. Packaging
        update("Creating Thumbnail & SEO...")
        meta, thumb_url = generate_packaging(transcript_text, dur)
        
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
        error_msg = f"Workflow failed: {str(e)}"
        # CRITICAL FIX: Log the full error but RETURN it as a success payload so Celery doesn't choke.
        logging.error(f"Task Exception: {traceback.format_exc()}")
        return {"status": "error", "message": error_msg}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
