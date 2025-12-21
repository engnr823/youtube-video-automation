# file: celery_worker.py
import os
import sys
import logging
import json
import uuid
import shutil
import math
import subprocess
import requests
from pathlib import Path
from datetime import timedelta

# Add current dir to path
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR)

# Libraries
import cloudinary
import cloudinary.uploader
import replicate
from openai import OpenAI
from celery_init import celery

# Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (EDITOR): %(message)s")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Cloudinary Setup
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

# -------------------------------------------------------------------------
# 🛠️ Helper Functions: The Editing Suite
# -------------------------------------------------------------------------

def download_file(url, dest_path):
    """Downloads a video from a URL to local disk."""
    logging.info(f"⬇️ Downloading raw video: {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return dest_path

def get_video_duration(file_path):
    """Gets duration using FFprobe."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    return float(subprocess.check_output(cmd).strip())

def format_timestamp(seconds):
    """Converts seconds to SRT timestamp format (00:00:00,000)."""
    td = timedelta(seconds=seconds)
    # Handle the comma for milliseconds
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# -------------------------------------------------------------------------
# ✂️ Feature 1: Silence Remover (Jump Cuts)
# -------------------------------------------------------------------------
def remove_silence(input_path, output_path, db_threshold=-30, min_silence_duration=0.5):
    """
    Uses FFmpeg silencedetect to find silent parts and skips them.
    """
    logging.info("✂️ Analyzing audio for silence removal...")
    
    # 1. Detect Silence
    cmd = [
        "ffmpeg", "-i", input_path, "-af", 
        f"silencedetect=noise={db_threshold}dB:d={min_silence_duration}", 
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    output = result.stderr

    # 2. Parse Silence Logs
    silence_starts = []
    silence_ends = []
    for line in output.splitlines():
        if "silence_start" in line:
            silence_starts.append(float(line.split("silence_start: ")[1]))
        if "silence_end" in line:
            silence_ends.append(float(line.split("silence_end: ")[1].split(" ")[0]))

    if not silence_starts:
        logging.info("No silence detected. Copying original.")
        shutil.copy(input_path, output_path)
        return

    # 3. Construct Keep Segments
    segments = []
    current_time = 0.0
    # Pair starts and ends
    silence_periods = list(zip(silence_starts, silence_ends))
    
    video_len = get_video_duration(input_path)
    
    # Filter Logic
    filter_complex = ""
    concat_n = 0
    
    for start, end in silence_periods:
        if start > current_time:
            # Keep segment from current_time to start of silence
            filter_complex += f"[0:v]trim=start={current_time}:end={start},setpts=PTS-STARTPTS[v{concat_n}];"
            filter_complex += f"[0:a]atrim=start={current_time}:end={start},asetpts=PTS-STARTPTS[a{concat_n}];"
            concat_n += 1
        current_time = end # Skip the silence
        
    # Add final segment if exists
    if current_time < video_len:
        filter_complex += f"[0:v]trim=start={current_time}:end={video_len},setpts=PTS-STARTPTS[v{concat_n}];"
        filter_complex += f"[0:a]atrim=start={current_time}:end={video_len},asetpts=PTS-STARTPTS[a{concat_n}];"
        concat_n += 1

    # Concat
    filter_complex += "".join([f"[v{i}][a{i}]" for i in range(concat_n)])
    filter_complex += f"concat=n={concat_n}:v=1:a=1[outv][outa]"

    logging.info(f"✂️ Cutting {len(silence_periods)} silent segments...")
    
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-preset", "fast", output_path
    ], check=True)

# -------------------------------------------------------------------------
# 📝 Feature 2: Transcribe & Generate Subtitles
# -------------------------------------------------------------------------
def generate_subtitles(audio_path):
    """Uses OpenAI Whisper to transcribe and creates an SRT file."""
    logging.info("🎙️ Transcribing audio...")
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="verbose_json"
        )
    
    srt_content = ""
    for i, segment in enumerate(transcript.segments):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_content += f"{i+1}\n{start} --> {end}\n{text}\n\n"
        
    return srt_content

# -------------------------------------------------------------------------
# 🖼️ Feature 3: Header/Footer Blurring (Logo Hider)
# -------------------------------------------------------------------------
def create_blur_filter(height):
    """Calculates crop parameters to blur top 15% and bottom 15%."""
    # Logic: Clone video -> Crop Top -> Blur -> Overlay. Clone -> Crop Bottom -> Blur -> Overlay
    # Simplified FFmpeg delogo is easier but requires coordinates.
    # We will use 'gblur' (Gaussian Blur) with masks.
    
    # Simple approach: Draw a blurred box at bottom (Footer) and top (Header)
    # We assume 1080x1920 video. 
    # Bottom 200px: y=1720, h=200. Top 150px.
    return (
        "[0:v]boxblur=luma_radius=20:luma_power=1:enable='between(y,0,180)+between(y,1700,1920)'[bg]"
    )

# -------------------------------------------------------------------------
# 🧠 Feature 4: Catchy Thumbnail & Metadata
# -------------------------------------------------------------------------
def generate_metadata_and_thumbnail(video_path, transcript_text):
    """
    1. Extracts a frame from the video.
    2. Sends frame + transcript to GPT-4o to write Metadata & Thumbnail Prompt.
    3. Generates Thumbnail with Flux.
    """
    logging.info("🧠 Generating SEO Metadata & Thumbnail Concept...")
    
    # 1. Extract Frame at 50% mark
    duration = get_video_duration(video_path)
    snapshot_path = "/tmp/snapshot.jpg"
    subprocess.run(["ffmpeg", "-y", "-ss", str(duration/2), "-i", video_path, "-vframes", "1", "-q:v", "2", snapshot_path], check=True)
    
    # 2. GPT-4 Analysis
    prompt = f"""
    Analyze this video frame and the transcript: "{transcript_text[:500]}..."
    
    Task 1: Write a Viral YouTube Shorts Title (Max 60 chars), Description (SEO optimized), and 5 Hashtags.
    Task 2: Write a detailed image prompt for 'Flux' to create a high-click-through-rate YouTube Thumbnail. It should be 'Hyper-realistic, Cinematic, 4k'.
    
    Output JSON: {{ "title": "", "description": "", "tags": [], "thumbnail_prompt": "" }}
    """
    
    # (Simplified for code brevity: assumes standard GPT call)
    # In production, send the image to GPT-4o-Vision. Here we stick to text for speed/cost if image upload is complex.
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    data = json.loads(response.choices[0].message.content)
    
    # 3. Generate Thumbnail via Flux
    logging.info(f"🎨 Generating Thumbnail: {data['thumbnail_prompt'][:30]}...")
    flux_output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": data['thumbnail_prompt'], "aspect_ratio": "9:16"}
    )
    thumb_url = str(flux_output[0])
    
    return data, thumb_url

# -------------------------------------------------------------------------
# 🏭 The Main Workflow
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    raw_path = os.path.join(temp_dir, "raw_input.mp4")
    cut_path = os.path.join(temp_dir, "silence_cut.mp4")
    final_path = os.path.join(temp_dir, "final_output.mp4")
    srt_path = os.path.join(temp_dir, "subtitles.srt")
    
    try:
        # 1. Download Video (From Flow/Drive URL)
        video_url = form_data.get("video_url")
        if not video_url: return {"status": "error", "message": "No Video URL provided"}
        
        logging.info("--- Step 1: Ingest ---")
        download_to_file(video_url, raw_path)
        
        # 2. Silence Removal (Optional toggle)
        logging.info("--- Step 2: Smart Cut ---")
        if form_data.get("remove_silence", "true") == "true":
            remove_silence(raw_path, cut_path)
        else:
            shutil.copy(raw_path, cut_path)
            
        # 3. Transcribe & Subtitles
        logging.info("--- Step 3: Transcription ---")
        # Extract audio for whisper
        audio_path = os.path.join(temp_dir, "audio.mp3")
        subprocess.run(["ffmpeg", "-y", "-i", cut_path, "-q:a", "0", "-map", "a", audio_path], check=True)
        
        srt_content = generate_subtitles(audio_path)
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
            
        # Read transcript for metadata generation
        transcript_text = srt_content.replace("-->", "") # Dirty clean for context
            
        # 4. Final Processing (Blur + Burn Subs)
        logging.info("--- Step 4: Visual Polish ---")
        
        # FFmpeg Filter Complex:
        # 1. boxblur: Blurs top 180px and bottom 200px (Hides TikTok/Veo logos)
        # 2. subtitles: Burns the SRT file with a specific style
        
        # Escape path for FFmpeg
        safe_srt = srt_path.replace(":", "\\:").replace("'", "'\\''")
        
        style = "Alignment=2,MarginV=30,Fontname=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0"
        
        # Note: boxblur requires simple filter syntax. 
        # For simplicity in this script, we assume vertical video (9:16).
        filters = f"boxblur=luma_radius=15:luma_power=1:enable='between(y,0,150)+between(y,h-200,h)',subtitles='{safe_srt}':force_style='{style}'"
        
        subprocess.run([
            "ffmpeg", "-y", "-i", cut_path, 
            "-vf", filters, 
            "-c:a", "copy", 
            final_path
        ], check=True)
        
        # 5. Metadata & Thumbnail
        logging.info("--- Step 5: SEO & Packaging ---")
        metadata, thumb_url = generate_metadata_and_thumbnail(final_path, transcript_text)
        
        # 6. Upload
        logging.info("--- Uploading ---")
        cloud_res = cloudinary.uploader.upload(final_path, folder="edited_videos", resource_type="video")
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": metadata,
            "transcript_srt": srt_content # Return SRT if user wants to download it
        }

    except Exception as e:
        logging.error(f"Editing Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
