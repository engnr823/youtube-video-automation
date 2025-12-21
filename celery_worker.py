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
from string import Template

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

# ===================================================================
# --- 🧠 SYSTEM PROMPTS V4.0 (INTEGRATED)
# ===================================================================

SEO_METADATA_PROMPT = """
# --- SYSTEM PROMPT V4.0 — SENIOR YOUTUBE SEO STRATEGIST
You are a Senior YouTube Growth Strategist and SEO Algorithm Expert.
Your goal is to engineer metadata that maximizes Click-Through Rate (CTR) and Search Ranking (SEO) by leveraging semantic keywords and psychological hooks.

# --- INPUT CONTEXT
VIDEO TYPE: ${video_type}
SCRIPT CONTEXT: ${transcript}

# --- OPTIMIZATION RULES (STRICT SEO)
1. TITLE ENGINEERING (Max 60 chars):
   - STRUCTURE: [Primary Keyword] + [Power Word/Emotional Hook].
   - STRATEGY: Create a "Curiosity Gap". Don't reveal the ending.
   - FORMAT: If VIDEO TYPE is "reel", strictly append "#Shorts".
   - LANGUAGE: English (Optimized for Global Reach).

2. DESCRIPTION "SEO SANDWICH" STRUCTURE:
   - SECTION 1 (The Hook): 2 sentences. First sentence MUST include the Primary Keyword verbatim.
   - SECTION 2 (The Deep Dive): A detailed 100-word summary of value. Use "LSI Keywords".
   - SECTION 3 (Key Takeaways): A bulleted list of 3-4 things the viewer will learn.
   - SECTION 4 (Call to Action): "Subscribe for more."

3. TAGS & KEYWORDS:
   - Mix of Broad, Exact Match, and Long-Tail tags.

# --- OUTPUT SCHEMA (STRICT JSON)
Return ONLY a valid JSON object:
{
  "title": "Optimized Clickable Title #Shorts",
  "description": "Full SEO-rich description...",
  "tags": ["tag1", "tag2", "tag3"],
  "primary_keyword": "extracted_keyword"
}
"""

THUMBNAIL_ARTIST_PROMPT = """
# --- SYSTEM PROMPT V4.0 — THUMBNAIL ARTIST
You are an expert YouTube Thumbnail Artist.
Your job is to write ONE detailed text-to-image prompt for the Flux AI generator based on this video context: ${context}

# --- OUTPUT REQUIREMENTS
Write a single, highly detailed image prompt. Focus on:
1. The most emotional/dramatic moment.
2. High Emotion in facial expressions (Fear, Joy, Shock).
3. Cinematic Lighting: "Rembrandt lighting", "Volumetric fog", or "Rim light".
4. Texture Quality: "Raw photo", "f/1.8 aperture", "4k", "detailed skin texture".
5. Composition: "Rule of thirds", "Depth of field".

STRICT NEGATIVE CONSTRAINTS:
--no text, words, logos, ui, cartoon, anime, plastic, 3d render, doll, low resolution, blurry

Output ONLY the raw prompt text.
"""

# -------------------------------------------------------------------------
# 🛠️ HELPER FUNCTIONS
# -------------------------------------------------------------------------

def transform_drive_url(url):
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
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        raise RuntimeError(f"Download failed: {str(e)}")

def get_video_info(file_path):
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,duration", "-of", "json", file_path]
        result = subprocess.check_output(cmd).decode('utf-8')
        info = json.loads(result)['streams'][0]
        return float(info.get('duration', 0)), int(info['width']), int(info['height'])
    except:
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
# ✂️ VIDEO PROCESSING (OPTIMIZED ENGINE)
# -------------------------------------------------------------------------

def remove_silence_optimized(input_path, output_path, db_threshold=-30, min_silence_duration=0.6):
    logging.info("✂️ Processing Jump Cuts (High Quality)...")
    try:
        cmd = ["ffmpeg", "-i", input_path, "-af", f"silencedetect=noise={db_threshold}dB:d={min_silence_duration}", "-f", "null", "-"]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
        
        silence_starts = []
        silence_ends = []
        for line in result.stderr.splitlines():
            if "silence_start" in line:
                silence_starts.append(float(line.split("silence_start: ")[1]))
            if "silence_end" in line:
                silence_ends.append(float(line.split("silence_end: ")[1].split(" ")[0]))

        if not silence_starts:
            logging.info("No silence found. Copying.")
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

        # OPTIMIZATION: Use -crf 18 to PRESERVE QUALITY in this intermediate step
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-filter_complex", filter_complex,
            "-map", "[outv]", "-map", "[outa]", "-c:v", "libx264", "-crf", "18", "-preset", "fast", output_path
        ], check=True)
    except Exception:
        logging.warning("Silence removal failed, using original.")
        shutil.copy(input_path, output_path)

def generate_subtitles_english(audio_path):
    logging.info("🎙️ Transcribing & Translating to English...")
    # Forces English translation for all input languages
    with open(audio_path, "rb") as audio_file:
        transcript = openai_client.audio.translations.create(
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

def apply_final_polish_optimized(input_path, srt_path, output_path, blur_watermarks=True, is_vertical=True, do_crop=False):
    """
    COMBINED PASS: Crop + Blur + Subtitles.
    Uses 'MarginV=550' (The 'First Code' Position) for safety.
    """
    logging.info(f"✨ Applying Combined Polish (Crop: {do_crop}, Vertical: {is_vertical})...")
    
    # --- STYLE (Safe Zone) ---
    if is_vertical:
        # MarginV=550 = Upper-Lower-Middle. Safe from TikTok/Reels UI.
        style = "Alignment=2,MarginV=550,FontSize=26,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Bold=1"
    else:
        style = "Alignment=2,MarginV=80,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=0,Bold=1"

    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", input_path]
    filter_chain = []
    last_label = "0:v"

    # 1. CROP
    if do_crop:
        filter_chain.append(f"[{last_label}]scale=-1:1920,crop=1080:1920:((iw-1080)/2):0,setsar=1[v_cropped]")
        last_label = "v_cropped"

    # 2. BLUR WATERMARKS
    if blur_watermarks and is_vertical:
        filter_chain.append(f"[{last_label}]crop=iw:ih*0.10:0:ih*0.90,boxblur=luma_radius=20[bot_blur]")
        filter_chain.append(f"[{last_label}][bot_blur]overlay=0:H-h[v_blurred]")
        last_label = "v_blurred"

    # 3. SUBTITLES
    if os.path.exists(srt_path):
        filter_chain.append(f"[{last_label}]subtitles='{safe_srt}':force_style='{style}'[v_final]")
        last_label = "v_final"

    # EXECUTE
    if filter_chain:
        full_filter = ";".join(filter_chain)
        # Use CRF 23 for final output
        cmd.extend(["-filter_complex", full_filter, "-map", f"[{last_label}]", "-map", "0:a?", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-c:a", "copy"])
    else:
        cmd.extend(["-c", "copy"])
        
    cmd.append(output_path)
    subprocess.run(cmd, check=True)

def generate_packaging_v4(transcript_text, duration, output_format="9:16"):
    logging.info("📦 Generating V4.0 Packaging...")
    is_short = (output_format == "9:16")
    video_type_str = "reel" if is_short else "youtube"

    # 1. METADATA (using V4 prompt)
    try:
        formatted_prompt = Template(SEO_METADATA_PROMPT).safe_substitute(
            video_type=video_type_str,
            transcript=transcript_text[:2500]
        )
        res = openai_client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role":"user", "content": formatted_prompt}], 
            response_format={"type": "json_object"}
        )
        meta = json.loads(res.choices[0].message.content)
    except Exception as e:
        logging.error(f"Metadata Gen Failed: {e}")
        meta = {"title": "Viral Video", "description": "", "tags": []}

    # 2. THUMBNAIL (using V4 prompt logic)
    try:
        # Pass the generated description to the Artist Prompt
        context_data = meta.get('description', transcript_text[:500])
        thumb_gen_prompt = Template(THUMBNAIL_ARTIST_PROMPT).safe_substitute(context=context_data)
        
        res_thumb = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user", "content": thumb_gen_prompt}]
        )
        flux_prompt = res_thumb.choices[0].message.content.strip()
        
        logging.info(f"🎨 Flux Prompt: {flux_prompt[:50]}...")
        
        flux_aspect = "9:16" if is_short else "16:9"
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": flux_prompt + ", photorealistic, 8k, highly detailed", 
                "aspect_ratio": flux_aspect
            }
        )
        thumb_url = str(output[0])
    except Exception as e:
        logging.error(f"Thumbnail Gen Failed: {e}")
        thumb_url = None
        flux_prompt = "Error"

    meta['thumbnail_prompt'] = flux_prompt
    return meta, thumb_url

# -------------------------------------------------------------------------
# 🏭 MAIN TASK
# -------------------------------------------------------------------------
@celery.task(bind=True)
def process_video_upload(self, form_data: dict):
    task_id = str(uuid.uuid4())
    temp_dir = f"/tmp/edit_{task_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
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
        
        target_format = form_data.get('output_format', '9:16')
        is_landscape = w > h
        need_crop = (target_format == '9:16' and is_landscape)
        is_vertical_output = (target_format == '9:16')
        
        current_video = raw_path

        # 2. Silence Removal (Pass 1 - High Quality)
        if form_data.get('remove_silence') == 'true':
            update("Removing Silence...")
            remove_silence_optimized(current_video, processed_path)
            current_video = processed_path
        
        # 3. Transcription (Force English)
        transcript_text = "Video Content"
        srt_exists = False
        if form_data.get('add_subtitles') == 'true':
            update("Transcribing to English...")
            subprocess.run(["ffmpeg", "-y", "-i", current_video, "-q:a", "0", "-map", "a", audio_path], check=True)
            srt_content, transcript_text = generate_subtitles_english(audio_path)
            with open(srt_path, "w", encoding="utf-8") as f: f.write(srt_content)
            srt_exists = True
            
        # 4. Final Polish (Pass 2 - Crop + Blur + Subs)
        update("Applying Polish...")
        apply_final_polish_optimized(
            current_video, 
            srt_path if srt_exists else None,
            final_path,
            blur_watermarks=(form_data.get('blur_watermarks') == 'true'),
            is_vertical=is_vertical_output,
            do_crop=need_crop
        )
        
        # 5. Packaging (V4.0 SEO)
        update("Generating Assets...")
        meta, thumb_url = generate_packaging_v4(transcript_text, dur, target_format)
        
        # 6. Upload
        update("Uploading...")
        cloud_res = cloudinary.uploader.upload(final_path, folder="viral_edits", resource_type="video")
        
        return {
            "status": "success",
            "video_url": cloud_res.get("secure_url"),
            "thumbnail_url": thumb_url,
            "metadata": meta,
            "transcript_srt": srt_content if srt_exists else None
        }

    except Exception as e:
        logging.error(f"Task Failed: {traceback.format_exc()}")
        return {"status": "error", "message": str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
