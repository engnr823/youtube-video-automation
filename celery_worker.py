# file: celery_worker.py
import os
import sys
import logging
import json
import re
import uuid
import shutil
import tempfile
import traceback
import concurrent.futures
import subprocess 
import random
import time
from string import Template
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# --- CRITICAL FIX: Ensure Python can find sibling packages ---
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR) 
# ---------------------------------------------------------------------------------

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError

# Celery app import
from celery_init import celery

# --- REPLICATE CLIENT (Added for Flux Images) ---
import replicate 

from openai import OpenAI

# --- IMPORT OPENAI TTS CLIENT (For Audio Fallback) ---
try:
    from video_clients.openai_client import generate_openai_speech
except ImportError:
    generate_openai_speech = None
    logging.warning("⚠️ OpenAI TTS client not found. Fallback will not work.")

# --- CONSTANTS FOR VOICES AND AVATARS ---
MALE_VOICE_ID = "ErXwobaYiN019PkySvjV" 
FEMALE_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# --- FIXED CAST LIST (YOUR CINEMATIC CUSTOM IDS) ---
CAST_LIST = {
    "MALE_LEAD": "4343bfb447bf4028a48b598ae297f5dc",    # Your Custom Male
    "FEMALE_LEAD": "16a811adf1cc4b12bc6edd04c8fecffa",  # Your Custom Female
    "NARRATOR": "4343bfb447bf4028a48b598ae297f5dc"      # Default Narrator
}

SAFE_FALLBACK_AVATAR_ID = "4343bfb447bf4028a48b598ae297f5dc"

# --- Utility Fix: Define missing ensure_dir function ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

# --- VOICE SETTINGS SAFE IMPORT ---
try:
    from elevenlabs import VoiceSettings
except Exception:
    VoiceSettings = None

# --- UTILS IMPORT SAFETY BLOCK ---
try:
    from utils.ffmpeg_utils import get_media_duration, composite_green_screen_scene
except Exception:
    def get_media_duration(file_path): return 0.0
    def composite_green_screen_scene(bg, fg, out): return False

# --- CLIENT IMPORT SAFETY BLOCK ---
try:
    from video_clients.elevenlabs_client import generate_audio_for_scene
except Exception:
    generate_audio_for_scene = None

# --- CRITICAL FIX: ROBUST HEYGEN CLIENT IMPORT ---
try:
    # REMOVED 'get_safe_fallback_id' to fix the import error
    from video_clients.heygen_client import (
        generate_heygen_video, 
        get_all_avatars, 
        HeyGenError
    )
    HEYGEN_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ HEYGEN CLIENT NOT FOUND: {e}")
    generate_heygen_video = None
    get_all_avatars = None
    HEYGEN_AVAILABLE = False


# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )
else:
    logging.warning("⚠️ Cloudinary not fully configured; uploads will fail.")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# -------------------------
# 🎶 Royalty-Free Music Library (Emotion Mapped)
# -------------------------
MUSIC_LIBRARY = {
    "motivational": "https://cdn.pixabay.com/audio/2022/07/22/powerful-8526.mp3",
    "intense": "https://cdn.pixabay.com/audio/2023/12/06/trailer-mood-176840.mp3", 
    "sad": "https://cdn.pixabay.com/audio/2022/09/20/emotional-128225.mp3",
    "ambient": "https://cdn.pixabay.com/audio/2022/07/26/cinematic-ambient-11634.mp3",
    "happy": "https://cdn.pixabay.com/audio/2022/10/05/upbeat-corporate-123.mp3",
    "default": "https://cdn.pixabay.com/audio/2022/07/22/powerful-8526.mp3"
}

# -------------------------
# Pydantic Schemas
# -------------------------
class SceneSchema(BaseModel):
    scene_id: int
    duration_seconds: float
    visual_prompt: str
    action_prompt: str
    audio_narration: Optional[str] = ""
    shot_type: Optional[str] = "medium"
    is_b_roll: Optional[bool] = False 
    characters_in_scene: Optional[List[str]] = []
    
class StoryboardSchema(BaseModel):
    video_title: str
    video_description: Optional[str] = ""
    main_character_profile: Optional[str] = ""
    characters: Optional[List[dict]] = []
    scenes: List[SceneSchema]

# --- CHARACTER DATABASE ---
CHAR_DB_PATH = os.getenv("CHAR_DB_PATH", "/var/data/character_db.json")
ensure_dir(str(Path(CHAR_DB_PATH).parent))

def ensure_character(name: str, appearance_prompt: Optional[str] = None, reference_image_url: Optional[str] = None, voice_id: Optional[str] = None) -> dict:
    try:
        with open(CHAR_DB_PATH, "r") as f: db = json.load(f)
    except Exception:
        db = {}
    if name in db: return db[name]
    db[name] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "appearance_prompt": appearance_prompt or f"{name}, photorealistic",
        "reference_image": reference_image_url,
        "voice_id": voice_id
    }
    try:
        with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    except Exception as e:
        logging.warning(f"Could not write char DB: {e}")
    return db[name]

# -------------------------
# Core Utils
# -------------------------

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def download_to_file(url: str, dest_path: str, timeout: int = 300):
    logging.info(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return dest_path

def safe_upload_to_cloudinary(filepath: str, resource_type="video", folder="automations"):
    try:
        res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
        url = res.get("secure_url") or res.get("url")
        logging.info(f"Uploaded to Cloudinary: {url}")
        return url
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return None

def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if not os.path.exists(path): return ""
    with open(path, "r", encoding="utf-8") as f: return f.read()

def extract_json_from_text(text: str) -> Optional[dict]:
    if not text: return None
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
    return None

def extract_json_list_from_text(text: str) -> Optional[list]:
    if not text: return None
    m = re.search(r'```(?:json)?\s*(\[.*\])\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
    return None

# -------------------------
# Image generation (Flux via Replicate)
# -------------------------
def generate_flux_image(prompt: str, aspect: str = "16:9", negative_prompt: str = "") -> str:
    """Generates a background image using Flux-Schnell (Replicate)."""
    try:
        logging.info(f"🎨 Generating background with Flux (Replicate): {prompt[:30]}...")
        if aspect not in ["16:9", "9:16", "1:1", "4:5", "2:3", "3:2"]: aspect = "16:9"
        
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt + ", cinematic lighting, 8k, photorealistic", "aspect_ratio": aspect, "output_format": "jpg", "output_quality": 90}
        )
        if output:
            image_url = str(output[0])
            logging.info("✅ Background generated successfully (Flux).")
            return image_url
    except Exception as e:
        logging.error(f"Flux generation failed: {e}")
    return "https://placeimg.com/480/832/abstract"

# -------------------------
# B-Roll Video Creator (Local FFmpeg)
# -------------------------
def create_static_video_from_image(image_path: str, duration: float, output_path: str) -> bool:
    try:
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path, "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
            "-preset", "ultrafast", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except Exception as e:
        logging.error(f"Failed to create B-Roll video: {e}")
        return False

# -------------------------
# Single scene processor
# -------------------------
def process_single_scene(
    scene: dict, index: int, character_profile: str, audio_path: str = None, 
    character_faces: dict = {}, aspect: str = "9:16", 
    fallback_avatar_id: str = None, background_url: str = None 
) -> dict:
    
    if not HEYGEN_AVAILABLE: raise RuntimeError("HeyGen client not available.")
    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("/tmp", f"scene_{index}_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        bg_image_local_path = os.path.join(temp_dir, f"bg_{index}.jpg")
        if background_url:
            try: download_to_file(background_url, bg_image_local_path)
            except Exception as e: logging.warning(f"BG Download failed: {e}")

        is_b_roll = scene.get("is_b_roll", False)
        if is_b_roll:
            logging.info(f"💰 Scene {index} is B-Roll. Skipping HeyGen.")
            final_scene_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
            duration = scene.get("duration_seconds", 5.0)
            if os.path.exists(bg_image_local_path):
                create_static_video_from_image(bg_image_local_path, duration, final_scene_path)
                return {"index": index, "video_path": final_scene_path, "status": "success"}
            return {"index": index, "video_path": None, "status": "failed"}

        character_list = scene.get('characters_in_scene')
        target_char_name = character_list[0] if character_list else list(character_faces.keys())[0] if character_faces else "MENTOR"
        char_data = character_faces.get(target_char_name, {})
        avatar_id = char_data.get("heygen_avatar_id") or fallback_avatar_id
        
        cloud_audio_url = None
        if audio_path and os.path.exists(audio_path):
            cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
        
        if not cloud_audio_url:
             logging.warning(f"Scene {index} skipped: Audio missing.")
             return {"index": index, "video_path": None, "status": "skipped"}

        logging.info(f"Calling HeyGen Scene {index} (Avatar: {avatar_id})")
        video_url = generate_heygen_video(avatar_id, cloud_audio_url, aspect_ratio=aspect)

        if not video_url: raise RuntimeError("HeyGen generation failed")

        green_screen_path = os.path.join(temp_dir, f"green_raw_{index}.mp4")
        download_to_file(str(video_url), green_screen_path, timeout=300)

        final_scene_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
        composite_success = False
        if os.path.exists(bg_image_local_path) and os.path.exists(green_screen_path):
            try: composite_success = composite_green_screen_scene(bg_image_local_path, green_screen_path, final_scene_path)
            except Exception as e: logging.error(f"Composite error: {e}")

        if not composite_success and os.path.exists(green_screen_path):
            shutil.copy(green_screen_path, final_scene_path)
        
        if not os.path.exists(final_scene_path): return {"index": index, "video_path": None, "status": "failed"}

        return {"index": index, "video_path": final_scene_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {traceback.format_exc()}")
        return {"index": index, "video_path": None, "status": "failed"}

# -------------------------
# Stitching (FIXED AUDIO CHANNELS)
# -------------------------
def stitch_video_audio_pairs_optimized(scene_pairs: List[Tuple[str, str]], output_path: str) -> bool:
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"render_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []
    
    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for stitching...")
        for i, (video, audio) in enumerate(scene_pairs):
            if not os.path.exists(video): continue
            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            
            # Duration logic
            audio_dur = get_media_duration(audio) if audio and os.path.exists(audio) else 0.0
            video_dur = get_media_duration(video)
            final_dur = audio_dur if audio_dur > 0.5 else video_dur
            if final_dur <= 0: final_dur = 5.0
            
            cmd = ["ffmpeg", "-y", "-stream_loop", "-1", "-i", video]
            
            # CRITICAL AUDIO SETTINGS
            audio_flags = ["-c:a", "aac", "-ac", "2", "-ar", "44100"] 
            
            if audio and os.path.exists(audio): 
                 cmd.extend(["-i", audio, "-map", "0:v", "-map", "1:a"])
            else: 
                 # Generate silent audio matching the format
                 cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-map", "0:v", "-map", "1:a"])

            cmd.extend([
                "-t", str(final_dur), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30"
            ])
            
            cmd.extend(audio_flags)
            cmd.extend(["-shortest", chunk_name])
            
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths: return False
        
        with open(input_list_path, "w") as f:
            for chunk in chunk_paths: 
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", input_list_path, "-c", "copy", output_path], check=True, capture_output=True)
        return os.path.exists(output_path)

    except Exception as e:
        logging.error(f"Stitching failed: {e}")
        return False
    finally: shutil.rmtree(temp_dir, ignore_errors=True)

# -------------------------
# Agents
# -------------------------
def get_openai_response(prompt, temp=0.7, is_json=False):
    if not openai_client: return ""
    try:
        res = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[{"role":"system","content":"Output JSON" if is_json else "You are a writer"},{"role":"user","content":prompt}],
            temperature=temp, response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return res.choices[0].message.content or ""
    except: return ""

def create_video_storyboard_agent(keyword, blueprint, form_data):
    prompt_template = load_prompt_template("prompt_video_storyboard_creator.txt") or "TASK: Create script JSON for '$keyword'"
    template = Template(prompt_template)
    prompt = template.safe_substitute(
        keyword=keyword, blueprint_json=json.dumps(blueprint),
        language=form_data.get("language", "english"), video_type=form_data.get("video_type", "reel"),
        max_scenes=str(form_data.get("max_scenes", 7))
    )
    raw = get_openai_response(prompt, 0.6, True)
    obj = extract_json_from_text(raw) or {}
    try: return StoryboardSchema(**obj).dict()
    except ValidationError: return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

def refine_script_with_roles(storyboard, form_data, char_faces):
    segments = []
    for scene in storyboard.get("scenes", []):
        text = scene.get("audio_narration", "")
        char_list = scene.get('characters_in_scene', [])
        char_name = char_list[0] if char_list else "Unknown"
        assigned_voice = MALE_VOICE_ID 
        if "female" in str(char_faces.get(char_name, {})).lower() or "Sana" in char_name:
             assigned_voice = FEMALE_VOICE_ID
        segments.append({"text": text, "voice_id": assigned_voice})
    return segments

# --- UPDATED: GENERATE THUMBNAIL (No skipping) ---
def generate_thumbnail_agent(storyboard: dict, aspect: str = "16:9") -> Optional[str]:
    # Try to load the high-quality prompt file
    prompt_template = load_prompt_template("prompt_image_synthesizer.txt")
    
    if prompt_template:
        summary = storyboard.get("video_description") or storyboard.get("video_title") or "Cinematic Scene"
        chars_desc = ", ".join([c.get('name','') + ": " + c.get('appearance_prompt','') for c in storyboard.get('characters', [])])
        
        template = Template(prompt_template)
        # Use safe substitution to avoid crashes
        prompt = template.safe_substitute(
            article_summary=summary,
            characters=chars_desc,
            orientation="Vertical 9:16" if aspect == "9:16" else "Cinematic 16:9"
        )
    else:
        # Fallback if file is missing
        summary = storyboard.get("video_description") or "Video"
        prompt = f"Cinematic movie poster for {summary}, detailed, 8k, dramatic lighting."

    try: 
        return generate_flux_image(prompt, aspect=aspect)
    except: 
        return None

# --- UPDATED: YOUTUBE METADATA (No skipping) ---
def youtube_metadata_agent(script, keyword, form_data, blueprint):
    prompt = load_prompt_template("prompt_youtube_metadata_generator.txt")
    if not prompt: return {}
    template = Template(prompt)
    
    # Ensure script isn't None
    safe_script = script[:3000] if script else "No script provided."
    
    raw = get_openai_response(template.safe_substitute(
        primary_keyword=keyword, 
        full_script=safe_script, 
        language=form_data.get("language", "english"), 
        video_type=form_data.get("video_type", "reel"), 
        blueprint_data=json.dumps(blueprint), 
        thumbnail_concept=""
    ), 0.4, True)
    return extract_json_from_text(raw) or {}

# -------------------------
# Celery Task
# -------------------------
@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    if not HEYGEN_AVAILABLE: return {"status": "error", "message": "HeyGen unavailable"}
    try:
        def update_status(msg): self.update_state(state="PROGRESS", meta={"message": msg}); logging.info(msg)
        
        keyword = form_data.get("keyword")
        update_status("Designing Video Concept...")
        storyboard = create_video_storyboard_agent(keyword, {}, form_data)
        scenes = storyboard.get("scenes", [])
        
        update_status("Casting Characters...")
        characters = storyboard.get("characters") or []
        character_faces = {}
        
        real_fallback_id = SAFE_FALLBACK_AVATAR_ID
        if get_all_avatars:
            try:
                avs = get_all_avatars()
                if avs: real_fallback_id = avs[0].get("avatar_id")
            except: pass

        for char in characters:
             name = char.get("name", "Unknown")
             prompt = char.get("appearance_prompt", "").lower()
             char_data = ensure_character(name, appearance_prompt=prompt)
             selected_id = CAST_LIST["NARRATOR"]
             if "man" in prompt or "detective" in prompt: selected_id = CAST_LIST.get("MALE_LEAD", real_fallback_id)
             elif "woman" in prompt or "female" in prompt: selected_id = CAST_LIST.get("FEMALE_LEAD", real_fallback_id)
             char_data["heygen_avatar_id"] = selected_id
             character_faces[name] = char_data
        
        update_status("Synthesizing Assets...")
        segments = refine_script_with_roles(storyboard, form_data, character_faces)
        scene_assets = []
        full_script = ""
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"

        for i, scene in enumerate(scenes):
            is_b_roll = scene.get("is_b_roll", False)
            text = segments[i].get("text")
            voice_id = segments[i].get("voice_id")
            if not is_b_roll: full_script += text + " "
            
            audio_path = None
            if not is_b_roll:
                try: 
                    if generate_audio_for_scene: audio_path = generate_audio_for_scene(text, voice_id).get("path")
                except: pass
                if not audio_path and generate_openai_speech:
                    voice_preset = "onyx" if voice_id == MALE_VOICE_ID else "nova"
                    audio_path = generate_openai_speech(text, voice_category=voice_preset)
            
            bg_url = generate_flux_image(scene.get("visual_prompt", "Background"), aspect=aspect)
            scene_assets.append({"index": i, "audio_path": audio_path, "scene_data": scene, "background_url": bg_url})

        update_status("Rendering Video Scenes...")
        final_pairs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(process_single_scene, asset["scene_data"], asset["index"], "", asset["audio_path"], character_faces, aspect, real_fallback_id, asset["background_url"]): asset for asset in scene_assets}
            results = {}
            for f in concurrent.futures.as_completed(futures):
                idx = futures[f]["index"]
                try: 
                    res = f.result()
                    if res["status"] == "success": results[idx] = (res["video_path"], futures[f]["audio_path"])
                except: pass
            
            for i in range(len(scenes)):
                if i in results: final_pairs.append(results[i])

        if not final_pairs: raise RuntimeError("Zero scenes generated.")

        update_status("Final Assembly...")
        desc_text = (storyboard.get("video_title", "") + " " + storyboard.get("video_description", "")).lower()
        detected_tone = "motivational"
        if any(x in desc_text for x in ["sad", "crying", "lost", "rain", "sorrow"]): detected_tone = "sad"
        elif any(x in desc_text for x in ["detective", "noir", "mystery", "crime", "dark"]): detected_tone = "intense"
        elif any(x in desc_text for x in ["peace", "calm", "nature"]): detected_tone = "ambient"
        elif any(x in desc_text for x in ["happy", "joy", "fun"]): detected_tone = "happy"
            
        logging.info(f"🎵 Smart Music Detector: Selected '{detected_tone}' track.")
        music_url = MUSIC_LIBRARY.get(detected_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try: download_to_file(music_url, music_path)
        except: music_path = None
        
        final_output = f"/tmp/final_{getattr(self.request, 'id', 'unknown')}.mp4"
        stitch_video_audio_pairs_optimized(final_pairs, final_output)
        
        if music_path and os.path.exists(music_path):
             mixed_out = final_output.replace(".mp4", "_mixed.mp4")
             subprocess.run([
                 "ffmpeg", "-y", "-i", final_output, "-stream_loop", "-1", "-i", music_path, 
                 "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first[aout]", 
                 "-map", "0:v", "-map", "[aout]", 
                 "-c:v", "copy", "-c:a", "aac", "-shortest", mixed_out
             ], check=True)
             final_output = mixed_out
        
        # --- 7. UPLOAD & FINISH ---
        update_status("Uploading & Finalizing...")
        final_video_url = safe_upload_to_cloudinary(final_output, folder="final_videos")
        
        # Call the FULLY IMPLEMENTED Agents
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        meta = youtube_metadata_agent(full_script, keyword, form_data, {})
        
        return {"status": "ready", "video_url": final_video_url, "thumbnail_url": thumbnail_url, "metadata": meta, "storyboard": storyboard}

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return {"status": "error", "message": str(e)}
