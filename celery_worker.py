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
# The new Smart HeyGen Client will detect these are Talking Photos.
CAST_LIST = {
    "MALE_LEAD": "4343bfb447bf4028a48b598ae297f5dc",    # Your Custom Male
    "FEMALE_LEAD": "16a811adf1cc4b12bc6edd04c8fecffa",  # Your Custom Female
    "NARRATOR": "4343bfb447bf4028a48b598ae297f5dc"      # Default Narrator
}

# Initial Fallback
SAFE_FALLBACK_AVATAR_ID = "4343bfb447bf4028a48b598ae297f5dc"

# --- Utility Fix: Define missing ensure_dir function ---
def ensure_dir(path):
    """Ensure that the given directory path exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
# --------------------------------------------------------

# --- VOICE SETTINGS SAFE IMPORT ---
try:
    from elevenlabs import VoiceSettings
except Exception:
    VoiceSettings = None
    logging.warning("⚠️ ElevenLabs VoiceSettings not found. Using default voice stability.")

# --- UTILS IMPORT SAFETY BLOCK ---
try:
    from utils.ffmpeg_utils import get_media_duration, composite_green_screen_scene
except Exception:
    def get_media_duration(file_path): return 0.0
    def composite_green_screen_scene(bg, fg, out): return False

# --- CLIENT IMPORT SAFETY BLOCK (ELEVENLABS) ---
try:
    from video_clients.elevenlabs_client import (
        generate_audio_for_scene
    )
except Exception:
    logging.warning("⚠️ ElevenLabs client not found. Voiceover generation will fail.")
    generate_audio_for_scene = None

# --- CRITICAL FIX: ROBUST HEYGEN CLIENT IMPORT ---
try:
    from video_clients.heygen_client import (
        generate_heygen_video, 
        get_all_avatars, 
        get_safe_fallback_id, 
        HeyGenError
    )
    HEYGEN_AVAILABLE = True
except ImportError as e:
    logging.error(f"❌ HEYGEN CLIENT NOT FOUND. Video generation will fail. IMPORT ERROR: {e}")
    generate_heygen_video = None
    get_all_avatars = None
    get_safe_fallback_id = lambda: SAFE_FALLBACK_AVATAR_ID
    HEYGEN_AVAILABLE = False


# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

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
    "intense": "https://cdn.pixabay.com/audio/2023/12/06/trailer-mood-176840.mp3", # Perfect for Noir
    "sad": "https://cdn.pixabay.com/audio/2022/09/20/emotional-128225.mp3",        # Perfect for Rain/Drama
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
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

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
    """
    Generates a background image using Flux-Schnell (Replicate) for low cost.
    """
    try:
        logging.info(f"🎨 Generating background with Flux (Replicate): {prompt[:30]}...")
        
        # Ensure aspect ratio is supported by Flux (16:9, 9:16, 1:1, etc.)
        if aspect not in ["16:9", "9:16", "1:1", "4:5", "2:3", "3:2"]:
            aspect = "16:9" # fallback

        # Call Flux-Schnell (Very fast & cheap)
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt + ", cinematic lighting, 8k, hyper-detailed, photorealistic, no text, bokeh background",
                "aspect_ratio": aspect,
                "output_format": "jpg",
                "output_quality": 90
            }
        )
        
        # Replicate usually returns a list of output URLs/objects
        if output:
            image_url = str(output[0])
            logging.info("✅ Background generated successfully (Flux).")
            return image_url
            
    except Exception as e:
        logging.error(f"Flux generation failed: {e}")

    # Fallback
    logging.warning("⚠️ Using placeholder image.")
    return "https://placeimg.com/480/832/abstract"

# -------------------------
# B-Roll Video Creator (Local FFmpeg)
# -------------------------
def create_static_video_from_image(image_path: str, duration: float, output_path: str) -> bool:
    """Creates a static video from an image to serve as B-Roll (Saving HeyGen credits)."""
    try:
        cmd = [
            "ffmpeg", "-y", "-loop", "1", "-i", image_path,
            "-t", str(duration),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
            "-preset", "ultrafast",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(output_path)
    except Exception as e:
        logging.error(f"Failed to create B-Roll video: {e}")
        return False

# -------------------------
# Single scene processor (UPDATED: HYBRID & COMPOSITING)
# -------------------------
def process_single_scene(
    scene: dict,
    index: int,
    character_profile: str,
    audio_path: str = None,
    character_faces: dict = {},
    aspect: str = "9:16",
    fallback_avatar_id: str = None,
    background_url: str = None 
) -> dict:
    
    if not HEYGEN_AVAILABLE:
        raise RuntimeError("HeyGen client not available. Cannot generate video scene.")

    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("/tmp", f"scene_{index}_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # --- 1. BACKGROUND PREP ---
        bg_image_local_path = os.path.join(temp_dir, f"bg_{index}.jpg")
        if background_url:
            try:
                download_to_file(background_url, bg_image_local_path)
            except Exception as e:
                logging.warning(f"Failed to download background for scene {index}: {e}")
        else:
            logging.warning(f"No background URL provided for scene {index}")

        # --- 2. CHECK FOR B-ROLL (COST SAVING MODE) ---
        is_b_roll = scene.get("is_b_roll", False)
        
        if is_b_roll:
            logging.info(f"💰 Scene {index} is B-Roll. Skipping HeyGen to save credits.")
            final_scene_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
            duration = scene.get("duration_seconds", 5.0)
            
            # Create simple video from Flux Background
            if os.path.exists(bg_image_local_path):
                create_static_video_from_image(bg_image_local_path, duration, final_scene_path)
                return {"index": index, "video_path": final_scene_path, "status": "success"}
            else:
                 logging.error(f"B-Roll scene {index} failed: No background image.")
                 return {"index": index, "video_path": None, "status": "failed"}

        # --- 3. DIALOGUE SCENE (HEYGEN) ---
        
        # Identify Character/Avatar ID for this scene
        character_list = scene.get('characters_in_scene')
        target_char_name = character_list[0] if character_list and len(character_list) > 0 else None
        
        if not target_char_name:
            target_char_name = list(character_faces.keys())[0] if character_faces else "MENTOR"
            
        char_data = character_faces.get(target_char_name, {})
        # FORCE fallback to default if ID is missing or known invalid
        avatar_id = char_data.get("heygen_avatar_id") or fallback_avatar_id
        
        # Upload Audio
        cloud_audio_url = None
        if audio_path and os.path.exists(audio_path):
            cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
            
        if not cloud_audio_url:
             logging.warning(f"Scene {index} skipped as required audio is missing.")
             return {"index": index, "video_path": None, "status": "skipped"}

        logging.info(f"Calling HeyGen for Green Screen Scene {index} (Avatar: {avatar_id})")
        
        video_url = generate_heygen_video(
            avatar_id=avatar_id,
            audio_url=cloud_audio_url,
            aspect_ratio=aspect, 
            background_image_url=None, 
            use_green_screen=True      
        )

        if not video_url:
            raise RuntimeError("Video generation failed via HeyGen")

        # Download Green Screen Video
        green_screen_path = os.path.join(temp_dir, f"green_raw_{index}.mp4")
        download_to_file(str(video_url), green_screen_path, timeout=300)

        # COMPOSITE (Flux BG + Green Screen Actor)
        final_scene_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
        
        composite_success = False
        if os.path.exists(bg_image_local_path) and os.path.exists(green_screen_path):
            logging.info(f"Compositing scene {index} with camera movement...")
            try:
                composite_success = composite_green_screen_scene(
                    bg_image_local_path, 
                    green_screen_path, 
                    final_scene_path
                )
            except Exception as e:
                logging.error(f"Composite crashed scene {index}: {e}")

        # Fallback
        if not composite_success:
            logging.warning(f"Composite failed for scene {index}, using raw video.")
            if os.path.exists(green_screen_path):
                shutil.copy(green_screen_path, final_scene_path)
        
        if not os.path.exists(final_scene_path):
             return {"index": index, "video_path": None, "status": "failed"}

        return {"index": index, "video_path": final_scene_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {traceback.format_exc()}")
        if os.path.exists(temp_dir): 
             try: shutil.rmtree(temp_dir)
             except: pass
        return {"index": index, "video_path": None, "status": "failed"}

# -------------------------
# Stitching
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
            
            # Calculate duration - if B-Roll (no audio), use video duration
            audio_dur = get_media_duration(audio) if audio and os.path.exists(audio) else 0.0
            video_dur = get_media_duration(video)
            
            # Use audio duration if available, else video duration
            final_dur = audio_dur if audio_dur > 0.5 else video_dur
            if final_dur <= 0: final_dur = 5.0
            
            cmd = [
                "ffmpeg", "-y", 
                "-stream_loop", "-1", "-i", video, 
            ]
            
            # Add audio input if exists
            if audio and os.path.exists(audio):
                 cmd.extend(["-i", audio, "-map", "0:v", "-map", "1:a"])
            else:
                 # Generate silent audio for B-Roll consistency
                 cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-map", "0:v", "-map", "1:a"])

            cmd.extend([
                "-t", str(final_dur),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
                "-c:a", "aac",
                "-shortest",
                chunk_name
            ])
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths: return False

        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)
        
        return os.path.exists(output_path)

    except Exception as e:
        logging.error(f"Stitching critical error: {e}")
        return False
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)


# -------------------------
# AGENTS
# -------------------------

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        system_content = "You are a professional screenwriter."
        if is_json: system_content += " You must output valid JSON."
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[{"role":"system","content": system_content},{"role":"user","content":prompt_content}],
            temperature=temperature,
            response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return ""

def create_video_storyboard_agent(keyword: str, blueprint: dict, form_data: dict) -> dict:
    prompt_template = load_prompt_template("prompt_video_storyboard_creator.txt")
    if not prompt_template:
        prompt_template = "TASK: Create script for '$keyword'. Output valid JSON."
    template = Template(prompt_template)
    full_context = keyword
    if form_data.get("characters"): full_context += f"\n\nUSER DEFINED CHARACTERS:\n{form_data.get('characters')}"
    target_scenes = form_data.get("max_scenes", 7)
    prompt = template.safe_substitute(
        keyword=full_context,
        blueprint_json=json.dumps(blueprint),
        language=form_data.get("language", "english"),
        video_type=form_data.get("video_type", "reel"),
        uploaded_assets_context="User uploaded images present" if form_data.get("uploaded_images") else "No uploads",
        max_scenes=str(target_scenes)
    )
    # Note: Prompt logic is now handled by the 9-Line Protocol file
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or (json.loads(raw) if raw else {})
    if not obj: raise RuntimeError("Storyboard generation failed")
    try: return StoryboardSchema(**obj).dict()
    except ValidationError: return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

def refine_script_with_roles(storyboard: dict, form_data: dict, char_faces: dict) -> List[dict]:
    characters = storyboard.get('characters', [])
    segments = []
    
    for scene in storyboard.get("scenes", []):
        text = scene.get("audio_narration", "")
        # Note: B-Roll might have empty text or [SFX], handled downstream
        
        char_list = scene.get('characters_in_scene', [])
        char_name = char_list[0] if char_list else "Unknown"
        char_data = char_faces.get(char_name, {})
        
        # Determine Voice Gender for Routing
        assigned_voice = MALE_VOICE_ID 
        if "female" in str(char_data) or "woman" in str(char_data) or "girl" in str(char_data) or "Sana" in char_name:
             assigned_voice = FEMALE_VOICE_ID
             
        segments.append({"text": text, "voice_id": assigned_voice})
        
    return segments

def generate_thumbnail_agent(storyboard: dict, orientation: str = "16:9") -> Optional[str]:
    summary = storyboard.get("video_description") or "Video"
    video_title = storyboard.get("video_title") or "New Video"
    
    cinematic_prompt = (
        f"Movie poster style for channel The Apex Archive, titled '{video_title}'. "
        f"Epic cinematic shot: {summary}, deep shadows, volumetric lighting, "
        f"bold typography placeholder, 8k, hyper-detailed, dramatic focus."
    )
    negative_prompt = "blurry, poor detail, cartoon, low contrast, text, signature, low resolution"
    
    try: 
        return generate_flux_image(cinematic_prompt, aspect=orientation, negative_prompt=negative_prompt)
    except: 
        return None

def youtube_metadata_agent(full_script: str, keyword: str, form_data: dict, blueprint: dict) -> dict:
    prompt_template = load_prompt_template("prompt_youtube_metadata_generator.txt")
    if not prompt_template: return {}
    template = Template(prompt_template)
    prompt = template.safe_substitute(
        primary_keyword=keyword,
        full_script=full_script[:3000],
        language=form_data.get("language", "english"),
        video_type=form_data.get("video_type", "reel"),
        blueprint_data=json.dumps(blueprint),
        thumbnail_concept=form_data.get("thumbnail_concept", "")
    )
    raw = get_openai_response(prompt, temperature=0.4, is_json=True)
    return extract_json_from_text(raw) or (json.loads(raw) if raw else {})

# -------------------------
# Celery Task (MODIFIED)
# -------------------------
@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] SaaS Task started.")
    if not HEYGEN_AVAILABLE:
        logging.error("Task aborted: HeyGen client is not configured.")
        return {"status": "error", "message": "HeyGen client not available"}

    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(msg)
        
        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")
        
        # --- 1. DYNAMIC AVATAR FETCH & FALLBACK SETUP ---
        update_status("Fetching Available Avatars...")
        real_fallback_id = SAFE_FALLBACK_AVATAR_ID
        
        try:
            if get_all_avatars:
                available_avatars = get_all_avatars()
                logging.info(f"✅ Found {len(available_avatars)} avatars in account.")
                
                # If we have avatars, pick the first one as a TRUE safety net if default is dead
                if available_avatars:
                    first_avail = available_avatars[0].get("avatar_id") or available_avatars[0].get("id")
                    if first_avail:
                        real_fallback_id = first_avail
                        logging.info(f"🛡️ System Fallback set to: {real_fallback_id}")
        except Exception as e:
            logging.error(f"Avatar fetch error: {e}.")

        # --- 2. SCRIPTING ---
        update_status("Designing Video Concept...")
        scraped = {} 
        blueprint = {} 
        
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)
        scenes = storyboard.get("scenes", [])
        if not scenes: raise RuntimeError("Failed to generate scenes.")

        # --- 3. CASTING & BACKGROUNDS ---
        update_status("Casting Characters & Scenes...")
        characters = storyboard.get("characters") or []
        uploaded_images = form_data.get("uploaded_images") or []
        character_faces = {}
        
        # --- FIXED CASTING LOGIC (HYBRID ENGINE) ---
        for i, char in enumerate(characters):
             name = char.get("name", "Unknown")
             prompt = char.get("appearance_prompt", "").lower()
             
             char_data = ensure_character(name, appearance_prompt=char.get("appearance_prompt"))
             
             # Default to Narrator
             selected_id = CAST_LIST["NARRATOR"]
            
             # Detect Gender/Role (Using User Configurable Map)
             if "man" in prompt or "detective" in prompt or "male" in prompt:
                # IMPORTANT: Replace CAST_LIST IDs with valid ones if not set
                selected_id = CAST_LIST.get("MALE_LEAD", real_fallback_id)
             elif "woman" in prompt or "female" in prompt or "girl" in prompt:
                selected_id = CAST_LIST.get("FEMALE_LEAD", real_fallback_id)
             
             # Save the assignment
             char_data["heygen_avatar_id"] = selected_id
             character_faces[name] = char_data
             logging.info(f"🎭 Cast {name} as ID: {selected_id}")
        
        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # --- 4. PREPARE ASSETS (AUDIO + BG) ---
        update_status("Synthesizing Audio & Backgrounds...")
        segments = refine_script_with_roles(storyboard, form_data, character_faces)
        scene_assets = []
        full_script_text = ""
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"

        for i, scene in enumerate(scenes):
            # Check B-Roll status
            is_b_roll = scene.get("is_b_roll", False)
            
            text = segments[i].get("text") if i < len(segments) else "..."
            voice_id = segments[i].get("voice_id") if i < len(segments) else MALE_VOICE_ID
            if not is_b_roll:
                full_script_text += text + " "
            
            # --- AUDIO GENERATION (WITH OPENAI FALLBACK) ---
            audio_path = None
            
            # If B-Roll, we skip speech generation (unless there's specific SFX logic later)
            if not is_b_roll:
                # Try ElevenLabs first
                try:
                    if generate_audio_for_scene:
                        audio_res = generate_audio_for_scene(text, voice_id)
                        audio_path = audio_res.get("path") if audio_res else None
                except Exception as e:
                    logging.warning(f"ElevenLabs error scene {i}: {e}")

                # Fallback to OpenAI TTS if ElevenLabs failed or returned nothing
                if not audio_path:
                    logging.info(f"⚠️ Switching to OpenAI TTS for scene {i} due to ElevenLabs failure/quota.")
                    try:
                        if generate_openai_speech:
                            # --- SMART GENDER MAPPING FOR OPENAI ---
                            # Maps the HeyGen voice ID logic to OpenAI Voices
                            voice_preset = "alloy" # default
                            if voice_id == MALE_VOICE_ID:
                                voice_preset = "onyx" # Gritty Male
                            elif voice_id == FEMALE_VOICE_ID:
                                voice_preset = "nova" # Soft Female
                            
                            audio_path = generate_openai_speech(text, voice_category=voice_preset)
                        else:
                            logging.error("OpenAI TTS client not available for fallback.")
                    except Exception as e:
                        logging.error(f"OpenAI TTS fallback failed: {e}")
            
            # B. Generate Background
            # Extract visual prompt from 9-Line Block
            visual_prompt = scene.get("visual_prompt", "Background")
            bg_url = generate_flux_image(visual_prompt, aspect=aspect)

            scene_assets.append({
                "index": i, 
                "audio_path": audio_path, 
                "duration": get_media_duration(audio_path) if audio_path else 5.0,
                "scene_data": scene,
                "background_url": bg_url
            })

        if not scene_assets: raise RuntimeError("Assets generation failed.")

        # --- 5. VIDEO RENDERING ---
        update_status("Rendering Video Scenes (Hybrid Engine)...")
        final_pairs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_asset = {
                executor.submit(
                    process_single_scene, 
                    asset["scene_data"], 
                    asset["index"], 
                    char_profile, 
                    asset["audio_path"], 
                    character_faces, 
                    aspect,
                    real_fallback_id, # Use Dynamic Fallback here
                    asset["background_url"] 
                ): asset
                for asset in scene_assets
            }
            results_map = {}
            for future in concurrent.futures.as_completed(future_to_asset):
                asset = future_to_asset[future]
                idx = asset["index"]
                try: res = future.result()
                except Exception as e:
                    logging.error(f"Scene {idx} crash: {e}")
                    res = {"status": "failed"}
                
                if res.get("status") == "success" and res.get("video_path"):
                    results_map[idx] = (res["video_path"], asset.get("audio_path"))
                else: 
                    logging.warning(f"Scene {idx} failed.")

            for i in range(len(scenes)):
                if i in results_map: final_pairs.append(results_map[i])

        if not final_pairs: raise RuntimeError("Zero video scenes generated.")

        # --- 6. STITCHING & MUSIC MIXING ---
        update_status("Final Assembly...")
        
        # Smart Emotion Detection Logic
        desc_text = (storyboard.get("video_title", "") + " " + storyboard.get("video_description", "")).lower()
        detected_tone = "motivational" # default
        
        if any(x in desc_text for x in ["sad", "crying", "lost", "rain", "breakup", "sorrow", "pain"]):
            detected_tone = "sad"
        elif any(x in desc_text for x in ["detective", "noir", "mystery", "crime", "dark", "night", "tension", "secret"]):
            detected_tone = "intense"
        elif any(x in desc_text for x in ["peace", "calm", "nature", "morning", "relax"]):
            detected_tone = "ambient"
        elif any(x in desc_text for x in ["happy", "joy", "fun", "party", "success"]):
            detected_tone = "happy"
            
        logging.info(f"🎵 Smart Music Detector: Selected '{detected_tone}' track based on script.")
        
        music_url = MUSIC_LIBRARY.get(detected_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try: download_to_file(music_url, music_path)
        except: music_path = None
        
        final_output_path = f"/tmp/final_render_{task_id}.mp4"
        temp_visual_path = f"/tmp/visual_base_{task_id}.mp4"
        
        # Optimized stitcher now handles silence for B-Roll clips automatically
        success = stitch_video_audio_pairs_optimized(final_pairs, temp_visual_path)
        
        if not success: 
            return {"status": "error", "message": "Stitching failed during final assembly."}
        
        # Audio Mix (Background Music)
        if music_path and os.path.exists(music_path):
            mixed_output_path = final_output_path.replace(".mp4", "_mixed.mp4")
            cmd = [
                "ffmpeg", "-y", "-i", temp_visual_path, "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", 
                # Mix voice (100%) with music (15%)
                "[0:a]volume=1.0[v];[1:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first",
                "-map", "0:v", "-map", "[outa]", # Note: filter output needs label [outa] or implicit use
                # Adjusted filter complex for correct mapping
                "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", # Copy video stream to save time (stitching already re-encoded if needed)
                "-c:a", "aac", "-shortest", final_output_path
            ]
            # Use a slightly different command to ensure stability with 'copy'
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_visual_path, "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", "[0:a]volume=1.0[v];[1:a]volume=0.15[m];[v][m]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "libx264", "-preset", "ultrafast", # Re-encode fast to ensure audio sync
                "-c:a", "aac", "-shortest", final_output_path
            ], check=True)
        else:
            shutil.move(temp_visual_path, final_output_path)
        
        # --- 7. UPLOAD & FINISH ---
        update_status("Uploading & Finalizing...")
        final_video_url = safe_upload_to_cloudinary(final_output_path, folder="final_videos")
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        metadata = youtube_metadata_agent(full_script_text, keyword, form_data, blueprint)
        
        return {
            "status": "ready",
            "video_url": final_video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard
        }

    except Exception as e:
        error_msg = f"Workflow failed: {str(e)}"
        logging.error(f"Task Exception Caught: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": error_msg})
        return {"status": "error", "message": error_msg}
