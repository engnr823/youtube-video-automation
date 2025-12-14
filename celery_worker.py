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

# --- REPLICATE REMOVAL (Kept primarily for compatibility) ---
replicate = None 
replicate_client = None 

from openai import OpenAI

# --- CONSTANTS FOR VOICES AND AVATARS ---
MALE_VOICE_ID = "ErXwobaYiN019PkySvjV" 
FEMALE_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

# Safe "Lite" Avatar (Josh) that works on almost all plans to avoid 404s
SAFE_FALLBACK_AVATAR_ID = "josh_lite3_20230714"

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
    from utils.ffmpeg_utils import get_media_duration
except Exception:
    def get_media_duration(file_path):
        # Fallback implementation
        try:
            if not os.path.exists(file_path): return 0.0
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            val = result.stdout.strip()
            return float(val) if val else 0.0
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {e}")
            return 0.0

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
    # Use local fallback if import fails
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
# Royalty-Free Music Library
# -------------------------
MUSIC_LIBRARY = {
    "motivational": "https://cdn.pixabay.com/audio/2022/07/22/powerful-8526.mp3",
    "intense": "https://cdn.pixabay.com/audio/2023/12/06/trailer-mood-176840.mp3",
    "sad": "https://cdn.pixabay.com/audio/2022/09/20/emotional-128225.mp3",
    "ambient": "https://cdn.pixabay.com/audio/2022/07/26/cinematic-ambient-11634.mp3",
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
    camera_angle: Optional[str] = "35mm"
    lighting: Optional[str] = "soft cinematic"
    emotion: Optional[str] = "neutral"
    characters_in_scene: Optional[List[str]] = []
    image_url: Optional[str] = None
    video_url: Optional[str] = None

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
# Image generation (UPGRADED: DALL-E 3)
# -------------------------
def generate_flux_image(prompt: str, aspect: str = "16:9", negative_prompt: str = "") -> str:
    """
    Generates a background image using DALL-E 3 or falls back to a placeholder.
    This creates the 'Scene' for the avatar to stand in.
    """
    if openai_client:
        try:
            logging.info(f"🎨 Generating background with DALL-E 3: {prompt[:30]}...")
            # Map aspect ratio to DALL-E supported sizes
            size = "1024x1792" if aspect == "9:16" else "1024x1024"
            
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Cinematic movie scene, {prompt}, photorealistic, 8k, highly detailed background, no text, atmospheric lighting",
                size=size,
                quality="standard",
                n=1,
            )
            image_url = response.data[0].url
            logging.info("✅ Background generated successfully.")
            return image_url
        except Exception as e:
            logging.warning(f"DALL-E generation failed: {e}")

    logging.warning("⚠️ Using placeholder image.")
    if 'old man' in prompt.lower() or 'mentor' in prompt.lower():
        return "https://placeimg.com/480/832/people" 
    return "https://placeimg.com/480/832/abstract"


# -------------------------
# Single scene processor (FIXED: BACKGROUNDS + NO PREMATURE DELETE)
# -------------------------
def process_single_scene(
    scene: dict,
    index: int,
    character_profile: str,
    audio_path: str = None,
    character_faces: dict = {},
    aspect: str = "9:16",
    fallback_avatar_id: str = None,
    background_url: str = None # NEW: Receive Generated Background
) -> dict:
    
    if not HEYGEN_AVAILABLE:
        raise RuntimeError("HeyGen client not available. Cannot generate video scene.")

    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("/tmp", f"scene_{index}_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # 1. Identify Character/Avatar ID for this scene
        character_list = scene.get('characters_in_scene')
        target_char_name = character_list[0] if character_list and len(character_list) > 0 else None
        
        if not target_char_name:
            target_char_name = list(character_faces.keys())[0] if character_faces else "MENTOR"
            
        char_data = character_faces.get(target_char_name, {})
        avatar_id = char_data.get("heygen_avatar_id") 
        
        # --- Fallback Logic ---
        if not avatar_id:
            logging.warning(f"No specific HeyGen Avatar ID found for {target_char_name}. Using dynamic fallback.")
            avatar_id = fallback_avatar_id

        if not avatar_id:
             avatar_id = SAFE_FALLBACK_AVATAR_ID

        # 2. Upload Audio
        cloud_audio_url = None
        if audio_path and os.path.exists(audio_path):
            cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
            
        if not cloud_audio_url:
             logging.warning(f"Scene {index} skipped as required audio/avatar is missing.")
             return {"index": index, "video_path": None, "status": "skipped"}

        # 3. Call the single HeyGen API function
        # PASS THE BACKGROUND URL HERE to fix "No Scenes"
        logging.info(f"Calling HeyGen for Scene {index} (Avatar: {avatar_id}, BG: {'Yes' if background_url else 'No'})")
        
        video_url = generate_heygen_video(
            avatar_id=avatar_id,
            audio_url=cloud_audio_url,
            aspect_ratio=aspect, 
            background_image_url=background_url # Pass the DALL-E image
        )

        if not video_url:
            raise RuntimeError("Video generation failed via HeyGen")

        # 4. Download and return
        temp_video_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
        download_to_file(str(video_url), temp_video_path, timeout=300)

        # CRITICAL: Verify file exists
        if not os.path.exists(temp_video_path):
             logging.error(f"Scene {index} reported success but file is missing at {temp_video_path}")
             return {"index": index, "video_path": None, "status": "failed"}

        # CRITICAL: DO NOT DELETE TEMP DIR HERE. It is needed for stitching.
        
        return {"index": index, "video_path": temp_video_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {traceback.format_exc()}")
        # Cleanup only on failure
        if os.path.exists(temp_dir): 
             try: shutil.rmtree(temp_dir)
             except: pass
        return {"index": index, "video_path": None, "status": "failed"}

# -------------------------
# Stitching (FIXED: ROBUST FFMPEG FILTER)
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
            if not os.path.exists(video):
                logging.error(f"Stitching Input Missing: {video}")
                continue

            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            audio_dur = get_media_duration(audio)
            
            # Use 5 seconds if audio duration fails (fallback)
            if audio_dur <= 0: audio_dur = 5.0
            
            # --- CRITICAL FIX: STANDARDIZE RESOLUTION & PIXEL FORMAT ---
            # This prevents the "Exit Status 254" caused by varying inputs.
            # We standardize to 1080x1920 (9:16) which matches the mobile reel target.
            
            cmd = [
                "ffmpeg", "-y", 
                "-stream_loop", "-1", "-i", video, 
                "-i", audio,
                "-t", str(audio_dur),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "ultrafast",
                # Complex filter to Scale, Pad, and Force FPS
                "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,fps=30",
                "-c:a", "aac",
                "-shortest",
                chunk_name
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                if os.path.exists(chunk_name):
                    chunk_paths.append(chunk_name)
                else:
                    logging.error(f"FFmpeg chunk {i} failed to create output file.")
            except subprocess.CalledProcessError as e:
                # Log detailed error from FFmpeg stderr
                logging.error(f"FFmpeg chunk {i} failed: {e.stderr.decode() if e.stderr else 'Unknown Error'}")
                continue

        if not chunk_paths: 
            logging.error("No chunks were successfully created for stitching.")
            return False

        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        # Final Concat
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
# AGENTS (Kept)
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
    prompt += f"\n\nIMPORTANT: The 'audio_narration' across all scenes MUST total at least 150 words (60+ seconds total length). Generate exactly {target_scenes} scenes. For dialogue, explicitly start the narration with the character name followed by a colon (e.g., KABIR: Hello. ZARA: Hi.). ALSO, include a 'duration_seconds' of 6 to 10 seconds for EACH scene object."
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or (json.loads(raw) if raw else {})
    if not obj: raise RuntimeError("Storyboard generation failed")
    try: return StoryboardSchema(**obj).dict()
    except ValidationError: return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

def refine_script_with_roles(storyboard: dict, form_data: dict, char_faces: dict) -> List[dict]:
    """
    Refines script and assigns correct MALE/FEMALE voices.
    """
    characters = storyboard.get('characters', [])
    full_script_text = " ".join([s.get('audio_narration', '') for s in storyboard.get('scenes', [])])
    
    # Simple list of segments
    segments = []
    
    for scene in storyboard.get("scenes", []):
        text = scene.get("audio_narration", "")
        if not text: continue
        
        # Determine character in this scene
        char_list = scene.get('characters_in_scene', [])
        char_name = char_list[0] if char_list else "Unknown"
        char_data = char_faces.get(char_name, {})
        
        # Assign Voice ID
        assigned_voice = MALE_VOICE_ID # Default to Male
        
        if "female" in str(char_data) or "woman" in str(char_data):
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
        # Return clean error dict to prevent Worker crash loop
        return {"status": "error", "message": "HeyGen client not available"}

    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(msg)
        
        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")
        
        # --- 1. DYNAMIC AVATAR FETCH ---
        update_status("Fetching Available Avatars...")
        available_avatars = []
        try:
            if get_all_avatars:
                available_avatars = get_all_avatars()
                logging.info(f"✅ Found {len(available_avatars)} avatars in account.")
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
        
        # Casting Logic: Round-Robin allocation for different faces
        for i, char in enumerate(characters):
             name = char.get("name", "Unknown")
             char_data = ensure_character(name, appearance_prompt=char.get("appearance_prompt"))
             ref_image = next((url for url in uploaded_images if name.lower() in url.lower()), None)
             
             # Pick different avatar for each character
             if available_avatars:
                 selected = available_avatars[i % len(available_avatars)]
                 char_data["heygen_avatar_id"] = selected.get("avatar_id")
             else:
                 char_data["heygen_avatar_id"] = SAFE_FALLBACK_AVATAR_ID
             
             char_data["reference_image"] = ref_image 
             character_faces[name] = char_data
        
        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # --- 4. PREPARE ASSETS (AUDIO + BG) ---
        update_status("Synthesizing Audio & Backgrounds...")
        segments = refine_script_with_roles(storyboard, form_data, character_faces)
        scene_assets = []
        full_script_text = ""
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"

        for i, scene in enumerate(scenes):
            text = segments[i].get("text") if i < len(segments) else "..."
            voice_id = segments[i].get("voice_id") if i < len(segments) else MALE_VOICE_ID
            full_script_text += text + " "
            
            # A. Generate Audio
            audio_path = None
            if generate_audio_for_scene:
                try: 
                    audio_res = generate_audio_for_scene(text, voice_id)
                    audio_path = audio_res.get("path") if audio_res else None
                except Exception as e:
                    logging.error(f"Audio error scene {i}: {e}")
            
            # B. Generate Background
            visual_prompt = scene.get("visual_prompt", "Background")
            bg_url = generate_flux_image(visual_prompt, aspect=aspect)

            if audio_path:
                scene_assets.append({
                    "index": i, 
                    "audio_path": audio_path, 
                    "duration": get_media_duration(audio_path),
                    "scene_data": scene,
                    "background_url": bg_url
                })

        if not scene_assets: raise RuntimeError("Assets generation failed.")

        # --- 5. VIDEO RENDERING ---
        update_status("Rendering Video Scenes (HeyGen)...")
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
                    SAFE_FALLBACK_AVATAR_ID,
                    asset["background_url"] # Pass generated BG
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
                    results_map[idx] = (res["video_path"], asset["audio_path"])
                else: 
                    logging.warning(f"Scene {idx} failed.")

            for i in range(len(scenes)):
                if i in results_map: final_pairs.append(results_map[i])

        if not final_pairs: raise RuntimeError("Zero video scenes generated.")

        # --- 6. STITCHING ---
        update_status("Final Assembly...")
        music_tone = blueprint.get("tone", "motivational")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try: download_to_file(music_url, music_path)
        except: music_path = None
        
        final_output_path = f"/tmp/final_render_{task_id}.mp4"
        temp_visual_path = f"/tmp/visual_base_{task_id}.mp4"
        
        success = stitch_video_audio_pairs_optimized(final_pairs, temp_visual_path)
        
        if not success: 
            return {"status": "error", "message": "Stitching failed during final assembly."}
        
        # Audio Mix
        if music_path and os.path.exists(music_path):
            cmd = [
                "ffmpeg", "-y", "-i", temp_visual_path, "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", 
                "[0:a]loudnorm=I=-14:LRA=7:tp=-2,volume=0.9[dia];[1:a]loudnorm=I=-22:LRA=7:tp=-2[bg];[dia][bg]amix=inputs=2:duration=first:weights=1 1[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "libx264", "-vf", "scale='min(720,iw)':-2", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-shortest", final_output_path
            ]
            subprocess.run(cmd, check=True)
        else: shutil.move(temp_visual_path, final_output_path)
        
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
