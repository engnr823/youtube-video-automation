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
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, wait_fixed
from pydantic import BaseModel, ValidationError

# Celery app import
try:
    from celery_init import celery
except ImportError:
    logging.critical("❌ Could not import 'celery' from 'celery_init'.")
    sys.exit(1)

# AI Clients
from openai import OpenAI
import replicate
from replicate.exceptions import ReplicateError

# --- CLIENT IMPORT SAFETY BLOCK ---
try:
    from video_clients.elevenlabs_client import generate_voiceover_and_upload
except ImportError:
    logging.error("⚠️ ElevenLabs client not found. Audio/Lip-sync will fail.")
    generate_voiceover_and_upload = None

try:
    from video_clients.replicate_client import generate_video_scene_with_replicate
except ImportError:
    logging.warning("⚠️ Standard video generation client not found (okay if using lip-sync only).")
    generate_video_scene_with_replicate = None
# ----------------------------------

# ==========================================
#  🚨 BUDGET SAFETY SWITCH 🚨
#  FALSE = Generates images only (FREE/CHEAP). Good for testing logic.
#  TRUE  = Generates Lip Sync Video (COSTS MONEY).
# ==========================================
USE_REAL_EXPENSIVE_GENERATION = False
# ==========================================


# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Cloudinary Config
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
# Utility Helpers
# -------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def download_to_file(url: str, dest_path: str, timeout: int = 300):
    logging.info(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    return dest_path

def run_subprocess(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, close_fds=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg Error: {e.stderr}")
        raise

def safe_upload_to_cloudinary(filepath: str, resource_type="video", folder="automations"):
    if not os.getenv("CLOUDINARY_API_KEY"): return filepath
    try:
        logging.info(f"Uploading to Cloudinary: {filepath}")
        res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
        return res.get("secure_url")
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        return filepath 

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

# -------------------------
# 1. Storyboard Agent (Prompt Engineering)
# -------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
def get_openai_response(prompt_content: str, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            messages=[{"role":"system","content":"You are a professional screenwriter."},{"role":"user","content":prompt_content}],
            response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise

def create_video_storyboard_agent(keyword: str, form_data: dict) -> dict:
    # We explicitly ask for 'character_name' in every scene to enable multi-character support
    prompt = f"""
    TASK: Create a 7-scene viral short film script for '{keyword}' formatted for a vertical Reel (9:16).
    
    REQUIREMENTS:
    1. Characters: Create 2 distinct characters (e.g., Host/Expert, Hero/Villain).
    2. Scenes: Exactly 7 scenes.
    3. Dialogue: Every scene MUST have spoken dialogue.
    
    OUTPUT JSON FORMAT:
    {{
      "video_title": "Title",
      "characters": [
         {{ "name": "Name1", "appearance_prompt": "Visual description...", "voice_gender": "male" }},
         {{ "name": "Name2", "appearance_prompt": "Visual description...", "voice_gender": "female" }}
      ],
      "scenes": [
         {{
            "scene_id": 1,
            "character_name": "Name1",  <-- CRITICAL: Who is in this scene?
            "visual_prompt": "Description of the setting...",
            "audio_narration": "Exact text to be spoken..."
         }}
      ]
    }}
    """

    try:
        raw = get_openai_response(prompt, is_json=True)
        obj = extract_json_from_text(raw) or json.loads(raw)
        return obj
    except Exception as e:
         raise RuntimeError(f"Storyboard generation failed: {e}")

# -------------------------
# 2. Image & Lip Sync Generators
# -------------------------
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=30), retry=retry_if_exception_type(ReplicateError))
def generate_flux_image_safe(prompt: str, aspect: str = "9:16") -> str:
    """Generates an image using Replicate (Flux Schnell - Fast/Cheap)."""
    if not REPLICATE_API_TOKEN: raise RuntimeError("Replicate Token Missing")
    logging.info(f"🎨 Generating Image...")
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg", "num_inference_steps": 4}
    )
    return str(output[0]) if isinstance(output, (list, tuple)) else str(output)

@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(ReplicateError))
def generate_lip_sync_safe(image_url: str, audio_url: str) -> str:
    """Generates lip-sync video using SadTalker (Budget Friendly)."""
    logging.info(f"👄 Generating Lip Sync Video...")
    output = replicate.run(
        "cjwbw/sadtalker:a519a502c74ac74325776184f17a54342880017f848988a641dd1e88e8945d81",
        input={
            "source_image": image_url, 
            "driven_audio": audio_url, 
            "still": True, # Keeps head stable, better for reels
            "enhancer": "gfpgan" # Face restoration
        }
    )
    return str(output)

# -------------------------
# 3. Scene Processor (Multi-Character Logic)
# -------------------------
def process_single_scene(
    scene: dict, 
    index: int, 
    character_map: Dict[str, dict], 
    default_char_key: str,
    aspect: str = "9:16"
) -> (int, Optional[str]):
    
    try:
        logging.info(f"--- Processing Scene {index+1} ---")

        # A. Resolve Character for this specific scene
        # The storyboard says "character_name": "Hero". We look up "Hero" in our map.
        scene_char_name = scene.get("character_name", default_char_key)
        
        # Fuzzy match character name (e.g. if script says "John" but map has "John Doe")
        matched_key = next((k for k in character_map.keys() if scene_char_name in k or k in scene_char_name), default_char_key)
        
        char_data = character_map.get(matched_key, character_map[default_char_key])
        
        logging.info(f"Scene {index+1} Character: {char_data['name']}")

        # B. Generate Visuals
        visual_setting = scene.get("visual_prompt", "Cinematic background")
        # Combine Character Prompt + Scene Setting
        full_image_prompt = f"A vertical portrait photograph of {char_data['appearance_prompt']}, {visual_setting}, looking directly at camera, 8k, photorealistic."
        
        keyframe_url = generate_flux_image_safe(full_image_prompt, aspect=aspect)

        # --- BUDGET CHECK ---
        if not USE_REAL_EXPENSIVE_GENERATION:
            logging.info(f"Scene {index+1}: [SAFETY MODE] Image generated. Skipping lip-sync.")
            return (index, keyframe_url)
        # --------------------

        # C. Generate Audio (Voice Switching)
        dialogue = scene.get("audio_narration", "").strip()
        voice_id = char_data.get("voice_id")
        
        if not dialogue:
            logging.warning(f"Scene {index+1} has no dialogue. Returning static image.")
            return (index, keyframe_url)

        if not generate_voiceover_and_upload: 
            raise RuntimeError("ElevenLabs client missing")

        logging.info(f"Scene {index+1}: Generating Audio with Voice ID {voice_id}...")
        scene_audio_url = generate_voiceover_and_upload(dialogue, voice_id)
        
        if not scene_audio_url: raise RuntimeError("Audio generation failed")

        # D. Generate Lip Sync
        video_url = generate_lip_sync_safe(keyframe_url, scene_audio_url)
        return (index, video_url)

    except Exception as e:
        logging.error(f"Scene {index+1} Error: {e}")
        return (index, None)

# -------------------------
# 4. Assembly (Robust)
# -------------------------
def concat_videos_robust(input_paths: List[str], output_path: str):
    logging.info(f"Concatenating {len(input_paths)} files...")
    list_file = os.path.join(tempfile.gettempdir(), f"concat_{uuid.uuid4()}.txt")
    
    with open(list_file, "w") as f:
        for path in input_paths:
            f.write(f"file '{path}'\n")
            f.write("duration 0.04\n") # Buffer to prevent glitches

    # FFmpeg concat demuxer (safest for joining clips)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file,
        "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
        output_path
    ]
    try:
        run_subprocess(cmd)
    finally:
        if os.path.exists(list_file): os.remove(list_file)
    return output_path

def add_background_music(video_path: str, output_path: str):
    # Just use a default track for now
    music_url = "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3" 
    music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
    
    try:
        download_to_file(music_url, music_path)
        # Mix audio: Voices 100%, Music 10%
        cmd = [
            "ffmpeg", "-y", "-i", video_path, "-stream_loop", "-1", "-i", music_path,
            "-filter_complex", "[0:a]volume=1.0[a1];[1:a]volume=0.1[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
        ]
        run_subprocess(cmd)
    except:
        shutil.copy(video_path, output_path)
    return output_path

# -------------------------
# Celery Task (Main)
# -------------------------
@celery.task(bind=True, time_limit=1800)
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] Task started.")

    if not OPENAI_API_KEY or not REPLICATE_API_TOKEN: 
        raise ValueError("Missing API Keys")

    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(f"STATUS: {msg}")

        # 1. Generate Storyboard
        update_status("Step 1/4: Writing Script...")
        storyboard = create_video_storyboard_agent(form_data.get("keyword"), form_data)
        scenes = storyboard.get("scenes", [])
        
        # 2. Build Character Map (Name -> {Prompt, VoiceID})
        update_status("Step 2/4: Casting Characters...")
        char_list = storyboard.get("characters", [])
        character_map = {}
        
        # Define some fallback voice IDs (Male/Female)
        # Replace these with your valid ElevenLabs Voice IDs
        VOICE_POOL = {
            "male": "21m00Tcm4TlvDq8ikWAM", # Adam
            "female": "EXAVITQu4vr4xnSDxMaL" # Bella
        }

        for c in char_list:
            name = c.get("name", "Unknown")
            gender = c.get("voice_gender", "male").lower()
            # If user provided a voice in form_data, use it for main char, else pick from pool
            vid = form_data.get("voice_selection") if name == char_list[0].get("name") else VOICE_POOL.get(gender, VOICE_POOL["male"])
            
            character_map[name] = {
                "name": name,
                "appearance_prompt": c.get("appearance_prompt", f"Portrait of {name}"),
                "voice_id": vid
            }
        
        if not character_map:
            # Fallback if AI didn't return characters
            character_map["Narrator"] = {"name": "Narrator", "appearance_prompt": "A presenter", "voice_id": VOICE_POOL["male"]}

        main_char_key = list(character_map.keys())[0] # Default fallback

        # 3. Process Scenes (Parallel-ish but limited to save rate limits)
        update_status("Step 3/4: Generating Scenes (Images + Audio + Lip Sync)...")
        scene_urls = [None] * len(scenes)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_idx = {
                executor.submit(
                    process_single_scene, 
                    scene, i, character_map, main_char_key
                ): i for i, scene in enumerate(scenes)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, url = future.result()
                    if url: scene_urls[idx] = url
                except Exception as e:
                    logging.error(f"Scene {idx} failed: {e}")

        # 4. Assembly
        update_status("Step 4/4: Assembling Reel...")
        valid_urls = [u for u in scene_urls if u]
        if not valid_urls: raise RuntimeError("No scenes generated.")

        # Standardize clips (image to video conversion if needed)
        tmpdir = tempfile.mkdtemp()
        local_paths = []
        for i, url in enumerate(valid_urls):
            ext = "mp4" if url.endswith(".mp4") else "jpg"
            path = os.path.join(tmpdir, f"clip_{i}.{ext}")
            download_to_file(url, path)
            
            out_path = os.path.join(tmpdir, f"norm_{i}.mp4")
            if ext == "jpg":
                # Convert static image to 4s video
                run_subprocess(["ffmpeg", "-y", "-loop", "1", "-i", path, "-t", "4", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path])
            else:
                # Ensure MP4 format
                run_subprocess(["ffmpeg", "-y", "-i", path, "-c:v", "libx264", "-c:a", "aac", out_path])
            local_paths.append(out_path)

        concat_path = os.path.join(tmpdir, "concat.mp4")
        concat_videos_robust(local_paths, concat_path)
        
        final_path = os.path.join(tmpdir, "final.mp4")
        add_background_music(concat_path, final_path)

        final_url = safe_upload_to_cloudinary(final_path)
        
        shutil.rmtree(tmpdir)
        
        return {"status": "completed", "video_url": final_url}

    except Exception as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
