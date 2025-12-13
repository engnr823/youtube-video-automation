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
from string import Template
from typing import Optional, List, Dict, Tuple
from pathlib import Path

# Fix import path
WORKER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKER_DIR) 

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError
from celery_init import celery
from openai import OpenAI

# --- Configuration & Imports ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

try:
    from video_clients.elevenlabs_client import generate_audio_for_scene
except:
    generate_audio_for_scene = None

try:
    from video_clients.heygen_client import generate_heygen_video, get_stock_avatar
    HEYGEN_AVAILABLE = True
except ImportError:
    HEYGEN_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

# --- Utils ---
def get_media_duration(file_path):
    try:
        if not os.path.exists(file_path): return 0.0
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip()) if result.stdout.strip() else 0.0
    except: return 0.0

def download_to_file(url, dest_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    return dest_path

def safe_upload_to_cloudinary(filepath, resource_type="video", folder="automations"):
    res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
    return res.get("secure_url")

def load_prompt_template(filename):
    path = os.path.join("prompts", filename)
    return open(path, "r", encoding="utf-8").read() if os.path.exists(path) else ""

# --- Scene Processing ---
def process_single_scene(scene, index, character_profile, audio_path, character_faces, aspect):
    if not HEYGEN_AVAILABLE: return {"index": index, "status": "failed"}
    
    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("/tmp", f"scene_{index}_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # Determine Character
        char_list = scene.get('characters_in_scene', [])
        char_name = char_list[0] if char_list else list(character_faces.keys())[0] if character_faces else "Unknown"
        
        char_data = character_faces.get(char_name, {})
        # DEFAULT TO MALE ID if missing (Your ID)
        avatar_id = char_data.get("heygen_avatar_id", "4343bfb447bf4028a48b598ae297f5dc")

        # Upload Audio
        if not audio_path or not os.path.exists(audio_path):
            logging.warning(f"Scene {index}: No audio. Skipping.")
            return {"index": index, "status": "skipped"}
            
        cloud_audio = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")

        # Generate Video
        logging.info(f"Scene {index}: Generating with Avatar {avatar_id} ({char_name})")
        video_url = generate_heygen_video(
            avatar_id=avatar_id,
            audio_url=cloud_audio,
            aspect_ratio=aspect
        )

        # Download Result
        dest = os.path.join(temp_dir, f"scene_{index}.mp4")
        download_to_file(video_url, dest)
        return {"index": index, "video_path": dest, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} Failed: {e}")
        return {"index": index, "status": "failed"}

# --- Stitching ---
def stitch_video(scene_pairs, output_path):
    temp_dir = os.path.join("/tmp", f"stitch_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    list_path = os.path.join(temp_dir, "inputs.txt")
    
    try:
        chunks = []
        for i, (video, audio) in enumerate(scene_pairs):
            dur = get_media_duration(audio)
            chunk = os.path.join(temp_dir, f"chunk_{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(dur), "-c:v", "libx264", "-c:a", "aac", "-shortest", chunk
            ], check=True, capture_output=True)
            chunks.append(chunk)
            
        with open(list_path, "w") as f:
            for c in chunks: f.write(f"file '{c}'\n")
            
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path
        ], check=True)
        return True
    except Exception as e:
        logging.error(f"Stitching failed: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# --- AI Agents ---
def create_storyboard(keyword, form_data):
    template = load_prompt_template("prompt_video_storyboard_creator.txt") or "{}"
    prompt = Template(template).safe_substitute(
        keyword=keyword, 
        blueprint_json="{}", 
        language=form_data.get("language", "english"),
        max_scenes=str(form_data.get("max_scenes", 5))
    )
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role":"user", "content": prompt}], response_format={"type": "json_object"}
    )
    return json.loads(completion.choices[0].message.content)

def refine_script(storyboard, form_data):
    # Simplified for robustness
    segments = []
    default_voice = form_data.get("voice_selection", "21m00Tcm4TlvDq8ikWAM")
    
    for scene in storyboard.get("scenes", []):
        text = scene.get("audio_narration", "")
        # Basic voice assignment logic could go here if needed
        segments.append({"text": text, "voice_id": default_voice})
    return segments

# --- Main Task ---
@celery.task(bind=True)
def background_generate_video(self, form_data):
    try:
        # 1. Concept
        self.update_state(state="PROGRESS", meta={"message": "Creating Storyboard..."})
        storyboard = create_storyboard(form_data.get("keyword"), form_data)
        
        # 2. Casting (Fixing Gender Mismatch)
        characters = storyboard.get("characters", [])
        char_faces = {}
        for char in characters:
            name = char.get("name", "Unknown")
            desc = char.get("appearance_prompt", "").lower()
            name_lower = name.lower()
            
            # IMPROVED GENDER DETECTION
            is_female = any(w in desc for w in ["woman", "female", "girl", "lady", "mother", "wife"]) or \
                        any(w in name_lower for w in ["zara", "sarah", "emily", "mom", "aisha", "fatima"])
            
            # Select ID
            if is_female:
                # Use Public Female Avatar
                avatar_id = "Avatar_Expressive_20240520_02"
            else:
                # Use Your Male Talking Photo
                avatar_id = "4343bfb447bf4028a48b598ae297f5dc"
            
            char_faces[name] = {"heygen_avatar_id": avatar_id}

        # 3. Audio
        self.update_state(state="PROGRESS", meta={"message": "Generating Audio..."})
        segments = refine_script(storyboard, form_data)
        scene_assets = []
        
        for i, scene in enumerate(storyboard.get("scenes", [])):
            text = segments[i]["text"]
            voice = segments[i]["voice_id"]
            
            # Generate Audio
            audio_res = generate_audio_for_scene(text, voice) if generate_audio_for_scene else None
            if audio_res:
                scene_assets.append({
                    "index": i, 
                    "audio_path": audio_res["path"], 
                    "scene_data": scene
                })

        # 4. Video Generation
        self.update_state(state="PROGRESS", meta={"message": "Rendering Video (HeyGen)..."})
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"
        final_pairs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(process_single_scene, 
                                asset["scene_data"], 
                                asset["index"], 
                                "", 
                                asset["audio_path"], 
                                char_faces, 
                                aspect): asset 
                for asset in scene_assets
            }
            
            results = {}
            for f in concurrent.futures.as_completed(futures):
                res = f.result()
                if res["status"] == "success":
                    # Map index to (video, audio)
                    results[res["index"]] = (res["video_path"], futures[f]["audio_path"])
            
            # Sort by index
            for i in range(len(storyboard["scenes"])):
                if i in results: final_pairs.append(results[i])

        if not final_pairs: raise RuntimeError("No scenes were generated successfully.")

        # 5. Stitching
        self.update_state(state="PROGRESS", meta={"message": "Stitching Final Video..."})
        final_path = f"/tmp/final_{uuid.uuid4()}.mp4"
        if stitch_video(final_pairs, final_path):
            video_url = safe_upload_to_cloudinary(final_path, folder="final_videos")
            return {"status": "ready", "video_url": video_url}
        else:
            raise RuntimeError("Stitching failed")

    except Exception as e:
        logging.error(f"Fatal Error: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
