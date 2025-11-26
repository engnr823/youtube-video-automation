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

# Celery app import (Assumes celery_init.py exists alongside this file)
try:
    from celery_init import celery
except ImportError:
    logging.critical("❌ Could not import 'celery' from 'celery_init'. Ensure the file exists.")
    sys.exit(1)

# AI Clients
from openai import OpenAI
import replicate
from replicate.exceptions import ReplicateError

# --- CLIENT IMPORT SAFETY BLOCK ---
# Assumes a folder named 'video_clients' exists with these files
try:
    # We need this imported specifically for lip-syncing now
    from video_clients.elevenlabs_client import generate_voiceover_and_upload, generate_multi_voice_audio
except ImportError:
    logging.error("⚠️ ElevenLabs client not found in video_clients/. Audio/Lip-sync will fail.")
    generate_voiceover_and_upload = None
    generate_multi_voice_audio = None

try:
    # Assuming you have a generic video generator here (like Wan or SVD)
    from video_clients.replicate_client import generate_video_scene_with_replicate
except ImportError:
    logging.warning("⚠️ standard video generation client not found.")
    generate_video_scene_with_replicate = None
# ----------------------------------

# ==========================================
#  🚨 BUDGET SAFETY SWITCH 🚨
#  SET TO FALSE TO TEST PIPELINE WITHOUT SPENDING REPLICATE CREDIT.
#  SET TO TRUE ONLY WHEN READY TO GENERATE REAL VIDEO.
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
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")

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
# Royalty-Free Music Library
# -------------------------
MUSIC_LIBRARY = {
    "motivational": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
    "sad": "https://cdn.pixabay.com/download/audio/2021/11/24/audio_8243a76035.mp3",
    "intense": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_07b04b67e0.mp3",
    "happy": "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3",
    "default": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3"
}

# -------------------------
# Utility Helpers
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_prompt_template(filename: str) -> str:
    # Ensure prompts directory exists or handle missing path
    prompts_dir = os.path.join(os.getcwd(), "prompts")
    path = os.path.join(prompts_dir, filename)
    if not os.path.exists(path):
        logging.warning(f"Prompt template not found: {path}. Using default.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def download_to_file(url: str, dest_path: str, timeout: int = 300):
    logging.info(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    return dest_path

def run_subprocess(cmd: List[str], check=True) -> subprocess.CompletedProcess:
    logging.debug("Running command: " + " ".join(cmd))
    try:
        # close_fds=True helps prevent file descriptor leaks in celery workers
        return subprocess.run(cmd, check=check, capture_output=True, text=True, close_fds=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess failed stdout: {e.stdout}")
        logging.error(f"Subprocess failed stderr: {e.stderr}")
        raise

def safe_upload_to_cloudinary(filepath: str, resource_type="video", folder="automations"):
    if not os.getenv("CLOUDINARY_API_KEY"):
         logging.error("Cloudinary keys missing. Skipping upload.")
         return filepath
    try:
        logging.info(f"Uploading to Cloudinary: {filepath}")
        res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
        return res.get("secure_url")
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        # Return local path as fallback if upload fails, so pipeline continues
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
# Scraping & Storyboard
# -------------------------
def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee", max_results: int = 3) -> List[dict]:
    # Simplified for stability. 
    results = []
    if provider.lower() == "scrapingbee" and SCRAPINGBEE_API_KEY:
         logging.info(f"Attempting ScrapingBee for '{keyword}' (Placeholder logic)")
         # --- Insert actual scraping logic here if you have it ---
         pass
    else:
         logging.info("Skipping scraping (no key or provider selected).")
    return results

def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {
        "hook_style": "intrigue", 
        "avg_scene_count": 7, 
        "tone": "motivational" 
    }

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        # Use a reliable, cheaper model for testing. Upgrade to gpt-4o when ready.
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125") 
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":"You are a professional screenwriter. Output valid JSON only."},{"role":"user","content":prompt_content}],
            temperature=temperature,
            response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise

def create_video_storyboard_agent(keyword: str, blueprint: dict, form_data: dict) -> dict:
    prompt_template = load_prompt_template("prompt_video_storyboard_creator.txt")
    if not prompt_template:
        prompt_template = """
        TASK: Create a 7-scene short film script for '$keyword' formatted for a vertical Reel/TikTok.
        The script must include dialogue for characters to speak.
        Output strictly valid JSON with keys: video_title, video_description, main_character_profile, characters (list of objects with name, voice_id), scenes (list of objects with visual_prompt, action_prompt, audio_narration).
        Ensure 'audio_narration' contains the exact text to be spoken.
        """
    
    template = Template(prompt_template)
    full_context = keyword
    if form_data.get("characters"):
        full_context += f"\n\nUSER DEFINED CHARACTERS:\n{form_data.get('characters')}"

    target_scenes = form_data.get("max_scenes", 7)

    try:
        prompt = template.safe_substitute(
            keyword=full_context,
            blueprint_json=json.dumps(blueprint),
            language=form_data.get("language", "english"),
            # Default to reel for 7 scenes
            video_type=form_data.get("video_type", "reel"), 
            max_scenes=str(target_scenes)
        )

        prompt += f"\n\nIMPORTANT: Generate exactly {target_scenes} scenes. Ensure JSON format. Scenes with dialogue must have 'audio_narration' filled."

        raw = get_openai_response(prompt, temperature=0.6, is_json=True)
        obj = extract_json_from_text(raw) or json.loads(raw)
        if not obj or not obj.get("scenes"): raise RuntimeError("Storyboard generation produced invalid JSON")
        
        # Ensure we actually got the requested number of scenes
        scenes = obj.get("scenes", [])[:target_scenes]
        obj["scenes"] = scenes

        return obj
                
    except Exception as e:
         logging.error(f"Storyboard generation failed: {e}")
         raise RuntimeError(f"Failed to generate storyboard: {e}")

# -------------------------
# Character & Image Generation
# -------------------------
CHAR_DB_PATH = os.getenv("CHAR_DB_PATH", os.path.join(tempfile.gettempdir(), "character_db.json"))

def ensure_character(name: str, appearance_prompt: Optional[str] = None, voice_id: Optional[str] = None) -> dict:
    # Simplified character management
    try:
        if os.path.exists(CHAR_DB_PATH):
             with open(CHAR_DB_PATH, "r") as f: db = json.load(f)
        else: db = {}
    except: db = {}

    if name in db: return db[name]
    
    db[name] = {
        "id": str(uuid.uuid4()),
        "name": name,
        # Ensure portrait aspect for reels
        "appearance_prompt": appearance_prompt or f"A vertical portrait photograph of {name}, looking at camera, 8k, photorealistic",
        "voice_id": voice_id
    }
    try:
        with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    except: pass
    return db[name]

# Specific retry for Replicate rate limits
@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=30), retry=retry_if_exception_type(ReplicateError))
def generate_flux_image_safe(prompt: str, aspect: str = "9:16") -> str:
    """Generates an image using Replicate with retry logic."""
    if not REPLICATE_API_TOKEN: raise RuntimeError("Replicate Token Missing")
    logging.info(f"Generating image via Replicate (Flux)...")
    # Using flux-schnell (faster/cheaper)
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg", "num_inference_steps": 4}
    )
    return str(output[0]) if isinstance(output, (list, tuple)) else str(output)

# Specific retry for Replicate Lip Sync
@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(ReplicateError))
def generate_lip_sync_safe(image_url: str, audio_url: str) -> str:
    """Generates lip-sync video using sadtalker on Replicate."""
    logging.info(f"👄 Starting Lip Sync generation (SadTalker)...")
    # Using a standard sadtalker deployment
    output = replicate.run(
        "cjwbw/sadtalker:a519a502c74ac74325776184f17a54342880017f848988a641dd1e88e8945d81",
        input={"source_image": image_url, "driven_audio": audio_url, "still": True, "enhancer": "gfpgan"}
    )
    return str(output)

# -------------------------
# [CRITICAL UPDATE] Scene Processor with Lip-Sync Logic
# -------------------------
def process_single_scene(scene: dict, index: int, character_profile: str, aspect: str = "9:16", default_voice_id: str = None) -> (int, Optional[str]):
    try:
        logging.info(f"--- Processing Scene {index+1} ---")

        # 1. Determine Visuals
        visual_setting = scene.get("visual_prompt", "")
        # Ensure the prompt is optimized for a portrait talking head
        full_image_prompt = f"A vertical portrait photograph of {character_profile}, {visual_setting}, looking directly at the camera, neutral expression, highly detailed, 8k."
        
        # Generate Base Image
        keyframe_url = generate_flux_image_safe(full_image_prompt, aspect=aspect)
        logging.info(f"Scene {index+1}: Base image generated.")

        # ==============================================================================
        # SAFETY SWITCH: If budget mode is ON, stop here and just return the image.
        # ==============================================================================
        if not USE_REAL_EXPENSIVE_GENERATION:
            logging.warning(f"Scene {index+1}: [BUDGET SAFETY MODE] Skipping video generation. Returning static image.")
            # We return the image URL. The assembly agent will handle converting it to a 3s clip.
            return (index, keyframe_url)
        # ==============================================================================


        # 2. Check for Dialogue (The Lip-Sync Decision)
        dialogue = scene.get("audio_narration", "").strip()
        
        if dialogue and len(dialogue) > 2:
            # --- PATH A: LIP SYNC ---
            logging.info(f"Scene {index+1}: Dialogue detected ('{dialogue[:20]}...'). Starting Lip Sync workflow.")
            
            # A. Generate Audio for THIS scene ONLY
            # Try to find a specific character voice, otherwise use default
            voice_id = scene.get("voice_id") or default_voice_id or "21m00Tcm4TlvDq8ikWAM"
            
            if not generate_voiceover_and_upload: raise RuntimeError("ElevenLabs client missing for lip sync.")
            
            # Note: This costs ElevenLabs credits
            scene_audio_url = generate_voiceover_and_upload(dialogue, voice_id)
            if not scene_audio_url: raise RuntimeError("Failed to generate scene audio.")
            logging.info(f"Scene {index+1}: Audio generated.")

            # B. Generate Lip Sync Video (Costs Replicate credits)
            video_url = generate_lip_sync_safe(keyframe_url, scene_audio_url)
            logging.info(f"Scene {index+1}: Lip sync video complete.")
            return (index, video_url)

        else:
            # --- PATH B: ATMOSPHERIC VIDEO (No talking) ---
            logging.info(f"Scene {index+1}: No dialogue. Generating atmospheric video.")
            action_prompt = scene.get("action_prompt", "subtle camera movement")
            
            if not generate_video_scene_with_replicate:
                 logging.warning("Video generation client missing, returning static image instead.")
                 return (index, keyframe_url)

            # (Costs Replicate credits - e.g., using Wan or SVD)
            video_url = generate_video_scene_with_replicate(prompt=action_prompt, image_url=keyframe_url, aspect=aspect)
            return (index, video_url)

    except Exception as e:
        logging.error(f"Scene {index+1} failed processing: {e}")
        logging.error(traceback.format_exc())
        return (index, None)

# -------------------------
# Assembly (Updated for Lip-Sync Clips)
# -------------------------
def concat_videos_robust(input_paths: List[str], output_path: str):
    """
    Concatenates videos using the demuxer method. 
    Crucial for mixing clips that have audio (lip-sync) with clips that don't.
    """
    logging.info(f"Concatenating {len(input_paths)} files...")
    
    # 1. Create a text list file for ffmpeg
    list_file_path = os.path.join(tempfile.gettempdir(), f"concat_list_{uuid.uuid4()}.txt")
    with open(list_file_path, "w") as f:
        for path in input_paths:
            # Escape paths for ffmpeg
            safe_path = path.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
            # Ensure uniform timescale to prevent glitches
            f.write("duration 0.04\n") # tiny buffer
            
    # 2. Run FFmpeg Concat Demuxer
    # -af apad: Adds silent audio to clips that don't have it so the stream doesn't break
    # -shortest: Ends video when the shortest stream ends (usually video)
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path,
        "-af", "apad",
        "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        output_path
    ]
    
    try:
        run_subprocess(cmd)
    finally:
        if os.path.exists(list_file_path): os.remove(list_file_path)
        
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
         raise RuntimeError("FFmpeg concatenation failed to produce a valid file.")
    return output_path

def add_background_music(video_path: str, music_url: str, output_path: str, tone: str):
    logging.info("Adding background music...")
    unique_music = f"bg_music_{uuid.uuid4()}.mp3"
    music_path = os.path.join(tempfile.gettempdir(), unique_music)
    
    try:
        download_to_file(music_url, music_path)
        # Mix audio: Keep original video audio (voices) at 100%, add music at 10% volume
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", music_path,
            "-filter_complex", "[0:a]volume=1.0[a1];[1:a]volume=0.1[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path
        ]
        run_subprocess(cmd)
    except Exception as e:
        logging.error(f"Music addition failed: {e}. Returning video without music.")
        shutil.copy(video_path, output_path)
    finally:
        if os.path.exists(music_path): os.remove(music_path)
    
    return output_path

def video_assembly_agent(scene_urls: List[str], aspect: str = "9:16", music_tone: str = "motivational"):
    tmpdir = tempfile.mkdtemp(prefix="assemble_")
    logging.info(f"Starting assembly in temp dir: {tmpdir}")
    try:
        # 1. Download and Standardize Clips
        local_scene_paths = []
        for i, url in enumerate(scene_urls):
            if not url: continue
            
            # Determine if it's an image (Budget Mode) or Video (Real Mode)
            is_video = url.endswith(".mp4") or url.endswith(".mov")
            ext = "mp4" if is_video else "jpg"
            local_path = os.path.join(tmpdir, f"scene_{i}_raw.{ext}")
            normalized_path = os.path.join(tmpdir, f"scene_{i}_norm.mp4")

            try:
                download_to_file(url, local_path)

                if is_video:
                    # It's a video (lip sync or atmospheric). Just copy it.
                    # Ideally, re-encode to standard format here to ensure compatibility.
                    run_subprocess([
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", local_path,  # Input 0: The Image
                    "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", # Input 1: Silent Audio Generator
                    "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p", "-vf", "scale=720:1280",
                    "-c:a", "aac", "-shortest",      # Encode audio and stop when image loop ends
                    normalized_path
                ])
                    local_scene_paths.append(normalized_path)
                else:
                    # It's an image (Budget Mode active). Convert to 3s static video.
                    logging.info(f"Converting image {i} to static video clip.")
                    run_subprocess(["ffmpeg", "-y", "-loop", "1", "-i", local_path, "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p", "-vf", "scale=720:1280", normalized_path])
                    local_scene_paths.append(normalized_path)
                    
            except Exception as e:
                logging.error(f"Failed to process scene {i}: {e}")

        if not local_scene_paths: raise RuntimeError("No valid scenes to assemble.")

        # 2. Concatenate
        concat_out = os.path.join(tmpdir, "concat_raw.mp4")
        concat_videos_robust(local_scene_paths, concat_out)

        # 3. Add Music
        final_out = os.path.join(tmpdir, "final_with_music.mp4")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        add_background_music(concat_out, music_url, final_out, music_tone)

        # 4. Upload
        return safe_upload_to_cloudinary(final_out, folder="final_videos_lipsync")
        
    except Exception as e:
        logging.error(f"Assembly failed: {e}")
        logging.error(traceback.format_exc())
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# -------------------------
# Celery Task (Main)
# -------------------------
@celery.task(bind=True, time_limit=1800) # 30 minute limit for 7 scenes
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] Task started.")

    # --- PRE-FLIGHT CHECKS ---
    if not OPENAI_API_KEY: raise ValueError("Missing OPENAI_API_KEY")
    if not REPLICATE_API_TOKEN: raise ValueError("Missing REPLICATE_API_TOKEN")
    if not os.getenv("ELEVENLABS_API_KEY"): logging.warning("ElevenLabs key missing. Lip sync will fail if tried.")
    
    if USE_REAL_EXPENSIVE_GENERATION:
         logging.warning("💰💰💰 REAL EXPENSIVE GENERATION IS ON. THIS WILL COST MONEY. 💰💰💰")
    else:
         logging.info("🛡️ SAFETY MODE ON. Generating images only. No video costs will be incurred.")
    # -------------------------
    
    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(f"[{task_id}] STATUS: {msg}")

        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")

        # 1. Storyboard (The blueprint for 7 scenes)
        update_status("Step 1/4: Writing 7-Scene Script...")
        # Force 7 scenes for a reel
        form_data["max_scenes"] = 7
        form_data["video_type"] = "reel" 
        blueprint = analyze_competitors([]) # Placeholder
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)
        
        scenes = storyboard.get("scenes", [])
        if len(scenes) < 7: logging.warning(f"Only generated {len(scenes)} scenes, desired 7.")

        # 2. Character Prep
        update_status("Step 2/4: Preparing Character...")
        chars_data = storyboard.get("characters", [])
        main_char_name = chars_data[0].get("name", "Narrator") if chars_data else "Narrator"
        # Use user provided voice if available, else the one from storyboard
        default_voice_id = form_data.get("voice_selection") or (chars_data[0].get("voice_id") if chars_data else None)

        # Ensure character exists in DB and get their look profile
        char_db_entry = ensure_character(main_char_name, voice_id=default_voice_id)
        char_profile = char_db_entry.get("appearance_prompt")

        # 3. Scene Generation Loop (Images -> Audio -> Lip Sync Video)
        update_status(f"Step 3/4: Generating {len(scenes)} Scenes (This takes time)...")
        scene_urls = [None] * len(scenes)
        aspect = "9:16" # Always portrait for reels

        # Use ThreadPool but keep workers low to prevent rate limit issues
        # Increase max_workers to 2 or 3 only if you have high tier Replicate account
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Pass necessary data to the processing function
            future_to_idx = {
                executor.submit(
                    process_single_scene, 
                    scene, 
                    i, 
                    char_profile, 
                    aspect, 
                    default_voice_id
                ): i for i, scene in enumerate(scenes)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, url = future.result()
                    if url:
                         logging.info(f"✅ Scene {idx+1} finished successfully.")
                         scene_urls[idx] = url
                    else:
                         logging.error(f"❌ Scene {idx+1} failed to produce a URL.")
                except Exception as e:
                    logging.error(f"❌ Scene {idx+1} raised critical exception: {e}")

        # 4. Final Assembly
        update_status("Step 4/4: Assembling Final Reel...")
        valid_urls = [u for u in scene_urls if u]
        if not valid_urls: raise RuntimeError("All scenes failed generation.")
        
        tone = blueprint.get("tone", "motivational")
        # Note: We no longer pass a single voiceover_url, as audio is now inside the clips
        final_url = video_assembly_agent(valid_urls, aspect=aspect, music_tone=tone)

        logging.info(f"[{task_id}] Finished. Final URL: {final_url}")
        return {
            "status": "completed",
            "video_url": final_url,
            "storyboard": storyboard
        }

    except Exception as e:
        err_msg = str(e)
        logging.error(f"[{task_id}] Task Failed: {err_msg}")
        logging.error(traceback.format_exc())
        self.update_state(state="FAILURE", meta={"error": err_msg, "traceback": traceback.format_exc()})
        raise Exception(err_msg)
