# celery_worker.py
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
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError

# Celery app import (ensure celery_init.py provides 'celery')
from celery_init import celery

# AI Clients
from openai import OpenAI
import replicate

# --- CLIENT IMPORT SAFETY BLOCK ---
# This prevents the worker from crashing immediately if a file is missing
try:
    from video_clients.elevenlabs_client import generate_voiceover_and_upload
except ImportError:
    logging.warning("⚠️ ElevenLabs client not found. Voiceover generation will fail.")
    generate_voiceover_and_upload = None

try:
    from video_clients.replicate_client import generate_video_scene_with_replicate
except ImportError:
    logging.warning("⚠️ Replicate client not found. Video generation will fail.")
    generate_video_scene_with_replicate = None
# ----------------------------------

# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

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
LOCAL_UPLOADED_FILE = "/mnt/data/v4afissrdr5krnzvpfvr.mp4"

# -------------------------
# Pydantic Schemas (The "Brain" of the Data)
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
    sfx_note: Optional[str] = ""

class StoryboardSchema(BaseModel):
    video_title: str
    video_description: Optional[str] = ""
    main_character_profile: Optional[str] = ""
    characters: Optional[List[dict]] = []
    scenes: List[SceneSchema]

# -------------------------
# Utility Helpers
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

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
        return subprocess.run(cmd, check=check, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error("Subprocess failed: " + str(e.stderr or e))
        raise

def safe_upload_to_cloudinary(filepath: str, resource_type="video", folder="automations"):
    try:
        res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
        return res.get("secure_url")
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        raise

def extract_json_from_text(text: str) -> Optional[dict]:
    if not text: return None
    # Try finding Markdown JSON
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    # Try finding raw JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try: return json.loads(text[start:end+1])
        except: pass
    return None

# -------------------------
# Scraping
# -------------------------
def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee", max_results: int = 8) -> List[dict]:
    results = []
    logging.info(f"Scraping YouTube for '{keyword}' provider={provider}")
    try:
        if provider.lower() == "scrapingbee":
            if not SCRAPINGBEE_API_KEY: raise RuntimeError("SCRAPINGBEE_API_KEY missing")
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(keyword)}"
            params = {"api_key": SCRAPINGBEE_API_KEY, "url": url, "render_js": "false"}
            r = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=30)
            r.raise_for_status()
            html = r.text
            match = re.search(r"ytInitialData\s*=\s*(\{.*\});", html, re.DOTALL) or re.search(r"var ytInitialData = (\{.*\})", html, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                def collect_vr(node):
                    if isinstance(node, dict):
                        if "videoRenderer" in node:
                            vr = node["videoRenderer"]
                            try:
                                vid = vr.get("videoId")
                                title = vr.get("title", {}).get("simpleText", "")
                                desc = vr.get("shortDescription", {}).get("simpleText", "")
                                thumb = vr.get("thumbnail", {}).get("thumbnails", [{}])[-1].get("url")
                                results.append({"videoId": vid, "title": title, "description": desc, "thumbnail": thumb, "url": f"https://www.youtube.com/watch?v={vid}"})
                            except: pass
                        for v in node.values(): collect_vr(v)
                    elif isinstance(node, list):
                        for item in node: collect_vr(item)
                collect_vr(data)
        elif provider.lower() == "serpapi":
            if not SERPAPI_KEY: raise RuntimeError("SERPAPI_KEY missing")
            params = {"engine": "youtube", "search_query": keyword, "api_key": SERPAPI_KEY, "num": max_results}
            r = requests.get("https://serpapi.com/search", params=params, timeout=30)
            r.raise_for_status()
            for item in r.json().get("video_results", [])[:max_results]:
                results.append({"videoId": item.get("id"), "title": item.get("title"), "description": item.get("description"), "url": item.get("link")})
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    
    # Deduplicate
    seen = set()
    filtered = []
    for r in results:
        if r.get("url") and r.get("url") not in seen:
            filtered.append(r)
            seen.add(r.get("url"))
    return filtered[:max_results]

# -------------------------
# Analysis & Storyboard
# -------------------------
def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    # Heuristics based on scraped data
    return {
        "hook_style": "intrigue",
        "avg_scene_count": 6,
        "avg_scene_duration": 5,
        "tone": "motivational",
        "recommended_length_seconds": 60
    }

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[{"role":"system","content":"You are a professional screenwriter."},{"role":"user","content":prompt_content}],
            temperature=temperature,
            response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return ""

def create_video_storyboard_agent(keyword: str, blueprint: dict, form_data: dict) -> dict:
    prompt = f"""
    You are an expert short film writer.
    BLUEPRINT: {json.dumps(blueprint)}
    TASK: Create an original short film script for '{keyword}'.
    Output valid JSON with keys: video_title, video_description, main_character_profile, characters (list), scenes (list).
    Each scene must have: scene_id, duration_seconds, visual_prompt, action_prompt, audio_narration, shot_type, camera_angle, lighting.
    """
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or json.loads(raw)
    if not obj: raise RuntimeError("Storyboard generation failed")
    
    try:
        return StoryboardSchema(**obj).dict()
    except ValidationError as e:
        logging.warning(f"Validation error: {e}, attempting repair...")
        # Basic repair: Ensure 'scenes' exists
        return {
            "video_title": obj.get("video_title", "Untitled"),
            "scenes": obj.get("scenes", []),
            "characters": obj.get("characters", [])
        }

# -------------------------
# Character DB
# -------------------------
CHAR_DB_PATH = os.getenv("CHAR_DB_PATH", "/var/data/character_db.json")
ensure_dir(str(Path(CHAR_DB_PATH).parent))

def ensure_character(name: str, appearance_prompt: Optional[str] = None, reference_image_url: Optional[str] = None, voice_id: Optional[str] = None) -> dict:
    try:
        with open(CHAR_DB_PATH, "r") as f: db = json.load(f)
    except: db = {}
    
    if name in db: return db[name]
    
    db[name] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "appearance_prompt": appearance_prompt or f"{name}, photorealistic",
        "reference_image": reference_image_url,
        "voice_id": voice_id or "21m00Tcm4TlvDq8ikWAM"
    }
    with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    return db[name]

# -------------------------
# Generation Logic
# -------------------------
def generate_flux_image(prompt: str, aspect: str = "16:9") -> str:
    logging.info(f"Generating keyframe: {prompt[:50]}...")
    output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg"})
    return str(output[0]) if isinstance(output, (list, tuple)) else str(output)

def process_single_scene(scene: dict, index: int, character_profile: str, aspect: str = "16:9") -> (int, Optional[str]):
    try:
        logging.info(f"Processing scene {index}")
        
        # --- [CRITICAL FEATURE] MP4 Passthrough ---
        # If the image_url is actually a video file, skip generation and use it directly.
        if scene.get("image_url") and str(scene.get("image_url")).endswith(".mp4"):
            logging.info(f"Scene {index}: Using pre-uploaded video asset.")
            return (index, scene.get("image_url"))
        # ------------------------------------------

        if scene.get("image_url"):
            keyframe_url = scene.get("image_url")
        else:
            visual_setting = scene.get("visual_prompt", "")
            full_image_prompt = f"{character_profile}, {visual_setting}, cinematic lighting"
            keyframe_url = generate_flux_image(full_image_prompt, aspect=aspect)

        action = scene.get("action_prompt", "") + f", camera:{scene.get('camera_angle','35mm')}"
        
        if generate_video_scene_with_replicate:
            video_url = generate_video_scene_with_replicate(prompt=action, image_url=keyframe_url, aspect=aspect)
            return (index, video_url)
        else:
            raise RuntimeError("Replicate Client function missing")
            
    except Exception as e:
        logging.exception(f"Scene {index} failed: {e}")
        return (index, None)

# -------------------------
# Assembly (FFmpeg)
# -------------------------
def concat_videos_safe(input_paths: List[str], output_path: str, width: int = 1280, height: int = 720):
    cmd = ["ffmpeg", "-y"]
    for p in input_paths: cmd.extend(["-i", p])
    
    # Complex filter for scaling and padding to ensure uniform resolution
    filter_complex = ""
    for i in range(len(input_paths)):
        filter_complex += f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    for i in range(len(input_paths)):
        filter_complex += f"[v{i}]"
    filter_complex += f"concat=n={len(input_paths)}:v=1:a=0[outv]"
    
    cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path])
    run_subprocess(cmd)
    return output_path

def merge_audio_video(video_path: str, audio_path: str, output_path: str):
    # Merge video with audio, cutting video to audio length (-shortest is optional depending on pref)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", output_path]
    run_subprocess(cmd)
    return output_path

def video_assembly_agent(scene_urls: List[str], voiceover_url: str, storyboard: dict, aspect: str = "16:9"):
    tmpdir = tempfile.mkdtemp(prefix="assemble_")
    try:
        local_scene_paths = []
        for i, url in enumerate(scene_urls):
            if not url: continue
            local_path = os.path.join(tmpdir, f"scene_{i}.mp4")
            download_to_file(url, local_path)
            local_scene_paths.append(local_path)

        if not local_scene_paths: raise RuntimeError("No videos to assemble")

        local_voice = os.path.join(tmpdir, "voice.mp3")
        download_to_file(voiceover_url, local_voice)

        width, height = (1280, 720) if aspect == "16:9" else (720, 1280)
        concat_out = os.path.join(tmpdir, "concat.mp4")
        concat_videos_safe(local_scene_paths, concat_out, width, height)

        final_out = os.path.join(tmpdir, "final.mp4")
        merge_audio_video(concat_out, local_voice, final_out)

        return safe_upload_to_cloudinary(final_out, folder="final_videos")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# -------------------------
# Celery Task (Main)
# -------------------------
@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] Task started.")
    
    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(msg)

        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")

        # 1. Scrape
        update_status("Scraping...")
        scraped = scrape_youtube_videos(keyword)
        blueprint = analyze_competitors(scraped)

        # 2. Storyboard
        update_status("Storyboarding...")
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)

        # 3. Characters & B-Roll Injection
        update_status("Characters...")
        characters = storyboard.get("characters") or []
        uploaded_images = form_data.get("uploaded_images") or []
        
        # Inject user uploaded image to main character
        if uploaded_images:
             if not characters:
                 characters = [{"name": "Main", "reference_image_url": uploaded_images[0]}]
             else:
                 characters[0]["reference_image_url"] = uploaded_images[0]
             storyboard["characters"] = characters

        for ch in characters:
            ensure_character(ch.get("name", "Main"), ch.get("appearance_prompt"), ch.get("reference_image_url"))
        
        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # 3b. Local Asset Injection (The "B-Roll" Feature)
        use_local_asset = form_data.get("use_local_asset", False)
        if use_local_asset and os.path.exists(LOCAL_UPLOADED_FILE):
             try:
                 local_url = safe_upload_to_cloudinary(LOCAL_UPLOADED_FILE, folder="user_assets")
                 scene0 = {
                     "scene_id": 0, "duration_seconds": 3, "visual_prompt": "Intro B-roll",
                     "action_prompt": "static", "image_url": local_url # Pass video url here
                 }
                 storyboard["scenes"].insert(0, scene0)
             except Exception as e:
                 logging.warning(f"Failed to inject local asset: {e}")

        # 4. Voiceover
        update_status("Voiceover...")
        full_script = " ".join([s.get("audio_narration","") for s in storyboard.get("scenes", [])])
        voice_id = form_data.get("voice_selection") or "21m00Tcm4TlvDq8ikWAM"
        
        if generate_voiceover_and_upload:
            voiceover_url = generate_voiceover_and_upload(full_script, voice_id)
        else:
            raise RuntimeError("Voiceover client unavailable")

        # 5. Scenes
        update_status("Scenes...")
        scenes = storyboard.get("scenes", [])
        scene_urls = [None] * len(scenes)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {executor.submit(process_single_scene, scenes[i], i, char_profile): i for i in range(len(scenes))}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, url = future.result()
                    scene_urls[idx] = url
                except Exception as e:
                    logging.error(f"Scene {idx} error: {e}")

        valid_urls = [u for u in scene_urls if u]
        if not valid_urls: raise RuntimeError("All scenes failed")

        # 6. Assembly
        update_status("Assembling...")
        final_url = video_assembly_agent(valid_urls, voiceover_url, storyboard)

        return {
            "status": "ready",
            "video_url": final_url,
            "storyboard": storyboard
        }

    except Exception as e:
        logging.error(f"Task failed: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
