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

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError

# Celery app import
from celery_init import celery

# --- AI CLIENTS & IMPORTS ---
try:
    import replicate
except ImportError:
    replicate = None

from openai import OpenAI

# --- ELEVENLABS SAFE IMPORT ---
try:
    from video_clients.elevenlabs_client import generate_audio_for_scene
    ELEVENLABS_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ ElevenLabs client not found. Audio generation will fallback to OpenAI.")
    generate_audio_for_scene = None
    ELEVENLABS_AVAILABLE = False

# --- UTILS IMPORT SAFETY BLOCK ---
try:
    from utils.ffmpeg_utils import get_media_duration
except ImportError:
    def get_media_duration(file_path):
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return 0.0

# -------------------------
# CONFIGURATION
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

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

if replicate and REPLICATE_API_TOKEN:
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
else:
    replicate_client = None

# -------------------------
# ROYALTY-FREE MUSIC LIBRARY
# -------------------------
MUSIC_LIBRARY = {
    "motivational": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3",
    "sad": "https://cdn.pixabay.com/download/audio/2021/11/24/audio_8243a76035.mp3",
    "intense": "https://cdn.pixabay.com/download/audio/2022/03/24/audio_07b04b67e0.mp3",
    "happy": "https://cdn.pixabay.com/download/audio/2022/01/18/audio_d0a13f69d2.mp3",
    "default": "https://cdn.pixabay.com/download/audio/2022/05/27/audio_1808fbf07a.mp3"
}

# -------------------------
# DATA SCHEMAS
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

# -------------------------
# CHARACTER DATABASE
# -------------------------
CHAR_DB_PATH = os.getenv("CHAR_DB_PATH", "/var/data/character_db.json")

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_character(name: str, appearance_prompt: Optional[str] = None, reference_image_url: Optional[str] = None) -> dict:
    ensure_dir(str(Path(CHAR_DB_PATH).parent))
    try:
        with open(CHAR_DB_PATH, "r") as f: db = json.load(f)
    except Exception:
        db = {}
    if name in db: return db[name]
    db[name] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "appearance_prompt": appearance_prompt or f"{name}, photorealistic",
        "reference_image": reference_image_url
    }
    try:
        with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    except Exception: pass
    return db[name]

# -------------------------
# CORE UTILS
# -------------------------
def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if not os.path.exists(path): return ""
    with open(path, "r", encoding="utf-8") as f: return f.read()

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

def safe_upload_to_cloudinary(filepath: str, resource_type="video", folder="automations"):
    try:
        res = cloudinary.uploader.upload(filepath, resource_type=resource_type, folder=folder)
        url = res.get("secure_url") or res.get("url")
        return url
    except Exception as e:
        logging.error(f"Cloudinary upload failed: {e}")
        raise

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
# REPLICATE SAFE RUNNER
# -------------------------
def normalize_replicate_output(raw):
    try:
        if raw is None: return None
        if isinstance(raw, str): return raw
        if isinstance(raw, (list, tuple)) and raw:
            for v in raw:
                if isinstance(v, str): return v
            return str(raw[0])
        if isinstance(raw, dict):
            for key in ("url", "output", "result", "video", "file"):
                if key in raw: return raw.get(key)
        return str(raw)
    except Exception: return str(raw)

def get_latest_model_version_id(model_name: str) -> Optional[str]:
    # Skip version check for models we know have issues or are official
    if "flux" in model_name or "wan-video" in model_name or "sadtalker" in model_name:
        return None
    if not replicate_client: raise RuntimeError("Replicate client not configured")
    try:
        model = replicate_client.models.get(model_name)
        versions = model.versions.list()
        if not versions: return None
        return versions[0].id if hasattr(versions[0], "id") else str(versions[0])
    except Exception as e:
        logging.warning(f"Unable to fetch versions for {model_name}: {e}")
        return None

def replicate_run_safe(model_name: str, **kwargs) -> Optional[str]:
    if not replicate_client: raise RuntimeError("Replicate client not configured")
    
    # 1. Try Direct Run
    try:
        logging.info(f"Replicate: running {model_name}")
        raw = replicate_client.run(model_name, **kwargs)
        return normalize_replicate_output(raw)
    except Exception as e:
        logging.warning(f"Direct run failed for {model_name}, attempting version resolve: {e}")
    
    # 2. Resolve Version (Fallback)
    version_id = get_latest_model_version_id(model_name)
    if not version_id:
        # If no version found and direct run failed, re-raise original error
        raise RuntimeError(f"Model run failed for {model_name}")
    
    model_ref = f"{model_name}:{version_id}"
    raw = replicate_client.run(model_ref, **kwargs)
    return normalize_replicate_output(raw)

# -------------------------
# AI GENERATORS
# -------------------------
def generate_flux_image(prompt: str, aspect: str = "16:9") -> str:
    model_name = "black-forest-labs/flux-schnell"
    input_payload = {"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg"}
    for i in range(2):
        try:
            return str(replicate_run_safe(model_name, input=input_payload))
        except Exception: time.sleep(1)
    raise RuntimeError("Flux image generation failed")

def generate_audio_robust(text: str, voice_id: str) -> str:
    # 1. ElevenLabs
    if ELEVENLABS_AVAILABLE and generate_audio_for_scene:
        try:
            res = generate_audio_for_scene(text, voice_id)
            if res and isinstance(res, dict) and res.get("path"):
                return res["path"]
        except Exception as e:
            logging.warning(f"⚠️ ElevenLabs failed: {e}. Switching to OpenAI.")

    # 2. OpenAI
    if openai_client:
        try:
            logging.info("🎙️ Using OpenAI TTS Fallback...")
            safe_id = str(uuid.uuid4())
            output_path = os.path.join(tempfile.gettempdir(), f"openai_audio_{safe_id}.mp3")
            
            v_lower = (voice_id or "").lower()
            openai_voice = "alloy"
            if "male" in v_lower: openai_voice = "onyx"
            elif "female" in v_lower: openai_voice = "nova"

            response = openai_client.audio.speech.create(
                model="tts-1", voice=openai_voice, input=text
            )
            response.stream_to_file(output_path)
            return output_path
        except Exception as e:
            logging.error(f"OpenAI TTS failed: {e}")

    raise RuntimeError("All audio generation methods failed.")

# -------------------------
# SCENE PROCESSING (FIXED LIP SYNC)
# -------------------------
def process_single_scene(
    scene: dict,
    index: int,
    character_profile: str,
    audio_path: str = None,
    character_faces: dict = {},
    aspect: str = "16:9"
) -> dict:
    request_id = str(uuid.uuid4())[:8]
    temp_dir = os.path.join("/tmp", f"scene_{index}_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    try:
        time.sleep(random.randint(1, 3))

        target_face_url = None
        visual_prompt = (scene.get("visual_prompt") or "").lower()

        # Face matching
        for char_name, face_url in character_faces.items():
            if char_name.lower() in visual_prompt:
                target_face_url = face_url
                break
        if not target_face_url and character_faces:
            target_face_url = list(character_faces.values())[0]

        shot_type = (scene.get("shot_type") or "medium").lower()
        has_dialogue = bool(audio_path and get_media_duration(audio_path) > 0.5)
        is_wide = "wide" in shot_type or "drone" in shot_type

        video_url = None

        # 1. Lip Sync (SadTalker) - Fixed Model Hash
        if has_dialogue and target_face_url and not is_wide:
            try:
                cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
                # FIXED: Added exact version hash for SadTalker
                model_name = "lucataco/sadtalker:85c698db7c0a66d5011435d0191bd32305a9c7499252a9041270252565697697"
                
                input_payload = {
                    "source_image": target_face_url,
                    "driven_audio": cloud_audio_url,
                    "still": True, "enhancer": "gfpgan", "expression_scale": 1.1
                }
                # Use direct run via client for pinned version
                raw = replicate_client.run(model_name, input=input_payload)
                video_url = normalize_replicate_output(raw)
            except Exception as e:
                logging.error(f"Lip-sync failed: {e}")

        # 2. Cinematic (Wan Video)
        if not video_url:
            action = f"{scene.get('action_prompt','cinematic movement')}, camera:{scene.get('camera_angle','35mm')}"
            try:
                model_name = "wan-video/wan-2.1-1.3b"
                input_payload = {"prompt": f"{visual_prompt}, {action}", "aspect_ratio": aspect}
                if target_face_url: input_payload["image"] = target_face_url
                video_url = replicate_run_safe(model_name, input=input_payload)
            except Exception as e:
                logging.error(f"Cinematic failed: {e}")

        if not video_url: raise RuntimeError("Video generation failed")

        temp_video_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
        download_to_file(str(video_url), temp_video_path, timeout=300)
        return {"index": index, "video_path": temp_video_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {traceback.format_exc()}")
        return {"index": index, "video_path": None, "status": "failed"}
    finally:
        if os.path.exists(temp_dir):
            try: shutil.rmtree(temp_dir)
            except: pass

# -------------------------
# STITCHING & AGENTS
# -------------------------
def stitch_video_audio_pairs_optimized(scene_pairs: List[Tuple[str, str]], output_path: str) -> bool:
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"render_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []
    try:
        logging.info(f"Stitching {len(scene_pairs)} pairs")
        for i, (video, audio) in enumerate(scene_pairs):
            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            audio_dur = get_media_duration(audio)
            if audio_dur == 0: continue 

            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(audio_dur), "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
                "-c:a", "aac", "-shortest", chunk_name
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths: return False

        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                safe_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)
        return True
    except Exception as e:
        logging.error(f"Stitching error: {e}")
        return False
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[{"role":"system","content": "You are a professional screenwriter." + (" Output valid JSON." if is_json else "")},{"role":"user","content":prompt_content}],
            temperature=temperature,
            response_format={"type": "json_object"} if is_json else {"type": "text"}
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI error: {e}")
        return "{}" if is_json else ""

def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee") -> List[dict]:
    if provider == "scrapingbee" and SCRAPINGBEE_API_KEY:
        try:
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(keyword)}"
            params = {"api_key": SCRAPINGBEE_API_KEY, "url": url, "render_js": "false"}
            requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=20)
            return [{"title": f"Viral {keyword}", "views": "1M"}] 
        except: pass
    return []

def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {"hook_style": "intrigue", "avg_scene_count": 6, "tone": "motivational"}

def create_video_storyboard_agent(keyword: str, blueprint: dict, form_data: dict) -> dict:
    prompt_template = load_prompt_template("prompt_video_storyboard_creator.txt")
    if not prompt_template: prompt_template = "TASK: Create script for '$keyword'. Output valid JSON."
    template = Template(prompt_template)
    full_context = keyword
    if form_data.get("characters"): full_context += f"\n\nUSER DEFINED CHARACTERS:\n{form_data.get('characters')}"
    target_scenes = form_data.get("max_scenes", 7)
    prompt = template.safe_substitute(
        keyword=full_context, blueprint_json=json.dumps(blueprint),
        language=form_data.get("language", "english"),
        video_type=form_data.get("video_type", "reel"),
        uploaded_assets_context="User uploaded images present" if form_data.get("uploaded_images") else "No uploads",
        max_scenes=str(target_scenes)
    )
    prompt += f"\n\nIMPORTANT: The 'audio_narration' MUST be at least 150 words. Generate {target_scenes} scenes."
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or (json.loads(raw) if raw else {})
    if not obj: raise RuntimeError("Storyboard generation failed")
    try: return StoryboardSchema(**obj).dict()
    except ValidationError: return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

def refine_script_with_roles(storyboard: dict, form_data: dict) -> List[dict]:
    scenes = storyboard.get('scenes', [])
    segments = []
    main_voice = form_data.get("voice_selection") or "neutral"
    for s in scenes:
        text = s.get("audio_narration", "")
        segments.append({"text": text, "voice_id": main_voice})
    return segments

def generate_thumbnail_agent(storyboard: dict, orientation: str = "16:9") -> Optional[str]:
    summary = storyboard.get("video_description") or "Video"
    try: return generate_flux_image(f"Movie poster: {summary}, 8k", aspect=orientation)
    except: return None

def youtube_metadata_agent(full_script: str, keyword: str, form_data: dict, blueprint: dict) -> dict:
    prompt_template = load_prompt_template("prompt_youtube_metadata_generator.txt")
    if not prompt_template: return {}
    template = Template(prompt_template)
    prompt = template.safe_substitute(
        primary_keyword=keyword, full_script=full_script[:3000],
        language=form_data.get("language", "english"), video_type=form_data.get("video_type", "reel"),
        blueprint_data=json.dumps(blueprint), thumbnail_concept=form_data.get("thumbnail_concept", "")
    )
    raw = get_openai_response(prompt, temperature=0.4, is_json=True)
    return extract_json_from_text(raw) or (json.loads(raw) if raw else {})

# -------------------------
# CELERY TASK MAIN
# -------------------------
@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    try:
        task_id = getattr(self.request, "id", "unknown")
        logging.info(f"[{task_id}] SaaS Task started.")
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(msg)

        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")

        # 1. Blueprint
        update_status("Designing Video Concept...")
        scraped = scrape_youtube_videos(keyword)
        blueprint = analyze_competitors(scraped)
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)
        scenes = storyboard.get("scenes", [])
        if not scenes: raise RuntimeError("Failed to generate scenes.")

        # 2. Casting
        update_status("Casting Characters...")
        characters = storyboard.get("characters", [])
        uploaded_images = form_data.get("uploaded_images") or []
        character_faces = {}
        for i, url in enumerate(uploaded_images):
            if i < len(characters):
                char_name = characters[i].get("name")
                character_faces[char_name] = url
        
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"
        for char in characters:
            name = char.get("name", "Unknown")
            ensure_character(name) 
            if name not in character_faces:
                desc = char.get("appearance_prompt") or f"Cinematic portrait of {name}"
                try:
                    character_faces[name] = generate_flux_image(f"{desc}, 8k", aspect=aspect)
                except Exception as e:
                    logging.error(f"Casting failed for {name}: {e}")

        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # 3. Audio
        update_status("Synthesizing Audio Dialogue...")
        segments = refine_script_with_roles(storyboard, form_data)
        scene_assets = []
        full_script_text = ""
        for i, scene in enumerate(scenes):
            if i >= len(segments): break
            text = segments[i].get("text")
            voice_id = segments[i].get("voice_id")
            full_script_text += text + " "
            try:
                audio_path = generate_audio_robust(text, voice_id)
                duration = get_media_duration(audio_path)
                scene_assets.append({"index": i, "audio_path": audio_path, "duration": duration, "scene_data": scene})
            except Exception as e:
                logging.error(f"Scene {i} audio failed: {e}")

        if not scene_assets: raise RuntimeError("Audio generation failed for all scenes.")

        # 4. Rendering
        update_status("Rendering Video Scenes (Synced)...")
        final_pairs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_asset = {
                executor.submit(process_single_scene, asset["scene_data"], asset["index"], char_profile, asset["audio_path"], character_faces, aspect): asset
                for asset in scene_assets
            }
            results_map = {}
            for future in concurrent.futures.as_completed(future_to_asset):
                try:
                    res = future.result()
                    if res["status"] == "success" and res["video_path"]:
                        results_map[res["index"]] = (res["video_path"], future_to_asset[future]["audio_path"])
                except Exception as e:
                    logging.error(f"Scene error: {e}")
            
            for i in range(len(scenes)):
                if i in results_map: final_pairs.append(results_map[i])

        if not final_pairs: raise RuntimeError("Video generation failed (no scenes).")

        # 5. Stitching
        update_status("Final Assembly...")
        music_tone = blueprint.get("tone", "motivational")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try: download_to_file(music_url, music_path)
        except: music_path = None
        
        final_output_path = f"/tmp/final_render_{task_id}.mp4"
        temp_visual_path = f"/tmp/visual_base_{task_id}.mp4"
        if not stitch_video_audio_pairs_optimized(final_pairs, temp_visual_path):
            raise RuntimeError("Stitching failed.")

        if music_path:
            cmd = [
                "ffmpeg", "-y", "-i", temp_visual_path, "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "libx264", "-vf", "scale='min(720,iw)':-2", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-shortest", final_output_path
            ]
            subprocess.run(cmd, check=True)
        else:
            shutil.move(temp_visual_path, final_output_path)

        # 6. Upload
        update_status("Uploading & Finalizing...")
        final_video_url = safe_upload_to_cloudinary(final_output_path, folder="final_videos")
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        metadata = youtube_metadata_agent(full_script_text, keyword, form_data, blueprint)

        # Cleanup
        try:
            if os.path.exists(temp_visual_path): os.remove(temp_visual_path)
            for v, a in final_pairs:
                if os.path.exists(v): os.remove(v)
        except: pass

        return {
            "status": "ready",
            "video_url": final_video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard
        }

    except Exception as e:
        logging.error(f"Task Crashed: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise RuntimeError(str(e))
