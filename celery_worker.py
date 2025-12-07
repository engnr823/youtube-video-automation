# file: video_worker_safe.py
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

# AI Clients
try:
    import replicate
except Exception:
    replicate = None

from openai import OpenAI

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
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logging.error(f"Error getting duration for {file_path}: {e}")
            return 5.0

# --- CLIENT IMPORT SAFETY BLOCK ---
try:
    from video_clients.elevenlabs_client import (
        generate_voiceover_and_upload, 
        generate_multi_voice_audio,
        generate_audio_for_scene
    )
except Exception:
    logging.warning("⚠️ ElevenLabs client not found. Voiceover generation will fail.")
    generate_audio_for_scene = None

try:
    from video_clients.replicate_client import (
        generate_video_scene_with_replicate, 
        generate_lip_sync_with_replicate
    )
except Exception:
    logging.warning("⚠️ Replicate client wrappers not found. Using internal replicate_run_safe wrappers.")
    generate_video_scene_with_replicate = None
    generate_lip_sync_with_replicate = None

# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
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

# Replicate client (stateful)
if replicate and REPLICATE_API_TOKEN:
    replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)
else:
    replicate_client = None
    if not replicate:
        logging.warning("⚠️ replicate package not installed.")
    elif not REPLICATE_API_TOKEN:
        logging.warning("⚠️ REPLICATE_API_TOKEN not set - replicate calls will fail.")

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

# -------------------------
# Helpers
# -------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if not os.path.exists(path):
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
# Replicate safe runner
# -------------------------
def normalize_replicate_output(raw):
    """
    Replicate API can return:
      - a string URL,
      - a list of strings,
      - a dict with 'url' or path,
      - nested structures.
    Normalize to a single URL string if possible; otherwise return str(raw).
    """
    try:
        if raw is None:
            return None
        if isinstance(raw, str):
            return raw
        if isinstance(raw, (list, tuple)) and raw:
            # find first string-looking element
            for v in raw:
                if isinstance(v, str):
                    return v
                if isinstance(v, dict) and ("url" in v or "output" in v):
                    return v.get("url") or v.get("output")
            return str(raw[0])
        if isinstance(raw, dict):
            for key in ("url", "output", "result", "video", "file"):
                if key in raw:
                    return raw.get(key)
            # sometimes outputs are nested lists
            for v in raw.values():
                if isinstance(v, str):
                    return v
                if isinstance(v, (list, tuple)) and v:
                    if isinstance(v[0], str):
                        return v[0]
            return json.dumps(raw)
        return str(raw)
    except Exception:
        return str(raw)

def get_latest_model_version_id(model_name: str) -> Optional[str]:
    if not replicate_client:
        raise RuntimeError("Replicate client not configured")
    try:
        model = replicate_client.models.get(model_name)
        versions = model.versions.list()
        if not versions:
            return None
        # versions.list() returns iterable-like; pick first/latest
        latest = versions[0].id if hasattr(versions[0], "id") else str(versions[0])
        return latest
    except Exception as e:
        logging.error(f"Unable to fetch versions for {model_name}: {e}")
        return None

def replicate_run_safe(model_name: str, version: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Run a Replicate model safely. If the provided version fails due to "invalid version or not permitted",
    attempt to fetch the latest version and retry once.
    Returns a normalized URL/string or raises.
    """
    if not replicate_client:
        raise RuntimeError("Replicate client not configured")

    tried_latest = False
    attempt_version = version
    for attempt in range(2):
        try:
            if attempt_version:
                model_ref = f"{model_name}:{attempt_version}"
            else:
                latest = get_latest_model_version_id(model_name)
                if not latest:
                    raise RuntimeError(f"No versions available for {model_name}")
                model_ref = f"{model_name}:{latest}"
                attempt_version = latest
                tried_latest = True

            logging.info(f"Replicate: running {model_ref} with input keys: {list(kwargs.get('input', {}).keys()) if 'input' in kwargs else 'N/A'}")
            # Use client.run to keep parity with earlier behavior
            raw = replicate_client.run(model_ref, **kwargs) if hasattr(replicate_client, "run") else replicate_client.predict(model_ref, **kwargs)
            url = normalize_replicate_output(raw)
            logging.info(f"Replicate returned: {url}")
            return url
        except Exception as e:
            msg = str(e).lower()
            logging.error(f"Replicate run failed for {model_name} (version={attempt_version}): {e}")
            # Detect invalid version / permission error from message or HTTP detail
            if ("invalid version" in msg or "not permitted" in msg or "does not exist" in msg or "422" in msg) and not tried_latest:
                logging.info("Detected invalid or not permitted version; fetching latest and retrying.")
                attempt_version = get_latest_model_version_id(model_name)
                if not attempt_version:
                    raise
                tried_latest = True
                continue
            raise

# -------------------------
# Character DB
# -------------------------
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
# Image generation (Flux) - safe wrapper
# -------------------------
def generate_flux_image(prompt: str, aspect: str = "16:9") -> str:
    """
    Uses replicate model black-forest-labs/flux-schnell (or similar).
    If replicate_client is missing, raises.
    """
    try:
        model_name = "black-forest-labs/flux-schnell"
        input_payload = {"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg"}
        url = replicate_run_safe(model_name, input=input_payload)
        if not url:
            raise RuntimeError("Flux returned empty response")
        return str(url)
    except Exception as e:
        logging.error(f"Flux image generation failed: {e}")
        raise

# -------------------------
# Single scene processor (robust)
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
        sleep_time = random.randint(2, 8)  # shorten random to be snappier
        time.sleep(sleep_time)

        target_face_url = None
        visual_prompt = (scene.get("visual_prompt") or "").lower()

        for char_name, face_url in character_faces.items():
            if char_name.lower() in visual_prompt:
                target_face_url = face_url
                break
        if not target_face_url and character_faces:
            target_face_url = list(character_faces.values())[0]

        shot_type = (scene.get("shot_type") or "medium").lower()
        has_dialogue = bool(scene.get("audio_narration") and len(scene.get("audio_narration")) > 2)
        is_cinematic_shot = "wide" in shot_type or "drone" in shot_type or "establish" in shot_type or not has_dialogue

        video_url = None

        # If user supplied mp4
        if scene.get("image_url") and str(scene.get("image_url")).endswith(".mp4"):
            temp_path = os.path.join(temp_dir, f"user_upload_{index}.mp4")
            download_to_file(scene.get("image_url"), temp_path)
            return {"index": index, "video_path": temp_path, "status": "success"}

        # Lip sync
        if not is_cinematic_shot and audio_path and (generate_lip_sync_with_replicate or replicate_client) and target_face_url:
            # Upload audio to cloudinary so replicate can fetch (some models require accessible URL)
            cloud_audio_url = None
            try:
                cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
            except Exception as e:
                logging.error(f"Audio upload for lip-sync failed: {e}")
                cloud_audio_url = None

            # prefer wrapper if present else call replicate directly
            try:
                if generate_lip_sync_with_replicate:
                    res = generate_lip_sync_with_replicate(image_url=target_face_url, audio_url=cloud_audio_url)
                    video_url = normalize_replicate_output(res)
                else:
                    # use sadtalker model - auto-detect latest if version causes error
                    model_name = "lucataco/sadtalker"
                    input_payload = {
                        "source_image": target_face_url,
                        "driven_audio": cloud_audio_url,
                        "still": True,
                        "enhancer": "gfpgan",
                        "preprocess": "full",
                        "expression_scale": 1.1,
                        "ref_eyeblink": None,
                        "ref_pose": None
                    }
                    video_url = replicate_run_safe(model_name, input=input_payload)
            except Exception as e:
                logging.error(f"Lip-sync generation failed for scene {index}: {e}")
                video_url = None

        # Cinematic B-roll
        if not video_url:
            action = f"{scene.get('action_prompt','cinematic movement')}, camera:{scene.get('camera_angle','35mm')}"
            start_image = target_face_url or None
            if not start_image:
                # generate cinematic starting image
                start_image = generate_flux_image(f"{scene.get('visual_prompt','cinematic scene')}, cinematic lighting, ultra-detailed", aspect=aspect)
            # prefer wrapper if available
            try:
                if generate_video_scene_with_replicate:
                    res = generate_video_scene_with_replicate(prompt=action, image_url=start_image, aspect=aspect)
                    video_url = normalize_replicate_output(res)
                else:
                    # fallback: example WAN-like model name; replace with your actual model if different
                    model_name = "wan/wan-video"  # <-- replace with actual model if you have
                    input_payload = {"prompt": action, "image": start_image, "aspect": aspect}
                    video_url = replicate_run_safe(model_name, input=input_payload)
            except Exception as e:
                logging.error(f"Cinematic generation failed for scene {index}: {e}")
                video_url = None

        if not video_url:
            raise RuntimeError("Video generation returned empty result")

        temp_video_path = os.path.join(temp_dir, f"scene_gen_{index}.mp4")
        download_to_file(str(video_url), temp_video_path, timeout=300)

        return {"index": index, "video_path": temp_video_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {traceback.format_exc()}")
        return {"index": index, "video_path": None, "status": "failed"}
    finally:
        # keep temp_dir for debugging if failure; remove only on success to help debugging logs if needed
        if os.path.exists(temp_dir):
            # if success remove, otherwise keep for investigation
            try:
                # determine success by checking for output files
                # remove if any video file exists -> assume success cleanup
                files = os.listdir(temp_dir)
                if any(f.endswith(".mp4") for f in files):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

# -------------------------
# Stitching (unchanged but with better logging)
# -------------------------
def stitch_video_audio_pairs_optimized(scene_pairs: List[Tuple[str, str]], output_path: str) -> bool:
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"render_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []
    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for Request ID: {request_id}")
        for i, (video, audio) in enumerate(scene_pairs):
            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            audio_dur = get_media_duration(audio)
            if audio_dur == 0:
                logging.warning(f"Audio duration is 0 for {audio}, skipping chunk.")
                continue
            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(audio_dur),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-vf", "scale='min(720,iw)':-2",
                "-preset", "ultrafast",
                "-crf", "28",
                "-c:a", "aac",
                "-shortest",
                chunk_name
            ]
            logging.info(f"Running ffmpeg chunk cmd for scene {i}: {' '.join(cmd[:6])} ...")
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        if not chunk_paths:
            logging.error("No chunk paths created; nothing to concatenate.")
            return False

        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        logging.info("Concatenating chunks...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", input_list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)

        logging.info(f"✅ Video successfully saved to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: returncode={getattr(e,'returncode',None)} stdout={getattr(e,'stdout',None)} stderr={getattr(e,'stderr',None)}")
        return False
    except Exception as e:
        logging.error(f"General stitching error: {e}")
        return False
    finally:
        # cleanup chunk files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass

# -------------------------
# Scraping & Storyboard + OpenAI wrappers (kept similar)
# -------------------------
def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee", max_results: int = 3) -> List[dict]:
    results = []
    logging.info(f"Scraping YouTube for '{keyword}' provider={provider}")
    try:
        if provider.lower() == "scrapingbee":
            if not SCRAPINGBEE_API_KEY:
                 logging.warning("SCRAPINGBEE_API_KEY missing, skipping scrape.")
                 return []
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(keyword)}"
            params = {"api_key": SCRAPINGBEE_API_KEY, "url": url, "render_js": "false"}
            r = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=30)
            if r.status_code == 200:
                # For now return empty; user may parse later
                pass
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    return results

def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {"hook_style": "intrigue", "avg_scene_count": 6, "tone": "motivational"}

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
        system_content = "You are a professional screenwriter."
        if is_json:
            system_content += " You must output valid JSON."
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[
                {"role":"system","content": system_content},
                {"role":"user","content":prompt_content}
            ],
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
        prompt_template = """
        TASK: Create an original short film script for '$keyword'.
        Output valid JSON with keys: video_title, video_description, main_character_profile, characters (list), scenes (list).
        """
    template = Template(prompt_template)
    full_context = keyword
    if form_data.get("characters"):
        full_context += f"\n\nUSER DEFINED CHARACTERS:\n{form_data.get('characters')}"

    target_scenes = form_data.get("max_scenes", 7)
    prompt = template.safe_substitute(
        keyword=full_context,
        blueprint_json=json.dumps(blueprint),
        language=form_data.get("language", "english"),
        video_type=form_data.get("video_type", "reel"),
        uploaded_assets_context="User uploaded images present" if form_data.get("uploaded_images") else "No uploads",
        max_scenes=str(target_scenes)
    )
    prompt += f"\n\nIMPORTANT: The 'audio_narration' across all scenes MUST total at least 150 words. Do not write short scripts. Generate exactly {target_scenes} scenes."
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or (json.loads(raw) if raw else {})
    if not obj:
        raise RuntimeError("Storyboard generation failed")
    try:
        return StoryboardSchema(**obj).dict()
    except ValidationError:
        return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

def refine_script_with_roles(storyboard: dict, form_data: dict) -> List[dict]:
    characters = storyboard.get('characters', [])
    full_script_text = " ".join([s.get('audio_narration', '') for s in storyboard.get('scenes', [])])
    prompt = f"""
    You are a Voice Director.
    TASK: Convert this raw script into a STRICT JSON list of audio segments.
    REAL CHARACTERS: {json.dumps(characters)}
    RAW SCRIPT: "{full_script_text}"
    OUTPUT FORMAT EXAMPLE:
    [
      {{"speaker": "Hero", "text": "Let's go."}},
      {{"speaker": "Narrator", "text": "They left."}}
    ]
    """
    try:
        raw = get_openai_response(prompt, temperature=0.1, is_json=True)
        segments = extract_json_list_from_text(raw) or []
    except Exception:
        segments = []
    if not isinstance(segments, list) or len(segments) == 0:
        return [{"speaker": "Narrator", "text": full_script_text, "voice_id": form_data.get("voice_selection")}]
    final_segments = []
    main_voice_id = form_data.get("voice_selection") or "21m00Tcm4TlvDq8ikWAM"
    for seg in segments:
        speaker_name = seg.get("speaker", "Narrator")
        text = seg.get("text", "")
        if not text: continue
        voice = main_voice_id 
        char_obj = next((c for c in characters if c.get("name") in speaker_name or speaker_name in c.get("name")), None)
        if char_obj and char_obj.get("voice_id"):
            voice = char_obj.get("voice_id")
        final_segments.append({"text": text, "voice_id": voice})
    return final_segments

def generate_thumbnail_agent(storyboard: dict, orientation: str = "16:9") -> Optional[str]:
    summary = storyboard.get("video_description") or "Video"
    prompt = f"Movie poster for: {summary}. High quality, 8k, textless."
    try:
        return generate_flux_image(prompt, aspect=orientation)
    except Exception:
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
# Celery Task
# -------------------------
@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] SaaS Task started.")
    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(msg)
        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")
        update_status("Designing Video Concept...")
        scraped = scrape_youtube_videos(keyword)
        blueprint = analyze_competitors(scraped)
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)
        characters = storyboard.get("characters") or []
        uploaded_images = form_data.get("uploaded_images") or []
        character_faces = {}
        for i, url in enumerate(uploaded_images):
            if i < len(characters):
                char_name = characters[i].get("name", f"Char_{i}")
                character_faces[char_name] = url
        for char in characters:
            name = char.get("name", "Unknown")
            if name not in character_faces:
                update_status(f"Casting {name}...")
                desc = char.get("appearance_prompt") or f"Cinematic portrait of {name}"
                aspect_ratio = "9:16" if form_data.get("video_type") == "reel" else "16:9"
                try:
                    face_url = generate_flux_image(f"{desc}, facing camera, high detailed, 8k", aspect=aspect_ratio)
                    character_faces[name] = face_url
                except Exception as e:
                    logging.error(f"Failed to generate face for {name}: {e}")
        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"
        update_status("Synthesizing Audio Dialogue...")
        segments = refine_script_with_roles(storyboard, form_data)
        scene_assets = []
        full_script_text = ""
        scenes = storyboard.get("scenes", [])
        for i, scene in enumerate(scenes):
            text = segments[i].get("text") if i < len(segments) else "..."
            voice_id = segments[i].get("voice_id") if i < len(segments) else "21m00Tcm4TlvDq8ikWAM"
            if voice_id:
                voice_id = voice_id.strip(" []'\"")
            full_script_text += text + " "
            if generate_audio_for_scene:
                try:
                    audio_path = generate_audio_for_scene(text, voice_id)
                except Exception as e:
                    logging.error(f"Audio generation failed for scene {i}: {e}")
                    audio_path = None
            else:
                audio_path = None
            if audio_path:
                duration = get_media_duration(audio_path)
                scene_assets.append({
                    "index": i,
                    "audio_path": audio_path,
                    "duration": duration,
                    "scene_data": scene
                })
        if not scene_assets:
            raise RuntimeError("Audio generation failed for all scenes.")
        update_status("Rendering Video Scenes (Synced)...")
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"
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
                    aspect
                ): asset
                for asset in scene_assets
            }
            results_map = {}
            for future in concurrent.futures.as_completed(future_to_asset):
                asset = future_to_asset[future]
                idx = asset["index"]
                try:
                    res = future.result()
                except Exception as e:
                    logging.error(f"Scene {idx} raised exception: {e}")
                    res = {"index": idx, "video_path": None, "status": "failed"}
                if res["status"] == "success" and res["video_path"]:
                    results_map[idx] = (res["video_path"], asset["audio_path"])
                else:
                    logging.warning(f"Scene {idx} video failed. Skipping.")
            for i in range(len(scenes)):
                if i in results_map:
                    final_pairs.append(results_map[i])
        if not final_pairs:
            raise RuntimeError("Video generation failed for all scenes.")
        update_status("Final Assembly (Audio-Video Sync)...")
        music_tone = blueprint.get("tone", "motivational")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try:
            download_to_file(music_url, music_path)
        except Exception:
            music_path = None
        final_output_path = f"/tmp/final_render_{task_id}.mp4"
        temp_visual_path = f"/tmp/visual_base_{task_id}.mp4"
        success = stitch_video_audio_pairs_optimized(final_pairs, temp_visual_path)
        if not success:
            raise RuntimeError("Stitching failed.")
        if music_path:
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_visual_path,
                "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "libx264", "-vf", "scale='min(720,iw)':-2", "-preset", "ultrafast", "-crf", "28",
                "-c:a", "aac", "-shortest",
                final_output_path
            ]
            subprocess.run(cmd, check=True)
        else:
            shutil.move(temp_visual_path, final_output_path)
        update_status("Uploading & Finalizing...")
        final_video_url = safe_upload_to_cloudinary(final_output_path, folder="final_videos")
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        metadata = youtube_metadata_agent(full_script_text, keyword, form_data, blueprint)
        try:
            if music_path and os.path.exists(music_path): os.remove(music_path)
            if os.path.exists(temp_visual_path): os.remove(temp_visual_path)
            if os.path.exists(final_output_path): os.remove(final_output_path)
            for v, a in final_pairs:
                if os.path.exists(v): os.remove(v)
                if a and os.path.exists(a): os.remove(a)
        except Exception:
            pass
        return {
            "status": "ready",
            "video_url": final_video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard
        }
    except Exception as e:
        logging.error(f"Task failed: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
