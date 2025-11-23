# worker.py
# YouTube Automation Worker v6.0 - Full end-to-end pipeline (Celery)
# - Requirements: celery, openai, replicate, cloudinary, tenacity, requests, pydantic, pillow (optional), youtube-transcript-api (optional)
# - Place this file in your worker service and configure env vars described below.

import os
import sys
import logging
import json
import re
import uuid
import shutil
import tempfile
import traceback
from pathlib import Path
from string import Template
from typing import Optional, List, Dict, Any
import concurrent.futures
import subprocess
import time

import requests
import cloudinary
import cloudinary.uploader
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from pydantic import BaseModel, ValidationError

# Celery app init - ensure celery_init.py exports "celery"
from celery_init import celery

# OpenAI & Replicate clients (wrappers)
from openai import OpenAI
import replicate

# Your existing video clients - adapt paths if different
from video_clients.elevenlabs_client import generate_voiceover_and_upload  # returns public mp3 URL
from video_clients.replicate_client import generate_video_scene_with_replicate  # accepts prompt, image_url, aspect

# ---------------------------------------------------------
# ---------------- Configuration & Logging ----------------
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Cloudinary config
if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )
else:
    logging.warning("Cloudinary not fully configured; uploads will fail.")

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Developer-provided uploaded file path (from conversation history)
LOCAL_UPLOADED_FILE = "/mnt/data/v4afissrdr5krnzvpfvr.mp4"

# ---------------------------------------------------------
# ---------------- Pydantic Schemas -----------------------
# ---------------------------------------------------------

class SceneSchema(BaseModel):
    scene_index: int
    duration: float  # seconds
    visual_prompt: str
    action_prompt: str
    audio_narration: Optional[str]
    shot_type: Optional[str] = "medium"  # closeup|medium|wide
    camera: Optional[str] = "35mm"
    lighting: Optional[str] = "soft"
    emotion: Optional[str] = "neutral"
    # optional override: use pre-uploaded image/video instead of generating
    image_url: Optional[str] = None
    video_url: Optional[str] = None

class StoryboardSchema(BaseModel):
    video_title: str
    video_description: Optional[str] = ""
    main_character_profile: Optional[str] = ""
    scenes: List[SceneSchema]

# ---------------------------------------------------------
# ---------------- Utility Helpers -----------------------
# ---------------------------------------------------------

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
                if chunk:
                    f.write(chunk)
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

# ---------------------------------------------------------
# ---------------- Scraping Helpers -----------------------
# ---------------------------------------------------------

def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee", max_results: int = 8) -> List[dict]:
    """
    Scrape top youtube results for a keyword using ScrapingBee or SerpAPI (user chooses).
    Returns list of simplified objects: {'videoId','title','description','thumbnail','duration','url'}
    """
    results = []
    logging.info(f"Scraping YouTube for '{keyword}' using provider={provider}")
    if provider.lower() == "scrapingbee":
        if not SCRAPINGBEE_API_KEY:
            raise RuntimeError("SCRAPINGBEE_API_KEY missing")
        url = f"https://www.youtube.com/results?search_query={requests.utils.quote(keyword)}"
        params = {"api_key": SCRAPINGBEE_API_KEY, "url": url, "render_js": "false"}
        r = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=30)
        r.raise_for_status()
        html = r.text
        # Extract ytInitialData JSON
        match = re.search(r"ytInitialData\s*=\s*(\{.*\});", html, re.DOTALL)
        if not match:
            # fallback: attempt to find JSON snippet
            match = re.search(r"var ytInitialData = (\{.*\})", html, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                # collect videoRenderer entries (helper traversal)
                def collect_vr(node):
                    if isinstance(node, dict):
                        if "videoRenderer" in node:
                            vr = node["videoRenderer"]
                            try:
                                vid = vr.get("videoId")
                                title = "".join([r.get("text","") for r in vr.get("title", {}).get("runs", [])]) if vr.get("title") else vr.get("title", {}).get("simpleText", "")
                                desc = vr.get("shortDescription", {}).get("simpleText", "") if vr.get("shortDescription") else ""
                                thumb = vr.get("thumbnail", {}).get("thumbnails", [{}])[-1].get("url")
                                length = vr.get("lengthText", {}).get("simpleText") if vr.get("lengthText") else None
                                results.append({"videoId": vid, "title": title, "description": desc, "thumbnail": thumb, "duration": length, "url": f"https://www.youtube.com/watch?v={vid}"})
                            except Exception:
                                pass
                        for v in node.values():
                            collect_vr(v)
                    elif isinstance(node, list):
                        for item in node:
                            collect_vr(item)
                collect_vr(data)
            except Exception:
                logging.exception("Failed to parse ytInitialData")
    elif provider.lower() == "serpapi":
        if not SERPAPI_KEY:
            raise RuntimeError("SERPAPI_KEY missing")
        params = {
            "engine": "youtube",
            "search_query": keyword,
            "api_key": SERPAPI_KEY,
            "num": max_results
        }
        r = requests.get("https://serpapi.com/search", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data.get("video_results", [])[:max_results]:
            results.append({"videoId": item.get("id"), "title": item.get("title"), "description": item.get("description"), "thumbnail": item.get("thumbnail"), "duration": item.get("duration"), "url": item.get("link")})
    else:
        raise ValueError("Unknown scrape provider")
    # dedupe & return top N
    seen = set()
    filtered = []
    for r in results:
        vid = r.get("videoId") or r.get("url")
        if vid and vid not in seen:
            filtered.append(r)
            seen.add(vid)
        if len(filtered) >= max_results:
            break
    logging.info(f"Scraped {len(filtered)} videos")
    return filtered

def fetch_captions_for_video(video_id: str, lang="en"):
    """
    Attempt to fetch captions via youtube timedtext endpoint through ScrapingBee.
    Returns list of caption dicts: [{'start', 'dur', 'text'}]
    """
    if not SCRAPINGBEE_API_KEY:
        return []
    url = f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}"
    r = requests.get("https://app.scrapingbee.com/api/v1/", params={"api_key": SCRAPINGBEE_API_KEY, "url": url}, timeout=20)
    if r.status_code != 200:
        return []
    text = r.text
    items = []
    for m in re.finditer(r'<text start="(?P<start>[^"]+)" dur="(?P<dur>[^"]*)">(?P<t>.*?)</text>', text, re.DOTALL):
        t = m.group("t")
        t = re.sub(r'&#39;|&quot;|&amp;', lambda s: {"&#39;":"'","&quot;":'"',"&amp;":"&"}[s.group(0)], t)
        items.append({"start": float(m.group("start")), "dur": float(m.group("dur") or 0), "text": re.sub(r'\s+', ' ', t).strip()})
    return items

# ---------------------------------------------------------
# ---------------- Analysis / Blueprint -------------------
# ---------------------------------------------------------

def analyze_competitors(scraped_videos: List[dict], fetch_subs: bool = True) -> Dict[str, Any]:
    """
    Divide & conquer analyzer: inspects scraped videos and returns a blueprint:
      - hook_style
      - avg_scene_count
      - avg_scene_duration
      - tone
      - recommended_length_range
    """
    logging.info("Analyzing competitors for blueprint")
    hooks = []
    tones = []
    scene_counts = []
    durations = []
    for v in scraped_videos:
        title = (v.get("title") or "").lower()
        desc = (v.get("description") or "").lower()
        if any(k in title for k in ["you won't believe", "shocking", "shocked", "surprising", "surprise", "don't miss"]):
            hooks.append("shock")
        elif any(k in title for k in ["how to", "learn", "tutorial", "tips"]):
            hooks.append("question")
        else:
            hooks.append("intrigue")
        tones.append("emotional" if any(w in (title+desc) for w in ["cry","emotion","father","mother","loss","lost","pain","feel"]) else "motivational")
        # duration heuristics (parse "12:34")
        dur = v.get("duration")
        if isinstance(dur, str) and ":" in dur:
            parts = [int(p) for p in dur.split(":")]
            seconds = parts[-1] + (parts[-2]*60 if len(parts)>1 else 0)
            durations.append(seconds)
            scene_counts.append(max(3, int(seconds // 8)))  # rough heuristic
        else:
            durations.append(180)
            scene_counts.append(5)
    blueprint = {
        "hook_style": max(set(hooks), key=hooks.count) if hooks else "intrigue",
        "avg_scene_count": int(sum(scene_counts)/len(scene_counts)) if scene_counts else 5,
        "avg_scene_duration": max(3, int(sum(durations)/len(durations)/ (sum(scene_counts)/len(scene_counts)) )) if durations else 5,
        "tone": max(set(tones), key=tones.count) if tones else "motivational",
        "recommended_length_seconds": int(sum(durations)/len(durations)) if durations else 180
    }
    logging.info(f"Blueprint: {blueprint}")
    return blueprint

# ---------------------------------------------------------
# ---------------- Storyboard / Script --------------------
# ---------------------------------------------------------

def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    if not os.path.exists(path):
        logging.warning(f"Prompt file missing: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI client not configured")
    response_format = {"type": "json_object"} if is_json else {"type": "text"}
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","gpt-4o"),
            messages=[{"role":"system","content":"You are a professional screenwriter and video producer."},{"role":"user","content":prompt_content}],
            temperature=temperature,
            response_format=response_format
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        return ""

def create_video_storyboard_agent(keyword: str, blueprint: dict, form_data: dict) -> dict:
    prompt_template = load_prompt_template("prompt_story_generator.txt")
    if not prompt_template:
        # fallback short prompt
        prompt_template = """
You are a professional short film writer. 
BLUEPRINT: $blueprint_json
TASK: Create an original short film script for the keyword: $keyword.
Output as JSON with keys: video_title, video_description, main_character_profile, scenes (list).
Each scene: scene_index, duration, visual_prompt, action_prompt, audio_narration, shot_type, camera, lighting, emotion.
"""
    template = Template(prompt_template)
    prompt = template.safe_substitute(blueprint_json=json.dumps(blueprint), keyword=keyword, video_style=form_data.get("video_style","Cinematic"))
    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    # extract JSON
    try:
        obj = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else extract_json_from_text(raw) or json.loads(raw)
    except Exception:
        obj = extract_json_from_text(raw)
    if not obj:
        raise RuntimeError("Failed to generate storyboard")
    # validate with pydantic
    try:
        sb = StoryboardSchema(**obj)
        return sb.dict()
    except ValidationError as e:
        logging.warning("Storyboard validation error, trying to repair: " + str(e))
        # attempt minimal repair: ensure scenes exist
        if isinstance(obj, dict) and "scenes" in obj:
            # coerce scene list
            scenes = []
            for i, s in enumerate(obj.get("scenes", [])):
                try:
                    scene = SceneSchema(scene_index=int(s.get("scene_index", i)), duration=float(s.get("duration", 5)), visual_prompt=s.get("visual_prompt",""), action_prompt=s.get("action_prompt",""), audio_narration=s.get("audio_narration",""))
                    scenes.append(scene.dict())
                except Exception:
                    continue
            repaired = {
                "video_title": obj.get("video_title", f"{keyword} - Short Film"),
                "video_description": obj.get("video_description",""),
                "main_character_profile": obj.get("main_character_profile",""),
                "scenes": scenes
            }
            return repaired
        raise

def extract_json_from_text(text: str) -> Optional[dict]:
    if not text: return None
    m = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # fallback: try to find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    return None

# ---------------------------------------------------------
# ---------------- Character DB / Consistency -------------
# ---------------------------------------------------------
CHAR_DB_PATH = os.getenv("CHAR_DB_PATH", "/var/data/character_db.json")
ensure_dir(str(Path(CHAR_DB_PATH).parent))

def load_char_db() -> dict:
    try:
        with open(CHAR_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_char_db(db: dict):
    with open(CHAR_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

def ensure_character(name: str, appearance_prompt: Optional[str] = None, reference_image_url: Optional[str] = None, voice_id: Optional[str] = None) -> dict:
    db = load_char_db()
    if name in db:
        return db[name]
    char_id = str(uuid.uuid4())
    db[name] = {
        "id": char_id,
        "name": name,
        "appearance_prompt": appearance_prompt or f"{name}, South Asian male, 30s, kind face, photorealistic",
        "reference_image": reference_image_url,
        "voice_id": voice_id or "21m00Tcm4TlvDq8ikWAM"
    }
    save_char_db(db)
    return db[name]

# ---------------------------------------------------------
# ---------------- Scene Generation -----------------------
# ---------------------------------------------------------

def generate_flux_image(prompt: str, aspect: str = "16:9") -> str:
    """
    Generate a keyframe image via Replicate Flux Schnell. Returns image URL (string).
    """
    logging.info(f"Generating keyframe: {prompt[:70]}...")
    try:
        # Using replicate.run. Keep inputs minimal but include aspect
        output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt, "aspect_ratio": aspect, "output_format":"jpg", "output_quality":90})
        # Output often list; cast to str
        image_url = str(output[0]) if isinstance(output, (list, tuple)) else str(output)
        return image_url
    except Exception as e:
        logging.error(f"Flux image generation error: {e}")
        raise

def process_single_scene(scene: dict, index: int, character_profile: str, aspect: str = "16:9") -> (int, Optional[str]):
    """
    Generate one scene: keyframe image -> replicate video scene generation -> upload -> return public url
    """
    try:
        logging.info(f"Processing scene {index}")
        # If the storyboard supplies an image_url (pre-uploaded), use it
        if scene.get("image_url"):
            # If image_url is already a video, optionally upload or pass through
            if scene.get("image_url").endswith(".mp4"):
                # pass the video through cloudinary if needed, otherwise return direct path
                return (index, scene.get("image_url"))
            else:
                # generate motion from image prompt
                keyframe_url = scene.get("image_url")
        else:
            visual_setting = scene.get("visual_prompt", "")
            full_image_prompt = f"{character_profile}, {visual_setting}, ultra-detailed, photorealistic, cinematic lighting"
            keyframe_url = generate_flux_image(full_image_prompt, aspect=aspect)

        # Action prompt used for motion; include camera & emotion
        action = scene.get("action_prompt", "") + f", camera:{scene.get('camera','35mm')}, shot:{scene.get('shot_type','medium')}, emotion:{scene.get('emotion','neutral')}"
        # generate motion scene using replicate client wrapper (user implemented)
        video_url = generate_video_scene_with_replicate(prompt=action, image_url=keyframe_url, aspect=aspect)
        logging.info(f"Scene {index} video_url: {video_url}")
        return (index, video_url)
    except Exception as e:
        logging.exception(f"Scene {index} failed: {e}")
        return (index, None)

# ---------------------------------------------------------
# ---------------- Assembly / FFmpeg ----------------------
# ---------------------------------------------------------

def normalize_video_to_mp4(input_path: str, output_path: str, width: int = 1280, height: int = 720, fps: int = 25):
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-ac", "2", "-ar", "44100",
        output_path
    ]
    run_subprocess(cmd)
    return output_path

def concat_videos_safe(input_paths: List[str], output_path: str, width: int = 1280, height: int = 720, fps: int = 25):
    tmpdir = tempfile.mkdtemp(prefix="concat_")
    try:
        normalized = []
        for i, p in enumerate(input_paths):
            norm = os.path.join(tmpdir, f"norm_{i}.mp4")
            normalize_video_to_mp4(p, norm, width, height, fps)
            normalized.append(norm)
        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for n in normalized:
                f.write(f"file '{n}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", output_path]
        run_subprocess(cmd)
        return output_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def merge_audio_video(video_path: str, audio_path: str, output_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
    ]
    run_subprocess(cmd)
    return output_path

def burn_subtitles(input_video: str, srt_path: str, output_video: str):
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", f"subtitles={srt_path}:force_style='Fontsize=36,PrimaryColour=&H00FFFFFF&'", "-c:a", "copy", output_video]
    run_subprocess(cmd)
    return output_video

def srt_from_narration(scenes: List[dict], out_path: str):
    """
    Very basic SRT builder using scene durations sequentially.
    """
    start = 0.0
    idx = 1
    with open(out_path, "w", encoding="utf-8") as f:
        for s in scenes:
            dur = max(1.0, float(s.get("duration", 3)))
            end = start + dur
            # format times
            def fmt(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                sss = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{sss:02},{ms:03}"
            text = s.get("audio_narration", "") or ""
            f.write(f"{idx}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
            start = end
            idx += 1
    return out_path

def video_assembly_agent(scene_urls: List[str], voiceover_url: str, storyboard: dict, aspect: str = "16:9"):
    logging.info("Assembling final video...")
    tmpdir = tempfile.mkdtemp(prefix="assemble_")
    try:
        # download scene video files
        local_scene_paths = []
        for i, url in enumerate(scene_urls):
            if not url:
                continue
            local_path = os.path.join(tmpdir, f"scene_{i}.mp4")
            download_to_file(url, local_path)
            local_scene_paths.append(local_path)

        if not local_scene_paths:
            raise RuntimeError("No scene videos downloaded")

        # download voiceover
        local_voice = os.path.join(tmpdir, "voice.mp3")
        download_to_file(voiceover_url, local_voice)

        # decide output resolution
        if aspect == "16:9":
            width, height = 1280, 720
        else:  # "9:16"
            width, height = 720, 1280

        # concat (normalize then concat)
        concat_output = os.path.join(tmpdir, "concat_out.mp4")
        concat_videos_safe(local_scene_paths, concat_output, width=width, height=height, fps=25)

        # merge voiceover
        merged_output = os.path.join(tmpdir, "merged.mp4")
        merge_audio_video(concat_output, local_voice, merged_output)

        # generate SRT from narration
        srt_path = os.path.join(tmpdir, "captions.srt")
        srt_from_narration(storyboard.get("scenes", []), srt_path)

        # burn subtitles
        final_out = os.path.join(tmpdir, "final_burned.mp4")
        burn_subtitles(merged_output, srt_path, final_out)

        # upload final
        cloud_url = safe_upload_to_cloudinary(final_out, resource_type="video", folder="final_videos")
        return cloud_url
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------------------------------------------------
# ---------------- Thumbnail & Metadata -------------------
# ---------------------------------------------------------

def generate_thumbnail_agent(storyboard: dict) -> Optional[str]:
    # Simple approach: generate close-up keyframe and composite with text (Pillow or OpenAI image)
    prompt = f"Thumbnail closeup: {storyboard.get('main_character_profile','')}, expression: surprised/emotional, cinematic lighting, 3:2 crop"
    try:
        # Use OpenAI or replicate to generate image or use Flux
        img_url = generate_flux_image(prompt, aspect="16:9")
        # Optionally overlay text with Pillow (left as TODO)
        # Save & upload to cloudinary (if needed we already have URL)
        return img_url
    except Exception as e:
        logging.warning(f"Thumbnail generation failed: {e}")
        return None

def youtube_metadata_agent(full_script: str, keyword: str) -> dict:
    prompt_template = load_prompt_template("prompt_youtube_metadata_generator.txt")
    if not prompt_template:
        prompt_template = "Create 5 titles, 1 description (200-300 words), 15 tags for this script: $script"
    prompt = Template(prompt_template).safe_substitute(script=full_script, keyword=keyword)
    res = get_openai_response(prompt, temperature=0.4, is_json=True)
    meta = extract_json_from_text(res) or {}
    return meta

# ---------------------------------------------------------
# ---------------- Celery Task (Main Pipeline) -------------
# ---------------------------------------------------------

@celery.task(bind=True)
def background_generate_video(self, form_data: dict):
    """
    form_data expected fields:
      - keyword: str
      - language: "english" | "urdu"
      - video_type: "reel" | "youtube"   # reel -> 30-60s ; youtube -> 60-300s
      - scrape_provider: "scrapingbee" | "serpapi"
      - uploaded_images: [urls] (optional)  # optional array of user-uploaded images to use for characters
      - use_local_asset: bool (if true, use LOCAL_UPLOADED_FILE as scene 0 B-roll)
      - voice_selection: optional voice id
      - max_scenes: optional int
    """
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"Task {task_id} started. form_data keys: {list(form_data.keys())}")
    try:
        # 0. Validate input
        keyword = form_data.get("keyword")
        if not keyword:
            raise ValueError("keyword is required")
        language = (form_data.get("language") or "english").lower()
        video_type = (form_data.get("video_type") or "reel").lower()  # reel or youtube
        scrape_provider = (form_data.get("scrape_provider") or "scrapingbee").lower()
        use_local_asset = bool(form_data.get("use_local_asset", False))
        uploaded_images = form_data.get("uploaded_images") or []
        voice_selection = form_data.get("voice_selection")
        max_scenes = int(form_data.get("max_scenes", 6))

        # Report progress
        def update_progress(msg, step=0):
            try:
                self.update_state(state="PROGRESS", meta={"message": msg})
            except Exception:
                pass
            logging.info(msg)

        update_progress("Checking FFmpeg...")
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
            update_progress("FFmpeg OK")
        except Exception:
            update_progress("FFmpeg missing or not executable", 1)
            raise RuntimeError("FFmpeg missing on server")

        # 1. Scrape
        update_progress("Scraping YouTube (competitors)...", 1)
        scraped = scrape_youtube_videos(keyword, provider=scrape_provider, max_results=8)
        # optional: fetch captions for top N
        for v in scraped[:3]:
            subs = fetch_captions_for_video(v.get("videoId"), lang="en")
            v["captions"] = subs

        # 2. Analyze blueprint
        update_progress("Analyzing competitors & building blueprint...", 2)
        blueprint = analyze_competitors(scraped)

        # Adjust blueprint according to requested video_type
        if video_type == "reel":
            target_length = 30 + (30 * 0.5)  # typical 30-60 short target
            blueprint["recommended_length_seconds"] = 45
            blueprint["avg_scene_count"] = min(max(3, blueprint.get("avg_scene_count",5)), max_scenes)
        else:
            # youtube 1-5 minutes
            blueprint["recommended_length_seconds"] = min(max(60, blueprint.get("recommended_length_seconds",120)), 300)
            blueprint["avg_scene_count"] = min(max(4, blueprint.get("avg_scene_count",6)), max_scenes)

        # 3. Storyboard generation
        update_progress("Generating storyboard...", 3)
        sb = create_video_storyboard_agent(keyword, blueprint, form_data)
        # pydantic validation already done in that function
        storyboard = sb

        # 4. Character DB & uploaded images assignment
        update_progress("Registering character(s)...", 3)
        main_char_prompt = storyboard.get("main_character_profile") or "Unnamed, South Asian male, 30s, kind face"
        main_char = ensure_character("Main", appearance_prompt=main_char_prompt, reference_image_url=(uploaded_images[0] if uploaded_images else None), voice_id=voice_selection)
        char_profile = main_char.get("appearance_prompt")

        # 4b. Optionally insert local uploaded file as B-roll first scene
        if use_local_asset and os.path.exists(LOCAL_UPLOADED_FILE):
            update_progress("Uploading local asset for use as B-roll...", 3)
            try:
                uploaded_local_url = safe_upload_to_cloudinary(LOCAL_UPLOADED_FILE, resource_type="video", folder="user_assets")
                # Inject as scene 0
                scene0 = {
                    "scene_index": 0,
                    "duration": 3,
                    "visual_prompt": "Intro B-roll establishing shot",
                    "action_prompt": "slow push-in",
                    "image_url": uploaded_local_url,
                    "audio_narration": storyboard.get("scenes", [{}])[0].get("audio_narration","") if storyboard.get("scenes") else ""
                }
                storyboard["scenes"].insert(0, scene0)
            except Exception as e:
                logging.warning(f"Failed to upload/use local asset: {e}")

        # 5. Voiceover creation (single combined track) - aggregate scene narrations
        update_progress("Generating voiceover...", 4)
        full_script = " ".join([s.get("audio_narration","") for s in storyboard.get("scenes", [])])
        # If Urdu selected, pass language flag to voice generator (client should support)
        # generate_voiceover_and_upload should return a public mp3 URL
        voice_id = main_char.get("voice_id") or voice_selection or "21m00Tcm4TlvDq8ikWAM"
        voiceover_url = generate_voiceover_and_upload(full_script, voice_id, language=language) if callable(generate_voiceover_and_upload) else None
        if not voiceover_url:
            raise RuntimeError("Voiceover generation failed")

        # 6. Parallel scene generation
        update_progress("Generating scenes in parallel...", 5)
        scenes = storyboard.get("scenes", [])[:max_scenes]
        aspect = "9:16" if video_type == "reel" else "16:9"
        scene_urls = [None] * len(scenes)
        # control concurrency to avoid rate limit
        max_workers = int(os.getenv("SCENE_WORKERS", "3"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_single_scene, scenes[i], i, char_profile, aspect): i for i in range(len(scenes))}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    i, url = future.result()
                    scene_urls[idx] = url
                except Exception as e:
                    logging.exception(f"Scene generation exception for idx {idx}: {e}")
                    scene_urls[idx] = None

        # Filter valid scenes
        valid_scene_urls = [u for u in scene_urls if u]
        if not valid_scene_urls:
            raise RuntimeError("All scenes failed to generate")

        # 7. Assembly: create two outputs if requested
        update_progress("Assembling final video (short/long variants)...", 6)
        final_url = video_assembly_agent(valid_scene_urls, voiceover_url, storyboard, aspect=aspect)
        # Optional: also assemble the other aspect ratio for cross-posting
        other_aspect = "16:9" if aspect == "9:16" else "9:16"
        try:
            other_final_url = video_assembly_agent(valid_scene_urls, voiceover_url, storyboard, aspect=other_aspect)
        except Exception as e:
            logging.warning(f"Other aspect assembly failed: {e}")
            other_final_url = None

        # 8. Thumbnail & Metadata
        update_progress("Generating thumbnail & metadata...", 7)
        thumbnail_url = generate_thumbnail_agent(storyboard)
        metadata = youtube_metadata_agent(full_script, keyword)

        payload = {
            "status": "ready",
            "video_url": final_url,
            "alt_aspect_video_url": other_final_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard
        }
        update_progress("Task complete", 8)
        return payload

    except Exception as e:
        logging.error(f"Task failed: {traceback.format_exc()}")
        try:
            self.update_state(state="FAILURE", meta={"error": str(e)})
        except Exception:
            pass
        raise

