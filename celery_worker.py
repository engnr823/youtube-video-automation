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

# Celery app import
from celery_init import celery

# AI Clients
from openai import OpenAI
import replicate

# --- CLIENT IMPORT SAFETY BLOCK ---
try:
    from video_clients.elevenlabs_client import generate_voiceover_and_upload, generate_multi_voice_audio
except ImportError:
    logging.warning("⚠️ ElevenLabs client not found. Voiceover generation will fail.")
    generate_voiceover_and_upload = None
    generate_multi_voice_audio = None

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
def scrape_youtube_videos(keyword: str, provider: str = "scrapingbee", max_results: int = 8) -> List[dict]:
    results = []
    logging.info(f"Scraping YouTube for '{keyword}' provider={provider}")
    try:
        if provider.lower() == "scrapingbee":
            if not SCRAPINGBEE_API_KEY: raise RuntimeError("SCRAPINGBEE_API_KEY missing")
            url = f"https://www.youtube.com/results?search_query={requests.utils.quote(keyword)}"
            params = {"api_key": SCRAPINGBEE_API_KEY, "url": url, "render_js": "false"}
            r = requests.get("https://app.scrapingbee.com/api/v1/", params=params, timeout=30)
            if r.status_code == 200:
                html = r.text
                match = re.search(r"ytInitialData\s*=\s*(\{.*\});", html, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                    # ... (Simplified scraping logic used previously) ...
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    return results

def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {"hook_style": "intrigue", "avg_scene_count": 6, "avg_scene_duration": 5, "tone": "motivational"}

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
    # [UPDATED] Use the file-based prompt if available
    prompt_template = load_prompt_template("prompt_video_storyboard_creator.txt")
    if not prompt_template:
        prompt_template = """
        You are an expert short film writer.
        BLUEPRINT: $blueprint_json
        TASK: Create an original short film script for '$keyword'.
        Output valid JSON with keys: video_title, video_description, main_character_profile, characters (list), scenes (list).
        IMPORTANT: List all speaking characters in 'characters' list with voice_id placeholders if needed.
        """
    
    template = Template(prompt_template)
    # Safe substitute to prevent crashing on missing keys
    prompt = template.safe_substitute(
        keyword=keyword,
        blueprint_json=json.dumps(blueprint),
        language=form_data.get("language", "english"),
        video_type=form_data.get("video_type", "reel"),
        uploaded_assets_context="User uploaded images present" if form_data.get("uploaded_images") else "No uploads"
    )

    raw = get_openai_response(prompt, temperature=0.6, is_json=True)
    obj = extract_json_from_text(raw) or json.loads(raw)
    if not obj: raise RuntimeError("Storyboard generation failed")
    
    try:
        return StoryboardSchema(**obj).dict()
    except ValidationError:
        return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

# -------------------------
# [FEATURE] Multi-Voice Logic
# -------------------------
def refine_script_with_roles(storyboard: dict, form_data: dict) -> List[dict]:
    """
    Splits the script into segments: [{'text': '...', 'voice_id': '...'}]
    """
    prompt = f"""
    You are a Voice Director. 
    Break this script into audio segments for different speakers.
    
    CHARACTERS: {json.dumps(storyboard.get('characters', []))}
    SCRIPT: {" ".join([s.get('audio_narration','') for s in storyboard.get('scenes', [])])}
    
    OUTPUT FORMAT (JSON List):
    [
      {{"speaker": "Narrator", "text": "Once upon a time..."}},
      {{"speaker": "Ali", "text": "I cannot believe it!"}}
    ]
    """
    raw = get_openai_response(prompt, temperature=0.3, is_json=True)
    segments = extract_json_from_text(raw) or []
    
    final_segments = []
    main_voice_id = form_data.get("voice_selection") or "21m00Tcm4TlvDq8ikWAM"
    
    # Simple fallback voice map
    fallback_voices = { "male": "pNInz6obpgDQGcFmaJgB", "female": "EXAVITQu4vr4xnSDxMaL" }

    for seg in segments:
        speaker_name = seg.get("speaker")
        text = seg.get("text")
        
        char_obj = next((c for c in storyboard.get("characters", []) if c.get("name") == speaker_name), None)
        
        if speaker_name == "Narrator":
            voice = main_voice_id
        elif char_obj and char_obj.get("voice_id"):
            voice = char_obj.get("voice_id")
        elif char_obj and "female" in char_obj.get("appearance_prompt", "").lower():
            voice = fallback_voices["female"]
        else:
            voice = main_voice_id

        final_segments.append({"text": text, "voice_id": voice})
        
    return final_segments

# -------------------------
# Character & Image Generation
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
        "voice_id": voice_id
    }
    with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    return db[name]

def generate_flux_image(prompt: str, aspect: str = "16:9") -> str:
    logging.info(f"Generating keyframe: {prompt[:50]}...")
    output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg"})
    return str(output[0]) if isinstance(output, (list, tuple)) else str(output)

def process_single_scene(scene: dict, index: int, character_profile: str, aspect: str = "16:9") -> (int, Optional[str]):
    try:
        if scene.get("image_url") and str(scene.get("image_url")).endswith(".mp4"):
            return (index, scene.get("image_url"))

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
        logging.error(f"Scene {index} error: {e}")
        return (index, None)

# -------------------------
# [FEATURE] Thumbnail & Metadata
# -------------------------
def generate_thumbnail_agent(storyboard: dict, orientation: str = "16:9") -> Optional[str]:
    prompt_template = load_prompt_template("prompt_image_synthesizer.txt")
    if not prompt_template:
        prompt_template = "A cinematic YouTube thumbnail for a story about: ${article_summary}. High contrast."

    summary = storyboard.get("video_description") or storyboard.get("video_title", "")
    characters_str = json.dumps(storyboard.get("characters", []), indent=2)

    template = Template(prompt_template)
    llm_prompt = template.safe_substitute(article_summary=summary, characters=characters_str, orientation=orientation)
    
    ready_to_use_prompt = get_openai_response(llm_prompt, temperature=0.7, is_json=False)
    ready_to_use_prompt = ready_to_use_prompt.strip().strip('"')

    try:
        logging.info(f"Generating Thumbnail...")
        img_url = generate_flux_image(ready_to_use_prompt, aspect=orientation)
        return img_url
    except Exception as e:
        logging.error(f"Thumbnail failed: {e}")
        return None

def youtube_metadata_agent(full_script: str, keyword: str, form_data: dict, blueprint: dict) -> dict:
    prompt_template = load_prompt_template("prompt_youtube_metadata_generator.txt")
    if not prompt_template:
        return {}

    language = form_data.get("language", "english")
    video_type = form_data.get("video_type", "reel")
    thumbnail_concept = form_data.get("thumbnail_concept", "Cinematic")
    
    template = Template(prompt_template)
    prompt = template.safe_substitute(
        primary_keyword=keyword,
        full_script=full_script[:3000],
        language=language,
        video_type=video_type,
        blueprint_data=json.dumps(blueprint),
        thumbnail_concept=thumbnail_concept
    )

    raw = get_openai_response(prompt, temperature=0.4, is_json=True)
    meta = extract_json_from_text(raw) or json.loads(raw)
    return meta or {}

# -------------------------
# Assembly (FFmpeg with Subtitles)
# -------------------------
def concat_videos_safe(input_paths: List[str], output_path: str, width: int = 1280, height: int = 720):
    cmd = ["ffmpeg", "-y"]
    for p in input_paths: cmd.extend(["-i", p])
    filter_complex = ""
    for i in range(len(input_paths)):
        filter_complex += f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}];"
    for i in range(len(input_paths)): filter_complex += f"[v{i}]"
    filter_complex += f"concat=n={len(input_paths)}:v=1:a=0[outv]"
    cmd.extend(["-filter_complex", filter_complex, "-map", "[outv]", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path])
    run_subprocess(cmd)
    return output_path

def merge_audio_video(video_path: str, audio_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", output_path]
    run_subprocess(cmd)
    return output_path

def srt_from_narration(segments: List[dict], out_path: str):
    """
    Generates an SRT file from the script segments. 
    This is an estimation (not word-perfect sync) but works well for Reels.
    """
    start = 0.0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            text = seg.get("text","")
            # Estimate duration: avg 15 chars per second + padding
            dur = max(1.5, len(text) / 15) 
            end = start + dur
            
            def fmt(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"
            
            f.write(f"{i+1}\n{fmt(start)} --> {fmt(end)}\n{text}\n\n")
            start = end

def burn_subtitles(input_video: str, srt_path: str, output_video: str):
    # Hardcode minimal style for now
    style = "Fontsize=24,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=3"
    # Escape the path for FFmpeg
    escaped_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    cmd = ["ffmpeg", "-y", "-i", input_video, "-vf", f"subtitles='{escaped_srt}':force_style='{style}'", "-c:a", "copy", output_video]
    run_subprocess(cmd)
    return output_video

def video_assembly_agent(scene_urls: List[str], voiceover_url: str, storyboard: dict, segments: List[dict], aspect: str = "16:9"):
    tmpdir = tempfile.mkdtemp(prefix="assemble_")
    try:
        # 1. Download scenes
        local_scene_paths = []
        for i, url in enumerate(scene_urls):
            if not url: continue
            local_path = os.path.join(tmpdir, f"scene_{i}.mp4")
            download_to_file(url, local_path)
            local_scene_paths.append(local_path)

        if not local_scene_paths: raise RuntimeError("No videos to assemble")

        # 2. Download Voiceover
        local_voice = os.path.join(tmpdir, "voice.mp3")
        download_to_file(voiceover_url, local_voice)

        # 3. Concat Scenes
        width, height = (1280, 720) if aspect == "16:9" else (720, 1280)
        concat_out = os.path.join(tmpdir, "concat.mp4")
        concat_videos_safe(local_scene_paths, concat_out, width, height)

        # 4. Merge Audio
        merged_out = os.path.join(tmpdir, "merged.mp4")
        merge_audio_video(concat_out, local_voice, merged_out)

        # 5. [RESTORED] Generate & Burn Subtitles
        srt_path = os.path.join(tmpdir, "subs.srt")
        srt_from_narration(segments, srt_path)
        
        final_out = os.path.join(tmpdir, "final_burned.mp4")
        try:
            burn_subtitles(merged_out, srt_path, final_out)
        except Exception as e:
            logging.error(f"Subtitle burning failed: {e}")
            final_out = merged_out # Fallback to non-subtitled video

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

        # 1. Scrape & Blueprint
        update_status("Analyzing...")
        scraped = scrape_youtube_videos(keyword)
        blueprint = analyze_competitors(scraped)

        # 2. Storyboard
        update_status("Storyboarding...")
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)

        # 3. Character Setup
        update_status("Characters...")
        characters = storyboard.get("characters") or []
        uploaded_images = form_data.get("uploaded_images") or []
        
        if uploaded_images:
             if not characters:
                 characters = [{"name": "Main", "reference_image_url": uploaded_images[0]}]
             else:
                 characters[0]["reference_image_url"] = uploaded_images[0]
             storyboard["characters"] = characters

        for ch in characters:
            ensure_character(ch.get("name", "Main"), ch.get("appearance_prompt"), ch.get("reference_image_url"))
        
        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # 4. Multi-Voice Generation
        update_status("Generating Voices...")
        segments = refine_script_with_roles(storyboard, form_data)
        full_script = " ".join([s.get("text","") for s in segments])
        
        if generate_multi_voice_audio:
            voiceover_url = generate_multi_voice_audio(segments)
        elif generate_voiceover_and_upload:
            voice_id = form_data.get("voice_selection") or "21m00Tcm4TlvDq8ikWAM"
            voiceover_url = generate_voiceover_and_upload(full_script, voice_id)
        else:
            raise RuntimeError("Voiceover client unavailable")

        # 5. Scenes (Parallel)
        update_status("Generating Scenes...")
        scenes = storyboard.get("scenes", [])
        scene_urls = [None] * len(scenes)
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {executor.submit(process_single_scene, scenes[i], i, char_profile, aspect): i for i in range(len(scenes))}
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, url = future.result()
                    scene_urls[idx] = url
                except Exception as e:
                    logging.error(f"Scene {idx} error: {e}")

        valid_urls = [u for u in scene_urls if u]
        if not valid_urls: raise RuntimeError("All scenes failed")

        # 6. Assembly with Subtitles
        update_status("Assembling Final Video...")
        final_url = video_assembly_agent(valid_urls, voiceover_url, storyboard, segments, aspect)

        # 7. [RESTORED] Thumbnail & Metadata
        update_status("Generating Metadata...")
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        metadata = youtube_metadata_agent(full_script, keyword, form_data, blueprint)

        return {
            "status": "ready",
            "video_url": final_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard
        }

    except Exception as e:
        logging.error(f"Task failed: {traceback.format_exc()}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
