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
from openai import OpenAI
import replicate

# --- UTILS IMPORT ---
try:
    from utils.ffmpeg_utils import get_media_duration
except ImportError:
    # Fallback if utils file is missing
    def get_media_duration(*args, **kwargs): return 5.0

# --- CLIENT IMPORT SAFETY BLOCK ---
try:
    from video_clients.elevenlabs_client import (
        generate_voiceover_and_upload, 
        generate_multi_voice_audio,
        generate_audio_for_scene
    )
except ImportError:
    logging.warning("⚠️ ElevenLabs client not found. Voiceover generation will fail.")
    generate_audio_for_scene = None

try:
    from video_clients.replicate_client import (
        generate_video_scene_with_replicate, 
        generate_lip_sync_with_replicate
    )
except ImportError:
    logging.warning("⚠️ Replicate client not found. Video generation will fail.")
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
# LOW-MEMORY VIDEO STITCHER (Fixes SIGKILL Crash)
# -------------------------
def stitch_video_audio_pairs_optimized(scene_pairs: List[Tuple[str, str]], output_path: str) -> bool:
    """
    Local optimized version of stitching to prevent OOM kills.
    Uses 'ultrafast' preset and 'crf 28' to minimize RAM usage.
    """
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join("/tmp", f"render_{request_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    input_list_path = os.path.join(temp_dir, "inputs.txt")
    chunk_paths = []

    try:
        logging.info(f"Processing {len(scene_pairs)} pairs for Request ID: {request_id}")

        # 1. Process chunks (Sync Video length to Audio length)
        for i, (video, audio) in enumerate(scene_pairs):
            chunk_name = os.path.join(temp_dir, f"chunk_{i}.mp4")
            
            audio_dur = get_media_duration(audio)
            if audio_dur == 0:
                logging.warning(f"Audio duration is 0 for {audio}, skipping chunk.")
                continue

            # COMMAND OPTIMIZED FOR LOW MEMORY ENVIRONMENTS
            cmd = [
                "ffmpeg", "-y", "-stream_loop", "-1", "-i", video, "-i", audio,
                "-t", str(audio_dur), 
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", 
                "-pix_fmt", "yuv420p", 
                "-preset", "ultrafast",  # <--- CHANGED: Uses minimal RAM
                "-crf", "28",            # <--- ADDED: Slightly lower quality to save RAM
                "-c:a", "aac", 
                "-shortest",
                chunk_name
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            chunk_paths.append(chunk_name)

        # 2. Write the Concat List
        with open(input_list_path, "w") as f:
            for chunk in chunk_paths:
                abs_path = os.path.abspath(chunk).replace("'", "'\\''")
                f.write(f"file '{abs_path}'\n")

        # 3. Concatenate Chunks
        logging.info("Concatenating chunks...")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", input_list_path, "-c", "copy", output_path
        ], check=True, capture_output=True)

        logging.info(f"✅ Video successfully saved to {output_path}")
        return True

    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed: {e}")
        return False
    except Exception as e:
        logging.error(f"General stitching error: {e}")
        return False
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# -------------------------
# Scraping & Storyboard
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
                pass 
    except Exception as e:
        logging.error(f"Scraping error: {e}")
    return results

def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {"hook_style": "intrigue", "avg_scene_count": 6, "tone": "motivational"}

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
    obj = extract_json_from_text(raw) or json.loads(raw)
    if not obj: raise RuntimeError("Storyboard generation failed")
    
    try:
        return StoryboardSchema(**obj).dict()
    except ValidationError:
        return {"video_title": "Untitled", "scenes": obj.get("scenes", []), "characters": obj.get("characters", [])}

# -------------------------
# Multi-Voice Logic
# -------------------------
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
    output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt, "aspect_ratio": aspect, "output_format": "jpg"})
    return str(output[0]) if isinstance(output, (list, tuple)) else str(output)

# -------------------------
# [UPDATED] Scene Processor 
# -------------------------
def process_single_scene(
    scene: dict, 
    index: int, 
    character_profile: str, 
    audio_path: str = None,   # [UPDATED] Pass audio for syncing
    reference_img_url: str = None, # [UPDATED] Pass master face for consistency
    aspect: str = "16:9"
) -> dict:
    """
    Generates a scene video. 
    1. Uses 'reference_img_url' if available (Consistency).
    2. If audio exists, runs 'generate_lip_sync_with_replicate' (Talking Head).
    3. If no audio, runs 'generate_video_scene_with_replicate' (Cinematic B-Roll).
    """
    try:
        # Rate Limit Buffer
        sleep_time = random.randint(5, 15)
        time.sleep(sleep_time)

        # --- A. CHARACTER CONSISTENCY LOGIC ---
        # If we have a Master Reference Image (generated once at start), use it!
        # Otherwise, generate a fresh one (Only for Scene 0 or B-Roll).
        keyframe_url = None
        
        if scene.get("image_url") and str(scene.get("image_url")).endswith(".mp4"):
             # If user uploaded a video directly
             temp_path = f"/tmp/user_upload_{index}_{uuid.uuid4()}.mp4"
             download_to_file(scene.get("image_url"), temp_path)
             return {"index": index, "video_path": temp_path, "status": "success"}

        if scene.get("image_url"):
            keyframe_url = scene.get("image_url") # User manual upload (per scene)
        elif reference_img_url:
            keyframe_url = reference_img_url # <--- USE THE CONSISTENT ACTOR FACE
        else:
            # Fallback: Only generate new face if we absolutely have to
            visual_setting = scene.get("visual_prompt", "")
            human_texture = "detailed skin pores, natural skin texture, 8k"
            negative = " --no plastic, doll, 3d render, cartoon"
            full_prompt = f"{character_profile}, {visual_setting}, {human_texture}, cinematic lighting {negative}"
            keyframe_url = generate_flux_image(full_prompt, aspect=aspect)

        # --- B. SELECT GENERATION ENGINE (Lip-Sync vs Cinematic) ---
        video_url = None
        
        # Check if this scene has dialogue AND we have audio
        has_dialogue = bool(scene.get("audio_narration") and len(scene.get("audio_narration")) > 2)
        
        if has_dialogue and audio_path and generate_lip_sync_with_replicate:
            # Upload local audio to cloud so Replicate can access it
            cloud_audio_url = safe_upload_to_cloudinary(audio_path, resource_type="video", folder="temp_audio")
            
            # CALL LIP-SYNC (SadTalker / Wav2Lip)
            # This makes the "keyframe_url" face talk using "cloud_audio_url"
            video_url = generate_lip_sync_with_replicate(keyframe_url, cloud_audio_url)
            
        else:
            # NO Dialogue -> Use Cinematic Action (Wan 2.1 / Luma)
            action = scene.get("action_prompt", "subtle movement") + f", camera:{scene.get('camera_angle','35mm')}"
            video_url = generate_video_scene_with_replicate(prompt=action, image_url=keyframe_url, aspect=aspect)
        
        # 3. Download Result
        temp_video_path = f"/tmp/scene_gen_{index}_{uuid.uuid4()}.mp4"
        download_to_file(video_url, temp_video_path)
        
        return {"index": index, "video_path": temp_video_path, "status": "success"}

    except Exception as e:
        logging.error(f"Scene {index} failed: {e}")
        return {"index": index, "video_path": None, "status": "failed"}

# -------------------------
# Metadata Helpers
# -------------------------
def generate_thumbnail_agent(storyboard: dict, orientation: str = "16:9") -> Optional[str]:
    summary = storyboard.get("video_description") or "Video"
    prompt = f"Movie poster for: {summary}. High quality, 8k, textless."
    try:
        return generate_flux_image(prompt, aspect=orientation)
    except: return None

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
    return extract_json_from_text(raw) or json.loads(raw)

# -------------------------
# Celery Task (SaaS Logic)
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

        # 1. Scrape & Storyboard
        update_status("Designing Video Concept...")
        scraped = scrape_youtube_videos(keyword)
        blueprint = analyze_competitors(scraped)
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)

        # 2. Characters & Assets
        characters = storyboard.get("characters") or []
        uploaded_images = form_data.get("uploaded_images") or []
        if uploaded_images:
             # Basic mapping of uploads to characters
             for i, url in enumerate(uploaded_images):
                 if i < len(characters): characters[i]["reference_image_url"] = url
                 else: characters.append({"name": f"Char {i}", "reference_image_url": url})
             storyboard["characters"] = characters

        char_profile = characters[0].get("appearance_prompt", "Cinematic") if characters else "Cinematic"

        # [UPDATED] GENERATE MASTER CHARACTER IMAGE ONCE (Consistency Fix)
        # Check if user uploaded a face, otherwise generate one "Actor" for the whole movie
        master_face_url = None
        if uploaded_images:
            master_face_url = uploaded_images[0]
        else:
            # Generate the Hero Face
            update_status("Casting AI Actor (Generating Consistent Face)...")
            aspect_ratio = "9:16" if form_data.get("video_type") == "reel" else "16:9"
            hero_prompt = f"Portrait of {char_profile}, facing camera, neutral expression, high detailed, 8k"
            master_face_url = generate_flux_image(hero_prompt, aspect=aspect_ratio)

        # 3. [NEW] Audio-First Pipeline
        update_status("Synthesizing Audio Dialogue...")
        segments = refine_script_with_roles(storyboard, form_data)
        
        # This list will hold: {'index': 0, 'audio_path': '...', 'duration': 4.5, 'scene_data': ...}
        scene_assets = [] 
        full_script_text = ""

        # Loop through storyboard scenes (not just segments, to ensure 1-to-1 mapping)
        scenes = storyboard.get("scenes", [])
        
        for i, scene in enumerate(scenes):
            # Match segment to scene (simple 1-to-1 mapping for now)
            # In a complex app, you might have multiple lines per scene, but for MVP SaaS, keep 1-to-1
            text = segments[i].get("text") if i < len(segments) else "..."
            voice_id = segments[i].get("voice_id") if i < len(segments) else "21m00Tcm4TlvDq8ikWAM"
            full_script_text += text + " "

            # Generate local audio
            if generate_audio_for_scene:
                audio_path = generate_audio_for_scene(text, voice_id)
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
            else:
                # Handle silent scenes or errors
                pass

        if not scene_assets:
            raise RuntimeError("Audio generation failed for all scenes.")

        # 4. [NEW] Video Generation (Targeted Duration)
        update_status("Rendering Video Scenes (Synced)...")
        aspect = "9:16" if form_data.get("video_type") == "reel" else "16:9"
        
        # Pairs for stitching: (video_path, audio_path)
        final_pairs = []
        
        # Parallel Execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # [UPDATED] Pass master_face_url and audio_path to scene processor
            future_to_asset = {
                executor.submit(
                    process_single_scene, 
                    asset["scene_data"], 
                    asset["index"], 
                    char_profile, 
                    asset["audio_path"],   # <--- PASS AUDIO PATH
                    master_face_url,       # <--- PASS CONSISTENT FACE URL
                    aspect
                ): asset
                for asset in scene_assets
            }
            
            # Collect results and keep them in order of index
            # We use a temp dict to store results then sort
            results_map = {}
            
            for future in concurrent.futures.as_completed(future_to_asset):
                asset = future_to_asset[future]
                idx = asset["index"]
                res = future.result() # returns {"index":..., "video_path":..., "status":...}
                
                if res["status"] == "success" and res["video_path"]:
                    results_map[idx] = (res["video_path"], asset["audio_path"])
                else:
                    logging.warning(f"Scene {idx} video failed. Skipping.")

            # Re-assemble in correct order
            for i in range(len(scenes)):
                if i in results_map:
                    final_pairs.append(results_map[i])

        if not final_pairs:
            raise RuntimeError("Video generation failed for all scenes.")

        # 5. [NEW] Stitching (SaaS Logic)
        update_status("Final Assembly (Audio-Video Sync)...")
        
        # Prepare Music
        music_tone = blueprint.get("tone", "motivational")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        music_path = os.path.join(tempfile.gettempdir(), f"music_{uuid.uuid4()}.mp3")
        try:
            download_to_file(music_url, music_path)
        except:
            music_path = None # Proceed without music if fail

        # Stitch - USING OPTIMIZED LOCAL FUNCTION
        final_output_path = f"/tmp/final_render_{task_id}.mp4"
        
        # Step A: Stitch Scenes using the new LOW-MEMORY function
        temp_visual_path = f"/tmp/visual_base_{task_id}.mp4"
        
        # [CRITICAL FIX] Use local optimized function instead of imported one
        success = stitch_video_audio_pairs_optimized(final_pairs, temp_visual_path)
        
        if not success:
            raise RuntimeError("Stitching failed.")
        
        # Step B: Add Background Music (Ducking)
        if music_path:
            # [CRITICAL FIX] Added -preset ultrafast and -crf 28 here too
            cmd = [
                "ffmpeg", "-y",
                "-i", temp_visual_path,
                "-stream_loop", "-1", "-i", music_path,
                "-filter_complex", "[1:a]volume=0.15[bg];[0:a][bg]amix=inputs=2:duration=first[outa]",
                "-map", "0:v", "-map", "[outa]",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", # Low Memory settings
                "-c:a", "aac", "-shortest",
                final_output_path
            ]
            subprocess.run(cmd, check=True)
        else:
            # No music, just rename
            shutil.move(temp_visual_path, final_output_path)

        # 6. Upload & Metadata
        update_status("Uploading & Finalizing...")
        final_video_url = safe_upload_to_cloudinary(final_output_path, folder="final_videos")
        thumbnail_url = generate_thumbnail_agent(storyboard, aspect)
        metadata = youtube_metadata_agent(full_script_text, keyword, form_data, blueprint)

        # Cleanup
        try:
            if music_path and os.path.exists(music_path): os.remove(music_path)
            if os.path.exists(temp_visual_path): os.remove(temp_visual_path)
            if os.path.exists(final_output_path): os.remove(final_output_path)
            for v, a in final_pairs:
                if os.path.exists(v): os.remove(v)
                if os.path.exists(a): os.remove(a)
        except: pass

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
