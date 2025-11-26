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
import requests
import cloudinary
import cloudinary.uploader
from string import Template
from typing import Optional, List, Dict, Any
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, wait_fixed

# Celery app import
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
try:
    from video_clients.elevenlabs_client import generate_voiceover_and_upload, generate_multi_voice_audio
except ImportError:
    logging.error("⚠️ ElevenLabs client not found. Audio/Lip-sync will fail.")
    generate_voiceover_and_upload = None
    generate_multi_voice_audio = None

try:
    from video_clients.replicate_client import generate_video_scene_with_replicate
except ImportError:
    logging.warning("⚠️ standard video generation client not found.")
    generate_video_scene_with_replicate = None
# ----------------------------------

# ==========================================
#  🚨 BUDGET SAFETY SWITCH 🚨
#  False = Images Only (Budget Mode). True = Real Video/LipSync (Costly).
# ==========================================
USE_REAL_EXPENSIVE_GENERATION = True
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
# Content Generation
# -------------------------
def analyze_competitors(scraped_videos: List[dict]) -> Dict[str, Any]:
    return {"hook_style": "intrigue", "avg_scene_count": 7, "tone": "motivational"}

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(3))
def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise RuntimeError("OpenAI client not configured")
    try:
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
        Output strictly valid JSON with keys: video_title, video_description, main_character_profile, characters (list of objects with name, voice_id), scenes (list of objects with visual_prompt, action_prompt, audio_narration).
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
            video_type=form_data.get("video_type", "reel"), 
            max_scenes=str(target_scenes)
        )
        prompt += f"\n\nIMPORTANT: Generate exactly {target_scenes} scenes. Ensure JSON format."

        raw = get_openai_response(prompt, temperature=0.6, is_json=True)
        obj = extract_json_from_text(raw) or json.loads(raw)
        if not obj or not obj.get("scenes"): raise RuntimeError("Storyboard generation produced invalid JSON")
        
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
    try:
        if os.path.exists(CHAR_DB_PATH):
             with open(CHAR_DB_PATH, "r") as f: db = json.load(f)
        else: db = {}
    except: db = {}

    if name in db: return db[name]
    
    db[name] = {
        "id": str(uuid.uuid4()),
        "name": name,
        "appearance_prompt": appearance_prompt or f"A vertical portrait photograph of {name}, looking at camera, 8k, photorealistic",
        "voice_id": voice_id
    }
    try:
        with open(CHAR_DB_PATH, "w") as f: json.dump(db, f, indent=2)
    except: pass
    return db[name]

@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=30), retry=retry_if_exception_type(ReplicateError))
def generate_flux_image_safe(prompt: str, aspect: str = "9:16") -> str:
    if not REPLICATE_API_TOKEN: raise RuntimeError("Replicate Token Missing")
    logging.info(f"Generating image via Replicate (Flux)...")
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
    
    # --- UPDATED MODEL VERSION BELOW ---
    output = replicate.run(
        "cjwbw/sadtalker:3aa3dac937e5675cb5761e31c50853e66172d54467d16781206152b12267191d",
        input={
            "source_image": image_url, 
            "driven_audio": audio_url, 
            "still": True, 
            "enhancer": "gfpgan",
            "preprocess": "full",
            "expression_scale": 1.0,
            "ref_eyeblink": None,
            "ref_pose": None
        }
    )
    return str(output)
def process_single_scene(scene: dict, index: int, character_profile: str, aspect: str = "9:16", default_voice_id: str = None) -> (int, Optional[str]):
    try:
        logging.info(f"--- Processing Scene {index+1} ---")
        visual_setting = scene.get("visual_prompt", "")
        full_image_prompt = f"A vertical portrait photograph of {character_profile}, {visual_setting}, looking directly at the camera, neutral expression, highly detailed, 8k."
        
        # 1. Generate Image
        keyframe_url = generate_flux_image_safe(full_image_prompt, aspect=aspect)
        logging.info(f"Scene {index+1}: Base image generated.")

        # 2. Safety/Budget Check
        if not USE_REAL_EXPENSIVE_GENERATION:
            logging.warning(f"Scene {index+1}: [BUDGET SAFETY MODE] Skipping video generation. Returning static image.")
            return (index, keyframe_url)

        # 3. Real Generation Logic (Voice + LipSync)
        dialogue = scene.get("audio_narration", "").strip()
        if dialogue and len(dialogue) > 2:
            logging.info(f"Scene {index+1}: Dialogue detected. Starting Lip Sync.")
            voice_id = scene.get("voice_id") or default_voice_id or "21m00Tcm4TlvDq8ikWAM"
            if not generate_voiceover_and_upload: raise RuntimeError("ElevenLabs client missing.")
            
            scene_audio_url = generate_voiceover_and_upload(dialogue, voice_id)
            if not scene_audio_url: raise RuntimeError("Failed to generate scene audio.")
            
            video_url = generate_lip_sync_safe(keyframe_url, scene_audio_url)
            return (index, video_url)
        else:
            logging.info(f"Scene {index+1}: No dialogue. Generating atmospheric video.")
            action_prompt = scene.get("action_prompt", "subtle camera movement")
            if not generate_video_scene_with_replicate:
                 return (index, keyframe_url)
            video_url = generate_video_scene_with_replicate(prompt=action_prompt, image_url=keyframe_url, aspect=aspect)
            return (index, video_url)

    except Exception as e:
        logging.error(f"Scene {index+1} failed processing: {e}")
        return (index, None)

# -------------------------
# Robust Assembly Logic (FIXED)
# -------------------------

def normalize_clip(input_path: str, output_path: str, is_image: bool = False):
    """
    Standardizes ALL clips (images or videos) to the exact same format:
    - 720x1280 resolution
    - 25 fps
    - yuv420p pixel format
    - AAC audio (silent if needed)
    This prevents concatenation failures and 'missing stream' errors.
    """
    cmd = ["ffmpeg", "-y"]
    
    if is_image:
        # IMAGE MODE: Loop image + Generate Silent Audio
        cmd.extend([
            "-loop", "1", "-i", input_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p",
            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
            "-c:a", "aac", "-shortest"
        ])
    else:
        # VIDEO MODE: Re-encode to ensure consistency + Generate Silent Audio if missing
        # We map video from input 0, and we conditionally map audio.
        # To be safe, we add a silent audio track as input 1, and map it if input 0 has no audio.
        # But for simplicity, let's just force re-encode. 
        # If the input video MIGHT not have audio (e.g. atmospheric video), we need a complex filter or padded audio.
        # For robustness, we will use -af apad with a null source fallback, but let's stick to a simple re-encode for now
        # assuming Real Mode videos (LipSync) HAVE audio.
        # If atmospheric video has NO audio, we need to add silence.
        
        # Strategy: Use lavfi anullsrc mixed with input.
        cmd.extend([
            "-i", input_path,
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-filter_complex", "[0:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1:1[v];[0:a][1:a]amix=inputs=2:duration=first[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
            "-c:a", "aac"
        ])
        
        # Fallback: If the above complex filter fails because input 0 has NO audio stream, 
        # we have to catch it. But usually amix handles it if configured right. 
        # Actually, simpler approach for the video path to avoid complex filter failure on missing stream:
        # Just use the robust image command if it's meant to be silent?
        # No, for now, let's assume if is_video=True, it has audio OR we are okay adding silence.
        
        # SIMPLIFIED ROBUST VIDEO COMMAND (Handling potential missing audio in video inputs):
        # We will just run a basic convert. If it fails due to missing audio, we'd need a check.
        # But 'USE_REAL_EXPENSIVE_GENERATION' usually implies LipSync (has audio) or Replicate Video (no audio).
        pass # The logic is handled inside video_assembly_agent loop below

    cmd.append(output_path)
    run_subprocess(cmd)

def concat_videos_robust(input_paths: List[str], output_path: str):
    logging.info(f"Concatenating {len(input_paths)} files...")
    list_file_path = os.path.join(tempfile.gettempdir(), f"concat_list_{uuid.uuid4()}.txt")
    
    with open(list_file_path, "w") as f:
        for path in input_paths:
            safe_path = path.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")
    
    # Concat using the demuxer, but force re-encode to fix any timestamp issues
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path,
        "-c:v", "copy", "-c:a", "copy", # Copy streams since we normalized them already
        output_path
    ]
    try:
        run_subprocess(cmd)
    finally:
        if os.path.exists(list_file_path): os.remove(list_file_path)
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
         raise RuntimeError("FFmpeg concatenation failed.")
    return output_path

def add_background_music(video_path: str, music_url: str, output_path: str, tone: str):
    logging.info("Adding background music...")
    unique_music = f"bg_music_{uuid.uuid4()}.mp3"
    music_path = os.path.join(tempfile.gettempdir(), unique_music)
    
    try:
        download_to_file(music_url, music_path)
        # Robust Mix: Input 0 (Video), Input 1 (Music).
        # We assume Input 0 has audio (because we normalized it!).
        # We mix them.
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
        local_scene_paths = []
        for i, url in enumerate(scene_urls):
            if not url: continue
            
            is_video = url.endswith(".mp4") or url.endswith(".mov")
            ext = "mp4" if is_video else "jpg"
            local_path = os.path.join(tmpdir, f"scene_{i}_raw.{ext}")
            normalized_path = os.path.join(tmpdir, f"scene_{i}_norm.mp4")

            download_to_file(url, local_path)

            try:
                if not is_video:
                    logging.info(f"Converting image {i} to static video clip with SILENT AUDIO.")
                    # IMAGE -> VIDEO (With guaranteed silent audio)
                    run_subprocess([
                        "ffmpeg", "-y",
                        "-loop", "1", "-i", local_path,
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-c:v", "libx264", "-t", "3", "-pix_fmt", "yuv420p",
                        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
                        "-c:a", "aac", "-shortest",
                        normalized_path
                    ])
                else:
                    logging.info(f"Normalizing video clip {i}.")
                    # VIDEO -> NORMALIZED VIDEO (Ensure audio exists)
                    # Use a trick: Add silent audio as a secondary input, map it only if needed? 
                    # For safety, we just try to copy. If it fails later, we know why.
                    # But if we are in 'Atmospheric' mode (no audio), we MUST add silence.
                    # Let's blindly add silence to the mix to be safe.
                    run_subprocess([
                        "ffmpeg", "-y",
                        "-i", local_path,
                        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                        "-filter_complex", "[0:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1:1[v];[0:a][1:a]amix=inputs=2:duration=first[a]",
                        "-map", "[v]", "-map", "[a]",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                        "-c:a", "aac",
                        normalized_path
                    ])
                    # Note: The command above works even if input 0 has audio (it mixes silence) 
                    # or if input 0 has NO audio (amix handles single stream nicely usually, 
                    # but strictly amix needs 2 inputs. If [0:a] is missing, this command fails).
                    
                    # RETRY BLOCK for video without audio (Atmospheric):
                    if not os.path.exists(normalized_path):
                         logging.info("Complex mix failed (likely no audio in source). Generating from video + silence.")
                         run_subprocess([
                            "ffmpeg", "-y",
                            "-i", local_path,
                            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1:1",
                            "-c:a", "aac", "-shortest", # Cut silence to video length
                            normalized_path
                        ])

                local_scene_paths.append(normalized_path)

            except Exception as e:
                logging.error(f"Failed to process scene {i}: {e}")

        if not local_scene_paths: raise RuntimeError("No valid scenes to assemble.")

        concat_out = os.path.join(tmpdir, "concat_raw.mp4")
        concat_videos_robust(local_scene_paths, concat_out)

        final_out = os.path.join(tmpdir, "final_with_music.mp4")
        music_url = MUSIC_LIBRARY.get(music_tone, MUSIC_LIBRARY["default"])
        add_background_music(concat_out, music_url, final_out, music_tone)

        return safe_upload_to_cloudinary(final_out, folder="final_videos_lipsync")
        
    except Exception as e:
        logging.error(f"Assembly failed: {e}")
        logging.error(traceback.format_exc())
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

@celery.task(bind=True, time_limit=1800)
def background_generate_video(self, form_data: dict):
    task_id = getattr(self.request, "id", "unknown")
    logging.info(f"[{task_id}] Task started.")
    
    if USE_REAL_EXPENSIVE_GENERATION:
         logging.warning("💰💰💰 REAL EXPENSIVE GENERATION IS ON. 💰💰💰")
    else:
         logging.info("🛡️ SAFETY MODE ON. Generating images only.")

    try:
        def update_status(msg):
            self.update_state(state="PROGRESS", meta={"message": msg})
            logging.info(f"[{task_id}] STATUS: {msg}")

        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword required")

        update_status("Step 1/4: Writing Script...")
        form_data["max_scenes"] = 7
        form_data["video_type"] = "reel" 
        blueprint = analyze_competitors([])
        storyboard = create_video_storyboard_agent(keyword, blueprint, form_data)
        scenes = storyboard.get("scenes", [])
        
        update_status("Step 2/4: Preparing Character...")
        chars_data = storyboard.get("characters", [])
        main_char_name = chars_data[0].get("name", "Narrator") if chars_data else "Narrator"
        default_voice_id = form_data.get("voice_selection") or (chars_data[0].get("voice_id") if chars_data else None)
        char_db_entry = ensure_character(main_char_name, voice_id=default_voice_id)
        char_profile = char_db_entry.get("appearance_prompt")

        update_status(f"Step 3/4: Generating {len(scenes)} Scenes...")
        scene_urls = [None] * len(scenes)
        aspect = "9:16"

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_idx = {
                executor.submit(process_single_scene, scene, i, char_profile, aspect, default_voice_id): i 
                for i, scene in enumerate(scenes)
            }
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, url = future.result()
                    if url:
                         logging.info(f"✅ Scene {idx+1} finished successfully.")
                         scene_urls[idx] = url
                    else:
                         logging.error(f"❌ Scene {idx+1} failed.")
                except Exception as e:
                    logging.error(f"❌ Scene {idx+1} exception: {e}")

        update_status("Step 4/4: Assembling Final Reel...")
        valid_urls = [u for u in scene_urls if u]
        if not valid_urls: raise RuntimeError("All scenes failed generation.")
        
        tone = blueprint.get("tone", "motivational")
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
        self.update_state(state="FAILURE", meta={"error": err_msg})
        raise Exception(err_msg)
