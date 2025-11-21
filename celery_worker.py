# ===================================================================
# ===== ✅ YOUTUBE AUTOMATION WORKER V3.0 (DRAMA/CONSISTENCY) =====
# ===================================================================
import os
import sys
import logging
import json
import re
import traceback
import uuid
import requests
import cloudinary
import cloudinary.uploader
import subprocess  # For FFMPEG
from typing import Optional
from string import Template
from celery_init import celery
from openai import OpenAI

# --- IMPORTS FOR CONSISTENCY PIPELINE ---
# We need 'replicate' directly here for the Image Generation step
import replicate 

# --- Global Configurations & Client Initialization ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] (WORKER): %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

if not OPENAI_API_KEY:
    logging.error("🔴 CRITICAL: OPENAI_API_KEY is not configured.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if not REPLICATE_API_TOKEN:
    logging.error("🔴 CRITICAL: REPLICATE_API_TOKEN is not configured. Video generation will fail.")

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )
    logging.info("✅ Cloudinary configured successfully.")
else:
    logging.warning("🔴 WARNING: Cloudinary credentials not fully set. Image/video uploads will fail.")

# ===================================================================
# ===== ✅ CLIENT IMPORTS ===========================================
# ===================================================================

from video_clients.elevenlabs_client import generate_voiceover_and_upload
# Ensure your replicate_client.py is updated to accept 'image_url'
from video_clients.replicate_client import generate_video_scene_with_replicate

# ===================================================================
# ===== ✅ HELPER FUNCTIONS =========================================
# ===================================================================

def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"🔴 Prompt file '{path}' not found!")
        return f"ERROR: Prompt file '{path}' is missing."

def extract_json_from_text(text: str) -> Optional[dict]:
    if not isinstance(text, str): return None
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except json.JSONDecodeError: pass
    try: return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try: return json.loads(text[start:end + 1])
            except: pass
    return None

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client: raise Exception("OpenAI client not initialized.")
    response_format = {"type": "json_object"} if is_json else {"type": "text"}
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{"role": "system", "content": "You are a helpful expert assistant."}, {"role": "user", "content": prompt_content}],
            temperature=temperature,
            response_format=response_format
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"🔴 OpenAI API call failed: {e}")
        return ""

def generate_flux_image(prompt: str) -> str:
    """
    Generates a high-quality keyframe image using Flux Schnell on Replicate.
    This is crucial for 'Drama' consistency before animating.
    """
    logging.info(f"📸 Generating FLUX Keyframe for: {prompt[:50]}...")
    try:
        # Using Flux Schnell (Fast & Cheap, Great Quality)
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "output_format": "jpg",
                "output_quality": 90
            }
        )
        # Flux returns a list of outputs [url, ...]
        image_url = output[0]
        logging.info(f"✅ Keyframe Image Ready: {image_url}")
        return image_url
    except Exception as e:
        logging.error(f"🔴 Flux Image Gen Failed: {e}")
        raise RuntimeError(f"Flux Image Gen Failed: {e}")

def generate_and_save_image(article_title: str, detailed_description: str) -> Optional[str]:
    # This is for THUMBNAILS only (DALL-E 3)
    if not openai_client: return None
    try:
        response = openai_client.images.generate(model="dall-e-3", prompt=detailed_description, size="1024x1024", quality="standard", n=1)
        upload_result = cloudinary.uploader.upload(response.data[0].url, folder="youtube_thumbnails", public_id=f"{uuid.uuid4()}")
        return upload_result.get('secure_url')
    except Exception as e:
        logging.error(f"🔴 Thumbnail Generation Failed: {e}")
        return None

# ===================================================================
# ===== ✅ VIDEO AGENT FUNCTIONS ====================================
# ===================================================================

def create_video_storyboard_agent(keyword: str, form_data: dict) -> dict:
    logging.info("Invoking Video Storyboard Agent...")
    prompt_str = load_prompt_template("prompt_video_storyboard_creator.txt")
    template = Template(prompt_str)
    prompt = template.safe_substitute(
        primary_keyword=keyword,
        competitor_analysis_summary="",
        video_style_guide=form_data.get("video_style", "Cinematic, realistic, 4K")
    )
    response_str = get_openai_response(prompt, temperature=0.5, is_json=True)
    storyboard = extract_json_from_text(response_str)
    if not storyboard or "scenes" not in storyboard:
        raise ValueError("Storyboard generation failed or invalid format.")
    return storyboard

def video_assembly_agent(scene_urls: list, voiceover_url: str) -> str:
    logging.info("Invoking Video Assembly Agent (FFMPEG)...")
    local_scene_paths = []
    
    try:
        # 1. Download Assets
        for i, url in enumerate(scene_urls):
            local_path = f"/tmp/scene_{i}_{uuid.uuid4()}.mp4"
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            local_scene_paths.append(local_path)

        local_voiceover_path = f"/tmp/voiceover_{uuid.uuid4()}.mp3"
        with requests.get(voiceover_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_voiceover_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

        # 2. Concat File
        concat_file_path = f"/tmp/concat_{uuid.uuid4()}.txt"
        with open(concat_file_path, "w") as f:
            for path in local_scene_paths: f.write(f"file '{path}'\n")

        output_file_path = f"/tmp/final_{uuid.uuid4()}.mp4"

        # 3. UPDATED FFMPEG COMMAND FOR HD (1280x720)
        # This ensures the video is standard 720p HD, which works best with Wan/Luma/Runway
        cmd = [
            "ffmpeg", "-y", 
            "-f", "concat", "-safe", "0", "-i", concat_file_path,
            "-i", local_voiceover_path,
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",  
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",        
            "-c:a", "aac", "-shortest",            
            output_file_path
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)
        upload_result = cloudinary.uploader.upload(output_file_path, resource_type="video")
        return upload_result['secure_url']

    except subprocess.CalledProcessError as e:
        logging.error(f"🔴 FFMPEG Error: {e.stderr}")
        raise
    finally:
        # Cleanup
        paths = local_scene_paths + [locals().get('local_voiceover_path'), locals().get('concat_file_path'), locals().get('output_file_path')]
        for p in paths:
            if p and os.path.exists(p): os.remove(p)

def generate_thumbnail_agent(storyboard: dict) -> str:
    logging.info("Generating YouTube Thumbnail...")
    title = storyboard.get('video_title', 'Video')
    desc = storyboard.get('video_description', 'Video')
    prompt = Template(load_prompt_template("prompt_image_synthesizer.txt")).safe_substitute(article_summary=f"{title}: {desc}")
    return generate_and_save_image(title, prompt)

def youtube_metadata_agent(full_script: str, keyword: str) -> dict:
    logging.info("Generating YouTube Metadata...")
    prompt = Template(load_prompt_template("prompt_youtube_metadata_generator.txt")).safe_substitute(full_script=full_script, primary_keyword=keyword)
    response_str = get_openai_response(prompt, temperature=0.4, is_json=True)
    return extract_json_from_text(response_str) or {"title": "AI Video", "tags": []}

# ===================================================================
# ===== ✅ MAIN VIDEO GENERATION TASK (DRAMA MODE) ==================
# ===================================================================

@celery.task(bind=True)
def background_generate_video(self, form_data):
    task_id = self.request.id
    total_steps = 6

    def update_status(message, step):
        self.update_state(state='PROGRESS', meta={'status': 'processing', 'message': f"Step {step}/{total_steps}: {message}"})
        logging.info(f"➡️  STATUS UPDATE: Step {step}/{total_steps}: {message}")

    try:
        update_status("Initializing...", 0)
        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword is required.")

        # --- Step 1: Storyboard ---
        update_status("Generating video storyboard...", 1)
        storyboard = create_video_storyboard_agent(keyword, form_data)

        # --- Step 2: Voiceover ---
        update_status("Generating voiceover...", 2)
        full_script = " ".join([scene.get('audio_narration', '') for scene in storyboard['scenes']])
        
        voice_map = {
            "Rachel": "21m00Tcm4TlvDq8ikWAM", "Adam": "pNInz6obpgDQGcFmaJgB",
            "Antoni": "ErXwobaYiN019PkySvjV", "Bella": "EXAVITQu4vr4xnSDxMaL",
            "Domi": "AZnzlk1XvdvUeBnXmlld", "Arnold": "VR6AewLTigWG4xSOukaG"
        }
        voice_id = voice_map.get(form_data.get("voice_selection", "Rachel"), "21m00Tcm4TlvDq8ikWAM") 
        voiceover_url = generate_voiceover_and_upload(full_script, voice_id)
        if not voiceover_url: raise RuntimeError("Voiceover generation failed.")

        # --- Step 3: THE CONSISTENCY LOOP (Image -> Video) ---
        update_status("Generating consistent scenes (Image -> Video)...", 3)
        
        # 1. Get the Master Character Profile (Fallbacks included)
        character_profile = storyboard.get('main_character_profile', 'A cinematic, photorealistic person')
        
        scene_urls = [None] * len(storyboard['scenes'])
        
        for i, scene in enumerate(storyboard['scenes']):
            logging.info(f"🎬 Processing Scene {i+1}/{len(storyboard['scenes'])}")
            
            try:
                # A. Construct Consistency Prompt
                # We combine the Character Profile + The Scene Setting
                visual_setting = scene.get('visual_setting', scene.get('visual_prompt', ''))
                full_image_prompt = f"{character_profile}, {visual_setting}, 8k, cinematic lighting, photorealistic"
                
                # B. Generate Keyframe Image (FLUX)
                keyframe_url = generate_flux_image(full_image_prompt)
                
                # C. Animate Image (WAN 2.1)
                # We pass the image_url to the updated replicate client
                action = scene.get('action_prompt', 'subtle cinematic movement')
                
                video_url = generate_video_scene_with_replicate(
                    prompt=action,
                    image_url=keyframe_url # PASSING THE IMAGE IS KEY FOR CONSISTENCY
                )
                
                if not video_url:
                    raise RuntimeError(f"Scene {i+1} animation failed.")
                
                scene_urls[i] = video_url
                logging.info(f"✅ Scene {i+1} Done: {video_url}")
                
            except Exception as exc:
                logging.error(f"❌ Scene {i+1} Failed: {exc}")
                raise RuntimeError(f"Scene {i+1} generation failed. Stopping to save credits.")

        # --- Step 4: Assembly ---
        update_status("Assembling final video...", 4)
        final_video_url = video_assembly_agent(scene_urls, voiceover_url)

        # --- Step 5 & 6: Metadata & Payload ---
        update_status("Finalizing metadata...", 5)
        thumbnail_url = generate_thumbnail_agent(storyboard)
        metadata = youtube_metadata_agent(full_script, keyword)

        final_payload = {
            "status": "ready",
            "video_url": final_video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard,
            "form_data": form_data
        }
        
        logging.info("✅✅ TASK COMPLETE.")
        return final_payload

    except Exception as e:
        logging.error(f"[{task_id}] CRITICAL ERROR: {traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'status': 'error', 'message': f"❌ Error: {e}"})
        raise
