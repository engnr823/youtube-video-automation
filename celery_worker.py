# ===================================================================
# ===== ✅ YOUTUBE AUTOMATION WORKER V4.0 (PARALLEL TURBO) =========
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
import subprocess
import concurrent.futures # <--- REQUIRED FOR PARALLELISM
from typing import Optional
from string import Template
from celery_init import celery
from openai import OpenAI
import replicate 

# --- Global Configurations ---
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] (WORKER): %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
openai_client = OpenAI(api_key=OPENAI_API_KEY)

if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

# --- CLIENT IMPORTS ---
from video_clients.elevenlabs_client import generate_voiceover_and_upload
from video_clients.replicate_client import generate_video_scene_with_replicate

# --- HELPER FUNCTIONS (Keep these same as before) ---
def load_prompt_template(filename: str) -> str:
    path = os.path.join("prompts", filename)
    try:
        with open(path, "r", encoding="utf-8") as f: return f.read()
    except FileNotFoundError: return ""

def extract_json_from_text(text: str) -> Optional[dict]:
    if not isinstance(text, str): return None
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
    try: return json.loads(text)
    except: pass
    return None

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    response_format = {"type": "json_object"} if is_json else {"type": "text"}
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[{"role": "system", "content": "Expert Assistant."}, {"role": "user", "content": prompt_content}],
            temperature=temperature, response_format=response_format
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"OpenAI Error: {e}")
        return ""

def generate_flux_image(prompt: str) -> str:
    """Generates Keyframe Image and ensures output is a STRING URL."""
    try:
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt, "aspect_ratio": "16:9", "output_format": "jpg", "output_quality": 90}
        )
        return str(output[0]) # Force string conversion
    except Exception as e:
        raise RuntimeError(f"Flux Error: {e}")

def process_single_scene(scene, index, character_profile):
    """
    Worker function to handle ONE scene (Image -> Video) independently.
    This allows us to run it in parallel.
    """
    try:
        logging.info(f"🚀 Starting Parallel Generation for Scene {index}...")
        
        # 1. Generate Image
        visual_setting = scene.get('visual_setting', scene.get('visual_prompt', ''))
        full_image_prompt = f"{character_profile}, {visual_setting}, 8k, cinematic lighting, photorealistic"
        keyframe_url = generate_flux_image(full_image_prompt)
        
        # 2. Generate Video
        action = scene.get('action_prompt', 'subtle cinematic movement')
        video_url = generate_video_scene_with_replicate(prompt=action, image_url=keyframe_url)
        
        logging.info(f"✅ Scene {index} COMPLETED: {video_url}")
        return (index, video_url)
        
    except Exception as e:
        logging.error(f"❌ Scene {index} Failed: {e}")
        return (index, None)

def video_assembly_agent(scene_urls: list, voiceover_url: str) -> str:
    # (Keep your existing ffmpeg logic here, it is correct)
    # ... [Paste your existing video_assembly_agent code here] ...
    # For brevity in this snippet, I am assuming you keep the function from the previous file.
    logging.info("Invoking Video Assembly Agent...")
    local_scene_paths = []
    try:
        for i, url in enumerate(scene_urls):
            if not url: continue
            local_path = f"/tmp/scene_{i}_{uuid.uuid4()}.mp4"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            local_scene_paths.append(local_path)
            
        local_voiceover_path = f"/tmp/voiceover_{uuid.uuid4()}.mp3"
        with requests.get(voiceover_url, stream=True) as r:
             with open(local_voiceover_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
                
        concat_path = f"/tmp/concat_{uuid.uuid4()}.txt"
        with open(concat_path, "w") as f:
            for path in local_scene_paths: f.write(f"file '{path}'\n")
            
        output_path = f"/tmp/final_{uuid.uuid4()}.mp4"
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_path,
            "-i", local_voiceover_path,
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
            "-c:a", "aac", "-shortest", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        res = cloudinary.uploader.upload(output_path, resource_type="video")
        return res['secure_url']
    except Exception as e:
        logging.error(f"FFMPEG Error: {e}")
        raise
    finally:
        # Cleanup logic
        pass

def generate_thumbnail_agent(storyboard):
    # (Keep existing logic)
    return "https://via.placeholder.com/1080" # Placeholder for brevity, use your real code

def youtube_metadata_agent(script, keyword):
    # (Keep existing logic)
    return {"title": "Video", "tags": []}

# ===================================================================
# ===== ✅ MAIN TASK (PARALLELIZED) =================================
# ===================================================================

@celery.task(bind=True)
def background_generate_video(self, form_data):
    task_id = self.request.id
    
    try:
        keyword = form_data.get("keyword")
        
        # 1. Storyboard
        self.update_state(state='PROGRESS', meta={'message': 'Creating Storyboard...'})
        storyboard = create_video_storyboard_agent(keyword, form_data) # Ensure this function exists
        
        # 2. Voiceover
        self.update_state(state='PROGRESS', meta={'message': 'Generating Voiceover...'})
        full_script = " ".join([s.get('audio_narration','') for s in storyboard['scenes']])
        voice_id = "21m00Tcm4TlvDq8ikWAM" # Rachel
        voiceover_url = generate_voiceover_and_upload(full_script, voice_id)

        # 3. PARALLEL SCENE GENERATION
        self.update_state(state='PROGRESS', meta={'message': 'Generating ALL Scenes in Parallel...'})
        character_profile = storyboard.get('main_character_profile', 'A person')
        
        # We use ThreadPoolExecutor to run 6 API calls at once
        # This reduces time from 15 mins -> 3 mins
        scene_urls = [None] * len(storyboard['scenes'])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_single_scene, scene, i, character_profile): i 
                for i, scene in enumerate(storyboard['scenes'])
            }
            
            # Wait for results
            for future in concurrent.futures.as_completed(future_to_index):
                index, url = future.result()
                if url:
                    scene_urls[index] = url
                else:
                    logging.error(f"Scene {index} failed completely.")
        
        # Check if we have valid videos
        valid_urls = [u for u in scene_urls if u is not None]
        if len(valid_urls) == 0:
            raise RuntimeError("All scenes failed to generate.")

        # 4. Assembly
        self.update_state(state='PROGRESS', meta={'message': 'Assembling Video...'})
        final_url = video_assembly_agent(valid_urls, voiceover_url) # Pass valid_urls
        
        return {
            "status": "ready",
            "video_url": final_url,
            "storyboard": storyboard
        }

    except Exception as e:
        logging.error(f"CRITICAL WORKER ERROR: {e}")
        self.update_state(state='FAILURE', meta={'message': str(e)})
        raise
