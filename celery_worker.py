# ===================================================================
# ===== ✅ YOUTUBE AUTOMATION WORKER V2.0 (SAFE MODE & LOGGING) ===
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

# --- Global Configurations & Client Initialization ---
# Configure logging to output to STDOUT so Railway captures it immediately
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] (WORKER): %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.error("🔴 CRITICAL: OPENAI_API_KEY is not configured.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

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
# ===== ✅ REAL API CLIENT IMPORTS ==================================
# ===================================================================

from video_clients.elevenlabs_client import generate_voiceover_and_upload
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
    if not isinstance(text, str):
        return None
    # Try to find JSON block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try parsing raw text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None

def get_openai_response(prompt_content: str, temperature: float = 0.7, is_json: bool = False) -> str:
    if not openai_client:
        raise Exception("OpenAI client not initialized.")
    if prompt_content.startswith("ERROR:"):
        raise FileNotFoundError(prompt_content)

    response_format = {"type": "json_object"} if is_json else {"type": "text"}
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-4o'),
            messages=[
                {"role": "system", "content": "You are a helpful expert assistant."},
                {"role": "user", "content": prompt_content}
            ],
            temperature=temperature,
            response_format=response_format
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"🔴 OpenAI API call failed: {e}")
        return ""

def generate_and_save_image(article_title: str, detailed_description: str) -> Optional[str]:
    if not openai_client or not cloudinary.config().api_key:
        logging.error("🔴 Cannot generate image, OpenAI or Cloudinary is not configured.")
        return None
    logging.info(f"Generating featured image for: '{article_title}'")
    try:
        response = openai_client.images.generate(model="dall-e-3", prompt=detailed_description, size="1024x1024", quality="standard", n=1)
        image_url_from_openai = response.data[0].url
        logging.info("Uploading generated image to Cloudinary...")
        upload_result = cloudinary.uploader.upload(image_url_from_openai, folder="youtube_thumbnails", public_id=f"{uuid.uuid4()}")
        secure_url = upload_result.get('secure_url')
        if secure_url:
            logging.info(f"✅ Successfully uploaded image to Cloudinary: {secure_url}")
            return secure_url
        else:
            logging.error("🔴 Cloudinary upload failed, no secure_url returned.")
            return None
    except Exception as e:
        logging.error(f"🔴 Failed to generate or upload featured image: {e}")
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
        competitor_analysis_summary="(Competitor analysis can be added here)",
        video_style_guide=form_data.get("video_style", "Cinematic, realistic, 4K")
    )
    
    # Get response from OpenAI
    response_str = get_openai_response(prompt, temperature=0.5, is_json=True)
    
    # --- 🔍 DEBUG LOGGING: SEE THE SCRIPT ---
    logging.info(f"📝 RAW OPENAI STORYBOARD RESPONSE:\n{response_str}")
    
    storyboard = extract_json_from_text(response_str)
    if not storyboard or "scenes" not in storyboard:
        raise ValueError("Storyboard generation failed or returned invalid format.")
    
    logging.info(f"✅ Video Storyboard created successfully. Scenes found: {len(storyboard['scenes'])}")
    return storyboard

def video_assembly_agent(scene_urls: list, voiceover_url: str) -> str:
    logging.info("Invoking Video Assembly Agent (FFMPEG)...")

    local_scene_paths = []
    # 1. Download all assets with timeouts
    try:
        for i, url in enumerate(scene_urls):
            if not url: 
                logging.warning(f"⚠️ Scene {i} URL is empty, skipping download.")
                continue
                
            local_path = f"/tmp/scene_{i}_{uuid.uuid4()}.mp4"
            logging.info(f"Downloading scene {i} from {url}...")
            
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            local_scene_paths.append(local_path)

        if not voiceover_url:
            raise ValueError("Voiceover URL is missing for assembly.")

        local_voiceover_path = f"/tmp/voiceover_{uuid.uuid4()}.mp3"
        logging.info("Downloading voiceover...")
        with requests.get(voiceover_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_voiceover_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 2. Create ffmpeg concat file list
        if not local_scene_paths:
            raise ValueError("No valid video scenes found to assemble.")

        concat_file_path = f"/tmp/concat_{uuid.uuid4()}.txt"
        with open(concat_file_path, "w") as f:
            for path in local_scene_paths:
                f.write(f"file '{path}'\n")

        output_file_path = f"/tmp/final_{uuid.uuid4()}.mp4"

        # 3. SAFER FFMPEG COMMAND (FIXES BLACK SCREEN ISSUES)
        # We use a filter complex to force scale all inputs to 16:9 (1024x576 is standard for Zeroscope)
        # 'setsar=1' fixes aspect ratio issues.
        cmd = [
            "ffmpeg", "-y", 
            "-f", "concat", "-safe", "0", "-i", concat_file_path,
            "-i", local_voiceover_path,
            "-vf", "scale=1024:576:force_original_aspect_ratio=decrease,pad=1024:576:(ow-iw)/2:(oh-ih)/2,setsar=1", 
            "-c:v", "libx264",      
            "-pix_fmt", "yuv420p",    
            "-preset", "fast",       
            "-c:a", "aac",          
            "-shortest",            
            output_file_path
        ]

        logging.info(f"Running FFMPEG Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info("✅ FFMPEG stitching complete.")

        # 4. Upload final video to Cloudinary
        upload_result = cloudinary.uploader.upload(output_file_path, resource_type="video")

        logging.info(f"✅ Final video uploaded: {upload_result['secure_url']}")
        return upload_result['secure_url']

    except subprocess.CalledProcessError as e:
        logging.error(f"🔴 FFMPEG Error: {e.stderr}")
        raise
    finally:
        # 5. Cleanup all temporary files
        paths_to_clean = local_scene_paths + [locals().get('local_voiceover_path'), locals().get('concat_file_path'), locals().get('output_file_path')]
        for path in paths_to_clean:
            if path and os.path.exists(path):
                os.remove(path)

def generate_thumbnail_agent(storyboard: dict) -> str:
    logging.info("Generating YouTube Thumbnail...")
    summary = f"YouTube Thumbnail for a video titled '{storyboard['video_title']}'. The video is about: {storyboard['video_description']}"
    image_prompt_str = load_prompt_template("prompt_image_synthesizer.txt")
    image_prompt_template = Template(image_prompt_str)
    final_image_prompt = image_prompt_template.safe_substitute(article_summary=summary)

    return generate_and_save_image(storyboard['video_title'], final_image_prompt)

def youtube_metadata_agent(full_script: str, keyword: str) -> dict:
    logging.info("Generating YouTube Metadata...")
    prompt_str = load_prompt_template("prompt_youtube_metadata_generator.txt")
    prompt = Template(prompt_str).safe_substitute(full_script=full_script, primary_keyword=keyword)
    response_str = get_openai_response(prompt, temperature=0.4, is_json=True)
    metadata = extract_json_from_text(response_str)
    if not metadata:
        return {"title": "AI Generated Video", "description": "Content created via automation.", "tags": [keyword]}
    logging.info("✅ YouTube metadata generated.")
    return metadata

# ===================================================================
# ===== ✅ MAIN VIDEO GENERATION TASK (SAFE MODE ENABLED) ===========
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

        # --- Step 1: Create Video Storyboard ---
        update_status("Generating video storyboard...", 1)
        storyboard = create_video_storyboard_agent(keyword, form_data)

        # --- Step 2: Sequential Asset Generation (Cost-Safe Mode) ---
        update_status("Generating voiceover & scenes sequentially...", 2)
        
        # 1. Generate Voiceover First (Cheapest Asset)
        full_script = " ".join([scene['audio_narration'] for scene in storyboard['scenes']])
        voice_id = form_data.get("voice_selection", "Rachel")
        
        logging.info(f"🎤 Generating full voiceover ({len(full_script)} chars)...")
        voiceover_url = generate_voiceover_and_upload(full_script, voice_id)
        if not voiceover_url:
             raise RuntimeError("Voiceover generation failed. Stopping before video generation.")

        # 2. Generate Video Scenes ONE BY ONE (To prevent mass failures)
        scene_urls = [None] * len(storyboard['scenes'])
        
        for i, scene in enumerate(storyboard['scenes']):
            logging.info(f"🎬 Generating Scene {i+1}/{len(storyboard['scenes'])}...")
            logging.info(f"   Prompt: {scene['visual_prompt'][:50]}...") # Log preview of prompt
            
            # Call Replicate specifically for this scene
            try:
                video_url = generate_video_scene_with_replicate(
                    scene['visual_prompt'], 
                    scene['duration_seconds']
                )
                
                if not video_url:
                    logging.error(f"❌ Scene {i+1} returned no URL. Stopping generation to save credits.")
                    # CRITICAL: Break the loop to stop spending money on broken prompts
                    raise RuntimeError(f"Scene {i+1} generation failed.")
                
                scene_urls[i] = video_url
                logging.info(f"✅ Scene {i+1} URL: {video_url}")
                
            except Exception as exc:
                logging.error(f"❌ Error generating Scene {i+1}: {exc}")
                raise RuntimeError(f"Failed to generate Scene {i+1}: {exc}")

        if None in scene_urls:
            raise RuntimeError("One or more scenes failed to generate.")

        # --- Step 3: Video Assembly ---
        update_status("Assembling final video with FFMPEG...", 3)
        final_video_url = video_assembly_agent(scene_urls, voiceover_url)

        # --- Step 4: Thumbnail Generation ---
        update_status("Generating AI thumbnail...", 4)
        thumbnail_url = generate_thumbnail_agent(storyboard)

        # --- Step 5: YouTube Metadata Generation ---
        update_status("Generating YouTube title & description...", 5)
        metadata = youtube_metadata_agent(full_script, keyword)

        # --- Step 6: Final Payload Assembly ---
        update_status("Assembling final payload...", 6)
        final_payload = {
            "status": "ready",
            "video_url": final_video_url,
            "thumbnail_url": thumbnail_url,
            "metadata": metadata,
            "storyboard": storyboard,
            "form_data": form_data
        }
        
        logging.info("✅✅ TASK COMPLETE: Video Generated Successfully.")
        return final_payload

    except Exception as e:
        logging.error(f"[{task_id}] A critical error occurred:\n{traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'status': 'error', 'message': f"❌ An unexpected error occurred: {e}"})
        # We re-raise to ensure Celery marks it as failed
        raise
