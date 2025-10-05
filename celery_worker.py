# ===================================================================
# ===== ✅ YOUTUBE AUTOMATION WORKER V1.1 (COMPLETE & CORRECTED) ====
# ===================================================================
import os
import logging
import concurrent.futures
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (WORKER): %(message)s")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.error("🔴 CRITICAL: OPENAI_API_KEY is not configured.")
    # You might want to raise an exception here or handle it gracefully
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
# ===== ✅ HELPER FUNCTIONS (COPIED FROM FOUNDATION CODE) ==========
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
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
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
# ===== ✅ PLACEHOLDER API CLIENTS (REPLACE WITH REAL LOGIC) =======
# ===================================================================

def generate_voiceover_and_upload(script: str, voice_id: str) -> str:
    """
    ## TODO: REPLACE THIS WITH YOUR REAL ELEVENLABS API LOGIC.
    This is a placeholder that returns a fake URL for testing.
    """
    logging.info(f"--- PLACEHOLDER: Generating voiceover for script (first 50 chars): '{script[:50]}...'")
    # In your real function, you would call the ElevenLabs API, get the audio bytes,
    # save to a temp file, and upload to Cloudinary, returning the secure_url.
    return "https://res.cloudinary.com/demo/video/upload/v1689235924/samples/elephants.mp3" # Fake URL

def generate_video_scene_and_upload(prompt: str, duration: int) -> str:
    """
    ## TODO: REPLACE THIS WITH YOUR REAL LUMA/RUNWAY/Pika API LOGIC.
    This is a placeholder that returns a fake URL for testing.
    """
    logging.info(f"--- PLACEHOLDER: Generating video scene for prompt: '{prompt[:50]}...'")
    # In your real function, you would call your chosen video API, wait for the result,
    # and return the final URL of the generated MP4 file.
    return "https://res.cloudinary.com/demo/video/upload/v1689235924/samples/elephants.mp4" # Fake URL

# ===================================================================
# ===== ✅ NEW VIDEO AGENT FUNCTIONS ================================
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
    response_str = get_openai_response(prompt, temperature=0.5, is_json=True)
    storyboard = extract_json_from_text(response_str)
    if not storyboard or "scenes" not in storyboard:
        raise ValueError("Storyboard generation failed or returned invalid format.")
    logging.info("✅ Video Storyboard created successfully.")
    return storyboard

def video_assembly_agent(scene_urls: list, voiceover_url: str) -> str:
    logging.info("Invoking Video Assembly Agent (FFMPEG)...")
    
    local_scene_paths = []
    # 1. Download all assets with timeouts
    try:
        for i, url in enumerate(scene_urls):
            local_path = f"/tmp/scene_{i}_{uuid.uuid4()}.mp4"
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            local_scene_paths.append(local_path)
        
        local_voiceover_path = f"/tmp/voiceover_{uuid.uuid4()}.mp3"
        with requests.get(voiceover_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(local_voiceover_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        # 2. Create ffmpeg concat file list
        concat_file_path = f"/tmp/concat_{uuid.uuid4()}.txt"
        with open(concat_file_path, "w") as f:
            for path in local_scene_paths:
                f.write(f"file '{path}'\n")

        output_file_path = f"/tmp/final_{uuid.uuid4()}.mp4"
        
        # 3. CORRECTED & ROBUST FFMPEG COMMAND
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path,
            "-i", local_voiceover_path,
            "-c:v", "libx264",      # Re-encode video to a standard format
            "-pix_fmt", "yuv420p",    # Ensures max player compatibility
            "-preset", "fast",        # Good balance of speed and quality
            "-c:a", "aac",          # Standard audio codec
            "-shortest",            # Finish encoding when the shortest stream (video) ends
            output_file_path
        ]
        
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
# ===== ✅ NEW MAIN VIDEO GENERATION TASK ===========================
# ===================================================================

@celery.task(bind=True)
def background_generate_video(self, form_data):
    task_id = self.request.id
    total_steps = 6

    def update_status(message, step):
        self.update_state(state='PROGRESS', meta={'status': 'processing', 'message': f"Step {step}/{total_steps}: {message}"})

    try:
        update_status("Initializing...", 0)
        keyword = form_data.get("keyword")
        if not keyword: raise ValueError("Keyword is required.")
        
        # --- Step 1: Create Video Storyboard ---
        update_status("Generating video storyboard...", 1)
        storyboard = create_video_storyboard_agent(keyword, form_data)
        
        # --- Step 2: Parallel Asset Generation ---
        update_status("Generating voiceover & video scenes...", 2)
        full_script = " ".join([scene['audio_narration'] for scene in storyboard['scenes']])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            voice_id = form_data.get("voice_selection", "Rachel")
            future_voiceover = executor.submit(generate_voiceover_and_upload, full_script, voice_id)
            
            future_scenes = {
                executor.submit(generate_video_scene_and_upload, scene['visual_prompt'], scene['duration_seconds']): i
                for i, scene in enumerate(storyboard['scenes'])
            }

            scene_urls = [None] * len(storyboard['scenes'])
            for future in concurrent.futures.as_completed(future_scenes):
                index = future_scenes[future]
                try:
                    scene_urls[index] = future.result()
                except Exception as exc:
                    logging.error(f"A scene generation task failed: {exc}")
                    # Handle failure, maybe by using a placeholder clip or stopping
            
            voiceover_url = future_voiceover.result()

        if None in scene_urls or not voiceover_url:
            raise RuntimeError("Failed to generate one or more media assets.")
            
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
        return final_payload

    except Exception as e:
        logging.error(f"[{task_id}] A critical error occurred:\n{traceback.format_exc()}")
        self.update_state(state='FAILURE', meta={'status': 'error', 'message': f"❌ An unexpected error occurred: {e}"})
        raise
