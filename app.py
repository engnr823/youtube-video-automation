import os
import logging
import cloudinary
import cloudinary.uploader
from flask import Flask, request, jsonify, render_template
from celery.result import AsyncResult
from celery_init import celery 

# Import the task from the worker
from celery_worker import process_video_upload

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Cloudinary (Required for direct file uploads)
if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """
    Receives the Editing Request (File Upload or URL) from the Frontend.
    """
    try:
        video_url = None
        
        # --- 1. Handle File Upload (Priority) ---
        if 'video_file' in request.files and request.files['video_file'].filename != '':
            file = request.files['video_file']
            logger.info(f"📤 Uploading file: {file.filename} to Cloudinary...")
            
            # Upload raw file to Cloudinary immediately
            upload_result = cloudinary.uploader.upload(
                file, 
                resource_type="video", 
                folder="raw_inputs"
            )
            video_url = upload_result['secure_url']
            logger.info(f"✅ Upload successful: {video_url}")
            
        # --- 2. Handle URL Input (Fallback) ---
        elif 'video_url' in request.form and request.form['video_url'].strip():
            video_url = request.form['video_url'].strip()
            
        # --- 3. Validation ---
        if not video_url:
            return jsonify({"status": "error", "message": "No Video File or URL provided."}), 400

        # --- 4. Get Toggles & Options ---
        output_format = request.form.get('output_format', '9:16') 
        remove_silence = request.form.get('remove_silence', 'false')
        blur_watermarks = request.form.get('blur_watermarks', 'false')
        add_subtitles = request.form.get('add_subtitles', 'false')
        
        # [NEW] Capture Channel Branding
        channel_name = request.form.get('channel_name', '@ViralShorts')

        # --- 5. Prepare Payload for Worker ---
        form_data = {
            "video_url": video_url,
            "output_format": output_format,
            "remove_silence": remove_silence,
            "blur_watermarks": blur_watermarks,
            "add_subtitles": add_subtitles,
            "channel_name": channel_name  # <--- Passed to Worker
        }

        logger.info(f"🚀 Dispatching Task for: {video_url} [Branding: {channel_name}]")

        # --- 6. Trigger Celery Task ---
        task = process_video_upload.delay(form_data)

        return jsonify({
            "status": "success", 
            "task_id": task.id,
            "message": "Video queued for processing."
        })

    except Exception as e:
        logger.error(f"Server Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/check_result/<task_id>', methods=['GET'])
def check_result(task_id):
    """
    Checks the status of the Celery task.
    """
    task = AsyncResult(task_id, app=celery)
    
    response = {'status': task.state.lower()}
    
    if task.state == 'PENDING':
        response['message'] = 'Job is in queue...'
    elif task.state == 'PROGRESS':
        response['message'] = task.info.get('message', 'Processing...')
    elif task.state == 'SUCCESS':
        # The worker returns a dictionary with video_url, metadata, etc.
        response = task.result 
        if not response: 
             response = {'status': 'error', 'message': 'Unknown error occurred (Empty Result).'}
    elif task.state == 'FAILURE':
        response = {
            'status': 'error',
            'message': str(task.info)
        }
    else:
        response['message'] = f'Unknown state: {task.state}'
        
    return jsonify(response)

if __name__ == '__main__':
    # Use PORT env for deployment (Railway/Heroku) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
