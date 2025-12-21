# file: main.py
import os
import logging
from flask import Flask, request, jsonify, render_template
from celery.result import AsyncResult
from celery_init import celery 

# Import the NEW task from your updated worker
# Ensure your celery_worker.py defines 'process_video_upload'
from celery_worker import process_video_upload

app = Flask(__name__)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_endpoint():
    """
    Receives the Editing Request from the Frontend.
    """
    try:
        # 1. Get Data from Form
        video_url = request.form.get('video_url')
        remove_silence = request.form.get('remove_silence', 'false')
        blur_watermarks = request.form.get('blur_watermarks', 'false')
        add_subtitles = request.form.get('add_subtitles', 'false')

        # 2. Validation
        if not video_url:
            return jsonify({"status": "error", "message": "Please provide a valid Video URL."}), 400

        # 3. Payload for Worker
        form_data = {
            "video_url": video_url,
            "remove_silence": remove_silence,
            "blur_watermarks": blur_watermarks,
            "add_subtitles": add_subtitles
        }

        logger.info(f"🚀 Starting Editing Job for: {video_url}")

        # 4. Trigger Celery Task (The Editor Engine)
        task = process_video_upload.delay(form_data)

        return jsonify({
            "status": "success", 
            "task_id": task.id,
            "message": "Video queued for editing."
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
    
    if task.state == 'PENDING':
        response = {
            'status': 'pending',
            'message': 'Job is in queue...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'status': 'progress',
            'message': task.info.get('message', 'Processing...')
        }
    elif task.state == 'SUCCESS':
        # task.result is the return value from the worker function
        response = task.result 
        if not response: # Safety check
             response = {'status': 'error', 'message': 'Unknown error occurred.'}
    elif task.state == 'FAILURE':
        response = {
            'status': 'error',
            'message': str(task.info)
        }
    else:
        response = {
            'status': 'unknown',
            'message': f'Unknown state: {task.state}'
        }
        
    return jsonify(response)

if __name__ == '__main__':
    # Use PORT env for deployment (Railway/Heroku) or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
