import os
import logging
import cloudinary
import cloudinary.uploader
# [UPDATED] Added session, redirect, url_for for OAuth
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from celery.result import AsyncResult
from celery_init import celery 

# [NEW] Google API Imports
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Import the task from the worker
from celery_worker import process_video_upload

app = Flask(__name__)

# --- CONFIGURATION ---

# 1. Secret Key (Required for Sessions)
# Reads from Railway Variable "SECRET_KEY" or defaults to a dev key
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_prod")

# 2. Allow HTTP for local dev (Railway uses HTTPS, so this is safe)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# 3. Google Client Config
CLIENT_SECRETS_FILE = "client_secret.json"
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']

# 4. [CRITICAL FOR RAILWAY] Create client_secret.json from Env Var if missing
if not os.path.exists(CLIENT_SECRETS_FILE):
    if os.environ.get("GOOGLE_CLIENT_SECRET_JSON"):
        with open(CLIENT_SECRETS_FILE, "w") as f:
            f.write(os.environ.get("GOOGLE_CLIENT_SECRET_JSON"))

# 5. Configure Cloudinary
if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True
    )

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- ROUTING ---

@app.route('/')
def index():
    # [UPDATED] Check if user is connected and pass info to UI
    channel_info = session.get('connected_channel')
    return render_template('index.html', channel_info=channel_info)

# --- [NEW] GOOGLE AUTH ROUTES ---

@app.route('/authorize_youtube')
def authorize_youtube():
    """Initiates the OAuth2 flow to log the user in."""
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return "Error: client_secret.json is missing. Check your Railway Variables.", 500

    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES)
    
    # The callback URI must match exactly what is in Google Cloud Console
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    # 'prompt' param forces the account chooser to appear every time
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent select_account' 
    )

    session['state'] = state
    return redirect(authorization_url)

@app.route('/oauth2callback')
def oauth2callback():
    """Handles the return from Google Login."""
    state = session['state']
    
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE, scopes=SCOPES, state=state)
    flow.redirect_uri = url_for('oauth2callback', _external=True)

    # Fetch token
    authorization_response = request.url
    flow.fetch_token(authorization_response=authorization_response)

    credentials = flow.credentials

    # [NEW] Verify Identity & Get Channel Name
    try:
        youtube = build('youtube', 'v3', credentials=credentials)
        # Request 'mine=True' to get the channel tied to this token
        request_channel = youtube.channels().list(part="snippet", mine=True)
        response_channel = request_channel.execute()
        
        if response_channel['items']:
            channel = response_channel['items'][0]
            # Store Channel Info in Session for the UI
            session['connected_channel'] = {
                'title': channel['snippet']['title'],
                'thumbnail': channel['snippet']['thumbnails']['default']['url'],
                'id': channel['id']
            }
    except Exception as e:
        logger.error(f"Failed to fetch channel info: {e}")

    # Store credentials in Session
    session['credentials'] = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

    return redirect(url_for('index'))

# --- GENERATE ENDPOINT ---

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
        channel_name = request.form.get('channel_name', '@ViralShorts')

        # [NEW] Check for Auto-Upload Toggle & Credentials
        auto_upload = request.form.get('auto_upload_youtube', 'false')
        youtube_creds = None
        
        if auto_upload == 'true':
            youtube_creds = session.get('credentials')
            if not youtube_creds:
                 return jsonify({"status": "error", "message": "You selected Auto-Upload but are not connected to YouTube. Please connect first."}), 401

        # --- 5. Prepare Payload for Worker ---
        form_data = {
            "video_url": video_url,
            "output_format": output_format,
            "remove_silence": remove_silence,
            "blur_watermarks": blur_watermarks,
            "add_subtitles": add_subtitles,
            "channel_name": channel_name,
            "youtube_creds": youtube_creds # <--- [NEW] Passed to Worker
        }

        logger.info(f"🚀 Dispatching Task. Auto-Upload: {auto_upload}")

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
        # The worker returns a dictionary with video_url, metadata, youtube_url, etc.
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
