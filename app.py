import os
import logging
import json
from functools import wraps
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

# Load environment
load_dotenv()

# Celery Imports
from celery_init import celery
from celery_worker import background_generate_video

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Flask
app = Flask(__name__, template_folder="templates")

# Cloudinary config
if all([os.getenv("CLOUDINARY_CLOUD_NAME"), os.getenv("CLOUDINARY_API_KEY"), os.getenv("CLOUDINARY_API_SECRET")]):
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )

# ------------------ SECURITY (Optional) ------------------
def check_auth(username, password):
    users_json_str = os.environ.get("APP_USERS_JSON")
    if not users_json_str: return True
    try:
        valid_users = json.loads(users_json_str)
        return username in valid_users and valid_users.get(username) == password
    except: return False

def authenticate():
    return Response('Login Required', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not os.environ.get("APP_USERS_JSON"): return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password): return authenticate()
        return f(*args, **kwargs)
    return wrapper

# ------------------ ROUTES ------------------

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"}), 200

@app.route("/")
@requires_auth
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
@requires_auth
def generate():
    form_data = request.form.to_dict()

    if not form_data.get("keyword"):
        return jsonify({"status": "error", "message": "Keyword is required"}), 400

    # --- IMAGE UPLOAD LOGIC ---
    uploaded_urls = []
    if "character_images" in request.files:
        files = request.files.getlist("character_images")
        for f in files:
            if f and f.filename.strip() != "":
                try:
                    upload = cloudinary.uploader.upload(f, folder="character_references", resource_type="image")
                    uploaded_urls.append(upload["secure_url"])
                except Exception as e:
                    logging.error(f"Image Upload Failed: {e}")

    # [CRITICAL FIX] Matching the key expected by celery_worker.py
    form_data["uploaded_images"] = uploaded_urls 

    # --- DISPATCH TASK ---
    task = background_generate_video.delay(form_data)
    
    # Return JSON so the frontend JS can handle the polling
    return jsonify({"status": "success", "task_id": task.id})

@app.route("/check_result/<task_id>")
def check_result(task_id):
    result = celery.AsyncResult(task_id)
    response = {"status": result.state.lower(), "message": "Processing..."}

    if result.state == 'PENDING':
        response["message"] = "Queued..."
    elif result.state == 'PROGRESS':
        info = result.info if isinstance(result.info, dict) else {}
        response.update(info)
    elif result.state == 'SUCCESS':
        response.update(result.result if isinstance(result.result, dict) else {})
        response['status'] = "ready"
    elif result.state == 'FAILURE':
        response["status"] = "error"
        response["message"] = str(result.info)

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
