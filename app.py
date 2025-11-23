# app.py (UPDATED FOR VIDEO SUITE 2025)

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

# Celery
from celery_init import celery
from celery_worker import background_generate_video

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Flask
app = Flask(__name__, template_folder="templates")

# Cloudinary config for uploads
if all([
    os.getenv("CLOUDINARY_CLOUD_NAME"),
    os.getenv("CLOUDINARY_API_KEY"),
    os.getenv("CLOUDINARY_API_SECRET")
]):
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )

# ------------------ SECURITY ------------------
def check_auth(username, password):
    users_json_str = os.environ.get("APP_USERS_JSON")
    if not users_json_str:
        return True
    try:
        valid_users = json.loads(users_json_str)
    except json.JSONDecodeError:
        logging.error("Invalid APP_USERS_JSON")
        return False
    return username in valid_users and valid_users.get(username) == password

def authenticate():
    return Response(
        'Authentication required.', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_enabled = os.environ.get("APP_USERS_JSON")
        if not auth_enabled:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
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
    """Handles new input form including file uploads, character system, scraping mode, etc."""
    
    # Convert text fields
    form_data = request.form.to_dict()

    # Mandatory field
    if not form_data.get("keyword"):
        return jsonify({"status": "error", "message": "Keyword is required"}), 400

    # ---------------------------
    # HANDLE CHARACTER IMAGES
    # ---------------------------
    uploaded_character_urls = []

    if "character_images" in request.files:
        files = request.files.getlist("character_images")

        for f in files:
            if f and f.filename.strip() != "":
                try:
                    upload = cloudinary.uploader.upload(
                        f,
                        folder="character_references",
                        resource_type="image"
                    )
                    uploaded_character_urls.append(upload["secure_url"])
                except Exception as e:
                    logging.error(f"Image Upload Failed: {e}")

    # Add to form_data for Celery
    form_data["character_image_urls"] = uploaded_character_urls

    # ---------------------------
    # SEND TO CELERY WORKER
    # ---------------------------
    task = background_generate_video.delay(form_data)

    return render_template("result.html", task_id=task.id)


@app.route("/check_result/<task_id>")
def check_result(task_id):
    result = celery.AsyncResult(task_id)

    response = {
        "status": result.state.lower(),
        "message": "",
        "video_url": None,
        "thumbnail_url": None,
        "metadata": None,
        "storyboard": None
    }

    # Pending
    if result.state == 'PENDING':
        response["message"] = "Task is queued."

    # Progress
    elif result.state == 'PROGRESS':
        info = result.info if isinstance(result.info, dict) else {}
        response.update(info)
        if not response.get("message"):
            response["message"] = "Processing..."

    # Success
    elif result.state == 'SUCCESS':
        final = result.result
        if isinstance(final, dict):
            response.update(final)
        response['status'] = "ready"

    # Failure
    elif result.state == 'FAILURE':
        response["status"] = "error"
        response["message"] = str(result.info)

    return jsonify(response)


@app.route("/video/<task_id>")
def video_player(task_id):
    result = celery.AsyncResult(task_id)
    if result.state != 'SUCCESS':
        return "Video not ready or failed.", 404
    return render_template("video_template.html", data=result.result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
