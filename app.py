# app.py

import os
import logging
import json
from functools import wraps
from flask import Flask, render_template, request, jsonify, Response
from dotenv import load_dotenv

load_dotenv()

# --- Celery app and task imports ---
from celery_init import celery
# Import the NEW video task
from celery_worker import background_generate_video

# If you still want to run the old text generator, you can import it too
# from celery_worker import background_generate

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
app = Flask(__name__, template_folder="templates")

# --- Multi-User Security (No changes needed here) ---
def check_auth(username, password):
    users_json_str = os.environ.get("APP_USERS_JSON")
    if not users_json_str:
        return True
    try:
        valid_users = json.loads(users_json_str)
    except json.JSONDecodeError:
        logging.error("CRITICAL: APP_USERS_JSON variable is not valid JSON.")
        return False
    if username in valid_users and valid_users.get(username) == password:
        return True
    return False

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not os.environ.get("APP_USERS_JSON"):
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# --- FLASK ROUTES ---

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
        return jsonify({"status": "error", "message": "Keyword is a required field."}), 400
    
    # --- MODIFIED: Call the new video generation task ---
    task = background_generate_video.delay(form_data)
    
    # The result page will now be used to display video generation progress
    return render_template("result.html", task_id=task.id)


@app.route("/check_result/<task_id>")
def check_result(task_id):
    result = celery.AsyncResult(task_id)
    
    # This response structure is now tailored for the video payload
    response = {
        "status": result.state.lower(),
        "message": "",
        "video_url": None,
        "thumbnail_url": None,
        "metadata": None,
        "storyboard": None
    }

    if result.state == 'PENDING':
        response["message"] = "Task is waiting to be processed."
    elif result.state == 'PROGRESS':
        progress_info = result.info if isinstance(result.info, dict) else {}
        response.update(progress_info)
        if not response.get("message"):
            response["message"] = "Processing in progress..."
    elif result.state == 'SUCCESS':
        final_result = result.result
        if isinstance(final_result, dict):
            response.update(final_result)
        response['status'] = 'ready' # Let the frontend know it's complete
    elif result.state == 'FAILURE':
        response["status"] = "error"
        try:
            error_info = str(result.info)
            response["message"] = f"Task failed. Please check worker logs. Error: {error_info}"
        except Exception:
            response["message"] = "Task failed with an unknown error."
        
    return jsonify(response)


# You may want a new route to display the final video, or handle it on the result page
@app.route("/video/<task_id>")
def video_player(task_id):
    result = celery.AsyncResult(task_id)
    if result.state != 'SUCCESS' or not result.result:
        return "<h1>Video Not Found or Not Ready</h1><p>This video may still be processing or failed.</p>", 404
        
    data = result.result
    # A new, simple template to just show the video
    return render_template("video_template.html", data=data)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
