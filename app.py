import os
import io
import uuid
import json
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import torch
import google.generativeai as genai

# ------------------- ENVIRONMENT -------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Gemini Client -------------------
genai.configure(api_key=GEMINI_API_KEY)

# ------------------- MediaPipe Tasks -------------------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

POSE_MODEL_PATH = "pose_landmarker_full.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE
)

POSE = PoseLandmarker.create_from_options(pose_options)

# ------------------- TryOnDiffusion Settings -------------------
MODEL_ID = os.environ.get("TRYON_MODEL_ID", "Kotiko-ua/tryondiffusion-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- TryOnDiffusion Loader -------------------
def load_tryon_pipeline(model_id=MODEL_ID, device=DEVICE):
    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN
    )

    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe

print("Loading TryOnDiffusion pipeline...")
PIPELINE = load_tryon_pipeline()

# ------------------- Helper Functions -------------------
def read_imagefile(file_storage):
    return Image.open(io.BytesIO(file_storage.read())).convert("RGB")

def extract_landmarks(img: Image.Image):
    mp_img = vision.Image(image_format=vision.ImageFormat.SRGB, data=np.array(img))
    result = POSE.detect(mp_img)

    if not result.pose_landmarks:
        return []

    landmarks = []
    for lm in result.pose_landmarks[0]:
        landmarks.append({
            "x": lm.x,
            "y": lm.y,
            "z": lm.z,
            "visibility": lm.visibility
        })

    return landmarks

def compute_pixel_distances(landmarks):
    def dist(a, b):
        return ((a['x'] - b['x'])**2 + (a['y'] - b['y'])**2) ** 0.5

    try:
        shoulder = dist(landmarks[11], landmarks[12])
        waist = dist(landmarks[23], landmarks[24])
        hip = waist
        torso = dist(landmarks[11], landmarks[23])
        arm = dist(landmarks[11], landmarks[15])
        leg = dist(landmarks[23], landmarks[27])
    except:
        shoulder = waist = hip = torso = arm = leg = 0

    return {
        "shoulder_width_px": shoulder,
        "waist_width_px": waist,
        "hip_width_px": hip,
        "torso_height_px": torso,
        "arm_length_px": arm,
        "leg_length_px": leg,
    }

# ------------------- Routes -------------------

@app.route("/measurements", methods=["POST"])
def measurements():
    if "front_image" not in request.files or "back_image" not in request.files:
        return jsonify({"error": "front_image and back_image are required"}), 400

    front_img = read_imagefile(request.files["front_image"])
    back_img = read_imagefile(request.files["back_image"])

    front_lm = extract_landmarks(front_img)
    back_lm = extract_landmarks(back_img)

    if not front_lm or not back_lm:
        return jsonify({"error": "Pose landmarks not detected"}), 400

    front_px = compute_pixel_distances(front_lm)
    back_px = compute_pixel_distances(back_lm)

    avg_px = {k: (front_px[k] + back_px[k]) / 2 for k in front_px}

    prompt = f"""
You are an AI model converting pose landmarks and pixel distances into
accurate human body measurements.

Front landmarks:
{json.dumps(front_lm)}

Back landmarks:
{json.dumps(back_lm)}

Pixel distances:
{json.dumps(avg_px)}

Return ONLY a JSON object with:
height_cm, shoulder_width_cm, chest_circumference_cm, waist_cm, hip_cm,
inseam_cm, arm_length_cm, leg_length_cm, torso_length_cm,
scaling_factor_cm_per_pixel, quality_score, notes.
"""

    try:
        response = genai.GenerativeModel("gemini-2.0-pro-vision").generate_content(prompt)
        data = json.loads(response.text)
    except Exception as e:
        return jsonify({"error": "Gemini failed", "details": str(e)}), 500

    return jsonify(data)

@app.route("/tryon", methods=["POST"])
def tryon():
    if "user_image" not in request.files or "cloth_image" not in request.files:
        return jsonify({"error": "user_image and cloth_image are required"}), 400

    user_img = read_imagefile(request.files["user_image"])
    cloth_img = read_imagefile(request.files["cloth_image"])

    user_img = user_img.resize((512, 1024), Image.LANCZOS)
    cloth_img = cloth_img.resize((512, 1024), Image.LANCZOS)

    try:
        result = PIPELINE(
            prompt="",
            user_image=user_img,
            cloth_image=cloth_img,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        out = result.images[0]
    except Exception as e:
        return jsonify({"error": "TryOnDiffusion error", "details": str(e)}), 500

    fname = f"{uuid.uuid4().hex}.png"
    fpath = os.path.join(OUTPUT_DIR, fname)
    out.save(fpath)

    return send_file(fpath, mimetype="image/png")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": DEVICE})

# ------------------- Run -------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
