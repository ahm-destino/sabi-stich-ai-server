import os
import io
import uuid
import json
from flask import Flask, request, jsonify, send_file
from PIL import Image
import mediapipe as mp
import numpy as np
import torch
import google.generativeai as genai


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- MediaPipe -------------------
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True)

client = genai.configure(api_key=GEMINI_API_KEY)   # will be replaced

# ------------------- TryOnDiffusion Settings -------------------
MODEL_ID = os.environ.get("TRYON_MODEL_ID", "Kotiko-ua/tryondiffusion-model")
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")      # <<<<<<<<<< NEW
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------- TryOnDiffusion Pipeline Loader ----------------
def load_tryon_pipeline(model_id=MODEL_ID, device=DEVICE):
    from diffusers import DiffusionPipeline
    
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_auth_token=HF_TOKEN  # <<<<<<<<<<<<<< IMPORTANT
    )

    pipe.to(device)
    pipe.enable_attention_slicing()
    return pipe


print("Loading TryOnDiffusion pipeline...")
PIPELINE = load_tryon_pipeline()


# ------------------- Helpers -------------------
def read_imagefile(file_storage):
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    return img


def extract_landmarks(img: Image.Image):
    img_np = np.array(img)
    result = POSE.process(img_np)
    if result.pose_landmarks:
        return [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            } for lm in result.pose_landmarks.landmark
        ]
    return []


def compute_pixel_distances(landmarks):
    def distance(a, b):
        return ((a['x']-b['x'])**2 + (a['y']-b['y'])**2)**0.5

    mp_landmark = mp.solutions.pose.PoseLandmark
    try:
        shoulder_width = distance(landmarks[mp_landmark.LEFT_SHOULDER.value],
                                  landmarks[mp_landmark.RIGHT_SHOULDER.value])
        waist_width = distance(landmarks[mp_landmark.LEFT_HIP.value],
                               landmarks[mp_landmark.RIGHT_HIP.value])
        hip_width = waist_width
        torso_height = distance(landmarks[mp_landmark.LEFT_SHOULDER.value],
                                landmarks[mp_landmark.LEFT_HIP.value])
        arm_length = distance(landmarks[mp_landmark.LEFT_SHOULDER.value],
                              landmarks[mp_landmark.LEFT_WRIST.value])
        leg_length = distance(landmarks[mp_landmark.LEFT_HIP.value],
                              landmarks[mp_landmark.LEFT_ANKLE.value])
    except Exception:
        shoulder_width = waist_width = hip_width = torso_height = arm_length = leg_length = 0.0

    return {
        "shoulder_width_px": shoulder_width,
        "waist_width_px": waist_width,
        "hip_width_px": hip_width,
        "torso_height_px": torso_height,
        "arm_length_px": arm_length,
        "leg_length_px": leg_length
    }


# ------------------- Routes -------------------

# ----- Measurements Route -----
@app.route("/measurements", methods=["POST"])
def measurements():
    if "front_image" not in request.files or "back_image" not in request.files:
        return jsonify({"error": "front_image and back_image are required"}), 400

    front_img = read_imagefile(request.files["front_image"])
    back_img = read_imagefile(request.files["back_image"])

    front_landmarks = extract_landmarks(front_img)
    back_landmarks = extract_landmarks(back_img)

    if not front_landmarks or not back_landmarks:
        return jsonify({"error": "Could not detect landmarks on one or both images"}), 400

    front_pixels = compute_pixel_distances(front_landmarks)
    back_pixels = compute_pixel_distances(back_landmarks)

    combined_pixel_measurements = {
        k: (front_pixels[k] + back_pixels[k]) / 2
        for k in front_pixels
    }

    GEMINI_PROMPT = f"""
You are an AI system that estimates accurate human body measurements
using two images (front and back) and raw pose landmarks from MediaPipe.

Input JSONs:
- mediapipe_landmarks_front: {json.dumps(front_landmarks)}
- mediapipe_landmarks_back: {json.dumps(back_landmarks)}
- pixel_measurements: {json.dumps(combined_pixel_measurements)}

Task:
1. Validate landmarks.
2. Correct errors.
3. Align front/back views.
4. Convert pixel distances to centimeters.
5. Return final JSON with:
    height_cm, shoulder_width_cm, chest_circumference_cm, waist_cm, hip_cm,
    inseam_cm, arm_length_cm, leg_length_cm, torso_length_cm,
    scaling_factor_cm_per_pixel, quality_score, notes.
Return ONLY JSON.
"""

    try:
        response = client.responses.create(
            model="gemini-2.0-pro-vision",
            input=GEMINI_PROMPT
        )
        measurements_json = json.loads(response.output_text)
    except Exception as e:
        return jsonify({"error": "Gemini API failed", "details": str(e)}), 500

    return jsonify(measurements_json)


# ----- Try-On Route -----
@app.route("/tryon", methods=["POST"])
def tryon():
    if "user_image" not in request.files or "cloth_image" not in request.files:
        return jsonify({"error": "user_image and cloth_image are required"}), 400

    user_img = read_imagefile(request.files["user_image"])
    cloth_img = read_imagefile(request.files["cloth_image"])

    def resize_for_pipeline(img):
        return img.resize((512, 1024), Image.LANCZOS)

    user_img = resize_for_pipeline(user_img)
    cloth_img = resize_for_pipeline(cloth_img)

    try:
        result = PIPELINE(
            prompt="",
            user_image=user_img,
            cloth_image=cloth_img,
            num_inference_steps=30,
            guidance_scale=7.5
        )
        out_img = result.images[0]
    except Exception as e:
        return jsonify({"error": "TryOnDiffusion failed", "details": str(e)}), 500

    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    out_img.save(out_path)

    return send_file(out_path, mimetype="image/png")


# ----- Health Route -----
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "device": DEVICE})


# ------------------- Run -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
