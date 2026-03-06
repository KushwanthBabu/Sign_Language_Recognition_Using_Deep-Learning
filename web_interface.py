import cv2
import base64
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from collections import deque
import json
from pathlib import Path
import tempfile
import os
import sys

# -----------------------------
# PATHS
# -----------------------------

ROOT = Path(__file__).parent

MODEL_PATH = ROOT / "src" / "best_kp_model.h5"
LABEL_MAP = ROOT / "data" / "label_map.json"

# Add src/ to path so we can import build_model
sys.path.insert(0, str(ROOT / "src"))

# -----------------------------
# CONFIG
# -----------------------------

SEQ_LEN = 60

NUM_HANDS = 2
HAND_LMS = 21
POSE_LMS = 15

FEATS = NUM_HANDS * HAND_LMS * 3 + POSE_LMS * 3  # = 171

CONF_THRESH = 0.6

IMG_SIZE = 640
MAX_VIDEO_FRAMES = 120

# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------

print("Loading model...")

# Build model from architecture definition, then load weights.
# This avoids Keras 2 vs Keras 3 deserialization conflicts.
from model_kp import build_model
model = build_model()
model.load_weights(str(MODEL_PATH))

with open(LABEL_MAP, "r") as f:
    id2label = json.load(f)

id2label = {int(k): v for k, v in id2label.items()}

print("✅ Model + labels loaded.")

# -----------------------------
# MEDIAPIPE INIT (GLOBAL)
# -----------------------------

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -----------------------------
# FLASK
# -----------------------------

app = Flask(__name__, template_folder="templates")

live_buffer = deque(maxlen=SEQ_LEN)

# -----------------------------
# HELPERS
# -----------------------------

def preprocess(frame):

    h, w, _ = frame.shape
    scale = IMG_SIZE / max(h, w)

    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return frame


def extract_frame_keypoints(frame):

    frame = preprocess(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hres = hands.process(rgb)
    pres = pose.process(rgb)

    feats = []

    # -------- HANDS --------

    hand_vals = []

    if hres.multi_hand_landmarks:
        for hand in hres.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                hand_vals.extend([lm.x, lm.y, lm.z])

    while len(hand_vals) < NUM_HANDS * HAND_LMS * 3:
        hand_vals.extend([0.0, 0.0, 0.0])

    feats.extend(hand_vals[: NUM_HANDS * HAND_LMS * 3])

    # -------- POSE --------

    pose_vals = []

    if pres.pose_landmarks:
        for lm in pres.pose_landmarks.landmark[:POSE_LMS]:
            pose_vals.extend([lm.x, lm.y, lm.z])
    else:
        pose_vals = [0.0] * (POSE_LMS * 3)

    feats.extend(pose_vals)

    return np.array(feats, dtype=np.float32)

# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------
# LIVE
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    img_bytes = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(img_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    kps = extract_frame_keypoints(frame)

    if kps.shape[0] == FEATS:
        live_buffer.append(kps)

    label = "Waiting..."
    conf = 0.0

    if len(live_buffer) == SEQ_LEN:

        X = np.expand_dims(np.stack(live_buffer), axis=0)

        probs = model.predict(X, verbose=0)[0]

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        if conf > CONF_THRESH:
            label = id2label.get(idx, "Unknown")

    return jsonify({"label": label, "confidence": conf})

# -----------------------------
# UPLOAD IMAGE / VIDEO
# -----------------------------

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file"}), 400

    suffix = Path(file.filename).suffix.lower()

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    file.save(path)

    seq = []

    try:

        # -------- IMAGE --------
        if suffix in [".jpg", ".jpeg", ".png"]:

            frame = cv2.imread(path)

            if frame is None:
                return jsonify({"error": "Bad image"}), 400

            kps = extract_frame_keypoints(frame)

            if kps.shape[0] == FEATS:
                seq.append(kps)

        # -------- VIDEO --------
        else:

            cap = cv2.VideoCapture(path)

            count = 0
            skip = 2

            while count < MAX_VIDEO_FRAMES and len(seq) < SEQ_LEN:

                ret, frame = cap.read()
                if not ret:
                    break

                if count % skip == 0:
                    kps = extract_frame_keypoints(frame)
                    if kps.shape[0] == FEATS:
                        seq.append(kps)

                count += 1

            cap.release()

    finally:
        if os.path.exists(path):
            os.remove(path)

    if len(seq) == 0:
        return jsonify({"label": "No hands detected", "confidence": 0})

    X = np.zeros((SEQ_LEN, FEATS), dtype=np.float32)
    X[: len(seq)] = seq

    probs = model.predict(np.expand_dims(X, 0), verbose=0)[0]

    idx = int(np.argmax(probs))

    return jsonify({
        "label": id2label.get(idx, "Unknown"),
        "confidence": float(probs[idx]),
    })

# -----------------------------
# RUN
# -----------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Open http://0.0.0.0:{port}\n")
    app.run(
        host="0.0.0.0",
        port=port,
        threaded=False,
        debug=False,
    )
