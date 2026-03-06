import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import json

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "best_kp_model.h5"

LABEL_MAP = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\label_map.json"

SEQ_LEN = 60

NUM_HANDS = 2
HAND_LMS = 21
POSE_LMS = 15

FEATS = NUM_HANDS * HAND_LMS * 3 + POSE_LMS * 3   # 171

CONF_THRESH = 0.70
SMOOTH_WINDOW = 8       # smooth predictions

FRAME_SKIP = 2          # speed
RESIZE_TO = (640, 480)

# -----------------------------
# Load model
# -----------------------------

print("✅ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded.")

with open(LABEL_MAP, "r") as f:
    id2label = json.load(f)

id2label = {int(k): v for k, v in id2label.items()}

# -----------------------------
# MediaPipe
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
# Feature extraction
# -----------------------------

def extract_frame_keypoints(frame):

    frame = cv2.resize(frame, RESIZE_TO)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    h = hands.process(rgb)
    p = pose.process(rgb)

    feats = []

    # -------- HANDS --------
    hand_vals = []

    if h.multi_hand_landmarks:
        for hand in h.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                hand_vals.extend([lm.x, lm.y, lm.z])

    while len(hand_vals) < NUM_HANDS * HAND_LMS * 3:
        hand_vals.extend([0.0, 0.0, 0.0])

    feats.extend(hand_vals[: NUM_HANDS * HAND_LMS * 3])

    # -------- POSE --------
    pose_vals = []

    if p.pose_landmarks:
        for lm in p.pose_landmarks.landmark[:POSE_LMS]:
            pose_vals.extend([lm.x, lm.y, lm.z])
        while len(pose_vals) < POSE_LMS * 3:
            pose_vals.extend([0.0, 0.0, 0.0])
    else:
        pose_vals = [0.0] * (POSE_LMS * 3)

    feats.extend(pose_vals)

    return np.array(feats, dtype=np.float32)

# -----------------------------
# Webcam loop
# -----------------------------

cap = cv2.VideoCapture(0)

seq = deque(maxlen=SEQ_LEN)

pred_hist = deque(maxlen=SMOOTH_WINDOW)

frame_count = 0

print("\n🎥 Webcam started.")
print("Hold sign steady for 2–3 seconds.")
print("Press Q to quit.\n")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    display = frame.copy()

    # -----------------
    # SPEED: skip frames
    # -----------------
    if frame_count % FRAME_SKIP == 0:

        kps = extract_frame_keypoints(frame)

        if kps.shape[0] == FEATS:
            seq.append(kps)

    # -----------------
    # Predict only when buffer full
    # -----------------

    if len(seq) == SEQ_LEN:

        X = np.expand_dims(np.stack(seq), axis=0)

        probs = model.predict(X, verbose=0)[0]

        pred_hist.append(probs)

        avg_probs = np.mean(pred_hist, axis=0)

        idx = int(np.argmax(avg_probs))
        conf = float(avg_probs[idx])

        if conf > CONF_THRESH:

            label = id2label.get(idx, "Unknown")

            text = f"{label} ({conf:.2f})"

            cv2.putText(
                display,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3,
            )

    cv2.imshow("Sign Recognition", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
