import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

# -----------------------------
# 🔴 PATHS (VAL SPLIT)
# -----------------------------

RAW_DIR = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\raw"

OUT_DIR = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\keypoints"

META = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\splits\test.csv"

SPLIT = "test"

MAX_FRAMES = 60

FEATURE_DIM = 171   # 2 hands + pose

# -----------------------------

os.makedirs(os.path.join(OUT_DIR, SPLIT), exist_ok=True)

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

def npy_to_video(npy_path):
    name = os.path.basename(npy_path)
    video = name.replace(".npy", ".mp4")
    return os.path.join(RAW_DIR, video)

# -----------------------------

def extract(video_path):

    cap = cv2.VideoCapture(video_path)

    seq = []

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h = hands.process(rgb)
        p = pose.process(rgb)

        features = np.zeros(FEATURE_DIM, dtype=np.float32)

        idx = 0

        # -------- HANDS (2 × 21 × 3 = 126) --------
        if h.multi_hand_landmarks:

            for hand in h.multi_hand_landmarks[:2]:
                for lm in hand.landmark:
                    features[idx:idx+3] = [lm.x, lm.y, lm.z]
                    idx += 3

        idx = 126  # force jump to pose section

        # -------- POSE (15 × 3 = 45) --------
        if p.pose_landmarks:

            for lm in p.pose_landmarks.landmark[:15]:
                features[idx:idx+3] = [lm.x, lm.y, lm.z]
                idx += 3

        seq.append(features)

        if len(seq) >= MAX_FRAMES:
            break

    cap.release()

    # -------- PAD SHORT VIDEOS --------
    while len(seq) < MAX_FRAMES:
        seq.append(np.zeros(FEATURE_DIM, dtype=np.float32))

    return np.stack(seq)

# -----------------------------

df = pd.read_csv(META)

save_dir = os.path.join(OUT_DIR, SPLIT)

rows = []

print(f"🚀 Extracting keypoints for {SPLIT} set...")
print("Total videos:", len(df))

for _, row in tqdm(df.iterrows(), total=len(df)):

    seq_path = row["sequence_path"]
    label = row["label"]

    video_path = npy_to_video(seq_path)

    out_name = os.path.basename(video_path).replace(".mp4", ".npy")

    out_path = os.path.join(save_dir, out_name)

    seq = extract(video_path)

    np.save(out_path, seq)

    rows.append({
        "kp_path": out_path,
        "label": label
    })

csv_out = os.path.join(OUT_DIR, f"{SPLIT}.csv")

pd.DataFrame(rows).to_csv(csv_out, index=False)

print(f"\n✅ DONE!")
print(f"{SPLIT.upper()} keypoints CSV saved to:")
print(csv_out)
