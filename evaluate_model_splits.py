import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import time
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# CONFIG
# ==============================

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data" / "splits_kp"
KP_ROOT = ROOT / "data" / "keypoints"

MODEL_PATH = ROOT / "src" / "best_kp_model.h5"

SEQ_LEN = 60
FEATS = 171

SPLITS = ["train", "test", "val"]

print("\nDATA:", DATA_DIR)
print("MODEL:", MODEL_PATH)

# ==============================
# LOAD MODEL
# ==============================

print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded")

# ==============================
# HELPERS
# ==============================

def fix_length(arr):
    """Pad or trim sequence to fixed frames"""
    if arr.shape[0] > SEQ_LEN:
        return arr[:SEQ_LEN]

    pad = SEQ_LEN - arr.shape[0]
    return np.pad(arr, ((0, pad), (0, 0)))


def load_split_csv(split):

    csv_path = DATA_DIR / f"{split}.csv"
    kp_dir = KP_ROOT / split

    print("\nLoading:", csv_path)

    df = pd.read_csv(csv_path)

    # assume first column has path, last is label
    path_col = df.columns[0]
    label_col = df.columns[-1]

    X = []
    y = []

    missing = 0

    for _, row in df.iterrows():

        stem = Path(row[path_col]).stem
        kp_file = kp_dir / f"{stem}.npy"

        if not kp_file.exists():
            missing += 1
            continue

        data = np.load(kp_file)

        # ensure correct shape
        if data.shape[1] != FEATS:
            continue

        data = fix_length(data)

        X.append(data)
        y.append(row[label_col])

    print(f"Found: {len(X)} samples | Missing: {missing}")

    return np.array(X), np.array(y)


# ==============================
# EVALUATION
# ==============================

def evaluate_split(split):

    X, y = load_split_csv(split)

    if len(X) == 0:
        print("❌ No valid samples in", split)
        return

    print(f"Evaluating {split.upper()} → {len(X)} samples")

    start = time.time()

    preds = model.predict(X, batch_size=16, verbose=0)

    end = time.time()

    y_pred = np.argmax(preds, axis=1)

    acc = accuracy_score(y, y_pred)

    latency = (end - start) / len(X)

    print("\nAccuracy:", round(acc * 100, 2), "%")
    print("Avg latency per sample:", round(latency, 4), "sec")

    print("\nClassification Report:")
    print(classification_report(y, y_pred))


# ==============================
# RUN ALL
# ==============================

for s in SPLITS:
    print("\n==============================")
    evaluate_split(s)
