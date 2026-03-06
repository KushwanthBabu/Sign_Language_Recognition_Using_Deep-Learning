import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\raw"

OUT_DIR = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\splits_kp"
os.makedirs(OUT_DIR, exist_ok=True)

rows = []

for fname in os.listdir(RAW_DIR):

    if not fname.endswith(".mp4"):
        continue

    # 001_002_004.mp4 → 001
    sign_id = int(fname.split("_")[0]) - 1

    rows.append({
        "video_path": os.path.join(RAW_DIR, fname),
        "label": sign_id
    })

df = pd.DataFrame(rows)

print("Total videos:", len(df))
print("Classes:", df.label.nunique())

# -------- stratified split --------

train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df.label,
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.67,
    stratify=temp.label,
    random_state=42
)

train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print("✅ Raw splits created:")
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
