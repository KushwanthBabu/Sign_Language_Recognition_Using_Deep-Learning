# 🤟 Sign Language Recognition System

> Real-time sign language gesture recognition using MediaPipe keypoints and a Bidirectional LSTM deep learning model — runs entirely on CPU with sub-10ms inference.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-green)
![Flask](https://img.shields.io/badge/Flask-Web%20API-lightgrey?logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project bridges the communication gap between the hearing-impaired community and the hearing world by building an end-to-end, real-time sign language recognition system. Instead of processing raw image pixels, the system extracts **171 skeletal keypoints** per frame using MediaPipe and classifies gesture sequences with a **2-layer Bidirectional LSTM** — achieving high accuracy at very low computational cost.

| Metric | Result |
|---|---|
| Training Accuracy | **99.33%** |
| Validation Accuracy | **97.22%** |
| Test Accuracy | **89.57%** |
| Inference Latency | **~0.009 seconds** |
| Gesture Classes | **64** |

---

## 🏗️ System Architecture

```
Webcam / Video Input
        ↓
Frame Extraction (OpenCV @ 640×480)
        ↓
MediaPipe Holistic (Hands + Pose)
        ↓
Keypoint Extraction — 171 features per frame
  ├── 2 hands × 21 landmarks × 3 (x,y,z) = 126
  └── 15 pose landmarks × 3 (x,y,z)      =  45
        ↓
Sequence Builder (60-frame sliding window)
        ↓
BiLSTM Model
  ├── BatchNormalization
  ├── BiLSTM(256, return_sequences=True)
  ├── BiLSTM(128, return_sequences=False)
  └── Dense(256) → Dropout(0.5)
        ↓
Softmax Classification (64 classes)
        ↓
Predicted Sign Text + Confidence Score
```

---

## 🚀 Features

- ✅ Real-time webcam sign recognition with confidence thresholding
- ✅ Video & image file upload via browser interface
- ✅ Lightweight — no GPU required, runs on standard CPU
- ✅ Flask-based web interface accessible from any browser
- ✅ Prediction smoothing via rolling average window (reduces jitter)
- ✅ Modular, clean codebase with separate train / infer / web pipelines

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow 2.x / Keras |
| **Pose Estimation** | MediaPipe (Hands + Pose) |
| **Computer Vision** | OpenCV |
| **Web Backend** | Flask |
| **Frontend** | HTML5 + JavaScript + WebRTC |
| **Data** | NumPy, Pandas |

---

## 📂 Project Structure

```
Sign_Recognition_Model/
├── src/
│   ├── model_kp.py           # BiLSTM model definition
│   ├── train_kp.py           # Training script
│   ├── dataset_kp.py         # TF Dataset loader
│   ├── extract_keypoints.py  # MediaPipe keypoint extraction
│   ├── webcam_live.py        # Standalone webcam inference
│   ├── organize_raw.py       # Dataset organization utility
│   └── build_raw_splits.py   # Train/Val/Test split builder
├── data/
│   ├── label_map.json        # Gesture ID → label mapping
│   └── splits/               # train.csv / val.csv / test.csv
├── templates/
│   └── index.html            # Web interface
├── web_interface.py          # Flask app entrypoint
├── evaluate_model_splits.py  # Model evaluation script
├── .gitignore
└── README.md
```

> ⚠️ `data/raw/`, `data/keypoints/`, and model `.h5` files are excluded from this repo due to size. Download them using the links below.

---

## 📥 Download Model & Dataset

| Resource | Link |
|---|---|
| 🧠 Trained Model (`best_kp_model.h5`) | [Google Drive — Add Link](#) |
| 📂 Raw Dataset (videos) | [Google Drive — Add Link](#) |
| 📂 Keypoints (`.npy` arrays) | [Google Drive — Add Link](#) |

Place downloaded files as follows:
```
src/best_kp_model.h5
data/raw/          ← extracted videos
data/keypoints/    ← extracted .npy files
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/sign-language-recognition.git
cd sign-language-recognition

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install tensorflow mediapipe opencv-python flask numpy pandas tqdm
```

---

## ▶️ Running the App

### Option A — Web Interface (Recommended)
```bash
python web_interface.py
```
Then open **http://127.0.0.1:5000** in your browser.  
Use your webcam or upload a video/image to get a prediction.

### Option B — Standalone Webcam (OpenCV Window)
```bash
cd src
python webcam_live.py
```
Hold a sign steady for 2–3 seconds. Press **Q** to quit.

---

## 🧠 Model Details

The model takes a sequence of **60 frames × 171 keypoint features** as input.

```
Input  →  (60, 171)
BatchNormalization
BiLSTM(256, return_sequences=True,  dropout=0.3)
BiLSTM(128, return_sequences=False, dropout=0.3)
Dense(256, relu) → Dropout(0.5)
Dense(64,  softmax)  ←  Output: 64 gesture classes
```

**Training config:** Adam (lr=1e-3), Sparse Categorical Crossentropy, batch=32, max 40 epochs with EarlyStopping + ReduceLROnPlateau.

---

## 🔁 Training From Scratch

```bash
# Step 1 — Organize raw videos
python src/organize_raw.py

# Step 2 — Build train/val/test splits
python src/build_raw_splits.py

# Step 3 — Extract keypoints (run for train, val, test splits)
python src/extract_keypoints.py

# Step 4 — Train the model
cd src
python train_kp.py

# Step 5 — Evaluate
python evaluate_model_splits.py
```

---

## 📊 Results

| Split | Accuracy |
|---|---|
| Train | 99.33% |
| Validation | 97.22% |
| Test | 89.57% |

Inference latency: **~0.009 sec/prediction** on CPU.

---

## 🌍 SDG Alignment

- **SDG 10 — Reduced Inequalities**: Reduces communication barriers for the hearing-impaired
- **SDG 3 — Good Health & Well-being**: Supports healthcare access for the deaf community
- **SDG 4 — Quality Education**: Enables educational accessibility for sign language learners

---

## 🔮 Future Work

- [ ] Scale to 500+ gesture vocabulary
- [ ] Sentence generation from gesture sequences (NLP integration)
- [ ] TensorFlow Lite export for mobile deployment
- [ ] Transformer-based sequence model (replace BiLSTM)
- [ ] Edge device deployment (Raspberry Pi / Jetson Nano)
- [ ] Multilingual sign language support (ASL, ISL, BSL)

---

## 📎 Project Links

| Resource | Link |
|---|---|
| 🎬 Demo Video | [YouTube / Google Drive — Add Link](#) |
| 🖥️ Live Demo | [Render / AWS — Add Link](#) |
| 📊 Presentation | [Google Drive — Add Link](#) |

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [Google MediaPipe](https://mediapipe.dev/) — Pose & hand landmark detection
- [TensorFlow](https://tensorflow.org/) — Deep learning framework
- [OpenCV](https://opencv.org/) — Computer vision utilities
