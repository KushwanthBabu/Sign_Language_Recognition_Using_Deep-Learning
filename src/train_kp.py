import os
import random
import numpy as np
import tensorflow as tf

from dataset_kp import load_split
from model_kp import build_model

# -----------------------------
# 🔒 Reproducibility
# -----------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# PATHS
# -----------------------------

TRAIN = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\keypoints\train.csv"
VAL   = r"C:\Users\AB\Downloads\Sign_Recognition_Model\data\keypoints\val.csv"

# -----------------------------
# Performance flags (CPU safe)
# -----------------------------

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

# -----------------------------
# Load datasets
# -----------------------------

train_ds = load_split(TRAIN, batch_size=32, shuffle=True)
val_ds   = load_split(VAL, batch_size=32, shuffle=False)

# -----------------------------
# Build model
# -----------------------------

model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Callbacks
# -----------------------------

callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath="best_kp_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1,
    ),
]

# -----------------------------
# Train
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=40,
    callbacks=callbacks,
)

# -----------------------------
# Save final model too
# -----------------------------

model.save("final_kp_model")

print("\n✅ Training complete.")
print("Best model saved as: best_kp_model.h5")
print("Final model folder: final_kp_model/")
