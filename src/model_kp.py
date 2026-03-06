import tensorflow as tf
from tensorflow.keras import layers, models

# -----------------------------
# CONFIG
# -----------------------------

NUM_CLASSES = 64
FRAMES = 60
FEATS = 171

# -----------------------------
# MODEL
# -----------------------------

def build_model():

    inp = layers.Input(shape=(FRAMES, FEATS), name="keypoints")

    # Normalize per feature
    x = layers.BatchNormalization()(inp)

    # Temporal modeling
    x = layers.Bidirectional(
        layers.LSTM(
            256,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.1,
        )
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            128,
            return_sequences=False,
            dropout=0.3,
            recurrent_dropout=0.1,
        )
    )(x)

    # Classifier head
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs=inp, outputs=out, name="BiLSTM_Keypoints")

    return model
