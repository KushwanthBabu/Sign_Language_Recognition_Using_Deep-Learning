import tensorflow as tf
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

FRAMES = 60
FEATS = 171

# -----------------------------
# DATASET LOADER
# -----------------------------

def load_split(csv_path, batch_size=32, shuffle=True):

    df = pd.read_csv(csv_path)

    paths = df["kp_path"].values.astype(str)
    labels = df["label"].values.astype(np.int32)

    def _load(path, label):

        x = np.load(path.numpy().decode()).astype(np.float32)

        # Ensure fixed length
        if x.shape[0] < FRAMES:
            pad = np.zeros((FRAMES, FEATS), dtype=np.float32)
            pad[: x.shape[0]] = x
            x = pad

        elif x.shape[0] > FRAMES:
            x = x[:FRAMES]

        # Ensure correct feature count
        if x.shape[1] != FEATS:
            fixed = np.zeros((FRAMES, FEATS), dtype=np.float32)
            m = min(x.shape[1], FEATS)
            fixed[:, :m] = x[:, :m]
            x = fixed

        return x, label

    def tf_load(path, label):

        x, y = tf.py_function(
            _load,
            [path, label],
            [tf.float32, tf.int32],
        )

        x.set_shape((FRAMES, FEATS))
        y.set_shape(())

        return x, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds
