# Facial embedder using OpenCV and ArcFace

import cv2
import numpy as py
import pickle
import os
from insightface.app import FaceAnalysis

DB_PATH = "face_db.pkl" # The database

# --------------- Helpers -----------------
def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

def cosine_sim(a, b):
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))

def save_db(db, path=DB_PATH):
    with open(path, "wb") as f:
        pickle.dump({k: v.tolist() for k, v in db.items()}, f)

def load_db(path=DB_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # convert lists back to numpy arrays
    return {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}

# Ensure the model folder is flattened
def flatten_model_folder(model_dir):
    entries = os.listdir(model_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(model_dir, entries[0])):
        subfolder = os.path.join(model_dir, entries[0])
        for f in os.listdir(subfolder):
            shutil.move(os.path.join(subfolder, f), model_dir)
        os.rmdir(subfolder)