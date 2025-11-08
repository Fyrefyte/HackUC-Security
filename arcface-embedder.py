# Facial embedder using OpenCV and ArcFace

import cv2
import numpy as py
import pickle
import os
from insightface.app import FaceAnalysis

main_font = cv2.FONT_HERSHEY_SIMPLEX
good_color = (0,255,0)
bad_color = (0,0,255)

DB_PATH = "face_db.pkl" # The database. Stores {name: embedding(np.array)}.

# --------------- Helpers -----------------
def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10) # The last bit ensures no divide by zero

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

# This part flattens the model directory where Antelope gets extracted.
def flatten_model_folder(model_dir):
    entries = os.listdir(model_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(model_dir, entries[0])):
        subfolder = os.path.join(model_dir, entries[0])
        for f in os.listdir(subfolder):
            shutil.move(os.path.join(subfolder, f), model_dir)
        os.rmdir(subfolder)

# Activate face detector via Antelope
model_dir = os.path.expanduser("~/.insightface/models/antelopev2")
flatten_model_folder(model_dir) # Ensure the models are where they should be

# Start up the facial recognition model. TODO Make sure to release this at the end
ctx_id = 0 # GPU mode
fa = FaceAnalysis(
    name="antelopev2",
    det_name = "2d106det.onnx", # det_name and rec_name may be unnecessary, remember to test it 
    rec_name = "1k3d68.onnx",
    providers=['CPUExecutionProvider'])

# Prepare the face analysis. 
fa.prepare(ctx_id=ctx_id, det_size=(640,640)) # det_size determines precision, 320,320 may be better for speed in this application

# Load up the database for later. This will store our saved faces. TODO Add ecryption
face_db = load_db();

# --------------- Enrollment function -----------------
def enroll_from_camera(name, n_samples=10, required_confidence=0.5, sample_delay=0.5):
    func_id = "ENROLL"

    cap = cv2.VideoCapture(0); # Default device camera
    embeddings = []; # The array of new embeddings
    print(f"[{func_id}] Look at the camera for {n_samples} samples ({n_samples*sample_delay} seconds)...")
    
    # Loop through each sample and average at the end
    while (len(embeddings) < n_samples):

        # Get a frame
        ret, frame = cap.read()
        if not ret:
            break
        bgr = frame

        # Get all the faces in frame
        faces = fa.get(bgr)

        if faces:
            # Get the biggest face in frame and add it to the list
            face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))

            # If the face isn't well recognized as a face:
            if face.det_score < required_confidence:
                cv2.putText(frame, f"Low conf {face.det_score:.2f}", (10,30), main_font, 0.8, bad_color, 2)
            else:
                emb = np.asarray(face.embedding, dtype=np.float32)
                embeddings.append(l2_normalize(emb))
                cv2.putText(frame, f"Captured {len(embeddings)}/{n_samples}", (10,30), main_font, 0.8, good_color, 2)

        # Allow the user to quit
        cv2.imshow("Enroll - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == ord('\x1b'): # Also allow "escape" because someone's going to try that
            break

    # When done, release the video capture and get rid of the window
    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings == 0):
        print(f"[{func_id}] No usable data.")
        return False
    
    # Takes the average and normalizes it
    avg_emb = l2_normalize(np.mean(np.stack(embeddings), axis=0))
    face_db[name] = avg_emb

    # Save the new face to the database
    save_db(face_db)
    print(f"[{func_id}] Saved {name} with {len(embeddings)} samples.")
    return True

