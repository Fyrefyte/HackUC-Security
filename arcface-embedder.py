# Facial embedder using OpenCV and ArcFace

import cv2
import numpy as np
import pickle
import os
os.environ["ORT_LOG_VERBOSE"] = "1"
import threading
import time
import torch
import smtplib
import onnxruntime as ort
from insightface.app import FaceAnalysis
from queue import Queue, Empty, Full
from email.message import EmailMessage
from twilio.rest import Client
import imghdr
from flask import Flask, Response, send_from_directory
import socket
import base64
from io import BytesIO
import json
import torch

# Limit GPU memory usage to 50%
# torch.cuda.set_per_process_memory_fraction(0.1, device=0)  # fraction of total

cap = cv2.VideoCapture(0)
latest_frame = None
latest_frame_annotated = None
app = Flask(__name__)
event_queue = Queue()

def camera_thread():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame

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

def notify(body, frame):
    # Encode frame to JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    event = json.dumps({
        "type": "face_detected",
        "body": body,
        "image": jpg_as_text
    })

    # Write event into your queue
    event_queue.put(event)

def get_local_ips():
    ips = []
    hostname = socket.gethostname()
    try:
        # Try to get the primary IP
        primary_ip = socket.gethostbyname(hostname)
        ips.append(primary_ip)
    except:
        pass

    # Scan all interfaces
    try:
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if ip not in ips and ":" not in ip:  # skip IPv6 for simplicity
                ips.append(ip)
    except:
        pass

    return ips

def generate_frames():
    while True:

        # Get a frame
        if latest_frame_annotated is None:
            continue
        frame = latest_frame_annotated.copy()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield in multipart format for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Activate face detector via Antelope
model_dir = os.path.expanduser("~/.insightface/models/antelopev2")
flatten_model_folder(model_dir) # Ensure the models are where they should be

# Debugging
print(ort.get_available_providers())
print(torch.cuda.is_available())

# Start up the facial recognition model.
ctx_id = 0 # GPU mode
fa = FaceAnalysis(
    name="antelopev2",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# Prepare the face analysis.
fa.prepare(ctx_id=ctx_id, det_size=(640,640)) # det_size determines precision, 320,320 may be better for speed in this application

# Load up the database for later. This will store our saved faces. TODO Add ecryption
face_db = load_db();

# --------------- Enrollment function -----------------
def enroll_from_camera(name, n_samples=10, required_confidence=0.5):
    func_id = "ENROLL"

    embeddings = []; # The array of new embeddings
    print(f"[{func_id}] {n_samples} are being collected. Move around, look in different directions, and make different expressions.")
    
    # Loop through each sample and average at the end
    while (len(embeddings) < n_samples):

        # Get a frame
        if latest_frame is None:
            continue
        frame = latest_frame.copy()
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
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key & 0xFF == ord('\x1b'): # Also allow "escape" because someone's going to try that
            break

    # When done, release the video capture and get rid of the window
    cv2.destroyAllWindows()

    # Chuck the data if there isn't any with faces
    if len(embeddings) == 0:
        print(f"[{func_id}] No usable data.")
        return False
    
    # Takes the average and normalizes it
    avg_emb = l2_normalize(np.mean(np.stack(embeddings), axis=0))
    face_db[name] = avg_emb

    # Save the new face to the database
    save_db(face_db)
    print(f"[{func_id}] Saved {name} with {len(embeddings)} samples.")
    return True

# ----------------- Remove enrollment -------------------
# In case you need to remove bad data or remove a person you no longer need in your system
def remove_from_database(name):
    func_id = "REMOVE"
    if name in face_db:
        del face_db[name]
        save_db(face_db)
        print(f"[{func_id}] Removed {name} from database.")
    else:
        print(f"[{func_id}] {name} not found in database.")

# ----------------- Clear database -------------------
# In case you need to remove all data (mostly for testing)
def clear_database():
    func_id = "CLEAR"
    face_db.clear()
    save_db(face_db)
    print(f"[{func_id}] Cleared database.")

# --------------- Recognize single frame -----------------
def recognize(bgr, sim_threshold=0.4):
    func_id = "RECOG"
    best_names = []
    best_scores = []

    # Get all the faces
    faces = fa.get(bgr)

    # Check each face and compile the results to return
    if faces:
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4]) # Get the bounds of the face
            emb = l2_normalize(np.asarray(face.embedding, dtype=np.float32))
            best_name = "Unknown" # Assume unknown until proven otherwise
            best_score = -1 # Arbitrarily low number

            # Compare the face to each face in the database and determine its score.
            # We take the highest score face.
            for name, db_emb in face_db.items():
                score = cosine_sim(emb, db_emb)
                if score > best_score and score > sim_threshold:
                    best_score = score
                    best_name = name
            
            # Add the results for the face to the array
            best_names.append(best_name)
            best_scores.append(best_score)
    
    # Return the results
    return faces, best_names, best_scores

# --------------- Recognition Thread ------------------
# Ensures that the recognition can run while the live stream stays live
frame_queue = Queue(maxsize=1)   # holds at most one frame (the most recent)
result = {
    "faces": [],
    "names": [],
    "scores": []
}
stop_event = threading.Event()   # allow clean shutdown if needed
def recognize_thread(sim_threshold):
    func_id = "RECOG THRD"

    global processing

    while not stop_event.is_set():
        try:
            # Wait for the newest frame (timeout so we can respond to stop_event)
            frame_copy = frame_queue.get(timeout=0.2)
        except Empty:
            continue

        # Get the faces and scores
        faces, best_names, best_scores = recognize(frame_copy, sim_threshold)

        # Update shared result
        result["faces"] = faces
        result["names"] = best_names
        result["scores"] = best_scores

        # Mark queue item done
        frame_queue.task_done()

# --------------- Recognition loop -----------------
cooldown = 2000
cooldown_timer = 0
def recognize_loop(sim_threshold=0.4, heat_threshold=240):
    func_id = "RECOG LOOP"
    heat = 0

    global latest_frame, cooldown, cooldown_timer, latest_frame_annotated

    # Configure the thread
    recogThread = threading.Thread(target=recognize_thread, kwargs={"sim_threshold": sim_threshold}, daemon = True)
    recogThread.start()

    print(f"[{func_id}] Starting webcam. Press q to quit.")
    while True:

        # Get a frame
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        try:
            frame_queue.put_nowait(frame.copy())
        except Full:
            try:
                _ = frame_queue.get_nowait() # drop the old frame
                frame_queue.task_done()
            except Empty:
                pass
            try:
                frame_queue.put_nowait(frame.copy())
            except Full:
                # If still full, skip; worker is busy and queue remains current
                pass

        # Update the faces with the global result
        faces = result["faces"]
        best_names = result["names"]
        best_scores = result["scores"]

        # Draw the indicators on each face
        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            label = f"{best_names[i]} {best_scores[i]:.3f}"
            color = good_color if best_scores[i] >= sim_threshold else bad_color
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, max(0,y1-10)), main_font, 0.6, color, 2)

            # My unidentified person "heat" system.
                # For every frame an unidentified person is in frame, heat increases by 1.
                # For every frame a known person is in frame, heat decreases by 10.
                    # This way, if someone in the system brings someone you don't know home, it won't ping you.
                    # The fact that this doesn't simply reset heat also prevents it from completely resetting if there is a brief malfunction.
                # If there are no faces in frame, heat slowly decreases.
                    # Thus, if the unidentified person is out of frame briefly, their heat won't go away completely and will still probably trigger a notification.
            if best_names[i] == "Unknown":
                heat += 1
            else:
                heat -= 10
        if not faces:
            heat -= 1
        heat = min(heat_threshold, max(0, heat))
        cv2.putText(frame, f"Heat: {heat}", (3, 23), main_font, 0.8, bad_color, 2)

        latest_frame_annotated = frame.copy()
        
        if cooldown_timer > 0:
            cooldown_timer -= 1

        # Emergency time!!!!
        if heat >= heat_threshold and cooldown_timer == 0:
            notify("ALERT: Unknown person detected", frame)
            send_email_alert_w_file(frame, "EchoGate Alert", "ALERT: Unknown person detected")
            heat = 0
            cooldown_timer = cooldown

        # Allow the user to quit
        cv2.imshow("Recognition - press q to quit", frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key & 0xFF == ord('\x1b'): # Also allow "escape" because someone's going to try that
            break
    
    # Clean shutdown
    stop_event.set()
    recogThread.join(timeout=1.0)
    stop_event.clear()

    # Give worker a moment to wake and exit
    try:
        # push a dummy frame to unblock the worker if it's waiting
        frame_queue.put_nowait(np.zeros((10,10,3), dtype=np.uint8))
    except Full:
        pass
    recogThread.join(timeout=1.0)

    # When done, release the video capture and get rid of the window
    cv2.destroyAllWindows()

# ----------------- Emergency alert system ------------------
# Email to SMS                                    DO NOT USE
# --- CONFIG ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "echo2gate@gmail.com" # dummy email
SMTP_PASS = "qeci zqpz nhbr kqkc" # TODO password here. THIS IS NOT PERMANENT, PLEASE FIX
# Phone number and carrier gateway:
# AT&T: @txt.att.net ---> discontinued in June 2025
# TMobile: @tmomail.net ---> discontinued in December 2024
# Verison: @vtext.com ---> technically exists but not consistent
# Sprint: @messaging.sprintpcs.com
PROVIDER_POSTFIX = "@vtext.com"
TO_SMS_ADDRESS = "6146801273" + PROVIDER_POSTFIX  # <-- put YOUR number and your carrier gateway here
FROM_EMAIL = "echo2gate@gmail.com"
# ----------------
def send_alert_via_email2sms(message_text: str):
    msg = EmailMessage()
    msg.set_content(message_text)
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_SMS_ADDRESS

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    # print("Email->SMS sent (if carrier accepted it).")

# Twilio (better alternative) (takes too long since they have to confirm it)
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = "" # TODO NEVER commit with this here
TWILIO_NUMBER = "+15136665816"     # Twilio phone number
ALERT_NUMBER = "+16147073765"      # Receiving phone number

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

def send_alert_twilio(message_text: str):
    client.messages.create(
        body=message_text,
        from_=TWILIO_NUMBER,
        to=ALERT_NUMBER
    )

# Normal old email. Lame but works for now.
TO_EMAIL = input("Input your email for notifications: ")
def send_email_alert(subject: str, body: str):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(FROM_EMAIL, SMTP_PASS)
            server.send_message(msg)
        print(f"[ALERT] Email sent successfully to {TO_EMAIL}")
    except Exception as e:
        print(f"[ALERT] Failed to send email: {e}")
def send_email_alert_w_file(frame, subject: str, body: str):
    # Encode frame as JPEG in memory
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        print("Failed to encode frame.")
        return

    img_bytes = encoded_image.tobytes()

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL

    # Determine image type for email attachment
    img_type = imghdr.what(None, img_bytes)  # should be "jpeg"

    msg.add_attachment(img_bytes, maintype="image", subtype=img_type, filename="capture.jpg")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(FROM_EMAIL, SMTP_PASS)
            server.send_message(msg)
        print(f"[ALERT] Email sent successfully to {TO_EMAIL}")
    except Exception as e:
        print(f"[ALERT] Failed to send email: {e}")

@app.route('/hls/<path:filename>')
def hls_files(filename):
    return send_from_directory('/path/to/output', filename)

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events')
def events():
    def stream():
        while True:
            msg = event_queue.get()
            yield f"data: {msg}\n\n"
    return Response(stream(), mimetype='text/event-stream')

@app.route('/')
def index():
    # Simple HTML page to view the video
    return '''
    <html>
        <head>
            <title>Live Camera Feed</title>
        </head>
        <body>
            <h1>Camera Feed</h1>
            <img src="/video" width="640" height="480" />
        </body>
    </html>
    '''

# ----------------- CLI-like entry -------------------
# This bit allows command line interaction with the script.
if __name__ == "__main__":
    print("\nServer started. Check for email confirmation.\nCommands: (r) recognize, (e) enroll, (p) print DB, (x) remove person, (c) clear database, (q) quit")
    send_email_alert("EchoGate Alert", "EchoGate has connected to this device for push notifications. Ignore this message if you didn't expect this.")
    threading.Thread(target=camera_thread, daemon=True).start() # Start the cam
    while True:
        cmd = input("cmd> ").strip().lower()
        if cmd in ("r", "recognize"):
            threading.Thread(target=lambda: app.run(host="0.0.0.0", port=8000), daemon=True).start() # Start the live feed
            recognize_loop(sim_threshold=0.4)
        elif cmd in ("e", "enroll"):
            n = input("Name to enroll: ").strip()
            if n in list(face_db.keys()):
                if not input("Person is already enrolled. Reenroll? (y/n): ").strip() in ("y"):
                    break
            s = input("Number of samples: ").strip()
            try:
                int(s)
            except ValueError:
                s = 50
            enroll_from_camera(n, n_samples=int(s))
        elif cmd in ("p", "print"):
            print("DB:", list(face_db.keys()))
        elif cmd in ("x", "remove person"):
            n = input("Name to remove: ").strip()
            remove_from_database(n)
        elif cmd in ("c", "clear database"):
            cont = input("Confirm clearing of data (y/n)? (This cannot be undone): ").strip()
            if cont in ("y", "confirm"):
                clear_database()
        elif cmd in ("q", "quit", "exit"):
            break
        else:
            print("Unknown command.")
    cap.release()