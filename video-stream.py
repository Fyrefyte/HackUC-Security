from flask import Flask, Response
import cv2

app = Flask(__name__)
cap = cv2.VideoCapture(0)  # Use default camera

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Optional: resize or process the frame here
        # frame = cv2.resize(frame, (640, 480))

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield in multipart format for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
