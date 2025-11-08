# Motion detector using OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the camera feed or video file
vid_path = input("Video path (enter 0 for camera): ")
if vid_path != "0":
    cap = cv2.VideoCapture(vid_path)
else:
    cap = cv2.VideoCapture(0)

# Setup background subtract to seperate moving/static objects
backSub = cv2.createBackgroundSubtractorMOG2()
if not cap.isOpened():
    print("Couldn't find video source")

# As long as the video is running, keep looking for movement
while cap.isOpened():

    # Get a frame
    ret, frame = cap.read()
    if not ret:
        break
    frame_out = frame.copy()

    # Apply the mask to the frame to remove the static background
    fg_mask = backSub.apply(frame)

    # Apply global threshold to remove shadows
    retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

    # Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_open = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_clean = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_dilated = cv2.dilate(mask_clean, kernel, iterations=1)

    # Contours
    contours, hierarchy = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter
    min_contour_area = 2000
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Draw everything
    for cnt in large_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame_out, (x, y), (x+w, y+h), (0, 0, 200), 3)
    
    # Display the resulting frame
    cv2.imshow('Motion Detection - press q to quit', frame_out)

    # Allow the user to quit
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key & 0xFF == ord('\x1b'): # Also allow "escape" because someone's going to try that
        break