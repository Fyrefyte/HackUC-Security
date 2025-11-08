# Motion detector using OpenCV

import cv2
# import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Get the camera feed or video file
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

    # Apply the mask to the frame to remove the static background
    fg_mask = backSub.apply(frame)

    # Find contours
    contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame_final', frame_ct)