# utils.py

from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

# Load YOLOv8 model from the weights path
def load_model(weights_path="runs/detect/yolov8_fabric_defect4/weights/best.pt"):
    model = YOLO(weights_path)
    return model

# Run prediction on uploaded image
def predict_image(model, image):
    results = model.predict(source=image, save=False, conf=0.25)
    annotated_frame = results[0].plot()  # Overlay boxes on the image
    return annotated_frame

# Run prediction from webcam (frame by frame)
def predict_webcam_frame(model, frame):
    results = model.predict(source=frame, save=False, conf=0.25)
    return results[0].plot()
