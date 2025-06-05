import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Live Webcam", layout="centered")

st.markdown("<h1 style='text-align:center;'>Live Mask Detection</h1>", unsafe_allow_html=True)
st.markdown("---")

model = tf.keras.models.load_model("mask_detection_model.h5")
label_map = {
    0: "Mask Worn Incorrectly",
    1: "Mask Worn Correctly",
    2: "No Mask"
}

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap and cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.error("Camera read failed.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = rgb_frame.shape
            x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
            x2, y2 = int((bbox.xmin + bbox.width) * w), int((bbox.ymin + bbox.height) * h)

            face_crop = rgb_frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            resized = cv2.resize(face_crop, (128, 128)) / 255.0
            pred = model.predict(np.expand_dims(resized, axis=0))
            idx = np.argmax(pred)
            label = label_map[idx]

            color = (0, 255, 0) if idx == 1 else ((0, 255, 255) if idx == 0 else (0, 0, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if cap:
    cap.release()
