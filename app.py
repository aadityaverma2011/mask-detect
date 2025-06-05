import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load trained model
model = tf.keras.models.load_model("mask_detection_model.h5")

# Label mapping
label_map = {
    0: "Mask Worn Incorrectly",
    1: "Mask Worn Correctly",
    2: "No Mask"
}

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Streamlit page setup
st.set_page_config(page_title="😷 Mask Detection", layout="centered")
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: white !important;
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, p, label {
            color: #111 !important;
        }
        .result-box {
            padding: 18px;
            border-radius: 10px;
            font-size: 1.1rem;
            text-align: center;
            margin-top: 16px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align:center;'>😷 Mask Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a face image to check mask usage and biometric access.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload
uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    (h, w) = img_array.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(img_array, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    face_found = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_found = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face_crop = img_array[y1:y2, x1:x2]

            # Preprocess
            face = cv2.resize(face_crop, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            # Predict
            pred = model.predict(face)
            label_index = np.argmax(pred)
            label = label_map[label_index]
            confidence = float(np.max(pred))

            # Log
            print(f"Prediction: {pred}")
            print(f"Index: {label_index} — Label: {label} — Confidence: {confidence:.4f}")

            # Display
            st.image(Image.fromarray(face_crop), caption="🖼️ Detected Face", use_container_width=True)

            if label_index == 1:
                st.success("✅ Biometrics Authorized")
                st.markdown(f"<div class='result-box' style='background-color:#e8f5e9;'>{label}</div>", unsafe_allow_html=True)
            elif label_index == 0:
                st.warning("⚠️ Biometrics Authorized — Please wear your mask correctly")
                st.markdown(f"<div class='result-box' style='background-color:#fff8e1;'>{label}</div>", unsafe_allow_html=True)
            else:
                st.error("❌ Biometrics Not Authorized")
                st.markdown(f"<div class='result-box' style='background-color:#ffebee;'>{label}</div>", unsafe_allow_html=True)
            break

    if not face_found:
        st.error("❌ No face detected with sufficient confidence.")
