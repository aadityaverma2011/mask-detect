import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Load trained mask detection model
model = tf.keras.models.load_model("mask_detection_model.h5")

# Label map (ensure order matches training)
label_map = {
    0: "Mask Worn Incorrectly",
    1: "Mask Worn Correctly",
    2: "No Mask"
}

# Load OpenCV DNN face detector
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

# Streamlit config
st.set_page_config(page_title="Mask Detection", layout="centered")
st.markdown("""
    <style>
        body, .stApp {
            background-color: white;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("😷 Mask Detection System")
st.markdown("Upload a face image to check mask status and biometric access.")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    (h, w) = img_array.shape[:2]

    # DNN face detection
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

            # Preprocess and predict
            face = cv2.resize(face_crop, (128, 128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            pred = model.predict(face)
            label_index = np.argmax(pred)
            label = label_map[label_index]
            confidence = float(np.max(pred))

            # Console output
            print(f"Prediction: {pred}")
            print(f"Index: {label_index} — Label: {label} — Confidence: {confidence:.4f}")

            # Feedback display
            st.image(Image.fromarray(face_crop), caption="Detected Face", use_container_width=True)
            if label_index == 1:
                st.success("✅ Biometrics Authorized")
            elif label_index == 0:
                st.warning("⚠️ Biometrics Authorized — Please wear your mask correctly")
            else:
                st.error("❌ Biometrics Not Authorized")

            break

    if not face_found:
        st.error("❌ No face detected with sufficient confidence.")
