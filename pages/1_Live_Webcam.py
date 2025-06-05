import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Live Webcam", layout="centered")
st.title("📷 Live Mask Detection")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mask_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

label_map = {
    0: "Mask Worn Incorrectly",
    1: "Mask Worn Correctly",
    2: "No Mask"
}

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

                resized = cv2.resize(face_crop, (128, 128)).astype(np.float32) / 255.0
                input_tensor = np.expand_dims(resized, axis=0)

                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

                idx = np.argmax(output_data)
                label = label_map[idx]

                color = (0, 255, 0) if idx == 1 else ((0, 255, 255) if idx == 0 else (0, 0, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return img

webrtc_streamer(key="live-mask-detect", video_processor_factory=VideoProcessor)
