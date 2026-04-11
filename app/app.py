import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from utils import predict_frame
from alerts import beep_alert

# Load model
model = YOLO("../model/best.pt")

st.set_page_config(page_title="Exam Cheating Detector", layout="wide")

st.title("🎓 AI Exam Cheating Detection System")

option = st.sidebar.selectbox("Choose Mode", ["Webcam", "Upload Video"])

# =======================
# 📷 WEBCAM MODE
# =======================
if option == "Webcam":
    st.subheader("📷 Live Webcam Monitoring")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf = predict_frame(model, frame)

        # Alert logic
        if label != "normal act" and conf > 0.7:
            beep_alert()
            cv2.putText(frame, f"🚨 {label}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()


# =======================
# 🎥 VIDEO UPLOAD MODE
# =======================
elif option == "Upload Video":
    st.subheader("🎥 Upload Video for Detection")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        cheating_timestamps = []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            label, conf = predict_frame(model, frame)

            time_sec = frame_count / fps

            if label != "normal act" and conf > 0.7:
                cheating_timestamps.append(round(time_sec, 2))
                beep_alert()

                cv2.putText(frame, f"🚨 {label}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            stframe.image(frame, channels="BGR")
            frame_count += 1

        cap.release()

        st.success("✅ Processing Complete")

        st.write("### 🚨 Cheating detected at (seconds):")
        st.write(cheating_timestamps)