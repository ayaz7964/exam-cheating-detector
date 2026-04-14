
# import streamlit as st
# import cv2
# import tempfile
# from ultralytics import YOLO
# from utils import predict_frame
# from alerts import beep_alert
# import time

# # =========================
# # LOAD MODEL (only once)
# # =========================
# @st.cache_resource
# def load_model():
#     return YOLO("../model/best.pt")

# model = load_model()

# # =========================
# # PAGE CONFIG
# # =========================
# st.set_page_config(page_title="Exam Cheating Detector", layout="wide")

# st.title("🎓 AI Exam Cheating Detection System")
# st.markdown("Real-time monitoring using Computer Vision")

# # =========================
# # SIDEBAR SETTINGS
# # =========================
# st.sidebar.title("⚙️ Settings")

# option = st.sidebar.selectbox("Choose Mode", ["Webcam", "Upload Video"])

# confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# frame_width = st.sidebar.slider("Frame Width", 320, 1280, 640)
# frame_height = st.sidebar.slider("Frame Height", 240, 720, 480)

# # =========================
# # FUNCTION: DRAW BOX (SIMULATED)
# # =========================
# def draw_ui_box(frame, label, conf):
#     """
#     Draws a full-frame bounding box + label (since classification model)
#     """

#     h, w, _ = frame.shape

#     # Color logic
#     if label != "normal act":
#         color = (0, 0, 255)  # Red for cheating
#     else:
#         color = (0, 255, 0)  # Green for normal

#     # Draw rectangle around full frame
#     cv2.rectangle(frame, (10, 10), (w-10, h-10), color, 3)

#     # Label text
#     text = f"{label} ({conf:.2f})"

#     # Draw background rectangle for text
#     cv2.rectangle(frame, (10, 10), (350, 50), color, -1)

#     # Put text
#     cv2.putText(frame, text, (15, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

#     return frame


# # =======================
# # 📷 WEBCAM MODE
# # =======================
# if option == "Webcam":
#     st.subheader("📷 Live Webcam Monitoring")

#     run = st.checkbox("Start Webcam")

#     FRAME_WINDOW = st.image([])

#     if run:
#         cap = cv2.VideoCapture(0)

#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to access webcam")
#                 break

#             # 🔥 Resize frame (IMPROVES SPEED)
#             frame = cv2.resize(frame, (frame_width, frame_height))

#             # 🔥 Prediction
#             # label, conf = predict_frame(model, frame)

#             detections = predict_frame(model, frame)


#             # 🔥 Draw UI box
#             frame = draw_ui_box(frame, label, conf)

#             # 🔥 Alert logic
#             if label != "normal act" and conf > confidence_threshold:
#                 beep_alert()

#             # Display frame
#             FRAME_WINDOW.image(frame, channels="BGR")

#             # 🔥 Small delay to reduce CPU load
#             time.sleep(0.01)

#         cap.release()


# # =======================
# # 🎥 VIDEO UPLOAD MODE
# # =======================
# elif option == "Upload Video":
#     st.subheader("🎥 Upload Video for Detection")

#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

#     if uploaded_file:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())

#         cap = cv2.VideoCapture(tfile.name)

#         stframe = st.empty()
#         cheating_timestamps = []

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         frame_count = 0

#         progress_bar = st.progress(0)

#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Resize frame
#             frame = cv2.resize(frame, (frame_width, frame_height))

#             # Prediction
#             label, conf = predict_frame(model, frame)

#             # Draw UI box
#             frame = draw_ui_box(frame, label, conf)

#             # Time calculation
#             time_sec = frame_count / fps if fps > 0 else 0

#             # Alert logic
#             if label != "normal act" and conf > confidence_threshold:
#                 cheating_timestamps.append(round(time_sec, 2))
#                 beep_alert()

#             # Display frame
#             stframe.image(frame, channels="BGR")

#             # Update progress bar
#             progress_bar.progress(min(frame_count / total_frames, 1.0))

#             frame_count += 1

#         cap.release()

#         st.success("✅ Processing Complete")

#         st.write("### 🚨 Cheating detected at (seconds):")
#         st.write(cheating_timestamps)






import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from alerts import beep_alert
import time

# =========================
# LOAD MODEL (ONLY ONCE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("../model/best.pt")  # Detection model

model = load_model()

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Exam Cheating Detector", layout="wide")

st.title("🎓 AI Exam Cheating Detection System")
st.markdown("Real-time cheating detection using YOLOv8 Detection Model")

# =========================
# SIDEBAR SETTINGS
# =========================
st.sidebar.title("⚙️ Settings")

option = st.sidebar.selectbox("Choose Mode", ["Webcam", "Upload Video"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

frame_width = st.sidebar.slider("Frame Width", 320, 1280, 640)
frame_height = st.sidebar.slider("Frame Height", 240, 720, 480)

# =========================
# DETECTION FUNCTION
# =========================
def detect_frame(model, frame):
    """
    Runs YOLO detection on a frame and returns detections
    """
    results = model(frame, imgsz=640, conf=0.4, verbose=False)[0]

    detections = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = results.names[cls]

            detections.append({
                "label": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })

    return detections


# =======================
# 📷 WEBCAM MODE
# =======================
if option == "Webcam":
    st.subheader("📷 Live Webcam Monitoring")

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access webcam")
                break

            # 🔥 Resize for speed
            frame = cv2.resize(frame, (frame_width, frame_height))

            # 🔥 Run detection
            detections = detect_frame(model, frame)

            # 🔥 Draw detections
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["label"]
                conf = det["confidence"]

                # Color based on class
                if label.lower() == "cheating":
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                # 🔥 Alert logic
                if label.lower() == "cheating" and conf > confidence_threshold:
                    beep_alert()

            # Show frame
            FRAME_WINDOW.image(frame, channels="BGR")

            # Reduce CPU load
            time.sleep(0.01)

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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))

            # Detection
            detections = detect_frame(model, frame)

            time_sec = frame_count / fps if fps > 0 else 0

            # Draw detections
            for det in detections:
                x1, y1, x2, y2 = det["box"]
                label = det["label"]
                conf = det["confidence"]

                if label.lower() == "cheating":
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame,
                            f"{label} {conf:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

                # Alert + timestamp
                if label.lower() == "cheating" and conf > confidence_threshold:
                    cheating_timestamps.append(round(time_sec, 2))
                    beep_alert()

            # Display frame
            stframe.image(frame, channels="BGR")

            # Progress bar
            progress_bar.progress(min(frame_count / total_frames, 1.0))

            frame_count += 1

        cap.release()

        st.success("✅ Processing Complete")

        st.write("### 🚨 Cheating detected at (seconds):")
        st.write(cheating_timestamps)