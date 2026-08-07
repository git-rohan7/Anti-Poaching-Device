import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


st.set_page_config(page_title="Human Detection (Thermal)", layout="wide")
st.title("🧍 Human Detection — Thermal Images")
st.caption("Live camera, photo upload, and video upload, powered by your YOLOv8 model.")

with st.sidebar:
    st.header("Model")
    weights_path = st.text_input(
        "Model weights (.pt)",
        value="best_human_thermal.pt",
        help="Path to the best.pt produced by train.py",
    )
    conf_thresh = st.slider("Confidence threshold", 0.05, 0.95, 0.35, 0.05)
    iou_thresh = st.slider("IoU threshold (NMS)", 0.1, 0.9, 0.45, 0.05)

    @st.cache_resource(show_spinner="Loading model...")
    def load_model(path):
        return YOLO(path)

    model = None
    if Path(weights_path).exists():
        model = load_model(weights_path)
        st.success(f"Loaded model: {weights_path}")
    else:
        st.warning("Model file not found yet. Enter a valid path to best.pt above.")


def annotate(frame_bgr, model, conf, iou):
    """Run YOLO inference on a BGR frame and return the annotated BGR frame."""
    results = model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)
    annotated = results[0].plot() 
    n_detections = len(results[0].boxes)
    return annotated, n_detections


tab_live, tab_photo, tab_video = st.tabs(
    ["📷 Live Camera", "🖼️ Photo Upload", "🎞️ Video Upload"]
)

## live camera testing
with tab_live:
    st.subheader("Live camera testing")

    if model is None:
        st.info("Load a model in the sidebar first.")
    elif WEBRTC_AVAILABLE:
        st.write("Grant camera permission in the browser prompt, then detections "
                 "will be drawn on the live feed in real time.")

        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        class Detector:
            def __init__(self):
                self.model = model
                self.conf = conf_thresh
                self.iou = iou_thresh

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                annotated, _ = annotate(img, self.model, self.conf, self.iou)
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="live-human-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=Detector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.warning(
            "For real-time streaming, install `streamlit-webrtc` "
            "(`pip install streamlit-webrtc`). Using snapshot mode instead."
        )
        cam_image = st.camera_input("Take a photo")
        if cam_image is not None:
            file_bytes = np.frombuffer(cam_image.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, n = annotate(frame, model, conf_thresh, iou_thresh)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                     caption=f"{n} person(s) detected")

#photo upload
with tab_photo:
    st.subheader("Photo upload")

    if model is None:
        st.info("Load a model in the sidebar first.")
    else:
        uploaded_img = st.file_uploader(
            "Upload a thermal image", type=["jpg", "jpeg", "png", "bmp"]
        )
        if uploaded_img is not None:
            file_bytes = np.frombuffer(uploaded_img.getvalue(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original")

            with st.spinner("Running detection..."):
                annotated, n = annotate(frame, model, conf_thresh, iou_thresh)

            with col2:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          caption=f"Detected: {n} person(s)")

            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                st.download_button(
                    "Download annotated image",
                    data=buf.tobytes(),
                    file_name="detected_" + uploaded_img.name,
                    mime="image/jpeg",
                )

#video upload
with tab_video:
    st.subheader("Video upload")

    if model is None:
        st.info("Load a model in the sidebar first.")
    else:
        uploaded_vid = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_vid is not None:
            tmp_in = Path("tmp_input_video") / uploaded_vid.name
            tmp_in.parent.mkdir(exist_ok=True)
            tmp_in.write_bytes(uploaded_vid.getvalue())

            st.video(str(tmp_in))
            run_btn = st.button("Run detection on this video")

            if run_btn:
                cap = cv2.VideoCapture(str(tmp_in))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                out_path = Path("tmp_output_video") / f"detected_{uploaded_vid.name}"
                out_path.parent.mkdir(exist_ok=True)
                # mp4v is broadly compatible; browser playback works for most cases
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

                progress = st.progress(0.0)
                status = st.empty()
                frame_idx = 0
                t0 = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotated, n = annotate(frame, model, conf_thresh, iou_thresh)
                    writer.write(annotated)
                    frame_idx += 1
                    if total_frames > 0:
                        progress.progress(min(frame_idx / total_frames, 1.0))
                    status.text(f"Frame {frame_idx}/{total_frames or '?'} — {n} person(s) detected")

                cap.release()
                writer.release()
                elapsed = time.time() - t0
                st.success(f"Done in {elapsed:.1f}s. Processed {frame_idx} frames.")

                st.video(str(out_path))
                with open(out_path, "rb") as f:
                    st.download_button(
                        "Download annotated video",
                        data=f.read(),
                        file_name=out_path.name,
                        mime="video/mp4",
                    )
