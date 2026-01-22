import streamlit as st
import cv2
import os
import ultralytics

model = ultralytics.YOLO("yolo26n.pt")

st.title("Object Tracking System")
video_file = st.file_uploader("Upload video", type=['mp4','avi','mov','mkv'])

if video_file:
    data = video_file.read()
    with open("temp.mp4", "wb") as f:
        f.write(data)

    st.video("temp.mp4",format="video/mp4")
    if st.button("Start Tracking"):
        with st.spinner("Loading video..."):
            cap = cv2.VideoCapture("temp.mp4")
            if cap:
                frame_placeholder = st.empty()
                ret = True
                while ret:
                    ret, frame = cap.read()
                    if ret:
                        results = model.track(frame, persist=True)
                        out = results[0].plot()
                        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(out)
                cap.release()
                os.remove("temp.mp4")