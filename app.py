import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
import wave
import io
import math
import base64
from utils.face_detection import get_face_mesh_model, extract_landmarks, draw_landmarks
from utils.eye_tracking import check_eyes_closed, get_head_pose, check_yawn
from utils.attention_logic import classify_attention
from utils.db_manager import init_db, insert_log, fetch_logs, clear_logs

# Initialize DB
init_db()

def get_beep_base64():
    """Generates a 0.5s 440Hz beep sound and returns it as a base64 encoded string."""
    sample_rate = 44100
    duration = 0.5
    freq = 440.0
    num_samples = int(duration * sample_rate)
    
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        samples = []
        for i in range(num_samples):
            t = float(i) / sample_rate
            value = int(32767.0 * math.sin(2.0 * math.pi * freq * t))
            samples.append(value)
            
        import struct
        packed_samples = struct.pack('<' + 'h' * num_samples, *samples)
        wav_file.writeframes(packed_samples)
        
    wav_bytes = wav_io.getvalue()
    return base64.b64encode(wav_bytes).decode()

BEEP_B64 = get_beep_base64()

def play_beep(placeholder):
    """Plays the beep sound via HTML5 audio."""
    md = f'<audio autoplay><source src="data:audio/wav;base64,{BEEP_B64}" type="audio/wav"></audio>'
    placeholder.markdown(md, unsafe_allow_html=True)

st.set_page_config(page_title="FocusAI - Attention Tracking", layout="wide", page_icon="🧠")

st.title("🧠 FocusAI: AI-Based Student Attention Tracking System")
st.markdown("Monitor student engagement in real-time using advanced Computer Vision and AI.")

tab1, tab2 = st.tabs(["🎥 Live Tracking", "📊 Teacher Dashboard"])

with tab1:
    st.sidebar.header("Settings")
    ear_threshold = st.sidebar.slider("Eye Aspect Ratio (EAR) Threshold", 0.15, 0.30, 0.22, 0.01)
    mar_threshold = st.sidebar.slider("Mouth Aspect Ratio (Yawn) Threshold", 0.30, 0.80, 0.50, 0.05)
    show_mesh = st.sidebar.checkbox("Show Face Mesh", value=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("Real-Time Attention Metrics")
        status_placeholder = st.empty()
        score_placeholder = st.empty()
        direction_placeholder = st.empty()
        alert_placeholder = st.empty()
        audio_placeholder = st.empty()
        
        st.markdown("---")
        st.subheader("History Log")
        log_placeholder = st.empty()
    
    start_btn = st.sidebar.button("Start Tracking")
    stop_btn = st.sidebar.button("Stop Tracking")
    
    if 'run' not in st.session_state:
        st.session_state['run'] = False
    
    if start_btn:
        st.session_state['run'] = True
    
    if stop_btn:
        st.session_state['run'] = False
    
    if st.session_state['run']:
        cap = cv2.VideoCapture(0)
        face_mesh = get_face_mesh_model()
        
        history_log = []
        
        # Track timers per face (assuming 1 primary face for alerts to avoid chaos, but drawing all)
        distracted_start_time = None
        drowsy_start_time = None
        last_beep_time = 0
        
        while st.session_state['run']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video. Please ensure webcam is connected.")
                break
                
            frame = cv2.flip(frame, 1) # Mirror image
            h, w, _ = frame.shape
            
            results = extract_landmarks(frame, face_mesh)
            
            primary_status = "No Face Detected"
            primary_score = 0
            primary_direction = "Unknown"
            
            if results.face_landmarks:
                if show_mesh:
                    frame = draw_landmarks(frame, results)
                    
                # Process all detected faces (up to 5)
                for idx, face_landmarks in enumerate(results.face_landmarks):
                    # 1. Eye Tracking & Yawn
                    is_eyes_closed, avg_ear = check_eyes_closed(face_landmarks, w, h, ear_threshold)
                    is_yawning, avg_mar = check_yawn(face_landmarks, w, h, mar_threshold)
                    
                    # 2. Head Pose
                    is_looking_away, direction, x_angle, y_angle = get_head_pose(face_landmarks, w, h)
                    
                    # 3. Classify
                    status, score = classify_attention(is_eyes_closed, is_looking_away, is_yawning)
                    
                    # Assume first face is primary for UI metrics and alerts
                    if idx == 0:
                        primary_status = status
                        primary_score = score
                        primary_direction = direction
                        
                        # 4. Timer Logic
                        current_time_sec = time.time()
                        if status == "Focused":
                            distracted_start_time = None
                            drowsy_start_time = None
                            alert_placeholder.empty()
                            audio_placeholder.empty()
                        elif status == "Distracted":
                            drowsy_start_time = None
                            if distracted_start_time is None:
                                distracted_start_time = current_time_sec
                            elif current_time_sec - distracted_start_time > 5:
                                alert_placeholder.error("🚨 ALERT: Distracted for more than 5 seconds!")
                                if current_time_sec - last_beep_time > 2: # Beep every 2s
                                    play_beep(audio_placeholder)
                                    last_beep_time = current_time_sec
                        elif status == "Drowsy":
                            distracted_start_time = None
                            if drowsy_start_time is None:
                                drowsy_start_time = current_time_sec
                            elif current_time_sec - drowsy_start_time > 5:
                                alert_placeholder.error("🚨 ALERT: Drowsy! Please wake up!")
                                if current_time_sec - last_beep_time > 2:
                                    play_beep(audio_placeholder)
                                    last_beep_time = current_time_sec
                                    
                        # Log to DB occasionally (every ~1s to avoid spam)
                        if int(current_time_sec * 10) % 10 == 0:
                            insert_log(status, score)
                    
                    # Render Box/Text on Image for each face
                    nose = face_landmarks[1]
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    color = (0, 255, 0) if status == "Focused" else ((0, 165, 255) if status == "Distracted" else (0, 0, 255))
                    cv2.putText(frame, status, (nx - 40, ny - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                cv2.putText(frame, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)
            
            html_color = "green" if primary_status == "Focused" else "orange" if primary_status == "Distracted" else "red"
            status_placeholder.markdown(f"### Status: <span style='color:{html_color}'>{primary_status}</span>", unsafe_allow_html=True)
            score_placeholder.metric("Attention Score", f"{primary_score}%")
            direction_placeholder.markdown(f"**Head Direction:** {primary_direction}")
            
            current_time_str = time.strftime("%H:%M:%S")
            if not history_log or history_log[-1]["Time"] != current_time_str:
                history_log.append({"Time": current_time_str, "Status": primary_status, "Score": primary_score})
                if len(history_log) > 10:
                    history_log.pop(0)
                log_placeholder.dataframe(pd.DataFrame(history_log), use_container_width=True)
            
        if cap.isOpened():
            cap.release()

with tab2:
    st.header("Teacher Analytics Dashboard")
    st.markdown("Review historical data of student attention metrics.")
    
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("Refresh Data"):
            st.rerun()
        if st.button("Clear Logs"):
            clear_logs()
            st.success("Logs cleared!")
            st.rerun()
            
    with col_b:
        df = fetch_logs()
        if df.empty:
            st.info("No data available yet. Start tracking to collect data.")
        else:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # KPI Cards
            avg_score = df['score'].mean()
            focused_time = len(df[df['status'] == 'Focused'])
            distracted_time = len(df[df['status'] == 'Distracted'])
            drowsy_time = len(df[df['status'] == 'Drowsy'])
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Average Score", f"{avg_score:.1f}%")
            k2.metric("Focus Ticks", focused_time)
            k3.metric("Distracted Ticks", distracted_time)
            k4.metric("Drowsy Ticks", drowsy_time)
            
            # Chart
            st.subheader("Attention Score Over Time")
            fig = px.line(df, x='timestamp', y='score', color='status', title="Session Engagement Timeline")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Raw Data")
            st.dataframe(df, use_container_width=True)
