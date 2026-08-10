import streamlit as st
import cv2
import time
import plotly.graph_objects as go
import pandas as pd
from collections import deque
from detector import EmotionDetector

EMOTION_COLORS = {
    "angry": "#E63946",
    "disgust": "#2A9D8F",
    "fear": "#6A4C93",
    "happy": "#F4A261",
    "sad": "#457B9D",
    "surprise": "#E9C46A",
    "neutral": "#8D99AE",
}

st.set_page_config(page_title="Emotion Detection Dashboard", layout="wide")
st.markdown('<p class="main-title">Emotion Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time facial expression analysis using computer vision</p>', unsafe_allow_html=True)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #2b2b33;
}
.stApp {
    background-color: #f5efe4;
}

.metric-card {
    border: 1px solid #ddd4c0 !important;
}


.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.01em;
    margin-bottom: 0.1rem;
}

.subtitle {
    font-size: 0.95rem;
    color: #6b7280;
    font-weight: 400;
    margin-bottom: 1.8rem;
}

.metric-card {
    background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
    border: 1px solid #ececec;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}

.metric-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    transform: translateY(-1px);
}

.metric-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.01em;
}

.metric-sub {
    font-size: 0.75rem;
    color: #a1a5ac;
    margin-top: 0.25rem;
}
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-top: 1.8rem;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #ececec;
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
status_ph = col1.empty()
total_ph = col2.empty()
acc_ph = col3.empty()
speed_ph = col4.empty()

def render_card(placeholder, label, value, sub):
    placeholder.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

render_card(status_ph, "Live Status", "Inactive", "Webcam not connected")
render_card(total_ph, "Total Detections", "0", "Since session start")
render_card(acc_ph, "Model Accuracy", "94.6%", "Validation benchmark")
render_card(speed_ph, "Avg Response Time", "0.00s", "Per frame")
if "detector" not in st.session_state:
    st.session_state.detector = EmotionDetector()
if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=30)
if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}
if "recent_log" not in st.session_state:
    st.session_state.recent_log = deque(maxlen=5)
if st.button("Reset Session Stats"):
    st.session_state.total_detections = 0
    st.session_state.emotion_counts = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0}
    st.session_state.history = deque(maxlen=30)
    st.session_state.recent_log = deque(maxlen=5)

run = st.checkbox("Start Detection")
video_col, panel_col = st.columns([2, 1])
frame_window = video_col.empty()
panel_window = panel_col.empty()

st.markdown('<p class="section-header">Emotion Distribution</p>', unsafe_allow_html=True)
chart_ph = st.empty()
st.markdown('<p class="section-header">Recent Detections</p>', unsafe_allow_html=True)
recent_ph = st.empty()
st.markdown('<p class="section-header">Real-time Emotion Trend</p>', unsafe_allow_html=True)
trend_ph = st.empty()
cap = cv2.VideoCapture(0)


while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Webcam not found.")
        break

    start = time.time()
    results = st.session_state.detector.analyze(frame)
    elapsed = time.time() - start

    label, conf = None, 0
    for face in results:
        x, y, w, h = face["box"]
        label, conf = st.session_state.detector.top_emotion_for(face["emotions"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf*100:.0f}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        st.session_state.total_detections += 1
        st.session_state.emotion_counts[label] += 1
        st.session_state.history.append((time.time(), label, conf))
        face_crop = frame[y:y+h, x:x+w].copy()
        st.session_state.recent_log.appendleft((face_crop, label, conf, time.strftime("%I:%M:%S %p")))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)
    render_card(status_ph, "Live Status", "Active", "Webcam connected")
    render_card(total_ph, "Total Detections", str(st.session_state.total_detections), "Since session start")
    render_card(speed_ph, "Avg Response Time", f"{elapsed:.2f}s", "Per frame")
    counts = st.session_state.emotion_counts
    if sum(counts.values()) > 0:
        fig = go.Figure(data=[go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.6,
            marker=dict(colors=[EMOTION_COLORS.get(e, "#999999") for e in counts.keys()])
        )])
        fig.update_layout(height=350, margin=dict(t=20, b=20))
        chart_ph.plotly_chart(fig, use_container_width=True, key=f"donut_{time.time()}")
        if st.session_state.recent_log:
            with recent_ph.container():
                cols = st.columns(len(st.session_state.recent_log))
                for i, (crop, lbl, cf, ts) in enumerate(st.session_state.recent_log):
                    with cols[i]:
                        if crop.size > 0:
                            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.caption(f"**{lbl.title()}** {cf*100:.0f}%  \n{ts}")
    if len(st.session_state.history) > 1:
        df = pd.DataFrame(st.session_state.history, columns=["t", "emotion", "conf"])
        df["t"] = pd.to_datetime(df["t"], unit="s")
        fig2 = go.Figure()
        for e in df["emotion"].unique():
            sub = df[df["emotion"] == e]
            fig2.add_trace(go.Scatter(
                x=sub["t"], y=sub["conf"], mode="lines+markers", name=e.title(),
                line=dict(color=EMOTION_COLORS.get(e, "#999999"))
            ))
        fig2.update_layout(height=300, xaxis_title="Time", yaxis_title="Confidence", margin=dict(t=20, b=20))
        trend_ph.plotly_chart(fig2, use_container_width=True, key=f"trend_{time.time()}")
    if label:
        with panel_window.container():
            st.subheader(f"{label.title()} — {conf*100:.0f}%")
            for e, c in results[0]["emotions"].items():
                st.progress(min(c, 1.0), text=f"{e.title()} {c*100:.0f}%")
cap.release()  
