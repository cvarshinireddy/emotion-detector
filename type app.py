import streamlit as st

st.set_page_config(page_title="Emotion Detection Dashboard", layout="wide")

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 600;
    color: #1a1a2e;
    margin-bottom: 0;
}
.subtitle {
    font-size: 1rem;
    color: #6b7280;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">Emotion Detection Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time facial expression analysis using computer vision</p>', unsafe_allow_html=True)