import streamlit as st
import numpy as np
import librosa
import tensorflow as tf

# ----------------------------
# STREAMLIT PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="CoughDetect",
    layout="wide",
    page_icon="🩺"
)

# ----------------------------
# GLOBAL STYLING (BACKGROUND + SIDE IMAGES)
# ----------------------------
st.markdown("""
    <style>
        /* Main content width */
        .main .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Soft gradient background */
        .stApp {
            background-color: #f4f6fa;
            background-image: radial-gradient(circle at 20% 20%, #ffffff 0%, #f4f6fa 70%);
        }

        /* Decorative left image */
        .left-img {
            position: fixed;
            top: 20%;
            left: 0;
            width: 250px;
            opacity: 0.18;
            z-index: -1;
        }

        /* Decorative right image */
        .right-img {
            position: fixed;
            top: 20%;
            right: 0;
            width: 250px;
            opacity: 0.18;
            z-index: -1;
        }

        /* Header images row */
        .header-img-row {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-bottom: 20px;
        }
        .header-img-row img {
            width: 160px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
        }

    </style>

    <!-- LEFT + RIGHT side images -->
    <img src="/mnt/data/a6380c32-c2fe-4ce8-9761-6c4d7d0dcc5f.png" class="left-img">
    <img src="/mnt/data/df7e5018-e460-48b8-a5dd-351fa64f29bb.png" class="right-img">
""", unsafe_allow_html=True)

# ----------------------------
# HEADER IMAGE ROW (top banner)
# Replace these URLs with any images you want
# ----------------------------
st.markdown("""
<div class="header-img-row">
    <img src="https://i.imgur.com/3ZQ3Zzb.png">
    <img src="https://i.imgur.com/PLQbF8H.png">
    <img src="https://i.imgur.com/6pQ0pUy.png">
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR EXPLANATION
# ----------------------------
st.sidebar.title("ℹ️ About CoughDetect")
st.sidebar.write("""
This tool analyzes **cough audio** using a deep learning model.

### How It Works:
1. Upload a `.wav` file  
2. Audio is preprocessed  
3. Model predicts: **Healthy** or **Abnormal**  

### Notes:
- Not medical advice  
- Best with clean, 1–2 second cough recordings  
""")

# ----------------------------
# LOAD MODEL (Your original logic)
# ----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# ----------------------------
# ORIGINAL PREPROCESSING (unchanged)
# ----------------------------
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    if len(y) > 1024:
        y = y[:1024]
    else:
        y = np.pad(y, (0, max(0, 1024 - len(y))))
    X = np.expand_dims(y, axis=0).astype(np.float32)
    return X

# ----------------------------
# MAIN CARD CONTAINER
# ----------------------------
st.markdown("""
<div style="
    background: white;
    padding: 35px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    width: 85%;
    margin: 20px auto;
">
""", unsafe_allow_html=True)

st.title("🩺 Welcome to CoughDetect!")
st.write("Upload a `.wav` file to check whether your cough is **Healthy** or **Abnormal**.")

# ----------------------------
# FILE UPLOAD
# ----------------------------
uploaded_file = st.file_uploader("Upload WAV File", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file)

    try:
        X = preprocess_audio("temp.wav")
        preds = model.predict(X)

        abnormal_prob = float(preds[0][0])
        healthy_prob = float(preds[0][1])

        threshold = 0.5

        if healthy_prob >= threshold:
            label = "Abnormal"
            confidence = healthy_prob
        else:
            label = "Healthy"
            confidence = abnormal_prob

        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}**")

    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("</div>", unsafe_allow_html=True)
