import streamlit as st
import tensorflow as tf
import numpy as np
import librosa

st.title("🩺 Lung Sound Classifier")
st.write("Upload a `.wav` file to classify it as **Healthy** or **Abnormal**.")

# -----------------------------
# Load model (cached for speed)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_sound_classifier.keras")

model = load_model()

# -----------------------------
# Preprocess audio
# -----------------------------
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    mel = librosa.feature.melspectrogram(y=y, sr=target_sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = tf.image.resize(mel_db[..., np.newaxis], [64, 94])  # shape (64, 94, 1)
    X = np.expand_dims(mel_db, axis=0).astype(np.float32)
    return X

# -----------------------------
# File upload section
# -----------------------------
uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        X = preprocess_audio("temp.wav")
        preds = model.predict(X)

        # -----------------------------
        # ✅ Step 1: Threshold-based decision
        # -----------------------------
        abnormal_prob = float(preds[0][0])  # assuming [Abnormal, Healthy]
        healthy_prob = float(preds[0][1])

        threshold = 0.6  # tune this value if needed

        if healthy_prob >= threshold:
            label = "Healthy"
            confidence = healthy_prob
        else:
            label = "Abnormal"
            confidence = abnormal_prob

        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2f}")
        st.write(f"(Abnormal={abnormal_prob:.2f}, Healthy={healthy_prob:.2f})")

        # Optional: play the uploaded audio
        st.audio(uploaded_file, format="audio/wav")

    except Exception as e:
        st.error(f"Error processing file: {e}")
