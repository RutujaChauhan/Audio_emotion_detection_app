import streamlit as st
import pickle
import librosa
import numpy as np
import pandas as pd
import pyaudio
import soundfile as sf
from PIL import Image

# Load the trained model

audio_model = pd.read_pickle("audio_detectionModel.pkl")
image = Image.open(
    "C:\\Users\\rutuj\\OneDrive\\Desktop\\Desktop\\audio_processing\\app1\\image.png",
)


# Define the emotion labels based on your encoding
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Pleasant surprised",
    "Sad",
]


# Function to extract audio features
def extract_features(audio_file):
    try:
        audio_data, _ = librosa.load(audio_file, duration=3, offset=0.5)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=_, n_mfcc=40)
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


# Function to predict emotion
def predict_emotion(audio_features):
    try:
        audio_features = np.mean(audio_features, axis=1)
        audio_features = audio_features.reshape(1, -1)
        prediction = audio_model.predict(audio_features)
        # return prediction[0]
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        return predicted_emotion
    except Exception as e:
        st.error(f"Error predicting emotion: {str(e)}")
        return None


# Streamlit UI

st.title(":violet[Audio Emotion Detection App] ")
st.subheader(":orange[✏️Rutuja]")
st.image(image, width=300)
st.write(":blue[Record or Upload an audio file to detect the emotion.]")


# Option to record audio
recorded_audio = None
recording = False
if st.button(":blue[Record Audio]"):
    recording = not recording

if recording:
    st.write(":blue[Recording... (Click again to stop)]")

if not recording and recorded_audio is not None:
    with open("temp_audio.wav", "wb") as audio_file:
        audio_file.write(recorded_audio)
        st.audio(audio_file, format="mp3/wav", start_time=0)

    st.write(":blue[Processing...]")
    audio_features = extract_features("temp_audio.wav")

    if audio_features is not None:
        emotion = predict_emotion(audio_features)
        if emotion is not None:
            st.success(f":blue[Emotion Detected: {emotion}]")

uploaded_file = st.file_uploader(":blue[Choose an audio file...]", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format="mp3/wav", start_time=0)

    if st.button("Detect Emotion"):
        st.balloons()
        st.write(":blue[Processing...]")
        audio_features = extract_features(uploaded_file)
        if audio_features is not None:
            emotion = predict_emotion(audio_features)
            if emotion is not None:
                st.success(f":blue[Emotion Detected: {emotion}]", icon="✅")
