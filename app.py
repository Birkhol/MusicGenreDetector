import os
import tempfile

import streamlit as st
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.model_2dcnn import GenreCNN2D
from src.config import NUM_CLASSES, GENRES, SAMPLE_RATE, MODELS_DIR


MODEL_PATH = os.path.join(MODELS_DIR, "best_2dcnn.pth")
IMAGE_SIZE = (128, 128)


@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GenreCNN2D(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# Gets 3 seconds of the audio file in the middle of the song
def load_middle_3_seconds(file_path, duration=3):
    total_duration = librosa.get_duration(path=file_path)
    start_time = max(0, (total_duration / 2) - (duration / 2))

    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start_time,
        duration=duration
    )

    return y, sr

def create_spectrogram_image(file_path):

    # Creates a spectrogram image similar to the GTZAN spectrogram PNGs.

    y, sr = load_middle_3_seconds(file_path, duration=3)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Create image without axes, similar to dataset spectrograms
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma")

    fig.canvas.draw()

    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    image = Image.fromarray(image).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    return image

def preprocess_image(image):
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.tensor(image, dtype=torch.float32)

    # Same normalization style as the data_loader.py
    mean = image.mean()
    std = image.std()

    if std > 0:
        image = (image - mean) / std

    image = torch.clamp(image, -2.0, 2.0)

    return image.unsqueeze(0)


def predict_genre(file_path):
    model, device = load_model()

    image = create_spectrogram_image(file_path)
    input_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_idx = torch.argmax(probabilities).item()
    predicted_genre = GENRES[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    top3 = torch.topk(probabilities, 3)

    top3_results = [
        (GENRES[top3.indices[i].item()], top3.values[i].item())
        for i in range(3)
    ]

    return predicted_genre, confidence, top3_results, image


st.title("Music Genre Detector")
st.write("Upload a song and the model will predict its genre.")

uploaded_file = st.file_uploader(
    "Upload a song file, both .wav and .mp3 files are accepted",
    type=["wav", "mp3"]
)

st.write("The first prediction may take up to a minute...")

if uploaded_file is not None:
    st.audio(uploaded_file)

    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        predicted_genre, confidence, top3_results, spectrogram = predict_genre(tmp_path)

        st.subheader("Prediction")
        st.write(f"**Genre:** {predicted_genre}")
        st.write(f"**Confidence:** {confidence:.2%}")

        st.subheader("Top 3 Predictions")
        for genre, prob in top3_results:
            st.write(f"{genre}: {prob:.2%}")

        st.subheader("Generated Spectrogram")
        st.image(spectrogram, caption="Spectrogram used by the model")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    finally:
        os.remove(tmp_path)