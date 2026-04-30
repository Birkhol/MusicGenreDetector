import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from src.config import SAMPLE_RATE

# Paths
RAW_AUDIO_DIR = "data/raw/genres_original"
OUTPUT_DIR = "data/processed/generated_spectograms"

IMAGE_SIZE = (128, 128)


def create_spectrogram(y, sr):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Create image without axes
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


def process_dataset():
    genres = os.listdir(RAW_AUDIO_DIR)

    for genre in genres:
        genre_path = os.path.join(RAW_AUDIO_DIR, genre)

        if not os.path.isdir(genre_path):
            continue

        output_genre_path = os.path.join(OUTPUT_DIR, genre)
        os.makedirs(output_genre_path, exist_ok=True)

        print(f"Processing genre: {genre}")

        for file in tqdm(os.listdir(genre_path)):
            if not file.endswith(".wav"):
                continue

            input_path = os.path.join(genre_path, file)

            try:
                y, sr = librosa.load(
                    input_path,
                    sr=SAMPLE_RATE,
                    mono=True,
                    duration=3
                )

                image = create_spectrogram(y, sr)

                name = file.replace(".wav", "")
                name = name.replace(".", "")
                output_filename = f"{name}.png"
                output_path = os.path.join(output_genre_path, output_filename)

                image.save(output_path)

            except Exception as e:
                print(f"Skipped {file}: {e}")


if __name__ == "__main__":
    process_dataset()