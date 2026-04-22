import numpy as np
import pandas as pd
import librosa
from PIL import Image
import torch
from torch.utils.data import Dataset
from src.config import SAMPLE_RATE, SAMPLES_PER_TRACK

# 1D CNN
class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Gets the row at position idx
        row = self.df.iloc[idx]
        
        # Only need signal, ignore samplerate value
        signal, _ = librosa.load(row["audio_path"], sr=SAMPLE_RATE, mono=True)

        # Checks if the audio is too short, calculates padding if needed
        if len(signal) < SAMPLES_PER_TRACK:
            pad_length = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad_length))
        else:
            signal = signal[:SAMPLES_PER_TRACK]

        # Adds an extra dimension to match what PyTorch expects
        signal = np.expand_dims(signal, axis=0)
        signal = torch.tensor(signal, dtype=torch.float32)

        # Gets the numeric class label from CSV and converts it into a tensor
        label = torch.tensor(int(row["label"]), dtype=torch.long) 
        return signal, label # Returns the input tensor and the correct label

# 2D CNN
class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, image_size=(128, 128)):
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        image = image.resize(self.image_size)

        # Converts image to NumPy array, dividing by 255 makes the range 0.0 to 1.0
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image, dtype=torch.float32) # Convert image to a tensor

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label