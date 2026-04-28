import numpy as np
import pandas as pd
import librosa
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from src.config import SAMPLE_RATE, SAMPLES_PER_TRACK

def spec_augment(image, time_mask_param=5, freq_mask_param=3, num_time_masks=1, num_freq_masks=1):

    # Apply SpecAugment to spectrogram image.
    # image: torch.Tensor of shape (C, H, W) where H=freq, W=time
    augmented = image.clone()
    
    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, time_mask_param)
        if t > 0 and image.shape[2] > t:
            t0 = random.randint(0, image.shape[2] - t)
            augmented[:, :, t0:t0+t] = 0
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        if f > 0 and image.shape[1] > f:
            f0 = random.randint(0, image.shape[1] - f)
            augmented[:, f0:f0+f, :] = 0
    
    return augmented

def add_noise(image, noise_factor=0.001):

    # Add Gaussian noise to the spectrogram.

    noise = torch.randn_like(image) * noise_factor
    return image + noise

def random_brightness(image, brightness_factor=0.05):

    # Randomly adjust brightness of spectrogram.

    factor = 1 + random.uniform(-brightness_factor, brightness_factor)
    return image * factor

def random_contrast(image, contrast_factor=0.05):

    # Randomly adjust contrast of spectrogram.

    mean = image.mean()
    factor = 1 + random.uniform(-contrast_factor, contrast_factor)
    return (image - mean) * factor + mean

def normalize_image(image):

    # Normalize spectrogram to have mean=0 and std=1.

    mean = image.mean()
    std = image.std()
    if std > 0:
        return (image - mean) / std
    return image

def random_crop(image, crop_size=(112, 112)):

    # Random crop the spectrogram image.

    _, h, w = image.shape
    crop_h, crop_w = crop_size
    if h > crop_h and w > crop_w:
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        return image[:, top:top+crop_h, left:left+crop_w]
    else:
        return image
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
    def __init__(self, csv_file, image_size=(128, 128), augment=False):
        self.df = pd.read_csv(csv_file)
        self.image_size = image_size
        self.augment = augment

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

        if self.augment:
            # Apply augmentations
            image = spec_augment(image)
            if random.random() < 0.1:
                image = add_noise(image)
            if random.random() < 0.05:
                image = random_brightness(image)
            if random.random() < 0.05:
                image = random_contrast(image)
        
        # Normalize spectrogram
        image = normalize_image(image)
        # Clip to prevent extreme values after normalization
        image = torch.clamp(image, -2.0, 2.0)

        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label