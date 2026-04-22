# Music Genre Detector

This project compares two deep learning approaches for music genre classification using the GTZAN dataset:

1. **1D CNN** on raw audio waveforms
2. **2D CNN** on spectrogram images

## Goal
To determine whether a 1D CNN trained on time-series audio performs better than a 2D CNN trained on image-based spectrograms for classifying songs into 10 genres.

## Dataset
GTZAN Dataset - Music Genre Classification  
Kaggle: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## Genres
- blues
- classical
- country
- disco
- hiphop
- jazz
- metal
- pop
- reggae
- rock

## Project Structure
- `src/` - source code
- `data/` - dataset and processed files
- `results/` - saved models, logs, and plots
- `notebooks/` - experiments and data exploration

## Dataset Setup

1. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

2. Extract the files

3. Create and place them in:

data/raw/

Final structure:

data/raw/
├── genres_original/
├── images_original/
├── features_30_sec.csv
└── features_3_sec.csv

## Setup

```bash
git clone https://github.com/Birkhol/MusicGenreDetector.git
cd MusicGenreDetector
python -m venv venv