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

## Setup

1.Clone the repository:
```bash
git clone https://github.com/Birkhol/MusicGenreDetector.git
After cloning the project, run:
```bash
cd MusicGenreDetector

2.Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

3.Install the required packages from requirements.txt:
```bash
pip install -r requirements.txt

4. Configure the Model & Hyperparameters:
Open src/config.py to customize your training run. You can toggle between different architectures and fine-tune the training settings:

Model Selection: Set MODEL_TYPE to either "1d" or "2d" to choose between the 1D CNN and 2D CNN architectures.

5. Before training the model, make sure to set the path to your project root in fix_paths.py by changing this variable: new_base = r'the-path-to-your-root'  and then run:
```bash
python -m fix_paths

6.Train the model by:
```bash
 python -m src.train to train the model
Test the model:
```bash
python -m src.evaluate to evaluate the model

Results:
To see the results of trained and evaluated model, go to results/figures for the images
and results/models for the models themselves. Additionally, the results will be shown in the terminal where the train and evaluate commands were run.