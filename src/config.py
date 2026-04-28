import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


# Change this one to "1d" or "2d" for choosing which model (1D CNN) or (2D CNN) to be trained/evaluated
MODEL_TYPE = "1d"


GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

AUDIO_DIR = os.path.join(RAW_DATA_DIR, "genres_original")
IMAGE_DIR = os.path.join(RAW_DATA_DIR, "images_original")

RANDOM_SEED = 42
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Adjust hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
DROPOUT = 0.25
WEIGHT_DECAY = 0
EPOCHS = 90

NUM_CLASSES = 10