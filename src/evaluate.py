import os
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import (
    SPLITS_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    NUM_CLASSES,
    GENRES
)

from src.utils import get_device
from src.data_loader import AudioDataset, SpectrogramDataset
from src.model_1dcnn import GenreCNN1D
from src.model_2dcnn import GenreCNN2D


MODEL_TYPE = "1d"  # Change between "1d" and "2d" to evaluate the models


def create_test_loader(model_type):
    test_csv = os.path.join(SPLITS_DIR, "test.csv")

    if model_type == "1d":
        test_dataset = AudioDataset(test_csv)
    elif model_type == "2d":
        test_dataset = SpectrogramDataset(test_csv)
    else:
        raise ValueError("MODEL_TYPE must be either '1d' or '2d'")

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


def create_model(model_type, device):
    if model_type == "1d":
        model = GenreCNN1D(num_classes=NUM_CLASSES)
    elif model_type == "2d":
        model = GenreCNN2D(num_classes=NUM_CLASSES)
    else:
        raise ValueError("MODEL_TYPE must be either '1d' or '2d'")

    return model.to(device)


def evaluate():
    device = get_device()
    print(f"Using device: {device}")
    print(f"Evaluating model type: {MODEL_TYPE.upper()} CNN")

    test_loader = create_test_loader(MODEL_TYPE)
    model = create_model(MODEL_TYPE, device)

    model_path = os.path.join(MODELS_DIR, f"best_{MODEL_TYPE}cnn.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find model file: {model_path}\n"
            f"Train the {MODEL_TYPE.upper()} CNN first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=GENRES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()