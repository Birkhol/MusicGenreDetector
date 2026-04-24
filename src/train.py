import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    SPLITS_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    NUM_CLASSES
)

from src.utils import set_seed, get_device, ensure_directories
from src.data_loader import AudioDataset, SpectrogramDataset
from src.model_1dcnn import GenreCNN1D
from src.model_2dcnn import GenreCNN2D


# Change between "1d" and "2d" to choose model
MODEL_TYPE = "1d"


def create_dataloaders(model_type):
    train_csv = os.path.join(SPLITS_DIR, "train.csv")
    val_csv = os.path.join(SPLITS_DIR, "val.csv")

    if model_type == "1d":
        train_dataset = AudioDataset(train_csv)
        val_dataset = AudioDataset(val_csv)

    elif model_type == "2d":
        train_dataset = SpectrogramDataset(train_csv)
        val_dataset = SpectrogramDataset(val_csv)

    else:
        raise ValueError("MODEL_TYPE must be either '1d' or '2d'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def create_model(model_type, device):
    if model_type == "1d":
        model = GenreCNN1D(num_classes=NUM_CLASSES)

    elif model_type == "2d":
        model = GenreCNN2D(num_classes=NUM_CLASSES)

    else:
        raise ValueError("MODEL_TYPE must be either '1d' or '2d'")

    return model.to(device)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader)
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


def train():
    set_seed(42)
    ensure_directories([MODELS_DIR])

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training model type: {MODEL_TYPE.upper()} CNN")

    train_loader, val_loader = create_dataloaders(MODEL_TYPE)

    model = create_model(MODEL_TYPE, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            save_name = f"best_{MODEL_TYPE}cnn.pth"
            save_path = os.path.join(MODELS_DIR, save_name)

            torch.save(model.state_dict(), save_path)

            print(f"Saved new best model to: {save_path}")

    print("\nTraining finished.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == "__main__":
    train()