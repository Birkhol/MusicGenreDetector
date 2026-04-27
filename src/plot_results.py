import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


def save_learning_curves(history, save_path):

    # history should be a dictionary with:
    # train_loss, val_loss, train_acc, val_acc

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace(".png", "_accuracy.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace(".png", "_loss.png"), bbox_inches="tight")
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_classification_report(report_dict, save_path):

    # report_dict comes from:
    # classification_report(y_true, y_pred, target_names=GENRES, output_dict=True)


    df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, 5))
    plt.axis("off")

    table = plt.table(
        cellText=df.round(3).values,
        rowLabels=df.index,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.title("Classification Report")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()