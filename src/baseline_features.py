import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.config import RAW_DATA_DIR, RANDOM_SEED, GENRES


def run_baseline():
    csv_path = os.path.join(RAW_DATA_DIR, "features_30_sec.csv")

    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Drop filename because it is not useful for prediction
    X = df.drop(columns=["filename", "label"])
    y = df["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED
        ))
    ])

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    print("\nValidation Accuracy:", accuracy_score(y_val, val_preds))
    print("Test Accuracy:", accuracy_score(y_test, test_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=GENRES))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))


if __name__ == "__main__":
    run_baseline()