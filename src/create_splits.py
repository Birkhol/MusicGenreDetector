import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import AUDIO_DIR, IMAGE_DIR, GENRES, SPLITS_DIR, RANDOM_SEED
from src.utils import ensure_directories


def collect_audio_files():
    rows = []

    for label, genre in enumerate(GENRES):
        genre_dir = os.path.join(AUDIO_DIR, genre)

        if not os.path.exists(genre_dir):
            print(f"Warning: Missing folder {genre_dir}")
            continue

        for filename in os.listdir(genre_dir):
            if filename.endswith(".wav"):
                stem = os.path.splitext(filename)[0]  # blues.00000
                rows.append({
                    "id": stem,
                    "genre": genre,
                    "label": label,
                    "audio_path": os.path.join(genre_dir, filename)
                })

    return pd.DataFrame(rows)


def attach_image_paths(df):
    image_paths = []

    for _, row in df.iterrows():
        genre = row["genre"]
        song_id = row["id"].replace(".", "")  # blues.00000 -> blues00000
        image_filename = f"{song_id}.png"
        image_path = os.path.join(IMAGE_DIR, genre, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Missing image {image_path}")

        image_paths.append(image_path)

    df["image_path"] = image_paths
    return df


def create_splits():
    ensure_directories([SPLITS_DIR])

    df = collect_audio_files()
    df = attach_image_paths(df)

    print(f"Total samples found: {len(df)}")
    print(df["genre"].value_counts())

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["genre"],
        random_state=RANDOM_SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["genre"],
        random_state=RANDOM_SEED
    )

    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

    print("\nSaved split files:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")


if __name__ == "__main__":
    create_splits()