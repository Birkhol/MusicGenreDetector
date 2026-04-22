from src.utils import set_seed, get_device, ensure_directories
from src.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    LOGS_DIR,
    FIGURES_DIR
)


def main():
    set_seed(42)

    ensure_directories([
        DATA_DIR,
        PROCESSED_DATA_DIR,
        SPLITS_DIR,
        RESULTS_DIR,
        MODELS_DIR,
        LOGS_DIR,
        FIGURES_DIR
    ])

    device = get_device()

    print("Project initialized successfully.")
    print(f"Using device: {device}")


if __name__ == "__main__":
    main()