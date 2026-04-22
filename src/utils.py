import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"