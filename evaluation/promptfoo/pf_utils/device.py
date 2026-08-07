import torch

from config import MODELS


def get_device() -> torch.device:
    if MODELS.device != "auto":
        return torch.device(MODELS.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
