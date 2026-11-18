"""Random seed setup."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """
    Seed the Python, NumPy and PyTorch random number generators.

    cuDNN benchmark mode is enabled and deterministic mode is disabled, so results can differ slightly between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass
