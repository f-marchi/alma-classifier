"""
Utility functions for ALMA Subtype v2.
"""
import os
import random


def set_deterministic(seed: int = 42) -> None:
    """Make torch / numpy / Python RNG deterministic (slow on CuDNN)."""
    try:
        import torch
        import numpy as np
        
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not available, just set basic seeds
        import numpy as np
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
