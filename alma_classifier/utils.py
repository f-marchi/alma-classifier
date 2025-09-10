import os, random, numpy as np, torch

def set_deterministic(seed: int = 42) -> None:
    """Make torch / numpy / Python RNG deterministic (slow on CuDNN)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

