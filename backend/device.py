import torch
import random
import numpy as np
import os

def get_device():
    """
    Device selection helper:
    Checks if CUDA is available and returns torch.device('cuda') or torch.device('cpu').
    Prints hardware details.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Device] Using GPU: {gpu_name} ({gpu_mem:.2f} GB VRAM, CUDA {torch.version.cuda})")
    else:
        device = torch.device('cpu')
        print("[Device] CUDA not available. Falling back to CPU.")
    return device

def set_seed(seed=42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
