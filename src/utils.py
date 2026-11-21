import os
import json
import random
import numpy as np
import torch

URBAN_INDUSTRIAL_CLASSES = {"Industrial", "Residential", "Highway"}
NATURAL_CLASSES = {
    "Forest", "HerbaceousVegetation", "Pasture", "River",
    "SeaLake", "AnnualCrop", "PermanentCrop"
}

CLASS_CATEGORIES = {
    "AnnualCrop": "Agricultural",
    "Forest": "Natural / Forestry",
    "HerbaceousVegetation": "Natural / Open Land",
    "Highway": "Infrastructure",
    "Industrial": "Urban / Industrial",
    "Pasture": "Agricultural / Grazing",
    "PermanentCrop": "Agricultural / Orchards",
    "Residential": "Urban / Built-up",
    "River": "Aquatic / Waterway",
    "SeaLake": "Aquatic / Waterbody"
}

def get_device():
    """Device selection helper with hardware diagnostic logging."""
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
    """Seed all pseudo-random generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def is_high_impact_error(true_class: str, pred_class: str) -> bool:
    """
    Cost of error rule:
    Misclassifying urban/industrial regions as natural, or natural as urban/industrial,
    is flagged as a high-impact error for geospatial planning and environmental policy.
    """
    if true_class == pred_class:
        return False
    is_true_urban = true_class in URBAN_INDUSTRIAL_CLASSES
    is_pred_urban = pred_class in URBAN_INDUSTRIAL_CLASSES
    return is_true_urban != is_pred_urban

def get_category_for_class(class_name: str) -> str:
    return CLASS_CATEGORIES.get(class_name, "General Land-Cover")

def save_json(data, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
