import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform(img_size=112):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=(0, 90)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def get_eval_transform(img_size=112):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class TransformedSubset(Dataset):
    """Wraps a dataset subset with a specific transform."""
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, target = self.base_dataset.samples[real_idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

def prepare_eurosat_data(
    data_dir="dataset/EuroSAT_RGB/EuroSAT_RGB",
    output_dir="outputs",
    img_size=112,
    seed=42,
    batch_size=64,
    num_workers=0,
    subsample_train_per_class: int = None,
    subsample_eval_per_class: int = None
):
    """
    Loads EuroSAT dataset, performs 70/15/15 stratified split,
    persists split indices, generates sample grid, and returns DataLoaders.
    Supports optional balanced subsampling for fast CPU training.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"EuroSAT dataset directory not found at: {data_dir}")

    full_dataset = datasets.ImageFolder(root=data_dir)
    classes = full_dataset.classes
    targets = full_dataset.targets
    n_samples = len(full_dataset)
    print(f"[Data] Found {n_samples} images across {len(classes)} classes: {classes}")

    # Save class names
    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(classes, f, indent=2)

    # Stratified Split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        list(range(n_samples)),
        test_size=0.30,
        stratify=targets,
        random_state=seed
    )
    # 15% val, 15% test from temp
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=temp_targets,
        random_state=seed
    )

    # Optional balanced subsample for CPU acceleration
    actual_train_idx = train_idx
    actual_val_idx = val_idx
    actual_test_idx = test_idx

    if subsample_train_per_class is not None:
        sampled_train = []
        class_counters = {i: 0 for i in range(len(classes))}
        for i in train_idx:
            t = targets[i]
            if class_counters[t] < subsample_train_per_class:
                sampled_train.append(i)
                class_counters[t] += 1
        actual_train_idx = sampled_train
        print(f"[Data] Subsampled training set: {len(actual_train_idx)} images ({subsample_train_per_class}/class)")

    if subsample_eval_per_class is not None:
        sampled_val = []
        sampled_test = []
        vc = {i: 0 for i in range(len(classes))}
        tc = {i: 0 for i in range(len(classes))}
        for i in val_idx:
            t = targets[i]
            if vc[t] < subsample_eval_per_class:
                sampled_val.append(i)
                vc[t] += 1
        for i in test_idx:
            t = targets[i]
            if tc[t] < subsample_eval_per_class:
                sampled_test.append(i)
                tc[t] += 1
        actual_val_idx = sampled_val
        actual_test_idx = sampled_test
        print(f"[Data] Subsampled val set: {len(actual_val_idx)} images, test set: {len(actual_test_idx)} images")

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "active_train_count": len(actual_train_idx),
        "active_val_count": len(actual_val_idx),
        "active_test_count": len(actual_test_idx),
        "classes": classes,
        "total_images": n_samples,
        "split_counts": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx)
        }
    }

    # Save split indices for reproducibility
    os.makedirs(output_dir, exist_ok=True)
    split_path = os.path.join(output_dir, "split_indices.json")
    with open(split_path, "w") as f:
        json.dump(split_indices, f, indent=2)
    print(f"[Data] Split saved to {split_path}: Train={len(actual_train_idx)}, Val={len(actual_val_idx)}, Test={len(actual_test_idx)}")

    # Create Transformed Subsets
    train_dataset = TransformedSubset(full_dataset, actual_train_idx, transform=get_train_transform(img_size))
    val_dataset = TransformedSubset(full_dataset, actual_val_idx, transform=get_eval_transform(img_size))
    test_dataset = TransformedSubset(full_dataset, actual_test_idx, transform=get_eval_transform(img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Save sample grid
    save_sample_grid(full_dataset, classes, os.path.join(output_dir, "samples", "class_samples_grid.png"))

    return {
        "classes": classes,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "full_dataset": full_dataset,
        "split_indices": split_indices,
        "active_test_indices": actual_test_idx
    }

def save_sample_grid(full_dataset, classes, save_path):
    """Saves a 2x5 grid of sample satellite images with labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    samples_per_class = {c: [] for c in classes}
    for path, target in full_dataset.samples:
        c = classes[target]
        if len(samples_per_class[c]) < 2:
            samples_per_class[c].append(path)
        if all(len(v) == 2 for v in samples_per_class.values()):
            break

    fig, axes = plt.subplots(2, len(classes), figsize=(20, 5))
    for col, c in enumerate(classes):
        for row in range(2):
            img_path = samples_per_class[c][row]
            img = Image.open(img_path).convert('RGB')
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
            if row == 0:
                ax.set_title(c, fontsize=10, fontweight='bold', pad=4)

    plt.suptitle("EuroSAT Land-Use Classification Sample Images", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Data] Sample grid saved to {save_path}")
