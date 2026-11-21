import os
import time
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.utils import get_device, set_seed, save_json
from src.dataset import prepare_eurosat_data
from src.models import build_model, prepare_model_for_finetuning, count_parameters

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.3f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train_model(
    model_name: str,
    train_loader,
    val_loader,
    num_classes: int = 10,
    epochs: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    patience: int = 3,
    label_smoothing: float = 0.1,
    freeze_early: bool = True,
    device=None,
    models_dir: str = "models",
    outputs_dir: str = "outputs"
):
    set_seed(42)
    if device is None:
        device = get_device()

    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"\n{'='*60}")
    print(f"Training Model: {model_name} on {device.type.upper()} (AMP: {use_amp})")
    print(f"{'='*60}")

    model = build_model(model_name, num_classes=num_classes, pretrained=True)
    model = prepare_model_for_finetuning(model, model_name, freeze_early_layers=freeze_early).to(device)
    param_info = count_parameters(model)
    print(f"Parameters: {param_info['total']:,} total | {param_info['trainable']:,} trainable")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epochs": 0,
        "best_epoch": 0,
        "best_val_acc": 0.0,
        "best_val_loss": float('inf'),
        "training_time_sec": 0.0,
        "parameters": param_info['total']
    }

    best_val_acc = 0.0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_weights_path = os.path.join(models_dir, f"{model_name}_best.pt")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device, use_amp
        )
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - LR: {current_lr:.1e} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["lr"].append(current_lr)

        if val_acc > best_val_acc or (abs(val_acc - best_val_acc) < 0.005 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            history["best_val_acc"] = round(best_val_acc, 4)
            history["best_val_loss"] = round(best_val_loss, 4)
            epochs_no_improve = 0

            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'history': history,
                'num_classes': num_classes
            }, best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path} (Val Acc: {val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[Early Stopping] No improvement in validation metrics for {patience} epochs.")
                break

    total_time = time.time() - start_time
    history["training_time_sec"] = round(total_time, 2)
    history["epochs"] = len(history["train_loss"])

    curves_dir = os.path.join(outputs_dir, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    history_json_path = os.path.join(curves_dir, f"{model_name}_history.json")
    save_json(history, history_json_path)

    plot_training_curves(history, model_name, os.path.join(curves_dir, f"{model_name}_curves.png"))

    print(f"[Training Complete] {model_name} finished in {total_time:.1f}s. Best Val Acc: {best_val_acc*100:.2f}%")
    return history, best_weights_path

def plot_training_curves(history, model_name: str, save_path: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1b6d24", linewidth=2, marker='o')
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ba1a1a", linewidth=2, linestyle="--", marker='s')
    ax1.set_title(f"{model_name.upper()} - Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#1b6d24", linewidth=2, marker='o')
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#0051d5", linewidth=2, linestyle="--", marker='s')
    ax2.set_title(f"{model_name.upper()} - Classification Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle(f"EuroSAT Fine-Tuning Dynamics: {model_name.upper()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Curves] Saved {save_path}")
