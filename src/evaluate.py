import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from src.utils import get_device, save_json
from src.models import build_model, count_parameters

@torch.no_grad()
def evaluate_model_on_test(model, test_loader, classes, device=None, model_name="model", outputs_dir="outputs"):
    if device is None:
        device = get_device()
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()
    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        all_probs.extend(probs)

    total_time = time.time() - start_time
    n_samples = len(all_targets)
    latency_ms = (total_time / n_samples) * 1000

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(all_targets, all_preds, average=None, zero_division=0)

    # Class report dictionary
    class_report = {}
    for i, c in enumerate(classes):
        class_report[c] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    # Confusion matrix
    cm_raw = confusion_matrix(all_targets, all_preds)
    cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]

    # Save Confusion Matrix Heatmap
    plot_confusion_matrix(cm_raw, cm_norm, classes, model_name, os.path.join(outputs_dir, f"{model_name}_confusion_matrix.png"))

    param_info = count_parameters(model)

    results = {
        "model_name": model_name,
        "test_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "latency_ms": round(float(latency_ms), 2),
        "throughput_fps": round(float(n_samples / total_time), 1),
        "parameters": param_info["total"],
        "class_report": class_report,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_norm": [[round(float(v), 4) for v in row] for row in cm_norm],
        "test_samples": n_samples
    }

    # Save per-model evaluation json
    eval_json_path = os.path.join(outputs_dir, f"{model_name}_eval.json")
    save_json(results, eval_json_path)

    print(f"\n[{model_name.upper()} Evaluation Results]")
    print(f"  Test Accuracy:     {acc * 100:.2f}%")
    print(f"  Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"  Macro Precision:   {macro_p * 100:.2f}%")
    print(f"  Macro Recall:      {macro_r * 100:.2f}%")
    print(f"  Inference Latency: {latency_ms:.2f} ms/sample")
    print(f"  Parameters:        {param_info['total']:,}")

    return results, (all_preds, all_targets, all_probs)

def plot_confusion_matrix(cm_raw, cm_norm, classes, model_name: str, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Normalized CM
    sns.heatmap(
        cm_norm, annot=True, fmt='.2f', cmap='Greens',
        xticklabels=classes, yticklabels=classes, ax=ax1, cbar=False
    )
    ax1.set_title(f"{model_name.upper()} - Normalized Confusion Matrix", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Predicted Class", fontweight='bold')
    ax1.set_ylabel("True Class", fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)

    # Raw CM
    sns.heatmap(
        cm_raw, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes, ax=ax2
    )
    ax2.set_title(f"{model_name.upper()} - Raw Misclassification Counts", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Predicted Class", fontweight='bold')
    ax2.set_ylabel("True Class", fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle(f"EuroSAT Test Set Evaluation: {model_name.upper()}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Eval] Saved confusion matrix to {save_path}")

def generate_comparison_table(all_results: list, outputs_dir: str = "outputs"):
    """
    Generates outputs/model_comparison.csv and outputs/model_comparison.json
    and comparison bar charts.
    """
    records = []
    for r in all_results:
        records.append({
            "Model": r["model_name"],
            "Accuracy": r["test_accuracy"],
            "Macro Precision": r["macro_precision"],
            "Macro Recall": r["macro_recall"],
            "Macro F1": r["macro_f1"],
            "Latency (ms)": r["latency_ms"],
            "Throughput (FPS)": r["throughput_fps"],
            "Parameters (M)": round(r["parameters"] / 1e6, 2)
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(outputs_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Comparison] Saved comparison table to {csv_path}")

    json_path = os.path.join(outputs_dir, "model_comparison.json")
    save_json({"models": all_results, "summary_table": records}, json_path)

    # Plot comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    model_names = [r["Model"] for r in records]

    # Accuracy & F1
    x = np.arange(len(model_names))
    width = 0.35
    axes[0].bar(x - width/2, [r["Accuracy"]*100 for r in records], width, label="Accuracy", color="#1b6d24")
    axes[0].bar(x + width/2, [r["Macro F1"]*100 for r in records], width, label="Macro F1", color="#3b6756")
    axes[0].set_title("Test Set Classification Accuracy & F1", fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_ylim(80, 100)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend()

    # Inference Latency
    axes[1].bar(model_names, [r["Latency (ms)"] for r in records], color="#0ea5e9")
    axes[1].set_title("Inference Latency (ms / sample)", fontweight='bold')
    axes[1].set_ylabel("Milliseconds")
    axes[1].grid(axis='y', alpha=0.3)

    # Parameters
    axes[2].bar(model_names, [r["Parameters (M)"] for r in records], color="#7138f5")
    axes[2].set_title("Model Parameters (Millions)", fontweight='bold')
    axes[2].set_ylabel("Parameters (M)")
    axes[2].grid(axis='y', alpha=0.3)

    plt.suptitle("EuroSAT Pretrained Architecture Benchmark Comparison", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    chart_path = os.path.join(outputs_dir, "model_comparison_chart.png")
    plt.savefig(chart_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Comparison] Saved comparison chart to {chart_path}")
    return df
