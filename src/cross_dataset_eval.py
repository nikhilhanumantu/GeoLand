import os
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from src.utils import get_device, set_seed, save_json
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models import build_model
from src.explainability import compute_gradcam_overlay

# Mapping from UC Merced classes to EuroSAT classes
UCMERCED_TO_EUROSAT_MAPPING = {
    "agricultural": "AnnualCrop",
    "forest": "Forest",
    "chaparral": "HerbaceousVegetation",
    "freeway": "Highway",
    "overpass": "Highway",
    "storagetanks": "Industrial",
    "golfcourse": "Pasture",
    "mediumresidential": "Residential",
    "denseresidential": "Residential",
    "sparseresidential": "Residential",
    "river": "River",
    "beach": "SeaLake"
}

# Out-Of-Distribution (OOD) novel classes with no direct EuroSAT equivalent
OOD_CLASSES = [
    "airplane", "baseballdiamond", "buildings", "harbor",
    "intersection", "mobilehomepark", "parkinglot", "runway", "tenniscourt"
]

def load_all_models(models_dir="models", device="cpu"):
    with open(os.path.join(models_dir, "class_names.json"), "r") as f:
        classes = json.load(f)

    models = {}
    for m_name in ["resnet50", "efficientnet_b0", "convnext_tiny"]:
        ckpt_path = os.path.join(models_dir, f"{m_name}_best.pt")
        if not os.path.exists(ckpt_path):
            continue
        model = build_model(m_name, num_classes=len(classes), pretrained=False)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        models[m_name] = model

    return models, classes

def evaluate_cross_dataset(
    ucmerced_dir="dataset/UCMerced_LandUse/Images",
    models_dir="models",
    outputs_dir="outputs/cross_dataset",
    img_size=112,
    uncertainty_threshold=0.60
):
    set_seed(42)
    device = get_device()
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "="*70)
    print("GEOLAND AI — CROSS-DATASET TESTING (EuroSAT Model on UC Merced)")
    print("="*70)

    models, classes = load_all_models(models_dir=models_dir, device=device)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # 1. Gather all UC Merced images
    mapped_samples = []
    ood_samples = []

    for ucm_class in sorted(os.listdir(ucmerced_dir)):
        c_path = os.path.join(ucmerced_dir, ucm_class)
        if not os.path.isdir(c_path):
            continue

        images = [os.path.join(c_path, f) for f in os.listdir(c_path) if f.lower().endswith(('.tif', '.tiff', '.jpg', '.png'))]

        if ucm_class in UCMERCED_TO_EUROSAT_MAPPING:
            target_eurosat_class = UCMERCED_TO_EUROSAT_MAPPING[ucm_class]
            for img_p in images:
                mapped_samples.append({
                    "image_path": img_p,
                    "ucm_class": ucm_class,
                    "target_class": target_eurosat_class,
                    "target_idx": class_to_idx[target_eurosat_class]
                })
        else:
            for img_p in images:
                ood_samples.append({
                    "image_path": img_p,
                    "ucm_class": ucm_class
                })

    print(f"[Dataset] Mapped in-domain test images: {len(mapped_samples)} across {len(set(s['ucm_class'] for s in mapped_samples))} UC Merced categories")
    print(f"[Dataset] Out-of-Distribution (OOD) test images: {len(ood_samples)} across {len(set(s['ucm_class'] for s in ood_samples))} novel categories")

    # 2. Evaluate Mapped Classes across Models
    comparison_records = []
    model_predictions = {}

    for model_name, model in models.items():
        print(f"\n--> Evaluating {model_name.upper()} on UC Merced Mapped Test Set...")
        correct = 0
        total = len(mapped_samples)
        all_preds = []
        all_targets = []
        all_confs = []
        uncertain_count = 0

        t0 = time.time()
        for s in mapped_samples:
            img = Image.open(s["image_path"]).convert("RGB")
            tensor = eval_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            all_preds.append(pred_idx)
            all_targets.append(s["target_idx"])
            all_confs.append(conf)

            if pred_idx == s["target_idx"]:
                correct += 1
            if conf < uncertainty_threshold:
                uncertain_count += 1

        elapsed = time.time() - t0
        latency_ms = (elapsed / total) * 1000.0

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        acc = correct / total
        avg_conf = float(np.mean(all_confs))

        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
        weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)

        print(f"  Accuracy on UC Merced: {acc*100:.2f}% ({correct}/{total})")
        print(f"  Macro Precision:       {prec*100:.2f}%")
        print(f"  Macro Recall:          {rec*100:.2f}%")
        print(f"  Macro F1-Score:        {f1*100:.2f}%")
        print(f"  Average Confidence:    {avg_conf*100:.2f}%")
        print(f"  Uncertain Flagged:     {uncertain_count}/{total} ({uncertain_count/total*100:.1f}%)")

        model_predictions[model_name] = {
            "preds": all_preds,
            "targets": all_targets,
            "confs": all_confs,
            "acc": acc,
            "macro_f1": f1
        }

        comparison_records.append({
            "Model": model_name,
            "EuroSAT Test Acc": 0.9333 if model_name == "convnext_tiny" else (0.8547 if model_name == "resnet50" else 0.8520),
            "UC Merced Transfer Acc": round(acc, 4),
            "Domain Gap Drop": round((0.9333 if model_name == "convnext_tiny" else (0.8547 if model_name == "resnet50" else 0.8520)) - acc, 4),
            "Macro Precision": round(prec, 4),
            "Macro Recall": round(rec, 4),
            "Macro F1": round(f1, 4),
            "Avg Confidence": round(avg_conf, 4),
            "Uncertain Pct": round(uncertain_count / total * 100, 2),
            "Latency (ms)": round(latency_ms, 2)
        })

    # Save Comparison CSV
    df_cmp = pd.DataFrame(comparison_records)
    csv_path = os.path.join(outputs_dir, "cross_dataset_comparison.csv")
    df_cmp.to_csv(csv_path, index=False)
    print(f"\n[Artifact] Saved comparison table to {csv_path}")

    # 3. Best Model Confusion Matrix on UC Merced
    best_model_name = "convnext_tiny" if "convnext_tiny" in model_predictions else list(model_predictions.keys())[0]
    best_preds = model_predictions[best_model_name]["preds"]
    best_targets = model_predictions[best_model_name]["targets"]

    cm = confusion_matrix(best_targets, best_preds, labels=range(len(classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(f"UC Merced Cross-Dataset Transfer Confusion Matrix ({best_model_name.upper()})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Predicted EuroSAT Class", fontweight='bold')
    plt.ylabel("Ground Truth Mapped Class", fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    cm_path = os.path.join(outputs_dir, "ucmerced_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Artifact] Saved confusion matrix to {cm_path}")

    # 4. Per-Class Transfer Accuracy Chart
    mapped_eurosat_classes = sorted(list(set(s['target_class'] for s in mapped_samples)))
    per_class_accs = {}

    for c_name in mapped_eurosat_classes:
        c_idx = class_to_idx[c_name]
        mask = (best_targets == c_idx)
        if mask.sum() > 0:
            c_acc = (best_preds[mask] == c_idx).sum() / mask.sum()
            per_class_accs[c_name] = round(float(c_acc), 4)

    plt.figure(figsize=(11, 5))
    bars = plt.bar(per_class_accs.keys(), [v * 100 for v in per_class_accs.values()], color='#006494', edgecolor='black', alpha=0.85)
    plt.axhline(df_cmp.loc[df_cmp["Model"] == best_model_name, "UC Merced Transfer Acc"].values[0] * 100, color='#ba1a1a', linestyle='--', label='Average Transfer Acc')
    plt.title(f"Zero-Shot Cross-Dataset Accuracy by Class on UC Merced ({best_model_name.upper()})", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(outputs_dir, "per_class_transfer_acc.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[Artifact] Saved per-class transfer chart to {bar_path}")

    # 5. Out-of-Distribution (OOD) Analysis on Novel Classes
    print(f"\n--> Running Out-of-Distribution (OOD) Analysis on Novel Classes...")
    best_model = models[best_model_name]
    ood_results_by_class = {}

    for ood_sample in ood_samples:
        u_class = ood_sample["ucm_class"]
        if u_class not in ood_results_by_class:
            ood_results_by_class[u_class] = {"total": 0, "uncertain": 0, "preds": {}, "confs": []}

        img = Image.open(ood_sample["image_path"]).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = best_model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        p_idx = int(np.argmax(probs))
        p_class = classes[p_idx]
        conf = float(probs[p_idx])

        ood_results_by_class[u_class]["total"] += 1
        ood_results_by_class[u_class]["confs"].append(conf)
        if conf < uncertainty_threshold:
            ood_results_by_class[u_class]["uncertain"] += 1

        ood_results_by_class[u_class]["preds"][p_class] = ood_results_by_class[u_class]["preds"].get(p_class, 0) + 1

    ood_summary = []
    for u_class, data in ood_results_by_class.items():
        top_pred = max(data["preds"].items(), key=lambda x: x[1])
        avg_c = float(np.mean(data["confs"]))
        unc_pct = float(data["uncertain"] / data["total"] * 100)
        ood_summary.append({
            "Novel Class": u_class,
            "Total Samples": data["total"],
            "Average Confidence": round(avg_c, 4),
            "Uncertain Detection Rate (%)": round(unc_pct, 1),
            "Dominant Semantic Analogy": f"{top_pred[0]} ({top_pred[1]/data['total']*100:.1f}%)"
        })

    df_ood = pd.DataFrame(ood_summary)
    ood_csv_path = os.path.join(outputs_dir, "ood_novel_classes_analysis.csv")
    df_ood.to_csv(ood_csv_path, index=False)
    print(f"[Artifact] Saved OOD analysis table to {ood_csv_path}")

    # 6. Grad-CAM Visualization Gallery on UC Merced
    print("\n--> Generating Grad-CAM Attribution Gallery on UC Merced Samples...")
    gallery_classes = ["freeway", "river", "forest", "agricultural", "denseresidential", "beach", "airplane", "harbor"]
    fig, axes = plt.subplots(len(gallery_classes), 3, figsize=(10, 2.5 * len(gallery_classes)))

    for i, c_name in enumerate(gallery_classes):
        c_dir = os.path.join(ucmerced_dir, c_name)
        img_file = sorted(os.listdir(c_dir))[0]
        full_img_p = os.path.join(c_dir, img_file)
        img_pil = Image.open(full_img_p).convert("RGB")

        cam_mask, overlay_pil, pred_idx, conf = compute_gradcam_overlay(
            model=best_model,
            image_pil=img_pil,
            model_name=best_model_name,
            device=device,
            use_gradcam_plus_plus=True
        )
        pred_label = classes[pred_idx]
        is_mapped = c_name in UCMERCED_TO_EUROSAT_MAPPING
        gt_label = UCMERCED_TO_EUROSAT_MAPPING.get(c_name, f"OOD ({c_name})")

        # Original
        axes[i, 0].imshow(img_pil.resize((img_size, img_size)))
        axes[i, 0].set_title(f"UC Merced: {c_name}\nGT: {gt_label}", fontsize=9, fontweight='bold')
        axes[i, 0].axis('off')

        # Heatmap
        axes[i, 1].imshow(cam_mask, cmap='jet')
        axes[i, 1].set_title("Grad-CAM++ Activation", fontsize=9)
        axes[i, 1].axis('off')

        # Overlay
        match = (pred_label == gt_label)
        color = 'green' if match else ('purple' if not is_mapped else 'red')
        axes[i, 2].imshow(overlay_pil)
        axes[i, 2].set_title(f"Pred: {pred_label} ({conf*100:.1f}%)\n{'CORRECT' if match else ('OOD' if not is_mapped else 'MISCLASS')}",
                             fontsize=9, fontweight='bold', color=color)
        axes[i, 2].axis('off')

    plt.suptitle(f"Cross-Dataset Feature Attribution on UC Merced ({best_model_name.upper()})", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    gallery_path = os.path.join(outputs_dir, "ucmerced_gradcam_gallery.png")
    plt.savefig(gallery_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Artifact] Saved Grad-CAM gallery to {gallery_path}")

    # 7. Write Full JSON Summary
    full_report = {
        "dataset_name": "UC Merced Land Use Dataset",
        "dataset_source": "USGS National Map (Aerial Orthophotos, USA)",
        "spatial_resolution": "1 foot (0.3m) per pixel (vs EuroSAT 10m/pixel)",
        "mapped_categories_count": len(mapped_samples),
        "ood_categories_count": len(ood_samples),
        "comparison_table": comparison_records,
        "per_class_transfer_accuracy": per_class_accs,
        "ood_analysis": ood_summary,
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    save_json(full_report, os.path.join(outputs_dir, "cross_dataset_report.json"))
    print(f"[Artifact] Saved full JSON report to {os.path.join(outputs_dir, 'cross_dataset_report.json')}")

    print("\n" + "="*70)
    print("CROSS-DATASET EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"Results directory: {outputs_dir}/")
    print("="*70)

    return full_report

if __name__ == "__main__":
    evaluate_cross_dataset()
