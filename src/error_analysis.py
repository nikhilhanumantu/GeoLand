import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils import is_high_impact_error, get_category_for_class, save_json

def analyze_errors(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    all_probs: np.ndarray,
    base_dataset,
    test_indices: list,
    classes: list,
    model_name: str = "best_model",
    outputs_dir: str = "outputs/errors"
):
    """
    Identifies top confused class pairs, flags high-impact cost-of-error misclassifications
    (urban/industrial vs natural), and generates a visual gallery of worst errors.
    """
    os.makedirs(outputs_dir, exist_ok=True)
    num_classes = len(classes)

    # 1. Bidirectional & Directional Confused Pairs
    confusion_pairs = {}
    misclassified_samples = []

    for i in range(len(all_targets)):
        true_idx = int(all_targets[i])
        pred_idx = int(all_preds[i])
        probs = all_probs[i]
        confidence = float(probs[pred_idx])

        if true_idx != pred_idx:
            true_name = classes[true_idx]
            pred_name = classes[pred_idx]

            pair_key = tuple(sorted([true_name, pred_name]))
            dir_key = f"{true_name} -> {pred_name}"

            if pair_key not in confusion_pairs:
                confusion_pairs[pair_key] = {"pair": f"{pair_key[0]} <-> {pair_key[1]}", "count": 0, "directions": {}}
            confusion_pairs[pair_key]["count"] += 1

            if dir_key not in confusion_pairs[pair_key]["directions"]:
                confusion_pairs[pair_key]["directions"][dir_key] = 0
            confusion_pairs[pair_key]["directions"][dir_key] += 1

            # Check Cost of Error
            high_impact = is_high_impact_error(true_name, pred_name)

            real_idx = test_indices[i]
            img_path = base_dataset.samples[real_idx][0]

            misclassified_samples.append({
                "test_index": i,
                "dataset_index": real_idx,
                "image_path": img_path,
                "true_class": true_name,
                "true_category": get_category_for_class(true_name),
                "pred_class": pred_name,
                "pred_category": get_category_for_class(pred_name),
                "confidence": round(confidence, 4),
                "high_impact_error": high_impact
            })

    # Rank top 10 confused pairs
    sorted_pairs = sorted(confusion_pairs.values(), key=lambda x: x["count"], reverse=True)
    top_10_pairs = sorted_pairs[:10]

    # Save Top Confused Pairs CSV
    pairs_df = pd.DataFrame([
        {
            "Rank": idx + 1,
            "Class Pair": p["pair"],
            "Total Misclassifications": p["count"],
            "Breakdown": "; ".join([f"{k}: {v}" for k, v in p["directions"].items()])
        }
        for idx, p in enumerate(top_10_pairs)
    ])
    pairs_csv_path = os.path.join(outputs_dir, "top_10_confused_pairs.csv")
    pairs_df.to_csv(pairs_csv_path, index=False)

    # 2. Separate High Impact Errors (Urban/Industrial <-> Natural)
    high_impact_errors = [s for s in misclassified_samples if s["high_impact_error"]]
    normal_errors = [s for s in misclassified_samples if not s["high_impact_error"]]

    # Sort worst misclassifications by highest model confidence in wrong prediction
    worst_errors = sorted(misclassified_samples, key=lambda x: x["confidence"], reverse=True)

    # Save Error Summary JSON
    error_summary = {
        "model_name": model_name,
        "total_test_samples": len(all_targets),
        "total_misclassifications": len(misclassified_samples),
        "overall_error_rate_pct": round((len(misclassified_samples) / len(all_targets)) * 100, 2),
        "high_impact_errors_count": len(high_impact_errors),
        "high_impact_error_pct": round((len(high_impact_errors) / max(1, len(misclassified_samples))) * 100, 2),
        "top_10_confused_pairs": top_10_pairs,
        "worst_confident_errors": worst_errors[:15],
        "high_impact_error_samples": high_impact_errors[:15]
    }
    save_json(error_summary, os.path.join(outputs_dir, "error_analysis.json"))

    # 3. Save Visual Gallery of Worst Misclassifications
    gallery_path = os.path.join(outputs_dir, "worst_misclassifications_gallery.png")
    plot_error_gallery(worst_errors[:12], gallery_path)

    # 4. Save High-Impact Error Gallery
    if high_impact_errors:
        hi_gallery_path = os.path.join(outputs_dir, "high_impact_errors_gallery.png")
        plot_error_gallery(high_impact_errors[:12], hi_gallery_path, title="High-Impact Cost-of-Error: Urban vs Natural Confusion")

    print(f"\n[Error Analysis] Total Errors: {len(misclassified_samples)} ({error_summary['overall_error_rate_pct']}%)")
    print(f"[Cost-of-Error] High-Impact Errors: {len(high_impact_errors)} ({error_summary['high_impact_error_pct']}% of errors)")
    print(f"[Error Analysis] Saved reports and gallery to {outputs_dir}")

    return error_summary

def plot_error_gallery(error_list: list, save_path: str, title: str = "Worst Confident Misclassifications"):
    n = min(len(error_list), 12)
    if n == 0:
        return
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = np.array([axes])

    for i in range(rows * cols):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        if i < n:
            err = error_list[i]
            img = Image.open(err["image_path"]).convert('RGB')
            ax.imshow(img)
            hi_badge = " [HIGH IMPACT]" if err.get("high_impact_error") else ""
            color = "#ba1a1a" if err.get("high_impact_error") else "#d97706"
            ax.set_title(
                f"True: {err['true_class']}\nPred: {err['pred_class']} ({err['confidence']*100:.1f}%){hi_badge}",
                fontsize=9, color=color, fontweight='bold', pad=4
            )
        ax.axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
