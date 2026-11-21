"""
GeoLand AI - Regenerate Robustness & Explainability Benchmark
Standardized at img_size=112 across ResNet50, EfficientNet-B0, and ConvNeXt-Tiny.
"""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import torch
from src.utils import get_device
from src.models import build_model
from src.dataset import prepare_eurosat_data
from src.robustness import run_robustness_benchmark
from src.explainability import generate_explainability_suite

def main():
    device = get_device()
    print(f"[Regenerate] Using device: {device}")

    models_dir = "models"
    outputs_dir = "outputs"
    classes = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    model_names = ["resnet50", "efficientnet_b0", "convnext_tiny"]

    # 1. Load trained models with float32 weight upcasting for CPU
    loaded_models = {}
    for m in model_names:
        ckpt_path = os.path.join(models_dir, f"{m}_best.pt")
        if not os.path.exists(ckpt_path):
            if m == "convnext_tiny" and os.path.exists(os.path.join(models_dir, "best_model.pt")):
                ckpt_path = os.path.join(models_dir, "best_model.pt")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[Regenerate] Loading {m} from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(m, num_classes=len(classes), pretrained=False)
        sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(sd)
        model.to(device).eval()
        loaded_models[m] = model

    # 2. Prepare EuroSAT data with standardized 112x112 resolution
    data_dir = "dataset/EuroSAT_RGB/EuroSAT_RGB"
    if not os.path.exists(data_dir):
        data_dir = "dataset/EuroSAT_RGB"

    print(f"[Regenerate] Loading EuroSAT from {data_dir} at img_size=112...")
    data_dict = prepare_eurosat_data(
        data_dir=data_dir,
        output_dir=outputs_dir,
        img_size=112,
        batch_size=64,
        seed=42,
        subsample_train_per_class=250,
        subsample_eval_per_class=75
    )
    base_dataset = data_dict["full_dataset"]
    test_indices = data_dict["active_test_indices"]
    print(f"[Regenerate] Test samples: {len(test_indices)}")

    # 3. Run Robustness Benchmark at 112x112
    print("\n" + "="*60)
    print("RUNNING STANDARDIZED 112x112 ROBUSTNESS BENCHMARK")
    print("="*60)
    robustness_results = run_robustness_benchmark(
        models_dict=loaded_models,
        base_dataset=base_dataset,
        test_indices=test_indices,
        classes=classes,
        device=device,
        outputs_dir=os.path.join(outputs_dir, "robustness"),
        batch_size=64
    )

    print("\n--> ROBUSTNESS CLEAN BASELINES:")
    for m in model_names:
        clean = robustness_results["benchmark_data"][m]["clean_acc"] * 100
        print(f"    {m.upper()}: Clean Accuracy = {clean:.2f}%")

    # 4. Run Explainability Suite with fixed ConvNeXt channel layout
    print("\n" + "="*60)
    print("GENERATING GRAD-CAM EXPLAINABILITY SUITE (ConvNeXt-Tiny)")
    print("="*60)
    error_summary = {}
    err_json_path = os.path.join(outputs_dir, "errors", "error_analysis.json")
    if os.path.exists(err_json_path):
        with open(err_json_path, "r") as f:
            error_summary = json.load(f)

    generate_explainability_suite(
        model=loaded_models["convnext_tiny"],
        classes=classes,
        error_summary=error_summary,
        base_dataset=base_dataset,
        model_name="convnext_tiny",
        device=device,
        outputs_dir=os.path.join(outputs_dir, "explainability")
    )
    print("\n[Regenerate] All robustness and explainability outputs regenerated successfully!")

if __name__ == "__main__":
    main()
