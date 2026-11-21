import os
import sys
import time
import json
import torch
import shutil

from src.utils import get_device, set_seed, save_json
from src.dataset import prepare_eurosat_data
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model_on_test, generate_comparison_table
from src.robustness import run_robustness_benchmark
from src.error_analysis import analyze_errors
from src.explainability import generate_explainability_suite

def run_complete_pipeline(
    data_dir: str = "dataset/EuroSAT_RGB/EuroSAT_RGB",
    outputs_dir: str = "outputs",
    models_dir: str = "models",
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 5e-4,
    img_size: int = 112,
    subsample_train_per_class: int = 250,
    subsample_eval_per_class: int = 75,
    quick_test: bool = False
):
    set_seed(42)
    device = get_device()

    print("\n" + "#"*70)
    print("GEOLAND AI — EUROSAT MACHINE LEARNING TRAINING & EVALUATION PIPELINE")
    print("#"*70)

    # STEP 2: DATA PIPELINE
    print("\n>>> STEP 2: DATA PIPELINE (Stratified Split 70/15/15, Sample Grid)")
    data_dict = prepare_eurosat_data(
        data_dir=data_dir,
        output_dir=outputs_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=42,
        subsample_train_per_class=subsample_train_per_class,
        subsample_eval_per_class=subsample_eval_per_class
    )
    classes = data_dict["classes"]
    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    test_loader = data_dict["test_loader"]
    base_dataset = data_dict["full_dataset"]
    test_indices = data_dict["active_test_indices"]

    # STEP 3: MODELS TRAINING (ResNet50, EfficientNet-B0, ConvNeXt-Tiny)
    print("\n>>> STEP 3: FINE-TUNING PRETRAINED MODELS")
    model_names = ["resnet50", "efficientnet_b0", "convnext_tiny"]
    if quick_test:
        epochs = 2
        print(f"[Quick Test Mode] Running {epochs} epochs per model.")

    training_histories = {}
    best_weights_paths = {}

    for m_name in model_names:
        history, ckpt_path = train_model(
            model_name=m_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=len(classes),
            epochs=epochs,
            lr=lr,
            patience=3,
            freeze_early=True,
            device=device,
            models_dir=models_dir,
            outputs_dir=outputs_dir
        )
        training_histories[m_name] = history
        best_weights_paths[m_name] = ckpt_path

    # Determine Best Model by Validation Accuracy
    best_model_name = max(training_histories, key=lambda k: training_histories[k]["best_val_acc"])
    best_val_acc = training_histories[best_model_name]["best_val_acc"]
    print(f"\n[Model Selection] Best Model: {best_model_name.upper()} (Best Val Accuracy: {best_val_acc*100:.2f}%)")

    # Copy best model to models/best_model.pt
    best_pt_path = os.path.join(models_dir, "best_model.pt")
    shutil.copyfile(best_weights_paths[best_model_name], best_pt_path)
    save_json({
        "best_model_name": best_model_name,
        "best_val_accuracy": best_val_acc,
        "selected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "img_size": img_size,
        "classes": classes
    }, os.path.join(models_dir, "best_model_metadata.json"))
    print(f"[Model Selection] Copied {best_weights_paths[best_model_name]} -> {best_pt_path}")

    # STEP 4: STANDARD EVALUATION ON TEST SET
    print("\n>>> STEP 4: STANDARD TEST SET EVALUATION")
    all_eval_results = []
    loaded_models = {}
    test_predictions_map = {}

    for m_name in model_names:
        model = build_model(m_name, num_classes=len(classes), pretrained=False)
        ckpt = torch.load(best_weights_paths[m_name], map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        loaded_models[m_name] = model

        eval_res, (preds, targets, probs) = evaluate_model_on_test(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=device,
            model_name=m_name,
            outputs_dir=outputs_dir
        )
        all_eval_results.append(eval_res)
        test_predictions_map[m_name] = (preds, targets, probs)

    # Comparison Table
    generate_comparison_table(all_eval_results, outputs_dir=outputs_dir)

    # STEP 5: ROBUSTNESS ANALYSIS
    print("\n>>> STEP 5: ROBUSTNESS ANALYSIS (Blur & Noise Stress-Testing)")
    robustness_results = run_robustness_benchmark(
        models_dict=loaded_models,
        base_dataset=base_dataset,
        test_indices=test_indices,
        classes=classes,
        device=device,
        outputs_dir=os.path.join(outputs_dir, "robustness"),
        batch_size=batch_size
    )

    # STEP 6: ERROR & CONFUSION ANALYSIS
    print("\n>>> STEP 6: ERROR AND CONFUSION ANALYSIS")
    best_preds, best_targets, best_probs = test_predictions_map[best_model_name]
    error_summary = analyze_errors(
        all_preds=best_preds,
        all_targets=best_targets,
        all_probs=best_probs,
        base_dataset=base_dataset,
        test_indices=test_indices,
        classes=classes,
        model_name=best_model_name,
        outputs_dir=os.path.join(outputs_dir, "errors")
    )

    # STEP 7: EXPLAINABILITY (Grad-CAM & Grad-CAM++)
    print("\n>>> STEP 7: EXPLAINABILITY (Grad-CAM & Visual Feature Attribution)")
    generate_explainability_suite(
        model=loaded_models[best_model_name],
        classes=classes,
        error_summary=error_summary,
        base_dataset=base_dataset,
        model_name=best_model_name,
        device=device,
        outputs_dir=os.path.join(outputs_dir, "explainability")
    )

    print("\n" + "="*70)
    print("ALL PIPELINE STEPS (2 through 7) COMPLETED SUCCESSFULLY!")
    print(f"Checkpoints: {models_dir}/")
    print(f"Artifacts:   {outputs_dir}/")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EuroSAT ML Training & Evaluation Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick run")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    run_complete_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        quick_test=args.quick
    )
