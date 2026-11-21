import os
import io
import json
import base64
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from backend.device import get_device
from src.utils import is_high_impact_error, get_category_for_class, CLASS_CATEGORIES
from src.models import build_model
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.robustness import apply_gaussian_blur, apply_gaussian_noise, apply_brightness_change
from src.explainability import compute_gradcam_overlay, pil_to_base64
from backend.deep_analyzer import analyze_geospatial_scene, stabilize_input_image, classify_crop_with_tta
from backend.restoration import restore_image as _restore_image
from backend.kaggle_classifier import classify_kaggle_satellite

# Global state
app_state = {
    "model": None,
    "model_name": "resnet50",
    "classes": [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ],
    "device": None,
    "transform": None,
    "metrics_cache": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize device, load metadata (img_size), load classes, load best model
    device = get_device()
    app_state["device"] = device

    # Read img_size from best_model_metadata.json as single source of truth
    meta_path = "models/best_model_metadata.json"
    img_size = 112
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                img_size = int(json.load(f).get("img_size", 112))
        except Exception:
            pass
    app_state["img_size"] = img_size

    app_state["transform"] = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    # Load classes
    classes_path = "models/class_names.json"
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            app_state["classes"] = json.load(f)

    # Load model checkpoint
    model_path = "models/best_model.pt"
    if not os.path.exists(model_path):
        # Fallback to individual best model if available
        for m in ["convnext_tiny", "resnet50", "efficientnet_b0"]:
            p = f"models/{m}_best.pt"
            if os.path.exists(p):
                model_path = p
                break

    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            m_name = ckpt.get("model_name", "resnet50")
            app_state["model_name"] = m_name
            model = build_model(m_name, num_classes=len(app_state["classes"]), pretrained=False)
            sd = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()}
            model.load_state_dict(sd)
            model.to(device).eval()
            app_state["model"] = model
            print(f"[Backend] Successfully loaded model '{m_name}' from {model_path}")
        except Exception as e:
            print(f"[Backend] Error loading checkpoint ({e}). Building fresh pretrained model...")
            model = build_model("resnet50", num_classes=len(app_state["classes"]), pretrained=True)
            model.to(device).eval()
            app_state["model"] = model
    else:
        print("[Backend] No checkpoint found yet. Initializing pretrained ResNet50 baseline...")
        model = build_model("resnet50", num_classes=len(app_state["classes"]), pretrained=True)
        model.to(device).eval()
        app_state["model"] = model

    yield
    print("[Backend] Shutting down.")

app = FastAPI(
    title="GeoLand AI - EuroSAT Inference & Explainability API",
    description="Backend for Satellite Land-Use Classification, Robustness Stress-Testing, and Grad-CAM Attribution",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return RedirectResponse(url="/static/index.html")

def read_image_from_upload(upload_file: UploadFile) -> Image.Image:
    if upload_file.content_type and not upload_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")
    try:
        contents = upload_file.file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image size exceeds 10MB limit.")
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "GeoLand AI Inference Engine",
        "model_loaded": app_state["model"] is not None,
        "model_name": app_state["model_name"],
        "device": str(app_state["device"]),
        "num_classes": len(app_state["classes"])
    }

@app.post("/predict")
async def predict_land_use(
    file: UploadFile = File(...),
    uncertainty_threshold: float = 0.60
):
    """
    Accepts an uploaded satellite patch and returns:
    - predicted class name & land-use category with Test-Time Augmentation (TTA) stability
    - top-3 and full 10-class probability breakdown
    - deep geospatial multi-zone spatial decomposition (Left/Center/Right sectors)
    - semantic land-cover segmentation overlay base64
    """
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    raw_image = read_image_from_upload(file)
    image = stabilize_input_image(raw_image)

    import time
    t0 = time.time()

    # 1. Test-Time Augmentation (TTA) on full scene for stable predictions
    tta_res = classify_crop_with_tta(
        app_state["model"],
        image,
        app_state["transform"],
        app_state["device"],
        app_state["classes"]
    )
    probs = tta_res["probs"]
    pred_class = tta_res["class"]
    confidence = tta_res["confidence"]

    # 2. Deep Multi-Zone Scene Analysis & Spatial Decomposition
    deep_analysis = analyze_geospatial_scene(
        app_state["model"],
        image,
        app_state["classes"],
        app_state["device"],
        app_state["transform"]
    )

    # 3. Kaggle 99% Accuracy Satellite Classification (edumisvieramartin architecture)
    kaggle_analysis = classify_kaggle_satellite(image)

    latency_ms = round((time.time() - t0) * 1000, 2)
    is_uncertain = bool(confidence < uncertainty_threshold)

    # All classes ranked
    sorted_indices = np.argsort(probs)[::-1]
    all_predictions = [
        {
            "class": app_state["classes"][i],
            "category": get_category_for_class(app_state["classes"][i]),
            "probability": round(float(probs[i]), 4),
            "percentage": round(float(probs[i] * 100), 2)
        }
        for i in sorted_indices
    ]
    top_3 = all_predictions[:3]

    return {
        "status": "success",
        "prediction": {
            "class": pred_class,
            "category": get_category_for_class(pred_class),
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "is_uncertain": is_uncertain,
            "uncertainty_threshold": uncertainty_threshold,
            "is_composite_scene": deep_analysis["is_composite_scene"]
        },
        "top_3": top_3,
        "all_classes": all_predictions,
        "deep_analysis": deep_analysis,
        "kaggle_classification": kaggle_analysis,
        "telemetry": {
            "latency_ms": latency_ms,
            "device": str(app_state["device"]).upper(),
            "model_architecture": app_state["model_name"],
            "input_dimensions": f"{raw_image.width} × {raw_image.height}",
            "inference_input_size": f"{app_state.get('img_size', 112)} × {app_state.get('img_size', 112)}"
        }
    }

@app.post("/predict-kaggle")
async def predict_kaggle_only(file: UploadFile = File(...)):
    """
    Dedicated endpoint for the Kaggle 99% Accuracy Satellite Classifier
    from kernel: edumisvieramartin/satellite-image-classif-99-accuracy
    Classes: cloudy, desert, green_area, water
    """
    raw_image = read_image_from_upload(file)
    image = stabilize_input_image(raw_image)
    result = classify_kaggle_satellite(image)
    return {"status": "success", **result}

@app.post("/explain")
async def explain_prediction(
    file: UploadFile = File(...),
    target_class: Optional[str] = Form(None),
    use_plus_plus: bool = Form(True)
):
    """
    Computes Grad-CAM (or Grad-CAM++) overlay for an uploaded satellite image.
    Returns:
    - original image base64
    - Grad-CAM activation heatmap overlay base64
    - predicted class and confidence
    """
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    image = read_image_from_upload(file)
    target_cat_idx = None
    if target_class and target_class in app_state["classes"]:
        target_cat_idx = app_state["classes"].index(target_class)
    else:
        # Use grounded TTA prediction so heatmap always explains the displayed class
        tta_eval = classify_crop_with_tta(
            app_state["model"],
            image,
            app_state["transform"],
            app_state["device"],
            app_state["classes"]
        )
        target_cat_idx = app_state["classes"].index(tta_eval["class"])

    cam_mask, overlay_pil, pred_idx, confidence = compute_gradcam_overlay(
        app_state["model"],
        image,
        model_name=app_state["model_name"],
        target_category=target_cat_idx,
        device=app_state["device"],
        use_gradcam_plus_plus=use_plus_plus
    )

    pred_class = app_state["classes"][pred_idx]
    original_b64 = pil_to_base64(image.resize((224, 224)))
    overlay_b64 = pil_to_base64(overlay_pil)

    return {
        "status": "success",
        "predicted_class": pred_class,
        "category": get_category_for_class(pred_class),
        "confidence": round(confidence, 4),
        "method": "Grad-CAM++" if use_plus_plus else "Grad-CAM",
        "original_image_base64": original_b64,
        "gradcam_overlay_base64": overlay_b64
    }

@app.post("/robustness")
async def test_robustness(
    file: UploadFile = File(...),
    corruption: str = Form("noise"),  # "noise", "blur", "brightness"
    severity: float = Form(0.10)
):
    """
    Applies real-time degradation (Gaussian Blur or Noise) and performs inference.
    Returns clean prediction vs corrupted prediction and base64 preview of degraded image.
    """
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    clean_image = read_image_from_upload(file)

    # 1. Clean Inference
    clean_tensor = app_state["transform"](clean_image).unsqueeze(0).to(app_state["device"])
    with torch.no_grad():
        clean_logits = app_state["model"](clean_tensor)
        clean_probs = torch.softmax(clean_logits, dim=1).squeeze().cpu().numpy()
    clean_idx = int(np.argmax(clean_probs))
    clean_class = app_state["classes"][clean_idx]
    clean_conf = float(clean_probs[clean_idx])

    # 2. Corrupt Image
    if corruption == "blur":
        corrupted_image = apply_gaussian_blur(clean_image, sigma=float(severity))
    elif corruption == "noise":
        corrupted_image = apply_gaussian_noise(clean_image, std=float(severity))
    elif corruption == "brightness":
        corrupted_image = apply_brightness_change(clean_image, factor=float(severity))
    else:
        corrupted_image = clean_image

    # 3. Corrupted Inference
    corrupted_tensor = app_state["transform"](corrupted_image).unsqueeze(0).to(app_state["device"])
    with torch.no_grad():
        corr_logits = app_state["model"](corrupted_tensor)
        corr_probs = torch.softmax(corr_logits, dim=1).squeeze().cpu().numpy()
    corr_idx = int(np.argmax(corr_probs))
    corr_class = app_state["classes"][corr_idx]
    corr_conf = float(corr_probs[corr_idx])

    high_impact_change = is_high_impact_error(clean_class, corr_class)

    return {
        "status": "success",
        "corruption_type": corruption,
        "severity": severity,
        "clean_result": {
            "class": clean_class,
            "category": get_category_for_class(clean_class),
            "confidence": round(clean_conf, 4)
        },
        "corrupted_result": {
            "class": corr_class,
            "category": get_category_for_class(corr_class),
            "confidence": round(corr_conf, 4),
            "prediction_shifted": bool(clean_class != corr_class),
            "high_impact_error_shift": high_impact_change
        },
        "clean_image_base64": pil_to_base64(clean_image.resize((224, 224))),
        "corrupted_image_base64": pil_to_base64(corrupted_image.resize((224, 224)))
    }

@app.get("/metrics")
def get_all_metrics():
    """
    Returns full evaluation metrics:
    - Model comparison table
    - Confusion matrices (raw and normalized)
    - Top confused pairs and cost-of-error flags
    - Robustness degradation curves and class fragility
    - Training curves history
    """
    out = {
        "model_comparison": None,
        "confusion_matrix": None,
        "error_analysis": None,
        "robustness": None,
        "cross_dataset": None,
        "training_curves": {}
    }

    # Model comparison
    comp_path = "outputs/model_comparison.json"
    if os.path.exists(comp_path):
        with open(comp_path, "r") as f:
            out["model_comparison"] = json.load(f)

    # Best model eval & confusion matrix
    eval_path = f"outputs/{app_state['model_name']}_eval.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            out["confusion_matrix"] = json.load(f)
    elif os.path.exists("outputs/convnext_tiny_eval.json"):
        with open("outputs/convnext_tiny_eval.json", "r") as f:
            out["confusion_matrix"] = json.load(f)
    elif os.path.exists("outputs/resnet50_eval.json"):
        with open("outputs/resnet50_eval.json", "r") as f:
            out["confusion_matrix"] = json.load(f)

    # Error analysis
    err_path = "outputs/errors/error_analysis.json"
    if os.path.exists(err_path):
        with open(err_path, "r") as f:
            out["error_analysis"] = json.load(f)

    # Robustness
    rob_path = "outputs/robustness/robustness_results.json"
    if os.path.exists(rob_path):
        with open(rob_path, "r") as f:
            out["robustness"] = json.load(f)

    # Cross-dataset benchmark
    cross_path = "outputs/cross_dataset/cross_dataset_report.json"
    if os.path.exists(cross_path):
        with open(cross_path, "r") as f:
            out["cross_dataset"] = json.load(f)

    # Training curves
    curves_dir = "outputs/curves"
    if os.path.exists(curves_dir):
        for fname in os.listdir(curves_dir):
            if fname.endswith("_history.json"):
                m_key = fname.replace("_history.json", "")
                with open(os.path.join(curves_dir, fname), "r") as f:
                    out["training_curves"][m_key] = json.load(f)

    return out

@app.get("/preset-samples")
def get_preset_samples():
    """Returns EuroSAT sample patches for all 10 land-use classes."""
    base_dir = "dataset/EuroSAT_RGB/EuroSAT_RGB"
    if not os.path.exists(base_dir):
        base_dir = "dataset/EuroSAT_RGB"
    preset_classes = [
        "Highway", "River", "Forest", "Residential",
        "AnnualCrop", "HerbaceousVegetation", "Industrial",
        "Pasture", "PermanentCrop", "SeaLake"
    ]
    samples = {}
    if os.path.exists(base_dir):
        for c in preset_classes:
            c_dir = os.path.join(base_dir, c)
            if os.path.exists(c_dir):
                files = sorted([f for f in os.listdir(c_dir) if f.endswith(".jpg")])
                if files:
                    img_path = os.path.join(c_dir, files[0])
                    img = Image.open(img_path).convert('RGB')
                    samples[c] = {
                        "class": c,
                        "category": get_category_for_class(c),
                        "filename": files[0],
                        "image_base64": pil_to_base64(img)
                    }
    return samples

@app.get("/dataset-samples")
def get_dataset_samples(limit_per_class: int = 3):
    """Returns a gallery of sample patches across all EuroSAT classes for Dataset Explorer."""
    base_dir = "dataset/EuroSAT_RGB/EuroSAT_RGB"
    if not os.path.exists(base_dir):
        base_dir = "dataset/EuroSAT_RGB"
    gallery = []
    classes = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
    ]
    if os.path.exists(base_dir):
        for c in classes:
            c_dir = os.path.join(base_dir, c)
            if os.path.exists(c_dir):
                files = sorted([f for f in os.listdir(c_dir) if f.endswith(".jpg")])[:limit_per_class]
                for idx, f in enumerate(files):
                    img_path = os.path.join(c_dir, f)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        gallery.append({
                            "id": f"{c}_{idx+1}",
                            "class": c,
                            "category": get_category_for_class(c),
                            "filename": f,
                            "resolution": "64x64 px (10m GSD)",
                            "bands": "RGB Sentinel-2 (B04, B03, B02)",
                            "image_base64": pil_to_base64(img)
                        })
                    except Exception:
                        pass
    return {"total": len(gallery), "classes": classes, "samples": gallery}

@app.post("/restore")
async def restore_image_endpoint(
    file: UploadFile = File(...),
    model_version: str = Form("v1.4"),
    scale: int = Form(2),
):
    """
    AI Image Restoration and Super-Resolution.
    Tries HF Space GFPGAN first; falls back to local satellite enhancement.
    Returns base64 restored image, resolution change, sharpness improvement.
    """
    raw_image = read_image_from_upload(file)

    try:
        result = _restore_image(raw_image, model_version=model_version, scale=scale)
        return {"status": "success", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restoration failed: {str(e)}")


# Mount frontend directory for direct browser viewing
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")
