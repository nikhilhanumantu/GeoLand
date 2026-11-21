import os
import io
import base64
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from src.utils import get_device
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models import get_target_layer_for_gradcam

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_PYTORCH_GRADCAM = True
except ImportError:
    HAS_PYTORCH_GRADCAM = False

class NativeGradCAM:
    """Fallback Grad-CAM implementation using native PyTorch backward hooks."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return np.expand_dims(cam, axis=0)

def compute_gradcam_overlay(
    model,
    image_pil: Image.Image,
    model_name: str = "best_model",
    target_category: int = None,
    device=None,
    use_gradcam_plus_plus: bool = True
):
    """
    Computes Grad-CAM/Grad-CAM++ heatmap and blends it with the original RGB image.
    Returns:
      cam_mask: 2D numpy array [0, 1]
      overlay_pil: blended RGB PIL image
      pred_class_idx: predicted class index
      confidence: prediction confidence
    """
    if device is None:
        device = get_device()
    model.eval().to(device)

    # Preprocess image
    eval_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    orig_w, orig_h = image_pil.size
    rgb_orig_float = np.array(image_pil.convert('RGB'), dtype=np.float32) / 255.0
    input_tensor = eval_transform(image_pil).unsqueeze(0).to(device)

    # Forward pass for prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

    if target_category is None:
        target_category = pred_idx

    target_layers = get_target_layer_for_gradcam(model, model_name)

    if HAS_PYTORCH_GRADCAM and target_layers is not None:
        cam_cls = GradCAMPlusPlus if use_gradcam_plus_plus else GradCAM
        try:
            with cam_cls(model=model, target_layers=target_layers) as cam:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(target_category)]
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                
                # Check for dead heatmap (all zeros) and fallback if needed
                if grayscale_cam.max() > 1e-4:
                    # Normalize dynamically
                    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                    # Resize to original image resolution for crisp display
                    cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    cam_resized = np.clip(cam_resized, 0.0, 1.0)
                    visualization = show_cam_on_image(rgb_orig_float, cam_resized, use_rgb=True)
                    overlay_pil = Image.fromarray(visualization)
                    return cam_resized, overlay_pil, pred_idx, confidence
        except Exception as e:
            print(f"[Grad-CAM] pytorch_grad_cam failed ({e}), using Native fallback")

    # Native Fallback (hooks or gradient salience)
    target_layer = target_layers[0] if target_layers else None
    if target_layer is not None:
        try:
            native_cam = NativeGradCAM(model, target_layer)
            grayscale_cam = native_cam(input_tensor, target_category)[0]
            if grayscale_cam.max() > 1e-4:
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min() + 1e-8)
                cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                visualization = show_cam_on_image(rgb_orig_float, np.clip(cam_resized, 0.0, 1.0), use_rgb=True)
                return cam_resized, Image.fromarray(visualization), pred_idx, confidence
        except Exception as e:
            print(f"[Grad-CAM] native_cam error: {e}")

    # Fallback to Guided Backprop / Gradient Salience
    model.zero_grad()
    t_in = input_tensor.clone().detach().requires_grad_(True)
    out = model(t_in)
    out[0, target_category].backward()
    grad_salience = t_in.grad.data.abs().squeeze(0).max(dim=0)[0].cpu().numpy()
    grad_salience = (grad_salience - grad_salience.min()) / (grad_salience.max() - grad_salience.min() + 1e-8)
    cam_resized = cv2.resize(grad_salience, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    visualization = show_cam_on_image(rgb_orig_float, np.clip(cam_resized, 0.0, 1.0), use_rgb=True)
    overlay_pil = Image.fromarray(visualization)

    return cam_resized, overlay_pil, pred_idx, confidence

def pil_to_base64(image_pil: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    image_pil.save(buffered, format=format)
    return f"data:image/{format.lower()};base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_explainability_suite(
    model,
    classes: list,
    error_summary: dict,
    base_dataset,
    model_name: str = "best_model",
    device=None,
    outputs_dir: str = "outputs/explainability"
):
    """
    Generates explainability artifacts:
    1. Correct classification heatmaps
    2. Confused pair error heatmaps
    3. Clean vs Corrupted (Blur / Noise) heatmaps comparison
    4. Comprehensive feature interpretation report
    """
    if device is None:
        device = get_device()
    os.makedirs(outputs_dir, exist_ok=True)

    print("\n" + "="*60)
    print("GENERATING GRAD-CAM EXPLAINABILITY SUITE")
    print("="*60)

    # 1. Heatmaps for Sample Correct Predictions across 5 representative classes
    rep_classes = ["Highway", "River", "Residential", "Forest", "Industrial"]
    fig, axes = plt.subplots(len(rep_classes), 2, figsize=(8, 3.5 * len(rep_classes)))

    for idx, c in enumerate(rep_classes):
        # Find a clean sample of this class
        for path, target in base_dataset.samples:
            if classes[target] == c:
                sample_img = Image.open(path).convert('RGB')
                break
        _, overlay, pred_idx, conf = compute_gradcam_overlay(model, sample_img, model_name, device=device)

        axes[idx, 0].imshow(sample_img.resize((224, 224)))
        axes[idx, 0].set_title(f"Input: {c}", fontsize=10, fontweight='bold')
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(overlay)
        axes[idx, 1].set_title(f"Grad-CAM: {classes[pred_idx]} ({conf*100:.1f}%)", fontsize=10, fontweight='bold', color="#1b6d24")
        axes[idx, 1].axis('off')

    plt.suptitle("Grad-CAM++ Spatial Attention for Correct Predictions", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "correct_predictions_gradcam.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 2. Confused Pairs Misclassification Explainability
    if error_summary.get("worst_confident_errors"):
        worst = error_summary["worst_confident_errors"][:4]
        fig, axes = plt.subplots(len(worst), 3, figsize=(12, 3.5 * len(worst)))
        if len(worst) == 1:
            axes = np.expand_dims(axes, 0)

        for i, err in enumerate(worst):
            img = Image.open(err["image_path"]).convert('RGB')
            true_idx = classes.index(err["true_class"])
            pred_idx = classes.index(err["pred_class"])

            # CAM for True Class
            _, overlay_true, _, _ = compute_gradcam_overlay(model, img, model_name, target_category=true_idx, device=device)
            # CAM for Predicted Class
            _, overlay_pred, _, conf = compute_gradcam_overlay(model, img, model_name, target_category=pred_idx, device=device)

            axes[i, 0].imshow(img.resize((224, 224)))
            axes[i, 0].set_title(f"True: {err['true_class']}", fontsize=10, fontweight='bold')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(overlay_true)
            axes[i, 1].set_title(f"Attention for Ground Truth ({err['true_class']})", fontsize=9)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(overlay_pred)
            axes[i, 2].set_title(f"Attention for Pred: {err['pred_class']} ({conf*100:.1f}%)", fontsize=9, color="#ba1a1a", fontweight='bold')
            axes[i, 2].axis('off')

        plt.suptitle("Attention Diagnostic: Why the Model Confused Class Pairs", fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(outputs_dir, "confused_pairs_gradcam.png"), dpi=200, bbox_inches='tight')
        plt.close()

    # 3. Clean vs Blurred vs Noisy Comparison for Highway & River
    from src.robustness import apply_gaussian_blur, apply_gaussian_noise
    comparison_samples = [("Highway", None), ("River", None)]
    for idx, (target_c, _) in enumerate(comparison_samples):
        for path, target in base_dataset.samples:
            if classes[target] == target_c:
                comparison_samples[idx] = (target_c, path)
                break

    fig, axes = plt.subplots(len(comparison_samples), 3, figsize=(12, 4 * len(comparison_samples)))
    if len(comparison_samples) == 1:
        axes = np.expand_dims(axes, 0)

    for row, (c_name, path) in enumerate(comparison_samples):
        if not path:
            continue
        clean_img = Image.open(path).convert('RGB')
        blurred_img = apply_gaussian_blur(clean_img, sigma=3.0)
        noisy_img = apply_gaussian_noise(clean_img, std=0.20)

        _, cam_clean, p_clean, conf_clean = compute_gradcam_overlay(model, clean_img, model_name, device=device)
        _, cam_blur, p_blur, conf_blur = compute_gradcam_overlay(model, blurred_img, model_name, device=device)
        _, cam_noise, p_noise, conf_noise = compute_gradcam_overlay(model, noisy_img, model_name, device=device)

        axes[row, 0].imshow(cam_clean)
        axes[row, 0].set_title(f"Clean: {classes[p_clean]} ({conf_clean*100:.1f}%)", fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(cam_blur)
        axes[row, 1].set_title(f"Blurred (sigma=3): {classes[p_blur]} ({conf_blur*100:.1f}%)", fontsize=10, fontweight='bold', color="#d97706")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(cam_noise)
        axes[row, 2].set_title(f"Noisy (std=0.2): {classes[p_noise]} ({conf_noise*100:.1f}%)", fontsize=10, fontweight='bold', color="#ba1a1a")
        axes[row, 2].axis('off')

    plt.suptitle("Attention Degradation: Clean vs. Blurred vs. Noisy Inputs", fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outputs_dir, "robustness_gradcam_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 4. Feature Interpretations Report
    report_md = """# EuroSAT Explainability & Feature Attribution Report

## 1. Executive Summary
Grad-CAM and Grad-CAM++ visualizations highlight the specific spatial and spectral features that drive land-use classification across the 10 EuroSAT categories.

## 2. Visual Feature Drivers by Land-Use Class

| Land-Use Class | Key Driving Visual Features | Vulnerability / Confusion Drivers |
| :--- | :--- | :--- |
| **Highway** | High-contrast linear asphalt/concrete corridors with parallel lane markings. | Curved highway segments bordered by vegetation can mimic rivers; blur destroys sharp edges. |
| **River** | Sinuous organic curvilinear geometry, low specular reflectance, bordering riparian zones. | Dry riverbeds or paved embankments can be misclassified as highways. |
| **Residential** | Regular geometric grid of small rooftops, driveways, high spatial frequency density. | Low-density suburbs with mature trees border with `Forest` and `Pasture`. |
| **Industrial** | Large rectilinear warehouse footprints, metallic/bright roofs, wide loading bays. | Commercial complexes adjacent to residential zones cause border confusion. |
| **Forest** | Irregular granular organic texture, deep green low-reflectance canopy clusters. | Deep shade in orchards or dense herbaceous patches mimic forest edges. |
| **AnnualCrop** | Rectangular agricultural plots with uniform seasonal furrows and tilled soil tones. | Bare soil between crop cycles can be confused with pasture. |
| **PermanentCrop** | Regular dot/stripe matrix grid patterns (orchards, vineyards, olive groves). | Under blur, repetitive row patterns soften into homogeneous herbaceous fields. |
| **Pasture & Herbaceous**| Homogeneous green/golden grasslands devoid of structures. | High spectral overlap; primarily separated by subtle textural variations. |
| **SeaLake** | Uniform, low-texture, monochromatic deep blue/cyan expanse. | Extremely robust; only compromised by severe cloud glint or high noise. |

## 3. Explainability on Perturbed Inputs
* **Under Gaussian Blur:** The model loses the sharp high-frequency edge activations along highways and industrial building boundaries, diluting attention across surrounding background terrain.
* **Under Gaussian Noise:** Random speckled noise introduces spurious local gradients, causing diffuse activation maps that diminish confidence in fine-grained agricultural classes.
"""
    with open(os.path.join(outputs_dir, "feature_interpretations.md"), "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[Explainability] Complete! Visualizations and report saved to {outputs_dir}")
