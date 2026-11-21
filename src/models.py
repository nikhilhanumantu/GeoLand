import torch
import torch.nn as nn
import timm
from torchvision import models as tv_models

MODEL_REGISTRY = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny"
}

def build_model(model_name: str, num_classes: int = 10, pretrained: bool = True):
    """
    Factory function to build one of the 3 specified models with pretrained weights:
    1. resnet50 (Baseline)
    2. efficientnet_b0 (Lightweight)
    3. convnext_tiny (Modern Vision Architecture)
    """
    name_clean = model_name.lower().replace("-", "_")
    if name_clean not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name '{model_name}'. Choose from {list(MODEL_REGISTRY.keys())}")
    
    timm_name = MODEL_REGISTRY[name_clean]
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    return model

def prepare_model_for_finetuning(model, model_name: str, freeze_early_layers: bool = True):
    """
    Freezes generic low-level layers and leaves the top domain-specific stages
    and classification head trainable for fast, stable transfer learning on EuroSAT.
    """
    if not freeze_early_layers:
        for p in model.parameters():
            p.requires_grad = True
        return model

    name_clean = model_name.lower().replace("-", "_")
    
    # Freeze all initially
    for p in model.parameters():
        p.requires_grad = False

    if "resnet" in name_clean:
        # Unfreeze layer4 and fc
        if hasattr(model, 'layer4'):
            for p in model.layer4.parameters(): p.requires_grad = True
        if hasattr(model, 'fc'):
            for p in model.fc.parameters(): p.requires_grad = True
        elif hasattr(model, 'head'):
            for p in model.head.parameters(): p.requires_grad = True

    elif "efficientnet" in name_clean:
        # Unfreeze last stages and classifier
        if hasattr(model, 'blocks'):
            for block in model.blocks[-2:]:
                for p in block.parameters(): p.requires_grad = True
        if hasattr(model, 'conv_head'):
            for p in model.conv_head.parameters(): p.requires_grad = True
        if hasattr(model, 'bn2'):
            for p in model.bn2.parameters(): p.requires_grad = True
        if hasattr(model, 'classifier'):
            for p in model.classifier.parameters(): p.requires_grad = True

    elif "convnext" in name_clean:
        # Unfreeze last stage and head
        if hasattr(model, 'stages'):
            for p in model.stages[-1].parameters(): p.requires_grad = True
        if hasattr(model, 'head'):
            for p in model.head.parameters(): p.requires_grad = True

    return model

def get_target_layer_for_gradcam(model, model_name: str):
    """Returns target convolutional layer for Grad-CAM explainability."""
    name_clean = model_name.lower().replace("-", "_")
    if "resnet" in name_clean:
        if hasattr(model, 'layer4'):
            return [model.layer4[-1]]
    elif "efficientnet" in name_clean:
        if hasattr(model, 'conv_head'):
            return [model.conv_head]
        elif hasattr(model, 'blocks'):
            return [model.blocks[-1]]
    elif "convnext" in name_clean:
        if hasattr(model, 'stages'):
            last_block = model.stages[-1].blocks[-1]
            if hasattr(last_block, 'norm'):
                return [last_block.norm]
            return [last_block]
    
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            return [module]
    return None

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
