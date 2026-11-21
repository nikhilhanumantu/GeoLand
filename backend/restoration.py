"""
GeoLand AI - Image Restoration Module
Connects to Hugging Face GFPGAN Space with robust local fallback.
"""

import io
import base64
import tempfile
import os
import time
import numpy as np
import cv2
from PIL import Image
from PIL import ImageFilter, ImageEnhance

# ─────────────────── HF Space client ────────────────────────────

_hf_client = None
_hf_client_tried = False

def _get_hf_client():
    global _hf_client, _hf_client_tried
    if _hf_client_tried:
        return _hf_client
    _hf_client_tried = True
    try:
        from gradio_client import Client
        _hf_client = Client(
            "randomtable/Image-Restoration-GFPGAN",
            httpx_kwargs={"timeout": 45.0},
        )
        print("[Restoration] Connected to HF Space randomtable/Image-Restoration-GFPGAN")
    except Exception as e:
        print(f"[Restoration] HF Space unavailable ({e}). Will use local fallback.")
        _hf_client = None
    return _hf_client


# ─────────────────── local satellite super-resolution ────────────

def _local_satellite_enhance(pil_img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Local high-quality satellite image enhancement using PIL only
    (avoids cv2 version compatibility issues):
    - Bicubic 2x upscaling
    - Contrast-Limited enhancement via ImageEnhance
    - Unsharp-mask edge sharpening
    """
    new_w = pil_img.width  * scale
    new_h = pil_img.height * scale

    # 1. Bicubic upscale
    upscaled = pil_img.resize((new_w, new_h), Image.BICUBIC)

    # 2. Mild contrast enhancement (CLAHE-like effect)
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.3)

    # 3. Mild sharpness boost
    upscaled = ImageEnhance.Sharpness(upscaled).enhance(1.6)

    # 4. Unsharp mask (best for satellite textures)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    # 5. Very slight saturation boost to restore color vividness
    upscaled = ImageEnhance.Color(upscaled).enhance(1.1)

    return upscaled


def _pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _sharpness_score(pil_img: Image.Image) -> float:
    """Laplacian variance as a proxy for image sharpness (using PIL only)."""
    gray = pil_img.convert("L")
    gray_np = np.array(gray, dtype=np.float32)
    # Simple Laplacian via numpy convolution
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    from scipy.ndimage import convolve
    try:
        result = convolve(gray_np, laplacian)
        return float(np.var(result))
    except Exception:
        # Fallback if scipy not available
        return float(np.var(gray_np))


# ─────────────────── public API ─────────────────────────────────

def restore_image(
    pil_img: Image.Image,
    model_version: str = "v1.4",
    scale: int = 2,
) -> dict:
    """
    Restore and upscale an image.
    Tries HF Space first; falls back to local satellite enhancement automatically.

    Returns:
        {
          restored_image_base64, original_resolution, restored_resolution,
          scale_factor, sharpness_before, sharpness_after, sharpness_improvement_pct,
          method_used, success
        }
    """
    pil_img = pil_img.convert("RGB")
    orig_w, orig_h = pil_img.size

    sharpness_before = round(_sharpness_score(pil_img), 2)
    method_used = "unknown"
    restored    = None

    # ── Try Hugging Face Space ──────────────────────────────────
    client = _get_hf_client()
    if client is not None:
        try:
            # Save to temp file — HF Space needs a file path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                pil_img.save(tmp_path, format="PNG")

            with open(tmp_path, "rb") as f:
                b64_input = "data:image/png;base64," + base64.b64encode(f.read()).decode()

            result = client.predict(
                input=b64_input,
                version=model_version,
                rescaling_factor=scale,
                api_name="/predict",
            )
            os.unlink(tmp_path)

            # result[0] is the restored image (base64 or path)
            restored_b64 = result[0]
            if isinstance(restored_b64, str) and restored_b64.startswith("data:image"):
                # Strip data-url header
                _, encoded = restored_b64.split(",", 1)
                restored = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
            elif isinstance(restored_b64, str) and os.path.exists(restored_b64):
                restored = Image.open(restored_b64).convert("RGB")

            if restored is not None:
                method_used = f"Hugging Face GFPGAN Space (model {model_version})"

        except Exception as e:
            print(f"[Restoration] HF Space call failed: {e}. Using local fallback.")
            restored = None

    # ── Local fallback ──────────────────────────────────────────
    if restored is None:
        restored    = _local_satellite_enhance(pil_img, scale=scale)
        method_used = "Local Satellite Super-Resolution (CLAHE + Unsharp + Bilateral)"

    new_w, new_h = restored.size
    sharpness_after = round(_sharpness_score(restored), 2)
    improvement_pct = round(
        (sharpness_after - sharpness_before) / max(sharpness_before, 1) * 100, 1
    )

    return {
        "success":                True,
        "restored_image_base64":  _pil_to_base64(restored),
        "original_resolution":    f"{orig_w} × {orig_h} px",
        "restored_resolution":    f"{new_w} × {new_h} px",
        "scale_factor":           scale,
        "sharpness_before":       sharpness_before,
        "sharpness_after":        sharpness_after,
        "sharpness_improvement_pct": improvement_pct,
        "method_used":            method_used,
    }
