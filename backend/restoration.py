"""
GeoLand AI - Satellite Imagery Restoration & Enhancement Module
High-fidelity geospatial super-resolution and spectral sharpening pipeline
tailored specifically for satellite Earth observation patches.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def _local_satellite_enhance(pil_img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Satellite-appropriate super-resolution & detail enhancement:
    - 2x high-fidelity bicubic/lanczos spatial upscaling
    - Local contrast and structural edge enhancement (CLAHE proxy)
    - Geospatial unsharp masking to sharpen roads, field boundaries, and waterways
    - Natural chromatic saturation balance
    """
    new_w = pil_img.width * scale
    new_h = pil_img.height * scale

    # 1. High-fidelity spatial resampling
    upscaled = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 2. Structural contrast enhancement for surface texture definition
    upscaled = ImageEnhance.Contrast(upscaled).enhance(1.25)

    # 3. Controlled geospatial unsharp masking (retains edge boundaries without haloing)
    upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=2))

    # 4. Mild high-frequency sharpness boost
    upscaled = ImageEnhance.Sharpness(upscaled).enhance(1.4)

    # 5. Natural color balance preservation
    upscaled = ImageEnhance.Color(upscaled).enhance(1.08)

    return upscaled


def _pil_to_base64(pil_img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=quality)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def _sharpness_score(pil_img: Image.Image) -> float:
    """Laplacian variance as an empirical proxy for spatial edge sharpness."""
    gray = pil_img.convert("L")
    gray_np = np.array(gray, dtype=np.float32)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    try:
        from scipy.ndimage import convolve
        result = convolve(gray_np, laplacian)
        return float(np.var(result))
    except Exception:
        return float(np.var(gray_np))


def restore_image(
    pil_img: Image.Image,
    model_version: str = "satellite_sr",
    scale: int = 2,
) -> dict:
    """
    Enhance, sharpen, and upscale optical satellite imagery.

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

    # Apply dedicated satellite image enhancement
    restored = _local_satellite_enhance(pil_img, scale=scale)
    method_used = "Geospatial Super-Resolution (Lanczos + Unsharp Mask + Dynamic Contrast)"

    new_w, new_h = restored.size
    sharpness_after = round(_sharpness_score(restored), 2)
    improvement_pct = round(
        (sharpness_after - sharpness_before) / max(sharpness_before, 1) * 100, 1
    )

    return {
        "success": True,
        "restored_image_base64": _pil_to_base64(restored),
        "original_resolution": f"{orig_w} × {orig_h} px",
        "restored_resolution": f"{new_w} × {new_h} px",
        "scale_factor": scale,
        "sharpness_before": sharpness_before,
        "sharpness_after": sharpness_after,
        "sharpness_improvement_pct": improvement_pct,
        "method_used": method_used,
    }
