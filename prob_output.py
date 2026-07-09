"""Soft-output export: per-pixel confidence maps as first-class outputs.

Canonical format: float32 .npy (values in [0, 1], native inference grid).
Viewing format:   8-bit grayscale PNG (quantized at export only).

PIPELINE-ORDER INVARIANT (see README): thresholding the exported map at
Config.PREDICTION_THRESHOLD reproduces the native-grid, PRE-morphology
binary mask bit-exactly. The shipped mask additionally passes through
post_process_grout (morphology) and, on the resize path, a bilinear resize;
those stages are NOT invertible from any probability map, so the export
happens at the sigmoid stage where the invariant provably holds.
"""
from pathlib import Path

import cv2
import numpy as np

from config import Config


def save_prob_outputs(prob: np.ndarray, out_dir, image_name: str) -> dict:
    """Write <name>_prob_<run>.npy (float32) + <name>_prob8_<run>.png.

    Returns {"npy": Path, "png8": Path}. `prob` must be a 2-D float array
    in [0, 1] at the native inference grid; it is stored as float32
    unmodified (quantization happens only for the PNG).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = np.asarray(prob, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"prob map must be 2-D, got shape {prob.shape}")

    npy_path = out_dir / Config.get_result_filename(image_name, "prob", "npy")
    png_path = out_dir / Config.get_result_filename(image_name, "prob8", "png")
    np.save(npy_path, prob)
    cv2.imwrite(str(png_path), np.round(prob * 255.0).astype(np.uint8))
    return {"npy": npy_path, "png8": png_path}


def save_var_outputs(var: np.ndarray, out_dir, image_name: str) -> dict:
    """TTA variance export: <name>_var_<run>.npy + <name>_var8_<run>.png.
    var8 is scaled by the theoretical max Bernoulli variance (0.25)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var = np.asarray(var, dtype=np.float32)
    npy_path = out_dir / Config.get_result_filename(image_name, "var", "npy")
    png_path = out_dir / Config.get_result_filename(image_name, "var8", "png")
    np.save(npy_path, var)
    cv2.imwrite(str(png_path),
                np.round(np.clip(var / 0.25, 0, 1) * 255.0).astype(np.uint8))
    return {"npy": npy_path, "png8": png_path}


def load_prob_map(path) -> np.ndarray:
    """Load a probability map as float32 in [0, 1].

    Accepts float .npy, 8-bit PNG (val/255) and 16-bit PNG (val/65535).
    """
    path = Path(path)
    if path.suffix.lower() == ".npy":
        m = np.load(path).astype(np.float32)
        return np.clip(m, 0.0, 1.0)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0)
    if img.dtype == np.uint16:
        return (img.astype(np.float32) / 65535.0)
    return np.clip(img.astype(np.float32), 0.0, 1.0)
