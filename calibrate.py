"""Temperature calibration for GROUT confidence maps.

Fits a single temperature T by pixel-NLL (LBFGS) on a folder of real
images + binary masks, writes calibration.json. Inference applies T when
calibration.json is present next to the model/cwd or --temperature is
passed; default T = 1.0 (exact no-op). Hook only - nothing blocks when no
calibration data exists.

Library:
    apply_temperature(logits, T) -> probabilities (float32)
    fit_temperature(logits, targets) -> T (float)
    nll(logits, targets, T) -> mean pixel NLL

CLI:
    python calibrate.py --images DIR --masks DIR --model CKPT
                        [--out calibration.json] [--device cpu]
                        [--max-pixels 2000000]
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """sigmoid(logits / T) as float32. T=1 is a no-op; any T>0 preserves
    the per-pixel ranking (monotone transform)."""
    t = float(temperature)
    if t <= 0:
        raise ValueError(f"temperature must be > 0, got {t}")
    z = np.asarray(logits, dtype=np.float32) / np.float32(t)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Mean pixel negative log-likelihood at temperature T."""
    import torch
    z = torch.as_tensor(np.asarray(logits, np.float32)) / float(temperature)
    y = torch.as_tensor(np.asarray(targets, np.float32))
    return float(torch.nn.functional.binary_cross_entropy_with_logits(
        z, y, reduction="mean"))


def fit_temperature(logits: np.ndarray, targets: np.ndarray,
                    max_iter: int = 100) -> float:
    """Single-parameter temperature fit by LBFGS on pixel NLL.
    Parametrized as T = exp(s) for positivity."""
    import torch
    z = torch.as_tensor(np.asarray(logits, np.float32).ravel())
    y = torch.as_tensor(np.asarray(targets, np.float32).ravel())
    s = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([s], lr=0.1, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z / torch.exp(s), y, reduction="mean")
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(s).item())


def resolve_temperature(explicit, model_path=None) -> float:
    """Inference-side resolution: explicit --temperature wins; else
    calibration.json next to the model, then cwd; else 1.0."""
    if explicit is not None:
        return float(explicit)
    candidates = []
    if model_path is not None:
        candidates.append(Path(model_path).parent / "calibration.json")
    candidates.append(Path("calibration.json"))
    for c in candidates:
        if c.exists():
            try:
                return float(json.loads(c.read_text())["temperature"])
            except Exception:
                pass
    return 1.0


def _collect_logits(images_dir: Path, masks_dir: Path, model_path: str,
                    device: str, max_pixels: int):
    import cv2
    import torch
    from config import Config
    from inference import GroutSegmenter

    seg = GroutSegmenter(model_path, device=device, temperature=1.0)
    logits_all, targets_all = [], []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in exts)
    for ip in images:
        mp = next((masks_dir / (ip.stem + e) for e in (".png", ip.suffix)
                   if (masks_dir / (ip.stem + e)).exists()), None)
        if mp is None:
            continue
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        mask = cv2.resize(mask, (Config.IMG_SIZE, Config.IMG_SIZE),
                          interpolation=cv2.INTER_NEAREST)
        x = seg.preprocess(img).to(seg.device)
        with torch.no_grad():
            out = seg.model(x).squeeze().cpu().numpy().astype(np.float32)
        logits_all.append(out.ravel())
        targets_all.append((mask > 127).astype(np.float32).ravel())
    if not logits_all:
        raise SystemExit("no image/mask pairs found")
    z = np.concatenate(logits_all)
    y = np.concatenate(targets_all)
    if z.size > max_pixels:
        idx = np.random.default_rng(0).choice(z.size, max_pixels, replace=False)
        z, y = z[idx], y[idx]
    return z, y, len(images)


def main():
    ap = argparse.ArgumentParser(description="Fit temperature calibration")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--model", type=str,
                    default="checkpoints/grout_b3_zeroshot_v1.pth")
    ap.add_argument("--out", type=Path, default=Path("calibration.json"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-pixels", type=int, default=2_000_000)
    args = ap.parse_args()

    z, y, n_imgs = _collect_logits(args.images, args.masks, args.model,
                                   args.device, args.max_pixels)
    t = fit_temperature(z, y)
    before, after = nll(z, y, 1.0), nll(z, y, t)
    args.out.write_text(json.dumps({
        "temperature": t, "fitted_on_images": n_imgs,
        "fitted_on_pixels": int(z.size),
        "nll_before": before, "nll_after": after,
        "model": str(args.model), "date": date.today().isoformat()}, indent=1))
    print(f"T = {t:.4f}  (pixel NLL {before:.4f} -> {after:.4f}, "
          f"{n_imgs} images) -> {args.out}")


if __name__ == "__main__":
    main()
