"""Probability-map fusion: combine a GROUT confidence map with an external
0-255 (or float) probability map.

Library:  fuse_prob_maps(a, b, *, mode, prior=0.5, resample="bilinear")
CLI:      python fuse_probs.py A B --mode gate|logit|min [--prior 0.5]
                                   [--nearest] [--out OUT_BASE]

Semantics (see README): `mode` is REQUIRED and has no default because the
right choice depends on how the maps relate:
  gate  : a*b            P(grout | mosaic) * P(mosaic) - conditional gating
  logit : sigmoid(logit(a) + logit(b) - logit(prior))  - INDEPENDENT evidence
  min   : min(a, b)      conservative; the safe choice when both maps are
                          derived from the same image (correlated evidence)
gate/logit double-count if the maps share image evidence.

All math float32; maps normalized to [0,1] then clamped to
[1e-6, 1-1e-6]; B is resampled to A's grid (bilinear default).
Quantization to 8-bit happens only at PNG export.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from prob_output import load_prob_map

EPS = 1e-6
_MODES = ("gate", "logit", "min")


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(p) - np.log1p(-p)


def fuse_prob_maps(a: np.ndarray, b: np.ndarray, *, mode: str,
                   prior: float = 0.5, resample: str = "bilinear") -> np.ndarray:
    """Fuse two probability maps; returns float32 on A's grid."""
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if b.shape != a.shape:
        interp = cv2.INTER_NEAREST if resample == "nearest" else cv2.INTER_LINEAR
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=interp)
    a = np.clip(a, EPS, 1.0 - EPS)
    b = np.clip(b, EPS, 1.0 - EPS)

    if mode == "gate":
        out = a * b
    elif mode == "min":
        out = np.minimum(a, b)
    else:  # logit
        prior = float(np.clip(prior, EPS, 1.0 - EPS))
        z = _logit(a) + _logit(b) - _logit(np.float32(prior))
        out = 1.0 / (1.0 + np.exp(-z))
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Fuse two probability maps")
    ap.add_argument("map_a", type=Path, help="reference map (grid kept)")
    ap.add_argument("map_b", type=Path, help="map resampled onto A's grid")
    ap.add_argument("--mode", required=True, choices=_MODES)
    ap.add_argument("--prior", type=float, default=0.5,
                    help="prior for logit mode (default 0.5)")
    ap.add_argument("--nearest", action="store_true",
                    help="nearest-neighbour resampling for B")
    ap.add_argument("--out", type=Path, default=None,
                    help="output basename (default: <A>_fused_<mode>)")
    args = ap.parse_args()

    a = load_prob_map(args.map_a)
    b = load_prob_map(args.map_b)
    fused = fuse_prob_maps(a, b, mode=args.mode, prior=args.prior,
                           resample="nearest" if args.nearest else "bilinear")
    base = args.out or args.map_a.with_suffix("").with_name(
        args.map_a.stem + f"_fused_{args.mode}")
    np.save(str(base) + ".npy", fused)
    cv2.imwrite(str(base) + ".png",
                np.round(fused * 255.0).astype(np.uint8))
    print(f"fused ({args.mode}) -> {base}.npy / {base}.png "
          f"[{fused.shape[1]}x{fused.shape[0]}, float32]")


if __name__ == "__main__":
    main()
